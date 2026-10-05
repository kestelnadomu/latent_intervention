"""Synthetic-only checks: no production encoding, training, or artifact replacement."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from test_oracle_workflow import StubGemma, config as config
from src import pipeline
from src.hz import (
    hz_benchmark as queue,
    hz_benchmark_matrix as matrix,
    hz_evaluation,
    hz_pilot,
    hz_training,
)
from src.artifact_io import sha256_file, write_json
from src.config import load_config
from src.hz.hz_validation import ValidationControl
from src.hz.oracle_targets import (
    encode_oracle_targets,
    load_oracle_targets,
    source_metadata,
)
from src.pair_encoding import load_latent_artifact
from src.schema import load_schema
from src.semantic_decoder import make_semantic_decoder


@pytest.fixture
def study(config, tmp_path, monkeypatch):
    """Eight fake CVs, tiny heads/flows, two epochs; all outputs stay in tmp_path."""
    config["sim_config"] = str(Path(config["sim_config"]).resolve())
    cfg = config["latent_intervention"]
    cfg.update(d_model=8, nhead=2, dim_feedforward=16, dropout=0)
    cfg["pre_additive"]["n_samples"] = cfg["noise_token"]["n_samples"] = 2
    cfg["dist"].update(top_k=2, embed_dim=2)
    cfg["particles"]["n_particles"] = 2
    cfg["flow"].update(
        n_blocks=2,
        hidden_dim=8,
        condition_dim=8,
        state_embed_dim=2,
        n_samples=2,
        support_bank_size=4,
    )
    cfg["oracle_regression"].update(hidden_dim=8, n_hidden=1, dropout=0)
    artifact = load_latent_artifact(config)
    columns, _ = load_schema(config["sim_config"])
    decoder = make_semantic_decoder(
        "independent",
        latent_dim=768,
        columns=columns,
        hidden_dim=8,
        n_hidden=1,
        dropout=0,
    )
    decoder.save(
        config["paths"]["decoder_model"],
        metadata=pipeline._decoder_metadata(config, artifact),
    )
    settings = matrix.read_settings()
    settings.update(
        dimensions=[768],
        max_epochs=2,
        dist_pretrain_epochs=1,
        patience=1,
        scheduler_patience=1,
        batch_size=3,
        evaluation_batch_size=1,
        evaluation_samples=2,
        validation_samples=2,
        learning_rates=[0.001],
        width_multipliers=[1],
        final_seeds=[42, 43],
        workers=2,
    )
    cases = []
    for variant in matrix.VARIANTS:
        current = copy.deepcopy(config)
        current["latent_intervention"]["variant"] = variant
        cases.append(
            dict(
                case=matrix.case_name(768, variant),
                dimension=768,
                variant=variant,
                config=current,
                supervision=matrix.SUPERVISION[variant],
                validation_metric=matrix.VALIDATION[variant],
            )
        )
        if variant in (*pipeline.FLOW_VARIANTS, "oracle_regression"):
            # Exercise the benchmark override (not the ordinary hidden_dim=8)
            # through training, checkpoint reload, evaluation, and resumption.
            cases[-1]["model_widths"] = {"1": dict(hidden_dim=6)}
        elif variant == "dist":
            cases[-1]["model_widths"] = {"1": dict(d_model=4, dim_feedforward=8)}
    plan = dict(
        settings=settings,
        cases=cases,
        split_ids=matrix.split_ids(config, artifact, settings),
        candidates=matrix.candidates(settings),
        intervention={"X": 3},
        code_sha256={},
        input_sha256={},
        oracle_sources=[
            dict(
                dimension=768,
                path=config["paths"]["oracle_targets"],
                expected_metadata=source_metadata(config, artifact),
            )
        ],
    )
    monkeypatch.chdir(tmp_path)
    return plan, tmp_path / "reports/test-run", config


def test_default_matrix_budget_and_graph():
    settings = matrix.read_settings()
    plan = dict(
        settings=settings,
        candidates=matrix.candidates(settings),
        cases=[
            dict(case=matrix.case_name(d, v), dimension=d, variant=v)
            for d in settings["dimensions"]
            for v in settings["variants"]
        ],
    )
    assert len(plan["cases"]) == 18
    assert queue.fit_count(plan) == 234
    search = queue.jobs_for(plan, "search")
    assert len({j["id"] for j in search}) == 144
    for job in search:
        if job["case"]["variant"] == "distilled_flow":
            assert len(job["dependencies"]) == 8
            assert all(
                f"embeddinggemma_{job['case']['dimension']}/g-independent/h-state_flow/"
                in dep
                for dep in job["dependencies"]
            )
        else:
            assert job["dependencies"] == []
    selection = dict(
        cases={c["case"]: dict(candidate=plan["candidates"][0]) for c in plan["cases"]}
    )
    for job in queue.jobs_for(plan, "final", selection):
        if job["case"]["variant"] == "distilled_flow":
            assert len(job["dependencies"]) == 1
            assert job["dependencies"][0].endswith(f"/seed-{job['seed']}")


def test_pilot_covers_all_families_and_preserves_validation_only_fit(study):
    plan, _, config = study
    plan["pilot_protocol"] = matrix.read_settings()["pilot"]
    jobs = hz_pilot.jobs_for_pilot(plan)
    assert len(jobs) == 13  # eight other families and five pre-additive noise levels
    assert all(job["stage"] == "pilot" and job["seed"] == 42 for job in jobs)
    assert all(job["candidate"]["width_multiplier"] == 1 for job in jobs)
    assert len({job["id"] for job in jobs}) == len(jobs)
    candidates = [job for job in jobs if job["case"]["variant"] == "pre_additive"]
    assert len(candidates) == 5
    columns, _ = load_schema(config["sim_config"])
    sigma = [
        hz_training.model_kwargs(job["case"], job["candidate"], 42, columns, {"X": 3})[
            "noise_std"
        ]
        for job in candidates
    ]
    assert sigma[:4] == pytest.approx([x / 768**0.5 for x in (0.1, 0.25, 0.5, 1.0)])
    assert sigma[-1] == pytest.approx(1.0)
    student = next(job for job in jobs if job["case"]["variant"] == "distilled_flow")
    assert student["dependencies"] == [
        "embeddinggemma_768/g-independent/h-state_flow/pilot/reference"
    ]
    assert not any(job["stage"] in {"final", "evaluation"} for job in jobs)


def test_pilot_noise_report_uses_saved_fits_and_validation_only(study):
    plan, root, _ = study
    plan["cases"] = [c for c in plan["cases"] if c["variant"] == "pre_additive"]
    plan["pilot_protocol"] = dict(
        matrix.read_settings()["pilot"],
        noise_rms=[0.25, 0.5],
        compare_legacy_noise_std_one=False,
    )
    jobs = hz_pilot.jobs_for_pilot(plan)
    reports = {
        job["id"]: hz_training.fit(
            root,
            plan,
            job["case"],
            job["candidate"],
            job["seed"],
            job["stage"],
            job["label"],
            [],
        )
        for job in jobs
    }
    result = hz_pilot.analyze(root, plan, reports)
    assert result["official_test_scoring"] is False
    assert result["completed_fits"] == 2
    assert result["selected_noise_rms"]["768"]["noise_rms"] in (0.25, 0.5)
    assert all(Path(row["training_report"]).exists() for row in result["rows"])
    assert len(result["noise_diagnostics"]) == 2
    assert all(row["test_targets_used"] is False for row in result["rows"])
    hz_pilot.render(root, plan, result, "complete")
    assert "Noise calibration" in (root / "summary.md").read_text()


@pytest.mark.parametrize("dimension", [256, 768])
def test_all_nine_sizes_match_without_changing_the_seven_aligned_models(dimension):
    settings = matrix.read_settings()
    cases = []
    for variant in matrix.VARIANTS:
        config = load_config(
            encoder_variant="embeddinggemma",
            embeddinggemma_dim=dimension,
            decoder_variant="independent",
            manipulator_variant=variant,
        )
        case = dict(
            case=matrix.case_name(dimension, variant),
            dimension=dimension,
            variant=variant,
            config=config,
        )
        if variant in matrix.MODEL_WIDTH_FIELDS:
            case["model_widths"] = settings["model_widths"][variant][str(dimension)]
        cases.append(case)
    plan = dict(cases=cases, settings=settings, intervention={"X": 3})
    rng_before = torch.random.get_rng_state()
    sizes = hz_training.model_sizes(plan)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert json.loads(json.dumps(sizes)) == sizes
    baseline = {
        row["width_multiplier"]: row["trainable_parameters"]
        for row in sizes
        if row["variant"] == "baseline"
    }
    unchanged = {
        256: {
            "baseline": [199552, 660992],
            "pre_additive": [199552, 660992],
            "noise_token": [200192, 662272],
            "particles": [201729, 665345],
            "state_flow": [201808, 648144],
            "distilled_flow": [192144, 672912],
            "direct_semantic_flow": [192144, 672912],
        },
        768: {
            "baseline": [331136, 923648],
            "pre_additive": [331136, 923648],
            "noise_token": [331776, 924928],
            "particles": [333313, 928001],
            "state_flow": [334672, 934096],
            "distilled_flow": [310672, 902928],
            "direct_semantic_flow": [310672, 902928],
        },
    }
    for row in sizes:
        variant, tier = row["variant"], row["width_multiplier"]
        assert abs(row["trainable_parameters"] / baseline[tier] - 1) <= 0.07
        kwargs = row["model_kwargs"]
        cfg = next(c for c in cases if c["variant"] == variant)["config"][
            "latent_intervention"
        ]
        if variant in matrix.MODEL_WIDTH_FIELDS:
            overrides = settings["model_widths"][variant][str(dimension)][str(tier)]
            assert all(kwargs[key] == value for key, value in overrides.items())
        if variant in unchanged[dimension]:
            assert (
                row["trainable_parameters"] == unchanged[dimension][variant][tier - 1]
            )
        else:
            assert abs(row["trainable_parameters"] / baseline[tier] - 1) <= 0.03
        if variant in pipeline.FLOW_VARIANTS:
            assert kwargs["n_blocks"] == 8
            assert kwargs["condition_dim"] == 64
            assert kwargs["state_embed_dim"] == 16
            assert cfg["flow"]["hidden_dim"] == 128
        elif variant == "dist":
            assert kwargs["nhead"] == 4 and kwargs["d_model"] % kwargs["nhead"] == 0
            assert kwargs["embed_dim"] == 16 and kwargs["top_k"] == 16
            assert cfg["d_model"] == 128 and cfg["dim_feedforward"] == 256
        elif variant == "oracle_regression":
            assert kwargs["n_hidden"] == 2
            assert cfg[variant]["hidden_dim"] == 512


@pytest.mark.parametrize("variant", matrix.MODEL_WIDTH_FIELDS)
@pytest.mark.parametrize("bad_width", [None, 0, -1, 1.5, True])
def test_model_width_configuration_fails_closed(tmp_path, variant, bad_width):
    settings = matrix.read_settings()
    widths = settings["model_widths"][variant]["256"]
    if bad_width is None:
        del widths["1"]
    else:
        widths["1"][next(iter(widths["1"]))] = bad_width
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="model_widths"):
        matrix.read_settings(path)


@pytest.mark.parametrize("variant", matrix.MODEL_WIDTH_FIELDS)
def test_model_width_overrides_cannot_change_nonwidth_settings(tmp_path, variant):
    settings = matrix.read_settings()
    settings["model_widths"][variant]["256"]["1"]["dropout"] = 1
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="model_widths"):
        matrix.read_settings(path)


@pytest.mark.parametrize("variant", matrix.VARIANTS)
def test_each_family_fit_reload_evaluate_and_resume(study, variant, monkeypatch):
    plan, root, config = study
    case = next(c for c in plan["cases"] if c["variant"] == variant)
    candidate = plan["candidates"][0]
    oracle_inputs, teacher = [], None
    original = load_latent_artifact(config)
    g_hash = sha256_file(config["paths"]["decoder_model"])
    if variant == "oracle_regression":
        encode_oracle_targets(config, encoder_factory=StubGemma)
        oracle_inputs = matrix.oracle_readiness(plan)
        monkeypatch.setattr(
            hz_training,
            "decoder_for",
            lambda *a: pytest.fail("oracle training must not load g"),
        )
        monkeypatch.setattr(
            hz_training,
            "load_symbolic_kernel",
            lambda *a: pytest.fail("oracle training must not load h_S"),
        )
    else:
        monkeypatch.setattr(
            hz_training,
            "load_oracle_targets",
            lambda *a: pytest.fail("non-oracle must never load true training Z'"),
        )
    # Poisoned official-test targets make accidental usage during training observable.
    monkeypatch.setattr(
        hz_training,
        "load_latent_artifact",
        lambda _: replace(
            original, z_prime=torch.full_like(original.z_prime, float("nan"))
        ),
    )
    if variant == "distilled_flow":
        teacher_case = next(c for c in plan["cases"] if c["variant"] == "state_flow")
        teacher = hz_training.fit(
            root, plan, teacher_case, candidate, 42, "search", "teacher", []
        )
    result = hz_training.fit(
        root, plan, case, candidate, 42, "search", "trial-000", oracle_inputs, teacher
    )
    assert result["test_targets_used"] is False
    assert result["training"]["epochs_run"] <= 2
    assert result["training"]["best_epoch"] >= (2 if variant == "dist" else 1)
    assert set(result["split_ids"]["fit"]).isdisjoint(original.test_ids)
    assert set(result["split_ids"]["validation"]).isdisjoint(original.test_ids)
    reloaded = hz_training.load_model(result["checkpoint"], result["signature"])
    assert result["trainable_parameters"] == sum(
        p.numel() for p in reloaded.parameters() if p.requires_grad
    )
    if variant in matrix.MODEL_WIDTH_FIELDS:
        assert all(
            result["model_kwargs"][key] == value
            for key, value in case["model_widths"]["1"].items()
        )
    assert (
        hz_training.fit(
            root,
            plan,
            case,
            candidate,
            42,
            "search",
            "trial-000",
            oracle_inputs,
            teacher,
        )["checkpoint_sha256"]
        == result["checkpoint_sha256"]
    )
    # Require a real selection freeze before any test scoring.
    with pytest.raises(ValueError, match="frozen"):
        hz_evaluation.evaluate(root, plan, case, result)
    write_json(
        root / "selection.json",
        dict(
            plan_sha256=matrix.digest(plan),
            cases={case["case"]: dict(candidate=candidate)},
        ),
    )
    evaluation = hz_evaluation.evaluate(root, plan, case, result)
    assert evaluation["regimes"]["latent_only"]["all"]["units"] == 2
    assert evaluation["regimes"]["latent_only"]["identity"]["units"] == 1
    assert ("observed_state" in evaluation["regimes"]) == (variant == "state_flow")
    assert hz_evaluation.evaluate(root, plan, case, result) == evaluation
    assert sha256_file(config["paths"]["decoder_model"]) == g_hash
    altered = dict(result, seed=123)
    with pytest.raises(ValueError, match="report checksum"):
        hz_training.verify_fit(altered)


def test_missing_oracle_blocks_launch_not_preparation(study, monkeypatch):
    plan, root, _ = study
    monkeypatch.setattr(queue, "run_root", lambda _: root)
    monkeypatch.setattr(queue, "discover", lambda *a: plan)
    # prepare's readable source snapshots use existing files, not synthetic config text.
    _, prepared, ready = queue.prepare("test-run")
    assert prepared == plan and not ready["ready"]
    assert len(prepared["model_sizes"]) == 9
    assert "## Model sizes" in (root / "summary.md").read_text()
    assert json.loads((root / "plan.json").read_text()) == prepared
    assert queue.prepare("test-run")[1] == prepared
    assert json.loads((root / "status.json").read_text())["training_started"] is False
    assert not list(root.glob("fits/**/*"))
    with pytest.raises(ValueError, match="targets are missing"):
        queue.preflight(root, plan)


def test_control_restores_literal_best_and_does_not_stop_dist_pretraining(tmp_path):
    settings = matrix.read_settings()
    settings.update(patience=1, scheduler_patience=1)
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scores = iter([9, 8, 3, 3.1])
    control = ValidationControl(
        lambda _: next(scores), settings, tmp_path / "progress.json"
    )
    assert not control(model, optimizer, "pretrain", dict(loss=1))
    assert not control(model, optimizer, "pretrain", dict(loss=1))
    with torch.no_grad():
        model.weight.fill_(2)
    assert not control(model, optimizer, "joint", dict(loss=1))
    with torch.no_grad():
        model.weight.fill_(5)
    assert control(model, optimizer, "joint", dict(loss=1))
    summary = control.restore(model)
    assert summary["best_epoch"] == 3
    assert float(model.weight.detach().item()) == 2


def test_queue_dependencies_failures_and_unrelated_progress(tmp_path, monkeypatch):
    plan = dict(settings=dict(workers=2))
    jobs = [
        dict(id=name, dependencies=deps, stage="search")
        for name, deps in (("teacher", []), ("student", ["teacher"]), ("unrelated", []))
    ]
    seen = []

    def execute(root, plan, job, inputs, teacher):
        seen.append(job["id"])
        if job["id"] == "teacher":
            raise ValueError("bad teacher")
        return dict(training=dict(best_validation=1), candidate={})

    monkeypatch.setattr(queue, "execute_job", execute)
    with (
        ThreadPoolExecutor(2) as pool,
        pytest.raises(RuntimeError, match="some fits failed"),
    ):
        queue.run_jobs(tmp_path, plan, jobs, [], pool)
    assert set(seen) == {"teacher", "unrelated"}
    error = json.loads((tmp_path / "errors.json").read_text())
    assert error["blocked"] == ["student"]


def test_settings_and_selection_fail_closed(tmp_path):
    settings = matrix.read_settings()
    settings["variants"] = ["distilled_flow"]
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="requires state_flow"):
        matrix.read_settings(path)
    with pytest.raises(ValueError, match="every search trial"):
        queue.freeze_selection(tmp_path, dict(cases=[{}], candidates=[{}]), {})


def test_preflight_freezes_target_hashes(study):
    plan, root, config = study
    encode_oracle_targets(config, encoder_factory=StubGemma)
    frozen = queue.preflight(root, plan, freeze=True)
    assert frozen[0]["sha256"] == load_oracle_targets(config).artifact_sha256
    assert queue.preflight(root, plan) == frozen
    frozen[0]["sha256"] = "changed"
    write_json(root / "oracle_inputs.json", frozen)
    with pytest.raises(ValueError, match="immutable"):
        queue.preflight(root, plan)


def test_spawned_queue_runs_all_nine_synthetic_cases_and_resumes(study, monkeypatch):
    """Actual spawn workers, tiny fake corpus; no production files or heavy encoder."""
    plan, root, config = study
    plan["settings"]["final_seeds"] = [42]
    encode_oracle_targets(config, encoder_factory=StubGemma)
    monkeypatch.setattr(queue, "run_root", lambda _: root)
    monkeypatch.setattr(
        queue, "prepare", lambda *a: (root, plan, queue.readiness(plan))
    )
    queue.run("test-run")
    result = json.loads((root / "results.json").read_text())
    assert len(result["evaluations"]) == 9
    assert len(result["final_fits"]) == 9
    assert json.loads((root / "status.json").read_text())["fits"] == 18
    published = json.loads((root / "published.json").read_text())
    assert len(published) == 9
    before = {r["checkpoint"]: sha256_file(r["checkpoint"]) for r in published}
    for row in published:
        hz_training.load_model(row["checkpoint"], row["signature"])
    assert "Observed State" in (root / "summary.md").read_text()
    assert "privileged supervised reference" in (root / "summary.md").read_text()
    queue.run("test-run")
    assert before == {r["checkpoint"]: sha256_file(r["checkpoint"]) for r in published}
    assert json.loads((root / "results.json").read_text()) == result


def test_common_metrics_have_known_scale_and_energy():
    from src.schema import ColumnSpec

    class Decoder:
        columns = [ColumnSpec("s", 2)]

        def log_joint(self, z):
            return torch.full((len(z), 2), -torch.log(torch.tensor(2.0)))

    class Kernel:
        def transition_matrix(self, intervention):
            return torch.eye(2).to_sparse()

    z = torch.zeros(3, 2)
    kwargs = dict(
        scale=torch.ones(2),
        decoder=Decoder(),
        h_s=Kernel(),
        intervention={},
        realized={"s": torch.zeros(3, dtype=torch.long)},
    )
    point = hz_evaluation.per_unit_metrics(
        z.unsqueeze(0), z, torch.ones_like(z), **kwargs
    )
    assert torch.allclose(point["energy_score"], torch.ones(3))
    assert torch.allclose(point["standardized_mean_mse"], torch.ones(3))
    assert torch.allclose(point["semantic_kl"], torch.zeros(3))
    pair = torch.stack([-torch.ones_like(z), torch.ones_like(z)])
    mixed = hz_evaluation.per_unit_metrics(pair, z, z, **kwargs)
    assert torch.allclose(mixed["energy_score"], torch.zeros(3), atol=1e-6)
    assert torch.allclose(mixed["sample_spread"], torch.ones(3))
