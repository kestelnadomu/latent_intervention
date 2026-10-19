"""Synthetic tests only: no corpus fitting or official-test evaluation."""

import copy
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from exp.benchmarks.misspecification import ACTIONS, EDITORS, METHODS
from exp.benchmarks.misspecification import (
    common,
    data,
    evaluation,
    metrics,
    models,
    reader,
    reporting,
    sampling,
    scenarios,
    training,
)
from exp.benchmarks.misspecification.run import parser, select_jobs
from src.schema import ColumnSpec, unflatten_state_index
from src.semantic_decoder.model import SemanticDecoder, load_semantic_decoder

COLS = (ColumnSpec("X", 4), ColumnSpec("T", 2))


@pytest.fixture(autouse=True)
def deterministic():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(19)
        yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny():
    states = unflatten_state_index(torch.arange(8), COLS).repeat(5, 1)
    z = torch.randn(40, 4) * 0.2 + states[:, :1] * 0.3
    fit = data.FactualPartition(z[:24], states[:24], tuple(range(24)))
    tune = data.FactualPartition(z[24:32], states[24:32], tuple(range(24, 32)))
    decoder = (
        SemanticDecoder(4, COLS, hidden_dim=8, n_hidden=1).eval().requires_grad_(False)
    )
    matrices = []
    for k in range(4):
        mat = torch.zeros(8, 8)
        mat[torch.arange(8), 2 * k + states[:8, 1]] = 1
        matrices.append(mat)
    cfg = common.read_config()
    cfg.update(
        max_epochs=3, batch_size=8, seed=42, learning_rate=0.003, support_bank_size=8
    )
    cfg["sinkhorn"] = dict(epsilon=0.05, tolerance=1e-5, max_iterations=10000)
    return SimpleNamespace(
        data=data.FactualData(fit, tune, COLS),
        guide=decoder,
        matrices=matrices,
        cfg=cfg,
        states=states,
        z=z,
    )


def fresh(method, tiny):
    return models.construct(method, COLS, dimension=4, seed=42, config=tiny.cfg)


def fit(method, tiny, folder, callback=None):
    kwargs = (
        dict(decoder=tiny.guide, matrices=tiny.matrices) if method in EDITORS else {}
    )
    return training.fit(
        fresh(method, tiny),
        tiny.data,
        tiny.cfg,
        folder / "selected.pt",
        folder / "training.json",
        folder / "resume.pt",
        dict(synthetic=True),
        on_epoch=callback,
        **kwargs,
    )


def test_grid_aliases_refits_and_all_actions():
    cfg = common.read_config()
    cases = scenarios.build()
    rows = scenarios.registry(cases)
    indexed = {r["name"]: r for r in rows}
    assert len(rows) == 15 and len({r["canonical"] for r in rows}) == 13
    assert indexed["O4/block_X_D"]["canonical"] == "O3/drop_X_D"
    assert indexed["O5/c0"]["canonical"] == "S6/true"
    jobs = scenarios.jobs(cfg, rows)
    assert len(jobs) == 196
    for method, count in dict(
        semantic_editor=65, ot_editor=65, state_flow=30, ot_flow=30, ot_gaussian=6
    ).items():
        assert sum(j["method"] == method for j in jobs) == count
    for case in cases:
        states = unflatten_state_index(
            torch.arange(len(case.matrices[0])), case.kernel.columns
        )
        assert len(case.matrices) == 4
        for k, matrix in enumerate(case.matrices):
            torch.testing.assert_close(
                matrix.sum(1), torch.ones(len(matrix)), atol=1e-5, rtol=0
            )
            assert matrix[:, states[:, 0] != k].count_nonzero() == 0
    pilot_jobs = scenarios.jobs(common.read_config(pilot=True), rows)
    assert len(pilot_jobs) == 15 and {j["space"] for j in pilot_jobs} == {
        "S1",
        "S3",
        "S6",
    }
    args = parser().parse_args(["train", "--operator", "O4/block_X_D", "--seed", "43"])
    chosen = select_jobs(dict(config=cfg, scenarios=rows, jobs=jobs), args)
    assert len(chosen) == 5
    assert all(j["operator"] in ("O3/drop_X_D", "factual") for j in chosen)
    assert next(j for j in chosen if j["method"] == "ot_gaussian")["seed"] == 42


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("space", [f"S{i}" for i in range(1, 7)])
def test_real_architecture_capacities(method, space):
    result = models.capacity(models.construct(method, data.columns_for(space)))
    if method != "ot_gaussian":
        assert abs(result["training_total"] - 331136) / 331136 <= 0.07


def test_coupled_probabilities_zeros_frequency_and_batch_order(tiny):
    p = torch.tensor([[0.2, 0.8, 0, 0, 0, 0, 0, 0], [0.7, 0.3, 0, 0, 0, 0, 0, 0]])
    matrix = tiny.matrices[2].clone()
    matrix[0, 4:6] = torch.tensor([0.3, 0.7])
    pairs = sampling.draw_pairs(
        p, matrix, [101, 202], COLS, action={"X": 2}, n_samples=30000
    )
    frequencies = (
        torch.bincount(
            pairs.source[:, 0] * 8 + pairs.target[:, 0], minlength=64
        ).reshape(8, 8)
        / 30000
    )
    torch.testing.assert_close(frequencies, p[0, :, None] * matrix, atol=0.012, rtol=0)
    assert pairs.source.max() == 1 and pairs.target.min() == 4
    for i in (1, 0):
        alone = sampling.draw_pairs(
            p[i : i + 1],
            matrix,
            [[101, 202][i]],
            COLS,
            action={"X": 2},
            n_samples=30000,
        )
        assert torch.equal(alone.source[:, 0], pairs.source[:, i])
        assert torch.equal(alone.target[:, 0], pairs.target[:, i])
    mass = sampling.support_mass(p, matrix, torch.tensor([10, 2, 0, 0, 10, 0, 0, 0]))
    torch.testing.assert_close(mass["unseen_either"], torch.tensor([0.94, 0.79]))
    torch.testing.assert_close(mass["well_supported"], torch.tensor([0.06, 0.21]))
    torch.testing.assert_close(
        sum(
            mass[k]
            for k in ("unseen_either", "rare_either_excluding_unseen", "well_supported")
        ),
        torch.ones(2),
    )
    for action in ({"X": 4}, {"X": 1.0}, {"T": 1}, {"X": True}, {"X": 2, "T": 0}):
        with pytest.raises(ValueError, match="supports"):
            sampling.draw_pairs(p, matrix, [1, 2], COLS, action=action)
    with pytest.raises(ValueError):
        sampling.draw_pairs(p * 0, matrix, [1, 2], COLS, action={"X": 1})


@pytest.mark.parametrize("method", METHODS)
def test_sampling_identity_roundtrip_and_caching(method, tiny, tmp_path):
    model = fresh(method, tiny).eval()
    kwargs = (
        dict(decoder=tiny.guide, matrices=tiny.matrices) if method in EDITORS else {}
    )
    training.prepare(model, tiny.data, tiny.cfg, **kwargs)
    if method == "ot_gaussian":
        model.fit(tiny.data.fit.z, tiny.data.fit.states)
    if method == "semantic_editor":
        with torch.no_grad():
            model.delta.out.bias.fill_(0.1)
    z = tiny.z[:2]
    p = sampling.posterior_rows(tiny.guide, z)
    for action in ACTIONS:
        pairs = sampling.draw_pairs(
            p, tiny.matrices[action["X"]], [0, 1], COLS, action=action, n_samples=11
        )
        samples, diagnostic = sampling.apply_pairs(model, z, pairs)
        assert samples.shape == (11, 2, 4) and diagnostic["sampled_pairs"] == 22
        split = torch.cat(
            [
                sampling.apply_pairs(
                    model,
                    z[i : i + 1],
                    sampling.Pairs(
                        pairs.source[:, i : i + 1], pairs.target[:, i : i + 1], COLS
                    ),
                )[0]
                for i in range(2)
            ],
            1,
        )
        torch.testing.assert_close(split, samples, atol=0, rtol=0)
    empty = sampling.draw_pairs(
        p, tiny.matrices[0], [0, 1], COLS, action={}, n_samples=11
    )
    assert torch.equal(
        sampling.apply_pairs(model, z, empty)[0], z[None].expand(11, -1, -1)
    )
    same = sampling.Pairs(
        torch.zeros(5, 2, dtype=torch.long), torch.zeros(5, 2, dtype=torch.long), COLS
    )
    values, _ = sampling.apply_pairs(model, z, same)
    if method not in EDITORS:
        assert torch.equal(values, z[None].expand_as(values))
    elif method == "semantic_editor":
        assert not torch.equal(values, z[None].expand_as(values))
    models.save(tmp_path / "new.pt", model, {"signature": "test"})
    restored = models.load(tmp_path / "new.pt", signature="test")
    torch.testing.assert_close(
        sampling.apply_pairs(restored, z, pairs)[0], samples, atol=0, rtol=0
    )
    model.save(tmp_path / "native.pt", metadata={"test": True})
    with pytest.raises(ValueError, match="multi-action"):
        models.load(tmp_path / "native.pt")


def test_flow_abducted_residual_is_reused_without_resampling(tiny, monkeypatch):
    model = fresh("state_flow", tiny).eval()
    model.set_latent_statistics(tiny.data.fit.z)
    calls = []
    original_abduct, original_render = model.abduct, model.render

    def abduct(z, source):
        value = original_abduct(z, source)
        calls.append(value.clone())
        return value

    def render(residual, target):
        assert torch.equal(residual, calls[-1])
        return original_render(residual, target)

    monkeypatch.setattr(model, "abduct", abduct)
    monkeypatch.setattr(model, "render", render)
    pairs = sampling.Pairs(
        torch.zeros(8, 1, dtype=torch.long),
        torch.full((8, 1), 7, dtype=torch.long),
        COLS,
    )
    sampling.apply_pairs(model, tiny.z[:1], pairs)
    assert len(calls) == 1 and calls[0].shape == (1, 4)


@pytest.mark.parametrize("method", METHODS)
def test_tiny_fit_restoration_resume_and_frozen_guides(method, tiny, tmp_path):
    before = copy.deepcopy(tiny.guide.state_dict())
    matrices = [m.clone() for m in tiny.matrices]
    full = fit(method, tiny, tmp_path / "full")
    assert full["status"] == "complete"
    if method != "ot_gaussian":
        snapshot = torch.load(tmp_path / "full" / "resume.pt", weights_only=True)
        assert snapshot["best_training_state"]["epoch"] == full["selected_epoch"]
        assert {"optimizer", "scheduler", "rng"} <= snapshot[
            "best_training_state"
        ].keys()
        scores = [
            full["initial_validation"]["objective"],
            *[r["validation"]["objective"] for r in full["history"]],
        ]
        assert full["selected_validation"]["objective"] == pytest.approx(
            min(scores), abs=1e-6
        )

        def stop(epoch):
            raise InterruptedError("synthetic interruption")

        with pytest.raises(InterruptedError):
            fit(method, tiny, tmp_path / "resume", stop)
        resumed = fit(method, tiny, tmp_path / "resume")
        restored = models.load(resumed["checkpoint"])
        for key, value in models.load(full["checkpoint"]).state_dict().items():
            torch.testing.assert_close(
                value, restored.state_dict()[key], atol=0, rtol=0
            )
        assert (
            full["selected_epoch"] == resumed["selected_epoch"]
            and full["optimizer_updates"] == 9
        )
        if method in EDITORS:
            assert full["history"][0]["training"]["action_objective_evaluations"] == 12
    else:
        assert full["optimizer_updates"] == 0 and not full["history"]
    assert (
        fit(method, tiny, tmp_path / "full")["checkpoint_sha256"]
        == full["checkpoint_sha256"]
    )
    for key, value in before.items():
        torch.testing.assert_close(value, tiny.guide.state_dict()[key], atol=0, rtol=0)
    assert all(torch.equal(a, b) for a, b in zip(matrices, tiny.matrices))
    assert all(p.grad is None for p in tiny.guide.parameters())


@pytest.mark.parametrize("method", EDITORS)
def test_four_action_gradient_accumulation_is_exact(method, tiny):
    model = fresh(method, tiny)
    prepared = training.prepare(model, tiny.data, tiny.cfg, tiny.guide, tiny.matrices)
    indices = torch.arange(8)

    def loss(k):
        terms = training.editor_terms(
            model,
            tiny.data.fit,
            indices,
            prepared["targets"]["fit"][k],
            k,
            tiny.guide,
            prepared["bank"],
            create_graph=True,
        )
        return training.total_loss(terms, tiny.cfg) / 4

    sum(loss(k) for k in range(4)).backward()
    expected = {
        n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
    }
    model.zero_grad()
    for k in range(4):
        loss(k).backward()
    for n, p in model.named_parameters():
        if n in expected:
            torch.testing.assert_close(p.grad, expected[n], atol=1e-7, rtol=1e-5)


def test_auxiliary_columns_excluded_and_reader_role_is_separate(tiny, tmp_path):
    artifact = SimpleNamespace(ids=list(range(40)), z=tiny.z)
    ids = dict(
        fit=list(range(24)),
        tune=list(range(24, 32)),
        official_test=list(range(32, 40)),
        reader_fit=list(range(8, 16)),
        reader_L=list(range(8)),
    )
    frame = pd.DataFrame(
        dict(
            id=range(40),
            X=tiny.states[:, 0],
            T=tiny.states[:, 1],
            P=0,
            L=1,
            H=2,
            A=0,
            Y=1,
        )
    )
    factual = tmp_path / "factual.csv"
    frame.to_csv(factual, index=False)
    original = data.select_factual(artifact, factual, ids, COLS)
    frame.drop(columns=["P", "L", "H", "A", "Y"]).to_csv(factual, index=False)
    changed = data.select_factual(artifact, factual, ids, COLS)
    assert not hasattr(changed, "oracle") and not hasattr(changed.fit, "auxiliary")
    for a, b in zip((original.fit, original.tune), (changed.fit, changed.tune)):
        assert (
            a.ids == b.ids and torch.equal(a.z, b.z) and torch.equal(a.states, b.states)
        )
    a = training.prepare(
        fresh("semantic_editor", tiny), original, tiny.cfg, tiny.guide, tiny.matrices
    )
    b = training.prepare(
        fresh("semantic_editor", tiny), changed, tiny.cfg, tiny.guide, tiny.matrices
    )
    assert all(
        torch.equal(x, y) for x, y in zip(a["targets"]["fit"], b["targets"]["fit"])
    )
    audit = data.select_factual(artifact, factual, ids, COLS, role="evaluation_reader")
    assert not set(audit.fit.ids) & set(ids["reader_L"])
    with pytest.raises(TypeError, match="factual"):
        training.prepare(
            fresh("semantic_editor", tiny), audit, tiny.cfg, tiny.guide, tiny.matrices
        )
    source = Path(training.__file__).read_text()
    assert (
        "from .reader" not in source
        and "z_prime" not in source
        and "audit_readers" not in source
    )


def test_evaluation_reader_development_selection_and_interrupted_resume(tiny, tmp_path):
    audit = data.FactualData(tiny.data.fit, tiny.data.tune, COLS, "evaluation_reader")
    settings = dict(
        hidden_dim=8,
        n_hidden=1,
        dropout=0.1,
        lr=0.003,
        weight_decay=0.0,
        batch_size=8,
        epochs=3,
        early_stopping=dict(patience=30, min_delta=1e-7),
        scheduler=dict(patience=10, factor=0.5, min_lr=1e-6),
    )
    meta = dict(latent_artifact_sha256="synthetic", encoder={"variant": "synthetic"})

    def run(where, callback=None):
        return reader.fit_trial(
            audit,
            settings,
            where / "model.pt",
            where / "training.json",
            meta,
            on_epoch=callback,
        )

    full = run(tmp_path / "full")

    def stop(epoch):
        raise InterruptedError

    with pytest.raises(InterruptedError):
        run(tmp_path / "resume", stop)
    resumed = run(tmp_path / "resume")
    assert full["validation_joint_nll"] == min(
        r["validation_joint_nll"] for r in full["history"]
    )
    assert full["selected_epoch"] == resumed["selected_epoch"]
    original = load_semantic_decoder(full["checkpoint"])
    restored = load_semantic_decoder(resumed["checkpoint"])
    for a, b in zip(original.state_dict().values(), restored.state_dict().values()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert not full["used_official_test"] and set(full["calibration"]) == {"X", "T"}
    assert run(tmp_path / "full")["checkpoint_sha256"] == full["checkpoint_sha256"]
    with pytest.raises(TypeError):
        reader.fit_trial(
            tiny.data, settings, tmp_path / "bad.pt", tmp_path / "bad.json", meta
        )


def test_target_matching_metrics_and_unit_bootstrap(tiny):
    ids, available = [1, 2, 3], {1: 0, 2: 1, 3: 0}
    assert metrics.embedding_match(ids, {"X": 0}, "total", available).tolist() == [
        True,
        False,
        True,
    ]
    for target in ("ps_D", "ps_DU"):
        assert not metrics.embedding_match(ids, {"X": 0}, target, available).any()
    z = tiny.z[:3]
    result = metrics.embedding_metrics(
        z[None].expand(64, -1, -1), z + 0.1, torch.ones(4)
    )
    torch.testing.assert_close(result["energy_score"], torch.full((3,), 0.1))
    assert reporting.bootstrap([2.0] * 15) == [2.0, 2.0]
    assert reporting.bootstrap([float("nan")]) is None
    assert reporting.bootstrap([1, 2, 3]) == reporting.bootstrap([1, 2, 3])
    summary = metrics.aggregate(
        dict(energy_score=torch.full((3,), float("nan"))),
        torch.tensor([True, False, False]),
    )
    assert summary["all"]["metrics"]["energy_score"] == dict(units=0, mean=None)


def test_symbolic_command_vs_truth_and_confusion_are_separate(tiny):
    z = tiny.z[:3]
    posterior = sampling.posterior_rows(tiny.guide, z)
    pairs = sampling.draw_pairs(
        posterior, tiny.matrices[3], [0, 1, 2], COLS, action={"X": 3}
    )
    factual = tiny.states[:3]
    truth = factual.clone()
    truth[:, 0] = 3
    gamma = metrics.factual_confusion(tiny.guide, tiny.z, tiny.states, COLS)
    values, strata = metrics.symbolic_metrics(
        z[None].expand(64, -1, -1),
        z,
        posterior,
        tiny.matrices[3],
        pairs,
        factual,
        truth,
        tiny.guide,
        tiny.guide,
        COLS,
        torch.arange(8),
        gamma,
    )
    assert {
        "own_g_semantic_kl",
        "command_accuracy_X",
        "true_accuracy_X",
        "preservation_accuracy_T",
    } <= set(values)
    assert sum(v["draws"] for v in strata.values()) == 64 * 3
    assert torch.isnan(values["preservation_accuracy_X"]).all()


def test_frozen_evaluation_gate_and_exclusive_lock(tmp_path):
    cfg = dict(reports_root=str(tmp_path), run_id="test")
    with pytest.raises(FileNotFoundError):
        evaluation.frozen(cfg, {})
    with common.exclusive(tmp_path / "run.lock"):
        with pytest.raises(RuntimeError, match="another writer"):
            with common.exclusive(tmp_path / "run.lock"):
                pass


def test_synthetic_evaluation_and_report_end_to_end(tiny, tmp_path, monkeypatch):
    from exp.benchmarks.misspecification import AUDIT_COLUMNS
    from src.artifact_io import save_torch, sha256_file

    trained = fit("semantic_editor", tiny, tmp_path / "fit")
    cfg = dict(
        tiny.cfg,
        reports_root=str(tmp_path / "reports"),
        run_id="synthetic",
        evaluation_batch_size=2,
    )
    ids, z = [32, 33, 34], tiny.z[32:35]
    columns = tuple(
        ColumnSpec(n, 4 if n == "X" else 2 if n == "T" else 3) for n in AUDIT_COLUMNS
    )
    auditor = (
        SemanticDecoder(4, columns, hidden_dim=8, n_hidden=1)
        .eval()
        .requires_grad_(False)
    )
    factual = torch.zeros(3, 9, dtype=torch.long)
    factual[:, :2] = tiny.states[32:35]
    cf_rows = []
    for i, unit in enumerate(ids):
        for k in range(4):
            for target in ("total", "ps_D"):
                values = dict(zip(AUDIT_COLUMNS, factual[i].tolist()))
                values["X"] = k
                values["D"] = 1 if target == "total" else 0
                cf_rows.append(dict(id=unit, do_value=k, target=target, **values))
    cf = pd.DataFrame(cf_rows).set_index(["id", "do_value", "target"])
    case = SimpleNamespace(
        name="S1/true",
        space="S1",
        targets=("total", "ps_D"),
        matrices=tiny.matrices,
        fingerprint="synthetic",
    )
    item = dict(
        checkpoint=trained["checkpoint"],
        checkpoint_sha256=trained["checkpoint_sha256"],
        fit_signature=trained["signature"],
    )
    job = dict(space="S1", method="semantic_editor", operator="S1/true", seed=42)
    selection = dict(models={common.job_key(job): item})
    monkeypatch.setattr(
        evaluation, "guide_path", lambda *args: Path(trained["checkpoint"])
    )
    monkeypatch.setattr(evaluation, "guide", lambda *args: tiny.guide)
    monkeypatch.setattr(evaluation, "load_factual", lambda *args: tiny.data)
    toy = dict(
        ids=ids,
        z=z,
        oracle=z + 0.1,
        factual=factual,
        cf=cf,
        columns=columns,
        auditor=auditor,
        available_actions={32: 0, 33: 1, 34: 2},
        scale=torch.ones(4),
        loaded=(None, None, None),
        confusion=metrics.factual_confusion(auditor, z, factual, columns),
    )
    signature = dict(
        jobs=[job],
        scenarios=[
            dict(name="S1/true", canonical="S1/true", targets=["total", "ps_D"])
        ],
    )
    records = []
    for target in case.targets:
        record = evaluation.evaluate_condition(
            cfg, signature, selection, toy, case, "semantic_editor", 42, 0, target
        )
        records.append(record)
        assert record["summary"]["all"]["metrics"]["energy_score"]["units"] == (
            1 if target == "total" else 0
        )
        assert evaluation.evaluate_condition(
            cfg, signature, selection, toy, case, "semantic_editor", 42, 0, target
        ) == common.read_record(
            evaluation.condition_path(cfg, case.name, "semantic_editor", 42, 0, target)
        )
    total, path = [torch.load(r["per_unit"], weights_only=True) for r in records]
    torch.testing.assert_close(
        total["values"]["own_g_semantic_kl"],
        path["values"]["own_g_semantic_kl"],
        atol=0,
        rtol=0,
    )
    result = reporting.report(cfg, signature)
    assert len(result["summaries"]) == 2
    assert (
        result["summaries"]["S1/true|ps_D|semantic_editor"]["metrics"]["energy_score"][
            "mean"
        ]
        is None
    )
    assert not result["paired_differences"]
    assert "unavailable" in (common.root(cfg) / "summary.md").read_text()
    # Per-unit corruption is rejected instead of silently reused.
    save_torch(records[0]["per_unit"], {"corrupt": True})
    assert sha256_file(records[0]["per_unit"]) != records[0]["per_unit_sha256"]
    with pytest.raises(ValueError, match="stale or changed"):
        evaluation.evaluate_condition(
            cfg, signature, selection, toy, case, "semantic_editor", 42, 0, "total"
        )


def test_reader_numerical_procedure_matches_existing_trainer(tiny, tmp_path):
    from src.semantic_decoder.training import train_semantic_decoder

    audit = data.FactualData(tiny.data.fit, tiny.data.tune, COLS, "evaluation_reader")
    settings = dict(
        hidden_dim=8,
        n_hidden=1,
        dropout=0.1,
        lr=0.003,
        weight_decay=0.0,
        batch_size=8,
        epochs=3,
        early_stopping=dict(patience=30, min_delta=1e-7),
        scheduler=dict(patience=10, factor=0.5, min_lr=1e-6),
    )
    meta = dict(latent_artifact_sha256="synthetic", encoder={"variant": "synthetic"})
    result = reader.fit_trial(
        audit, settings, tmp_path / "reader.pt", tmp_path / "training.json", meta
    )
    original = SemanticDecoder(4, COLS, hidden_dim=8, n_hidden=1, dropout=0.1)
    train_semantic_decoder(
        original,
        audit.fit.z,
        reader.labels(audit.fit, COLS),
        epochs=3,
        batch_size=8,
        lr=0.003,
        seed=42,
        verbose=False,
        min_delta=1e-7,
        validation_latents=audit.tune.z,
        validation_targets=reader.labels(audit.tune, COLS),
    )
    reconstructed = load_semantic_decoder(result["checkpoint"])
    for key, value in original.state_dict().items():
        torch.testing.assert_close(
            value, reconstructed.state_dict()[key], atol=0, rtol=0
        )


def test_tracking_boundaries_and_no_automatic_evaluation():
    base = "models/talent_motivation/embeddinggemma_768/g-independent/S1/h-ot_editor/runs/hz-misspec-v1/S1--true/"
    report = "reports/talent_motivation/hz_misspecification/hz-misspec-v1/"
    checks = {
        base + "seed-42/selected/latent_intervention.pt": False,
        base + "seed-43/selected/latent_intervention.pt": True,
        report + "fits/S1/ot_editor/S1--true/seed-43/training.json": False,
        report + "fits/S1/ot_editor/S1--true/seed-43/resume.pt": True,
        report
        + "evaluation/S1--true/ot_editor/seed-43/action-0/total/metrics.json": False,
        report
        + "evaluation/S1--true/ot_editor/seed-43/action-0/total/per_unit.pt": True,
        report + "logs/queue.log": True,
    }
    for path, ignored in checks.items():
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "-v", path],
            check=False,
            capture_output=True,
            text=True,
        )
        matched = result.stdout.split("\t")[0]
        assert (bool(matched) and ":!" not in matched) is ignored, path
    launcher = Path("scripts/run_hz_misspecification.sh").read_text()
    assert (
        "run preflight" in launcher
        and "run train --all" in launcher
        and "run evaluate" not in launcher
    )
