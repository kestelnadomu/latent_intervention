"""Offline checks for the bounded 1B pilot and its sealed data boundaries."""

import copy
from pathlib import Path

import pytest
import torch
import yaml

from exp.text_realizer.data import load_bundle, prepare, prepared_path
from exp.text_realizer.pilot_protocol import (
    baseline_material,
    compare_tune,
    prepare_development,
)
from exp.text_realizer.protocol import identity, read_json
from exp.text_realizer.settings import (
    PILOT_CONFIG,
    read_config,
    require_command,
    run_paths,
)
from src.artifact_io import sha256_file, write_json


def result_for(ids, *, character=0.33, accuracy=0.8):
    fields = ["country", "degree", "university_tier", "project_count"]
    return {
        "complete": True,
        "conditions": {
            "correct": {
                "rows": [
                    {"id": key, "text": f"CV {key}", "character_similarity": character}
                    for key in ids
                ]
            }
        },
        "conditioning": {"nll_gap": 0.1, "character_gap": 0.03},
        "facts": {
            "reference": [{"country": "ghana"} for _ in ids],
            "fields": {
                name: {
                    "reference_scorable": len(ids),
                    "total": len(ids),
                    "accuracy_on_scorable_references": accuracy,
                }
                for name in fields
            },
        },
    }


@pytest.fixture
def pilot_case(factual_config, tmp_path):
    source = prepare(factual_config)
    baseline = tmp_path / "baseline_report"
    ids = load_bundle(source)["tune"]["ids"]
    write_json(
        baseline / "plan.json",
        {
            "identity": identity(factual_config, source),
            "variant": "lora-compact",
            "seed": 42,
            "stage": "train",
        },
    )
    write_json(baseline / "factual_tune.json", result_for(ids))
    template = read_config(PILOT_CONFIG)
    config = copy.deepcopy(factual_config)
    for key in ("model", "selection", "preflight", "pilot"):
        config[key] = copy.deepcopy(template[key])
    config["paths"]["prepared"] = str(tmp_path / "pilot_prepared")
    config["pilot"].update(
        baseline_prepared=str(source),
        baseline_report=str(baseline),
        baseline_sha256={
            "manifest": sha256_file(source / "manifest.json"),
            "plan": sha256_file(baseline / "plan.json"),
            "factual_tune": sha256_file(baseline / "factual_tune.json"),
        },
    )
    return config


def test_default_stays_270m_and_pilot_is_one_seed():
    original, pilot = read_config(), read_config(PILOT_CONFIG)
    assert original["model"]["name"] == "google/gemma-3-270m"
    assert original["selection"]["seeds"] == [42, 43, 44]
    assert pilot["selection"]["seeds"] == [42]
    assert pilot["model"]["attention_implementation"] == "eager"
    assert original["model"].get("attention_implementation", "sdpa") == "sdpa"
    for key in ("inputs", "encoder", "training", "split", "generation"):
        assert original[key] == pilot[key]
    assert prepared_path(original) != prepared_path(pilot)
    for command in ("prepare", "campaign", "freeze", "audit", "oracle-evaluate"):
        with pytest.raises(ValueError, match="closed"):
            require_command(pilot, command)
    for command in ("pilot", "preflight", "train"):
        require_command(pilot, command)
    with pytest.raises(ValueError, match="separate"):
        require_command(original, "pilot")


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "0" * 40),
        ("name", "unapproved/model"),
        ("attention_implementation", "sdpa"),
    ],
)
def test_pilot_backbone_must_be_pinned(pilot_case, tmp_path, field, value):
    pilot_case["model"][field] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(pilot_case))
    with pytest.raises(ValueError):
        read_config(path)


def test_prepare_only_copies_development_never_reads_oracle_or_audit(
    pilot_case, monkeypatch
):
    blocked = {
        Path(pilot_case["inputs"][key])
        for key in ("latents", "factual", "pairs", "info", "split_plan")
    }
    original_open = Path.open

    def guarded(path, *args, **kwargs):
        if path in blocked or path.name in {"audit.pt", "audit_groups.json"}:
            raise AssertionError(f"forbidden input: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    target = prepare_development(pilot_case)
    assert not (target / "audit.pt").exists()
    assert not (target / "audit_groups.json").exists()
    original = load_bundle(Path(pilot_case["pilot"]["baseline_prepared"]))
    copied = load_bundle(target)
    for name in ("fit", "tune"):
        assert copied[name]["ids"] == original[name]["ids"]
        assert copied[name]["texts"] == original[name]["texts"]
        assert torch.equal(copied[name]["z"], original[name]["z"])
    assert prepare_development(pilot_case) == target


def test_baseline_tamper_and_changed_comparison_refused(pilot_case):
    changed = copy.deepcopy(pilot_case)
    changed["training"]["lora_lr"] *= 2
    with pytest.raises(ValueError, match="comparison"):
        baseline_material(changed)
    changed = copy.deepcopy(pilot_case)
    changed["model"]["lora_rank"] *= 2
    with pytest.raises(ValueError, match="comparison"):
        baseline_material(changed)
    report = Path(pilot_case["pilot"]["baseline_report"]) / "factual_tune.json"
    report.write_text(report.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_development(pilot_case)


def test_pilot_decision_requires_text_facts_and_conditioning(pilot_case):
    baseline = result_for([1, 2, 3])
    candidate = result_for([1, 2, 3], character=0.36, accuracy=0.85)
    assert compare_tune(candidate, baseline, pilot_case)["promising"]
    candidate["facts"]["fields"]["country"]["accuracy_on_scorable_references"] = 0.77
    decision = compare_tune(candidate, baseline, pilot_case)
    assert decision["comparison_scope"] == "historical_270m_not_pure_capacity_ablation"
    assert not decision["promising"]
    assert not decision["gates"]["no_material_field_regression"]
    assert decision["gates"]["macro_factual_gain"]
    assert not compare_tune(result_for([1, 2, 3], character=0.4), baseline, pilot_case)[
        "promising"
    ]
    candidate["conditioning"]["nll_gap"] = 0.01
    assert not compare_tune(candidate, baseline, pilot_case)["gates"][
        "conditioning_nll"
    ]
    with pytest.raises(ValueError, match="ordered IDs"):
        compare_tune(result_for([3, 2, 1]), baseline, pilot_case)
    candidate["conditions"]["correct"]["rows"][0]["character_similarity"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        compare_tune(candidate, baseline, pilot_case)


@pytest.mark.parametrize("backend", ["sdpa", "eager"])
def test_real_backbone_checks_work_with_tiny_offline_model(
    tiny_factory, numerical_settings, backend
):
    from exp.text_realizer.sanity import (
        check_gradients_and_frozen,
        check_native_forward,
        frozen_digest,
    )
    from src.text_realizer.training import optimizer_for, update

    model = tiny_factory(lora=True)
    model.language_model.set_attn_implementation(backend)
    z = torch.randn(1, 768)
    checks = check_native_forward(model, z, [4, 5, 2])
    assert checks["cache_test_sequence_length"] > checks["sliding_window"]
    original = frozen_digest(model)
    update(
        model,
        optimizer_for(model, numerical_settings),
        z,
        [[4, 5, 2]],
        [0],
        microbatch=1,
        clip=1,
    )
    assert check_gradients_and_frozen(model, original)["lora_gradient"]


@pytest.mark.parametrize("backend", [None, "eager"])
def test_loading_preserves_old_backend_and_honors_pilot_backend(
    tiny_factory, monkeypatch, backend
):
    import transformers

    from src.text_realizer import model as module

    tiny = tiny_factory()
    spec, received = dict(tiny.spec), {}
    if backend is not None:
        spec["attention_implementation"] = backend

    def base(*args, **kwargs):
        received.update(kwargs)
        return tiny.language_model

    monkeypatch.setattr(module, "cached_decoder", lambda *args: "offline")
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", base)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tiny.tokenizer
    )
    loaded = module.build_model(spec, tiny.generation)
    assert received["attn_implementation"] == (backend or "sdpa")
    assert loaded.spec == spec


def test_pilot_requires_opt_in_and_stops_after_one_tune_comparison(
    pilot_case, monkeypatch, tmp_path
):
    from exp.text_realizer import pilot as module

    config_path = tmp_path / "pilot.yaml"
    config_path.write_text(yaml.safe_dump(pilot_case))
    stages = []
    run_id = "tiny-pilot"
    models, reports = run_paths(pilot_case, run_id + ".lora-compact.seed-42")

    def child(command, root, phase, progress_root):
        stages.append(phase)
        assert not any(
            item in command
            for item in ("campaign", "audit", "oracle-evaluate", "freeze")
        )
        if phase == "preflight":
            prepared = prepared_path(pilot_case)
            write_json(
                prepared / "preflight.json",
                {
                    "ready": True,
                    "identity": identity(pilot_case, prepared),
                    "seconds_per_validation_cv": 0.1,
                    "seconds_per_optimizer_update": 1,
                    "seconds_per_generated_cv": 2,
                },
            )
        elif phase == "training_and_tune":
            models.mkdir(parents=True, exist_ok=True)
            (models / "best.pt").write_bytes(b"mock checkpoint verified by mock child")
            part = load_bundle(prepared_path(pilot_case))["tune"]
            result = result_for(part["ids"], character=0.36, accuracy=0.85)
            result["checkpoint_sha256"] = sha256_file(models / "best.pt")
            write_json(reports / "factual_tune.json", result)
        elif phase == "verifying":
            output = Path(command[command.index("--output") + 1])
            write_json(
                output,
                {
                    "verified": True,
                    "checkpoint_sha256": sha256_file(models / "best.pt"),
                },
            )

    monkeypatch.setattr(module, "_child", child)
    ready = module.pilot(pilot_case, config_path, run_id, device="cuda:0")
    assert ready["training_started"] is False and stages == ["preflight"]
    with pytest.raises(ValueError, match="run exists"):
        module.pilot(pilot_case, config_path, run_id, device="cuda:0")
    result = module.pilot(
        pilot_case,
        config_path,
        run_id,
        device="cuda:0",
        resume=True,
        train_after_preflight=True,
    )
    assert stages == ["preflight", "training_and_tune", "verifying"]
    assert result["promising"] and result["audit_informed"]
    assert not result["replication_authorized"]
    root = run_paths(pilot_case, run_id)[1]
    assert read_json(root / "status.json")["state"] == "complete"
    assert not (root / "frozen.json").exists()
    assert not (root / "factual_audit.json").exists()
