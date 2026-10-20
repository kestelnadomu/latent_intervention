import json
from pathlib import Path

import pytest
import torch

from exp.text_realizer import experiment
from exp.text_realizer.data import load_bundle, prepare
from exp.text_realizer.protocol import oracle_gate
from exp.text_realizer.settings import run_paths
from src.artifact_io import sha256_file
from src.text_realizer.artifacts import read_checkpoint


def test_full_fit_resume_and_changed_config_refusal(
    factual_config, tiny_factory, monkeypatch
):
    from exp.text_realizer import preflight
    from src.text_realizer.model import TextRealizer

    prepare(factual_config)
    factual_config["preflight"]["smoke_updates"] = 2
    factual_config["training"]["checkpoint_every"] = 1
    monkeypatch.setattr(
        preflight,
        "token_audit",
        lambda *args: {
            "hidden_size": 16,
            "generation": {"max_new_tokens": 12, "batch_size": 1},
        },
    )

    def tiny(spec, generation, **kwargs):
        base = tiny_factory(lora=spec["use_lora"])
        return TextRealizer(base.language_model, base.tokenizer, spec, generation)

    monkeypatch.setattr(experiment, "build_model", tiny)
    # Training must work from the derived bundle even without access to originals.
    for name in ("factual", "pairs", "latents", "info", "split_plan"):
        Path(factual_config["inputs"][name]).rename(
            Path(factual_config["inputs"][name]).with_suffix(".sealed")
        )
    uninterrupted = experiment.train(
        factual_config, "continuous", "lora-compact", 42, device="cpu", smoke=True
    )
    assert uninterrupted["epochs"] == 2
    original_update = experiment.update
    count = 0

    def interrupt(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise KeyboardInterrupt("injected test interruption")
        return original_update(*args, **kwargs)

    monkeypatch.setattr(experiment, "update", interrupt)
    with pytest.raises(KeyboardInterrupt):
        experiment.train(
            factual_config, "resumed", "lora-compact", 42, device="cpu", smoke=True
        )
    monkeypatch.setattr(experiment, "update", original_update)
    resumed = experiment.train(
        factual_config,
        "resumed",
        "lora-compact",
        42,
        device="cpu",
        smoke=True,
        resume=True,
    )
    assert resumed["best_tune_nll"] == uninterrupted["best_tune_nll"]
    a = read_checkpoint(run_paths(factual_config, "continuous")[0] / "best.pt")[
        "weights"
    ]
    b = read_checkpoint(run_paths(factual_config, "resumed")[0] / "best.pt")["weights"]
    assert all(torch.equal(a[name], b[name]) for name in a)
    # An interrupted best/last replacement is repaired from the committed last.
    best = run_paths(factual_config, "resumed")[0] / "best.pt"
    best.write_bytes(b"injected interrupted replacement")
    experiment.train(
        factual_config,
        "resumed",
        "lora-compact",
        42,
        device="cpu",
        smoke=True,
        resume=True,
    )
    repaired = read_checkpoint(best)["weights"]
    assert all(torch.equal(a[name], repaired[name]) for name in a)
    with pytest.raises(ValueError, match="run exists"):
        experiment.train(
            factual_config, "resumed", "lora-compact", 42, device="cpu", smoke=True
        )
    factual_config["training"]["projector_lr"] *= 2
    with pytest.raises(ValueError, match="immutable"):
        experiment.train(
            factual_config,
            "resumed",
            "lora-compact",
            42,
            device="cpu",
            smoke=True,
            resume=True,
        )


def test_audit_requires_matching_freeze_and_oracle_requires_completed_audit(
    factual_config, tmp_path
):
    prepared = prepare(factual_config)
    reports = tmp_path / "frozen-run"
    reports.mkdir()
    frozen = {
        "used_counterfactual_text": False,
        "used_counterfactual_latent": False,
        "reports_sha256": {},
        "reload_verified": True,
        "prepared_manifest_sha256": sha256_file(prepared / "manifest.json"),
    }
    (reports / "frozen.json").write_text(json.dumps(frozen))
    assert (
        len(load_bundle(prepared, audit=True, frozen_root=reports)["audit"]["ids"]) == 2
    )
    with pytest.raises(FileNotFoundError):
        oracle_gate(reports, allowed=True)
    audit = {
        "complete": True,
        "frozen_sha256": sha256_file(reports / "frozen.json"),
        "seed_report_sha256": {},
    }
    (reports / "factual_audit.json").write_text(json.dumps(audit))
    assert oracle_gate(reports, allowed=True) == frozen
    audit["frozen_sha256"] = "changed"
    (reports / "factual_audit.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="audit"):
        oracle_gate(reports, allowed=True)


def test_oracle_evaluator_cannot_open_data_before_opt_in(factual_config, monkeypatch):
    import exp.text_realizer.oracle as oracle

    def forbidden(*args, **kwargs):
        raise AssertionError("oracle input was opened before the gate")

    monkeypatch.setattr(oracle, "csv_rows", forbidden)
    monkeypatch.setattr(oracle.torch, "load", forbidden)
    with pytest.raises(PermissionError):
        oracle.evaluate_oracle(
            factual_config, "not-frozen", allowed=False, device="cpu"
        )


def test_no_canonical_or_oracle_artifact_accepted_for_generic_generation(tmp_path):
    from exp.text_realizer.inference import read_latents

    path = tmp_path / "vectors.pt"
    torch.save(
        {"ids": [1], "z": torch.zeros(1, 768), "z_prime": torch.zeros(1, 768)}, path
    )
    with pytest.raises(ValueError, match="paired artifacts forbidden"):
        read_latents(path)


def test_freeze_reload_three_seed_audit_and_report_tamper_gate(
    factual_config, tiny_factory, monkeypatch
):
    from types import SimpleNamespace
    from exp.text_realizer import campaign, setup
    from exp.text_realizer.protocol import identity, snapshot
    from exp.text_realizer.settings import VARIANTS
    from src.artifact_io import write_json
    from src.text_realizer import artifacts

    pytest.importorskip("rapidfuzz")
    pytest.importorskip("sacrebleu")
    prepared = prepare(factual_config)
    _, reports = run_paths(factual_config, "tiny-campaign")
    snapshot(reports, {"identity": identity(factual_config, prepared)})
    variant = "prefix-compact"
    monkeypatch.setattr(setup, "model_files", lambda config: {})
    monkeypatch.setattr(artifacts, "build_model", lambda *a, **k: tiny_factory())

    class TinyEncoder:
        def encode(self, texts, **kwargs):
            return torch.ones(len(texts), 768) / 768**0.5

    monkeypatch.setattr(campaign, "make_cycle_encoder", lambda *args: TinyEncoder())

    def verify_child(command, **kwargs):
        checkpoint = Path(command[command.index("--checkpoint") + 1])
        output = Path(command[command.index("--output") + 1])
        write_json(output, artifacts.verify_checkpoint(checkpoint))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(campaign.subprocess, "run", verify_child)
    candidates = {}
    for name in VARIANTS:
        result = {
            "conditioning": {"nll_gap": 0.1, "character_gap": 0.03},
            "conditions": {
                "correct": {
                    "summary": {"character_similarity": 0.8 if name == variant else 0.5}
                }
            },
            "cycle": {"cosine_invalid_scored_minus_one": 0.7},
            "checkpoint_sha256": "unselected-fixture",
        }
        candidates[name] = result
    for seed in (42, 43, 44):
        model = tiny_factory()
        paths = run_paths(
            factual_config, campaign.trial_id("tiny-campaign", variant, seed)
        )
        source_models, source_reports = paths
        snapshot(
            source_reports,
            {
                "identity": identity(factual_config, prepared),
                "variant": variant,
                "seed": seed,
                "model": model.spec,
                "generation": model.generation,
            },
        )
        artifacts.save_checkpoint(
            source_models / "best.pt",
            model,
            encoder=factual_config["encoder"],
            provenance={"stage": "train", "seed": seed, "base_files_sha256": {}},
        )
        result = {
            **candidates[variant],
            "checkpoint_sha256": sha256_file(source_models / "best.pt"),
        }
        write_json(source_reports / "factual_tune.json", result)
        if seed == 42:
            candidates[variant] = result
    for name, result in candidates.items():
        source_reports = run_paths(
            factual_config, campaign.trial_id("tiny-campaign", name, 42)
        )[1]
        write_json(source_reports / "factual_tune.json", result)
    selection = campaign.choose(candidates, factual_config["selection"])
    write_json(reports / "selection.json", selection)
    frozen = campaign.freeze(factual_config, "tiny-campaign", device="cpu")
    assert frozen["deployment_seed"] == 42 and frozen["reload_verified"]
    with pytest.raises(FileNotFoundError):
        oracle_gate(reports, allowed=True)
    result = campaign.audit(factual_config, "tiny-campaign", device="cpu")
    assert result["complete"] and result["seeds"] == [42, 43, 44]
    assert campaign.audit(factual_config, "tiny-campaign", device="cpu") == result
    assert oracle_gate(reports, allowed=True) == frozen
    with (reports / "audit-seed-42.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="audit report changed"):
        campaign.audit(factual_config, "tiny-campaign", device="cpu")
    with pytest.raises(ValueError, match="audit report changed"):
        oracle_gate(reports, allowed=True)
