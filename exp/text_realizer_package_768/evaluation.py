"""Development-only diagnostics and a data-free, verified inference export."""

from __future__ import annotations

from pathlib import Path

import torch

from exp.text_realizer.campaign import make_cycle_encoder
from exp.text_realizer.evaluation import evaluate_model
from exp.text_realizer.protocol import (
    immutable_json,
    progress_callback,
    read_json,
    status,
)
from exp.text_realizer.report import examples
from src.artifact_io import copy_atomic, sha256_file, write_json
from src.text_realizer.artifacts import (
    load_text_realizer,
    read_checkpoint,
    verify_checkpoint,
)
from src.text_realizer.generation import generate
from src.text_realizer.metrics import paired_bootstrap

from .data import load_bundle, prepared_path
from .preflight import readiness
from .settings import digest, require_current, resolve, run_paths


def finalize(config: dict, run_id: str, *, device: str) -> dict:
    models, reports = run_paths(config, run_id)
    prepared = prepared_path(config)
    plan = require_current(reports, config, prepared)
    ready = readiness(config, reports, device=device)
    training = read_json(reports / "training.json")
    checkpoint = models / "best.pt"
    if not training["complete"] or training["checkpoint_sha256"] != sha256_file(
        checkpoint
    ):
        raise ValueError(
            "completed training and its unchanged best checkpoint are required"
        )
    bundle = load_bundle(prepared)
    verification = verify_checkpoint(
        checkpoint, device=device, cache_dir=resolve(config["paths"]["cache"])
    )
    write_json(reports / "reload.json", verification)
    model = load_text_realizer(
        checkpoint, device=device, cache_dir=resolve(config["paths"]["cache"])
    )
    encoder = make_cycle_encoder(config, device)
    group_path = prepared / "development_groups.json"
    if (
        sha256_file(group_path)
        != read_json(prepared / "manifest.json")["files"][group_path.name]
    ):
        raise ValueError("development groups changed")
    groups = read_json(group_path)
    result = evaluate_model(
        model,
        bundle["tune"],
        bundle["fit"],
        encoder=encoder,
        groups=groups["tune"],
        fit_groups=groups["fit"],
        controls=True,
        microbatch=ready["microbatch"],
        progress=progress_callback(reports, "development_diagnostics"),
    )
    correct = result["conditions"]["correct"]["rows"]
    retrieved = result["retrieval"]["rows"]
    differences = [
        a["character_similarity"] - b["character_similarity"]
        for a, b in zip(correct, retrieved, strict=True)
    ]
    comparison = paired_bootstrap(
        differences,
        samples=config["evaluation"]["bootstrap_samples"],
        seed=config["evaluation"]["bootstrap_seed"],
    )
    result.update(
        {
            "scope": "200 development CVs; selection-biased, not a held-out evaluation",
            "conditioning_eligible": result["conditioning"]["nll_gap"]
            >= config["selection"]["nll_gap"]
            and result["conditioning"]["character_gap"]
            >= config["selection"]["character_gap"],
            "retrieval_comparison": comparison,
            "development_retrieval_gain_passed": comparison["mean"]
            >= config["evaluation"]["minimum_retrieval_gain"]
            and comparison["lower_95"] > 0,
            "factual_check_scope": "country, degree, university tier, project count; motivation and behavioral episodes require human review",
        }
    )
    write_json(reports / "development.json", result)
    examples(
        reports / "development_examples.md",
        bundle["tune"],
        result,
        count=config["evaluation"]["examples"],
    )
    require_current(reports, config, prepared)
    portable = models / "inference"
    portable.mkdir(parents=True, exist_ok=True)
    target = portable / "model.pt"
    if target.exists() and sha256_file(target) != sha256_file(checkpoint):
        raise ValueError("portable export exists with different weights")
    if not target.exists():
        copy_atomic(checkpoint, target)
    metadata = read_checkpoint(target)["metadata"]
    frozen = {
        "format_version": 1,
        "checkpoint_sha256": sha256_file(target),
        "plan_sha256": digest(plan),
        "encoder": metadata["encoder"],
        "spec": metadata["spec"],
        "generation": metadata["generation"],
        "reload_verified": True,
        "evaluation_scope": "development-only",
        "conditioning_eligible": result["conditioning_eligible"],
        "used_counterfactual_text": False,
        "used_counterfactual_latent": False,
        "used_evaluation_content": False,
        "prepared_manifest_sha256": plan["identity"]["prepared_manifest_sha256"],
        "reports_sha256": {
            name: sha256_file(reports / name)
            for name in (
                "plan.json",
                "training.json",
                "development.json",
                "reload.json",
            )
        },
        "note": "Portable projector+LoRA weights; pinned cached base/tokenizer still required. Diagnostic success is not evidence of faithful counterfactual reconstruction.",
    }
    immutable_json(portable / "manifest.json", frozen)
    immutable_json(reports / "frozen.json", frozen)
    status(
        reports,
        "complete",
        checkpoint=str(target),
        conditioning_eligible=result["conditioning_eligible"],
    )
    return frozen


def read_vectors(path: Path, encoder: dict) -> tuple[list[int], torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {
        "ids",
        "z",
        "encoder",
        "provenance",
    }:
        raise ValueError("input must contain exactly ids, z, encoder and provenance")
    ids, z = payload["ids"], payload["z"]
    if (
        not isinstance(ids, list)
        or not ids
        or any(type(key) is not int for key in ids)
        or len(ids) != len(set(ids))
        or payload["encoder"] != encoder
        or not isinstance(payload["provenance"], dict)
        or not payload["provenance"]
        or not isinstance(z, torch.Tensor)
        or not z.is_floating_point()
        or z.shape != (len(ids), 768)
        or not torch.isfinite(z).all()
        or not torch.isfinite(z.float()).all()
    ):
        raise ValueError("invalid or incompatible ID-aligned 768-D input")
    # Edited vectors are deliberately not projected, normalized or otherwise altered.
    return ids, z.float(), payload["provenance"]


def generate_vectors(
    checkpoint: Path, vectors: Path, output: Path, *, device: str
) -> None:
    if output.exists():
        raise FileExistsError(output)
    metadata = read_checkpoint(checkpoint)["metadata"]
    if metadata["spec"]["latent_dim"] != 768:
        raise ValueError("use a package 768-D checkpoint")
    ids, z, provenance = read_vectors(vectors, metadata["encoder"])
    model = load_text_realizer(checkpoint, device=device)
    rows = generate(model, z)
    write_json(
        output,
        {
            "checkpoint_sha256": sha256_file(checkpoint),
            "vectors_sha256": sha256_file(vectors),
            "input_provenance": provenance,
            "rows": [{"id": key, **row} for key, row in zip(ids, rows, strict=True)],
        },
    )
