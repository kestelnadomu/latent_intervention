"""Development-only preparation and predeclared decisions for the 1B pilot."""

from __future__ import annotations

import math
import re
from pathlib import Path
from statistics import mean

from src.artifact_io import copy_atomic, sha256_file
from src.text_realizer.metrics import paired_bootstrap

from .data import load_bundle, prepared_path
from .protocol import lock, read_json
from .settings import PILOT_MODEL, PILOT_REVISION, ROOT, resolve


def validate_pilot_config(config: dict) -> None:
    pilot = config["pilot"]
    if (
        config["model"]["name"] != PILOT_MODEL
        or config["model"]["revision"] != PILOT_REVISION
        or config["model"].get("attention_implementation") != "eager"
        or pilot["variant"] != "lora-compact"
        or pilot["seed"] != 42
        or pilot["audit_informed"] is not True
        or config["preflight"].get("variant") != "lora-compact"
        or config["preflight"].get("microbatches") != [1]
        or config["preflight"].get("native_checks") is not True
    ):
        raise ValueError(
            "1B pilot requires pinned compact LoRA, eager attention, seed 42, microbatch 1"
        )
    if pilot["factual_fields"] != [
        "country",
        "degree",
        "university_tier",
        "project_count",
    ]:
        raise ValueError("pilot factual fields must not double-count region")
    for name in (
        "minimum_character_gain",
        "minimum_macro_factual_gain",
        "maximum_field_regression",
    ):
        value = pilot[name]
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value < 1
        ):
            raise ValueError(f"invalid pilot threshold: {name}")
    if set(pilot["baseline_sha256"]) != {"manifest", "plan", "factual_tune"} or any(
        not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in pilot["baseline_sha256"].values()
    ):
        raise ValueError("baseline manifest, plan and tune report hashes are required")
    if prepared_path(config).resolve() == resolve(pilot["baseline_prepared"]).resolve():
        raise ValueError("pilot preparation must not overwrite baseline diagnostics")


def baseline_material(config: dict) -> tuple[dict, dict]:
    """Read only the existing development manifest/plan/tune result, never audit."""
    pilot = config["pilot"]
    source = resolve(pilot["baseline_prepared"])
    report = resolve(pilot["baseline_report"])
    paths = {
        "manifest": source / "manifest.json",
        "plan": report / "plan.json",
        "factual_tune": report / "factual_tune.json",
    }
    for name, path in paths.items():
        if sha256_file(path) != pilot["baseline_sha256"][name]:
            raise ValueError(f"baseline hash mismatch: {name}")
    manifest, plan, result = (read_json(paths[key]) for key in paths)
    original = plan["identity"]["config"]
    if (
        plan["identity"]["prepared_manifest_sha256"]
        != pilot["baseline_sha256"]["manifest"]
        or plan["variant"] != "lora-compact"
        or plan["seed"] != 42
        or plan["stage"] != "train"
        or original["model"]["name"] != "google/gemma-3-270m"
        or any(
            original["model"][key] != config["model"][key]
            for key in (
                "latent_dim",
                "prefix_tokens",
                "input_scaling",
                "lora_rank",
                "lora_alpha",
                "lora_dropout",
                "lora_targets",
            )
        )
        or any(
            original[key] != config[key]
            for key in ("inputs", "encoder", "split", "training", "generation")
        )
        or manifest["encoder"] != config["encoder"]
        or any(
            manifest["input_sha256"][key] != value
            for key, value in config["inputs"]["sha256"].items()
        )
    ):
        raise ValueError("pilot no longer matches the 270M data/training comparison")
    # A paired comparison is not valid if the metrics or generation changed.
    for name in (
        "exp/text_realizer/evaluation.py",
        "exp/text_realizer/facts.py",
        "src/text_realizer/metrics.py",
        "src/text_realizer/generation.py",
        "exp/sim/codebook.yaml",
    ):
        if sha256_file(ROOT / name) != plan["identity"]["source_sha256"][name]:
            raise ValueError(f"baseline comparison implementation changed: {name}")
    ids = [row["id"] for row in result["conditions"]["correct"]["rows"]]
    if not result["complete"] or ids != manifest["split_ids"]["tune"]:
        raise ValueError(
            "baseline must be the complete, ID-aligned factual tune report"
        )
    return manifest, result


def prepare_development(config: dict) -> Path:
    """Copy hash-verified fit/tune artifacts; do not even open the audit bundle."""
    manifest, _ = baseline_material(config)
    source = resolve(config["pilot"]["baseline_prepared"])
    target = prepared_path(config)
    with lock(target / ".prepare.lock"):
        files = {
            name: manifest["files"][name]
            for name in ("development.pt", "development_groups.json")
        }
        files["manifest.json"] = config["pilot"]["baseline_sha256"]["manifest"]
        for name, expected in files.items():
            if sha256_file(source / name) != expected:
                raise ValueError(f"baseline development hash mismatch: {name}")
            if (target / name).exists():
                if sha256_file(target / name) != expected:
                    raise ValueError(f"pilot development hash mismatch: {name}")
            else:
                copy_atomic(source / name, target / name)
        load_bundle(target)
    return target


def compare_tune(candidate: dict, baseline: dict, config: dict) -> dict:
    """Operational pilot gate, not a fresh audit or a significance claim."""
    settings = config["pilot"]
    rows = candidate["conditions"]["correct"]["rows"]
    previous = baseline["conditions"]["correct"]["rows"]
    ids = [row["id"] for row in rows]
    if not ids or len(set(ids)) != len(ids) or ids != [row["id"] for row in previous]:
        raise ValueError("paired tune comparison requires identical ordered IDs")
    if candidate["facts"]["reference"] != baseline["facts"]["reference"]:
        raise ValueError("factual reference extraction changed")
    differences = [
        row["character_similarity"] - old["character_similarity"]
        for row, old in zip(rows, previous, strict=True)
    ]
    fields = {}
    for name in settings["factual_fields"]:
        current, old = (
            candidate["facts"]["fields"][name],
            baseline["facts"]["fields"][name],
        )
        if (
            current["reference_scorable"] != old["reference_scorable"]
            or current["total"] != old["total"]
            or not current["reference_scorable"]
        ):
            raise ValueError(f"factual comparison coverage changed: {name}")
        left, right = (
            current["accuracy_on_scorable_references"],
            old["accuracy_on_scorable_references"],
        )
        fields[name] = {"pilot": left, "baseline": right, "gain": left - right}
    gap = candidate["conditioning"]
    numeric = (
        differences
        + [item["gain"] for item in fields.values()]
        + [gap["nll_gap"], gap["character_gap"]]
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("nonfinite pilot comparison metric")
    intervals = paired_bootstrap(
        differences,
        samples=config["evaluation"]["bootstrap_samples"],
        seed=config["evaluation"]["bootstrap_seed"],
    )
    factual_gain = mean(item["gain"] for item in fields.values())
    gates = {
        "character_gain": intervals["mean"] >= settings["minimum_character_gain"],
        "macro_factual_gain": factual_gain >= settings["minimum_macro_factual_gain"],
        "no_material_field_regression": min(item["gain"] for item in fields.values())
        >= -settings["maximum_field_regression"],
        "conditioning_nll": gap["nll_gap"] >= config["selection"]["nll_gap"],
        "conditioning_character": gap["character_gap"]
        >= config["selection"]["character_gap"],
    }
    return {
        "complete": True,
        "promising": all(gates.values()),
        "gates": gates,
        "paired_character_gain": intervals,
        "macro_factual_gain": factual_gain,
        "factual_fields": fields,
        "conditioning": gap,
        "baseline_sha256": settings["baseline_sha256"],
        "audit_informed": True,
        "evaluation_partition": "tune",
        "audit_gate": "closed",
        "oracle_gate": "sealed",
        "replication_authorized": False,
        "comparison_scope": "historical_270m_not_pure_capacity_ablation",
        "baseline_numerics_caveat": (
            "The existing 270M checkpoint failed cached/full SDPA parity on this "
            "GPU/software stack; eager attention passed. The pilot uses eager. "
            "Improvement against historical results cannot be attributed solely "
            "to model size; the 270M baseline needs numerical reassessment."
        ),
        "note": "Tune is selection-biased; bootstrap intervals are descriptive. No confirmatory or counterfactual claim.",
    }
