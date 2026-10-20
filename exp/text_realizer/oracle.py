"""Explicit post-freeze test evaluation. Never imported by the training workflow."""

from __future__ import annotations

import torch

from src.artifact_io import sha256_file, write_json
from src.text_realizer.artifacts import load_text_realizer
from src.text_realizer.metrics import aggregate

from .campaign import frozen_checkpoints, make_cycle_encoder
from .data import csv_rows, load_bundle, prepared_path
from .evaluation import evaluate_model
from .inference import read_latents
from .protocol import (
    immutable_json,
    lock,
    oracle_gate,
    read_json,
    require_current,
    progress_callback,
)
from .settings import ROOT, digest, resolve, run_paths


def evaluate_oracle(
    config: dict, run_id: str, *, allowed: bool, device: str, edited=None
) -> dict:
    _, reports = run_paths(config, run_id)
    prepared = prepared_path(config)
    # These checks precede even resolving/opening the counterfactual data.
    oracle_gate(reports, allowed=allowed)
    require_current(reports, config, prepared)
    frozen, checkpoints = frozen_checkpoints(config, run_id)
    manifest = read_json(prepared / "manifest.json")
    test_ids = manifest["split_ids"]["official_test"]
    expected = set(test_ids)
    counterfactual = ROOT / "data/text_talent/cv_counterfactual.csv"
    canonical = resolve(config["inputs"]["latents"])
    info_path = resolve(config["inputs"]["info"])
    if sha256_file(info_path) != manifest["input_sha256"]["info"]:
        raise ValueError("canonical provenance changed")
    info = read_json(info_path)
    for name in ("latents", "factual", "pairs"):
        if (
            sha256_file(resolve(config["inputs"][name]))
            != manifest["input_sha256"][name]
        ):
            raise ValueError(f"final-evaluation input changed: {name}")
    if sha256_file(counterfactual) != info["input_sha256"]["counterfactual_text"]:
        raise ValueError("counterfactual text hash mismatch")
    declaration = {
        "frozen_sha256": sha256_file(reports / "frozen.json"),
        "counterfactual_sha256": sha256_file(counterfactual),
        "edited_sha256": sha256_file(edited) if edited else None,
    }
    output_root = reports / "oracle" / digest(declaration)
    with lock(output_root / ".oracle.lock"):
        immutable_json(output_root / "plan.json", declaration)
        if (output_root / "evaluation.json").exists():
            return read_json(output_root / "evaluation.json")
        raw = torch.load(canonical, map_location="cpu", weights_only=True)
        ids = [int(value) for value in raw["ids"]]
        oracle_ids = [int(value) for value in raw["test_ids"]]
        if set(oracle_ids) != expected or len(oracle_ids) != len(expected):
            raise ValueError("canonical oracle IDs differ from sealed test")
        index, oracle_index = {key: i for i, key in enumerate(ids)}, {
            key: i for i, key in enumerate(oracle_ids)
        }
        z = raw["z"][[index[key] for key in test_ids]].clone()
        z_prime = raw["z_prime"][[oracle_index[key] for key in test_ids]].clone()
        if (
            z.shape != (len(test_ids), 768)
            or z_prime.shape != z.shape
            or not torch.isfinite(z_prime).all()
        ):
            raise ValueError("invalid official-test latent tensors")
        factual_rows = csv_rows(
            resolve(config["inputs"]["factual"]), allowed_ids=expected
        )
        oracle_rows = csv_rows(counterfactual, allowed_ids=expected)
        pairs = csv_rows(resolve(config["inputs"]["pairs"]), allowed_ids=expected)
        identity = [
            key
            for key in test_ids
            if pairs[key]["is_identity"].lower() in {"true", "1"}
        ]
        for key in identity:
            position = test_ids.index(key)
            if not torch.equal(z[position], z_prime[position]):
                raise ValueError("identity vectors differ")
            if factual_rows[key]["text"] != oracle_rows[key]["text"]:
                raise ValueError("identity reference texts differ")
        variants = {
            "factual": {
                "ids": test_ids,
                "z": z,
                "texts": [factual_rows[key]["text"] for key in test_ids],
            },
            "oracle": {
                "ids": test_ids,
                "z": z_prime,
                "texts": [oracle_rows[key]["text"] for key in test_ids],
            },
        }
        if edited:
            edited_ids, edited_z, payload = read_latents(edited, expected_ids=test_ids)
            if payload["encoder"] != config["encoder"]:
                raise ValueError("edited vectors have incompatible encoder provenance")
            mapping = {key: i for i, key in enumerate(edited_ids)}
            variants["learned_intervention"] = {
                **variants["oracle"],
                "z": edited_z[[mapping[key] for key in test_ids]],
            }
        model = load_text_realizer(
            checkpoints[42],
            device=device,
            cache_dir=str(resolve(config["paths"]["cache"])),
        )
        encoder, fit = make_cycle_encoder(config, device), load_bundle(prepared)["fit"]
        results = {}
        for name, part in variants.items():
            results[name] = evaluate_model(
                model,
                part,
                fit,
                encoder=encoder,
                controls=False,
                progress=progress_callback(
                    output_root, "oracle_evaluation", variant=name
                ),
            )
            rows = results[name]["conditions"]["correct"]["rows"]
            results[name]["identity_strata"] = {
                "identity": aggregate([row for row in rows if row["id"] in identity]),
                "changed": aggregate(
                    [row for row in rows if row["id"] not in identity]
                ),
            }
        before = {
            row["id"]: row["token_ids"]
            for row in results["factual"]["conditions"]["correct"]["rows"]
        }
        after = {
            row["id"]: row["token_ids"]
            for row in results["oracle"]["conditions"]["correct"]["rows"]
        }
        if any(before[key] != after[key] for key in identity):
            raise ValueError(
                "identical input vectors produced different deterministic outputs"
            )
        if edited:
            results["learned_intervention"]["input_to_oracle_mean_l2"] = float(
                (variants["learned_intervention"]["z"] - z_prime).norm(dim=1).mean()
            )
        output = {
            "complete": True,
            "declaration": declaration,
            "results": results,
            "identity_count": len(identity),
            "changed_count": len(test_ids) - len(identity),
            "oracle_gate": "opened-after-freeze",
            "decoder_updates_allowed": False,
        }
        write_json(output_root / "evaluation.json", output)
        return output
