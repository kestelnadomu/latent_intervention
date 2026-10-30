"""Export the provided package fits, never call cache builders or train a model."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.artifact_io import atomic_output, copy_atomic, save_torch, write_json
from src.text_realizer.artifacts import read_checkpoint
from src.encoder.protocols import embeddinggemma_protocol

from .common import (
    checked_file,
    checked_source,
    csv_records,
    digest,
    file_record,
    immutable_json,
    load_manifest,
    lock,
    read_json,
    source_hashes,
    versions,
)


def select_units(
    partition: dict, pairs: dict, references: dict, count: int
) -> list[int]:
    # Match the notebook's literal partition order, including the SCM random draws.
    ids = [key for key, row in partition.items() if row["partition"] == "eval"]
    official = {key for key, row in pairs.items() if row["split"] == "test"}
    if len(ids) != count or set(ids) != official or set(ids) != set(references):
        raise ValueError("evaluation IDs, official test and oracle text IDs must match")
    if any(not references[key]["text"].strip() for key in ids):
        raise ValueError("empty oracle reference")
    return ids


def aligned_vectors(
    payload: dict, ids: list[int], dim: int, *, oracle: bool
) -> torch.Tensor:
    keys = [int(key) for key in payload["test_ids" if oracle else "ids"]]
    values = payload["z_prime" if oracle else "z"]
    if len(keys) != len(set(keys)) or values.shape != (len(keys), 768):
        raise ValueError("invalid canonical latent IDs or shape")
    positions = {key: i for i, key in enumerate(keys)}
    if not set(ids).issubset(positions):
        raise ValueError("missing canonical latent IDs")
    z = values[[positions[key] for key in ids]].float()
    if not torch.isfinite(z).all() or not torch.allclose(
        z.norm(dim=1), torch.ones(len(z)), atol=2e-6, rtol=2e-6
    ):
        raise ValueError("canonical vectors must be finite and unit norm")
    if dim == 256:
        if (z[:, :256].norm(dim=1) <= 1e-12).any():
            raise ValueError("degenerate canonical 256-D prefix")
        return F.normalize(z[:, :256], dim=1)
    if dim != 768:
        raise ValueError("unsupported latent dimension")
    return z


def check_training_record(actual: dict, expected: dict) -> None:
    for key in ("history", "best_epoch", "epochs_run", "max_epochs"):
        if actual.get(key) != expected.get(key):
            raise ValueError(f"checkpoint does not match its package report: {key}")


def predict_vectors(
    config: dict,
    pipeline: dict,
    dim: int,
    z: torch.Tensor,
    do_values: list[int],
    kind: str = "flow",
    target: pd.DataFrame | None = None,
) -> tuple[torch.Tensor, dict]:
    """The edits of the renderer ``pipeline[kind]`` (S1, trained on psi(z)); with
    ``target`` (one row per unit), conditioned on these target states instead of the
    operator's draw."""
    from latent_lab import LinearSCM, Pipeline, mean_predict

    report = read_json(checked_source(pipeline["report"]))
    # These are the trusted repository's pinned Python-object caches, not arbitrary downloads.
    _, readers = torch.load(
        checked_source(pipeline["reader"]), map_location="cpu", weights_only=False
    )
    flow = torch.load(
        checked_source(pipeline[kind]), map_location="cpu", weights_only=False
    )
    if len(readers) != 5 or (
        kind == "flow" and flow.flow.state_dict()["mean"].numel() != dim
    ):
        raise ValueError("wrong fold count or flow dimension")
    records = [
        r
        for r in report["training"]
        if r["model"] == "psi"
        and r["seed"] == config["seed"]
        and r.get("encoder", f"embeddinggemma-{dim}") == f"embeddinggemma-{dim}"
    ]
    records.sort(key=lambda r: r["fold"])
    if len(records) != len(readers):
        raise ValueError("missing matching reader training records")
    for reader, record in zip(readers, records, strict=True):
        if reader.model.trunk[0].in_features != dim:
            raise ValueError("reader dimension mismatch")
        check_training_record(reader.fit_log, record)
        reader.model.eval()
    record = [
        r
        for r in report["training"]
        if r["model"] == f"rho-{kind}"
        and r["S"] == "S1"
        and r["train"] == "ψ(z)"
        and r["seed"] == config["seed"]
    ]
    if len(record) != 1:
        raise ValueError(f"missing matching {kind} training record")
    check_training_record(flow.fit_log, record[0])
    scm_config = config["scm"]
    if flow.columns != list(scm_config["levels"]):
        raise ValueError("flow/SCM state columns differ")
    states = mean_predict(readers, z)
    if target is None:
        scm = LinearSCM(**scm_config, seed=config["seed"])
        operator = scm.do(X=lambda _: np.asarray(do_values))
    else:

        def operator(s: pd.DataFrame) -> pd.DataFrame:
            return target[s.columns].reset_index(drop=True)

    result = Pipeline(None, operator, flow)(z, states)
    predicted = result.z_prime.detach().cpu().float()
    if predicted.shape != z.shape or not torch.isfinite(predicted).all():
        raise ValueError("invalid predicted latent vectors")
    return predicted, {
        "source": result.s.to_dict(orient="records"),
        "target": result.s_prime.to_dict(orient="records"),
        f"{kind}_settings": getattr(flow, "settings", None),
    }


def replay(
    config: dict,
    pipeline: dict,
    dim: int,
    payload: dict,
    ids: list[int],
    do_values: list[int],
    kind: str = "flow",
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Oracle and predicted vectors of the evaluation units, checked against the package."""
    z = aligned_vectors(payload, ids, dim, oracle=False)
    oracle = aligned_vectors(payload, ids, dim, oracle=True)
    predicted, states = predict_vectors(config, pipeline, dim, z, do_values, kind)
    # Check the replay against the already-published scalar result, without
    # choosing a checkpoint or seed from the reconstructed text.
    report = read_json(checked_source(pipeline["report"]))
    expected = [
        row
        for row in report["results"]
        if row["S"] == "S1"
        and row["rho"] == kind
        and row["train"] == "ψ(z)"
        and row["test"] == "ψ(z)"
        and row["seed"] == config["seed"]
    ]
    if len(expected) != 1:
        raise ValueError("missing package replay reference")
    measured = {
        "cosine": float(F.cosine_similarity(predicted, oracle).mean()),
        "distance": float(
            ((predicted - oracle) / z.std(0).clamp(min=1e-6)).norm(dim=1).mean()
            / math.sqrt(dim)
        ),
    }
    for metric, value in measured.items():
        if not math.isclose(value, expected[0][metric], abs_tol=2e-5, rel_tol=2e-5):
            raise ValueError(
                f"{dim}-D package replay differs on {metric}: {value} vs {expected[0][metric]}"
            )
    states["reproduced_package_metrics"] = measured
    return oracle, predicted, states


def prepare(config: dict, output: Path, *, allow_oracle: bool = False) -> dict:
    if not allow_oracle:
        raise PermissionError("official test export requires --allow-oracle-evaluation")
    torch.set_num_threads(1)
    provenance = {
        "config": config,
        "source_sha256": source_hashes(),
        "versions": versions(("torch", "numpy", "pandas", "scipy")),
    }
    with lock(output / ".prepare.lock"):
        if (output / "manifest.json").exists():
            saved = load_manifest(output)
            if saved["provenance"] != provenance:
                raise ValueError(
                    "bundle recipe/source changed; choose a new output directory"
                )
            for item in saved["files"].values():
                checked_file(output, item)
            return saved
        paths = {name: checked_source(item) for name, item in config["inputs"].items()}
        info, canonical = read_json(paths["info"]), read_json(paths["manifest"])
        if (
            info["artifact_sha256"] != config["inputs"]["latents"]["sha256"]
            or info["input_sha256"]["counterfactual_text"]
            != config["inputs"]["counterfactual"]["sha256"]
            or info["input_sha256"]["factual_text"]
            != config["inputs"]["factual"]["sha256"]
            or not canonical["complete"]
        ):
            raise ValueError("canonical latent/text provenance mismatch")
        factual, reference = csv_records(paths["factual"]), csv_records(
            paths["counterfactual"]
        )
        ids = select_units(
            csv_records(paths["partition"]),
            csv_records(paths["pairs"]),
            reference,
            config["unit_count"],
        )
        interventions = csv_records(paths["interventions"])
        if not set(ids).issubset(factual) or not set(ids).issubset(interventions):
            raise ValueError("missing source CV or intervention")
        do_values = [int(interventions[key]["do_value"]) for key in ids]
        payload = torch.load(paths["latents"], map_location="cpu", weights_only=True)
        if set(int(key) for key in payload["test_ids"]) != set(ids):
            raise ValueError("oracle latent IDs differ from official test IDs")
        files, models, encoders = {}, {}, {}
        for key, spec in config["decoders"].items():
            checkpoint = checked_source(spec["checkpoint"])
            frozen = read_json(checked_source(spec["manifest"]))
            metadata = read_checkpoint(checkpoint)["metadata"]
            dim = spec["latent_dim"]
            if (
                not frozen.get("reload_verified")
                or frozen["checkpoint_sha256"] != spec["checkpoint"]["sha256"]
                or metadata["spec"]["latent_dim"] != dim
            ):
                raise ValueError("requires the verified, dimension-matched decoder")
            if (
                frozen.get("used_counterfactual_text")
                or frozen.get("used_counterfactual_latent")
                or frozen.get("used_evaluation_content")
            ):
                raise ValueError(
                    "requires factual-only decoders with unopened test data"
                )
            encoder = metadata["encoder"]
            if (
                encoder["embeddinggemma_model_revision"] != info["model_revision"]
                or encoder["embeddinggemma_model_name"] != info["encoder"]
                or encoder["embeddinggemma_latent_dim"] != dim
            ):
                raise ValueError("decoder/canonical encoder mismatch")
            if (
                embeddinggemma_protocol({**encoder, "embeddinggemma_latent_dim": 768})
                != info["embeddinggemma_protocol"]
            ):
                raise ValueError("decoder/canonical embedding protocols differ")
            if any(
                frozen[key] != metadata[key]
                for key in ("encoder", "spec", "generation")
            ):
                raise ValueError("decoder metadata differs from its frozen manifest")
            if str(dim) in encoders and encoders[str(dim)] != encoder:
                raise ValueError("decoder encoder protocols differ")
            encoders[str(dim)] = encoder
            destination = output / "checkpoints" / f"{key}.pt"
            copy_atomic(checkpoint, destination)
            files[key] = file_record(output, destination)
            models[key] = {
                **files[key],
                "latent_dim": dim,
                "encoder": encoder,
                "spec": metadata["spec"],
                "generation": metadata["generation"],
            }
        for width, pipeline in config["pipelines"].items():
            dim = int(width)
            oracle, predicted, states = replay(
                config, pipeline, dim, payload, ids, do_values
            )
            path = output / f"vectors-{dim}.pt"
            save_torch(
                path,
                {
                    "ids": ids,
                    "oracle": oracle,
                    "predicted": predicted,
                    "encoder": encoders[width],
                },
            )
            files[f"vectors-{dim}"] = file_record(output, path)
            path = output / f"states-{dim}.json"
            write_json(path, {"ids": ids, "do_value": do_values, **states})
            files[f"states-{dim}"] = file_record(output, path)
        path = output / "references.jsonl"
        with atomic_output(path) as temporary:
            with temporary.open("w", encoding="utf-8") as stream:
                for key, value in zip(ids, do_values, strict=True):
                    stream.write(
                        json.dumps(
                            {
                                "id": key,
                                "do_value": value,
                                "original": factual[key]["text"],
                                "reference": reference[key]["text"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        files["references"] = file_record(output, path)
        manifest = {
            "format_version": 1,
            "complete": True,
            "ids": ids,
            "models": models,
            "files": files,
            "provenance": provenance,
            "bundle_id": digest(provenance),
            "scope": "official test: oracle inversion and predicted-vector end-to-end reconstruction; provided 256-D and 768-D pipelines have different training protocols",
        }
        immutable_json(output / "manifest.json", manifest)
        return manifest
