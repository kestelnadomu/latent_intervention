"""Hash-bound, ID-filtered privileged pairs; training never receives eval data."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from exp.text_realizer.data import csv_rows, load_bundle
from src.artifact_io import save_torch, sha256_file, write_json

from .protocol import lock, read_json
from .settings import prepared_path, resolve


def verified_sources(config: dict) -> tuple[dict, dict, dict]:
    inputs = config["inputs"]
    paths = {
        "factual_manifest": resolve(inputs["factual_prepared"]) / "manifest.json",
        **{
            name: resolve(inputs[name])
            for name in (
                "oracle_vectors",
                "oracle_info",
                "counterfactual_text",
                "pair_index",
                "baseline_checkpoint",
            )
        },
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    if observed != inputs["sha256"]:
        raise ValueError(
            "source checksum mismatch; do not reuse this pilot configuration"
        )
    factual = read_json(paths["factual_manifest"])
    oracle = read_json(paths["oracle_info"])
    if (
        oracle.get("format_version") != 1
        or oracle.get("purpose") != "oracle_regression_only"
        or oracle.get("split") != "train"
        or oracle.get("artifact_sha256") != observed["oracle_vectors"]
        or oracle.get("input_sha256", {}).get("counterfactual_text")
        != observed["counterfactual_text"]
        or oracle.get("input_sha256", {}).get("pair_index") != observed["pair_index"]
        or oracle.get("input_sha256", {}).get("factual_text")
        != factual["input_sha256"]["factual"]
        or oracle.get("source_latent_sha256") != factual["input_sha256"]["latents"]
        or oracle.get("encoder", {}).get("model_revision")
        != factual["encoder"]["embeddinggemma_model_revision"]
        or oracle.get("encoder", {}).get("embeddinggemma_protocol")
        != factual["encoder_protocol"]
        or oracle.get("intervention") != {"X": 3}
    ):
        raise ValueError("oracle/factual provenance or intervention mismatch")
    from src.text_realizer.artifacts import read_checkpoint

    baseline = read_checkpoint(paths["baseline_checkpoint"])["metadata"]
    spec = baseline["spec"]
    expected_model = config["model"]
    if (
        spec["name"] != expected_model["name"]
        or spec["revision"] != expected_model["revision"]
        or spec["variant"] != expected_model["variant"]
        or spec["latent_dim"] != 768
        or spec["prefix_tokens"] != 16
        or spec["projector_hidden"] != 768
        or spec["lora_rank"] != 16
        or spec["lora_targets"] != expected_model["lora_targets"]
        or baseline["generation"] != config["generation"]
        or baseline["encoder"] != factual["encoder"]
        or baseline["provenance"].get("used_counterfactual_text") is not False
        or baseline["provenance"].get("used_counterfactual_latent") is not False
    ):
        raise ValueError("baseline is not the unchanged factual-only 270M decoder")
    return observed, factual, oracle


def _part(ids: list[int], z: torch.Tensor, texts: list[str]) -> dict:
    if z.shape != (len(ids), 768) or z.dtype != torch.float32 or len(texts) != len(ids):
        raise ValueError("misaligned privileged partition")
    if not torch.isfinite(z).all() or not torch.allclose(
        z.norm(dim=1), torch.ones(len(ids)), atol=2e-6, rtol=2e-6
    ):
        raise ValueError("nonfinite or non-unit privileged vectors")
    if any(not text.strip() for text in texts):
        raise ValueError("empty privileged CV")
    return {"ids": ids, "z": z.clone(), "texts": texts}


def assemble(
    split_ids: dict,
    factual_parts: dict,
    oracle_payload: dict,
    pair_rows: dict[int, dict],
    cf_rows: dict[int, dict],
) -> tuple[dict, dict, dict, dict]:
    """Pure join/partition function, shared by preparation and offline tests."""
    expected_train = set(split_ids["fit"] + split_ids["tune"] + split_ids["audit"])
    official_test = set(split_ids["official_test"])
    if (
        expected_train & official_test
        or set(pair_rows) != expected_train | official_test
    ):
        raise ValueError("official split or pair-index IDs disagree")
    if set(oracle_payload) != {"train_ids", "z_prime", "is_identity"}:
        raise ValueError("unexpected oracle payload keys")
    ids = [int(key) for key in oracle_payload["train_ids"]]
    z_prime, flags = oracle_payload["z_prime"], oracle_payload["is_identity"]
    if (
        len(ids) != 4000
        or len(ids) != len(set(ids))
        or set(ids) != expected_train
        or z_prime.shape != (len(ids), 768)
        or z_prime.dtype != torch.float32
        or flags.shape != (len(ids),)
        or flags.dtype != torch.bool
        or not torch.isfinite(z_prime).all()
        or not torch.allclose(
            z_prime.norm(dim=1), torch.ones(len(ids)), atol=2e-6, rtol=2e-6
        )
    ):
        raise ValueError("oracle tensor is malformed or leaks official-test IDs")
    positions = {key: i for i, key in enumerate(ids)}
    if set(cf_rows) != set(split_ids["fit"] + split_ids["tune"]):
        raise ValueError("counterfactual development CSV IDs disagree")
    for key, row in pair_rows.items():
        if row["split"] != ("test" if key in official_test else "train"):
            raise ValueError("pair-index official split mismatch")
        if row["is_identity"] not in {"True", "False"}:
            raise ValueError("invalid identity flag")
    for key in ids:
        if bool(flags[positions[key]]) != (pair_rows[key]["is_identity"] == "True"):
            raise ValueError(f"oracle identity flag differs for ID {key}")
    factual = {}
    for name in ("fit", "tune"):
        part = factual_parts[name]
        if part["ids"] != split_ids[name]:
            raise ValueError("factual development split changed")
        factual[name] = part

    def world(keys: list[int], base: dict) -> tuple[dict, torch.Tensor]:
        base_pos = {key: i for i, key in enumerate(base["ids"])}
        expected_texts = [base["texts"][base_pos[key]] for key in keys]
        texts = [cf_rows[key]["text"] for key in keys]
        vectors = z_prime[[positions[key] for key in keys]]
        identities = torch.tensor([bool(flags[positions[key]]) for key in keys])
        for key, identity, reference, changed, vector in zip(
            keys, identities.tolist(), expected_texts, texts, vectors, strict=True
        ):
            if identity:
                if changed != reference or not torch.equal(
                    vector, base["z"][base_pos[key]]
                ):
                    raise ValueError(f"identity text/vector mismatch for ID {key}")
            elif changed == reference:
                raise ValueError(f"changed counterfactual text equals factual ID {key}")
        return _part(keys, vectors, texts), identities

    fit_oracle, fit_identity = world(split_ids["fit"], factual["fit"])
    tune_oracle, tune_identity = world(split_ids["tune"], factual["tune"])
    changed = (~fit_identity).nonzero().flatten().tolist()
    selection_count = 80
    selection_ids = split_ids["tune"][:selection_count]
    evaluation_ids = split_ids["tune"][selection_count:]
    fit = {
        "factual": factual["fit"],
        "changed": _part(
            [fit_oracle["ids"][i] for i in changed],
            fit_oracle["z"][changed],
            [fit_oracle["texts"][i] for i in changed],
        ),
    }
    select = {
        "factual": _part(
            selection_ids,
            factual["tune"]["z"][:selection_count],
            factual["tune"]["texts"][:selection_count],
        )
    }
    evaluation = {
        "factual": _part(
            evaluation_ids,
            factual["tune"]["z"][selection_count:],
            factual["tune"]["texts"][selection_count:],
        ),
        "oracle": _part(
            evaluation_ids,
            tune_oracle["z"][selection_count:],
            tune_oracle["texts"][selection_count:],
        ),
        "is_identity": tune_identity[selection_count:].clone(),
    }
    splits = {
        "fit": split_ids["fit"],
        "selection": selection_ids,
        "evaluation": evaluation_ids,
        "audit": split_ids["audit"],
        "official_test": split_ids["official_test"],
    }
    return fit, select, evaluation, splits


def prepare(config: dict) -> Path:
    root = prepared_path(config)
    with lock(root.parent / ".cf-prepare.lock"):
        observed, factual_manifest, oracle_info = verified_sources(config)
        if (root / "manifest.json").exists():
            manifest = read_json(root / "manifest.json")
            if manifest["input_sha256"] != observed:
                raise ValueError("prepared inputs changed")
            for name in ("fit", "selection", "evaluation"):
                load_partition(root, name)
            return root
        if root.exists():
            raise ValueError("incomplete prepared directory; inspect before retrying")
        factual = load_bundle(resolve(config["inputs"]["factual_prepared"]))
        pair_rows = csv_rows(resolve(config["inputs"]["pair_index"]))
        allowed = set(
            factual_manifest["split_ids"]["fit"] + factual_manifest["split_ids"]["tune"]
        )
        cf_rows = csv_rows(
            resolve(config["inputs"]["counterfactual_text"]), allowed_ids=allowed
        )
        payload = torch.load(
            resolve(config["inputs"]["oracle_vectors"]),
            map_location="cpu",
            weights_only=True,
        )
        fit, select, evaluation, splits = assemble(
            factual_manifest["split_ids"], factual, payload, pair_rows, cf_rows
        )
        if oracle_info["training_units"] != len(payload["train_ids"]) or oracle_info[
            "identity_units"
        ] != int(payload["is_identity"].sum()):
            raise ValueError("oracle metadata count mismatch")
        root.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            prefix=".cf-prepared-", dir=root.parent
        ) as staging_name:
            staging = Path(staging_name)
            for name, part in (
                ("fit", fit),
                ("selection", select),
                ("evaluation", evaluation),
            ):
                save_torch(staging / f"{name}.pt", part)
            manifest = {
                "format_version": 1,
                "purpose": "privileged_text_decoder",
                "input_sha256": observed,
                "split_ids": splits,
                "counts": {
                    "fit_factual": len(fit["factual"]["ids"]),
                    "fit_changed": len(fit["changed"]["ids"]),
                    "selection_factual": len(select["factual"]["ids"]),
                    "evaluation_changed": int((~evaluation["is_identity"]).sum()),
                    "evaluation_identity": int(evaluation["is_identity"].sum()),
                },
                "files": {
                    f"{name}.pt": sha256_file(staging / f"{name}.pt")
                    for name in ("fit", "selection", "evaluation")
                },
                "encoder": factual_manifest["encoder"],
                "encoder_protocol": factual_manifest["encoder_protocol"],
                "original_oracle_purpose": oracle_info["purpose"],
                "used_official_test_for_selection": False,
            }
            write_json(staging / "manifest.json", manifest)
            os.rename(staging, root)
    return root


def load_partition(root: Path, name: str) -> dict:
    if name not in {"fit", "selection", "evaluation"}:
        raise ValueError("unknown privileged partition")
    manifest = read_json(root / "manifest.json")
    filename = f"{name}.pt"
    if sha256_file(root / filename) != manifest["files"][filename]:
        raise ValueError("prepared partition checksum mismatch")
    part = torch.load(root / filename, map_location="cpu", weights_only=True)
    expected = {
        "fit": {"factual", "changed"},
        "selection": {"factual"},
        "evaluation": {"factual", "oracle", "is_identity"},
    }[name]
    if set(part) != expected:
        raise ValueError("prepared partition contains forbidden fields")
    for key in expected - {"is_identity"}:
        row = part[key]
        if set(row) != {"ids", "z", "texts"}:
            raise ValueError("malformed prepared world")
        if row["z"].shape != (len(row["ids"]), 768) or len(row["texts"]) != len(
            row["ids"]
        ):
            raise ValueError("prepared world dimensions disagree")
        if set(row["ids"]) & set(
            manifest["split_ids"]["audit"] + manifest["split_ids"]["official_test"]
        ):
            raise ValueError("audit/test IDs leaked into development partition")
    if name == "fit":
        if part["factual"]["ids"] != manifest["split_ids"]["fit"]:
            raise ValueError("fit IDs changed")
        if not set(part["changed"]["ids"]) <= set(manifest["split_ids"]["fit"]):
            raise ValueError("changed training IDs leave the fit split")
    else:
        ids = manifest["split_ids"][name]
        if any(part[key]["ids"] != ids for key in expected - {"is_identity"}):
            raise ValueError("tune IDs changed")
        if name == "evaluation" and part["is_identity"].shape != (len(ids),):
            raise ValueError("evaluation identity flags are misaligned")
    return part
