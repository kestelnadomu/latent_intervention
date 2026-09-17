"""Encode and validate the canonical paired-latent artifact."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral
from pathlib import Path
from typing import Any

import pandas as pd
import torch

_ARTIFACT_VERSION = 1
_LATENT_DIMENSION = 128
_NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"
_ENCODER_INFO_KEYS = (
    "encoder_variant",
    "encoder",
    "model_revision",
    "local_checkpoint",
    "local_checkpoint_sha256",
    "code_revision",
    "task",
    "max_length",
    "deterministic",
    "latent_dimension",
)


@dataclass(frozen=True)
class LatentArtifact:
    """Validated latent tensors and the provenance needed by later checkpoints."""

    ids: list[int]
    z: torch.Tensor
    test_ids: list[int]
    z_prime: torch.Tensor
    is_identity: torch.Tensor
    artifact_sha256: str
    encoder_info: dict[str, Any]

    @property
    def train_ids(self) -> list[int]:
        """IDs assigned to the official training split by ``pair_index.csv``."""
        test_ids = set(self.test_ids)
        return [row_id for row_id in self.ids if row_id not in test_ids]


def _load_csv(path: str | Path, label: str, columns: set[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    frame = pd.read_csv(path)
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns {sorted(missing)}")
    ids = pd.to_numeric(frame["id"], errors="coerce")
    if ids.isna().any() or not ids.eq(ids.round()).all():
        raise ValueError(f"{label} IDs must be integers")
    frame["id"] = ids.astype(int)
    if frame["id"].duplicated().any():
        raise ValueError(f"{label} contains duplicate IDs")
    return frame.sort_values("id").reset_index(drop=True)


def _bool(value: Any, row_id: int) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"is_identity for id={row_id} must be true/false")


def _load_pair_index(path: str | Path) -> pd.DataFrame:
    pairs = _load_csv(path, "pair index", {"id", "split", "is_identity"})
    pairs["split"] = pairs["split"].astype(str).str.lower()
    if set(pairs["split"]) != {"train", "test"}:
        raise ValueError("pair index must contain non-empty train and test splits")
    pairs["is_identity"] = [
        _bool(value, row_id)
        for row_id, value in zip(pairs["id"], pairs["is_identity"], strict=True)
    ]
    return pairs


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_directory(path: Path) -> str:
    """Hash relative names and bytes of every file in a checkpoint directory."""
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"local checkpoint contains no files: {path}")
    for file_path in files:
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _active_encoder_info(config: dict[str, Any]) -> dict[str, Any]:
    variant = config.get("variant", "langvae")
    if variant not in {"langvae", "nomic"}:
        raise ValueError(f"unknown encoder variant: {variant}")
    if config.get("deterministic", True) is not True:
        raise ValueError("paired latent artifacts require deterministic encoding")

    if variant == "nomic":
        model = config.get("nomic_model_name", _NOMIC_MODEL)
        model_revision = config.get("nomic_model_revision")
        local_checkpoint = None
        code_revision = config.get("nomic_code_revision")
        task = config.get("nomic_task", "classification")
    else:
        model = config["model_name"]
        local_checkpoint = config.get("local_checkpoint")
        model_revision = None if local_checkpoint else config.get("model_revision")
        code_revision = None
        task = None

    local_path = Path(local_checkpoint).resolve() if local_checkpoint else None
    if local_path is not None:
        if not local_path.is_dir():
            raise ValueError(f"local checkpoint directory not found: {local_path}")
        local_checkpoint_sha256 = _sha256_directory(local_path)
    else:
        local_checkpoint_sha256 = None

    return {
        "encoder_variant": variant,
        "encoder": model,
        "model_revision": model_revision,
        "local_checkpoint": str(local_path) if local_path is not None else None,
        "local_checkpoint_sha256": local_checkpoint_sha256,
        "code_revision": code_revision,
        "task": task,
        "max_length": int(config["max_len"]),
        "deterministic": True,
        "latent_dimension": _LATENT_DIMENSION,
    }


def _reencode(message: str) -> ValueError:
    return ValueError(
        f"latent artifact {message}; re-encode it with `python -m src.pipeline encode`"
    )


def _integer_ids(value: Any, label: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise _reencode(f"has invalid {label}")
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in value):
        raise _reencode(f"has non-integer {label}")
    ids = [int(item) for item in value]
    if len(ids) != len(set(ids)):
        raise _reencode(f"has duplicate {label}")
    return ids


def _latent_tensor(
    value: Any,
    label: str,
    rows: int,
    latent_dimension: int,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or not torch.is_floating_point(value)
        or value.ndim != 2
        or tuple(value.shape) != (rows, latent_dimension)
        or not torch.isfinite(value).all().item()
    ):
        raise _reencode(
            f"has invalid {label}; expected finite shape ({rows}, {latent_dimension})"
        )
    return value


def load_latent_artifact(config: dict[str, Any]) -> LatentArtifact:
    """Load ``z_pairs.pt`` only after validating its data and provenance."""
    paths = config["paths"]
    artifact_path = Path(paths["latents"])
    info_path = artifact_path.with_suffix(".info.json")
    if not artifact_path.is_file() or not info_path.is_file():
        raise _reencode("or its .info.json sidecar is missing")

    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _reencode("has an unreadable .info.json sidecar") from exc
    required_info = {
        "artifact_version",
        "artifact_sha256",
        "input_sha256",
        "factual_units",
        "counterfactual_test_units",
        "identity_test_units",
        *_ENCODER_INFO_KEYS,
    }
    if not isinstance(info, dict) or required_info - set(info):
        raise _reencode("uses missing or legacy metadata")
    if info["artifact_version"] != _ARTIFACT_VERSION:
        raise _reencode("uses an unsupported metadata version")

    try:
        artifact_sha256 = sha256_file(artifact_path)
    except OSError as exc:
        raise _reencode("cannot be hashed") from exc
    if info["artifact_sha256"] != artifact_sha256:
        raise _reencode("does not match its recorded SHA-256")

    input_paths = {
        "pair_index": paths["pair_index"],
        "factual_text": paths["texts"],
        "counterfactual_text": paths["texts_counterfactual"],
    }
    recorded_input_hashes = info["input_sha256"]
    if not isinstance(recorded_input_hashes, dict):
        raise _reencode("has invalid input hashes")
    try:
        current_input_hashes = {
            label: sha256_file(path) for label, path in input_paths.items()
        }
    except OSError as exc:
        raise _reencode("cannot verify its source inputs") from exc
    if recorded_input_hashes != current_input_hashes:
        raise _reencode("does not match the current pair/text inputs")

    try:
        expected_encoder_info = _active_encoder_info(config["encoder"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _reencode("cannot be matched to the active encoder config") from exc
    encoder_info = {key: info[key] for key in _ENCODER_INFO_KEYS}
    if encoder_info != expected_encoder_info:
        raise _reencode("does not match the active encoder config")

    try:
        payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise _reencode("payload cannot be loaded") from exc
    required_payload = {"ids", "z", "test_ids", "z_prime", "is_identity"}
    if not isinstance(payload, dict) or required_payload - set(payload):
        raise _reencode("payload is incomplete")

    ids = _integer_ids(payload["ids"], "IDs")
    test_ids = _integer_ids(payload["test_ids"], "test IDs")
    latent_dimension = encoder_info["latent_dimension"]
    z = _latent_tensor(payload["z"], "z", len(ids), latent_dimension)
    z_prime = _latent_tensor(
        payload["z_prime"], "z_prime", len(test_ids), latent_dimension
    )
    identity = payload["is_identity"]
    if (
        not isinstance(identity, torch.Tensor)
        or identity.dtype != torch.bool
        or identity.ndim != 1
        or identity.shape[0] != len(test_ids)
    ):
        raise _reencode("has an invalid is_identity vector")

    try:
        pairs = _load_pair_index(paths["pair_index"])
    except (FileNotFoundError, ValueError) as exc:
        raise _reencode("cannot be aligned with pair_index.csv") from exc
    expected_ids = pairs["id"].tolist()
    test = pairs.loc[pairs["split"] == "test"]
    expected_test_ids = test["id"].tolist()
    if ids != expected_ids:
        raise _reencode("IDs do not match pair_index.csv")
    if test_ids != expected_test_ids:
        raise _reencode("test IDs do not match pair_index.csv")
    if identity.tolist() != test["is_identity"].tolist():
        raise _reencode("identity flags do not match pair_index.csv")
    positions = {row_id: position for position, row_id in enumerate(ids)}
    for test_position, (row_id, is_identity) in enumerate(
        zip(test_ids, identity.tolist(), strict=True)
    ):
        if is_identity and not torch.equal(
            z_prime[test_position], z[positions[row_id]]
        ):
            raise _reencode(f"identity latent differs for id={row_id}")
    if (
        info["factual_units"] != len(ids)
        or info["counterfactual_test_units"] != len(test_ids)
        or info["identity_test_units"] != int(identity.sum().item())
    ):
        raise _reencode("counts do not match its payload")

    return LatentArtifact(
        ids=ids,
        z=z,
        test_ids=test_ids,
        z_prime=z_prime,
        is_identity=identity,
        artifact_sha256=artifact_sha256,
        encoder_info=encoder_info,
    )


def encode_pairs(config: dict[str, Any], encoder_factory=None) -> dict[str, Any]:
    """Encode all X and test X', copying identity latents exactly."""
    paths = config["paths"]
    pairs = _load_pair_index(paths["pair_index"])
    factual = _load_csv(paths["texts"], "factual texts", {"id", "text"})
    counterfactual = _load_csv(
        paths["texts_counterfactual"], "counterfactual texts", {"id", "text"}
    )

    all_ids = pairs["id"].tolist()
    test = pairs.loc[pairs["split"] == "test"].copy()
    test_ids = test["id"].tolist()
    if factual["id"].tolist() != all_ids:
        raise ValueError("factual text IDs must equal all pair-index IDs")
    counterfactual_ids = counterfactual["id"].tolist()
    if counterfactual_ids not in (test_ids, all_ids):
        raise ValueError(
            "counterfactual text IDs must equal either test IDs or all "
            "pair-index IDs exactly"
        )
    for label, frame in (("factual", factual), ("counterfactual", counterfactual)):
        if (
            frame["text"].isna().any()
            or not frame["text"].astype(str).str.strip().all()
        ):
            raise ValueError(f"{label} texts must be non-empty")

    factual_by_id = factual.set_index("id")
    counterfactual_by_id = counterfactual.set_index("id")
    identity = test["is_identity"].astype(bool).tolist()
    for row_id, is_identity in zip(test_ids, identity, strict=True):
        if (
            is_identity
            and counterfactual_by_id.at[row_id, "text"]
            != factual_by_id.at[row_id, "text"]
        ):
            raise ValueError(f"identity text differs for id={row_id}")

    nonidentity_ids = [
        row_id for row_id, flag in zip(test_ids, identity, strict=True) if not flag
    ]
    texts = factual["text"].astype(str).tolist() + [
        str(counterfactual_by_id.at[row_id, "text"]) for row_id in nonidentity_ids
    ]
    enc = config["encoder"]
    encoder_info = _active_encoder_info(enc)
    if encoder_factory is None:
        from src.encoder import make_encoder

        encoder = make_encoder(enc)
    else:
        encoder = encoder_factory(
            model_name=enc["model_name"],
            model_revision=enc.get("model_revision"),
            device=enc["device"],
            max_len=int(enc["max_len"]),
            local_checkpoint=enc.get("local_checkpoint"),
        )
    if int(encoder.latent_dim) != _LATENT_DIMENSION:
        raise ValueError(
            f"encoder latent dimension must be 128, got {encoder.latent_dim}"
        )
    encoded = (
        encoder.encode(
            texts,
            deterministic=encoder_info["deterministic"],
            batch_size=int(enc["batch_size"]),
        )
        .detach()
        .cpu()
    )
    if (
        not torch.is_floating_point(encoded)
        or encoded.shape != (len(texts), _LATENT_DIMENSION)
        or not torch.isfinite(encoded).all()
    ):
        raise ValueError("encoder returned invalid latent vectors")

    # Detach factual storage from the appended counterfactual encodings before saving.
    z = encoded[: len(all_ids)].clone()
    z_prime = torch.empty((len(test_ids), z.shape[1]), dtype=z.dtype)
    factual_position = {row_id: position for position, row_id in enumerate(all_ids)}
    generated_position = {
        row_id: len(all_ids) + i for i, row_id in enumerate(nonidentity_ids)
    }
    for position, (row_id, is_identity) in enumerate(
        zip(test_ids, identity, strict=True)
    ):
        source = factual_position[row_id] if is_identity else generated_position[row_id]
        z_prime[position] = encoded[source]

    payload = {
        "ids": all_ids,
        "z": z,
        "test_ids": test_ids,
        "z_prime": z_prime,
        "is_identity": torch.tensor(identity, dtype=torch.bool),
    }
    output = Path(paths["latents"])
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    info_path = output.with_suffix(".info.json")
    is_nomic = encoder_info["encoder_variant"] == "nomic"
    info_path.write_text(
        json.dumps(
            {
                "artifact_version": _ARTIFACT_VERSION,
                "artifact_sha256": sha256_file(output),
                **encoder_info,
                "task_prefix": f"{encoder_info['task']}: " if is_nomic else None,
                "normalization": ("layer_norm+truncate_128+l2" if is_nomic else None),
                "langvae_version": _package_version("langvae"),
                "transformers_version": _package_version("transformers"),
                "factual_units": len(all_ids),
                "counterfactual_test_units": len(test_ids),
                "identity_test_units": sum(identity),
                "input_sha256": {
                    "pair_index": sha256_file(paths["pair_index"]),
                    "factual_text": sha256_file(paths["texts"]),
                    "counterfactual_text": sha256_file(paths["texts_counterfactual"]),
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote {output}: {len(all_ids)} factual and {len(test_ids)} test pairs")
    return payload
