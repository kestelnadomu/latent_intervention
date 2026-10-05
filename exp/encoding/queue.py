"""Serial, resumable Matryoshka dimension queues using one full-width pass.

python -m src.encoders.nomic_encoding --prepare
python -m src.encoders.nomic_encoding --run

Completed stages resume after validation. An interrupted transformer pass restarts;
partially written staging directories are retained, never silently overwritten.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
from importlib.metadata import version
import os
import re
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
import torch.nn.functional as F

from src.artifact_io import write_json
from src.config import (
    CONFIG_PATH,
    NOMIC_DIMENSIONS,
    QWEN3_DIMENSIONS,
    EMBEDDINGGEMMA_DIMENSIONS,
    load_config,
)
from src.pair_encoding import (
    LatentArtifact,
    _active_encoder_info,
    encode_pairs,
    load_latent_artifact,
    sha256_file,
    write_latent_artifact,
)

DIMENSIONS = {
    "nomic": NOMIC_DIMENSIONS,
    "qwen3": QWEN3_DIMENSIONS,
    "embeddinggemma": EMBEDDINGGEMMA_DIMENSIONS,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dimension_configs(
    path: str | Path = CONFIG_PATH, *, variant: str = "nomic"
) -> dict[int, dict]:
    if variant not in DIMENSIONS:
        raise ValueError(f"dimension queues support {tuple(DIMENSIONS)}")
    dimensions = DIMENSIONS[variant]
    configs = {
        dim: load_config(path=path, encoder_variant=variant, **{f"{variant}_dim": dim})
        for dim in dimensions
    }
    for config in configs.values():
        config["encoder"]["progress"] = True
    if any(config["encoder"].get("tag") for config in configs.values()):
        raise ValueError("the dimension queue requires encoder.tag: null")
    outputs = [
        Path(config["paths"]["latents"]).resolve() for config in configs.values()
    ]
    if (
        len(set(outputs)) != len(outputs)
        or len({p.parent.parent for p in outputs}) != 1
    ):
        raise ValueError(
            "queue latent paths must use distinct sibling {encoder} directories"
        )
    if any(
        p.parent.name != f"{variant}_{d}"
        for d, p in zip(dimensions, outputs, strict=True)
    ):
        raise ValueError(
            f"queue output directories must be named {variant}_<dimension>"
        )
    return configs


def verify_artifact(config: dict) -> LatentArtifact:
    artifact = load_latent_artifact(config)
    for name, tensor in (("z", artifact.z), ("z_prime", artifact.z_prime)):
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32 for this dimension comparison")
        if not torch.allclose(
            tensor.norm(dim=1), torch.ones(len(tensor)), atol=1e-6, rtol=1e-6
        ):
            raise ValueError(f"{name} must contain unit-normalized embedding vectors")
    return artifact


def projected_payload(source: LatentArtifact, dim: int) -> dict[str, Any]:
    """Slice a full-width normalized embedding source, then renormalize.

    Do NOT reapply layer norm to the truncated prefix. Full-vector L2 scaling
    cancels during the prefix normalization, up to floating-point rounding.
    """
    variant = source.encoder_info["encoder_variant"]
    dimensions = DIMENSIONS.get(variant, ())
    if not dimensions or source.z.shape[1] != max(dimensions) or dim not in dimensions:
        raise ValueError(
            "projection requires a full-width source and a supported dimension"
        )
    z = F.normalize(source.z[:, :dim], p=2, dim=1).contiguous()
    z_prime = F.normalize(source.z_prime[:, :dim], p=2, dim=1).contiguous()
    positions = {row_id: i for i, row_id in enumerate(source.ids)}
    for i, (row_id, identity) in enumerate(
        zip(source.test_ids, source.is_identity.tolist(), strict=True)
    ):
        if identity:
            z_prime[i] = z[positions[row_id]]
    return {
        "ids": source.ids,
        "z": z,
        "test_ids": source.test_ids,
        "z_prime": z_prime,
        "is_identity": source.is_identity.clone(),
    }


def compare_projection(
    source: LatentArtifact, target: LatentArtifact
) -> dict[str, float]:
    if source.ids != target.ids or source.test_ids != target.test_ids:
        raise ValueError("cross-dimension IDs differ")
    expected = projected_payload(source, target.z.shape[1])
    differences = {}
    for name, tensor in (("z", target.z), ("z_prime", target.z_prime)):
        differences[name] = (tensor - expected[name]).abs().max().item()
        if not torch.allclose(tensor, expected[name], atol=1e-6, rtol=1e-5):
            raise ValueError(
                f"{target.z.shape[1]}D {name} disagrees with {source.z.shape[1]}D source: {differences[name]}"
            )
    return differences


def inspect_queue(configs: dict[int, dict]) -> list[dict[str, Any]]:
    rows = []
    full_dim = max(configs)
    for dim in (
        full_dim,
        128,
        *(d for d in sorted(configs) if d not in (full_dim, 128)),
    ):
        config = configs[dim]
        path = Path(config["paths"]["latents"])
        # Detect partial outputs as well as complete ones; never silently replace either.
        exists = path.exists() or path.with_suffix(".info.json").exists()
        if path.parent.exists() and not exists:
            raise FileExistsError(
                f"output directory already exists without an artifact: {path.parent}"
            )
        artifact = verify_artifact(config) if exists else None
        rows.append(
            {
                "dimension": dim,
                "path": str(path),
                "action": (
                    "verify/reuse"
                    if artifact
                    else ("encode" if dim == full_dim else "derive")
                ),
                "artifact_sha256": artifact.artifact_sha256 if artifact else None,
            }
        )
    return rows


def protocol(configs: dict[int, dict], threads: int) -> dict:
    base = configs[max(configs)]
    is_qwen3 = base["encoder"]["variant"] == "qwen3"
    is_gemma = base["encoder"]["variant"] == "embeddinggemma"
    extra = {}
    if is_qwen3:
        from src.encoders.qwen3_encoder import MODEL_FILES, cached_snapshot

        snapshot = Path(cached_snapshot(base["encoder"]))
        extra["model_files_sha256"] = {
            name: sha256_file(snapshot / name) for name in MODEL_FILES
        }
    if is_gemma:
        from src.encoders.embeddinggemma_encoder import MODEL_FILES, cached_snapshot

        snapshot = Path(cached_snapshot(base["encoder"]))
        extra["model_files_sha256"] = {
            name: sha256_file(snapshot / name) for name in MODEL_FILES
        }
        extra["sentence_transformers_version"] = version("sentence-transformers")
    return {
        "format_version": 1,
        "encoder": _active_encoder_info(base["encoder"]),
        "batch_size": base["encoder"]["batch_size"],
        "device": base["encoder"]["device"],
        "threads": threads,
        "outputs": {
            str(d): str(Path(c["paths"]["latents"]).resolve())
            for d, c in configs.items()
        },
        "input_sha256": {
            key: sha256_file(base["paths"][key])
            for key in ("texts", "texts_counterfactual", "pair_index")
        },
        "source_sha256": {
            str(path): sha256_file(path)
            for path in (
                Path(__file__),
                Path("src/config.py"),
                Path("src/encoder.py"),
                Path("src/pair_encoding.py"),
                Path("src/artifact_io.py"),
                Path("src/encoder_protocols.py"),
                Path("src/encoding_progress.py"),
                Path("src/latent_writer.py"),
                Path("uv.lock"),
                *(
                    [
                        Path("src/encoders/qwen3_encoder.py"),
                        Path("src/encoders/qwen3_encoding.py"),
                        Path("src/encoders/qwen3_preflight.py"),
                        Path("requirements/qwen3.lock"),
                    ]
                    if is_qwen3
                    else []
                ),
                *(
                    [
                        Path(f"src/encoders/embeddinggemma_{part}.py")
                        for part in ("encoder", "encoding", "preflight", "setup")
                    ]
                    + [
                        Path("requirements/embeddinggemma.lock"),
                        Path("scripts/run_embeddinggemma_dimensions.sh"),
                    ]
                    if is_gemma
                    else []
                ),
            )
        },
        "torch_version": str(torch.__version__),
        "transformers_version": version("transformers"),
        "derivation": (
            "normalize(first_d_coordinates(normalize(last_nonpadding_token)))"
            if is_qwen3
            else (
                "normalize(first_d_coordinates(normalize(dense2(dense1(mean_pool_bidirectional_tokens)))))"
                if is_gemma
                else "normalize(first_d_coordinates(normalize(layer_norm(mean_pool(tokens)))))"
            )
        ),
        "comparison_tolerance": {"atol": 1e-6, "rtol": 1e-5},
        **extra,
    }


def publish_dimension(
    config: dict,
    run_id: str,
    source: LatentArtifact | None = None,
    *,
    encoder_factory=None,
) -> LatentArtifact:
    """Stage, validate, then atomically rename a complete directory into place."""
    destination = Path(config["paths"]["latents"])
    staging_dir = destination.parent.with_name(
        f".{destination.parent.name}.{run_id}.pending"
    )
    staging = copy.deepcopy(config)
    staging["paths"]["latents"] = str(staging_dir / destination.name)
    if staging_dir.exists():
        try:
            verify_artifact(staging)
        except (ValueError, OSError):
            archived = staging_dir.with_name(
                f"{staging_dir.name}.interrupted-{uuid4().hex[:8]}"
            )
            staging_dir.rename(archived)
            print(f"retained incomplete staging directory at {archived}", flush=True)
    if not staging_dir.exists():
        if source is None:
            encode_pairs(staging, encoder_factory=encoder_factory)
        else:
            dim = _active_encoder_info(config["encoder"])["latent_dimension"]
            write_latent_artifact(
                staging,
                projected_payload(source, dim),
                derivation={
                    "method": "prefix_truncation_then_l2",
                    "source_dimension": source.z.shape[1],
                    "source_artifact_sha256": source.artifact_sha256,
                },
            )
    staged = verify_artifact(staging)
    if source is not None:
        compare_projection(source, staged)
    if destination.parent.exists():
        raise FileExistsError(f"refusing to replace {destination.parent}")
    staging_dir.rename(destination.parent)
    directory = os.open(destination.parent.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return verify_artifact(config)


def run_queue(
    configs: dict[int, dict], root: Path, run_id: str, *, encoder_factory=None
) -> dict:
    rows = inspect_queue(
        configs
    )  # validate every existing dimension before any encoding
    state = {"status": "running", "started_at": now(), "queue": rows, "completed": []}

    def save():
        state["updated_at"] = now()
        write_json(root / "status.json", state)

    save()
    try:
        source = None
        for row in rows:
            dim = row["dimension"]
            state["active_dimension"] = dim
            save()
            print(f"{now()} {dim}D: {row['action']}", flush=True)
            if row["action"] == "verify/reuse":
                artifact = verify_artifact(configs[dim])
            else:
                artifact = publish_dimension(
                    configs[dim], run_id, source, encoder_factory=encoder_factory
                )
            if dim == max(configs):
                source = artifact
                difference = None
            else:
                difference = compare_projection(source, artifact)
            state["completed"].append(
                {
                    "dimension": dim,
                    "path": row["path"],
                    "artifact_sha256": artifact.artifact_sha256,
                    "factual_shape": list(artifact.z.shape),
                    "counterfactual_shape": list(artifact.z_prime.shape),
                    "max_projection_difference": difference,
                }
            )
            save()
        state.update(status="complete", active_dimension=None, finished_at=now())
        save()
        return state
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


def main(*, variant: str = "nomic") -> None:
    parser = argparse.ArgumentParser(
        description=f"Prepare/run the serial, resumable {variant} embedding-dimension queue."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prepare",
        action="store_true",
        help="validate/save the plan; no model inference",
    )
    mode.add_argument(
        "--run", action="store_true", help="execute/resume the prepared queue"
    )
    if variant in {"qwen3", "embeddinggemma"}:
        mode.add_argument(
            "--preflight",
            action="store_true",
            help=f"audit all input lengths and smoke-test the cached {variant} model; no full run",
        )
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--run-id", default=f"{variant}-dimensions-v1")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", args.run_id) or args.threads < 1:
        parser.error(
            "run-id must be a simple directory name and threads must be positive"
        )
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    configs = dimension_configs(args.config, variant=variant)
    root = Path(f"reports/talent/{variant}_dimensions") / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    # One lock across run IDs: two queues must never write the same dimension.
    lock_path = (
        Path(configs[max(configs)]["paths"]["latents"]).parent.parent
        / f".{variant}-encoding.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"another {variant} dimension queue holds the output lock")
        signature = protocol(configs, args.threads)
        if variant in {"qwen3", "embeddinggemma"}:
            label = "Qwen3" if variant == "qwen3" else "EmbeddingGemma"
            preflight_path = root / "preflight.json"
            if getattr(args, "preflight", False):
                if variant == "qwen3":
                    from src.encoders.qwen3_preflight import run_preflight
                else:
                    from src.encoders.embeddinggemma_preflight import run_preflight

                report = run_preflight(configs)
                report["protocol_signature"] = signature
                write_json(preflight_path, report)
                print(
                    json.dumps(
                        {k: v for k, v in report.items() if k != "protocol_signature"},
                        indent=2,
                    )
                )
                return
            if not preflight_path.is_file():
                raise ValueError(
                    f"{label} requires a passing --preflight before --prepare or --run"
                )
            preflight = json.loads(preflight_path.read_text())
            if (
                preflight.get("status") != "passed"
                or preflight.get("protocol_signature") != signature
            ):
                raise ValueError(
                    f"{label} preflight is stale; repeat --preflight with the current code/config/environment"
                )
        manifest = root / "protocol.json"
        if manifest.exists() and json.loads(manifest.read_text()) != signature:
            raise ValueError("queue code/config/inputs changed; use a new run-id")
        rows = inspect_queue(configs)
        if not manifest.exists():
            write_json(manifest, signature)
        if args.prepare:
            plan = {"status": "prepared", "prepared_at": now(), "queue": rows}
            write_json(root / "plan.json", plan)
            print(json.dumps(plan, indent=2))
            return
        print(f"Queue log: {root / 'run.log'}", flush=True)
        with (root / "run.log").open("a", buffering=1) as log:
            with redirect_stdout(log), redirect_stderr(log):
                try:
                    state = run_queue(configs, root, args.run_id)
                except BaseException:
                    traceback.print_exc()
                    raise
        print(f"Queue {state['status']}: {root / 'status.json'}", flush=True)


if __name__ == "__main__":
    main()
