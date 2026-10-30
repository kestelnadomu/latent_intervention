"""One model per worker; append complete CVs and resume from the last saved ID."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

import torch

from src.artifact_io import sha256_file, write_json
from src.text_realizer.artifacts import load_text_realizer
from src.text_realizer.generation import generate

from .common import (
    CONDITIONS,
    checked_file,
    immutable_json,
    load_manifest,
    lock,
    source_hashes,
    versions,
)


def read_completed(
    path: Path, ids: list[int], *, repair_tail: bool = False
) -> list[dict]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        if not repair_tail:
            raise ValueError(f"incomplete final record: {path}")
        length = raw.rfind(b"\n") + 1
        with path.open("r+b") as stream:
            stream.truncate(length)
        raw = raw[:length]
    rows = [json.loads(line) for line in raw.splitlines()]
    if [row["id"] for row in rows] != ids[: len(rows)] or len(rows) > len(ids):
        raise ValueError(
            "saved IDs are duplicated, out of order or belong to another shard"
        )
    if any(
        not isinstance(row.get("text"), str)
        or row.get("termination") not in {"eos", "max_length"}
        for row in rows
    ):
        raise ValueError("malformed saved generation")
    return rows


def conditions(manifest: dict) -> tuple[str, ...]:
    """The bundle's vector sets: oracle and predicted, or e.g. one per renderer."""
    return tuple(manifest.get("conditions", CONDITIONS))


def load_vectors(bundle: Path, manifest: dict, model: dict) -> dict:
    dim = model["latent_dim"]
    payload = torch.load(
        checked_file(bundle, manifest["files"][f"vectors-{dim}"]),
        map_location="cpu",
        weights_only=True,
    )
    if (
        set(payload) != {"ids", "encoder", *conditions(manifest)}
        or payload["ids"] != manifest["ids"]
        or payload["encoder"] != model["encoder"]
    ):
        raise ValueError("incompatible vector bundle")
    for condition in conditions(manifest):
        z = payload[condition]
        if (
            not isinstance(z, torch.Tensor)
            or z.dtype != torch.float32
            or z.shape != (len(manifest["ids"]), dim)
            or not torch.isfinite(z).all()
        ):
            raise ValueError("invalid vectors for decoder")
    # In particular, never normalize or truncate the predicted editor outputs.
    return payload


def run_generation(
    bundle: Path,
    output: Path,
    model_key: str,
    *,
    condition: str = "both",
    device: str = "cpu",
    shard_index: int = 0,
    num_shards: int = 1,
    limit: int | None = None,
    cache_dir: Path | None = None,
    threads: int = 1,
) -> None:
    if (
        num_shards < 1
        or not 0 <= shard_index < num_shards
        or threads < 1
        or (limit is not None and limit < 1)
    ):
        raise ValueError("invalid shard, limit or CPU thread count")
    if device != "cpu" and not device.startswith("cuda:"):
        raise ValueError("use cpu or an explicit visible CUDA device")
    torch.set_num_threads(threads)
    manifest = load_manifest(bundle)
    if condition not in (*conditions(manifest), "both", "all"):
        raise ValueError("unknown conditioning arm")
    specification = manifest["models"][model_key]
    checkpoint = checked_file(bundle, specification)
    payload = load_vectors(bundle, manifest, specification)
    positions = list(range(len(manifest["ids"])))[shard_index::num_shards]
    if not positions:
        raise ValueError("empty shard")
    ids = [manifest["ids"][i] for i in positions]
    execution = {
        "torch": torch.__version__,
        **versions(("transformers", "peft")),
        "device_type": device.split(":")[0],
    }
    model = None
    for arm in conditions(manifest) if condition in ("both", "all") else (condition,):
        folder = output / model_key / arm
        path = folder / f"part-{shard_index:03d}-of-{num_shards:03d}.jsonl"
        meta = {
            "bundle_sha256": sha256_file(bundle / "manifest.json"),
            "checkpoint_sha256": specification["sha256"],
            "model": model_key,
            "condition": arm,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "ids": ids,
            "generation": specification["generation"],
            "execution": execution,
            "source_sha256": source_hashes(),
        }
        with lock(path.with_suffix(".lock")):
            # A different sharding scheme must use a different output directory.
            others = list(folder.glob("part-*.meta.json"))
            for other in others:
                if json.loads(other.read_text())["num_shards"] != num_shards:
                    raise ValueError("sharding changed; use a new output directory")
            immutable_json(path.with_suffix(".meta.json"), meta)
            rows = read_completed(path, ids, repair_tail=True)
            if len(rows) == len(ids):
                write_json(
                    path.with_suffix(".done.json"),
                    {
                        "complete": True,
                        "records": len(rows),
                        "sha256": sha256_file(path),
                    },
                )
                print(
                    f"{model_key}/{arm}: already complete ({len(rows)} CVs)", flush=True
                )
                continue
            if model is None:
                model = load_text_realizer(
                    checkpoint, device=device, cache_dir=cache_dir
                )
            started, new = perf_counter(), 0
            with path.open("a", encoding="utf-8") as stream:
                for index in range(len(rows), len(ids)):
                    z = payload[arm][positions[index] : positions[index] + 1]
                    row = {"id": ids[index], **generate(model, z, batch_size=1)[0]}
                    stream.write(
                        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                    new += 1
                    elapsed = perf_counter() - started
                    progress = {
                        "completed": index + 1,
                        "total": len(ids),
                        "seconds_per_cv": elapsed / new,
                        "remaining_seconds": (len(ids) - index - 1) * elapsed / new,
                        "device": device,
                    }
                    write_json(path.with_suffix(".progress.json"), progress)
                    if new == 1 or new % 10 == 0 or index + 1 == len(ids):
                        print(
                            f"{model_key}/{arm}: {index + 1}/{len(ids)}; {elapsed / new:.1f} s/CV",
                            flush=True,
                        )
                    if limit is not None and new >= limit:
                        break
            completed = read_completed(path, ids)
            if len(completed) == len(ids):
                write_json(
                    path.with_suffix(".done.json"),
                    {
                        "complete": True,
                        "records": len(ids),
                        "sha256": sha256_file(path),
                    },
                )
