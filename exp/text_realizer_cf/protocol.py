"""Small immutable run records and fail-closed readiness checks."""

from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.artifact_io import copy_atomic, sha256_file, write_json

from .settings import ROOT, digest, source_hashes


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


@contextmanager
def lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process owns {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def immutable_json(path: Path, payload: dict) -> None:
    if path.exists():
        if read_json(path) != payload:
            raise ValueError(f"immutable artifact changed: {path}; use a new run ID")
    else:
        write_json(path, payload)


def status(root: Path, state: str, **details: object) -> None:
    write_json(
        root / "status.json",
        {
            "state": state,
            "heartbeat": datetime.now(timezone.utc).isoformat(),
            **details,
        },
    )


def identity(config: dict, prepared: Path) -> dict:
    from exp.text_realizer.settings import packages

    return {
        "config_sha256": digest(config),
        "prepared_manifest_sha256": sha256_file(prepared / "manifest.json"),
        "source_sha256": source_hashes(),
        "packages": packages(),
    }


def snapshot(root: Path, plan: dict) -> None:
    immutable_json(root / "plan.json", plan)
    for name, expected in plan["identity"]["source_sha256"].items():
        source = ROOT / name
        if sha256_file(source) != expected:
            raise ValueError(f"source changed during snapshot: {name}")
        target = root / "source" / name
        if target.exists() and sha256_file(target) != expected:
            raise ValueError(f"source snapshot changed: {name}")
        if not target.exists():
            copy_atomic(source, target)


def require_current(root: Path, config: dict, prepared: Path) -> dict:
    plan = read_json(root / "plan.json")
    if plan["identity"] != identity(config, prepared):
        raise ValueError(
            "code, config, inputs, or environment changed; use a new run ID"
        )
    return plan
