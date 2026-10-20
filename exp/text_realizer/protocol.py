"""Atomic, hash-bound phase transitions and advisory process locks."""

from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from src.artifact_io import copy_atomic, sha256_file, write_json

from .settings import ROOT, packages, source_hashes


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def immutable_json(path: Path, payload: dict) -> None:
    if path.exists():
        if read_json(path) != payload:
            raise ValueError(f"immutable artifact changed: {path}; use a new run ID")
    else:
        write_json(path, payload)


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


def status(root: Path, state: str, **details) -> None:
    write_json(
        root / "status.json",
        {
            "state": state,
            "heartbeat": datetime.now(timezone.utc).isoformat(),
            **details,
        },
    )


def identity(config: dict, prepared: Path) -> dict:
    return {
        "config": config,
        "prepared_manifest_sha256": sha256_file(prepared / "manifest.json"),
        "source_sha256": source_hashes(),
        "packages": packages(),
    }


def progress_callback(root: Path, phase: str, **details):
    """Throttle fsync-heavy heartbeat updates during lengthy evaluation loops."""
    previous = 0.0

    def emit(event: str, done: int, total: int):
        nonlocal previous
        now = perf_counter()
        if done == total or now - previous >= 15:
            status(root, phase, event=event, completed=done, total=total, **details)
            previous = now

    return emit


def snapshot(root: Path, plan: dict) -> None:
    immutable_json(root / "plan.json", plan)
    for relative, expected in plan["identity"]["source_sha256"].items():
        source = ROOT / relative
        if sha256_file(source) != expected:
            raise ValueError(f"source changed during snapshot: {relative}")
        target = root / "source" / relative
        if target.exists() and sha256_file(target) != expected:
            raise ValueError(f"source snapshot changed: {relative}")
        if not target.exists():
            copy_atomic(source, target)


def require_current(root: Path, config: dict, prepared: Path) -> dict:
    plan = read_json(root / "plan.json")
    if plan["identity"] != identity(config, prepared):
        raise ValueError(
            "code, inputs, environment or configuration changed; use a new run ID"
        )
    return plan


def require_frozen(root: Path) -> dict:
    frozen = read_json(root / "frozen.json")
    if frozen.get("used_counterfactual_text") or frozen.get(
        "used_counterfactual_latent"
    ):
        raise ValueError("not a factual-only frozen model")
    for name, expected in frozen["reports_sha256"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe report reference")
        if sha256_file(root / name) != expected:
            raise ValueError(f"frozen report changed: {name}")
    return frozen


def oracle_gate(root: Path, *, allowed: bool) -> dict:
    """No caller may open oracle files before this check completes."""
    if not allowed:
        raise PermissionError("requires literal --allow-oracle-evaluation")
    frozen = require_frozen(root)
    audit = read_json(root / "factual_audit.json")
    if audit["frozen_sha256"] != sha256_file(root / "frozen.json"):
        raise ValueError("audit does not belong to the frozen model")
    if not frozen.get("reload_verified") or not audit.get("complete"):
        raise ValueError(
            "fresh reload verification and completed factual audit required"
        )
    for seed, expected in audit["seed_report_sha256"].items():
        if (
            not str(seed).isdigit()
            or sha256_file(root / f"audit-seed-{seed}.json") != expected
        ):
            raise ValueError("factual audit report changed")
    return frozen
