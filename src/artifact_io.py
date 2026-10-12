"""Atomic writes for training checkpoints, reports, and manifests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterator

import torch


@contextmanager
def atomic_output(path: str | Path) -> Iterator[Path]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        yield temporary
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: str | Path, payload: Any) -> None:
    with atomic_output(path) as temporary:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def save_torch(path: str | Path, payload: Any) -> None:
    with atomic_output(path) as temporary:
        # A file object gives torch.save a stable archive name across temp paths.
        with temporary.open("wb") as stream:
            torch.save(payload, stream)


def copy_atomic(source: str | Path, destination: str | Path) -> None:
    with atomic_output(destination) as temporary:
        shutil.copyfile(source, temporary)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
