"""Small file/provenance helpers shared by the three evaluation commands."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path

import yaml

from exp.text_realizer.protocol import immutable_json as immutable_json, lock as lock
from src.artifact_io import sha256_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/text_realizer_package_eval.yaml"
DEFAULT_BUNDLE = ROOT / "models/talent_motivation/text-realizer-evaluation/package-v1"
DEFAULT_OUTPUT = ROOT / "reports/talent_motivation/text-realizer-evaluation/package-v1"
CONDITIONS = ("oracle", "predicted")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def read_config(path: Path = DEFAULT_CONFIG) -> dict:
    config = yaml.safe_load(path.read_text())
    if config["format_version"] != 1 or config["unit_count"] < 1:
        raise ValueError("invalid evaluation configuration")
    if set(config["pipelines"]) != {"256", "768"}:
        raise ValueError("requires the explicit 256-D and 768-D pipelines")
    if set(config["decoders"]) != {"256-270m", "256-1b", "768-270m", "768-1b"}:
        raise ValueError("requires the four named decoder checkpoints")
    return config


def checked_source(item: dict) -> Path:
    path = Path(item["path"])
    path = path if path.is_absolute() else ROOT / path
    # A CRLF checkout (core.autocrlf on Windows) of a pinned text file still verifies.
    lf = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if item["sha256"] not in (sha256_file(path), lf):
        raise ValueError(f"source checksum mismatch: {path}")
    return path


def checked_file(root: Path, item: dict) -> Path:
    path = Path(item["file"])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe bundle file name")
    path = root / path
    if sha256_file(path) != item["sha256"]:
        raise ValueError(f"bundle checksum mismatch: {path}")
    return path


def file_record(root: Path, path: Path) -> dict:
    return {"file": str(path.relative_to(root)), "sha256": sha256_file(path)}


def csv_records(path: Path) -> dict[int, dict]:
    result = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = int(row["id"])
            if key in result:
                raise ValueError(f"duplicate ID {key}: {path}")
            result[key] = row
    return result


def jsonl_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def versions(names: tuple[str, ...]) -> dict:
    return {name: importlib.metadata.version(name) for name in names}


def source_hashes() -> dict:
    files = list((ROOT / "exp/text_realizer_package_eval").glob("*.py"))
    files.extend((ROOT / "latent_lab").glob("*.py"))
    files.extend((ROOT / "src/text_realizer").glob("*.py"))
    files += [
        ROOT / "src/artifact_io.py",
        ROOT / "exp/text_realizer/facts.py",
        ROOT / "exp/sim/codebook.yaml",
        ROOT / "exp/text_realizer/environment/pyproject.toml",
        ROOT / "exp/text_realizer/environment/uv.lock",
        ROOT / "exp/text_realizer_package_eval/environment/pyproject.toml",
        ROOT / "exp/text_realizer_package_eval/environment/uv.lock",
    ]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(files)}


def load_manifest(root: Path) -> dict:
    manifest = read_json(root / "manifest.json")
    if manifest.get("format_version") != 1 or not manifest.get("complete"):
        raise ValueError("requires a completed evaluation bundle")
    return manifest
