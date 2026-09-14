"""Shared plumbing for the data-generation stages.

Generic file, digest, and billed-generation helpers with no knowledge of the
CV-screening experiment: resumable CSV outputs guarded by a sibling
``*.generation.json`` journal, a per-ID retry budget, and the generated-pool
contract shared by the pool and CV stages.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from exp.sim.generate_text import DEFAULT_BASE_URL, GenerationResult
from exp.sim.pairing import checked_ids

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_sim_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the data-generation configuration."""
    with open(Path(path), "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


# --- small conversions --------------------------------------------------------------


def slug(text: str) -> str:
    """Lowercase, underscore-joined form of ``text`` for use as a CSV header."""
    return "_".join("".join(char if char.isalnum() else " " for char in text).split()).lower()


def coerce_bool(value: Any, label: str) -> bool:
    """Read a CSV round-tripped boolean, rejecting anything ambiguous."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{label} must be true/false, got {value!r}")


# --- file IO ------------------------------------------------------------------------


def read_csv(
    path: str | Path,
    label: str,
    required: Iterable[str] = (),
) -> pd.DataFrame:
    """Read a CSV with an ``id`` column, checking that ``required`` columns exist."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        frame = pd.read_csv(path, float_precision="round_trip")
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"{label} is malformed: {path}") from exc
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing columns {missing}")
    return checked_ids(frame, label)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` as stable, sorted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 digest of a file's contents."""
    digest = hashlib.sha256()
    with open(Path(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_string_list(path: str | Path, label: str) -> list[str]:
    """Load a non-empty YAML list of non-empty strings."""
    with open(Path(path), "r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(value, str) and value.strip() for value in values)
    ):
        raise ValueError(f"{label} must be a non-empty YAML list of strings: {path}")
    return values


def input_digest(settings: Mapping[str, Any], files: Mapping[str, Path]) -> str:
    """Digest of the settings and file contents a generated output depends on."""
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8"))
    for name, path in sorted(files.items()):
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


# --- generation settings ------------------------------------------------------------


def generation_limit(config: Mapping[str, Any], count: int) -> int:
    """Cap the number of billed calls for one stage run."""
    limit = config.get("generation", {}).get("limit")
    if limit is None:
        return count
    limit = int(limit)
    if limit < 0:
        raise ValueError("generation.limit must be non-negative or null")
    return min(limit, count)


def max_generation_attempts(config: Mapping[str, Any]) -> int:
    """Per-ID retry budget for billed calls."""
    attempts = int(config.get("generation", {}).get("max_attempts", 3))
    if not 1 <= attempts <= 3:
        raise ValueError("generation.max_attempts must be between 1 and 3")
    return attempts


def include_train_counterfactual_texts(config: Mapping[str, Any]) -> bool:
    """Whether the X' artifact includes train units as well as test units."""
    value = config.get("generation", {}).get("include_train_counterfactual_texts", True)
    if not isinstance(value, bool):
        raise ValueError("generation.include_train_counterfactual_texts must be true or false")
    return value


def validate_generation_settings(config: Mapping[str, Any]) -> None:
    """Fail on invalid billing settings before any row is written or billed."""
    generation_limit(config, 0)
    max_generation_attempts(config)
    include_train_counterfactual_texts(config)


def resolve_llm(config: Mapping[str, Any]) -> tuple[str, str | None]:
    """Return the configured endpoint and API key, failing before any call."""
    llm = config.get("llm", {})
    base_url = llm.get("base_url", DEFAULT_BASE_URL)
    if "<resource-name>" in base_url:
        raise ValueError("Fill in llm.base_url in exp/sim/config.yaml")
    env_name = llm.get("api_key_env")
    api_key = os.getenv(env_name) if env_name else None
    if env_name and not api_key:
        raise ValueError(f"Environment variable {env_name} is not set")
    return base_url, api_key


# --- resumable billed outputs -------------------------------------------------------


def generation_info_path(output: Path) -> Path:
    """Path of the journal sitting beside a generated CSV."""
    return output.with_suffix(".generation.json")


def prepare_generated_csv(
    output: Path,
    schema: list[str],
    id_column: str,
    expected_ids: set[int],
    digest: str,
    *,
    create: bool,
) -> pd.DataFrame:
    """Validate a resumable billed output before any API call."""
    info_path = generation_info_path(output)
    expected_info = {
        "input_digest": digest,
        "expected_ids": sorted(expected_ids),
        "id_column": id_column,
        "columns": schema,
    }
    if output.exists() != info_path.exists():
        raise RuntimeError(f"{output} and {info_path} must either both exist or both be absent")
    if not output.exists():
        if not create:
            raise FileNotFoundError(f"generated output not found: {output}")
        write_json(
            info_path,
            {**expected_info, "complete": False, "n_completed": 0, "attempts": {}},
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=schema).to_csv(output, index=False)

    info = json.loads(info_path.read_text(encoding="utf-8"))
    if any(info.get(key) != value for key, value in expected_info.items()):
        raise RuntimeError(f"{info_path} is incompatible with current generation inputs")
    frame = pd.read_csv(output)
    if list(frame.columns) != schema:
        raise ValueError(f"{output} has an incompatible schema")
    if frame.empty:
        return frame
    ids = pd.to_numeric(frame[id_column], errors="coerce")
    if ids.isna().any() or not ids.eq(ids.round()).all():
        raise ValueError(f"{output} has non-integer {id_column} values")
    frame[id_column] = ids.astype(int)
    if frame[id_column].duplicated().any():
        raise ValueError(f"{output} contains duplicate {id_column} values")
    observed = set(frame[id_column])
    if not observed <= expected_ids:
        raise ValueError(f"{output} contains unexpected IDs: {sorted(observed - expected_ids)}")
    if frame["text"].isna().any() or not frame["text"].astype(str).str.strip().all():
        raise ValueError(f"{output} contains empty text")
    return frame.sort_values(id_column).reset_index(drop=True)


def completed_ids(frame: pd.DataFrame, id_column: str) -> set[int]:
    """IDs already present in a resumable output."""
    return set(frame[id_column].astype(int)) if not frame.empty else set()


def generate_with_attempts(
    output: Path,
    row_id: int,
    maximum: int,
    call: Callable[[], GenerationResult],
) -> GenerationResult:
    """Return one accepted response, making at most ``maximum`` calls per ID."""
    info_path = generation_info_path(output)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    attempts = info.setdefault("attempts", {})
    used = int(attempts.get(str(row_id), 0))
    if used < 0:
        raise ValueError(f"{info_path} has an invalid attempt count for id={row_id}")
    for attempt in range(used + 1, maximum + 1):
        attempts[str(row_id)] = attempt
        write_json(info_path, info)
        try:
            result = call()
            if not result.text.strip():
                raise RuntimeError("empty text")
            if result.finish_reason != "stop":
                raise RuntimeError(f"finish_reason={result.finish_reason!r}")
            if not result.model or not result.model.strip():
                raise RuntimeError("response model is missing")
            return result
        except RuntimeError as exc:
            print(f"id={row_id} attempt {attempt}/{maximum} failed: {exc}")
    raise RuntimeError(f"id={row_id} exhausted its {maximum} generation attempts")


def finish_generation(output: Path, completed: set[int], expected: set[int]) -> None:
    """Record coverage in the journal after a stage run."""
    info_path = generation_info_path(output)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info.update({"n_completed": len(completed), "complete": completed == expected})
    write_json(info_path, info)


# --- generated pools ----------------------------------------------------------------


def reset_failed_attempts(output: Path, id_column: str) -> int:
    """Clear the retry budget of IDs that never produced a row, so they can be retried.

    A per-ID budget is spent permanently, so an ID that burned its attempts on a
    transient outage would otherwise fail every later run without calling.
    """
    info_path = generation_info_path(output)
    if not output.exists() or not info_path.exists():
        return 0
    info = json.loads(info_path.read_text(encoding="utf-8"))
    attempts = info.get("attempts", {})
    if not attempts:
        return 0
    written = {str(row_id) for row_id in completed_ids(pd.read_csv(output), id_column)}
    stale = [row_id for row_id in attempts if row_id not in written]
    for row_id in stale:
        del attempts[row_id]
    if stale:
        info["attempts"] = attempts
        write_json(info_path, info)
    return len(stale)


def pool_contract(
    config: Mapping[str, Any], kind: str
) -> tuple[Path, list[str], str, set[int], str]:
    """Output path, schema, ID column, expected IDs, and input digest of a pool."""
    if kind == "templates":
        output = Path(config["paths"]["templates"])
        schema = ["template_id", "seed_id", "text"]
        id_column = "template_id"
        count = int(config["generation"]["n_templates"])
        files = {
            "source": Path(config["pools"]["seed_statements"]),
            "prompt": Path(config["prompts"]["templates"]),
        }
        settings = {"stage": kind}
    elif kind == "personas":
        output = Path(config["paths"]["personas"])
        schema = ["persona_id", "job_title", "text"]
        id_column = "persona_id"
        count = int(config["generation"]["n_personas"])
        files = {
            "source": Path(config["pools"]["job_titles"]),
            "prompt": Path(config["prompts"]["personas"]),
        }
        settings = {"stage": kind, "seed": config.get("seed")}
    else:
        raise ValueError(f"unknown generated pool {kind!r}")
    settings["endpoint"] = config.get("llm", {}).get("base_url", DEFAULT_BASE_URL)
    return output, schema, id_column, set(range(1, count + 1)), input_digest(settings, files)


def load_pool(config: Mapping[str, Any], kind: str) -> pd.DataFrame:
    """Load a complete generated pool, failing if it still has gaps."""
    output, schema, id_column, expected, digest = pool_contract(config, kind)
    frame = prepare_generated_csv(output, schema, id_column, expected, digest, create=False)
    if set(frame[id_column]) != expected:
        raise ValueError(f"{kind} pool is incomplete; run generate-{kind}")
    if kind == "personas" and (
        frame["job_title"].isna().any() or not frame["job_title"].astype(str).str.strip().all()
    ):
        raise ValueError("persona job titles must be non-empty")
    return frame
