"""SCM simulation and LIBERTy-style factual/counterfactual CV generation.

Stages:
    simulate                       S and S' for every unit, plus a train/test index
    generate-templates             narrative template pool
    generate-personas              persona pool
    generate-texts                 factual X for every unit
    generate-counterfactual-texts  X' for test units only
    validate-pairs                 validate the complete paired-data contract

Generated CVs share one persisted render plan (template, persona, and one
quantile per binned field). Identity counterfactuals are copied without an API
call. Billed stages append one row at a time and resume from IDs already present.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from exp.sim.generate_text import (
    DEFAULT_BASE_URL,
    GenerationResult,
    generate_text_result,
    load_codebook_spec,
    load_prompts,
)
from exp.sim.pairing import (
    PairingError,
    build_pair_index,
    build_render_plan,
    checked_ids,
    materialize_binned_values,
)

CONFIG_PATH = Path(__file__).parent / "config.yaml"
RNG_PERSONAS = 1
RENDERER_VERSION = "liberty-fixed-context-v1"
RESPONSE_COLUMNS = [
    "response_id",
    "model",
    "system_fingerprint",
    "finish_reason",
    "generation_mode",
]


def load_sim_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the data-generation configuration."""
    with open(Path(path), "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _state_columns(config: Mapping[str, Any]) -> list[str]:
    return [str(column) for column in config["schema"]["columns"]]


def _read_csv(
    path: str | Path,
    label: str,
    required: Iterable[str] = (),
) -> pd.DataFrame:
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(Path(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(text: str) -> str:
    return "_".join("".join(char if char.isalnum() else " " for char in text).split()).lower()


def _coerce_bool(value: Any, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{label} must be true/false, got {value!r}")


def _generation_limit(config: Mapping[str, Any], count: int) -> int:
    limit = config.get("generation", {}).get("limit")
    if limit is None:
        return count
    limit = int(limit)
    if limit < 0:
        raise ValueError("generation.limit must be non-negative or null")
    return min(limit, count)


def _max_generation_attempts(config: Mapping[str, Any]) -> int:
    attempts = int(config.get("generation", {}).get("max_attempts", 3))
    if not 1 <= attempts <= 3:
        raise ValueError("generation.max_attempts must be between 1 and 3")
    return attempts


def _input_digest(settings: Mapping[str, Any], files: Mapping[str, Path]) -> str:
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8"))
    for name, path in sorted(files.items()):
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _generation_info_path(output: Path) -> Path:
    return output.with_suffix(".generation.json")


def _archive_existing(paths: Iterable[str | Path]) -> None:
    """Move existing run artifacts aside using one date-only suffix."""
    existing = list(dict.fromkeys(Path(path) for path in paths if Path(path).exists()))
    if not existing:
        return
    stamp = datetime.now().strftime("%y-%m-%d")
    destinations = {}
    for path in existing:
        suffixes = "".join(path.suffixes)
        basename = path.name[: -len(suffixes)] if suffixes else path.name
        destinations[path] = path.with_name(f"{basename}_{stamp}{suffixes}")
    collisions = [destination for destination in destinations.values() if destination.exists()]
    if collisions:
        raise FileExistsError(f"dated archive already exists: {collisions[0]}")
    for path, destination in destinations.items():
        path.rename(destination)
        print(f"archived {path} -> {destination}")


def _prepare_generated_csv(
    output: Path,
    schema: list[str],
    id_column: str,
    expected_ids: set[int],
    digest: str,
    *,
    create: bool,
) -> pd.DataFrame:
    """Validate a resumable billed output before any API call."""
    info_path = _generation_info_path(output)
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
        _write_json(
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


def _generate_with_attempts(
    output: Path,
    row_id: int,
    maximum: int,
    call: Callable[[], GenerationResult],
) -> GenerationResult:
    """Return one accepted response, making at most ``maximum`` calls per ID."""
    info_path = _generation_info_path(output)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    attempts = info.setdefault("attempts", {})
    used = int(attempts.get(str(row_id), 0))
    if used < 0:
        raise ValueError(f"{info_path} has an invalid attempt count for id={row_id}")
    for attempt in range(used + 1, maximum + 1):
        attempts[str(row_id)] = attempt
        _write_json(info_path, info)
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


def _finish_generation(output: Path, completed: set[int], expected: set[int]) -> None:
    info_path = _generation_info_path(output)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info.update({"n_completed": len(completed), "complete": completed == expected})
    _write_json(info_path, info)


# --- structured simulation ---------------------------------------------------------


def _simulation_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    sim_dir = Path(config["paths"]["sim_dir"])
    return {
        "factual": sim_dir / "sim_data_factual.csv",
        "counterfactual": sim_dir / "sim_data_counterfactual.csv",
        "epsilon": sim_dir / "sim_data_epsilon.csv",
        "pair_index": Path(config["paths"]["pair_index"]),
    }


def _simulation_record(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = _simulation_paths(config)
    return {
        "n": int(config["n"]),
        "simulation_seed": config.get("seed"),
        "split_seed": int(config["split"]["seed"]),
        "test_fraction": float(config["split"].get("test_fraction", 0.2)),
        "intervention": config["intervention"],
        "schema": config["schema"],
        "objects": config["objects"],
        "files": {name: _sha256(path) for name, path in paths.items()},
    }


def _validate_fixed_query(
    factual: pd.DataFrame,
    counterfactual: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    columns = _state_columns(config)
    intervention = {str(key): int(value) for key, value in config["intervention"].items()}
    if not set(intervention) <= set(columns):
        raise ValueError("the fixed text-rendering intervention must target schema.columns")
    for column, target in intervention.items():
        if not counterfactual[column].eq(target).all():
            raise ValueError(f"counterfactual {column} does not equal target {target}")
    target_rows = pd.Series(True, index=factual.index)
    for column, target in intervention.items():
        target_rows &= factual[column].eq(target)
    if not factual.loc[target_rows, columns].equals(counterfactual.loc[target_rows, columns]):
        raise ValueError("units already at all intervention targets must be structured identities")


def stage_simulate(config: dict[str, Any]) -> None:
    """Generate aligned S/S' for all units and a seeded test index."""
    from src.schema import load_object

    factory = load_object(config["objects"]["scm"])
    scm = factory(config)
    factual, counterfactual, epsilon = scm.simulate(
        n=int(config["n"]),
        intervention=config["intervention"],
        seed=config.get("seed"),
    )
    factual = checked_ids(factual, "factual simulation")
    counterfactual = checked_ids(counterfactual, "counterfactual simulation")
    epsilon = checked_ids(epsilon, "simulation noise")
    expected_ids = factual["id"].tolist()
    if (
        len(expected_ids) != int(config["n"])
        or counterfactual["id"].tolist() != expected_ids
        or epsilon["id"].tolist() != expected_ids
    ):
        raise PairingError("configured SCM must return n aligned factual/counterfactual/noise IDs")
    _validate_fixed_query(factual, counterfactual, config)
    pair_index = build_pair_index(
        factual,
        counterfactual,
        _state_columns(config),
        seed=int(config["split"]["seed"]),
        test_fraction=float(config["split"].get("test_fraction", 0.2)),
    )

    sim_dir = Path(config["paths"]["sim_dir"])
    text_outputs = [
        Path(config["paths"]["texts"]),
        Path(config["paths"]["texts_counterfactual"]),
    ]
    archive_paths = [
        *_simulation_paths(config).values(),
        sim_dir / "simulation_info.json",
        Path(config["paths"]["render_plan"]),
    ]
    for output in text_outputs:
        archive_paths.extend([output, _generation_info_path(output)])
    _archive_existing(archive_paths)

    sim_dir.mkdir(parents=True, exist_ok=True)
    outputs = _simulation_paths(config)
    for name, frame in (
        ("factual", factual),
        ("counterfactual", counterfactual),
        ("epsilon", epsilon),
        ("pair_index", pair_index),
    ):
        outputs[name].parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(outputs[name], index=False)
        print(f"wrote {outputs[name]} ({len(frame)} rows)")

    _write_json(
        sim_dir / "simulation_info.json",
        _simulation_record(config),
    )
    changed = ~pair_index["is_identity"]
    print(f"intervention {config['intervention']}: {changed.mean():.1%} of units changed")


# --- reusable billed generation ----------------------------------------------------


def _load_string_list(path: str | Path, label: str) -> list[str]:
    with open(Path(path), "r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(value, str) and value.strip() for value in values)
    ):
        raise ValueError(f"{label} must be a non-empty YAML list of strings: {path}")
    return values


def _resolve_llm(config: Mapping[str, Any]) -> tuple[str, str | None]:
    llm = config.get("llm", {})
    base_url = llm.get("base_url", DEFAULT_BASE_URL)
    if "<resource-name>" in base_url:
        raise ValueError("Fill in llm.base_url in exp/sim/config.yaml")
    env_name = llm.get("api_key_env")
    api_key = os.getenv(env_name) if env_name else None
    if env_name and not api_key:
        raise ValueError(f"Environment variable {env_name} is not set")
    return base_url, api_key


def _pool_contract(
    config: Mapping[str, Any], kind: str
) -> tuple[Path, list[str], str, set[int], str]:
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
    return output, schema, id_column, set(range(1, count + 1)), _input_digest(settings, files)


def _generate_rows(
    config: Mapping[str, Any],
    out_path: Path,
    todo: list[tuple[dict[str, Any], list[Any]]],
    prompt_path: str | Path,
    label: str,
) -> set[int]:
    maximum = _max_generation_attempts(config)
    todo = todo[: _generation_limit(config, len(todo))]
    if not todo:
        return set()
    prompts = load_prompts(prompt_path)
    if len(prompts["prompts"]) != 1:
        raise ValueError(f"{label} prompt file must define exactly one prompt")
    base_url, api_key = _resolve_llm(config)
    written: set[int] = set()
    with open(out_path, "a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        for position, (sample, prefix) in enumerate(todo, 1):
            row_id = int(prefix[0])
            result = _generate_with_attempts(
                out_path,
                row_id,
                maximum,
                lambda: generate_text_result(
                    sample,
                    prompts["prompts"][0],
                    prompts["templates"],
                    api_key=api_key,
                    base_url=base_url,
                ),
            )
            writer.writerow([*prefix, result.text])
            stream.flush()
            written.add(row_id)
            if position % 25 == 0 or position == len(todo):
                print(f"  {position}/{len(todo)}")
    return written


def stage_generate_templates(config: dict[str, Any]) -> None:
    """Generate the configured narrative-template pool."""
    seeds = _load_string_list(config["pools"]["seed_statements"], "Seed statements")
    out_path, schema, id_column, expected, digest = _pool_contract(config, "templates")
    existing = _prepare_generated_csv(
        out_path, schema, id_column, expected, digest, create=True
    )
    done = set(existing[id_column].astype(int)) if not existing.empty else set()
    todo = [
        (
            {"sampled_statement": seeds[(template_id - 1) % len(seeds)]},
            [template_id, (template_id - 1) % len(seeds) + 1],
        )
        for template_id in sorted(expected)
        if template_id not in done
    ]
    done |= _generate_rows(
        config,
        out_path,
        todo,
        config["prompts"]["templates"],
        "templates",
    )
    _finish_generation(out_path, done, expected)
    print(f"{len(done)}/{len(expected)} templates available in {out_path}")


def stage_generate_personas(config: dict[str, Any]) -> None:
    """Generate the configured persona pool."""
    titles = _load_string_list(config["pools"]["job_titles"], "Job titles")
    out_path, schema, id_column, expected, digest = _pool_contract(config, "personas")
    existing = _prepare_generated_csv(
        out_path, schema, id_column, expected, digest, create=True
    )
    done = set(existing[id_column].astype(int)) if not existing.empty else set()
    todo = []
    for persona_id in sorted(expected):
        if persona_id in done:
            continue
        rng = np.random.default_rng([int(config["seed"]), RNG_PERSONAS, persona_id])
        title = titles[int(rng.integers(len(titles)))]
        todo.append(({"job_title": title}, [persona_id, title]))
    done |= _generate_rows(
        config,
        out_path,
        todo,
        config["prompts"]["personas"],
        "personas",
    )
    _finish_generation(out_path, done, expected)
    print(f"{len(done)}/{len(expected)} personas available in {out_path}")


# --- paired CV rendering ------------------------------------------------------------


def _load_pair_inputs(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sim_dir = Path(config["paths"]["sim_dir"])
    info_path = sim_dir / "simulation_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"simulation information not found: {info_path}; run simulate")
    if json.loads(info_path.read_text(encoding="utf-8")) != _simulation_record(config):
        raise RuntimeError("simulation_info.json is incompatible with the current configuration or files")
    columns = _state_columns(config)
    factual = _read_csv(sim_dir / "sim_data_factual.csv", "factual simulation", columns)
    counterfactual = _read_csv(
        sim_dir / "sim_data_counterfactual.csv", "counterfactual simulation", columns
    )
    pairs = _read_csv(
        config["paths"]["pair_index"], "pair index", ("split", "is_identity")
    )
    ids = factual["id"].tolist()
    if counterfactual["id"].tolist() != ids or pairs["id"].tolist() != ids:
        raise PairingError("S, S', and pair_index.csv must contain exactly the same IDs")
    if len(ids) != int(config["n"]):
        raise PairingError(f"simulation has {len(ids)} units, expected n={config['n']}")
    _validate_fixed_query(factual, counterfactual, config)
    pairs["split"] = pairs["split"].astype(str).str.lower()
    if set(pairs["split"]) != {"train", "test"}:
        raise PairingError("pair index must contain non-empty train and test splits")
    pairs["is_identity"] = [
        _coerce_bool(value, f"pair index id={row_id} is_identity")
        for row_id, value in zip(pairs["id"], pairs["is_identity"])
    ]
    expected_pairs = build_pair_index(
        factual,
        counterfactual,
        columns,
        seed=int(config["split"]["seed"]),
        test_fraction=float(config["split"].get("test_fraction", 0.2)),
    )
    try:
        pd.testing.assert_frame_equal(pairs, expected_pairs, check_dtype=False)
    except AssertionError as exc:
        raise PairingError("pair_index.csv does not match the configured seeded split") from exc
    return factual, counterfactual, pairs


def _load_pool(config: Mapping[str, Any], kind: str) -> pd.DataFrame:
    output, schema, id_column, expected, digest = _pool_contract(config, kind)
    frame = _prepare_generated_csv(
        output, schema, id_column, expected, digest, create=False
    )
    if set(frame[id_column]) != expected:
        raise ValueError(f"{kind} pool is incomplete; run generate-{kind}")
    if kind == "personas" and (
        frame["job_title"].isna().any()
        or not frame["job_title"].astype(str).str.strip().all()
    ):
        raise ValueError("persona job titles must be non-empty")
    return frame


def _load_render_spec(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = load_codebook_spec(config["codebook"])
    cardinalities = {str(k): int(v) for k, v in config["schema"]["columns"].items()}
    if set(spec["columns"]) != set(cardinalities):
        raise ValueError("codebook columns must exactly match schema.columns")
    for column, cardinality in cardinalities.items():
        levels = set(range(cardinality))
        if set(spec["columns"][column]) != levels:
            raise ValueError(f"codebook levels for {column} must be 0..{cardinality - 1}")
        if column in spec["bins"] and set(spec["bins"][column]) != levels:
            raise ValueError(f"bins for {column} must cover every category")
    return spec


def _ensure_render_plan(
    config: Mapping[str, Any],
    pairs: pd.DataFrame,
    templates: pd.DataFrame,
    personas: pd.DataFrame,
    binned: list[str],
) -> pd.DataFrame:
    expected = build_render_plan(
        pairs["id"],
        templates["template_id"],
        personas["persona_id"],
        binned,
        seed=int(config["seed"]),
    )
    path = Path(config["paths"]["render_plan"])
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        expected.to_csv(path, index=False)
        print(f"wrote {path} ({len(expected)} rows)")
        return expected
    observed = _read_csv(path, "render plan", expected.columns.drop("id"))
    try:
        pd.testing.assert_frame_equal(observed, expected, check_dtype=False, atol=0, rtol=0)
    except AssertionError as exc:
        raise RuntimeError("render_plan.csv is incompatible with current inputs") from exc
    return observed


def _candidate_info(
    state: Mapping[str, Any],
    spec: Mapping[str, Any],
    concrete: Mapping[str, int],
) -> str:
    parts = []
    for column, levels in spec["columns"].items():
        category = int(state[column])
        label = spec["labels"].get(column, column)
        value = concrete[column] if column in concrete else levels[category]
        parts.append(f"{label}: {value}")
    return "[" + ", ".join(parts) + "]"


def _render_inputs(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    templates: pd.DataFrame,
    personas: pd.DataFrame,
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    template_id = int(plan["template_id"])
    persona_id = int(plan["persona_id"])
    template = templates.loc[template_id]
    persona = personas.loc[persona_id]
    concrete = materialize_binned_values(state, spec["bins"], plan)
    sample = {
        "cv_template": template["text"],
        "candidate_info": _candidate_info(state, spec, concrete),
        "persona_details": f"Job Title: {persona['job_title']}\n{persona['text']}",
    }
    return sample, concrete


def _cv_schema(spec: Mapping[str, Any]) -> tuple[list[str], dict[str, str]]:
    headers = {
        column: _slug(spec["labels"].get(column, column))
        for column in spec["bins"]
    }
    return [
        "id",
        "template_id",
        "persona_id",
        *headers.values(),
        "text",
        *RESPONSE_COLUMNS,
    ], headers


def _cv_prompt(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    prompts = load_prompts(config["prompts"]["cv"])
    if len(prompts["prompts"]) != 1:
        raise ValueError("CV prompt file must define exactly one prompt")
    prompt = prompts["prompts"][0]
    if float(prompt.get("metadata", {}).get("temperature", 0.7)) != 0:
        raise ValueError("paired CV generation requires temperature: 0")
    return prompt, prompts["templates"]


def _generation_digest(config: Mapping[str, Any], counterfactual: bool) -> str:
    sim_dir = Path(config["paths"]["sim_dir"])
    files = {
        "states": sim_dir / ("sim_data_counterfactual.csv" if counterfactual else "sim_data_factual.csv"),
        "render_plan": Path(config["paths"]["render_plan"]),
        "templates": Path(config["paths"]["templates"]),
        "personas": Path(config["paths"]["personas"]),
        "prompt": Path(config["prompts"]["cv"]),
        "codebook": Path(config["codebook"]),
    }
    if counterfactual:
        files["pair_index"] = Path(config["paths"]["pair_index"])
        files["factual_text"] = Path(config["paths"]["texts"])
    settings = {
        "renderer": RENDERER_VERSION,
        "world": "counterfactual" if counterfactual else "factual",
        "endpoint": config.get("llm", {}).get("base_url", DEFAULT_BASE_URL),
    }
    if counterfactual:
        settings["intervention"] = config["intervention"]
    return _input_digest(settings, files)


def _append_cv_row(output: Path, row: list[Any]) -> None:
    with open(output, "a", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(row)
        stream.flush()


def _cv_row(
    row_id: int,
    plan: Mapping[str, Any],
    concrete: Mapping[str, int],
    binned: list[str],
    text: str,
    result: GenerationResult | None,
    mode: str,
) -> list[Any]:
    return [
        row_id,
        int(plan["template_id"]),
        int(plan["persona_id"]),
        *(concrete[column] for column in binned),
        text,
        result.response_id if result else None,
        result.model if result else None,
        result.system_fingerprint if result else None,
        result.finish_reason if result else None,
        mode,
    ]


def _validate_grounding(
    output: pd.DataFrame,
    states: pd.DataFrame,
    pairs: pd.DataFrame,
    plan: pd.DataFrame,
    templates: pd.DataFrame,
    personas: pd.DataFrame,
    spec: Mapping[str, Any],
    headers: Mapping[str, str],
    *,
    counterfactual: bool,
    factual_text: pd.DataFrame | None = None,
) -> None:
    if output.empty:
        return
    states = states.set_index("id")
    pairs = pairs.set_index("id")
    plan = plan.set_index("id")
    factual_by_id = factual_text.set_index("id") if factual_text is not None else None
    for row in output.to_dict(orient="records"):
        row_id = int(row["id"])
        plan_row = plan.loc[row_id]
        _, concrete = _render_inputs(
            states.loc[row_id], plan_row, templates, personas, spec
        )
        if int(row["template_id"]) != int(plan_row["template_id"]):
            raise ValueError(f"incorrect template for id={row_id}")
        if int(row["persona_id"]) != int(plan_row["persona_id"]):
            raise ValueError(f"incorrect persona for id={row_id}")
        for column, value in concrete.items():
            if int(row[headers[column]]) != value:
                raise ValueError(f"incorrect concrete {column} for id={row_id}")
        identity = bool(pairs.at[row_id, "is_identity"])
        expected_mode = "identity_copy" if counterfactual and identity else "generated"
        if row["generation_mode"] != expected_mode:
            raise ValueError(f"incorrect generation mode for id={row_id}")
        if expected_mode == "generated":
            if row["finish_reason"] != "stop":
                raise ValueError(f"generated text did not finish normally for id={row_id}")
            if pd.isna(row["model"]) or not str(row["model"]).strip():
                raise ValueError(f"generated text has no response model for id={row_id}")
        else:
            if factual_by_id is None or row["text"] != factual_by_id.at[row_id, "text"]:
                raise ValueError(f"identity counterfactual text differs for id={row_id}")
            if any(pd.notna(row[column]) for column in RESPONSE_COLUMNS[:-1]):
                raise ValueError(f"identity counterfactual has response metadata for id={row_id}")


def _generate_cv(config: dict[str, Any], *, counterfactual: bool) -> None:
    factual, counterfactual_states, pairs = _load_pair_inputs(config)
    states = counterfactual_states if counterfactual else factual
    spec = _load_render_spec(config)
    binned = list(spec["bins"])
    schema, headers = _cv_schema(spec)
    templates = _load_pool(config, "templates")
    personas = _load_pool(config, "personas")
    plan = _ensure_render_plan(config, pairs, templates, personas, binned)
    templates = templates.set_index("template_id")
    personas = personas.set_index("persona_id")
    prompt, prompt_templates = _cv_prompt(config)
    _generation_limit(config, 0)  # validate before identity rows or billed calls
    maximum = _max_generation_attempts(config)

    all_ids = set(pairs["id"])
    factual_text: pd.DataFrame | None = None
    if counterfactual:
        factual_path = Path(config["paths"]["texts"])
        factual_text = _prepare_generated_csv(
            factual_path,
            schema,
            "id",
            all_ids,
            _generation_digest(config, False),
            create=False,
        )
        if set(factual_text["id"]) != all_ids:
            raise ValueError("factual texts must be complete before generating X'")
        _validate_grounding(
            factual_text,
            factual,
            pairs,
            plan,
            templates,
            personas,
            spec,
            headers,
            counterfactual=False,
        )
        expected_ids = set(pairs.loc[pairs["split"] == "test", "id"])
        output = Path(config["paths"]["texts_counterfactual"])
    else:
        expected_ids = all_ids
        output = Path(config["paths"]["texts"])

    existing = _prepare_generated_csv(
        output,
        schema,
        "id",
        expected_ids,
        _generation_digest(config, counterfactual),
        create=True,
    )
    _validate_grounding(
        existing,
        states,
        pairs,
        plan,
        templates,
        personas,
        spec,
        headers,
        counterfactual=counterfactual,
        factual_text=factual_text,
    )
    done = set(existing["id"].astype(int)) if not existing.empty else set()
    missing = sorted(expected_ids - done)
    state_by_id = states.set_index("id")
    pair_by_id = pairs.set_index("id")
    plan_by_id = plan.set_index("id")

    if counterfactual:
        factual_by_id = factual_text.set_index("id")  # type: ignore[union-attr]
        for row_id in [unit_id for unit_id in missing if pair_by_id.at[unit_id, "is_identity"]]:
            factual_row = factual_by_id.loc[row_id]
            concrete = {column: int(factual_row[headers[column]]) for column in binned}
            _append_cv_row(
                output,
                _cv_row(
                    row_id,
                    plan_by_id.loc[row_id],
                    concrete,
                    binned,
                    str(factual_row["text"]),
                    None,
                    "identity_copy",
                ),
            )
            done.add(row_id)
        missing = [unit_id for unit_id in missing if unit_id not in done]

    selected = missing[: _generation_limit(config, len(missing))]
    if selected:
        base_url, api_key = _resolve_llm(config)
        print(f"generating {len(selected)} {'counterfactual' if counterfactual else 'factual'} CVs")
        for position, row_id in enumerate(selected, 1):
            plan_row = plan_by_id.loc[row_id]
            sample, concrete = _render_inputs(
                state_by_id.loc[row_id], plan_row, templates, personas, spec
            )
            result = _generate_with_attempts(
                output,
                row_id,
                maximum,
                lambda: generate_text_result(
                    sample,
                    prompt,
                    prompt_templates,
                    api_key=api_key,
                    base_url=base_url,
                ),
            )
            _append_cv_row(
                output,
                _cv_row(
                    row_id,
                    plan_row,
                    concrete,
                    binned,
                    result.text,
                    result,
                    "generated",
                ),
            )
            done.add(row_id)
            if position % 25 == 0 or position == len(selected):
                print(f"  {position}/{len(selected)}")
    _finish_generation(output, done, expected_ids)
    print(f"{len(done)}/{len(expected_ids)} rows available in {output}")


def stage_generate_texts(config: dict[str, Any]) -> None:
    """Generate factual X for every unit."""
    _generate_cv(config, counterfactual=False)


def stage_generate_counterfactual_texts(config: dict[str, Any]) -> None:
    """Generate X' for test units, copying structured identities for free."""
    _generate_cv(config, counterfactual=True)


def stage_validate_pairs(config: dict[str, Any]) -> None:
    """Validate coverage, grounding, and identity-copy semantics end to end."""
    factual, counterfactual, pairs = _load_pair_inputs(config)
    spec = _load_render_spec(config)
    binned = list(spec["bins"])
    schema, headers = _cv_schema(spec)
    templates = _load_pool(config, "templates")
    personas = _load_pool(config, "personas")
    plan = _ensure_render_plan(config, pairs, templates, personas, binned)
    templates = templates.set_index("template_id")
    personas = personas.set_index("persona_id")
    all_ids = set(pairs["id"])
    test_ids = set(pairs.loc[pairs["split"] == "test", "id"])
    factual_text = _prepare_generated_csv(
        Path(config["paths"]["texts"]),
        schema,
        "id",
        all_ids,
        _generation_digest(config, False),
        create=False,
    )
    counterfactual_text = _prepare_generated_csv(
        Path(config["paths"]["texts_counterfactual"]),
        schema,
        "id",
        test_ids,
        _generation_digest(config, True),
        create=False,
    )
    if set(factual_text["id"]) != all_ids:
        raise ValueError("factual text coverage/schema is incomplete")
    if set(counterfactual_text["id"]) != test_ids:
        raise ValueError("counterfactual text coverage must equal test IDs exactly")

    _validate_grounding(
        factual_text,
        factual,
        pairs,
        plan,
        templates,
        personas,
        spec,
        headers,
        counterfactual=False,
    )
    _validate_grounding(
        counterfactual_text,
        counterfactual,
        pairs,
        plan,
        templates,
        personas,
        spec,
        headers,
        counterfactual=True,
        factual_text=factual_text,
    )
    factual_by_id = factual_text.set_index("id")
    counterfactual_by_id = counterfactual_text.set_index("id")
    factual_states = factual.set_index("id")
    counterfactual_states = counterfactual.set_index("id")
    for row_id in test_ids:
        if int(factual_by_id.at[row_id, "template_id"]) != int(
            counterfactual_by_id.at[row_id, "template_id"]
        ) or int(factual_by_id.at[row_id, "persona_id"]) != int(
            counterfactual_by_id.at[row_id, "persona_id"]
        ):
            raise ValueError(f"factual/counterfactual context differs for id={row_id}")
        for column in binned:
            if factual_states.at[row_id, column] == counterfactual_states.at[
                row_id, column
            ] and int(factual_by_id.at[row_id, headers[column]]) != int(
                counterfactual_by_id.at[row_id, headers[column]]
            ):
                raise ValueError(f"unchanged bin was rendered differently for id={row_id}")
    print(f"validated {len(all_ids)} factual units and {len(test_ids)} counterfactual test pairs")


STAGES = {
    "simulate": stage_simulate,
    "generate-templates": stage_generate_templates,
    "generate-personas": stage_generate_personas,
    "generate-texts": stage_generate_texts,
    "generate-counterfactual-texts": stage_generate_counterfactual_texts,
    "validate-pairs": stage_validate_pairs,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", default=CONFIG_PATH)
    args = parser.parse_args()
    STAGES[args.stage](load_sim_config(args.config))


if __name__ == "__main__":
    main()
