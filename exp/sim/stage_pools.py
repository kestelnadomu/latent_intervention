"""Stages ``generate-templates`` and ``generate-personas``: the billed text pools.

Both stages fill a resumable CSV one row at a time from a single-prompt file;
they differ only in how a row's prompt sample and CSV prefix are built.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from exp.sim.generate_text import generate_text_result, load_prompts
from exp.sim.helpers import (
    completed_ids,
    finish_generation,
    generate_with_attempts,
    generation_limit,
    load_string_list,
    max_generation_attempts,
    pool_contract,
    prepare_generated_csv,
    resolve_llm,
)

RNG_PERSONAS = 1


def _generate_rows(
    config: Mapping[str, Any],
    out_path: Path,
    todo: list[tuple[dict[str, Any], list[Any]]],
    prompt_path: str | Path,
    label: str,
) -> set[int]:
    """Generate and append the pending rows of a pool, respecting the billing cap."""
    maximum = max_generation_attempts(config)
    todo = todo[: generation_limit(config, len(todo))]
    if not todo:
        return set()
    prompts = load_prompts(prompt_path)
    if len(prompts["prompts"]) != 1:
        raise ValueError(f"{label} prompt file must define exactly one prompt")
    base_url, api_key = resolve_llm(config)
    written: set[int] = set()
    with open(out_path, "a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        for position, (sample, prefix) in enumerate(todo, 1):
            row_id = int(prefix[0])
            result = generate_with_attempts(
                out_path,
                row_id,
                maximum,
                lambda sample=sample: generate_text_result(
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
    seeds = load_string_list(config["pools"]["seed_statements"], "Seed statements")
    out_path, schema, id_column, expected, digest = pool_contract(config, "templates")
    existing = prepare_generated_csv(out_path, schema, id_column, expected, digest, create=True)
    done = completed_ids(existing, id_column)
    todo = [
        (
            {"sampled_statement": seeds[(template_id - 1) % len(seeds)]},
            [template_id, (template_id - 1) % len(seeds) + 1],
        )
        for template_id in sorted(expected)
        if template_id not in done
    ]
    done |= _generate_rows(config, out_path, todo, config["prompts"]["templates"], "templates")
    finish_generation(out_path, done, expected)
    print(f"{len(done)}/{len(expected)} templates available in {out_path}")


def stage_generate_personas(config: dict[str, Any]) -> None:
    """Generate the configured persona pool."""
    titles = load_string_list(config["pools"]["job_titles"], "Job titles")
    out_path, schema, id_column, expected, digest = pool_contract(config, "personas")
    existing = prepare_generated_csv(out_path, schema, id_column, expected, digest, create=True)
    done = completed_ids(existing, id_column)
    todo = []
    for persona_id in sorted(expected):
        if persona_id in done:
            continue
        rng = np.random.default_rng([int(config["seed"]), RNG_PERSONAS, persona_id])
        title = titles[int(rng.integers(len(titles)))]
        todo.append(({"job_title": title}, [persona_id, title]))
    done |= _generate_rows(config, out_path, todo, config["prompts"]["personas"], "personas")
    finish_generation(out_path, done, expected)
    print(f"{len(done)}/{len(expected)} personas available in {out_path}")
