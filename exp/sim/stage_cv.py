"""Stages ``generate-texts`` and ``generate-counterfactual-texts``.

Both worlds run through one routine: the factual pass renders X for every unit,
the counterfactual pass renders X' for test units only and copies structured
identities from the factual file without an API call.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from exp.sim.generate_text import generate_text_result, load_prompts
from exp.sim.helpers import (
    completed_ids,
    finish_generation,
    generate_with_attempts,
    generation_limit,
    max_generation_attempts,
    prepare_generated_csv,
    resolve_llm,
)
from exp.sim.render import (
    RenderContext,
    generation_digest,
    render_context,
    validate_grounding,
)


def _cv_prompt(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """The single CV prompt, which must be sampled deterministically."""
    prompts = load_prompts(config["prompts"]["cv"])
    if len(prompts["prompts"]) != 1:
        raise ValueError("CV prompt file must define exactly one prompt")
    prompt = prompts["prompts"][0]
    if float(prompt.get("metadata", {}).get("temperature", 0.7)) != 0:
        raise ValueError("paired CV generation requires temperature: 0")
    return prompt, prompts["templates"]


def _append_cv_row(output: Path, row: list[Any]) -> None:
    with open(output, "a", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(row)
        stream.flush()


def _load_factual_text(ctx: RenderContext):
    """Load the complete, grounded factual file the counterfactual pass builds on."""
    factual_text = prepare_generated_csv(
        ctx.output_path(False),
        ctx.schema,
        "id",
        ctx.all_ids,
        generation_digest(ctx.config, False),
        create=False,
    )
    if set(factual_text["id"]) != ctx.all_ids:
        raise ValueError("factual texts must be complete before generating X'")
    validate_grounding(ctx, factual_text, counterfactual=False)
    return factual_text


def _copy_identities(
    ctx: RenderContext, output: Path, factual_text, missing: list[int]
) -> set[int]:
    """Copy X to X' for units the intervention leaves structurally unchanged."""
    factual_by_id = factual_text.set_index("id")
    copied: set[int] = set()
    for row_id in missing:
        if not ctx.pairs_by_id.at[row_id, "is_identity"]:
            continue
        factual_row = factual_by_id.loc[row_id]
        concrete = {column: int(factual_row[ctx.headers[column]]) for column in ctx.binned}
        _append_cv_row(
            output,
            ctx.cv_row(row_id, concrete, str(factual_row["text"]), None, "identity_copy"),
        )
        copied.add(row_id)
    return copied


def _generate_cv(config: dict[str, Any], *, counterfactual: bool) -> None:
    ctx = render_context(config)
    prompt, prompt_templates = _cv_prompt(config)
    generation_limit(config, 0)  # validate before identity rows or billed calls
    maximum = max_generation_attempts(config)

    factual_text = _load_factual_text(ctx) if counterfactual else None
    expected_ids = ctx.test_ids if counterfactual else ctx.all_ids
    output = ctx.output_path(counterfactual)

    existing = prepare_generated_csv(
        output,
        ctx.schema,
        "id",
        expected_ids,
        generation_digest(config, counterfactual),
        create=True,
    )
    validate_grounding(
        ctx,
        existing,
        counterfactual=counterfactual,
        factual_text=factual_text,
    )
    done = completed_ids(existing, "id")
    missing = sorted(expected_ids - done)

    if counterfactual:
        done |= _copy_identities(ctx, output, factual_text, missing)
        missing = [unit_id for unit_id in missing if unit_id not in done]

    selected = missing[: generation_limit(config, len(missing))]
    if selected:
        base_url, api_key = resolve_llm(config)
        world = "counterfactual" if counterfactual else "factual"
        print(f"generating {len(selected)} {world} CVs")
        for position, row_id in enumerate(selected, 1):
            sample, concrete = ctx.render_inputs(row_id, counterfactual)
            result = generate_with_attempts(
                output,
                row_id,
                maximum,
                lambda sample=sample: generate_text_result(
                    sample,
                    prompt,
                    prompt_templates,
                    api_key=api_key,
                    base_url=base_url,
                ),
            )
            _append_cv_row(
                output,
                ctx.cv_row(row_id, concrete, result.text, result, "generated"),
            )
            done.add(row_id)
            if position % 25 == 0 or position == len(selected):
                print(f"  {position}/{len(selected)}")
    finish_generation(output, done, expected_ids)
    print(f"{len(done)}/{len(expected_ids)} rows available in {output}")


def stage_generate_texts(config: dict[str, Any]) -> None:
    """Generate factual X for every unit."""
    _generate_cv(config, counterfactual=False)


def stage_generate_counterfactual_texts(config: dict[str, Any]) -> None:
    """Generate X' for test units, copying structured identities for free."""
    _generate_cv(config, counterfactual=True)
