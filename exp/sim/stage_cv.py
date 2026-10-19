"""Stages ``generate-texts``, ``generate-counterfactual-texts`` and ``generate-floor-texts``.

All worlds run through one routine: the factual pass renders X for every unit,
the counterfactual pass always renders test X' and optionally includes train X'.
Structured identities are copied from the factual file without an API call. The
floor pass (per-unit designs only) re-renders the factual state of the flagged
evaluation units with the next option of the ``design.floor_swap`` choices column,
for the noise floor of latent proximity. The held-out pass re-renders the factual
state of every test (evaluation) unit with ``codebook_held_out`` (V_B) and the same
render plan, for the reader robustness check (a seeded ``held_out_fraction`` of them).
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
    validate_generation_settings,
    prepare_generated_csv,
    resolve_llm,
)
from exp.sim.design import floor_ids, load_design
from exp.sim.render import (
    RenderContext,
    floor_digest,
    generation_digest,
    held_out_config,
    held_out_digest,
    held_out_ids,
    render_context,
    validate_grounding,
)
from exp.sim.text_length import length_rejection, load_token_counter


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
        concrete = {column: int(factual_row[ctx.headers[column]]) for column in ctx.sampled}
        _append_cv_row(
            output,
            ctx.cv_row(row_id, concrete, str(factual_row["text"]), None, "identity_copy"),
        )
        copied.add(row_id)
    return copied


def floor_swap(config: Mapping[str, Any]) -> str:
    """The choices column swapped for noise-floor texts; fails without a floor design."""
    spec = load_design(config)
    if spec is None or spec.floor_swap is None:
        raise ValueError("noise-floor texts need a design: block with floor_swap")
    return spec.floor_swap


def _generate_cv(config: dict[str, Any], world: str) -> None:
    validate_generation_settings(config)  # before rows are written or calls are billed
    counterfactual = world == "counterfactual"
    swap = floor_swap(config) if world == "floor" else None
    if world == "held_out":
        # the factual file (production codebook) must be complete and grounded first
        _load_factual_text(render_context(config))
        ctx = render_context(held_out_config(config))
    else:
        ctx = render_context(config)
    if swap is not None and swap not in ctx.spec["choices"]:
        raise ValueError(f"design.floor_swap {swap!r} must be a codebook choices column")
    prompt, prompt_templates = _cv_prompt(config)
    maximum = max_generation_attempts(config)

    factual_text = _load_factual_text(ctx) if counterfactual else None
    if world == "floor":
        expected_ids = floor_ids(config)
        output = Path(config["paths"]["texts_floor"])
        digest = floor_digest(config)
    elif world == "held_out":
        expected_ids = held_out_ids(config, ctx.test_ids)
        output = Path(config["paths"]["texts_held_out"])
        digest = held_out_digest(config)
    else:
        expected_ids = ctx.counterfactual_text_ids if counterfactual else ctx.all_ids
        output = ctx.output_path(counterfactual)
        digest = generation_digest(config, counterfactual)

    existing = prepare_generated_csv(
        output,
        ctx.schema,
        "id",
        expected_ids,
        digest,
        create=True,
        allow_extension=world == "held_out",  # raising held_out_fraction extends the subset
    )
    validate_grounding(
        ctx,
        existing,
        counterfactual=counterfactual,
        factual_text=factual_text,
        swap=swap,
    )
    done = completed_ids(existing, "id")
    missing = sorted(expected_ids - done)

    if counterfactual:
        done |= _copy_identities(ctx, output, factual_text, missing)
        missing = [unit_id for unit_id in missing if unit_id not in done]

    selected = missing[: generation_limit(config, len(missing))]
    try:
        if selected:
            base_url, api_key = resolve_llm(config)
            budget = load_token_counter(config)
            reject = length_rejection(*budget) if budget is not None else None
            print(f"generating {len(selected)} {world} CVs")
            for position, row_id in enumerate(selected, 1):
                sample, concrete = ctx.render_inputs(row_id, counterfactual, swap)
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
                    reject,
                )
                _append_cv_row(
                    output,
                    ctx.cv_row(row_id, concrete, result.text, result, "generated"),
                )
                done.add(row_id)
                if position % 25 == 0 or position == len(selected):
                    print(f"  {position}/{len(selected)}")
    finally:
        # rows already on disk must be reflected in the journal even if a row failed
        finish_generation(output, done, expected_ids)
    print(f"{len(done)}/{len(expected_ids)} rows available in {output}")


def stage_generate_texts(config: dict[str, Any]) -> None:
    """Generate factual X for every unit."""
    _generate_cv(config, "factual")


def stage_generate_counterfactual_texts(config: dict[str, Any]) -> None:
    """Generate configured X' rows, copying structured identities for free."""
    _generate_cv(config, "counterfactual")


def stage_generate_floor_texts(config: dict[str, Any]) -> None:
    """Generate the noise-floor texts: unchanged state, swapped choice option."""
    _generate_cv(config, "floor")


def stage_generate_held_out_texts(config: dict[str, Any]) -> None:
    """Generate the test units' factual texts in the held-out (V_B) codebook."""
    _generate_cv(config, "held_out")
