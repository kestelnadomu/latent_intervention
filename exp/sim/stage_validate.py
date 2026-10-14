"""Stage ``validate-pairs``: the end-to-end paired-data contract.

``render_context`` and ``validate_grounding`` already re-derive the split and
check each file against the render plan; this stage adds coverage and the
cross-world checks that only make sense with X and X' side by side, and reports
texts that use a codebook ``forbidden_terms`` entry, plus texts over the
``text_length`` token budget.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from exp.sim.design import floor_ids, load_design
from exp.sim.helpers import prepare_generated_csv
from exp.sim.render import (
    floor_digest,
    generation_digest,
    held_out_config,
    held_out_digest,
    held_out_ids,
    render_context,
    validate_grounding,
)
from exp.sim.text_length import load_token_counter, report_text_lengths


def report_forbidden_terms(frame: pd.DataFrame, terms: list[str], world: str) -> None:
    """Print which texts use a codebook forbidden term (a report, not a failure)."""
    if not terms or frame.empty:
        return
    pattern = re.compile(r"\b(" + "|".join(re.escape(term) for term in terms) + r")\b", re.IGNORECASE)
    hits = {
        int(row_id): sorted({match.lower() for match in pattern.findall(str(text))})
        for row_id, text in zip(frame["id"], frame["text"])
        if pattern.search(str(text))
    }
    print(f"forbidden terms in {len(hits)}/{len(frame)} {world} texts")
    for row_id, found in sorted(hits.items()):
        print(f"  id={row_id}: {', '.join(found)}")


def stage_validate_pairs(config: dict[str, Any]) -> None:
    """Validate coverage, grounding, and identity-copy semantics end to end."""
    ctx = render_context(config)
    all_ids = ctx.all_ids
    counterfactual_ids = ctx.counterfactual_text_ids
    factual_text = prepare_generated_csv(
        ctx.output_path(False),
        ctx.schema,
        "id",
        all_ids,
        generation_digest(config, False),
        create=False,
    )
    counterfactual_text = prepare_generated_csv(
        ctx.output_path(True),
        ctx.schema,
        "id",
        counterfactual_ids,
        generation_digest(config, True),
        create=False,
    )
    if set(factual_text["id"]) != all_ids:
        raise ValueError("factual text coverage/schema is incomplete")
    if set(counterfactual_text["id"]) != counterfactual_ids:
        raise ValueError("counterfactual text coverage does not match the configured IDs")

    validate_grounding(ctx, factual_text, counterfactual=False)
    validate_grounding(
        ctx,
        counterfactual_text,
        counterfactual=True,
        factual_text=factual_text,
    )

    factual_by_id = factual_text.set_index("id")
    counterfactual_by_id = counterfactual_text.set_index("id")
    for row_id in counterfactual_ids:
        same_context = int(factual_by_id.at[row_id, "template_id"]) == int(
            counterfactual_by_id.at[row_id, "template_id"]
        ) and int(factual_by_id.at[row_id, "persona_id"]) == int(
            counterfactual_by_id.at[row_id, "persona_id"]
        )
        if not same_context:
            raise ValueError(f"factual/counterfactual context differs for id={row_id}")
        for column in ctx.sampled:
            unchanged_state = (
                ctx.factual_by_id.at[row_id, column] == ctx.counterfactual_by_id.at[row_id, column]
            )
            rendered_differently = int(factual_by_id.at[row_id, ctx.headers[column]]) != int(
                counterfactual_by_id.at[row_id, ctx.headers[column]]
            )
            if unchanged_state and rendered_differently:
                raise ValueError(f"unchanged bin was rendered differently for id={row_id}")
    for world, frame in (("factual", factual_text), ("counterfactual", counterfactual_text)):
        report_forbidden_terms(frame, ctx.spec["forbidden_terms"], world)
    budget = load_token_counter(config)
    if budget is not None:
        report_text_lengths(factual_text, *budget, "factual", ctx.factual_by_id)
        report_text_lengths(
            counterfactual_text, *budget, "counterfactual", ctx.counterfactual_by_id
        )
    n_floor = _validate_floor(config, ctx, factual_text, budget)
    n_held_out = _validate_held_out(config, ctx, factual_text, budget)
    floor_note = f" and {n_floor} noise-floor texts" if n_floor else ""
    if n_held_out:
        floor_note += f" and {n_held_out} held-out texts"
    print(
        f"validated {len(all_ids)} factual units and "
        f"{len(counterfactual_ids)} counterfactual pairs{floor_note}"
    )


def _validate_held_out(config, ctx, factual_text: pd.DataFrame, budget) -> int:
    """Coverage and grounding of the held-out (V_B) texts, if configured."""
    if "codebook_held_out" not in config or "texts_held_out" not in config["paths"]:
        return 0
    if not Path(config["paths"]["texts_held_out"]).exists():
        print("held-out texts not generated yet (generate-held-out-texts); skipped")
        return 0
    held_ctx = render_context(held_out_config(config))
    if held_ctx.schema != ctx.schema:
        raise ValueError("the held-out codebook must keep the production CV layout")
    expected = held_out_ids(config, held_ctx.test_ids)
    held_out_text = prepare_generated_csv(
        Path(config["paths"]["texts_held_out"]),
        held_ctx.schema,
        "id",
        expected,
        held_out_digest(config),
        create=False,
    )
    if set(held_out_text["id"]) != expected:
        raise ValueError("held-out text coverage does not match the held-out subset of the test units")
    validate_grounding(held_ctx, held_out_text, counterfactual=False)
    factual_by_id = factual_text.set_index("id")
    columns = ["template_id", "persona_id", *(ctx.headers[c] for c in ctx.sampled)]
    for row in held_out_text.to_dict(orient="records"):
        factual_row = factual_by_id.loc[int(row["id"])]
        if any(int(row[c]) != int(factual_row[c]) for c in columns):
            raise ValueError(f"held-out rendering plan differs from the factual one for id={row['id']}")
    report_forbidden_terms(held_out_text, held_ctx.spec["forbidden_terms"], "held-out")
    if budget is not None:
        report_text_lengths(held_out_text, *budget, "held-out", held_ctx.factual_by_id)
    return len(expected)


def _validate_floor(config, ctx, factual_text: pd.DataFrame, budget) -> int:
    """Coverage and grounding of the noise-floor texts of a per-unit design."""
    spec = load_design(config)
    if spec is None or spec.floor_swap is None:
        return 0
    expected = floor_ids(config)
    floor_text = prepare_generated_csv(
        Path(config["paths"]["texts_floor"]),
        ctx.schema,
        "id",
        expected,
        floor_digest(config),
        create=False,
    )
    if set(floor_text["id"]) != expected:
        raise ValueError("noise-floor text coverage does not match the flagged evaluation units")
    validate_grounding(ctx, floor_text, counterfactual=False, swap=spec.floor_swap)
    header = ctx.headers[spec.floor_swap]
    factual_by_id = factual_text.set_index("id")
    for row in floor_text.to_dict(orient="records"):
        factual_row = factual_by_id.loc[int(row["id"])]
        if (int(row["template_id"]), int(row["persona_id"])) != (
            int(factual_row["template_id"]), int(factual_row["persona_id"])
        ):
            raise ValueError(f"noise-floor context differs from the factual one for id={row['id']}")
        if int(row[header]) == int(factual_row[header]):
            raise ValueError(f"noise-floor text did not swap {spec.floor_swap} for id={row['id']}")
    report_forbidden_terms(floor_text, ctx.spec["forbidden_terms"], "noise-floor")
    if budget is not None:
        report_text_lengths(floor_text, *budget, "noise-floor", ctx.factual_by_id)
    return len(expected)
