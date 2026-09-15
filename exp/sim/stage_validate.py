"""Stage ``validate-pairs``: the end-to-end paired-data contract.

``render_context`` and ``validate_grounding`` already re-derive the split and
check each file against the render plan; this stage adds coverage and the
cross-world checks that only make sense with X and X' side by side, and reports
texts that use a codebook ``forbidden_terms`` entry.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from exp.sim.helpers import prepare_generated_csv
from exp.sim.render import generation_digest, render_context, validate_grounding


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
    print(
        f"validated {len(all_ids)} factual units and "
        f"{len(counterfactual_ids)} counterfactual pairs"
    )
