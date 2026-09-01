"""Stage ``reset-attempts``: make a spent retry budget recoverable.

``generation.max_attempts`` is counted across resumes, so an ID that burned its
attempts on a transient outage fails every later run without making a call. This
stage clears the budget of IDs that never produced a row, leaving written rows
and every digest untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from exp.sim.helpers import pool_contract, reset_failed_attempts


def stage_reset_attempts(config: dict[str, Any]) -> None:
    """Clear the retry budget of every ID that is still missing from its output."""
    outputs: list[tuple[Path, str]] = []
    for kind in ("templates", "personas"):
        output, _, id_column, _, _ = pool_contract(config, kind)
        outputs.append((output, id_column))
    outputs.append((Path(config["paths"]["texts"]), "id"))
    outputs.append((Path(config["paths"]["texts_counterfactual"]), "id"))

    total = 0
    for output, id_column in outputs:
        cleared = reset_failed_attempts(output, id_column)
        total += cleared
        if cleared:
            print(f"cleared {cleared} spent attempt budgets in {output}")
    print(f"{total} IDs can be retried; re-run the stage that owns them")
