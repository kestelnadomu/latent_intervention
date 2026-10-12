"""Token-length budget for generated CV texts.

The LangVAE encoder truncates its input at ``encoder.max_len`` tokens
(src/config.yaml), so a text longer than that is silently cut before encoding.
The ``text_length:`` config section sets a stricter budget, counted with the
tokenizer LangVAE uses. The CV stages reject over-budget responses (the retry
counts against ``generation.max_attempts``), and ``validate-pairs`` reports rows
that exceed it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

TokenCounter = Callable[[str], int]


def load_token_counter(config: Mapping[str, Any]) -> tuple[TokenCounter, int] | None:
    """Token counter and budget from ``text_length:``, or None if the section is absent."""
    settings = config.get("text_length")
    if not settings:
        return None
    max_tokens = int(settings["max_tokens"])
    if max_tokens < 1:
        raise ValueError("text_length.max_tokens must be positive")
    from transformers import AutoTokenizer  # heavy import, only when a budget is set

    tokenizer = AutoTokenizer.from_pretrained(
        settings["tokenizer"], revision=settings.get("tokenizer_revision")
    )
    return (lambda text: len(tokenizer.encode(text, add_special_tokens=False))), max_tokens


def length_rejection(count: TokenCounter, max_tokens: int) -> Callable[[str], str | None]:
    """A response check for ``generate_with_attempts``: a reason if the text is too long."""

    def check(text: str) -> str | None:
        n_tokens = count(text)
        return f"{n_tokens} tokens > {max_tokens}" if n_tokens > max_tokens else None

    return check


def report_text_lengths(
    frame: pd.DataFrame,
    count: TokenCounter,
    max_tokens: int,
    world: str,
    states: pd.DataFrame | None = None,
) -> pd.Series:
    """Print length statistics and over-budget IDs (a report, not a failure).

    With ``states`` (indexed by id), also prints the Spearman correlation of
    length with every state column: length should not carry information about S.
    """
    if frame.empty:
        return pd.Series(dtype=int)
    lengths = pd.Series([count(str(text)) for text in frame["text"]], index=frame["id"])
    over = lengths[lengths > max_tokens]
    print(
        f"{world} token lengths: mean {lengths.mean():.0f}, min {lengths.min()}, "
        f"max {lengths.max()}; {len(over)}/{len(lengths)} over {max_tokens}"
    )
    for row_id, n_tokens in over.items():
        print(f"  id={row_id}: {n_tokens} tokens")
    if states is not None and len(lengths) > 2:
        aligned = states.loc[lengths.index].drop(columns=["id"], errors="ignore")
        aligned = aligned.loc[:, aligned.nunique() > 1]  # e.g. the intervened column in S'
        correlations = aligned.apply(
            lambda column: lengths.corr(column, method="spearman")
        ).dropna()
        if not correlations.empty:
            print(
                "  length~state Spearman: "
                +", ".join(f"{name} {value:+.2f}" for name, value in correlations.items())
            )
    return lengths
