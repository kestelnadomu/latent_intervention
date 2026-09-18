from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from exp.sim import helpers
from exp.sim.generate_text import GenerationResult
from exp.sim.text_length import length_rejection, load_token_counter, report_text_lengths


def _count_words(text: str) -> int:
    return len(text.split())


def test_over_budget_responses_spend_attempts_until_one_fits(tmp_path: Path) -> None:
    output = tmp_path / "texts.csv"
    helpers.write_json(helpers.generation_info_path(output), {"attempts": {}})
    texts = iter(["one two three four", "one two three", "one two"])

    def call() -> GenerationResult:
        return GenerationResult(next(texts), model="m", finish_reason="stop")

    result = helpers.generate_with_attempts(
        output, 5, 3, call, length_rejection(_count_words, 2)
    )

    assert result.text == "one two"
    info = json.loads(helpers.generation_info_path(output).read_text(encoding="utf-8"))
    assert info["attempts"] == {"5": 3}


def test_budget_exhaustion_raises(tmp_path: Path) -> None:
    output = tmp_path / "texts.csv"
    helpers.write_json(helpers.generation_info_path(output), {"attempts": {}})

    def call() -> GenerationResult:
        return GenerationResult("far too many words", model="m", finish_reason="stop")

    with pytest.raises(RuntimeError, match="exhausted"):
        helpers.generate_with_attempts(output, 1, 2, call, length_rejection(_count_words, 2))


def test_report_flags_over_budget_rows(capsys) -> None:
    frame = pd.DataFrame({"id": [1, 2, 3], "text": ["a", "a b c", "a b"]})
    states = pd.DataFrame({"id": [1, 2, 3], "X": [0, 2, 1]}).set_index("id", drop=False)

    lengths = report_text_lengths(frame, _count_words, 2, "factual", states)

    assert lengths.to_dict() == {1: 1, 2: 3, 3: 2}
    out = capsys.readouterr().out
    assert "1/3 over 2" in out
    assert "id=2: 3 tokens" in out
    assert "X +1.00" in out


def test_budget_is_optional_and_validated() -> None:
    assert load_token_counter({}) is None
    with pytest.raises(ValueError, match="positive"):
        load_token_counter({"text_length": {"tokenizer": "gpt2", "max_tokens": 0}})
