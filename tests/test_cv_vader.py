"""Offline, synthetic checks for paired sentiment scoring and artifact guards."""

import csv
import json
from types import SimpleNamespace

import pytest

from exp import text_realizer_vader as vader


@pytest.fixture
def comparisons(tmp_path):
    path = tmp_path / "comparisons.jsonl"
    rows = [
        {
            "id": key,
            "reference": reference,
            "generated": {
                f"{decoder}/{condition}": (
                    reference if condition == "oracle" else prediction
                )
                for decoder in vader.DECODERS
                for condition in vader.CONDITIONS
            },
        }
        for key, reference, prediction in (
            (7, "Good  CV!", "Bad CV!"),
            (2, "Bad CV!", "Good CV!"),
        )
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.fixture
def analyzer(monkeypatch):
    calls = []

    def polarity_scores(text):
        calls.append(text)
        return {"compound": {"Good CV!": 0.6, "Bad CV!": -0.6}[text]}

    fake = SimpleNamespace(polarity_scores=polarity_scores)
    monkeypatch.setattr(vader, "load_analyzer", lambda: (fake, {"test": True}))
    return calls


def test_signed_and_absolute_changes_alignment_and_rerun(
    comparisons, analyzer, tmp_path
):
    output = tmp_path / "scores"
    report = vader.score(comparisons, output)
    assert report["complete"] and report["pairs"] == 16
    assert analyzer == ["Good CV!", "Bad CV!"]
    with (output / "per_cv.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [(int(row["id"]), row["decoder"], row["condition"]) for row in rows] == [
        (key, decoder, condition)
        for decoder in vader.DECODERS
        for key in (7, 2)
        for condition in vader.CONDITIONS
    ]
    assert float(rows[1]["sentiment_delta"]) == pytest.approx(-1.2)
    assert float(rows[3]["sentiment_delta"]) == pytest.approx(1.2)
    for row in report["metrics"]:
        assert row["units"] == 2
        assert row["mean_sentiment_delta"] == pytest.approx(0)
        assert row["mean_absolute_sentiment_delta"] == pytest.approx(
            0 if row["condition"] == "oracle" else 1.2
        )
    assert vader.score(comparisons, output) == report
    assert len(analyzer) == 2
    (output / "per_cv.csv").write_text("corrupted\n")
    with pytest.raises(ValueError, match="results changed"):
        vader.score(comparisons, output)


def test_changed_input_and_invalid_ids_refused(comparisons, analyzer, tmp_path):
    output = tmp_path / "scores"
    vader.score(comparisons, output)
    comparisons.write_text(comparisons.read_text().replace("Good", "GREAT"))
    with pytest.raises(ValueError, match="immutable artifact changed"):
        vader.score(comparisons, output)
    assert len(analyzer) == 2
    with comparisons.open("a") as stream:
        stream.write(comparisons.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        vader.score(comparisons, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_interruption_restarts_without_publishing_partial_results(
    comparisons, analyzer, monkeypatch, tmp_path
):
    output = tmp_path / "scores"
    original = vader.load_analyzer

    def interrupted(text):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(
        vader,
        "load_analyzer",
        lambda: (SimpleNamespace(polarity_scores=interrupted), {"test": True}),
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        vader.score(comparisons, output)
    assert (output / "plan.json").exists()
    assert not (output / "summary.json").exists()
    assert not (output / "per_cv.csv").exists()
    monkeypatch.setattr(vader, "load_analyzer", original)
    assert vader.score(comparisons, output)["complete"]


@pytest.mark.parametrize("value", [float("nan"), 1.1])
def test_invalid_analyzer_output_refused(comparisons, monkeypatch, tmp_path, value):
    monkeypatch.setattr(
        vader,
        "load_analyzer",
        lambda: (
            SimpleNamespace(polarity_scores=lambda text: {"compound": value}),
            {"test": True},
        ),
    )
    output = tmp_path / "scores"
    with pytest.raises(RuntimeError, match="invalid VADER"):
        vader.score(comparisons, output)
    assert not (output / "summary.json").exists()
