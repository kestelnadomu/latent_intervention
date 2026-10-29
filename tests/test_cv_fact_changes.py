"""Synthetic tests for intended changes, preservation, coverage and artifact guards."""

import csv
import json

import pytest

from exp import text_realizer_fact_changes as changes
from exp.text_realizer_bertscore.__main__ import CONDITIONS


@pytest.fixture
def comparisons(tmp_path):
    original = (
        "From Nigeria, a Bachelor's degree from a university in the top 500, "
        "with two projects."
    )
    reference = (
        "From Germany, a Master's degree from a university in the top 500, "
        "with two projects."
    )
    examples = [
        (7, original, reference, original),
        (2, reference, reference, "From Germany, a Master's degree; three projects."),
        (9, "No details stated.", reference, reference),
        (4, original, "From Nigeria and Germany, a Master's degree.", "From Spain."),
    ]
    rows = [
        {
            "id": key,
            "original": before,
            "reference": oracle,
            "generated": {
                f"{decoder}/{condition}": oracle
                if condition == "oracle"
                else prediction
                for decoder in changes.DECODERS
                for condition in CONDITIONS
            },
        }
        for key, before, oracle, prediction in examples
    ]
    path = tmp_path / "comparisons.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_changed_preserved_missing_and_empty_denominators(comparisons, tmp_path):
    output = tmp_path / "scores"
    report = changes.score(comparisons, output)
    assert report["complete"] and report["pairs"] == 32 and report["field_rows"] == 160
    assert len(report["metrics"]) == 80
    metrics = {
        (row["field"], row["status"]): row
        for row in report["metrics"]
        if row["decoder"] == "256-270m" and row["condition"] == "predicted"
    }
    country = metrics["country", "changed"]
    assert country["units"] == 1 and country["accuracy"] == 0
    assert country["jointly_scorable"] == 2 and country["reference_coverage"] == 0.5
    assert metrics["country", "preserved"]["accuracy"] == 1
    assert metrics["degree", "changed"]["units"] == 2
    assert metrics["degree", "changed"]["unparseable_predictions"] == 1
    assert metrics["degree", "changed"]["accuracy"] == 0
    assert metrics["university_tier", "preserved"]["accuracy"] == 0.5
    assert metrics["university_tier", "preserved"]["unparseable_predictions"] == 1
    assert metrics["project_count", "preserved"]["accuracy"] == 0.5
    assert metrics["project_count", "changed"]["units"] == 0
    assert metrics["project_count", "changed"]["accuracy"] is None
    for row in report["metrics"]:
        if row["condition"] == "oracle" and row["units"]:
            assert row["accuracy"] == 1
    with (output / "per_cv.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["id"]) for row in rows[::10]][:4] == [7, 2, 9, 4]
    zero_region = next(
        row
        for row in rows
        if row["id"] == "7"
        and row["field"] == "region"
        and row["condition"] == "predicted"
    )
    assert zero_region["original_value"] == "0" and zero_region["status"] == "changed"
    unknown = [row for row in rows if row["id"] == "9"]
    assert all(
        row["status"] == "unscorable" and row["correct"] == "" for row in unknown
    )


def test_verified_rerun_and_corrupt_output_refusal(comparisons, tmp_path, monkeypatch):
    output = tmp_path / "scores"
    report = changes.score(comparisons, output)

    def unexpected(*args):
        raise AssertionError("completed run must not be rescored")

    monkeypatch.setattr(changes, "evaluate", unexpected)
    assert changes.score(comparisons, output) == report
    (output / "per_cv.csv").write_text("corrupt\n")
    with pytest.raises(ValueError, match="results changed"):
        changes.score(comparisons, output)


def test_input_and_codebook_hash_guards(comparisons, tmp_path, monkeypatch):
    output = tmp_path / "scores"
    changes.score(comparisons, output)
    original_bytes = comparisons.read_bytes()
    comparisons.write_text(comparisons.read_text().replace("Germany", "France"))
    with pytest.raises(ValueError, match="immutable artifact changed"):
        changes.score(comparisons, output)
    comparisons.write_bytes(original_bytes)
    codebook = tmp_path / "codebook.yaml"
    codebook.write_text(changes.CODEBOOK.read_text() + "\n# changed fixture\n")
    monkeypatch.setattr(changes, "CODEBOOK", codebook)
    with pytest.raises(ValueError, match="immutable artifact changed"):
        changes.score(comparisons, output)


@pytest.mark.parametrize(
    "invalid", ["missing_original", "duplicate_id", "missing_variant"]
)
def test_invalid_comparisons_refused(comparisons, tmp_path, invalid):
    rows = [json.loads(line) for line in comparisons.read_text().splitlines()]
    if invalid == "missing_original":
        rows[0].pop("original")
    elif invalid == "duplicate_id":
        rows[1]["id"] = rows[0]["id"]
    else:
        rows[0]["generated"].pop("768-1b/predicted")
    comparisons.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        changes.score(comparisons, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
