"""Synthetic, dependency-free checks for portable paper aggregates and LaTeX."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from exp.paper_tables import decoder_package_table as table


@pytest.fixture
def reports(tmp_path):
    comparisons = b"".join(
        json.dumps({"id": key, "reference": "SYNTHETIC-CV-TEXT"}).encode() + b"\n"
        for key in range(10)
    )
    identity = {"input_sha256": hashlib.sha256(comparisons).hexdigest()}
    reports = {
        name: {"complete": True, "identity": identity, "metrics": []}
        for name in table.SOURCES
    }
    for decoder in table.DECODERS:
        for condition in table.CONDITIONS:
            common = {"decoder": decoder, "condition": condition, "units": 10}
            base = {
                "model": decoder,
                "condition": condition,
                "units": 10,
                "character_similarity": 0.321,
                "chrf": 0.555,
                "rouge_l": 0.31,
            }
            reports["scores"]["metrics"].append(base)
            reports["bertscore-roberta-large"]["metrics"].append(
                {**common, "precision": 0.9, "recall": 0.8, "f1": 0.8456}
            )
            reports["vader"]["metrics"].append(
                {
                    **common,
                    "mean_sentiment_delta": -0.00004,
                    "mean_absolute_sentiment_delta": 0.007,
                }
            )
            for field in table.FIELDS:
                base[f"{field}_accuracy_on_scorable_references"] = 0.9
                base[f"{field}_reference_coverage"] = 1.0
                changed_n = (
                    10
                    if field in ("country", "region")
                    else 1
                    if field == "project_count"
                    else 3
                )
                for status, n in (
                    ("changed", changed_n),
                    ("preserved", 10 - changed_n),
                ):
                    correct = max(0, n - 1)
                    reports["fact-changes"]["metrics"].append(
                        {
                            **common,
                            "field": field,
                            "status": status,
                            "total_units": 10,
                            "original_scorable": 10,
                            "oracle_scorable": 10,
                            "jointly_scorable": 10,
                            "reference_coverage": 1.0,
                            "excluded_reference_pairs": 0,
                            "units": n,
                            "correct": correct,
                            "accuracy": correct / n if n else None,
                            "unparseable_predictions": 0,
                        }
                    )
    for name, report in reports.items():
        folder = tmp_path / name
        folder.mkdir()
        (folder / "summary.json").write_text(json.dumps(report))
    (tmp_path / "scores/comparisons.jsonl").write_bytes(comparisons)
    return tmp_path


def test_portable_collection_and_rendering(reports):
    summary = table.collect(reports)
    assert "SYNTHETIC-CV-TEXT" not in json.dumps(summary)
    panels = table.render(summary)
    assert set(panels) == {"text", "facts", "changed", "preserved"}
    assert all(text.count("256 & 270M") == 2 for text in panels.values())
    assert "0.8456" in panels["text"] and "-0.00004" in panels["text"]
    assert "90.0" in panels["facts"] and "$n=10$" in panels["facts"]
    assert "$n=0$" in panels["preserved"] and " & -- & -- & " in panels["preserved"]
    assert "$n=1$" in panels["changed"]
    for text in panels.values():
        assert "Oracle $Z'$" in text and "Predicted $\\hat{Z}'$" in text
        assert "Oracle-exposed decoder" not in text
        assert text.count(r"\begin{tabular}") == text.count(r"\end{tabular}") == 1
        assert all(line.endswith(r"\\") for line in text.splitlines() if " & " in line)


@pytest.mark.parametrize(
    "problem",
    [
        "mixed_input",
        "missing_row",
        "duplicate_row",
        "unit_count",
        "nan",
        "empty_accuracy",
        "wrong_accuracy",
    ],
)
def test_incompatible_reports_refused(reports, problem):
    summary = table.collect(reports)
    data = summary["reports"]
    if problem == "mixed_input":
        data["vader"]["identity"]["input_sha256"] = "other"
    elif problem == "missing_row":
        data["scores"]["metrics"].pop()
    elif problem == "duplicate_row":
        data["scores"]["metrics"].append(copy.deepcopy(data["scores"]["metrics"][0]))
    elif problem == "unit_count":
        data["vader"]["metrics"][0]["units"] = 11
    elif problem == "nan":
        data["bertscore-roberta-large"]["metrics"][0]["f1"] = float("nan")
    else:
        row = next(
            r
            for r in data["fact-changes"]["metrics"]
            if (r["units"] == 0) == (problem == "empty_accuracy")
        )
        row["accuracy"] = 1.0
    with pytest.raises(ValueError):
        summary["reports_sha256"] = table._digest(data)
        table.render(summary)


def test_modified_aggregate_checksum_refused(reports):
    summary = table.collect(reports)
    summary["reports"]["scores"]["metrics"][0]["chrf"] = 0.9
    with pytest.raises(ValueError, match="checksum"):
        table.render(summary)


def test_cli_renders_without_site_packages_or_raw_reports(reports, tmp_path):
    summary = tmp_path / "portable.json"
    summary.write_text(json.dumps(table.collect(reports)))
    (reports / "scores/comparisons.jsonl").unlink()
    output = tmp_path / "paper/decoder.tex"
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "exp.paper_tables.decoder_package_table",
            "--summary",
            str(summary),
            "--output",
            str(output),
        ],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in output.parent.glob("*.tex")) == [
        "decoder.tex",
        "decoder_changed.tex",
        "decoder_facts.tex",
        "decoder_preserved.tex",
    ]
