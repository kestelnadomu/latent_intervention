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


def reference_rows(summary):
    """Synthetic retrieval/unedited rows sharing the fixture's units and fact cohorts."""
    changed = {
        f: 10 if f in ("country", "region") else 1 if f == "project_count" else 3
        for f in table.FIELDS
    }
    row = {
        "units": 10,
        "character_similarity": 0.4,
        "chrf": 0.6,
        "rouge_l": 0.3,
        "f1": 0.87,
        "mean_sentiment_delta": 0.001,
        "mean_absolute_sentiment_delta": 0.01,
        "facts": {f: {"accuracy": 0.5, "reference_scorable": 10} for f in table.FIELDS},
        "changed": {f: {"units": changed[f], "accuracy": 0.0} for f in table.FIELDS},
        "preserved": {
            f: {"units": 10 - changed[f], "accuracy": None if changed[f] == 10 else 1.0}
            for f in table.FIELDS
        },
    }
    keys = [("retrieval", d, c) for d in table.DIMENSIONS for c in table.CONDITIONS]
    return {
        "format_version": 1,
        "unit_ids_sha256": summary["unit_ids_sha256"],
        "rows": [
            {"method": m, "dim": d, "condition": c, **copy.deepcopy(row)}
            for m, d, c in [*keys, ("unedited", None, None)]
        ],
    }


# Synthetic noise floor: 3 same-state rewrites against their factual CVs.
FLOOR = {
    "format_version": 1,
    "units": 3,
    "unit_ids_sha256": "floor-ids",
    "row": {
        "character_similarity": 0.5,
        "chrf": 0.7,
        "rouge_l": 0.5,
        "f1": 0.93,
        "mean_sentiment_delta": 0.0,
        "mean_absolute_sentiment_delta": 0.005,
        "facts": {f: {"accuracy": 0.8, "reference_scorable": 3} for f in table.FIELDS},
    },
}


def joint_rows(summary):
    """Synthetic joint O/D/U accuracies for every row of the fact table."""
    keys = [(d, None, c) for d in table.DECODERS for c in table.CONDITIONS]
    keys += [("retrieval", d, c) for d in table.DIMENSIONS for c in table.CONDITIONS]
    keys += [("unedited", None, None), ("same_state_rewrite", None, None)]
    return {
        "format_version": 1,
        "unit_ids_sha256": summary["unit_ids_sha256"],
        "comparisons_sha256": summary["input_sha256"],
        "floor_unit_ids_sha256": FLOOR["unit_ids_sha256"],
        "rows": [
            {
                "method": m,
                "dim": d,
                "condition": c,
                "joint": {
                    "accuracy": 0.6 if m == "same_state_rewrite" else 0.75,
                    "reference_scorable": 3 if m == "same_state_rewrite" else 9,
                },
            }
            for m, d, c in keys
        ],
    }


def test_portable_collection_and_rendering(reports):
    summary = table.collect(reports)
    assert "SYNTHETIC-CV-TEXT" not in json.dumps(summary)
    joint = joint_rows(summary)
    panels = table.render(summary, reference_rows(summary), FLOOR, joint)
    assert set(panels) == {"main", "changed", "preserved"}
    assert all(text.count("256 & Gemma 3 & 270M &") == 2 for text in panels.values())
    assert all(text.count("& Retrieval & -- &") == 4 for text in panels.values())
    assert all(text.count("-- & Unedited $x$ & --") == 1 for text in panels.values())
    # joint, O, D and U under one fidelity header left of the text metrics; their n go
    # to the caption; no country, projects or VADER
    main = panels["main"]
    assert r"\multicolumn{4}{c}{{\bfseries\boldmath Fidelity $\uparrow$}}" in main
    assert r"\cmidrule(lr){4-7}" in main and r"\boldmath Size}" in main
    assert r"\shortstack" not in main and "$n=" not in main
    assert "Country" not in main and "Projects" not in main and "VADER" not in main
    assert (
        "256 & Gemma 3 & 270M & 75.0 & 90.0 & 90.0 & 90.0 & 0.321 & 0.555 & 0.310 & 0.8456 \\"
        in main
    )
    assert "-- & Same-state rewrite & -- & 60.0 & 80.0 & 80.0 & 80.0 & 0.500 & 0.700" in main
    # the rewrite has no changed/preserved split
    assert "-- & Same-state rewrite & -- & -- & -- & -- \\" in panels["changed"]
    assert "Fidelity" not in panels["changed"]
    assert r"\boldmath Joint}" not in panels["changed"]
    assert "$n=0$" in panels["preserved"] and " & -- & -- & " in panels["preserved"]
    assert "$n=3$" in panels["changed"]
    with pytest.raises(ValueError, match="other texts"):
        table.render(
            summary, reference_rows(summary), FLOOR, {**joint, "comparisons_sha256": "1"}
        )
    for text in panels.values():
        assert r"$\varphi^{-1}(\varphi(x'))$ against $x'$" in text
        assert r"$\varphi^{-1}(\hat{z}')$ against $x'$" in text
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
        "baseline_units",
        "baseline_cohort",
    ],
)
def test_incompatible_reports_refused(reports, problem):
    summary = table.collect(reports)
    data = summary["reports"]
    baselines = reference_rows(summary)
    if problem == "baseline_units":
        baselines["unit_ids_sha256"] = "other"
    elif problem == "baseline_cohort":
        baselines["rows"][0]["changed"]["degree"]["units"] = 4
    elif problem == "mixed_input":
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
        table.render(summary, baselines, FLOOR, joint_rows(summary))


def test_modified_aggregate_checksum_refused(reports):
    summary = table.collect(reports)
    summary["reports"]["scores"]["metrics"][0]["chrf"] = 0.9
    with pytest.raises(ValueError, match="checksum"):
        table.render(summary, reference_rows(summary), FLOOR, joint_rows(summary))


def test_cli_renders_without_site_packages_or_raw_reports(reports, tmp_path):
    summary = tmp_path / "portable.json"
    collected = table.collect(reports)
    summary.write_text(json.dumps(collected))
    baselines = tmp_path / "baselines.json"
    baselines.write_text(json.dumps(reference_rows(collected)))
    floor = tmp_path / "floor.json"
    floor.write_text(json.dumps(FLOOR))
    joint = tmp_path / "joint.json"
    joint.write_text(json.dumps(joint_rows(collected)))
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
            "--baselines",
            str(baselines),
            "--floor",
            str(floor),
            "--joint",
            str(joint),
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
        "decoder_preserved.tex",
    ]


def test_renderer_table_pairs_true_and_drawn_target_states(reports):
    from exp.paper_tables import decoder_renderers_table as renderer_table

    summary = table.collect(reports)
    kinds = ("editor", "ot_editor", "flow", "ot_flow", "state_effect", "group_ot")
    methods = [*(k for k in kinds if k != "flow"), "factual"]
    methods += [f"{k}-true-target" for k in kinds]
    renderers = {
        "format_version": 1,
        "decoder": "768-270m",
        "unit_ids_sha256": summary["unit_ids_sha256"],
        "comparisons_sha256": "0" * 64,
        "bertscore_cohort": {"excluded_ids": [2333]},
        "rows": [
            {
                "method": m,
                "units": 10,
                "against": "x" if m == "factual" else "x'",
                **{name: 0.5 for name, _, _ in table.TEXT_COLUMNS},
                "facts": {
                    f: {
                        "accuracy": 0.4 if m.endswith("target") else 0.2,
                        "reference_scorable": 10,
                    }
                    for f in table.FIELDS
                },
            }
            for m in methods
        ],
    }
    decoded = {
        "format_version": 1,
        "unit_ids_sha256": summary["unit_ids_sha256"],
        "comparisons_sha256": "0" * 64,
        "row": {
            "method": "decoded_unedited",
            "units": 10,
            "against": "x'",
            "facts": {
                f: {"accuracy": 0.1, "reference_scorable": 10} for f in table.FIELDS
            },
        },
    }
    joint = joint_rows(summary)
    joint["renderers"] = {
        "comparisons_sha256": renderers["comparisons_sha256"],
        "unit_ids_sha256": renderers["unit_ids_sha256"],
        "rows": [
            {
                "method": m,
                "against": "x" if m == "factual" else "x'",
                "joint": {
                    "accuracy": 0.35
                    if m.endswith("target")
                    else {"factual": 0.25, "decoded_unedited": 0.05}.get(m, 0.15),
                    "reference_scorable": 8 if m == "factual" else 7,
                },
            }
            for m in [*methods, "decoded_unedited"]
        ],
    }
    commanded = [(k, c) for k in kinds for c in ("s'", "omega")]
    joint["commands"] = {
        "operator_equals_true": {
            "region": 1,
            "degree": 0.68,
            "university_tier": 0.7,
            "joint": 0.48,
        },
        "rows": [
            {
                "method": m,
                "command": c,
                "facts": {f: 0.95 if c == "s'" else 0.85 for f in table.FIELDS},
                "joint": 0.72 if c == "s'" else 0.62,
                "changed": {
                    f: {
                        "units": 4,
                        "realized": 0.05 if m == "factual" else 0.75 if c == "s'" else 0.5,
                        "kept": 0.9 if m == "factual" else 0.2,
                    }
                    for f in (*table.FIELDS, "joint")
                },
            }
            for m, c in [*commanded, ("oracle", "s'"), ("factual", "s'")]
        ],
    }
    # an own table under the true s': per fact, where s' changes it, the shares of CVs that
    # state the new value, still x's value or another; each fact's three add up to 100
    commands = renderer_table.render_commands(renderers, joint)
    lines = commands.splitlines()
    assert "  Editor & " + " & ".join(["75.0", "20.0", "5.0"] * 4) + " \\\\" in lines
    no_edit = " & ".join([r"  $\varphi^{-1}(\varphi(x))$ (no edit)", *["5.0", "90.0", "5.0"] * 4])
    assert no_edit + " \\\\" in lines
    assert r"\multicolumn{3}{c}{{\bfseries\boldmath $\texttt O$}}" in commands
    assert r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}" in commands
    assert r"\multicolumn{3}{c}{{\bfseries\boldmath Joint}}" in commands
    assert r"{\bfseries\boldmath Kept $\downarrow$}" in commands
    assert "omega" not in commands and "50.0" not in commands  # s' only
    assert all(line.endswith(r"\\") for line in lines if " & " in line)
    with pytest.raises(ValueError, match="invalid metric value"):  # shares beyond 100
        renderer_table.render_commands(
            renderers,
            {**joint, "commands": {**joint["commands"], "rows": [
                {**r, "changed": {f: {**v, "realized": 0.75, "kept": 0.9} for f, v in r["changed"].items()}}
                if r["method"] == "factual" else r
                for r in joint["commands"]["rows"]
            ]}},
        )
    with pytest.raises(ValueError, match="commanded-state scores cover"):
        other = {**joint, "commands": {**joint["commands"], "rows": joint["commands"]["rows"][1:]}}
        renderer_table.render_commands(renderers, other)
    text = renderer_table.render(summary, renderers, decoded, FLOOR, joint)
    assert "commanded" not in text and "Keeps" not in text
    # true s' first, then omega_S(psi(z)); the state flow's drawn edits come from package-v1;
    # O (region), D, U and the joint, then the text metrics without VADER; no n in the header
    assert (
        "Editor & 40.0 & 20.0 & 40.0 & 20.0 & 40.0 & 20.0 & 35.0 & 15.0"
        " & 0.500 & 0.500 & 0.500 & 0.500 & 0.500 & 0.500 & 0.5000 & 0.5000 \\" in text
    )
    assert (
        "State flow & 40.0 & 90.0 & 40.0 & 90.0 & 40.0 & 90.0 & 35.0 & 75.0"
        " & 0.500 & 0.321" in text
    )
    assert r"\multicolumn{2}{c}{{\bfseries\boldmath Joint}}" in text
    assert "$n=" not in text and "Mean" not in text and "Country" not in text
    assert r"VADER $" not in text and "0.50000" not in text
    assert "against x': 10/10/10/7; against x: one more for O and D, joint 8." in text
    assert text.count(r"\multicolumn{2}{c}{20.0}") == 3  # phi^-1(phi(x)) against x
    assert text.count(r"\multicolumn{2}{c}{90.0}") == 3  # phi^-1(phi(x')), package-v1
    assert text.count(r"\multicolumn{2}{c}{10.0}") == 3  # the decoded do-nothing
    assert text.count(r"\multicolumn{2}{c}{80.0}") == 3  # the same-state rewrite
    assert all(line.endswith(r"\\") for line in text.splitlines() if " & " in line)
    reference = {
        "(x))$ against $x$": r"\multicolumn{2}{c}{25.0}",
        "(x'))$ against $x'$": r"\multicolumn{2}{c}{75.0}",
        "rewrite against $x$": r"\multicolumn{2}{c}{60.0}",
    }
    for label, joint_cell in reference.items():
        line = next(line for line in text.splitlines() if label in line)
        assert joint_cell in line
    # the do-nothing's joint, and no text metrics in this fixture
    decoded_line = next(
        line for line in text.splitlines() if "(x))$ against $x'$" in line
    )
    assert r"\multicolumn{2}{c}{5.0}" in decoded_line
    assert decoded_line.endswith(r"\multicolumn{2}{c}{--} \\")  # BERTScore, the last
    with pytest.raises(ValueError, match="other texts"):
        renderer_table.render(
            summary, renderers, {**decoded, "comparisons_sha256": "1"}, FLOOR, joint
        )
    with pytest.raises(ValueError, match="other renderer texts"):
        other = {**joint, "renderers": {**joint["renderers"], "comparisons_sha256": "1"}}
        renderer_table.render(summary, renderers, decoded, FLOOR, other)
    renderers["unit_ids_sha256"] = "other"
    with pytest.raises(ValueError, match="other test units"):
        renderer_table.render(summary, renderers, decoded, FLOOR, joint)
