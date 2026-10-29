"""Portable LaTeX tables for the four package CV decoders; standard library only.

Render from the shipped aggregate JSON, without CV texts or model downloads:
    python -m exp.paper_tables.decoder_package_table --output tables/decoder.tex

Refresh the aggregate on a machine holding the completed evaluation reports:
    python -m exp.paper_tables.decoder_package_table --reports REPORT_DIR --output tables/decoder.tex
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from .decoder_table import bold, cell

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "docs/experiments/results/decoder_package_v1.json"
DECODERS = ("256-270m", "256-1b", "768-270m", "768-1b")
CONDITIONS = ("oracle", "predicted")
FIELDS = ("country", "region", "degree", "university_tier", "project_count")
LABELS = ("Country", "Region", "Degree", "Univ. tier", "Projects")
SOURCES = ("scores", "bertscore-roberta-large", "vader", "fact-changes")
ROW_END = r" \\"
TEXT_COLUMNS = (
    ("character_similarity", r"Char. sim. $\uparrow$", 3),
    ("chrf", r"chrF $\uparrow$", 3),
    ("rouge_l", r"ROUGE-L $\uparrow$", 3),
    ("f1", r"BERTScore $F_1\uparrow$", 4),
    ("mean_sentiment_delta", r"VADER $\Delta$", 5),
    ("mean_absolute_sentiment_delta", r"VADER $|\Delta|\downarrow$", 5),
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _index(rows: list[dict], keys: tuple[str, ...], expected: set) -> dict:
    indexed = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        if key in indexed:
            raise ValueError(f"duplicate result row: {key}")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("missing or unexpected decoder/condition/field rows")
    return indexed


def _number(value: object, low: float = 0, high: float = 1) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"invalid metric value: {value}")
    return value


def validate(summary: dict) -> tuple[dict, dict, dict, dict]:
    """Reject mixed inputs, incomplete variants, invalid values and denominators."""
    if summary.get("format_version") != 1:
        raise ValueError("unsupported package table summary format")
    reports = summary["reports"]
    if summary["reports_sha256"] != _digest(reports):
        raise ValueError("aggregate checksum mismatch; regenerate from reports")
    expected = {(d, c) for d in DECODERS for c in CONDITIONS}
    base = _index(reports["scores"]["metrics"], ("model", "condition"), expected)
    bert, vader = (
        _index(reports[name]["metrics"], ("decoder", "condition"), expected)
        for name in ("bertscore-roberta-large", "vader")
    )
    changed = _index(
        reports["fact-changes"]["metrics"],
        ("decoder", "condition", "field", "status"),
        {
            (d, c, f, s)
            for d, c in expected
            for f in FIELDS
            for s in ("changed", "preserved")
        },
    )
    units = summary["units"]
    if type(units) is not int or units < 1:
        raise ValueError("invalid evaluation unit count")
    for name in SOURCES[1:]:
        report = reports[name]
        if (
            not report["complete"]
            or report["identity"]["input_sha256"] != summary["input_sha256"]
        ):
            raise ValueError("metric reports are incomplete or refer to different CVs")
    coverage = {}
    for key in expected:
        if any(rows[key]["units"] != units for rows in (base, bert, vader)):
            raise ValueError("metrics use different evaluation unit counts")
        for field in ("character_similarity", "chrf", "rouge_l"):
            _number(base[key][field])
        for field in ("precision", "recall", "f1"):
            _number(bert[key][field])
        _number(vader[key]["mean_sentiment_delta"], -2, 2)
        _number(vader[key]["mean_absolute_sentiment_delta"], 0, 2)
        for field in FIELDS:
            _number(base[key][f"{field}_accuracy_on_scorable_references"])
            oracle_n = _number(base[key][f"{field}_reference_coverage"]) * units
            pair = [changed[*key, field, status] for status in ("changed", "preserved")]
            if sum(row["units"] for row in pair) != pair[0]["jointly_scorable"]:
                raise ValueError(
                    "changed/preserved denominators do not sum to joint coverage"
                )
            for row in pair:
                for count in (
                    "total_units",
                    "original_scorable",
                    "oracle_scorable",
                    "jointly_scorable",
                    "excluded_reference_pairs",
                    "units",
                    "correct",
                    "unparseable_predictions",
                ):
                    if type(row[count]) is not int or not 0 <= row[count] <= units:
                        raise ValueError("invalid factual count")
                n, correct = row["units"], row["correct"]
                if (
                    row["total_units"] != units
                    or correct + row["unparseable_predictions"] > n
                    or row["jointly_scorable"] + row["excluded_reference_pairs"]
                    != units
                    or not math.isclose(oracle_n, row["oracle_scorable"], abs_tol=1e-9)
                    or not math.isclose(
                        row["reference_coverage"], row["jointly_scorable"] / units
                    )
                ):
                    raise ValueError("inconsistent factual counts or coverage")
                if n == 0:
                    if row["accuracy"] is not None:
                        raise ValueError(
                            "an empty factual group must have null accuracy"
                        )
                elif not math.isclose(
                    _number(row["accuracy"]), correct / n, abs_tol=1e-12
                ):
                    raise ValueError("factual accuracy differs from correct/count")
                counts = tuple(
                    row[name]
                    for name in (
                        "original_scorable",
                        "oracle_scorable",
                        "jointly_scorable",
                        "units",
                    )
                )
                group = field, row["status"]
                if coverage.setdefault(group, counts) != counts:
                    raise ValueError("reference cohorts differ across decoder variants")
    return base, bert, vader, changed


def collect(reports_dir: Path) -> dict:
    """Package existing aggregate reports and their provenance, excluding CV texts."""
    reports, source_hashes = {}, {}
    for name in SOURCES:
        path = reports_dir / name / "summary.json"
        data = path.read_bytes()
        reports[name] = json.loads(data)
        source_hashes[f"{name}/summary.json"] = hashlib.sha256(data).hexdigest()
    comparisons = (reports_dir / "scores/comparisons.jsonl").read_bytes()
    ids = [json.loads(line)["id"] for line in comparisons.splitlines()]
    if not ids or any(type(key) is not int for key in ids) or len(set(ids)) != len(ids):
        raise ValueError("invalid or duplicate comparison IDs")
    summary = {
        "format_version": 1,
        "run": reports_dir.name,
        "decoder_training": "factual-only; oracle/predicted describes inference vectors",
        "units": len(ids),
        "input_sha256": hashlib.sha256(comparisons).hexdigest(),
        "unit_ids_sha256": _digest(ids),
        "source_files_sha256": source_hashes,
        "reports_sha256": _digest(reports),
        "reports": reports,
    }
    validate(summary)
    return summary


def render(summary: dict) -> dict[str, str]:
    """Render text, overall-fact, changed-fact and preserved-fact tabulars."""
    base, bert, vader, changed = validate(summary)
    result = {}
    first = DECODERS[0], CONDITIONS[0]
    for panel in ("text", "facts", "changed", "preserved"):
        headers = [label for _, label, _ in TEXT_COLUMNS]
        if panel != "text":
            denominators = [
                round(base[first][f"{field}_reference_coverage"] * summary["units"])
                if panel == "facts"
                else changed[*first, field, panel]["units"]
                for field in FIELDS
            ]
            headers = [
                rf"\shortstack{{{label}\\$n={n}$}}"
                for label, n in zip(LABELS, denominators, strict=True)
            ]
        columns = len(headers) + 2
        lines = [
            "% Generated by exp.paper_tables.decoder_package_table; do not edit numbers.",
            f"% Source aggregate SHA256: {summary['reports_sha256']}",
            "% All four decoders are factual-only. Oracle/predicted refers to input vectors.",
            "% 256-D and 768-D reader/flow training protocols differ; not a dimension-only comparison.",
            "% Text scores use raw 0--1 scales; facts are percentages; VADER differences are raw.",
            "% Fact n is the scorable denominator; -- means n=0. Interpret small groups cautiously.",
            "% Bare tabular: provide the table float/caption and load booktabs in the paper.",
            r"\begin{tabular}{@{}rl" + "r" * len(headers) + "@{}}",
            r"  \toprule",
            "  "
            + " & ".join([bold(r"$d_Z$"), bold("Gemma 3"), *map(bold, headers)])
            + ROW_END,
        ]
        for condition, label in (
            ("oracle", r"Oracle $Z'$"),
            ("predicted", r"Predicted $\hat{Z}'$"),
        ):
            lines.extend(
                [
                    r"  \midrule",
                    rf"  \multicolumn{{{columns}}}{{l}}{{\textit{{{label}; {summary['units']:,} test CVs}}}} \\",
                ]
            )
            for decoder in DECODERS:
                key = decoder, condition
                dimension, size = decoder.split("-")
                cells = [dimension, size.upper()]
                if panel == "text":
                    values = {**base[key], **bert[key], **vader[key]}
                    cells += [
                        f"{values[name]:+.{digits}f}"
                        if name == "mean_sentiment_delta"
                        else cell(values[name], digits, 1)
                        for name, _, digits in TEXT_COLUMNS
                    ]
                else:
                    for field in FIELDS:
                        value = (
                            base[key][f"{field}_accuracy_on_scorable_references"]
                            if panel == "facts"
                            else changed[*key, field, panel]["accuracy"]
                        )
                        cells.append("--" if value is None else cell(value, 1, 100))
                lines.append("  " + " & ".join(cells) + ROW_END)
        lines.extend([r"  \bottomrule", r"\end{tabular}"])
        result[panel] = "\n".join(lines) + "\n"
    return result


def _write(path: Path, text: str) -> None:
    # Keep this renderer independent of the torch-backed artifact helpers.
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, mode="w", encoding="utf-8", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument(
        "--reports",
        type=Path,
        help="refresh the portable summary from completed reports",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="text table .tex path; companion tables share its stem",
    )
    args = parser.parse_args()
    if args.output.suffix != ".tex" or args.output.resolve() == args.summary.resolve():
        parser.error("--output must be a .tex file distinct from --summary")
    summary = (
        collect(args.reports) if args.reports else json.loads(args.summary.read_text())
    )
    tables = render(summary)
    if args.reports:
        _write(
            args.summary,
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        print(f"wrote portable aggregates: {args.summary}")
    for panel, text in tables.items():
        path = (
            args.output
            if panel == "text"
            else args.output.with_stem(f"{args.output.stem}_{panel}")
        )
        _write(path, text)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
