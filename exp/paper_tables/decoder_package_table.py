"""Portable LaTeX tables for the four package CV decoders; standard library only.

Render from the shipped aggregate JSON, its non-generative reference rows (retrieval,
unedited x) and the noise floor (same-state rewrite; both from
``exp.text_realizer_package_eval.baselines``), without CV texts or downloads:
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
BASELINES = ROOT / "docs/experiments/results/decoder_package_baselines_v1.json"
FLOOR = ROOT / "docs/experiments/results/decoder_package_floor_v1.json"
DIMENSIONS = (256, 768)
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


def validate_baselines(
    summary: dict, baselines: dict, base: dict, changed: dict
) -> dict:
    """Index the reference rows; they must share the decoders' units and fact cohorts."""
    if (
        baselines.get("format_version") != 1
        or baselines["unit_ids_sha256"] != summary["unit_ids_sha256"]
    ):
        raise ValueError("baselines refer to other test units")
    expected = {("retrieval", d, c) for d in DIMENSIONS for c in CONDITIONS}
    rows = _index(
        baselines["rows"],
        ("method", "dim", "condition"),
        expected | {("unedited", None, None)},
    )
    first = DECODERS[0], CONDITIONS[0]
    for row in rows.values():
        if row["units"] != summary["units"]:
            raise ValueError("baselines use a different evaluation unit count")
        for name in ("character_similarity", "chrf", "rouge_l", "f1"):
            _number(row[name])
        _number(row["mean_sentiment_delta"], -2, 2)
        _number(row["mean_absolute_sentiment_delta"], 0, 2)
        for field in FIELDS:
            scorable = base[first][f"{field}_reference_coverage"] * summary["units"]
            if row["facts"][field]["reference_scorable"] != round(scorable) or any(
                row[status][field]["units"] != changed[*first, field, status]["units"]
                for status in ("changed", "preserved")
            ):
                raise ValueError("baseline fact cohorts differ from the decoders'")
            for value in (
                row["facts"][field]["accuracy"],
                row["changed"][field]["accuracy"],
                row["preserved"][field]["accuracy"],
            ):
                if value is not None:
                    _number(value)
    return rows


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


def _row(panel: str, dim: str, label: str, values: dict) -> str:
    """One table row; ``values`` holds the text metrics, or one accuracy per fact field."""
    if panel == "text":
        cells = [
            f"{values[name]:+.{digits}f}"
            if name == "mean_sentiment_delta"
            else cell(values[name], digits, 1)
            for name, _, digits in TEXT_COLUMNS
        ]
    else:
        cells = [
            "--" if values[field] is None else cell(values[field], 1, 100)
            for field in FIELDS
        ]
    return "  " + " & ".join([dim, label, *cells]) + ROW_END


def validate_floor(floor: dict) -> dict:
    """The noise-floor row: same-state rewrites against their factual CVs (own units)."""
    if floor.get("format_version") != 1:
        raise ValueError("unsupported noise-floor format")
    row = floor["row"]
    for name in ("character_similarity", "chrf", "rouge_l", "f1"):
        _number(row[name])
    _number(row["mean_sentiment_delta"], -2, 2)
    _number(row["mean_absolute_sentiment_delta"], 0, 2)
    for field in FIELDS:
        if row["facts"][field]["accuracy"] is not None:
            _number(row["facts"][field]["accuracy"])
    return row


def render(summary: dict, baselines: dict, floor: dict) -> dict[str, str]:
    """Render text, overall-fact, changed-fact and preserved-fact tabulars."""
    base, bert, vader, changed = validate(summary)
    references = validate_baselines(summary, baselines, base, changed)
    rewrite = validate_floor(floor)

    def floor_values(panel: str) -> dict:
        if panel == "text":
            return rewrite
        # the rewrite changes the country by construction; it has no changed/preserved split
        return {
            f: rewrite["facts"][f]["accuracy"]
            if panel == "facts" and f != "country"
            else None
            for f in FIELDS
        }

    def decoder_values(key: tuple, panel: str) -> dict:
        if panel == "text":
            return {**base[key], **bert[key], **vader[key]}
        if panel == "facts":
            return {
                f: base[key][f"{f}_accuracy_on_scorable_references"] for f in FIELDS
            }
        return {f: changed[*key, f, panel]["accuracy"] for f in FIELDS}

    def reference_values(key: tuple, panel: str) -> dict:
        row = references[key]
        if panel == "text":
            return row
        return {
            f: row["facts" if panel == "facts" else panel][f]["accuracy"]
            for f in FIELDS
        }

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
            "% Retrieval: the fit CV with the closest latent; unedited x: the factual CV as x'.",
            f"% Reference rows SHA256: {_digest(baselines['rows'])}",
            f"% Same-state rewrite (noise floor): {floor['units']} floor texts against their"
            " factual CVs; the country differs by construction.",
            "% 256-D and 768-D reader/flow training protocols differ; not a dimension-only comparison.",
            "% Text scores use raw 0--1 scales; facts are percentages; VADER differences are raw.",
            "% Fact n is the scorable denominator; -- means n=0. Interpret small groups cautiously.",
            "% Bare tabular: provide the table float/caption and load booktabs in the paper.",
            r"\begin{tabular}{@{}rl" + "r" * len(headers) + "@{}}",
            r"  \toprule",
            "  "
            + " & ".join([bold(r"$d_Z$"), bold("Decoder"), *map(bold, headers)])
            + ROW_END,
        ]
        for condition, label in (
            ("oracle", r"$\varphi^{-1}(\varphi(x'))$ against $x'$"),
            ("predicted", r"$\varphi^{-1}(z')$ against $x'$"),
        ):
            lines.extend(
                [
                    r"  \midrule",
                    rf"  \multicolumn{{{columns}}}{{l}}{{\textit{{{label}}}}} \\",
                ]
            )
            for dim in DIMENSIONS:
                for decoder in DECODERS:
                    if decoder.startswith(f"{dim}-"):
                        size = decoder.split("-")[1].upper()
                        values = decoder_values((decoder, condition), panel)
                        lines.append(_row(panel, str(dim), f"Gemma 3 {size}", values))
                values = reference_values(("retrieval", dim, condition), panel)
                lines.append(_row(panel, str(dim), "Retrieval", values))
        values = reference_values(("unedited", None, None), panel)
        lines.extend(
            [
                r"  \midrule",
                _row(panel, "--", r"Unedited $x$", values),
                _row(panel, "--", "Same-state rewrite", floor_values(panel)),
                r"  \bottomrule",
                r"\end{tabular}",
            ]
        )
        result[panel] = "\n".join(lines) + "\n"
    return result


def _write(path: Path, text: str) -> None:
    # Keep this renderer independent of the torch-backed artifact helpers.
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile(
        dir=path.parent, mode="w", encoding="utf-8", newline="\n", delete=False
    )
    temporary = Path(stream.name)
    try:
        with stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(
            temporary, path
        )  # after closing: Windows cannot replace an open file
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--baselines", type=Path, default=BASELINES)
    parser.add_argument("--floor", type=Path, default=FLOOR)
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
    baselines, floor = (
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.baselines, args.floor)
    )
    tables = render(summary, baselines, floor)
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
