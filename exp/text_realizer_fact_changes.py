"""CPU accuracy of changed and preserved facts in the saved CV comparisons."""

from __future__ import annotations

import argparse
import csv
import platform
from collections import defaultdict
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from time import perf_counter

import yaml

from exp.text_realizer import facts
from exp.text_realizer.protocol import immutable_json, lock
from exp.text_realizer_bertscore.__main__ import (
    DECODERS,
    INPUT,
    RESULTS,
    ROOT,
    load_pairs,
)
from exp.text_realizer_package_eval.common import jsonl_records, read_json, versions
from src.artifact_io import atomic_output, sha256_file, write_json

OUTPUT = RESULTS / "fact-changes"
CODEBOOK = ROOT / "exp/sim/codebook.yaml"


def evaluate(
    pairs: list[dict], originals: dict[int, str], codebook: dict
) -> tuple[list[dict], list[dict]]:
    """Return per-field rows and accuracies, excluding unscorable reference pairs."""

    @cache
    def extract(text: str) -> dict:
        return facts.extract(text, codebook)

    rows, groups = [], defaultdict(list)
    for pair in pairs:
        original = extract(" ".join(originals[pair["id"]].split()))
        oracle = extract(pair["reference"])
        generated = extract(pair["prediction"])
        for field, expected in oracle.items():
            before, observed = original[field], generated[field]
            status = (
                "unscorable"
                if before is None or expected is None
                else "preserved"
                if before == expected
                else "changed"
            )
            row = {
                "id": pair["id"],
                "decoder": pair["decoder"],
                "condition": pair["condition"],
                "field": field,
                "status": status,
                "original_value": before,
                "oracle_value": expected,
                "generated_value": observed,
                "correct": observed == expected if status != "unscorable" else None,
            }
            rows.append(row)
            groups[pair["decoder"], pair["condition"], field].append(row)
    summaries = []
    for (decoder, condition, field), group in groups.items():
        jointly_scorable = sum(row["status"] != "unscorable" for row in group)
        for status in ("changed", "preserved"):
            selected = [row for row in group if row["status"] == status]
            correct = sum(row["correct"] for row in selected)
            summaries.append(
                {
                    "decoder": decoder,
                    "condition": condition,
                    "field": field,
                    "status": status,
                    "total_units": len(group),
                    "original_scorable": sum(
                        row["original_value"] is not None for row in group
                    ),
                    "oracle_scorable": sum(
                        row["oracle_value"] is not None for row in group
                    ),
                    "jointly_scorable": jointly_scorable,
                    "reference_coverage": jointly_scorable / len(group),
                    "excluded_reference_pairs": len(group) - jointly_scorable,
                    "units": len(selected),
                    "correct": correct,
                    "accuracy": correct / len(selected) if selected else None,
                    "unparseable_predictions": sum(
                        row["generated_value"] is None for row in selected
                    ),
                }
            )
    return rows, summaries


def score(path: Path, output: Path) -> dict:
    """Score all eight variants with the existing extractor and immutable inputs."""
    started = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "input_sha256": sha256_file(path),
        "codebook_sha256": sha256_file(CODEBOOK),
        "python": platform.python_version(),
        "versions": versions(("pyyaml",)),
        "source_sha256": {
            name: sha256_file(ROOT / name)
            for name in (
                "exp/text_realizer_fact_changes.py",
                "exp/text_realizer/facts.py",
                "exp/text_realizer_bertscore/__main__.py",
                "exp/text_realizer_package_eval/common.py",
                "exp/text_realizer/protocol.py",
                "src/artifact_io.py",
            )
        },
        "reference_policy": "both original and oracle values must be extractable",
        "target": "oracle value for both changed and preserved facts",
        "missing_generated_value": "failure on scorable reference pairs",
        "empty_group_accuracy": None,
        "normalization": "collapse whitespace; unchanged factual extractor",
    }
    pairs = [
        {"decoder": decoder, **pair}
        for decoder in DECODERS
        for pair in load_pairs(path, decoder)
    ]
    originals = {row["id"]: row.get("original") for row in jsonl_records(path)}
    if any(
        not isinstance(text, str) or not text.strip() for text in originals.values()
    ):
        raise ValueError("requires nonempty original CV text for every candidate")
    with lock(output / ".lock"):
        immutable_json(output / "plan.json", metadata)
        target = output / "summary.json"
        if target.exists():
            saved = read_json(target)
            if (
                not saved["complete"]
                or saved["identity"] != metadata
                or any(
                    not (output / name).is_file()
                    or sha256_file(output / name) != saved["files_sha256"][name]
                    for name in ("per_cv.csv", "summary.csv")
                )
            ):
                raise ValueError(
                    "fact-change results changed; use a new output directory"
                )
            print(
                "Fact changes: already complete; output checksums verified", flush=True
            )
            return saved
        # shortcut: restart incomplete CPU runs; checkpoint rows if runtime grows materially.
        rows, summaries = evaluate(
            pairs, originals, yaml.safe_load(CODEBOOK.read_text())
        )
        if (
            sha256_file(path) != metadata["input_sha256"]
            or sha256_file(CODEBOOK) != metadata["codebook_sha256"]
        ):
            raise ValueError("inputs changed during fact-change scoring")
        for name, records in (("per_cv.csv", rows), ("summary.csv", summaries)):
            with atomic_output(output / name) as temporary:
                with temporary.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows(records)
        saved = {
            "complete": True,
            "identity": metadata,
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": perf_counter() - started,
            "pairs": len(pairs),
            "field_rows": len(rows),
            "metrics": summaries,
            "files_sha256": {
                name: sha256_file(output / name)
                for name in ("per_cv.csv", "summary.csv")
            },
            "limitation": (
                "Text-extractable facts only; reference changes are not a separate SCM audit. "
                "No pooled score: country and region overlap. No behavioral or fairness claim."
            ),
        }
        write_json(target, saved)
        print(
            f"Complete: {len(pairs)} pairs, {len(rows)} field rows in "
            f"{saved['elapsed_seconds']:.1f}s; {output}",
            flush=True,
        )
        return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    score(args.input, args.output)


if __name__ == "__main__":
    main()
