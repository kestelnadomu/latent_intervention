"""CPU VADER sentiment differences for the saved eight-way CV comparison."""

from __future__ import annotations

import argparse
import csv
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

from exp.text_realizer.protocol import immutable_json, lock
from exp.text_realizer_bertscore.__main__ import (
    CONDITIONS,
    DECODERS,
    INPUT,
    RESULTS,
    ROOT,
    load_pairs,
)
from exp.text_realizer_package_eval.common import read_json, versions
from src.artifact_io import atomic_output, sha256_file, write_json

OUTPUT = RESULTS / "vader"
VERSION = "3.3.2"
METRICS = (
    "reference_compound",
    "generated_compound",
    "sentiment_delta",
    "absolute_sentiment_delta",
)


def load_analyzer() -> tuple[object, dict]:
    """Load the pinned reference implementation and identify its bundled lexicons."""
    from vaderSentiment import vaderSentiment

    installed = versions(("vaderSentiment",))
    if installed["vaderSentiment"] != VERSION:
        raise ValueError(f"requires vaderSentiment=={VERSION}")
    folder = Path(vaderSentiment.__file__).parent
    return vaderSentiment.SentimentIntensityAnalyzer(), {
        "versions": installed,
        "python": platform.python_version(),
        "package_files_sha256": {
            name: sha256_file(folder / name)
            for name in (
                "vaderSentiment.py",
                "vader_lexicon.txt",
                "emoji_utf8_lexicon.txt",
            )
        },
    }


def score(path: Path, output: Path) -> dict:
    """Score complete CVs, retaining ID-aligned compound scores and paired changes."""
    started = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    analyzer, package = load_analyzer()
    metadata = {
        **package,
        "input_sha256": sha256_file(path),
        "source_sha256": {
            name: sha256_file(ROOT / name)
            for name in (
                "exp/text_realizer_vader.py",
                "exp/text_realizer_bertscore/__main__.py",
                "exp/text_realizer_package_eval/common.py",
                "exp/text_realizer/protocol.py",
                "src/artifact_io.py",
            )
        },
        "base_environment_sha256": sha256_file(
            ROOT / "exp/text_realizer_bertscore/environment/uv.lock"
        ),
        "normalization": "collapse whitespace; preserve case and punctuation",
        "scoring_unit": "whole CV; no truncation or sentence splitting",
        "reference": "oracle counterfactual CV",
        "sentiment_delta": "generated_compound - reference_compound",
    }
    pairs = [
        {"decoder": decoder, **pair}
        for decoder in DECODERS
        for pair in load_pairs(path, decoder)
    ]
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
                raise ValueError("VADER results changed; use a new output directory")
            print("VADER: already complete; output checksums verified", flush=True)
            return saved
        rows, cache = [], {}
        # shortcut: restart incomplete CPU runs; checkpoint rows if runtime grows materially.
        for pair in pairs:
            for text in (pair["reference"], pair["prediction"]):
                if text not in cache:
                    value = float(analyzer.polarity_scores(text)["compound"])
                    if not math.isfinite(value) or not -1 <= value <= 1:
                        raise RuntimeError("invalid VADER compound score")
                    cache[text] = value
            reference = cache[pair["reference"]]
            generated = cache[pair["prediction"]]
            delta = generated - reference
            rows.append(
                {
                    "id": pair["id"],
                    "decoder": pair["decoder"],
                    "condition": pair["condition"],
                    "reference_compound": reference,
                    "generated_compound": generated,
                    "sentiment_delta": delta,
                    "absolute_sentiment_delta": abs(delta),
                }
            )
            if len(rows) % 1000 == 0:
                print(f"VADER: {len(rows)}/{len(pairs)} pairs", flush=True)
        summaries = []
        for decoder in DECODERS:
            for condition in CONDITIONS:
                group = [
                    row
                    for row in rows
                    if row["decoder"] == decoder and row["condition"] == condition
                ]
                summaries.append(
                    {
                        "decoder": decoder,
                        "condition": condition,
                        "units": len(group),
                        **{
                            f"mean_{key}": mean(row[key] for row in group)
                            for key in METRICS
                        },
                    }
                )
        if sha256_file(path) != metadata["input_sha256"]:
            raise ValueError("comparison input changed during scoring")
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
            "pairs": len(rows),
            "unique_texts_scored": len(cache),
            "metrics": summaries,
            "files_sha256": {
                name: sha256_file(output / name)
                for name in ("per_cv.csv", "summary.csv")
            },
            "limitation": (
                "Whole-CV sentiment may saturate; it is not factual accuracy, "
                "personality, employability or evidence of counterfactual fairness."
            ),
        }
        write_json(target, saved)
        print(
            f"Complete: {len(rows)} pairs in {saved['elapsed_seconds']:.1f}s; {output}",
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
