"""CPU text/fact scoring and readable comparisons, after every shard is complete."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from exp.text_realizer import facts
from src.artifact_io import atomic_output, sha256_file, write_json
from src.text_realizer.metrics import aggregate, text_metrics

from .common import (
    CONDITIONS,
    checked_file,
    jsonl_records,
    load_manifest,
    read_json,
    source_hashes,
    versions,
)
from .generate import read_completed


def collect_predictions(
    folder: Path,
    bundle_hash: str,
    model_key: str,
    arm: str,
    ids: list[int],
    checkpoint_hash: str,
    generation: dict,
) -> dict[int, dict]:
    paths = sorted((folder / model_key / arm).glob("part-*.meta.json"))
    if not paths:
        raise ValueError(f"no completed predictions for {model_key}/{arm}")
    records, shard_ids, execution = {}, set(), None
    expected_shards = None
    for path in paths:
        meta = read_json(path)
        if (
            meta["bundle_sha256"] != bundle_hash
            or meta["checkpoint_sha256"] != checkpoint_hash
            or meta["model"] != model_key
            or meta["condition"] != arm
            or meta["generation"] != generation
        ):
            raise ValueError("prediction provenance mismatch")
        shards, index = meta["num_shards"], meta["shard_index"]
        if (
            shards < 1
            or not 0 <= index < shards
            or index in shard_ids
            or (expected_shards is not None and expected_shards != shards)
        ):
            raise ValueError("mixed or duplicated shards")
        expected_shards = shards
        shard_ids.add(index)
        expected = ids[index::shards]
        if meta["ids"] != expected:
            raise ValueError("shard IDs do not match the frozen bundle")
        comparable = {
            key: meta[key] for key in ("generation", "execution", "source_sha256")
        }
        if execution is not None and comparable != execution:
            raise ValueError("generation settings or runtime differ between shards")
        execution = comparable
        data_path = path.with_name(path.name.removesuffix(".meta.json") + ".jsonl")
        done = read_json(data_path.with_suffix(".done.json"))
        if (
            not done.get("complete")
            or done["records"] != len(expected)
            or done["sha256"] != sha256_file(data_path)
        ):
            raise ValueError("incomplete or modified prediction shard")
        rows = read_completed(data_path, expected)
        if len(rows) != len(expected):
            raise ValueError("incomplete prediction shard")
        for row in rows:
            if row["id"] in records:
                raise ValueError("duplicate generated ID")
            records[row["id"]] = row
    if shard_ids != set(range(expected_shards)) or set(records) != set(ids):
        raise ValueError("missing prediction shards or IDs")
    return records


def score(bundle: Path, predictions: Path, output: Path) -> dict:
    manifest = load_manifest(bundle)
    reference_rows = jsonl_records(
        checked_file(bundle, manifest["files"]["references"])
    )
    ids = manifest["ids"]
    if [row["id"] for row in reference_rows] != ids:
        raise ValueError("reference IDs differ from frozen bundle")
    references = [row["reference"] for row in reference_rows]
    all_outputs, summaries, prediction_files = {}, [], {}
    bundle_hash = sha256_file(bundle / "manifest.json")
    for model_key, model in manifest["models"].items():
        for arm in CONDITIONS:
            rows = collect_predictions(
                predictions,
                bundle_hash,
                model_key,
                arm,
                ids,
                model["sha256"],
                model["generation"],
            )
            for path in sorted((predictions / model_key / arm).glob("part-*")):
                if path.name.endswith((".jsonl", ".meta.json", ".done.json")):
                    prediction_files[str(path.relative_to(predictions))] = sha256_file(
                        path
                    )
            texts = [rows[key]["text"] for key in ids]
            metrics = aggregate(
                [text_metrics(a, b) for a, b in zip(references, texts, strict=True)]
            )
            factual = facts.evaluate(references, texts)
            summary = {
                "model": model_key,
                "condition": arm,
                "units": len(ids),
                **metrics,
                "eos_rate": sum(rows[key]["termination"] == "eos" for key in ids)
                / len(ids),
                "length_limit_rate": sum(
                    rows[key]["termination"] == "max_length" for key in ids
                )
                / len(ids),
                "mean_output_tokens": sum(rows[key]["tokens"] for key in ids)
                / len(ids),
            }
            for field, values in factual["fields"].items():
                for key in (
                    "accuracy_on_scorable_references",
                    "reference_coverage",
                    "unparseable_predictions",
                ):
                    summary[f"{field}_{key}"] = values[key]
            summaries.append(summary)
            all_outputs[f"{model_key}/{arm}"] = texts
    result = {
        "scope": manifest["scope"],
        "bundle_sha256": bundle_hash,
        "selection": manifest["provenance"]["config"]["selection"],
        "pipelines": manifest["provenance"]["config"]["pipelines"],
        "metrics": summaries,
        "prediction_sha256": prediction_files,
        "source_sha256": source_hashes(),
        "versions": versions(("rapidfuzz", "sacrebleu")),
        "limitations": "Text-based extraction reports coverage; unparseable predictions fail scorable references. These checks do not establish behavioral fidelity or counterfactual fairness. The provided flow protocols differ across dimensions.",
    }
    write_json(output / "summary.json", result)
    with atomic_output(output / "summary.csv") as temporary:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    with atomic_output(output / "comparisons.jsonl") as temporary:
        with temporary.open("w", encoding="utf-8") as stream:
            for i, row in enumerate(reference_rows):
                stream.write(
                    json.dumps(
                        {
                            **row,
                            "generated": {
                                key: texts[i] for key, texts in all_outputs.items()
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    with atomic_output(output / "examples.md") as temporary:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(
                "# First five test IDs in the fixed package order\n\nReview occupation, behavioral episodes, changed and preserved facts, invented details and readability. These examples were not selected by quality.\n"
            )
            for i, row in enumerate(reference_rows[:5]):
                stream.write(f"\n## ID {row['id']}; do(X={row['do_value']})\n")
                for label, text in [
                    ("Original", row["original"]),
                    ("Oracle counterfactual", row["reference"]),
                    *[(key, texts[i]) for key, texts in all_outputs.items()],
                ]:
                    stream.write(
                        f"\n### {label}\n\n"
                        + "\n".join("> " + line for line in text.splitlines())
                        + "\n"
                    )
    return result
