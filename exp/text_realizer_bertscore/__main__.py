"""Score saved CV pairs with pinned, unmodified RoBERTa-large BERTScore."""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from exp.text_realizer.protocol import lock
from exp.text_realizer.runtime import gpu_lease
from exp.text_realizer_package_eval.common import (
    digest,
    jsonl_records,
    read_json,
    versions,
)
from src.artifact_io import atomic_output, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "reports/talent_motivation/text-realizer-evaluation/package-v1"
INPUT = RESULTS / "scores/comparisons.jsonl"
OUTPUT = RESULTS / "bertscore-roberta-large"
CACHE = ROOT / ".cache/huggingface/hub"
MODEL = "FacebookAI/roberta-large"
REVISION = "722cf37b1afa9454edce342e7895e588b6ff1d59"
DECODERS = ("256-270m", "256-1b", "768-270m", "768-1b")
CONDITIONS = ("oracle", "predicted")
METRICS = ("precision", "recall", "f1")
FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
)
BATCH_SIZE = 16
CHUNK_SIZE = 64


def snapshot(cache: Path, *, download: bool = False) -> Path:
    """Resolve the exact model revision, downloading only when explicitly requested."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            MODEL,
            revision=REVISION,
            cache_dir=str(cache),
            allow_patterns=list(FILES),
            local_files_only=not download,
            max_workers=2,
        )
    )


def load_pairs(path: Path, decoder: str) -> list[dict]:
    """Validate the complete eight-output comparison file and preserve its ID order."""
    rows = jsonl_records(path)
    ids = [row["id"] for row in rows]
    expected = {f"{model}/{arm}" for model in DECODERS for arm in CONDITIONS}
    if not ids or any(type(key) is not int for key in ids) or len(set(ids)) != len(ids):
        raise ValueError("empty, invalid or duplicate candidate IDs")
    pairs = []
    for row in rows:
        if set(row["generated"]) != expected:
            raise ValueError("requires all eight decoder/condition outputs")
        texts = [row["reference"], *row["generated"].values()]
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("empty or invalid comparison text")
        for arm in CONDITIONS:
            pairs.append(
                {
                    "id": row["id"],
                    "condition": arm,
                    "reference": " ".join(row["reference"].split()),
                    "prediction": " ".join(
                        row["generated"][f"{decoder}/{arm}"].split()
                    ),
                }
            )
    return pairs


def check_lengths(pairs: list[dict], tokenizer) -> int:
    """Reject long inputs before BERTScore's tokenizer can truncate them."""
    texts = {row[key] for row in pairs for key in ("reference", "prediction")}
    maximum = max(
        len(
            tokenizer.encode(
                text, add_special_tokens=True, add_prefix_space=True, truncation=False
            )
        )
        for text in texts
    )
    if maximum > 512:
        raise ValueError(
            f"CV has {maximum} tokens; standard RoBERTa BERTScore permits 512; truncation is forbidden"
        )
    return maximum


def identity(path: Path, model_path: Path) -> dict:
    """Bind scores to input bytes, evaluator weights, settings and implementation."""
    return {
        "input_sha256": sha256_file(path),
        "model": MODEL,
        "revision": REVISION,
        "model_files_sha256": {
            name: sha256_file(model_path / name)
            for name in FILES
            if (model_path / name).is_file()
        },
        "num_layers": 17,
        "idf": False,
        "rescale_with_baseline": False,
        "use_fast_tokenizer": False,
        "dtype": "float32",
        "seed": 42,
        "normalization": "collapse whitespace",
        "max_tokens_including_special": 512,
        "batch_size": BATCH_SIZE,
        "chunk_size": CHUNK_SIZE,
        "versions": versions(
            ("bert-score", "torch", "transformers", "tokenizers", "numpy", "pandas")
        ),
        "source_sha256": sha256_file(__file__),
        "environment_sha256": sha256_file(
            Path(__file__).parent / "environment/uv.lock"
        ),
    }


def checked_rows(
    saved: dict, expected: list[dict], metadata: dict, decoder: str
) -> list[dict]:
    """Reject stale, reordered or corrupted partial results."""
    rows = saved["rows"]
    if saved["identity"] != metadata or saved["decoder"] != decoder:
        raise ValueError(
            "BERTScore inputs/settings changed; use a new output directory"
        )
    if saved["rows_sha256"] != digest(rows) or len(rows) > len(expected):
        raise ValueError("corrupted BERTScore rows")
    for row, pair in zip(rows, expected, strict=False):
        if any(row[key] != pair[key] for key in ("id", "condition")) or any(
            not isinstance(row[key], (int, float)) or not math.isfinite(row[key])
            for key in METRICS
        ):
            raise ValueError("invalid BERTScore row or ID order")
    return rows


def score(path: Path, output: Path, decoder: str, device: str, cache: Path) -> None:
    """Resume one decoder's two conditions; save after each small group of pairs."""
    from bert_score import BERTScorer

    torch.set_num_threads(1)
    torch.manual_seed(42)
    pairs = load_pairs(path, decoder)
    model_path = snapshot(cache)
    metadata = identity(path, model_path)
    target = output / f"{decoder}.json"
    with lock(target.with_suffix(".lock")), gpu_lease(device):
        rows = (
            checked_rows(read_json(target), pairs, metadata, decoder)
            if target.exists()
            else []
        )
        if len(rows) == len(pairs):
            print(f"{decoder}: already complete", flush=True)
            return
        scorer = BERTScorer(
            model_type=str(model_path),
            num_layers=17,
            lang="en",
            device=device,
            batch_size=BATCH_SIZE,
            idf=False,
            rescale_with_baseline=False,
            use_fast_tokenizer=False,
        )
        maximum = check_lengths(pairs, scorer._tokenizer)
        started, offset = perf_counter(), len(rows)
        for start in range(offset, len(pairs), CHUNK_SIZE):
            batch = pairs[start : start + CHUNK_SIZE]
            values = scorer.score(
                [row["prediction"] for row in batch],
                [row["reference"] for row in batch],
                batch_size=BATCH_SIZE,
            )
            for pair, scores in zip(
                batch,
                zip(*(value.tolist() for value in values), strict=True),
                strict=True,
            ):
                if not all(math.isfinite(value) for value in scores):
                    raise RuntimeError("non-finite BERTScore")
                rows.append(
                    {
                        "id": pair["id"],
                        "condition": pair["condition"],
                        **dict(zip(METRICS, scores, strict=True)),
                    }
                )
            elapsed = perf_counter() - started
            per_pair = elapsed / (len(rows) - offset)
            write_json(
                target,
                {
                    "identity": metadata,
                    "decoder": decoder,
                    "rows": rows,
                    "rows_sha256": digest(rows),
                    "complete": len(rows) == len(pairs),
                    "completed": len(rows),
                    "total": len(pairs),
                    "max_input_tokens": maximum,
                    "heartbeat": datetime.now(timezone.utc).isoformat(),
                    "seconds_per_pair": per_pair,
                    "remaining_seconds": per_pair * (len(pairs) - len(rows)),
                    "device": device,
                    "bert_score_hash": scorer.hash.replace(str(model_path), MODEL),
                },
            )
            print(
                f"{decoder}: {len(rows)}/{len(pairs)}; {per_pair:.3f} s/pair; {per_pair * (len(pairs)-len(rows))/60:.1f} min remaining",
                flush=True,
            )


def merge(path: Path, output: Path, cache: Path) -> None:
    """Require all four complete workers and write per-CV scores plus eight means."""
    metadata = identity(path, snapshot(cache))
    summaries, all_rows, files = [], [], {}
    for decoder in DECODERS:
        target = output / f"{decoder}.json"
        saved = read_json(target)
        pairs = load_pairs(path, decoder)
        rows = checked_rows(saved, pairs, metadata, decoder)
        if not saved["complete"] or len(rows) != len(pairs):
            raise ValueError(f"incomplete BERTScore worker: {decoder}")
        files[decoder] = sha256_file(target)
        all_rows.extend({"decoder": decoder, **row} for row in rows)
        for arm in CONDITIONS:
            group = [row for row in rows if row["condition"] == arm]
            summaries.append(
                {
                    "decoder": decoder,
                    "condition": arm,
                    "units": len(group),
                    **{key: mean(row[key] for row in group) for key in METRICS},
                }
            )
    for name, rows in (("summary.csv", summaries), ("per_cv.csv", all_rows)):
        with atomic_output(output / name) as temporary:
            with temporary.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    write_json(
        output / "summary.json",
        {
            "complete": True,
            "identity": metadata,
            "worker_sha256": files,
            "metrics": summaries,
            "limitation": "Semantic similarity is not factual accuracy or evidence of counterfactual fairness.",
        },
    )
    print(f"Complete: {len(all_rows)} pairs in {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "score", "merge"))
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--decoder", choices=DECODERS)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "download":
        print(snapshot(args.cache_dir, download=True))
    elif args.command == "merge":
        merge(args.input, args.output, args.cache_dir)
    else:
        if args.decoder is None:
            parser.error("score requires --decoder")
        score(args.input, args.output, args.decoder, args.device, args.cache_dir)


if __name__ == "__main__":
    main()
