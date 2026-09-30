"""Split CV identities first, then reconstruct complete short passages."""

import csv
import hashlib
import random
import re
from collections import defaultdict
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path, required):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(required) <= set(reader.fieldnames or []):
            raise ValueError(f"missing required columns in {path}")
        result = {}
        for row in reader:
            row_id = int(row["id"])
            if row_id in result:
                raise ValueError(f"duplicate CV id {row_id}")
            result[row_id] = row
    return result


def split_cvs(paths, seed=42, validation_fraction=0.1):
    """Only factual official-training CVs become adaptation examples.

    Test text is read only to detect duplicate-document leakage, never encoded.
    No counterfactual or structured-label file is opened.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    pairs = _rows(paths["pair_index"], {"id", "split"})
    factual = _rows(paths["texts"], {"id", "text"})
    if set(pairs) != set(factual):
        raise ValueError("factual IDs must exactly match pair_index")
    if {row["split"] for row in pairs.values()} != {"train", "test"}:
        raise ValueError("expected nonempty official train and test splits")
    official_train = sorted(i for i, row in pairs.items() if row["split"] == "train")
    test_ids = sorted(set(pairs) - set(official_train))
    shuffled = official_train.copy()
    random.Random(seed).shuffle(shuffled)
    count = max(1, int(len(shuffled) * validation_fraction))
    ids = {"train": sorted(shuffled[count:]), "validation": sorted(shuffled[:count])}
    if min(map(len, ids.values())) < 2:
        raise ValueError("at least two CVs per adaptation partition are required")
    seen = {}
    examples = {key: [] for key in ids}
    for partition, members in {"test": test_ids, **ids}.items():
        for row_id in members:
            text = factual[row_id]["text"].strip()
            normalized = " ".join(text.split())
            if not normalized:
                raise ValueError(f"empty CV {row_id}")
            if normalized in seen and seen[normalized] != partition:
                raise ValueError("duplicate full CV across data partitions")
            seen[normalized] = partition
            if partition != "test":
                examples[partition].append({"id": row_id, "text": text})
    return examples, {
        "seed": seed,
        "train_ids": ids["train"],
        "validation_ids": ids["validation"],
        "excluded_official_test_ids": test_ids,
        "source_sha256": {
            key: file_hash(paths[key]) for key in ("pair_index", "texts")
        },
        "supervision": "factual passage self-reconstruction only",
    }


def passage_records(examples, tokenizer, max_length=32):
    """Greedy whitespace-boundary packing, without dropping document content.

    Each input equals the reconstruction target before its final EOS. Character
    spans cover the stripped original CV exactly. No word/Unicode character is
    split. An individually over-budget word fails loudly instead of truncating.
    The native decoder tokenizer's own normalization/prefix-space policy remains
    in effect for both encoding and reconstruction.
    """
    if max_length != 32:
        raise ValueError("native baseline must retain 32 output slots")
    result = []
    for example in examples:
        text = example["text"].strip()
        if not text:
            raise ValueError("empty CV")
        start = end = 0
        passage = 0

        def emit(stop, text=text, row_id=example["id"]):
            nonlocal start, passage
            content = text[start:stop]
            tokens = tokenizer(content, add_special_tokens=False)["input_ids"]
            if not 0 < len(tokens) < max_length:
                raise ValueError("empty or over-budget passage")
            if tokenizer.eos_token_id in tokens:
                raise ValueError("CV content contains the reserved EOS token")
            result.append(
                {
                    "id": row_id,
                    "passage": passage,
                    "char_start": start,
                    "char_end": stop,
                    "text": content,
                    "input_ids": tokens,
                    "target_ids": tokens + [tokenizer.eos_token_id],
                }
            )
            start = stop
            passage += 1

        for word in re.finditer(r"\S+", text):
            candidate = tokenizer(text[start : word.end()], add_special_tokens=False)[
                "input_ids"
            ]
            if len(candidate) >= max_length:
                if end == start:
                    raise ValueError(
                        f"single word exceeds passage budget in CV {example['id']}"
                    )
                emit(end)
                candidate = tokenizer(
                    text[start : word.end()], add_special_tokens=False
                )["input_ids"]
                if len(candidate) >= max_length:
                    raise ValueError(
                        f"single word exceeds passage budget in CV {example['id']}"
                    )
            end = word.end()
        emit(end)
    return result


def wrong_cv_indices(records):
    """Deterministic negative controls from the next *different* CV, not neighbors.

    Not claimed to be a permutation: different CVs can have different passage
    counts. This is evaluation only, never a training loss.
    """
    groups = defaultdict(list)
    for index, row in enumerate(records):
        groups[row["id"]].append(index)
    ids = sorted(groups)
    if len(ids) < 2:
        raise ValueError("wrong-CV controls require at least two CVs")
    result = [0] * len(records)
    for pos, row_id in enumerate(ids):
        other = groups[ids[(pos + 1) % len(ids)]]
        for offset, index in enumerate(groups[row_id]):
            result[index] = other[offset % len(other)]
    return result


def audit_indices(records, count, seed):
    """One seeded passage per CV, then a seeded subset of CVs."""
    groups = defaultdict(list)
    for index, row in enumerate(records):
        groups[row["id"]].append(index)
    rng = random.Random(seed)
    chosen = [rng.choice(groups[row_id]) for row_id in sorted(groups)]
    rng.shuffle(chosen)
    return chosen[:count]
