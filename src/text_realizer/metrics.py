"""Lexical reconstruction, independent of encoder similarity or semantic g."""

from __future__ import annotations

import difflib
import math
import random
from statistics import mean


def normalize(text: str) -> str:
    return " ".join(text.split())


def text_metrics(reference: str, prediction: str) -> dict:
    from rapidfuzz.distance import LCSseq, Levenshtein
    from sacrebleu.metrics import CHRF

    truth, output = normalize(reference), normalize(prediction)
    a, b = truth.split(), output.split()
    # Unit-cost Levenshtein alignment, with rapidfuzz's deterministic tie policy.
    operations = Levenshtein.editops(a, b)
    errors_on_reference = sum(op.tag in {"replace", "delete"} for op in operations)
    lcs = LCSseq.similarity(a, b)
    return {
        "exact_raw": float(reference == prediction),
        "exact_normalized": float(truth == output),
        "character_similarity": Levenshtein.normalized_similarity(truth, output),
        "aligned_token_accuracy": (len(a) - errors_on_reference) / max(1, len(a)),
        "token_error_rate": len(operations) / max(1, len(a)),
        "rouge_l": 2 * lcs / max(1, len(a) + len(b)),
        "chrf": CHRF().sentence_score(prediction, [reference]).score / 100,
        "reference_words": len(a),
        "generated_words": len(b),
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot summarize an empty evaluation")
    keys = set.intersection(*(set(row) for row in rows))
    return {
        key: mean(row[key] for row in rows)
        for key in sorted(keys)
        if all(
            isinstance(row[key], (int, float)) and math.isfinite(row[key])
            for row in rows
        )
    }


def nearest_copy(prediction: str, fit_texts: list[str]) -> dict:
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein

    normalized = [normalize(text) for text in fit_texts]
    text = normalize(prediction)
    _, score, index = process.extractOne(
        text, normalized, scorer=Levenshtein.normalized_similarity
    )
    a, b = text.split(), normalized[index].split()
    span = (
        difflib.SequenceMatcher(None, a, b, autojunk=False)
        .find_longest_match(0, len(a), 0, len(b))
        .size
    )
    return {
        "fit_index": index,
        "similarity": score,
        "exact_fit_copy": text in normalized,
        "longest_shared_words_nearest_lexical_fit": span,
        "shared_span_ge_20_words": span >= 20,
    }


def paired_bootstrap(differences: list[float], *, samples: int, seed: int) -> dict:
    rng = random.Random(seed)
    estimates = sorted(
        mean(rng.choices(differences, k=len(differences))) for _ in range(samples)
    )
    return {
        "mean": mean(differences),
        "lower_95": estimates[int(0.025 * (samples - 1))],
        "upper_95": estimates[int(0.975 * (samples - 1))],
        "samples": samples,
        "seed": seed,
    }
