"""Factual reconstruction and conditioning controls; no oracle imports or paths."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

import torch
import torch.nn.functional as F

from src.text_realizer.generation import generate
from src.text_realizer.metrics import aggregate, nearest_copy, normalize, text_metrics
from src.text_realizer.training import evaluate_nll, tokenize

from . import facts


def derangement(size: int, seed: int = 20261007) -> torch.Tensor:
    if size < 2:
        raise ValueError("conditioning controls need at least two distinct IDs")
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed))
    result = torch.empty(size, dtype=torch.long)
    result[order] = order.roll(1)
    return result


def nearest_indices(z: torch.Tensor, fit_z: torch.Tensor) -> torch.Tensor:
    return (F.normalize(z.float(), dim=1) @ F.normalize(fit_z.float(), dim=1).T).argmax(
        1
    )


def retrieval(part: dict, fit: dict) -> dict:
    nearest = nearest_indices(part["z"], fit["z"]).tolist()
    rows = [
        {
            "id": key,
            "text": fit["texts"][index],
            "fit_id": fit["ids"][index],
            **text_metrics(reference, fit["texts"][index]),
        }
        for key, reference, index in zip(
            part["ids"], part["texts"], nearest, strict=True
        )
    ]
    return {
        "kind": "retrieval, not a generative decoder",
        "summary": aggregate(rows),
        "rows": rows,
    }


def cycle_metrics(
    texts: list[str], z: torch.Tensor, fit: dict, encoder, progress=None
) -> tuple[dict, list[dict]]:
    vectors = torch.zeros(len(texts), z.shape[1])
    valid = torch.zeros(len(texts), dtype=torch.bool)
    errors = {}
    # Per-example failures (empty/overlong generation) remain explicit; no truncation or dropping.
    for i, text in enumerate(texts):
        try:
            vectors[i] = encoder.encode([text], batch_size=1)[0]
            valid[i] = True
        except ValueError as exc:
            errors[str(i)] = str(exc)
        if progress is not None:
            progress("cycle_encoding", i + 1, len(texts))
    similarity = F.normalize(vectors, dim=1) @ F.normalize(z.float(), dim=1).T
    nearest = nearest_indices(vectors, fit["z"])
    rows = []
    for i in range(len(texts)):
        order = similarity[i].argsort(descending=True, stable=True)
        rows.append(
            {
                "cycle_valid": bool(valid[i]),
                "cycle_cosine": float(similarity[i, i]) if valid[i] else -1.0,
                "cycle_l2": float((vectors[i] - z[i]).norm()) if valid[i] else None,
                "target_retrieval_rank": (
                    int((order == i).nonzero()[0]) + 1 if valid[i] else None
                ),
                "nearest_fit_embedding_id": (
                    fit["ids"][int(nearest[i])] if valid[i] else None
                ),
            }
        )
    good = [row for row in rows if row["cycle_valid"]]
    return {
        "valid_fraction": float(valid.float().mean()),
        "target_retrieval_candidates": len(texts),
        "target_retrieval_scope": "input vectors in this evaluated partition; stable input-order tie break",
        "cosine_invalid_scored_minus_one": mean(row["cycle_cosine"] for row in rows),
        "l2_valid_only": mean(row["cycle_l2"] for row in good) if good else None,
        "retrieval_rank_valid_only": (
            mean(row["target_retrieval_rank"] for row in good) if good else None
        ),
        "errors": errors,
    }, rows


def group_report(rows: list[dict], groups: dict, fit_groups: dict) -> dict:
    seen = {(row["template_id"], row["persona_id"]) for row in fit_groups.values()}
    result = {}
    for field in ("template_id", "persona_id", "combination_seen_in_fit"):
        buckets = defaultdict(list)
        for row in rows:
            group = groups[str(row["id"])]
            value = (
                str((group["template_id"], group["persona_id"]) in seen)
                if field == "combination_seen_in_fit"
                else group[field]
            )
            buckets[value].append(row["character_similarity"])
        result[field] = {
            key: {"count": len(values), "character_similarity": mean(values)}
            for key, values in sorted(buckets.items())
        }
    return result


def evaluate_model(
    model,
    part: dict,
    fit: dict,
    *,
    encoder,
    groups: dict | None = None,
    fit_groups: dict | None = None,
    controls=True,
    microbatch=1,
    progress=None,
) -> dict:
    targets = tokenize(model.tokenizer, part["texts"])
    near = nearest_indices(part["z"], fit["z"])
    conditions = {"correct": part["z"]}
    if controls:
        conditions.update(
            {
                "shuffled": part["z"][derangement(len(part["ids"]))],
                "mean_fit": fit["z"].mean(0).expand(len(part["ids"]), -1),
                "zero": torch.zeros_like(part["z"]),
                "nearest_fit_vector": fit["z"][near],
            }
        )
    outputs = {}
    for name, vectors in conditions.items():
        nll = evaluate_nll(
            model, vectors, targets, microbatch=microbatch, progress=progress
        )
        constant = name in {"mean_fit", "zero"}
        generated = generate(
            model, vectors[:1] if constant else vectors, progress=progress
        )
        if constant:
            generated = generated * len(vectors)
        rows = []
        for key, reference, target, row in zip(
            part["ids"], part["texts"], targets, generated, strict=True
        ):
            rows.append(
                {
                    "id": key,
                    **row,
                    **text_metrics(reference, row["text"]),
                    "reference_tokens": len(target),
                    "eos": float(row["termination"] == "eos"),
                    "max_length": float(row["termination"] == "max_length"),
                    "premature_eos": float(
                        row["termination"] == "eos" and row["tokens"] < len(target) / 2
                    ),
                }
            )
        outputs[name] = {"nll": nll["nll"], "summary": aggregate(rows), "rows": rows}
    correct = outputs["correct"]
    texts = [row["text"] for row in correct["rows"]]
    cycle, cycle_rows = cycle_metrics(texts, part["z"], fit, encoder, progress=progress)
    for row, extra in zip(correct["rows"], cycle_rows, strict=True):
        copy = nearest_copy(row["text"], fit["texts"])
        copy["fit_id"] = fit["ids"][copy.pop("fit_index")]
        row.update(extra)
        row["memorization"] = copy
    result = {
        "conditions": outputs,
        "cycle": cycle,
        "facts": facts.evaluate(part["texts"], texts),
        "retrieval": retrieval(part, fit),
        "output_uniqueness": len(set(map(normalize, texts))) / len(texts),
        "exact_fit_copy_rate": mean(
            row["memorization"]["exact_fit_copy"] for row in correct["rows"]
        ),
        "metric_policy": "whitespace-normalized character similarity; unit-cost token Levenshtein alignment; ROUGE-L word LCS F1; chrF beta=2/order=6",
    }
    if controls:
        result["conditioning"] = {
            "nll_gap": outputs["shuffled"]["nll"] - correct["nll"],
            "character_gap": correct["summary"]["character_similarity"]
            - outputs["shuffled"]["summary"]["character_similarity"],
        }
    if groups is not None:
        result["groups"] = group_report(correct["rows"], groups, fit_groups)
    return result
