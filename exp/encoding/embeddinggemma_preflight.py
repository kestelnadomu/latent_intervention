"""Audit the corpus and run bounded inference; never publish production latents."""

from datetime import datetime, timezone
from time import perf_counter

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.config import EMBEDDINGGEMMA_PROMPT
from src.encoder.embeddinggemma import (
    cached_snapshot,
    format_text,
    make_embeddinggemma_encoder,
)
from src.pair_encoding import _input_hashes, _load_csv, _load_pair_index


def run_preflight(configs: dict[int, dict]) -> dict:
    started = perf_counter()
    config = configs[768]
    enc, paths = config["encoder"], config["paths"]
    before = _input_hashes(paths)
    pairs = _load_pair_index(paths["pair_index"])
    factual = _load_csv(paths["texts"], "factual texts", {"id", "text"})
    counterfactual = _load_csv(
        paths["texts_counterfactual"], "counterfactual texts", {"id", "text"}
    )
    ids = pairs["id"].tolist()
    test = pairs.loc[pairs["split"] == "test"]
    test_ids = test["id"].tolist()
    if factual["id"].tolist() != ids or counterfactual["id"].tolist() not in (
        ids,
        test_ids,
    ):
        raise ValueError(
            "EmbeddingGemma corpus IDs do not align with the official pairs"
        )
    cf = counterfactual.set_index("id").loc[test_ids]
    factual_indexed = factual.set_index("id")
    for row in test.itertuples():
        if (
            row.is_identity
            and factual_indexed.at[row.id, "text"] != cf.at[row.id, "text"]
        ):
            raise ValueError(f"identity text differs for id={row.id}")
    raw = factual["text"].tolist() + cf["text"].tolist()
    formatted = [format_text(text) for text in raw]
    tokenizer = AutoTokenizer.from_pretrained(
        cached_snapshot(enc),
        padding_side="right",
        trust_remote_code=False,
        local_files_only=True,
    )
    lengths = []
    for start in range(0, len(formatted), 128):
        tokens = tokenizer(
            formatted[start : start + 128],
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )
        lengths.extend(map(len, tokens["input_ids"]))
    if max(lengths) > int(enc["max_len"]):
        raise ValueError(
            f"EmbeddingGemma corpus needs {max(lengths)} tokens, configured limit is {enc['max_len']}"
        )
    print(
        f"Token audit: {len(raw)} texts; maximum {max(lengths)} / {enc['max_len']} tokens; no truncation",
        flush=True,
    )
    selected = list(range(min(4, len(factual))))
    selected += [
        len(factual) + i
        for i, identity in enumerate(test["is_identity"].tolist())
        if not identity
    ][:2]
    selected.append(max(range(len(raw)), key=lambda i: lengths[i]))
    selected = list(dict.fromkeys(selected))
    sample = [raw[i] for i in selected]
    model = make_embeddinggemma_encoder(enc)
    model_lengths = (
        model.model.tokenize([format_text(text) for text in sample])["attention_mask"]
        .sum(dim=1)
        .tolist()
    )
    if model_lengths != [lengths[i] for i in selected]:
        raise ValueError(
            "EmbeddingGemma token audit differs from the actual SentenceTransformer tokenizer"
        )
    timed = perf_counter()
    full = model.encode(sample, batch_size=int(enc["batch_size"]))
    inference_seconds = perf_counter() - timed
    repeated = model.encode(sample, batch_size=1)
    if not torch.allclose(full, repeated, atol=1e-6, rtol=1e-5):
        raise ValueError(
            "EmbeddingGemma embeddings depend materially on batch size/padding"
        )
    # Independent call through the official library's named prompt, without our formatter.
    official = (
        model.model.encode(
            sample[:2],
            prompt_name="Classification",
            normalize_embeddings=True,
            convert_to_tensor=True,
            batch_size=2,
            show_progress_bar=False,
        )
        .cpu()
        .float()
    )
    if model.model.prompts.get(
        "Classification"
    ) != EMBEDDINGGEMMA_PROMPT or not torch.allclose(
        full[:2], official, atol=1e-6, rtol=1e-5
    ):
        raise ValueError(
            "EmbeddingGemma wrapper differs from the official classification recipe"
        )
    dimensions = []
    for dim in sorted(configs):
        model.latent_dim = dim
        direct = model.encode(sample[:2], batch_size=2)
        derived = F.normalize(full[:2, :dim], p=2, dim=1)
        if not torch.allclose(direct, derived, atol=1e-6, rtol=1e-5):
            raise ValueError(
                f"EmbeddingGemma {dim}D direct/derived embeddings disagree"
            )
        dimensions.append(
            {
                "dimension": dim,
                "max_direct_vs_derived_difference": (direct - derived)
                .abs()
                .max()
                .item(),
            }
        )
    if _input_hashes(paths) != before:
        raise ValueError("source inputs changed during EmbeddingGemma preflight")
    texts_to_encode = len(factual) + int((~test["is_identity"]).sum())
    return {
        "status": "passed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus_texts_audited": len(raw),
        "factual_units": len(factual),
        "counterfactual_test_units": len(cf),
        "identity_test_units": int(test["is_identity"].sum()),
        "maximum_tokens": max(lengths),
        "configured_max_length": enc["max_len"],
        "truncated_texts": 0,
        "smoke_texts": len(sample),
        "max_batch_size_difference": (full - repeated).abs().max().item(),
        "max_official_recipe_difference": (full[:2] - official).abs().max().item(),
        "dimensions": dimensions,
        "input_sha256": before,
        "sample_inference_seconds": inference_seconds,
        "texts_to_encode": texts_to_encode,
        "rough_full_pass_seconds_from_small_sample": inference_seconds
        / len(sample)
        * texts_to_encode,
        "timing_caveat": "Small sample including longest CV; not a full-run benchmark.",
        "elapsed_seconds": perf_counter() - started,
        "full_encoding_started": False,
    }
