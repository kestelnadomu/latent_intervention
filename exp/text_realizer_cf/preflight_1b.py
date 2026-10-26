"""1B reference integrity and inference capacity, using four fit examples only."""

from __future__ import annotations

import gc
from time import perf_counter

import torch

from exp.text_realizer.campaign import make_cycle_encoder
from src.text_realizer.artifacts import verify_checkpoint
from src.text_realizer.generation import generate
from src.text_realizer.training import evaluate_nll, tokenize

from .model import clear_gpu, load_reference
from .protocol import read_json
from .settings import resolve


def inference_check(config: dict, prepared, fit: dict, device: str) -> dict:
    verification = verify_checkpoint(
        resolve(config["inputs"]["baseline_checkpoint"]),
        device=device,
        cache_dir=str(resolve(config["paths"]["cache"])),
    )
    gc.collect()
    clear_gpu(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = load_reference(config, device=device)
    encoder = make_cycle_encoder(
        {
            "encoder": read_json(prepared / "manifest.json")["encoder"],
            "paths": config["paths"],
        },
        device,
    )
    z = torch.cat((fit["factual"]["z"][:2], fit["changed"]["z"][:2]))
    texts = fit["factual"]["texts"][:2] + fit["changed"]["texts"][:2]
    targets = tokenize(model.tokenizer, texts)
    started = perf_counter()
    evaluate_nll(model, z, targets)
    torch.cuda.synchronize(device)
    nll_seconds = perf_counter() - started
    started = perf_counter()
    rows = generate(model, z)
    torch.cuda.synchronize(device)
    generation_seconds = perf_counter() - started
    started = perf_counter()
    cycles = encoder.encode([row["text"] for row in rows], batch_size=1)
    torch.cuda.synchronize(device)
    cycle_seconds = perf_counter() - started
    if cycles.shape != z.shape or not torch.isfinite(cycles).all():
        raise ValueError("invalid cycle encoder output during joint memory check")
    free, total = torch.cuda.mem_get_info(device)
    peak = torch.cuda.max_memory_reserved(device)
    headroom = config["preflight"]["memory_headroom"]
    if peak > total * (1 - headroom) or free < total * headroom:
        raise RuntimeError("joint 1B decoder/cycle inference lacks 15% GPU headroom")
    result = {
        "baseline_reload_verified": verification["verified"],
        "baseline_checkpoint_sha256": verification["checkpoint_sha256"],
        "examples": len(z),
        "partitions": ["fit_factual", "fit_changed"],
        "seconds_per_generated_cv": generation_seconds / len(z),
        "seconds_per_nll_cv": nll_seconds / len(z),
        "seconds_per_cycle_cv": cycle_seconds / len(z),
        "generated_tokens": [row["tokens"] for row in rows],
        "peak_reserved_bytes": peak,
        "total_bytes": total,
        "free_bytes": free,
    }
    del model, encoder
    gc.collect()
    clear_gpu(device)
    return result
