"""The established GPU checks, applied to full 768-D package vectors."""

from __future__ import annotations

import gc
import subprocess
import sys
from time import perf_counter

import torch

from exp.text_realizer.campaign import make_cycle_encoder
from exp.text_realizer.experiment import enable_checkpointing
from exp.text_realizer.preflight import token_audit
from exp.text_realizer.protocol import read_json, status
from exp.text_realizer_package.preflight import (
    environment_check as environment_check,
    estimate,
    learning_checks,
    memory_check,
)
from src.artifact_io import sha256_file, write_json
from src.text_realizer.artifacts import load_text_realizer, save_checkpoint
from src.text_realizer.generation import generate
from src.text_realizer.model import build_model, model_spec
from src.text_realizer.training import optimizer_for, seed_all, tokenize, update

from .data import load_bundle, prepared_path
from .settings import ROOT, identity, require_current, resolve


def preflight(config: dict, reports, *, device: str) -> dict:
    root = reports / "preflight"
    write_json(root / "ready.json", {"ready": False, "reason": "checks in progress"})
    try:
        return _checks(config, reports, device=device)
    except BaseException as exc:
        reason = f"{type(exc).__name__}: {exc}"
        write_json(root / "ready.json", {"ready": False, "reason": reason})
        status(root, "failed", error=reason)
        raise


def _checks(config: dict, reports, *, device: str) -> dict:
    prepared = prepared_path(config)
    require_current(reports, config, prepared)
    root = reports / "preflight"
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("accessible CUDA required for preflight")
    base_files = environment_check(config)
    seed_all(config["seed"])
    torch.set_num_threads(1)
    bundle = load_bundle(prepared)
    audit = token_audit(config, bundle)
    write_json(root / "token_audit.json", audit)
    spec = model_spec(config["model"], "lora-compact", audit["hidden_size"])
    model = build_model(
        spec,
        audit["generation"],
        device=device,
        cache_dir=resolve(config["paths"]["cache"]),
    )
    enable_checkpointing(model, True)
    fit = bundle["fit"]
    targets = tokenize(model.tokenizer, fit["texts"])
    status(root, "memory_check")
    microbatch, memory = memory_check(model, fit, targets, config, device)
    write_json(root / "memory.json", {"microbatch": microbatch, "measurements": memory})
    measured = learning_checks(model, fit, targets, config, root, microbatch)
    write_json(root / "learning_checks.json", measured)
    checkpoint = root / "diagnostic.pt"
    save_checkpoint(
        checkpoint,
        model,
        encoder=config["encoder"],
        provenance={"stage": "diagnostic-never-deploy"},
    )
    started = perf_counter()
    generate(model, fit["z"][:4])
    torch.cuda.synchronize(device)
    measured["seconds_per_generated_cv"] = (perf_counter() - started) / 4
    optimizer = optimizer_for(model, config["training"])
    started = perf_counter()
    batch = config["training"]["effective_batch"]
    for i in range(config["preflight"]["benchmark_steps"]):
        indices = [(i * batch + j) % len(targets) for j in range(batch)]
        update(
            model,
            optimizer,
            fit["z"],
            targets,
            indices,
            microbatch=microbatch,
            clip=config["training"]["clip"],
        )
        torch.cuda.synchronize(device)
        measured["seconds_per_optimizer_update"] = (perf_counter() - started) / (i + 1)
        status(
            root,
            "benchmark",
            step=i + 1,
            seconds_per_update=measured["seconds_per_optimizer_update"],
        )
        if (i + 1) % 5 == 0:
            write_json(reports / "eta.json", estimate(config, bundle, measured))
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "exp.text_realizer",
            "verify",
            "--checkpoint",
            str(checkpoint),
            "--device",
            device,
            "--cache-dir",
            str(resolve(config["paths"]["cache"])),
            "--output",
            str(root / "reload.json"),
            "--inherited-gpu-lease",
        ],
        cwd=ROOT,
        check=True,
    )
    model = load_text_realizer(
        checkpoint, device=device, cache_dir=resolve(config["paths"]["cache"])
    )
    torch.cuda.reset_peak_memory_stats(device)
    texts = model.decode(fit["z"][:2])
    encoder = make_cycle_encoder(config, device)
    original = encoder.encode(fit["texts"][:2], batch_size=1)
    torch.testing.assert_close(original, fit["z"][:2], atol=2e-5, rtol=2e-5)
    cycle = encoder.encode(texts, batch_size=1)
    if cycle.shape != (2, 768) or not torch.isfinite(cycle).all():
        raise ValueError("cycle encoding is incompatible with the package vectors")
    free, total = torch.cuda.mem_get_info(device)
    peak = torch.cuda.max_memory_reserved(device)
    if (
        peak > total * (1 - config["preflight"]["memory_headroom"])
        or free < total * config["preflight"]["memory_headroom"]
    ):
        raise RuntimeError("decoder plus cycle encoder lack 15% evaluation headroom")
    require_current(reports, config, prepared)
    result = {
        "ready": True,
        "identity": identity(config, prepared),
        **audit,
        **measured,
        "gpu_name": torch.cuda.get_device_name(device),
        "microbatch": microbatch,
        "memory_measurements": memory,
        "joint_evaluation_peak_reserved_bytes": peak,
        "base_files_sha256": base_files,
        "encoder_parity_max_abs": float((original - fit["z"][:2]).abs().max()),
        "reload_verification_sha256": sha256_file(root / "reload.json"),
    }
    write_json(root / "ready.json", result)
    write_json(reports / "eta.json", estimate(config, bundle, result))
    status(root, "ready")
    return result


def readiness(config: dict, reports, *, device: str) -> dict:
    require_current(reports, config, prepared_path(config))
    ready = read_json(reports / "preflight/ready.json")
    if (
        not ready["ready"]
        or ready["identity"] != identity(config, prepared_path(config))
        or ready["gpu_name"] != torch.cuda.get_device_name(device)
        or ready["base_files_sha256"] != environment_check(config)
        or ready["reload_verification_sha256"]
        != sha256_file(reports / "preflight/reload.json")
    ):
        raise ValueError(
            "passing preflight for this code, data, environment and hardware is required"
        )
    return ready
