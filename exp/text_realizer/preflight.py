"""Bounded GPU readiness checks on development data only; never publish a model."""

from __future__ import annotations

import gc
import math
import subprocess
import sys
from time import perf_counter

import torch

from src.artifact_io import sha256_file, write_json
from src.text_realizer.artifacts import load_text_realizer, save_checkpoint
from src.text_realizer.generation import generate
from src.text_realizer.metrics import text_metrics
from src.text_realizer.model import build_model, cached_decoder, model_spec
from src.text_realizer.training import (
    evaluate_nll,
    optimizer_for,
    restore_resume,
    save_resume,
    scheduler_for,
    seed_all,
    tokenize,
    update,
)

from .data import load_bundle, prepared_path
from .experiment import enable_checkpointing
from .protocol import identity, lock, status
from .settings import ROOT, resolve
from .setup import model_files


def token_audit(config: dict, bundle: dict) -> dict:
    from transformers import AutoConfig, AutoTokenizer

    source = cached_decoder(config["model"], str(resolve(config["paths"]["cache"])))
    common = {"local_files_only": True, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(source, **common)
    architecture = AutoConfig.from_pretrained(source, **common)
    lengths, mismatches = [], 0
    for name in ("fit", "tune"):
        texts = bundle[name]["texts"]
        targets = tokenize(tokenizer, texts)
        lengths.extend(map(len, targets))
        for text, target in zip(texts, targets, strict=True):
            decoded = tokenizer.decode(
                target[:-1],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            mismatches += decoded != text
    lengths.sort()
    multiple = config["generation"]["budget_multiple"]
    maximum = max(
        config["generation"]["minimum_budget"],
        math.ceil(max(lengths) / multiple) * multiple,
    )
    if (
        config["model"]["prefix_tokens"] + 1 + maximum
        > architecture.max_position_embeddings
    ):
        raise ValueError(
            "complete-CV generation budget exceeds context; no truncation allowed"
        )
    return {
        "hidden_size": architecture.hidden_size,
        "generation": {
            "max_new_tokens": maximum,
            "batch_size": config["generation"]["batch_size"],
        },
        "token_lengths_cv_including_eos": {
            "min": lengths[0],
            "median": lengths[len(lengths) // 2],
            "p95": lengths[int(0.95 * (len(lengths) - 1))],
            "max": lengths[-1],
        },
        "tokenizer_raw_roundtrip_mismatches": mismatches,
        "audited_partitions": ["fit", "tune"],
    }


def preflight(config: dict, *, device: str) -> dict:
    prepared = prepared_path(config)
    root = prepared / "diagnostics"
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(
            "GPU preflight requires accessible CUDA; offline tests do not"
        )
    with lock(root / ".preflight.lock"):
        started_identity = identity(config, prepared)
        # A failed rerun cannot leave a previous ready=true marker usable.
        write_json(
            prepared / "preflight.json",
            {"ready": False, "reason": "preflight in progress"},
        )
        try:
            result = _checks(config, prepared, root, device)
            if result["identity"] != started_identity:
                raise ValueError(
                    "source/config/environment changed during preflight; rerun on stable code"
                )
            write_json(prepared / "preflight.json", result)
            status(root, "ready", ready=True)
            return result
        except BaseException as exc:
            status(root, "failed", error=f"{type(exc).__name__}: {exc}")
            raise


def _checks(config, prepared, root, device):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/text_realizer",
            "tests/test_embeddinggemma_encoder.py",
            "tests/test_langvae_ft.py",
            "tests/test_config_paths.py",
            "tests/test_config_propagation.py",
        ],
        cwd=ROOT,
        check=True,
    )
    seed_all(42)
    torch.set_num_threads(1)
    bundle = load_bundle(prepared)
    audit = token_audit(config, bundle)
    checkpoint = root / "diagnostic.pt"
    spec = model_spec(
        config["model"],
        config["preflight"].get("variant", "lora-compact"),
        audit["hidden_size"],
    )
    model = build_model(
        spec,
        audit["generation"],
        device=device,
        cache_dir=str(resolve(config["paths"]["cache"])),
    )
    enable_checkpointing(model, True)
    fit = bundle["fit"]
    targets = tokenize(model.tokenizer, fit["texts"])
    settings, checks = config["training"], config["preflight"]
    native = {}
    if checks.get("native_checks", False):
        from .sanity import check_native_forward, frozen_digest

        status(root, "native_backbone_checks")
        native = check_native_forward(model, fit["z"][:1], targets[0])
        frozen = frozen_digest(model)
    longest = sorted(range(len(targets)), key=lambda i: len(targets[i]), reverse=True)[
        :2
    ]
    microbatch, measurements = None, []
    for size in checks.get("microbatches", (1, 2)):
        optimizer = optimizer_for(model, settings)
        torch.cuda.reset_peak_memory_stats(device)
        try:
            update(
                model,
                optimizer,
                fit["z"],
                targets,
                longest[:size],
                microbatch=size,
                clip=settings["clip"],
            )
            torch.cuda.synchronize(device)
            free, total = torch.cuda.mem_get_info(device)
            peak = torch.cuda.max_memory_reserved(device)
            safe = (
                peak <= total * (1 - checks["memory_headroom"])
                and free >= total * checks["memory_headroom"]
            )
            measurements.append(
                {
                    "microbatch": size,
                    "peak_reserved_bytes": peak,
                    "total_bytes": total,
                    "safe": safe,
                }
            )
            if safe:
                microbatch = size
        except torch.cuda.OutOfMemoryError:
            measurements.append({"microbatch": size, "safe": False, "oom": True})
        finally:
            model.zero_grad(set_to_none=True)
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()
    if microbatch is None:
        raise RuntimeError(
            "FP32 complete targets lack 15% GPU headroom even at microbatch 1"
        )
    optimizer = optimizer_for(model, settings)
    scheduler = scheduler_for(optimizer, 1000, settings["warmup_fraction"])
    before = evaluate_nll(model, fit["z"][:2], targets[:2])["nll"]
    for step in range(checks["smoke_updates"]):
        update(
            model,
            optimizer,
            fit["z"],
            targets,
            [0, 1],
            microbatch=microbatch,
            clip=settings["clip"],
        )
        scheduler.step()
        status(root, "smoke", step=step + 1)
    after = evaluate_nll(model, fit["z"][:2], targets[:2])["nll"]
    if checks.get("native_checks", False):
        from .sanity import check_gradients_and_frozen

        native.update(check_gradients_and_frozen(model, frozen))
    if after >= before:
        raise RuntimeError("20-update smoke did not reduce loss")
    save_resume(
        root / "resume.pt",
        model,
        optimizer,
        scheduler,
        {"step": checks["smoke_updates"], "bad_evaluations": 2},
        identity="diagnostic",
    )
    update(
        model,
        optimizer,
        fit["z"],
        targets,
        [0, 1],
        microbatch=microbatch,
        clip=settings["clip"],
    )
    scheduler.step()
    continued = model.trainable_state()
    restored = restore_resume(
        root / "resume.pt", model, optimizer, scheduler, identity="diagnostic"
    )
    if restored["bad_evaluations"] != 2:
        raise RuntimeError("resume lost early-stopping state")
    update(
        model,
        optimizer,
        fit["z"],
        targets,
        [0, 1],
        microbatch=microbatch,
        clip=settings["clip"],
    )
    scheduler.step()
    resumed_weights = model.trainable_state()
    for name, expected in continued.items():
        torch.testing.assert_close(
            resumed_weights[name], expected, atol=1e-6, rtol=1e-6
        )
    del continued, resumed_weights
    similarities = []
    for step in range(checks["overfit_updates"]):
        update(
            model,
            optimizer,
            fit["z"],
            targets,
            [0, 1],
            microbatch=microbatch,
            clip=settings["clip"],
        )
        if (step + 1) % 25 == 0:
            rows = generate(model, fit["z"][:2])
            similarities = [
                text_metrics(ref, row["text"])["character_similarity"]
                for ref, row in zip(fit["texts"][:2], rows, strict=True)
            ]
            status(
                root,
                "two_example_overfit",
                step=step + 1,
                character_similarity=similarities,
            )
            if min(similarities) >= checks["overfit_character_similarity"]:
                break
    if not similarities or min(similarities) < checks["overfit_character_similarity"]:
        raise RuntimeError(
            "two-example overfit gate failed; diagnose before a full campaign"
        )
    started = perf_counter()
    correct = evaluate_nll(model, fit["z"][:2], targets[:2])["nll"]
    seconds_per_validation_cv = (perf_counter() - started) / 2
    wrong = evaluate_nll(model, fit["z"][:2].flip(0), targets[:2])["nll"]
    if (
        wrong - correct < config["selection"]["nll_gap"]
        or rows[0]["text"] == rows[1]["text"]
    ):
        raise RuntimeError("smoke outputs do not use distinct latent vectors")
    save_checkpoint(
        checkpoint, model, encoder=config["encoder"], provenance={"stage": "diagnostic"}
    )
    # Benchmark optimizer updates at the actual effective batch, not just forward passes.
    started = perf_counter()
    batch = settings["effective_batch"]
    for step in range(checks["benchmark_steps"]):
        indices = [(step * batch + i) % len(targets) for i in range(batch)]
        update(
            model,
            optimizer,
            fit["z"],
            targets,
            indices,
            microbatch=microbatch,
            clip=settings["clip"],
        )
        status(root, "benchmark", step=step + 1)
    torch.cuda.synchronize(device)
    seconds_per_update = (perf_counter() - started) / checks["benchmark_steps"]
    started = perf_counter()
    generate(model, fit["z"][:4])
    torch.cuda.synchronize(device)
    seconds_per_generation = (perf_counter() - started) / 4
    del optimizer, scheduler, model
    gc.collect()
    torch.cuda.empty_cache()
    # Fresh process checks the pinned base + exported trainables, not the live model.
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
    reloaded = load_text_realizer(
        checkpoint, device=device, cache_dir=str(resolve(config["paths"]["cache"]))
    )
    torch.cuda.reset_peak_memory_stats(device)
    texts = reloaded.decode(fit["z"][:2])
    if not checks.get("native_checks", False):
        del reloaded
        gc.collect()
        torch.cuda.empty_cache()
    from .campaign import make_cycle_encoder

    encoder = make_cycle_encoder(config, device)
    original_cycle = encoder.encode(fit["texts"][:2], batch_size=1)
    torch.testing.assert_close(original_cycle, fit["z"][:2], atol=2e-5, rtol=2e-5)
    cycle = encoder.encode(texts, batch_size=1)
    if cycle.shape != (2, 768) or not torch.isfinite(cycle).all():
        raise ValueError("generated CVs cannot be re-encoded by the canonical protocol")
    if checks.get("native_checks", False):
        free, total = torch.cuda.mem_get_info(device)
        peak = torch.cuda.max_memory_reserved(device)
        if (
            peak > total * (1 - checks["memory_headroom"])
            or free < total * checks["memory_headroom"]
        ):
            raise RuntimeError(
                "decoder + frozen cycle encoder lack evaluation memory headroom"
            )
        native["joint_evaluation_peak_reserved_bytes"] = peak
    result = {
        "ready": True,
        "native_checks": native,
        "identity": identity(config, prepared),
        **audit,
        "gpu_name": torch.cuda.get_device_name(device),
        "microbatch": microbatch,
        "memory_measurements": measurements,
        "base_files_sha256": model_files(config),
        "smoke_nll_before": before,
        "smoke_nll_after": after,
        "overfit_character_similarity": similarities,
        "conditioning_nll_gap": wrong - correct,
        "encoder_parity_max_abs": float((original_cycle - fit["z"][:2]).abs().max()),
        "reload_verification_sha256": sha256_file(root / "reload.json"),
        "seconds_per_optimizer_update": seconds_per_update,
        "seconds_per_validation_cv": seconds_per_validation_cv,
        "seconds_per_generated_cv": seconds_per_generation,
        "single_run_training_seconds_upper_bound": seconds_per_update
        * math.ceil(len(targets) / batch)
        * settings["epochs"],
        "note": "ETA excludes setup; epoch validation and final controls add overhead. Diagnostic weights are never promoted.",
    }
    write_json(root / "measurements.json", result)
    return result
