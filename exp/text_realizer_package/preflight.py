"""Real-GPU readiness checks for complete-CV targets at the package's 256-D width."""

from __future__ import annotations

import gc
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from time import perf_counter

import torch

from exp.text_realizer.campaign import make_cycle_encoder
from exp.text_realizer.experiment import enable_checkpointing
from exp.text_realizer.preflight import token_audit
from exp.text_realizer.protocol import read_json, status
from exp.text_realizer.sanity import (
    check_gradients_and_frozen,
    check_native_forward,
    frozen_digest,
)
from exp.text_realizer.setup import model_files
from src.artifact_io import sha256_file, write_json
from src.text_realizer.artifacts import load_text_realizer, save_checkpoint
from src.text_realizer.generation import generate
from src.text_realizer.metrics import text_metrics
from src.text_realizer.model import build_model, model_spec
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
from .settings import ROOT, identity, packages, require_current, resolve


def environment_check(config: dict) -> dict:
    versions = packages()
    expected = {
        "torch": "2.6.0",
        "transformers": "4.57.6",
        "sentence-transformers": "5.1.2",
        "peft": "0.17.1",
    }
    if not versions["python"].startswith("3.12.") or any(
        versions[key].split("+")[0] != value for key, value in expected.items()
    ):
        raise ValueError(
            "use the existing pinned text-realizer Python 3.12 environment"
        )
    files = model_files(config)
    manifest = read_json(resolve(config["inputs"]["manifest"]))
    expected_encoder = manifest["protocol"]["encoding"]["model_files_sha256"]
    if files["encoder"] != expected_encoder:
        raise ValueError(
            "cached encoder files differ from the package encoding provenance"
        )
    return files


def estimate(config: dict, bundle: dict, measurements: dict) -> dict:
    epochs = config["training"]["epochs"]
    steps = math.ceil(len(bundle["fit"]["ids"]) / config["training"]["effective_batch"])
    count = len(bundle["tune"]["ids"])
    fit_seconds = epochs * steps * measurements["seconds_per_optimizer_update"]
    validation = epochs * count * measurements["seconds_per_validation_cv"]
    # Correct/shuffled/nearest plus two constant conditions; cycle encoding is separate.
    diagnostics = (3 * count + 2) * measurements[
        "seconds_per_generated_cv"
    ] + 5 * count * measurements["seconds_per_validation_cv"]
    nominal = fit_seconds + validation + diagnostics
    padded = nominal * 1.25
    return {
        "maximum_epochs": epochs,
        "training_seconds_at_maximum_epochs": fit_seconds,
        "validation_seconds_at_maximum_epochs": validation,
        "development_diagnostics_seconds": diagnostics,
        "remaining_seconds_with_25_percent_margin": padded,
        "estimated_finish_utc_if_training_starts_now": (
            datetime.now(timezone.utc) + timedelta(seconds=padded)
        ).isoformat(),
        "note": "Measured estimate, not a guarantee; early stopping may finish sooner. Excludes GPU queue and remaining preflight time. Margin covers cycle encoding, saving and evaluation overhead.",
    }


def memory_check(
    model, fit: dict, targets: list, config: dict, device: str
) -> tuple[int, list]:
    checks, settings = config["preflight"], config["training"]
    longest = sorted(range(len(targets)), key=lambda i: len(targets[i]), reverse=True)[
        :2
    ]
    measurements, chosen = [], None
    for size in (1, 2):
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
                    "safe": safe,
                    "peak_reserved_bytes": peak,
                    "total_bytes": total,
                }
            )
            if safe:
                chosen = size
        except torch.cuda.OutOfMemoryError:
            measurements.append({"microbatch": size, "safe": False, "oom": True})
        finally:
            model.zero_grad(set_to_none=True)
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()
    if chosen is None:
        raise RuntimeError("complete CV targets cannot fit with 15% memory headroom")
    return chosen, measurements


def learning_checks(
    model, fit: dict, targets: list, config: dict, root, microbatch: int
) -> dict:
    settings, checks = config["training"], config["preflight"]
    status(root, "native_backbone_checks")
    native = check_native_forward(model, fit["z"][:1], targets[0])
    write_json(root / "native_checks.json", native)
    frozen = frozen_digest(model)
    optimizer = optimizer_for(model, settings)
    scheduler = scheduler_for(optimizer, 1000, settings["warmup_fraction"])

    def step():
        update(
            model,
            optimizer,
            fit["z"],
            targets,
            [0, 1],
            microbatch=microbatch,
            clip=settings["clip"],
        )

    before = evaluate_nll(model, fit["z"][:2], targets[:2])["nll"]
    for i in range(checks["smoke_updates"]):
        step()
        scheduler.step()
        status(root, "smoke", step=i + 1)
    after = evaluate_nll(model, fit["z"][:2], targets[:2])["nll"]
    native.update(check_gradients_and_frozen(model, frozen))
    if after >= before:
        raise RuntimeError("smoke updates failed to reduce reconstruction loss")
    position = {
        "step": checks["smoke_updates"],
        "order": [1, 0],
        "offset": 2,
        "bad_evaluations": 2,
    }
    save_resume(
        root / "resume.pt",
        model,
        optimizer,
        scheduler,
        position,
        identity="package-diagnostic",
    )
    step()
    scheduler.step()
    continued = model.trainable_state()
    if (
        restore_resume(
            root / "resume.pt",
            model,
            optimizer,
            scheduler,
            identity="package-diagnostic",
        )
        != position
    ):
        raise RuntimeError("resume lost progress or early-stopping history")
    step()
    scheduler.step()
    for name, value in model.trainable_state().items():
        torch.testing.assert_close(value, continued[name], atol=1e-6, rtol=1e-6)
    del continued
    similarities = []
    for i in range(checks["overfit_updates"]):
        step()
        if (i + 1) % 25 == 0 or i + 1 == checks["overfit_updates"]:
            rows = generate(model, fit["z"][:2])
            similarities = [
                text_metrics(text, row["text"])["character_similarity"]
                for text, row in zip(fit["texts"][:2], rows, strict=True)
            ]
            status(
                root,
                "two_example_overfit",
                step=i + 1,
                character_similarity=similarities,
            )
            if min(similarities) >= checks["overfit_character_similarity"]:
                break
    if not similarities or min(similarities) < checks["overfit_character_similarity"]:
        raise RuntimeError("two-example overfit failed; full training remains blocked")
    started = perf_counter()
    correct = evaluate_nll(model, fit["z"][:2], targets[:2])["nll"]
    validation_seconds = (perf_counter() - started) / 2
    wrong = evaluate_nll(model, fit["z"][:2].flip(0), targets[:2])["nll"]
    if (
        wrong - correct < config["selection"]["nll_gap"]
        or rows[0]["text"] == rows[1]["text"]
    ):
        raise RuntimeError(
            "two-example model does not distinguish conditioning vectors"
        )
    return {
        "native_checks": native,
        "smoke_nll_before": before,
        "smoke_nll_after": after,
        "overfit_character_similarity": similarities,
        "conditioning_nll_gap": wrong - correct,
        "resume_verified": True,
        "seconds_per_validation_cv": validation_seconds,
    }


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
    if cycle.shape != (2, 256) or not torch.isfinite(cycle).all():
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
