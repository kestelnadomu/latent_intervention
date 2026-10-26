"""CPU data checks followed by bounded, non-promotable GPU readiness checks."""

from __future__ import annotations

import gc
import subprocess
import sys
from time import perf_counter

import torch

from exp.text_realizer.experiment import enable_checkpointing
from exp.text_realizer.runtime import gpu_lease
from exp.text_realizer.sanity import (
    check_gradients_and_frozen,
    check_native_forward,
    frozen_digest,
)
from src.artifact_io import sha256_file, write_json
from src.text_realizer.artifacts import save_checkpoint, verify_checkpoint
from src.text_realizer.model import cached_decoder
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

from .data import load_partition, prepare
from .model import clear_gpu, load_reference, numerical_backend
from .protocol import identity, lock, read_json, status
from .settings import GEMMA1B_MODEL, ROOT, resolve


def token_audit(config: dict, fit: dict, selection: dict) -> dict:
    from transformers import AutoConfig, AutoTokenizer

    source = cached_decoder(config["model"], str(resolve(config["paths"]["cache"])))
    common = {"local_files_only": True, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(source, **common)
    architecture = AutoConfig.from_pretrained(source, **common)
    lengths, mismatches = [], 0
    for part in (fit["factual"], fit["changed"], selection["factual"]):
        encoded = tokenize(tokenizer, part["texts"])
        lengths.extend(map(len, encoded))
        for text, tokens in zip(part["texts"], encoded, strict=True):
            roundtrip = tokenizer.decode(
                tokens[:-1],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            mismatches += roundtrip != text
    if not lengths or mismatches:
        raise ValueError("empty targets or tokenizer round-trip mismatch")
    if (
        config["model"]["prefix_tokens"] + 1 + max(lengths)
        > architecture.max_position_embeddings
    ):
        raise ValueError("complete CV target exceeds decoder context")
    if (
        config["model"]["prefix_tokens"] + 1 + config["generation"]["max_new_tokens"]
        > architecture.max_position_embeddings
    ):
        raise ValueError("generation budget exceeds decoder context")
    return {
        "target_count": len(lengths),
        "target_min_including_eos": min(lengths),
        "target_max_including_eos": max(lengths),
        "tokenizer_roundtrip_mismatches": mismatches,
        "hidden_size": architecture.hidden_size,
        "context": architecture.max_position_embeddings,
        "target_partitions": ["fit_factual", "fit_changed", "selection_factual"],
    }


def preflight(config: dict, *, device: str) -> dict:
    prepared = prepare(config)
    root = prepared / "diagnostics"
    with lock(root / ".preflight.lock"):
        fit = load_partition(prepared, "fit")
        selection = load_partition(prepared, "selection")
        token = token_audit(config, fit, selection)
        before = identity(config, prepared)
        marker = prepared / "preflight.json"
        result = {
            "ready": False,
            "data_ready": True,
            "identity": before,
            "token_audit": token,
            "device": device,
        }
        write_json(marker, result)
        status(root, "data_ready", token_audit=token)
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/text_realizer_cf",
                    "tests/text_realizer/test_model.py",
                    "tests/text_realizer/test_artifacts.py",
                ],
                cwd=ROOT,
                check=True,
            )
            result["offline_tests_passed"] = True
            if not device.startswith("cuda:") or not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA driver/device unavailable; GPU preflight not run"
                )
            with gpu_lease(device):
                result.update(_gpu_checks(config, prepared, fit, device))
                if config["model"]["name"] == GEMMA1B_MODEL:
                    from .preflight_1b import inference_check

                    status(root, "inference_benchmark")
                    result["inference"] = inference_check(config, prepared, fit, device)
            if identity(config, prepared) != before:
                raise ValueError("code/config/environment changed during preflight")
            result["ready"] = True
            write_json(marker, result)
            status(
                root,
                "ready",
                gpu_name=result["gpu_name"],
                microbatch=result["microbatch"],
            )
            return result
        except BaseException as exc:
            result["reason"] = f"{type(exc).__name__}: {exc}"
            write_json(marker, result)
            status(root, "failed", error=result["reason"], data_ready=True)
            raise


def _gpu_checks(config: dict, prepared, fit: dict, device: str) -> dict:
    seed_all(42)
    torch.set_num_threads(1)
    factual, changed = fit["factual"], fit["changed"]
    model = load_reference(config, device=device)
    if numerical_backend(model) != "eager":
        raise RuntimeError("the common eager numerical backend was not activated")
    enable_checkpointing(model, True)
    factual_targets = tokenize(model.tokenizer, factual["texts"])
    changed_targets = tokenize(model.tokenizer, changed["texts"])
    longest_factual = max(
        range(len(factual_targets)), key=lambda i: len(factual_targets[i])
    )
    longest_changed = max(
        range(len(changed_targets)), key=lambda i: len(changed_targets[i])
    )
    inputs_z = torch.stack(
        (factual["z"][longest_factual], changed["z"][longest_changed])
    )
    inputs_t = [factual_targets[longest_factual], changed_targets[longest_changed]]
    native = check_native_forward(model, inputs_z[:1], inputs_t[0])
    settings = config["training"]
    headroom = config["preflight"]["memory_headroom"]
    measurements, microbatch = [], None
    for size in config["preflight"]["microbatches"]:
        optimizer = optimizer_for(model, settings)
        torch.cuda.reset_peak_memory_stats(device)
        try:
            update(
                model,
                optimizer,
                inputs_z,
                inputs_t,
                list(range(size)),
                microbatch=size,
                clip=settings["clip"],
            )
            torch.cuda.synchronize(device)
            free, total = torch.cuda.mem_get_info(device)
            peak = torch.cuda.max_memory_reserved(device)
            safe = peak <= total * (1 - headroom) and free >= total * headroom
            measurements.append(
                {
                    "microbatch": size,
                    "peak_reserved_bytes": peak,
                    "total_bytes": total,
                    "free_bytes": free,
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
    del model
    gc.collect()
    clear_gpu(device)
    if microbatch is None:
        raise RuntimeError(
            "complete-CV training lacks the required 15% GPU-memory headroom"
        )
    model = load_reference(config, device=device)
    enable_checkpointing(model, True)
    frozen = frozen_digest(model)
    optimizer = optimizer_for(model, settings)
    scheduler = scheduler_for(
        optimizer, settings["updates"], settings["warmup_fraction"]
    )
    smoke_z = torch.stack((factual["z"][0], changed["z"][0]))
    smoke_targets = [factual_targets[0], changed_targets[0]]
    initial = evaluate_nll(model, smoke_z, smoke_targets)["nll"]
    start = perf_counter()
    for step in range(config["preflight"]["smoke_updates"]):
        update(
            model,
            optimizer,
            smoke_z,
            smoke_targets,
            [0, 1],
            microbatch=microbatch,
            clip=settings["clip"],
        )
        scheduler.step()
        status(prepared / "diagnostics", "smoke", step=step + 1)
    smoke_seconds = perf_counter() - start
    final = evaluate_nll(model, smoke_z, smoke_targets)["nll"]
    if final >= initial:
        raise RuntimeError("20-update warm-start smoke did not reduce paired loss")
    gradients = check_gradients_and_frozen(model, frozen)
    resume_path = prepared / "diagnostics" / "resume.pt"
    save_resume(
        resume_path, model, optimizer, scheduler, {"step": 20}, identity="cf-diagnostic"
    )
    update(
        model,
        optimizer,
        smoke_z,
        smoke_targets,
        [0, 1],
        microbatch=microbatch,
        clip=settings["clip"],
    )
    scheduler.step()
    continued = model.trainable_state()
    progress = restore_resume(
        resume_path, model, optimizer, scheduler, identity="cf-diagnostic"
    )
    if progress != {"step": 20}:
        raise RuntimeError("diagnostic resume metadata changed")
    update(
        model,
        optimizer,
        smoke_z,
        smoke_targets,
        [0, 1],
        microbatch=microbatch,
        clip=settings["clip"],
    )
    scheduler.step()
    for name, expected in continued.items():
        torch.testing.assert_close(
            model.trainable_state()[name], expected, atol=1e-6, rtol=1e-6
        )
    # Match a real optimizer update: 32 target sequences accumulated through
    # the selected microbatch, rather than timing only the two-example smoke.
    benchmark_z = torch.cat((factual["z"][:16], changed["z"][:16]))
    benchmark_targets = factual_targets[:16] + changed_targets[:16]
    bench_start = perf_counter()
    for step in range(config["preflight"]["benchmark_updates"]):
        update(
            model,
            optimizer,
            benchmark_z,
            benchmark_targets,
            list(range(32)),
            microbatch=microbatch,
            clip=settings["clip"],
        )
        scheduler.step()
        status(prepared / "diagnostics", "benchmark", step=step + 1)
    benchmark_seconds = perf_counter() - bench_start
    checkpoint = prepared / "diagnostics" / "diagnostic-only.pt"
    save_checkpoint(
        checkpoint,
        model,
        encoder=read_json(prepared / "manifest.json")["encoder"],
        provenance={"stage": "diagnostic-only", "promotable": False},
    )
    del model, optimizer, scheduler
    gc.collect()
    clear_gpu(device)
    verification = verify_checkpoint(
        checkpoint, device=device, cache_dir=str(resolve(config["paths"]["cache"]))
    )
    if not verification["verified"]:
        raise RuntimeError("diagnostic checkpoint failed fresh reload")
    return {
        "gpu_name": torch.cuda.get_device_name(device),
        "microbatch": microbatch,
        "memory": measurements,
        "native": native,
        "gradients": gradients,
        "smoke_nll_before": initial,
        "smoke_nll_after": final,
        "smoke_seconds": smoke_seconds,
        "benchmark_updates": config["preflight"]["benchmark_updates"],
        "benchmark_seconds": benchmark_seconds,
        "seconds_per_diagnostic_update": benchmark_seconds
        / config["preflight"]["benchmark_updates"],
        "diagnostic_checkpoint_sha256": sha256_file(checkpoint),
        "reload_verified": True,
    }
