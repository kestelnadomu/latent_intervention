"""One explicit 1B run; stops at tune comparison, never audit or replication."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

from src.artifact_io import sha256_file, write_json

from .pilot_protocol import baseline_material, compare_tune, prepare_development
from .protocol import identity, lock, read_json, require_current, snapshot, status
from .settings import ROOT, read_config, resolve, run_paths


def _child(command: list[str], root: Path, phase: str, progress_root: Path) -> None:
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        # GPU checks are explicitly performed in preflight, not pytest children
        # competing for the already leased GPU.
        "TEXT_REALIZER_GPU_TESTS": "0",
    }
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / f"{phase}.log").open("a") as stream:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
        )
        try:
            while True:
                progress = (
                    read_json(progress_root / "status.json")
                    if (progress_root / "status.json").exists()
                    else {}
                )
                status(
                    root,
                    phase,
                    child_pid=process.pid,
                    progress=progress,
                    audit_gate="closed",
                    oracle_gate="sealed",
                )
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if code:
                raise RuntimeError(
                    f"{phase} failed (exit {code}); inspect {logs / (phase + '.log')}"
                )
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def runtime_estimate(config: dict, ready: dict, manifest: dict) -> dict:
    count = len(manifest["split_ids"]["fit"])
    tune = len(manifest["split_ids"]["tune"])
    updates = math.ceil(count / config["training"]["effective_batch"])
    validation = ready["seconds_per_validation_cv"] * tune
    epoch = updates * ready["seconds_per_optimizer_update"] + validation
    # Correct, shuffled and nearest-fit conditioning each generate all tune IDs;
    # mean-fit and zero require one generation each. All five require tune NLL.
    evaluation = (3 * tune + 2) * ready["seconds_per_generated_cv"] + 5 * validation
    return {
        "seconds_per_epoch_estimate": epoch,
        "tune_evaluation_seconds_estimate": evaluation,
        "maximum_epochs": config["training"]["epochs"],
        "planning_seconds_with_20_percent_margin": 1.2
        * (epoch * config["training"]["epochs"] + evaluation),
        "note": "Measured preflight extrapolation, not a deadline; excludes preflight, includes a heuristic overhead margin. Early stopping may shorten training.",
    }


def pilot(
    config: dict,
    config_path: Path,
    run_id: str,
    *,
    device: str,
    resume: bool = False,
    train_after_preflight: bool = False,
) -> dict:
    if "pilot" not in config or read_config(config_path) != config:
        raise ValueError("pilot requires the exact on-disk pilot configuration")
    if not device.startswith("cuda:"):
        raise ValueError("pilot requires an explicit CUDA device")
    _, reports = run_paths(config, run_id)
    with lock(reports / ".pilot.lock"):
        if (reports / "plan.json").exists() and not resume:
            raise ValueError("pilot run exists; use --resume or a new run ID")
        prepared = prepare_development(config)
        manifest, baseline = baseline_material(config)
        trial = f"{run_id}.lora-compact.seed-42"
        models, trial_reports = run_paths(config, trial)
        plan = {
            "identity": identity(config, prepared),
            "kind": "one-seed-1b-pilot",
            "trial_id": trial,
            "variant": "lora-compact",
            "seeds": [42],
            "baseline_sha256": config["pilot"]["baseline_sha256"],
            "audit_informed": True,
            "audit_gate": "closed",
            "oracle_gate": "sealed",
            "replication_authorized": False,
        }
        snapshot(reports, plan)
        cli = [sys.executable, "-u", "-m", "exp.text_realizer"]
        common = ["--config", str(config_path), "--device", device]
        try:
            ready_path = prepared / "preflight.json"
            ready = read_json(ready_path) if ready_path.exists() else {}
            if not ready.get("ready") or ready.get("identity") != plan["identity"]:
                _child(
                    cli + ["preflight"] + common,
                    reports,
                    "preflight",
                    prepared / "diagnostics",
                )
                ready = read_json(ready_path)
            if not ready["ready"] or ready["identity"] != plan["identity"]:
                raise ValueError("preflight did not verify this pilot identity")
            estimate = runtime_estimate(config, ready, manifest)
            write_json(reports / "eta.json", estimate)
            if not train_after_preflight:
                status(reports, "preflight_ready", training_started=False, **estimate)
                return {"preflight_ready": True, "training_started": False, **estimate}
            command = (
                cli
                + ["train"]
                + common
                + [
                    "--run-id",
                    trial,
                    "--variant",
                    "lora-compact",
                    "--seed",
                    "42",
                ]
            )
            if (trial_reports / "plan.json").exists():
                command.append("--resume")
            _child(command, reports, "training_and_tune", trial_reports)
            # Verify the actual selected checkpoint in a fresh process, without
            # manufacturing a campaign freeze that could unlock audit/oracle.
            verify_path = reports / "verify.json"
            _child(
                cli
                + [
                    "verify",
                    "--checkpoint",
                    str(models / "best.pt"),
                    "--device",
                    device,
                    "--cache-dir",
                    str(resolve(config["paths"]["cache"])),
                    "--output",
                    str(verify_path),
                ],
                reports,
                "verifying",
                trial_reports,
            )
            verification = read_json(verify_path)
            checkpoint_hash = sha256_file(models / "best.pt")
            result = read_json(trial_reports / "factual_tune.json")
            if (
                not verification["verified"]
                or verification["checkpoint_sha256"] != checkpoint_hash
                or result["checkpoint_sha256"] != checkpoint_hash
                or not result["complete"]
            ):
                raise ValueError("pilot checkpoint/tune/reload mismatch")
            require_current(reports, config, prepared)
            _, baseline = baseline_material(config)
            comparison = compare_tune(result, baseline, config)
            comparison.update(
                checkpoint=str(models / "best.pt"),
                checkpoint_sha256=checkpoint_hash,
                reload_verified=True,
                verification_sha256=sha256_file(verify_path),
                tune_report_sha256=sha256_file(trial_reports / "factual_tune.json"),
            )
            write_json(reports / "pilot_result.json", comparison)
            from .data import load_bundle

            part = load_bundle(prepared)["tune"]
            write_json(
                reports / "tune_examples.json",
                [
                    {
                        "id": key,
                        "reference": reference,
                        "baseline_270m": old["text"],
                        "pilot_1b": new["text"],
                    }
                    for key, reference, old, new in list(
                        zip(
                            part["ids"],
                            part["texts"],
                            baseline["conditions"]["correct"]["rows"],
                            result["conditions"]["correct"]["rows"],
                            strict=True,
                        )
                    )[: config["evaluation"]["examples"]]
                ],
            )
            status(
                reports,
                "complete",
                promising=comparison["promising"],
                audit_gate="closed",
                oracle_gate="sealed",
            )
            return comparison
        except BaseException as exc:
            status(
                reports,
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                error=f"{type(exc).__name__}: {exc}",
                audit_gate="closed",
                oracle_gate="sealed",
            )
            raise
