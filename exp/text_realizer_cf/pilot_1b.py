"""Detached 1B pilot: preflight, two independent continuations, then tune comparison."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from src.artifact_io import write_json

from .data import prepare
from .protocol import identity, lock, read_json, require_current, snapshot, status
from .settings import GEMMA1B_CONFIG, GEMMA1B_MODEL, ROOT, read_config, run_paths


def runtime_estimate(config: dict, ready: dict, manifest: dict) -> dict:
    """Parallel training, followed by sequential A/B/C factual+oracle evaluation."""
    timing = ready["inference"]
    generation = timing["seconds_per_generated_cv"]
    nll = timing["seconds_per_nll_cv"]
    cycle = timing["seconds_per_cycle_cv"]
    updates = config["training"]["updates"]
    checks = updates // config["training"]["evaluate_every"]
    selected = manifest["counts"]["selection_factual"]
    evaluated = len(manifest["split_ids"]["evaluation"])
    training = updates * ready["seconds_per_diagnostic_update"]
    selection = checks * selected * (generation + nll)
    evaluation = 3 * 2 * evaluated * (generation + nll + cycle)
    now = datetime.now(timezone.utc)
    total = training + selection + evaluation
    return {
        "measured_at": now.isoformat(),
        "parallel_training_seconds": training + selection,
        "final_comparison_seconds": evaluation,
        "total_seconds_before_margin": total,
        "planning_seconds_with_25_percent_margin": total * 1.25,
        "planning_completion_berlin": (now + timedelta(seconds=total * 1.25))
        .astimezone(ZoneInfo("Europe/Berlin"))
        .isoformat(),
        "note": "Extrapolated from 10 full updates and four fit generations; includes six 80-CV selections per arm and 1920 final generations. A planning estimate, not a deadline.",
    }


def _run_children(tasks: list[dict], controller: Path, phase: str) -> None:
    """Own only our subprocesses; each GPU worker takes the existing GPU lease."""
    processes = []
    with ExitStack() as stack:
        try:
            for task in tasks:
                log = controller / "logs" / f"{task['name']}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                stream = stack.enter_context(log.open("a"))
                env = {
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": task["gpu"],
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "TEXT_REALIZER_GPU_TESTS": "0",
                    "TEXT_REALIZER_1B_GPU_TESTS": "0",
                }
                process = subprocess.Popen(
                    task["command"],
                    cwd=ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                processes.append((task, process))
            while True:
                progress = {}
                for task, process in processes:
                    code = process.poll()
                    if code not in (None, 0):
                        raise RuntimeError(
                            f"{task['name']} failed with exit {code}; inspect controller/logs"
                        )
                    path = task["progress"] / "status.json"
                    progress[task["name"]] = {
                        "pid": process.pid,
                        "exit_code": code,
                        "gpu": task["gpu"],
                        "progress": read_json(path) if path.exists() else None,
                    }
                status(controller, phase, workers=progress)
                if all(process.poll() == 0 for _, process in processes):
                    return
                time.sleep(10)
        finally:
            for _, process in processes:
                if process.poll() is None:
                    process.terminate()
            for _, process in processes:
                if process.poll() is None:
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


def run(
    config_path: Path,
    run_id: str,
    gpus: list[str],
    *,
    resume: bool = False,
    train_after_preflight: bool = False,
) -> dict:
    if len(gpus) != 2 or len(set(gpus)) != 2 or any(not gpu.isdigit() for gpu in gpus):
        raise ValueError("select two distinct physical GPU indices")
    config = read_config(config_path)
    if config["model"]["name"] != GEMMA1B_MODEL:
        raise ValueError("this controller is only for the separate 1B pilot")
    _, reports = run_paths(config, run_id)
    controller = reports / "controller"
    with lock(controller / ".controller.lock"):
        if (controller / "plan.json").exists() and not resume:
            raise ValueError("pilot already exists; use --resume or a new run ID")
        prepared = prepare(config)
        expected = identity(config, prepared)
        snapshot(
            controller,
            {
                "identity": expected,
                "config": config,
                "kind": "gemma1b-oracle-exposure-pilot",
                "physical_gpus": {"factual": gpus[0], "oracle": gpus[1]},
                "audit_gate": "closed",
                "official_test_gate": "sealed",
            },
        )
        cli = [sys.executable, "-u", "-m", "exp.text_realizer_cf"]
        common = ["--config", str(config_path.resolve()), "--device", "cuda:0"]
        try:
            marker = prepared / "preflight.json"
            ready = read_json(marker) if marker.exists() else {}
            if not ready.get("ready") or ready.get("identity") != expected:
                _run_children(
                    [
                        {
                            "name": "preflight",
                            "gpu": gpus[0],
                            "command": cli + ["preflight"] + common,
                            "progress": prepared / "diagnostics",
                        }
                    ],
                    controller,
                    "preflight",
                )
                ready = read_json(marker)
            if not ready.get("ready") or ready.get("identity") != expected:
                raise ValueError("1B preflight did not verify this exact identity")
            require_current(controller, config, prepared)
            estimate = runtime_estimate(
                config, ready, read_json(prepared / "manifest.json")
            )
            write_json(reports / "eta.json", estimate)
            if not train_after_preflight:
                status(
                    controller, "preflight_ready", training_started=False, **estimate
                )
                return {"preflight_ready": True, "training_started": False, **estimate}
            tasks = []
            for arm, gpu in zip(("factual", "oracle"), gpus, strict=True):
                models, arm_reports = run_paths(config, run_id, arm)
                command = cli + ["train"] + common + ["--run-id", run_id, "--arm", arm]
                if (arm_reports / "plan.json").exists():
                    if (
                        not (models / "last.pt").exists()
                        and not (arm_reports / "training.json").exists()
                    ):
                        raise ValueError(
                            f"{arm} has a plan but no resume state; use a new run ID"
                        )
                    command.append("--resume")
                tasks.append(
                    {
                        "name": arm,
                        "gpu": gpu,
                        "command": command,
                        "progress": arm_reports,
                    }
                )
            status(reports, "training", gpus=gpus, **estimate)
            _run_children(tasks, controller, "training")
            require_current(controller, config, prepared)
            _run_children(
                [
                    {
                        "name": "evaluation",
                        "gpu": gpus[0],
                        "command": cli + ["evaluate"] + common + ["--run-id", run_id],
                        "progress": reports,
                    }
                ],
                controller,
                "evaluating",
            )
            result = read_json(reports / "result.json")
            if not result.get("complete") or result["identity"] != expected:
                raise ValueError("evaluation did not finish for this exact identity")
            status(
                controller,
                "complete",
                worth_replication=result["decision"]["worth_replication"],
            )
            return result["decision"]
        except BaseException as exc:
            status(controller, "failed", error=f"{type(exc).__name__}: {exc}")
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=GEMMA1B_CONFIG)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpus", nargs=2, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--train-after-preflight", action="store_true")
    args = parser.parse_args()
    result = run(
        args.config,
        args.run_id,
        args.gpus,
        resume=args.resume,
        train_after_preflight=args.train_after_preflight,
    )
    print(result)


if __name__ == "__main__":
    main()
