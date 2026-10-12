"""Prepare or explicitly launch the resumable, dependency-aware h_Z CPU queue."""

from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing
import os
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.artifact_io import copy_atomic, sha256_file, write_json
from src.config import CONFIG_PATH
from src.hz_benchmark_matrix import (
    PROTOCOL,
    ROOT,
    assert_unchanged,
    case_name,
    digest,
    discover,
    oracle_readiness,
    run_root,
)
from src.hz_evaluation import evaluate
from src.hz_report import render
from src.hz_training import fit, job_paths, model_sizes, verify_fit

DEFAULT_RUN = "hz-gemma-v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def immutable_json(path, payload):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(
                f"immutable run artifact differs: {path}; use a new run ID"
            )
    else:
        write_json(path, payload)


def freeze_preparation(root, plan, config_path, protocol_path, report=render):
    """Save a reusable, immutable plan and its source/input provenance."""
    immutable_json(root / "plan.json", plan)
    for filename, expected in plan["code_sha256"].items():
        destination = root / "source" / filename
        if destination.exists():
            if sha256_file(destination) != expected:
                raise ValueError(f"source snapshot differs: {filename}")
        else:
            copy_atomic(ROOT / filename, destination)
    # Full resolved configs/settings are already in plan.json; keep source YAML readable too.
    for name, path in (
        ("pipeline.yaml", config_path),
        ("benchmark.yaml", protocol_path),
    ):
        destination = root / "configs" / name
        if destination.exists() and sha256_file(destination) != sha256_file(path):
            raise ValueError(f"saved configuration differs: {name}")
        if not destination.exists():
            copy_atomic(path, destination)
    ready = readiness(plan)
    write_json(root / "readiness.json", ready)
    if not (root / "status.json").exists():
        write_json(
            root / "status.json",
            dict(status="prepared", training_started=False, updated_at=now()),
        )
        report(root, plan, [], "prepared; no training or target encoding started")
    return ready


def prepare(run_id=DEFAULT_RUN, config_path=CONFIG_PATH, protocol_path=PROTOCOL):
    torch.set_num_threads(1)
    root = run_root(run_id)
    plan = discover(config_path, protocol_path)
    plan["model_sizes"] = model_sizes(plan)
    ready = freeze_preparation(root, plan, config_path, protocol_path)
    print(
        f"Prepared {len(plan['cases'])} cases / {fit_count(plan)} fits. Launch prerequisites: {ready['ready']}. {root}",
        flush=True,
    )
    return root, plan, ready


def fit_count(plan):
    return len(plan["cases"]) * (
        len(plan["candidates"]) + len(plan["settings"]["final_seeds"])
    )


def readiness(plan):
    oracle = oracle_readiness(plan)
    return dict(
        ready=all(x["status"] == "verified" for x in oracle),
        oracle_targets=oracle,
        next_step="If targets are missing, explicitly launch scripts/run_oracle_targets.sh first. This benchmark never starts encoding.",
    )


def preflight(root, plan, *, freeze=False):
    assert_unchanged(plan)
    ready = readiness(plan)
    write_json(Path(root) / "readiness.json", ready)
    if not ready["ready"]:
        raise ValueError(
            "oracle training targets are missing; run scripts/run_oracle_targets.sh separately before launching all nine variants"
        )
    if freeze:
        immutable_json(Path(root) / "oracle_inputs.json", ready["oracle_targets"])
    elif (Path(root) / "oracle_inputs.json").exists():
        immutable_json(Path(root) / "oracle_inputs.json", ready["oracle_targets"])
    return ready["oracle_targets"]


def job_id(case, stage, label):
    return f"{case['case']}/{stage}/{label}"


def jobs_for(plan, stage, selection=None):
    """Job graph: search teachers before students; final teachers matched by seed."""
    jobs = []
    ordered = sorted(
        plan["cases"], key=lambda c: (c["variant"] != "state_flow", c["case"])
    )
    for case in ordered:
        if stage == "search":
            settings = [
                (f"trial-{i:03d}", plan["settings"]["search_seed"], candidate)
                for i, candidate in enumerate(plan["candidates"])
            ]
        else:
            settings = [
                (f"seed-{seed}", seed, selection["cases"][case["case"]]["candidate"])
                for seed in plan["settings"]["final_seeds"]
            ]
        for label, seed, candidate in settings:
            dependencies = []
            if case["variant"] == "distilled_flow":
                teacher_case = dict(case=case_name(case["dimension"], "state_flow"))
                labels = (
                    [f"trial-{i:03d}" for i in range(len(plan["candidates"]))]
                    if stage == "search"
                    else [label]
                )
                dependencies = [job_id(teacher_case, stage, item) for item in labels]
            jobs.append(
                dict(
                    id=job_id(case, stage, label),
                    case=case,
                    stage=stage,
                    label=label,
                    seed=seed,
                    candidate=candidate,
                    dependencies=dependencies,
                )
            )
    return jobs


def best(reports):
    return min(
        reports,
        key=lambda r: (r["training"]["best_validation"], digest(r["candidate"])),
    )


def freeze_selection(root, plan, results):
    if len(results) != len(plan["cases"]) * len(plan["candidates"]):
        raise ValueError("every search trial must complete before freezing choices")
    choices = {}
    for case in plan["cases"]:
        trials = [r for r in results.values() if r["case"] == case["case"]]
        if len(trials) != len(plan["candidates"]):
            raise ValueError("incomplete search for a family")
        chosen = best(trials)
        choices[case["case"]] = dict(
            candidate=chosen["candidate"],
            search_validation=chosen["training"]["best_validation"],
            validation_metric=case["validation_metric"],
            search_checkpoint_sha256=chosen["checkpoint_sha256"],
        )
    selection = dict(
        plan_sha256=digest(plan),
        cases=choices,
        test_metrics_used=False,
        final_seeds=plan["settings"]["final_seeds"],
        deployment_seed=plan["settings"]["deployment_seed"],
        cross_family_validation_ranking=False,
    )
    immutable_json(Path(root) / "selection.json", selection)
    return selection


def execute_job(root, plan, job, oracle_inputs, teacher):
    reports, _ = job_paths(root, job["case"], job["stage"], job["label"])
    try:
        return fit(
            root,
            plan,
            job["case"],
            job["candidate"],
            job["seed"],
            job["stage"],
            job["label"],
            oracle_inputs,
            teacher,
        )
    except Exception:
        write_json(
            reports / "failure.json",
            dict(job=job["id"], traceback=traceback.format_exc(), time=now()),
        )
        raise


def run_jobs(root, plan, jobs, oracle_inputs, executor):
    """Bounded concurrency, dependency ordering, verified resumption, visible failures."""
    pending = list(jobs)
    active, completed, errors, blocked = {}, {}, {}, []
    while pending or active:
        for job in list(pending):
            if any(dep in errors or dep in blocked for dep in job["dependencies"]):
                blocked.append(job["id"])
                pending.remove(job)
            elif len(active) < plan["settings"]["workers"] and all(
                dep in completed for dep in job["dependencies"]
            ):
                teacher = (
                    best([completed[dep] for dep in job["dependencies"]])
                    if job["dependencies"]
                    else None
                )
                future = executor.submit(
                    execute_job, root, plan, job, oracle_inputs, teacher
                )
                active[future] = job
                pending.remove(job)
        if not active and pending:
            raise ValueError("unresolvable h_Z job dependency graph")
        write_json(
            Path(root) / "status.json",
            dict(
                status="running",
                phase=jobs[0]["stage"],
                pid=os.getpid(),
                updated_at=now(),
                completed=len(completed),
                total=len(jobs),
                active=[j["id"] for j in active.values()],
                queued=len(pending),
                failed=errors,
                blocked=blocked,
            ),
        )
        if not active:
            break
        done, _ = wait(active, timeout=20, return_when=FIRST_COMPLETED)
        for future in done:
            job = active.pop(future)
            try:
                completed[job["id"]] = future.result()
                print(
                    f"[{job['stage']}] {len(completed)}/{len(jobs)} complete: {job['id']}",
                    flush=True,
                )
            except Exception as exc:
                errors[job["id"]] = f"{type(exc).__name__}: {exc}"
                print(f"FAILED: {job['id']}: {errors[job['id']]}", flush=True)
    if errors or blocked:
        write_json(Path(root) / "errors.json", dict(errors=errors, blocked=blocked))
        raise RuntimeError(
            "some fits failed; dependent jobs were blocked and unrelated jobs finished; inspect errors.json and relaunch to resume"
        )
    return completed


def publish(root, plan, finals):
    """Track one predeclared seed per case, never overwrite canonical prior models."""
    published = []
    for report in finals.values():
        if report["seed"] != plan["settings"]["deployment_seed"]:
            continue
        verify_fit(report)
        destination = (
            Path(report["checkpoint"]).parents[2]
            / "selected"
            / "latent_intervention.pt"
        )
        if (
            destination.exists()
            and sha256_file(destination) != report["checkpoint_sha256"]
        ):
            raise ValueError(
                f"refusing to replace an existing selected checkpoint: {destination}"
            )
        if not destination.exists():
            copy_atomic(report["checkpoint"], destination)
        info = dict(
            case=report["case"],
            seed=report["seed"],
            signature=report["signature"],
            checkpoint_sha256=report["checkpoint_sha256"],
            source_checkpoint=report["checkpoint"],
            plan_sha256=digest(plan),
            loader="src.hz_training.load_model",
        )
        immutable_json(destination.with_suffix(".info.json"), info)
        published.append(dict(**info, checkpoint=str(destination)))
    immutable_json(
        Path(root) / "published.json", sorted(published, key=lambda r: r["case"])
    )


def run(run_id=DEFAULT_RUN, config_path=CONFIG_PATH, protocol_path=PROTOCOL):
    root = run_root(run_id)
    root.parent.mkdir(parents=True, exist_ok=True)
    # A global lock also rejects two different tmux sessions sharing the same machine/run outputs.
    with (root.parent / ".benchmark.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root, plan, _ = prepare(run_id, config_path, protocol_path)
        oracle_inputs = preflight(root, plan, freeze=True)
        if plan["settings"]["workers"] * plan["settings"]["threads_per_worker"] > len(
            os.sched_getaffinity(0)
        ):
            raise ValueError(
                "worker/thread budget exceeds this process's available CPUs"
            )
        try:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=plan["settings"]["workers"], mp_context=context
            ) as executor:
                search = run_jobs(
                    root, plan, jobs_for(plan, "search"), oracle_inputs, executor
                )
                assert_unchanged(plan)
                selection = freeze_selection(root, plan, search)
                finals = run_jobs(
                    root,
                    plan,
                    jobs_for(plan, "final", selection),
                    oracle_inputs,
                    executor,
                )
                preflight(root, plan, freeze=True)
                evaluations = []
                cases = {case["case"]: case for case in plan["cases"]}
                # Keep at most `workers` tasks in flight, including during test scoring.
                remaining = iter(finals.values())
                active = {}
                while True:
                    while len(active) < plan["settings"]["workers"]:
                        report = next(remaining, None)
                        if report is None:
                            break
                        future = executor.submit(
                            evaluate, root, plan, cases[report["case"]], report
                        )
                        active[future] = report["case"]
                    if not active:
                        break
                    write_json(
                        root / "status.json",
                        dict(
                            status="running",
                            phase="evaluation",
                            updated_at=now(),
                            completed=len(evaluations),
                            total=len(finals),
                            active=list(active.values()),
                        ),
                    )
                    done, _ = wait(active, timeout=20, return_when=FIRST_COMPLETED)
                    for future in done:
                        active.pop(future)
                        evaluations.append(future.result())
            preflight(root, plan, freeze=True)
            publish(root, plan, finals)
            write_json(
                root / "results.json",
                dict(
                    plan_sha256=digest(plan),
                    selection=selection,
                    final_fits=[
                        {
                            k: v
                            for k, v in report.items()
                            if k not in {"training", "record_sha256"}
                        }
                        | {
                            "best_epoch": report["training"]["best_epoch"],
                            "best_validation": report["training"]["best_validation"],
                            "training_record_sha256": report["record_sha256"],
                        }
                        for report in sorted(
                            finals.values(), key=lambda r: (r["case"], r["seed"])
                        )
                    ],
                    evaluations=sorted(
                        evaluations, key=lambda r: (r["case"], r["seed"])
                    ),
                ),
            )
            render(root, plan, evaluations, "complete")
            write_json(
                root / "status.json",
                dict(
                    status="complete",
                    updated_at=now(),
                    fits=len(search) + len(finals),
                    evaluations=len(evaluations),
                ),
            )
        except BaseException:
            write_json(
                root / "status.json",
                dict(
                    status="failed",
                    updated_at=now(),
                    traceback=traceback.format_exc(),
                    resume="Relaunch the same run ID after resolving the error; completed verified jobs are reused.",
                ),
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--check-ready", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--config", default=CONFIG_PATH, type=Path)
    parser.add_argument("--protocol", default=PROTOCOL, type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.run:
        run(args.run_id, args.config, args.protocol)
    else:
        root, plan, _ = prepare(args.run_id, args.config, args.protocol)
        if args.check_ready:
            preflight(root, plan)


if __name__ == "__main__":
    main()
