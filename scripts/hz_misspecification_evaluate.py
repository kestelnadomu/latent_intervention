#!/usr/bin/env python3
"""Parallel scheduling of the frozen official-test evaluator; no fitting."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from pathlib import Path
import socket
import sys
from time import perf_counter
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

from exp.benchmarks.misspecification import METHODS  # noqa: E402
from exp.benchmarks.misspecification.common import (  # noqa: E402
    CONFIG,
    digest,
    exclusive,
    now,
    read_config,
    root,
    write_record,
)
from src.artifact_io import sha256_file  # noqa: E402

_CONTEXT = None


def tasks(signature):
    """Exactly the native evaluator's cases, with one early timing per method."""
    canonical = {r["canonical"] for r in signature["scenarios"]}
    cases = {r["name"]: r for r in signature["scenarios"] if r["name"] in canonical}
    if set(cases) != canonical:
        raise ValueError("missing canonical evaluation scenario")
    methods = (
        "ot_flow",
        "semantic_editor",
        "ot_editor",
        "state_flow",
        "ot_gaussian",
        "no_edit",
    )
    if set(methods) != {*METHODS, "no_edit"}:
        raise ValueError("unexpected method registry")
    names = sorted(cases, key=lambda n: (n != "S1/true", n))
    return [
        (name, method, seed, action, target)
        for name in names
        for seed in signature["config"]["seeds"]
        for action in range(4)
        for target in cases[name]["targets"]
        for method in methods
    ]


def task_key(task):
    case, method, seed, action, target = task
    return f"{case}/{method}/seed-{seed}/action-{action}/{target}"


def initialize(cfg, signature, selection, data, cases):
    import torch

    global _CONTEXT
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _CONTEXT = (cfg, signature, selection, data, cases)


def worker(task):
    from exp.benchmarks.misspecification.evaluation import (
        evaluate_condition,
        condition_path,
    )

    cfg, signature, selection, data, cases = _CONTEXT
    name, method, seed, action, target = task
    key = task_key(task)
    progress = root(cfg) / "evaluation" / "workers" / f"{os.getpid()}.json"
    started = now()
    write_record(progress, dict(status="running", task=key, started_at=started))
    print(f"{started} START {key}", flush=True)
    clock = perf_counter()
    cached = condition_path(cfg, name, method, seed, action, target).exists()
    try:
        record = evaluate_condition(
            cfg, signature, selection, data, cases[name], method, seed, action, target
        )
        result = dict(
            task=key,
            method=method,
            status="complete",
            cached=cached,
            elapsed_seconds=perf_counter() - clock,
            evaluation_seconds=record["elapsed_seconds"],
            metrics_sha256=sha256_file(
                condition_path(cfg, name, method, seed, action, target)
            ),
        )
    except Exception:
        result = dict(
            task=key,
            method=method,
            status="failed",
            elapsed_seconds=perf_counter() - clock,
            traceback=traceback.format_exc(),
        )
        print(result["traceback"], flush=True)
    write_record(progress, dict(result, started_at=started, finished_at=now()))
    return result


def run(cfg, signature, workers):
    import torch
    from exp.benchmarks.misspecification.evaluation import frozen, evaluation_data
    from exp.benchmarks.misspecification.reporting import report
    from exp.benchmarks.misspecification.scenarios import build

    if cfg["is_pilot"]:
        raise ValueError("pilot runs cannot evaluate the official test set")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    queue = tasks(signature)
    status_path = root(cfg) / "evaluation" / "queue.json"
    # Same global lock as the native sequential command; workers also use its
    # per-condition locks and checksum/identity validation for resumability.
    with exclusive(root(cfg) / "evaluation" / "run.lock"):
        selection = frozen(cfg, signature)
        data = evaluation_data(cfg, signature, selection)
        cases = {c.name: c for c in build(data["loaded"][0]["sim_config"])}
        state = dict(
            status="running",
            started_at=now(),
            host=socket.gethostname(),
            protocol_sha256=digest(signature),
            selection_sha256=digest(selection),
            runner_sha256=sha256_file(__file__),
            workers=workers,
            threads_per_worker=1,
            total=len(queue),
            completed=0,
            failures=[],
            completed_by_method={},
            timings_by_method={},
            official_test_evaluation=True,
            fitting=False,
        )
        write_record(status_path, state)
        print(
            f"{state['started_at']} EVALUATION {len(queue)} conditions; {workers} workers",
            flush=True,
        )
        counts, timings = Counter(), {}
        clock = perf_counter()
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize,
                initargs=(cfg, signature, selection, data, cases),
            ) as pool:
                futures = {pool.submit(worker, task): task for task in queue}
                for future in as_completed(futures):
                    result = future.result()
                    if result["status"] == "complete":
                        counts[result["method"]] += 1
                        if not result["cached"]:
                            timings.setdefault(result["method"], []).append(
                                result["evaluation_seconds"]
                            )
                    else:
                        state["failures"].append(result)
                    state.update(
                        completed=sum(counts.values()),
                        completed_by_method=dict(counts),
                        timings_by_method=timings,
                        elapsed_seconds=perf_counter() - clock,
                        updated_at=now(),
                    )
                    write_record(status_path, state)
                    print(
                        f"{now()} {result['status'].upper()} {result['task']} "
                        f"{result['elapsed_seconds']:.1f}s; {state['completed']}/{len(queue)} complete; "
                        f"{len(state['failures'])} failed",
                        flush=True,
                    )
            frozen(cfg, signature)  # Recheck models, reports, inputs and frozen source.
            state.update(status="reporting", updated_at=now())
            write_record(status_path, state)
            report(cfg, signature)
            state.update(
                status="failed" if state["failures"] else "complete",
                finished_at=now(),
                elapsed_seconds=perf_counter() - clock,
                report=str(root(cfg) / "summary.md"),
            )
            write_record(status_path, state)
        except BaseException:
            state.update(
                status="interrupted_or_failed",
                updated_at=now(),
                traceback=traceback.format_exc(),
            )
            write_record(status_path, state)
            raise
        if state["failures"]:
            raise RuntimeError(
                f"{len(state['failures'])} conditions failed; completed metrics retained; see {status_path}"
            )
        print(f"{now()} COMPLETE; report: {state['report']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--run-id", default="hz-misspec-v1")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    os.chdir(REPO)
    from exp.benchmarks.misspecification.workflow import read_protocol

    cfg = read_config(args.config, run_id=args.run_id)
    run(cfg, read_protocol(cfg), args.workers)


if __name__ == "__main__":
    main()
