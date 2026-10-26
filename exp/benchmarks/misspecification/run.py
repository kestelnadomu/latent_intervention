"""Standalone queue; deliberately no routing changes to the main pipeline."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import json
import multiprocessing
import os
import traceback

from . import METHODS
from .common import (
    CONFIG,
    digest,
    exclusive,
    job_key,
    paths,
    read_config,
    read_record,
    root,
    write_record,
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "command",
        choices=("preflight", "train", "train-reader", "evaluate", "report", "status"),
    )
    result.add_argument("--config", default=CONFIG)
    result.add_argument("--run-id")
    result.add_argument("--pilot", action="store_true")
    result.add_argument("--all", action="store_true")
    result.add_argument("--method", choices=METHODS, action="append")
    result.add_argument(
        "--state-space", choices=[f"S{i}" for i in range(1, 7)], action="append"
    )
    result.add_argument(
        "--operator",
        action="append",
        help="exact scenario label, e.g. O3/drop_X_D (aliases resolve automatically)",
    )
    result.add_argument("--seed", type=int, choices=range(42, 47), action="append")
    return result


def select_jobs(signature, args):
    rows = signature["scenarios"]
    if args.operator and set(args.operator) - {r["name"] for r in rows}:
        raise ValueError("unknown operator/scenario selector")
    if args.seed and set(args.seed) - set(signature["config"]["seeds"]):
        raise ValueError("seed is outside this protocol")
    cases = [
        r
        for r in rows
        if (not args.operator or r["name"] in args.operator)
        and (not args.state_space or r["space"] in args.state_space)
    ]
    operators, spaces = {r["canonical"] for r in cases}, {r["space"] for r in cases}
    selected = [
        j
        for j in signature["jobs"]
        if j["space"] in spaces
        and (j["operator"] == "factual" or j["operator"] in operators)
        and (not args.method or j["method"] in args.method)
        and (not args.seed or j["seed"] in args.seed or j["method"] == "ot_gaussian")
    ]
    if not selected:
        raise ValueError("selectors matched no fitting jobs")
    return selected


def worker(cfg, signature, job):
    import torch
    from .workflow import fit_job
    from .reader import train

    torch.set_num_threads(1)
    log = (
        root(cfg)
        / "logs"
        / ("evaluation_reader.log" if job is None else job_key(job) + ".log")
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    with (
        log.open("a", buffering=1) as stream,
        redirect_stdout(stream),
        redirect_stderr(stream),
    ):
        try:
            return (
                train(cfg, signature) if job is None else fit_job(cfg, signature, job)
            )
        except Exception:
            traceback.print_exc()
            raise


def queue(cfg, signature, selected, *, include_reader):
    from .workflow import freeze

    if not (root(cfg) / "preflight.json").exists():
        raise ValueError("run preflight before fitting")
    checked = read_record(root(cfg) / "preflight.json")
    if (
        checked["signature_sha256"] != digest(signature)
        or checked["status"] != "passed"
    ):
        raise ValueError("preflight does not match this queue")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    with exclusive(root(cfg) / "queue.lock"):
        states = {job_key(j): "queued" for j in selected}
        if include_reader:
            states["evaluation_reader"] = "queued"

        def publish():
            write_record(
                root(cfg) / "status.json",
                dict(protocol_sha256=digest(signature), jobs=states),
            )

        publish()
        failures = []
        with ProcessPoolExecutor(
            max_workers=cfg["workers"], mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            futures = {}
            # Reader occupies at most one of the same four slots; no hidden oversubscription.
            for job in ([None] if include_reader else []) + selected:
                key = "evaluation_reader" if job is None else job_key(job)
                futures[executor.submit(worker, cfg, signature, job)] = key
            for future in as_completed(futures):
                key = futures[future]
                try:
                    future.result()
                    states[key] = "complete"
                except Exception as error:
                    states[key] = f"failed: {type(error).__name__}: {error}"
                    failures.append(key)
                publish()
                print(f"{key}: {states[key]}", flush=True)
        if failures:
            raise RuntimeError(
                f"{len(failures)} jobs failed; successful jobs are preserved; see status.json/logs"
            )
        frozen = freeze(cfg, signature)
        print(
            "All selections frozen; test evaluation NOT launched."
            if frozen
            else "Selected jobs complete; full selections not yet frozen."
        )


def status(cfg, signature):
    totals, entries = {}, {}
    for job in signature["jobs"]:
        _, report, resume = paths(cfg, job)
        state = "pending"
        if report.exists():
            record = read_record(report)
            state = record["status"]
            if state == "complete":
                from src.artifact_io import sha256_file

                if sha256_file(record["checkpoint"]) != record["checkpoint_sha256"]:
                    state = "invalid checkpoint"
        elif resume.exists():
            state = "resumable"
        entries[job_key(job)] = state
        totals[state] = totals.get(state, 0) + 1
    return dict(
        run_id=cfg["run_id"],
        fits=totals,
        jobs=entries,
        evaluation_reader_complete=(
            root(cfg) / "evaluation_reader" / "selection.json"
        ).exists(),
        selections_frozen=(root(cfg) / "selection.json").exists(),
    )


def main(argv=None):
    args = parser().parse_args(argv)
    cfg = read_config(args.config, run_id=args.run_id, pilot=args.pilot)
    from .workflow import preflight, read_protocol

    if args.command == "preflight":
        result = preflight(cfg, args.config)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "status",
                        "neural_fits",
                        "analytical_fits",
                        "optimizer_steps",
                    )
                },
                indent=2,
            )
        )
        return
    signature = read_protocol(cfg)
    selected = select_jobs(signature, args)
    if args.command == "train":
        if not args.all and not any(
            (args.method, args.state_space, args.operator, args.seed, args.pilot)
        ):
            raise ValueError("use --all or explicit fitting selectors")
        if args.all and any((args.method, args.state_space, args.operator, args.seed)):
            raise ValueError("--all cannot be combined with narrowing selectors")
        queue(cfg, signature, selected, include_reader=args.all)
    elif args.command == "train-reader":
        queue(cfg, signature, [], include_reader=True)
    elif args.command == "evaluate":
        if cfg["is_pilot"]:
            raise ValueError(
                "the pilot is development-only; official-test evaluation is disabled"
            )
        from .evaluation import evaluate

        evaluate(
            cfg,
            signature,
            methods=args.method or METHODS,
            spaces=args.state_space,
            operators=args.operator,
            seeds=args.seed or cfg["seeds"],
        )
    elif args.command == "report":
        from .reporting import report

        report(cfg, signature)
        print(root(cfg) / "summary.md")
    else:
        print(json.dumps(status(cfg, signature), indent=2))


if __name__ == "__main__":
    main()
