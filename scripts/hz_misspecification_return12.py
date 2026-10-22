#!/usr/bin/env python3
"""Run the authorized 38/12 subdivision of remote50 without changing its protocol."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from pathlib import Path
import socket
import sys
import json

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import hz_misspecification_shards as parent  # noqa: E402

BASE = parent.BASE / "return12-20261022"
ALLOCATION = BASE / "allocation.json"
RUNNER = "scripts/hz_misspecification_return12.py"
LAUNCHER = "scripts/run_hz_misspecification_return12.sh"
ROLES = {"keep38": 38, "return12": 12}


def returned(job):
    return (
        job["space"] == "S1"
        and job["method"] == "ot_editor"
        and (
            job["operator"] in {"O3/drop_X_D", "O3/drop_X_U"}
            or (
                job["operator"] == "O2/noise_sigma_x0.5"
                and job["seed"] in {45, 46}
            )
        )
    )


def artifacts(cfg, job):
    return (*parent.paths(cfg, job), parent.root(cfg) / "logs" / (parent.job_key(job) + ".log"))


def validate_allocation(allocation, assignment, signature):
    if (
        allocation["format"] != "hz_misspecification_return12_v1"
        or allocation["parent_assignment_sha256"] != assignment["record_sha256"]
        or allocation["protocol_sha256"] != parent.digest(signature)
        or set(allocation["assignments"]) != set(ROLES)
    ):
        raise ValueError("allocation belongs to a different parent/protocol")
    expected = {parent.job_key(j): j for j in assignment["assignments"]["remote"]}
    seen = set()
    for role, count in ROLES.items():
        jobs = allocation["assignments"][role]
        if len(jobs) != count:
            raise ValueError(f"{role} must contain exactly {count} fits")
        for job in jobs:
            key = parent.job_key(job)
            if key in seen or expected.get(key) != job or returned(job) != (role == "return12"):
                raise ValueError("duplicate, modified, unknown or wrongly allocated fit")
            seen.add(key)
    if seen != set(expected):
        raise ValueError("allocation must cover the original 50 exactly once")


def load_allocation():
    cfg, signature, assignment = parent.load_assignment()
    allocation = parent.read_record(ALLOCATION)
    validate_allocation(allocation, assignment, signature)
    expected_sources = allocation["orchestration_sha256"]
    if set(expected_sources) != {RUNNER, LAUNCHER}:
        raise ValueError("missing scheduling source hashes")
    for name, expected in expected_sources.items():
        if parent.sha256_file(name) != expected:
            raise ValueError(f"return12 orchestration changed: {name}")
    return cfg, signature, assignment, allocation


def allocate():
    """Run once on cpuserver16, only after the old queue and its workers stop."""
    cfg, signature, assignment = parent.load_assignment()
    with parent.exclusive(parent.root(cfg) / "queue.lock"):
        if ALLOCATION.exists():
            raise FileExistsError(ALLOCATION)
        selected = [j for j in assignment["assignments"]["remote"] if returned(j)]
        for job in selected:
            if any(p.exists() for p in artifacts(cfg, job)):
                raise ValueError(f"refusing to delegate a started fit: {parent.job_key(job)}")
        snapshot = {}
        for job in assignment["assignments"]["remote"]:
            _, report, resume = parent.paths(cfg, job)
            record = parent.read_record(report) if report.exists() else {}
            snapshot[parent.job_key(job)] = dict(
                status="complete" if parent.completed_record(cfg, signature, job)
                else "resumable" if resume.exists() else "unstarted",
                epochs=len(record.get("history", [])),
            )
        allocation = dict(
            format="hz_misspecification_return12_v1",
            created_at=parent.now(),
            created_on=socket.gethostname(),
            parent_assignment_sha256=assignment["record_sha256"],
            protocol_sha256=parent.digest(signature),
            reason="User requested returning 12 untouched fits to the originating agent because CPU fitting on cpuserver16 is too slow.",
            assignments=dict(
                keep38=[j for j in assignment["assignments"]["remote"] if not returned(j)],
                return12=selected,
            ),
            intended_hosts=dict(keep38="cpuserver16", return12="cpuserver17 / original agent"),
            snapshot=snapshot,
            orchestration_sha256={p: parent.sha256_file(p) for p in (RUNNER, LAUNCHER)},
        )
        validate_allocation(allocation, assignment, signature)
        parent.write_record(ALLOCATION, allocation)
    print(f"Allocated 38 here and 12 to the original agent: {ALLOCATION}")


def check(role, numerical=False):
    cfg, signature, _, allocation = load_allocation()
    prepared = parent.read_record(parent.root(cfg) / "preflight.json")
    if prepared["status"] != "passed" or prepared["signature_sha256"] != parent.digest(signature):
        raise ValueError("missing or incompatible original preflight")
    jobs = allocation["assignments"][role]
    complete = sum(parent.completed_record(cfg, signature, j) is not None for j in jobs)
    results = {}
    if numerical:
        import torch
        from exp.benchmarks.misspecification.data import bundle, guide, load_factual
        from exp.benchmarks.misspecification.scenarios import build
        from exp.benchmarks.misspecification.workflow import numerical_check

        torch.set_num_threads(1)
        loaded = bundle(cfg)
        cases = {c.name: c for c in build(loaded[0]["sim_config"])}
        for job in jobs:
            key = f"{job['space']}/{job['method']}/{job['operator']}"
            if key in results or parent.completed_record(cfg, signature, job):
                continue
            results[key] = numerical_check(
                cfg, job["method"], load_factual(cfg, job["space"], loaded),
                guide(cfg, job["space"], 42, loaded[1]), cases[job["operator"]].matrices,
            )
    record = dict(
        checked_at=parent.now(), role=role, status="passed", jobs=len(jobs), complete=complete,
        allocation_sha256=allocation["record_sha256"], protocol_sha256=parent.digest(signature),
        numerical=results, numerical_requested=numerical, optimizer_steps=0,
        official_test_evaluation=False,
    )
    parent.write_record(BASE / role / "check.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "numerical"}, indent=2))


def train(role, workers):
    if workers < 1:
        raise ValueError("workers must be positive")
    cfg, signature, assignment, allocation = load_allocation()
    checked = parent.read_record(BASE / role / "check.json")
    if (
        checked["status"] != "passed" or checked["role"] != role
        or checked["allocation_sha256"] != allocation["record_sha256"]
        or checked["protocol_sha256"] != parent.digest(signature)
        or not checked["numerical_requested"]
    ):
        raise ValueError("run check --numerical for this role before fitting")
    with parent.exclusive(parent.root(cfg) / "queue.lock"), parent.exclusive(BASE / role / "queue.lock"):
        selected = allocation["assignments"][role]
        states, remaining, resumed = {}, [], {}
        for job in selected:
            key = parent.job_key(job)
            complete = parent.completed_record(cfg, signature, job)
            states[key] = "complete" if complete else "queued"
            if not complete:
                remaining.append(job)
                _, report, resume = parent.paths(cfg, job)
                if resume.exists():
                    resumed[key] = len(parent.read_record(report).get("history", [])) if report.exists() else 0
        remaining.sort(key=lambda j: not parent.paths(cfg, j)[2].exists())
        execution = dict(
            started_at=parent.now(), hostname=socket.gethostname(), pid=os.getpid(), role=role,
            allocation_sha256=allocation["record_sha256"], parent_assignment_sha256=assignment["record_sha256"],
            protocol_sha256=parent.digest(signature), execution_workers=workers, threads_per_worker=1,
            cpu_affinity=sorted(os.sched_getaffinity(0)), resumed_epochs=resumed,
            submitted_jobs=[parent.job_key(j) for j in remaining],
        )
        parent.write_record(BASE / role / "executions" / f"{parent.now().replace(':', '-')}-{os.getpid()}.json", execution)

        def publish():
            parent.write_record(BASE / role / "status.json", dict(
                updated_at=parent.now(), role=role, workers=workers, jobs=states,
                allocation_sha256=allocation["record_sha256"], official_test_evaluation=False,
            ))

        publish()
        print(f"{role}: {len(selected)} assigned, {len(remaining)} unfinished, {len(resumed)} resumable; workers={workers}", flush=True)
        failures = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = {executor.submit(parent.worker, cfg, signature, j): parent.job_key(j) for j in remaining}
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
            raise RuntimeError(f"{len(failures)} failed fits; preserve snapshots and resume this role")
        print(f"{role} complete. No official-test evaluation.", flush=True)


def status(role):
    cfg, signature, _, allocation = load_allocation()
    counts = Counter()
    for job in allocation["assignments"][role]:
        counts["complete" if parent.completed_record(cfg, signature, job)
               else "started/resumable" if any(p.exists() for p in artifacts(cfg, job))
               else "unstarted"] += 1
    print(json.dumps(dict(role=role, jobs=dict(counts), allocation_sha256=allocation["record_sha256"]), indent=2))


def result_paths(cfg, allocation, role):
    files = {str(parent.ASSIGNMENT), str(ALLOCATION)}
    for job in allocation["assignments"][role]:
        files.update(map(str, parent.paths(cfg, job)[:2]))
    return files


def pack_results(role, output):
    cfg, signature, assignment, allocation = load_allocation()
    for job in allocation["assignments"][role]:
        if parent.completed_record(cfg, signature, job) is None:
            raise ValueError(f"incomplete {role} fit: {parent.job_key(job)}")
    files = result_paths(cfg, allocation, role)
    files.update(str(p) for p in (BASE / role).rglob("*.json"))
    parent.archive(output, files, f"return12-{role}-results", assignment, signature)


def merge_results(role, directory):
    cfg, signature, assignment, allocation = load_allocation()
    incoming = Path(directory)
    manifest = parent.read_record(incoming / "TRANSFER_MANIFEST.json")
    if (
        manifest["kind"] != f"return12-{role}-results" or manifest["shard"] != "remote"
        or manifest["protocol_sha256"] != parent.digest(signature)
        or manifest["assignment_sha256"] != assignment["record_sha256"]
        or parent.read_record(parent.safe_path(incoming, ALLOCATION)) != allocation
    ):
        raise ValueError("returned bundle belongs to a different allocation/protocol")
    required = result_paths(cfg, allocation, role)
    allowed = required | {
        name for name in manifest["files"]
        if Path(name).is_relative_to(BASE / role) and Path(name).suffix == ".json"
    }
    if not required <= manifest["files"].keys() or set(manifest["files"]) - allowed:
        raise ValueError("returned bundle has missing or unexpected files")
    with parent.exclusive(parent.root(cfg) / "queue.lock"):
        # Validate the complete bundle, collisions and model loads before any copy.
        for name, expected in manifest["files"].items():
            if parent.sha256_file(parent.safe_path(incoming, name)) != expected:
                raise ValueError(f"returned checksum mismatch: {name}")
            destination = parent.safe_path(".", name)
            if destination.exists() and parent.sha256_file(destination) != expected:
                raise FileExistsError(f"refusing to overwrite different bytes: {name}")
        from exp.benchmarks.misspecification.models import load
        for job in allocation["assignments"][role]:
            record = parent.completed_record(cfg, signature, job, incoming)
            if record is None:
                raise ValueError(f"missing completed fit: {parent.job_key(job)}")
            load(parent.safe_path(incoming, parent.paths(cfg, job)[0]), signature=record["signature"])
        for name in manifest["files"]:
            if not Path(name).exists():
                parent.copy_atomic(parent.safe_path(incoming, name), name)
        from exp.benchmarks.misspecification.workflow import freeze
        frozen = freeze(cfg, signature)
        parent.write_record(BASE / f"merge-{role}.json", dict(
            merged_at=parent.now(), role=role, jobs=ROLES[role], selections_frozen=bool(frozen),
            return_manifest_sha256=manifest["record_sha256"], official_test_evaluation=False,
        ))
    print(f"Imported {ROLES[role]} fits ({role}); full selection frozen: {bool(frozen)}. No test evaluation.")


def main():
    os.chdir(REPO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("allocate", "check", "train", "status", "pack-results", "merge-results"))
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--numerical", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--from-directory")
    args = parser.parse_args()
    if args.command == "allocate":
        allocate()
        return
    if not args.role:
        parser.error("--role is required")
    if args.command == "check":
        check(args.role, args.numerical)
    elif args.command == "train":
        train(args.role, args.workers)
    elif args.command == "status":
        status(args.role)
    elif args.command == "pack-results":
        if not args.output:
            parser.error("--output is required")
        pack_results(args.role, args.output)
    else:
        if not args.from_directory:
            parser.error("--from-directory is required")
        merge_results(args.role, args.from_directory)


if __name__ == "__main__":
    main()
