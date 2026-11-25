#!/usr/bin/env python3
"""Operational sharding of an existing frozen run; never change fit signatures."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import io
import json
import multiprocessing
import os
from pathlib import Path
import platform
import socket
import sys
import tarfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

from src.artifact_io import atomic_output, copy_atomic, sha256_file  # noqa: E402
from exp.benchmarks.misspecification.common import (  # noqa: E402
    digest,
    exclusive,
    job_key,
    now,
    paths,
    read_config,
    read_record,
    root,
    write_record,
)
from exp.benchmarks.misspecification.run import worker  # noqa: E402
from exp.benchmarks.misspecification.workflow import read_protocol  # noqa: E402

RUN = "hz-misspec-v1"
SPLIT = "remote50-20261020"
BASE = Path("reports/talent_motivation/hz_misspecification") / RUN / "shards" / SPLIT
ASSIGNMENT = BASE / "assignment.json"
RUNNER = "scripts/hz_misspecification_shards.py"
LAUNCHER = "scripts/run_hz_misspecification_shard.sh"


def safe_path(base, name):
    """Reject traversal and symlink escapes in portable bundle paths."""
    relative = Path(name)
    base = Path(base).resolve()
    if relative.is_absolute() or ".." in relative.parts or str(relative) in ("", "."):
        raise ValueError(f"unsafe bundle path: {name}")
    result = base / relative
    if not result.resolve().is_relative_to(base):
        raise ValueError(f"bundle path escapes its root: {name}")
    return result


def remote_jobs(jobs):
    """Exactly 50 untouched jobs, favoring the expensive OT/S6 fits."""
    s1_operators = {
        "O1/unaware",
        "O2/coef_X_x0.5",
        "O2/noise_sigma_x0.5",
        "O3/drop_X_D",
        "O3/drop_X_U",
    }
    chosen = [
        j
        for j in jobs
        if (
            j["space"] == "S6"
            and j["operator"] in {"O5/c0.5", "O5/c1"}
            and j["method"] in {"semantic_editor", "ot_editor"}
        )
        or (
            j["space"] == "S1"
            and j["method"] == "ot_editor"
            and j["operator"] in s1_operators
        )
        or (
            j["space"] == "S1"
            and j["method"] == "semantic_editor"
            and j["operator"] == "O2/noise_sigma_x0.5"
        )
    ]
    if len(chosen) != 50:
        raise ValueError(f"expected exactly 50 remote jobs, found {len(chosen)}")
    # Start expensive jobs early to reduce the last-worker tail.
    return sorted(
        chosen,
        key=lambda j: (
            j["space"] != "S6",
            j["method"] != "ot_editor",
            j["operator"],
            j["seed"],
        ),
    )


def validate_partition(assignment, signature):
    expected = {job_key(j): j for j in signature["jobs"]}
    seen = {}
    for shard in ("local", "remote"):
        for job in assignment["assignments"][shard]:
            key = job_key(job)
            if key in seen or expected.get(key) != job:
                raise ValueError("duplicate, unknown or modified assigned job")
            seen[key] = shard
    if set(seen) != set(expected) or len(assignment["assignments"]["remote"]) != 50:
        raise ValueError("assignment must cover all jobs once, including 50 remote")
    if assignment["protocol_sha256"] != digest(signature):
        raise ValueError("assignment belongs to a different frozen protocol")


def completed_record(cfg, signature, job, base=Path(".")):
    checkpoint, report, _ = paths(cfg, job)
    report = safe_path(base, report)
    if not report.exists():
        return None
    record = read_record(report)
    if record["status"] != "complete":
        return None
    if (
        record["metadata"]["protocol_sha256"] != digest(signature)
        or record["metadata"]["job"] != job
        or record["checkpoint"] != str(checkpoint)
        or sha256_file(safe_path(base, checkpoint)) != record["checkpoint_sha256"]
    ):
        raise ValueError(f"incompatible or corrupt completed fit: {job_key(job)}")
    return record


def load_assignment():
    cfg = read_config(run_id=RUN)
    signature = read_protocol(cfg)
    assignment = read_record(ASSIGNMENT)
    validate_partition(assignment, signature)
    for path, expected in assignment["orchestration_sha256"].items():
        if sha256_file(path) != expected:
            raise ValueError(f"shard orchestration changed: {path}")
    for package, expected in signature["packages"].items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise ValueError(f"{package}: expected {expected}, found {actual}")
    return cfg, signature, assignment


def split():
    cfg = read_config(run_id=RUN)
    signature = read_protocol(cfg)
    # This deliberately fails while the original all-jobs queue is alive.
    with exclusive(root(cfg) / "queue.lock"):
        if ASSIGNMENT.exists():
            raise FileExistsError("assignment already exists; never reassign live jobs")
        remote = remote_jobs(signature["jobs"])
        for job in remote:
            if any(p.exists() for p in paths(cfg, job)):
                raise ValueError(f"remote job has already started: {job_key(job)}")
        remote_keys = {job_key(j) for j in remote}
        local = [j for j in signature["jobs"] if job_key(j) not in remote_keys]
        completed, active = [], {}
        for job in local:
            if completed_record(cfg, signature, job):
                completed.append(job_key(job))
                continue
            _, report, resume = paths(cfg, job)
            if resume.exists():
                import torch

                saved = torch.load(resume, map_location="cpu", weights_only=True)
                if saved.get("format") != "hz_misspecification_resume_v1":
                    raise ValueError("unexpected resume format")
                if (
                    report.exists()
                    and read_record(report)["signature"] != saved["signature"]
                ):
                    raise ValueError("resume/report signatures differ")
                active[job_key(job)] = len(saved["history"])
        assignment = dict(
            format="hz_misspecification_shards_v1",
            run_id=RUN,
            split_id=SPLIT,
            created_at=now(),
            protocol_sha256=digest(signature),
            assignments=dict(local=local, remote=remote),
            completed_at_split=completed,
            resumed_epochs=active,
            orchestration_sha256={p: sha256_file(p) for p in (RUNNER, LAUNCHER)},
            policy="disjoint jobs; unchanged scientific protocol; no test evaluation",
        )
        validate_partition(assignment, signature)
        old_status = root(cfg) / "status.json"
        if old_status.exists():
            copy_atomic(old_status, BASE / "status-before-split.json")
        write_record(ASSIGNMENT, assignment)
    print(
        json.dumps(
            dict(
                remote=len(remote),
                local_total=len(local),
                completed=len(completed),
                local_remaining=len(local) - len(completed),
                resumed_epochs=active,
                assignment=str(ASSIGNMENT),
            ),
            indent=2,
        )
    )


def check(shard, numerical=False):
    cfg, signature, assignment = load_assignment()
    prepared = read_record(root(cfg) / "preflight.json")
    if prepared["status"] != "passed" or prepared["signature_sha256"] != digest(
        signature
    ):
        raise ValueError("missing or incompatible original preflight")
    jobs = assignment["assignments"][shard]
    complete = sum(completed_record(cfg, signature, j) is not None for j in jobs)
    numerical_results = {}
    if numerical:
        import torch
        from exp.benchmarks.misspecification.data import bundle, guide, load_factual
        from exp.benchmarks.misspecification.scenarios import build
        from exp.benchmarks.misspecification.workflow import numerical_check

        torch.set_num_threads(1)
        loaded = bundle(cfg)
        cases = {c.name: c for c in build(loaded[0]["sim_config"])}
        # Check each assigned schema/method/operator without any optimizer step.
        for job in jobs:
            if completed_record(cfg, signature, job):
                continue
            key = (job["space"], job["method"], job["operator"])
            if key in numerical_results:
                continue
            data = load_factual(cfg, job["space"], loaded)
            decoder = guide(cfg, job["space"], 42, loaded[1])
            case = (
                next(
                    c
                    for name, c in cases.items()
                    if name.startswith(job["space"] + "/")
                )
                if job["operator"] == "factual"
                else cases[job["operator"]]
            )
            numerical_results[key] = numerical_check(
                cfg,
                job["method"],
                data,
                decoder,
                case.matrices,
            )
    result = dict(
        checked_at=now(),
        shard=shard,
        assignment_sha256=assignment["record_sha256"],
        protocol_sha256=digest(signature),
        jobs=len(jobs),
        complete=complete,
        remaining=len(jobs) - complete,
        status="passed",
        optimizer_steps=0,
        numerical={"/".join(k): v for k, v in numerical_results.items()},
        official_test_evaluation=False,
    )
    write_record(BASE / shard / "check.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "numerical"}, indent=2))


def train(shard, workers):
    if workers < 1:
        raise ValueError("workers must be positive")
    cfg, signature, assignment = load_assignment()
    selected = assignment["assignments"][shard]
    checked = read_record(BASE / shard / "check.json")
    if (
        checked["status"] != "passed"
        or checked["assignment_sha256"] != assignment["record_sha256"]
    ):
        raise ValueError("run the shard check before fitting")
    with exclusive(root(cfg) / "queue.lock"), exclusive(BASE / shard / "queue.lock"):
        states, remaining = {}, []
        for job in selected:
            complete = completed_record(cfg, signature, job)
            states[job_key(job)] = "complete" if complete else "queued"
            if not complete:
                remaining.append(job)
        # Resume interrupted fits first, without changing the assigned membership.
        remaining.sort(key=lambda j: not paths(cfg, j)[2].exists())
        execution = dict(
            started_at=now(),
            hostname=socket.gethostname(),
            python=sys.version,
            platform=platform.platform(),
            pid=os.getpid(),
            shard=shard,
            protocol_sha256=digest(signature),
            assignment_sha256=assignment["record_sha256"],
            configured_workers=cfg["workers"],
            execution_workers=workers,
            threads_per_worker=1,
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            orchestration_sha256=assignment["orchestration_sha256"],
            resumed_epochs=assignment["resumed_epochs"] if shard == "local" else {},
            note="Worker count is scheduling-only; fit config/signatures stay unchanged.",
        )
        attempt = f"{now().replace(':', '-')}-{os.getpid()}.json"
        write_record(BASE / shard / "executions" / attempt, execution)

        def publish():
            write_record(
                BASE / shard / "status.json",
                dict(
                    updated_at=now(),
                    protocol_sha256=digest(signature),
                    assignment_sha256=assignment["record_sha256"],
                    workers=workers,
                    jobs=states,
                ),
            )
            if shard == "local":
                global_states = dict(states)
                for job in assignment["assignments"]["remote"]:
                    global_states[job_key(job)] = (
                        "complete"
                        if completed_record(cfg, signature, job)
                        else "delegated: remote"
                    )
                write_record(
                    root(cfg) / "status.json",
                    dict(
                        protocol_sha256=digest(signature),
                        jobs=global_states,
                        note="Remote progress is reported on the remote host until imported.",
                    ),
                )

        publish()
        failures = []
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = {
                executor.submit(worker, cfg, signature, j): job_key(j)
                for j in remaining
            }
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
                f"{len(failures)} failed fits; rerun this shard to resume"
            )
        print(
            f"{shard} shard complete; no official-test evaluation was launched.",
            flush=True,
        )


def archive(output, files, kind, assignment, signature):
    """Explicit file allowlist; no credentials, environment, logs or optimizer blobs."""
    output = Path(output)
    if output.exists() or Path(str(output) + ".sha256").exists():
        raise FileExistsError(output)
    hashes = {str(p): sha256_file(safe_path(".", p)) for p in sorted(set(files))}
    manifest = dict(
        format="hz_misspecification_transfer_v1",
        kind=kind,
        shard="remote",
        protocol_sha256=digest(signature),
        assignment_sha256=assignment["record_sha256"],
        files=hashes,
    )
    manifest["record_sha256"] = digest(manifest)
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    with atomic_output(output) as temporary:
        with tarfile.open(temporary, "w:gz") as bundle:
            for name in hashes:
                bundle.add(name, arcname=name, recursive=False)
            entry = tarfile.TarInfo("TRANSFER_MANIFEST.json")
            entry.size = len(payload)
            bundle.addfile(entry, io.BytesIO(payload))
        with tarfile.open(temporary, "r:gz") as bundle:
            if set(bundle.getnames()) != set(hashes) | {"TRANSFER_MANIFEST.json"}:
                raise ValueError("archive contents differ from the allowlist")
            for name, expected in hashes.items():
                with bundle.extractfile(name) as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                        raise ValueError(f"archive verification failed: {name}")
    checksum = sha256_file(output)
    with atomic_output(str(output) + ".sha256") as temporary:
        temporary.write_text(f"{checksum}  {output.name}\n")
    print(
        json.dumps(
            dict(
                archive=str(output),
                files=len(hashes),
                bytes=output.stat().st_size,
                sha256=checksum,
                verified=True,
            ),
            indent=2,
        )
    )


def pack_inputs(output):
    cfg, signature, assignment = load_assignment()
    files = set(signature["inputs"]) | set(signature["source"])
    # Include import dependencies that the historical signature does not enumerate.
    for directory in ("src", "exp"):
        files.update(str(p) for p in Path(directory).rglob("*.py"))
    files.update(
        [
            "README.md",
            RUNNER,
            LAUNCHER,
            str(ASSIGNMENT),
            "docs/experiments/hz_misspecification.md",
            "docs/experiments/hz_misspecification_remote50_handoff.md",
            "tests/test_misspecification_shards.py",
            str(root(cfg) / "protocol.json"),
            str(root(cfg) / "preflight.json"),
        ]
    )
    archive(output, files, "inputs", assignment, signature)


def pack_results(output):
    cfg, signature, assignment = load_assignment()
    files = {str(ASSIGNMENT)}
    for job in assignment["assignments"]["remote"]:
        if completed_record(cfg, signature, job) is None:
            raise ValueError(f"remote shard is not complete: {job_key(job)}")
        checkpoint, report, _ = paths(cfg, job)
        files.update(map(str, (checkpoint, report)))
    files.update(str(p) for p in (BASE / "remote").rglob("*.json"))
    archive(output, files, "results", assignment, signature)


def merge_results(directory):
    cfg, signature, assignment = load_assignment()
    incoming = Path(directory)
    manifest = read_record(incoming / "TRANSFER_MANIFEST.json")
    if (
        manifest["kind"] != "results"
        or manifest["shard"] != "remote"
        or manifest["protocol_sha256"] != digest(signature)
        or manifest["assignment_sha256"] != assignment["record_sha256"]
    ):
        raise ValueError("returned bundle belongs to a different shard/protocol")
    allowed, required = {str(ASSIGNMENT)}, {str(ASSIGNMENT)}
    for job in assignment["assignments"]["remote"]:
        if completed_record(cfg, signature, job, incoming) is None:
            raise ValueError(f"missing complete returned fit: {job_key(job)}")
        checkpoint, report, _ = paths(cfg, job)
        required.update(map(str, (checkpoint, report)))
    allowed |= required
    for name in manifest["files"]:
        relative = Path(name)
        if relative.is_relative_to(BASE / "remote") and relative.suffix == ".json":
            allowed.add(name)
    if not required <= manifest["files"].keys() or set(manifest["files"]) - allowed:
        raise ValueError("returned bundle contains missing/unexpected files")
    with exclusive(root(cfg) / "queue.lock"):
        # Validate every source and every collision before copying anything.
        for name, expected in manifest["files"].items():
            if sha256_file(safe_path(incoming, name)) != expected:
                raise ValueError(f"returned checksum mismatch: {name}")
            destination = safe_path(".", name)
            if destination.exists() and sha256_file(destination) != expected:
                raise FileExistsError(f"refusing to overwrite different bytes: {name}")
        from exp.benchmarks.misspecification.models import load

        for job in assignment["assignments"]["remote"]:
            record = completed_record(cfg, signature, job, incoming)
            load(safe_path(incoming, paths(cfg, job)[0]), signature=record["signature"])
        for name in manifest["files"]:
            if not Path(name).exists():
                copy_atomic(safe_path(incoming, name), name)
        from exp.benchmarks.misspecification.workflow import freeze

        frozen = freeze(cfg, signature)
        states = {
            job_key(j): "complete" if completed_record(cfg, signature, j) else "pending"
            for j in signature["jobs"]
        }
        write_record(
            root(cfg) / "status.json",
            dict(
                protocol_sha256=digest(signature),
                jobs=states,
            ),
        )
        write_record(
            BASE / "merge.json",
            dict(
                merged_at=now(),
                remote_jobs=50,
                selections_frozen=bool(frozen),
                return_manifest_sha256=manifest["record_sha256"],
                official_test_evaluation=False,
            ),
        )
    print(f"Imported 50 fits; selections frozen: {bool(frozen)}. No test evaluation.")


def main():
    os.chdir(REPO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "split",
            "check",
            "train",
            "status",
            "pack-inputs",
            "pack-results",
            "merge-results",
        ),
    )
    parser.add_argument("--shard", choices=("local", "remote"), default="remote")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--numerical", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--from-directory")
    args = parser.parse_args()
    if args.command == "split":
        split()
    elif args.command == "check":
        check(args.shard, args.numerical)
    elif args.command == "train":
        train(args.shard, args.workers)
    elif args.command == "status":
        cfg, signature, assignment = load_assignment()
        from collections import Counter

        counts = Counter()
        for job in assignment["assignments"][args.shard]:
            _, report, resume = paths(cfg, job)
            state = "pending"
            if completed_record(cfg, signature, job):
                state = "complete"
            elif report.exists() or resume.exists():
                state = "started/resumable"
            counts[state] += 1
        print(json.dumps(dict(shard=args.shard, jobs=dict(counts)), indent=2))
    elif args.command in ("pack-inputs", "pack-results"):
        if not args.output:
            parser.error("--output is required")
        (pack_inputs if args.command == "pack-inputs" else pack_results)(args.output)
    else:
        if not args.from_directory:
            parser.error("--from-directory is required")
        merge_results(args.from_directory)


if __name__ == "__main__":
    main()
