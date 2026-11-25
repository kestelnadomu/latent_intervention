"""Queue/transfer tests only; no real-data fitting or official-test evaluation."""

import copy
from concurrent.futures import Future
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from exp.benchmarks.misspecification import common, scenarios
from scripts import hz_misspecification_shards as shard


@pytest.fixture(scope="module")
def protocol():
    return dict(
        jobs=scenarios.jobs(
            common.read_config(),
            scenarios.registry(scenarios.build()),
        )
    )


def assignment_for(protocol):
    remote = shard.remote_jobs(protocol["jobs"])
    remote_keys = {common.job_key(j) for j in remote}
    return dict(
        protocol_sha256=common.digest(protocol),
        assignments=dict(
            remote=remote,
            local=[j for j in protocol["jobs"] if common.job_key(j) not in remote_keys],
        ),
    )


def test_exact_fifty_disjoint_complete_partition(protocol):
    assignment = assignment_for(protocol)
    shard.validate_partition(assignment, protocol)
    assert len(assignment["assignments"]["remote"]) == 50
    assert len(assignment["assignments"]["local"]) == 146
    remote = assignment["assignments"]["remote"]
    assert sum(j["space"] == "S6" for j in remote) == 20
    assert sum(j["method"] == "ot_editor" for j in remote) == 35
    assert {j["seed"] for j in remote} == {42, 43, 44, 45, 46}
    assert all(j["operator"].startswith("O") for j in remote)


@pytest.mark.parametrize("change", ["duplicate", "missing", "modified", "protocol"])
def test_invalid_assignment_rejected(protocol, change):
    assignment = copy.deepcopy(assignment_for(protocol))
    if change == "duplicate":
        assignment["assignments"]["local"].append(
            assignment["assignments"]["remote"][0]
        )
    elif change == "missing":
        assignment["assignments"]["local"].pop()
    elif change == "modified":
        assignment["assignments"]["remote"][0]["seed"] = 900
    else:
        assignment["protocol_sha256"] = "wrong"
    with pytest.raises(ValueError):
        shard.validate_partition(assignment, protocol)


@pytest.mark.parametrize("name", ["/tmp/escape", "../escape", "ok/../../escape", "."])
def test_bundle_path_traversal_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        shard.safe_path(tmp_path, name)


def test_bundle_symlink_escape_rejected(tmp_path):
    (tmp_path / "outside").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        shard.safe_path(tmp_path, "outside/escape")


def test_archive_checksums_allowlist_and_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("allowed").write_bytes(b"checkpoint bytes")
    Path("secret").write_bytes(b"must not be copied")
    shard.archive(
        Path("transfer.tar.gz"),
        {"allowed"},
        "inputs",
        {"record_sha256": "assigned"},
        {},
    )
    with tarfile.open("transfer.tar.gz") as bundle:
        assert set(bundle.getnames()) == {"allowed", "TRANSFER_MANIFEST.json"}
        manifest = json.load(bundle.extractfile("TRANSFER_MANIFEST.json"))
    assert manifest["files"] == {"allowed": shard.sha256_file("allowed")}
    assert (
        Path("transfer.tar.gz.sha256")
        .read_text()
        .startswith(shard.sha256_file("transfer.tar.gz"))
    )
    with pytest.raises(FileExistsError):
        shard.archive("transfer.tar.gz", {"allowed"}, "inputs", {}, {})


def test_runtime_worker_override_never_changes_fitting_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(shard, "BASE", Path("shards/split"))
    cfg = dict(workers=4, reports_root="reports", models_root="models", run_id="test")
    original = copy.deepcopy(cfg)
    jobs = [
        dict(space="S1", method="semantic_editor", operator="O1/unaware", seed=s)
        for s in (42, 43)
    ]
    assignment = dict(
        assignments=dict(remote=jobs),
        record_sha256="assignment",
        orchestration_sha256={},
        resumed_epochs={},
    )
    monkeypatch.setattr(shard, "load_assignment", lambda: (cfg, {}, assignment))
    monkeypatch.setattr(shard, "completed_record", lambda *args: None)
    common.write_record(
        shard.BASE / "remote/check.json",
        dict(
            status="passed",
            assignment_sha256="assignment",
        ),
    )
    calls = []

    class Executor:
        def __init__(self, *, max_workers, mp_context):
            assert max_workers == 8

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, config, signature, job):
            calls.append((copy.deepcopy(config), job))
            result = Future()
            result.set_result(None)
            return result

    monkeypatch.setattr(shard, "ProcessPoolExecutor", Executor)
    shard.train("remote", 8)
    assert cfg == original and all(config == original for config, _ in calls)
    status = common.read_record(shard.BASE / "remote/status.json")
    assert set(status["jobs"].values()) == {"complete"}
    assert status["workers"] == 8
    assert not Path("reports/test/status.json").exists()


def test_numerical_check_skips_completed_and_routes_factual(tmp_path, monkeypatch):
    from exp.benchmarks.misspecification import data, workflow

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(shard, "BASE", Path("shards/split"))
    cfg = dict(reports_root="reports", run_id="test")
    jobs = [
        dict(space="S1", method=method, operator="factual", seed=42)
        for method in ("state_flow", "ot_flow")
    ]
    assignment = dict(assignments=dict(local=jobs), record_sha256="assignment")
    monkeypatch.setattr(shard, "load_assignment", lambda: (cfg, {}, assignment))
    monkeypatch.setattr(
        shard,
        "completed_record",
        lambda cfg, signature, job: (
            {"status": "complete"} if job["method"] == "state_flow" else None
        ),
    )
    common.write_record(
        Path("reports/test/preflight.json"),
        dict(
            status="passed",
            signature_sha256=common.digest({}),
        ),
    )
    monkeypatch.setattr(data, "bundle", lambda cfg: ({"sim_config": "synthetic"}, None))
    monkeypatch.setattr(data, "load_factual", lambda *args: None)
    monkeypatch.setattr(data, "guide", lambda *args: None)
    monkeypatch.setattr(
        scenarios,
        "build",
        lambda path: [SimpleNamespace(name="S1/true", matrices="expected")],
    )
    calls = []

    def numerical(config, method, data, decoder, matrices):
        calls.append((method, matrices))
        return dict(optimizer_steps=0)

    monkeypatch.setattr(workflow, "numerical_check", numerical)
    shard.check("local", numerical=True)
    assert calls == [("ot_flow", "expected")]


def test_split_refuses_active_original_queue(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = dict(reports_root="reports", run_id="test")
    monkeypatch.setattr(shard, "read_config", lambda **kwargs: cfg)
    monkeypatch.setattr(shard, "read_protocol", lambda _: {})
    with common.exclusive(Path("reports/test/queue.lock")):
        with pytest.raises(RuntimeError, match="another writer"):
            shard.split()


def test_completed_checkpoint_integrity_and_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = dict(models_root="models", reports_root="reports", run_id="test")
    job = dict(space="S1", method="ot_editor", operator="O1/unaware", seed=42)
    checkpoint, report, _ = common.paths(cfg, job)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"model")
    record = dict(
        status="complete",
        checkpoint=str(checkpoint),
        checkpoint_sha256=shard.sha256_file(checkpoint),
        metadata=dict(protocol_sha256=common.digest({}), job=job),
    )
    common.write_record(report, record)
    assert shard.completed_record(cfg, {}, job)["status"] == "complete"
    checkpoint.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        shard.completed_record(cfg, {}, job)


def test_merge_validates_all_files_before_writing(tmp_path, monkeypatch, protocol):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(shard, "BASE", Path("shards/split"))
    monkeypatch.setattr(shard, "ASSIGNMENT", Path("shards/split/assignment.json"))
    cfg = dict(models_root="models", reports_root="reports", run_id="test")
    assignment = assignment_for(protocol)
    common.write_record(shard.ASSIGNMENT, assignment)
    assignment = common.read_record(shard.ASSIGNMENT)
    monkeypatch.setattr(shard, "load_assignment", lambda: (cfg, protocol, assignment))
    incoming = Path("incoming")
    files = {}
    for job in assignment["assignments"]["remote"]:
        checkpoint, report, _ = common.paths(cfg, job)
        saved = incoming / checkpoint
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(b"synthetic model")
        common.write_record(
            incoming / report,
            dict(
                status="complete",
                checkpoint=str(checkpoint),
                signature="fit",
                checkpoint_sha256=shard.sha256_file(saved),
                metadata=dict(protocol_sha256=common.digest(protocol), job=job),
            ),
        )
        for path in (checkpoint, report):
            files[str(path)] = shard.sha256_file(incoming / path)
    shard.copy_atomic(shard.ASSIGNMENT, incoming / shard.ASSIGNMENT)
    files[str(shard.ASSIGNMENT)] = shard.sha256_file(shard.ASSIGNMENT)
    common.write_record(
        incoming / "TRANSFER_MANIFEST.json",
        dict(
            kind="results",
            shard="remote",
            protocol_sha256=common.digest(protocol),
            assignment_sha256=assignment["record_sha256"],
            files=files,
        ),
    )
    last = common.paths(cfg, assignment["assignments"]["remote"][-1])[0]
    last.parent.mkdir(parents=True, exist_ok=True)
    last.write_bytes(b"existing unrelated model")
    with pytest.raises(FileExistsError):
        shard.merge_results(incoming)
    first = common.paths(cfg, assignment["assignments"]["remote"][0])[0]
    assert not first.exists()
    assert last.read_bytes() == b"existing unrelated model"
    # No model deserialization or fitting is needed for this transfer unit test.
    shard.copy_atomic(incoming / last, last)
    from exp.benchmarks.misspecification import models, workflow

    monkeypatch.setattr(models, "load", lambda *args, **kwargs: None)
    monkeypatch.setattr(workflow, "freeze", lambda *args: None)
    shard.merge_results(incoming)
    assert first.read_bytes() == b"synthetic model"
    assert common.read_record(shard.BASE / "merge.json")["remote_jobs"] == 50
    # Exact repeated imports are idempotent.
    shard.merge_results(incoming)
