"""Scheduling changes cannot add, omit or duplicate native evaluation conditions."""

import pytest

from scripts.hz_misspecification_evaluate import tasks, task_key
from exp.benchmarks.misspecification import METHODS


def test_queue_matches_native_conditions_and_deduplicates_aliases():
    signature = dict(
        config=dict(seeds=[42, 43]),
        scenarios=[
            dict(name="S1/true", canonical="S1/true", targets=["total", "ps_D"]),
            dict(
                name="O3/drop_X_D", canonical="O3/drop_X_D", targets=["total", "ps_D"]
            ),
            dict(
                name="O4/block_X_D", canonical="O3/drop_X_D", targets=["total", "ps_D"]
            ),
        ],
    )
    queue = tasks(signature)
    expected = {
        (case, method, seed, action, target)
        for case in ("S1/true", "O3/drop_X_D")
        for method in (*METHODS, "no_edit")
        for seed in (42, 43)
        for action in range(4)
        for target in ("total", "ps_D")
    }
    assert set(queue) == expected
    assert len(queue) == len(expected) == len({task_key(t) for t in queue})
    assert {t[1] for t in queue[:6]} == {*METHODS, "no_edit"}
    assert all(t[0] == "S1/true" and t[2:] == (42, 0, "total") for t in queue[:6])


@pytest.mark.parametrize("fail_first", [False, True])
def test_worker_preserves_native_cache_and_records_failures(tmp_path, monkeypatch, fail_first):
    from scripts import hz_misspecification_evaluate as queue
    from exp.benchmarks.misspecification import evaluation
    from exp.benchmarks.misspecification.common import read_record, write_record

    cfg = dict(reports_root=str(tmp_path), run_id="synthetic")
    case = object()
    task = ("S1/true", "semantic_editor", 42, 0, "total")
    path = evaluation.condition_path(cfg, *task)
    monkeypatch.setattr(queue, "_CONTEXT", (cfg, {}, {}, {}, {"S1/true": case}))

    def native(*args):
        assert args[4:] == (case, "semantic_editor", 42, 0, "total")
        if path.exists():
            return read_record(path)
        record = dict(elapsed_seconds=0.5)
        write_record(path, record)
        return record  # Native first-run return does not contain record_sha256.

    if fail_first:
        def broken(*_):
            raise FloatingPointError("synthetic solver failure")

        monkeypatch.setattr(evaluation, "evaluate_condition", broken)
        result = queue.worker(task)
        assert result["status"] == "failed"
        assert "synthetic solver failure" in result["traceback"]
        assert not path.exists()

    monkeypatch.setattr(evaluation, "evaluate_condition", native)
    first = queue.worker(task)
    assert first["status"] == "complete" and not first["cached"]
    saved = path.read_bytes()
    second = queue.worker(task)
    assert second["status"] == "complete" and second["cached"]
    assert first["metrics_sha256"] == second["metrics_sha256"]
    assert path.read_bytes() == saved
