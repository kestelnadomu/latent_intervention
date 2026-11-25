"""Scheduling/transfer safety tests; use synthetic artifacts, never fit real data."""

import copy
from concurrent.futures import Future
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from exp.benchmarks.misspecification import common, scenarios
from scripts import hz_misspecification_return12 as sub


class Return12Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.signature = dict(jobs=scenarios.jobs(common.read_config(), scenarios.registry(scenarios.build())))
        cls.remote = sub.parent.remote_jobs(cls.signature["jobs"])
        cls.assignment = dict(record_sha256="parent", assignments=dict(remote=cls.remote))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.temporary.name)
        self.cfg = dict(reports_root="reports", models_root="models", run_id="test", workers=4)
        self.allocation = dict(
            format="hz_misspecification_return12_v1",
            parent_assignment_sha256="parent", protocol_sha256=common.digest(self.signature),
            assignments={role: [j for j in self.remote if sub.returned(j) == (role == "return12")]
                         for role in sub.ROLES},
        )
        common.write_record(sub.ALLOCATION, self.allocation)
        self.allocation = common.read_record(sub.ALLOCATION)

    def loaded(self):
        return self.cfg, self.signature, self.assignment, self.allocation

    def test_partition_matches_exact_requested_twelve(self):
        sub.validate_allocation(self.allocation, self.assignment, self.signature)
        jobs = self.allocation["assignments"]["return12"]
        self.assertEqual(len(jobs), 12)
        self.assertEqual({j["seed"] for j in jobs if j["operator"] == "O2/noise_sigma_x0.5"}, {45, 46})
        for operator in ("O3/drop_X_D", "O3/drop_X_U"):
            self.assertEqual({j["seed"] for j in jobs if j["operator"] == operator}, set(range(42, 47)))
        self.assertTrue(all(j["space"] == "S1" and j["method"] == "ot_editor" for j in jobs))

    def test_invalid_partition_or_parent_rejected(self):
        for change in ("duplicate", "missing", "modified", "swapped", "parent", "protocol"):
            with self.subTest(change=change):
                bad = copy.deepcopy(self.allocation)
                keep, returned = bad["assignments"]["keep38"], bad["assignments"]["return12"]
                if change == "duplicate": returned[-1] = returned[0]
                elif change == "missing": keep.pop()
                elif change == "modified": returned[0]["seed"] = 99
                elif change == "swapped": keep[0], returned[0] = returned[0], keep[0]
                elif change == "parent": bad["parent_assignment_sha256"] = "wrong"
                else: bad["protocol_sha256"] = "wrong"
                with self.assertRaises(ValueError):
                    sub.validate_allocation(bad, self.assignment, self.signature)

    def test_allocation_refuses_live_queue_and_started_returned_job(self):
        sub.ALLOCATION.unlink()
        with patch.object(sub.parent, "load_assignment", return_value=self.loaded()[:3]):
            with common.exclusive(common.root(self.cfg) / "queue.lock"):
                with self.assertRaisesRegex(RuntimeError, "another writer"):
                    sub.allocate()
            job = self.allocation["assignments"]["return12"][0]
            log = sub.artifacts(self.cfg, job)[-1]
            log.parent.mkdir(parents=True)
            log.touch()
            with self.assertRaisesRegex(ValueError, "started fit"):
                sub.allocate()
        self.assertFalse(sub.ALLOCATION.exists())

    def test_queue_submits_only_retained_jobs_and_resumes_first(self):
        original = copy.deepcopy(self.cfg)
        retained = self.allocation["assignments"]["keep38"]
        completed = retained[0]
        resumed = retained[-1]
        _, report, resume = common.paths(self.cfg, resumed)
        common.write_record(report, dict(history=[{}, {}]))
        resume.touch()
        common.write_record(sub.BASE / "keep38/check.json", dict(
            status="passed", role="keep38", allocation_sha256=self.allocation["record_sha256"],
            protocol_sha256=common.digest(self.signature), numerical_requested=True,
        ))
        calls = []
        case = self

        class Executor:
            def __init__(self, *, max_workers, mp_context):
                case.assertEqual(max_workers, 16)
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def submit(self, function, cfg, signature, job):
                case.assertEqual(cfg, original)
                calls.append(job)
                result = Future()
                result.set_result(None)
                return result

        with patch.object(sub, "load_allocation", side_effect=self.loaded), \
             patch.object(sub.parent, "completed_record", side_effect=lambda c, s, j: {"status": "complete"} if j == completed else None), \
             patch.object(sub, "ProcessPoolExecutor", Executor):
            sub.train("keep38", 16)
        self.assertEqual(calls[0], resumed)
        self.assertEqual({common.job_key(j) for j in calls}, {common.job_key(j) for j in retained if j != completed})
        self.assertFalse(any(sub.returned(j) for j in calls))
        self.assertEqual(self.cfg, original)

    def make_incoming(self):
        incoming = Path("incoming")
        common.write_record(incoming / sub.ALLOCATION, self.allocation)
        original_parent = {"test": "parent assignment"}
        common.write_record(incoming / sub.parent.ASSIGNMENT, original_parent)
        for job in self.allocation["assignments"]["return12"]:
            checkpoint, report, _ = common.paths(self.cfg, job)
            target = incoming / checkpoint
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(common.job_key(job).encode())
            common.write_record(incoming / report, dict(
                status="complete", signature="fit-signature", checkpoint=str(checkpoint),
                checkpoint_sha256=sub.parent.sha256_file(target),
                metadata=dict(protocol_sha256=common.digest(self.signature), job=job),
            ))
        files = sub.result_paths(self.cfg, self.allocation, "return12")
        common.write_record(incoming / "TRANSFER_MANIFEST.json", dict(
            kind="return12-return12-results", shard="remote", protocol_sha256=common.digest(self.signature),
            assignment_sha256="parent", files={name: sub.parent.sha256_file(incoming / name) for name in sorted(files)},
        ))
        return incoming

    def test_merge_imports_only_twelve_and_is_idempotent(self):
        incoming = self.make_incoming()
        with patch.object(sub, "load_allocation", side_effect=self.loaded), \
             patch("exp.benchmarks.misspecification.models.load") as load, \
             patch("exp.benchmarks.misspecification.workflow.freeze", return_value=None):
            sub.merge_results("return12", incoming)
            sub.merge_results("return12", incoming)
        self.assertEqual(load.call_count, 24)
        for role, jobs in self.allocation["assignments"].items():
            for job in jobs:
                checkpoint, report, resume = common.paths(self.cfg, job)
                self.assertEqual(checkpoint.exists(), role == "return12")
                self.assertEqual(report.exists(), role == "return12")
                self.assertFalse(resume.exists())

    def test_corrupt_or_conflicting_bundle_never_partially_imports(self):
        for change in ("corrupt", "collision", "extra"):
            with self.subTest(change=change):
                incoming = self.make_incoming()
                name = str(common.paths(self.cfg, self.allocation["assignments"]["return12"][-1])[1])
                destination = None
                if change == "corrupt":
                    (incoming / name).write_bytes(b"corrupt")
                elif change == "collision":
                    destination = Path(name)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(b"different")
                else:
                    manifest = common.read_record(incoming / "TRANSFER_MANIFEST.json")
                    manifest["files"]["unexpected"] = "invalid"
                    common.write_record(incoming / "TRANSFER_MANIFEST.json", manifest)
                with patch.object(sub, "load_allocation", side_effect=self.loaded):
                    with self.assertRaises((ValueError, FileExistsError)):
                        sub.merge_results("return12", incoming)
                self.assertFalse(Path("models").exists())
                if destination:
                    destination.unlink()

    def test_pack_refuses_missing_fit(self):
        with patch.object(sub, "load_allocation", side_effect=self.loaded):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                sub.pack_results("return12", "results.tar.gz")
        self.assertFalse(Path("results.tar.gz").exists())


if __name__ == "__main__":
    unittest.main()
