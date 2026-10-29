"""Independent locked 768-D preflight, training and development diagnostics."""

from __future__ import annotations

import os
import subprocess
import sys

from exp.text_realizer.protocol import lock, read_json, status
from exp.text_realizer_package.workflow import TESTS as REFERENCE_TESTS, available_gpu
from src.artifact_io import sha256_file

from .data import prepare
from .settings import ROOT, create_plan, require_current, run_paths

TESTS = [
    *REFERENCE_TESTS,
    "tests/text_realizer_package_1b",
    "tests/text_realizer_package_768",
]


def run(config: dict, run_id: str, *, device: str, resume: bool, wait: bool) -> None:
    _, reports = run_paths(config, run_id)
    with lock(reports / ".run.lock"):
        existed = (reports / "plan.json").exists()
        if existed and not resume:
            raise ValueError("run ID exists; use --resume or a new run ID")
        try:
            status(reports, "preparing")
            prepared = prepare(config)
            create_plan(config, prepared, reports)
            status(reports, "offline_tests")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    *TESTS,
                    "-m",
                    "not text_realizer_gpu",
                ],
                cwd=ROOT,
                check=True,
            )
            # Cache/model/environment checks are CPU-only and fail before queueing.
            from .preflight import environment_check

            environment_check(config)
            with available_gpu(device, reports, wait=wait):
                require_current(reports, config, prepared)
                stages = ["preflight", "train", "finalize"]
                ready = reports / "preflight/ready.json"
                if resume and ready.exists() and read_json(ready).get("ready"):
                    stages.remove("preflight")
                for stage in stages:
                    require_current(reports, config, prepared)
                    status(reports, stage)
                    command = [
                        sys.executable,
                        "-u",
                        "-m",
                        "exp.text_realizer_package_768",
                        "_stage",
                        stage,
                        "--run-id",
                        run_id,
                        "--device",
                        device,
                        "--config",
                        str(reports / "config.json"),
                    ]
                    if resume:
                        command.append("--resume")
                    environment = {
                        **os.environ,
                        "TEXT_REALIZER_PACKAGE_768_LEASE": str(reports.resolve()),
                    }
                    subprocess.run(command, cwd=ROOT, env=environment, check=True)
                frozen = read_json(reports / "frozen.json")
                if not frozen["reload_verified"]:
                    raise ValueError("missing final reload verification")
                status(
                    reports,
                    "complete",
                    frozen_manifest_sha256=sha256_file(reports / "frozen.json"),
                )
        except BaseException as exc:
            status(
                reports,
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
