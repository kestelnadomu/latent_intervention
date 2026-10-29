"""Independent GPU worker; shared preparation, preflight, fitting and evaluation."""

from __future__ import annotations

import os
import subprocess
import sys

from exp.text_realizer.protocol import lock, read_json, status
from exp.text_realizer_package.data import prepare, prepared_path
from exp.text_realizer_package.workflow import TESTS, available_gpu
from src.artifact_io import sha256_file

from .settings import ROOT, create_plan, require_current, run_paths


def run(config: dict, run_id: str, *, device: str, resume: bool, wait: bool) -> None:
    _, reports = run_paths(config, run_id)
    with lock(reports / ".run.lock"):
        if (reports / "plan.json").exists() and not resume:
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
                    "tests/text_realizer_package_1b",
                    "-m",
                    "not text_realizer_gpu",
                ],
                cwd=ROOT,
                check=True,
            )
            from exp.text_realizer_package.preflight import environment_check

            environment_check(config)
            with available_gpu(device, reports, wait=wait):
                require_current(reports, config, prepared)
                stages = ["preflight", "train", "finalize"]
                ready = reports / "preflight/ready.json"
                if resume and ready.exists() and read_json(ready).get("ready"):
                    stages.remove("preflight")
                for phase in stages:
                    require_current(reports, config, prepared)
                    status(reports, phase)
                    command = [
                        sys.executable,
                        "-u",
                        "-m",
                        "exp.text_realizer_package_1b",
                        "_stage",
                        phase,
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
                        "TEXT_REALIZER_PACKAGE_1B_LEASE": str(reports.resolve()),
                    }
                    subprocess.run(command, cwd=ROOT, env=environment, check=True)
                    require_current(reports, config, prepared)
                if not read_json(reports / "frozen.json")["reload_verified"]:
                    raise ValueError("missing reload verification")
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


def stage(config: dict, run_id: str, phase: str, *, device: str, resume: bool) -> None:
    _, reports = run_paths(config, run_id)
    if os.environ.get("TEXT_REALIZER_PACKAGE_1B_LEASE") != str(reports.resolve()):
        raise PermissionError("internal stages require the 1B workflow GPU lease")
    prepared = prepared_path(config)
    require_current(reports, config, prepared)
    if phase == "preflight":
        from exp.text_realizer_package.preflight import preflight

        preflight(config, reports, device=device)
    elif phase == "train":
        from exp.text_realizer_package.training import train

        train(config, run_id, device=device, resume=resume)
    elif phase == "finalize":
        from exp.text_realizer_package.evaluation import finalize

        finalize(config, run_id, device=device)
    else:
        raise ValueError("unknown 1B workflow phase")
    require_current(reports, config, prepared)
