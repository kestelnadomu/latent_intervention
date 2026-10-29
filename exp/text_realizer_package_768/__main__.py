"""Explicit CLI; importing the package never starts training or model downloads."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .settings import DEFAULT_CONFIG, read_config, run_paths


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="create a factual fit/development bundle only"
    )
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run = commands.add_parser(
        "run", help="tests, GPU preflight, train, diagnose and export"
    )
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--run-id", required=True)
    run.add_argument("--device", required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--wait-for-gpu", action="store_true")
    stage = commands.add_parser(
        "_stage", help="internal worker; requires workflow GPU lease"
    )
    stage.add_argument("stage", choices=("preflight", "train", "finalize"))
    stage.add_argument("--config", type=Path, required=True)
    stage.add_argument("--run-id", required=True)
    stage.add_argument("--device", required=True)
    stage.add_argument("--resume", action="store_true")
    inference = commands.add_parser("generate", help="decode ID-aligned 768-D vectors")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--latents", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--device", default="cpu")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "generate":
        from exp.text_realizer.runtime import gpu_lease
        from .evaluation import generate_vectors

        with gpu_lease(args.device):
            generate_vectors(
                args.checkpoint, args.latents, args.output, device=args.device
            )
        return
    config = read_config(args.config)
    if args.command == "prepare":
        from .data import prepare

        print(prepare(config))
    elif args.command == "run":
        from .workflow import run

        if not args.device.startswith("cuda:"):
            raise ValueError("explicit CUDA device required")
        run(
            config,
            args.run_id,
            device=args.device,
            resume=args.resume,
            wait=args.wait_for_gpu,
        )
    else:
        _, reports = run_paths(config, args.run_id)
        if os.environ.get("TEXT_REALIZER_PACKAGE_768_LEASE") != str(reports.resolve()):
            raise PermissionError(
                "internal stages require the parent workflow GPU lease"
            )
        if args.stage == "preflight":
            from .preflight import preflight

            preflight(config, reports, device=args.device)
        elif args.stage == "train":
            from .training import train

            train(config, args.run_id, device=args.device, resume=args.resume)
        else:
            from .evaluation import finalize

            finalize(config, args.run_id, device=args.device)


if __name__ == "__main__":
    main()
