"""Separate 1B CLI. Shared numerical code and the running 270M files stay intact."""

from __future__ import annotations

import argparse
from pathlib import Path

from .settings import DEFAULT_CONFIG, read_config


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="verify/reuse factual development data"
    )
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    for name in ("run", "_stage"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--run-id", required=True)
        command.add_argument("--device", required=True)
        command.add_argument("--resume", action="store_true")
        if name == "run":
            command.add_argument("--wait-for-gpu", action="store_true")
        else:
            command.add_argument("stage", choices=("preflight", "train", "finalize"))
    return result


def main() -> None:
    args = parser().parse_args()
    config = read_config(args.config)
    if args.command == "prepare":
        from exp.text_realizer_package.data import prepare

        print(prepare(config))
        return
    if not args.device.startswith("cuda:"):
        raise ValueError("explicit CUDA device required")
    if args.command == "run":
        from .workflow import run

        run(
            config,
            args.run_id,
            device=args.device,
            resume=args.resume,
            wait=args.wait_for_gpu,
        )
    else:
        from .workflow import stage

        stage(config, args.run_id, args.stage, device=args.device, resume=args.resume)


if __name__ == "__main__":
    main()
