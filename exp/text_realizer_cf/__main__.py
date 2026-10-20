"""Explicit commands for the isolated counterfactual-exposed text-decoder pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .settings import ARMS, DEFAULT_CONFIG


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    for name in ("prepare", "preflight", "train", "evaluate", "status"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        if name in {"train", "evaluate", "status"}:
            command.add_argument("--run-id", required=True)
        if name in {"preflight", "train", "evaluate"}:
            command.add_argument("--device", default="cuda:0")
        if name == "train":
            command.add_argument("--arm", choices=ARMS, required=True)
            command.add_argument("--resume", action="store_true")
    return cli


def main() -> None:
    args = parser().parse_args()
    from .settings import prepared_path, read_config, run_paths

    config = read_config(args.config)
    if args.command == "prepare":
        from .data import prepare

        print(prepare(config))
        return
    if args.command == "status":
        from .protocol import read_json

        _, root = run_paths(config, args.run_id)
        paths = {"pilot": root / "status.json"}
        for arm in ARMS:
            paths[arm] = root / arm / "status.json"
        paths["preflight"] = prepared_path(config) / "preflight.json"
        print(
            json.dumps(
                {
                    name: read_json(path) if path.exists() else None
                    for name, path in paths.items()
                },
                indent=2,
            )
        )
        return
    if args.command == "preflight":
        from .preflight import preflight

        result = preflight(config, device=args.device)
    else:
        from exp.text_realizer.runtime import gpu_lease

        with gpu_lease(args.device):
            if args.command == "train":
                from .training import train_arm

                result = train_arm(
                    config,
                    args.run_id,
                    args.arm,
                    device=args.device,
                    resume=args.resume,
                )
            else:
                from .evaluation import evaluate_pilot

                result = evaluate_pilot(config, args.run_id, device=args.device)
    print(json.dumps({k: v for k, v in result.items() if k != "identity"}, indent=2))


if __name__ == "__main__":
    main()
