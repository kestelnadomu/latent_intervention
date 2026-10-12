"""python -m exp.langvae_ft {smoke,train,verify,configure}; never encodes pairs."""

import argparse
import json
from pathlib import Path

import yaml

from .artifacts import ROOT, verify_checkpoint, write_inference_config
from .training import device_for, fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "train"):
        sub = commands.add_parser(name)
        sub.add_argument(
            "--config", type=Path, default=ROOT / "configs/langvae_ft.yaml"
        )
        sub.add_argument("--device", default=None)
        sub.add_argument("--resume", type=Path, default=None)
    verify = commands.add_parser("verify")
    verify.add_argument("--checkpoint", type=Path, required=True)
    verify.add_argument("--device", default="cpu")
    configure = commands.add_parser(
        "configure", help="write opt-in pipeline YAML; does not run encoding"
    )
    configure.add_argument("--checkpoint", type=Path, required=True)
    configure.add_argument("--base-config", type=Path, default=ROOT / "src/config.yaml")
    configure.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command in {"smoke", "train"}:
        directory, result = fit(
            args.config, stage=args.command, device_name=args.device, resume=args.resume
        )
        print(json.dumps({"run_dir": str(directory), **result}, indent=2))
    elif args.command == "verify":
        print(
            json.dumps(
                verify_checkpoint(args.checkpoint, device_for(args.device)), indent=2
            )
        )
    else:
        raw = yaml.safe_load(args.base_config.read_text())
        write_inference_config(raw, args.checkpoint, args.output)
        print(f"Wrote {args.output}; no encoding was run.")


if __name__ == "__main__":
    main()
