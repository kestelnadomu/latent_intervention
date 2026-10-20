"""Explicit entry points for the isolated text decoder; never invokes the pipeline."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

from .settings import DEFAULT_CONFIG


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    for name in (
        "prepare",
        "preflight",
        "train",
        "campaign",
        "freeze",
        "audit",
        "oracle-evaluate",
    ):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        if name not in {"prepare", "preflight"}:
            sub.add_argument("--run-id", required=True)
        if name not in {"prepare", "campaign"}:
            sub.add_argument("--device", default="cuda:0")
        if name == "train":
            from .settings import VARIANTS

            sub.add_argument("--variant", choices=VARIANTS, required=True)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--smoke", action="store_true")
        if name == "campaign":
            sub.add_argument("--gpus", nargs="+", required=True)
        if name == "oracle-evaluate":
            sub.add_argument("--allow-oracle-evaluation", action="store_true")
            sub.add_argument("--edited-latents", type=Path)
    for name in ("verify", "generate"):
        sub = commands.add_parser(name)
        sub.add_argument("--checkpoint", type=Path, required=True)
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--cache-dir", type=Path)
        sub.add_argument("--output", type=Path, required=True)
        if name == "verify":
            sub.add_argument(
                "--inherited-gpu-lease", action="store_true", help=argparse.SUPPRESS
            )
        else:
            sub.add_argument("--latents", type=Path, required=True)
    return cli


def main() -> None:
    args = parser().parse_args()
    from src.artifact_io import write_json

    from .runtime import gpu_lease
    from .settings import read_config, require_command

    if hasattr(args, "config"):
        require_command(read_config(args.config), args.command)
    if args.command == "prepare":
        from .data import prepare

        print(prepare(read_config(args.config)))
        return
    if args.command == "campaign":
        from .campaign import campaign

        campaign(
            read_config(args.config), args.config.resolve(), args.run_id, args.gpus
        )
        return
    if args.command == "freeze":
        # Each fresh verification subprocess acquires its own GPU lease.
        from .campaign import freeze

        result = freeze(read_config(args.config), args.run_id, device=args.device)
        print(json.dumps(result, indent=2))
        return
    lease = (
        nullcontext()
        if getattr(args, "inherited_gpu_lease", False)
        else gpu_lease(args.device)
    )
    with lease:
        if args.command == "verify":
            from src.text_realizer.artifacts import verify_checkpoint

            result = verify_checkpoint(
                args.checkpoint, device=args.device, cache_dir=args.cache_dir
            )
            write_json(args.output, result)
        elif args.command == "generate":
            from src.text_realizer.artifacts import load_text_realizer, read_checkpoint

            from .inference import read_latents

            ids, z, payload = read_latents(args.latents)
            metadata = read_checkpoint(args.checkpoint)["metadata"]
            if payload["encoder"] != metadata["encoder"]:
                raise ValueError("input encoder differs from the decoder's encoder")
            model = load_text_realizer(
                args.checkpoint, device=args.device, cache_dir=args.cache_dir
            )
            result = {"ids": ids, "texts": model.decode(z)}
            write_json(args.output, result)
        elif args.command == "preflight":
            from .preflight import preflight

            result = preflight(read_config(args.config), device=args.device)
        elif args.command == "train":
            from .experiment import train

            config = read_config(args.config)
            result = train(
                config,
                args.run_id,
                args.variant,
                args.seed,
                device=args.device,
                resume=args.resume,
                smoke=args.smoke,
            )
            if not args.smoke:
                from .campaign import evaluate_trial

                evaluate_trial(config, args.run_id, device=args.device)
        elif args.command == "audit":
            from .campaign import audit

            result = audit(read_config(args.config), args.run_id, device=args.device)
        else:
            from .oracle import evaluate_oracle

            result = evaluate_oracle(
                read_config(args.config),
                args.run_id,
                allowed=args.allow_oracle_evaluation,
                device=args.device,
                edited=args.edited_latents,
            )
    # Large evaluation rows remain in reports, not terminal output.
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"results", "config", "identity", "base_files_sha256"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
