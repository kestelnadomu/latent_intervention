"""Prepare once on CPU; generate on allocated devices; score once on CPU."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import DEFAULT_BUNDLE, DEFAULT_CONFIG, DEFAULT_OUTPUT, read_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="export provided flows, vectors, references and decoder weights"
    )
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    prepare.add_argument("--allow-oracle-evaluation", action="store_true")
    generate = commands.add_parser(
        "generate", help="resume one decoder's oracle/predicted reconstructions"
    )
    generate.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    generate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "predictions")
    generate.add_argument(
        "--model", required=True, choices=("256-270m", "256-1b", "768-270m", "768-1b")
    )
    generate.add_argument(
        "--condition", choices=("oracle", "predicted", "both"), default="both"
    )
    generate.add_argument(
        "--device", required=True, help="cpu or a visible device such as cuda:0"
    )
    generate.add_argument("--shard-index", type=int, default=0)
    generate.add_argument("--num-shards", type=int, default=1)
    generate.add_argument(
        "--limit",
        type=int,
        help="stop after this many NEW CVs per arm; rerun without it to continue",
    )
    generate.add_argument("--cache-dir", type=Path)
    generate.add_argument("--threads", type=int, default=1)
    scoring = commands.add_parser(
        "score", help="require complete outputs and score against oracle text"
    )
    scoring.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    scoring.add_argument(
        "--predictions", type=Path, default=DEFAULT_OUTPUT / "predictions"
    )
    scoring.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "scores")
    args = parser.parse_args()
    if args.command == "prepare":
        from .prepare import prepare as run

        manifest = run(
            read_config(args.config),
            args.bundle,
            allow_oracle=args.allow_oracle_evaluation,
        )
        print(
            f"Prepared {len(manifest['ids'])} units and {len(manifest['models'])} decoders at {args.bundle}"
        )
    elif args.command == "generate":
        from .generate import run_generation

        run_generation(
            args.bundle,
            args.output,
            args.model,
            condition=args.condition,
            device=args.device,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            limit=args.limit,
            cache_dir=args.cache_dir,
            threads=args.threads,
        )
    else:
        from .score import score

        report = score(args.bundle, args.predictions, args.output)
        print(
            f"Scored {len(report['metrics'])} model/condition combinations at {args.output}"
        )


if __name__ == "__main__":
    main()
