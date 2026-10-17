"""Individual or sequential clean-editor commands; test evaluation is explicit."""

import argparse
import json

from src.config import CONFIG_PATH, load_config
from . import EDITOR_VARIANTS
from .protocol import DEFAULT_PROTOCOL, read_settings


def cases_for(args, parser):
    if args.all and (args.variant or args.dimension):
        parser.error("--all cannot be combined with --variant or --dimension")
    if not args.all and not args.variant:
        parser.error("choose --variant or --all")
    overrides = {
        key: getattr(args, key)
        for key in ("run_id", "seed", "tier")
        if getattr(args, key) is not None
    }
    settings = read_settings(args.protocol, overrides)
    dimensions = settings["dimensions"] if args.all else [args.dimension or 768]
    variants = EDITOR_VARIANTS if args.all else [args.variant]
    seeds = settings["seeds"] if args.all and args.seed is None else [settings["seed"]]
    cases = []
    for dimension in dimensions:
        for variant in variants:
            for seed in seeds:
                config = load_config(
                    path=args.config,
                    encoder_variant="embeddinggemma",
                    embeddinggemma_dim=dimension,
                    decoder_variant="independent",
                    manipulator_variant=variant,
                )
                config["latent_intervention"]["editor"] = dict(
                    protocol=args.protocol, overrides=dict(overrides, seed=seed)
                )
                cases.append((config, dict(settings, seed=seed)))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "train", "evaluate", "report"))
    parser.add_argument("--variant", choices=EDITOR_VARIANTS)
    parser.add_argument("--dimension", type=int, choices=(256, 768))
    parser.add_argument(
        "--all",
        action="store_true",
        help="both methods, two dimensions, five seeds (20 default fits)",
    )
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tier", type=int, choices=(1, 2))
    parser.add_argument("--output", help="optional Markdown destination for report")
    args = parser.parse_args()
    if args.output and args.stage != "report":
        parser.error("--output is only valid for report")
    cases = cases_for(args, parser)
    if args.stage == "report":
        from .reporting import report

        print(report(cases, args.output))
        return
    from .workflow import preflight, train_manipulator, evaluate_manipulator

    if args.stage == "preflight" or (args.stage == "train" and args.all):
        records = [preflight(config) for config, _ in cases]
        print(
            json.dumps(records, indent=2, sort_keys=True, allow_nan=False), flush=True
        )
        if args.stage == "preflight":
            return
    action = train_manipulator if args.stage == "train" else evaluate_manipulator
    for config, settings in cases:
        print(
            f"{args.stage}: {config['latent_intervention']['variant']} seed-{settings['seed']}",
            flush=True,
        )
        action(config)


if __name__ == "__main__":
    main()
