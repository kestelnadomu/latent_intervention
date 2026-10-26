"""Individual OT preflight, fitting, independent audit and explicit test evaluation."""

import argparse
import json

from src.config import CONFIG_PATH, encoder_tag, load_config
from . import OT_VARIANTS
from .protocol import DEFAULT_PROTOCOL, read_settings
from .workflow import (
    preflight,
    train_manipulator,
    audit_manipulator,
    evaluate_manipulator,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "train", "audit", "evaluate"))
    parser.add_argument("--variant", choices=OT_VARIANTS)
    parser.add_argument("--dimension", type=int, choices=(256, 768))
    parser.add_argument(
        "--all",
        action="store_true",
        help="Gaussian transport and OT flow at both configured dimensions",
    )
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tier", type=int, choices=(1, 2))
    parser.add_argument("--learning-rate", type=float)
    args = parser.parse_args()
    if args.all and (args.variant or args.dimension):
        parser.error("--all cannot be combined with --variant or --dimension")
    if not args.all and not args.variant:
        parser.error("choose --variant or --all")
    overrides = {
        key: getattr(args, key)
        for key in ("run_id", "seed", "tier", "learning_rate")
        if getattr(args, key) is not None
    }
    settings = read_settings(args.protocol, overrides)
    dimensions = settings["dimensions"] if args.all else [args.dimension or 768]
    variants = OT_VARIANTS if args.all else [args.variant]
    configs = []
    for dimension in dimensions:
        for variant in variants:
            config = load_config(
                path=args.config,
                encoder_variant="embeddinggemma",
                embeddinggemma_dim=dimension,
                decoder_variant="independent",
                manipulator_variant=variant,
            )
            config["latent_intervention"]["ot"] = dict(
                protocol=args.protocol, overrides=overrides
            )
            configs.append(config)
    # An all-case launch verifies every input before starting the first fit.
    if args.stage == "preflight" or (args.stage == "train" and args.all):
        records = [preflight(config) for config in configs]
        print(
            json.dumps(records, indent=2, sort_keys=True, allow_nan=False), flush=True
        )
        if args.stage == "preflight":
            return
    action = dict(
        train=train_manipulator, audit=audit_manipulator, evaluate=evaluate_manipulator
    )[args.stage]
    for config in configs:
        print(
            f"{args.stage}: {encoder_tag(config['encoder'])} / {config['latent_intervention']['variant']}",
            flush=True,
        )
        action(config)


if __name__ == "__main__":
    main()
