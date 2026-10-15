"""Single-method CLI, also used by the thin detached two-dimension launcher."""

import argparse

from src.config import CONFIG_PATH, load_config
from .model import VARIANT
from .protocol import DEFAULT_PROTOCOL, read_settings
from .workflow import evaluate_manipulator, train_manipulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate"))
    parser.add_argument("--dimension", choices=(256, 768), type=int, default=768)
    parser.add_argument("--all-dimensions", action="store_true")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument(
        "--tag", help="fresh output component tag; never overwrites published weights"
    )
    args = parser.parse_args()
    dimensions = (
        read_settings(args.protocol)["dimensions"]
        if args.all_dimensions
        else [args.dimension]
    )
    for dimension in dimensions:
        config = load_config(
            path=args.config,
            encoder_variant="embeddinggemma",
            embeddinggemma_dim=dimension,
            decoder_variant="independent",
            manipulator_variant=VARIANT,
        )
        config["latent_intervention"][VARIANT] = {"protocol": args.protocol}
        if args.tag:
            # Paths have already been resolved. Substitute the component, not arbitrary substrings.
            if not args.tag.replace("-", "").replace("_", "").isalnum():
                parser.error(
                    "tag must contain only letters, numbers, hyphens and underscores"
                )
            for name, value in config["paths"].items():
                if isinstance(value, str):
                    config["paths"][name] = value.replace(
                        f"/h-{VARIANT}/", f"/h-{args.tag}/"
                    )
            config["latent_intervention"]["tag"] = args.tag
        (train_manipulator if args.stage == "train" else evaluate_manipulator)(config)


if __name__ == "__main__":
    main()
