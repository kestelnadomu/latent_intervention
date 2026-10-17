"""Summarize existing records; never fit or evaluate to fill missing cases."""

from collections import defaultdict
import json
from pathlib import Path
import statistics

from src.artifact_io import atomic_output
from src.config import encoder_dimension
from .protocol import paths_for


def report(cases, output=None):
    rows, groups = [], defaultdict(list)
    for config, settings in cases:
        paths = paths_for(config, settings)
        variant, dimension, seed = (
            config["latent_intervention"]["variant"],
            encoder_dimension(config["encoder"]),
            settings["seed"],
        )
        training = (
            json.loads(paths["training"].read_text())
            if paths["training"].exists()
            else None
        )
        evaluation = (
            json.loads(paths["evaluation"].read_text())
            if paths["evaluation"].exists()
            else None
        )
        tier = (
            training["capacity"]["tier"]
            if training
            else evaluation["tier"]
            if evaluation
            else settings["tier"]
        )
        if evaluation and (
            (
                evaluation["variant"],
                evaluation["dimension"],
                evaluation["seed"],
                evaluation["tier"],
            )
            != (variant, dimension, seed, tier)
            or (
                training
                and evaluation["checkpoint_sha256"] != training["checkpoint_sha256"]
            )
        ):
            raise ValueError(
                "editor evaluation does not match its case/selected checkpoint"
            )
        rows.append((variant, dimension, seed, tier, training, evaluation))
        if evaluation:
            groups[(variant, dimension, tier)].append(evaluation)
    lines = [
        "# Clean semantic editors",
        "",
        "Existing records only; missing fits/evaluations are not launched.",
        "",
        "| Method | Dimension | Tier | Seed | Selected epoch | Validation objective | Test evaluation |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for variant, dimension, seed, tier, training, evaluation in rows:
        epoch = training["selected_epoch"] if training else "—"
        value = (
            f"{training['selected_validation']['objective']:.6g}" if training else "—"
        )
        lines.append(
            f"| {variant} | {dimension} | {tier} | {seed} | {epoch} | {value} | {'available' if evaluation else 'not run'} |"
        )
    lines += [
        "",
        "## Explicit test evaluations",
        "",
        "g-consistency, realized-state accuracy and oracle recovery are different criteria. Means ± sample standard deviations below are across training seeds, not test-unit errors. Previously viewed test results remain exploratory.",
        "",
    ]
    for (variant, dimension, tier), records in sorted(groups.items()):
        lines += [
            f"### {variant}, {dimension}D, tier {tier}",
            "",
            f"{len(records)} training seed(s): {'single-seed only' if len(records) == 1 else 'multi-seed'}.",
            "",
        ]
        for subset in ("all", "identity", "nonidentity"):
            lines += [f"{subset}:", ""]
            for metric in records[0]["metrics"][subset]["metrics"]:
                values = [r["metrics"][subset]["metrics"].get(metric) for r in records]
                if any(v is None for v in values):
                    continue
                spread = (
                    f" ± {statistics.stdev(values):.6g}"
                    if len(values) > 1
                    else " (single seed)"
                )
                lines.append(f"- {metric}: {statistics.mean(values):.6g}{spread}")
            lines.append("")
    text = "\n".join(lines) + "\n"
    if output is not None:
        with atomic_output(Path(output)) as temporary:
            temporary.write_text(text)
    return text
