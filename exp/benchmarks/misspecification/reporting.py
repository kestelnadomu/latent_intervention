"""Compact summaries; seed variation and paired-unit uncertainty stay separate."""

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from src.artifact_io import atomic_output, sha256_file
from .common import digest, read_record, root, write_record

PRIMARY = (
    "energy_score",
    "own_g_semantic_kl",
    "mean_l2",
    "mean_cosine",
    *[f"true_accuracy_{n}" for n in ("X", "T", "D", "U", "M", "P", "L", "H", "A")],
    *[f"preservation_accuracy_drop_{n}" for n in ("T", "M", "P", "L", "H", "A")],
)


def bootstrap(values, *, seed=20260930, resamples=2000):
    """Resample units, not draws or seeds; callers pass paired differences."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, resamples, 100):
        indices = rng.integers(
            len(values), size=(min(100, resamples - start), len(values))
        )
        means.extend(values[indices].mean(1).tolist())
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def collapse(records):
    """Average actions within units, then seeds; do not multiply sample size."""
    seeds = defaultdict(list)
    for record in records:
        if sha256_file(record["per_unit"]) != record["per_unit_sha256"]:
            raise ValueError("per-unit metric checksum mismatch")
        payload = torch.load(record["per_unit"], map_location="cpu", weights_only=True)
        seeds[record["reader_seed"]].append(payload)
    ids = next(iter(seeds.values()))[0]["ids"]
    for rows in seeds.values():
        if any(r["ids"] != ids for r in rows):
            raise ValueError("paired report IDs are misaligned")
    averaged = {
        key: torch.stack(
            [
                torch.stack([r["values"][key] for r in rows]).nanmean(0)
                for _, rows in sorted(seeds.items())
            ]
        )
        for key in PRIMARY
    }
    return ids, sorted(seeds), averaged


def report(cfg, signature):
    grouped = defaultdict(list)
    for path in (root(cfg) / "evaluation").glob("*/*/seed-*/action-*/*/metrics.json"):
        record = read_record(path)
        if record["identity"]["protocol"] != digest(signature):
            raise ValueError("evaluation belongs to another protocol")
        grouped[(record["scenario"], record["target"], record["method"])].append(record)
    summaries, arrays = {}, {}
    for key, records in sorted(grouped.items()):
        ids, seeds, values = collapse(records)
        metrics = {}
        for name, value in values.items():
            unit_mean = value.nanmean(0)
            seed_means = value.nanmean(1)
            valid = seed_means[torch.isfinite(seed_means)]
            metrics[name] = dict(
                mean=float(valid.mean()) if len(valid) else None,
                seed_std=float(valid.std()) if len(valid) > 1 else None,
                seed_means=[
                    float(v) if torch.isfinite(v) else None for v in seed_means
                ],
                paired_unit_ci95=bootstrap(
                    unit_mean.numpy(),
                    seed=cfg["evaluation_seed"],
                    resamples=cfg["bootstrap_resamples"],
                ),
                units=int(torch.isfinite(unit_mean).sum()),
            )
        label = "|".join(key)
        summaries[label] = dict(
            scenario=key[0],
            target=key[1],
            method=key[2],
            seeds=seeds,
            completed_conditions=len(records),
            expected_conditions=4 * len(cfg["seeds"]),
            metrics=metrics,
        )
        arrays[key] = (ids, seeds, values)
    differences = []
    for scenario, target in sorted({k[:2] for k in arrays}):
        methods = sorted(k[2] for k in arrays if k[:2] == (scenario, target))
        for a, b in combinations(methods, 2):
            aid, aseed, av = arrays[(scenario, target, a)]
            bid, bseed, bv = arrays[(scenario, target, b)]
            if aid != bid or aseed != bseed:
                continue  # Incomplete queues must not pretend to be paired comparisons.
            if any(
                summaries["|".join((scenario, target, m))]["completed_conditions"]
                != 4 * len(aseed)
                for m in (a, b)
            ):
                continue
            metrics = {}
            for name in PRIMARY:
                delta = (av[name] - bv[name]).nanmean(0)
                valid = delta[torch.isfinite(delta)]
                metrics[name] = dict(
                    mean_difference=float(valid.mean()) if len(valid) else None,
                    paired_unit_ci95=bootstrap(
                        delta.numpy(),
                        seed=cfg["evaluation_seed"],
                        resamples=cfg["bootstrap_resamples"],
                    ),
                    units=len(valid),
                )
            differences.append(
                dict(
                    scenario=scenario,
                    target=target,
                    contrast=f"{a} minus {b}",
                    seeds=aseed,
                    metrics=metrics,
                )
            )
    result = dict(
        protocol_sha256=digest(signature),
        summaries=summaries,
        paired_differences=differences,
        aliases={r["name"]: r["canonical"] for r in signature["scenarios"]},
        uncertainty="SD across reader/training seeds; separate 2000 paired-unit bootstrap resamples after seed/action averaging",
        warning="Gaussian has one deterministic fit: its seed variation comes from readers and shared symbolic draws",
    )
    write_record(root(cfg) / "results.json", result)
    lines = [
        "# Five-method misspecification benchmark",
        "",
        f"Run `{cfg['run_id']}`. {len(signature['jobs'])} hZ fits prepared; "
        "independent evaluation reader selected separately.",
        "",
        "True factual labels are available for hZ. This is not an end-to-end 1,000-label experiment.",
        "",
        "Results use full predictive distributions; no rare/unseen transition mass is removed. "
        "Energy is available only for matching total-effect embedding pairs. Lower energy and KL are better.",
        "",
        "| Scenario | Target | Method | Seeds | Conditions | Energy | Own-g KL | Independent X accuracy |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def show(value):
        return "unavailable" if value is None else f"{value:.5g}"

    for row in signature["scenarios"]:
        for value in summaries.values():
            if (
                value["scenario"] != row["canonical"]
                or value["target"] not in row["targets"]
            ):
                continue
            metric = value["metrics"]
            lines.append(
                f"| {row['name']} | {value['target']} | {value['method']} | {len(value['seeds'])} | "
                f"{value['completed_conditions']}/{value['expected_conditions']} | "
                f"{show(metric['energy_score']['mean'])} | {show(metric['own_g_semantic_kl']['mean'])} | "
                f"{show(metric['true_accuracy_X']['mean'])} |"
            )
    if not summaries:
        lines += ["", "No official-test results have been produced."]
    lines += [
        "",
        "See `results.json` for seed SD, separate 95% paired-unit intervals, and paired method differences. "
        "Per-condition records contain all head recoveries, preservation, calibration-reference diagnostics, "
        "support strata, identity/nonidentity groups, and numerical/inference cost. "
        "Intervals quantify unit sampling, not the uncertainty of training a new seed. "
        "Single-seed/pilot results are not five-seed findings. "
        "Disjoint P/L labels alone do not establish wording robustness.",
        "",
    ]
    with atomic_output(root(cfg) / "summary.md") as temporary:
        Path(temporary).write_text("\n".join(lines))
    return result
