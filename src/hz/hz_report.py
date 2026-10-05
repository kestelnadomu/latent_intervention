"""Human-readable benchmark protocol and seed-aggregated, separated comparisons."""

import json
import statistics

from src.artifact_io import atomic_output


def render(root, plan, evaluations, status):
    settings = plan["settings"]
    lines = [
        "# h_Z benchmark",
        "",
        f"Status: {status}.",
        "",
        f"{len(plan['cases'])} cases; {len(plan['candidates'])} search trials each; final seeds {settings['final_seeds']}. "
        f"Up to {settings['max_epochs']} epochs, patience {settings['patience']}; best validation weights restored.",
        "",
        f"Split: {len(plan['split_ids']['fit'])} fit / {len(plan['split_ids']['validation'])} validation / "
        f"{len(plan['split_ids']['official_test'])} official test. g is frozen; h_S is analytic.",
        "",
        "## Protocol",
        "",
        "- Hyperparameters and early stopping use each family's declared validation metric. These unlike objectives are not ranked across families.",
        "- Dist: fixed pretraining, then validation-controlled joint fitting within the total epoch ceiling.",
        "- State-flow search winners supply search teachers; final distillation uses the same-seed final state-flow teacher. Teachers fit only fit IDs.",
        "- All choices are frozen before test scoring. Five seeds measure training variability, not independent test datasets or confidence intervals.",
        f"- Stochastic test evaluation: {settings['evaluation_samples']} draws/unit, batches of {settings['evaluation_batch_size']}; deterministic methods are point masses.",
        "- Energy score uses fit-only per-coordinate standard deviations, divides distances by sqrt(d), and uses the unbiased off-diagonal sample estimator (it can be slightly negative).",
        "- Raw MSE is not dimension-comparable by itself. Compare normalized energy/recovery, semantics, identity behavior, and the no-edit reference together.",
        "- g-based metrics are descriptive and may favor models trained through g. A paired counterfactual sample cannot establish full distributional calibration.",
        "- State-aware results are separate from latent-only results. The oracle is a privileged supervised reference, not a guaranteed upper bound.",
        "- Frozen g was tuned on the same inner validation split; this is not independent nested end-to-end validation. The official test remains held out from h_Z selection.",
        "- Trials restart after interruption; completed checkpoint/report pairs are verified and reused. No canonical artifacts are replaced.",
        "",
        "## Cases",
        "",
        "| Case | Validation criterion | Supervision |",
        "|---|---|---|",
    ]
    for case in plan["cases"]:
        lines.append(
            f"| {case['case']} | {case['validation_metric']} | {case['supervision']} |"
        )
    if plan.get("model_sizes"):
        widths = settings["width_multipliers"]
        counts = {
            (row["case"], row["width_multiplier"]): row["trainable_parameters"]
            for row in plan["model_sizes"]
        }
        lines += [
            "",
            "## Model sizes",
            "",
            "Actual trainable h_Z parameters, excluding frozen g and separate distillation teachers. "
            "Widths are explicitly set per dimension/family/tier in the benchmark YAML; "
            "all nine default models match the same-tier baseline transformer within 7%. "
            "Flows keep eight blocks, dist keeps both components, and the oracle keeps two hidden layers. "
            "This matches parameter budgets, not architecture, FLOPs, supervision, or guaranteed convergence. "
            "Exact constructor arguments are in `plan.json`.",
            "",
            "| Case | " + " | ".join(f"Tier {width}" for width in widths) + " |",
            "|---|" + "---:|" * len(widths),
        ]
        for case in plan["cases"]:
            lines.append(
                f"| {case['case']} | "
                + " | ".join(f"{counts[case['case'], width]:,}" for width in widths)
                + " |"
            )
    ready_path = root / "readiness.json"
    if ready_path.exists():
        ready = json.loads(ready_path.read_text())
        lines += ["", "## Launch prerequisites", ""]
        lines.append(
            "Oracle training targets: "
            + "; ".join(
                f"{item['dimension']}D {item['status']}"
                for item in ready["oracle_targets"]
            )
            + "."
        )
        if not ready["ready"]:
            lines += [
                "",
                "The full run is blocked until the separate oracle-target queue is explicitly launched and completes. "
                "Encode nonidentity training counterfactuals once at 768D, copy identity pairs, "
                "and derive 256D by prefix truncation plus normalization; counts are in `readiness.json`. "
                "This benchmark does not start encoding or training during preparation.",
            ]
    for regime in ("latent_only", "observed_state"):
        rows = [r for r in evaluations if regime in r["regimes"]]
        if not rows:
            continue
        lines += [
            "",
            f"## {regime.replace('_', ' ').title()}: test mean ± seed SD",
            "",
            "Descriptive comparisons, not a test-selected deployment winner. Lower energy/MSE/KL and higher cosine are better.",
            "",
            "| Case | Seeds | Energy ↓ | Standardized mean MSE ↓ | Cosine ↑ | Semantic KL ↓ | Identity shift L2 ↓ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for case in sorted({r["case"] for r in rows}):
            group = [r["regimes"][regime] for r in rows if r["case"] == case]

            def fmt(metric, subset="all"):
                values = [
                    r[subset]["metrics"][metric]
                    for r in group
                    if metric in r[subset]["metrics"]
                ]
                if not values:
                    return "n/a"
                return f"{statistics.mean(values):.5f} ± {statistics.stdev(values) if len(values) > 1 else 0:.5f}"

            lines.append(
                f"| {case} | {len(group)} | {fmt('energy_score')} | {fmt('standardized_mean_mse')} | {fmt('mean_cosine')} | {fmt('semantic_kl')} | {fmt('mean_shift_l2', 'identity')} |"
            )
    lines += [
        "",
        "## Files",
        "",
        "`plan.json` records exact splits/configs/hashes; `selection.json` freezes choices before test; "
        "`fits/<case>/{search,final}/<trial-or-seed>/training.json` holds every epoch and best-epoch decision; "
        "`evaluation/<case>/seed-*.json` holds all metrics, identity/nonidentity subsets and no-edit references. "
        "`published.json` points to predeclared deployment-seed checkpoints under "
        "`models/talent/<case>/benchmarks/<run>/selected/`. Load these with `src.hz.hz_training.load_model`. "
        "Selected weights plus all completed reports/histories are Git-trackable; trial/nondeployment weights, logs and live progress remain local.",
        "",
    ]
    with atomic_output(root / "summary.md") as temporary:
        temporary.write_text("\n".join(lines), encoding="utf-8")
