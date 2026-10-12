"""Readable, regenerable comparison of all encoder/decoder combinations."""

import os
from pathlib import Path

from src.artifact_io import atomic_output


def _value(moment, percent=False):
    scale, digits = (100, 2) if percent else (1, 4)
    return f"{scale * moment['mean']:.{digits}f} ± {scale * moment['std']:.{digits}f}"


def render(root, plan, results, evaluations, selection, state):
    root = Path(root)
    evaluation = {r["case"]: r for r in evaluations}
    ranked = sorted(
        results, key=lambda r: (r["summary"]["validation_joint_nll"]["mean"], r["case"])
    )
    cfg = plan["configs"][0]["semantic_decoder"]
    split = plan["split_ids"]
    lines = [
        "# Semantic decoder benchmark",
        "",
        f"Run: `{root.name}`. Status: **{state}**.",
        "",
        f"Trained cases: {len(results)}/{len(plan['configs'])}; evaluated cases: {len(evaluations)}/{len(plan['configs'])}.",
        "",
        "## Selection",
        "",
    ]
    if selection:
        lines += [
            f"Validation-selected recommendation: **`{selection['recommended_case']}`**.",
            "",
            "Selection was frozen before reading any official-test performance. The lowest mean full joint validation NLL across the five predeclared final seeds wins; ties use case-name order. Seed 42 is the active checkpoint, not the best-looking seed.",
            "",
            *[
                f"- Best `{variant}`: `{case}`."
                for variant, case in selection["best_by_decoder"].items()
            ],
            "",
        ]
    else:
        lines += [
            "No final recommendation yet. Any ranking below is partial; all cases must finish before selection and test evaluation.",
            "",
        ]
    lines += [
        "## Protocol",
        "",
        f"- {len(plan['inventory'])} frozen latent spaces × two decoder variants; no encoders are retrained.",
        f"- Same official IDs throughout: {len(split['fit'])} decoder-fit / {len(split['validation'])} validation / {len(split['official_test'])} test units.",
        f"- {cfg['search']['trials']} identical hyperparameter candidates per case; learning rate, weight decay, dropout and hidden width are tuned only on validation joint NLL.",
        f"- At most {cfg['epochs']} epochs, validation early stopping (patience {cfg['early_stopping']['patience']}), plateau LR reduction, best-checkpoint restoration.",
        f"- Final initialization seeds: {cfg['search']['final_seeds']}. Hyperparameters are selected using search seed {cfg['search']['initialization_seed']}.",
        "- Full joint NLL is in nats/example and comparable across independent and autoregressive decoders, despite their different optimization-loss scales.",
        "- Mean ± sample SD summarizes initialization variability, not a confidence interval over test units. Validation is selection-biased; small ranking differences are not evidence of statistical significance.",
        "- Higher accuracy is better; lower NLL, Brier and ECE are better. ECE is descriptive, uses fixed equal-width confidence bins, and no post-hoc calibrator is fitted.",
        "- Test results are a final check of the frozen selection. Choosing another winner from this test table would reuse the test set for model selection.",
        "- Parameter counts may increase with latent width/selected hidden width; the compute budget is equal per case, not equal per encoder family (families have different numbers of widths).",
        "",
    ]
    caveat = plan.get("langvae_ft_validation_caveat")
    if caveat:
        lines += [
            "### LangVAE fine-tuning caveat",
            "",
            f"Of the decoder-validation texts, {caveat['g_validation_texts_used_in_adaptation_training']} were used for unsupervised LangVAE adaptation training and {caveat['g_validation_texts_used_in_adaptation_validation']} for its validation/checkpoint selection. Decoder-validation structured labels were not used for fitting g, but these texts are not completely untouched end-to-end validation examples. All official-test IDs were excluded from adaptation. Account for this asymmetry when interpreting validation rankings.",
            "",
        ]
    lines += [
        "## Main comparison (ordered by validation NLL)",
        "",
        "Joint accuracy uses the exact joint MAP state, not greedy autoregressive decoding. Macro ECE is the unweighted mean across the structured-variable marginals. Accuracy is in percent; ECE is on [0, 1].",
        "",
        "| Rank | Encoder | Dim | g | Validation NLL | Test NLL | Test joint accuracy (%) | Test macro ECE |",
        "|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for rank, result in enumerate(ranked, 1):
        metrics = evaluation.get(result["case"], {}).get("factual_test")
        fields = [
            _value(metrics[key], percent=key == "joint_map_accuracy")
            if metrics
            else "pending"
            for key in ("joint_nll", "joint_map_accuracy", "macro_marginal_ece")
        ]
        lines.append(
            f"| {rank} | {result['encoder_tag']} | {result['latent_dimension']} | {result['decoder']} | {_value(result['summary']['validation_joint_nll'])} | {' | '.join(fields)} |"
        )
    if evaluations:
        columns = list(evaluations[0]["factual_test"]["marginals"])
        lines += [
            "",
            "## Factual-test marginal accuracy (%)",
            "",
            "These use each exact marginal's argmax, without true-prefix teacher forcing. T is hidden in the text and must be inferred from noisy proxies; 100% is not an appropriate expected target. Y is not a decoder target.",
            "",
            "| Case | " + " | ".join(columns) + " | Joint Brier |",
            "|---|" + "---:|" * (len(columns) + 1),
        ]
        for result in ranked:
            metric = evaluation.get(result["case"], {}).get("factual_test")
            if metric:
                fields = [
                    _value(metric["marginals"][name]["accuracy"], True)
                    for name in columns
                ]
                lines.append(
                    f"| {result['case']} | {' | '.join(fields)} | {_value(metric['joint_brier'])} |"
                )
        lines += [
            "",
            "## Counterfactual-test diagnostics",
            "",
            "These are real encoded counterfactual texts, not outputs of a trained latent manipulator. They do not demonstrate h_Z performance or fairness. The non-identity subset removes copied identity pairs. Factual and counterfactual rows share unit IDs and must not be treated as independent samples.",
            "",
            "| Case | All CF NLL | Non-identity CF NLL | Non-identity joint accuracy (%) | Non-identity macro ECE |",
            "|---|---:|---:|---:|---:|",
        ]
        for result in ranked:
            item = evaluation.get(result["case"])
            if not item:
                continue
            metric = item.get("counterfactual_nonidentity_test")
            fields = [
                _value(metric[key], key == "joint_map_accuracy") if metric else "n/a"
                for key in ("joint_nll", "joint_map_accuracy", "macro_marginal_ece")
            ]
            lines.append(
                f"| {result['case']} | {_value(item['counterfactual_test']['joint_nll'])} | {' | '.join(fields)} |"
            )
    lines += [
        "",
        "## Selected settings and checkpoints",
        "",
        "These links point to the retained seed-42 experiment checkpoints. After successful publication, the same weights also occupy the canonical active paths. Every other final seed and tuning trial is retained.",
        "",
        "| Case | Hidden width | LR | Weight decay | Dropout | Seed-42 best/run epochs | Parameters | Checkpoint |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in ranked:
        settings = result["selected_trial"]["settings"]
        final = next(
            r for r in result["final_runs"] if r["seed"] == result["deployment_seed"]
        )
        target = os.path.relpath(Path(final["checkpoint"]).resolve(), root.resolve())
        parameters = evaluation.get(result["case"], {}).get("parameters", "pending")
        lines.append(
            f"| {result['case']} | {settings['hidden_dim']} | {settings['lr']:g} | {settings['weight_decay']:g} | {settings['dropout']:g} | {final['best_epoch']}/{final['epochs_run']} | {parameters} | [model]({target}) |"
        )
    lines += [
        "",
        "## Storage and reproducibility",
        "",
        "```text",
        "models/talent/<encoder>/g-<variant>/semantic_decoder.pt        # active seed 42",
        f"models/talent/<encoder>/g-<variant>/experiments/{root.name}/",
        "  search/trial-NNN/semantic_decoder.pt",
        "  final/seed-NN/semantic_decoder.pt",
        f"models/talent/<encoder>/g-<variant>/archive/{root.name}/       # previous active model",
        "reports/talent/<encoder>/g-<variant>/                         # matching reports/history",
        f"reports/talent/decoder_benchmarks/{root.name}/",
        "  summary.md, results.json, evaluation.json, selection.json",
        "  protocol.json, plan.json, status.json, published.json",
        "  configs/<encoder>.yaml, source/, evaluation/<encoder>/g-<variant>/",
        "```",
        "",
        "The protocol records source/input hashes, all cases, common candidates, exact splits and package versions. Each training report includes learning curves, best epoch and checkpoint checksum; per-seed evaluation JSON includes reliability bins. Existing active models are archived before replacement. Original embeddings are never rewritten.",
        "",
        "Models and reports are Git-ignored: they are saved locally but must be copied explicitly for another machine. Use the saved per-encoder YAML with `src.pipeline` and the matching `--decoder-variant` for downstream work. Do not mix checkpoints from different latent spaces. No h_Z training has been started.",
        "",
    ]
    with atomic_output(root / "summary.md") as temporary:
        temporary.write_text("\n".join(lines), encoding="utf-8")
