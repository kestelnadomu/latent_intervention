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
    complete = state.startswith("complete")
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
        (
            "These links point to the canonical active seed-42 checkpoints, which are eligible for Git tracking. Check their hashes against [published.json](published.json) if later training replaces them. Other final seeds and tuning trials remain retained locally."
            if complete
            else "These links point to the retained seed-42 experiment checkpoints. Canonical active models are not yet published for this run."
        ),
        "",
        "| Case | Hidden width | LR | Weight decay | Dropout | Seed-42 best/run epochs | Parameters | Checkpoint |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in ranked:
        settings = result["selected_trial"]["settings"]
        final = next(
            r for r in result["final_runs"] if r["seed"] == result["deployment_seed"]
        )
        checkpoint = (
            result["active_paths"]["decoder_model"] if complete else final["checkpoint"]
        )
        target = os.path.relpath(Path(checkpoint).resolve(), root.resolve())
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
        "Canonical active g checkpoints and training reports, this summary, benchmark JSON, per-seed evaluation JSON, configs and the frozen source snapshot are eligible for Git tracking; add, commit and push them to transfer them. Experiment checkpoints/reports, archives, logs and temporary files remain ignored. Historical references to local-only files are intentional; a clone is not a full resumable copy of every fit. The separately distributed LangVAE-FT encoder checkpoint is still required by its provenance checks.",
        "",
        "Use the saved per-encoder YAML with `src.pipeline` and the matching `--decoder-variant` for downstream work. Do not mix checkpoints from different latent spaces. No h_Z training has been started.",
        "",
        "## Data behind the statistics",
        "",
        "g predicts structured X, T, D, U labels from the saved `data/latents/talent/<encoder>/z_pairs.pt` embeddings. The factual and counterfactual ground-truth labels are ID-aligned from `data/sim_talent/sim_data_factual.csv` and `sim_data_counterfactual.csv`, not reconstructed CV text. `data/sim_talent/pair_index.csv` is the official split authority; the original CVs are in `data/text_talent/cv_factual.csv` and `cv_counterfactual.csv`.",
        "",
        "| Saved record | Statistics or evidence |",
        "|---|---|",
        "| [results.json](results.json) | Each final seed's validation NLL, selected settings, validation mean and sample SD |",
        "| [evaluation.json](evaluation.json) | Aggregated factual-test and counterfactual-test scores for all cases |",
        "| [evaluation/](evaluation/) | Individual-seed metrics, per-attribute scores, reliability bins and input/model hashes |",
        "| [protocol.json](protocol.json), [plan.json](plan.json) | Exact fit/validation/test IDs, input hashes, configurations and search candidates |",
        "| [selection.json](selection.json) | Validation-only ranking frozen before test evaluation |",
        "| [published.json](published.json) | Active checkpoint and training-report paths and checksums |",
        "",
        "The ± values are sample standard deviations across training seeds on the same data split, not confidence intervals over CVs. Per-CV predictions are not saved here; the JSON records contain aggregate metrics and reliability bins. The original source snapshot and protocol remain unchanged when this Markdown is regenerated.",
        "",
    ]
    with atomic_output(root / "summary.md") as temporary:
        temporary.write_text("\n".join(lines), encoding="utf-8")
