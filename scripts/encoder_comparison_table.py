"""Export the completed latent_lab encoder comparison without loading or fitting models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports/latent_lab/run-20261027-225936.json"
DEFAULT_OUTPUT = ROOT / "reports/latent_lab/encoder-comparison-20261027"
FAMILIES = {
    "langvae": ("LangVAE", (128,)),
    "nomic": ("Nomic Embed v1.5", (64, 128, 256, 512, 768)),
    "qwen3": ("Qwen3-Embedding-0.6B", (32, 64, 128, 256, 512, 768, 1024)),
    "embeddinggemma": ("EmbeddingGemma-300m", (128, 256, 512, 768)),
}
METRICS = {
    "pool_oof_joint_acc": ("oof_joint_acc", "psi (mean)"),
    "pool_oof_joint_acc_sd": ("oof_joint_acc", "psi (sd over seeds)"),
    "eval_joint_nll": ("joint_nll", "psi (mean)"),
    "eval_joint_nll_sd": ("joint_nll", "psi (sd over seeds)"),
    "judge_oof_joint_nll": ("joint_nll", "judge"),
    "eval_joint_acc": ("joint_acc", "psi (mean)"),
    "eval_joint_acc_sd": ("joint_acc", "psi (sd over seeds)"),
    "judge_oof_joint_acc": ("joint_acc", "judge"),
}
DISPLAY = (
    ("pool_oof_joint_acc", "pool_oof_joint_acc_sd", 100, 1, "max"),
    ("eval_joint_nll", "eval_joint_nll_sd", 1, 3, "min"),
    ("judge_oof_joint_nll", None, 1, 3, "min"),
    ("eval_joint_acc", "eval_joint_acc_sd", 100, 1, "max"),
    ("judge_oof_joint_acc", None, 100, 1, "max"),
)

CAPTION = r"""Frozen-encoder comparison on the extended-DAG factual CVs.
All readers have independent categorical heads and two width-256 GELU hidden layers;
the two reader columns denote roles, not independent versus autoregressive architectures.
Joint metrics cover $S=(X,T,D,U)$; the additional fitted $M$ head is excluded from them.
For $\psi$, pool accuracy uses out-of-fold predictions on 2,800 units, while evaluation
on 1,000 separate units averages the five fold models' marginal probabilities.
Entries for $\psi$ are means $\pm$ sample standard deviations across seeds 0--4.
The evaluation reader $\psi_{\mathrm{eval}}$ uses a disjoint labelled half of the data,
including evaluation units, with five-fold cross-fitting at seed 0: each unit is
scored by the model that excluded that unit. Its entries have no across-seed deviation.
NLL is in nats; accuracies are percentages. Bold entries mark the best mean per column,
without asserting statistical significance. All results use the original wording
$\mathcal V_A$; paraphrase robustness is a separate analysis."""


def load_rows(report: dict) -> list[dict]:
    """Reject other protocols instead of silently applying this table's caption."""
    settings = report["settings"]
    expected = {
        f"{family}-{dim}" for family, (_, dims) in FAMILIES.items() for dim in dims
    }
    if set(settings["psi_encoders"]) != expected:
        raise ValueError("Expected the completed 17-space encoder comparison")
    if settings["seeds"] != list(range(5)) or settings["read"] != [
        "X",
        "T",
        "D",
        "U",
        "M",
    ]:
        raise ValueError("Reader seeds or heads differ from the documented protocol")
    if settings["states"][settings["main"]] != ["X", "T", "D", "U"]:
        raise ValueError("Joint metrics must cover X, T, D, U")
    if settings["units"] != {"pool": 2800, "psi": 2000, "judge": 2000, "eval": 1000}:
        raise ValueError("Unexpected split sizes")

    metrics = {row["index"]: row for row in report["psi"]}
    if len(metrics) != len(report["psi"]):
        raise ValueError("Duplicate metric rows")
    rows = []
    for family, (label, dims) in FAMILIES.items():
        for dim in dims:
            key = f"{family}-{dim}"
            row = {"encoder_key": key, "encoder": label, "dimension": dim}
            for field, (metric, role) in METRICS.items():
                value = float(metrics[metric][f"{key} | {role}"])
                if (
                    not math.isfinite(value)
                    or value < 0
                    or ("acc" in field and value > 1)
                ):
                    raise ValueError(f"Invalid {key}/{field}: {value}")
                row[field] = value
            for role, seeds in (("psi", range(5)), ("judge", (None,))):
                fits = [
                    f
                    for f in report["training"]
                    if f["model"] == role and f.get("encoder") == key
                ]
                identities = [(f.get("seed"), f["fold"]) for f in fits]
                expected_fits = {(seed, fold) for seed in seeds for fold in range(5)}
                if (
                    len(identities) != len(expected_fits)
                    or set(identities) != expected_fits
                ):
                    raise ValueError(f"Incomplete or duplicate {role} fits for {key}")
            rows.append(row)
    return rows


def best_values(rows: list[dict]) -> dict[str, float]:
    return {
        field: (min if direction == "min" else max)(r[field] for r in rows)
        for field, _, _, _, direction in DISPLAY
    }


def cells(row: dict, best: dict[str, float], latex: bool) -> list[str]:
    result = []
    for field, sd, scale, places, _ in DISPLAY:
        mean = f"{row[field] * scale:.{places}f}"
        if row[field] == best[field]:
            mean = rf"\textbf{{{mean}}}" if latex else f"**{mean}**"
        if sd:
            spread = f"{row[sd] * scale:.{places}f}"
            mean += rf"{{\tiny$\pm${spread}}}" if latex else f" ± {spread}"
        result.append(mean)
    return result


def latex_table(rows: list[dict], best: dict[str, float], source: str, sha: str) -> str:
    lines = [
        "% Generated by scripts/encoder_comparison_table.py; do not edit by hand.",
        f"% Source: {source}",
        f"% Source SHA-256: {sha}",
        "% Requires booktabs. See encoder_comparison.md for the caption and protocol.",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"  \toprule",
        r"  & & \textbf{Pool OOF} & \multicolumn{2}{c}{\textbf{Eval. joint NLL} $\downarrow$}"
        r" & \multicolumn{2}{c}{\textbf{Eval. joint acc. (\%)} $\uparrow$} \\",
        r"  \cmidrule(lr){3-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"  \textbf{Encoder $\varphi$} & \textbf{$d$} & \textbf{$\psi$ joint acc. (\%)} $\uparrow$"
        r" & $\psi$ & $\psi_{\mathrm{eval}}$ & $\psi$ & $\psi_{\mathrm{eval}}$ \\",
    ]
    previous = None
    for row in rows:
        if row["encoder"] != previous:
            lines.append(r"  \midrule")
        label = row["encoder"] if row["encoder"] != previous else ""
        lines.append(
            "  "
            + " & ".join([label, str(row["dimension"]), *cells(row, best, True)])
            + r" \\"
        )
        previous = row["encoder"]
    lines.extend([r"  \bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def markdown_table(
    rows: list[dict], best: dict[str, float], source: str, sha: str
) -> str:
    leaders = {
        r["encoder_key"]
        for field, value in best.items()
        for r in rows
        if r[field] == value
    }
    if len(leaders) == 1:
        leader = next(row for row in rows if row["encoder_key"] in leaders)
        finding = (
            f"{leader['encoder']} at {leader['dimension']}D has the best mean pool out-of-fold "
            "accuracy, evaluation joint NLL and evaluation joint accuracy among the 17 tested "
            "spaces. It also leads the separately trained evaluation-reader columns."
        )
    else:
        finding = "Different columns favour different encoders; bold entries identify each column's best mean."
    lines = [
        "# Encoder comparison on the extended-DAG corpus",
        "",
        f"Generated from `{source}` (SHA-256 `{sha}`). No model fitting or new evaluation is performed.",
        "",
        finding,
        "",
        "| Encoder | d | Pool OOF joint acc. ψ (%) ↑ | Eval. joint NLL ψ ↓ | "
        "Eval. joint NLL ψ_eval ↓ | Eval. joint acc. ψ (%) ↑ | Eval. joint acc. ψ_eval (%) ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [row["encoder"], str(row["dimension"]), *cells(row, best, False)]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation and differences from the old table",
            "",
            "- All readers have independent heads. ψ_eval is an evaluation role, **not an autoregressive model**.",
            "- ψ is fitted on 2,000 labelled pool units in five folds per seed. Each fold supplies 1,600 units "
            "to Reader.fit: 1,280 for optimization and 320 for internal epoch selection. The best internal "
            "validation checkpoint is restored. There are 25 fold fits per encoder/dimension.",
            "- ψ_eval uses the other 2,000 units (800 pool + 200 development + 1,000 evaluation), with five "
            "folds at seed 0 and the same internal split. A unit's evaluation reader excluded that unit "
            "from both optimization and internal validation. This is out-of-fold evaluation, not a reader "
            "trained entirely outside the evaluation set; it also uses one fold model per unit rather "
            "than ψ's five-model ensemble.",
            "- Joint NLL and accuracy refer to X, T, D, U. The networks additionally learn M, whose "
            "per-head results remain in the source report. Accuracies in the CSV are fractions; "
            "the displayed tables convert them to percentages. NLL uses natural logarithms.",
            "- ± is the sample standard deviation across five ψ seeds, not a confidence interval or "
            "variation across folds. The evaluation reader has one seed, so no seed deviation is reported.",
            "- The saved internal validation losses cover all **five** heads and different within-fold "
            "validation subsets. They are not the same quantity as the four-variable NLL in this table; "
            "they are therefore not relabelled as the old table's validation column.",
            "- Autoregressive readers and LangVAE-FT were not run on this corpus. Null and Bayes-optimal "
            "joint reference rows were not computed in this comparison. None of their old-data numbers "
            "is carried forward.",
            "- These descriptive results concern factual CVs in the original wording, not downstream "
            "counterfactual quality. The source report separately contains held-out wording results. "
            "Best mean does not establish statistically significant superiority; the test comparison "
            "is exploratory, and the pool OOF column provides a training-pool-only diagnostic.",
            "",
            "## Suggested LaTeX caption",
            "",
            "```latex",
            CAPTION,
            "```",
            "",
            "## Regenerate",
            "",
            "```bash",
            "python scripts/encoder_comparison_table.py",
            "```",
            "",
            "The `.tex` file contains only the tabular, ready for the paper's existing table float. "
            "The CSV stores full-precision values, and the metadata records the source and generator hashes.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    report = json.loads(raw)
    rows = load_rows(report)
    best = best_values(rows)
    source = args.report.resolve()
    source = (
        str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source)
    )
    sha = hashlib.sha256(raw).hexdigest()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "encoder_comparison.tex").write_text(latex_table(rows, best, source, sha))
    (output / "encoder_comparison.md").write_text(
        markdown_table(rows, best, source, sha)
    )
    with (output / "encoder_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "source": source,
        "source_sha256": sha,
        "run_created": report["created"],
        "run_git_commit": report["git_commit"],
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "joint_variables": report["settings"]["states"][report["settings"]["main"]],
        "seeds": report["settings"]["seeds"],
        "rows": len(rows),
        "psi_fold_fits": len(rows) * 25,
        "judge_fold_fits": len(rows) * 5,
        "csv_accuracy_units": "fraction",
        "display_accuracy_units": "percent",
        "nll_units": "nats",
        "sd": "sample standard deviation across five psi seeds (ddof=1)",
        "best_by_column": {
            field: [r["encoder_key"] for r in rows if r[field] == value]
            for field, value in best.items()
        },
        "caption_latex": CAPTION,
    }
    (output / "encoder_comparison.metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    print(f"Exported {len(rows)} encoder/dimension rows to {output}")
    print(json.dumps(metadata["best_by_column"], indent=2))


if __name__ == "__main__":
    main()
