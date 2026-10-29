"""Export the completed latent_lab encoder comparison without loading or fitting models.

Writes the paper table ``overleaf/tables/encoder_decoder.tex`` and figure
``overleaf/figures/encoder_comparison.pdf`` (floats and captions in
``overleaf/appendix/encoder.tex``) and a Markdown/CSV/metadata record of them:

    uv run --with matplotlib==3.10.3 python scripts/encoder_comparison_table.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports/latent_lab/run-20261027-225936.json"
DEFAULT_OUTPUT = ROOT / "reports/latent_lab/encoder-comparison-20261027"
PAPER_TABLE = ROOT / "overleaf/tables/encoder_decoder.tex"
PAPER_FIGURE = ROOT / "overleaf/figures/encoder_comparison.pdf"
# Colour (categorical slots 1-4, checked with the dataviz palette validator) and marker per
# encoder, in legend order: the order of the curves at their right end.
STYLE = {
    "embeddinggemma": ("#2A78D6", "o"),
    "qwen3": ("#EB6834", "s"),
    "nomic": ("#1BAF7A", "^"),
    "langvae": ("#EDA100", "D"),
}
T95 = 2.776  # Student-t 0.975 quantile with 4 df: 95 % interval over the five psi seeds
SIM = ROOT / "data/sim_talent_motivation"
JOINT = ["X", "T", "D", "U"]
VERBALIZED = ["X", "D", "U", "P", "L", "H", "A"]  # what the CVs show; T only through its proxies
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
    ("eval_joint_nll", "eval_joint_nll_sd", 1, 3, "min"),
    ("judge_oof_joint_nll", None, 1, 3, "min"),
    ("eval_joint_acc", "eval_joint_acc_sd", 100, 1, "max"),
    ("judge_oof_joint_acc", None, 100, 1, "max"),
)


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


def lf_sha256(data: bytes) -> str:
    """Hash with LF line endings, so a CRLF checkout (core.autocrlf) gives the same digest."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def references() -> dict[str, dict[str, float]]:
    """Latent-free rows on the evaluation units: the null model and the Bayes-optimal reader."""
    from exp.sim_talent.talent_sfm import _talent_log_posterior
    from exp.sim_talent_motivation.talent_motivation import T_PRIOR

    factual = pd.read_csv(SIM / "sim_data_factual.csv", index_col="id")
    partition = pd.read_csv(SIM / "partition.csv", index_col="id")
    pool, evaluation = (
        factual.loc[partition.index[partition["partition"] == name]]
        for name in ("pool", "eval")
    )
    cards = {c: int(factual[c].max()) + 1 for c in factual}
    # null model: the Laplace-smoothed empirical joint of s on the pool, ignoring z
    counts = pool.groupby(JOINT).size()
    total = counts.sum() + np.prod([cards[c] for c in JOINT])
    cells = list(evaluation[JOINT].itertuples(index=False, name=None))
    null = {
        "joint_nll": -np.mean([np.log((counts.get(c, 0) + 1) / total) for c in cells]),
        "joint_acc": np.mean([c == counts.idxmax() for c in cells]),
    }
    # Bayes-optimal: X, D and U are read exactly, T by its exact posterior under the known SCM
    log_post = _talent_log_posterior(evaluation, VERBALIZED, cards, T_PRIOR)
    log_post -= np.logaddexp.reduce(log_post, axis=1, keepdims=True)
    t = evaluation["T"].to_numpy()
    bayes = {
        "joint_nll": -log_post[np.arange(len(t)), t].mean(),
        "joint_acc": (log_post.argmax(axis=1) == t).mean(),
    }
    return {
        "Null model": {k: float(v) for k, v in null.items()},
        "Bayes-optimal": {k: float(v) for k, v in bayes.items()},
    }


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
            mean += (
                rf"{{\tiny$\pm${spread.removeprefix('0')}}}" if latex else f" ± {spread}"
            )
        result.append(mean)
    return result


def latex_table(
    rows: list[dict], best: dict[str, float], refs: dict, source: str, sha: str
) -> str:
    def bold(text: str) -> str:
        return rf"{{\bfseries\boldmath {text}}}"

    lines = [
        f"% Generated by scripts/encoder_comparison_table.py from {source}",
        f"% (SHA-256 {sha}).",
        "% Do not edit by hand; regenerate instead. The table float and caption live in",
        "% appendix/encoder.tex.",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"  \toprule",
        rf"  & & \multicolumn{{2}}{{c}}{{{bold(r'Joint NLL $\downarrow$')}}}"
        rf" & \multicolumn{{2}}{{c}}{{{bold(r'Joint acc.\ (\%) $\uparrow$')}}} \\",
        r"  \cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        "  "
        + " & ".join(
            bold(text)
            for text in (r"Encoder $\varphi$", "$d$", *2 * [r"$\psi$", r"$\psi_{\mathrm{eval}}$"])
        )
        + r" \\",
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
    lines.append(r"  \midrule")
    for name, ref in refs.items():
        lines.append(
            rf"  {name} & -- & \multicolumn{{2}}{{c}}{{{ref['joint_nll']:.3f}}}"
            rf" & \multicolumn{{2}}{{c}}{{{100 * ref['joint_acc']:.1f}}} \\"
        )
    lines.extend([r"  \bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def figure(rows: list[dict], path: Path) -> None:
    """psi's joint accuracy over d (log scale) per encoder, 95 % seed intervals as bands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # TrueType fonts: ACM rejects the Type 3 fonts matplotlib embeds by default.
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.1, 2.1))
    for family, (color, marker) in STYLE.items():
        series = [r for r in rows if r["encoder_key"].startswith(f"{family}-")]
        d = [r["dimension"] for r in series]
        acc = np.array([100 * r["eval_joint_acc"] for r in series])
        half = np.array([100 * T95 * r["eval_joint_acc_sd"] / math.sqrt(5) for r in series])
        if len(series) > 1:
            ax.fill_between(d, acc - half, acc + half, color=color, alpha=0.2, linewidth=0)
        else:  # a single width: its interval as an error bar
            ax.errorbar(d, acc, yerr=half, fmt="none", ecolor=color, elinewidth=1)
        ax.plot(d, acc, color=color, marker=marker, markersize=5, linewidth=1.5,
                markeredgecolor="white", markeredgewidth=0.6, label=FAMILIES[family][0])
    ticks = [32, 64, 128, 256, 512, 1024]  # 768 sits unlabelled between 512 and 1024
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(0, 100)
    ax.set_xlabel("Latent width $d$ (log scale)")
    ax.set_ylabel(r"Joint accuracy of $\psi$ (%)")
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    # No creation date: the same report gives the same PDF bytes.
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def markdown_table(
    rows: list[dict], best: dict[str, float], refs: dict, source: str, sha: str
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
            f"{leader['encoder']} at {leader['dimension']}D has the best mean evaluation joint "
            "NLL and joint accuracy among the 17 tested spaces, for ψ and for ψ_eval."
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
        "| Encoder | d | Eval. joint NLL ψ ↓ | Eval. joint NLL ψ_eval ↓ | "
        "Eval. joint acc. ψ (%) ↑ | Eval. joint acc. ψ_eval (%) ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [row["encoder"], str(row["dimension"]), *cells(row, best, False)]
            )
            + " |"
        )
    for name, ref in refs.items():
        nll, acc = f"{ref['joint_nll']:.3f}", f"{100 * ref['joint_acc']:.1f}"
        lines.append(f"| {name} | – | {nll} | {nll} | {acc} | {acc} |")
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
            "- Autoregressive readers and LangVAE-FT were not run on this corpus; none of their "
            "old-data numbers is carried forward.",
            "- The reference rows are recomputed on this corpus and need no latents. Null model: the "
            "Laplace-smoothed empirical joint of (X, T, D, U) on the 2,800 pool units. Bayes-optimal: "
            "X, D and U read exactly, T by its exact posterior under the known SCM given X, D, U and "
            "the proxies P, L, H, A.",
            "- These descriptive results concern factual CVs in the original wording, not downstream "
            "counterfactual quality. The source report separately contains held-out wording results. "
            "Best mean does not establish statistically significant superiority; the test comparison "
            "is exploratory.",
            "",
            "## Regenerate",
            "",
            "```bash",
            "python scripts/encoder_comparison_table.py",
            "```",
            "",
            "The LaTeX tabular is written to `overleaf/tables/encoder_decoder.tex`; its float and "
            "caption live in `overleaf/appendix/encoder.tex`. The CSV stores full-precision values "
            "(including the pool out-of-fold joint accuracy of ψ), and the metadata records the "
            "source and generator hashes.",
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
    refs = references()
    source = args.report.resolve()
    source = (
        source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else str(source)
    )
    sha = lf_sha256(raw)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    PAPER_TABLE.write_text(latex_table(rows, best, refs, source, sha), newline="\n")
    figure(rows, PAPER_FIGURE)
    (output / "encoder_comparison.md").write_text(
        markdown_table(rows, best, refs, source, sha), encoding="utf-8", newline="\n"
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
        "generator_sha256": lf_sha256(Path(__file__).read_bytes()),
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
        "references": refs,
    }
    (output / "encoder_comparison.metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    print(f"Exported {len(rows)} encoder/dimension rows to {output}")
    print(json.dumps(metadata["best_by_column"], indent=2))


if __name__ == "__main__":
    main()
