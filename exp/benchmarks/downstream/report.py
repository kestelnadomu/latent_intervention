"""Markdown summary: per-group accuracy, predicted-level rates and confusion matrices."""

from __future__ import annotations

from pathlib import Path

import torch

from src.artifact_io import atomic_output


def aggregate(rows: list[dict]) -> list[dict]:
    """Keep p(Z) / p(Z'); sum each h_Z family's per-seed matrices into one row."""
    out, families = [], {}
    for row in rows:
        if "family" not in row:
            out.append(dict(name=row["row"], seeds=None, confusion=row["confusion"]))
            continue
        entry = families.setdefault(
            row["family"],
            dict(name=f"p(h_Z(Z)) {row['family']}", seeds=[], confusion={}),
        )
        entry["seeds"].append(row["seed"])
        for label, groups in row["confusion"].items():
            target = entry["confusion"].setdefault(label, {})
            for group, matrix in groups.items():
                current = torch.tensor(matrix)
                target[group] = (
                    current if group not in target else target[group] + current
                )
    for entry in families.values():
        entry["confusion"] = {
            label: {g: m.tolist() for g, m in groups.items()}
            for label, groups in entry["confusion"].items()
        }
    return out + list(families.values())


def accuracy(matrix) -> float:
    m = torch.tensor(matrix, dtype=torch.float64)
    return float(m.trace() / m.sum()) if m.sum() else float("nan")


def predicted_rates(matrix) -> list[float]:
    m = torch.tensor(matrix, dtype=torch.float64)
    return (m.sum(0) / m.sum()).tolist() if m.sum() else [float("nan")] * len(m)


def _rates(rates) -> str:
    return " / ".join(f"{r:.2f}" for r in rates)


def interpretation(rows: list[dict], n: int) -> list[str]:
    """One-paragraph reading of p(Z) / p(Z') / best edit against the Bayes ceilings."""
    acc = {
        (row["name"], label): accuracy(groups["all"])
        for row in rows
        for label, groups in row["confusion"].items()
    }
    text = acc.get(("Bayes: text content (X, D, U, proxies)", "Y"))
    full = acc.get(("Bayes: all parents (T known)", "Y"))
    none = acc.get(("Bayes: no talent information (X, D, U)", "Y"))
    text_cf = acc.get(("Bayes: text content (X, D, U, proxies)", "Y'"))
    if text is None:
        return []
    se = (text * (1 - text) / n) ** 0.5
    p_z, p_zp = acc[("p(Z)", "Y")], acc[("p(Z')", "Y'")]
    edits = {
        row["name"].removeprefix("p(h_Z(Z)) "): acc[(row["name"], "Y'")]
        for row in rows
        if row["name"].startswith("p(h_Z(Z))")
    }
    lines = [
        f"Ceilings vs Y: {full:.3f} with T known, {text:.3f} from the text content, {none:.3f} without talent information. "
        f"p(Z) reaches {p_z:.3f}, {text - p_z:.3f} below the text ceiling ({(text - p_z) / se:.1f} SE; SE ≈ {se:.3f}) "
        f"and recovers {(p_z - none) / (text - none):.0%} of the gain the proxies can provide over X, D, U alone."
        if text > none
        else f"Ceilings vs Y: {full:.3f} with T known, {text:.3f} from the text content; p(Z) reaches {p_z:.3f}.",
    ]
    if text_cf is not None:
        lines[0] += f" Against Y' the text ceiling is {text_cf:.3f}; p(Z') reaches {p_zp:.3f}"
        if edits:
            best = max(edits, key=edits.get)
            lines[0] += f", the best edit ({best}) {edits[best]:.3f}"
        lines[0] += "."
    return lines


def render(root: str | Path, payload: dict) -> None:
    lines = [
        "# Downstream-predictor benchmark",
        "",
        f"h_Z run `{payload['hz_run_id']}`; intervention do({', '.join(f'{k}={v}' for k, v in payload['intervention'].items())}). "
        "One MLP p: Z -> Y per latent dimension, fit on the h_Z fit split and early-stopped on its validation split; "
        "all numbers are on the official test split.",
        "",
        "- `p(Z)`: p on factual latents, scored against factual Y.",
        "- `p(Z')`: p on the encoded counterfactual texts (reference for a perfect h_Z).",
        "- `p(h_Z(Z)) <family>`: p on h_Z edits; stochastic families average p's class probabilities "
        f"over {payload['evaluation']['evaluation_samples']} draws. Matrices are summed over the final seeds.",
        "- Groups are the factual group column; confusion matrices have rows = true Y, columns = predicted Y.",
        "- Predicted-level rates P(Ŷ=k | group) do not depend on the label; they are the multi-class analogue of statistical parity.",
        "",
        "## Reading the accuracies",
        "",
        "Y is not a deterministic function of the latents, so 1.0 is not the reference. "
        "Y = clip_round(0.15·X + 0.40·T + 0.20·D + 0.20·U + ε) with ε ~ N(0, 0.3): "
        "the noise moves a unit across a rounding boundary often enough that even the full parents do not determine Y.",
        "",
        "- `Bayes: …` rows are the exact Bayes-optimal predictors under the known SCM, evaluated on the same test units "
        "(argmax of P(Y | information set)). On factual states they predict Y; on counterfactual states they predict Y'.",
        "- **all parents (T known)** is the irreducible-noise ceiling; no predictor can beat it in expectation.",
        "- **text content (X, D, U, proxies)** is the ceiling for any reader of the CV: T is hidden but inferable "
        "through the proxies P, L, H, A. This is the right reference for p(Z), and for p(h_Z(Z)) vs Y'.",
        "- **no talent information (X, D, U)** shows how much of the accuracy comes from talent: "
        "a predictor between this row and the text ceiling uses some, but not all, of the proxy evidence.",
        "- Sampling error: with n test units an accuracy near 0.75 has a standard error of about sqrt(0.19/n) "
        "(≈ 0.014 overall for n = 1000, ≈ 0.027 for a group of 250). Small gaps to the ceiling are within noise.",
        "- p(h_Z(Z)) vs factual Y is expected to fall: the intervention changes the outcome, so even the true "
        "counterfactual latents (`p(Z')`) score poorly against Y. Compare edited rows against Y' and the CF ceiling instead.",
        "- The Bayes rows' predicted-level rates (factual states) show the disparity the outcome itself carries: "
        "even the optimal decision rule differs strongly by X, so p(Z)'s gap is not a predictor artefact.",
        "",
    ]
    for dimension, block in payload["dimensions"].items():
        rows = aggregate(block["rows"])
        groups = list(rows[0]["confusion"]["Y"])
        predictor = block["predictor"]
        lines += [
            f"## {block['encoder']} ({block['test_units']} test units)",
            "",
            f"Predictor: best epoch {predictor['best_epoch']}, validation CE {predictor['best_validation_ce']:.4f}.",
            "",
        ]
        if block.get("missing_fits"):
            lines += [
                f"**{len(block['missing_fits'])} final h_Z fits were not scored** (checkpoint not available locally): "
                + ", ".join(f"{Path(m['case']).name} seed {m['seed']}" for m in block["missing_fits"])
                + ".",
                "",
            ]
        lines += [
            "### Accuracy",
            "",
            "| Row | Label | " + " | ".join(groups) + " |",
            "|---|---|" + "---:|" * len(groups),
        ]
        for row in rows:
            for label, by_group in row["confusion"].items():
                lines.append(
                    f"| {row['name']} | {label} | "
                    + " | ".join(f"{accuracy(by_group[g]):.3f}" for g in groups)
                    + " |"
                )
        lines += ["", *interpretation(rows, block["test_units"])]
        lines += [
            "",
            "### Predicted-level rates P(Ŷ=0 / 1 / 2 | group)",
            "",
            "| Row | " + " | ".join(groups) + " | max gap per level |",
            "|---|" + "---|" * (len(groups) + 1),
        ]
        for row in rows:
            by_group = row["confusion"]["Y"]
            rates = {g: predicted_rates(by_group[g]) for g in groups}
            per_group = [rates[g] for g in groups if g != "all"]
            gaps = [max(level) - min(level) for level in zip(*per_group)]
            lines.append(
                f"| {row['name']} | "
                + " | ".join(_rates(rates[g]) for g in groups)
                + f" | {_rates(gaps)} |"
            )
        lines += ["", "### Confusion matrices", ""]
        for row in rows:
            for label, by_group in row["confusion"].items():
                n = len(by_group["all"])
                lines += [
                    f"**{row['name']} vs {label}**"
                    + (f" (seeds {row['seeds']})" if row["seeds"] else ""),
                    "",
                    "| Group | True | " + " | ".join(f"Ŷ={k}" for k in range(n)) + " |",
                    "|---|---|" + "---:|" * n,
                ]
                for g in groups:
                    for true, counts in enumerate(by_group[g]):
                        lines.append(
                            f"| {g if true == 0 else ''} | {label}={true} | "
                            + " | ".join(str(c) for c in counts)
                            + " |"
                        )
                lines.append("")
    with atomic_output(Path(root) / "summary.md") as temporary:
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
