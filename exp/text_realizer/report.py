"""Compact local reports and fixed-ID human-review sheets."""

from __future__ import annotations

from pathlib import Path

from src.artifact_io import atomic_output


def write_text(path: Path, text: str) -> None:
    with atomic_output(path) as temporary:
        temporary.write_text(text, encoding="utf-8")


def examples(path: Path, part: dict, result: dict, count=20) -> None:
    references = dict(zip(part["ids"], part["texts"], strict=True))
    rows = sorted(result["conditions"]["correct"]["rows"], key=lambda row: row["id"])[
        :count
    ]
    text = "# Fixed-ID factual reconstruction review\n\n"
    text += (
        "Review occupation, country, degree/tier, project count, learning/problem-solving/advice episodes, "
        "persona details, invented facts, and readability. Score each as preserved / missing / contradicted / unclear. "
        "These examples are not selected for quality and must not be used to retune the frozen model.\n\n"
    )
    for row in rows:
        text += f"## ID {row['id']}\n\n### Reference\n\n{references[row['id']]}\n\n"
        text += f"### Reconstruction\n\n{row['text']}\n\n"
    write_text(path, text)


def campaign_summary(path: Path, selection: dict, audit: dict | None = None) -> None:
    lines = [
        "# EmbeddingGemma text realization",
        "",
        "No counterfactual data were used for development.",
        "",
        "| Configuration | Eligible | Character similarity | NLL gap | Character gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in selection["candidates"]:
        lines.append(
            f"| {row['variant']} | {row['eligible']} | {row['character_similarity']:.4f} | "
            f"{row['nll_gap']:.4f} | {row['character_gap']:.4f} |"
        )
    lines += [
        "",
        f"Selected configuration: {selection['selected'] or 'none qualified'}.",
        "Deployment seed: 42. Seeds 43 and 44 are repetitions, not selection candidates.",
    ]
    if audit:
        lines += [
            "",
            f"Audit improvement criterion passed: {audit['improvement_passed']}.",
            f"Paired retrieval comparison: {audit['retrieval_comparison']}.",
            "Exact inversion and counterfactual faithfulness are not guaranteed.",
        ]
    write_text(path, "\n".join(lines) + "\n")
