"""Conservative text-to-text checks, never training targets or a learned judge."""

from __future__ import annotations

import re

import yaml

from .settings import ROOT


def extract(text: str, codebook: dict) -> dict:
    text = text.lower().replace("’", "'")
    countries = []
    for region, names in codebook["choices"]["X"].items():
        for name in names:
            name = name.lower().removeprefix("the ")
            if re.search(r"\b" + re.escape(name) + r"\b", text):
                countries.append((name, int(region)))
    degrees = [
        name
        for name, pattern in {
            "bachelor": r"\bbachelor(?:'s|s)?\b",
            "master": r"\bmaster(?:'s|s)?\b",
            "doctorate": r"\b(?:doctorate|doctoral|ph\.?d\.?)\b",
        }.items()
        if re.search(pattern, text)
    ]
    tiers = [
        name
        for name, pattern in {
            "unranked": r"\b(?:unranked|not listed in (?:the )?(?:major )?global rankings)\b",
            "top500": r"\b(?:global\s+)?top\s*500\b",
            "top100": r"\b(?:global\s+)?top\s*100\b",
        }.items()
        if re.search(pattern, text)
    ]
    words = {
        word: i
        for i, word in enumerate(
            (
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
            )
        )
    }
    matches = re.findall(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine)\s+(?:completed\s+)?projects?\b",
        text,
    )
    counts = {int(value) if value.isdigit() else words[value] for value in matches}
    return {
        "country": countries[0][0] if len(countries) == 1 else None,
        "region": countries[0][1] if len(countries) == 1 else None,
        "degree": degrees[0] if len(degrees) == 1 else None,
        "university_tier": tiers[0] if len(tiers) == 1 else None,
        "project_count": next(iter(counts)) if len(counts) == 1 else None,
    }


def evaluate(references: list[str], predictions: list[str]) -> dict:
    codebook = yaml.safe_load((ROOT / "exp/sim/codebook.yaml").read_text())
    expected = [extract(text, codebook) for text in references]
    observed = [extract(text, codebook) for text in predictions]
    report = {}
    for key in expected[0]:
        valid = [i for i, row in enumerate(expected) if row[key] is not None]
        correct = sum(observed[i][key] == expected[i][key] for i in valid)
        report[key] = {
            "reference_scorable": len(valid),
            "total": len(references),
            "reference_coverage": len(valid) / len(references),
            "accuracy_on_scorable_references": correct / len(valid) if valid else None,
            "unparseable_predictions": sum(observed[i][key] is None for i in valid),
        }
    return {
        "method": "conservative rules; ambiguous reference excluded, missing prediction incorrect",
        "fields": report,
        "reference": expected,
        "prediction": observed,
    }
