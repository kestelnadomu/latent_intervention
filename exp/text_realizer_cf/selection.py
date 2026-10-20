"""Factual-only generated-text checkpoint selection for both continuations."""

from __future__ import annotations

from statistics import mean

from exp.text_realizer.facts import evaluate as factual_checks
from src.text_realizer.generation import generate
from src.text_realizer.metrics import text_metrics
from src.text_realizer.training import evaluate_nll, tokenize


def macro_accuracy(facts: dict, names: list[str]) -> float:
    values = [
        facts["fields"][name]["accuracy_on_scorable_references"] for name in names
    ]
    if any(value is None for value in values):
        raise ValueError("unscorable factual selection field")
    return mean(values)


def evaluate_selection(model, part: dict, fields: list[str], *, progress=None) -> dict:
    generated = generate(model, part["z"], progress=progress)
    predictions = [row["text"] for row in generated]
    similarities = [
        text_metrics(reference, prediction)["character_similarity"]
        for reference, prediction in zip(part["texts"], predictions, strict=True)
    ]
    facts = factual_checks(part["texts"], predictions)
    macro = macro_accuracy(facts, fields)
    nll = evaluate_nll(model, part["z"], tokenize(model.tokenizer, part["texts"]))[
        "nll"
    ]
    character = mean(similarities)
    return {
        "character_similarity": character,
        "four_field_macro_accuracy": macro,
        "score": (character + macro) / 2,
        "factual_token_nll": nll,
        "field_accuracy": {
            name: facts["fields"][name]["accuracy_on_scorable_references"]
            for name in fields
        },
        "eos_fraction": mean(row["termination"] == "eos" for row in generated),
    }


def selection_key(record: dict) -> tuple[float, float, float, int]:
    return (
        record["score"],
        record["character_similarity"],
        -record["factual_token_nll"],
        -record["step"],
    )
