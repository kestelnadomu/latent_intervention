"""Tune-only paired comparison; audit, official test, and h_Z remain outside scope."""

from __future__ import annotations

import difflib
from statistics import mean

import torch

from exp.text_realizer.campaign import make_cycle_encoder
from exp.text_realizer.evaluation import cycle_metrics, group_report
from exp.text_realizer.facts import evaluate as factual_checks
from src.artifact_io import sha256_file, write_json
from src.text_realizer.artifacts import load_text_realizer
from src.text_realizer.generation import generate
from src.text_realizer.metrics import (
    aggregate,
    normalize,
    paired_bootstrap,
    text_metrics,
)
from src.text_realizer.training import evaluate_nll, tokenize

from .data import load_partition, verified_sources
from .model import load_reference
from .protocol import identity, lock, read_json, require_current, status
from .selection import macro_accuracy
from .settings import prepared_path, resolve, run_paths


def _copy_checks(texts: list[str], fit_texts: list[str]) -> list[dict]:
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein

    reference = [normalize(text) for text in fit_texts]
    reference_set = set(reference)
    result = []
    for text in texts:
        value = normalize(text)
        _, score, index = process.extractOne(
            value, reference, scorer=Levenshtein.normalized_similarity
        )
        a, b = value.split(), reference[index].split()
        span = (
            difflib.SequenceMatcher(None, a, b, autojunk=False)
            .find_longest_match(0, len(a), 0, len(b))
            .size
        )
        result.append(
            {
                "exact_fit_copy": value in reference_set,
                "nearest_fit_similarity": score,
                "nearest_fit_longest_shared_words": span,
                "shared_span_ge_20_words": span >= 20,
            }
        )
    return result


def score_world(
    model,
    part: dict,
    fit: dict,
    encoder,
    *,
    groups: dict,
    fit_groups: dict,
    progress=None,
) -> dict:
    targets = tokenize(model.tokenizer, part["texts"])
    nll = evaluate_nll(model, part["z"], targets, progress=progress)["nll"]
    outputs = generate(model, part["z"], progress=progress)
    predictions = [row["text"] for row in outputs]
    copies = _copy_checks(predictions, fit["texts"])
    rows = []
    for key, reference, output, copy in zip(
        part["ids"], part["texts"], outputs, copies, strict=True
    ):
        rows.append(
            {
                "id": key,
                "text": output["text"],
                "tokens": output["tokens"],
                "termination": output["termination"],
                **text_metrics(reference, output["text"]),
                **copy,
                "eos": float(output["termination"] == "eos"),
                "max_length": float(output["termination"] == "max_length"),
            }
        )
    cycle, cycle_rows = cycle_metrics(
        predictions, part["z"], fit, encoder, progress=progress
    )
    for row, extra in zip(rows, cycle_rows, strict=True):
        row.update(extra)
    facts = factual_checks(part["texts"], predictions)
    return {
        "ids": list(part["ids"]),
        "nll": nll,
        "summary": aggregate(rows),
        "cycle": cycle,
        "facts": facts,
        "groups": group_report(rows, groups, fit_groups),
        "output_uniqueness": len(set(map(normalize, predictions))) / len(predictions),
        "rows": rows,
    }


def subset_scores(
    report: dict, references: list[str], mask: list[bool], fields: list[str]
) -> dict:
    rows = [row for row, keep in zip(report["rows"], mask, strict=True) if keep]
    truths = [text for text, keep in zip(references, mask, strict=True) if keep]
    if not rows:
        raise ValueError("empty comparison subset")
    facts = factual_checks(truths, [row["text"] for row in rows])
    return {
        "ids": [row["id"] for row in rows],
        "character_similarity": mean(row["character_similarity"] for row in rows),
        "macro_factual_accuracy": macro_accuracy(facts, fields),
        "facts": {name: facts["fields"][name] for name in fields},
        "max_length_rate": mean(row["max_length"] for row in rows),
        "exact_fit_copy_rate": mean(row["exact_fit_copy"] for row in rows),
        "rows": rows,
    }


def decision(reports: dict, evaluation: dict, settings: dict) -> dict:
    fields = settings["factual_fields"]
    changed = [not value for value in evaluation["is_identity"].tolist()]
    for arm, worlds in reports.items():
        for i, identity_flag in enumerate(evaluation["is_identity"].tolist()):
            if (
                identity_flag
                and worlds["factual"]["rows"][i]["text"]
                != worlds["oracle"]["rows"][i]["text"]
            ):
                raise ValueError(
                    f"{arm} generated different text for identical vectors"
                )
    scores = {
        arm: {
            "oracle_changed": subset_scores(
                worlds["oracle"], evaluation["oracle"]["texts"], changed, fields
            ),
            "factual": subset_scores(
                worlds["factual"],
                evaluation["factual"]["texts"],
                [True] * len(changed),
                fields,
            ),
        }
        for arm, worlds in reports.items()
    }
    primary = scores["oracle"]["oracle_changed"]
    baseline = scores["original"]["oracle_changed"]
    control = scores["factual"]["oracle_changed"]
    intervals = {}
    for other, reference in (("original", baseline), ("factual", control)):
        if primary["ids"] != reference["ids"]:
            raise ValueError("paired counterfactual comparison IDs differ")
        differences = [
            new["character_similarity"] - old["character_similarity"]
            for new, old in zip(primary["rows"], reference["rows"], strict=True)
        ]
        intervals[other] = paired_bootstrap(
            differences,
            samples=settings["bootstrap_samples"],
            seed=settings["bootstrap_seed"],
        )
    factual_new = scores["oracle"]["factual"]
    factual_old = scores["original"]["factual"]
    field_changes = {
        name: primary["facts"][name]["accuracy_on_scorable_references"]
        - baseline["facts"][name]["accuracy_on_scorable_references"]
        for name in fields
    }
    gates = {
        "character_vs_original": intervals["original"]["mean"]
        >= settings["minimum_character_gain"]
        and intervals["original"]["lower_95"] > 0,
        "character_vs_factual_continuation": intervals["factual"]["mean"]
        >= settings["minimum_character_gain"]
        and intervals["factual"]["lower_95"] > 0,
        "counterfactual_macro_gain": primary["macro_factual_accuracy"]
        - baseline["macro_factual_accuracy"]
        >= settings["minimum_macro_factual_gain"],
        "no_counterfactual_field_regression": all(
            change >= -settings["maximum_field_regression"]
            for change in field_changes.values()
        ),
        "factual_character_retention": factual_new["character_similarity"]
        - factual_old["character_similarity"]
        > -settings["maximum_factual_regression"],
        "factual_macro_retention": factual_new["macro_factual_accuracy"]
        - factual_old["macro_factual_accuracy"]
        > -settings["maximum_factual_regression"],
    }
    return {
        "comparison_scope": "development_tune_oracle_supervised_not_hz",
        "changed_units": len(primary["ids"]),
        "identity_units": int(evaluation["is_identity"].sum()),
        "arm_scores": {
            arm: {
                world: {
                    key: value
                    for key, value in score.items()
                    if key not in {"rows", "ids"}
                }
                for world, score in worlds.items()
            }
            for arm, worlds in scores.items()
        },
        "paired_character_intervals": intervals,
        "counterfactual_field_changes_vs_original": field_changes,
        "gates": gates,
        "worth_replication": all(gates.values()),
        "audit_gate": "closed",
        "official_test_gate": "sealed",
        "note": "Tune is exploratory: the original model was already selected on these units. No h_Z, audit, or official-test conclusion.",
    }


def evaluate_pilot(config: dict, run_id: str, *, device: str) -> dict:
    prepared = prepared_path(config)
    _, root = run_paths(config, run_id)
    require_current(root, config, prepared)
    verified_sources(config)
    with lock(root / ".evaluate.lock"):
        if (root / "result.json").exists():
            result = read_json(root / "result.json")
            if result["identity"] != identity(config, prepared):
                raise ValueError("existing pilot result belongs to other inputs")
            return result["decision"]
        checkpoints = {}
        for arm in ("factual", "oracle"):
            models, reports = run_paths(config, run_id, arm)
            require_current(reports, config, prepared)
            trained = read_json(reports / "training.json")
            checkpoint = models / "best.pt"
            if (
                not trained["complete"]
                or not trained["reload_verified"]
                or trained["checkpoint_sha256"] != sha256_file(checkpoint)
            ):
                raise ValueError(f"{arm} must finish and verify before evaluation")
            checkpoints[arm] = checkpoint
        fit = load_partition(prepared, "fit")
        evaluation = load_partition(prepared, "evaluation")
        fit_reference = {
            "ids": fit["factual"]["ids"] + fit["changed"]["ids"],
            "z": torch.cat((fit["factual"]["z"], fit["changed"]["z"])),
            "texts": fit["factual"]["texts"] + fit["changed"]["texts"],
        }
        factual_prepared = resolve(config["inputs"]["factual_prepared"])
        factual_manifest = read_json(factual_prepared / "manifest.json")
        groups_file = factual_prepared / "development_groups.json"
        if (
            sha256_file(groups_file)
            != factual_manifest["files"]["development_groups.json"]
        ):
            raise ValueError("factual group metadata changed")
        all_groups = read_json(groups_file)
        groups = {
            str(key): all_groups["tune"][str(key)]
            for key in evaluation["factual"]["ids"]
        }
        fit_groups = all_groups["fit"]
        encoder = make_cycle_encoder(
            {
                "encoder": read_json(prepared / "manifest.json")["encoder"],
                "paths": config["paths"],
            },
            device,
        )
        reports = {}
        for arm in ("original", "factual", "oracle"):
            model = (
                load_reference(config, device=device)
                if arm == "original"
                else load_text_realizer(
                    checkpoints[arm],
                    device=device,
                    cache_dir=str(resolve(config["paths"]["cache"])),
                )
            )
            worlds = {}
            for world in ("factual", "oracle"):
                status(root, "evaluating", arm=arm, world=world)
                worlds[world] = score_world(
                    model,
                    evaluation[world],
                    fit_reference,
                    encoder,
                    groups=groups,
                    fit_groups=fit_groups,
                    progress=lambda event, done, total: (
                        status(
                            root,
                            "evaluating",
                            arm=arm,
                            world=world,
                            event=event,
                            completed=done,
                            total=total,
                        )
                        if done == total or done % 40 == 0
                        else None
                    ),
                )
            reports[arm] = worlds
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        for arm, worlds in reports.items():
            for world in ("factual", "oracle"):
                if worlds[world]["ids"] != evaluation[world]["ids"]:
                    raise ValueError(f"{arm}/{world} evaluation IDs changed")
        result = decision(reports, evaluation, config["selection"])
        examples = [
            {
                "id": key,
                "reference_cv_prime": reference,
                **{
                    arm: reports[arm]["oracle"]["rows"][i]["text"]
                    for arm in ("original", "factual", "oracle")
                },
            }
            for i, (key, reference) in enumerate(
                zip(
                    evaluation["oracle"]["ids"],
                    evaluation["oracle"]["texts"],
                    strict=True,
                )
            )
            if i < 20
        ]
        write_json(root / "examples.json", examples)
        write_json(
            root / "result.json",
            {
                "identity": identity(config, prepared),
                "checkpoint_sha256": {
                    "original": config["inputs"]["sha256"]["baseline_checkpoint"],
                    **{arm: sha256_file(path) for arm, path in checkpoints.items()},
                },
                "complete": True,
                "decision": result,
                "worlds": reports,
            },
        )
        status(
            root,
            "complete",
            worth_replication=result["worth_replication"],
            audit_gate="closed",
            official_test_gate="sealed",
        )
        return result
