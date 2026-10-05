"""Downstream-predictor benchmark: confusion matrices of p(Z), p(Z') and p(h_Z(Z)).

Consumes a finished h_Z benchmark run read-only (its plan, frozen split and final
fits). Per latent dimension, one MLP p: Z -> Y is fit on the h_Z run's fit IDs and
early-stopped on its validation IDs; every row is scored on the official test IDs
only, against factual Y and (for edited/counterfactual latents) counterfactual Y'.
Per-fit scores are cached and verified, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
import yaml

from src import pipeline
from src.artifact_io import sha256_file, write_json
from src.downstream import (
    OutcomePredictor,
    grouped_confusion,
    load_outcome_predictor,
    save_outcome_predictor,
    train_outcome_predictor,
)
from src.pair_encoding import load_latent_artifact
from src.schema import load_object, load_schema
from src.symbolic_intervention import load_symbolic_kernel
from exp.benchmarks.downstream.report import render
from exp.benchmarks.latent_intervention.evaluation import draw
from exp.benchmarks.latent_intervention.matrix import ROOT, digest, run_root
from exp.benchmarks.latent_intervention.training import decoder_for, load_model

PROTOCOL = ROOT / "configs/downstream_benchmark.yaml"
REPORTS = ROOT / "reports/talent/downstream_benchmarks"
DEFAULT_RUN = "downstream-gemma-v1"


def output_root(run_id: str) -> Path:
    # Same run-ID validation as the h_Z benchmark.
    return REPORTS / run_root(run_id).name


def read_protocol(path: str | Path = PROTOCOL) -> dict:
    protocol = yaml.safe_load(Path(path).read_text())
    for name in ("hz_run_id", "group_column", "seed", "predictor"):
        if name not in protocol:
            raise ValueError(f"downstream protocol is missing {name!r}")
    return protocol


def load_hz_run(hz_run_id: str) -> tuple[dict, dict]:
    """Return the h_Z plan and results; refuse incomplete or inconsistent runs."""
    root = run_root(hz_run_id)
    if not (root / "results.json").exists():
        raise ValueError(f"h_Z run is not complete (no results.json): {root}")
    plan = json.loads((root / "plan.json").read_text())
    results = json.loads((root / "results.json").read_text())
    if results["plan_sha256"] != digest(plan):
        raise ValueError("h_Z results do not belong to the saved plan")
    return plan, results


def eval_settings(protocol: dict, plan: dict) -> dict:
    return {
        key: plan["settings"][key] if protocol.get(key) is None else protocol[key]
        for key in ("evaluation_samples", "evaluation_seed", "evaluation_batch_size")
    }


def dimension_data(case: dict, plan: dict, group_column: str) -> dict:
    """Latents, outcome labels and factual groups, aligned to the frozen split."""
    config = case["config"]
    artifact = load_latent_artifact(config)
    if artifact.test_ids != plan["split_ids"]["official_test"]:
        raise ValueError("official test order changed since the h_Z run")
    columns, outcome = load_schema(config["sim_config"])
    group = {c.name: c for c in columns}.get(group_column)
    if group is None:
        raise ValueError(f"group column {group_column!r} is not in the schema")
    factual = pipeline._aligned_targets(
        config["paths"]["sim_factual"], artifact.ids, [outcome, group]
    )
    counterfactual = pipeline._aligned_targets(
        config["paths"]["sim_counterfactual"], artifact.ids, [outcome]
    )
    idx = {
        split: pipeline._indices_for_ids(artifact.ids, ids)
        for split, ids in plan["split_ids"].items()
    }
    test = idx["official_test"]
    return dict(
        artifact=artifact,
        outcome=outcome,
        group=group,
        z=artifact.z,
        z_test=artifact.z[test],
        z_prime=artifact.z_prime,  # aligned to artifact.test_ids
        y=factual[outcome.name],
        idx=idx,
        labels={
            "Y": factual[outcome.name][test],
            "Y'": counterfactual[outcome.name][test],
        },
        groups=factual[group.name][test],
    )


def fit_predictor(root: Path, protocol: dict, case: dict, data: dict) -> tuple:
    """Train (or reload a verified) p for one latent dimension."""
    settings = protocol["predictor"]
    signature = digest(
        dict(
            predictor=settings,
            seed=protocol["seed"],
            latents=data["artifact"].artifact_sha256,
            sim_factual=sha256_file(case["config"]["paths"]["sim_factual"]),
            fit=[int(i) for i in data["idx"]["fit"]],
            validation=[int(i) for i in data["idx"]["validation"]],
        )
    )
    encoder = Path(case["case"]).parts[0]
    checkpoint = (
        ROOT / "models/talent" / encoder / "downstream" / root.name / "outcome_predictor.pt"
    )
    report_path = root / "predictors" / f"{encoder}.json"
    if checkpoint.exists() and report_path.exists():
        report = json.loads(report_path.read_text())
        if (
            report["signature"] == signature
            and sha256_file(checkpoint) == report["checkpoint_sha256"]
        ):
            return load_outcome_predictor(checkpoint), report
        raise ValueError(f"saved predictor has stale inputs: {checkpoint}; use a new run ID")
    fit, val = data["idx"]["fit"], data["idx"]["validation"]
    torch.manual_seed(protocol["seed"])
    model = OutcomePredictor(
        latent_dim=data["z"].shape[1],
        n_classes=data["outcome"].n_categories,
        hidden_dim=settings["hidden_dim"],
        n_hidden=settings["n_hidden"],
        dropout=settings["dropout"],
    )
    training = train_outcome_predictor(
        model,
        data["z"][fit],
        data["y"][fit],
        data["z"][val],
        data["y"][val],
        lr=settings["lr"],
        batch_size=settings["batch_size"],
        max_epochs=settings["max_epochs"],
        patience=settings["patience"],
        seed=protocol["seed"],
    )
    save_outcome_predictor(model, checkpoint, dict(signature=signature, case=encoder))
    report = dict(
        signature=signature,
        encoder=encoder,
        checkpoint=str(checkpoint.relative_to(ROOT)),
        checkpoint_sha256=sha256_file(checkpoint),
        best_epoch=training["best_epoch"],
        best_validation_ce=training["best_validation_ce"],
        epochs_run=len(training["history"]),
        test_targets_used=False,
    )
    write_json(report_path, report)
    return model, report


@torch.no_grad()
def edited_probabilities(
    predictor, model, variant, z, *, intervention, decoder, h_s, settings
) -> torch.Tensor:
    """E[p(h_Z(z))] over h_Z draws (a single draw for point-mass families)."""
    generator = torch.Generator().manual_seed(settings["evaluation_seed"])
    out = []
    for start in range(0, len(z), settings["evaluation_batch_size"]):
        samples = draw(
            model,
            variant,
            z[start : start + settings["evaluation_batch_size"]],
            intervention,
            settings["evaluation_samples"],
            generator,
            decoder,
            h_s,
        )
        if not torch.isfinite(samples).all():
            raise FloatingPointError("non-finite h_Z samples")
        out.append(predictor.probabilities(samples).mean(0))
    return torch.cat(out)


def score(probabilities: torch.Tensor, data: dict, labels: tuple[str, ...]) -> dict:
    pred = probabilities.argmax(-1)
    return {
        label: grouped_confusion(
            pred,
            data["labels"][label],
            data["groups"],
            data["outcome"].n_categories,
            group_name=data["group"].name,
            n_groups=data["group"].n_categories,
        )
        for label in labels
    }


def available_checkpoints(hz_run_id: str, results: dict) -> dict[str, Path]:
    """checkpoint sha -> a local file with exactly those bytes, or no entry.

    Falls back to the published ``selected/`` copy (deployment seed only) when a
    final fit's own checkpoint was not kept on this machine.
    """
    published_path = run_root(hz_run_id) / "published.json"
    published = (
        json.loads(published_path.read_text()) if published_path.exists() else []
    )
    candidates: dict[str, list[Path]] = {}
    for fit in results["final_fits"]:
        candidates.setdefault(fit["checkpoint_sha256"], []).append(Path(fit["checkpoint"]))
    for entry in published:
        candidates.setdefault(entry["checkpoint_sha256"], []).append(Path(entry["checkpoint"]))
    found = {}
    for sha, paths in candidates.items():
        for path in paths:
            if path.exists() and sha256_file(path) == sha:
                found[sha] = path
                break
    return found


def bayes_rows(case: dict, plan: dict, data: dict, dotted: str) -> list[dict]:
    """Bayes-optimal reference rows on the same test units: factual states -> Y, CF states -> Y'."""
    reference = load_object(dotted)
    test_ids = plan["split_ids"]["official_test"]
    predictions = {}
    for label, key in (("Y", "sim_factual"), ("Y'", "sim_counterfactual")):
        frame = pd.read_csv(case["config"]["paths"][key]).set_index("id").loc[test_ids]
        for name, probs in reference(frame.reset_index(), case["config"]["sim_config"]).items():
            predictions.setdefault(name, {})[label] = torch.as_tensor(probs)
    return [
        dict(
            row=f"Bayes: {name}",
            reference=True,
            confusion={
                label: score(probs, data, (label,))[label]
                for label, probs in by_label.items()
            },
        )
        for name, by_label in predictions.items()
    ]


def score_fit(root, plan, fit, checkpoint, case, predictor, predictor_report, data, decoder, h_s, settings):
    """Score one final h_Z fit, reusing a verified cached record if present."""
    signature = digest(
        dict(
            predictor=predictor_report["checkpoint_sha256"],
            hz_checkpoint=fit["checkpoint_sha256"],
            settings=settings,
            intervention=plan["intervention"],
        )
    )
    path = root / "scores" / fit["case"] / f"seed-{fit['seed']}.json"
    if path.exists():
        record = json.loads(path.read_text())
        if record["signature"] == signature:
            return record
    model = load_model(checkpoint, fit["signature"])
    probabilities = edited_probabilities(
        predictor,
        model,
        case["variant"],
        data["z_test"],
        intervention=plan["intervention"],
        decoder=decoder,
        h_s=h_s,
        settings=settings,
    )
    record = dict(
        signature=signature,
        row="p(h_Z(Z))",
        family=case["variant"],
        seed=fit["seed"],
        case=fit["case"],
        confusion=score(probabilities, data, ("Y", "Y'")),
    )
    write_json(path, record)
    return record


def run(run_id=DEFAULT_RUN, protocol_path=PROTOCOL, hz_run_id=None) -> Path:
    protocol = read_protocol(protocol_path)
    if hz_run_id is not None:
        protocol["hz_run_id"] = hz_run_id
    plan, results = load_hz_run(protocol["hz_run_id"])
    settings = eval_settings(protocol, plan)
    root = output_root(run_id)
    root.mkdir(parents=True, exist_ok=True)
    cases = {case["case"]: case for case in plan["cases"]}
    by_dimension: dict[int, list[dict]] = {}
    for case in plan["cases"]:
        by_dimension.setdefault(case["dimension"], []).append(case)
    h_s = load_symbolic_kernel(plan["cases"][0]["config"]["sim_config"])
    checkpoints = available_checkpoints(protocol["hz_run_id"], results)
    dimensions = {}
    for dimension, dim_cases in sorted(by_dimension.items()):
        reference = dim_cases[0]  # every family shares encoder, latents and g
        data = dimension_data(reference, plan, protocol["group_column"])
        predictor, predictor_report = fit_predictor(root, protocol, reference, data)
        decoder = decoder_for(reference, data["artifact"])
        rows = [
            dict(row="p(Z)", confusion=score(predictor.probabilities(data["z_test"]), data, ("Y",))),
            dict(row="p(Z')", confusion=score(predictor.probabilities(data["z_prime"]), data, ("Y", "Y'"))),
        ]
        if protocol.get("outcome_reference"):
            rows = bayes_rows(reference, plan, data, protocol["outcome_reference"]) + rows
        names = {case["case"] for case in dim_cases}
        fits = sorted(
            (f for f in results["final_fits"] if f["case"] in names),
            key=lambda f: (f["case"], f["seed"]),
        )
        missing = []
        for i, fit in enumerate(fits, 1):
            checkpoint = checkpoints.get(fit["checkpoint_sha256"])
            if checkpoint is None:
                missing.append(dict(case=fit["case"], seed=fit["seed"]))
                print(f"[{dimension}] {i}/{len(fits)} MISSING checkpoint: {fit['case']} seed {fit['seed']}", flush=True)
                continue
            rows.append(
                score_fit(
                    root, plan, fit, checkpoint, cases[fit["case"]], predictor,
                    predictor_report, data, decoder, h_s, settings,
                )
            )
            print(f"[{dimension}] {i}/{len(fits)} scored: {fit['case']} seed {fit['seed']}", flush=True)
        dimensions[str(dimension)] = dict(
            encoder=predictor_report["encoder"],
            test_units=len(data["z_test"]),
            predictor=predictor_report,
            rows=rows,
            missing_fits=missing,
        )
    payload = dict(
        hz_run_id=protocol["hz_run_id"],
        hz_plan_sha256=digest(plan),
        protocol=protocol,
        protocol_sha256=sha256_file(protocol_path),
        evaluation=settings,
        intervention=plan["intervention"],
        confusion_layout="rows = true class, columns = predicted class",
        dimensions=dimensions,
    )
    write_json(root / "results.json", payload)
    render(root, payload)
    print(f"Wrote {root / 'summary.md'}", flush=True)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--protocol", default=PROTOCOL, type=Path)
    parser.add_argument("--hz-run-id", default=None, help="override the protocol's hz_run_id")
    args = parser.parse_args()
    os.chdir(ROOT)
    run(args.run_id, args.protocol, args.hz_run_id)


if __name__ == "__main__":
    main()
