"""Resumable fit/validation pilot for the 18 EmbeddingGemma h_Z cases.

The pilot never scores official-test targets or publishes deployment checkpoints.
It compares pre-additive noise magnitudes before the full benchmark is frozen.
"""

from __future__ import annotations

import argparse
import fcntl
import math
import multiprocessing
import os
import statistics
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from src import pipeline
from src.artifact_io import atomic_output, write_json
from src.config import CONFIG_PATH
from src.hz.hz_benchmark import (
    freeze_preparation,
    job_id,
    now,
    preflight,
    run_jobs,
)
from src.hz.hz_benchmark_matrix import (
    PROTOCOL,
    ROOT,
    assert_unchanged,
    case_name,
    digest,
    discover,
    read_settings,
    run_root,
)
from src.hz.hz_training import decoder_for, job_paths, load_model, model_sizes, verify_fit
from src.hz.flow_intervention import sample_counterfactual
from src.latent_intervention import consistency_target
from src.pair_encoding import load_latent_artifact
from src.symbolic_intervention import load_symbolic_kernel


def pilot_config(settings):
    value = settings["pilot"]
    for key in (
        "seed",
        "width_multiplier",
        "max_epochs",
        "dist_pretrain_epochs",
        "patience",
    ):
        if type(value[key]) is not int or value[key] < (
            0 if key == "dist_pretrain_epochs" else 1
        ):
            raise ValueError(f"pilot.{key} must be a valid integer")
    if value["seed"] not in settings["final_seeds"]:
        raise ValueError("pilot.seed must be a declared benchmark seed")
    if value["width_multiplier"] not in settings["width_multipliers"]:
        raise ValueError("pilot width must be in the full benchmark grid")
    if not 0 < value["dist_pretrain_epochs"] < value["max_epochs"]:
        raise ValueError("pilot must include both dist pretraining and joint training")
    lr = value["learning_rate"]
    if type(lr) not in (int, float) or not math.isfinite(lr) or lr < settings["min_lr"]:
        raise ValueError("pilot.learning_rate must be finite and at least min_lr")
    levels = value["noise_rms"]
    if (
        not isinstance(levels, list)
        or not levels
        or len(levels) != len(set(levels))
        or any(
            type(x) not in (int, float) or not math.isfinite(x) or x <= 0
            for x in levels
        )
    ):
        raise ValueError("pilot.noise_rms must contain distinct positive finite values")
    if type(value["compare_legacy_noise_std_one"]) is not bool:
        raise ValueError("pilot.compare_legacy_noise_std_one must be Boolean")
    run_root(value["run_id"])
    return value


def build_plan(config_path=CONFIG_PATH, protocol_path=PROTOCOL):
    plan = discover(config_path, protocol_path)
    protocol = pilot_config(plan["settings"])
    plan["settings"] = dict(
        plan["settings"],
        max_epochs=protocol["max_epochs"],
        dist_pretrain_epochs=protocol["dist_pretrain_epochs"],
        patience=protocol["patience"],
    )
    plan["pilot_protocol"] = protocol
    plan["model_sizes"] = model_sizes(plan)
    return plan


def jobs_for_pilot(plan):
    protocol = plan["pilot_protocol"]
    base = dict(
        lr=protocol["learning_rate"], width_multiplier=protocol["width_multiplier"]
    )
    jobs = []
    ordered = sorted(
        plan["cases"], key=lambda case: (case["variant"] != "state_flow", case["case"])
    )
    for case in ordered:
        choices = [("reference", base)]
        if case["variant"] == "pre_additive":
            choices = [
                (f"rms-{str(rms).replace('.', 'p')}", dict(base, noise_rms=rms))
                for rms in protocol["noise_rms"]
            ]
            if protocol["compare_legacy_noise_std_one"]:
                choices.append(
                    (
                        "legacy-std-one",
                        dict(base, noise_rms=math.sqrt(case["dimension"])),
                    )
                )
        for label, candidate in choices:
            teacher = (
                [f"{case_name(case['dimension'], 'state_flow')}/pilot/reference"]
                if case["variant"] == "distilled_flow"
                else []
            )
            jobs.append(
                dict(
                    id=job_id(case, "pilot", label),
                    case=case,
                    stage="pilot",
                    label=label,
                    seed=protocol["seed"],
                    candidate=candidate,
                    dependencies=teacher,
                )
            )
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("pilot job IDs are not unique")
    return jobs


@torch.no_grad()
def noise_diagnostics(plan, reports):
    """Use fit/validation latents only; preserve every candidate's native score."""
    diagnostics = {}
    for dimension in plan["settings"]["dimensions"]:
        case = next(
            c
            for c in plan["cases"]
            if c["dimension"] == dimension and c["variant"] == "pre_additive"
        )
        artifact = load_latent_artifact(case["config"])
        indices = pipeline._indices_for_ids(
            artifact.ids, plan["split_ids"]["validation"]
        )
        validation = artifact.z[indices]
        decoder = decoder_for(case, artifact)
        h_s = load_symbolic_kernel(case["config"]["sim_config"])
        no_edit = 0.0
        batch_size = plan["settings"]["evaluation_batch_size"]
        for start in range(0, len(validation), batch_size):
            batch = validation[start : start + batch_size]
            target = consistency_target(decoder, h_s, batch, plan["intervention"])
            log_q = decoder.log_joint(batch)
            no_edit += float(
                (target * (target.clamp_min(1e-30).log() - log_q)).sum(-1).sum()
            )
        no_edit /= len(validation)
        for report in reports:
            if report["case"] != case["case"]:
                continue
            verify_fit(report)
            model = load_model(report["checkpoint"], report["signature"])
            sample = validation[: min(64, len(validation))]
            draws = sample_counterfactual(
                model,
                sample,
                plan["intervention"],
                n_samples=16,
                generator=torch.Generator().manual_seed(20260930),
            )
            key = report["checkpoint_sha256"]
            diagnostics[key] = dict(
                dimension=dimension,
                noise_rms=report["candidate"]["noise_rms"],
                noise_std=report["model_kwargs"]["noise_std"],
                no_edit_validation_kl=no_edit,
                mean_shift_l2=float((draws.mean(0) - sample).norm(dim=-1).mean()),
                sample_spread_l2=float(
                    draws.std(dim=0, unbiased=False).norm(dim=-1).mean()
                ),
                sample_units=len(sample),
                sample_draws=len(draws),
            )
    return diagnostics


def analyze(root, plan, completed):
    reports = sorted(
        completed.values(),
        key=lambda row: (row["case"], row["candidate"].get("noise_rms", 0)),
    )
    expected = jobs_for_pilot(plan)
    if len(reports) != len(expected) or set(completed) != {
        job["id"] for job in expected
    }:
        raise ValueError("pilot cannot be summarized before every fit completes")
    rows = []
    labels = {
        (job["case"]["case"], digest(job["candidate"])): job["label"]
        for job in expected
    }
    for report in reports:
        verify_fit(report)
        history = report["training"]["history"]
        eligible = [row for row in history if row.get("selection_eligible", True)]
        if not eligible:
            raise ValueError("pilot fit has no eligible validation epochs")
        validation_key = (
            "validation_mse"
            if report["variant"] == "oracle_regression"
            else "validation"
        )
        train_first = eligible[0].get("training", {"mse": eligible[0].get("train_mse")})
        train_last = eligible[-1].get(
            "training", {"mse": eligible[-1].get("train_mse")}
        )
        rows.append(
            dict(
                case=report["case"],
                variant=report["variant"],
                dimension=int(report["case"].split("/")[0].rsplit("_", 1)[1]),
                candidate=report["candidate"],
                validation_metric=report["validation_metric"],
                first_validation=eligible[0][validation_key],
                best_validation=report["training"]["best_validation"],
                last_validation=eligible[-1][validation_key],
                best_epoch=report["training"]["best_epoch"],
                epochs_run=report["training"]["epochs_run"],
                phases=list(
                    dict.fromkeys(row.get("phase", "train") for row in history)
                ),
                first_training=train_first,
                last_training=train_last,
                seconds=report["seconds"],
                checkpoint_sha256=report["checkpoint_sha256"],
                training_report=str(
                    job_paths(
                        root,
                        next(c for c in plan["cases"] if c["case"] == report["case"]),
                        "pilot",
                        labels[report["case"], digest(report["candidate"])],
                    )[0]
                    / "training.json"
                ),
                test_targets_used=report["test_targets_used"],
            )
        )
    if any(row["test_targets_used"] for row in rows):
        raise ValueError("pilot may not use official-test targets")
    noise = noise_diagnostics(plan, reports)
    selected = {}
    for dimension in plan["settings"]["dimensions"]:
        group = [
            r
            for r in rows
            if r["dimension"] == dimension and r["variant"] == "pre_additive"
        ]
        choice = min(
            group,
            key=lambda row: (row["best_validation"], row["candidate"]["noise_rms"]),
        )
        selected[str(dimension)] = dict(
            noise_rms=choice["candidate"]["noise_rms"],
            noise_std=noise[choice["checkpoint_sha256"]]["noise_std"],
            validation_kl=choice["best_validation"],
            checkpoint_sha256=choice["checkpoint_sha256"],
        )
    return dict(
        format="hz_pilot_v1",
        plan_sha256=digest(plan),
        official_test_scoring=False,
        input_splits={k: len(plan["split_ids"][k]) for k in ("fit", "validation")},
        completed_fits=len(rows),
        total_fit_seconds=sum(row["seconds"] for row in rows),
        median_fit_seconds=statistics.median(row["seconds"] for row in rows),
        rows=rows,
        noise_diagnostics=noise,
        selected_noise_rms=selected,
    )


def render(root, plan, evaluations, status):
    """Write the pilot protocol and its training curves without test metrics."""
    result = evaluations if isinstance(evaluations, dict) else None
    protocol = plan["pilot_protocol"]
    lines = [
        "# h_Z training pilot",
        "",
        f"Status: {status}.",
        "",
        f"Fit/validation units: {len(plan['split_ids']['fit'])}/{len(plan['split_ids']['validation'])}; "
        f"{len(jobs_for_pilot(plan))} fits, one seed {protocol['seed']}, "
        f"{plan['settings']['max_epochs']} epoch ceiling ({plan['settings']['dist_pretrain_epochs']} "
        "pretraining for dist). No official-test scoring or deployment selection.",
        "",
        "All nine families run at both dimensions. The separate pre-additive candidates "
        "use per-coordinate noise_std = noise_rms / sqrt(latent_dim), with the original "
        "noise_std=1 as a diagnostic. Noise levels are compared by that family's "
        "native validation semantic KL only. Exact inputs, source hashes and constructor "
        "settings are in `plan.json`; epoch histories are in `fits/.../pilot/.../training.json`.",
    ]
    if result:
        lines += [
            "",
            "## Noise calibration",
            "",
            "| Dimension | RMS noise | Per-coordinate std | Best validation KL | No-edit KL | Shift L2 | Sample spread L2 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in result["rows"]:
            if row["variant"] != "pre_additive":
                continue
            diagnostic = result["noise_diagnostics"][row["checkpoint_sha256"]]
            lines.append(
                f"| {row['dimension']} | {row['candidate']['noise_rms']:.3f} "
                f"| {diagnostic['noise_std']:.6f} | {row['best_validation']:.6f} "
                f"| {diagnostic['no_edit_validation_kl']:.6f} "
                f"| {diagnostic['mean_shift_l2']:.4f} "
                f"| {diagnostic['sample_spread_l2']:.4f} |"
            )
        lines += [
            "",
            "Selected RMS noise by minimum pilot validation KL: "
            + "; ".join(
                f"{dimension}D = {choice['noise_rms']:.4g}"
                for dimension, choice in result["selected_noise_rms"].items()
            )
            + ". This is a pilot choice at one learning rate and size tier, "
            "not an estimate of final test performance.",
            "",
            "## Training diagnostics",
            "",
            "The first/best/last validation values below use each family's own criterion; "
            "do not rank unlike criteria across families. Dist values start with its first "
            "joint epoch. Full epoch-level loss components and learning rates are in the saved training reports.",
            "",
            "| Case | RMS noise | Epochs | Phases | Validation first / best / last | Best epoch | Final train loss components | Seconds |",
            "|---|---:|---:|---|---|---:|---|---:|",
        ]
        for row in result["rows"]:
            components = ", ".join(
                f"{key}={value:.4g}"
                for key, value in row["last_training"].items()
                if type(value) in (int, float)
            )
            noise = row["candidate"].get("noise_rms")
            lines.append(
                f"| {row['case']} | {noise if noise is not None else '—'} "
                f"| {row['epochs_run']} | {', '.join(row['phases'])} "
                f"| {row['first_validation']:.5g} / {row['best_validation']:.5g} / {row['last_validation']:.5g} "
                f"| {row['best_epoch']} | {components} | {row['seconds']:.1f} |"
            )
        lines += [
            "",
            f"Aggregate fitting time: {result['total_fit_seconds']/3600:.2f} worker-hours "
            f"across {result['completed_fits']} fits; median {result['median_fit_seconds']:.1f} seconds/fit. "
            "This pilot establishes feasibility and scale; its short learning curves "
            "cannot certify convergence under the full 500-epoch ceiling.",
        ]
    lines.append("")
    with atomic_output(Path(root) / "summary.md") as temporary:
        temporary.write_text("\n".join(lines), encoding="utf-8")


def prepare(config_path=CONFIG_PATH, protocol_path=PROTOCOL):
    torch.set_num_threads(1)
    plan = build_plan(config_path, protocol_path)
    root = run_root(plan["pilot_protocol"]["run_id"])
    ready = freeze_preparation(root, plan, config_path, protocol_path, render)
    print(
        f"Prepared {len(jobs_for_pilot(plan))} fit/validation pilot jobs; targets ready: {ready['ready']}. {root}",
        flush=True,
    )
    return root, plan


def run(config_path=CONFIG_PATH, protocol_path=PROTOCOL):
    root = run_root(pilot_config(read_settings(protocol_path))["run_id"])
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / ".benchmark.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root, plan = prepare(config_path, protocol_path)
        oracle_inputs = preflight(root, plan, freeze=True)
        if plan["settings"]["workers"] * plan["settings"]["threads_per_worker"] > len(
            os.sched_getaffinity(0)
        ):
            raise ValueError("pilot worker/thread budget exceeds available CPUs")
        try:
            with ProcessPoolExecutor(
                max_workers=plan["settings"]["workers"],
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor:
                reports = run_jobs(
                    root, plan, jobs_for_pilot(plan), oracle_inputs, executor
                )
            assert_unchanged(plan)
            result = analyze(root, plan, reports)
            write_json(root / "results.json", result)
            render(root, plan, result, "complete")
            write_json(
                root / "status.json",
                dict(
                    status="complete",
                    training_started=True,
                    fits=result["completed_fits"],
                    updated_at=now(),
                ),
            )
            print(
                f"Pilot complete: {result['completed_fits']} fit/validation runs. {root}",
                flush=True,
            )
        except BaseException:
            write_json(
                root / "status.json",
                dict(
                    status="failed",
                    training_started=True,
                    updated_at=now(),
                    traceback=traceback.format_exc(),
                ),
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--prepare", action="store_true")
    choice.add_argument("--check-ready", action="store_true")
    choice.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.run:
        run(args.config, args.protocol)
    else:
        root, plan = prepare(args.config, args.protocol)
        if args.check_ready:
            preflight(root, plan)


if __name__ == "__main__":
    main()
