"""Tune/final-fit independent S1-S6 readers. No official-test evaluation command.

Completed fits resume after reload verification. An interrupted fit restarts from
its deterministic seed, matching the established g benchmark's resume policy.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics
import subprocess
from time import perf_counter
import traceback

import torch

from exp.benchmarks.reader.protocol import CONFIG, digest, load_data, read_config
from exp.benchmarks.semantic_decoder.experiment import candidate_settings
from src.artifact_io import atomic_output, copy_atomic, sha256_file, write_json
from src.semantic_decoder.model import (
    SemanticDecoder,
    calibration_metrics,
    joint_nll,
    load_semantic_decoder,
)
from src.semantic_decoder.training import train_semantic_decoder


def now():
    return datetime.now(timezone.utc).isoformat()


def report_root(cfg):
    return Path(cfg["paths"]["reports"]) / cfg["run_id"]


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another writer holds {path}") from error
        yield


def unchanged(signature):
    for group in ("inputs", "source"):
        for path, expected in signature[group].items():
            if sha256_file(path) != expected:
                raise ValueError(f"{group} changed during the run: {path}")


def prepare(cfg, config_path=CONFIG):
    torch.set_num_threads(cfg["threads_per_worker"])
    cases, inputs = load_data(cfg)
    inputs[str(config_path)] = sha256_file(config_path)
    source = {
        *Path("exp/benchmarks/reader").glob("*.py"),
        *Path("src/semantic_decoder").glob("*.py"),
        *Path("src").glob("*.py"),
        Path("exp/benchmarks/semantic_decoder/experiment.py"),
        Path("scripts/run_reader_s1_s6.sh"),
        Path("pyproject.toml"),
        Path("uv.lock"),
    }
    candidates = candidate_settings(cfg["training"])
    signature = {
        "format": "reader_s1_s6_v1",
        "config": cfg,
        "inputs": inputs,
        "source": {str(p): sha256_file(p) for p in sorted(source)},
        "packages": {
            n: importlib.metadata.version(n)
            for n in ("torch", "numpy", "pandas", "pyyaml")
        },
        "split_ids": cases["S1"].split_ids,
        "candidates": candidates,
        "selection": "development joint NLL within each state space; predeclared deployment seed",
        "test_evaluation": False,
        "resume_policy": "verify/skip complete fits; restart incomplete fits deterministically",
    }
    root = report_root(cfg)
    manifest = root / "protocol.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["signature"] != signature:
            raise ValueError("run signature changed; use a new run ID")
    else:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
        ).stdout.strip()
        write_json(
            manifest,
            {
                "signature": signature,
                "sha256": digest(signature),
                "created_at": now(),
                "git_revision": revision,
            },
        )
    for path, expected in signature["source"].items():
        target = root / "source" / path
        if target.exists():
            if sha256_file(target) != expected:
                raise ValueError(f"source snapshot checksum differs: {target}")
        else:
            copy_atomic(path, target)
    coverage = {}
    with torch.random.fork_rng():
        torch.manual_seed(0)
        for name, data in cases.items():
            model = SemanticDecoder(
                data.fit_z.shape[1],
                data.columns,
                hidden_dim=cfg["training"]["hidden_dim"],
                n_hidden=cfg["training"]["n_hidden"],
                dropout=cfg["training"]["dropout"],
            ).eval()
            with torch.no_grad():
                probabilities = model.log_joint(data.validation_z[:8]).exp()
            if not torch.isfinite(probabilities).all() or not torch.allclose(
                probabilities.sum(-1), torch.ones(len(probabilities)), atol=1e-5
            ):
                raise ValueError(f"invalid probabilities in {name}")
            coverage[name] = {
                "columns": {c.name: c.n_categories for c in data.columns},
                "fitting_class_counts": {
                    c.name: torch.bincount(
                        data.fit_targets[c.name], minlength=c.n_categories
                    ).tolist()
                    for c in data.columns
                },
                "fitting_joint_states": len(
                    torch.unique(
                        torch.stack(list(data.fit_targets.values()), dim=1), dim=0
                    )
                ),
                "possible_joint_states": math.prod(
                    c.n_categories for c in data.columns
                ),
                "baseline_parameters": sum(p.numel() for p in model.parameters()),
            }
    unchanged(signature)
    write_json(
        root / "preflight.json",
        {
            "status": "passed",
            "checked_at": now(),
            "signature_sha256": digest(signature),
            "split_counts": {k: len(v) for k, v in cases["S1"].split_ids.items()},
            "coverage": coverage,
            "optimization_performed": False,
            "test_evaluation": False,
            "search_fits": len(cases) * len(candidates),
            "final_fits": len(cases) * len(cfg["training"]["search"]["final_seeds"]),
        },
    )
    return cases, signature


def fit_paths(cfg, state, phase, name):
    relative = Path(state) / "runs" / cfg["run_id"] / phase / name
    return (
        Path(cfg["paths"]["models"]) / relative / "semantic_decoder.pt",
        report_root(cfg) / state / phase / name / "training.json",
    )


def verify_fit(path, data, expected):
    report = json.loads(path.read_text())
    if report["identity"] != expected:
        raise ValueError(f"fit signature differs: {path}")
    if sha256_file(report["checkpoint"]) != report["checkpoint_sha256"]:
        raise ValueError(f"checkpoint checksum differs: {path}")
    model = load_semantic_decoder(
        report["checkpoint"],
        expected_variant="independent",
        expected_columns=data.columns,
        expected_metadata={**data.metadata, "reader_fit": expected},
    )
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("non-finite checkpoint")
    settings = expected["settings"]
    if (model.hidden_dim, model.n_hidden, model.dropout) != (
        settings["hidden_dim"],
        settings["n_hidden"],
        settings["dropout"],
    ):
        raise ValueError("checkpoint architecture differs")
    observed = joint_nll(model, data.validation_z, data.validation_targets)
    for recorded in (
        report["validation_joint_nll"],
        report["training"]["best_validation_joint_nll"],
    ):
        if not math.isclose(observed, recorded, abs_tol=1e-6, rel_tol=1e-6):
            raise ValueError("checkpoint is not the selected validation optimum")
    return report


def execute_fit(cfg, state, data, settings, seed, phase, name, signature):
    checkpoint, report_path = fit_paths(cfg, state, phase, name)
    identity = {
        "run_id": cfg["run_id"],
        "state_space": state,
        "phase": phase,
        "name": name,
        "seed": seed,
        "settings": settings,
        "signature_sha256": digest(signature),
    }
    if report_path.exists():
        print(f"{now()} verified resume {state}/{phase}/{name}", flush=True)
        return verify_fit(report_path, data, identity)
    metadata = {**data.metadata, "reader_fit": identity}
    progress = report_path.with_name("progress.json")
    model = SemanticDecoder(
        data.fit_z.shape[1],
        data.columns,
        hidden_dim=settings["hidden_dim"],
        n_hidden=settings["n_hidden"],
        dropout=settings["dropout"],
    )
    started = now()

    def epoch_callback(current, record):
        if record["improved"]:
            current.save(checkpoint, metadata={**metadata, "selection": record})
        write_json(progress, {"status": "running", "updated_at": now(), **record})

    stopping, scheduler = settings["early_stopping"], settings["scheduler"]
    train_semantic_decoder(
        model,
        data.fit_z,
        data.fit_targets,
        epochs=settings["epochs"],
        batch_size=settings["batch_size"],
        lr=settings["lr"],
        device="cpu",
        seed=seed,
        verbose=False,
        validation_latents=data.validation_z,
        validation_targets=data.validation_targets,
        patience=stopping["patience"],
        min_delta=stopping["min_delta"],
        scheduler_patience=scheduler["patience"],
        scheduler_factor=scheduler["factor"],
        min_lr=scheduler["min_lr"],
        weight_decay=settings["weight_decay"],
        epoch_callback=epoch_callback,
    )
    model.save(
        checkpoint,
        metadata={
            **metadata,
            "selection": {
                k: v for k, v in model.training_summary.items() if k != "history"
            },
        },
    )
    report = {
        "identity": identity,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "started_at": started,
        "completed_at": now(),
        "training": model.training_summary,
        "validation_joint_nll": joint_nll(
            model, data.validation_z, data.validation_targets
        ),
        "validation_marginals": calibration_metrics(
            model,
            data.validation_z,
            data.validation_targets,
            settings["calibration_bins"],
        ),
        "parameters": sum(p.numel() for p in model.parameters()),
        "test_evaluation": False,
        "validation_note": "Used for selection; not an unbiased test estimate. Compare NLL within a state space only.",
    }
    write_json(report_path, report)
    verified = verify_fit(report_path, data, identity)
    write_json(
        progress,
        {
            "status": "complete",
            "updated_at": now(),
            "epochs_run": model.training_summary["epochs_run"],
            "seconds": model.training_summary["seconds"],
        },
    )
    print(
        f"{now()} {state}/{phase}/{name}: NLL={report['validation_joint_nll']:.6f}, epochs={model.training_summary['epochs_run']}, seconds={model.training_summary['seconds']:.1f}",
        flush=True,
    )
    return verified


def moments(values):
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def run_case(cfg, state, data, signature):
    torch.set_num_threads(cfg["threads_per_worker"])
    torch.use_deterministic_algorithms(True)
    root = report_root(cfg) / state
    started = perf_counter()
    with (
        exclusive(root / ".case.lock"),
        (root / "train.log").open("a", buffering=1) as log,
        redirect_stdout(log),
        redirect_stderr(log),
    ):
        try:
            unchanged(signature)
            trials = []
            base = cfg["training"]
            search = base["search"]
            for index, candidate in enumerate(signature["candidates"]):
                name = f"trial-{index:03d}"
                write_json(
                    root / "status.json",
                    {
                        "status": "running",
                        "phase": "search",
                        "name": name,
                        "updated_at": now(),
                    },
                )
                settings = {**base, **candidate}
                report = execute_fit(
                    cfg,
                    state,
                    data,
                    settings,
                    search["initialization_seed"],
                    "search",
                    name,
                    signature,
                )
                trials.append(
                    {
                        "trial": index,
                        "settings": candidate,
                        "validation_joint_nll": report["validation_joint_nll"],
                        "seconds": report["training"]["seconds"],
                        "epochs_run": report["training"]["epochs_run"],
                    }
                )
                write_json(root / "search.json", trials)
            selected = min(
                trials, key=lambda x: (x["validation_joint_nll"], x["trial"])
            )
            selection = {
                "selected_trial": selected,
                "deployment_seed": search["deployment_seed"],
                "rule": "minimum development joint NLL; seed fixed before fitting",
            }
            selection_path = root / "selection.json"
            if (
                selection_path.exists()
                and json.loads(selection_path.read_text()) != selection
            ):
                raise ValueError("previous selection changed")
            write_json(selection_path, selection)
            finals = []
            for seed in search["final_seeds"]:
                name = f"seed-{seed}"
                write_json(
                    root / "status.json",
                    {
                        "status": "running",
                        "phase": "final",
                        "name": name,
                        "updated_at": now(),
                    },
                )
                finals.append(
                    execute_fit(
                        cfg,
                        state,
                        data,
                        {**base, **selected["settings"]},
                        seed,
                        "final",
                        name,
                        signature,
                    )
                )
            unchanged(signature)
            result = {
                "state_space": state,
                "columns": [c.name for c in data.columns],
                **selection,
                "validation_joint_nll": moments(
                    [r["validation_joint_nll"] for r in finals]
                ),
                "validation_marginals": {
                    c.name: {
                        m: moments(
                            [r["validation_marginals"][c.name][m] for r in finals]
                        )
                        for m in ("accuracy", "ece")
                    }
                    for c in data.columns
                },
                "final_runs": [
                    {
                        "seed": r["identity"]["seed"],
                        "checkpoint": r["checkpoint"],
                        "checkpoint_sha256": r["checkpoint_sha256"],
                        "validation_joint_nll": r["validation_joint_nll"],
                        "best_epoch": r["training"]["best_epoch"],
                        "epochs_run": r["training"]["epochs_run"],
                    }
                    for r in finals
                ],
                "seconds": perf_counter() - started,
                "completed_at": now(),
                "test_evaluation": False,
            }
            write_json(root / "result.json", result)
            write_json(
                root / "status.json", {"status": "complete", "updated_at": now()}
            )
            return result
        except Exception:
            write_json(
                root / "status.json",
                {
                    "status": "failed",
                    "updated_at": now(),
                    "error": traceback.format_exc(),
                },
            )
            raise


def progress(cfg):
    cases = {}
    for state in cfg["state_spaces"]:
        root = report_root(cfg) / state
        status = root / "status.json"
        entry = (
            json.loads(status.read_text()) if status.exists() else {"status": "queued"}
        )
        entry["completed_fits"] = len(list(root.glob("search/*/training.json"))) + len(
            list(root.glob("final/*/training.json"))
        )
        if entry["status"] == "running":
            path = root / entry["phase"] / entry["name"] / "progress.json"
            if path.exists():
                entry["epoch_progress"] = json.loads(path.read_text())
        cases[state] = entry
    total = len(cases) * (
        cfg["training"]["search"]["trials"]
        + len(cfg["training"]["search"]["final_seeds"])
    )
    return {
        "completed_fits": sum(c["completed_fits"] for c in cases.values()),
        "total_fits": total,
        "cases": cases,
    }


def render(cfg, results, status):
    lines = [
        "# Independent readers: S1-S6",
        "",
        f"Run: `{cfg['run_id']}`. Status: **{status}**.",
        "",
        "EmbeddingGemma 768D; factual L labels only; fixed development set. No official-test, held-out-wording, counterfactual or downstream evaluation.",
        "",
        "Mean ± sample SD over seeds 42–46 (not confidence intervals). Development data select the models: these are not held-out performance estimates. NLLs across different state spaces are not comparable rankings.",
        "",
        "| State | Heads | Development joint NLL | Selected trial |",
        "|---|---|---:|---:|",
    ]
    for r in sorted(results, key=lambda r: r["state_space"]):
        m = r["validation_joint_nll"]
        lines.append(
            f"| {r['state_space']} | {', '.join(r['columns'])} | {m['mean']:.6f} ± {m['std']:.6f} | {r['selected_trial']['trial']} |"
        )
    lines += [
        "",
        "## Development marginal accuracy",
        "",
        "| State | Variable | Accuracy | ECE |",
        "|---|---|---:|---:|",
    ]
    for r in sorted(results, key=lambda r: r["state_space"]):
        for column, metrics in r["validation_marginals"].items():
            a, e = metrics["accuracy"], metrics["ece"]
            lines.append(
                f"| {r['state_space']} | {column} | {a['mean']:.4f} ± {a['std']:.4f} | {e['mean']:.4f} ± {e['std']:.4f} |"
            )
    lines += [
        "",
        "Each state's published checkpoint uses predeclared seed 42. All five selected seed checkpoints remain in its run directory. `protocol.json` records split IDs, input/code hashes, candidates and settings. Each fit's `training.json` contains its history and reload-verified development score.",
        "",
        "Scope ends at g. Evaluation readers, cross-fitting, label-budget sweeps, symbolic operator fitting and h_Z training are not run.",
    ]
    with atomic_output(report_root(cfg) / "summary.md") as temporary:
        temporary.write_text("\n".join(lines) + "\n")


def publish(cfg, results, signature):
    unchanged(signature)
    published = []
    for result in results:
        selected = next(
            r for r in result["final_runs"] if r["seed"] == result["deployment_seed"]
        )
        source = Path(selected["checkpoint"])
        if sha256_file(source) != selected["checkpoint_sha256"]:
            raise ValueError("selected checkpoint changed before publication")
        target = (
            Path(cfg["paths"]["models"]) / result["state_space"] / "semantic_decoder.pt"
        )
        if target.exists() and sha256_file(target) != selected["checkpoint_sha256"]:
            raise ValueError(
                f"refusing to replace an existing published model: {target}"
            )
        published.append(
            {
                "state_space": result["state_space"],
                "checkpoint": str(target),
                "source": str(source),
                "sha256": selected["checkpoint_sha256"],
                "seed": result["deployment_seed"],
                "signature_sha256": digest(signature),
            }
        )
    for item in published:
        copy_atomic(item["source"], item["checkpoint"])
        if sha256_file(item["checkpoint"]) != item["sha256"]:
            raise ValueError("publication checksum mismatch")
        write_json(Path(item["checkpoint"]).with_suffix(".info.json"), item)
    write_json(report_root(cfg) / "published.json", published)


def run(cfg, config_path=CONFIG):
    with exclusive(report_root(cfg) / ".run.lock"):
        cases, signature = prepare(cfg, config_path)
        results, errors = [], []
        started = now()
        root = report_root(cfg)

        def status(phase):
            write_json(
                root / "status.json",
                {
                    "status": phase,
                    "started_at": started,
                    "updated_at": now(),
                    "pid": os.getpid(),
                    "workers": cfg["workers"],
                    "test_evaluation": False,
                    **progress(cfg),
                    "errors": errors,
                },
            )

        status("running")
        render(cfg, [], "running")
        try:
            with ProcessPoolExecutor(
                max_workers=min(cfg["workers"], len(cases)),
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                pending = {
                    pool.submit(run_case, cfg, name, data, signature): name
                    for name, data in cases.items()
                }
                while pending:
                    done, _ = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
                    for future in done:
                        name = pending.pop(future)
                        try:
                            results.append(future.result())
                            results.sort(key=lambda r: r["state_space"])
                            write_json(root / "results.json", results)
                            print(f"{now()} completed {name}", flush=True)
                        except Exception:
                            errors.append(
                                {"state_space": name, "error": traceback.format_exc()}
                            )
                            print(
                                f"{now()} failed {name}; other cases continue",
                                flush=True,
                            )
                    status("running")
                    if done:
                        render(cfg, results, "running")
                    counts = progress(cfg)
                    print(
                        f"{now()} {counts['completed_fits']}/{counts['total_fits']} fits complete",
                        flush=True,
                    )
            if errors:
                raise RuntimeError(
                    f"{len(errors)} reader cases failed; see status.json"
                )
            publish(cfg, results, signature)
            render(cfg, results, "complete")
            status("complete")
            print(
                f"{now()} complete: six independent readers published; no test evaluation",
                flush=True,
            )
        except Exception:
            errors.append({"error": traceback.format_exc()})
            status("failed")
            render(cfg, results, "failed")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "run", "status"))
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    cfg = read_config(args.config, args.run_id)
    if args.stage == "run":
        run(cfg, args.config)
    elif args.stage == "preflight":
        with exclusive(report_root(cfg) / ".run.lock"):
            prepare(cfg, args.config)
        print(f"Preflight passed: {report_root(cfg)}")
    else:
        print(json.dumps(progress(cfg), indent=2))


if __name__ == "__main__":
    main()
