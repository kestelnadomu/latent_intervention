"""Tune all four g baselines, repeat selected settings, and publish verified models.

Run: python -m src.decoder_experiment --run-id cv-g500-v1 --workers 4
Repeat the same command to resume completed trials. Incomplete trials restart.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import itertools
import json
import math
import multiprocessing
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from src import pipeline
from src.artifact_io import copy_atomic, write_json
from src.config import CONFIG_PATH, load_config
from src.pair_encoding import sha256_file
from src.semantic_decoder import joint_nll, load_semantic_decoder

SEARCH_KEYS = ("lr", "weight_decay", "dropout", "hidden_dim")
CASES = tuple(
    itertools.product(("langvae", "nomic"), ("independent", "autoregressive"))
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def candidate_settings(settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Common deterministic budget: include the old baseline, sample the remainder."""
    search = settings["search"]
    baseline = {key: settings[key] for key in SEARCH_KEYS}
    candidates = [
        dict(zip(SEARCH_KEYS, row))
        for row in itertools.product(*(search[key] for key in SEARCH_KEYS))
    ]
    candidates = [row for row in candidates if row != baseline]
    random.Random(int(search["seed"])).shuffle(candidates)
    count = int(search["trials"])
    if not 1 <= count <= len(candidates) + 1:
        raise ValueError("search.trials exceeds the distinct search space")
    return [baseline, *candidates[: count - 1]]


def verify_run(config: dict[str, Any]) -> dict[str, Any]:
    """Reload and check exact provenance, split isolation, weights, and validation NLL."""
    report_path = Path(config["paths"]["decoder_report"])
    report = json.loads(report_path.read_text())
    if report["config"] != config:
        raise ValueError(f"configuration changed for existing run: {report_path}")
    if report["checkpoint_sha256"] != sha256_file(config["paths"]["decoder_model"]):
        raise ValueError(f"checkpoint checksum mismatch: {report_path}")
    artifact = pipeline.load_latent_artifact(config)
    columns, _ = pipeline.load_schema(config.get("sim_config"))
    model = load_semantic_decoder(
        config["paths"]["decoder_model"],
        expected_variant=config["semantic_decoder"]["variant"],
        expected_columns=columns,
        expected_metadata=pipeline._decoder_metadata(config, artifact),
    )
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("checkpoint contains non-finite parameters")
    train_idx, _ = pipeline._official_indices(artifact)
    fit_idx, val_idx = pipeline._fit_calibration_split(
        train_idx,
        float(config["semantic_decoder"]["calibration_split"]),
        int(config["semantic_decoder"]["split_seed"]),
    )
    expected_ids = {
        "fit": [artifact.ids[i] for i in fit_idx.tolist()],
        "validation": [artifact.ids[i] for i in val_idx.tolist()],
        "official_test": list(artifact.test_ids),
    }
    if report["split_ids"] != expected_ids or report[
        "split_sha256"
    ] != pipeline._config_sha256(expected_ids):
        raise ValueError("decoder split manifest mismatch")
    used = set(expected_ids["fit"]) | set(expected_ids["validation"])
    if used & set(artifact.test_ids) or used != set(artifact.train_ids):
        raise ValueError("decoder split leaks test IDs or omits training IDs")
    targets = pipeline._aligned_targets(
        config["paths"]["sim_factual"], artifact.ids, columns
    )
    observed = joint_nll(model, artifact.z[val_idx], pipeline._subset(targets, val_idx))
    best = report["training"]["best_validation_joint_nll"]
    if not math.isclose(observed, best, abs_tol=1e-6, rel_tol=1e-6):
        raise ValueError("reloaded weights do not match the best validation checkpoint")
    if not math.isclose(
        observed, report["validation_joint_nll"], abs_tol=1e-6, rel_tol=1e-6
    ):
        raise ValueError("validation report does not match checkpoint")
    with torch.no_grad():
        probabilities = model.log_joint(artifact.z[val_idx[:16]]).exp()
    if not torch.isfinite(probabilities).all() or not torch.allclose(
        probabilities.sum(-1), torch.ones(len(probabilities)), atol=1e-5
    ):
        raise ValueError("checkpoint produces invalid joint probabilities")
    return report


def execute_run(config: dict[str, Any]) -> dict[str, Any]:
    if Path(config["paths"]["decoder_report"]).exists():
        print(f"resume verified {config['paths']['decoder_report']}", flush=True)
        return verify_run(config)
    pipeline.stage_train_decoder(config)
    return verify_run(config)


def run_config(
    base: dict[str, Any],
    run_id: str,
    phase: str,
    name: str,
    settings: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["seed"] = int(seed)
    config["semantic_decoder"].update(settings)
    for key in ("decoder_model", "decoder_report"):
        original = Path(base["paths"][key])
        config["paths"][key] = str(
            original.parent / "experiments" / run_id / phase / name / original.name
        )
    return config


def metric_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    def moments(values):
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    return {
        "n_seeds": len(reports),
        "variability": "Sample standard deviation across initialization/shuffle/dropout seeds on a fixed validation split; not a test confidence interval.",
        "validation_joint_nll": moments([r["validation_joint_nll"] for r in reports]),
        "marginals": {
            name: {
                metric: moments([r["calibration"][name][metric] for r in reports])
                for metric in ("accuracy", "ece")
            }
            for name in reports[0]["calibration"]
        },
    }


def run_case(base: dict[str, Any], run_id: str, threads: int = 1) -> dict[str, Any]:
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    settings = base["semantic_decoder"]
    search = settings["search"]
    finals = [int(seed) for seed in search["final_seeds"]]
    if (
        not finals
        or len(set(finals)) != len(finals)
        or int(search["deployment_seed"]) not in finals
    ):
        raise ValueError("final seeds must be unique and include deployment_seed")
    experiment_dir = (
        Path(base["paths"]["decoder_report"]).parent / "experiments" / run_id
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    with (
        (experiment_dir / "train.log").open("a", buffering=1) as log,
        redirect_stdout(log),
        redirect_stderr(log),
    ):
        try:
            trials = []
            for index, candidate in enumerate(candidate_settings(settings)):
                name = f"trial-{index:03d}"
                print(f"{now()} {name} {candidate}", flush=True)
                config = run_config(
                    base,
                    run_id,
                    "search",
                    name,
                    candidate,
                    search["initialization_seed"],
                )
                write_json(
                    experiment_dir / "status.json",
                    {
                        "status": "running",
                        "phase": "search",
                        "trial": index,
                        "total_trials": search["trials"],
                        "updated_at": now(),
                    },
                )
                report = execute_run(config)
                trials.append(
                    {
                        "trial": index,
                        "settings": candidate,
                        "validation_joint_nll": report["validation_joint_nll"],
                        "best_epoch": report["training"]["best_epoch"],
                        "epochs_run": report["training"]["epochs_run"],
                        "report": config["paths"]["decoder_report"],
                    }
                )
                write_json(experiment_dir / "search.json", trials)
            best = min(
                trials, key=lambda row: (row["validation_joint_nll"], row["trial"])
            )
            final_reports = []
            selected_config = None
            for seed in finals:
                print(
                    f"{now()} final seed={seed}, selected trial={best['trial']}",
                    flush=True,
                )
                config = run_config(
                    base, run_id, "final", f"seed-{seed}", best["settings"], seed
                )
                write_json(
                    experiment_dir / "status.json",
                    {
                        "status": "running",
                        "phase": "final",
                        "seed": seed,
                        "updated_at": now(),
                    },
                )
                final_reports.append(execute_run(config))
                if seed == int(search["deployment_seed"]):
                    selected_config = config
            result = {
                "encoder": base["encoder"]["variant"],
                "decoder": settings["variant"],
                "run_id": run_id,
                "selected_trial": best,
                "selected_config": selected_config,
                "active_paths": {
                    key: base["paths"][key]
                    for key in ("decoder_model", "decoder_report")
                },
                "deployment_seed": int(search["deployment_seed"]),
                "deployment_rule": "Predeclared seed; never select a seed by validation or test performance.",
                "summary": metric_summary(final_reports),
                "final_runs": [
                    {
                        "seed": r["seed"],
                        "best_epoch": r["training"]["best_epoch"],
                        "epochs_run": r["training"]["epochs_run"],
                        "validation_joint_nll": r["validation_joint_nll"],
                        "checkpoint": r["config"]["paths"]["decoder_model"],
                        "report": r["config"]["paths"]["decoder_report"],
                    }
                    for r in final_reports
                ],
                "seconds": perf_counter() - started,
                "completed_at": now(),
            }
            write_json(experiment_dir / "summary.json", result)
            write_json(
                experiment_dir / "status.json",
                {"status": "complete", "updated_at": now()},
            )
            return result
        except BaseException:
            traceback.print_exc()
            write_json(
                experiment_dir / "status.json",
                {
                    "status": "failed",
                    "updated_at": now(),
                    "error": traceback.format_exc(),
                },
            )
            raise


def publish_results(results: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    """Verify all candidates, archive old active files, then replace individual files atomically."""
    prepared = []
    for result in results:
        report = verify_run(result["selected_config"])
        prepared.append((result, report))
    backups = []
    for result, _ in prepared:
        for key, destination in result["active_paths"].items():
            active = Path(destination)
            archive = active.parent / "archive" / result["run_id"] / active.name
            if active.exists():
                if not archive.exists():
                    copy_atomic(active, archive)
                backups.append(
                    {
                        "active": str(active),
                        "archive": str(archive),
                        "archive_sha256": sha256_file(archive),
                    }
                )
    write_json(root / "backups.json", backups)
    published = []
    for result, report in prepared:
        selected = result["selected_config"]
        active = result["active_paths"]
        copy_atomic(selected["paths"]["decoder_model"], active["decoder_model"])
        if sha256_file(active["decoder_model"]) != report["checkpoint_sha256"]:
            raise ValueError("published checkpoint checksum mismatch")
        active_report = {**report, "experiment": result}
        active_report["config"] = copy.deepcopy(report["config"])
        active_report["config"]["paths"].update(active)
        write_json(active["decoder_report"], active_report)
        verify_run(active_report["config"])
        item = {
            "encoder": result["encoder"],
            "decoder": result["decoder"],
            "paths": active,
            "model_sha256": report["checkpoint_sha256"],
            "report_sha256": sha256_file(active["decoder_report"]),
            "best_epoch": report["training"]["best_epoch"],
            "epochs_run": report["training"]["epochs_run"],
            "validation_joint_nll": report["validation_joint_nll"],
        }
        published.append(item)
        write_json(root / "published.json", published)
    return published


def provenance(configs: list[dict[str, Any]]) -> dict[str, Any]:
    code_paths = [
        Path("uv.lock"),
        *sorted(Path("src").glob("*.py")),
        Path("src/config.yaml"),
    ]
    inputs = sorted(
        {
            str(config["paths"][key])
            for config in configs
            for key in (
                "latents",
                "texts",
                "texts_counterfactual",
                "pair_index",
                "sim_factual",
            )
        }
    )
    inputs += sorted({str(config["sim_config"]) for config in configs})
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return {
        "format_version": 1,
        "configs": configs,
        "code_sha256": {str(path): sha256_file(path) for path in code_paths},
        "input_sha256": {path: sha256_file(path) for path in inputs},
        "git_revision": revision,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "selection": "Minimum validation joint NLL, fixed split; five final seeds; no official-test evaluation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", args.run_id):
        parser.error("run-id must be a simple directory name")
    if args.workers < 1 or args.threads_per_worker < 1:
        parser.error("workers and threads must be positive")
    configs = [
        load_config(path=args.config, encoder_variant=e, decoder_variant=d)
        for e, d in CASES
    ]
    root = Path("reports/talent/decoder_experiments") / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    with (root / "experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signature = provenance(configs)
        signature["threads_per_worker"] = args.threads_per_worker
        manifest_path = root / "protocol.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != signature:
                raise ValueError(
                    "code/config/input provenance changed; use a new run-id"
                )
        else:
            write_json(manifest_path, signature)
            for filename in signature["code_sha256"]:
                target = root / "source" / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(filename, target)
        if (root / "status.json").exists() and json.loads(
            (root / "status.json").read_text()
        )["status"] == "complete":
            for item in json.loads((root / "published.json").read_text()):
                report = json.loads(Path(item["paths"]["decoder_report"]).read_text())
                verify_run(report["config"])
            print(f"Already complete and verified: {root}")
            return
        started = now()
        write_json(
            root / "status.json",
            {
                "status": "running",
                "started_at": started,
                "pid": os.getpid(),
                "workers": args.workers,
            },
        )
        results = []
        try:
            with ProcessPoolExecutor(
                max_workers=min(args.workers, len(configs)),
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                futures = [
                    pool.submit(run_case, config, args.run_id, args.threads_per_worker)
                    for config in configs
                ]
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    print(
                        f"Completed {result['encoder']}/{result['decoder']}: {result['summary']['validation_joint_nll']}",
                        flush=True,
                    )
                    write_json(root / "results.json", results)
            results.sort(key=lambda row: (row["encoder"], row["decoder"]))
            published = publish_results(results, root)
            write_json(
                root / "status.json",
                {
                    "status": "complete",
                    "started_at": started,
                    "completed_at": now(),
                    "published": published,
                },
            )
            print(
                f"All four decoders verified and published; reports: {root}", flush=True
            )
        except BaseException:
            write_json(
                root / "status.json",
                {
                    "status": "failed",
                    "started_at": started,
                    "failed_at": now(),
                    "error": traceback.format_exc(),
                },
            )
            raise


if __name__ == "__main__":
    main()
