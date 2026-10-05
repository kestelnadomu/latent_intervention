"""Tune both g variants for every saved encoder, then evaluate and write a report.

The existing decoder_experiment owns numerical training, checkpoint verification,
resumption and publication. This module adds the all-encoder matrix, a selection
freeze before test evaluation, and automatic comparison reporting.
"""

import argparse
import fcntl
import importlib.metadata
import json
import multiprocessing
import os
import re
import shutil
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import torch
import yaml

from src.artifact_io import atomic_output, sha256_file, write_json
from src.config import CONFIG_PATH
from src.decoder_benchmark.decoder_benchmark_evaluation import evaluate_case
from src.decoder_benchmark.decoder_benchmark_matrix import (
    ROOT,
    annotate_result,
    case_name,
    discover,
    freeze_selection,
)
from src.decoder_benchmark.decoder_benchmark_report import render
from src.decoder_benchmark.decoder_experiment import (
    now,
    provenance,
    publish_results,
    run_case,
    verify_run,
)

DEFAULT_RUN = "g-all-encoders-v1"
REPORTS = ROOT / "reports/talent/decoder_benchmarks"


def training_case(config, run_id, threads):
    return annotate_result(run_case(config, run_id, threads))


def _root(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run-id must be a simple directory name")
    return REPORTS / run_id


def _assert_unchanged(signature):
    for group in ("code_sha256", "input_sha256"):
        for path, expected in signature[group].items():
            if sha256_file(path) != expected:
                raise ValueError(
                    f"benchmark {group} changed: {path}; do not publish mixed runs"
                )


def prepare(run_id, config_path=CONFIG_PATH, threads=1):
    if threads < 1:
        raise ValueError("thread count must be positive")
    torch.set_num_threads(1)
    root = _root(run_id)
    plan = discover(config_path)
    signature = provenance(plan["configs"])
    signature.update(
        format_version=2,
        threads_per_worker=threads,
        selection="Hyperparameters by validation NLL; encoder/decoder by mean validation NLL over final seeds. Freeze all choices before factual/CF official-test evaluation.",
        inventory=plan["inventory"],
        split_ids=plan["split_ids"],
        langvae_ft_validation_caveat=plan["langvae_ft_validation_caveat"],
        packages={
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pandas", "pyyaml")
        },
    )
    extras = {str(config_path)}
    for config in plan["configs"]:
        extras.add(str(Path(config["paths"]["latents"]).with_suffix(".info.json")))
        extras.add(config["paths"]["sim_counterfactual"])
    signature["input_sha256"].update(
        {path: sha256_file(path) for path in sorted(extras)}
    )
    for path in sorted(Path("src/encoders/langvae_ft").glob("*.py")):
        signature["code_sha256"][str(path)] = sha256_file(path)
    for path in ("configs/langvae_ft_scope.json", "scripts/run_decoder_benchmark.sh"):
        signature["code_sha256"][path] = sha256_file(path)
    manifest = root / "protocol.json"
    if manifest.exists():
        recorded = json.loads(manifest.read_text())

        # A commit with identical source bytes does not invalidate a running job.
        def comparable(value):
            return {k: v for k, v in value.items() if k != "git_revision"}

        if comparable(recorded) != comparable(signature):
            raise ValueError("code/config/input provenance changed; use a new run ID")
        signature = recorded
    else:
        write_json(manifest, signature)
    # Finish an interrupted preparation, but never rewrite a mismatched snapshot.
    for filename, digest in signature["code_sha256"].items():
        destination = root / "source" / filename
        if destination.exists():
            if sha256_file(destination) != digest:
                raise ValueError(f"source snapshot checksum mismatch: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(filename, destination)
    for tag, template in plan["templates"].items():
        output = root / "configs" / f"{tag}.yaml"
        if output.exists():
            if yaml.safe_load(output.read_text()) != template:
                raise ValueError(f"saved encoder configuration changed: {output}")
        else:
            with atomic_output(root / "configs" / f"{tag}.yaml") as temporary:
                temporary.write_text(
                    yaml.safe_dump(template, sort_keys=False), encoding="utf-8"
                )
    if (root / "plan.json").exists():
        if json.loads((root / "plan.json").read_text()) != plan:
            raise ValueError("saved benchmark plan changed")
    else:
        write_json(root / "plan.json", plan)
    if not (root / "summary.md").exists():
        render(root, plan, [], [], None, "prepared; training not yet started")
    print(
        f"Verified {len(plan['inventory'])} latent spaces / {len(plan['configs'])} decoder cases: {root}",
        flush=True,
    )
    return root, plan, signature


def progress(plan, run_id):
    details = []
    completed_fits = 0
    budget = 0
    for config in plan["configs"]:
        search = config["semantic_decoder"]["search"]
        budget += int(search["trials"]) + len(search["final_seeds"])
        folder = Path(config["paths"]["decoder_report"]).parent / "experiments" / run_id
        search_file = folder / "search.json"
        trials = len(json.loads(search_file.read_text())) if search_file.exists() else 0
        finals = sum(
            (folder / "final" / f"seed-{seed}" / "semantic_decoder.json").exists()
            for seed in search["final_seeds"]
        )
        completed_fits += trials + finals
        status_path = folder / "status.json"
        status = (
            json.loads(status_path.read_text())
            if status_path.exists()
            else {"status": "queued"}
        )
        if status.get("phase") == "search" and status.get("status") == "running":
            current = (
                folder
                / "search"
                / f"trial-{status['trial']:03d}"
                / "semantic_decoder.progress.json"
            )
        elif status.get("phase") == "final" and status.get("status") == "running":
            current = (
                folder
                / "final"
                / f"seed-{status['seed']}"
                / "semantic_decoder.progress.json"
            )
        else:
            current = None
        details.append(
            {
                "case": case_name(config),
                "completed_search": trials,
                "completed_final_seeds": finals,
                **status,
                "epoch_progress": json.loads(current.read_text())
                if current and current.exists()
                else None,
            }
        )
    return {"completed_fits": completed_fits, "total_fits": budget, "cases": details}


def _frozen_selection(root, results, configs):
    selection = freeze_selection(results, [case_name(c) for c in configs])
    path = root / "selection.json"
    if path.exists() and json.loads(path.read_text()) != selection:
        raise ValueError("previously frozen selection changed; refusing test reuse")
    if not path.exists():
        write_json(path, selection)
    return selection


def run(run_id, config_path=CONFIG_PATH, workers=4, threads=1):
    if workers < 1 or threads < 1:
        raise ValueError("workers and threads must be positive")
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (REPORTS / ".benchmark.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root, plan, signature = prepare(run_id, config_path, threads)
        status_path = root / "status.json"
        if (
            status_path.exists()
            and json.loads(status_path.read_text())["status"] == "complete"
        ):
            for item in json.loads((root / "published.json").read_text()):
                report = json.loads(Path(item["paths"]["decoder_report"]).read_text())
                verify_run(report["config"])
            print(f"Already complete and verified: {root}", flush=True)
            return
        started = now()
        results, evaluations, errors = [], [], []
        selection = None

        def status(phase, **extra):
            write_json(
                status_path,
                {
                    "status": "running",
                    "phase": phase,
                    "run_id": run_id,
                    "pid": os.getpid(),
                    "started_at": started,
                    "updated_at": now(),
                    "workers": workers,
                    "completed_cases": len(results),
                    "evaluated_cases": len(evaluations),
                    "total_cases": len(plan["configs"]),
                    **extra,
                },
            )

        try:
            status("training", **progress(plan, run_id))
            render(root, plan, [], [], None, "training")
            # Wider cases first reduce the long tail; the scientific candidate/seed order is unchanged.
            widths = {row["encoder_tag"]: row["dimension"] for row in plan["inventory"]}
            ordered = sorted(
                plan["configs"],
                key=lambda c: (-widths[case_name(c).split("/")[0]], case_name(c)),
            )
            with ProcessPoolExecutor(
                max_workers=min(workers, len(ordered)),
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                futures = {
                    pool.submit(training_case, c, run_id, threads): case_name(c)
                    for c in ordered
                }
                while futures:
                    completed, _ = wait(
                        futures, timeout=30, return_when=FIRST_COMPLETED
                    )
                    for future in completed:
                        name = futures.pop(future)
                        try:
                            result = future.result()
                            results.append(result)
                            results.sort(key=lambda row: row["case"])
                            write_json(root / "results.json", results)
                            print(
                                f"{now()} Trained {name}: {result['summary']['validation_joint_nll']}",
                                flush=True,
                            )
                        except Exception:
                            errors.append(
                                {"case": name, "error": traceback.format_exc()}
                            )
                            write_json(root / "errors.json", errors)
                            print(
                                f"{now()} FAILED {name}; other independent cases continue.",
                                flush=True,
                            )
                    counts = progress(plan, run_id)
                    status("training", errors=errors, **counts)
                    if completed:
                        render(
                            root, plan, results, [], None, "training (partial results)"
                        )
                    print(
                        f"{now()} Progress: {counts['completed_fits']}/{counts['total_fits']} fits, {len(results)}/{len(ordered)} cases",
                        flush=True,
                    )
            if errors:
                raise RuntimeError(
                    f"{len(errors)} cases failed; inspect errors.json and resume verified completed trials"
                )
            _assert_unchanged(signature)
            selection = _frozen_selection(root, results, plan["configs"])
            render(
                root,
                plan,
                results,
                [],
                selection,
                "selection frozen; evaluating official test",
            )
            status("evaluating", **progress(plan, run_id))
            with ProcessPoolExecutor(
                max_workers=min(workers, len(results)),
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                futures = {
                    pool.submit(evaluate_case, result, root): result["case"]
                    for result in results
                }
                while futures:
                    completed, _ = wait(
                        futures, timeout=30, return_when=FIRST_COMPLETED
                    )
                    for future in completed:
                        name = futures.pop(future)
                        evaluations.append(future.result())
                        evaluations.sort(key=lambda row: row["case"])
                        write_json(root / "evaluation.json", evaluations)
                        render(
                            root,
                            plan,
                            results,
                            evaluations,
                            selection,
                            "evaluating official test",
                        )
                        print(f"{now()} Evaluated {name}", flush=True)
                    status("evaluating")
            _assert_unchanged(signature)
            status("publishing")
            published = publish_results(results, root)
            tags = {r["active_paths"]["decoder_model"]: r for r in results}
            for item in published:
                item["encoder_tag"] = tags[item["paths"]["decoder_model"]][
                    "encoder_tag"
                ]
            write_json(root / "published.json", published)
            render(
                root,
                plan,
                results,
                evaluations,
                selection,
                "complete; all active models verified",
            )
            write_json(
                status_path,
                {
                    "status": "complete",
                    "phase": "complete",
                    "run_id": run_id,
                    "started_at": started,
                    "completed_at": now(),
                    "completed_cases": len(results),
                    "evaluated_cases": len(evaluations),
                    "recommended_case": selection["recommended_case"],
                    "report": str(root / "summary.md"),
                    "published": published,
                },
            )
            print(
                f"All {len(results)} decoders verified and published. Report: {root / 'summary.md'}",
                flush=True,
            )
        except BaseException:
            write_json(
                status_path,
                {
                    "status": "failed",
                    "run_id": run_id,
                    "started_at": started,
                    "failed_at": now(),
                    "completed_cases": len(results),
                    "evaluated_cases": len(evaluations),
                    "error": traceback.format_exc(),
                },
            )
            render(
                root,
                plan,
                results,
                evaluations,
                selection,
                "FAILED / incomplete; see status.json",
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--report", action="store_true")
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.prepare:
        prepare(args.run_id, args.config, args.threads_per_worker)
    elif args.run:
        run(args.run_id, args.config, args.workers, args.threads_per_worker)
    else:
        root = _root(args.run_id)

        def read(name, default):
            path = root / name
            return json.loads(path.read_text()) if path.exists() else default

        render(
            root,
            read("plan.json", {}),
            read("results.json", []),
            read("evaluation.json", []),
            read("selection.json", None),
            read("status.json", {"status": "prepared"})["status"],
        )
        print(root / "summary.md")


if __name__ == "__main__":
    main()
