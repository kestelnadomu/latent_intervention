"""Four predeclared candidates, three selected seeds, and one sealed factual audit."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from statistics import mean, stdev

from src.artifact_io import copy_atomic, sha256_file, write_json
from src.text_realizer.artifacts import load_text_realizer, read_checkpoint
from src.text_realizer.metrics import paired_bootstrap

from .data import load_bundle, load_groups, prepared_path
from .evaluation import evaluate_model
from .protocol import (
    identity,
    immutable_json,
    lock,
    progress_callback,
    read_json,
    require_current,
    require_frozen,
    snapshot,
    status,
)
from .report import campaign_summary, examples
from .settings import ROOT, VARIANTS, resolve, run_paths


def make_cycle_encoder(config: dict, device: str):
    from src.encoder.embeddinggemma import EmbeddingGemmaTextEncoder

    return EmbeddingGemmaTextEncoder(
        {
            **config["encoder"],
            "device": device,
            "embeddinggemma_cache_dir": str(resolve(config["paths"]["cache"])),
        }
    )


def evaluate_trial(config: dict, run_id: str, *, device: str) -> dict:
    prepared = prepared_path(config)
    models, reports = run_paths(config, run_id)
    require_current(reports, config, prepared)
    training = read_json(reports / "training.json")
    checkpoint = models / "best.pt"
    if training["stage"] != "train" or training["checkpoint_sha256"] != sha256_file(
        checkpoint
    ):
        raise ValueError(
            "only completed, unmodified training checkpoints may enter selection"
        )
    output = reports / "factual_tune.json"
    if output.exists():
        previous = read_json(output)
        if previous["checkpoint_sha256"] != sha256_file(checkpoint):
            raise ValueError("tune report refers to different weights")
        return previous
    bundle, groups = load_bundle(prepared), load_groups(prepared)
    model = load_text_realizer(
        checkpoint, device=device, cache_dir=str(resolve(config["paths"]["cache"]))
    )
    encoder = make_cycle_encoder(config, device)
    status(reports, "evaluating_factual_tune")
    result = evaluate_model(
        model,
        bundle["tune"],
        bundle["fit"],
        encoder=encoder,
        groups=groups["tune"],
        fit_groups=groups["fit"],
        progress=progress_callback(reports, "evaluating_factual_tune"),
    )
    result["checkpoint_sha256"] = sha256_file(checkpoint)
    result["complete"] = True
    write_json(output, result)
    status(reports, "tune_complete")
    return result


def choose(candidates: dict[str, dict], thresholds: dict) -> dict:
    rows = []
    for variant, result in sorted(candidates.items()):
        gap = result["conditioning"]
        rows.append(
            {
                "variant": variant,
                "eligible": gap["nll_gap"] >= thresholds["nll_gap"]
                and gap["character_gap"] >= thresholds["character_gap"],
                **gap,
                "character_similarity": result["conditions"]["correct"]["summary"][
                    "character_similarity"
                ],
                "cycle_cosine": result["cycle"]["cosine_invalid_scored_minus_one"],
                "checkpoint_sha256": result["checkpoint_sha256"],
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    eligible.sort(
        key=lambda row: (
            -row["character_similarity"],
            -row["cycle_cosine"],
            row["variant"],
        )
    )
    return {
        "selected": eligible[0]["variant"] if eligible else None,
        "candidates": rows,
        "deployment_seed": 42,
        "used_audit_for_selection": False,
        "used_official_test_for_selection": False,
    }


def trial_id(run_id: str, variant: str, seed: int) -> str:
    return f"{run_id}.{variant}.seed-{seed}"


def _worker(
    config_path: Path, run_id: str, variant: str, seed: int, gpu: str, log: Path
):
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpu,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    command = [
        sys.executable,
        "-u",
        "-m",
        "exp.text_realizer",
        "train",
        "--config",
        str(config_path),
        "--run-id",
        trial_id(run_id, variant, seed),
        "--variant",
        variant,
        "--seed",
        str(seed),
        "--device",
        "cuda:0",
        "--resume",
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_jobs(config_path, root, run_id, jobs, gpus):
    # Explicit waves keep a maximum of one child per selected physical GPU.
    for start in range(0, len(jobs), len(gpus)):
        wave = jobs[start : start + len(gpus)]
        with ThreadPoolExecutor(max_workers=len(wave)) as pool:
            pending = {
                pool.submit(
                    _worker,
                    config_path,
                    run_id,
                    variant,
                    seed,
                    gpu,
                    root / "logs" / f"{variant}-seed-{seed}.log",
                )
                for (variant, seed), gpu in zip(wave, gpus, strict=False)
            }
            while pending:
                finished, pending = wait(
                    pending, timeout=30, return_when=FIRST_COMPLETED
                )
                status(
                    root,
                    "campaign_training",
                    active=len(pending),
                    wave=start // len(gpus) + 1,
                )
                for future in finished:
                    future.result()


def campaign(config: dict, config_path: Path, run_id: str, gpus: list[str]) -> dict:
    if (
        not gpus
        or len(gpus) != len(set(gpus))
        or any(not gpu.isdigit() for gpu in gpus)
    ):
        raise ValueError("provide unique physical GPU indices, e.g. --gpus 0 1 2 3")
    _, reports = run_paths(config, run_id)
    prepared = prepared_path(config)
    load_bundle(prepared)
    with lock(reports / ".campaign.lock"):
        plan = {
            "identity": identity(config, prepared),
            "variants": list(VARIANTS),
            "seeds": config["selection"]["seeds"],
            "deployment_seed": 42,
            "oracle_gate": "sealed",
        }
        snapshot(reports, plan)
        try:
            run_jobs(
                config_path,
                reports,
                run_id,
                [(variant, 42) for variant in VARIANTS],
                gpus,
            )
            candidates = {
                variant: read_json(
                    run_paths(config, trial_id(run_id, variant, 42))[1]
                    / "factual_tune.json"
                )
                for variant in VARIANTS
            }
            selection = choose(candidates, config["selection"])
            immutable_json(reports / "selection.json", selection)
            campaign_summary(reports / "summary.md", selection)
            if selection["selected"] is None:
                status(reports, "no_qualified_model", oracle_gate="sealed")
                return selection
            variant = selection["selected"]
            run_jobs(config_path, reports, run_id, [(variant, 43), (variant, 44)], gpus)
            freeze(config, run_id, device=f"cuda:{gpus[0]}")
            # Audit owns a GPU lease itself; no oracle command is ever scheduled here.
            from .runtime import gpu_lease

            with gpu_lease(f"cuda:{gpus[0]}"):
                result = audit(config, run_id, device=f"cuda:{gpus[0]}")
            status(
                reports,
                "complete",
                oracle_gate="sealed",
                improvement_passed=result["improvement_passed"],
            )
            return result
        except BaseException as exc:
            status(
                reports,
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                error=f"{type(exc).__name__}: {exc}",
                oracle_gate="sealed",
            )
            raise


def freeze(config: dict, run_id: str, *, device: str) -> dict:
    models, reports = run_paths(config, run_id)
    prepared = prepared_path(config)
    require_current(reports, config, prepared)
    with lock(reports / ".freeze.lock"):
        if (reports / "frozen.json").exists():
            return frozen_checkpoints(config, run_id)[0]
        selection = read_json(reports / "selection.json")
        candidates = {
            name: read_json(
                run_paths(config, trial_id(run_id, name, 42))[1] / "factual_tune.json"
            )
            for name in VARIANTS
        }
        if selection != choose(candidates, config["selection"]):
            raise ValueError(
                "selection no longer matches the predeclared tune-only rule"
            )
        variant = selection["selected"]
        if variant is None:
            raise ValueError("no eligible model to freeze")
        checkpoints, report_hashes = [], {
            "selection.json": sha256_file(reports / "selection.json")
        }
        expected_spec, expected_generation = None, None
        for seed in config["selection"]["seeds"]:
            source_models, source_reports = run_paths(
                config, trial_id(run_id, variant, seed)
            )
            source_plan = require_current(source_reports, config, prepared)
            result = read_json(source_reports / "factual_tune.json")
            source = source_models / "best.pt"
            payload = read_checkpoint(source)
            if (
                result["checkpoint_sha256"] != sha256_file(source)
                or payload["metadata"]["provenance"]["stage"] != "train"
                or payload["metadata"]["provenance"]["seed"] != seed
                or source_plan["variant"] != variant
                or source_plan["seed"] != seed
                or source_plan["model"] != payload["metadata"]["spec"]
                or source_plan["generation"] != payload["metadata"]["generation"]
                or payload["metadata"]["encoder"] != config["encoder"]
            ):
                raise ValueError("invalid selected checkpoint")
            if expected_spec is None:
                expected_spec = payload["metadata"]["spec"]
                expected_generation = payload["metadata"]["generation"]
            elif (
                expected_spec != payload["metadata"]["spec"]
                or expected_generation != payload["metadata"]["generation"]
            ):
                raise ValueError(
                    "selected seeds must share architecture and generation settings"
                )
            target = models / "selected" / f"seed-{seed}.pt"
            copy_atomic(source, target)
            verify_path = reports / f"verify-seed-{seed}.json"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "exp.text_realizer",
                    "verify",
                    "--checkpoint",
                    str(target),
                    "--device",
                    device,
                    "--cache-dir",
                    str(resolve(config["paths"]["cache"])),
                    "--output",
                    str(verify_path),
                ],
                cwd=ROOT,
                check=True,
            )
            verification = read_json(verify_path)
            if not verification["verified"] or verification[
                "checkpoint_sha256"
            ] != sha256_file(target):
                raise ValueError("fresh-process reload failed")
            report_name = f"tune-seed-{seed}.json"
            copy_atomic(source_reports / "factual_tune.json", reports / report_name)
            report_hashes[report_name] = sha256_file(reports / report_name)
            report_hashes[verify_path.name] = sha256_file(verify_path)
            checkpoints.append(
                {"seed": seed, "file": target.name, "sha256": sha256_file(target)}
            )
        frozen = {
            "format_version": 1,
            "run_id": run_id,
            "variant": variant,
            "deployment_seed": 42,
            "checkpoints": checkpoints,
            "reports_sha256": report_hashes,
            "reload_verified": True,
            "prepared_manifest_sha256": sha256_file(prepared / "manifest.json"),
            "plan_sha256": sha256_file(reports / "plan.json"),
            "used_counterfactual_text": False,
            "used_counterfactual_latent": False,
            "used_official_test_for_selection": False,
            "oracle_gate": "sealed",
        }
        immutable_json(reports / "frozen.json", frozen)
        return frozen


def frozen_checkpoints(config: dict, run_id: str) -> tuple[dict, dict[int, Path]]:
    from .setup import model_files

    models, reports = run_paths(config, run_id)
    frozen = require_frozen(reports)
    if frozen["plan_sha256"] != sha256_file(reports / "plan.json"):
        raise ValueError("frozen campaign plan changed")
    paths = {}
    base_files = model_files(config)
    for entry in frozen["checkpoints"]:
        if Path(entry["file"]).name != entry["file"]:
            raise ValueError("unsafe checkpoint reference")
        path = models / "selected" / entry["file"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("frozen weights changed")
        metadata = read_checkpoint(path)["metadata"]
        if metadata["provenance"]["base_files_sha256"] != base_files:
            raise ValueError("frozen decoder/tokenizer/encoder cache changed")
        paths[entry["seed"]] = path
    return frozen, paths


def audit(config: dict, run_id: str, *, device: str) -> dict:
    import gc
    import torch

    _, reports = run_paths(config, run_id)
    prepared = prepared_path(config)
    require_current(reports, config, prepared)
    frozen, paths = frozen_checkpoints(config, run_id)
    with lock(reports / ".audit.lock"):
        output = reports / "factual_audit.json"
        if output.exists():
            existing = read_json(output)
            if existing["frozen_sha256"] != sha256_file(reports / "frozen.json"):
                raise ValueError("audit belongs to a different freeze")
            for seed, expected in existing["seed_report_sha256"].items():
                if (
                    not str(seed).isdigit()
                    or sha256_file(reports / f"audit-seed-{seed}.json") != expected
                ):
                    raise ValueError("factual audit report changed")
            return existing
        part = load_bundle(prepared, audit=True, frozen_root=reports)["audit"]
        fit = load_bundle(prepared)["fit"]
        groups = load_groups(prepared, audit=True, frozen_root=reports)["audit"]
        fit_groups = load_groups(prepared)["fit"]
        results = {}
        for seed, checkpoint in paths.items():
            partial = reports / f"audit-seed-{seed}.json"
            if partial.exists():
                result = read_json(partial)
                if result["checkpoint_sha256"] != sha256_file(checkpoint):
                    raise ValueError("partial audit checkpoint changed")
            else:
                model = load_text_realizer(
                    checkpoint,
                    device=device,
                    cache_dir=str(resolve(config["paths"]["cache"])),
                )
                encoder = make_cycle_encoder(config, device)
                status(reports, "factual_audit", seed=seed, oracle_gate="sealed")
                result = evaluate_model(
                    model,
                    part,
                    fit,
                    encoder=encoder,
                    groups=groups,
                    fit_groups=fit_groups,
                    progress=progress_callback(reports, "factual_audit", seed=seed),
                )
                result["checkpoint_sha256"] = sha256_file(checkpoint)
                write_json(partial, result)
                del model, encoder
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            results[seed] = result
        deployment = results[frozen["deployment_seed"]]
        rows = deployment["conditions"]["correct"]["rows"]
        differences = [
            row["character_similarity"] - baseline["character_similarity"]
            for row, baseline in zip(rows, deployment["retrieval"]["rows"], strict=True)
        ]
        intervals = paired_bootstrap(
            differences,
            samples=config["evaluation"]["bootstrap_samples"],
            seed=config["evaluation"]["bootstrap_seed"],
        )
        values = [
            result["conditions"]["correct"]["summary"]["character_similarity"]
            for result in results.values()
        ]
        summary = {
            "complete": True,
            "frozen_sha256": sha256_file(reports / "frozen.json"),
            "deployment_seed": 42,
            "seeds": list(results),
            "character_similarity_mean": mean(values),
            "character_similarity_std": stdev(values),
            "retrieval_comparison": intervals,
            "improvement_passed": intervals["mean"]
            >= config["evaluation"]["minimum_retrieval_gain"]
            and intervals["lower_95"] > 0,
            "seed_report_sha256": {
                str(seed): sha256_file(reports / f"audit-seed-{seed}.json")
                for seed in results
            },
            "oracle_gate": "sealed",
            "quality_claim": "factual inversion experiment, not an exact-recovery or fairness guarantee",
        }
        examples(
            reports / "audit_examples.md",
            part,
            deployment,
            config["evaluation"]["examples"],
        )
        immutable_json(output, summary)
        campaign_summary(
            reports / "summary.md", read_json(reports / "selection.json"), summary
        )
        return summary
