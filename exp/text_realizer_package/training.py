"""One fresh factual-only fit; resumable at optimizer-update boundaries."""

from __future__ import annotations

import math
from time import perf_counter

import torch

from exp.text_realizer.experiment import enable_checkpointing
from exp.text_realizer.protocol import immutable_json, progress_callback, status
from src.artifact_io import save_torch, sha256_file, write_json
from src.text_realizer.artifacts import read_checkpoint, save_checkpoint
from src.text_realizer.model import build_model, model_spec
from src.text_realizer.training import (
    evaluate_nll,
    optimizer_for,
    restore_resume,
    save_resume,
    scheduler_for,
    seed_all,
    tokenize,
    update,
)

from .data import load_bundle, prepared_path
from .preflight import readiness
from .settings import digest, require_current, resolve, run_paths


def initial_progress(size: int, seed: int) -> dict:
    return {
        "epoch": 0,
        "offset": 0,
        "step": 0,
        "order": torch.randperm(
            size, generator=torch.Generator().manual_seed(seed)
        ).tolist(),
        "best": None,
        "patience_best": None,
        "bad_evaluations": 0,
        "epoch_sum": 0.0,
        "epoch_tokens": 0,
        "history": [],
        "best_checkpoint": None,
        "done": False,
    }


def train(config: dict, run_id: str, *, device: str, resume: bool = False) -> dict:
    models, reports = run_paths(config, run_id)
    prepared = prepared_path(config)
    plan = require_current(reports, config, prepared)
    ready = readiness(config, reports, device=device)
    spec = model_spec(config["model"], "lora-compact", ready["hidden_size"])
    training_plan = {
        "plan_sha256": digest(plan),
        "preflight_sha256": sha256_file(reports / "preflight/ready.json"),
        "spec": spec,
        "generation": ready["generation"],
        "microbatch": ready["microbatch"],
    }
    if (reports / "training_plan.json").exists() and not resume:
        raise ValueError("training already started; use --resume or a new run ID")
    immutable_json(reports / "training_plan.json", training_plan)
    signature = digest(training_plan)
    seed_all(config["seed"])
    torch.set_num_threads(1)
    model = build_model(
        spec,
        ready["generation"],
        device=device,
        cache_dir=resolve(config["paths"]["cache"]),
    )
    enable_checkpointing(model, True)
    bundle = load_bundle(prepared)
    fit, tune = bundle["fit"], bundle["tune"]
    targets = tokenize(model.tokenizer, fit["texts"])
    validation = tokenize(model.tokenizer, tune["texts"])
    settings = config["training"]
    total = math.ceil(len(targets) / settings["effective_batch"]) * settings["epochs"]
    optimizer = optimizer_for(model, settings)
    scheduler = scheduler_for(optimizer, total, settings["warmup_fraction"])
    progress = initial_progress(len(targets), config["seed"])
    last, best = models / "last.pt", models / "best.pt"
    if resume and last.exists():
        progress = restore_resume(last, model, optimizer, scheduler, identity=signature)
        # last.pt is the transaction boundary, including the committed best.
        # Repair a best.pt replaced immediately before an interrupted last save.
        if progress["best_checkpoint"] is not None and (
            not best.exists() or sha256_file(best) != progress["best_sha256"]
        ):
            save_torch(best, progress["best_checkpoint"])
            read_checkpoint(best)
            progress["best_sha256"] = sha256_file(best)
    started, first_step = perf_counter(), progress["step"]
    while not progress["done"]:
        while progress["offset"] < len(targets):
            chosen = progress["order"][
                progress["offset"] : progress["offset"] + settings["effective_batch"]
            ]
            record = update(
                model,
                optimizer,
                fit["z"],
                targets,
                chosen,
                microbatch=ready["microbatch"],
                clip=settings["clip"],
            )
            scheduler.step()
            progress["offset"] += len(chosen)
            progress["step"] += 1
            progress["epoch_sum"] += record["nll"] * record["tokens"]
            progress["epoch_tokens"] += record["tokens"]
            if progress["step"] % settings["checkpoint_every"] == 0:
                save_resume(
                    last, model, optimizer, scheduler, progress, identity=signature
                )
            seconds = (perf_counter() - started) / max(1, progress["step"] - first_step)
            status(
                reports,
                "training",
                epoch=progress["epoch"] + 1,
                step=progress["step"],
                nll=record["nll"],
                best=progress["best"],
                seconds_per_update=seconds,
                eta_training_seconds_at_maximum_epochs=max(0, total - progress["step"])
                * seconds,
            )
        epoch = progress["epoch"] + 1
        measured = evaluate_nll(
            model,
            tune["z"],
            validation,
            microbatch=ready["microbatch"],
            progress=progress_callback(reports, "validation", epoch=epoch),
        )["nll"]
        if progress["best"] is None or measured < progress["best"]:
            save_checkpoint(
                best,
                model,
                encoder=config["encoder"],
                provenance={
                    "plan_sha256": digest(plan),
                    "training_plan_sha256": signature,
                    "epoch": epoch,
                    "tune_nll": measured,
                    "seed": config["seed"],
                    "stage": "package-factual-training",
                    "corpus": "talent_motivation",
                    "initialization": plan["initialization"],
                    "prepared_manifest_sha256": plan["identity"][
                        "prepared_manifest_sha256"
                    ],
                    "used_counterfactual_text": False,
                    "used_counterfactual_latent": False,
                    "used_official_test_for_selection": False,
                    "base_files_sha256": ready["base_files_sha256"],
                },
            )
            progress["best"] = measured
            progress["best_sha256"] = sha256_file(best)
            progress["best_checkpoint"] = read_checkpoint(best)
        if (
            progress["patience_best"] is None
            or measured < progress["patience_best"] - settings["min_delta"]
        ):
            progress["patience_best"], progress["bad_evaluations"] = measured, 0
        else:
            progress["bad_evaluations"] += 1
        progress["history"].append(
            {
                "epoch": epoch,
                "step": progress["step"],
                "tune_nll": measured,
                "fit_nll": progress["epoch_sum"] / progress["epoch_tokens"],
            }
        )
        progress.update(
            epoch=epoch,
            offset=0,
            epoch_sum=0.0,
            epoch_tokens=0,
            done=epoch >= settings["epochs"]
            or progress["bad_evaluations"] >= settings["patience"],
        )
        progress["order"] = torch.randperm(
            len(targets),
            generator=torch.Generator().manual_seed(config["seed"] + epoch),
        ).tolist()
        save_resume(last, model, optimizer, scheduler, progress, identity=signature)
        write_json(reports / "history.json", progress["history"])
        print(
            f"epoch={epoch} tune_nll={measured:.6f} best={progress['best']:.6f}",
            flush=True,
        )
    require_current(reports, config, prepared)
    result = {
        "complete": True,
        "epochs": progress["epoch"],
        "best_tune_nll": progress["best"],
        "checkpoint_sha256": sha256_file(best),
        "training_plan_sha256": signature,
    }
    write_json(reports / "training.json", result)
    status(reports, "trained", **result)
    return result
