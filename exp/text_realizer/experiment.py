"""One bounded fit, with immutable provenance and optimizer-boundary recovery."""

from __future__ import annotations

import math
from time import perf_counter

import torch

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
from .protocol import (
    identity,
    lock,
    progress_callback,
    read_json,
    require_current,
    snapshot,
    status,
)
from .settings import digest, resolve, run_paths


def enable_checkpointing(model, enabled: bool) -> None:
    if enabled:
        model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )


def readiness(config: dict, prepared, device: str) -> dict:
    ready = read_json(prepared / "preflight.json")
    if not ready["ready"] or ready["identity"] != identity(config, prepared):
        raise ValueError(
            "passing preflight for current code/config/environment is required"
        )
    if (
        not str(device).startswith("cuda")
        or torch.cuda.get_device_name(device) != ready["gpu_name"]
    ):
        raise ValueError("training hardware differs from GPU preflight")
    from .setup import model_files

    if ready["base_files_sha256"] != model_files(config):
        raise ValueError(
            "cached decoder/tokenizer/encoder files changed after preflight"
        )
    return ready


def train(
    config: dict,
    run_id: str,
    variant: str,
    seed: int,
    *,
    device: str,
    resume: bool = False,
    smoke: bool = False,
) -> dict:
    if "pilot" in config and (variant != "lora-compact" or seed != 42):
        raise ValueError("the 1B pilot authorizes only lora-compact / seed 42")
    prepared = prepared_path(config)
    bundle = load_bundle(prepared)
    models, reports = run_paths(config, run_id)
    with lock(reports / ".train.lock"):
        if smoke:
            from .preflight import token_audit

            ready = token_audit(config, bundle)
            ready["microbatch"] = 1
        else:
            ready = readiness(config, prepared, device)
        spec = model_spec(config["model"], variant, ready["hidden_size"])
        plan = {
            "identity": identity(config, prepared),
            "variant": variant,
            "seed": seed,
            "stage": "smoke" if smoke else "train",
            "model": spec,
            "generation": ready["generation"],
            "microbatch": ready["microbatch"],
            "preflight_sha256": (
                None if smoke else sha256_file(prepared / "preflight.json")
            ),
        }
        existed = (reports / "plan.json").exists()
        if existed and not resume:
            raise ValueError("run exists; use --resume or a new run ID")
        snapshot(reports, plan)
        try:
            return _fit(
                config, bundle, prepared, models, reports, plan, ready, device, resume
            )
        except BaseException as exc:
            status(
                reports,
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise


def _fit(config, bundle, prepared, models, reports, plan, ready, device, resume):
    seed_all(plan["seed"])
    torch.set_num_threads(1)
    model = build_model(
        plan["model"],
        plan["generation"],
        device=device,
        cache_dir=str(resolve(config["paths"]["cache"])),
    )
    enable_checkpointing(model, config["training"]["gradient_checkpointing"])
    fit, tune = bundle["fit"], bundle["tune"]
    smoke = plan["stage"] == "smoke"
    if smoke:
        fit = {key: value[:32] for key, value in fit.items()}
        tune = {key: value[:16] for key, value in tune.items()}
    targets, validation = (
        tokenize(model.tokenizer, fit["texts"]),
        tokenize(model.tokenizer, tune["texts"]),
    )
    settings = config["training"]
    epochs = config["preflight"]["smoke_updates"] if smoke else settings["epochs"]
    effective = min(32, len(targets)) if smoke else settings["effective_batch"]
    total_updates = math.ceil(len(targets) / effective) * epochs
    optimizer = optimizer_for(model, settings)
    scheduler = scheduler_for(optimizer, total_updates, settings["warmup_fraction"])
    progress = {
        "epoch": 0,
        "offset": 0,
        "step": 0,
        "best": None,
        "patience_best": None,
        "bad_evaluations": 0,
        "history": [],
        "epoch_sum": 0.0,
        "epoch_tokens": 0,
        "done": False,
        "best_checkpoint": None,
    }
    progress["order"] = torch.randperm(
        len(targets), generator=torch.Generator().manual_seed(plan["seed"])
    ).tolist()
    signature = digest(plan)
    last = models / "last.pt"
    if resume and last.exists():
        progress = restore_resume(last, model, optimizer, scheduler, identity=signature)
        if progress.get("best_checkpoint") is not None:
            # last is the transaction boundary; it carries its committed best.
            # An interruption between replacing best and saving last rolls back
            # to that committed pair before replaying outstanding updates.
            best_path = models / "best.pt"
            if (
                not best_path.exists()
                or sha256_file(best_path) != progress["best_sha256"]
            ):
                save_torch(best_path, progress["best_checkpoint"])
                read_checkpoint(best_path)
                progress["best_sha256"] = sha256_file(best_path)
    started, first_step = perf_counter(), progress["step"]
    while not progress["done"]:
        order = progress["order"]
        while progress["offset"] < len(order):
            chosen = order[progress["offset"] : progress["offset"] + effective]
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
                eta_seconds_upper_bound=max(0, total_updates - progress["step"])
                * seconds,
            )
        measured = evaluate_nll(
            model,
            tune["z"],
            validation,
            microbatch=ready["microbatch"],
            progress=progress_callback(
                reports, "validation", epoch=progress["epoch"] + 1
            ),
        )["nll"]
        epoch = progress["epoch"] + 1
        if progress["best"] is None or measured < progress["best"]:
            save_checkpoint(
                models / "best.pt",
                model,
                encoder=config["encoder"],
                provenance={
                    "plan_sha256": signature,
                    "epoch": epoch,
                    "tune_nll": measured,
                    "seed": plan["seed"],
                    "stage": plan["stage"],
                    "prepared_manifest_sha256": plan["identity"][
                        "prepared_manifest_sha256"
                    ],
                    "used_counterfactual_text": False,
                    "used_counterfactual_latent": False,
                    "used_official_test_for_selection": False,
                    "base_files_sha256": ready.get("base_files_sha256", {}),
                    **(
                        {"audit_informed": True, "evaluation_scope": "tune-only-pilot"}
                        if "pilot" in config
                        else {}
                    ),
                },
            )
            progress["best"] = measured
            progress["best_sha256"] = sha256_file(models / "best.pt")
            progress["best_checkpoint"] = read_checkpoint(models / "best.pt")
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
            done=epoch >= epochs
            or (not smoke and progress["bad_evaluations"] >= settings["patience"]),
        )
        progress["order"] = torch.randperm(
            len(targets), generator=torch.Generator().manual_seed(plan["seed"] + epoch)
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
        "stage": plan["stage"],
        "epochs": progress["epoch"],
        "best_tune_nll": progress["best"],
        "checkpoint_sha256": sha256_file(models / "best.pt"),
    }
    write_json(reports / "training.json", result)
    status(reports, "trained", **result)
    return result
