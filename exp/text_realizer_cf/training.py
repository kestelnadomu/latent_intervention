"""One explicit 600-update continuation; evaluation data are not imported here."""

from __future__ import annotations

import gc
from time import perf_counter

import torch

from exp.text_realizer.experiment import enable_checkpointing
from src.artifact_io import save_torch, sha256_file, write_json
from src.text_realizer.artifacts import (
    read_checkpoint,
    save_checkpoint,
    verify_checkpoint,
)
from src.text_realizer.training import (
    optimizer_for,
    restore_resume,
    save_resume,
    scheduler_for,
    seed_all,
    tokenize,
    update,
)

from .data import load_partition
from .model import clear_gpu, load_reference
from .protocol import identity, lock, read_json, require_current, snapshot, status
from .sampling import ArmSampler
from .selection import evaluate_selection, selection_key
from .settings import digest, prepared_path, resolve, run_paths


def require_preflight(config: dict, prepared, device: str) -> dict:
    ready = read_json(prepared / "preflight.json")
    if not ready.get("ready") or ready.get("identity") != identity(config, prepared):
        raise ValueError("passing GPU preflight for this code/config is required")
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("a preflighted CUDA GPU is required")
    if torch.cuda.get_device_name(device) != ready["gpu_name"]:
        raise ValueError("training GPU differs from preflight")
    return ready


def train_arm(
    config: dict, run_id: str, arm: str, *, device: str, resume: bool = False
) -> dict:
    prepared = prepared_path(config)
    ready = require_preflight(config, prepared, device)
    fit = load_partition(prepared, "fit")
    select = load_partition(prepared, "selection")["factual"]
    models_root, reports_root = run_paths(config, run_id)
    models, reports = run_paths(config, run_id, arm)
    with lock(reports_root / ".plan.lock"):
        snapshot(
            reports_root,
            {
                "identity": identity(config, prepared),
                "kind": "counterfactual-exposed-text-decoder-pilot",
                "arms": ["factual", "oracle"],
                "seed": 42,
                "audit_gate": "closed",
                "official_test_gate": "sealed",
            },
        )
    with lock(reports / ".train.lock"):
        plan = {
            "identity": identity(config, prepared),
            "arm": arm,
            "seed": 42,
            "updates": 600,
            "microbatch": ready["microbatch"],
            "preflight_sha256": sha256_file(prepared / "preflight.json"),
        }
        if (reports / "plan.json").exists() and not resume:
            raise ValueError("arm already exists; use --resume or a new run ID")
        snapshot(reports, plan)
        if (reports / "training.json").exists():
            result = read_json(reports / "training.json")
            if result["checkpoint_sha256"] != sha256_file(models / "best.pt"):
                raise ValueError("completed arm checkpoint changed")
            return result
        try:
            return _fit(
                config,
                prepared,
                fit,
                select,
                models,
                reports,
                plan,
                ready,
                device,
                resume,
            )
        except BaseException as exc:
            status(
                reports,
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise


def _fit(config, prepared, fit, select, models, reports, plan, ready, device, resume):
    seed_all(plan["seed"])
    torch.set_num_threads(1)
    model = load_reference(config, device=device)
    enable_checkpointing(model, config["training"]["gradient_checkpointing"])
    factual, changed = fit["factual"], fit["changed"]
    if plan["arm"] == "oracle":
        vectors = torch.cat((factual["z"], changed["z"]))
        texts = factual["texts"] + changed["texts"]
    else:
        vectors, texts = factual["z"], factual["texts"]
    targets = tokenize(model.tokenizer, texts)
    sampler = ArmSampler(
        len(factual["ids"]), len(changed["ids"]), plan["arm"], plan["seed"]
    )
    settings = config["training"]
    optimizer = optimizer_for(model, settings)
    scheduler = scheduler_for(
        optimizer, settings["updates"], settings["warmup_fraction"]
    )
    progress = {
        "step": 0,
        "sampler": sampler.state_dict(),
        "history": [],
        "best_selection": None,
        "best_sha256": None,
        "best_checkpoint": None,
    }
    signature = digest(plan)
    last = models / "last.pt"
    if resume:
        if not last.exists():
            raise ValueError(
                "resume requested without a committed optimizer checkpoint"
            )
        progress = restore_resume(last, model, optimizer, scheduler, identity=signature)
        sampler.load_state_dict(progress["sampler"])
        if progress["best_checkpoint"] is not None:
            best = models / "best.pt"
            if not best.exists() or sha256_file(best) != progress["best_sha256"]:
                save_torch(best, progress["best_checkpoint"])
                read_checkpoint(best)
                progress["best_sha256"] = sha256_file(best)
    started, initial = perf_counter(), progress["step"]
    while progress["step"] < settings["updates"]:
        indices = sampler.take_batch()
        record = update(
            model,
            optimizer,
            vectors,
            targets,
            indices,
            microbatch=ready["microbatch"],
            clip=settings["clip"],
        )
        scheduler.step()
        progress["step"] += 1
        progress["sampler"] = sampler.state_dict()
        step = progress["step"]
        if step % settings["evaluate_every"] == 0:
            selection = evaluate_selection(
                model,
                select,
                config["selection"]["factual_fields"],
                progress=lambda event, done, total: (
                    status(
                        reports,
                        "factual_selection",
                        step=step,
                        event=event,
                        completed=done,
                        total=total,
                    )
                    if done == total or done % 20 == 0
                    else None
                ),
            )
            selection["step"] = step
            progress["history"].append({**selection, "last_fit_nll": record["nll"]})
            if progress["best_selection"] is None or selection_key(
                selection
            ) > selection_key(progress["best_selection"]):
                save_checkpoint(
                    models / "best.pt",
                    model,
                    encoder=read_json(prepared / "manifest.json")["encoder"],
                    provenance={
                        "plan_sha256": signature,
                        "step": step,
                        "seed": 42,
                        "arm": plan["arm"],
                        "selection": selection,
                        "prepared_manifest_sha256": plan["identity"][
                            "prepared_manifest_sha256"
                        ],
                        "baseline_checkpoint_sha256": config["inputs"]["sha256"][
                            "baseline_checkpoint"
                        ],
                        "used_counterfactual_text": plan["arm"] == "oracle",
                        "used_counterfactual_latent": plan["arm"] == "oracle",
                        "used_audit_for_selection": False,
                        "used_official_test_for_selection": False,
                        "base_files_sha256": read_checkpoint(
                            resolve(config["inputs"]["baseline_checkpoint"])
                        )["metadata"]["provenance"].get("base_files_sha256", {}),
                    },
                )
                progress["best_selection"] = selection
                progress["best_sha256"] = sha256_file(models / "best.pt")
                progress["best_checkpoint"] = read_checkpoint(models / "best.pt")
            write_json(reports / "history.json", progress["history"])
        if step % settings["checkpoint_every"] == 0 or step == settings["updates"]:
            save_resume(last, model, optimizer, scheduler, progress, identity=signature)
        elapsed = (perf_counter() - started) / max(1, step - initial)
        status(
            reports,
            "training",
            step=step,
            updates=settings["updates"],
            arm=plan["arm"],
            last_nll=record["nll"],
            best_selection=progress["best_selection"],
            eta_seconds_upper_bound=(settings["updates"] - step) * elapsed,
        )
    require_current(reports, config, prepared)
    require_current(reports.parent, config, prepared)
    best = models / "best.pt"
    best_hash = sha256_file(best)
    chosen = progress["best_selection"]
    del model, optimizer, scheduler
    gc.collect()
    clear_gpu(device)
    verification = verify_checkpoint(
        best, device=device, cache_dir=str(resolve(config["paths"]["cache"]))
    )
    if not verification["verified"] or verification["checkpoint_sha256"] != best_hash:
        raise ValueError("fresh checkpoint reload failed")
    write_json(reports / "verify.json", verification)
    result = {
        "complete": True,
        "arm": plan["arm"],
        "updates": settings["updates"],
        "selected_step": chosen["step"],
        "selection": chosen,
        "checkpoint_sha256": best_hash,
        "reload_verified": True,
        "used_counterfactual_text": plan["arm"] == "oracle",
        "used_counterfactual_latent": plan["arm"] == "oracle",
    }
    write_json(reports / "training.json", result)
    status(reports, "trained", **result)
    return result
