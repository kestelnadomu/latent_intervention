"""Factual warmup and exact-mixture fitting, with validation selection and atomic resume."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from time import perf_counter

import torch

from src.artifact_io import save_torch, sha256_file, write_json
from .model import ProjectedCSym
from .objective import loss_terms, selection_key
from .protocol import digest


def terms_for(model, part, source_probs, matrix, decoder, scale, bank):
    z = part["z"]
    samples, weights = model.components(z, source_probs, matrix)
    terms = loss_terms(
        samples,
        weights,
        z,
        source_probs @ matrix,
        decoder,
        model.guide,
        scale,
        bank,
        part["labels"][:, 0] == 3,
    )
    terms["factual"] = model.factual_loss(z, part["labels"][:, : len(model.columns)])
    return terms


def objective(terms, settings):
    total = terms["semantic_kl"]
    for name in ("factual", "movement", "support", "identity", "radial", "entropy"):
        total = total + settings["loss"][name] * terms[name]
    return total


@torch.no_grad()
def validate(model, part, posterior, matrix, data, scale, bank, settings):
    model.eval()
    totals, denominators = {}, {}
    for indices in torch.arange(len(part["z"])).split(settings["batch_size"]):
        batch = {k: v[indices] for k, v in part.items()}
        terms = terms_for(
            model,
            batch,
            posterior[indices],
            matrix,
            data["decoder"],
            scale,
            bank,
        )
        for name, value in terms.items():
            size = (
                int((batch["labels"][:, 0] == 3).sum())
                if name == "identity"
                else len(indices)
            )
            totals[name] = totals.get(name, 0.0) + float(value) * size
            denominators[name] = denominators.get(name, 0) + size
    result = {key: value / max(denominators[key], 1) for key, value in totals.items()}
    if not all(torch.isfinite(torch.tensor(v)) for v in result.values()):
        raise FloatingPointError("nonfinite factual validation")
    return result


def warmup(model, data, settings, progress_path):
    """Fit the state effect to factual embeddings; restart deterministically if interrupted."""
    fit, tune = data["fit"], data["tune"]
    model.set_latent_statistics(fit["z"])
    optimizer = torch.optim.AdamW(
        model.effect.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    rng = torch.Generator().manual_seed(settings["seed"] + 901)
    best, state, stale, history = float("inf"), None, 0, []
    started = perf_counter()
    for epoch in range(settings["warmup_epochs"]):
        model.train()
        total = 0.0
        for indices in torch.randperm(len(fit["z"]), generator=rng).split(
            settings["batch_size"]
        ):
            loss = model.factual_loss(
                fit["z"][indices], fit["labels"][indices, : len(model.columns)]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.effect.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        model.eval()
        with torch.no_grad():
            validation = float(
                model.factual_loss(tune["z"], tune["labels"][:, : len(model.columns)])
            )
        if not torch.isfinite(torch.tensor(validation)):
            raise FloatingPointError("nonfinite factual warmup")
        improved = validation < best - 1e-6
        if improved:
            best, state, stale = validation, copy.deepcopy(model.effect.state_dict()), 0
        else:
            stale += 1
        history.append(
            dict(
                epoch=epoch + 1,
                factual_train=total / len(fit["z"]),
                factual_validation=validation,
                selected=improved,
            )
        )
        write_json(
            progress_path,
            dict(
                phase="factual_warmup",
                epoch=epoch + 1,
                elapsed_seconds=perf_counter() - started,
                validation=validation,
            ),
        )
        if stale >= settings["warmup_patience"]:
            break
    model.effect.load_state_dict(state)
    return history


def fit(data, guide, settings, checkpoint, report_path, metadata):
    """Fit one editor; resume epoch state and restore the validation-selected model."""
    started = perf_counter()
    checkpoint, report_path = Path(checkpoint), Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    signature = digest(metadata)
    resume_path = report_path.parent / "resume.pt"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if (
            report["signature"] != signature
            or report["record_sha256"]
            != digest({k: v for k, v in report.items() if k != "record_sha256"})
            or sha256_file(checkpoint) != report["checkpoint_sha256"]
        ):
            raise ValueError("completed training provenance mismatch")
        ProjectedCSym.load(checkpoint, expected_metadata=metadata)
        return report
    if checkpoint.exists():
        if not resume_path.exists():
            raise FileExistsError(
                "checkpoint exists without a completed training report"
            )
        # Recover the narrow interruption window between publishing the weights
        # and their report, but never overwrite an unrelated checkpoint.
        ProjectedCSym.load(checkpoint, expected_metadata=metadata)
    data["decoder"].eval().requires_grad_(False)
    guide.eval().requires_grad_(False)
    torch.manual_seed(settings["seed"])
    dimension = data["fit"]["z"].shape[1]
    model = ProjectedCSym(
        dimension,
        data["columns"],
        guide=guide,
        hidden_dim=settings["hidden_dim"],
        state_embed_dim=settings["state_embed_dim"],
    )
    job = f"embeddinggemma_{dimension}/c_sym_project"
    z = data["fit"]["z"]
    scale = z.std(0).clamp_min(1e-4)  # identical movement convention to the old pilot
    bank = z[
        torch.randperm(len(z), generator=torch.Generator().manual_seed(99))[
            : settings["support_bank_size"]
        ]
    ]
    with torch.no_grad():
        posterior = {
            part: data["decoder"].log_joint(data[part]["z"]).exp().detach()
            for part in ("fit", "tune")
        }
        matrix = data["kernel"].transition_matrix(model.intervention).to_dense()
    progress_path = report_path.parent / "progress.json"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5
    )
    rng = torch.Generator().manual_seed(settings["seed"])
    history, best_key, best_state, best_validation, best_epoch, stale, prior_seconds = (
        [],
        None,
        None,
        None,
        0,
        0,
        0.0,
    )
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        if saved["signature"] != signature or saved["job"] != job:
            raise ValueError("resume checkpoint provenance mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        rng.set_state(saved["rng"])
        torch.set_rng_state(saved["torch_rng"])
        history, warmup_history = saved["history"], saved["warmup_history"]
        best_key, best_state, best_validation = (
            saved["best_key"],
            saved["best_state"],
            saved["best_validation"],
        )
        best_epoch, stale, prior_seconds = (
            saved["best_epoch"],
            saved["stale"],
            saved["elapsed_seconds"],
        )
    else:
        warmup_history = warmup(model, data, settings, progress_path)
    for epoch in range(len(history), settings["max_epochs"]):
        if stale >= settings["patience"]:
            break
        epoch_start = perf_counter()
        model.train()
        totals, denominators = {}, {}
        for indices in torch.randperm(len(z), generator=rng).split(
            settings["batch_size"]
        ):
            batch = {k: v[indices] for k, v in data["fit"].items()}
            terms = terms_for(
                model,
                batch,
                posterior["fit"][indices],
                matrix,
                data["decoder"],
                scale,
                bank,
            )
            total = objective(terms, settings)
            if not torch.isfinite(total):
                raise FloatingPointError("nonfinite standalone objective")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            for name, value in {**terms, "total": total, "grad_norm": gradient}.items():
                size = (
                    int((batch["labels"][:, 0] == 3).sum())
                    if name == "identity"
                    else len(indices)
                )
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * size
                denominators[name] = denominators.get(name, 0) + size
        validation = validate(
            model,
            data["tune"],
            posterior["tune"],
            matrix,
            data,
            scale,
            bank,
            settings,
        )
        key = selection_key(validation, settings["semantic_tolerance"])
        scheduler.step(validation["semantic_kl"])
        improved = (
            best_key is None
            or key[0] < best_key[0]
            or (key[0] == best_key[0] and key[1] < best_key[1] - 1e-6)
        )
        if improved:
            best_key, best_state, best_validation = (
                key,
                copy.deepcopy(model.state_dict()),
                validation,
            )
            best_epoch, stale = epoch + 1, 0
        else:
            stale += 1
        elapsed = prior_seconds + perf_counter() - started
        history.append(
            dict(
                epoch=epoch + 1,
                train={k: v / max(denominators[k], 1) for k, v in totals.items()},
                validation=validation,
                learning_rate=optimizer.param_groups[0]["lr"],
                selected=improved,
                epoch_seconds=perf_counter() - epoch_start,
                seconds=elapsed,
            )
        )
        # Optimizer, RNG, scheduler, current AND best weights survive an interruption.
        save_torch(
            resume_path,
            dict(
                signature=signature,
                job=job,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                rng=rng.get_state(),
                torch_rng=torch.get_rng_state(),
                history=history,
                warmup_history=warmup_history,
                best_key=best_key,
                best_state=best_state,
                best_validation=best_validation,
                best_epoch=best_epoch,
                stale=stale,
                elapsed_seconds=elapsed,
            ),
        )
        write_json(
            progress_path,
            dict(
                job=job,
                phase="joint_fit",
                epoch=epoch + 1,
                best_epoch=best_epoch,
                stale_epochs=stale,
                validation=validation,
                epoch_seconds=history[-1]["epoch_seconds"],
                elapsed_seconds=elapsed,
            ),
        )
        print(
            f"{job} epoch={epoch + 1} KL={validation['semantic_kl']:.5f} factual={validation['factual']:.5f} seconds={history[-1]['epoch_seconds']:.2f}",
            flush=True,
        )
    model.load_state_dict(best_state)
    model.eval()
    model.save(checkpoint, metadata=metadata)
    restored = ProjectedCSym.load(checkpoint, expected_metadata=metadata)
    with torch.no_grad():
        a, aw = model.components(z[:3], posterior["fit"][:3], matrix)
        b, bw = restored.components(z[:3], posterior["fit"][:3], matrix)
        if not torch.equal(a, b) or not torch.equal(aw, bw):
            raise ValueError("saved C-sym checkpoint changed predictions")
    report = dict(
        signature=signature,
        job=job,
        dimension=dimension,
        variant="c_sym_project",
        parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256_file(checkpoint),
        best_epoch=best_epoch,
        epochs=len(history),
        warmup_history=warmup_history,
        history=history,
        selected_validation=best_validation,
        selection_key=list(best_key),
        semantic_feasible=best_key[0] == 0,
        seconds=prior_seconds + perf_counter() - started,
        stopped_early=len(history) < settings["max_epochs"],
        used_official_test=False,
        used_oracle_targets=False,
        used_flow_teacher=False,
        component_count=int((matrix > 0).sum()),
        roundtrip_verified=True,
    )
    report["record_sha256"] = digest(report)
    write_json(report_path, report)
    return report
