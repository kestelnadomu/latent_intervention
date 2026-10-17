"""One shared factual objective, validation selector and resumable optimizer loop."""

import copy
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch

from src.artifact_io import save_torch, sha256_file, write_json
from . import load_model
from .capacity import capacity_report
from .common import probabilities
from .objective import terms_for, total_loss
from .protocol import FittingData, digest, support_report
from src.latent_intervention.flow.common import _state_tensor


def prepare(model, data, decoder, settings):
    decoder.eval().requires_grad_(False)
    model.fit_normalization(data.fit.z)
    order = torch.randperm(
        len(data.fit.z),
        generator=torch.Generator().manual_seed(settings["support_seed"]),
    )
    bank_ids = order[: settings["support_bank_size"]]
    bank = ((data.fit.z[bank_ids] - model.mean) / model.scale).detach()
    size = len(data.matrix)
    matrix = probabilities(data.matrix, rows=size, states=size)
    with torch.no_grad():
        targets = {
            name: probabilities(
                decoder.log_joint(getattr(data, name).z).exp() @ matrix,
                rows=len(getattr(data, name).z),
                states=size,
            )
            for name in ("fit", "tune")
        }
    return bank, targets, bank_ids


def batch_terms(model, part, indices, target, decoder, bank, *, create_graph):
    x_index = next(i for i, c in enumerate(model.columns) if c.name == "X")
    return terms_for(
        model,
        part.z[indices],
        target[indices],
        part.states[indices, x_index] == 3,
        decoder,
        bank,
        create_graph=create_graph,
    )


def accumulate(totals, denominators, terms, count, identity_count):
    for name, value in terms.items():
        weight = identity_count if name == "identity" else count
        totals[name] = totals.get(name, 0.0) + float(value.detach()) * weight
        denominators[name] = denominators.get(name, 0) + weight


def averages(totals, denominators, settings):
    result = {
        name: value / max(denominators[name], 1) for name, value in totals.items()
    }
    result["objective"] = float(total_loss(result, settings))
    if not all(np.isfinite(v) for v in result.values()):
        raise FloatingPointError("nonfinite editor loss")
    return result


@torch.no_grad()
def validate(model, part, target, decoder, bank, settings):
    model.eval()
    totals, denominators = {}, {}
    x_index = next(i for i, c in enumerate(model.columns) if c.name == "X")
    for idx in torch.arange(len(part.z)).split(settings["batch_size"]):
        terms = batch_terms(model, part, idx, target, decoder, bank, create_graph=False)
        accumulate(
            totals,
            denominators,
            terms,
            len(idx),
            int((part.states[idx, x_index] == 3).sum()),
        )
    return averages(totals, denominators, settings)


def rng_state(generator):
    np_state = np.random.get_state()
    return dict(
        torch=torch.get_rng_state(),
        order=generator.get_state(),
        python=random.getstate(),
        numpy=(np_state[0], np_state[1].tolist(), *np_state[2:]),
    )


def restore_rng(saved, generator):
    torch.set_rng_state(saved["torch"])
    generator.set_state(saved["order"])
    random.setstate(saved["python"])
    state = saved["numpy"]
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))


def fit(
    model,
    data,
    decoder,
    settings,
    checkpoint,
    report_path,
    resume_path,
    metadata,
    *,
    on_epoch=None,
):
    """Resume at epoch boundaries, then deploy the lowest factual-validation objective."""
    if not isinstance(data, FittingData):
        raise TypeError("editor fitting accepts only factual FittingData")
    if tuple(model.columns) != data.columns or list(decoder.columns) != list(
        data.columns
    ):
        raise ValueError("editor, data and g schemas differ")
    if set(data.fit.ids) & set(data.tune.ids):
        raise ValueError("overlapping fitting/tuning IDs")
    for part in (data.fit, data.tune):
        if (
            not len(part.z)
            or len(part.z) != len(part.ids)
            or len(set(part.ids)) != len(part.ids)
        ):
            raise ValueError("invalid factual IDs")
        model.validate_z(part.z)
        _state_tensor(
            part.states, data.columns, batch_size=len(part.z), device=part.z.device
        )
    if model.mean.device.type != "cpu":
        raise ValueError("reproducible editor fitting currently requires CPU")
    capacity = capacity_report(
        model, tier=settings["tier"], enforce=model.latent_dim in (256, 768)
    )
    checkpoint, report_path, resume_path = map(
        Path, (checkpoint, report_path, resume_path)
    )
    signature = digest(
        dict(metadata=metadata, constructor=model.constructor(), settings=settings)
    )
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if (
            report["signature"] != signature
            or report["record_sha256"]
            != digest({k: v for k, v in report.items() if k != "record_sha256"})
            or report["checkpoint_sha256"] != sha256_file(checkpoint)
        ):
            raise ValueError("completed editor provenance mismatch")
        selected = load_model(
            checkpoint, expected_variant=model.variant, expected_metadata=metadata
        )
        model.load_state_dict(selected.state_dict())
        return report
    if checkpoint.exists():
        if not resume_path.exists():
            raise FileExistsError(
                "selected editor exists without a report or resume snapshot"
            )
        load_model(
            checkpoint, expected_variant=model.variant, expected_metadata=metadata
        )
    started = perf_counter()
    torch.manual_seed(settings["seed"])
    np.random.seed(settings["seed"])
    random.seed(settings["seed"])
    generator = torch.Generator().manual_seed(settings["seed"])
    bank, targets, bank_ids = prepare(model, data, decoder, settings)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=settings["scheduler_patience"],
        factor=settings["scheduler_factor"],
        min_lr=settings["min_lr"],
        threshold=settings["min_delta"],
        threshold_mode="abs",
    )
    history, stale, best_epoch, prior_seconds, updates = [], 0, 0, 0.0, 0
    initial = validate(model, data.tune, targets["tune"], decoder, bank, settings)
    best, best_state = initial["objective"], copy.deepcopy(model.state_dict())

    def snapshot():
        save_torch(
            resume_path,
            dict(
                format="semantic_editor_resume_v1",
                signature=signature,
                model=model.state_dict(),
                best_state=best_state,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                epoch=len(history),
                rng=rng_state(generator),
                history=history,
                initial_validation=initial,
                best=best,
                best_epoch=best_epoch,
                stale=stale,
                optimizer_updates=updates,
                elapsed_seconds=prior_seconds + perf_counter() - started,
            ),
        )

    if resume_path.exists():
        saved = torch.load(resume_path, weights_only=True, map_location="cpu")
        if (
            saved["format"] != "semantic_editor_resume_v1"
            or saved["signature"] != signature
        ):
            raise ValueError("editor resume signature mismatch")
        if saved["epoch"] != len(saved["history"]):
            raise ValueError("invalid resume epoch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"], generator)
        history, initial, best_state = (
            saved["history"],
            saved["initial_validation"],
            saved["best_state"],
        )
        best, best_epoch, stale = saved["best"], saved["best_epoch"], saved["stale"]
        prior_seconds, updates = saved["elapsed_seconds"], saved["optimizer_updates"]
    else:
        snapshot()  # Even an interruption in epoch one can be restarted exactly.
    x_index = next(i for i, c in enumerate(model.columns) if c.name == "X")
    for epoch in range(len(history), settings["max_epochs"]):
        if stale >= settings["patience"]:
            break
        epoch_start = perf_counter()
        model.train()
        totals, denominators, max_gradient = {}, {}, 0.0
        for idx in torch.randperm(len(data.fit.z), generator=generator).split(
            settings["batch_size"]
        ):
            optimizer.zero_grad(set_to_none=True)
            terms = batch_terms(
                model, data.fit, idx, targets["fit"], decoder, bank, create_graph=True
            )
            loss = total_loss(terms, settings)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings["grad_clip"], error_if_nonfinite=True
            )
            max_gradient = max(max_gradient, float(norm))
            optimizer.step()
            updates += 1
            accumulate(
                totals,
                denominators,
                terms,
                len(idx),
                int((data.fit.states[idx, x_index] == 3).sum()),
            )
        validation = validate(
            model, data.tune, targets["tune"], decoder, bank, settings
        )
        value = validation["objective"]
        significant = value < best - settings["min_delta"]
        selected = (
            value < best
        )  # Store the actual minimum, not a tolerance-rounded minimum.
        if selected:
            best, best_epoch, best_state = (
                value,
                epoch + 1,
                copy.deepcopy(model.state_dict()),
            )
        stale = 0 if significant else stale + 1
        scheduler.step(value)
        history.append(
            dict(
                epoch=epoch + 1,
                train=averages(totals, denominators, settings),
                validation=validation,
                selected=selected,
                gradient_norm_max=max_gradient,
                learning_rate=optimizer.param_groups[0]["lr"],
                elapsed_seconds=perf_counter() - epoch_start,
            )
        )
        snapshot()
        print(
            f"{model.variant}/{model.latent_dim}/seed-{settings['seed']} epoch={epoch + 1} validation={value:.7g} best={best:.7g}",
            flush=True,
        )
        if on_epoch is not None:
            on_epoch(epoch + 1)
    model.load_state_dict(best_state, strict=True)
    model.eval()
    selected_validation = validate(
        model, data.tune, targets["tune"], decoder, bank, settings
    )
    if abs(selected_validation["objective"] - best) > 1e-6:
        raise ValueError("selected-checkpoint restoration failed")
    model.save(checkpoint, metadata=metadata)
    report = dict(
        signature=signature,
        variant=model.variant,
        dimension=model.latent_dim,
        seed=settings["seed"],
        selected_epoch=best_epoch,
        selected_validation=selected_validation,
        initial_validation=initial,
        history=history,
        epochs_completed=len(history),
        optimizer_updates=updates,
        early_stopped=stale >= settings["patience"],
        elapsed_seconds=prior_seconds + perf_counter() - started,
        capacity=capacity,
        support=support_report(data, targets["tune"]),
        frozen_auxiliary_parameters=dict(
            g=sum(p.numel() for p in decoder.parameters()), h_s=0
        ),
        support_bank_ids=[data.fit.ids[i] for i in bank_ids.tolist()],
        selection="lowest factual-validation objective, including epoch zero",
        used_auxiliary_labels=False,
        used_oracle_targets=False,
        used_official_test=False,
        checkpoint_sha256=sha256_file(checkpoint),
    )
    report["record_sha256"] = digest(report)
    write_json(report_path, report)
    return report
