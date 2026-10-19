"""Individual factual-only fits, deterministic validation and epoch-boundary resume."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from time import perf_counter

import torch

from src.artifact_io import save_torch, sha256_file, write_json
from src.schema import flat_state_index
from .capacity import capacity_report
from .common import coupled_weights
from .protocol import FittingData, digest, support_report
from .transport import sinkhorn


def groups(states, columns, size):
    index = flat_state_index(states, columns)
    return [(index == s).nonzero().flatten() for s in range(size)]


def draw_group(x, indices, batch_size, rng, smoothing_std):
    selected = indices[torch.randint(len(indices), (batch_size,), generator=rng)]
    points = x[selected]
    return points + smoothing_std * torch.randn(
        points.shape, generator=rng, dtype=points.dtype
    )


def flow_batch(x, indices, state, settings, rng):
    """Every pair in this OT problem has ONE fixed state; no cross-state matching."""
    target = draw_group(
        x, indices, settings["batch_size"], rng, settings["smoothing_std"]
    )
    reference = torch.randn(target.shape, generator=rng, dtype=target.dtype)
    plan, diagnostics = sinkhorn(reference, target, **settings["sinkhorn"])
    picks = torch.multinomial(
        plan.flatten(), len(target), replacement=True, generator=rng
    )
    u, endpoint = reference[picks // len(target)], target[picks % len(target)]
    t = torch.rand(len(target), generator=rng, dtype=target.dtype)
    return dict(
        x=(1 - t[:, None]) * u + t[:, None] * endpoint,
        time=t,
        states=state.expand(len(target), -1),
        velocity=endpoint - u,
        diagnostics=diagnostics,
    )


def validation_bank(model, data, settings):
    """Fixed augmentation/noise/times/couplings; independent from fitting RNG."""
    x = (data.tune.z - model.mean) / model.scale
    bank = groups(data.tune.states, data.columns, len(model.states))
    rng = torch.Generator().manual_seed(settings["seed"] + 3001)
    return [
        (len(ids) / len(x), flow_batch(x, ids, model.states[s], settings, rng))
        for s, ids in enumerate(bank)
        if len(ids)
    ]


@torch.no_grad()
def validation_score(model, bank, settings):
    result = sum(
        weight
        * float(
            (model.velocity(b["x"], b["time"], b["states"]) - b["velocity"])
            .square()
            .mean()
        )
        for weight, b in bank
    )
    if not math.isfinite(result):
        raise FloatingPointError("nonfinite OT validation loss")
    return result


def optimizer_step(loss, optimizer, parameters, clip):
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite OT training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    optimizer.step()


def fit(
    model,
    data,
    settings,
    checkpoint,
    report_path,
    resume_path,
    metadata,
    *,
    epoch_callback=None,
):
    if not isinstance(data, FittingData):
        raise TypeError("OT fitting accepts only the factual FittingData boundary")
    if (
        set(data.fit.ids) & set(data.tune.ids)
        or len(set(data.fit.ids)) != len(data.fit.ids)
        or len(set(data.tune.ids)) != len(data.tune.ids)
        or len(data.fit.ids) != len(data.fit.z)
        or len(data.tune.ids) != len(data.tune.z)
        or tuple(model.columns) != data.columns
    ):
        raise ValueError("invalid/overlapping OT fitting and tuning partitions")
    model.validate_z(data.fit.z)
    model.validate_z(data.tune.z)
    model.validate_states(data.fit.states, len(data.fit.z))
    model.validate_states(data.tune.states, len(data.tune.z))
    if model.mean.device.type != "cpu":
        raise ValueError("the reproducible OT fitting workflow currently uses CPU")
    coupled_weights(torch.ones(1, len(model.states)) / len(model.states), data.matrix)
    signature = digest(dict(metadata=metadata, constructor=model.constructor()))
    checkpoint, report_path, resume_path = map(
        Path, (checkpoint, report_path, resume_path)
    )
    if report_path.exists():
        record = json.loads(report_path.read_text())
        if record["signature"] != signature:
            raise ValueError("OT run signature changed; choose a new run ID")
        if record["status"] == "complete":
            if record["checkpoint_sha256"] != sha256_file(checkpoint):
                raise ValueError("selected OT checkpoint checksum mismatch")
            return record
        if not resume_path.exists():
            raise FileNotFoundError("incomplete OT run is missing its resume snapshot")
    model.fit_normalization(data.fit.z, data.fit.states)
    capacity = capacity_report(
        model, tier=settings["tier"], enforce=model.latent_dim in (256, 768)
    )
    support = support_report(data)
    started = perf_counter()
    common = dict(
        signature=signature,
        variant=model.variant,
        dimension=model.latent_dim,
        seed=settings["seed"],
        capacity=capacity,
        support=support,
        non_oracle=True,
        preservation="audit_only",
    )
    if model.variant == "ot_gaussian":
        model.fit(data.fit.z, data.fit.states)
        model.save(checkpoint, metadata=metadata)
        record = dict(
            common,
            status="complete",
            selection="analytical fit; no epoch selection",
            factual_validation_nll=model.nll(data.tune.z, data.tune.states),
            history=[],
            optimizer_updates=0,
            elapsed_seconds=perf_counter() - started,
            checkpoint_sha256=sha256_file(checkpoint),
        )
        write_json(report_path, record)
        return record

    x = ((data.fit.z - model.mean) / model.scale).detach()
    bank = groups(data.fit.states, data.columns, len(model.states))
    validation = validation_bank(model, data, settings)
    probabilities = model.fit_counts.float()
    optimizers = [torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])]
    schedulers = [
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            patience=settings["scheduler_patience"],
            factor=settings["scheduler_factor"],
            threshold=settings["min_delta"],
            threshold_mode="abs",
            min_lr=settings["min_lr"],
        )
        for opt in optimizers
    ]
    rng = torch.Generator().manual_seed(settings["seed"] + 2001)
    history, stale, start_epoch, elapsed, updates = [], 0, 0, 0.0, 0
    best = validation_score(model, validation, settings)
    reference, best_epoch, best_state = best, 0, copy.deepcopy(model.state_dict())
    initial_validation = best
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        if saved.get("format") != "ot_resume_v1" or saved["signature"] != signature:
            raise ValueError("OT resume signature mismatch")
        model.load_state_dict(saved["current"], strict=True)
        for opt, state in zip(optimizers, saved["optimizers"], strict=True):
            opt.load_state_dict(state)
        for scheduler, state in zip(schedulers, saved["schedulers"], strict=True):
            scheduler.load_state_dict(state)
        rng.set_state(saved["rng"])
        torch.set_rng_state(saved["torch_rng"])
        history, stale, start_epoch = saved["history"], saved["stale"], saved["epoch"]
        best, reference, best_epoch = (
            saved["best"],
            saved["reference"],
            saved["best_epoch"],
        )
        best_state, initial_validation = (
            saved["best_state"],
            saved["initial_validation"],
        )
        elapsed, updates = saved["elapsed_seconds"], saved["optimizer_updates"]
    steps = math.ceil(len(x) / settings["batch_size"])
    for epoch in range(start_epoch + 1, settings["max_epochs"] + 1):
        if stale >= settings["patience"]:
            break
        model.train()
        totals = {}
        for _ in range(steps):
            pick = int(torch.multinomial(probabilities, 1, generator=rng))
            batch = flow_batch(x, bank[pick], model.states[pick], settings, rng)
            loss = (
                (
                    model.velocity(batch["x"], batch["time"], batch["states"])
                    - batch["velocity"]
                )
                .square()
                .mean()
            )
            optimizer_step(
                loss, optimizers[0], model.parameters(), settings["grad_clip"]
            )
            values = dict(
                flow_matching=float(loss.detach()),
                sinkhorn_iterations=batch["diagnostics"]["iterations"],
                sinkhorn_marginal_error=batch["diagnostics"]["marginal_error"],
            )
            updates += 1
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + value / steps
        model.eval()
        score = validation_score(model, validation, settings)
        if not math.isfinite(score):
            raise FloatingPointError("nonfinite OT validation score")
        if score < best:
            best, best_epoch, best_state = (
                score,
                epoch,
                copy.deepcopy(model.state_dict()),
            )
        if score < reference - settings["min_delta"]:
            reference, stale = score, 0
        else:
            stale += 1
        for scheduler in schedulers:
            scheduler.step(score)
        row = dict(
            epoch=epoch,
            training=totals,
            validation=score,
            best_epoch=best_epoch,
            stale_epochs=stale,
            learning_rates=[opt.param_groups[0]["lr"] for opt in optimizers],
            outer_steps=steps,
            optimizer_updates=updates,
        )
        history.append(row)
        duration = elapsed + perf_counter() - started
        save_torch(
            resume_path,
            dict(
                format="ot_resume_v1",
                signature=signature,
                current=model.state_dict(),
                dual=None,  # Preserve the existing OT-flow resume format.
                optimizers=[opt.state_dict() for opt in optimizers],
                schedulers=[scheduler.state_dict() for scheduler in schedulers],
                rng=rng.get_state(),
                torch_rng=torch.get_rng_state(),
                history=history,
                stale=stale,
                epoch=epoch,
                best=best,
                reference=reference,
                best_epoch=best_epoch,
                best_state=best_state,
                initial_validation=initial_validation,
                elapsed_seconds=duration,
                optimizer_updates=updates,
            ),
        )
        write_json(
            report_path,
            dict(
                common,
                status="running",
                history=history,
                elapsed_seconds=duration,
                optimizer_updates=updates,
            ),
        )
        if epoch_callback is not None:
            epoch_callback(row)
    model.load_state_dict(best_state, strict=True)
    model.eval()
    model.save(checkpoint, metadata=metadata)
    record = dict(
        common,
        status="complete",
        history=history,
        initial_validation=initial_validation,
        best_validation=best,
        best_epoch=best_epoch,
        optimizer_updates=updates,
        selection="held-out flow-matching MSE",
        elapsed_seconds=elapsed + perf_counter() - started,
        stopped_early=len(history) < settings["max_epochs"],
        checkpoint_sha256=sha256_file(checkpoint),
    )
    write_json(report_path, record)
    return record
