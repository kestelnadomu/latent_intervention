"""Experiment-local resumable loops; reuse the native losses and OT batches."""

import copy
import math
from pathlib import Path
from time import perf_counter

import torch

from src.artifact_io import save_torch, sha256_file
from src.latent_intervention.editor.objective import terms_for, total_loss
from src.latent_intervention.editor.training import (
    accumulate,
    averages,
    rng_state,
    restore_rng,
)
from src.latent_intervention.ot.training import (
    groups,
    flow_batch,
    validation_bank,
    validation_score,
)
from . import EDITORS
from .common import digest, exclusive, read_record, seeded, tensor_digest, write_record
from .data import FactualData, support_summary
from . import models


def prepare(model, data, settings, decoder=None, matrices=None):
    if not isinstance(data, FactualData) or data.role != "h_z":
        raise TypeError("h_Z fitting accepts only its factual visible-state boundary")
    if (
        tuple(model.columns) != data.columns
        or set(data.fit.ids) & set(data.tune.ids)
        or not len(data.fit.ids)
        or not len(data.tune.ids)
    ):
        raise ValueError("invalid schema or fitting/development split")
    for part in (data.fit, data.tune):
        if (
            len(part.ids) != len(part.z)
            or len(set(part.ids)) != len(part.ids)
            or part.states.shape != (len(part.z), len(data.columns))
            or not torch.isfinite(part.z).all()
            or part.z.device.type != "cpu"
        ):
            raise ValueError("invalid factual partition")
        for i, c in enumerate(data.columns):
            if (part.states[:, i] < 0).any() or (
                part.states[:, i] >= c.n_categories
            ).any():
                raise ValueError("invalid fitting state")
    if model.variant in EDITORS:
        if decoder is None or matrices is None or len(matrices) != 4:
            raise ValueError("editors require frozen g and all four action kernels")
        if tuple(decoder.columns) != data.columns:
            raise ValueError("guide/schema mismatch")
        decoder.eval().requires_grad_(False)
        model.fit_normalization(data.fit.z)
        rng = torch.Generator().manual_seed(settings["support_seed"])
        bank_ids = torch.randperm(len(data.fit.z), generator=rng)[
            : settings["support_bank_size"]
        ]
        bank = ((data.fit.z[bank_ids] - model.mean) / model.scale).detach()
        with torch.no_grad():
            targets = {}
            for name in ("fit", "tune"):
                probabilities = decoder.log_joint(getattr(data, name).z).exp()
                targets[name] = []
                for matrix in matrices:
                    if (
                        matrix.shape != (probabilities.shape[-1],) * 2
                        or not torch.isfinite(matrix).all()
                        or (matrix < 0).any()
                        or not torch.allclose(
                            matrix.sum(-1), torch.ones(len(matrix)), atol=1e-5
                        )
                    ):
                        raise ValueError("invalid training action matrix")
                    targets[name].append((probabilities @ matrix).detach())
        return dict(bank=bank, bank_ids=bank_ids, targets=targets)
    if decoder is not None or matrices is not None:
        raise ValueError(
            "factual renderers must not receive g or an operator in training"
        )
    if model.variant == "state_flow":
        model.set_latent_statistics(data.fit.z)
        return {}
    model.fit_normalization(data.fit.z, data.fit.states)
    if model.variant == "ot_gaussian":
        return {}
    return dict(
        x=((data.fit.z - model.mean) / model.scale).detach(),
        groups=groups(data.fit.states, data.columns, len(model.states)),
        validation=validation_bank(model, data, settings),
    )


def editor_terms(model, part, indices, target, action, decoder, bank, *, create_graph):
    x_index = next(i for i, col in enumerate(model.columns) if col.name == "X")
    return terms_for(
        model,
        part.z[indices],
        target[indices],
        part.states[indices, x_index] == action,
        decoder,
        bank,
        create_graph=create_graph,
    )


@torch.no_grad()
def validate(model, data, settings, prepared, decoder=None):
    model.eval()
    if model.variant in EDITORS:
        actions = []
        x_index = next(i for i, col in enumerate(model.columns) if col.name == "X")
        for action, target in enumerate(prepared["targets"]["tune"]):
            totals, denominators = {}, {}
            for indices in torch.arange(len(data.tune.z)).split(settings["batch_size"]):
                terms = editor_terms(
                    model,
                    data.tune,
                    indices,
                    target,
                    action,
                    decoder,
                    prepared["bank"],
                    create_graph=False,
                )
                accumulate(
                    totals,
                    denominators,
                    terms,
                    len(indices),
                    int((data.tune.states[indices, x_index] == action).sum()),
                )
            actions.append(averages(totals, denominators, settings))
        result = {k: sum(a[k] for a in actions) / 4 for k in actions[0]}
        result["by_action"] = actions
    elif model.variant == "state_flow":
        value = sum(
            float(-model.log_prob(data.tune.z[i], data.tune.states[i]).sum())
            for i in torch.arange(len(data.tune.z)).split(settings["batch_size"])
        )
        result = dict(objective=value / len(data.tune.z) / model.latent_dim)
    elif model.variant == "ot_gaussian":
        result = dict(objective=float(model.nll(data.tune.z, data.tune.states)))
    else:
        result = dict(
            objective=validation_score(model, prepared["validation"], settings)
        )
    if not math.isfinite(result["objective"]):
        raise FloatingPointError("non-finite development objective")
    return result


def one_epoch(model, data, settings, prepared, decoder, optimizer, generator):
    model.train()
    updates, losses, max_gradient = 0, [], 0.0
    if model.variant == "ot_flow":
        batches = range(math.ceil(len(data.fit.z) / settings["batch_size"]))
    else:
        batches = torch.randperm(len(data.fit.z), generator=generator).split(
            settings["batch_size"]
        )
    diagnostics = dict(sinkhorn_iterations=0, sinkhorn_marginal_error_max=0.0)
    for indices in batches:
        optimizer.zero_grad(set_to_none=True)
        if model.variant in EDITORS:
            value = 0.0
            # Four exact-mixture losses, one optimizer step; no sampled targets.
            for action, target in enumerate(prepared["targets"]["fit"]):
                terms = editor_terms(
                    model,
                    data.fit,
                    indices,
                    target,
                    action,
                    decoder,
                    prepared["bank"],
                    create_graph=True,
                )
                loss = total_loss(terms, settings) / 4
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite editor objective")
                loss.backward()
                value += float(loss.detach())
        else:
            if model.variant == "state_flow":
                loss = -model.log_prob(
                    data.fit.z[indices], data.fit.states[indices]
                ).mean()
            else:
                state = int(
                    torch.multinomial(model.fit_counts.float(), 1, generator=generator)
                )
                batch = flow_batch(
                    prepared["x"],
                    prepared["groups"][state],
                    model.states[state],
                    settings,
                    generator,
                )
                loss = (
                    (
                        model.velocity(batch["x"], batch["time"], batch["states"])
                        - batch["velocity"]
                    )
                    .square()
                    .mean()
                )
                diagnostics["sinkhorn_iterations"] += batch["diagnostics"]["iterations"]
                diagnostics["sinkhorn_marginal_error_max"] = max(
                    diagnostics["sinkhorn_marginal_error_max"],
                    batch["diagnostics"]["marginal_error"],
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite factual objective")
            loss.backward()
            value = float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), settings["grad_clip"], error_if_nonfinite=True
        )
        max_gradient = max(max_gradient, float(norm))
        optimizer.step()
        updates += 1
        losses.append(value)
    return dict(
        objective=sum(losses) / len(losses),
        updates=updates,
        action_objective_evaluations=updates * (4 if model.variant in EDITORS else 1),
        gradient_norm_max=max_gradient,
        **diagnostics,
    )


def fit(
    model,
    data,
    settings,
    checkpoint,
    report_path,
    resume_path,
    metadata,
    *,
    decoder=None,
    matrices=None,
    on_epoch=None,
):
    with exclusive(Path(report_path).with_name("fit.lock")):
        return _fit(
            model,
            data,
            settings,
            checkpoint,
            report_path,
            resume_path,
            metadata,
            decoder=decoder,
            matrices=matrices,
            on_epoch=on_epoch,
        )


def _fit(
    model,
    data,
    settings,
    checkpoint,
    report_path,
    resume_path,
    metadata,
    *,
    decoder=None,
    matrices=None,
    on_epoch=None,
):
    if not isinstance(data, FactualData) or data.role != "h_z":
        raise TypeError("h_Z fitting cannot consume audit/evaluation data")
    signature = digest(
        dict(
            metadata=metadata,
            settings=settings,
            architecture=models.architecture(model),
            factual=[
                tensor_digest(t) for p in (data.fit, data.tune) for t in (p.z, p.states)
            ],
            ids=[data.fit.ids, data.tune.ids],
            matrices=None if matrices is None else [tensor_digest(m) for m in matrices],
            guide=None
            if decoder is None
            else {k: tensor_digest(v) for k, v in decoder.state_dict().items()},
        )
    )
    checkpoint, report_path, resume_path = map(
        Path, (checkpoint, report_path, resume_path)
    )
    if report_path.exists():
        old = read_record(report_path)
        if old["signature"] != signature:
            raise ValueError("fit signature changed; choose a new run ID")
        if old["status"] == "complete":
            if sha256_file(checkpoint) != old["checkpoint_sha256"]:
                raise ValueError("checkpoint checksum mismatch")
            model.load_state_dict(
                models.load(checkpoint, signature=signature).state_dict()
            )
            prepared = prepare(model, data, settings, decoder, matrices)
            restored = validate(model, data, settings, prepared, decoder)["objective"]
            if abs(restored - old["selected_validation"]["objective"]) > 1e-5:
                raise ValueError("selected development score does not reproduce")
            return old
        if not resume_path.exists():
            raise FileNotFoundError("incomplete fit is missing its resume snapshot")
    if checkpoint.exists():
        models.load(checkpoint, signature=signature)
    if (
        checkpoint.exists()
        and not resume_path.exists()
        and model.variant != "ot_gaussian"
    ):
        raise FileExistsError(
            "selected checkpoint exists without complete record or resume"
        )
    started = perf_counter()
    seeded(settings["seed"])
    prepared = prepare(model, data, settings, decoder, matrices)
    common = dict(
        signature=signature,
        method=model.variant,
        seed=settings["seed"],
        capacity=models.capacity(model),
        support=support_summary(data),
        split_ids=dict(fit=list(data.fit.ids), tune=list(data.tune.ids)),
        training_regime="true factual states available; no counterfactual supervision",
        frozen_auxiliary_parameters=0
        if decoder is None
        else sum(p.numel() for p in decoder.parameters()),
        checkpoint=str(checkpoint),
        metadata=metadata,
    )
    model_metadata = dict(
        metadata,
        signature=signature,
        support=common["support"],
        split_ids=common["split_ids"],
    )
    if model.variant == "ot_gaussian":
        model.fit(data.fit.z, data.fit.states)
        selected = validate(model, data, settings, prepared)
        models.save(checkpoint, model, model_metadata)
        result = dict(
            common,
            status="complete",
            selected_epoch=None,
            selected_validation=selected,
            history=[],
            optimizer_updates=0,
            elapsed_seconds=perf_counter() - started,
            selection="analytical fit; factual development NLL is diagnostic, not epoch selection",
            checkpoint_sha256=sha256_file(checkpoint),
        )
        write_record(report_path, result)
        return read_record(report_path)
    optimizer = (
        torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
        if model.variant == "ot_flow"
        else torch.optim.AdamW(
            model.parameters(),
            lr=settings["learning_rate"],
            weight_decay=settings["weight_decay"],
        )
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=settings["scheduler_patience"],
        factor=settings["scheduler_factor"],
        min_lr=settings["min_lr"],
        threshold=settings["min_delta"],
        threshold_mode="abs",
    )
    generator = torch.Generator().manual_seed(
        settings["seed"] + (2001 if model.variant == "ot_flow" else 1)
    )
    initial = validate(model, data, settings, prepared, decoder)
    best, reference, best_epoch = initial["objective"], initial["objective"], 0
    best_state = copy.deepcopy(model.state_dict())
    best_training_state = dict(
        optimizer=copy.deepcopy(optimizer.state_dict()),
        scheduler=copy.deepcopy(scheduler.state_dict()),
        rng=rng_state(generator),
        epoch=0,
    )
    history, stale, updates, prior_seconds = [], 0, 0, 0.0

    def snapshot():
        save_torch(
            resume_path,
            dict(
                format="hz_misspecification_resume_v1",
                signature=signature,
                current=model.state_dict(),
                best_state=best_state,
                best_training_state=best_training_state,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                rng=rng_state(generator),
                history=history,
                initial=initial,
                best=best,
                reference=reference,
                best_epoch=best_epoch,
                stale=stale,
                optimizer_updates=updates,
                elapsed_seconds=prior_seconds + perf_counter() - started,
            ),
        )

    if resume_path.exists():
        previous = torch.load(resume_path, map_location="cpu", weights_only=True)
        if (
            previous.get("format") != "hz_misspecification_resume_v1"
            or previous["signature"] != signature
        ):
            raise ValueError("incompatible resume snapshot")
        model.load_state_dict(previous["current"], strict=True)
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        restore_rng(previous["rng"], generator)
        history, initial, best_state = (
            previous["history"],
            previous["initial"],
            previous["best_state"],
        )
        best_training_state = previous["best_training_state"]
        best, reference, best_epoch, stale = (
            previous[k] for k in ("best", "reference", "best_epoch", "stale")
        )
        updates, prior_seconds = (
            previous["optimizer_updates"],
            previous["elapsed_seconds"],
        )
    else:
        snapshot()
    for epoch in range(len(history), settings["max_epochs"]):
        if stale >= settings["patience"]:
            break
        epoch_start = perf_counter()
        training = one_epoch(
            model, data, settings, prepared, decoder, optimizer, generator
        )
        updates += training["updates"]
        validation = validate(model, data, settings, prepared, decoder)
        score = validation["objective"]
        if score < best:
            best, best_epoch, best_state = (
                score,
                epoch + 1,
                copy.deepcopy(model.state_dict()),
            )
        if score < reference - settings["min_delta"]:
            reference, stale = score, 0
        else:
            stale += 1
        scheduler.step(score)
        if best_epoch == epoch + 1:
            best_training_state = dict(
                optimizer=copy.deepcopy(optimizer.state_dict()),
                scheduler=copy.deepcopy(scheduler.state_dict()),
                rng=rng_state(generator),
                epoch=best_epoch,
            )
        history.append(
            dict(
                epoch=epoch + 1,
                training=training,
                validation=validation,
                best_epoch=best_epoch,
                learning_rate=optimizer.param_groups[0]["lr"],
                elapsed_seconds=perf_counter() - epoch_start,
            )
        )
        snapshot()
        write_record(
            report_path,
            dict(common, status="running", history=history, optimizer_updates=updates),
        )
        print(
            f"{model.variant} seed={settings['seed']} epoch={epoch + 1} validation={score:.7g} best={best:.7g}",
            flush=True,
        )
        if on_epoch is not None:
            on_epoch(epoch + 1)
    model.load_state_dict(best_state, strict=True)
    selected = validate(model, data, settings, prepared, decoder)
    if abs(selected["objective"] - best) > 1e-6:
        raise ValueError("validation-selected weights did not restore")
    models.save(checkpoint, model, model_metadata)
    result = dict(
        common,
        status="complete",
        history=history,
        initial_validation=initial,
        selected_validation=selected,
        selected_epoch=best_epoch,
        optimizer_updates=updates,
        elapsed_seconds=prior_seconds + perf_counter() - started,
        checkpoint_sha256=sha256_file(checkpoint),
        early_stopped=stale >= settings["patience"],
        selection="minimum factual-development objective; unlike family objectives are not ranked",
    )
    write_record(report_path, result)
    return read_record(report_path)
