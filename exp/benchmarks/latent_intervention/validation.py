"""Training-only validation and early stopping; no official-test targets accepted."""

import math
from time import perf_counter

import torch

from src.artifact_io import write_json
from src.hz.flow_intervention import multivariate_energy_distance, sample_counterfactual
from src.latent_intervention import consistency_target, make_objective


@torch.no_grad()
def validation_score(
    model,
    variant,
    z,
    *,
    states,
    s_prime,
    decoder,
    h_s,
    intervention,
    teacher=None,
    n_samples=16,
    batch_size=32,
    seed=42,
):
    """Native selection criteria, deliberately not a cross-family ranking metric.

    baseline uses S'; state_flow factual likelihood; distilled_flow teacher samples;
    remaining semantic variants forward KL (exact finite mixtures where available).
    Oracle validation lives in its existing MSE trainer, isolated from this function.
    """
    generator = torch.Generator().manual_seed(seed)
    target_rng = torch.Generator().manual_seed(seed + 1)
    total = 0.0
    for start in range(0, len(z), batch_size):
        batch = z[start : start + batch_size]
        state = {k: v[start : start + batch_size] for k, v in states.items()}
        if variant == "state_flow":
            score = -model.log_prob(batch, state).mean() / batch.shape[-1]
        elif variant == "distilled_flow":
            samples = sample_counterfactual(
                model, batch, intervention, n_samples=n_samples, generator=generator
            )
            targets = sample_counterfactual(
                teacher,
                batch,
                intervention,
                n_samples=n_samples,
                source_states=state,
                h_s=h_s,
                generator=target_rng,
            )
            score = multivariate_energy_distance(samples, targets, model.latent_scale)
        elif variant == "baseline":
            values, mask = make_objective(
                intervention, model.columns, batch_size=len(batch)
            )
            score = decoder.nll(
                model(batch, values, mask),
                {k: v[start : start + batch_size] for k, v in s_prime.items()},
            )
        else:
            target = consistency_target(decoder, h_s, batch, intervention)
            if variant in {"dist", "particles"}:
                values, mask = make_objective(
                    intervention, model.columns, batch_size=len(batch)
                )
                composed = model.composed_log_joint(batch, values, mask, decoder)[0]
            else:
                samples = sample_counterfactual(
                    model, batch, intervention, n_samples=n_samples, generator=generator
                )
                logs = decoder.log_joint(samples.flatten(0, 1)).reshape(
                    n_samples, len(batch), -1
                )
                composed = torch.logsumexp(logs, dim=0) - math.log(n_samples)
            score = (target * (target.clamp_min(1e-30).log() - composed)).sum(-1).mean()
        total += float(score) * len(batch)
    value = total / len(z)
    if not math.isfinite(value):
        raise FloatingPointError("non-finite h_Z validation metric")
    return value


class ValidationControl:
    """Callback shared by the eight legacy trainers, without resetting optimizers."""

    def __init__(self, score, settings, progress_path):
        self.score, self.settings, self.progress_path = score, settings, progress_path
        self.history, self.best_state, self.scheduler = [], None, None
        self.best, self.reference = float("inf"), float("inf")
        self.best_epoch, self.stale = 0, 0
        self.started = perf_counter()

    def __call__(self, model, optimizer, phase, logs):
        if not all(math.isfinite(float(v)) for v in logs.values()):
            raise FloatingPointError("non-finite training diagnostics")
        was_training = model.training
        try:
            model.eval()
            # Validation must not change dropout or training-noise random streams.
            with torch.random.fork_rng(devices=[]):
                value = self.score(model)
        finally:
            model.train(was_training)
        epoch = len(self.history) + 1
        lr = optimizer.param_groups[0]["lr"]
        eligible = phase != "pretrain"
        if eligible:
            if self.scheduler is None:
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=self.settings["scheduler_patience"],
                    factor=self.settings["scheduler_factor"],
                    threshold=self.settings["min_delta"],
                    threshold_mode="abs",
                    min_lr=self.settings["min_lr"],
                )
            if value < self.best:
                self.best, self.best_epoch = value, epoch
                self.best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            if value < self.reference - self.settings["min_delta"]:
                self.reference, self.stale = value, 0
            else:
                self.stale += 1
            self.scheduler.step(value)
        row = dict(
            epoch=epoch,
            phase=phase,
            training=logs,
            validation=value,
            learning_rate=lr,
            best_epoch=self.best_epoch,
            stale_epochs=self.stale,
            selection_eligible=eligible,
            elapsed_seconds=perf_counter() - self.started,
        )
        self.history.append(row)
        write_json(self.progress_path, dict(status="running", history=self.history))
        return eligible and self.stale >= self.settings["patience"]

    def restore(self, model):
        if self.best_state is None:
            raise ValueError("no eligible validation checkpoint was produced")
        model.load_state_dict(self.best_state)
        model.eval()
        return dict(
            history=self.history,
            best_epoch=self.best_epoch,
            best_validation=self.best,
            epochs_run=len(self.history),
            max_epochs=self.settings["max_epochs"],
            stopped_early=len(self.history) < self.settings["max_epochs"],
        )
