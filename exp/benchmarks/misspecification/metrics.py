"""Evaluation-only metrics: command consistency is not counterfactual truth."""

import math

import torch
import torch.nn.functional as F

from src.schema import unflatten_state_index
from .reader import probabilities


@torch.no_grad()
def embedding_metrics(samples, truth, scale):
    """Same fitting-scaled, unbiased sample energy estimator as the old benchmark."""
    m, n, d = samples.shape
    normalized = (samples / scale).transpose(0, 1)
    first = ((samples - truth) / scale).norm(dim=-1).mean(0) / math.sqrt(d)
    within = torch.zeros(n)
    if m > 1:
        within = torch.cdist(
            normalized, normalized, compute_mode="donot_use_mm_for_euclid_dist"
        ).sum((1, 2))
        within /= m * (m - 1) * math.sqrt(d)
    mean = samples.mean(0)
    return dict(
        energy_score=first - 0.5 * within,
        mean_l2=(mean - truth).norm(dim=-1),
        mean_mse=(mean - truth).square().mean(-1),
        mean_cosine=F.cosine_similarity(mean, truth, dim=-1),
    )


def embedding_match(ids, action, target, available_actions):
    """An embedding target exists for only one *total* intervention per test ID."""
    return torch.tensor(
        [target == "total" and available_actions.get(i) == action["X"] for i in ids]
    )


@torch.no_grad()
def factual_confusion(reader, z, states, columns):
    """Soft confusion Gamma, pooled by true head level, not full sparse cells."""
    head = probabilities(reader, z)
    result = {}
    for j, col in enumerate(columns):
        levels = []
        for k in range(col.n_categories):
            mask = states[:, j] == k
            levels.append(
                dict(
                    level=k,
                    units=int(mask.sum()),
                    probabilities=head[col.name][mask].mean(0).tolist()
                    if mask.any()
                    else None,
                )
            )
        result[col.name] = levels
    return result


@torch.no_grad()
def symbolic_metrics(
    samples,
    z,
    posterior,
    matrix,
    pairs,
    factual,
    truth,
    decoder,
    auditor,
    audit_columns,
    fit_counts,
    confusion,
):
    m, n, d = samples.shape
    logs = decoder.log_joint(samples.reshape(m * n, d)).reshape(m, n, -1)
    if not torch.isfinite(samples).all() or not torch.isfinite(logs).all():
        raise FloatingPointError("non-finite semantic evaluation input/output")
    composed = logs.logsumexp(0) - math.log(m)
    q = posterior @ matrix
    values = dict(
        own_g_semantic_kl=(q * (q.clamp_min(1e-30).log() - composed)).sum(-1),
        movement_l2=(samples - z).norm(dim=-1).mean(0),
        sample_spread=(samples - samples.mean(0)).square().mean((0, 2)).sqrt(),
    )
    heads = probabilities(auditor, samples.reshape(m * n, d))
    factual_heads = probabilities(auditor, z)
    if any(
        not torch.isfinite(p).all() for p in [*heads.values(), *factual_heads.values()]
    ):
        raise FloatingPointError("non-finite independent evaluation reader")
    command = unflatten_state_index(pairs.target.flatten(), pairs.columns).reshape(
        m, n, -1
    )
    command_columns = {c.name: j for j, c in enumerate(pairs.columns)}
    support = unflatten_state_index(torch.arange(matrix.shape[0]), pairs.columns)
    source_counts, target_counts = fit_counts[pairs.source], fit_counts[pairs.target]
    strata = {
        "unseen": (source_counts == 0) | (target_counts == 0),
        "rare": (source_counts > 0)
        & (target_counts > 0)
        & ((source_counts < 5) | (target_counts < 5)),
        "frequent": (source_counts >= 5) & (target_counts >= 5),
    }
    transition = {
        key: dict(draws=int(mask.sum()), command_correct_sum=0.0, command_checks=0)
        for key, mask in strata.items()
    }
    for j, col in enumerate(audit_columns):
        name = col.name
        p = heads[name].reshape(m, n, -1)
        mean = p.mean(0)
        actual = truth[:, j]
        values[f"true_accuracy_{name}"] = mean.argmax(-1).eq(actual).float()
        values[f"true_nll_{name}"] = (
            -mean.gather(1, actual[:, None]).squeeze(1).clamp_min(1e-30).log()
        )
        unchanged = actual.eq(factual[:, j])
        values[f"unchanged_{name}"] = unchanged.float()
        values[f"changed_accuracy_{name}"] = values[
            f"true_accuracy_{name}"
        ].masked_fill(unchanged, float("nan"))
        values[f"change_probability_{name}"] = 1 - mean.gather(
            1, factual[:, j : j + 1]
        ).squeeze(1)
        recovered = mean.argmax(-1).eq(factual[:, j]).float()
        baseline = factual_heads[name].argmax(-1).eq(factual[:, j]).float()
        values[f"preservation_accuracy_{name}"] = recovered.masked_fill(
            ~unchanged, float("nan")
        )
        values[f"preservation_accuracy_drop_{name}"] = (
            baseline - recovered
        ).masked_fill(~unchanged, float("nan"))
        gamma = confusion[name]
        available = torch.tensor(
            [gamma[int(k)]["probabilities"] is not None for k in actual]
        )
        expected = torch.stack(
            [
                torch.tensor(gamma[int(k)]["probabilities"] or [0.0] * col.n_categories)
                for k in actual
            ]
        )
        values[f"confusion_reference_tv_{name}"] = (
            0.5 * (mean - expected).abs().sum(-1)
        ).masked_fill(~available, float("nan"))
        if name in command_columns:
            target = command[:, :, command_columns[name]]
            marginal = torch.stack(
                [
                    q[:, support[:, command_columns[name]] == k].sum(-1)
                    for k in range(col.n_categories)
                ],
                -1,
            )
            values[f"command_marginal_kl_{name}"] = (
                marginal
                * (marginal.clamp_min(1e-30).log() - mean.clamp_min(1e-30).log())
            ).sum(-1)
            correct = p.argmax(-1).eq(target).float()
            values[f"command_accuracy_{name}"] = correct.mean(0)
            values[f"command_nll_{name}"] = (
                -p.gather(2, target[:, :, None])
                .squeeze(-1)
                .clamp_min(1e-30)
                .log()
                .mean(0)
            )
            for key, mask in strata.items():
                transition[key]["command_correct_sum"] += float(correct[mask].sum())
                transition[key]["command_checks"] += int(mask.sum())
    return values, transition


def aggregate(values, identity):
    result = {}
    for group, mask in (
        ("all", torch.ones_like(identity)),
        ("identity", identity),
        ("nonidentity", ~identity),
    ):
        result[group] = dict(units=int(mask.sum()), metrics={})
        for key, value in values.items():
            valid = mask & torch.isfinite(value)
            result[group]["metrics"][key] = dict(
                units=int(valid.sum()),
                mean=float(value[valid].mean()) if valid.any() else None,
            )
    return result
