"""Exact semantic mixture and factual-geometry objective; no paired Z-prime targets."""

import math

import torch
import torch.nn.functional as F


def preservation_kl(probe, z, samples):
    """Per-component, per-unit mean KL over protected attribute heads."""
    with torch.no_grad():
        reference = probe.log_probs(z)
    edited = probe.log_probs(samples.flatten(0, 1))
    return (
        torch.stack(
            [
                (
                    source.exp()[None]
                    * (source[None] - output.reshape(*samples.shape[:2], -1))
                ).sum(-1)
                for source, output in zip(reference, edited, strict=True)
            ]
        )
        .mean(0)
        .clamp_min(0)
    )


def weighted_mean(values, log_weights):
    return (values * log_weights.exp().t()).sum(0).mean()


def loss_terms(samples, log_weights, z, target, decoder, guide, scale, bank, identity):
    """Semantic consistency, factual geometry and protected-guide diagnostics."""
    count, batch, dimension = samples.shape
    logs = decoder.log_joint(samples.flatten(0, 1)).reshape(count, batch, -1)
    composed = torch.logsumexp(logs + log_weights.t()[:, :, None], 0)
    kl_per_unit = (target * (target.clamp_min(1e-30).log() - composed)).sum(-1)
    standardized_shift = (samples - z[None]) / scale
    move = F.smooth_l1_loss(
        standardized_shift, torch.zeros_like(standardized_shift), reduction="none"
    ).mean(-1)
    # No test data, oracle Z', or target-state nearest neighbors enter the support bank.
    distances = torch.cdist((samples / scale).flatten(0, 1), bank / scale)
    support = distances.min(-1).values.reshape(count, batch) / math.sqrt(dimension)
    preservation = preservation_kl(guide, z, samples)
    identity_per_unit = (move * log_weights.exp().t()).sum(0)
    return {
        "semantic_kl": kl_per_unit.mean().clamp_min(0),
        "preservation_kl": weighted_mean(preservation, log_weights),
        "movement": weighted_mean(move, log_weights),
        "support": weighted_mean(support, log_weights),
        "radial": weighted_mean((samples.norm(dim=-1) - 1).square(), log_weights),
        "identity": (
            identity_per_unit[identity].mean() if identity.any() else samples.sum() * 0
        ),
        "entropy": weighted_mean(-(logs.exp() * logs).sum(-1), log_weights),
    }


def selection_key(metrics, semantic_tolerance):
    """Feasible semantics first; then preservation, identity, and collateral movement.

    Audit-probe metrics and oracle counterfactual embeddings NEVER enter selection.
    Infeasible runs remain explicitly infeasible, rather than claiming preservation
    success by returning the input unchanged.
    """
    kl = metrics["semantic_kl"]
    if kl > semantic_tolerance:
        return (1, kl)
    return (
        0,
        metrics["preservation_kl"] + metrics["identity"] + 0.1 * metrics["movement"],
    )
