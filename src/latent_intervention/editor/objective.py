"""One exact semantic/geometry objective for both editors; no protected attributes."""

import math

import torch


def terms_for(model, z, target, identity, decoder, bank, *, create_graph=False):
    samples, log_weights = model.components(z, target, create_graph=create_graph)
    weights = log_weights.exp().t()
    logs = decoder.log_joint(samples.flatten(0, 1)).reshape(*samples.shape[:2], -1)
    composed = torch.logsumexp(logs + log_weights.t()[:, :, None], 0)
    semantic = (target * (target.clamp_min(1e-30).log() - composed)).sum(-1).mean()
    shifts = (samples - z[None]) / model.scale
    movement = (shifts.square().mean(-1) * weights).sum(0)
    points = ((samples - model.mean) / model.scale).flatten(0, 1)
    support = torch.cdist(points, bank).min(-1).values.reshape(
        samples.shape[:2]
    ) / math.sqrt(model.latent_dim)
    return dict(
        semantic_kl=semantic.clamp_min(0),
        movement=movement.mean(),
        support=(support * weights).sum(0).mean(),
        identity=movement[identity].mean() if identity.any() else movement.sum() * 0,
    )


def total_loss(terms, settings):
    return terms["semantic_kl"] + sum(
        settings["loss"][key] * terms[key]
        for key in ("movement", "support", "identity")
    )
