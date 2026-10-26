"""Common random coupled pairs, separate from deterministic model application."""

from dataclasses import dataclass
import math

import torch

from src.schema import unflatten_state_index
from src.latent_intervention.ot.common import coupled_weights
from . import ACTIONS, EDITORS
from .common import digest


@dataclass(frozen=True)
class Pairs:
    source: torch.Tensor
    target: torch.Tensor
    columns: tuple
    empty: bool = False


def check_action(action):
    if (
        not isinstance(action, dict)
        or action not in [{}, *ACTIONS]
        or any(type(v) is not int for v in action.values())
    ):
        raise ValueError("experiment supports empty action and do(X=k), k=0..3")


@torch.no_grad()
def posterior_rows(decoder, z):
    if decoder.training:
        raise ValueError("guiding reader must be frozen in evaluation mode")
    # A fixed one-row forward makes probabilities independent of outer batching.
    return torch.cat([decoder.log_joint(row[None]).exp() for row in z])


def draw_pairs(
    posterior, matrix, ids, columns, *, action, context="", seed=20260930, n_samples=64
):
    check_action(action)
    if matrix.shape != (math.prod(c.n_categories for c in columns),) * 2:
        raise ValueError("transition matrix differs from sampling schema")
    if type(n_samples) is not int or n_samples < 1 or len(ids) != len(posterior):
        raise ValueError("invalid draw count or IDs")
    if len(set(ids)) != len(ids):
        raise ValueError("sampling IDs must be unique")
    coupled_weights(posterior, matrix)  # Validate, retaining exact zero probabilities.
    source, target = [], []
    for i, identity in enumerate(ids):
        local = int(
            digest(dict(seed=seed, context=context, id=identity, action=action))[:16],
            16,
        ) % (2**63 - 1)
        rng = torch.Generator().manual_seed(local)
        s = torch.multinomial(
            posterior[i].cpu(), n_samples, replacement=True, generator=rng
        )
        t = (
            s.clone()
            if not action
            else torch.multinomial(matrix.cpu()[s], 1, generator=rng).flatten()
        )
        source.append(s)
        target.append(t)
    shape = (n_samples, 0)
    return Pairs(
        torch.stack(source, 1) if source else torch.empty(shape, dtype=torch.long),
        torch.stack(target, 1) if target else torch.empty(shape, dtype=torch.long),
        tuple(columns),
        not action,
    )


def support_mass(posterior, matrix, fit_counts):
    coupled_weights(posterior, matrix)
    if len(fit_counts) != len(matrix) or (fit_counts < 0).any():
        raise ValueError("invalid fitting support counts")
    known, frequent = fit_counts > 0, fit_counts >= 5
    known_mass = (posterior[:, known] * matrix[known][:, known].sum(-1)).sum(-1)
    frequent_mass = (
        posterior[:, frequent] * matrix[frequent][:, frequent].sum(-1)
    ).sum(-1)
    return dict(
        unseen_source=posterior[:, ~known].sum(-1),
        unseen_target=(posterior @ matrix)[:, ~known].sum(-1),
        unseen_either=(1 - known_mass).clamp(0, 1),
        rare_either_excluding_unseen=(known_mass - frequent_mass).clamp(0, 1),
        well_supported=frequent_mass.clamp(0, 1),
    )


@torch.no_grad()
def apply_pairs(model, z, pairs):
    if (
        tuple(model.columns) != pairs.columns
        or pairs.source.shape != pairs.target.shape
        or pairs.source.ndim != 2
        or pairs.source.shape[1] != len(z)
    ):
        raise ValueError("model/pair schema or batch mismatch")
    m, n = pairs.source.shape
    if pairs.empty:
        return z[None].expand(m, -1, -1).clone(), dict(
            sampled_pairs=m * n, unique_pairs=0, nfe=0
        )
    size = 1
    for col in pairs.columns:
        size *= col.n_categories
    if (
        (pairs.source < 0).any()
        or (pairs.target < 0).any()
        or (pairs.source >= size).any()
        or (pairs.target >= size).any()
    ):
        raise ValueError("pair index outside state space")
    states = unflatten_state_index(torch.arange(size), pairs.columns)
    outputs, unique_count, nfe, point_evaluations = [], 0, 0, 0
    for i in range(n):
        s, t = pairs.source[:, i], pairs.target[:, i]
        keys = t if model.variant in EDITORS else s * size + t
        unique, inverse = torch.unique(keys, sorted=True, return_inverse=True)
        unique_count += len(unique)
        points = z[i : i + 1].expand(len(unique), -1)
        if model.variant in EDITORS:
            result = model.transport(points, states[unique])
        else:
            src, dst = unique // size, unique % size
            changed = src != dst
            result = points.clone()
            if changed.any():
                result[changed] = model.transport(
                    points[changed], states[src[changed]], states[dst[changed]]
                )
                diagnostic = getattr(model, "last_diagnostics", {})
                nfe += diagnostic.get("nfe", 0)
                point_evaluations += diagnostic.get("velocity_point_evaluations", 0)
        outputs.append(result[inverse])
    samples = torch.stack(outputs, 1) if outputs else z.new_empty((m, 0, z.shape[-1]))
    if not torch.isfinite(samples).all():
        raise FloatingPointError("non-finite counterfactual samples")
    return samples, dict(
        sampled_pairs=m * n,
        unique_pairs=unique_count,
        nfe=nfe,
        velocity_point_evaluations=point_evaluations,
    )
