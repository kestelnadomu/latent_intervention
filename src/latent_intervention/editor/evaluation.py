"""Explicit oracle evaluation only; no auxiliary readers or selection feedback."""

from time import perf_counter

import pandas as pd
import torch

from src.symbolic_intervention import load_symbolic_kernel
from .protocol import decoder_for


def aggregate(values, identity):
    if any(not torch.isfinite(v).all() for v in values.values()):
        raise FloatingPointError("nonfinite editor evaluation metrics")
    result = {}
    for name, mask in dict(
        all=torch.ones_like(identity, dtype=torch.bool),
        identity=identity.bool(),
        nonidentity=~identity.bool(),
    ).items():
        count = int(mask.sum())
        result[name] = dict(
            units=count,
            metrics={k: float(v[mask].mean()) for k, v in values.items()}
            if count
            else {},
        )
    return result


@torch.no_grad()
def evaluate(config, artifact, model):
    from src.pipeline import _indices_for_ids
    from exp.benchmarks.latent_intervention.evaluation import per_unit_metrics

    metadata = model.checkpoint_metadata
    settings = metadata["settings"]
    decoder = decoder_for(config, artifact)
    kernel = load_symbolic_kernel(config["sim_config"])
    ids = metadata["ids"]["official_test"]
    test = _indices_for_ids(artifact.ids, ids)
    fit = _indices_for_ids(artifact.ids, metadata["ids"]["fit"])
    z, truth = artifact.z[test], artifact.z_prime
    if len(truth) != len(ids) or list(artifact.test_ids) != ids:
        raise ValueError("oracle targets are not aligned to the official test IDs")
    scale = artifact.z[fit].std(0, unbiased=False).clamp_min(1e-6)
    names = [c.name for c in model.columns]
    frame = pd.read_csv(
        config["paths"]["sim_counterfactual"], usecols=["id", *names]
    ).set_index("id")
    if not frame.index.is_unique:
        raise ValueError("duplicate counterfactual IDs")
    realized = {}
    for col in model.columns:
        raw = frame.loc[ids, col.name].to_numpy()
        if (
            not (raw == raw.astype(int)).all()
            or (raw < 0).any()
            or (raw >= col.n_categories).any()
        ):
            raise ValueError("invalid realized counterfactual state")
        realized[col.name] = torch.tensor(raw, dtype=torch.long)
    generator = torch.Generator().manual_seed(settings["evaluation_seed"])
    collected, started = {}, perf_counter()
    for idx in torch.arange(len(z)).split(settings["evaluation_batch_size"]):
        samples = model.sample_symbolic(
            z[idx],
            {"X": 3},
            decoder=decoder,
            h_s=kernel,
            n_samples=settings["evaluation_samples"],
            generator=generator,
        )
        values = per_unit_metrics(
            samples,
            z[idx],
            truth[idx],
            scale,
            decoder,
            kernel,
            {"X": 3},
            {k: v[idx] for k, v in realized.items()},
        )
        for name, value in values.items():
            collected.setdefault(name, []).append(value)
    return dict(
        variant=model.variant,
        dimension=model.latent_dim,
        seed=settings["seed"],
        tier=settings["tier"],
        regime="latent_only",
        official_test_ids=ids,
        evaluation_samples=settings["evaluation_samples"],
        evaluation_seed=settings["evaluation_seed"],
        evaluation_batch_size=settings["evaluation_batch_size"],
        energy_scale="fitting-only coordinate std (unbiased=False), floor 1e-6; normalized by sqrt(d)",
        metrics=aggregate(
            {k: torch.cat(v) for k, v in collected.items()}, artifact.is_identity
        ),
        elapsed_seconds=perf_counter() - started,
        selection_frozen=True,
        used_official_test=True,
        used_auxiliary_labels=False,
        used_oracle_targets="evaluation metrics only",
        reserved_audit="unused",
        interpretation=dict(
            semantic_kl="consistency with frozen g/h_S, not realized truth",
            realized="accuracy/NLL against realized visible S-prime",
            energy="paired oracle embedding recovery",
        ),
        caveats=[
            "Previously viewed test results make these comparisons exploratory.",
            "A single training seed is not a multi-seed result.",
            "The existing g was trained separately; this is not a new end-to-end holdout.",
            "Neither editor guarantees true individual-counterfactual identification or auxiliary-attribute preservation.",
        ],
    )
