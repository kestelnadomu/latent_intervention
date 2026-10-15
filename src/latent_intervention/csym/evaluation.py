"""Independent factual audits and the unchanged common paired-oracle evaluation."""

from __future__ import annotations

import math

import torch

from src.pair_encoding import load_latent_artifact
from src.symbolic_intervention import load_symbolic_kernel
from src.latent_intervention.base.common import consistency_target
from exp.benchmarks.latent_intervention.evaluation import per_unit_metrics
from .protocol import decoder_for, load_data, validate_ids, verify_inputs
from .probes import probe_metrics


@torch.no_grad()
def audit_metrics(model, part, data, guide, auditor, settings):
    matrix = data["kernel"].transition_matrix({"X": 3}).to_dense()
    names = [c.name for c in data["columns"]]
    totals, counts, max_guide_change = {}, dict(all=0, identity=0, changed=0), 0.0
    for indices in torch.arange(len(part["z"])).split(settings["batch_size"]):
        z, labels = part["z"][indices], part["labels"][indices]
        posterior = data["decoder"].log_joint(z).exp()
        if model is None:
            samples, weights = z[None], z.new_zeros(len(z), 1)
        else:
            samples, weights = model.components(z, posterior, matrix)
        if not torch.isfinite(samples).all():
            raise FloatingPointError("nonfinite audit predictions")
        count, w = len(samples), weights.exp().t()
        metrics = {}
        for name, reference, output in (
            (
                "semantic_kl",
                posterior,
                data["decoder"].log_joint(samples.flatten(0, 1)),
            ),
            (
                "independent_semantic_kl",
                auditor.joint(z, names).exp(),
                auditor.joint(samples.flatten(0, 1), names),
            ),
        ):
            target = reference @ matrix
            composed = torch.logsumexp(
                output.reshape(count, len(z), -1) + weights.t()[:, :, None], 0
            )
            metrics[name] = (
                (target * (target.clamp_min(1e-30).log() - composed))
                .sum(-1)
                .clamp_min(0)
            )
        metrics["shift_l2"] = ((samples - z[None]).norm(dim=-1) * w).sum(0)
        metrics["norm_deviation"] = ((samples.norm(dim=-1) - 1).abs() * w).sum(0)
        before, after = auditor.log_probs(z), auditor.log_probs(samples.flatten(0, 1))
        flips, corruptions = [], []
        for j, col in enumerate(auditor.columns):
            original = before[j].argmax(-1)
            edited = after[j].reshape(count, len(z), -1).argmax(-1)
            if col.name == "X":
                metrics["independent_action_success_rate"] = (
                    edited.eq(3).float() * w
                ).sum(0)
            if col.name in ["T", *settings["protected_columns"]]:
                flip = (edited.ne(original[None]).float() * w).sum(0)
                corruption = (
                    (original == labels[:, j])[None] * edited.ne(labels[:, j][None]) * w
                ).sum(0)
                metrics[f"{col.name}_flip_rate"] = flip
                metrics[f"{col.name}_corruption_rate_unconditional"] = corruption
                if col.name != "T":
                    flips.append(flip)
                    corruptions.append(corruption)
        metrics["protected_flip_rate"] = torch.stack(flips).mean(0)
        metrics["protected_corruption_rate_unconditional"] = torch.stack(
            corruptions
        ).mean(0)
        for a, b in zip(
            guide.log_probs(z), guide.log_probs(samples.flatten(0, 1)), strict=True
        ):
            max_guide_change = max(
                max_guide_change,
                float((a.exp()[None] - b.reshape(count, len(z), -1).exp()).abs().max()),
            )
        for subset, mask in (
            ("all", torch.ones(len(z), dtype=torch.bool)),
            ("identity", labels[:, 0] == 3),
            ("changed", labels[:, 0] != 3),
        ):
            counts[subset] += int(mask.sum())
            for name, values in metrics.items():
                key = f"{subset}/{name}"
                totals[key] = totals.get(key, 0.0) + float(values[mask].sum())
    if model is not None and max_guide_change > 2e-5:
        raise ValueError(f"projection failed its probability check: {max_guide_change}")
    return dict(
        metrics={k: v / max(counts[k.split("/")[0]], 1) for k, v in totals.items()},
        counts=counts,
        max_guide_probability_change=max_guide_change,
        projection_verified=max_guide_change <= 2e-5 if model is not None else None,
    )


@torch.no_grad()
def exact_metrics(locations, log_weights, z, truth, scale, decoder, kernel, action):
    weights = log_weights.exp()
    first = (((locations - truth) / scale).norm(dim=-1).t() * weights).sum(-1)
    normalized = (locations / scale).transpose(0, 1)
    distances = torch.cdist(
        normalized, normalized, compute_mode="donot_use_mm_for_euclid_dist"
    )
    within = (distances * weights[:, :, None] * weights[:, None, :]).sum((1, 2))
    logs = decoder.log_joint(locations.flatten(0, 1)).reshape(
        len(locations), len(z), -1
    )
    composed = torch.logsumexp(logs + log_weights.t()[:, :, None], 0)
    target = consistency_target(decoder, kernel, z, action)
    return dict(
        energy_score=(first - 0.5 * within) / math.sqrt(z.shape[-1]),
        semantic_kl=(target * (target.clamp_min(1e-30).log() - composed)).sum(-1),
    )


def aggregate(collected, identity):
    return {
        name: dict(
            units=int(mask.sum()),
            metrics={k: float(v[mask].mean()) for k, v in collected.items()}
            if mask.any()
            else {},
        )
        for name, mask in (
            ("all", torch.ones_like(identity)),
            ("identity", identity),
            ("nonidentity", ~identity),
        )
    }


@torch.no_grad()
def audit(config, model, readers, settings):
    metadata = model.checkpoint_metadata
    verify_inputs(metadata)
    data = load_data(config, metadata["ids"], parts=("audit",))
    part = data["audit"]
    return dict(
        variant=model.variant,
        audit_ids=metadata["ids"]["audit"],
        used_official_test=False,
        used_oracle_targets=False,
        readers={
            name: dict(
                quality=probe_metrics(readers[name], part["z"], part["labels"]),
                no_edit=audit_metrics(
                    None, part, data, model.guide, readers[name], settings
                ),
                edited=audit_metrics(
                    model, part, data, model.guide, readers[name], settings
                ),
            )
            for name in ("auditor", "stress_auditor")
        },
    )


@torch.no_grad()
def evaluate(config, model):
    """Truth enters metrics only; g and h_S alone determine the prediction mixture."""
    from src import pipeline

    metadata = model.checkpoint_metadata
    verify_inputs(metadata)
    artifact = load_latent_artifact(config)
    validate_ids(artifact, metadata["ids"])
    settings = metadata["settings"]
    decoder = decoder_for(config, artifact)
    kernel = load_symbolic_kernel(config["sim_config"])
    fit = pipeline._indices_for_ids(artifact.ids, metadata["ids"]["fit"])
    test = pipeline._indices_for_ids(artifact.ids, artifact.test_ids)
    z, truth, identity = artifact.z[test], artifact.z_prime, artifact.is_identity
    scale = artifact.z[fit].std(0, unbiased=False).clamp_min(1e-6)
    realized = pipeline._subset(
        pipeline._aligned_targets(
            config["paths"]["sim_counterfactual"], artifact.ids, decoder.columns
        ),
        test,
    )
    matrix = kernel.transition_matrix(model.intervention).to_dense()
    generator = torch.Generator().manual_seed(settings["evaluation_seed"])
    collected, exact = {}, {}
    for start in range(0, len(z), settings["evaluation_batch_size"]):
        sl = slice(start, start + settings["evaluation_batch_size"])
        batch = z[sl]
        locations, weights = model.components(
            batch, decoder.log_joint(batch).exp(), matrix
        )
        picks = torch.multinomial(
            weights.exp(),
            settings["evaluation_samples"],
            replacement=True,
            generator=generator,
        )
        samples = locations[picks.t(), torch.arange(len(batch))]
        values = per_unit_metrics(
            samples,
            batch,
            truth[sl],
            scale,
            decoder,
            kernel,
            model.intervention,
            {k: v[sl] for k, v in realized.items()},
        )
        diagnostics = exact_metrics(
            locations,
            weights,
            batch,
            truth[sl],
            scale,
            decoder,
            kernel,
            model.intervention,
        )
        for output, scores in ((collected, values), (exact, diagnostics)):
            for key, value in scores.items():
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f"nonfinite evaluation metric: {key}")
                output.setdefault(key, []).append(value)
    collected = {k: torch.cat(v) for k, v in collected.items()}
    exact = {k: torch.cat(v) for k, v in exact.items()}
    return dict(
        variant=model.variant,
        dimension=model.latent_dim,
        test_ids=artifact.test_ids,
        identity=identity.tolist(),
        metrics=aggregate(collected, identity),
        per_unit={k: v.tolist() for k, v in collected.items()},
        exact_finite_mixture=dict(
            metrics=aggregate(exact, identity),
            per_unit={k: v.tolist() for k, v in exact.items()},
        ),
        evaluation_only=True,
        used_official_test=True,
        protocol={
            k: settings[k]
            for k in ("evaluation_samples", "evaluation_seed", "evaluation_batch_size")
        },
        interpretation="Exploratory, previously viewed test; one training seed. Exact-mixture scores are supplementary.",
    )
