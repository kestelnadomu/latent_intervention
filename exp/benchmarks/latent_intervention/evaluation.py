"""Batched, common held-out metrics for point, mixture and flow h_Z models."""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from src import pipeline
from src.artifact_io import sha256_file, write_json
from src.latent_intervention.dispatch import sample_counterfactual
from exp.benchmarks.latent_intervention.matrix import digest
from exp.benchmarks.latent_intervention.training import decoder_for, load_model, verify_fit
from src.latent_intervention.base import consistency_target
from src.latent_intervention.oracle.regression import OracleRegression
from src.pair_encoding import load_latent_artifact
from src.schema import flat_state_index, unflatten_state_index
from src.symbolic_intervention import load_symbolic_kernel


def draw(model, variant, z, action, n_samples, generator, decoder, h_s, states=None):
    if model is None:  # no-edit reference, not a tenth fitted family
        return z.unsqueeze(0)
    if isinstance(model, OracleRegression):
        return model.predict(z, intervention=action).unsqueeze(0)
    return sample_counterfactual(
        model,
        z,
        action,
        n_samples=1 if variant == "baseline" else n_samples,
        generator=generator,
        decoder=decoder,
        h_s=h_s,
        source_states=states,
    )


@torch.no_grad()
def per_unit_metrics(samples, z, truth, scale, decoder, h_s, intervention, realized):
    """Unbiased sample energy score; raw and scale-/dimension-normalized recovery.

    Only distances *within the same unit* are compared. No held-out nearest-neighbor
    support bank, pooled test fitting, or quadratic cost in the number of test units.
    """
    if not torch.isfinite(samples).all():
        raise FloatingPointError("non-finite h_Z samples")
    m, n, dimension = samples.shape
    mean = samples.mean(0)
    error = mean - truth
    normalized = (samples / scale).transpose(0, 1)
    first = ((samples - truth) / scale).norm(dim=-1).mean(0) / math.sqrt(dimension)
    if m > 1:
        distances = torch.cdist(
            normalized, normalized, compute_mode="donot_use_mm_for_euclid_dist"
        )
        within = distances.sum((1, 2)) / (m * (m - 1) * math.sqrt(dimension))
    else:
        within = torch.zeros(n)
    # g is applied to individual draws, not to their possibly off-manifold mean.
    logs = decoder.log_joint(samples.flatten(0, 1)).reshape(m, n, -1)
    composed = torch.logsumexp(logs, dim=0) - math.log(m)
    target = consistency_target(decoder, h_s, z, intervention)
    result = dict(
        energy_score=first - 0.5 * within,
        mean_mse=error.square().mean(-1),
        standardized_mean_mse=(error / scale).square().mean(-1),
        mean_l2=error.norm(dim=-1),
        mean_cosine=F.cosine_similarity(mean, truth, dim=-1),
        sample_spread=(samples - mean).square().mean((0, 2)).sqrt(),
        mean_shift_l2=(samples - z).norm(dim=-1).mean(0),
        semantic_kl=(target * (target.clamp_min(1e-30).log() - composed)).sum(-1),
    )
    state = torch.stack([realized[c.name] for c in decoder.columns], -1)
    flat = flat_state_index(state, decoder.columns)
    result["realized_joint_nll"] = -composed[torch.arange(n), flat]
    result["realized_joint_accuracy"] = composed.argmax(-1).eq(flat).float()
    support = unflatten_state_index(torch.arange(composed.shape[1]), decoder.columns)
    for i, col in enumerate(decoder.columns):
        marginal = torch.stack(
            [
                composed.exp()[:, support[:, i] == k].sum(-1)
                for k in range(col.n_categories)
            ],
            -1,
        )
        result[f"accuracy_{col.name}"] = (
            marginal.argmax(-1).eq(realized[col.name]).float()
        )
    return result


@torch.no_grad()
def evaluate_regime(
    model,
    variant,
    z,
    truth,
    identity,
    factual_states,
    realized,
    *,
    scale,
    decoder,
    h_s,
    intervention,
    settings,
    observed_states=False,
):
    generator = torch.Generator().manual_seed(settings["evaluation_seed"])
    collected = {}
    for start in range(0, len(z), settings["evaluation_batch_size"]):
        sl = slice(start, start + settings["evaluation_batch_size"])
        states = (
            {k: v[sl] for k, v in factual_states.items()} if observed_states else None
        )
        samples = draw(
            model,
            variant,
            z[sl],
            intervention,
            settings["evaluation_samples"],
            generator,
            decoder,
            h_s,
            states,
        )
        values = per_unit_metrics(
            samples,
            z[sl],
            truth[sl],
            scale,
            decoder,
            h_s,
            intervention,
            {k: v[sl] for k, v in realized.items()},
        )
        for key, value in values.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"non-finite evaluation metric: {key}")
            collected.setdefault(key, []).append(value.cpu())
    collected = {k: torch.cat(v) for k, v in collected.items()}
    result = {}
    for name, mask in (
        ("all", torch.ones(len(z), dtype=torch.bool)),
        ("identity", identity),
        ("nonidentity", ~identity),
    ):
        result[name] = dict(
            units=int(mask.sum()),
            metrics=(
                {k: float(v[mask].mean()) for k, v in collected.items()}
                if mask.any()
                else {}
            ),
        )
    return result


def evaluate(root, plan, case, fit_report):
    torch.set_num_threads(plan["settings"]["threads_per_worker"])
    selection_path = Path(root) / "selection.json"
    if not selection_path.exists():
        raise ValueError("all selections must be frozen before any test evaluation")
    frozen = json.loads(selection_path.read_text())
    if (
        frozen["plan_sha256"] != digest(plan)
        or frozen["cases"][case["case"]]["candidate"] != fit_report["candidate"]
    ):
        raise ValueError("fit does not match frozen validation selection")
    verify_fit(fit_report)
    signature = digest(
        dict(
            plan=digest(plan),
            checkpoint=fit_report["checkpoint_sha256"],
            selection=sha256_file(selection_path),
        )
    )
    path = Path(root) / "evaluation" / case["case"] / f"seed-{fit_report['seed']}.json"
    if path.exists():
        record = json.loads(path.read_text())
        checksum = record.pop("record_sha256")
        if record["signature"] != signature or digest(record) != checksum:
            raise ValueError("saved evaluation was changed or has stale inputs")
        return dict(**record, record_sha256=checksum)
    artifact = load_latent_artifact(case["config"])
    model = load_model(fit_report["checkpoint"], fit_report["signature"])
    decoder = decoder_for(case, artifact)
    h_s = load_symbolic_kernel(case["config"]["sim_config"])
    fit_idx = pipeline._indices_for_ids(artifact.ids, plan["split_ids"]["fit"])
    test_idx = pipeline._indices_for_ids(
        artifact.ids, plan["split_ids"]["official_test"]
    )
    if artifact.test_ids != plan["split_ids"]["official_test"]:
        raise ValueError("official test order changed")
    factual, realized = (
        pipeline._subset(
            pipeline._aligned_targets(
                case["config"]["paths"][key], artifact.ids, decoder.columns
            ),
            test_idx,
        )
        for key in ("sim_factual", "sim_counterfactual")
    )
    scale = artifact.z[fit_idx].std(0, unbiased=False).clamp_min(1e-6)
    kwargs = dict(
        z=artifact.z[test_idx],
        truth=artifact.z_prime,
        identity=artifact.is_identity,
        factual_states=factual,
        realized=realized,
        scale=scale,
        decoder=decoder,
        h_s=h_s,
        intervention=plan["intervention"],
        settings=plan["settings"],
    )
    regimes = {"latent_only": evaluate_regime(model, case["variant"], **kwargs)}
    if case["variant"] == "state_flow":
        regimes["observed_state"] = evaluate_regime(
            model, case["variant"], observed_states=True, **kwargs
        )
    reference = evaluate_regime(None, "no_edit", **kwargs)
    record = dict(
        signature=signature,
        case=case["case"],
        seed=fit_report["seed"],
        supervision=case["supervision"],
        checkpoint_sha256=fit_report["checkpoint_sha256"],
        test_ids=artifact.test_ids,
        evaluation_samples=plan["settings"]["evaluation_samples"],
        point_mass=case["variant"] in {"baseline", "oracle_regression"},
        regimes=regimes,
        no_edit_reference=reference,
        notes="Energy score uses fit-only coordinate scales and sqrt(d) normalization. Test results are descriptive, not selection inputs. g-based metrics favor methods trained through g; paired latent recovery is independent of g.",
    )
    record["record_sha256"] = digest(record)
    write_json(path, record)
    return record
