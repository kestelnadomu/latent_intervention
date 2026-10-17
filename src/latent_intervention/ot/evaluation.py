"""Independent factual audits; oracle targets enter ONLY explicit final evaluation."""

import math
from time import perf_counter

import pandas as pd
import torch

from src.artifact_io import save_torch, sha256_file
from src.schema import ColumnSpec
from src.symbolic_intervention import load_symbolic_kernel
from src.latent_intervention.base.common import consistency_target
from src.latent_intervention.audit_probes import (
    fit_independent_auditor,
    fit_probe,
    pack_probe,
    unpack_probe,
    probe_metrics,
)
from .protocol import decoder_for, digest, load_factual


def audit_data(config, metadata):
    parts, columns = load_factual(
        config, metadata["ids"], parts=("fit", "tune", "audit")
    )
    protected = [ColumnSpec(name, 3) for name in ("P", "L", "H", "A")]
    names = [c.name for c in protected]
    frame = pd.read_csv(
        config["paths"]["sim_factual"], usecols=["id", *names]
    ).set_index("id")
    if not frame.index.is_unique:
        raise ValueError("duplicate audit IDs")
    result = {}
    for name, part in parts.items():
        raw = frame.loc[list(part.ids), names].to_numpy()
        if not (raw == raw.astype(int)).all() or (raw < 0).any() or (raw > 2).any():
            raise ValueError("invalid audit attributes")
        labels = torch.cat([part.states, torch.tensor(raw, dtype=torch.long)], -1)
        result[name] = dict(z=part.z, labels=labels)
    return result, [*columns, *protected]


def readers_for(paths, data, columns, metadata):
    # Reader fitting is independent of editor variant/weights. Same split/seed/config
    # produces the same audit instrument across all three methods.
    settings = metadata["settings"]
    signature = digest(
        dict(
            inputs=metadata["inputs"],
            ids=metadata["ids"],
            probe=settings["probe"],
            seed=settings["seed"],
            probe_source=sha256_file("src/latent_intervention/audit_probes.py"),
        )
    )
    if paths["readers"].exists():
        payload = torch.load(paths["readers"], map_location="cpu", weights_only=True)
        if (
            payload.get("format") != "ot_audit_readers_v1"
            or payload["signature"] != signature
        ):
            raise ValueError("independent OT audit reader signature mismatch")
    else:
        probe, seed = settings["probe"], settings["seed"] + 211
        with torch.random.fork_rng(devices=[]):
            auditor, history = fit_independent_auditor(
                data["fit"], data["tune"], columns, probe, seed
            )
            stress, stress_history = fit_probe(
                data["fit"]["z"],
                data["fit"]["labels"],
                data["tune"]["z"],
                data["tune"]["labels"],
                columns,
                probe["auditor_hidden"],
                probe,
                seed,
                probe["auditor_lr"],
                skip=probe["auditor_skip"],
            )
        payload = dict(
            format="ot_audit_readers_v1",
            signature=signature,
            readers={"independent": pack_probe(auditor), "stress": pack_probe(stress)},
            histories=dict(independent=history, stress=stress_history),
        )
        save_torch(paths["readers"], payload)
    return {
        name: unpack_probe(value) for name, value in payload["readers"].items()
    }, payload


@torch.no_grad()
def audit_predictions(model, data, readers, decoder, kernel, settings):
    z, labels = data["z"], data["labels"]
    generator = torch.Generator().manual_seed(settings["evaluation_seed"])
    collected = {name: {} for name in readers}
    diagnostics = []
    started = perf_counter()
    for idx in torch.arange(len(z)).split(settings["evaluation_batch_size"]):
        original = z[idx]
        samples = model.sample_symbolic(
            original,
            {"X": 3},
            decoder=decoder,
            h_s=kernel,
            n_samples=settings["evaluation_samples"],
            generator=generator,
        )
        if model.last_diagnostics:
            diagnostics.append(dict(model.last_diagnostics))
        for name, reader in readers.items():
            before, after = (
                reader.log_probs(original),
                reader.log_probs(samples.flatten(0, 1)),
            )
            values = {}
            for j, col in enumerate(reader.columns):
                predictions = after[j].reshape(len(samples), len(idx), -1).argmax(-1)
                factual = before[j].argmax(-1)
                if col.name == "X":
                    values["action_success"] = predictions.eq(3).float().mean(0)
                if col.name in ("T", "P", "L", "H", "A"):
                    values[f"{col.name}_flip_rate"] = (
                        predictions.ne(factual[None]).float().mean(0)
                    )
                    values[f"{col.name}_corruption_unconditional"] = (
                        (
                            factual.eq(labels[idx, j])[None]
                            & predictions.ne(labels[idx, j][None])
                        )
                        .float()
                        .mean(0)
                    )
            values["norm_deviation"] = (samples.norm(dim=-1) - 1).abs().mean(0)
            values["shift_l2"] = (samples - original).norm(dim=-1).mean(0)
            logs = decoder.log_joint(samples.flatten(0, 1)).reshape(
                len(samples), len(idx), -1
            )
            composed = torch.logsumexp(logs, 0) - math.log(len(samples))
            target = consistency_target(decoder, kernel, original, {"X": 3})
            values["semantic_kl"] = (
                target * (target.clamp_min(1e-30).log() - composed)
            ).sum(-1)
            for key, value in values.items():
                collected[name].setdefault(key, []).append(value)
    result = {}
    identity = labels[:, 0] == 3
    for name, values in collected.items():
        merged = {key: torch.cat(items) for key, items in values.items()}
        result[name] = aggregate(merged, identity)
    return dict(
        readers=result,
        elapsed_seconds=perf_counter() - started,
        solver_diagnostics=diagnostics,
    )


def aggregate(values, identity):
    if any(not torch.isfinite(v).all() for v in values.values()):
        raise FloatingPointError("nonfinite OT evaluation metrics")
    return {
        name: dict(
            units=int(mask.sum()),
            metrics={k: float(v[mask].mean()) for k, v in values.items()}
            if mask.any()
            else {},
        )
        for name, mask in (
            ("all", torch.ones_like(identity)),
            ("identity", identity),
            ("nonidentity", ~identity),
        )
    }


def audit(config, artifact, model, paths):
    metadata = model.checkpoint_metadata
    data, columns = audit_data(config, metadata)
    readers, payload = readers_for(paths, data, columns, metadata)
    quality = {
        name: probe_metrics(reader, data["audit"]["z"], data["audit"]["labels"])
        for name, reader in readers.items()
    }
    qualified = {
        name: all(metrics[c]["balanced_accuracy"] >= 0.6 for c in ("P", "L", "H", "A"))
        for name, metrics in quality.items()
    }
    return dict(
        variant=model.variant,
        used_official_test=False,
        used_oracle_targets=False,
        audit_ids=metadata["ids"]["audit"],
        reader_quality=quality,
        readers_qualified=qualified,
        reader_histories=payload["histories"],
        reader_sha256=sha256_file(paths["readers"]),
        checkpoint_sha256=sha256_file(paths["model"]),
        preservation_guaranteed=False,
        predictions=audit_predictions(
            model,
            data["audit"],
            readers,
            decoder_for(config, artifact),
            load_symbolic_kernel(config["sim_config"]),
            metadata["settings"],
        ),
    )


@torch.no_grad()
def evaluate(config, artifact, model):
    """The sole OT entrypoint that accesses paired test targets/counterfactual states."""
    from src import pipeline
    from exp.benchmarks.latent_intervention.evaluation import per_unit_metrics

    metadata, settings = (
        model.checkpoint_metadata,
        model.checkpoint_metadata["settings"],
    )
    decoder = decoder_for(config, artifact)
    kernel = load_symbolic_kernel(config["sim_config"])
    test = pipeline._indices_for_ids(artifact.ids, metadata["ids"]["official_test"])
    fit = pipeline._indices_for_ids(artifact.ids, metadata["ids"]["fit"])
    z, truth = artifact.z[test], artifact.z_prime
    scale = artifact.z[fit].std(0, unbiased=False).clamp_min(1e-6)
    realized = pipeline._subset(
        pipeline._aligned_targets(
            config["paths"]["sim_counterfactual"], artifact.ids, decoder.columns
        ),
        test,
    )
    generator = torch.Generator().manual_seed(settings["evaluation_seed"])
    collected, diagnostics = {}, []
    started = perf_counter()
    for idx in torch.arange(len(z)).split(settings["evaluation_batch_size"]):
        samples = model.sample_symbolic(
            z[idx],
            {"X": 3},
            decoder=decoder,
            h_s=kernel,
            n_samples=settings["evaluation_samples"],
            generator=generator,
        )
        if model.last_diagnostics:
            diagnostics.append(dict(model.last_diagnostics))
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
        for key, value in values.items():
            collected.setdefault(key, []).append(value)
    return dict(
        variant=model.variant,
        dimension=model.latent_dim,
        regime="latent_only",
        official_test_ids=metadata["ids"]["official_test"],
        evaluation_samples=settings["evaluation_samples"],
        evaluation_seed=settings["evaluation_seed"],
        metrics=aggregate(
            {k: torch.cat(v) for k, v in collected.items()}, artifact.is_identity
        ),
        elapsed_seconds=perf_counter() - started,
        solver_diagnostics=diagnostics,
        selection_frozen=True,
        used_oracle_targets="metrics only",
        caveats=[
            "Existing test results were previously viewed; these comparisons are exploratory.",
            "Original nine-method benchmark used 800 tuning IDs; OT/C-sym use 400 tuning + 400 audit.",
            "The audit split is not an untouched end-to-end holdout for the previously fitted g.",
        ],
    )
