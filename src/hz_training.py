"""Isolated per-fit workers and portable checkpoints for the h_Z benchmark.

The established model definitions/losses are reused. No canonical encoder/g/h_Z
artifact is overwritten. Interrupted fits restart; completed fits are hash verified.
"""

from __future__ import annotations

import fcntl
import json
import math
from pathlib import Path
from time import perf_counter

import torch

from src import pipeline
from src.artifact_io import save_torch, sha256_file, write_json
from src.flow_intervention import (
    make_latent_intervention,
    train_latent_intervention_model,
)
from src.hz_benchmark_matrix import case_name, digest
from src.hz_validation import ValidationControl, validation_score
from src.oracle_regression import OracleRegression, train_oracle_regression
from src.oracle_targets import load_oracle_targets
from src.pair_encoding import load_latent_artifact
from src.schema import ColumnSpec, load_schema
from src.semantic_decoder import load_semantic_decoder
from src.symbolic_intervention import load_symbolic_kernel


def job_paths(root, case, stage, label):
    reports = Path(root) / "fits" / case["case"] / stage / label
    models = (
        Path("models/talent")
        / case["case"]
        / "benchmarks"
        / Path(root).name
        / stage
        / label
    )
    return reports, models / "latent_intervention.pt"


def model_kwargs(case, candidate, seed, columns, intervention):
    cfg = case["config"]["latent_intervention"]
    variant, width = case["variant"], candidate["width_multiplier"]
    # Absolute, parameter-matched widths override the ordinary multiplier rule.
    # Missing tiers fail closed; cases without overrides keep pipeline defaults.
    overrides = case["model_widths"][str(width)] if "model_widths" in case else {}
    common = dict(latent_dim=case["dimension"])
    if variant == "oracle_regression":
        oracle = cfg[variant]
        return (
            dict(
                **common,
                intervention=intervention,
                hidden_dim=int(oracle["hidden_dim"] * width),
                n_hidden=oracle["n_hidden"],
                dropout=oracle["dropout"],
            )
            | overrides
        )
    common.update(columns=[[c.name, c.n_categories] for c in columns])
    if variant in pipeline.FLOW_VARIANTS:
        flow = cfg["flow"]
        common.update(
            {
                k: flow[k]
                for k in (
                    "n_blocks",
                    "hidden_dim",
                    "condition_dim",
                    "state_embed_dim",
                    "scale_floor",
                    "base_std",
                )
            }
        )
        common.update(hidden_dim=int(flow["hidden_dim"] * width), permutation_seed=seed)
    else:
        common.update(
            {k: cfg[k] for k in ("d_model", "nhead", "dim_feedforward", "dropout")}
        )
        common.update(
            d_model=int(cfg["d_model"] * width),
            dim_feedforward=int(cfg["dim_feedforward"] * width),
        )
        for key in (
            "noise_std",
            "noise_dim",
            "embed_dim",
            "top_k",
            "n_particles",
            "uniform_weights",
        ):
            if key in cfg.get(variant, {}):
                common[key] = cfg[variant][key]
    common.update(overrides)
    if variant == "pre_additive":
        rms = candidate.get("noise_rms", case.get("noise_rms"))
        if rms is not None:
            if type(rms) not in (int, float) or not math.isfinite(rms) or rms <= 0:
                raise ValueError("pre_additive noise RMS must be positive and finite")
            common["noise_std"] = rms / math.sqrt(case["dimension"])
    return common


def construct(variant, kwargs, seed):
    if variant == "oracle_regression":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return OracleRegression(**kwargs)
    values = dict(kwargs)
    values["columns"] = [ColumnSpec(*c) for c in values["columns"]]
    return make_latent_intervention(variant, seed=seed, **values)


def model_sizes(plan):
    """Count actual models without fitting or loading g/a distillation teacher."""
    rows = []
    seed = plan["settings"]["search_seed"]
    for case in plan["cases"]:
        columns, _ = load_schema(case["config"]["sim_config"])
        for width in plan["settings"]["width_multipliers"]:
            kwargs = model_kwargs(
                case, dict(width_multiplier=width), seed, columns, plan["intervention"]
            )
            model = construct(case["variant"], kwargs, seed)
            rows.append(
                dict(
                    case=case["case"],
                    dimension=case["dimension"],
                    variant=case["variant"],
                    width_multiplier=width,
                    model_kwargs=kwargs,
                    trainable_parameters=sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                )
            )
    return rows


def load_model(path, expected_signature=None):
    """Reload a benchmark checkpoint, including its construction/provenance data."""
    payload = torch.load(path, weights_only=True, map_location="cpu")
    if payload.get("format") != "hz_benchmark_v1":
        raise ValueError("not an h_Z benchmark checkpoint")
    if expected_signature is not None and payload["signature"] != expected_signature:
        raise ValueError("h_Z checkpoint provenance mismatch")
    model = construct(payload["variant"], payload["model_kwargs"], payload["seed"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.supported_interventions = payload["supported_interventions"]
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("non-finite h_Z checkpoint weights")
    return model.eval()


def verify_fit(report, expected_signature=None):
    if report.get("record_sha256") != digest(
        {k: v for k, v in report.items() if k != "record_sha256"}
    ):
        raise ValueError("completed training report checksum mismatch")
    if expected_signature is not None and report["signature"] != expected_signature:
        raise ValueError("completed fit has different inputs/settings")
    if sha256_file(report["checkpoint"]) != report["checkpoint_sha256"]:
        raise ValueError("completed checkpoint checksum mismatch")
    load_model(report["checkpoint"], report["signature"])
    return report


def decoder_for(case, artifact):
    config = case["config"]
    columns, _ = load_schema(config["sim_config"])
    return (
        load_semantic_decoder(
            config["paths"]["decoder_model"],
            expected_variant="independent",
            expected_columns=columns,
            expected_metadata=pipeline._decoder_metadata(config, artifact),
        )
        .eval()
        .requires_grad_(False)
    )


def fit(
    root, plan, case, candidate, seed, stage, label, oracle_inputs, teacher_report=None
):
    """One CPU process, one immutable fit. No test targets enter any loss/selector."""
    torch.set_num_threads(plan["settings"]["threads_per_worker"])
    reports, checkpoint = job_paths(root, case, stage, label)
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / ".fit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _fit(
            reports,
            checkpoint,
            plan,
            case,
            candidate,
            seed,
            oracle_inputs,
            teacher_report,
        )


def _fit(
    reports, checkpoint, plan, case, candidate, seed, oracle_inputs, teacher_report
):
    variant, config, settings = case["variant"], case["config"], plan["settings"]
    teacher_hash = teacher_report["checkpoint_sha256"] if teacher_report else None
    if variant == "distilled_flow":
        if (
            teacher_report is None
            or teacher_report["case"] != case_name(case["dimension"], "state_flow")
            or teacher_report["seed"] != seed
            or teacher_report["plan_sha256"] != digest(plan)
        ):
            raise ValueError(
                "distillation teacher must match the dimension, seed and run protocol"
            )
    target_hash = None
    if variant == "oracle_regression":
        target_hash = next(
            x["sha256"] for x in oracle_inputs if x["dimension"] == case["dimension"]
        )
    signature = digest(
        dict(
            plan=digest(plan),
            case=case["case"],
            candidate=candidate,
            seed=seed,
            teacher=teacher_hash,
            oracle_targets=target_hash,
        )
    )
    report_path = reports / "training.json"
    if report_path.exists():
        return verify_fit(json.loads(report_path.read_text()), signature)
    started = perf_counter()
    write_json(reports / "progress.json", dict(status="loading", seed=seed))
    artifact = load_latent_artifact(config)
    columns, _ = load_schema(config["sim_config"])
    indices = {
        name: pipeline._indices_for_ids(artifact.ids, plan["split_ids"][name])
        for name in ("fit", "validation")
    }
    z_fit, z_val = (artifact.z[indices[name]] for name in ("fit", "validation"))
    kwargs = model_kwargs(case, candidate, seed, columns, plan["intervention"])
    model = construct(variant, kwargs, seed)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    torch.manual_seed(seed)
    if variant == "oracle_regression":
        targets = load_oracle_targets(config, artifact)
        if targets.artifact_sha256 != target_hash:
            raise ValueError("oracle targets changed after launch")
        target_idx = {
            name: pipeline._indices_for_ids(targets.ids, plan["split_ids"][name])
            for name in indices
        }
        summary = train_oracle_regression(
            model,
            z_fit,
            targets.z_prime[target_idx["fit"]],
            validation_latents=z_val,
            validation_targets=targets.z_prime[target_idx["validation"]],
            epochs=settings["max_epochs"],
            lr=candidate["lr"],
            seed=seed,
            device="cpu",
            weight_decay=config["latent_intervention"][variant]["weight_decay"],
            epoch_callback=lambda _model, row: write_json(
                reports / "progress.json",
                dict(status="running", elapsed_seconds=perf_counter() - started, **row),
            ),
            **{
                k: settings[k]
                for k in (
                    "batch_size",
                    "patience",
                    "min_delta",
                    "scheduler_patience",
                    "scheduler_factor",
                    "min_lr",
                    "grad_clip",
                )
            },
        )
        summary["best_validation"] = summary["best_validation_mse"]
    else:
        # Do not even load the privileged oracle training artifact in these branches.
        factual = pipeline._aligned_targets(
            config["paths"]["sim_factual"], artifact.ids, columns
        )
        states = {name: pipeline._subset(factual, idx) for name, idx in indices.items()}
        s_prime = {name: {} for name in indices}
        if variant in {"baseline", "dist"}:
            structured_cf = pipeline._aligned_targets(
                config["paths"]["sim_counterfactual"], artifact.ids, columns
            )
            s_prime = {
                name: pipeline._subset(structured_cf, idx)
                for name, idx in indices.items()
            }
        decoder = (
            None
            if variant in {"state_flow", "distilled_flow"}
            else decoder_for(case, artifact)
        )
        h_s = load_symbolic_kernel(config["sim_config"])
        teacher = None
        if variant == "distilled_flow":
            if teacher_report is None:
                raise ValueError(
                    "distillation requires the matching state-flow teacher"
                )
            verify_fit(teacher_report)
            teacher = load_model(
                teacher_report["checkpoint"], teacher_report["signature"]
            ).requires_grad_(False)

        def score(current):
            return validation_score(
                current,
                variant,
                z_val,
                states=states["validation"],
                s_prime=s_prime["validation"],
                decoder=decoder,
                h_s=h_s,
                intervention=plan["intervention"],
                teacher=teacher,
                n_samples=settings["validation_samples"],
                batch_size=settings["evaluation_batch_size"],
                seed=settings["split_seed"] + 10000,
            )

        control = ValidationControl(score, settings, reports / "progress.json")
        cfg = config["latent_intervention"]
        flow = cfg["flow"]
        training = dict(
            epochs=settings["max_epochs"],
            batch_size=settings["batch_size"],
            lr=candidate["lr"],
            grad_clip=settings["grad_clip"],
            seed=seed,
            device="cpu",
            verbose=False,
            epoch_callback=control,
        )
        if variant in pipeline.FLOW_VARIANTS:
            training.update(
                weight_decay=flow["weight_decay"],
                n_samples=flow["n_samples"],
                identity_fraction=flow["no_op_fraction"],
                support_bank_size=flow["support_bank_size"],
            )
            training.update(
                flow["direct_semantic"] if variant == "direct_semantic_flow" else {}
            )
        else:
            training.update(
                proximity_weight=cfg["proximity_weight"],
                sparsity_weight=cfg["sparsity_weight"],
            )
            training.update(
                {
                    k: cfg.get(variant, {})[k]
                    for k in (
                        "n_samples",
                        "entropy_weight",
                        "realiser_l1",
                        "realiser_l2",
                    )
                    if k in cfg.get(variant, {})
                }
            )
            if variant == "dist":
                training.update(
                    pretrain_epochs=settings["dist_pretrain_epochs"],
                    joint_epochs=settings["max_epochs"]
                    - settings["dist_pretrain_epochs"],
                )
        train_latent_intervention_model(
            model,
            latents=z_fit,
            intervention=plan["intervention"],
            decoder=decoder,
            h_s=h_s,
            states=states["fit"],
            s_prime=s_prime["fit"],
            teacher=teacher,
            **training,
        )
        summary = control.restore(model)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise FloatingPointError("non-finite restored model")
    save_torch(
        checkpoint,
        dict(
            format="hz_benchmark_v1",
            variant=variant,
            seed=seed,
            signature=signature,
            model_kwargs=kwargs,
            state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
            supported_interventions=getattr(
                model, "supported_interventions", [plan["intervention"]]
            ),
        ),
    )
    report = dict(
        signature=signature,
        plan_sha256=digest(plan),
        case=case["case"],
        variant=variant,
        seed=seed,
        candidate=candidate,
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256_file(checkpoint),
        model_kwargs=kwargs,
        trainable_parameters=trainable_parameters,
        teacher_sha256=teacher_hash,
        oracle_targets_sha256=target_hash,
        validation_metric=case["validation_metric"],
        supervision=case["supervision"],
        split_ids={k: plan["split_ids"][k] for k in ("fit", "validation")},
        test_targets_used=False,
        training=summary,
        seconds=perf_counter() - started,
    )
    report["record_sha256"] = digest(report)
    verify_fit(report, signature)
    write_json(report_path, report)
    write_json(
        reports / "progress.json",
        dict(status="complete", best_epoch=summary["best_epoch"]),
    )
    return report
