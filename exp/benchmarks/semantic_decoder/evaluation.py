"""Held-out distribution metrics, evaluated only after benchmark selection freezes."""

import json
import math
import statistics
from pathlib import Path

import torch

from src import pipeline
from src.artifact_io import sha256_file, write_json
from src.decoder_benchmark.decoder_experiment import verify_run
from src.schema import flat_state_index
from src.semantic_decoder import calibration_metrics, load_semantic_decoder


def moments(values):
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


@torch.no_grad()
def distribution_metrics(model, z, targets, n_bins=10):
    """Exact joint MAP, proper scores and true marginals (not AR greedy heads)."""
    if len(z) == 0:
        raise ValueError("evaluation set must be nonempty")
    model.eval()
    logp = torch.cat([model.log_joint(batch) for batch in z.split(256)])
    p = logp.exp()
    if not torch.isfinite(logp).all() or not torch.allclose(
        p.sum(-1), torch.ones(len(z)), atol=1e-5, rtol=1e-5
    ):
        raise ValueError("invalid joint probabilities during final evaluation")
    flat = flat_state_index(
        torch.stack([targets[c.name] for c in model.columns], dim=-1), model.columns
    )
    truth_logp = logp.gather(1, flat[:, None]).squeeze(1)
    marginals = calibration_metrics(model, z, targets, n_bins=n_bins)
    return {
        "n_units": len(z),
        "joint_nll": float(-truth_logp.mean()),
        "joint_map_accuracy": float(logp.argmax(-1).eq(flat).float().mean()),
        "joint_brier": float((p.square().sum(-1) - 2 * truth_logp.exp() + 1).mean()),
        "macro_marginal_accuracy": statistics.mean(
            m["accuracy"] for m in marginals.values()
        ),
        "macro_marginal_ece": statistics.mean(m["ece"] for m in marginals.values()),
        "marginals": marginals,
    }


def summarize(runs, population):
    metrics = [r[population] for r in runs]
    return {
        "n_units": metrics[0]["n_units"],
        "n_seeds": len(runs),
        **{
            key: moments([m[key] for m in metrics])
            for key in (
                "joint_nll",
                "joint_map_accuracy",
                "joint_brier",
                "macro_marginal_accuracy",
                "macro_marginal_ece",
            )
        },
        "marginals": {
            name: {
                metric: moments([m["marginals"][name][metric] for m in metrics])
                for metric in ("accuracy", "ece")
            }
            for name in metrics[0]["marginals"]
        },
    }


def evaluate_case(result, root):
    """Read final checkpoints only; never train, calibrate or select using test data."""
    torch.set_num_threads(1)
    root = Path(root)
    selection = json.loads((root / "selection.json").read_text())
    if not selection.get("frozen_before_test_evaluation") or result["case"] not in {
        row["case"] for row in selection["ranking"]
    }:
        raise ValueError("selection must be frozen before test evaluation")
    base = result["selected_config"]
    artifact = pipeline.load_latent_artifact(base)
    columns, _ = pipeline.load_schema(base["sim_config"])
    _, test_idx = pipeline._official_indices(artifact)
    factual = pipeline._aligned_targets(
        base["paths"]["sim_factual"], artifact.ids, columns
    )
    # Validate all CF IDs, then select only official test IDs, in latent order.
    counterfactual = pipeline._aligned_targets(
        base["paths"]["sim_counterfactual"], artifact.ids, columns
    )
    factual = pipeline._subset(factual, test_idx)
    counterfactual = pipeline._subset(counterfactual, test_idx)
    nonidentity = (~artifact.is_identity).nonzero().flatten()
    population = {
        "factual_test": (artifact.z[test_idx], factual),
        "counterfactual_test": (artifact.z_prime, counterfactual),
    }
    if len(nonidentity):
        population["counterfactual_nonidentity_test"] = (
            artifact.z_prime[nonidentity],
            pipeline._subset(counterfactual, nonidentity),
        )
    report_dir = root / "evaluation" / result["encoder_tag"] / f"g-{result['decoder']}"
    signature = {
        "case": result["case"],
        "protocol_sha256": sha256_file(root / "protocol.json"),
        "selection_sha256": sha256_file(root / "selection.json"),
        "latent_sha256": artifact.artifact_sha256,
        "test_ids_sha256": pipeline._config_sha256({"ids": artifact.test_ids}),
        "targets_sha256": {
            key: sha256_file(base["paths"][key])
            for key in ("sim_factual", "sim_counterfactual")
        },
        "calibration_bins": int(base["semantic_decoder"]["calibration_bins"]),
    }
    runs = []
    for final in result["final_runs"]:
        report = json.loads(Path(final["report"]).read_text())
        config = report["config"]
        verify_run(config)
        seed_signature = {
            **signature,
            "seed": final["seed"],
            "checkpoint_sha256": sha256_file(final["checkpoint"]),
        }
        output = report_dir / f"seed-{final['seed']}.json"
        if output.exists():
            evaluation = json.loads(output.read_text())
            if evaluation["signature"] != seed_signature:
                raise ValueError(f"evaluation provenance changed: {output}")
        else:
            model = load_semantic_decoder(
                final["checkpoint"],
                expected_variant=result["decoder"],
                expected_columns=columns,
                expected_metadata=pipeline._decoder_metadata(config, artifact),
            )
            validation = {
                "macro_marginal_accuracy": statistics.mean(
                    v["accuracy"] for v in report["calibration"].values()
                ),
                "macro_marginal_ece": statistics.mean(
                    v["ece"] for v in report["calibration"].values()
                ),
            }
            evaluation = {
                "signature": seed_signature,
                "parameters": sum(p.numel() for p in model.parameters()),
                "validation": validation,
                **{
                    name: distribution_metrics(
                        model, z, targets, n_bins=signature["calibration_bins"]
                    )
                    for name, (z, targets) in population.items()
                },
            }
            if not all(
                math.isfinite(evaluation[name]["joint_nll"]) for name in population
            ):
                raise ValueError("non-finite evaluation loss")
            write_json(output, evaluation)
        runs.append(evaluation)
    summary = {
        "case": result["case"],
        "signature": signature,
        "parameters": runs[0]["parameters"],
        "variability": "Mean and sample SD across five training seeds on the same test IDs; not confidence intervals over units.",
        "validation": {
            key: moments([r["validation"][key] for r in runs])
            for key in ("macro_marginal_accuracy", "macro_marginal_ece")
        },
        **{name: summarize(runs, name) for name in population},
    }
    write_json(report_dir / "summary.json", summary)
    return summary
