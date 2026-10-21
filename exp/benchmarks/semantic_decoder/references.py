"""Latent-free reference rows for a completed semantic-decoder benchmark.

Two references bracket every encoder/decoder case on the benchmark's own split:

- ``prior``: the Laplace-smoothed empirical joint of the structured state, fitted on
  the decoder-fit IDs, ignoring the latent entirely (the no-information floor).
- ``bayes_optimal``: the Bayes-optimal predictor of the state given every verbalized
  variable (X, D, U, P, L, H, A). X, D and U are read exactly; T, which the CVs show
  only through its proxies, gets its exact posterior under the known SCM. Nothing is
  fitted. The CVs are generated from these variables plus T-independent randomness,
  so by data processing no reader of the text or its embedding can do better in
  expectation.

Both use the frozen ``split_ids`` and verify the simulation input hash of the run's
protocol. The output is ``references.json`` in the run directory.

    .venv/bin/python -m exp.benchmarks.semantic_decoder.references --run-id g-all-encoders-v1
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp.sim_talent.talent_sfm import _nodes, _talent_log_posterior

ROOT = Path("reports/talent/decoder_benchmarks")
SIM_FACTUAL = "data/sim_talent/sim_data_factual.csv"
STATE = ("X", "T", "D", "U")
VERBALIZED = ("X", "D", "U", "P", "L", "H", "A")
T_PRIOR = (0.25, 0.50, 0.25)  # the root sampler of T in exp/sim_talent/talent_sfm.py
SPLITS = {"validation": "validation", "test": "official_test"}


def lf_sha256(path):
    """Hash with LF line endings, so a CRLF checkout (core.autocrlf) still verifies."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def talent_log_posterior(frame):
    """Normalized exact log P(T | X, D, U, P, L, H, A) under the known SCM, shape (n, 3)."""
    cardinalities = {column.name: column.n_categories for column in _nodes()}
    frame = frame.reset_index(drop=True)
    log_post = _talent_log_posterior(frame, list(VERBALIZED), cardinalities, T_PRIOR)
    peak = log_post.max(axis=1, keepdims=True)
    return log_post - peak - np.log(np.exp(log_post - peak).sum(axis=1, keepdims=True))


def _prior(fit, frames):
    """Laplace-smoothed joint over the observed state cells; all cells get mass."""
    cardinalities = [int(fit[c].max()) + 1 for c in STATE]
    counts = fit.groupby(list(STATE)).size()
    total = counts.sum() + int(np.prod(cardinalities))
    mode = counts.idxmax()
    result = {}
    for name, frame in frames.items():
        cells = [tuple(row) for row in frame[list(STATE)].to_numpy()]
        log_p = [np.log((counts.get(cell, 0) + 1) / total) for cell in cells]
        result[name] = {
            "joint_nll": float(-np.mean(log_p)),
            "joint_map_accuracy": float(np.mean([cell == mode for cell in cells])),
            "marginal_accuracy": {
                c: float((frame[c] == fit[c].mode().iloc[0]).mean()) for c in STATE
            },
        }
    return {"cardinalities": dict(zip(STATE, cardinalities)), "metrics": result}


def _bayes_optimal(frames):
    metrics = {}
    for name, frame in frames.items():
        log_post = talent_log_posterior(frame)
        t = frame["T"].to_numpy()
        accuracy = float((log_post.argmax(axis=1) == t).mean())
        # X, D and U are read with probability one, so joint NLL/accuracy reduce to T's.
        metrics[name] = {
            "joint_nll": float(-log_post[np.arange(len(t)), t].mean()),
            "joint_map_accuracy": accuracy,
            "marginal_accuracy": {"X": 1.0, "T": accuracy, "D": 1.0, "U": 1.0},
        }
    return {"given": list(VERBALIZED), "t_prior": list(T_PRIOR), "metrics": metrics}


def compute(run_id):
    run = ROOT / run_id
    protocol = json.loads((run / "protocol.json").read_text())
    expected = protocol["input_sha256"][SIM_FACTUAL]
    if lf_sha256(SIM_FACTUAL) != expected:
        raise ValueError(f"{SIM_FACTUAL} differs from the run's protocol input hash")

    sim = pd.read_csv(SIM_FACTUAL).set_index("id")
    ids = protocol["split_ids"]
    fit = sim.loc[ids["fit"]]
    frames = {name: sim.loc[ids[key]] for name, key in SPLITS.items()}
    return {
        "run_id": run_id,
        "source": {"sim_factual": SIM_FACTUAL, "sim_factual_sha256": expected},
        "split_sizes": {"fit": len(fit), **{n: len(f) for n, f in frames.items()}},
        "note": (
            "Latent-free references on the benchmark split. The Bayes-optimal predictor "
            "uses the exact SCM posterior given the verbalized variables; it bounds any "
            "reader of the text in expectation, not on every finite sample."
        ),
        "prior": _prior(fit, frames),
        "bayes_optimal": _bayes_optimal(frames),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = compute(args.run_id)
    path = ROOT / args.run_id / "references.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name in ("prior", "bayes_optimal"):
        for split, m in result[name]["metrics"].items():
            print(f"{name:14s} {split:10s} NLL {m['joint_nll']:.4f}  joint acc {100 * m['joint_map_accuracy']:.2f}%")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
