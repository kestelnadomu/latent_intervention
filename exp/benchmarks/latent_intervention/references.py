"""Bayes-optimal counterfactual-state reference for the standard-DAG h_Z comparison.

Latent editors are scored against the realized counterfactual state S' of each test
unit, but no editor can predict S' perfectly from the factual CV: talent T reaches
the text only through its proxies, and the abducted noise of D and U is only known
up to an interval. This reference is the Bayes-optimal prediction of S' from
everything the factual CV contains: the exact posterior of the factual state given
X, D, U, P, L, H and A under the known SCM, pushed through the analytic kernel h_S
for ``do(X=3)``. Nothing is fitted.

It verifies the simulation inputs against the source hashes of the comparison run,
uses exactly the test units of its evaluation records, and writes
``references.json`` next to ``comparison.json``.

    .venv/bin/python -m exp.benchmarks.latent_intervention.references --run-id editor-clean-v1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp.benchmarks.semantic_decoder.references import lf_sha256, talent_log_posterior
from exp.sim_talent.talent_sfm import build_symbolic_kernel

ROOT = Path("reports/talent/editor_benchmarks")
PAIR_INDEX = "data/sim_talent/pair_index.csv"
SIM_FACTUAL = "data/sim_talent/sim_data_factual.csv"
SIM_COUNTERFACTUAL = "data/sim_talent/sim_data_counterfactual.csv"
INTERVENTION = {"X": 3}


def _test_units(comparison):
    """Official test IDs shared by every evaluation record, with their identity flags."""
    test_ids = None
    for group in comparison["groups"]:
        for path in group["evaluation_records"]:
            record = json.loads(Path(path).read_text(encoding="utf-8"))
            ids = record.get("official_test_ids", record.get("test_ids"))
            if test_ids is None:
                test_ids = sorted(ids)
            elif sorted(ids) != test_ids:
                raise ValueError(f"{path} uses different test units")
    pairs = pd.read_csv(PAIR_INDEX).set_index("id")
    if sorted(pairs.index[pairs["split"] == "test"]) != test_ids:
        raise ValueError("evaluation test units differ from the official test split")
    return test_ids, pairs.loc[test_ids, "is_identity"].to_numpy().astype(bool)


def compute(run_id):
    run = ROOT / run_id
    comparison = json.loads((run / "comparison.json").read_text(encoding="utf-8"))
    for path in (PAIR_INDEX, SIM_FACTUAL, SIM_COUNTERFACTUAL):
        if lf_sha256(path) != comparison["sources"][path]:
            raise ValueError(f"{path} differs from the comparison's source hash")
    test_ids, identity = _test_units(comparison)

    kernel = build_symbolic_kernel()
    transition = kernel.transition_matrix(INTERVENTION).to_dense().numpy()
    factual = pd.read_csv(SIM_FACTUAL).set_index("id").loc[test_ids].reset_index(drop=True)
    realized = pd.read_csv(SIM_COUNTERFACTUAL).set_index("id").loc[test_ids].reset_index(drop=True)
    columns = [column.name for column in kernel.columns]

    # Factual posterior: X, D and U are read exactly, T has its exact posterior.
    posterior = np.exp(talent_log_posterior(factual))
    source = np.zeros((len(factual), transition.shape[0]))
    for i, row in factual.iterrows():
        for t, weight in enumerate(posterior[i]):
            source[i, kernel.state_index({**{c: int(row[c]) for c in columns}, "T": t})] += weight
    target = source @ transition

    truth = np.array(
        [kernel.state_index({c: int(row[c]) for c in columns}) for _, row in realized.iterrows()]
    )
    rows = np.arange(len(truth))
    hit = target.argmax(1) == truth
    nll = -np.log(target[rows, truth])
    metrics = {}
    for name, mask in (("all", np.ones_like(identity)), ("identity", identity), ("nonidentity", ~identity)):
        metrics[name] = {
            "units": int(mask.sum()),
            "realized_joint_accuracy": float(hit[mask].mean()),
            "realized_joint_nll": float(nll[mask].mean()),
        }
    return {
        "run_id": run_id,
        "intervention": INTERVENTION,
        "sources": {p: comparison["sources"][p] for p in (PAIR_INDEX, SIM_FACTUAL, SIM_COUNTERFACTUAL)},
        "note": (
            "Bayes-optimal prediction of the realized counterfactual state from the factual "
            "CV content (exact SCM posterior pushed through h_S). It bounds every latent "
            "editor's realized-state accuracy in expectation, not on every finite sample."
        ),
        "bayes_optimal": {"given": ["X", "D", "U", "P", "L", "H", "A"], "metrics": metrics},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = compute(args.run_id)
    path = ROOT / args.run_id / "references.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, m in result["bayes_optimal"]["metrics"].items():
        print(
            f"bayes_optimal {name:11s} n={m['units']:4d}  joint acc "
            f"{100 * m['realized_joint_accuracy']:.2f}%  NLL {m['realized_joint_nll']:.3f}"
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
