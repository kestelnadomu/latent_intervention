"""Assumed operators (exp/sim/assumed.py) and the operator benchmark (simulation only)."""

import json

import numpy as np
import pytest
import yaml

from exp.benchmarks.operator import run
from exp.sim import assumed
from exp.sim.helpers import load_sim_config
from exp.sim_talent_motivation import talent_motivation as tm

CONFIG = load_sim_config(tm.CONFIG_PATH)
KERNEL = tm.build_symbolic_kernel(CONFIG)
PRIORS = {"X": [0.25] * 4, "T": list(tm.T_PRIOR), "M": list(tm.M_PRIOR)}


def test_state_distribution_is_normalized_and_matches_simulation():
    probs = assumed.state_distribution(KERNEL, PRIORS)
    assert probs.shape == (108,) and np.isclose(probs.sum(), 1.0)
    factual, _, _ = tm.build_scm(CONFIG).simulate(100_000, {"X": 0}, seed=4)
    from exp.benchmarks.operator.metrics import indices

    empirical = np.bincount(indices(factual, KERNEL), minlength=108) / len(factual)
    assert 0.5 * np.abs(probs - empirical).sum() < 0.03


def test_blocked_edge_keeps_the_mediator_and_restriction_drops_parents():
    from exp.benchmarks.operator.metrics import all_states, dense

    states = all_states(KERNEL)
    blocked = dense(assumed.block(KERNEL, {"D": ["X"]}), {"X": 3})
    expected_d = blocked @ states["D"].to_numpy(float)
    assert np.allclose(expected_d, states["D"].to_numpy(float))  # D' = D for every factual state
    restricted = assumed.restrict(KERNEL, ["X", "U"])
    assert [c.name for c in restricted.columns] == ["X", "U"]
    assert restricted.mechanism_coeffs == {"U": {"X": KERNEL.mechanism_coeffs["U"]["X"]}}


def test_matched_root_noise_keeps_the_middle_level():
    mu, sigma = assumed.matched_root_noise(tm.M_PRIOR, 1.5, 0.0)
    from exp.sim.symbolic import _category_probs

    probs = _category_probs(0.0, -np.inf, np.inf, mu, sigma, 3)
    assert np.isclose(probs[1], tm.M_PRIOR[1], atol=1e-3)
    assert np.isclose(mu, 1.0)


def test_intercept_refit_detects_the_false_edge_but_the_naive_fit_does_not():
    from exp.sim.symbolic import SymbolicIntervention
    from src.schema import ColumnSpec

    kernel = SymbolicIntervention.from_scm(
        [*KERNEL.columns, ColumnSpec("M", 3)], tm.base.MECHANISM_COEFFS, tm.base.NOISE_PARAMS
    )
    graph = assumed.add_mechanism(kernel, "M", {"X": 0.0}, assumed.matched_root_noise(tm.M_PRIOR, 1.5, 0.0))
    factual, _, _ = tm.build_scm(CONFIG).simulate(50_000, {"X": 0}, seed=5)
    naive = assumed.refit(graph, factual, fit_mechanisms=True, only=["M"])
    intercept = assumed.refit(graph, factual, fit_mechanisms=True, estimator="intercept", only=["M"])
    mle = assumed.refit(graph, factual, fit_mechanisms=True, estimator="mle", only=["M"])
    assert naive.mechanism_coeffs["M"]["X"] > 0.3  # no intercept: E[X]E[M]/E[X^2] ~ 0.43
    assert abs(intercept.mechanism_coeffs["M"]["X"]) < 0.02
    assert abs(mle.mechanism_coeffs["M"]["X"]) < 0.02
    assert intercept.noise_params["M"][1] < 0.05  # interior-only sigma collapses for a 3-level root
    assert abs(mle.noise_params["M"][1] - assumed.matched_root_noise(tm.M_PRIOR, 1.5, 0.0)[1]) < 0.03
    assert naive.mechanism_coeffs["D"] == KERNEL.mechanism_coeffs["D"]  # only M refitted


def test_ordered_probit_recovers_the_true_parameters():
    factual, _, _ = tm.build_scm(CONFIG).simulate(100_000, {"X": 0}, seed=6)
    fitted = assumed.refit(KERNEL, factual, fit_mechanisms=True, estimator="mle")
    for node, parents in KERNEL.mechanism_coeffs.items():
        for parent, coef in parents.items():
            assert fitted.mechanism_coeffs[node][parent] == pytest.approx(coef, abs=0.02)
        for est, true in zip(fitted.noise_params[node], KERNEL.noise_params[node]):
            assert est == pytest.approx(true, abs=0.02)


def test_benchmark_references_and_structural_identities(tmp_path, monkeypatch):
    protocol = yaml.safe_load(run.PROTOCOL.read_text(encoding="utf-8"))
    protocol.update(n_score=2000, o2a_n_fit=5000)
    protocol["o2b"] = {"n_fit": [300], "resamples": 1}
    protocol["estimators"] = ["naive", "mle"]
    protocol["o5"].update(refit_n_fit=[300], refit_resamples=1)
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    monkeypatch.setattr(run, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr("sys.argv", ["run", "--run-id", "t", "--protocol", str(path)])
    run.main()
    rows = json.loads((tmp_path / "reports" / "t" / "results.json").read_text())
    tv = {(r["condition"], r["target"]): r["tv"] for r in rows}
    for key in [("S1/true", "total"), ("S6/true", "total"), ("O4/block_X_D", "ps_D"),
                ("O3/drop_X_D", "ps_D"), ("O5/c0.0", "total"), ("S4/unaware", "total"),
                ("S3/unaware", "ps_DU")]:
        assert tv[key] == pytest.approx(0.0, abs=1e-6), key
    # dropping and blocking X -> D are indistinguishable at state level: D has no state descendant
    assert tv[("O3/drop_X_D", "total")] == pytest.approx(tv[("O4/block_X_D", "total")], abs=1e-9)
    shift_m = [r["mean_shift"]["M"]["directional"] for r in rows if r["group"] == "O5"]
    assert shift_m[0] == pytest.approx(0.0, abs=1e-9) and all(np.diff(shift_m) > 0)
    with pytest.raises(FileExistsError):
        run.main()  # results are never overwritten
