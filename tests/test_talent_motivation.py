"""Checks for the talent-motivation sibling experiment (simulation only; no API calls)."""

import numpy as np
import pandas as pd
import pytest

from exp.sim import talent_motivation as tm
from exp.sim import talent_sfm
from exp.sim.helpers import load_sim_config
from exp.sim.render import load_render_spec
from src.schema import load_intervention

CONFIG = load_sim_config(tm.CONFIG_PATH)
CARD = {c.name: c.n_categories for c in tm._nodes(CONFIG)}


def _simulate(n=4000, seed=3):
    return tm.build_scm(CONFIG).simulate(n, intervention=load_intervention(CONFIG), seed=seed)


def test_extension_leaves_the_talent_sfm_variables_unchanged():
    base_factual, _, _ = talent_sfm.build_scm().simulate(2000, intervention={"X": 3}, seed=7)
    factual, _, _ = tm.build_scm(CONFIG).simulate(2000, intervention={"X": 3}, seed=7)
    shared = ["X", "T", "D", "U", "P", "L", "H", "A"]
    assert factual[shared].equals(base_factual[shared])


def test_motivation_is_an_independent_root_invariant_under_do_x():
    factual, counterfactual, _ = _simulate(n=20_000)
    assert factual["M"].equals(counterfactual["M"])
    assert factual["M"].between(0, 2).all()
    marginal = factual["M"].value_counts(normalize=True).sort_index().to_numpy()
    assert np.allclose(marginal, tm.M_PRIOR, atol=0.02)
    for other in ["X", "T", "D", "U"]:
        assert abs(factual["M"].corr(factual[other])) < 0.03


def test_outcome_marginal_is_recentred():
    factual, _, _ = _simulate(n=20_000)
    marginal = factual["Y"].value_counts(normalize=True).sort_index().to_numpy()
    assert np.allclose(marginal, [0.24, 0.52, 0.24], atol=0.02)


def test_outcome_posterior_marginalizes_m_over_its_prior():
    factual, _, _ = _simulate(n=500)
    given = ["X", "T", "D", "U"]
    marginal = tm.outcome_posterior(factual, given, CARD)
    assert np.allclose(marginal.sum(axis=1), 1.0)
    mixture = sum(
        prior * tm.outcome_posterior(factual.assign(M=m), [*given, "M"], CARD)
        for m, prior in enumerate(tm.M_PRIOR)
    )
    assert np.allclose(marginal, mixture)


def test_knowing_m_improves_the_bayes_optimal_outcome_prediction():
    factual, _, _ = _simulate(n=20_000)
    y = factual["Y"].to_numpy()
    accuracy = {
        name: (probs.argmax(axis=1) == y).mean()
        for name, probs in tm.outcome_bayes_predictions(factual, CONFIG).items()
    }
    assert accuracy["all parents (T, M known)"] > accuracy["state only (T known, M unknown)"] + 0.05


def test_symbolic_kernel_is_the_talent_sfms():
    delta = load_intervention(CONFIG)
    ours = tm.build_symbolic_kernel(CONFIG).transition_matrix(delta).to_dense()
    base = talent_sfm.build_symbolic_kernel().transition_matrix(delta).to_dense()
    assert ours.shape == base.shape == (108, 108)
    assert np.allclose(ours.numpy(), base.numpy())


def test_codebook_covers_motivation_as_evidence():
    spec = load_render_spec(CONFIG)
    assert set(spec["evidence"]) == {"L", "H", "A", "M"}
    assert set(spec["evidence"]["M"]) == {0, 1, 2}
    assert "motivated" in spec["forbidden_terms"]
    assert "M" not in spec["hidden"]


def test_outcome_posterior_requires_observed_parents():
    frame = pd.DataFrame({"X": [0], "T": [1], "D": [1]})
    with pytest.raises(ValueError, match="U"):
        tm.outcome_posterior(frame, ["X", "T", "D"], CARD)
