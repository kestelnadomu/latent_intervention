"""latent_lab on synthetic data: renderers keep unchanged states, the pipeline runs."""

import numpy as np
import pandas as pd
import pytest
import torch

from latent_lab import (
    RENDERERS, Encoder, EditorRenderer, IdentityRenderer, OTEditorRenderer, LinearSCM, Pipeline, Reader, Renderer, StateEffectRenderer,
    OutOfFold, cross_fit, evaluate, factual_score, mean_predict, state_metrics,
)


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    states = pd.DataFrame({"X": rng.integers(0, 3, 400), "D": rng.integers(0, 2, 400)})
    z = torch.randn(400, 6) + torch.tensor(states.to_numpy(), dtype=torch.float32).repeat(1, 3)
    return z, states


TRAINED = ("flow", "editor", "ot_editor", "ot_flow")


@pytest.mark.parametrize("name", RENDERERS)
def test_renderers_keep_unchanged_states(toy, name):
    z, states = toy
    options = {"epochs": 2} if name in TRAINED else {}
    renderer = RENDERERS[name](["X", "D"], **options).fit(z, states)
    assert isinstance(renderer, Renderer)
    z_prime = renderer(z, states, states)
    assert z_prime.shape == z.shape
    if name != "constant":
        assert torch.allclose(z_prime, z, atol=1e-5)
    moved = renderer(z, states, states.assign(X=(states["X"] + 1) % 3))
    assert moved.shape == z.shape and torch.isfinite(moved).all()


def test_unfitted_renderer_raises(toy):
    z, states = toy
    assert torch.equal(IdentityRenderer(["X"])(z, states, states), z)
    with pytest.raises(ValueError):
        StateEffectRenderer(["X", "D"])(z, states, states)


def test_pipeline_with_a_plain_function(toy):
    z, states = toy
    reader = Reader(["X", "D"]).fit(z, states, epochs=5)
    renderer = StateEffectRenderer(["X", "D"]).fit(z, states)
    result = Pipeline(reader, lambda s: s.assign(X=2), renderer)(z)
    assert result.s_prime["X"].eq(2).all() and result.z_prime.shape == z.shape
    true_states = Pipeline(None, lambda s: s, renderer)(z, states)
    assert torch.allclose(true_states.z_prime, z, atol=1e-5)
    # the operator sees every column; a renderer on a subset ignores the others
    narrow = StateEffectRenderer(["X"]).fit(z, states)
    result = Pipeline(None, lambda s: s.assign(D=1 - s["D"]), narrow)(z, states)
    assert result.s_prime["D"].ne(states["D"].to_numpy()).all() and torch.allclose(result.z_prime, z, atol=1e-5)


def test_reader_activation(toy):
    z, states = toy
    relu = Reader(["X", "D"], hidden=1, activation="relu").fit(z, states, epochs=5)
    assert isinstance(relu.model.trunk[1], torch.nn.ReLU) and relu.accuracy(z, states).shape == (2,)
    with pytest.raises(ValueError):
        Reader(["X"], activation="tanh")


def test_cross_fit_reads_out_of_fold(toy):
    z, states = toy
    folds = np.arange(len(z)) % 4
    readings, readers = cross_fit(Reader(["X", "D"]), z, states, folds, labelled=folds != 3, seed=0)
    assert len(readers) == 4 and readings.shape == (len(z), 2)
    assert (readings["X"].to_numpy() == states["X"].to_numpy()).mean() > 0.5
    assert mean_predict(readers, z).shape == (len(z), 2)


def test_out_of_fold_reads_each_row_with_its_fold_reader(toy):
    z, states = toy
    folds = np.arange(len(z)) % 4
    readings, readers = cross_fit(Reader(["X", "D"]), z, states, folds, seed=0)
    oof = OutOfFold(readers, folds)
    assert (oof.predict(z).to_numpy() == readings.to_numpy()).all()
    assert (oof.rows(folds == 2).predict(z[folds == 2]).to_numpy() == readings[folds == 2].to_numpy()).all()
    assert evaluate(z, z, z, oof, states)["joint_acc"] == oof.metrics(z, states)["joint_acc"]
    with pytest.raises(ValueError):
        oof.proba(z[:10])


@pytest.mark.parametrize("editor_class", [EditorRenderer, OTEditorRenderer])
def test_editor_moves_towards_target(toy, editor_class):
    z, states = toy
    editor = editor_class(["X", "D"], epochs=100, lr=1e-2).fit(z, states)
    target = states.assign(X=(states["X"] + 1) % 3)
    hit = editor.critic.accuracy(editor(z, states, target), target)["X"]
    assert hit > editor.critic.accuracy(z, target)["X"] + 0.3


def test_state_metrics_and_evaluate(toy):
    z, states = toy
    perfect = {c: torch.nn.functional.one_hot(torch.tensor(states[c].to_numpy()), 3).float() for c in states}
    m = state_metrics(perfect, states)
    assert m["joint_acc"] == 1 and m["joint_nll"] < 1e-6 and m["ece_X"] < 1e-6
    uniform = {c: torch.full((len(z), 3), 1 / 3) for c in states}
    assert abs(state_metrics(uniform, states, joint=["X"])["joint_nll"] - np.log(3)) < 1e-5
    reader = Reader(["X", "D"]).fit(z, states, epochs=5)
    predict = lambda latent: (latent[:, 0] > 0).long()  # noqa: E731
    out = evaluate(z, z, z, reader, states, predict=predict, outcome=states["D"], groups=states["X"])
    assert out["energy"] == 0 and out["shift"] == 0 and out["down_agree"] == 1
    assert {"joint_acc", "real_joint_acc", "down_acc", "parity_gap"} <= set(out.index)


def test_latents_project_a_wider_cache(tmp_path):
    z = torch.nn.functional.normalize(torch.randn(5, 768), dim=1)
    torch.save({"ids": list(range(5)), "z": z, "test_ids": [0, 1], "z_prime": z[:2]}, tmp_path / "z.pt")
    latents = Encoder("embeddinggemma", 256).latents(path=tmp_path / "z.pt")
    assert latents.z.shape == (5, 256) and torch.allclose(latents.z.norm(dim=1), torch.ones(5))
    assert torch.allclose(latents.z_cf, latents.z[:2])


def test_factual_score_prefers_a_real_edit(toy):
    z, states = toy
    reader = Reader(["X", "D"]).fit(z, states, epochs=50)
    identity = factual_score(IdentityRenderer(["X", "D"]), z, states, reader)
    effect = factual_score(StateEffectRenderer(["X", "D"]).fit(z, states), z, states, reader)
    assert identity["shift"] == 0 and effect["realism"] < identity["realism"] and effect["fidelity"] > identity["fidelity"]


LEVELS = {"X": 4, "T": 3, "D": 3, "U": 3}
PARENTS = {"D": {"X": 0.30, "T": 0.60}, "U": {"X": 0.35, "T": 0.50}}
NOISE = {"D": (-0.10, 0.50), "U": (-0.15, 0.50)}


def _all_states(reps: int) -> pd.DataFrame:
    from itertools import product

    grid = pd.DataFrame(list(product(*[range(k) for k in LEVELS.values()])), columns=list(LEVELS))
    return grid.loc[grid.index.repeat(reps)].reset_index(drop=True)


def _flat(df: pd.DataFrame) -> np.ndarray:
    return (((df["X"] * 3 + df["T"]) * 3 + df["D"]) * 3 + df["U"]).to_numpy()


@pytest.mark.parametrize("blocked", [{}, {"D": ["X"]}])
def test_linear_scm_matches_exact_kernel(blocked):
    from exp.sim.symbolic import SymbolicIntervention
    from exp.sim.schema import ColumnSpec

    kernel = SymbolicIntervention.from_scm([ColumnSpec(c, k) for c, k in LEVELS.items()], PARENTS, NOISE)
    kernel.blocked = {child: set(parents) for child, parents in blocked.items()}
    exact = kernel.transition_matrix({"X": 3}).to_dense().numpy()
    reps = 2000
    states = _all_states(reps)
    out = LinearSCM(LEVELS, PARENTS, NOISE).counterfactual(states, {"X": 3}, blocked)
    empirical = np.zeros_like(exact)
    np.add.at(empirical, (_flat(states), _flat(out)), 1 / reps)
    assert np.abs(empirical - exact).max() < 0.05


def test_linear_scm_blocking_and_validation():
    scm = LinearSCM(LEVELS, PARENTS, NOISE)
    states = _all_states(5)
    out = scm.do(X=0, blocked={"D": ["X"], "U": ["X"]})(states)
    assert out["X"].eq(0).all() and out[["T", "D", "U"]].equals(states[["T", "D", "U"]])
    assert "D = clip(round(0.30·X + 0.60·T + ε), 0, 2)" in repr(scm)
    per_unit = scm.do(X=lambda s: (s["X"] + 1) % 4, blocked={"D": ["X"], "U": ["X"]})(states)
    assert per_unit["X"].equals((states["X"] + 1) % 4) and per_unit["D"].equals(states["D"])
    with pytest.raises(ValueError):
        scm.do(X=4)(states)
    with pytest.raises(ValueError):
        LinearSCM(LEVELS, {"U": {"D": 1.0}, "D": {"X": 1.0}}, {"U": (0, 1), "D": (0, 1)})  # not topological
    with pytest.raises(ValueError):
        LinearSCM(LEVELS, PARENTS, {"D": (0, 1)})


def test_cache_reloads_and_invalidates(tmp_path):
    from latent_lab.cache import cached

    calls = []

    def tick():  # defined outside __main__, so counting does not enter the key
        calls.append(1)

    # a notebook: its functions live in __main__, where the cache follows them
    g = {"__name__": "__main__", "cached": cached, "torch": torch, "pd": pd, "folder": tmp_path, "tick": tick}
    exec(
        "data = torch.arange(4.0)\n"
        "frame = pd.DataFrame({'a': [1, 2]})\n"
        "def helper():\n    return frame['a'].sum()\n"
        "def run(scale=2):\n    tick()\n    return data * scale + helper()\n",
        g,
    )
    step = "result = cached('x', lambda: run(), folder=folder)"
    exec(step, g)
    first = g["result"]
    exec(step, g)  # unchanged: reloaded
    assert torch.equal(g["result"], first) and len(calls) == 1
    exec("data = torch.arange(5.0)\n" + step, g)  # an input the function reads
    exec("frame = pd.DataFrame({'a': [1, 3]})\n" + step, g)  # an input of a notebook function it calls
    exec("def run(scale=3):\n    tick()\n    return data * scale + helper()\n" + step, g)  # a hyperparameter
    exec(step.replace("run()", "run() + 1"), g)  # the cached code itself
    assert len(calls) == 5 and len(list(tmp_path.glob("x-*.pt"))) == 5
    assert torch.equal(g["result"], torch.arange(5.0) * 3 + 4 + 1)


def test_cache_key_follows_only_used_modules(tmp_path, monkeypatch):
    import shutil

    import latent_lab.cache as cache
    from latent_lab import FlowRenderer

    package = tmp_path / "latent_lab"
    shutil.copytree(cache._PACKAGE, package, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(cache, "_PACKAGE", package)
    g = {"__name__": "__main__", "Reader": Reader, "FlowRenderer": FlowRenderer}
    exec("read = lambda: Reader(['X'])\nrender = lambda: FlowRenderer(['X'])", g)
    before = cache.digest(g["read"]), cache.digest(g["render"])
    (package / "renderer.py").write_text((package / "renderer.py").read_text() + "\n# edited\n")
    assert cache.digest(g["read"]) == before[0] and cache.digest(g["render"]) != before[1]
    (package / "reader.py").write_text((package / "reader.py").read_text() + "\n# edited\n")
    assert cache.digest(g["read"]) != before[0]  # and renderer keys too: renderer imports reader


def test_fit_logs_record_time_and_convergence(toy):
    z, states = toy
    reader = Reader(["X", "D"], hidden=1).fit(z, states, epochs=40, patience=3)
    log = reader.fit_log
    assert log["seconds"] > 0 and log["epochs_run"] == len(log["history"]) <= 40
    assert 1 <= log["best_epoch"] <= log["epochs_run"] and "validation" in log["history"][0]
    flow = RENDERERS["flow"](["X", "D"], epochs=3).fit(z, states)
    assert flow.fit_log["epochs_run"] == 3 and "validation" in flow.fit_log["history"][-1]
    closed = StateEffectRenderer(["X", "D"]).fit(z, states)
    assert set(closed.fit_log) == {"seconds"}


@pytest.mark.parametrize("name", ["editor", "ot_editor", "ot_flow"])
def test_early_stopping(toy, name):
    z, states = toy
    options = {"lr": 1e-2} if name == "ot_editor" else {}  # it learns slowly from its near-identity start
    renderer = RENDERERS[name](["X", "D"], epochs=200, patience=2, **options).fit(z, states)
    log = renderer.fit_log
    assert log["stopped_early"] and log["epochs_run"] < 200 and "validation" in log["history"][0]
    assert log["best_epoch"] == min(range(len(log["history"])), key=lambda i: log["history"][i]["validation"]) + 1
