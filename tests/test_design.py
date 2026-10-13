"""Per-unit counterfactual design: SCM.generate with blocked edges, interventions, partitions."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from exp.sim import paired_data
from exp.sim.design import build_partition, draw_interventions, load_design
from exp.sim.helpers import load_sim_config
from exp.sim.pairing import PairingError
from exp.sim.render import load_render_spec
from exp.sim.scm import SCM, linear_mechanism
from exp.sim.stage_simulate import stage_simulate
from exp.sim_talent_motivation.talent_motivation import CONFIG_PATH
from src.schema import ColumnSpec


def _chain_scm() -> SCM:
    """a -> b, a -> c, b -> c: c has a direct and a mediated path from a."""
    nodes = [ColumnSpec("a", 4), ColumnSpec("b", 3), ColumnSpec("c", 3)]
    noise = {
        "a": lambda rng, n: rng.integers(0, 4, n),
        "b": lambda rng, n: rng.normal(0.0, 0.4, n),
        "c": lambda rng, n: rng.normal(0.0, 0.4, n),
    }
    mechanisms = {
        "b": linear_mechanism({"a": 0.5}, 0, 2),
        "c": linear_mechanism({"a": 0.3, "b": 0.5}, 0, 2),
    }
    return SCM(nodes, ("a",), noise, mechanisms)


def _noise(epsilon: pd.DataFrame) -> dict[str, np.ndarray]:
    return {c.removeprefix("eps_"): epsilon[c].to_numpy() for c in epsilon if c.startswith("eps_")}


def test_generate_reproduces_simulate():
    scm = _chain_scm()
    factual, counterfactual, epsilon = scm.simulate(500, {"a": 3}, seed=2)
    noise = _noise(epsilon)
    assert scm.generate(noise).equals(factual.drop(columns="id"))
    assert scm.generate(noise, {"a": 3}).equals(counterfactual.drop(columns="id"))


def test_blocked_edge_keeps_the_mediator_factual():
    scm = _chain_scm()
    factual, _, epsilon = scm.simulate(2000, {"a": 3}, seed=4)
    reference = factual.drop(columns="id")
    path = scm.generate(_noise(epsilon), {"a": 0}, {"b": ["a"]}, reference)
    total = scm.generate(_noise(epsilon), {"a": 0})
    assert path["b"].equals(reference["b"])  # mediated path cut
    assert (path["a"] == 0).all()
    assert (total["b"] != reference["b"]).any()
    # c still sees a' directly, so it can differ from both the factual and the total world
    assert (path["c"] != total["c"]).any()


def test_generate_rejects_invalid_blocks():
    scm = _chain_scm()
    factual, _, epsilon = scm.simulate(50, {"a": 3}, seed=1)
    noise, reference = _noise(epsilon), factual.drop(columns="id")
    with pytest.raises(ValueError, match="root"):
        scm.generate(noise, {"a": 0}, {"a": ["b"]}, reference)
    with pytest.raises(ValueError, match="not parents"):
        scm.generate(noise, {"a": 0}, {"b": ["c"]}, reference)
    with pytest.raises(ValueError, match="reference"):
        scm.generate(noise, {"a": 0}, {"b": ["a"]})


def test_interventions_are_non_identity_and_order_independent():
    factual = pd.DataFrame({"id": [3, 1, 2, 4], "X": [0, 1, 3, 2]})
    drawn = draw_interventions(factual, "X", 4, seed=5)
    assert (drawn["do_value"].to_numpy() != factual["X"].to_numpy()).all()
    shuffled = draw_interventions(factual.iloc[::-1], "X", 4, seed=5).set_index("id")
    assert drawn.set_index("id").loc[shuffled.index].equals(shuffled)


def _small_config(tmp_path):
    config = copy.deepcopy(load_sim_config(CONFIG_PATH))
    config["n"] = 400
    config["design"]["partition"].update(dev=40, L=100, P=100, folds=5, floor_fraction=0.2)
    for key in ("sim_dir", "pair_index", "render_plan", "texts", "texts_counterfactual"):
        config["paths"][key] = str(tmp_path / key)
    config["paths"]["sim_dir"] = str(tmp_path / "sim")
    return config


def test_partition_roles(tmp_path):
    config = _small_config(tmp_path)
    spec = load_design(config)
    ids = np.arange(1, 401)
    pairs = pd.DataFrame({"id": ids, "split": ["test" if i % 4 == 0 else "train" for i in ids]})
    partition = build_partition(pairs, spec, seed=9)
    sizes = partition["partition"].value_counts().to_dict()
    assert sizes == {"pool": 260, "eval": 100, "dev": 40}
    pool = partition[partition["partition"] == "pool"]
    in_l = pool["L_rank"] > 0
    assert sorted(pool.loc[in_l, "L_rank"]) == list(range(1, 101))
    assert not (in_l & pool["in_P"]).any() and pool["in_P"].sum() == 100
    assert set(pool["fold"]) == set(range(5))
    small_budget = pool[(pool["L_rank"] > 0) & (pool["L_rank"] <= 10)]
    assert set(small_budget["fold"]) == set(range(5))  # every prefix spans all folds
    evaluation = partition[partition["partition"] == "eval"]
    assert set(evaluation["id"]) == set(pairs.loc[pairs["split"] == "test", "id"])
    assert evaluation["floor"].sum() == 20
    assert not partition.loc[partition["partition"] != "eval", "floor"].any()


def test_simulate_writes_and_validates_the_design(tmp_path):
    config = _small_config(tmp_path)
    stage_simulate(config)
    sim = tmp_path / "sim"
    factual = pd.read_csv(sim / "sim_data_factual.csv")
    counterfactual = pd.read_csv(sim / "sim_data_counterfactual.csv")
    interventions = pd.read_csv(sim / "interventions.csv")
    states = pd.read_csv(sim / "sim_data_counterfactual_states.csv")

    assert (counterfactual["X"] == interventions["do_value"]).all()
    assert (interventions["do_value"] != factual["X"]).all()
    assert len(states) == 400 * 4 * 3
    by_target = {t: g.set_index(["id", "do_value"]) for t, g in states.groupby("target")}
    factual_by_id = factual.set_index("id")
    for target, blocked in (("ps_D", ["D"]), ("ps_DU", ["D", "U"])):
        frame = by_target[target]
        ids = frame.index.get_level_values("id")
        for column in blocked:
            assert (frame[column].to_numpy() == factual_by_id.loc[ids, column].to_numpy()).all()
    for column in ("T", "M", "P", "L", "H", "A"):
        ids = by_target["total"].index.get_level_values("id")
        assert (by_target["total"][column].to_numpy() == factual_by_id.loc[ids, column].to_numpy()).all()
    drawn = by_target["total"].loc[list(zip(interventions["id"], interventions["do_value"]))]
    assert drawn.drop(columns="target").reset_index(drop=True).equals(
        counterfactual.drop(columns="id").reset_index(drop=True)
    )

    paired_data.load_pair_inputs(config)  # re-derives interventions and partition
    partition = pd.read_csv(sim / "partition.csv")
    partition.loc[0, "fold"] = (partition.loc[0, "fold"] + 1) % 5
    partition.to_csv(sim / "partition.csv", index=False)
    info = json.loads((sim / "simulation_info.json").read_text())
    info["files"]["partition"] = paired_data.sha256_file(sim / "partition.csv")
    (sim / "simulation_info.json").write_text(json.dumps(info, indent=2, sort_keys=True))
    with pytest.raises(PairingError, match="design"):
        paired_data.load_pair_inputs(config)


def test_fixed_query_configs_are_unchanged():
    config = load_sim_config("exp/sim_talent/config.yaml")
    assert load_design(config) is None
    assert set(paired_data.simulation_paths(config)) == {
        "factual", "counterfactual", "epsilon", "pair_index",
    }


def test_existing_talent_simulation_still_validates():
    """The talent SFM's recorded simulation must still match its fixed-query record."""
    config = load_sim_config("exp/sim_talent/config.yaml")
    if not (Path(config["paths"]["sim_dir"]) / "simulation_info.json").exists():
        pytest.skip("talent simulation not present")
    paired_data.load_pair_inputs(config)


def _pipeline_config(tmp_path):
    """The sibling config at n=40 with temporary pools and outputs, ready for fake generation."""
    from exp.sim import helpers

    config = _small_config(tmp_path)
    config["n"] = 40
    config["design"]["partition"].update(dev=4, L=10, P=10, folds=5, floor_fraction=0.5)
    config["llm"] = {"base_url": "https://example.invalid/v1", "api_key_env": None}
    config.pop("text_length", None)  # no tokenizer download
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    config["paths"].update(
        templates=str(text_dir / "templates.csv"),
        personas=str(text_dir / "personas.csv"),
        texts=str(text_dir / "cv_factual.csv"),
        texts_counterfactual=str(text_dir / "cv_counterfactual.csv"),
        texts_floor=str(text_dir / "cv_floor.csv"),
        render_plan=str(tmp_path / "sim" / "render_plan.csv"),
        pair_index=str(tmp_path / "sim" / "pair_index.csv"),
    )
    config["generation"].update(n_templates=2, n_personas=2, limit=None)
    pools = {
        "templates": pd.DataFrame({"template_id": [1, 2], "seed_id": [1, 2], "text": ["T one", "T two"]}),
        "personas": pd.DataFrame(
            {"persona_id": [1, 2], "job_title": ["Engineer", "Analyst"], "text": ["P one", "P two"]}
        ),
    }
    for kind, frame in pools.items():
        output, schema, id_column, expected, digest = helpers.pool_contract(config, kind)
        helpers.prepare_generated_csv(output, schema, id_column, expected, digest, create=True)
        frame.to_csv(output, index=False)
        helpers.finish_generation(output, expected, expected)
    return config


def test_floor_texts_swap_the_country_within_the_region(tmp_path, monkeypatch):
    from exp.sim import run, stage_cv
    from exp.sim.generate_text import GenerationResult

    config = _pipeline_config(tmp_path)
    calls = []

    def fake_generate(sample, *args, **kwargs):
        calls.append(sample)
        return GenerationResult(
            text=f"CV::{sample['candidate_info']}::{sample['evidence_guidance']}",
            response_id=f"r{len(calls)}",
            model="mock-model",
            finish_reason="stop",
        )

    monkeypatch.setattr(stage_cv, "generate_text_result", fake_generate)
    stage_simulate(config)
    with pytest.raises(FileNotFoundError):
        run.stage_generate_counterfactual_texts(config)  # factual texts must exist first
    run.stage_generate_texts(config)
    run.stage_generate_counterfactual_texts(config)
    run.stage_generate_floor_texts(config)
    partition = pd.read_csv(tmp_path / "sim" / "partition.csv")
    n_eval = int((partition["partition"] == "eval").sum())
    n_floor = int(partition["floor"].sum())
    assert n_floor == 5 and len(calls) == 40 + n_eval + n_floor
    run.stage_generate_floor_texts(config)
    assert len(calls) == 40 + n_eval + n_floor  # complete resume makes no calls

    factual = pd.read_csv(config["paths"]["texts"]).set_index("id")
    floor = pd.read_csv(config["paths"]["texts_floor"]).set_index("id")
    states = pd.read_csv(tmp_path / "sim" / "sim_data_factual.csv").set_index("id")
    countries = load_render_spec(config)["choices"]["X"]
    assert set(floor.index) == set(partition.loc[partition["floor"], "id"])
    for unit_id, row in floor.iterrows():
        region = int(states.at[unit_id, "X"])
        before = countries[region][int(factual.at[unit_id, "country_of_origin"])]
        after = countries[region][int(row["country_of_origin"])]
        assert before != after and f"Country of origin: {after}" in row["text"]
        assert row["template_id"] == factual.at[unit_id, "template_id"]
        assert row["persona_id"] == factual.at[unit_id, "persona_id"]
        # everything except the country is the factual rendering
        assert row["text"].replace(after, before) == factual.at[unit_id, "text"]
    run.stage_validate_pairs(config)


def test_floor_stage_requires_a_floor_design(tmp_path):
    from exp.sim import run

    config = _pipeline_config(tmp_path)
    config["design"]["floor_swap"] = None
    config["design"]["partition"]["floor_fraction"] = 0.0
    with pytest.raises(ValueError, match="floor_swap"):
        run.stage_generate_floor_texts(config)


def test_evidence_variants_are_planned_shared_and_replayed(tmp_path, monkeypatch):
    from exp.sim import run, stage_cv
    from exp.sim.generate_text import GenerationResult
    from exp.sim.render import cv_schema, sampled_columns

    config = _pipeline_config(tmp_path)
    spec = load_render_spec(config)
    assert sampled_columns(spec) == ["P", "X", "L", "H", "A", "M"]
    headers = cv_schema(spec)[1]

    def fake_generate(sample, *args, **kwargs):
        return GenerationResult(
            text=sample["evidence_guidance"], response_id="r", model="mock-model", finish_reason="stop"
        )

    monkeypatch.setattr(stage_cv, "generate_text_result", fake_generate)
    stage_simulate(config)
    run.stage_generate_texts(config)
    run.stage_generate_counterfactual_texts(config)
    factual = pd.read_csv(config["paths"]["texts"]).set_index("id")
    counterfactual = pd.read_csv(config["paths"]["texts_counterfactual"]).set_index("id")
    states = pd.read_csv(tmp_path / "sim" / "sim_data_factual.csv").set_index("id")

    for column in ("L", "H", "A", "M"):
        index = factual[headers[column]]
        assert index.nunique() > 1  # units differ in the variant they get
        # the guidance in the prompt is the planned variant at the unit's level
        for unit_id, variant in index.items():
            guidance = spec["evidence"][column][int(states.at[unit_id, column])][int(variant)]
            assert guidance in factual.at[unit_id, "text"]
        # X' keeps the variant: the evidence columns do not change under do(X)
        shared = counterfactual.index
        assert (counterfactual[headers[column]] == index.loc[shared]).all()
    run.stage_generate_floor_texts(config)
    run.stage_validate_pairs(config)

    tampered = factual.reset_index()
    tampered.loc[0, headers["M"]] = (tampered.loc[0, headers["M"]] + 1) % 3
    from exp.sim.render import render_context, validate_grounding

    with pytest.raises(ValueError, match="incorrect concrete M"):
        validate_grounding(render_context(config), tampered, counterfactual=False)
