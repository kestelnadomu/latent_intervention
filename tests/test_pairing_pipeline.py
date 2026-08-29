from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from exp.sim import run
from exp.sim.generate_text import GenerationResult
from exp.sim.pairing import build_pair_index, build_render_plan, materialize_binned_values


def test_pair_index_and_render_plan_are_order_independent() -> None:
    factual = pd.DataFrame(
        {"id": range(1, 11), "G": [0, 1] * 5, "E": [0, 1, 2, 0, 1] * 2}
    )
    counterfactual = factual.copy()
    counterfactual["G"] = 1
    counterfactual.loc[counterfactual["id"] == 1, "E"] = 1

    first = build_pair_index(factual, counterfactual, ["G", "E"], seed=17)
    second = build_pair_index(
        factual.sample(frac=1, random_state=3),
        counterfactual.sample(frac=1, random_state=4),
        ["G", "E"],
        seed=17,
    )

    pd.testing.assert_frame_equal(first, second)
    assert (first["split"] == "test").sum() == 2
    assert first.loc[first["id"] == 2, "is_identity"].item()
    assert not first.loc[first["id"] == 1, "is_identity"].item()

    plan = build_render_plan(first["id"], [3, 1, 2], [2, 1], ["A", "W"], seed=9)
    reordered = build_render_plan(
        reversed(first["id"].tolist()), [2, 3, 1], [1, 2], ["A", "W"], seed=9
    )
    pd.testing.assert_frame_equal(plan, reordered)

    row = plan.iloc[0]
    bins = {"A": {0: (24, 32), 1: (33, 44)}, "W": {0: (2, 5), 1: (6, 10)}}
    factual_values = materialize_binned_values({"A": 0, "W": 0}, bins, row)
    counterfactual_values = materialize_binned_values({"A": 0, "W": 1}, bins, row)
    assert factual_values["A"] == counterfactual_values["A"]
    assert 2 <= factual_values["W"] <= 5
    assert 6 <= counterfactual_values["W"] <= 10


def test_simulation_writes_shared_noise_pairs_and_bounded_split(tmp_path: Path) -> None:
    config = deepcopy(run.load_sim_config())
    config["n"] = 11
    config["seed"] = 19
    config["split"] = {"seed": 7, "test_fraction": 0.2}
    config["paths"]["sim_dir"] = str(tmp_path / "sim")
    config["paths"]["pair_index"] = str(tmp_path / "sim" / "pair_index.csv")

    run.stage_simulate(config)

    factual = pd.read_csv(tmp_path / "sim" / "sim_data_factual.csv")
    counterfactual = pd.read_csv(tmp_path / "sim" / "sim_data_counterfactual.csv")
    epsilon = pd.read_csv(tmp_path / "sim" / "sim_data_epsilon.csv")
    pairs = pd.read_csv(tmp_path / "sim" / "pair_index.csv")

    assert set(factual["id"]) == set(counterfactual["id"]) == set(epsilon["id"])
    assert (counterfactual["G"] == 1).all()
    assert factual["R"].equals(counterfactual["R"])
    assert factual["A"].equals(counterfactual["A"])
    assert (pairs["split"] == "test").sum() == 2
    assert (pairs["split"] == "train").sum() == 9

    columns = list(config["schema"]["columns"])
    target_rows = factual["G"] == 1
    assert factual.loc[target_rows, columns].equals(counterfactual.loc[target_rows, columns])
    assert pairs.set_index("id").loc[factual.loc[target_rows, "id"], "is_identity"].all()


def _paired_config(tmp_path: Path) -> dict:
    config = deepcopy(run.load_sim_config())
    sim_dir = tmp_path / "sim"
    text_dir = tmp_path / "text"
    sim_dir.mkdir()
    text_dir.mkdir()

    config["llm"] = {"base_url": "https://example.invalid/v1", "api_key_env": None}
    config["n"] = 4
    config["split"] = {"seed": 25, "test_fraction": 0.5}
    config["generation"].update({"n_templates": 2, "n_personas": 2, "limit": None})
    config["paths"].update(
        {
            "sim_dir": str(sim_dir),
            "pair_index": str(sim_dir / "pair_index.csv"),
            "render_plan": str(sim_dir / "render_plan.csv"),
            "templates": str(text_dir / "templates.csv"),
            "personas": str(text_dir / "personas.csv"),
            "texts": str(text_dir / "cv_factual.csv"),
            "texts_counterfactual": str(text_dir / "cv_counterfactual.csv"),
        }
    )
    for name in ("templates", "personas", "cv"):
        prompt_copy = tmp_path / f"{name}_prompt.yaml"
        prompt_copy.write_text(
            Path(config["prompts"][name]).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        config["prompts"][name] = str(prompt_copy)

    factual = pd.DataFrame(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [3, 2, 0, 2, 2, 2, 2, 0, 0, 2],
            [4, 3, 1, 0, 3, 2, 1, 1, 0, 1],
        ],
        columns=["id", "R", "G", "A", "E", "S", "W", "V", "C", "Q"],
    )
    counterfactual = factual.copy()
    counterfactual.loc[counterfactual["id"].isin([1, 3]), "G"] = 1
    counterfactual.loc[counterfactual["id"] == 1, "W"] = 1
    factual.to_csv(sim_dir / "sim_data_factual.csv", index=False)
    counterfactual.to_csv(sim_dir / "sim_data_counterfactual.csv", index=False)
    pd.DataFrame({"id": [1, 2, 3, 4], "eps": [0.0] * 4}).to_csv(
        sim_dir / "sim_data_epsilon.csv", index=False
    )
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "split": ["test", "test", "train", "train"],
            "is_identity": [False, True, False, True],
        }
    ).to_csv(sim_dir / "pair_index.csv", index=False)
    run._write_json(sim_dir / "simulation_info.json", run._simulation_record(config))

    template_frame = pd.DataFrame(
        {
            "template_id": [1, 2],
            "seed_id": [1, 2],
            "text": ["Template one", "Template two"],
        }
    )
    persona_frame = pd.DataFrame(
        {
            "persona_id": [1, 2],
            "job_title": ["Engineer", "Researcher"],
            "text": ["Persona one", "Persona two"],
        }
    )
    for kind, frame in (("templates", template_frame), ("personas", persona_frame)):
        output, schema, id_column, expected, digest = run._pool_contract(config, kind)
        run._prepare_generated_csv(
            output, schema, id_column, expected, digest, create=True
        )
        frame.to_csv(output, index=False)
        run._finish_generation(output, expected, expected)
    return config


def test_test_only_counterfactual_generation_and_identity_copy(
    tmp_path: Path, monkeypatch
) -> None:
    config = _paired_config(tmp_path)
    calls = []

    def fake_generate(sample, *args, **kwargs):
        calls.append(sample)
        return GenerationResult(
            text=f"CV::{sample['candidate_info']}",
            response_id=f"response-{len(calls)}",
            model="mock-model",
            finish_reason="stop",
        )

    monkeypatch.setattr(run, "generate_text_result", fake_generate)

    run.stage_generate_texts(config)
    assert len(calls) == 4
    run.stage_generate_counterfactual_texts(config)
    assert len(calls) == 5  # one nonidentity test call; the identity is copied
    run.stage_generate_counterfactual_texts(config)
    assert len(calls) == 5  # complete resume makes no calls

    factual = pd.read_csv(config["paths"]["texts"]).set_index("id")
    counterfactual = pd.read_csv(config["paths"]["texts_counterfactual"]).set_index("id")
    assert set(factual.index) == {1, 2, 3, 4}
    assert set(counterfactual.index) == {1, 2}
    assert counterfactual.at[2, "text"] == factual.at[2, "text"]
    assert counterfactual.at[2, "generation_mode"] == "identity_copy"
    assert counterfactual.at[1, "generation_mode"] == "generated"
    assert counterfactual.at[1, "response_id"] == "response-5"
    assert counterfactual.at[1, "model"] == "mock-model"
    assert counterfactual.at[1, "template_id"] == factual.at[1, "template_id"]
    assert counterfactual.at[1, "persona_id"] == factual.at[1, "persona_id"]
    assert counterfactual.at[1, "age"] == factual.at[1, "age"]
    assert "Gender: Male" in counterfactual.at[1, "text"]
    run.stage_validate_pairs(config)


def test_generation_resumes_and_rejects_changed_prompt_before_a_call(
    tmp_path: Path, monkeypatch
) -> None:
    config = _paired_config(tmp_path)
    config["generation"]["limit"] = 1
    calls = []

    def fake_generate(sample, *args, **kwargs):
        calls.append(sample)
        return GenerationResult(text=f"CV {len(calls)}")

    monkeypatch.setattr(run, "generate_text_result", fake_generate)
    run.stage_generate_texts(config)
    run.stage_generate_texts(config)
    assert len(calls) == 2

    config["generation"]["limit"] = None  # scheduling is deliberately not semantic
    run.stage_generate_texts(config)
    assert len(calls) == 4

    template_prompt = Path(config["prompts"]["templates"])
    original_template_prompt = template_prompt.read_text(encoding="utf-8")
    template_prompt.write_text(original_template_prompt + "\n# changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="incompatible"):
        run.stage_generate_texts(config)
    assert len(calls) == 4

    template_prompt.write_text(original_template_prompt, encoding="utf-8")
    cv_prompt = Path(config["prompts"]["cv"])
    cv_prompt.write_text(cv_prompt.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="incompatible"):
        run.stage_generate_counterfactual_texts(config)
    assert len(calls) == 4


def test_pair_index_must_match_the_recorded_seeded_split(tmp_path: Path) -> None:
    config = _paired_config(tmp_path)
    pairs = pd.read_csv(config["paths"]["pair_index"])
    pairs["split"] = ["train", "test", "test", "train"]
    pairs.to_csv(config["paths"]["pair_index"], index=False)
    run._write_json(
        Path(config["paths"]["sim_dir"]) / "simulation_info.json",
        run._simulation_record(config),
    )

    with pytest.raises(ValueError, match="seeded split"):
        run._load_pair_inputs(config)


def test_generation_info_and_csv_must_exist_together(tmp_path: Path) -> None:
    output = tmp_path / "texts.csv"
    run._write_json(
        run._generation_info_path(output),
        {"input_digest": "digest"},
    )

    with pytest.raises(RuntimeError, match="both exist or both be absent"):
        run._prepare_generated_csv(
            output,
            ["id", "text"],
            "id",
            {1},
            "digest",
            create=True,
        )
