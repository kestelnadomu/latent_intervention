from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from exp.sim import helpers, paired_data, run, stage_cv, stage_simulate
from exp.sim.generate_text import GenerationResult
from exp.sim.pairing import build_pair_index, build_render_plan, materialize_binned_values


def _saved_attempts(output: Path) -> dict[int, int]:
    info = json.loads(helpers.generation_info_path(output).read_text(encoding="utf-8"))
    return {int(row_id): int(count) for row_id, count in info["attempts"].items()}


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
    # simulate archives the text outputs and the render plan too: every path it
    # touches must be redirected, or the test rewrites the real data/ directory.
    config["paths"]["render_plan"] = str(tmp_path / "sim" / "render_plan.csv")
    config["paths"]["texts"] = str(tmp_path / "text" / "cv_factual.csv")
    config["paths"]["texts_counterfactual"] = str(tmp_path / "text" / "cv_counterfactual.csv")

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
    helpers.write_json(sim_dir / "simulation_info.json", paired_data.simulation_record(config))

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
        output, schema, id_column, expected, digest = helpers.pool_contract(config, kind)
        helpers.prepare_generated_csv(
            output, schema, id_column, expected, digest, create=True
        )
        frame.to_csv(output, index=False)
        helpers.finish_generation(output, expected, expected)
    return config


@pytest.mark.parametrize(
    ("include_train", "expected_ids", "generated_ids", "identity_ids"),
    [
        pytest.param(None, {1, 2, 3, 4}, {1, 3}, {2, 4}, id="default-all"),
        pytest.param(True, {1, 2, 3, 4}, {1, 3}, {2, 4}, id="explicit-all"),
        pytest.param(False, {1, 2}, {1}, {2}, id="test-only"),
    ],
)
def test_configured_counterfactual_generation_and_identity_copy(
    tmp_path: Path,
    monkeypatch,
    include_train: bool | None,
    expected_ids: set[int],
    generated_ids: set[int],
    identity_ids: set[int],
) -> None:
    config = _paired_config(tmp_path)
    if include_train is None:
        config["generation"].pop("include_train_counterfactual_texts", None)
    else:
        config["generation"]["include_train_counterfactual_texts"] = include_train
    calls = []

    def fake_generate(sample, *args, **kwargs):
        calls.append(sample)
        return GenerationResult(
            text=f"CV::{sample['candidate_info']}",
            response_id=f"response-{len(calls)}",
            model="mock-model",
            finish_reason="stop",
        )

    monkeypatch.setattr(stage_cv, "generate_text_result", fake_generate)

    run.stage_generate_texts(config)
    assert len(calls) == 4
    run.stage_generate_counterfactual_texts(config)
    expected_calls = 4 + len(generated_ids)
    assert len(calls) == expected_calls
    run.stage_generate_counterfactual_texts(config)
    assert len(calls) == expected_calls  # complete resume makes no calls

    factual = pd.read_csv(config["paths"]["texts"]).set_index("id")
    counterfactual = pd.read_csv(config["paths"]["texts_counterfactual"]).set_index("id")
    assert set(factual.index) == {1, 2, 3, 4}
    assert set(counterfactual.index) == expected_ids
    for row_id in identity_ids:
        assert counterfactual.at[row_id, "text"] == factual.at[row_id, "text"]
        assert counterfactual.at[row_id, "generation_mode"] == "identity_copy"
        assert pd.isna(counterfactual.at[row_id, "response_id"])
    for row_id in generated_ids:
        assert counterfactual.at[row_id, "generation_mode"] == "generated"
        assert counterfactual.at[row_id, "model"] == "mock-model"
        assert "Gender: Male" in counterfactual.at[row_id, "text"]
    assert counterfactual.at[1, "response_id"] == "response-5"
    assert counterfactual.at[1, "template_id"] == factual.at[1, "template_id"]
    assert counterfactual.at[1, "persona_id"] == factual.at[1, "persona_id"]
    assert counterfactual.at[1, "age"] == factual.at[1, "age"]
    attempts = _saved_attempts(Path(config["paths"]["texts_counterfactual"]))
    assert attempts == {row_id: 1 for row_id in generated_ids}
    run.stage_validate_pairs(config)


def test_counterfactual_coverage_change_is_rejected_before_an_api_call(
    tmp_path: Path, monkeypatch
) -> None:
    config = _paired_config(tmp_path)
    calls = []

    def fake_generate(sample, *args, **kwargs):
        calls.append(sample)
        return GenerationResult(text="CV", model="mock-model", finish_reason="stop")

    monkeypatch.setattr(stage_cv, "generate_text_result", fake_generate)
    run.stage_generate_texts(config)
    run.stage_generate_counterfactual_texts(config)
    completed_calls = len(calls)

    config["generation"]["include_train_counterfactual_texts"] = False
    with pytest.raises(RuntimeError, match="incompatible"):
        run.stage_generate_counterfactual_texts(config)
    assert len(calls) == completed_calls


def test_invalid_counterfactual_coverage_setting_fails_before_generation(
    tmp_path: Path, monkeypatch
) -> None:
    config = _paired_config(tmp_path)
    config["generation"]["include_train_counterfactual_texts"] = "yes"
    calls = []

    monkeypatch.setattr(
        stage_cv,
        "generate_text_result",
        lambda *args, **kwargs: calls.append(True),
    )
    with pytest.raises(ValueError, match="must be true or false"):
        run.stage_generate_texts(config)
    assert calls == []
    assert not Path(config["paths"]["texts"]).exists()


@pytest.mark.parametrize(
    ("problem", "message"),
    [("missing", "coverage does not match"), ("unexpected", "unexpected IDs")],
)
def test_validation_rejects_incorrect_counterfactual_coverage(
    tmp_path: Path, monkeypatch, problem: str, message: str
) -> None:
    config = _paired_config(tmp_path)

    monkeypatch.setattr(
        stage_cv,
        "generate_text_result",
        lambda sample, *args, **kwargs: GenerationResult(
            text=f"CV::{sample['candidate_info']}",
            model="mock-model",
            finish_reason="stop",
        ),
    )
    run.stage_generate_texts(config)
    run.stage_generate_counterfactual_texts(config)

    output = Path(config["paths"]["texts_counterfactual"])
    frame = pd.read_csv(output)
    if problem == "missing":
        frame = frame.iloc[:-1]
    else:
        extra = frame.iloc[[0]].copy()
        extra["id"] = 99
        frame = pd.concat([frame, extra], ignore_index=True)
    frame.to_csv(output, index=False)

    with pytest.raises(ValueError, match=message):
        run.stage_validate_pairs(config)


def test_generation_resumes_and_rejects_changed_prompt_before_a_call(
    tmp_path: Path, monkeypatch
) -> None:
    config = _paired_config(tmp_path)
    config["generation"]["limit"] = 1
    calls = []

    def fake_generate(sample, *args, **kwargs):
        calls.append(sample)
        return GenerationResult(
            text=f"CV {len(calls)}",
            model="mock-model",
            finish_reason="stop",
        )

    monkeypatch.setattr(stage_cv, "generate_text_result", fake_generate)
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
    helpers.write_json(
        Path(config["paths"]["sim_dir"]) / "simulation_info.json",
        paired_data.simulation_record(config),
    )

    with pytest.raises(ValueError, match="seeded split"):
        paired_data.load_pair_inputs(config)


def test_generation_info_and_csv_must_exist_together(tmp_path: Path) -> None:
    output = tmp_path / "texts.csv"
    helpers.write_json(
        helpers.generation_info_path(output),
        {"input_digest": "digest"},
    )

    with pytest.raises(RuntimeError, match="both exist or both be absent"):
        helpers.prepare_generated_csv(
            output,
            ["id", "text"],
            "id",
            {1},
            "digest",
            create=True,
        )


def test_generation_attempts_are_persisted_and_capped(tmp_path: Path) -> None:
    output = tmp_path / "texts.csv"
    info_path = helpers.generation_info_path(output)
    helpers.write_json(info_path, {"attempts": {}})
    calls = []
    rejected_results = [
        GenerationResult("unfinished", model="mock-model", finish_reason="length"),
        GenerationResult("finished", model="", finish_reason="stop"),
        GenerationResult("", model="mock-model", finish_reason="stop"),
    ]

    def rejected() -> GenerationResult:
        calls.append(True)
        assert _saved_attempts(output) == {7: len(calls)}
        return rejected_results[len(calls) - 1]

    with pytest.raises(RuntimeError, match="exhausted its 3 generation attempts"):
        helpers.generate_with_attempts(output, 7, 3, rejected)
    assert len(calls) == 3
    assert _saved_attempts(output) == {7: 3}
    with pytest.raises(RuntimeError, match="exhausted its 3 generation attempts"):
        helpers.generate_with_attempts(output, 7, 3, rejected)
    assert len(calls) == 3


def test_archive_existing_uses_one_date_only_suffix(tmp_path: Path) -> None:
    first = tmp_path / "sim_data_factual.csv"
    second = tmp_path / "cv_factual.generation.json"
    first.write_text("data", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")

    stamp = stage_simulate.datetime.now().strftime("%y-%m-%d")
    stage_simulate._archive_existing([first, second])

    assert not first.exists()
    assert not second.exists()
    assert (tmp_path / f"sim_data_factual_{stamp}.csv").read_text(encoding="utf-8") == "data"
    archived_info = tmp_path / f"cv_factual_{stamp}.generation.json"
    assert archived_info.read_text(encoding="utf-8") == "{}"


def test_spent_attempt_budget_is_cleared_only_for_unwritten_ids(tmp_path: Path) -> None:
    output = tmp_path / "pool.csv"
    helpers.write_json(
        helpers.generation_info_path(output),
        {"input_digest": "d", "attempts": {"1": 3, "2": 3}},
    )
    pd.DataFrame({"pool_id": [1], "text": ["written"]}).to_csv(output, index=False)

    assert helpers.reset_failed_attempts(output, "pool_id") == 1

    info = json.loads(helpers.generation_info_path(output).read_text(encoding="utf-8"))
    assert info["attempts"] == {"1": 3}          # a written row keeps its history
    assert info["input_digest"] == "d"           # digests are untouched

    assert helpers.reset_failed_attempts(output, "pool_id") == 0


def test_coverage_is_recorded_when_a_row_exhausts_its_attempts(
    tmp_path: Path, monkeypatch
) -> None:
    config = _paired_config(tmp_path)
    calls = {"n": 0}

    def flaky_generate(sample, prompt, templates, api_key=None, base_url=None):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("endpoint down")
        return GenerationResult(
            text=f"cv {calls['n']}",
            response_id=f"r{calls['n']}",
            model="m",
            system_fingerprint="fp",
            finish_reason="stop",
        )

    monkeypatch.setattr(stage_cv, "generate_text_result", flaky_generate)

    with pytest.raises(RuntimeError, match="exhausted"):
        run.stage_generate_texts(config)

    output = Path(config["paths"]["texts"])
    info = json.loads(helpers.generation_info_path(output).read_text(encoding="utf-8"))
    assert info["n_completed"] == len(pd.read_csv(output))
    assert info["n_completed"] > 0
    assert info["complete"] is False
