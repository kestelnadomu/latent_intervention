"""Regression checks for integrating the editor variants with saved latent spaces."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src import pipeline
from src.latent_intervention.flow import workflow as flow_workflow
from src.config import load_config

VARIANTS = (
    *pipeline.BASE_VARIANTS,
    *flow_workflow.FLOW_VARIANTS,
    pipeline.ORACLE_VARIANT,
)


@pytest.mark.parametrize("dimension", [256, 768])
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("stage", ["train-manipulator", "evaluate"])
def test_cli_accepts_every_editor_for_both_candidate_dimensions(
    monkeypatch, dimension: int, variant: str, stage: str
) -> None:
    observed = []
    monkeypatch.setitem(pipeline.STAGES, stage, observed.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pipeline",
            stage,
            "--encoder-variant",
            "embeddinggemma",
            "--embeddinggemma-dim",
            str(dimension),
            "--decoder-variant",
            "independent",
            "--manipulator-variant",
            variant,
        ],
    )

    pipeline.main()

    assert len(observed) == 1
    config = observed[0]
    assert config["latent_intervention"]["variant"] == variant
    assert config["encoder"]["embeddinggemma_latent_dim"] == dimension
    assert config["paths"]["latents"] == (
        f"data/latents/talent/embeddinggemma_{dimension}/z_pairs.pt"
    )
    assert config["paths"]["decoder_model"] == (
        f"models/talent/embeddinggemma_{dimension}/g-independent/semantic_decoder.pt"
    )


def test_editor_paths_and_flow_teachers_do_not_mix_candidate_latent_spaces() -> None:
    model_paths = set()
    report_paths = set()
    teacher_paths = set()
    for dimension in (256, 768):
        for decoder in ("independent", "autoregressive"):
            namespace = f"talent/embeddinggemma_{dimension}/g-{decoder}"
            for variant in VARIANTS:
                config = load_config(
                    encoder_variant="embeddinggemma",
                    embeddinggemma_dim=dimension,
                    decoder_variant=decoder,
                    manipulator_variant=variant,
                )
                if variant in flow_workflow.FLOW_VARIANTS:
                    model_path = flow_workflow._flow_model_path(config, variant)
                    report_path = flow_workflow._flow_report_path(config, variant)
                else:
                    model_path = Path(config["paths"]["manipulator_model"])
                    report_path = Path(config["paths"]["eval_report"])
                assert model_path == Path(
                    f"models/{namespace}/h-{variant}/latent_intervention.pt"
                )
                assert report_path == Path(f"reports/{namespace}/h-{variant}/eval.json")
                assert model_path == Path(config["paths"]["manipulator_model"])
                assert report_path == Path(config["paths"]["eval_report"])
                assert model_path not in model_paths
                assert report_path not in report_paths
                model_paths.add(model_path)
                report_paths.add(report_path)
                teacher = flow_workflow._flow_teacher_path(config)
                assert teacher == Path(
                    f"models/{namespace}/h-state_flow/latent_intervention.pt"
                )
                teacher_paths.add(teacher)

    assert len(model_paths) == len(report_paths) == 4 * len(VARIANTS)
    assert len(teacher_paths) == 4
    assert teacher_paths.issubset(model_paths)
