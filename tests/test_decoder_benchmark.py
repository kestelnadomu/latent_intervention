import copy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import yaml

from src import pipeline
from src.artifact_io import write_json
from src.config import CONFIG_PATH, load_config, resolve_paths
from src.decoder_benchmark.decoder_benchmark_evaluation import distribution_metrics, evaluate_case
from src.decoder_benchmark.decoder_benchmark_matrix import (
    annotate_result,
    case_name,
    discover,
    encoding_config,
    freeze_selection,
)
from src.decoder_benchmark.decoder_benchmark_report import render
from src.decoder_benchmark.decoder_experiment import publish_results, run_case, verify_run
from src.schema import ColumnSpec
from src.semantic_decoder import joint_nll, make_semantic_decoder


@pytest.mark.parametrize(
    "tag,dimension",
    [
        ("langvae", 128),
        ("nomic_64", 64),
        ("nomic_768", 768),
        ("qwen3_32", 32),
        ("qwen3_1024", 1024),
        ("embeddinggemma_128", 128),
        ("embeddinggemma_768", 768),
    ],
)
def test_configuration_namespaces_and_dimensions(tag, dimension):
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    before = copy.deepcopy(raw)
    template = encoding_config(raw, Path(tag))
    from src.encoder_protocols import encoder_dimension

    assert encoder_dimension(template["encoder"]) == dimension
    assert "{decoder}" in template["paths"]["decoder_model"]
    for decoder in ("independent", "autoregressive"):
        config = copy.deepcopy(template)
        config["semantic_decoder"]["variant"] = decoder
        resolve_paths(config)
        assert f"/{tag}/g-{decoder}/" in config["paths"]["decoder_model"]
    assert raw == before


def test_selection_requires_complete_matrix_and_ignores_test_scores():
    def result(case, variant, val, test):
        return {
            "case": case,
            "decoder": variant,
            "deployment_seed": 42,
            "summary": {"validation_joint_nll": {"mean": val, "std": 0.01}},
            "test_nll": test,
        }

    rows = [
        result("a", "independent", 1.0, 9.0),
        result("b", "autoregressive", 2.0, 0.1),
    ]
    with pytest.raises(ValueError, match="every case"):
        freeze_selection(rows[:1], ["a", "b"])
    selection = freeze_selection(rows, ["a", "b"])
    assert selection["recommended_case"] == "a"
    assert selection["test_metrics_used_for_selection"] is False
    assert selection["best_by_decoder"]["autoregressive"] == "b"


def test_exact_distribution_metrics():
    columns = [ColumnSpec("a", 2), ColumnSpec("b", 2)]
    model = make_semantic_decoder(
        "independent", latent_dim=2, columns=columns, hidden_dim=4
    )
    p = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.6, 0.1, 0.2, 0.1]])
    model.log_joint = lambda z: p[z[:, 0].long()].log()
    metrics = distribution_metrics(
        model,
        torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        {"a": torch.tensor([0, 1]), "b": torch.tensor([1, 0])},
    )
    assert metrics["joint_nll"] == pytest.approx(math.log(5))
    assert metrics["joint_map_accuracy"] == 0
    assert metrics["joint_brier"] == pytest.approx(0.96)
    assert metrics["marginals"]["a"]["accuracy"] == 0
    assert metrics["marginals"]["b"]["accuracy"] == 1
    assert metrics["macro_marginal_ece"] == pytest.approx(0.5)


@pytest.mark.parametrize("variant", ["independent", "autoregressive"])
def test_report_nll_matches_training_scale(variant):
    model = make_semantic_decoder(
        variant,
        latent_dim=3,
        columns=[ColumnSpec("a", 2), ColumnSpec("b", 3)],
        hidden_dim=4,
    )
    z = torch.arange(21).reshape(7, 3).float() / 10
    targets = {"a": torch.arange(7) % 2, "b": torch.arange(7) % 3}
    metrics = distribution_metrics(model, z, targets)
    assert metrics["joint_nll"] == pytest.approx(joint_nll(model, z, targets), abs=1e-6)


def test_discovery_rejects_mismatched_source_data(tmp_path, monkeypatch):
    import src.decoder_benchmark.decoder_benchmark_matrix as matrix

    latent_root = tmp_path / "latents"
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["paths"]["latents"] = str(latent_root / "{encoder}/z_pairs.pt")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    for tag in ("langvae", "nomic_64"):
        folder = latent_root / tag
        folder.mkdir(parents=True)
        (folder / "z_pairs.pt").write_bytes(b"fixture")
        write_json(folder / "z_pairs.info.json", {"input_sha256": {"data": "same"}})
    artifact = SimpleNamespace(
        ids=list(range(20)),
        train_ids=list(range(16)),
        test_ids=list(range(16, 20)),
        z=torch.zeros(20, 128),
        is_identity=torch.zeros(4, dtype=torch.bool),
        artifact_sha256="fixture",
    )
    monkeypatch.setattr(matrix, "load_latent_artifact", lambda config: artifact)
    monkeypatch.setattr(pipeline, "_aligned_targets", lambda *args: {})
    plan = discover(config_path, latent_root)
    assert len(plan["configs"]) == 4
    assert len({case_name(c) for c in plan["configs"]}) == 4
    write_json(
        latent_root / "nomic_64/z_pairs.info.json",
        {"input_sha256": {"data": "changed"}},
    )
    with pytest.raises(ValueError, match="different text/pair inputs"):
        discover(config_path, latent_root)


def test_tiny_full_workflow_freezes_evaluates_resumes_reports_and_archives(
    tmp_path, monkeypatch
):
    base = load_config()
    base["semantic_decoder"].update(epochs=3, batch_size=8, hidden_dim=8, verbose=False)
    base["semantic_decoder"]["search"].update(
        trials=2, final_seeds=[42, 43], hidden_dim=[8, 12]
    )
    sim = tmp_path / "sim.yaml"
    sim.write_text(
        yaml.safe_dump({"schema": {"columns": {"a": 2, "b": 2}, "outcome": {"Y": 2}}})
    )
    factual_path, cf_path = tmp_path / "factual.csv", tmp_path / "cf.csv"
    frame = pd.DataFrame(
        {
            "id": range(32),
            "a": [i % 2 for i in range(32)],
            "b": [i % 3 == 0 for i in range(32)],
        }
    )
    frame["b"] = frame["b"].astype(int)
    frame.to_csv(factual_path, index=False)
    cf = frame.copy()
    cf.loc[28:, "a"] = 1 - cf.loc[28:, "a"]
    cf.to_csv(cf_path, index=False)
    z = torch.arange(32 * 128).reshape(32, 128).float() / (32 * 128)
    artifact = SimpleNamespace(
        ids=list(range(32)),
        train_ids=list(range(24)),
        test_ids=list(range(24, 32)),
        z=z,
        z_prime=z[24:].clone(),
        is_identity=torch.tensor([True] * 4 + [False] * 4),
        artifact_sha256="fixture",
        encoder_info={"variant": "fixture"},
    )
    monkeypatch.setattr(pipeline, "load_latent_artifact", lambda config: artifact)
    base["sim_config"] = str(sim)
    base["paths"].update(sim_factual=str(factual_path), sim_counterfactual=str(cf_path))
    configs, results = [], []
    old_threads, old_determinism = (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
    )
    try:
        for variant in ("independent", "autoregressive"):
            config = copy.deepcopy(base)
            config["semantic_decoder"]["variant"] = variant
            config["paths"]["decoder_model"] = str(
                tmp_path / "models" / variant / "semantic_decoder.pt"
            )
            config["paths"]["decoder_report"] = str(
                tmp_path / "reports" / variant / "semantic_decoder.json"
            )
            model_path = Path(config["paths"]["decoder_model"])
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"old-active-model")
            configs.append(config)
            results.append(annotate_result(run_case(config, "tiny")))
        root = tmp_path / "benchmark"
        write_json(root / "protocol.json", {"fixture": True})
        with pytest.raises(FileNotFoundError):
            evaluate_case(results[0], root)
        selection = freeze_selection(results, [case_name(c) for c in configs])
        write_json(root / "selection.json", selection)
        evaluations = [evaluate_case(result, root) for result in results]
        assert evaluate_case(results[0], root) == evaluations[0]
        for item in evaluations:
            assert item["factual_test"]["n_units"] == 8
            assert item["counterfactual_nonidentity_test"]["n_units"] == 4
            assert item["factual_test"]["n_seeds"] == 2
        first_report = json.loads(
            Path(results[0]["final_runs"][0]["report"]).read_text()
        )
        plan = {
            "configs": configs,
            "inventory": [{"encoder_tag": "langvae", "dimension": 128}],
            "split_ids": first_report["split_ids"],
            "langvae_ft_validation_caveat": None,
        }
        render(root, plan, results, evaluations, selection, "evaluating official test")
        assert "final/seed-42/semantic_decoder.pt)" in (root / "summary.md").read_text()
        published = publish_results(results, root)
        render(root, plan, results, evaluations, selection, "complete")
        markdown = (root / "summary.md").read_text()
        assert selection["recommended_case"] in markdown
        assert "Non-identity CF NLL" in markdown and "Factual-test marginal" in markdown
        assert "Data behind the statistics" in markdown
        assert "[evaluation.json](evaluation.json)" in markdown
        assert "[model](../models/independent/semantic_decoder.pt)" in markdown
        assert "final/seed-42/semantic_decoder.pt)" not in markdown
        for result in results:
            model_link = os.path.relpath(result["active_paths"]["decoder_model"], root)
            assert f"[model]({model_link})" in markdown
        assert len(published) == 2
        for config in configs:
            path = Path(config["paths"]["decoder_model"])
            assert (
                path.parent / "archive/tiny/semantic_decoder.pt"
            ).read_bytes() == b"old-active-model"
            active = json.loads(Path(config["paths"]["decoder_report"]).read_text())
            verify_run(active["config"])
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_determinism)
