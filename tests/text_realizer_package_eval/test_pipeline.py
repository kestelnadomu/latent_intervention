"""Offline tests for joins, phase separation, portable resume and complete scoring."""

from __future__ import annotations

import json
import shutil

import pytest
import torch

from exp.text_realizer_package_eval import generate as generation
from exp.text_realizer_package_eval.common import file_record, read_json
from exp.text_realizer_package_eval.prepare import (
    aligned_vectors,
    check_training_record,
    prepare,
    select_units,
)
from exp.text_realizer_package_eval.score import score
from src.artifact_io import save_torch, write_json


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    ids = [7, 2, 9, 4]
    models, files = {}, {}
    for dim in (256, 768):
        z = torch.zeros(len(ids), dim)
        z[:, 0] = torch.tensor(ids)
        encoder = {"embeddinggemma_latent_dim": dim}
        path = root / f"vectors-{dim}.pt"
        save_torch(
            path, {"ids": ids, "oracle": z, "predicted": z + 100, "encoder": encoder}
        )
        files[f"vectors-{dim}"] = file_record(root, path)
        for size in ("270m", "1b"):
            key = f"{dim}-{size}"
            path = root / f"{key}.pt"
            path.write_bytes(key.encode())
            models[key] = {
                **file_record(root, path),
                "latent_dim": dim,
                "encoder": encoder,
                "generation": {"max_new_tokens": 1024, "batch_size": 1},
            }
    path = root / "references.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "id": key,
                    "do_value": 1,
                    "original": "Original CV",
                    "reference": f"CV {key}: France, master, top100, 3 projects.",
                }
            )
            + "\n"
            for key in ids
        )
    )
    files["references"] = file_record(root, path)
    write_json(
        root / "manifest.json",
        {
            "format_version": 1,
            "complete": True,
            "ids": ids,
            "models": models,
            "files": files,
            "scope": "fixture",
            "provenance": {"config": {"selection": "fixture", "pipelines": {}}},
        },
    )
    return root


@pytest.fixture
def fake_decoder(monkeypatch):
    calls = []
    monkeypatch.setattr(generation, "load_text_realizer", lambda *a, **kw: object())

    def generate(model, z, **kwargs):
        calls.append(z.clone())
        key = int(z[0, 0])
        return [
            {
                "text": f"CV {key}: France, master, top100, 3 projects.",
                "tokens": 12,
                "token_ids": [3] * 11 + [2],
                "termination": "eos",
            }
        ]

    monkeypatch.setattr(generation, "generate", generate)
    return calls


def test_test_export_requires_explicit_gate(tmp_path):
    with pytest.raises(PermissionError, match="allow-oracle-evaluation"):
        prepare({}, tmp_path / "never-opened")
    assert not (tmp_path / "never-opened").exists()


def test_join_preserves_notebook_order_and_rejects_missing_reference():
    partition = {
        7: {"partition": "eval"},
        1: {"partition": "pool"},
        2: {"partition": "eval"},
    }
    pairs = {7: {"split": "test"}, 1: {"split": "train"}, 2: {"split": "test"}}
    references = {2: {"text": "second"}, 7: {"text": "first"}}
    assert select_units(partition, pairs, references, 2) == [7, 2]
    with pytest.raises(ValueError, match="IDs"):
        select_units(partition, pairs, {7: references[7]}, 2)


def test_oracle_alignment_and_exact_canonical_width_conversion():
    z = torch.nn.functional.normalize(torch.randn(3, 768), dim=1)
    payload = {"test_ids": [3, 1, 2], "z_prime": z, "ids": [3, 1, 2], "z": z}
    actual = aligned_vectors(payload, [1, 3], 768, oracle=True)
    torch.testing.assert_close(actual, z[[1, 0]], atol=0, rtol=0)
    torch.testing.assert_close(
        aligned_vectors(payload, [1, 3], 256, oracle=True),
        torch.nn.functional.normalize(z[[1, 0], :256], dim=1),
        atol=0,
        rtol=0,
    )
    payload["test_ids"] = [3, 3, 2]
    with pytest.raises(ValueError, match="IDs"):
        aligned_vectors(payload, [1, 3], 768, oracle=True)


def test_wrong_reader_experiment_is_rejected():
    record = {
        "history": [{"epoch": 1}],
        "best_epoch": 1,
        "epochs_run": 1,
        "max_epochs": 300,
    }
    check_training_record(record, record)
    with pytest.raises(ValueError, match="package report"):
        check_training_record({**record, "best_epoch": 2}, record)


def test_generation_never_needs_reference_text_and_preserves_edited_vectors(
    bundle, tmp_path, fake_decoder
):
    (bundle / "references.jsonl").unlink()
    generation.run_generation(
        bundle, tmp_path / "predictions", "256-270m", condition="predicted", limit=1
    )
    assert len(fake_decoder) == 1
    assert fake_decoder[0][0, 0] == 107
    assert fake_decoder[0].norm() > 1


def test_portable_resume_repairs_only_uncommitted_tail(bundle, tmp_path, fake_decoder):
    output = tmp_path / "predictions"
    generation.run_generation(bundle, output, "256-270m", limit=1)
    assert len(fake_decoder) == 2
    path = output / "256-270m/oracle/part-000-of-001.jsonl"
    with path.open("ab") as stream:
        stream.write(b'{"id": 2, "tex')
    moved = tmp_path / "another-cluster"
    shutil.copytree(bundle, moved)
    generation.run_generation(moved, output, "256-270m")
    assert len(fake_decoder) == 8
    assert read_json(path.with_suffix(".done.json"))["records"] == 4
    generation.run_generation(moved, output, "256-270m")
    assert len(fake_decoder) == 8


def test_changed_sharding_or_bundle_cannot_resume(bundle, tmp_path, fake_decoder):
    output = tmp_path / "predictions"
    generation.run_generation(bundle, output, "768-1b", limit=1)
    with pytest.raises(ValueError, match="sharding changed"):
        generation.run_generation(bundle, output, "768-1b", num_shards=2)
    manifest = read_json(bundle / "manifest.json")
    manifest["scope"] = "modified experiment"
    write_json(bundle / "manifest.json", manifest)
    with pytest.raises(ValueError, match="immutable artifact changed"):
        generation.run_generation(bundle, output, "768-1b")


def test_complete_sharded_matrix_scores_all_units(bundle, tmp_path, fake_decoder):
    predictions, reports = tmp_path / "predictions", tmp_path / "scores"
    for model in read_json(bundle / "manifest.json")["models"]:
        for shard in (0, 1):
            generation.run_generation(
                bundle, predictions, model, shard_index=shard, num_shards=2
            )
    result = score(bundle, predictions, reports)
    assert len(fake_decoder) == 32
    assert len(result["metrics"]) == 8
    assert all(row["units"] == 4 for row in result["metrics"])
    assert all(
        row["exact_raw"] == 1
        for row in result["metrics"]
        if row["condition"] == "oracle"
    )
    comparisons = [
        json.loads(line)
        for line in (reports / "comparisons.jsonl").read_text().splitlines()
    ]
    assert [row["id"] for row in comparisons] == [7, 2, 9, 4]
    assert all(len(row["generated"]) == 8 for row in comparisons)
    assert (reports / "examples.md").exists()


def test_score_refuses_missing_or_modified_shards(bundle, tmp_path, fake_decoder):
    predictions = tmp_path / "predictions"
    for model in read_json(bundle / "manifest.json")["models"]:
        generation.run_generation(bundle, predictions, model)
    path = predictions / "256-270m/oracle/part-000-of-001.jsonl"
    path.write_text(path.read_text().replace("France", "Germany"))
    with pytest.raises(ValueError, match="modified"):
        score(bundle, predictions, tmp_path / "scores")
    assert not (tmp_path / "scores").exists()


def test_missing_inputs_and_corrupt_weights_fail_before_model_load(
    bundle, tmp_path, fake_decoder
):
    (bundle / "256-1b.pt").write_bytes(b"different weights")
    with pytest.raises(ValueError, match="checksum mismatch"):
        generation.run_generation(bundle, tmp_path / "output", "256-1b")
    assert not fake_decoder


def test_importing_cli_does_not_require_pandas3_or_download_models():
    from exp.text_realizer_package_eval.__main__ import main

    assert callable(main)
