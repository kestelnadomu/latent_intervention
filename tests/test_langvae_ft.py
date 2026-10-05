"""Offline tests for the sole new baseline; no V1/V2/V3 imports."""

import ast
import copy
import csv
import json
import random
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from src.config import load_config, resolve_paths
from src.encoders.langvae_ft import ARCHITECTURE, TAG
from src.encoders.langvae_ft.artifacts import (
    ROOT,
    inference_config,
    read_checkpoint,
    restore_resume,
    save_resume,
    source_hashes,
    trainable_schema,
    write_inference_config,
)
from src.encoders.langvae_ft.data import (
    audit_indices,
    file_hash,
    passage_records,
    split_cvs,
    wrong_cv_indices,
)
from src.encoders.langvae_ft.evaluation import evaluate
from src.encoders.langvae_ft.model import NativeAdapter, loss_terms, parameter_hash, train_update
from src.encoders.langvae_ft.training import read_settings


class WordTokenizer:
    eos_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(1, len(text.split()) + 1))}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def data_paths(tmp_path):
    pairs, texts = tmp_path / "pairs.csv", tmp_path / "texts.csv"
    write_csv(
        pairs,
        [
            {"id": i, "split": "train" if i < 10 else "test", "is_identity": i == 11}
            for i in range(12)
        ],
    )
    write_csv(
        texts,
        [
            {"id": i, "text": f"CV {i} " + " ".join(f"word{j}" for j in range(75))}
            for i in range(12)
        ],
    )
    return {
        "pair_index": str(pairs),
        "texts": str(texts),
        "texts_counterfactual": str(tmp_path / "DO_NOT_OPEN.csv"),
    }


def test_cv_split_precedes_chunking_and_never_uses_cf_or_test(data_paths):
    examples, manifest = split_cvs(data_paths, validation_fraction=0.2)
    assert (examples, manifest) == split_cvs(data_paths, validation_fraction=0.2)
    records = {
        key: passage_records(rows, WordTokenizer()) for key, rows in examples.items()
    }
    ids = {key: {r["id"] for r in rows} for key, rows in records.items()}
    assert ids["train"].isdisjoint(ids["validation"])
    assert ids["train"] | ids["validation"] == set(range(10))
    assert manifest["excluded_official_test_ids"] == [10, 11]
    assert len(ids["validation"]) == 2


def test_duplicate_document_leakage_is_rejected(data_paths):
    with Path(data_paths["texts"]).open() as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["text"] = rows[10]["text"]
    write_csv(Path(data_paths["texts"]), rows)
    with pytest.raises(ValueError, match="duplicate full CV"):
        split_cvs(data_paths, validation_fraction=0.2)


def test_passages_cover_start_middle_end_and_self_reconstruct():
    text = "  " + "\n".join(f"élève{n} Ω{n}" for n in range(55)) + "  "
    rows = passage_records([{"id": 42, "text": text}], WordTokenizer())
    assert "".join(row["text"] for row in rows) == text.strip()
    assert rows[0]["char_start"] == 0 and rows[-1]["char_end"] == len(text.strip())
    assert len(rows) == 4
    for previous, current in pairwise(rows):
        assert previous["char_end"] == current["char_start"]
    for row in rows:
        assert row["input_ids"] == row["target_ids"][:-1]
        assert row["target_ids"][-1] == 0
        assert len(row["target_ids"]) <= 32
        assert row["text"] == text.strip()[row["char_start"] : row["char_end"]]


def test_single_long_word_is_not_silently_truncated():
    class CharacterTokenizer(WordTokenizer):
        def __call__(self, text, **kwargs):
            return {"input_ids": [1] * len(text)}

    with pytest.raises(ValueError, match="single word"):
        passage_records([{"id": 1, "text": "x" * 40}], CharacterTokenizer())
    with pytest.raises(ValueError, match="32 output"):
        passage_records([], WordTokenizer(), max_length=512)


def test_wrong_codes_come_from_different_cvs_and_audit_is_cv_balanced():
    rows = [
        {"id": i, "passage": j} for i, n in [(0, 8), (1, 2), (2, 3)] for j in range(n)
    ]
    negatives = wrong_cv_indices(rows)
    assert all(row["id"] != rows[other]["id"] for row, other in zip(rows, negatives))
    selected = audit_indices(rows, 64, 42)
    assert len(selected) == 3
    assert len({rows[i]["id"] for i in selected}) == 3
    assert selected == audit_indices(rows, 64, 42)


@pytest.fixture
def adapter():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    lm = GPT2LMHeadModel(
        GPT2Config(
            n_layer=2,
            n_head=2,
            n_embd=16,
            vocab_size=25,
            n_positions=64,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
        )
    )
    native = SimpleNamespace(
        encoder=SimpleNamespace(linear=nn.Linear(8, 8, bias=False)),
        decoder=SimpleNamespace(
            context_hidden=nn.ModuleList([nn.Linear(4, 2 * 16 * 5) for _ in range(2)]),
            decoder=lm,
            dropout=nn.Dropout(0.1),
            max_len=4,
            tokenizer=SimpleNamespace(bos_token_id=0, eos_token_id=0, pad_token_id=0),
            pkv_dims=(2, 1, 8),
            conditional=False,
        ),
        model_config=SimpleNamespace(latent_dim=4),
    )
    return NativeAdapter(native)


def test_wrapper_has_only_original_trainable_parameters(adapter):
    expected = {
        id(p)
        for layer in (adapter.projection, adapter.adapters)
        for p in layer.parameters()
    }
    assert {id(p) for p in adapter.parameters() if p.requires_grad} == expected
    assert all(
        name.startswith(("projection.", "adapters."))
        for name in trainable_schema(adapter)
    )
    adapter.train()
    assert adapter.dropout.training and not adapter.language_model.training


def test_causality_eos_and_generation_teacher_parity(adapter):
    adapter.eval()
    z = torch.randn(2, 4)
    labels = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]])
    changed = labels.clone()
    changed[:, 1] = 12
    before, after = (
        adapter.teacher_logits(z, labels),
        adapter.teacher_logits(z, changed),
    )
    torch.testing.assert_close(before[:, :2], after[:, :2], atol=0, rtol=0)
    assert not torch.allclose(before[:, 2], after[:, 2])
    generated = adapter.generate(z)
    predictions = adapter.teacher_logits(z, generated).argmax(-1)
    for i, row in enumerate(generated):
        eos = row.eq(0).nonzero().flatten()
        length = int(eos[0]) + 1 if len(eos) else len(row)
        assert torch.equal(predictions[i, :length], row[:length])
    with pytest.raises(ValueError, match="unchanged decoder length"):
        adapter.teacher_logits(z, torch.ones(2, 5, dtype=torch.long))


def test_padding_not_eos_is_masked_and_loss_is_standard_vae():
    logits = torch.tensor(
        [[[3.0, 0.0], [0.0, 3.0], [100.0, -100.0]]], requires_grad=True
    )
    labels = torch.tensor([[0, 1, -100]])
    mu, logvar = torch.ones(1, 4), torch.zeros(1, 4)
    nll, kl, tokens = loss_terms(logits, labels, mu, logvar)
    assert tokens.item() == 2 and kl.item() == 2
    assert nll.item() == pytest.approx(
        2 * torch.nn.functional.softplus(torch.tensor(-3.0)).item()
    )
    (nll + kl).mean().backward()
    assert logits.grad[0, 0].abs().sum() > 0
    assert logits.grad[0, 2].abs().sum() == 0


def test_update_and_resume_leave_base_frozen_and_reproduce_next_step(adapter, tmp_path):
    optimizer = torch.optim.AdamW(
        [p for p in adapter.parameters() if p.requires_grad], lr=1e-3
    )
    batches = [(torch.randn(2, 8), torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]]))]
    base_hash = parameter_hash(adapter.language_model)
    projection, decoder = (
        parameter_hash(adapter.projection),
        parameter_hash(adapter.adapters),
    )
    train_update(adapter, optimizer, batches, 0.5)
    assert projection != parameter_hash(adapter.projection)
    assert decoder != parameter_hash(adapter.adapters)
    path = tmp_path / "resume.pt"
    save_resume(path, adapter, optimizer, epoch=1)
    expected_random = random.random()
    train_update(adapter, optimizer, batches, 1.0)
    expected = parameter_hash(adapter)
    assert restore_resume(path, adapter, optimizer) == {"epoch": 1}
    assert random.random() == expected_random
    train_update(adapter, optimizer, batches, 1.0)
    assert parameter_hash(adapter) == expected
    assert parameter_hash(adapter.language_model) == base_hash
    assert all(p.grad is None for p in adapter.language_model.parameters())


def test_validation_is_fixed_mc_objective_and_does_not_advance_training_rng(adapter):
    features = torch.randn(4, 8)
    records = [{"target_ids": [i + 1, i + 2, 0]} for i in range(4)]
    state = torch.get_rng_state().clone()
    kwargs = {"batch_size": 2, "device": torch.device("cpu"), "seed": 42, "samples": 2}
    first = evaluate(adapter, features, records, **kwargs)
    assert torch.equal(state, torch.get_rng_state())
    assert first == evaluate(adapter, features, records, **kwargs)
    assert first["negative_elbo"] == pytest.approx(
        first["sampled_reconstruction_nll"] + first["kl"]
    )


def fake_checkpoint(tmp_path, stage="train"):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    raw = yaml.safe_load((ROOT / "src/config.yaml").read_text())
    metadata = {
        "architecture": ARCHITECTURE,
        "tag": TAG,
        "stage": stage,
        "encoder": {**raw["encoder"], "local_checkpoint": None, "tag": None},
    }
    (checkpoint / "langvae_ft.json").write_text(json.dumps(metadata))
    for name in (
        "encoder.pt",
        "decoder.pt",
        "encoder_cfg.json",
        "decoder_cfg.json",
        "model_config.json",
        "environment.json",
        "reload_probe.pt",
    ):
        (checkpoint / name).write_bytes(b"fixture")
    hashes = {p.name: file_hash(p) for p in checkpoint.iterdir()}
    (checkpoint / "sha256.json").write_text(json.dumps(hashes))
    return checkpoint, raw


def test_opt_in_config_is_native_portable_and_has_separate_output_paths(tmp_path):
    checkpoint, raw = fake_checkpoint(tmp_path)
    original = copy.deepcopy(raw)
    config = inference_config(raw, checkpoint)
    assert config["encoder"]["variant"] == "langvae"
    assert config["encoder"]["tag"] == "langvae_ft"
    assert config["encoder"]["max_len"] == 512
    assert "finetune_vae" not in config
    resolved = resolve_paths(copy.deepcopy(config))
    assert resolved["paths"]["latents"] == "data/latents/talent/langvae_ft/z_pairs.pt"
    for key in ("decoder_model", "decoder_report", "manipulator_model", "eval_report"):
        assert "/langvae_ft/" in resolved["paths"][key]
    assert raw == original
    output = tmp_path / "encoding.yaml"
    write_inference_config(raw, checkpoint, output)
    assert load_config(path=output)["paths"]["latents"] == resolved["paths"]["latents"]
    with pytest.raises(FileExistsError):
        write_inference_config(raw, checkpoint, output)
    with pytest.raises(ValueError, match="outside the immutable"):
        write_inference_config(raw, checkpoint, checkpoint / "encoding.yaml")


def test_diagnostic_checkpoint_cannot_be_promoted_and_corruption_is_detected(tmp_path):
    checkpoint, raw = fake_checkpoint(tmp_path, stage="smoke")
    with pytest.raises(ValueError, match="diagnostic"):
        inference_config(raw, checkpoint)
    (checkpoint / "encoder.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        read_checkpoint(checkpoint)


def test_recipe_ignores_legacy_finetuning_and_keeps_root_config_unchanged():
    path = ROOT / "src/config.yaml"
    before = file_hash(path)
    recipe, raw, encoder = read_settings(ROOT / "configs/langvae_ft.yaml", "cpu")
    assert recipe["tag"] == TAG and recipe["device"] == "cpu"
    assert "finetune_vae" not in raw
    assert encoder["local_checkpoint"] is None and encoder["max_len"] == 512
    assert file_hash(path) == before


def test_package_has_no_legacy_experiment_dependencies():
    forbidden = ("src.vae_", "src.finetune_vae")
    for path in (ROOT / "src/encoders/langvae_ft").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(forbidden)
            elif isinstance(node, ast.Import):
                assert not any(alias.name.startswith(forbidden) for alias in node.names)
    assert not any(
        Path(name).name.startswith(("vae_", "finetune_vae")) for name in source_hashes()
    )


def test_transfer_allowlist_is_complete_and_excludes_legacy_work():
    scope = json.loads((ROOT / "configs/langvae_ft_scope.json").read_text())
    keep = set(scope["keep_for_code_transfer"])
    assert len(keep) == len(scope["keep_for_code_transfer"])
    assert all((ROOT / name).is_file() for name in keep)
    assert all(
        str(path.relative_to(ROOT)) in keep
        for path in (ROOT / "src/encoders/langvae_ft").glob("*.py")
    )
    assert keep.isdisjoint(
        {"src/config.yaml", "src/finetune_vae.py", "src/pipeline.py"}
    )
    assert not any(
        name.startswith(("src/vae_", "src/finetune_vae", "tests/test_vae_"))
        or "langvae_cv_" in name
        for name in keep
    )
    assert not any(name.startswith(("models/", "data/")) for name in keep)
    assert scope["baseline"] == "langvae_ft"
    assert len(scope["checkpoint_manifest_sha256"]) == 64
