"""Tiny offline fixtures. Real model weights/network access are never test defaults."""

import copy
import csv
import json

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "text_realizer_gpu: opt-in pinned-model GPU integration checks"
    )


class TinyTokenizer:
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def __call__(self, texts, **kwargs):
        assert kwargs.get("truncation") is False
        assert kwargs.get("add_special_tokens") is False
        if isinstance(texts, str):
            texts = [texts]
        return {
            "input_ids": [
                [
                    int(word) if word.isdigit() else 3 + sum(map(ord, word)) % 25
                    for word in text.split()
                ]
                for text in texts
            ]
        }

    def decode(self, tokens, skip_special_tokens=True, **kwargs):
        return " ".join(
            str(token)
            for token in tokens
            if not skip_special_tokens or token not in {0, 1, 2}
        )


@pytest.fixture
def tiny_factory():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers", minversion="4.57.6")
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    from src.text_realizer.model import TextRealizer

    torch.set_num_threads(1)

    def make(*, lora=False):
        torch.manual_seed(17)
        cfg = Gemma3TextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=128,
            sliding_window=8,
            sliding_window_pattern=2,
            layer_types=["sliding_attention", "full_attention"],
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        cfg._attn_implementation = "sdpa"
        base = Gemma3ForCausalLM(cfg).requires_grad_(False)
        if lora:
            pytest.importorskip("peft", minversion="0.17.1")
            from peft import LoraConfig, get_peft_model

            base = get_peft_model(
                base,
                LoraConfig(
                    r=2,
                    lora_alpha=4,
                    lora_dropout=0.1,
                    target_modules=["q_proj", "v_proj"],
                    task_type="CAUSAL_LM",
                ),
            )
        spec = {
            "latent_dim": 768,
            "prefix_tokens": 2,
            "projector_hidden": 24,
            "decoder_hidden_size": 16,
            "use_lora": lora,
            "name": "tiny-offline",
        }
        return TextRealizer(
            base, TinyTokenizer(), spec, {"max_new_tokens": 20, "batch_size": 1}
        )

    return make


@pytest.fixture
def numerical_settings():
    return {"projector_lr": 0.01, "lora_lr": 0.01, "weight_decay": 0.01, "clip": 1.0}


@pytest.fixture
def factual_config(tmp_path):
    torch = pytest.importorskip("torch")
    from exp.text_realizer.settings import read_config
    from src.artifact_io import sha256_file
    from src.encoder.protocols import embeddinggemma_protocol

    config = copy.deepcopy(read_config())
    ids = list(range(12))

    def csv_file(name, rows):
        path = tmp_path / name
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    factual = csv_file(
        "factual.csv",
        [
            {
                "id": i,
                "text": f"CV number {i}",
                "template_id": i % 3,
                "persona_id": i % 2,
            }
            for i in reversed(ids)
        ],
    )
    pairs = csv_file(
        "pairs.csv",
        [
            {"id": i, "split": "train" if i < 10 else "test", "is_identity": i == 11}
            for i in ids
        ],
    )
    latents = tmp_path / "canonical.pt"
    z = torch.nn.functional.normalize(
        torch.randn(12, 768, generator=torch.Generator().manual_seed(42)), dim=1
    )
    torch.save({"ids": ids, "z": z, "z_prime": "POISON: DO NOT ACCESS"}, latents)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "split_ids": {
                    "fit": list(range(6)),
                    "validation": [6, 7, 8, 9],
                    "official_test": [10, 11],
                }
            }
        )
    )
    inputs = {
        "factual": str(factual),
        "pairs": str(pairs),
        "latents": str(latents),
        "split_plan": str(plan),
    }
    inputs["sha256"] = {key: sha256_file(value) for key, value in inputs.items()}
    info = tmp_path / "info.json"
    info.write_text(
        json.dumps(
            {
                "artifact_sha256": inputs["sha256"]["latents"],
                "input_sha256": {
                    "factual_text": inputs["sha256"]["factual"],
                    "pair_index": inputs["sha256"]["pairs"],
                },
                "encoder": "google/embeddinggemma-300m",
                "model_revision": config["encoder"]["embeddinggemma_model_revision"],
                "embeddinggemma_protocol": embeddinggemma_protocol(config["encoder"]),
                "latent_dimension": 768,
                "deterministic": True,
            }
        )
    )
    inputs["info"] = str(info)
    config["inputs"] = inputs
    config["paths"] = {
        key: str(tmp_path / key) for key in ("prepared", "models", "reports", "cache")
    }
    config["split"].update(fit_count=6, validation_count=4, test_count=2)
    return config
