"""Small factual-only package fixtures; no model downloads or GPU required."""

import csv
import json

import pytest
import torch


@pytest.fixture
def package_config(tmp_path):
    from exp.text_realizer_package.settings import read_config
    from src.artifact_io import sha256_file
    from src.encoder.protocols import embeddinggemma_protocol

    config = read_config()
    ids = list(range(8))

    def table(name, rows):
        path = tmp_path / name
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return str(path)

    paths = {
        "factual": table(
            "factual.csv",
            [
                {
                    "id": i,
                    "text": f"4 {i + 3} 6",
                    "template_id": i % 2,
                    "persona_id": i % 3,
                }
                for i in reversed(ids)
            ],
        ),
        "pairs": table(
            "pairs.csv", [{"id": i, "split": "train" if i < 6 else "test"} for i in ids]
        ),
        "partition": table(
            "partition.csv",
            [
                {"id": i, "partition": "pool" if i < 4 else "dev" if i < 6 else "eval"}
                for i in ids
            ],
        ),
        "latents": str(tmp_path / "latents.pt"),
        "info": str(tmp_path / "info.json"),
        "manifest": str(tmp_path / "source_manifest.json"),
    }
    z = torch.nn.functional.normalize(
        torch.randn(8, 768, generator=torch.Generator().manual_seed(37)), dim=1
    )
    # A protected access test below additionally traps __getitem__('z_prime').
    torch.save({"ids": ids, "z": z, "z_prime": "FORBIDDEN"}, paths["latents"])
    hashes = {
        name: sha256_file(paths[name])
        for name in ("factual", "pairs", "partition", "latents")
    }
    info = {
        "artifact_sha256": hashes["latents"],
        "input_sha256": {
            "factual_text": hashes["factual"],
            "pair_index": hashes["pairs"],
        },
        "encoder": config["encoder"]["embeddinggemma_model_name"],
        "model_revision": config["encoder"]["embeddinggemma_model_revision"],
        "embeddinggemma_protocol": embeddinggemma_protocol(
            {**config["encoder"], "embeddinggemma_latent_dim": 768}
        ),
        "latent_dimension": 768,
        "max_length": 2048,
        "deterministic": True,
    }
    with open(paths["info"], "w") as stream:
        json.dump(info, stream)
    with open(paths["manifest"], "w") as stream:
        json.dump(
            {
                "complete": True,
                "protocol": {
                    "input_sha256": {
                        "partition": hashes["partition"],
                        "texts": hashes["factual"],
                        "pair_index": hashes["pairs"],
                    }
                },
            },
            stream,
        )
    hashes.update({name: sha256_file(paths[name]) for name in ("info", "manifest")})
    config["inputs"] = {**paths, "sha256": hashes}
    config["paths"] = {
        name: str(tmp_path / name)
        for name in ("prepared", "models", "reports", "cache")
    }
    config["split"] = {"fit_count": 4, "tune_count": 2, "evaluation_count": 2}
    return config


class TinyTokenizer:
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def __call__(self, texts, **kwargs):
        assert kwargs["truncation"] is False and kwargs["add_special_tokens"] is False
        return {"input_ids": [[int(token) for token in text.split()] for text in texts]}

    def decode(self, tokens, **kwargs):
        return " ".join(str(token) for token in tokens if token not in (0, 1, 2))


@pytest.fixture
def tiny_model():
    def build(*args, **kwargs):
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig
        from peft import LoraConfig, get_peft_model
        from src.text_realizer.model import TextRealizer

        torch.set_num_threads(1)
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
        return TextRealizer(
            base,
            TinyTokenizer(),
            {
                "latent_dim": 256,
                "prefix_tokens": 2,
                "projector_hidden": 24,
                "decoder_hidden_size": 16,
                "use_lora": True,
                "name": "tiny-offline",
            },
            {"max_new_tokens": 12, "batch_size": 1},
        )

    return build
