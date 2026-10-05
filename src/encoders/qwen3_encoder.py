"""Frozen Qwen3-Embedding-0.6B; intentionally independent of LangVAE imports.

Run inference with .venv-qwen3/bin/python. Use the official last-token pooling
recipe, not Nomic's mean-pooling/extra-layer-normalization recipe.
"""

from __future__ import annotations

from importlib.metadata import version
from time import perf_counter

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from packaging.version import Version
from transformers import AutoModel, AutoTokenizer

from src.config import QWEN3_MODEL, QWEN3_REVISION, qwen3_dimension, qwen3_protocol
from src.encoding_progress import log_encoding_progress

MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
)


def cached_snapshot(config: dict) -> str:
    qwen3_protocol(config)
    return snapshot_download(
        repo_id=config.get("qwen3_model_name", QWEN3_MODEL),
        revision=config.get("qwen3_model_revision", QWEN3_REVISION),
        cache_dir=config.get("qwen3_cache_dir", ".cache/huggingface/hub"),
        allow_patterns=list(MODEL_FILES),
        local_files_only=True,
    )


def format_text(text: str, instruction: str) -> str:
    return f"Instruct: {instruction}\nQuery:{text}" if instruction else text


def last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Locate the last nonpadding token; works with left/right padding."""
    if (
        mask.ndim != 2
        or hidden.shape[:2] != mask.shape
        or not mask.bool().any(dim=1).all()
    ):
        raise ValueError("invalid/empty attention mask for last-token pooling")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    last = positions.masked_fill(~mask.bool(), -1).max(dim=1).values
    return hidden[torch.arange(len(hidden), device=hidden.device), last]


class Qwen3TextEncoder:
    def __init__(self, config: dict) -> None:
        self.protocol = qwen3_protocol(config)
        if Version(version("transformers")) < Version("4.51.0"):
            raise RuntimeError(
                "Qwen3 requires transformers>=4.51.0; use .venv-qwen3/bin/python"
            )
        self.latent_dim = qwen3_dimension(config.get("qwen3_latent_dim", 128))
        self.max_len = int(config.get("qwen3_max_len", config.get("max_len", 1024)))
        if not 1 <= self.max_len <= 32768:
            raise ValueError("Qwen3 max_len must be between 1 and 32768")
        self.device = torch.device(config.get("device", "cpu"))
        self.instruction = self.protocol["instruction"]
        self.progress = bool(config.get("progress", False))
        source = cached_snapshot(config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            source, padding_side="left", trust_remote_code=False, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            source,
            dtype=torch.float32,
            attn_implementation="sdpa",
            trust_remote_code=False,
            local_files_only=True,
        )
        if (
            self.model.config.hidden_size != 1024
            or self.model.config.model_type != "qwen3"
        ):
            raise ValueError("expected the Qwen3-Embedding-0.6B 1024-D backbone")
        self.model.eval().requires_grad_(False).to(self.device)

    @torch.inference_mode()
    def encode(
        self, texts: list[str], deterministic: bool = True, batch_size: int = 8
    ) -> torch.Tensor:
        if deterministic is not True:
            raise ValueError("Qwen3 encoding must be deterministic")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not texts:
            return torch.empty((0, self.latent_dim), dtype=torch.float32)
        chunks = []
        started = perf_counter()
        for start in range(0, len(texts), batch_size):
            batch = [
                format_text(text, self.instruction)
                for text in texts[start : start + batch_size]
            ]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=False,
                add_special_tokens=True,
                return_tensors="pt",
            )
            lengths = inputs["attention_mask"].sum(dim=1)
            if lengths.max().item() > self.max_len:
                raise ValueError(
                    f"Qwen3 input exceeds max_len={self.max_len}; refusing silent truncation"
                )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            hidden = self.model(**inputs, use_cache=False).last_hidden_state
            pooled = last_token_pool(hidden, inputs["attention_mask"]).float()
            embedding = F.normalize(pooled[:, : self.latent_dim], p=2, dim=1)
            if not torch.isfinite(embedding).all() or not torch.allclose(
                embedding.norm(dim=1),
                torch.ones(len(embedding), device=self.device),
                atol=1e-6,
                rtol=1e-6,
            ):
                raise ValueError("Qwen3 produced non-finite or non-unit embeddings")
            chunks.append(embedding.cpu())
            log_encoding_progress(
                "Qwen3", start, batch_size, len(texts), started, enabled=self.progress
            )
        return torch.cat(chunks)


def make_qwen3_encoder(config: dict) -> Qwen3TextEncoder:
    return Qwen3TextEncoder(config)
