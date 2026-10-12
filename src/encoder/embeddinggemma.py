"""Frozen EmbeddingGemma-300m with the complete official SentenceTransformer stack.

Importable without sentence-transformers; only inference needs the isolated environment.
No model code is fetched/executed remotely. All inference uses the pinned local snapshot.
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import torch
from huggingface_hub import snapshot_download
from packaging.version import Version
from src.encoder.progress import log_encoding_progress

from src.config import (
    EMBEDDINGGEMMA_MODEL,
    EMBEDDINGGEMMA_PROMPT,
    EMBEDDINGGEMMA_REVISION,
    embeddinggemma_dimension,
    embeddinggemma_protocol,
)

MODEL_FILES = (
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "model.safetensors",
    "1_Pooling/config.json",
    "2_Dense/config.json",
    "2_Dense/model.safetensors",
    "3_Dense/config.json",
    "3_Dense/model.safetensors",
)


def cached_snapshot(config: dict) -> str:
    embeddinggemma_protocol(config)
    snapshot = snapshot_download(
        repo_id=config.get("embeddinggemma_model_name", EMBEDDINGGEMMA_MODEL),
        revision=config.get("embeddinggemma_model_revision", EMBEDDINGGEMMA_REVISION),
        cache_dir=config.get("embeddinggemma_cache_dir", ".cache/huggingface/hub"),
        allow_patterns=list(MODEL_FILES),
        local_files_only=True,
    )
    missing = [name for name in MODEL_FILES if not (Path(snapshot) / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"EmbeddingGemma cache incomplete ({missing}); run bash scripts/setup_embeddinggemma.sh"
        )
    return snapshot


def format_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("EmbeddingGemma expects non-empty text strings")
    return (EMBEDDINGGEMMA_PROMPT + text).strip()


def load_sentence_model(source: str, device: torch.device):
    try:
        if Version(version("transformers")) < Version("4.56.0") or Version(
            version("sentence-transformers")
        ) < Version("5.1.0"):
            raise RuntimeError("incompatible embedding libraries")
        from sentence_transformers import SentenceTransformer
    except (PackageNotFoundError, ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "EmbeddingGemma requires .venv-embeddinggemma/bin/python; run bash scripts/setup_embeddinggemma.sh"
        ) from exc
    return SentenceTransformer(
        source,
        device=str(device),
        trust_remote_code=False,
        local_files_only=True,
        model_kwargs={"dtype": torch.float32, "attn_implementation": "sdpa"},
        tokenizer_kwargs={"padding_side": "right"},
        config_kwargs={"use_cache": False},
    )


def validate_model(model) -> None:
    """Fail rather than accidentally use just the backbone or a causal attention mask."""
    if [type(layer).__name__ for layer in model] != [
        "Transformer",
        "Pooling",
        "Dense",
        "Dense",
        "Normalize",
    ]:
        raise ValueError(
            "EmbeddingGemma requires the complete released pooling/projection/normalization stack"
        )
    cfg = model[0].auto_model.config
    if (
        cfg.model_type != "gemma3_text"
        or cfg.hidden_size != 768
        or not cfg.use_bidirectional_attention
        or cfg.use_cache
    ):
        raise ValueError(
            "EmbeddingGemma requires the 768-D bidirectional Gemma3 backbone without KV caching"
        )
    pool = model[1]
    if pool.get_pooling_mode_str() != "mean" or not pool.include_prompt:
        raise ValueError("EmbeddingGemma requires mean pooling including the prompt")
    for layer, dims in ((model[2], (768, 3072)), (model[3], (3072, 768))):
        if (
            (layer.in_features, layer.out_features) != dims
            or layer.linear.bias is not None
            or not isinstance(layer.activation_function, torch.nn.Identity)
        ):
            raise ValueError(
                "EmbeddingGemma dense projection heads differ from the released architecture"
            )
    if model.get_sentence_embedding_dimension() != 768:
        raise ValueError(
            "EmbeddingGemma native sentence embeddings must have 768 dimensions"
        )


class EmbeddingGemmaTextEncoder:
    def __init__(self, config: dict) -> None:
        self.protocol = embeddinggemma_protocol(config)
        self.latent_dim = embeddinggemma_dimension(
            config.get("embeddinggemma_latent_dim", 128)
        )
        self.max_len = int(
            config.get("embeddinggemma_max_len", config.get("max_len", 2048))
        )
        if not 1 <= self.max_len <= 2048:
            raise ValueError("EmbeddingGemma max_len must be between 1 and 2048")
        self.device = torch.device(config.get("device", "cpu"))
        self.progress = bool(config.get("progress", False))
        self.model = load_sentence_model(cached_snapshot(config), self.device)
        validate_model(self.model)
        self.model.max_seq_length = self.max_len
        self.model.eval().requires_grad_(False).float()
        self.tokenizer = self.model.tokenizer

    @torch.inference_mode()
    def encode(
        self, texts: list[str], deterministic: bool = True, batch_size: int = 8
    ) -> torch.Tensor:
        if deterministic is not True:
            raise ValueError("EmbeddingGemma encoding must be deterministic")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        dimension = embeddinggemma_dimension(self.latent_dim)
        if not texts:
            return torch.empty((0, dimension), dtype=torch.float32)
        chunks = []
        started = perf_counter()
        for start in range(0, len(texts), batch_size):
            batch = [format_text(text) for text in texts[start : start + batch_size]]
            tokens = self.tokenizer(
                batch, padding=False, truncation=False, add_special_tokens=True
            )
            if max(map(len, tokens["input_ids"])) > self.max_len:
                raise ValueError(
                    f"EmbeddingGemma input exceeds max_len={self.max_len}; refusing silent truncation"
                )
            # Already formatted above. An explicit empty prompt prevents a second prefix.
            embedding = (
                self.model.encode(
                    batch,
                    batch_size=batch_size,
                    prompt="",
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    truncate_dim=dimension,
                )
                .detach()
                .cpu()
                .float()
            )
            if (
                embedding.shape != (len(batch), dimension)
                or not torch.isfinite(embedding).all()
                or not torch.allclose(
                    embedding.norm(dim=1), torch.ones(len(batch)), atol=1e-6, rtol=1e-6
                )
            ):
                raise ValueError("EmbeddingGemma produced invalid/non-unit embeddings")
            chunks.append(embedding)
            log_encoding_progress(
                "EmbeddingGemma",
                start,
                batch_size,
                len(texts),
                started,
                enabled=self.progress,
            )
        return torch.cat(chunks)


def make_embeddinggemma_encoder(config: dict) -> EmbeddingGemmaTextEncoder:
    return EmbeddingGemmaTextEncoder(config)
