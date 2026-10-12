"""Frozen text encoder f: X -> Z = R^d (docs/architecture/text_encoder.md).

Backends (``encoder.variant`` in src/config.yaml), one module each:
    langvae         TextEncoder                LangVAE posterior mean mu(x) (default)
    nomic           NomicTextEncoder           Nomic Embed v1.5, configurable Matryoshka width
    qwen3           Qwen3TextEncoder           Qwen3-Embedding-0.6B (isolated environment)
    embeddinggemma  EmbeddingGemmaTextEncoder  EmbeddingGemma-300m (isolated environment)

This package entry point exposes the pure-Python protocols (widths, pinned revisions,
metadata) and the lazy ``make_encoder`` / ``get_encoder_factory`` dispatch; importing it
never loads torch or LangVAE, so the isolated environments can use it.
"""

from src.encoder.protocols import (
    NOMIC_DIMENSIONS,
    QWEN3_DIMENSIONS,
    QWEN3_MODEL,
    QWEN3_REVISION,
    EMBEDDINGGEMMA_DIMENSIONS,
    EMBEDDINGGEMMA_MODEL,
    EMBEDDINGGEMMA_REVISION,
    EMBEDDINGGEMMA_PROMPT,
    BASE_MODEL_PATTERNS,
    embeddinggemma_dimension,
    embeddinggemma_protocol,
    qwen3_dimension,
    qwen3_protocol,
    nomic_dimension,
    encoder_dimension,
    configure_encoder,
    embedding_encoder_info,
    encoding_metadata,
    validate_encoding_metadata,
    get_encoder_factory,
    make_encoder,
    add_encoder_arguments,
)

__all__ = [
    "NOMIC_DIMENSIONS",
    "QWEN3_DIMENSIONS",
    "QWEN3_MODEL",
    "QWEN3_REVISION",
    "EMBEDDINGGEMMA_DIMENSIONS",
    "EMBEDDINGGEMMA_MODEL",
    "EMBEDDINGGEMMA_REVISION",
    "EMBEDDINGGEMMA_PROMPT",
    "BASE_MODEL_PATTERNS",
    "embeddinggemma_dimension",
    "embeddinggemma_protocol",
    "qwen3_dimension",
    "qwen3_protocol",
    "nomic_dimension",
    "encoder_dimension",
    "configure_encoder",
    "embedding_encoder_info",
    "encoding_metadata",
    "validate_encoding_metadata",
    "get_encoder_factory",
    "make_encoder",
    "add_encoder_arguments",
]
