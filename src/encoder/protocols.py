"""Encoder-specific protocols and dimension selection; no ML libraries at import time.

Keep config.py a loader. These settings describe the same latent spaces as existing
artifacts; changing a protocol requires new outputs, not relabeling old embeddings.
"""

from numbers import Integral
import re
from typing import Any

NOMIC_DIMENSIONS = (64, 128, 256, 512, 768)
QWEN3_DIMENSIONS = (32, 64, 128, 256, 512, 768, 1024)
QWEN3_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN3_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
EMBEDDINGGEMMA_DIMENSIONS = (128, 256, 512, 768)
EMBEDDINGGEMMA_MODEL = "google/embeddinggemma-300m"
EMBEDDINGGEMMA_REVISION = "57c266a740f537b4dc058e1b0cda161fd15afa75"
EMBEDDINGGEMMA_PROMPT = "task: classification | query: "

# Hugging Face snapshot allowlist for pinned base-model downloads (LangVAE, Nomic).
BASE_MODEL_PATTERNS = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
)


def embeddinggemma_dimension(value: Any = 128) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value not in EMBEDDINGGEMMA_DIMENSIONS
    ):
        raise ValueError(
            f"EmbeddingGemma dimension must be one of {EMBEDDINGGEMMA_DIMENSIONS}, got {value!r}"
        )
    return int(value)


def embeddinggemma_protocol(encoder: dict[str, Any]) -> dict[str, Any]:
    """Fixed classification baseline, including the released dense projection heads."""
    if (
        encoder.get("embeddinggemma_model_name", EMBEDDINGGEMMA_MODEL)
        != EMBEDDINGGEMMA_MODEL
    ):
        raise ValueError(
            "embeddinggemma_<dimension> is reserved for EmbeddingGemma-300m"
        )
    revision = encoder.get("embeddinggemma_model_revision", EMBEDDINGGEMMA_REVISION)
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("EmbeddingGemma requires a pinned 40-character model revision")
    if (
        encoder.get("embeddinggemma_prompt", EMBEDDINGGEMMA_PROMPT)
        != EMBEDDINGGEMMA_PROMPT
    ):
        raise ValueError(
            "EmbeddingGemma baseline requires the fixed classification prompt"
        )
    return {
        "prompt": EMBEDDINGGEMMA_PROMPT,
        "strip_formatted_text": True,
        "pooling": "attention_mask_mean_including_prompt",
        "projection": "dense_768_to_3072_to_768_identity_no_bias",
        "normalization": f"truncate_{embeddinggemma_dimension(encoder.get('embeddinggemma_latent_dim', 128))}+l2",
        "inference_dtype": "float32",
        "attention_implementation": "sdpa",
        "bidirectional_attention": True,
        "padding_side": "right",
        "add_special_tokens": True,
        "truncation": "error",
        "use_cache": False,
    }


def qwen3_dimension(value: Any = 128) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value not in QWEN3_DIMENSIONS
    ):
        raise ValueError(
            f"Qwen3 dimension must be one of {QWEN3_DIMENSIONS}, got {value!r}"
        )
    return int(value)


def qwen3_protocol(encoder: dict[str, Any]) -> dict[str, Any]:
    """Only the pinned 0.6B encoder is allowed to use the qwen3_<d> names."""
    if encoder.get("qwen3_model_name", QWEN3_MODEL) != QWEN3_MODEL:
        raise ValueError("qwen3_<dimension> is reserved for Qwen3-Embedding-0.6B")
    revision = encoder.get("qwen3_model_revision", QWEN3_REVISION)
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Qwen3 requires a pinned 40-character model revision")
    instruction = encoder.get("qwen3_instruction", "")
    if not isinstance(instruction, str):
        raise ValueError(
            "qwen3_instruction must be a string (empty means plain CV text)"
        )
    return {
        "instruction": instruction,
        "pooling": "last_nonpadding_token",
        "normalization": f"truncate_{qwen3_dimension(encoder.get('qwen3_latent_dim', 128))}+l2",
        "inference_dtype": "float32",
        "attention_implementation": "sdpa",
        "padding_side": "left",
        "add_special_tokens": True,
        "truncation": "error",
        "use_cache": False,
    }


def nomic_dimension(value: Any = 128) -> int:
    """Validate a Matryoshka dimension without silently rounding/coercing it."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value not in NOMIC_DIMENSIONS
    ):
        raise ValueError(
            f"Nomic dimension must be one of {NOMIC_DIMENSIONS}, got {value!r}"
        )
    return int(value)


def encoder_dimension(encoder: dict[str, Any]) -> int:
    """LangVAE stays 128-D; the embedding encoders have configurable widths."""
    if encoder.get("variant", "langvae") == "nomic":
        return nomic_dimension(encoder.get("nomic_latent_dim", 128))
    if encoder.get("variant") == "qwen3":
        qwen3_protocol(encoder)
        return qwen3_dimension(encoder.get("qwen3_latent_dim", 128))
    if encoder.get("variant") == "embeddinggemma":
        embeddinggemma_protocol(encoder)
        return embeddinggemma_dimension(encoder.get("embeddinggemma_latent_dim", 128))
    return 128


def configure_encoder(
    config: dict[str, Any], *, nomic_dim=None, qwen3_dim=None, embeddinggemma_dim=None
) -> None:
    """Apply dimension/runtime overrides before resolving artifact paths."""
    if nomic_dim is not None:
        if config["encoder"].get("variant", "langvae") != "nomic":
            raise ValueError("--nomic-dim requires the nomic encoder variant")
        config["encoder"]["nomic_latent_dim"] = nomic_dimension(nomic_dim)
    if qwen3_dim is not None:
        if config["encoder"].get("variant") != "qwen3":
            raise ValueError("--qwen3-dim requires the qwen3 encoder variant")
        config["encoder"]["qwen3_latent_dim"] = qwen3_dimension(qwen3_dim)
    if config["encoder"].get("variant") == "qwen3":
        config["encoder"]["max_len"] = config["encoder"].get("qwen3_max_len", 1024)
        config["encoder"]["batch_size"] = config["encoder"].get("qwen3_batch_size", 8)
    if embeddinggemma_dim is not None:
        if config["encoder"].get("variant") != "embeddinggemma":
            raise ValueError(
                "--embeddinggemma-dim requires the embeddinggemma encoder variant"
            )
        config["encoder"]["embeddinggemma_latent_dim"] = embeddinggemma_dimension(
            embeddinggemma_dim
        )
    if config["encoder"].get("variant") == "embeddinggemma":
        config["encoder"]["max_len"] = config["encoder"].get(
            "embeddinggemma_max_len", 2048
        )
        config["encoder"]["batch_size"] = config["encoder"].get(
            "embeddinggemma_batch_size", 8
        )
    encoder_dimension(config["encoder"])


def embedding_encoder_info(config: dict[str, Any]) -> dict[str, Any]:
    """Additional encoder identities, with exactly the established metadata keys."""
    variant = config["variant"]
    if variant == "qwen3":
        model = config.get("qwen3_model_name", QWEN3_MODEL)
        revision = config.get("qwen3_model_revision", QWEN3_REVISION)
        task = config.get("qwen3_instruction", "")
        protocol = qwen3_protocol(config)
    elif variant == "embeddinggemma":
        model = config.get("embeddinggemma_model_name", EMBEDDINGGEMMA_MODEL)
        revision = config.get("embeddinggemma_model_revision", EMBEDDINGGEMMA_REVISION)
        task = "classification"
        protocol = embeddinggemma_protocol(config)
    else:
        raise ValueError(f"unknown embedding encoder variant: {variant}")
    return {
        "encoder_variant": variant,
        "encoder": model,
        "model_revision": revision,
        "local_checkpoint": None,
        "local_checkpoint_sha256": None,
        "code_revision": None,
        "task": task,
        "max_length": int(config["max_len"]),
        "deterministic": True,
        "latent_dimension": encoder_dimension(config),
        "langvae_encoder_model": None,
        "langvae_encoder_revision": None,
        "langvae_decoder_model": None,
        "langvae_decoder_revision": None,
        f"{variant}_protocol": protocol,
    }


def encoding_metadata(info: dict[str, Any]) -> dict[str, Any]:
    """Task-prefix and postprocessing fields written to each latent sidecar."""
    variant, dimension = info["encoder_variant"], info["latent_dimension"]
    if variant == "nomic":
        return {
            "task_prefix": f"{info['task']}: ",
            "normalization": f"layer_norm+truncate_{dimension}+l2",
        }
    if variant == "qwen3":
        prefix = f"Instruct: {info['task']}\nQuery:" if info["task"] else None
        return {
            "task_prefix": prefix,
            "normalization": info["qwen3_protocol"]["normalization"],
        }
    if variant == "embeddinggemma":
        return {
            "task_prefix": info["embeddinggemma_protocol"]["prompt"],
            "normalization": info["embeddinggemma_protocol"]["normalization"],
        }
    return {"task_prefix": None, "normalization": None}


def validate_encoding_metadata(
    info: dict[str, Any], encoder_info: dict[str, Any]
) -> None:
    """Preserve encoder-specific sidecar checks, without importing the encoders."""
    variant = encoder_info["encoder_variant"]
    expected = encoding_metadata(encoder_info)
    if variant == "nomic" and info.get("normalization") != expected["normalization"]:
        raise ValueError("has incompatible Nomic normalization")
    if variant == "qwen3" and info.get("normalization") != expected["normalization"]:
        raise ValueError("has incompatible Qwen3 normalization")
    if variant == "embeddinggemma" and any(
        info.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("has incompatible EmbeddingGemma normalization/prompt")


def get_encoder_factory(config: dict[str, Any], variant: str | None = None):
    """Return the configured backend's factory; ``variant`` optionally overrides config.

    Backends are imported lazily, so isolated embedding environments never import LangVAE.
    """
    variant = variant or config.get("variant", "langvae")
    if variant == "langvae":
        from src.encoder.langvae import make_langvae_encoder

        return make_langvae_encoder
    if variant == "nomic":
        from src.encoder.nomic import make_nomic_encoder

        return make_nomic_encoder
    if variant == "qwen3":
        from src.encoder.qwen3 import make_qwen3_encoder

        return make_qwen3_encoder
    if variant == "embeddinggemma":
        from src.encoder.embeddinggemma import make_embeddinggemma_encoder

        return make_embeddinggemma_encoder
    raise ValueError(f"unknown encoder variant: {variant}")


def make_encoder(config: dict[str, Any], variant: str | None = None):
    """Construct the configured encoder; ``variant`` optionally overrides config."""
    return get_encoder_factory(config, variant)(config)


def add_encoder_arguments(parser) -> None:
    """The unchanged encoder CLI, shared by the lightweight pipeline entry point."""
    parser.add_argument(
        "--encoder-variant",
        choices=("langvae", "nomic", "qwen3", "embeddinggemma"),
        default=None,
        help="override encoder.variant before artifact paths are resolved",
    )
    parser.add_argument(
        "--nomic-dim",
        type=int,
        choices=(64, 128, 256, 512, 768),
        default=None,
        help="Nomic output dimension; also selects nomic_<dimension> artifact paths",
    )
    parser.add_argument(
        "--qwen3-dim",
        type=int,
        choices=(32, 64, 128, 256, 512, 768, 1024),
        default=None,
        help="Qwen3-Embedding-0.6B width; selects qwen3_<dimension> paths",
    )
    parser.add_argument(
        "--embeddinggemma-dim",
        type=int,
        choices=(128, 256, 512, 768),
        default=None,
        help="EmbeddingGemma-300m width; selects embeddinggemma_<dimension> paths",
    )
