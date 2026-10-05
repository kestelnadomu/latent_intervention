"""Prefetch only the pinned Qwen3-Embedding-0.6B files; never run full inference."""

from huggingface_hub import snapshot_download

from src.config import load_config
from src.encoder.qwen3 import MODEL_FILES


def main() -> None:
    encoder = load_config(encoder_variant="qwen3")["encoder"]
    path = snapshot_download(
        repo_id=encoder["qwen3_model_name"],
        revision=encoder["qwen3_model_revision"],
        cache_dir=encoder["qwen3_cache_dir"],
        allow_patterns=list(MODEL_FILES),
        max_workers=4,
    )
    print(f"Qwen3 checkpoint cached: {path}")


if __name__ == "__main__":
    main()
