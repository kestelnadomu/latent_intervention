"""Download the pinned gated checkpoint after the user personally accepts its terms."""

from huggingface_hub import snapshot_download

from src.config import load_config
from src.embeddinggemma_encoder import MODEL_FILES, cached_snapshot


def main() -> None:
    enc = load_config(encoder_variant="embeddinggemma")["encoder"]
    snapshot_download(
        repo_id=enc["embeddinggemma_model_name"],
        revision=enc["embeddinggemma_model_revision"],
        cache_dir=enc["embeddinggemma_cache_dir"],
        allow_patterns=list(MODEL_FILES),
        max_workers=4,
    )
    print(f"EmbeddingGemma checkpoint cached: {cached_snapshot(enc)}")


if __name__ == "__main__":
    main()
