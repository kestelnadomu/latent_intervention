"""Explicit gated-model downloads; never accept terms or print credentials."""

from pathlib import Path

from .settings import resolve

DECODER_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)


def snapshots(config: dict, *, download: bool = False) -> dict[str, Path]:
    from huggingface_hub import snapshot_download
    from src.encoder.embeddinggemma import MODEL_FILES

    choices = {
        "decoder": (
            config["model"]["name"],
            config["model"]["revision"],
            DECODER_FILES,
        ),
        "encoder": (
            config["encoder"]["embeddinggemma_model_name"],
            config["encoder"]["embeddinggemma_model_revision"],
            MODEL_FILES,
        ),
    }
    result = {}
    for kind, (name, revision, files) in choices.items():
        result[kind] = Path(
            snapshot_download(
                repo_id=name,
                revision=revision,
                cache_dir=str(resolve(config["paths"]["cache"])),
                allow_patterns=list(files),
                local_files_only=not download,
                max_workers=2,
            )
        )
    return result


def model_files(config: dict) -> dict:
    from src.artifact_io import sha256_file

    return {
        kind: {
            str(path.relative_to(root)): sha256_file(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
        for kind, root in snapshots(config).items()
    }


def main():
    import argparse
    from .settings import DEFAULT_CONFIG, read_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--download-models", action="store_true")
    args = parser.parse_args()
    if args.download_models:
        snapshots(read_config(args.config), download=True)
        print("Pinned model snapshots cached; no training or encoding was run.")
    else:
        print(
            "Environment only. Use --download-models after personally accepting both Gemma licenses."
        )


if __name__ == "__main__":
    main()
