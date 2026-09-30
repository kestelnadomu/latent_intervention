"""Atomic paired-latent publication, separate from ID alignment and encoding."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from src.artifact_io import save_torch, write_json, sha256_file
from src.encoder_protocols import encoding_metadata

ARTIFACT_VERSION = 1


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _require_new_output(path: str | Path) -> None:
    output = Path(path)
    if output.exists() or output.with_suffix(".info.json").exists():
        raise FileExistsError(
            f"refusing to overwrite existing latent artifact: {output}"
        )


def _input_hashes(paths: dict) -> dict[str, str]:
    return {
        "pair_index": sha256_file(paths["pair_index"]),
        "factual_text": sha256_file(paths["texts"]),
        "counterfactual_text": sha256_file(paths["texts_counterfactual"]),
    }


def write_latent_payload(
    paths: dict,
    payload: dict,
    encoder_info: dict,
    *,
    derivation: dict | None = None,
    expected_input_hashes: dict[str, str] | None = None,
) -> None:
    """Publish validated tensors without overwriting or mixing changed source inputs."""
    output = Path(paths["latents"])
    _require_new_output(output)
    input_hashes = _input_hashes(paths)
    if expected_input_hashes is not None and input_hashes != expected_input_hashes:
        raise ValueError("source inputs changed during encoding; refusing to publish")
    save_torch(output, payload)
    write_json(
        output.with_suffix(".info.json"),
        {
            "artifact_version": ARTIFACT_VERSION,
            "artifact_sha256": sha256_file(output),
            **encoder_info,
            **encoding_metadata(encoder_info),
            "langvae_version": _package_version("langvae"),
            "transformers_version": _package_version("transformers"),
            **(
                {
                    "sentence_transformers_version": _package_version(
                        "sentence-transformers"
                    )
                }
                if encoder_info["encoder_variant"] == "embeddinggemma"
                else {}
            ),
            "factual_units": len(payload["ids"]),
            "counterfactual_test_units": len(payload["test_ids"]),
            "identity_test_units": int(payload["is_identity"].sum()),
            "input_sha256": input_hashes,
            **({"derivation": derivation} if derivation else {}),
        },
    )
