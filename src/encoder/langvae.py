"""LangVAE encoder: the posterior mean mu(x) of a pinned LangVAE checkpoint (default f).

``local_checkpoint`` loads a fine-tuned folder (see exp/langvae_ft/).
"""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from huggingface_hub import snapshot_download

from langvae import LangVAE
from langvae.data_conversion.tokenization import TokenizedDataSet

from src.encoder.protocols import BASE_MODEL_PATTERNS

DEFAULT_MODEL = "neuro-symbolic-ai/eb-langvae-bert-base-cased-gpt2-l128"

DEFAULT_LANGVAE_ENCODER_MODEL = "bert-base-cased"

DEFAULT_LANGVAE_ENCODER_REVISION = "cd5ef92a9fb2f889e972770a36d4ed042daf221e"

DEFAULT_LANGVAE_DECODER_MODEL = "gpt2"

DEFAULT_LANGVAE_DECODER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"



def _load_pinned_langvae(
    checkpoint: str | Path,
    *,
    encoder_model_name: str,
    encoder_model_revision: str,
    decoder_model_name: str,
    decoder_model_revision: str,
) -> LangVAE:
    """Load LangVAE while pinning the base BERT and GPT-2 snapshots it reconstructs."""
    checkpoint = Path(checkpoint).resolve()
    encoder_snapshot = Path(
        snapshot_download(
            repo_id=encoder_model_name,
            revision=encoder_model_revision,
            allow_patterns=list(BASE_MODEL_PATTERNS),
        )
    )
    decoder_snapshot = Path(
        snapshot_download(
            repo_id=decoder_model_name,
            revision=decoder_model_revision,
            allow_patterns=list(BASE_MODEL_PATTERNS),
        )
    )
    with TemporaryDirectory(prefix="latent-intervention-langvae-") as overlay_dir:
        overlay = Path(overlay_dir)
        for filename in (
            "environment.json",
            "model_config.json",
            "encoder.pt",
            "decoder.pt",
        ):
            source = checkpoint / filename
            if not source.is_file():
                raise FileNotFoundError(f"LangVAE checkpoint is missing {source}")
            os.symlink(source, overlay / filename)
        for filename, expected_model, snapshot in (
            ("encoder_cfg.json", encoder_model_name, encoder_snapshot),
            ("decoder_cfg.json", decoder_model_name, decoder_snapshot),
        ):
            config_path = checkpoint / filename
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("model_path") != expected_model:
                raise ValueError(
                    f"{config_path} uses base model {config.get('model_path')!r}; "
                    f"expected {expected_model!r}"
                )
            config["model_path"] = str(snapshot)
            (overlay / filename).write_text(json.dumps(config), encoding="utf-8")
        model = LangVAE.load_from_folder(str(overlay))
        # Keep later fine-tuned checkpoints portable; our loader re-applies the pins.
        model.encoder.model_path = encoder_model_name
        model.decoder.model_path = decoder_model_name
        return model



class TextEncoder:
    """`langvae` variant: f(x) = mu(x), the LangVAE posterior mean (optionally a fine-tuned checkpoint)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | torch.device | None = None,
        max_len: int = 512,
        local_checkpoint: str | Path | None = None,
        model_revision: str | None = None,
        encoder_model_name: str = DEFAULT_LANGVAE_ENCODER_MODEL,
        encoder_model_revision: str = DEFAULT_LANGVAE_ENCODER_REVISION,
        decoder_model_name: str = DEFAULT_LANGVAE_DECODER_MODEL,
        decoder_model_revision: str = DEFAULT_LANGVAE_DECODER_REVISION,
    ) -> None:
        """Load a LangVAE checkpoint from the HF hub, or from a local folder
        (e.g. produced by exp/langvae_ft/) if `local_checkpoint` is given."""
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.max_len = max_len
        checkpoint = (
            Path(local_checkpoint)
            if local_checkpoint is not None
            else Path(snapshot_download(repo_id=model_name, revision=model_revision))
        )
        self.model = _load_pinned_langvae(
            checkpoint,
            encoder_model_name=encoder_model_name,
            encoder_model_revision=encoder_model_revision,
            decoder_model_name=decoder_model_name,
            decoder_model_revision=decoder_model_revision,
        )
        self.model.eval()
        self.model.to(self.device)

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the latent space."""
        return self.model.model_config.latent_dim

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        deterministic: bool = True,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Encode texts into latent vectors of shape (len(texts), latent_dim).

        With deterministic=True (default) returns the posterior mean; otherwise
        samples z from the posterior.
        """
        dataset = TokenizedDataSet(texts, self.model.decoder.tokenizer, self.max_len)
        chunks = []
        for start in range(0, len(texts), batch_size):
            x = dataset[start : start + batch_size]["data"].to(self.device)
            if deterministic:
                z = self.model.encoder(x).embedding
            else:
                z, _ = self.model.encode_z(x)
            chunks.append(z)
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> list[str]:
        """Decode latent vectors back into sentences (useful for sanity checks)."""
        return self.model.decode_sentences(z.to(self.device))



def make_langvae_encoder(config: dict[str, Any]) -> TextEncoder:
    """Construct the LangVAE encoder from an ``encoder`` config section."""
    return TextEncoder(
        model_name=config["model_name"],
        model_revision=config.get("model_revision"),
        local_checkpoint=config.get("local_checkpoint"),
        encoder_model_name=config.get(
            "langvae_encoder_model_name", DEFAULT_LANGVAE_ENCODER_MODEL
        ),
        encoder_model_revision=config.get(
            "langvae_encoder_model_revision", DEFAULT_LANGVAE_ENCODER_REVISION
        ),
        decoder_model_name=config.get(
            "langvae_decoder_model_name", DEFAULT_LANGVAE_DECODER_MODEL
        ),
        decoder_model_revision=config.get(
            "langvae_decoder_model_revision", DEFAULT_LANGVAE_DECODER_REVISION
        ),
        device=config["device"],
        max_len=int(config["max_len"]),
    )
