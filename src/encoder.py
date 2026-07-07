"""
Latent text representations from a pre-trained LangVAE model.

TextEncoder wraps a LangVAE checkpoint from the HuggingFace hub and exposes:
- encode(texts) -> (n, latent_dim) tensor of latent representations
- decode(z) -> list of sentences (round-trip through the VAE decoder)

Pre-trained checkpoints: https://huggingface.co/neuro-symbolic-ai
"""

from pathlib import Path

import torch

from langvae import LangVAE
from langvae.data_conversion.tokenization import TokenizedDataSet

DEFAULT_MODEL = "neuro-symbolic-ai/eb-langvae-bert-base-cased-gpt2-l128"


class TextEncoder:
    """Pre-trained text encoder producing latent representations via LangVAE."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | torch.device | None = None,
        max_len: int = 512,
        local_checkpoint: str | Path | None = None,
    ) -> None:
        """Load a LangVAE checkpoint from the HF hub, or from a local folder
        (e.g. produced by src/finetune_vae.py) if `local_checkpoint` is given."""
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.max_len = max_len
        if local_checkpoint is not None:
            self.model = LangVAE.load_from_folder(str(local_checkpoint))
        else:
            self.model = LangVAE.load_from_hf_hub(model_name)
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
        dataset = TokenizedDataSet(texts, self.model.encoder.tokenizer, self.max_len)
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
