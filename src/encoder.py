"""128-dimensional text encoders used by the latent intervention pipeline."""

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

from langvae import LangVAE
from langvae.data_conversion.tokenization import TokenizedDataSet
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "neuro-symbolic-ai/eb-langvae-bert-base-cased-gpt2-l128"
DEFAULT_NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"


class TextEncoder:
    """Pre-trained text encoder producing latent representations via LangVAE."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | torch.device | None = None,
        max_len: int = 512,
        local_checkpoint: str | Path | None = None,
        model_revision: str | None = None,
    ) -> None:
        """Load a LangVAE checkpoint from the HF hub, or from a local folder
        (e.g. produced by src/finetune_vae.py) if `local_checkpoint` is given."""
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.max_len = max_len
        if local_checkpoint is not None:
            self.model = LangVAE.load_from_folder(str(local_checkpoint))
        elif model_revision is not None:
            checkpoint = snapshot_download(repo_id=model_name, revision=model_revision)
            self.model = LangVAE.load_from_folder(checkpoint)
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


class NomicTextEncoder:
    """Nomic Embed v1.5 reduced to its trained 128-D Matryoshka output."""

    latent_dim = 128

    def __init__(
        self,
        model_name: str = DEFAULT_NOMIC_MODEL,
        model_revision: str | None = None,
        code_revision: str | None = None,
        device: str | torch.device | None = None,
        max_len: int = 512,
        task: str = "classification",
    ) -> None:
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.max_len = max_len
        self.task_prefix = f"{task}: "
        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if model_revision is not None:
            load_kwargs["revision"] = model_revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
        if code_revision is not None:
            load_kwargs["code_revision"] = code_revision
        self.model = AutoModel.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        deterministic: bool = True,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Encode texts using Nomic's documented 128-D pooling procedure."""
        if not deterministic:
            raise ValueError("NomicTextEncoder only supports deterministic encoding")
        chunks = []
        prefixed = [self.task_prefix + text for text in texts]
        for start in range(0, len(prefixed), batch_size):
            inputs = self.tokenizer(
                prefixed[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            token_embeddings = self.model(**inputs)[0]
            mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
            embeddings = (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            embeddings = F.layer_norm(embeddings, (embeddings.shape[1],))
            chunks.append(F.normalize(embeddings[:, : self.latent_dim], p=2, dim=1))
        return torch.cat(chunks, dim=0)


def make_encoder(config: dict[str, Any], variant: str | None = None):
    """Construct the configured encoder; ``variant`` optionally overrides config."""
    variant = variant or config.get("variant", "langvae")
    common = {
        "device": config["device"],
        "max_len": int(config["max_len"]),
    }
    if variant == "langvae":
        return TextEncoder(
            model_name=config["model_name"],
            model_revision=config.get("model_revision"),
            local_checkpoint=config.get("local_checkpoint"),
            **common,
        )
    if variant == "nomic":
        return NomicTextEncoder(
            model_name=config.get("nomic_model_name", DEFAULT_NOMIC_MODEL),
            model_revision=config.get("nomic_model_revision"),
            code_revision=config.get("nomic_code_revision"),
            task=config.get("nomic_task", "classification"),
            **common,
        )
    raise ValueError(f"unknown encoder variant: {variant}")
