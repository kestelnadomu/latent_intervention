"""Nomic Embed v1.5 encoder with a configurable Matryoshka output width; no decoder."""

from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

from transformers import AutoModel, AutoTokenizer

from src.encoder.protocols import BASE_MODEL_PATTERNS, nomic_dimension
from src.encoder.progress import log_encoding_progress

DEFAULT_NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"



class NomicTextEncoder:
    """f(x) = L2(first d dims of LN(mean-pooled Nomic Embed v1.5)); no decoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_NOMIC_MODEL,
        model_revision: str | None = None,
        code_revision: str | None = None,
        device: str | torch.device | None = None,
        max_len: int = 512,
        task: str = "classification",
        latent_dim: int = 128,
        progress: bool = False,
    ) -> None:
        self.latent_dim = nomic_dimension(latent_dim)
        self.progress = progress
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.max_len = max_len
        self.task_prefix = f"{task}: "
        source = model_name
        if model_revision is not None:
            source = snapshot_download(
                repo_id=model_name,
                revision=model_revision,
                allow_patterns=list(BASE_MODEL_PATTERNS),
            )
        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        self.tokenizer = AutoTokenizer.from_pretrained(source, **load_kwargs)
        if code_revision is not None:
            load_kwargs["code_revision"] = code_revision
        self.model = AutoModel.from_pretrained(source, **load_kwargs)
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        deterministic: bool = True,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Apply layer norm at full width, then truncate and L2-normalize."""
        if not deterministic:
            raise ValueError("NomicTextEncoder only supports deterministic encoding")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not texts:
            return torch.empty((0, self.latent_dim), device=self.device)
        chunks = []
        prefixed = [self.task_prefix + text for text in texts]
        started = perf_counter()
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
            if embeddings.shape[1] != 768:
                raise ValueError("Nomic Embed v1.5 must produce 768 pooled coordinates")
            embeddings = F.layer_norm(embeddings, (embeddings.shape[1],))
            chunks.append(F.normalize(embeddings[:, : self.latent_dim], p=2, dim=1))
            log_encoding_progress(
                "Nomic",
                start,
                batch_size,
                len(prefixed),
                started,
                enabled=self.progress,
            )
        return torch.cat(chunks, dim=0)



def make_nomic_encoder(config: dict[str, Any]) -> NomicTextEncoder:
    """Construct the Nomic encoder from an ``encoder`` config section."""
    return NomicTextEncoder(
        model_name=config.get("nomic_model_name", DEFAULT_NOMIC_MODEL),
        model_revision=config.get("nomic_model_revision"),
        code_revision=config.get("nomic_code_revision"),
        task=config.get("nomic_task", "classification"),
        latent_dim=config.get("nomic_latent_dim", 128),
        progress=bool(config.get("progress", False)),
        device=config["device"],
        max_len=int(config["max_len"]),
    )
