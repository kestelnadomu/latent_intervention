"""phi: a frozen text encoder, preselected to the three embedding models we use.

``Encoder("nomic", 128).latents("talent")`` loads the published encodings of an
experiment's texts (``data/latents/<experiment>/<name>_<dim>/z_pairs.pt``), which is
what a notebook normally needs. ``Encoder.encode(texts)`` runs the model itself;
Qwen3 and EmbeddingGemma only run in their isolated environments
(``scripts/setup_qwen3.sh``, ``scripts/setup_embeddinggemma.sh``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

ENCODERS: dict[str, tuple[int, ...]] = {
    "nomic": (64, 128, 256, 512, 768),
    "qwen3": (32, 64, 128, 256, 512, 768, 1024),
    "embeddinggemma": (128, 256, 512, 768),
}


@dataclass
class Latents:
    """Factual latents ``z`` of the units ``ids``, and the paired counterfactual
    latents ``z_cf`` = phi(x') of the units ``cf_ids`` (where a counterfactual text exists)."""

    ids: list[int]
    z: torch.Tensor
    cf_ids: list[int]
    z_cf: torch.Tensor

    def factual(self, ids) -> torch.Tensor:
        """z for the given unit ids, in that order."""
        row = {unit: i for i, unit in enumerate(self.ids)}
        return self.z[[row[int(unit)] for unit in ids]]

    def counterfactual(self, ids) -> torch.Tensor:
        """phi(x') for the given unit ids, in that order."""
        row = {unit: i for i, unit in enumerate(self.cf_ids)}
        return self.z_cf[[row[int(unit)] for unit in ids]]


class Encoder:
    """One embedding model at one Matryoshka width, e.g. ``Encoder("qwen3", 256)``."""

    def __init__(self, name: str = "nomic", dim: int = 128, device: str = "cpu") -> None:
        if name not in ENCODERS:
            raise ValueError(f"unknown encoder {name!r}; choose from {sorted(ENCODERS)}")
        if dim not in ENCODERS[name]:
            raise ValueError(f"{name} supports dims {ENCODERS[name]}, got {dim}")
        self.name, self.dim, self.device = name, dim, device
        self._model = None

    @property
    def tag(self) -> str:
        return f"{self.name}_{self.dim}"

    def __repr__(self) -> str:
        return f"Encoder({self.name!r}, {self.dim})"

    def latents(self, experiment: str = "talent", path: str | Path | None = None) -> Latents:
        """Load cached encodings: ``data/latents/<experiment>/<tag>/z_pairs.pt`` or ``path``."""
        path = Path(path) if path else ROOT / "data/latents" / experiment / self.tag / "z_pairs.pt"
        if not path.exists():
            raise FileNotFoundError(f"no cached latents at {path}; encode the texts first")
        payload = torch.load(path, weights_only=False)
        return Latents(
            ids=[int(i) for i in payload["ids"]],
            z=payload["z"].float(),
            cf_ids=[int(i) for i in payload["test_ids"]],
            z_cf=payload["z_prime"].float(),
        )

    def encode(self, texts: list[str], batch_size: int = 8) -> torch.Tensor:
        """Encode texts with the pinned model (downloads it on first use)."""
        if self._model is None:
            from src.config import load_config
            from src.encoder import configure_encoder, make_encoder

            config = {"encoder": copy.deepcopy(load_config("encoder"))}
            config["encoder"].update(variant=self.name, device=self.device)
            configure_encoder(config, **{f"{self.name}_dim": self.dim})
            self._model = make_encoder(config["encoder"])
        return self._model.encode(list(texts), batch_size=batch_size)
