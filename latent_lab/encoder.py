"""phi: the cached encodings of a frozen text encoder, one of the three embedding models we use.

``Encoder("nomic", 128).latents("talent")`` loads the published encodings of an
experiment's texts (``data/latents/<experiment>/<name>_<dim>/z_pairs.pt``). latent_lab
does not run the models; the texts are encoded by ``exp/encoding/``.
"""

from __future__ import annotations

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
    """Source latents ``z`` of the units ``ids``, and the paired target latents
    ``z_cf`` = phi(x') of the units ``cf_ids`` (where a text with the target state exists)."""

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

    def __init__(self, name: str = "nomic", dim: int = 128) -> None:
        if name not in ENCODERS:
            raise ValueError(f"unknown encoder {name!r}; choose from {sorted(ENCODERS)}")
        if dim not in ENCODERS[name]:
            raise ValueError(f"{name} supports dims {ENCODERS[name]}, got {dim}")
        self.name, self.dim = name, dim

    @property
    def tag(self) -> str:
        return f"{self.name}_{self.dim}"

    def __repr__(self) -> str:
        return f"Encoder({self.name!r}, {self.dim})"

    def latents(self, experiment: str = "talent", path: str | Path | None = None) -> Latents:
        """Load cached encodings: ``data/latents/<experiment>/<tag>/z_pairs.pt`` or ``path``.

        Without a cache at this width, the full-width cache is projected instead: the
        Matryoshka prefix of ``dim`` coordinates, renormalized (how the narrower caches were made).
        """
        payload = self._load(experiment, "z_pairs.pt", path)
        return Latents(
            ids=[int(i) for i in payload["ids"]],
            z=payload["z"],
            cf_ids=[int(i) for i in payload["test_ids"]],
            z_cf=payload["z_prime"],
        )

    def auxiliary(self, experiment: str, file: str = "z_factual_vb.pt") -> Latents:
        """Other cached source encodings next to ``z_pairs.pt``, e.g. the held-out
        paraphrase ``z_factual_vb.pt`` or the noise floor ``z_floor.pt``; read them
        with ``.factual(ids)``."""
        payload = self._load(experiment, file)
        return Latents([int(i) for i in payload["ids"]], payload["z"], [], payload["z"][:0])

    def _load(self, experiment: str, file: str, path: str | Path | None = None) -> dict:
        """A cached payload with its ``z``-like tensors at this width (projected if needed)."""
        if not path:
            folder = ROOT / "data/latents" / experiment
            path = folder / self.tag / file
            if not path.exists():
                path = folder / f"{self.name}_{max(ENCODERS[self.name])}" / file
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no cached latents at {path}; encode the texts first")
        payload = torch.load(path, weights_only=False)
        for key in ("z", "z_prime"):
            if key in payload:
                z = payload[key].float()
                if z.shape[1] < self.dim:
                    raise ValueError(f"cached latents at {path} have width {z.shape[1]} < {self.dim}")
                payload[key] = z if z.shape[1] == self.dim else torch.nn.functional.normalize(z[:, : self.dim], dim=1)
        return payload
