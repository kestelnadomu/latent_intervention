"""Warm-start all arms from identical factual weights on a common eager backend."""

from __future__ import annotations

import torch

from src.artifact_io import sha256_file
from src.text_realizer.artifacts import read_checkpoint
from src.text_realizer.model import build_model

from .settings import resolve


def load_reference(config: dict, *, device: str):
    source = resolve(config["inputs"]["baseline_checkpoint"])
    if sha256_file(source) != config["inputs"]["sha256"]["baseline_checkpoint"]:
        raise ValueError("factual reference checkpoint changed")
    payload = read_checkpoint(source)
    spec = dict(payload["metadata"]["spec"])
    spec["attention_implementation"] = config["model"]["attention_implementation"]
    if (
        spec["name"] != config["model"]["name"]
        or spec["revision"] != config["model"]["revision"]
        or spec["variant"] != "lora-compact"
    ):
        raise ValueError("reference architecture changed")
    model = build_model(
        spec,
        payload["metadata"]["generation"],
        device=device,
        cache_dir=str(resolve(config["paths"]["cache"])),
    )
    model.restore_trainable(payload["weights"])
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not (name.startswith("projector.") or "lora_" in name)
    ):
        raise ValueError("unexpected trainable base parameter")
    return model.eval()


def numerical_backend(model) -> str:
    return str(model.language_model.config._attn_implementation)


def clear_gpu(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
