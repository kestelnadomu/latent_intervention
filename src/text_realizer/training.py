"""Small token-weighted PyTorch updates and exact optimizer-boundary resume."""

from __future__ import annotations

import math
import random
from pathlib import Path

import torch

from src.artifact_io import save_torch
from .model import TextRealizer


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(tokenizer, texts: list[str]) -> list[list[int]]:
    sequences = tokenizer(
        texts, add_special_tokens=False, padding=False, truncation=False
    )["input_ids"]
    if any(not sequence for sequence in sequences):
        raise ValueError("empty tokenized CV")
    return [list(sequence) + [tokenizer.eos_token_id] for sequence in sequences]


def optimizer_for(model: TextRealizer, settings: dict) -> torch.optim.AdamW:
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        rate = (
            settings["projector_lr"]
            if name.startswith("projector.")
            else settings["lora_lr"]
        )
        decay = settings["weight_decay"] if parameter.ndim >= 2 else 0.0
        groups.setdefault((rate, decay), []).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": values, "lr": rate, "weight_decay": decay}
            for (rate, decay), values in groups.items()
        ]
    )


def scheduler_for(
    optimizer: torch.optim.Optimizer, total_updates: int, warmup_fraction: float
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(1, math.ceil(total_updates * warmup_fraction))

    def scale(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_updates - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def update(
    model: TextRealizer,
    optimizer: torch.optim.Optimizer,
    z: torch.Tensor,
    targets: list[list[int]],
    indices: list[int],
    *,
    microbatch: int,
    clip: float,
) -> dict:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    token_count = sum(len(targets[i]) for i in indices)
    total = 0.0
    for start in range(0, len(indices), microbatch):
        chosen = indices[start : start + microbatch]
        sums, _ = model.loss_sums(z[chosen], [targets[i] for i in chosen])
        if not torch.isfinite(sums).all():
            raise FloatingPointError("nonfinite reconstruction loss")
        (sums.sum() / token_count).backward()
        total += float(sums.detach().sum())
    parameters = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    optimizer.step()
    return {
        "nll": total / token_count,
        "tokens": token_count,
        "gradient_norm": float(norm),
    }


@torch.no_grad()
def evaluate_nll(model, z, targets, *, microbatch=1, progress=None) -> dict:
    previous = model.training
    model.eval()
    sums, counts = [], []
    try:
        for start in range(0, len(targets), microbatch):
            losses, tokens = model.loss_sums(
                z[start : start + microbatch], targets[start : start + microbatch]
            )
            sums.extend(losses.cpu().tolist())
            counts.extend(tokens.cpu().tolist())
            if progress is not None:
                progress(
                    "teacher_nll", min(start + microbatch, len(targets)), len(targets)
                )
    finally:
        model.train(previous)
    if not sums or not all(math.isfinite(value) for value in sums):
        raise FloatingPointError("nonfinite or empty validation")
    return {"nll": sum(sums) / sum(counts), "nll_sums": sums, "token_counts": counts}


def save_resume(
    path: str | Path,
    model: TextRealizer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    progress: dict,
    *,
    identity: str,
) -> None:
    save_torch(
        path,
        {
            "identity": identity,
            "weights": model.trainable_state(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state(model.device)
                if model.device.type == "cuda"
                else None
            ),
            "progress": progress,
        },
    )


def restore_resume(
    path: str | Path,
    model: TextRealizer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    *,
    identity: str,
) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["identity"] != identity:
        raise ValueError("resume identity changed")
    model.restore_trainable(state["weights"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"], model.device)
    return state["progress"]
