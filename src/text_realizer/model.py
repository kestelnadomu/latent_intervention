"""A learned latent prefix and optional LoRA; no changes to the frozen encoder."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class TextRealizer(nn.Module):
    def __init__(self, language_model, tokenizer, spec: dict, generation: dict):
        super().__init__()
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.spec = dict(spec)
        self.generation = dict(generation)
        self.latent_dim = spec["latent_dim"]
        self.prefix_tokens = spec["prefix_tokens"]
        width = language_model.config.hidden_size
        hidden = spec["projector_hidden"]
        self.projector = nn.Sequential(
            nn.Linear(self.latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.prefix_tokens * width),
        )
        self.hidden_size = width
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id
        if any(value is None for value in (self.bos_id, self.eos_id, self.pad_id)):
            raise ValueError("explicit BOS/EOS/PAD tokens are required")

    @property
    def device(self) -> torch.device:
        return next(self.projector.parameters()).device

    def train(self, mode: bool = True) -> TextRealizer:
        super().train(mode)
        if not self.spec["use_lora"]:
            # Frozen parameters still need a training-mode graph for activation
            # checkpointing. Gemma's functional attention dropout is zero; keep
            # any module dropout disabled without disabling checkpointing.
            for layer in self.language_model.modules():
                if isinstance(layer, nn.Dropout):
                    layer.eval()
        return self

    def prefix(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.latent_dim or not torch.isfinite(z).all():
            raise ValueError(f"expected finite [batch, {self.latent_dim}] vectors")
        z = z.to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(z).all():
            raise ValueError("latent values cannot be represented as finite FP32")
        return self.projector(z * math.sqrt(self.latent_dim)).reshape(
            len(z), self.prefix_tokens, self.hidden_size
        )

    def teacher_inputs(self, z: torch.Tensor, targets: list[list[int]]) -> dict:
        if len(z) != len(targets) or not targets or any(not row for row in targets):
            raise ValueError("one nonempty CV+EOS target per vector is required")
        width = max(map(len, targets))
        if (
            self.prefix_tokens + 1 + width
            > self.language_model.config.max_position_embeddings
        ):
            raise ValueError("target exceeds decoder context; truncation is forbidden")
        tokens = torch.full(
            (len(z), width + 1), self.pad_id, device=self.device, dtype=torch.long
        )
        mask = torch.zeros_like(tokens)
        labels = torch.full(
            (len(z), self.prefix_tokens + width + 1),
            -100,
            device=self.device,
            dtype=torch.long,
        )
        tokens[:, 0] = self.bos_id
        for i, row in enumerate(targets):
            if row[-1] != self.eos_id:
                raise ValueError("target must include EOS")
            tokens[i, 1 : len(row) + 1] = torch.tensor(row, device=self.device)
            mask[i, : len(row) + 1] = 1
            labels[i, self.prefix_tokens + 1 : self.prefix_tokens + 1 + len(row)] = (
                tokens[i, 1 : len(row) + 1]
            )
        # Gemma's embedding module already applies its native sqrt(hidden_size) scale.
        embeddings = self.language_model.get_input_embeddings()(tokens)
        prefix = self.prefix(z)
        return {
            "inputs_embeds": torch.cat((prefix, embeddings), dim=1),
            "attention_mask": torch.cat(
                (mask.new_ones((len(z), self.prefix_tokens)), mask), dim=1
            ),
            "labels": labels,
        }

    def loss_sums(
        self, z: torch.Tensor, targets: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.teacher_inputs(z, targets)
        labels = inputs.pop("labels")
        output = self.language_model(**inputs, use_cache=False)
        # One shift only: BOS predicts the first CV token. EOS remains supervised.
        losses = F.cross_entropy(
            output.logits[:, :-1].float().transpose(1, 2),
            labels[:, 1:],
            ignore_index=-100,
            reduction="none",
        )
        return losses.sum(1), labels[:, 1:].ne(-100).sum(1)

    def decode(self, z: torch.Tensor) -> list[str]:
        from .generation import generate

        return [row["text"] for row in generate(self, z)]

    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def restore_trainable(self, state: dict) -> None:
        expected = {
            name: parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        if set(state) != set(expected):
            raise ValueError("trainable checkpoint schema mismatch")
        with torch.no_grad():
            for name, parameter in expected.items():
                value = state[name]
                if value.shape != parameter.shape or not torch.isfinite(value).all():
                    raise ValueError(f"invalid checkpoint parameter: {name}")
                parameter.copy_(value)


def model_spec(config: dict, variant: str, hidden_size: int) -> dict:
    if variant not in {"prefix-compact", "prefix-wide", "lora-compact", "lora-wide"}:
        raise ValueError("unknown decoder variant")
    return {
        **config,
        "variant": variant,
        "use_lora": variant.startswith("lora-"),
        "projector_hidden": 768 if variant.endswith("compact") else 4 * hidden_size,
        "decoder_hidden_size": hidden_size,
    }


def cached_decoder(spec: dict, cache_dir: str | Path | None = None) -> str:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    choices = (
        [cache_dir]
        if cache_dir is not None
        else [Path(__file__).resolve().parents[2] / ".cache/huggingface/hub", None]
    )
    for location in choices:
        try:
            return snapshot_download(
                repo_id=spec["name"],
                revision=spec["revision"],
                cache_dir=location,
                local_files_only=True,
            )
        except LocalEntryNotFoundError:
            if location == choices[-1]:
                raise
    raise FileNotFoundError("pinned decoder snapshot is not cached")


def build_model(
    spec: dict,
    generation: dict,
    *,
    device: str | torch.device = "cpu",
    cache_dir: str | Path | None = None,
) -> TextRealizer:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Pass the resolved local path: this Transformers version's tokenizer
    # compatibility hook can otherwise call model_info despite local_files_only.
    source = cached_decoder(spec, cache_dir)
    common = {"local_files_only": True, "trust_remote_code": False}
    attention = spec.get("attention_implementation", "sdpa")
    if attention not in {"sdpa", "eager"}:
        raise ValueError("unsupported decoder attention implementation")
    tokenizer = AutoTokenizer.from_pretrained(source, **common)
    base = AutoModelForCausalLM.from_pretrained(
        source, **common, dtype=torch.float32, attn_implementation=attention
    )
    if (
        base.config.hidden_size != spec["decoder_hidden_size"]
        or base.config.model_type != "gemma3_text"
        or base.config.attention_dropout != 0
    ):
        raise ValueError("unexpected decoder backbone")
    base.requires_grad_(False)
    if spec["use_lora"]:
        from peft import LoraConfig, get_peft_model

        found = {
            name.rsplit(".", 1)[-1]
            for name, layer in base.named_modules()
            if isinstance(layer, nn.Linear)
        }
        if not set(spec["lora_targets"]) <= found:
            raise ValueError("missing requested LoRA target modules")
        base = get_peft_model(
            base,
            LoraConfig(
                r=spec["lora_rank"],
                lora_alpha=spec["lora_alpha"],
                lora_dropout=spec["lora_dropout"],
                target_modules=spec["lora_targets"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    model = TextRealizer(base, tokenizer, spec, generation).to(device).float()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not (
            name.startswith("projector.") or "lora_" in name
        ):
            raise ValueError(f"unexpected trainable backbone parameter: {name}")
    return model
