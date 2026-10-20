"""Small real-backbone interface checks, independent of reconstruction quality."""

from __future__ import annotations

import torch

from src.text_realizer.artifacts import tensor_digest
from src.text_realizer.generation import generate


def frozen_digest(model) -> str:
    return tensor_digest(
        {
            name: parameter
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        }
    )


def check_gradients_and_frozen(model, expected: str) -> dict:
    if frozen_digest(model) != expected:
        raise RuntimeError("frozen backbone parameters changed during preflight")
    projector = any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for parameter in model.projector.parameters()
    )
    lora = any(
        "lora_" in name
        and parameter.grad is not None
        and bool(parameter.grad.abs().sum() > 0)
        for name, parameter in model.named_parameters()
    )
    if not projector or (model.spec["use_lora"] and not lora):
        raise RuntimeError("projector/LoRA gradients are missing")
    return {
        "frozen_parameters_unchanged": True,
        "projector_gradient": projector,
        "lora_gradient": lora,
    }


@torch.inference_mode()
def check_native_forward(model, z: torch.Tensor, target: list[int]) -> dict:
    previous = model.training
    model.eval()
    try:
        ids = torch.tensor([[model.bos_id] + target[:4]], device=model.device)
        embeddings = model.language_model.get_input_embeddings()(ids)
        native = model.language_model(
            input_ids=ids, use_cache=False, logits_to_keep=1
        ).logits
        embedded = model.language_model(
            inputs_embeds=embeddings, use_cache=False, logits_to_keep=1
        ).logits
        torch.testing.assert_close(native, embedded, rtol=2e-5, atol=2e-4)
        prefix = model.prefix(z[:1])
        # Force a short continuation across the real model's local-attention
        # boundary, regardless of whether a randomly initialized prefix emits EOS.
        window = model.language_model.config.sliding_window
        count = window + 3 - model.prefix_tokens
        sequence = [model.bos_id] + (target * (count // len(target) + 1))[: count - 1]
        tokens = torch.tensor([sequence], device=model.device)
        all_embeddings = torch.cat(
            (prefix, model.language_model.get_input_embeddings()(tokens)), 1
        )
        cached = model.language_model(
            inputs_embeds=all_embeddings[:, :window],
            attention_mask=torch.ones(1, window, device=model.device, dtype=torch.long),
            use_cache=True,
            logits_to_keep=1,
        )
        differences = []
        for position in range(window, all_embeddings.shape[1]):
            index = torch.tensor([position], device=model.device)
            cached = model.language_model(
                input_ids=tokens[
                    :,
                    position - model.prefix_tokens : position - model.prefix_tokens + 1,
                ],
                attention_mask=torch.ones(
                    1, position + 1, device=model.device, dtype=torch.long
                ),
                past_key_values=cached.past_key_values,
                cache_position=index,
                position_ids=index[None, :],
                use_cache=True,
                logits_to_keep=1,
            )
            full = model.language_model(
                inputs_embeds=all_embeddings[:, : position + 1],
                attention_mask=torch.ones(
                    1, position + 1, device=model.device, dtype=torch.long
                ),
                use_cache=False,
                logits_to_keep=1,
            )
            torch.testing.assert_close(cached.logits, full.logits, rtol=2e-5, atol=2e-4)
            differences.append(float((cached.logits - full.logits).abs().max()))
        rows = generate(model, z[:1].repeat(2, 1), batch_size=1, max_new_tokens=16)
        repeated = generate(model, z[:1], batch_size=1, max_new_tokens=16)
        if rows[0] != rows[1] or rows[0] != repeated[0]:
            raise RuntimeError("identical vectors do not generate identical outputs")
        return {
            "native_embedding_scaling": True,
            "attention_implementation": model.language_model.config._attn_implementation,
            "cached_logits_max_abs": max(differences),
            "cache_test_sequence_length": all_embeddings.shape[1],
            "sliding_window": window,
            "deterministic_generation": True,
        }
    finally:
        model.train(previous)
