"""Greedy, cached, source-free generation with explicit prefix positions."""

from __future__ import annotations

import torch


@torch.inference_mode()
def generate(
    model,
    z: torch.Tensor,
    *,
    max_new_tokens: int | None = None,
    batch_size: int | None = None,
    progress=None,
) -> list[dict]:
    maximum = (
        model.generation["max_new_tokens"] if max_new_tokens is None else max_new_tokens
    )
    size = model.generation.get("batch_size", 1) if batch_size is None else batch_size
    if maximum < 1 or size < 1:
        raise ValueError("generation limits must be positive")
    if (
        model.prefix_tokens + 1 + maximum
        > model.language_model.config.max_position_embeddings
    ):
        raise ValueError("generation exceeds context limit")
    # Validate even empty batches; edited vectors are not silently renormalized.
    if z.ndim != 2 or z.shape[1] != model.latent_dim or not torch.isfinite(z).all():
        raise ValueError("invalid latent batch")
    was_training = model.training
    model.eval()
    result = []
    try:
        for start in range(0, len(z), size):
            batch = z[start : start + size]
            prefix = model.prefix(batch)
            bos = torch.full(
                (len(batch), 1), model.bos_id, dtype=torch.long, device=model.device
            )
            embeddings = torch.cat(
                (prefix, model.language_model.get_input_embeddings()(bos)), 1
            )
            mask = torch.ones(
                embeddings.shape[:2], dtype=torch.long, device=model.device
            )
            output = model.language_model(
                inputs_embeds=embeddings,
                attention_mask=mask,
                use_cache=True,
                logits_to_keep=1,
            )
            finished = torch.zeros(len(batch), dtype=torch.bool, device=model.device)
            sequences = [[] for _ in range(len(batch))]
            for _ in range(maximum):
                next_token = output.logits[:, -1].argmax(-1)
                for i, token in enumerate(next_token.tolist()):
                    if not finished[i]:
                        sequences[i].append(token)
                finished |= next_token.eq(model.eos_id)
                if finished.all() or max(map(len, sequences)) >= maximum:
                    break
                # A fixed batch shape preserves active rows; finished rows no longer contribute.
                next_token = next_token.masked_fill(finished, model.pad_id)
                mask = torch.cat((mask, (~finished).long()[:, None]), dim=1)
                position = torch.tensor([mask.shape[1] - 1], device=model.device)
                output = model.language_model(
                    input_ids=next_token[:, None],
                    attention_mask=mask,
                    past_key_values=output.past_key_values,
                    cache_position=position,
                    position_ids=position[None, :],
                    use_cache=True,
                    logits_to_keep=1,
                )
            for tokens in sequences:
                eos = bool(tokens and tokens[-1] == model.eos_id)
                result.append(
                    {
                        "text": model.tokenizer.decode(
                            tokens,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        ),
                        "token_ids": tokens,
                        "tokens": len(tokens),
                        "termination": "eos" if eos else "max_length",
                    }
                )
            if progress is not None:
                progress("generation", min(start + size, len(z)), len(z))
    finally:
        model.train(was_training)
    return result
