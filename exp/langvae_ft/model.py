"""Correct training of existing LangVAE tensors; no new model parameters."""

import hashlib

import torch
from torch import nn
from torch.nn import functional as F


def parameter_hash(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_native(encoder_config, device):
    from src.encoder import make_encoder

    model = make_encoder({**encoder_config, "device": "cpu"}, variant="langvae").model
    if (
        model.model_config.latent_dim != 128
        or model.decoder.max_len != 32
        or model.decoder.conditional
        or model.encoder.linear.bias is not None
        or tuple(model.encoder.linear.weight.shape) != (256, 768)
    ):
        raise ValueError("expected unchanged BERT/GPT-2 LangVAE l128, length 32")
    model.to(device)
    # Upstream stores these transformers in Python lists, outside registration.
    model.encoder.to(device)
    model.decoder.to(device)
    model.encoder.encoder.requires_grad_(False).eval()
    model.decoder.decoder.requires_grad_(False).eval()
    model.encoder.linear.requires_grad_(True)
    model.decoder.context_hidden.requires_grad_(True)
    model.encoder.caching = False
    return model, NativeAdapter(model).to(device)


class NativeAdapter(nn.Module):
    """Training wrapper sharing only the original projection/adapters/LM.

    The stock rolling positional-KV convention is preserved in both teacher
    forcing and generation. Registered shapes and exported architecture do not
    change. Unlike an auxiliary adapter architecture, this wrapper adds no W.
    """

    def __init__(self, model):
        super().__init__()
        self.projection = model.encoder.linear
        self.adapters = model.decoder.context_hidden
        self.language_model = model.decoder.decoder.requires_grad_(False).eval()
        self.dropout = model.decoder.dropout
        self.max_length = model.decoder.max_len
        self.latent_dim = model.model_config.latent_dim
        self.bos_id = model.decoder.tokenizer.bos_token_id
        self.eos_id = model.decoder.tokenizer.eos_token_id
        self.pad_id = model.decoder.tokenizer.pad_token_id
        self.heads, slots, self.head_width = model.decoder.pkv_dims
        if slots != 1 or model.decoder.conditional:
            raise ValueError("expected native unconditional one-slot KV injection")

    def train(self, mode=True):
        super().train(mode)
        self.language_model.eval()
        return self

    def posterior(self, pooled):
        return self.projection(pooled.float()).chunk(2, dim=-1)

    def _memory(self, z):
        shape = (len(z), self.heads, self.max_length + 1, self.head_width)
        return [
            tuple(t.reshape(shape) for t in self.dropout(layer(z)).chunk(2, -1))
            for layer in self.adapters
        ]

    def _step(self, tokens, past, memory, position):
        output = self.language_model(
            input_ids=tokens, past_key_values=past, use_cache=True
        )
        past = tuple(
            tuple(
                torch.cat(
                    (
                        cached[:, :, :-2],
                        latent[:, :, position + 1 : position + 2],
                        cached[:, :, -1:],
                    ),
                    dim=2,
                )
                for cached, latent in zip(cache, context, strict=True)
            )
            for cache, context in zip(output.past_key_values, memory, strict=True)
        )
        return output.logits[:, -1], past

    def teacher_logits(self, z, labels):
        if not 1 <= labels.shape[1] <= self.max_length:
            raise ValueError("target exceeds unchanged decoder length")
        memory = self._memory(z)
        past = tuple(tuple(t[:, :, :1] for t in layer) for layer in memory)
        previous = labels.new_full((len(z), 1), self.bos_id)
        logits = []
        for position in range(labels.shape[1]):
            predicted, past = self._step(previous, past, memory, position)
            logits.append(predicted)
            previous = labels[:, position : position + 1].masked_fill(
                labels[:, position : position + 1].eq(-100), self.pad_id
            )
        return torch.stack(logits, 1)

    @torch.no_grad()
    def generate(self, z):
        if self.training:
            raise RuntimeError("generation requires eval()")
        memory = self._memory(z)
        past = tuple(tuple(t[:, :, :1] for t in layer) for layer in memory)
        previous = torch.full(
            (len(z), 1), self.bos_id, dtype=torch.long, device=z.device
        )
        result = previous.new_full((len(z), self.max_length), self.pad_id)
        finished = torch.zeros(len(z), dtype=torch.bool, device=z.device)
        for position in range(self.max_length):
            logits, past = self._step(previous, past, memory, position)
            token = logits.argmax(-1).masked_fill(finished, self.pad_id)
            result[:, position] = token
            finished |= token.eq(self.eos_id)
            previous = token[:, None]
            if finished.all():
                break
        return result


def loss_terms(logits, labels, mu, logvar):
    """Token-sum categorical reconstruction and Gaussian KL, per passage.

    Padding alone is -100; an actual EOS is a supervised token, even when its
    vocabulary ID equals the padding ID. No free-bits/auxiliary losses.
    """
    valid = labels.ne(-100)
    if not valid.any(1).all():
        raise ValueError("empty reconstruction target")
    nll = F.cross_entropy(
        logits.float().transpose(1, 2), labels, ignore_index=-100, reduction="none"
    ).sum(1)
    kl = -0.5 * (1 + logvar.float() - mu.float().square() - logvar.float().exp()).sum(1)
    if not torch.isfinite(nll).all() or not torch.isfinite(kl).all():
        raise FloatingPointError("nonfinite reconstruction/KL")
    return nll, kl, valid.sum(1)


def batch_tensors(features, records, indices, device):
    sequences = [records[i]["target_ids"] for i in indices]
    labels = torch.full(
        (len(indices), max(map(len, sequences))), -100, dtype=torch.long, device=device
    )
    for i, sequence in enumerate(sequences):
        labels[i, : len(sequence)] = torch.tensor(sequence, device=device)
    return features[indices].to(device), labels


def train_update(adapter, optimizer, batches, beta, clip=1.0):
    adapter.train()
    optimizer.zero_grad(set_to_none=True)
    count = sum(len(pooled) for pooled, _ in batches)
    totals = {"loss": 0.0, "reconstruction_nll": 0.0, "kl": 0.0}
    for pooled, labels in batches:
        mu, logvar = adapter.posterior(pooled)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        nll, kl, _ = loss_terms(adapter.teacher_logits(z, labels), labels, mu, logvar)
        loss = (nll + beta * kl).mean()
        (loss * len(pooled) / count).backward()
        for key, value in (
            ("loss", loss),
            ("reconstruction_nll", nll.mean()),
            ("kl", kl.mean()),
        ):
            totals[key] += float(value.detach()) * len(pooled) / count
    trainable = [p for p in adapter.parameters() if p.requires_grad]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):
        raise FloatingPointError("missing/nonfinite trainable gradient")
    for module in (adapter.projection, adapter.adapters):
        if not any(p.grad.abs().sum() > 0 for p in module.parameters()):
            raise FloatingPointError("encoder/decoder component has no gradient")
    if any(p.grad is not None for p in adapter.language_model.parameters()):
        raise RuntimeError("frozen GPT-2 received gradients")
    totals["gradient_norm"] = float(
        nn.utils.clip_grad_norm_(trainable, clip, error_if_nonfinite=True)
    )
    optimizer.step()
    totals["beta"] = beta
    return totals


@torch.no_grad()
def pooled_features(base, records, batch_size, device):
    """Frozen BERT only, never cache the trainable 128-D projection."""
    base.encoder.encoder.eval()
    result = []
    for start in range(0, len(records), batch_size):
        tokens = [row["input_ids"] for row in records[start : start + batch_size]]
        texts = base.decoder.tokenizer.batch_decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        lengths = base.encoder.tokenizer(texts, truncation=False)["input_ids"]
        if max(map(len, lengths)) > base.encoder.encoder.config.max_position_embeddings:
            raise ValueError("BERT input would be truncated")
        result.append(base.encoder.recode(tokens).detach().float().cpu())
        if start % (batch_size * 100) == 0:
            print(
                f"BERT passage features: {min(start + batch_size, len(records))}/{len(records)}",
                flush=True,
            )
    features = torch.cat(result)
    if not torch.isfinite(features).all():
        raise FloatingPointError("nonfinite frozen features")
    return features
