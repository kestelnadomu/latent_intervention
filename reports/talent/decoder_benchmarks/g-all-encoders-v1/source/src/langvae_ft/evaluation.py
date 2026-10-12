"""Fixed-objective validation and passage-only latent-use diagnostics."""

import math
from collections import Counter

import torch

from .data import audit_indices, wrong_cv_indices
from .model import batch_tensors, loss_terms


@torch.no_grad()
def evaluate(adapter, features, records, *, batch_size, device, seed, samples=1):
    """Fixed-seed Monte Carlo negative ELBO, always beta=1.

    Posterior-mean NLL is reported separately: it is NOT the ELBO. A private
    generator prevents validation from advancing the training random stream.
    """
    adapter.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    totals = {
        "negative_elbo": 0.0,
        "sampled_reconstruction_nll": 0.0,
        "kl": 0.0,
        "mean_nll_sum": 0.0,
    }
    token_count = 0
    means = []
    for start in range(0, len(records), batch_size):
        indices = list(range(start, min(start + batch_size, len(records))))
        pooled, labels = batch_tensors(features, records, indices, device)
        mu, logvar = adapter.posterior(pooled)
        means.append(mu.cpu())
        mean_nll, kl, tokens = loss_terms(
            adapter.teacher_logits(mu, labels), labels, mu, logvar
        )
        sampled_nll = torch.zeros_like(mean_nll)
        for _ in range(samples):
            noise = torch.randn(mu.shape, device=device, generator=generator)
            nll, _, _ = loss_terms(
                adapter.teacher_logits(mu + (logvar / 2).exp() * noise, labels),
                labels,
                mu,
                logvar,
            )
            sampled_nll += nll / samples
        totals["negative_elbo"] += float((sampled_nll + kl).sum())
        totals["sampled_reconstruction_nll"] += float(sampled_nll.sum())
        totals["kl"] += float(kl.sum())
        totals["mean_nll_sum"] += float(mean_nll.sum())
        token_count += int(tokens.sum())
    result = {
        key: value / len(records)
        for key, value in totals.items()
        if key != "mean_nll_sum"
    }
    result.update(
        mean_token_nll=totals["mean_nll_sum"] / token_count,
        active_mean_dimensions=int(
            (torch.cat(means).var(0, unbiased=False) > 0.01).sum()
        ),
        passages=len(records),
        tokens=token_count,
        validation_samples=samples,
    )
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("nonfinite validation metric")
    return result


def overlap(reference, prediction):
    ref, pred = Counter(reference.lower().split()), Counter(prediction.lower().split())
    return (
        2 * sum((ref & pred).values()) / max(1, sum(ref.values()) + sum(pred.values()))
    )


@torch.no_grad()
def audit(adapter, features, records, tokenizer, *, count, batch_size, device, seed):
    """Correct/wrong-CV/zero controls; never used as auxiliary training losses."""
    adapter.eval()
    indices = audit_indices(records, count, seed)
    negatives = wrong_cv_indices(records)
    rows = []
    nll_sums = dict.fromkeys(("correct", "wrong_cv", "zero"), 0.0)
    tokens = 0
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        pooled, labels = batch_tensors(features, records, batch, device)
        mu, logvar = adapter.posterior(pooled)
        wrong_mu, _ = adapter.posterior(
            features[[negatives[i] for i in batch]].to(device)
        )
        conditions = {"correct": mu, "wrong_cv": wrong_mu, "zero": torch.zeros_like(mu)}
        texts = {}
        for name, latent in conditions.items():
            nll, _, valid = loss_terms(
                adapter.teacher_logits(latent, labels), labels, mu, logvar
            )
            nll_sums[name] += float(nll.sum())
            if name == "correct":
                tokens += int(valid.sum())
            texts[name] = tokenizer.batch_decode(
                adapter.generate(latent).cpu(), skip_special_tokens=True
            )
        references = tokenizer.batch_decode(
            [records[i]["input_ids"] for i in batch], skip_special_tokens=True
        )
        for offset, index in enumerate(batch):
            rows.append(
                {
                    "id": records[index]["id"],
                    "passage": records[index]["passage"],
                    "wrong_cv_id": records[negatives[index]]["id"],
                    "reference": references[offset],
                    **{name: values[offset] for name, values in texts.items()},
                }
            )
    metrics = {f"{key}_token_nll": value / tokens for key, value in nll_sums.items()}
    metrics.update(
        {
            f"{key}_word_f1": sum(overlap(row["reference"], row[key]) for row in rows)
            / len(rows)
            for key in nll_sums
        }
    )
    metrics["wrong_cv_nll_gap"] = (
        metrics["wrong_cv_token_nll"] - metrics["correct_token_nll"]
    )
    metrics["zero_nll_gap"] = metrics["zero_token_nll"] - metrics["correct_token_nll"]
    metrics["audited_cvs"] = len(rows)
    metrics["scope"] = (
        "short passage reconstruction, not full-CV generation or fairness"
    )
    return metrics, rows
