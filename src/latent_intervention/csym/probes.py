"""Factual-only preservation guide and separately fitted audit readers."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.schema import ColumnSpec


class AttributeProbe(nn.Module):
    """Factual-only reader: fit-only standardization, tuning-only calibration."""

    def __init__(
        self,
        latent_dim: int,
        columns: Sequence[ColumnSpec],
        hidden: Sequence[int] = (),
        skip: bool = False,
    ):
        super().__init__()
        self.columns, self.hidden = list(columns), tuple(hidden)
        if type(skip) is not bool or (skip and not hidden):
            raise ValueError("a skip-connected auditor needs a nonlinear trunk")
        self.skip = skip
        if (
            latent_dim < 1
            or not self.columns
            or len({c.name for c in self.columns}) != len(self.columns)
            or any(c.n_categories < 2 for c in self.columns)
            or any(type(width) is not int or width < 1 for width in self.hidden)
        ):
            raise ValueError("invalid probe dimensions/columns")
        self.latent_dim = latent_dim
        self.register_buffer("mean", torch.zeros(latent_dim))
        self.register_buffer("scale", torch.ones(latent_dim))
        self.register_buffer("temperature", torch.ones(len(columns)))
        layers = []
        previous = latent_dim
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.GELU()])
            previous = width
        self.trunk = nn.Sequential(*layers)
        if self.skip:
            previous += latent_dim
        self.heads = nn.ModuleList(
            [nn.Linear(previous, c.n_categories) for c in columns]
        )

    def logits(self, z):
        normalized = (z - self.mean) / self.scale
        features = self.trunk(normalized)
        if self.skip:
            features = torch.cat([normalized, features], dim=-1)
        return [
            head(features) / self.temperature[i] for i, head in enumerate(self.heads)
        ]

    def log_probs(self, z):
        return [F.log_softmax(values, -1) for values in self.logits(z)]

    def joint(self, z, names):
        """Independent joint in the explicitly requested schema order."""
        logs = dict(zip([c.name for c in self.columns], self.log_probs(z), strict=True))
        result = z.new_zeros(len(z), 1)
        for name in names:
            result = (result[:, :, None] + logs[name][:, None, :]).flatten(1)
        return result

    def constructor(self):
        return dict(
            latent_dim=self.latent_dim,
            columns=[(c.name, c.n_categories) for c in self.columns],
            hidden=list(self.hidden),
            skip=self.skip,
        )


class IndependentAttributeProbe(nn.Module):
    """One separately fitted nonlinear reader per attribute; no shared-trunk transfer.

    Each reader may start from its own fitted linear model. The linear guide is
    never copied, and the selected epoch can be zero if nonlinear corrections do
    not improve tuning NLL. This is an audit model, not a preservation constraint.
    """

    def __init__(self, latent_dim, columns, hidden, skip=True):
        super().__init__()
        self.latent_dim, self.columns = latent_dim, list(columns)
        self.hidden, self.skip = tuple(hidden), skip
        if (
            not self.hidden
            or not self.columns
            or len({c.name for c in columns}) != len(columns)
        ):
            raise ValueError(
                "independent auditor requires nonlinear, uniquely named readers"
            )
        self.readers = nn.ModuleList(
            [
                AttributeProbe(latent_dim, [col], self.hidden, skip=skip)
                for col in columns
            ]
        )

    def log_probs(self, z):
        return [reader.log_probs(z)[0] for reader in self.readers]

    def joint(self, z, names):
        # Same marginal-product convention and column ordering as the shared reader.
        return AttributeProbe.joint(self, z, names)

    def constructor(self):
        return dict(
            latent_dim=self.latent_dim,
            columns=[(c.name, c.n_categories) for c in self.columns],
            hidden=list(self.hidden),
            skip=self.skip,
            independent_heads=True,
        )


@torch.no_grad()
def probe_metrics(probe, z, labels):
    result = {}
    for i, (col, lp) in enumerate(zip(probe.columns, probe.log_probs(z), strict=True)):
        truth = labels[:, i]
        predicted = lp.argmax(-1)
        recalls = [
            predicted[truth == k].eq(k).float().mean().item()
            for k in range(col.n_categories)
            if (truth == k).any()
        ]
        result[col.name] = dict(
            accuracy=predicted.eq(truth).float().mean().item(),
            balanced_accuracy=sum(recalls) / len(recalls),
            nll=F.nll_loss(lp, truth).item(),
        )
    return result


def fit_probe(
    z,
    y,
    val_z,
    val_y,
    columns,
    hidden,
    settings,
    seed,
    lr,
    *,
    skip=False,
    warm_start=False,
):
    baseline, baseline_history = None, None
    if warm_start:
        if not hidden or not skip:
            raise ValueError(
                "a linear warm start requires a nonlinear skip-connected reader"
            )
        baseline, baseline_history = fit_probe(
            z,
            y,
            val_z,
            val_y,
            columns,
            (),
            settings,
            seed + 10000,
            settings["guide_lr"],
        )
    torch.manual_seed(seed)
    probe = AttributeProbe(z.shape[-1], columns, hidden, skip=skip)
    probe.mean.copy_(z.mean(0))
    probe.scale.copy_(z.std(0).clamp_min(1e-4))
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    rng = torch.Generator().manual_seed(seed + 1)
    best, stale, best_state, history = float("inf"), 0, None, []
    if baseline is not None:
        with torch.no_grad():
            # The independently fitted linear predictor is epoch zero, not an
            # irreversible pretraining step. Never substitute guidance weights.
            probe.temperature.copy_(baseline.temperature)
            for head, source in zip(probe.heads, baseline.heads, strict=True):
                head.weight.zero_()
                head.weight[:, : z.shape[-1]].copy_(source.weight)
                head.bias.copy_(source.bias)
            best = sum(
                F.nll_loss(lp, val_y[:, i]).item()
                for i, lp in enumerate(probe.log_probs(val_z))
            ) / len(columns)
            initial_train = sum(
                F.nll_loss(lp, y[:, i]).item()
                for i, lp in enumerate(probe.log_probs(z))
            ) / len(columns)
        best_state = copy.deepcopy(probe.state_dict())
        history.append(
            dict(
                epoch=0,
                training_nll=initial_train,
                tuning_nll=best,
                selected=True,
                warm_start_history=baseline_history,
            )
        )
    for epoch in range(settings["max_epochs"]):
        probe.train()
        order = torch.randperm(len(z), generator=rng)
        total = 0.0
        for idx in order.split(settings["batch_size"]):
            logits = probe.logits(z[idx])
            loss = torch.stack(
                [F.cross_entropy(value, y[idx, j]) for j, value in enumerate(logits)]
            ).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite probe loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                probe.parameters(), 5, error_if_nonfinite=True
            )
            optimizer.step()
            total += loss.item() * len(idx)
        probe.eval()
        with torch.no_grad():
            val = (
                torch.stack(
                    [
                        F.nll_loss(lp, val_y[:, j])
                        for j, lp in enumerate(probe.log_probs(val_z))
                    ]
                )
                .mean()
                .item()
            )
        improved = val < best - 1e-5
        history.append(
            dict(
                epoch=epoch + 1,
                training_nll=total / len(z),
                tuning_nll=val,
                selected=improved,
            )
        )
        if improved:
            best, stale, best_state = val, 0, copy.deepcopy(probe.state_dict())
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    probe.load_state_dict(best_state)
    probe.eval()
    # Per-head temperature selected on tuning data only; retain genuine uncertainty.
    with torch.no_grad():
        logits = probe.logits(val_z)
        grid = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
        for i, values in enumerate(logits):
            chosen = min(
                grid, key=lambda t: F.cross_entropy(values / t, val_y[:, i]).item()
            )
            # Warm starts can already have calibrated temperatures.
            probe.temperature[i] *= chosen
    return probe.requires_grad_(False), history


def fit_independent_auditor(fit, tune, columns, settings, seed):
    probe = IndependentAttributeProbe(
        fit["z"].shape[-1],
        columns,
        settings["auditor_hidden"],
        skip=settings.get("auditor_skip", False),
    )
    histories = {}
    for i, column in enumerate(columns):
        reader, history = fit_probe(
            fit["z"],
            fit["labels"][:, i : i + 1],
            tune["z"],
            tune["labels"][:, i : i + 1],
            [column],
            settings["auditor_hidden"],
            settings,
            seed + i,
            settings["auditor_lr"],
            skip=settings.get("auditor_skip", False),
            warm_start=settings.get("auditor_warm_start", False),
        )
        probe.readers[i] = reader
        histories[column.name] = history
    return probe.eval().requires_grad_(False), histories


def pack_probe(probe):
    return dict(config=probe.constructor(), state_dict=probe.state_dict())


def unpack_probe(payload):
    config = dict(payload["config"])
    config["columns"] = [ColumnSpec(*c) for c in config["columns"]]
    cls = (
        IndependentAttributeProbe
        if config.pop("independent_heads", False)
        else AttributeProbe
    )
    probe = cls(**config)
    probe.load_state_dict(payload["state_dict"], strict=True)
    if any(not torch.isfinite(value).all() for value in probe.state_dict().values()):
        raise ValueError("nonfinite probe checkpoint")
    if any(
        (reader.scale <= 0).any() or (reader.temperature <= 0).any()
        for reader in (
            probe.readers if isinstance(probe, IndependentAttributeProbe) else [probe]
        )
    ):
        raise ValueError("invalid probe normalization")
    return probe.eval().requires_grad_(False)


def fit_readers(data, settings):
    """Fit a linear guide, independent per-attribute readers, and a stress reader."""
    fit, tune = data["fit"], data["tune"]
    cfg, seed = settings["probe"], settings["seed"]
    offset = len(data["columns"])
    guide, guide_history = fit_probe(
        fit["z"],
        fit["labels"][:, offset:],
        tune["z"],
        tune["labels"][:, offset:],
        data["protected"],
        (),
        cfg,
        seed + 101,
        cfg["guide_lr"],
    )
    columns = [*data["columns"], *data["protected"]]
    auditor, audit_history = fit_independent_auditor(
        fit, tune, columns, cfg, seed + 211
    )
    stress, stress_history = fit_probe(
        fit["z"],
        fit["labels"],
        tune["z"],
        tune["labels"],
        columns,
        cfg["auditor_hidden"],
        cfg,
        seed + 211,
        cfg["auditor_lr"],
        skip=cfg["auditor_skip"],
    )
    report = dict(
        guide_history=guide_history,
        auditor_history=audit_history,
        stress_history=stress_history,
        guide_tuning=probe_metrics(guide, tune["z"], tune["labels"][:, offset:]),
        auditor_tuning=probe_metrics(auditor, tune["z"], tune["labels"]),
        stress_tuning=probe_metrics(stress, tune["z"], tune["labels"]),
        auditor_selected_epochs={
            name: next(row["epoch"] for row in reversed(history) if row["selected"])
            for name, history in audit_history.items()
        },
    )
    report["auditor_linear_fallback_attributes"] = [
        name for name, epoch in report["auditor_selected_epochs"].items() if epoch == 0
    ]
    report["readers_qualified"] = all(
        report[reader][column]["balanced_accuracy"] >= 0.6
        for reader in ("guide_tuning", "auditor_tuning")
        for column in settings["protected_columns"]
    )
    return dict(guide=guide, auditor=auditor, stress_auditor=stress), report
