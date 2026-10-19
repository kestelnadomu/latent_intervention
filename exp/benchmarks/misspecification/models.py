"""New experiment checkpoints; native model and historical loader behavior unchanged."""

import torch

from src.artifact_io import save_torch
from src.schema import ColumnSpec
from src.latent_intervention.editor import make_model as make_editor
from src.latent_intervention.editor.capacity import capacity_report as editor_capacity
from src.latent_intervention.ot import make_model as make_ot
from src.latent_intervention.ot.capacity import capacity_report as ot_capacity
from src.latent_intervention.flow.state_conditional import StateConditionalFlow
from . import ACTIONS, EDITORS, FORMAT, METHODS


def construct(
    method, columns, *, dimension=768, seed=42, config=None, architecture=None
):
    cfg = config or {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if architecture is not None:
            options = dict(architecture)
            options.pop("latent_dim", None)
            options.pop("columns", None)
        elif method in EDITORS:
            options = (
                {"tier": 1}
                if dimension in (256, 768)
                else (
                    dict(d_model=8, nhead=2, dim_feedforward=16)
                    if method == "semantic_editor"
                    else dict(hidden_dim=8)
                )
            )
        elif method == "state_flow":
            options = dict(
                n_blocks=8,
                hidden_dim=32,
                condition_dim=64,
                state_embed_dim=16,
                permutation_seed=seed,
                scale_floor=1e-6,
            )
            if dimension not in (256, 768):
                options.update(
                    n_blocks=2, hidden_dim=8, condition_dim=8, state_embed_dim=4
                )
        elif method == "ot_flow":
            options = dict(cfg.get("ot_flow", {}))
            options.update(tier=1) if dimension in (256, 768) else options.update(
                hidden_dim=8
            )
        elif method == "ot_gaussian":
            options = dict(cfg.get("gaussian", {}))
        else:
            raise ValueError(f"unsupported method {method}")
        if method in EDITORS:
            return make_editor(
                method, latent_dim=dimension, columns=columns, settings=options
            )
        if method in ("ot_flow", "ot_gaussian"):
            return make_ot(
                method, latent_dim=dimension, columns=columns, settings=options
            )
        return StateConditionalFlow(dimension, columns, **options)


def capacity(model):
    enforce = model.latent_dim in (256, 768)
    if model.variant in EDITORS:
        return editor_capacity(model, enforce=enforce)
    if model.variant.startswith("ot_"):
        return ot_capacity(model, enforce=enforce)
    count = sum(p.numel() for p in model.parameters())
    difference = abs(count - 331136) / 331136
    if enforce and difference > 0.07:
        raise ValueError("state-flow parameter budget mismatch")
    return dict(
        inference_active=count,
        training_total=count,
        training_only=0,
        target=331136,
        relative_difference=difference,
        buffer_bytes=sum(v.numel() * v.element_size() for v in model.buffers()),
    )


def architecture(model):
    return model.constructor() if hasattr(model, "constructor") else model._config


def validate(model):
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise FloatingPointError("non-finite model state")
    if hasattr(model, "validate_buffers"):
        model.validate_buffers()
    elif (model.latent_scale <= 0).any():
        raise ValueError("invalid flow normalization")


def save(path, model, metadata):
    validate(model)
    save_torch(
        path,
        dict(
            format=FORMAT,
            method=model.variant,
            architecture=architecture(model),
            state_dict=model.state_dict(),
            supported_actions=[{}, *ACTIONS],
            metadata=metadata,
        ),
    )


def load(path, *, signature=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("format") != FORMAT
        or payload.get("method") not in METHODS
        or payload.get("supported_actions") != [{}, *ACTIONS]
    ):
        raise ValueError("not a multi-action misspecification checkpoint")
    metadata = payload["metadata"]
    if signature is not None and metadata.get("signature") != signature:
        raise ValueError("checkpoint signature mismatch")
    config = payload["architecture"]
    model = construct(
        payload["method"],
        [ColumnSpec(*c) for c in config["columns"]],
        dimension=config["latent_dim"],
        architecture=config,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.checkpoint_metadata = metadata
    validate(model)
    # The experiment sampler owns the all-X-actions contract. Native legacy
    # sample_symbolic/load methods are deliberately not widened by this wrapper.
    return model.eval()
