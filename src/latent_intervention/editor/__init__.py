"""Proxy-free semantic editors; model-specific imports remain lazy."""

EDITOR_VARIANTS = ("semantic_editor", "ot_editor")
CHECKPOINT_FORMAT = "semantic_editor_v1"


def make_model(variant, *, latent_dim, columns, settings=None):
    from .capacity import architecture
    from .models import SemanticEditor, OTStructuredEditor

    classes = dict(semantic_editor=SemanticEditor, ot_editor=OTStructuredEditor)
    if variant not in classes:
        raise ValueError(f"unknown semantic editor: {variant}")
    return classes[variant](
        latent_dim,
        columns,
        **architecture(variant, latent_dim, columns, settings or {}),
    )


def load_model(
    path,
    *,
    device=None,
    expected_variant=None,
    expected_columns=None,
    expected_metadata=None,
):
    import torch
    from src.schema import ColumnSpec
    from .common import metadata_matches

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            "not a clean semantic-editor checkpoint; historical editors are incompatible"
        )
    variant = payload.get("variant")
    if variant not in EDITOR_VARIANTS or (
        expected_variant is not None and variant != expected_variant
    ):
        raise ValueError("semantic editor variant mismatch")
    config = dict(payload["config"])
    columns = [ColumnSpec(*value) for value in config.pop("columns")]
    if expected_columns is not None and columns != list(expected_columns):
        raise ValueError("semantic editor schema mismatch")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or not metadata_matches(
        metadata, expected_metadata or {}
    ):
        raise ValueError("semantic editor provenance mismatch")
    with torch.random.fork_rng(devices=[]):
        model = make_model(
            variant,
            latent_dim=config.pop("latent_dim"),
            columns=columns,
            settings=config,
        )
    if payload.get("supported_interventions") != model.supported_interventions:
        raise ValueError("semantic editor intervention mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.validate_buffers()
    model.checkpoint_metadata = metadata
    return model.to(device or "cpu").eval()
