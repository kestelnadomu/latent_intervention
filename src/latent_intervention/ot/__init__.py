"""Independent, factual-only optimal-transport baselines (lazy public facade)."""

OT_VARIANTS = ("ot_gaussian", "ot_flow")
CHECKPOINT_FORMAT = "latent_ot_v1"


def make_model(variant, *, latent_dim, columns, settings=None):
    from .capacity import architecture
    from .flow import ReferenceOTFlow
    from .gaussian import GaussianOT

    classes = dict(ot_gaussian=GaussianOT, ot_flow=ReferenceOTFlow)
    if variant not in classes:
        raise ValueError(f"unknown OT variant: {variant}")
    options = dict(settings or {})
    if variant != "ot_gaussian":
        options = architecture(variant, latent_dim, columns, options)
    return classes[variant](latent_dim, columns, **options)


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
        raise ValueError("not an OT checkpoint")
    variant = payload.get("variant")
    if variant == "ot_editor":
        raise ValueError(
            "retired distributional OT editor checkpoint; incompatible with semantic_editor_v1"
        )
    if variant not in OT_VARIANTS or (
        expected_variant is not None and variant != expected_variant
    ):
        raise ValueError("OT checkpoint variant mismatch")
    config = dict(payload["config"])
    columns = [ColumnSpec(*c) for c in config.pop("columns")]
    if expected_columns is not None and list(expected_columns) != columns:
        raise ValueError("OT checkpoint schema mismatch")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or not metadata_matches(
        metadata, expected_metadata or {}
    ):
        raise ValueError("OT checkpoint provenance mismatch")
    model = make_model(
        variant, latent_dim=config.pop("latent_dim"), columns=columns, settings=config
    )
    if payload.get("supported_interventions") != model.supported_interventions:
        raise ValueError("OT checkpoint intervention contract mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.validate_buffers()
    model.checkpoint_metadata = metadata
    return model.to(device or "cpu").eval()
