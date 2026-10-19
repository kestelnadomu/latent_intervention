"""Match actual flow parameters and report analytical Gaussian storage."""

TARGETS = {256: {1: 199552, 2: 660992}, 768: {1: 331136, 2: 923648}}


def parameter_count(variant, dimension, columns, width, embedding):
    """Closed form only selects a width; capacity_report verifies real tensors."""
    embedded = len(columns) * embedding
    lookup = sum(c.n_categories for c in columns) * embedding
    return width * width + (2 * dimension + embedded + 18) * width + dimension + lookup


def architecture(variant, dimension, columns, settings):
    options = dict(settings)
    tier = options.pop("tier", 1)
    if tier not in (1, 2):
        raise ValueError("OT capacity tier must be 1 or 2")
    if "hidden_dim" not in options:
        if dimension not in TARGETS:
            raise ValueError("non-benchmark dimensions require explicit hidden_dim")
        target = TARGETS[dimension][tier]
        embedding = options.get("state_embed_dim", 16)
        options["hidden_dim"] = min(
            range(1, 4097),
            key=lambda h: (
                abs(
                    parameter_count(variant, dimension, columns, h, embedding) - target
                ),
                h,
            ),
        )
    return options


def capacity_report(model, *, tier=1, enforce=True):
    inference = sum(p.numel() for p in model.parameters() if p.requires_grad)
    training_only = 0
    total = inference + training_only
    result = dict(
        inference_active=inference,
        training_only=training_only,
        training_total=total,
        frozen_auxiliary_in_model=0,
        buffer_values=sum(b.numel() for b in model.buffers()),
        buffer_bytes=sum(b.numel() * b.element_size() for b in model.buffers()),
    )
    if model.variant == "ot_gaussian":
        result.update(
            statistical_mean_values=model.state_mean.numel(),
            statistical_variance_values=model.state_variance.numel(),
            basis_values=model.basis.numel(),
            capacity_matched=False,
        )
    else:
        target = TARGETS.get(model.latent_dim, {}).get(tier)
        result.update(
            tier=tier,
            target=target,
            hidden_dim=model.hidden_dim,
            relative_difference=None
            if target is None
            else abs(total - target) / target,
        )
        if enforce and (target is None or abs(total - target) / target > 0.07):
            raise ValueError(f"OT parameter budget mismatch: {total} vs {target}")
        result["capacity_matched"] = (
            target is not None and abs(total - target) / target <= 0.07
        )
    return result
