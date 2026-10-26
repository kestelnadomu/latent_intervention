"""Matched deployed capacity; the OT editor has no second/training-only potential."""

TARGETS = {256: {1: 199552, 2: 660992}, 768: {1: 331136, 2: 923648}}


def potential_count(dimension, columns, width, embedding):
    context = len(columns) * embedding
    return (
        width * width
        + (2 * dimension + 2 * context + 3) * width
        + 2 * dimension
        + sum(c.n_categories for c in columns) * embedding
    )


def architecture(variant, dimension, columns, settings):
    options = dict(settings)
    tier = options.pop("tier", 1)
    if tier not in (1, 2):
        raise ValueError("editor tier must be 1 or 2")
    if variant == "semantic_editor":
        width = options.pop("d_model", 128 if tier == 1 else 256)
        return dict(
            d_model=width,
            nhead=options.pop("nhead", 4),
            dim_feedforward=options.pop("dim_feedforward", 2 * width),
            **options,
        )
    if "hidden_dim" not in options:
        if dimension not in TARGETS:
            raise ValueError("synthetic dimensions require explicit hidden_dim")
        embedding = options.get("state_embed_dim", 16)
        options["hidden_dim"] = min(
            range(1, 4097),
            key=lambda w: (
                abs(
                    potential_count(dimension, columns, w, embedding)
                    - TARGETS[dimension][tier]
                ),
                w,
            ),
        )
    return options


def capacity_report(model, *, tier=1, enforce=True):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    target = TARGETS.get(model.latent_dim, {}).get(tier)
    difference = None if target is None else abs(total - target) / target
    if enforce and (difference is None or difference > 0.07):
        raise ValueError(f"editor parameter budget mismatch: {total} vs {target}")
    return dict(
        inference_active=total,
        training_total=total,
        training_only=0,
        target=target,
        relative_difference=difference,
        tier=tier,
        buffer_values=sum(b.numel() for b in model.buffers()),
        buffer_bytes=sum(b.numel() * b.element_size() for b in model.buffers()),
    )
