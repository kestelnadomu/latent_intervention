"""
Generic structured-state schema shared by the framework modules.

`src/` contains no knowledge of any specific experiment. The structured state S
(its columns and their cardinalities, in decoding order) and the SCM behind it
live in the data-generation half; the framework reads them at run time:

- the column schema is data in ``exp/sim/config.yaml`` (``schema:`` block), read
  by :func:`load_schema`;
- the concrete SCM and its closed-form counterfactual kernel are Python objects
  in ``exp/sim/``, referenced by dotted path in the same config (``objects:``
  block) and imported via :func:`load_object`.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

# Default location of the data-generation config that carries the schema and the
# dotted-path pointers. A repo-relative path, not experiment knowledge.
SIM_CONFIG_PATH = Path(__file__).resolve().parent.parent / "exp" / "sim" / "config.yaml"


@dataclass(frozen=True)
class ColumnSpec:
    """One structured-state column: name and number of discrete categories (values 0..n-1)."""

    name: str
    n_categories: int


def _as_config(sim_config: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    """Accept an already-loaded config dict or a path (default: SIM_CONFIG_PATH)."""
    if isinstance(sim_config, dict):
        return sim_config
    path = Path(sim_config) if sim_config is not None else SIM_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_schema(
    sim_config: dict[str, Any] | str | Path | None = None,
) -> tuple[list[ColumnSpec], ColumnSpec]:
    """
    Read the structured-state schema from the ``schema:`` block of the config.

    Returns ``(columns, outcome)`` where ``columns`` is the ordered list of S
    columns (the autoregressive decoding order) and ``outcome`` is the single
    downstream outcome column Y (simulated but excluded from the decoder heads
    and the consistency loss).
    """
    config = _as_config(sim_config)
    try:
        schema = config["schema"]
        columns = [ColumnSpec(str(name), int(card)) for name, card in schema["columns"].items()]
        (out_name, out_card), = schema["outcome"].items()
    except (KeyError, ValueError) as e:
        raise ValueError(f"Malformed 'schema:' block in the sim config: {e}") from e
    if not columns:
        raise ValueError("schema.columns is empty")
    return columns, ColumnSpec(str(out_name), int(out_card))


def load_intervention(
    sim_config: dict[str, Any] | str | Path | None = None,
) -> dict[str, int]:
    """Read the do() spec the counterfactual data was generated under (``intervention:``)."""
    config = _as_config(sim_config)
    if "intervention" not in config:
        raise ValueError("No 'intervention:' block in the sim config.")
    return {str(k): int(v) for k, v in config["intervention"].items()}


def load_object(dotted: str) -> Any:
    """
    Import ``"package.module:attribute"`` and return the attribute.

    Used to resolve the ``objects:`` pointers in the sim config (the concrete SCM
    and symbolic-kernel builders defined in the data-generation half).
    """
    if ":" not in dotted:
        raise ValueError(f"Expected 'module:attribute', got {dotted!r}")
    module_name, attr = dotted.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise ValueError(f"{module_name!r} has no attribute {attr!r}") from e


def load_config_object(
    key: str,
    sim_config: dict[str, Any] | str | Path | None = None,
) -> Any:
    """Resolve one dotted-path pointer from the ``objects:`` block of the config."""
    config = _as_config(sim_config)
    try:
        dotted = config["objects"][key]
    except KeyError as e:
        raise ValueError(f"No objects.{key} pointer in the sim config.") from e
    return load_object(dotted)


# --- flat state <-> per-column indices -------------------------------------------------
# Column-major over ``columns``: idx = ((s_0 * k_1) + s_1) * k_2 + ...
# Matches SemanticDecoder.log_joint and the symbolic kernel's state_index order.


def flat_state_index(values: torch.Tensor, columns: list[ColumnSpec]) -> torch.Tensor:
    """(..., n_cols) category indices -> (...,) flat state index."""
    idx = torch.zeros(values.shape[:-1], dtype=torch.long, device=values.device)
    for i, col in enumerate(columns):
        idx = idx * col.n_categories + values[..., i]
    return idx


def unflatten_state_index(flat: torch.Tensor, columns: list[ColumnSpec]) -> torch.Tensor:
    """(...,) flat state index -> (..., n_cols) category indices."""
    rem = flat.clone()
    cols: list[torch.Tensor] = []
    for col in reversed(columns):
        cols.append(rem % col.n_categories)
        rem = torch.div(rem, col.n_categories, rounding_mode="floor")
    return torch.stack(list(reversed(cols)), dim=-1)


def n_states(columns: list[ColumnSpec]) -> int:
    """Size of the joint structured-state space |S| = prod(cardinalities)."""
    total = 1
    for col in columns:
        total *= col.n_categories
    return total