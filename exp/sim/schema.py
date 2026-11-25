"""The structured-state schema and object pointers of a sim config.

- the column schema is data in the sim config (default ``exp/sim_talent/config.yaml``,
  ``schema:`` block), read by :func:`load_schema`;
- the concrete SCM and its closed-form symbolic kernel are Python objects in
  ``exp/``, referenced by dotted path in the same config (``objects:`` block) and
  imported via :func:`load_object`.

The data-generation half reads its configs through this module. ``src/schema.py`` is
the framework's own copy; the two ``ColumnSpec`` classes compare equal by value.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SIM_CONFIG_PATH = Path(__file__).resolve().parent.parent / "sim_talent" / "config.yaml"


@dataclass(frozen=True, eq=False)
class ColumnSpec:
    """One structured-state column: name and number of discrete categories (values 0..n-1)."""

    name: str
    n_categories: int

    # By value, so that src.schema.ColumnSpec("X", 4) == ColumnSpec("X", 4) in both directions
    # (src's dataclass __eq__ returns NotImplemented for this class, so Python asks this one).
    def __eq__(self, other: object) -> bool:
        if not (hasattr(other, "name") and hasattr(other, "n_categories")):
            return NotImplemented
        return (self.name, self.n_categories) == (other.name, other.n_categories)

    def __hash__(self) -> int:
        return hash((self.name, self.n_categories))  # = hash of src's frozen dataclass


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
    """``(columns, outcome)`` from the ``schema:`` block: the ordered S columns and the
    downstream outcome column Y (simulated, but not part of S)."""
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
    """The do() spec the paired data were generated under (``intervention:``)."""
    config = _as_config(sim_config)
    if "intervention" not in config:
        raise ValueError("No 'intervention:' block in the sim config.")
    return {str(k): int(v) for k, v in config["intervention"].items()}


def load_object(dotted: str) -> Any:
    """Import ``"package.module:attribute"`` and return the attribute."""
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
