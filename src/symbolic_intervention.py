"""
Access to the symbolic counterfactual kernel h_S(s' | s, delta).

The framework does not implement h_S: the concrete kernel is a Python object in
the data-generation half, referenced by dotted path in ``exp/sim/config.yaml``
(``objects.symbolic_kernel``). This module defines the interface the framework
relies on and a loader that resolves the configured object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch

from src.schema import ColumnSpec, load_config_object


@runtime_checkable
class SymbolicKernel(Protocol):
    """
    Minimal interface for a symbolic counterfactual kernel h_S(s' | s, delta).

    ``columns`` is the structured-state schema in column-major flat-index order
    (matching ``SemanticDecoder.log_joint``).
    """

    columns: list[ColumnSpec]

    def state_index(self, state: dict[str, int]) -> int:
        """Flat column-major index of a structured state."""
        ...

    def transition_matrix(self, delta: dict[str, int]) -> torch.Tensor:
        """Sparse ``(|S|, |S|)`` tensor M with ``M[s, s'] = h_S(s' | s, delta)``; rows sum to 1."""
        ...

    def compose(self, g_probs: torch.Tensor, delta: dict[str, int]) -> torch.Tensor:
        """Pushforward ``(h_S . g)``: ``(batch, |S|)`` distributions through M_delta."""
        ...


def load_symbolic_kernel(
    sim_config: dict[str, Any] | str | Path | None = None,
) -> SymbolicKernel:
    """Instantiate the kernel pointed to by ``objects.symbolic_kernel`` in the sim config."""
    builder = load_config_object("symbolic_kernel", sim_config)
    kernel = builder() if sim_config is None else builder(sim_config)
    if not isinstance(kernel, SymbolicKernel):
        raise TypeError(f"{builder!r} did not return a SymbolicKernel-compatible object")
    return kernel
