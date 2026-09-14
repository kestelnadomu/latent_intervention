from __future__ import annotations

import torch

from exp.sim.cv_screening import build_scm, build_symbolic_kernel
from src import symbolic_intervention


def _config() -> dict:
    return {
        "schema": {
            "columns": {
                "R": 5,
                "G": 2,
                "A": 3,
                "E": 4,
                "S": 3,
                "W": 3,
                "V": 2,
                "C": 2,
            },
            "outcome": {"Q": 4},
        }
    }


def test_cv_builders_use_the_supplied_schema() -> None:
    scm = build_scm(_config())
    kernel = build_symbolic_kernel(_config())

    assert [(node.name, node.n_categories) for node in scm.nodes][:2] == [
        ("R", 5),
        ("G", 2),
    ]
    assert (scm.nodes[-1].name, scm.nodes[-1].n_categories) == ("Q", 4)
    assert [(column.name, column.n_categories) for column in kernel.columns][:2] == [
        ("R", 5),
        ("G", 2),
    ]


def test_symbolic_loader_passes_the_supplied_config(monkeypatch) -> None:
    config = _config()
    received = []

    class StubKernel:
        columns = []

        def state_index(self, state: dict[str, int]) -> int:
            return 0

        def transition_matrix(self, delta: dict[str, int]) -> torch.Tensor:
            return torch.eye(1).to_sparse()

        def compose(self, g_probs: torch.Tensor, delta: dict[str, int]) -> torch.Tensor:
            return g_probs

    def builder(sim_config):
        received.append(sim_config)
        return StubKernel()

    monkeypatch.setattr(
        symbolic_intervention,
        "load_config_object",
        lambda key, sim_config: builder,
    )

    symbolic_intervention.load_symbolic_kernel(config)

    assert received == [config]
