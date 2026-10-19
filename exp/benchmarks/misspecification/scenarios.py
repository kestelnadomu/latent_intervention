"""Only the predeclared kernels: no simulation, refitting or operator sweep."""

from dataclasses import dataclass
from functools import cached_property

import torch

from exp.benchmarks.reader.protocol import STATE_SPACES
from exp.sim import assumed
from exp.sim.helpers import load_sim_config
from exp.sim.symbolic import SymbolicIntervention
from exp.sim_talent_motivation import talent_motivation as tm
from src.schema import ColumnSpec
from . import ACTIONS, EDITORS, METHODS
from .common import digest, tensor_digest


def dense(kernel, action):
    value = kernel.transition_matrix(action)
    if value.is_sparse:
        value = value.to_dense()
    value = value.detach().cpu().float()
    if (
        value.ndim != 2
        or value.shape[0] != value.shape[1]
        or not torch.isfinite(value).all()
        or (value < 0).any()
        or not torch.allclose(value.sum(-1), torch.ones(len(value)), atol=1e-5)
    ):
        raise ValueError("invalid symbolic transition matrix")
    return value


@dataclass
class Scenario:
    name: str
    space: str
    kernel: object
    targets: tuple

    @cached_property
    def matrices(self):
        return tuple(dense(self.kernel, action) for action in ACTIONS)

    @cached_property
    def fingerprint(self):
        return digest(
            dict(
                schema=[(c.name, c.n_categories) for c in self.kernel.columns],
                matrices=[tensor_digest(m) for m in self.matrices],
            )
        )


def build(sim_config="exp/sim_talent_motivation/config.yaml"):
    config = load_sim_config(sim_config)
    full = tm.build_symbolic_kernel(config)
    with_m = SymbolicIntervention.from_scm(
        [*full.columns, ColumnSpec("M", 3)],
        tm.base.MECHANISM_COEFFS,
        tm.base.NOISE_PARAMS,
    )
    targets = {
        "S1": ("total", "ps_D", "ps_DU"),
        "S2": ("total",),
        "S3": ("total", "ps_DU"),
        "S4": ("total", "ps_D"),
        "S5": ("total", "ps_D"),
        "S6": ("total",),
    }
    cases = [
        Scenario(
            f"{s}/{'true' if s in ('S1', 'S6') else 'unaware'}",
            s,
            with_m
            if s == "S6"
            else full
            if s == "S1"
            else assumed.restrict(full, names),
            targets[s],
        )
        for s, names in STATE_SPACES.items()
    ]
    cases += [
        Scenario(
            "O1/unaware",
            "S1",
            assumed.drop_parents(full, {"D": ["T"], "U": ["T"]}),
            ("total",),
        ),
        Scenario(
            "O2/coef_X_x0.5", "S1", assumed.scale(full, 0.5, parent="X"), ("total",)
        ),
        Scenario(
            "O2/noise_sigma_x0.5",
            "S1",
            assumed.scale(full, 0.5, noise="sigma"),
            ("total",),
        ),
        Scenario(
            "O3/drop_X_D",
            "S1",
            assumed.drop_parents(full, {"D": ["X"]}),
            ("total", "ps_D"),
        ),
        Scenario(
            "O3/drop_X_U", "S1", assumed.drop_parents(full, {"U": ["X"]}), ("total",)
        ),
        Scenario(
            "O4/block_X_D", "S1", assumed.block(full, {"D": ["X"]}), ("total", "ps_D")
        ),
    ]
    for relative in (0, 0.5, 1):
        coefficient = relative * tm.base.MECHANISM_COEFFS["D"]["X"]
        noise = assumed.matched_root_noise(tm.M_PRIOR, 1.5, coefficient)
        cases.append(
            Scenario(
                f"O5/c{relative}",
                "S6",
                assumed.add_mechanism(with_m, "M", {"X": coefficient}, noise),
                ("total",),
            )
        )
    return cases


def registry(cases):
    canonical, rows = {}, []
    for case in cases:
        fingerprint = case.fingerprint
        owner = canonical.setdefault((case.space, fingerprint), case.name)
        rows.append(
            dict(
                name=case.name,
                space=case.space,
                targets=list(case.targets),
                fingerprint=fingerprint,
                canonical=owner,
                schema=[(c.name, c.n_categories) for c in case.kernel.columns],
                construction=(
                    "restricted parameter-preserving unaware kernel; no marginalization/refit"
                    if case.space in ("S2", "S3", "S4", "S5")
                    else "matched-noise false X->M edge"
                    if case.name.startswith("O5/")
                    else "specified parameter-preserving kernel modification"
                    if case.name.startswith("O")
                    else "correct kernel"
                ),
            )
        )
    return rows


def jobs(cfg, rows):
    if cfg.get("is_pilot"):
        rows = [
            r
            for r in rows
            if r["space"] in cfg["pilot"]["spaces"]
            and r["name"].startswith(r["space"] + "/")
        ]
    result, seen = [], set()
    for row in rows:
        for method in METHODS:
            operator = row["canonical"] if method in EDITORS else "factual"
            for seed in [42] if method == "ot_gaussian" else cfg["seeds"]:
                key = (row["space"], method, operator, seed)
                if key not in seen:
                    result.append(
                        dict(space=key[0], method=method, operator=operator, seed=seed)
                    )
                    seen.add(key)
    return result
