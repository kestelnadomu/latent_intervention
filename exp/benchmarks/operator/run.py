"""Operator benchmark: assumed vs. true symbolic operator, on states alone.

    uv run python -m exp.benchmarks.operator.run --run-id operator-v1

Design: docs/experiments/operator_benchmark.md; protocol: configs/operator_benchmark.yaml.
No texts, encoders or billed calls. Every metric is computed exactly from transition
matrices and the exact factual state distribution, except the log score of realized
counterfactuals and the fitted conditions (O2a/b, refit variants, O5 refit), which use
simulated units. O2c (fitting on the reader's labels) needs texts and is not run here.

Writes ``reports/talent_motivation/operator_benchmarks/<run-id>/``: ``protocol.json``,
``results.json`` (one row per condition x target) and ``summary.md``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from exp.benchmarks.operator import metrics
from exp.sim import assumed
from exp.sim.design import counterfactual_states, draw_interventions, load_design, noise_from_epsilon
from exp.sim.helpers import load_sim_config
from exp.sim.symbolic import SymbolicIntervention
from exp.sim_talent_motivation import talent_motivation as tm
from src.schema import ColumnSpec

PROTOCOL = Path("configs/operator_benchmark.yaml")
REPORTS = Path("reports/talent_motivation/operator_benchmarks")
NODE = "X"
SPACES = {  # name -> state columns
    "S1": ["X", "T", "D", "U"],
    "S2": ["X", "D", "U"],
    "S3": ["X"],
    "S4": ["X", "T", "U"],
    "S5": ["X", "U"],
    "S6": ["X", "T", "D", "U", "M"],
}
TARGETS = {"total": {}, "ps_D": {"D": ["X"]}, "ps_DU": {"D": ["X"], "U": ["X"]}}
OUTCOME_GIVEN = {"S1": ["X", "T", "D", "U"], "S2": ["X", "D", "U"], "S6": ["X", "T", "D", "U", "M"]}


@dataclass
class Condition:
    name: str
    group: str
    space: str
    kernel: SymbolicIntervention
    targets: list[str]
    params: dict[str, Any] = field(default_factory=dict)


class World:
    """The true model: kernels, exact state distributions, truth matrices, scored units."""

    def __init__(self, protocol: dict[str, Any]):
        self.config = load_sim_config(protocol["sim_config"])
        self.seed = int(protocol["seed"])
        self.scm = tm.build_scm(self.config)
        self.full = tm.build_symbolic_kernel(self.config)
        self.with_m = SymbolicIntervention.from_scm(
            [*self.full.columns, ColumnSpec("M", 3)], tm.base.MECHANISM_COEFFS, tm.base.NOISE_PARAMS
        )
        self.priors = {NODE: [0.25] * 4, "T": list(tm.T_PRIOR), "M": list(tm.M_PRIOR)}
        self.card = {c.name: c.n_categories for c in tm._nodes(self.config)}
        self.n_values = self.card[NODE]
        self._truth: dict[tuple[str, str, int], np.ndarray] = {}
        self.spaces = {name: self._space(name) for name in SPACES}
        self.units = self._score_units(int(protocol["n_score"]))

    def kernel(self, space: str) -> SymbolicIntervention:
        if space == "S6":
            return self.with_m
        if space == "S1":
            return self.full
        return assumed.restrict(self.full, SPACES[space])

    def _space(self, name: str) -> dict[str, Any]:
        kernel = self.kernel(name)
        base = self.with_m if name == "S6" else self.full
        base_probs = assumed.state_distribution(base, self.priors)
        if name in ("S1", "S6"):
            probs, index = base_probs, None
        else:
            index = assumed.projection(self.full, kernel)
            probs = np.bincount(index, weights=base_probs, minlength=len(metrics.all_states(kernel)))
        states = metrics.all_states(kernel)
        outcome = None
        if name in OUTCOME_GIVEN:
            outcome = tm.outcome_posterior(states, OUTCOME_GIVEN[name], self.card)
        return {"kernel": kernel, "states": states, "probs": probs, "index": index,
                "base_probs": base_probs, "outcome": outcome}

    def truth(self, space: str, target: str, value: int) -> np.ndarray:
        key = (space, target, value)
        if key not in self._truth:
            delta = {NODE: value}
            if space == "S6":
                if target != "total":
                    raise ValueError("S6 is evaluated against the total target only")
                matrix = metrics.dense(self.with_m, delta)
            else:
                full = metrics.dense(assumed.block(self.full, TARGETS[target]) if TARGETS[target] else self.full, delta)
                if space == "S1":
                    matrix = full
                else:
                    info = self.spaces[space]
                    matrix, _ = assumed.project(full, info["base_probs"], info["index"], len(info["states"]))
            self._truth[key] = matrix
        return self._truth[key]

    def factual_sample(self, n: int, seed: int) -> pd.DataFrame:
        factual, _, _ = self.scm.simulate(n, {NODE: 0}, seed=seed)
        return factual

    def _score_units(self, n: int) -> dict[str, Any]:
        factual, _, epsilon = self.scm.simulate(n, {NODE: 0}, seed=self.seed + 1)
        spec = load_design(self.config)
        states = counterfactual_states(self.scm, noise_from_epsilon(epsilon), factual, spec, self.n_values)
        drawn = draw_interventions(factual, NODE, self.n_values, self.seed + 1)
        realized = {}
        for target in TARGETS:
            rows = states[states["target"] == target].merge(drawn, on=["id", "do_value"])
            realized[target] = rows.sort_values("id").reset_index(drop=True)
        return {"factual": factual, "drawn": drawn["do_value"].to_numpy(), "realized": realized}


def evaluate(world: World, condition: Condition, target: str) -> dict[str, Any]:
    info = world.spaces[condition.space]
    states, probs = info["states"], info["probs"]
    kernel = condition.kernel
    totals = {"tv": 0.0, "missing_mass": 0.0}
    shift = {c: {"directional": 0.0, "absolute": 0.0} for c in states.columns}
    down = {"y_tv": 0.0, "y_shift": 0.0, "y_abs": 0.0}
    tv_by_x = np.zeros(world.n_values)
    mass_by_x = np.zeros(world.n_values)
    units = world.units
    source_all = metrics.indices(units["factual"], kernel)
    target_all = metrics.indices(units["realized"][target], kernel)
    nll_a, nll_t, n_scored = 0.0, 0.0, 0
    for value in range(world.n_values):
        matrix_a = metrics.dense(kernel, {NODE: value})
        matrix_t = world.truth(condition.space, target, value)
        weights = metrics.intervention_weights(states, probs, NODE, value)
        direction = np.sign(value - states[NODE].to_numpy(float))
        result = metrics.compare(matrix_a, matrix_t, weights)
        totals["tv"] += result["tv"] / world.n_values
        totals["missing_mass"] += result["missing_mass"] / world.n_values
        for column, errors in metrics.mean_shift(matrix_a, matrix_t, weights, states, direction).items():
            for key, number in errors.items():
                shift[column][key] += number / world.n_values
        if info["outcome"] is not None:
            for key, number in metrics.downstream(matrix_a, matrix_t, weights, info["outcome"], direction).items():
                down[key] += number / world.n_values
        raw = probs * (states[NODE].to_numpy() != value)
        for x in range(world.n_values):
            rows = states[NODE].to_numpy() == x
            tv_by_x[x] += raw[rows] @ result["tv_rows"][rows]
            mass_by_x[x] += raw[rows].sum()
        chosen = units["drawn"] == value
        if chosen.any():
            nll_a += metrics.log_score(matrix_a, source_all[chosen], target_all[chosen]) * chosen.sum()
            nll_t += metrics.log_score(matrix_t, source_all[chosen], target_all[chosen]) * chosen.sum()
            n_scored += int(chosen.sum())
    row = {
        "condition": condition.name,
        "group": condition.group,
        "space": condition.space,
        "target": target,
        **totals,
        "tv_by_factual_x": {str(x): float(tv_by_x[x] / mass_by_x[x]) for x in range(world.n_values)},
        "mean_shift": {c: v for c, v in shift.items() if c != NODE},
        "nll_assumed": nll_a / n_scored,
        "nll_true": nll_t / n_scored,
        "params": condition.params,
    }
    if info["outcome"] is not None:
        row.update(down)
    return row


def parameter_error(kernel: SymbolicIntervention, true: SymbolicIntervention) -> dict[str, float]:
    """Mean relative error per parameter group of the S1 mechanisms."""
    groups: dict[str, list[float]] = {"coef_X": [], "coef_T": [], "noise_mu": [], "noise_sigma": []}
    for child, parents in true.mechanism_coeffs.items():
        for parent in ("X", "T"):
            est = kernel.mechanism_coeffs[child].get(parent, 0.0)
            groups[f"coef_{parent}"].append(abs(est - parents[parent]) / abs(parents[parent]))
        (mu_t, sd_t), (mu_e, sd_e) = true.noise_params[child], kernel.noise_params[child]
        groups["noise_mu"].append(abs(mu_e - mu_t) / abs(mu_t))
        groups["noise_sigma"].append(abs(sd_e - sd_t) / abs(sd_t))
    return {k: float(np.mean(v)) for k, v in groups.items()}


def build_conditions(world: World, protocol: dict[str, Any]) -> list[Condition]:
    full, with_m = world.full, world.with_m
    big = world.factual_sample(int(protocol["o2a_n_fit"]), world.seed + 2)
    out = [
        Condition("S1/true", "reference", "S1", full, list(TARGETS)),
        Condition("S6/true", "reference", "S6", with_m, ["total"]),
    ]
    # state-space family: the kernel restricted to S_k, unaware and refit
    for space, targets in (("S2", ["total"]), ("S3", ["total", "ps_DU"]), ("S4", ["total", "ps_D"]), ("S5", ["total", "ps_D"])):
        unaware = assumed.restrict(full, SPACES[space])
        out.append(Condition(f"{space}/unaware", space, space, unaware, targets))
        if unaware.mechanism_coeffs != {c: p for c, p in full.mechanism_coeffs.items() if c in SPACES[space]}:
            out.append(Condition(f"{space}/refit", space, space, assumed.refit(unaware, big), targets))
            out.append(Condition(f"{space}/refit_mle", space, space,
                                 assumed.refit(unaware, big, estimator="mle"), targets))
    # O1: T read but ignored by the operator
    o1 = assumed.drop_parents(full, {"D": ["T"], "U": ["T"]})
    out += [Condition("O1/unaware", "O1", "S1", o1, ["total"]),
            Condition("O1/refit", "O1", "S1", assumed.refit(o1, big), ["total"]),
            Condition("O1/refit_mle", "O1", "S1", assumed.refit(o1, big, estimator="mle"), ["total"])]
    estimators = list(protocol.get("estimators", ["naive"]))
    # O2: estimated parameters
    for tag in estimators:
        kernel = assumed.refit(full, big, fit_mechanisms=True, estimator=tag)
        out.append(Condition(f"O2a/{tag}", "O2a", "S1", kernel, ["total"],
                             {"estimator": tag, "n_fit": len(big), **parameter_error(kernel, full)}))
    rng = np.random.default_rng(world.seed + 3)
    for n_fit in protocol["o2b"]["n_fit"]:
        for resample in range(int(protocol["o2b"]["resamples"])):
            sample = world.factual_sample(int(n_fit), int(rng.integers(2**31)))
            for tag in estimators:
                kernel = assumed.refit(full, sample, fit_mechanisms=True, estimator=tag)
                out.append(Condition(f"O2b/{tag}/n{n_fit}/r{resample}", "O2b", "S1", kernel, ["total"],
                                     {"estimator": tag, "n_fit": int(n_fit), "resample": resample,
                                      **parameter_error(kernel, full)}))
    for group in protocol["o2d"]["groups"]:
        for factor in protocol["o2d"]["factors"]:
            if group.startswith("coef_"):
                kernel = assumed.scale(full, factor, parent=group.removeprefix("coef_"))
            else:
                kernel = assumed.scale(full, factor, noise=group.removeprefix("noise_"))
            out.append(Condition(f"O2d/{group}/x{factor}", "O2d", "S1", kernel, ["total"],
                                 {"group": group, "factor": factor}))
    # O3: dropped edges; O4: blocked edge
    out += [Condition("O3/drop_X_D", "O3", "S1", assumed.drop_parents(full, {"D": ["X"]}), ["total", "ps_D"]),
            Condition("O3/drop_X_U", "O3", "S1", assumed.drop_parents(full, {"U": ["X"]}), ["total"]),
            Condition("O4/block_X_D", "O4", "S1", assumed.block(full, {"D": ["X"]}), ["ps_D", "total"])]
    # O5: false edge X -> M on S6, c relative to the true X -> D coefficient
    reference = tm.base.MECHANISM_COEFFS["D"]["X"]
    mean_x = float(np.dot(np.arange(world.n_values), world.priors[NODE]))
    for relative in protocol["o5"]["c_relative"]:
        c = relative * reference
        noise = assumed.matched_root_noise(tm.M_PRIOR, mean_x, c)
        kernel = assumed.add_mechanism(with_m, "M", {"X": c}, noise)
        out.append(Condition(f"O5/c{relative}", "O5", "S6", kernel, ["total"], {"c": c, "c_relative": relative}))
    graph = assumed.add_mechanism(with_m, "M", {"X": 0.0}, assumed.matched_root_noise(tm.M_PRIOR, mean_x, 0.0))
    for n_fit in protocol["o5"]["refit_n_fit"]:
        for resample in range(int(protocol["o5"]["refit_resamples"])):
            sample = world.factual_sample(int(n_fit), int(rng.integers(2**31)))
            for tag in estimators:
                # only M's mechanism is estimated; D and U keep their true parameters, so the
                # result isolates the false edge from the estimator's bias on D and U (O2a)
                kernel = assumed.refit(graph, sample, fit_mechanisms=True, estimator=tag, only=["M"])
                out.append(Condition(f"O5refit/{tag}/n{n_fit}/r{resample}", "O5refit", "S6", kernel, ["total"],
                                     {"estimator": tag, "n_fit": int(n_fit), "resample": resample,
                                      "c_hat": kernel.mechanism_coeffs["M"]["X"]}))
    return out


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def summarize(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> str:
    frame = pd.DataFrame(rows)
    single = frame[~frame["group"].isin(["O2b", "O5refit"])]
    lines = ["# Operator benchmark", "",
             "Assumed vs. true symbolic operator on states (exact transition matrices; the log score "
             f"uses {protocol['n_score']:,} simulated units). TV: expected total-variation distance of "
             "the counterfactual distributions over factual states and non-identity do(X=k). "
             "Missing: true mass outside the assumed support. Shift: error of E[V'] along the intervention "
             "(positive = the assumed operator moves V further in the direction of the change in X than the "
             "truth). Y': Bayes-optimal outcome prediction (TV; E[Y'] error along the intervention). NLL: log "
             "score of realized counterfactuals, assumed vs. true kernel.", "",
             "## Conditions", "",
             "| condition | space | target | TV | missing | shift D | shift U | shift M | Y' TV | Y' shift | NLL assumed | NLL true |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in single.to_dict(orient="records"):
        shift = {c: v["directional"] for c, v in row["mean_shift"].items()}
        lines.append(
            f"| {row['condition']} | {row['space']} | {row['target']} | {_fmt(row['tv'])} | {_fmt(row['missing_mass'])} | "
            f"{_fmt(shift.get('D', float('nan')), 3)} | {_fmt(shift.get('U', float('nan')), 3)} | {_fmt(shift.get('M', float('nan')), 3)} | "
            f"{_fmt(row.get('y_tv', float('nan')))} | {_fmt(row.get('y_shift', float('nan')), 3)} | "
            f"{_fmt(row['nll_assumed'], 3)} | {_fmt(row['nll_true'], 3)} |")
    lines += ["", "## O2a: estimator bias (large sample)", "",
              "| estimator | TV | rel. err coef X | coef T | noise mu | noise sigma |", "|---|---|---|---|---|---|"]
    for row in frame[frame["group"] == "O2a"].to_dict(orient="records"):
        p = row["params"]
        lines.append(f"| {p['estimator']} | {_fmt(row['tv'])} | {_fmt(p['coef_X'], 3)} | {_fmt(p['coef_T'], 3)} | "
                     f"{_fmt(p['noise_mu'], 3)} | {_fmt(p['noise_sigma'], 3)} |")
    o2b = frame[frame["group"] == "O2b"]
    if len(o2b):
        o2b = o2b.assign(estimator=o2b["params"].map(lambda p: p["estimator"]), n_fit=o2b["params"].map(lambda p: p["n_fit"]),
                         coef_X=o2b["params"].map(lambda p: p["coef_X"]))
        lines += ["", "## O2b: sampling error (mean ± SD over resamples)", "",
                  "| estimator | n_fit | TV | rel. err coef X |", "|---|---|---|---|"]
        for (estimator, n_fit), group in o2b.groupby(["estimator", "n_fit"]):
            lines.append(f"| {estimator} | {n_fit} | {_fmt(group['tv'].mean())} ± {_fmt(group['tv'].std())} | "
                         f"{_fmt(group['coef_X'].mean(), 3)} ± {_fmt(group['coef_X'].std(), 3)} |")
    o2d = frame[frame["group"] == "O2d"]
    if len(o2d):
        table = o2d.assign(g=o2d["params"].map(lambda p: p["group"]), f=o2d["params"].map(lambda p: p["factor"]))
        pivot = table.pivot(index="g", columns="f", values="tv")
        lines += ["", "## O2d: sensitivity (TV; one parameter group scaled)", "",
                  "| group | " + " | ".join(f"×{f}" for f in pivot.columns) + " |",
                  "|---|" + "---|" * len(pivot.columns)]
        for group, values in pivot.iterrows():
            lines.append(f"| {group} | " + " | ".join(_fmt(v) for v in values) + " |")
    o5 = frame[frame["group"] == "O5refit"]
    if len(o5):
        o5 = o5.assign(estimator=o5["params"].map(lambda p: p["estimator"]), n_fit=o5["params"].map(lambda p: p["n_fit"]),
                       c_hat=o5["params"].map(lambda p: p["c_hat"]))
        o5 = o5.assign(shift_m=o5["mean_shift"].map(lambda s: s["M"]["directional"]))
        lines += ["", "## O5 refit: estimated false coefficient c (true 0); only M's mechanism refitted", "",
                  "| estimator | n_fit | c_hat | TV | shift M |", "|---|---|---|---|---|"]
        for (estimator, n_fit), group in o5.groupby(["estimator", "n_fit"]):
            lines.append(f"| {estimator} | {n_fit} | {_fmt(group['c_hat'].mean(), 3)} ± {_fmt(group['c_hat'].std(), 3)} | "
                         f"{_fmt(group['tv'].mean())} ± {_fmt(group['tv'].std())} | "
                         f"{_fmt(group['shift_m'].mean(), 3)} ± {_fmt(group['shift_m'].std(), 3)} |")
    threshold = float(protocol["screening_tv"])
    screened = sorted({r["condition"] for r in single.to_dict(orient="records")
                       if r["group"] not in ("reference",) and r["tv"] > threshold})
    lines += ["", f"## Screening (TV > {threshold})", "",
              "Enter the end-to-end runs: " + (", ".join(screened) if screened else "none") + ". "
              "The S family enters regardless, because omission errors are invisible at state level.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protocol", default=str(PROTOCOL))
    args = parser.parse_args()
    protocol = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))
    out = REPORTS / args.run_id
    if (out / "results.json").exists():
        raise FileExistsError(f"{out} already holds results; use a new run ID")
    world = World(protocol)
    conditions = build_conditions(world, protocol)
    rows = []
    for position, condition in enumerate(conditions, 1):
        for target in condition.targets:
            rows.append(evaluate(world, condition, target))
        if position % 25 == 0 or position == len(conditions):
            print(f"  {position}/{len(conditions)} conditions")
    out.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (out / "protocol.json").write_text(json.dumps({**protocol, "git_revision": revision}, indent=2), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(summarize(rows, protocol), encoding="utf-8")
    print(f"wrote {out}/summary.md ({len(rows)} rows)")


if __name__ == "__main__":
    main()
