"""Figures of the results section, from the latent_lab run (S1, rho trained on psi(z)):

- ``latent_tradeoff.pdf``: joint fidelity against preservation of M, per renderer and seed,
  for the renderer in isolation (true source state s, true counterfactual state s');
- ``latent_gap.pdf``: per-unit gap closed to phi(x') per renderer, in isolation, with the
  noise floor;
- ``latent_downstream.pdf``: counterfactual fairness, do(O = Western Europe) for every unit in
  the deployment setting: mean predicted qualification per factual region.

The trade-off reads the run report. The gap and the downstream figure recompute the edits
from the cached fits without refitting; their gap closed (against the report) and SPD
(against the notebook's cached fairness evaluation) per renderer and seed must match. Writes
to ``overleaf_paper/figures/``:

    uv run --with matplotlib==3.10.3 python scripts/latent_figures.py
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from latent_lab import Encoder, LinearSCM, Pipeline, mean_predict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/latent_lab/run-20261030-183501.json"
FIGURES = ROOT / "overleaf_paper/figures"
CACHE = ROOT / "models/latent_lab"
DATA = ROOT / "data/sim_talent_motivation"
CODEBOOK = ROOT / "exp/sim_talent_motivation/codebook.yaml"
# Colour and marker per renderer family (categorical slots 1-3, the ones that stay apart
# for every pair in a scatter); the OT variant of each family is hollow.
RENDERERS = {
    "editor": ("Editor", "#2A78D6", "o", True),
    "ot_editor": ("OT editor", "#2A78D6", "o", False),
    "flow": ("State flow", "#EB6834", "s", True),
    "ot_flow": ("OT flow", "#EB6834", "s", False),
    "state_effect": ("State effect", "#1BAF7A", "^", True),
    "group_ot": ("Group OT", "#1BAF7A", "^", False),
}
# The operator of experiment.ipynb: the SCM of D and U given X and T.
LEVELS = {"X": 4, "T": 3, "D": 3, "U": 3}
PARENTS = {"D": {"X": 0.30, "T": 0.60}, "U": {"X": 0.35, "T": 0.50}}
NOISE = {"D": (-0.10, 0.50), "U": (-0.15, 0.50)}
FAIR_X = 3  # Western Europe: the counterfactual fairness use case sets every unit to it


def isolated(r: dict) -> bool:
    """The renderer in isolation: trained on psi(z), given the true s and the true s'."""
    return (r["S"] == "S1" and r.get("target") == "true s'"
            and r["train"] == "ψ(z)" and r["test"] == "true s")


def save(fig, name: str) -> None:
    # No creation date: the same inputs give the same PDF bytes.
    fig.savefig(FIGURES / name, bbox_inches="tight", metadata={"CreationDate": None})
    print(f"wrote {FIGURES / name}")


def tradeoff_figure(results: list[dict]) -> None:
    alone = [r for r in results if isolated(r)]
    (identity,) = [r for r in results if r["rho"] == "identity"]
    fig, ax = plt.subplots(figsize=(2.9, 2.3))
    for rho, (label, color, marker, filled) in RENDERERS.items():
        rows = [r for r in alone if r["rho"] == rho]
        ax.scatter([100 * r["joint_acc"] for r in rows], [100 * r["acc_M"] for r in rows],
                   s=18, marker=marker, facecolors=color if filled else "white",
                   edgecolors=color, linewidths=0.8, label=label, zorder=3)
    references = (
        (r"identity $z' = z$", identity["joint_acc"], identity["acc_M"], "left", 4),
        (r"real $\varphi(x')$", alone[0]["real_joint_acc"], alone[0]["real_acc_M"], "center", 0),
    )
    for label, joint, m, align, dx in references:
        ax.scatter(100 * joint, 100 * m, s=40, marker="*", color="0.25", zorder=3)
        ax.annotate(label, (100 * joint, 100 * m), xytext=(dx, 5), textcoords="offset points", ha=align)
    ax.set_xlim(-3, 103)
    ax.set_xlabel("Joint fidelity (%)")
    ax.set_ylabel(r"Preservation of $\mathtt{M}$ (%)")
    ax.grid(color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False,
              handletextpad=0.2, columnspacing=0.8)
    save(fig, "latent_tradeoff.pdf")


def run_fit(pattern: str, records: list[dict], logs) -> object:
    """This run's fit among the cached versions in ``CACHE``: the one whose training logs
    (``logs(fit)``) are the report's ``records``, including the fit's duration."""
    def same(fit) -> bool:
        return all(all(record.get(k) == v for k, v in log.items())
                   for log, record in zip(logs(fit), records, strict=True))

    found = [fit for path in sorted(CACHE.glob(pattern))
             if same(fit := torch.load(path, weights_only=False))]
    if len(found) != 1:
        raise ValueError(f"{len(found)} cached fits {pattern} match the run report")
    return found[0]


def replay(report: dict) -> dict:
    """The evaluation units, the isolated edits rho(z; s, s') and the deployment edits under
    do(X = FAIR_X) of every renderer and seed."""
    partition = pd.read_csv(DATA / "partition.csv", index_col="id")
    eval_ids = partition.index[partition["partition"] == "eval"]
    do_eval = pd.read_csv(DATA / "interventions.csv", index_col="id")["do_value"].loc[eval_ids].to_numpy()
    s = pd.read_csv(DATA / "sim_data_factual.csv", index_col="id").loc[eval_ids]
    s_cf = pd.read_csv(DATA / "sim_data_counterfactual.csv", index_col="id").loc[eval_ids]
    encoder = Encoder("embeddinggemma", 768)
    latents = encoder.latents("talent_motivation")
    z = latents.factual(eval_ids)
    training = report["training"]
    alone, fair = {}, {}
    for seed in range(5):
        records = sorted((t for t in training if t["model"] == "psi" and t["seed"] == seed
                          and t.get("encoder") == "embeddinggemma-768"), key=lambda t: t["fold"])
        _, readers = run_fit(f"psi-embeddinggemma-768-seed{seed}-*.pt", records,
                             lambda fit: [r.fit_log for r in fit[1]])
        states = mean_predict(readers, z)
        for kind in RENDERERS:
            records = [t for t in training if t["model"] == f"rho-{kind}" and t.get("S") == "S1"
                       and t.get("train") == "ψ(z)" and t["seed"] == seed]
            renderer = run_fit(f"rho-S1-{kind}-psi-seed{seed}-*.pt", records, lambda fit: [fit.fit_log])
            alone[kind, seed] = renderer(z, s.reset_index(drop=True), s_cf[s.columns].reset_index(drop=True))
            # a fresh operator per renderer, as in the notebook
            operator = LinearSCM(LEVELS, PARENTS, NOISE, seed=seed).do(X=FAIR_X)
            fair[kind, seed] = Pipeline(None, operator, renderer)(z, states).z_prime
    return {"ids": eval_ids, "do": do_eval, "x": s["X"].to_numpy(), "encoder": encoder, "z": z,
            "z_cf": latents.counterfactual(eval_ids), "alone": alone, "fair": fair}


def check(report: dict, metric: str, values: dict) -> None:
    """The replayed ``metric`` of the isolated renderers per seed must be the report's."""
    for r in report["results"]:
        if isolated(r) and abs(values[r["rho"], r["seed"]] - r[metric]) > 1e-6:
            raise ValueError(f"{r['rho']}, seed {r['seed']}: the replay does not reproduce {metric}")


def gap_figure(report: dict, run: dict) -> None:
    z, z_cf = run["z"], run["z_cf"]
    before = (z - z_cf).norm(dim=1)  # raw coordinates, as latent_lab.pipeline.evaluate
    per_unit = {key: (1 - (z_prime - z_cf).norm(dim=1) / before).numpy() for key, z_prime in run["alone"].items()}
    check(report, "gap_closed", {key: float(v.mean()) for key, v in per_unit.items()})
    gaps = {kind: np.concatenate([per_unit[kind, seed] for seed in range(5)]) for kind in RENDERERS}
    floor = run["encoder"].auxiliary("talent_motivation", "z_floor.pt")
    rows = run["ids"].get_indexer(floor.ids)
    # a perfect edit still lies as far from phi(x') as a same-state rewrite from its text
    ceiling = (1 - (floor.factual(floor.ids) - z[rows]).norm(dim=1) / before[rows]).numpy()
    fig, axes = plt.subplots(2, 3, figsize=(2.9, 2.2), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.45})
    # the OT flow's long tail below -0.4 is collected in the leftmost bin
    bins = np.arange(-0.4, 0.75, 0.05)
    # columns: the three families; rows: base variant, then its hollow OT variant
    for ax, kind in zip(axes.T.flat, RENDERERS):
        label, color, _, filled = RENDERERS[kind]
        values = np.clip(gaps[kind], bins[0], bins[-1])
        ax.hist(values, bins=bins, weights=np.full(len(values), 100 / len(values)),
                color=color if filled else "white", edgecolor=color, linewidth=0.6)
        ax.axvline(ceiling.mean(), color="0.25", linewidth=0.8)
        ax.set_title(label, fontsize=7, pad=2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1, 1].annotate("noise\nfloor", (ceiling.mean(), 1), xycoords=("data", "axes fraction"),
                        xytext=(-2, 0), textcoords="offset points", ha="right", va="top", fontsize=6)
    fig.supxlabel("Gap closed to $\\varphi(x')$ per unit", fontsize=7, y=-0.06)
    fig.supylabel("Units (%)", fontsize=7, x=0.0)
    save(fig, "latent_gap.pdf")
    print(f"noise floor: mean {ceiling.mean():.3f} over {len(ceiling)} units")


def downstream_figure(report: dict, run: dict) -> None:
    """Counterfactual fairness: mean predicted qualification per factual region under
    do(O = Western Europe), for the unedited, edited and real latents."""
    records = [t for t in report["training"] if t["model"] == "pi"]
    pi = run_fit("pi-*.pt", records, lambda fit: [fit.fit_log])

    def predict(z: torch.Tensor) -> np.ndarray:
        return pi.proba(z)["Y"].argmax(-1).numpy()

    x = run["x"]  # the factual region of each unit

    def by_region(values: np.ndarray, groups: np.ndarray = x) -> pd.Series:
        return pd.Series(values).groupby(groups).mean()

    edited = {key: by_region(predict(z_prime)) for key, z_prime in run["fair"].items()}
    # the notebook's cached fairness evaluation (cell-fair) must agree with the replay
    (path,) = CACHE.glob("fairness-*.pt")
    cached = torch.load(path, weights_only=False)
    for r in cached[(cached["train"] == "ψ(z)") & (cached["test"] == "ψ(z)")].itertuples():
        means = edited[r.rho, r.seed]
        if abs(means[3] - means[0] - r.spd) > 1e-6:
            raise ValueError(f"{r.rho}, seed {r.seed}: the replay does not reproduce the SPD")
    # the real latents under X = 3: the unit's own text if it is from there, else its target
    # text if its drawn do() was X = 3; the other units have none
    real = (x == FAIR_X) | (run["do"] == FAIR_X)
    z_real = torch.where(torch.as_tensor(x == FAIR_X)[:, None], run["z"], run["z_cf"])[torch.as_tensor(real)]
    states = pd.read_csv(DATA / "sim_data_counterfactual_states.csv")
    y_fair = (states[(states["do_value"] == FAIR_X) & (states["target"] == "total")]
              .set_index("id").loc[run["ids"], "Y"].to_numpy())
    # benchmarks: label, colour, marker, filled, mean per region
    series = [(r"unedited $\pi(z)$", "0.6", "o", True, by_region(predict(run["z"])))]
    for kind, (label, color, marker, filled) in RENDERERS.items():
        series.append((label, color, marker, filled,
                       sum(edited[kind, seed] for seed in range(5)) / 5))
    series.append((r"real latents", "0.25", "*", True, by_region(predict(z_real), x[real])))
    references = [(r"true $\mathtt{Y}'$ under $do(\mathsf{O} = $ WE$)$", "#6250D6", by_region(y_fair))]
    names = yaml.safe_load(CODEBOOK.read_text(encoding="utf-8"))["columns"]["X"]
    offsets = np.linspace(-0.36, 0.36, len(series))
    width = 0.8 * (offsets[1] - offsets[0])
    fig, ax = plt.subplots(figsize=(5.3, 2.3))
    for offset, (label, color, marker, filled, means) in zip(offsets, series):
        xs = means.index + offset
        # a transparent bar from 0 up to the benchmark's symbol, behind everything
        ax.bar(xs, means.to_numpy(), width=width, color=color, alpha=0.25, linewidth=0, zorder=1)
        ax.scatter(xs, means.to_numpy(), marker=marker, label=label, s=40 if marker == "*" else 18,
                   facecolors=color if filled else "white", edgecolors=color, linewidths=0.8, zorder=3)
    for label, color, means in references:  # a line across each region's benchmarks
        ax.hlines(means.to_numpy(), means.index + offsets[0] - width, means.index + offsets[-1] + width,
                  colors=color, linewidth=1.2, label=label, zorder=2)
    ax.set_ylim(0.5, 1.6)
    for boundary in np.arange(0.5, len(names) - 1):
        ax.axvline(boundary, color="0.9", linewidth=0.6)
    ax.set_xticks(range(len(names)), [names[k] for k in range(len(names))])
    ax.set_xlabel("Factual region $\\mathsf{O}$")  # a mathtt O looks like a zero
    ax.set_ylabel("Mean predicted qualification")
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=5, frameon=False,
              handletextpad=0.2, columnspacing=0.8)
    save(fig, "latent_downstream.pdf")
    for label, *_, means in series + references:
        print(f"{label:28s} " + "  ".join(f"{m:.2f}" for m in means) + f"  SPD {means[3] - means[0]:+.2f}")


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    results = [r for r in report["results"] if r["S"] == "S1"]
    alone = [r for r in results if isolated(r)]
    if sorted({r["rho"] for r in alone}) != sorted(RENDERERS) or len(alone) != 30:
        raise ValueError("expected six renderers with five seeds each")
    # TrueType fonts: ACM rejects the Type 3 fonts matplotlib embeds by default.
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
    tradeoff_figure(results)
    run = replay(report)
    gap_figure(report, run)
    downstream_figure(report, run)


if __name__ == "__main__":
    main()
