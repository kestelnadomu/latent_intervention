"""Figures of the results section, from the latent_lab run (S1, rho trained on psi(z)):

- ``latent_tradeoff.pdf``: joint fidelity against preservation of M, per renderer and seed,
  for the renderer in isolation (true source state s, true counterfactual state s');
- ``latent_gap.pdf``: per-unit gap closed to phi(x') per renderer, in isolation;
- ``latent_triangle.pdf``: the triangle (z, edit, phi(x')) per renderer, in isolation, next to
  same-state texts, with the spread over units as 1-SD ellipses;
- ``latent_downstream.pdf``: counterfactual fairness, do(O = Western Europe) for every unit in
  the deployment setting: boxplots of the per-unit expected predicted qualification per factual region.
- ``latent_downstream_hist.pdf``: the predicted class per factual region, one panel per method (seed 0),
  with the true Y and Y' as reference panels.

The trade-off reads the run report. The gap, triangle and downstream figures recompute the edits
from the cached fits without refitting; their gap closed (against the report) and region
means (against the notebook's cached fairness evaluation) per renderer and seed must match. Writes
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
from matplotlib.patches import Ellipse  # noqa: E402

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
# Same-state texts in the triangle figure: label, cached encodings, filled
TRIANGLE_REFERENCES = {
    "swap": ("Country swap", "z_floor.pt", True),
    "vb": (r"$\mathcal{V}_B$ rewording", "z_factual_vb.pt", False),
}
# Offset (points) of each label from its marker in the triangle figure
TRIANGLE_LABELS = {
    "editor": (-5, -1), "ot_editor": (-5, 3), "flow": (5, -1), "state_effect": (5, 1), "ot_flow": (5, 2),
    "group_ot": (-5, -3), "swap": (-5, 3), "vb": (-5, 1),
}


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
        ax.set_title(label, fontsize=7, pad=2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("Gap closed to $\\varphi(x')$ per unit", fontsize=7, y=-0.06)
    fig.supylabel("Units (%)", fontsize=7, x=0.0)
    save(fig, "latent_gap.pdf")


def triangle_positions(edit: torch.Tensor, z: torch.Tensor, z_cf: torch.Tensor) -> np.ndarray:
    """Per-unit position of an edit in the frame z = (0, 0), phi(x') = (1, 0): its part along
    phi(x') - z and the length of the rest, both relative to |phi(x') - z|."""
    gap, shift = z_cf - z, edit - z
    along = (shift * gap).sum(1) / (gap * gap).sum(1)
    off = (shift - along[:, None] * gap).norm(dim=1) / gap.norm(dim=1)
    return torch.stack([along, off], 1).numpy()


def triangle_apex(positions: np.ndarray) -> tuple[float, float]:
    """The point at the units' mean distances from z and from phi(x'), so that its distance
    from phi(x') is 1 - the mean gap closed and its distance from z the mean relative shift."""
    to_z = np.hypot(positions[:, 0], positions[:, 1]).mean()
    to_cf = np.hypot(positions[:, 0] - 1, positions[:, 1]).mean()
    x = (to_z**2 - to_cf**2 + 1) / 2
    return x, np.sqrt(to_z**2 - x**2)


def triangle_figure(report: dict, run: dict) -> None:
    z, z_cf = run["z"], run["z_cf"]
    seeds = {key: triangle_positions(edit, z, z_cf) for key, edit in run["alone"].items()}
    check(report, "gap_closed", {key: float(1 - np.hypot(p[:, 0] - 1, p[:, 1]).mean()) for key, p in seeds.items()})
    # label, marker, filled, positions, label offset (points)
    series = {kind: (RENDERERS[kind][0], RENDERERS[kind][1], RENDERERS[kind][2], RENDERERS[kind][3],
                     np.concatenate([seeds[kind, seed] for seed in range(5)]), TRIANGLE_LABELS[kind])
              for kind in RENDERERS}
    for name, (label, source, filled) in TRIANGLE_REFERENCES.items():
        texts = run["encoder"].auxiliary("talent_motivation", source)
        rows = run["ids"].get_indexer(texts.ids)
        series[name] = (label, "0.25", "D", filled, triangle_positions(texts.factual(texts.ids), z[rows], z_cf[rows]),
                        TRIANGLE_LABELS[name])
    fig, ax = plt.subplots(figsize=(2.9, 2.9))
    # the arcs of equal gap closed around phi(x'); each label sits on its arc in the free upper right
    arc = np.linspace(np.pi / 2, np.pi, 200)
    for gap in (0.0, 0.1, 0.2):
        radius = 1 - gap
        ax.plot(1 + radius * np.cos(arc), radius * np.sin(arc), color="0.82", linewidth=0.6, linestyle=(0, (3, 3)), zorder=1)
        angle = np.radians(101)
        ax.text(1 + radius * np.cos(angle), radius * np.sin(angle), f"gap closed = {gap:.1f}", fontsize=5, color="0.4",
                ha="center", va="center", rotation=np.degrees(angle) - 90, rotation_mode="anchor", zorder=1,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8})
    for label, color, marker, filled, positions, offset in series.values():
        x, y = triangle_apex(positions)
        line = (0, (1, 1.5)) if color == "0.25" else ("-" if filled else "--")
        ax.plot([0, x, 1], [0, y, 0], color=color, linewidth=0.5, alpha=0.55, linestyle=line, zorder=2)
        values, vectors = np.linalg.eigh(np.cov(positions.T))
        ellipse = {"xy": (x, y), "width": 2 * values[1] ** 0.5, "height": 2 * values[0] ** 0.5,
                   "angle": np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))}
        if filled and color != "0.25":
            ax.add_patch(Ellipse(**ellipse, facecolor=color, alpha=0.1, edgecolor="none", zorder=2))
        ax.add_patch(Ellipse(**ellipse, fill=False, edgecolor=color, linewidth=0.6, linestyle=line, zorder=3))
        ax.scatter(x, y, s=16, marker=marker, facecolors=color if filled else "white", edgecolors=color,
                   linewidths=0.8, zorder=4, label=label)
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=5.5, color="0.15", zorder=6,
                    ha="left" if offset[0] > 0 else "right", va="center",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.4})
    ax.plot([0, 1], [0, 0], color="0.2", linewidth=1, zorder=3)
    ax.scatter([0, 1], [0, 0], s=16, color="0.2", zorder=5)
    ax.annotate("$z$", (0, 0), xytext=(0, -4), textcoords="offset points", ha="center", va="top", fontsize=8)
    ax.annotate(r"$z' = \varphi(x')$", (1, 0), xytext=(-2, -4), textcoords="offset points", ha="center", va="top", fontsize=8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.14, 1.08)
    ax.set_ylim(-0.1, 1.03)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=4, frameon=False, fontsize=5.5,
              handletextpad=0.1, columnspacing=0.6)
    save(fig, "latent_triangle.pdf")


def downstream_figure(report: dict, run: dict) -> None:
    """Counterfactual fairness: per-unit expected predicted qualification per factual region
    under do(O = Western Europe), for the unedited, edited and real latents."""
    records = [t for t in report["training"] if t["model"] == "pi"]
    pi = run_fit("pi-*.pt", records, lambda fit: [fit.fit_log])
    levels = torch.arange(3, dtype=torch.float32)

    def predict(z: torch.Tensor) -> np.ndarray:
        return pi.proba(z)["Y"].argmax(-1).numpy()

    def score(z: torch.Tensor) -> np.ndarray:  # E[Y] under pi's predicted distribution
        return (pi.proba(z)["Y"] @ levels).numpy()

    x = run["x"]  # the factual region of each unit
    # the notebook's cached fairness evaluation (cell-fair) must agree with the replay
    (path,) = CACHE.glob("fairness-*.pt")
    cached = torch.load(path, weights_only=False)
    for r in cached[(cached["train"] == "ψ(z)") & (cached["test"] == "ψ(z)")].itertuples():
        means = pd.Series(predict(run["fair"][r.rho, r.seed])).groupby(x).mean()
        if any(abs(means[k] - getattr(r, f"mean_X{k}")) > 1e-6 for k in means.index):
            raise ValueError(f"{r.rho}, seed {r.seed}: the replay does not reproduce the region means")
    # the real latents under X = 3: the unit's own text if it is from there, else its target
    # text if its drawn do() was X = 3; the other units have none
    real = (x == FAIR_X) | (run["do"] == FAIR_X)
    z_real = torch.where(torch.as_tensor(x == FAIR_X)[:, None], run["z"], run["z_cf"])[torch.as_tensor(real)]
    states = pd.read_csv(DATA / "sim_data_counterfactual_states.csv")
    y_fair = (states[(states["do_value"] == FAIR_X) & (states["target"] == "total")]
              .set_index("id").loc[run["ids"], "Y"].to_numpy())
    y = pd.read_csv(DATA / "sim_data_factual.csv", index_col="id").loc[run["ids"], "Y"].to_numpy()
    we = x == FAIR_X
    print(f"WE units: pi(z) accuracy against Y {(predict(run['z'])[we] == y[we]).mean():.3f}, "
          f"Y'_(O=3) == Y for {(y_fair[we] == y[we]).mean():.3f}")
    # benchmarks: label, colour, filled, per-unit scores, their regions (renderers: all seeds)
    series = [(r"unedited $\pi(z)$", "0.6", True, score(run["z"]), x)]
    for kind, (label, color, _, filled) in RENDERERS.items():
        series.append((label, color, filled, np.concatenate([score(run["fair"][kind, seed]) for seed in range(5)]),
                       np.tile(x, 5)))
    series.append((r"real latents", "0.25", True, score(z_real), x[real]))
    names = yaml.safe_load(CODEBOOK.read_text(encoding="utf-8"))["columns"]["X"]
    offsets = np.linspace(-0.36, 0.36, len(series))
    width = 0.8 * (offsets[1] - offsets[0])
    fig, ax = plt.subplots(figsize=(5.3, 2.3))
    for offset, (label, color, filled, values, groups) in zip(offsets, series):
        regions = sorted(set(groups))
        ax.boxplot([values[groups == k] for k in regions], positions=np.array(regions) + offset,
                   widths=width, whis=(5, 95), showfliers=False, patch_artist=True, manage_ticks=False,
                   boxprops={"facecolor": color if filled else "white", "edgecolor": color, "linewidth": 0.6},
                   medianprops={"color": "white" if filled else color, "linewidth": 0.8},
                   whiskerprops={"color": color, "linewidth": 0.6}, capprops={"color": color, "linewidth": 0.6})
        ax.plot([], [], "s", color=color, markerfacecolor=color if filled else "white", label=label)
    true_means = pd.Series(y_fair).groupby(x).mean()  # a line across each region's boxes
    ax.hlines(true_means.to_numpy(), true_means.index + offsets[0] - width, true_means.index + offsets[-1] + width,
              colors="#6250D6", linewidth=1.2, zorder=4,
              label=r"mean true $\mathtt{Y}'$ under $do(\mathsf{O} = $ WE$)$")
    ax.set_ylim(0, 2)
    for boundary in np.arange(0.5, len(names) - 1):
        ax.axvline(boundary, color="0.9", linewidth=0.6)
    ax.set_xticks(range(len(names)), [names[k] for k in range(len(names))])
    ax.set_xlabel("Factual region $\\mathsf{O}$")  # a mathtt O looks like a zero
    ax.set_ylabel("Expected predicted\nqualification")
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=5, frameon=False,
              handletextpad=0.2, columnspacing=0.8)
    save(fig, "latent_downstream.pdf")
    # the alternative: per method, the predicted class per factual region for one seed
    # (no pooling); fair = the regions' bars are equal. The true Y' is the reference panel.
    seed = 0
    renderer = {kind: (RENDERERS[kind][0], predict(run["fair"][kind, seed]), x) for kind in RENDERERS}
    # row 1: the outcomes and the minimal/constant-noise/closed-form base variants; row 2: the
    # real latents and the OT variants
    classes = [(r"true $\mathtt{Y}$", y, x), (r"true $\mathtt{Y}'$", y_fair, x),
               renderer["editor"], renderer["flow"], renderer["state_effect"],
               (r"$\pi(\varphi(x))$", predict(run["z"]), x), (r"$\pi(\varphi(x'))$", predict(z_real), x[real]),
               renderer["ot_editor"], renderer["ot_flow"], renderer["group_ot"]]
    fig, axes = plt.subplots(2, 5, figsize=(5.3, 2.5), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.45, "wspace": 0.12})
    # regions ordered by factual qualification: a sequential ramp, light (SSA) to dark (WE)
    region_colors = ["#F2A93B", "#D1495B", "#7B3294", "#1B1B6B"]
    bar = 0.8 / len(names)
    for ax, (label, values, groups) in zip(axes.flat, classes):
        for k in range(len(names)):
            share = np.bincount(values[groups == k], minlength=3) / (groups == k).sum() * 100
            ax.bar(np.arange(3) + (k - (len(names) - 1) / 2) * bar, share, width=bar,
                   color=region_colors[k], linewidth=0)
        ax.set_title(label, fontsize=7, pad=2)
        ax.set_xticks(range(3))
        ax.grid(axis="y", color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("Qualification (class)", fontsize=7, y=-0.05)
    fig.supylabel("Units of the region (%)", fontsize=7, x=0.02)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, linewidth=0) for c in region_colors]
    fig.legend(handles, [names[k] for k in range(len(names))], loc="upper center",
               bbox_to_anchor=(0.5, -0.09), ncol=len(names), frameon=False, handlelength=1.2,
               title="Factual region $\\mathsf{O}$", title_fontsize=7)
    save(fig, "latent_downstream_hist.pdf")
    for label, _, _, values, groups in series:
        means = pd.Series(values).groupby(groups).mean()
        print(f"{label:28s} " + "  ".join(f"{m:.2f}" for m in means) + f"  var {means.var(ddof=0) * 100:.2f}e-2")


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
    triangle_figure(report, run)
    downstream_figure(report, run)


if __name__ == "__main__":
    main()
