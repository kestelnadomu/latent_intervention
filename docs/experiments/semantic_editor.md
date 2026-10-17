# Clean semantic editors

`semantic_editor` and `ot_editor` implement the original commutative-pipeline
question: among embeddings whose decoded distribution agrees with the symbolic
counterfactual, which edit should be made to a factual individual's embedding?
They share supervision, objective, split and selection. Their only methodological
difference is the map class. Both are standalone: no teacher or state-flow model.

The first clean-editor run, `editor-clean-v1`, completed on 2026-10-17: 20 tier-1
fits (both methods, 256D/768D, seeds 42–46), followed by explicitly requested
official-test evaluation. All checkpoints were selected on factual validation
before test scoring. See the [results and benchmark comparison](../../reports/talent/editor_benchmarks/editor-clean-v1/comparison.md)
and [per-seed report](../../reports/talent/editor_benchmarks/editor-clean-v1/summary.md).
Existing encodings, semantic decoders, original nine-method benchmarks, Gaussian
OT and OT flow are unchanged; compatible historical evaluations were reused.

## Definition

For frozen semantic decoder $g$ and analytic kernel $h_S$, use the complete target
marginal, not a sampled or independently paired source state:

$$q_a(s'\mid z)=\sum_s g(s\mid z)h_S(s'\mid s,a),\qquad
h_Z(\cdot\mid z,a)=\sum_{s'}q_a(s'\mid z)\delta_{T_{s'}(z)}.$$

Every positive target weight is retained, including small weights; exact zeros
remain zero. Target components have no learned mixture weights or top-$k$ cutoff.
Inference requires the existing frozen $g$ and analytic $h_S$ as separate inputs.
The public sampling interface is latent-only and rejects observed-state overrides.

Let $x=(z-\mu)/\sigma$, where only the 3,200 fitting embeddings determine $\mu$
and the coordinate standard deviations (`unbiased=False`, floor `1e-4`). Maps are
defined in these standardized coordinates; their outputs are unstandardized
before applying $g$ or returning an embedding. No output projection or unit-norm
renormalization is applied.

- `semantic_editor`: $T(x;s')=x+\Delta_\theta(x,s')$. It reuses the existing,
  unchanged `_DeltaNet`: one transformer layer, four attention heads, no dropout,
  and zero-initialized residual output.
- `ot_editor`: $T(x;s')=\nabla_x\psi_\theta(x;s')$. One target-conditioned
  input-convex potential has two Softplus hidden layers, affine input skips,
  nonnegative hidden/output connections, categorical state embeddings of width
  16, and a learnable positive diagonal quadratic with floor `1e-3`. The quadratic
  is not fixed at identity, so the map can contract. There is no source-state
  conditioning, dual potential, minimax optimization, or factual target-distribution
  matching.

The convexity statement concerns each **fixed target in standardized coordinates**.
For a sufficiently regular input distribution, a convex gradient is a quadratic-cost
OT map to *its own induced output distribution*. That does not establish that this
output is the desired factual state-conditional distribution or the true individual
counterfactual. The state mixture and input-dependent weights are not themselves
one globally invertible OT map. This is **OT-structured semantic editing**, not a
new OT algorithm or a guarantee of true counterfactual recovery.

Only `do(X=3)` and the empty action `{}` are supported. Empty actions return the
input exactly. Unsupported interventions are rejected. Individual target components
and factual units whose true X is already 3 are **not** forced to be exact identities.

## Objective and data boundary

Both maps minimize exactly the same loss:

$$L=\operatorname{KL}\!\left(q_a\;\middle\|\;
\sum_{s'}q_a(s'\mid z)g(\cdot\mid T_{s'}(z))\right)
+0.1R_{\rm movement}+0.05R_{\rm support}+R_{\rm identity}.$$

- Movement is mixture-weighted squared displacement per standardized coordinate.
- Support is mixture-weighted nearest-bank Euclidean distance in standardized
  coordinates, divided by $\sqrt d$. The bank contains 128 fitting embeddings
  sampled once with seed 42 and is identical across model/training seeds.
- Identity is the same movement cost averaged over factual units with X=3; it is
  zero in batches without these units. Validation averages it over all tuning X=3
  units rather than averaging unequal batch counts.

KL is computed by exact finite-mixture composition through $g$, not Monte Carlo.
The parameters of $g$ are frozen, while gradients through its inputs reach the
editor. The analytic transition matrix is detached. There are no preservation,
radial, entropy, component-label, or factual state-effect losses; no warmup or
Gaussian smoothing. Mixture consistency does not identify which target-labelled
component decodes to which state, or guarantee semantic correctness of each draw.

The fitting interface exposes only factual Z, visible S=(X,T,D,U), IDs, schema
and frozen symbolic components. Visible X supplies the identity regularizer;
visible states also provide support diagnostics. P/L/H/A are not parsed, imported,
used as gates, fitted as readers, or used in primary evaluation. There is no
paired Z-prime supervision. Hashes cover the consumed tensors/visible states and
symbolic components, so changing unused auxiliary CSV columns cannot affect fits
or checkpoint selection. Encoded texts can still implicitly contain those
attributes; excluding their explicit labels is not proof of invariance.

The legacy g checkpoint includes hashes of a whole CSV and paired latent container.
Those remain historical provenance rather than proxy-dependent loading gates here:
the clean adapter verifies g's encoder identity, schema and dimension, then pins
its exact checkpoint bytes and the consumed factual tensors in each run signature.
It does not rewrite g or claim to re-certify its original training data.

The fixed split is reconstructed from official IDs with split seed 42 and audit
split seed 20261007:

| Partition | Units | Use |
|---|---:|---|
| Fitting | 3,200 | Objective, normalization, support bank |
| Tuning | 400 | Common validation objective, selection, scheduler, stopping |
| Reserved audit | 400 | Unused in this workflow |
| Official test | 1,000 | Only explicitly requested evaluation |

The canonical latent container also holds test targets; loading the container
does not expose them to `FittingData`, the objective or the selector. Counterfactual
CSV states and target embeddings are accessed only by explicit evaluation. Test
and reserved-audit embeddings do not determine normalization or support.
Missing factual states are reported, but their target probability mass is never
dropped; predictions there involve extrapolation. Frozen $g$ was previously
trained/selected separately, so these splits are not a new end-to-end holdout.

## Capacity and training

| Dimension | Tier | Residual parameters / target | OT potential parameters |
|---|---:|---:|---:|
| 256 | 1 | 199,552 | 199,308 |
| 256 | 2 | 660,992 | 660,360 |
| 768 | 1 | 331,136 | 330,154 |
| 768 | 2 | 923,648 | 923,734 |

Residual widths are 128/256 and feedforward widths 256/512. OT widths are chosen
deterministically by nearest instantiated parameter count (tie: smaller width),
with a ±7% hard limit. The single potential is both training-active and deployed;
there are no training-only parameters. Frozen $g$ is reported separately. Equal
parameter counts are not equal compute, inductive bias or effective capacity.

`configs/hz_editor.yaml` specifies tier 1, dimensions 768 and 256, seeds 42–46,
CPU with one numerical thread per process, AdamW (`lr=3e-4`, decay `1e-5`), at most
500 epochs, batch 32, gradient clipping 5, patience 30 and minimum improvement
`1e-7`. The plateau scheduler has patience 10, factor 0.5 and minimum LR `1e-6`.
An epoch is one shuffled pass through the 3,200 fitting units (100 optimizer steps).
Validation uses fixed data, the same support bank, and no sampling randomness.
The actual lowest validation objective is restored, including the initial epoch-zero
model as a candidate; small improvements are saved even when below stopping tolerance.

## Commands and module layout

All commands run from the repository root using the existing virtual environment.

```bash
# Read-only; checks both capacity tiers, forward/backward, splits and input compatibility.
.venv/bin/python -m src.latent_intervention.editor preflight --all

# Twenty-fit sequential detached queue; completed matching fits are verified/reused.
bash scripts/run_hz_editor.sh editor-clean-v1

# Individual resumable fit, if desired instead of the queue.
.venv/bin/python -m src.latent_intervention.editor train \
  --variant semantic_editor --dimension 768 --seed 42 --tier 1 --run-id editor-clean-v1

# Explicit evaluation only, after fitting/selection are complete and frozen.
.venv/bin/python -m src.latent_intervention.editor evaluate --all --run-id editor-clean-v1
.venv/bin/python -m src.latent_intervention.editor report --all --run-id editor-clean-v1 \
  --output reports/talent/editor_benchmarks/editor-clean-v1/summary.md
```

Use `--variant ot_editor` for its matched counterpart. `--protocol` selects a YAML
protocol and `--config` selects the established base config. `--all` expands to
20 tier-1 cases by default, or four cases when a single `--seed` is supplied.
A different tier/protocol requires a new run ID to avoid conflicting signatures.
The launcher preflights before fitting, uses detached tmux, stops on failure,
and never starts audit/test evaluation. Rerunning the same valid configuration
resumes interrupted fits or validates/skips completed fits. It is a sequential
queue, not a tuning framework. `src.pipeline` also exposes both names through
`--manipulator-variant`, using the dedicated protocol rather than base-trainer defaults.

| Module | Responsibility |
|---|---|
| `editor/models.py`, `common.py` | Maps, target marginal, components, sampling, persistence |
| `editor/objective.py` | One shared exact semantic/geometry objective |
| `editor/protocol.py`, `capacity.py` | Factual boundary, splits, hashes, paths, architecture budgets |
| `editor/training.py` | Optimization, validation selection, atomic resume |
| `editor/evaluation.py` | Explicit common oracle/visible-state metrics only |
| `editor/workflow.py`, `__main__.py`, `reporting.py` | Thin pipeline/CLI adapters and existing-record summaries |

Public components return locations `(components, batch, dimension)` and log
weights `(batch, components)`; components are in ascending positive target-index
order. Public seeded samples have shape `(samples, batch, dimension)` and work
inside both `no_grad` and inference mode. The ICNN locally enables input derivatives
without requiring callers to manage autograd.

## Artifacts, resume and evaluation

Both methods use checkpoint format `semantic_editor_v1`. Selected deployments:

```text
models/talent/embeddinggemma_D/g-independent/h-V/runs/R/seed-N/
  selected/latent_intervention.pt
  selected/latent_intervention.info.json
  resume.pt                         # local only
reports/talent/embeddinggemma_D/g-independent/h-V/runs/R/seed-N/
  protocol.json
  training.json
  evaluation.json                  # only after explicit evaluation
```

Each selected checkpoint is self-contained apart from frozen g/analytic h_S: it
contains normalization, architecture, schema, intervention, selected weights,
consumed-input/source/config hashes and split IDs. Records include histories,
selected epoch, capacity, runtime and optimizer counts. Atomic resume snapshots
contain current/best weights, optimizer, scheduler, epoch, stopping state and
Python/NumPy/Torch/shuffle RNG states. Epoch-boundary replay is deterministic;
an interruption within an epoch replays that epoch from its last atomic snapshot.
Exclusive locks reject concurrent writers; conflicting signatures are never
overwritten. No historical experimental directory is a runtime dependency.

Only seed-42 deployment checkpoints and compact records for every seed are
Git-trackable. Other selected checkpoints, optimizer snapshots, locks and logs
stay local. The queue does not stage or commit anything. All 20 selected
checkpoints and their compact training/evaluation records now exist locally.
The combined comparison and its source-hash manifest are also Git-trackable.

Explicit evaluation uses 64 IID draws, seed 20260930, batches of 32 and the existing
common per-unit metrics. Energy scaling uses fitting-only coordinate standard
deviations (`unbiased=False`, floor `1e-6`) and $\sqrt d$, matching the established
comparison convention. Report all/identity/nonidentity subsets separately.
Semantic KL measures consistency with frozen g/h_S; realized joint/coordinate
accuracy measures actual S-prime prediction; energy and embedding errors measure
paired oracle Z-prime recovery. None substitutes for the others. Reports distinguish
single-seed outcomes from across-training-seed mean ± sample standard deviation.

## Implementation verification (2026-10-16)

- Regression suite: `458 passed, 3 skipped` via `.venv/bin/pytest -q tests`.
  Synthetic coverage includes conditional convexity/contraction, exact target
  weights and zero handling, shared loss, frozen symbolic components, proxy/oracle
  exclusion, selected-checkpoint restoration, interrupted resume, archive-independent
  loading/evaluation and tracking policy. No synthetic test fits real CV data.
- `ruff check src tests exp scripts`, changed-package formatting checks, shell
  syntax checks and `git diff --check` pass. No active import or command depends
  on a retired implementation.
- The actual `preflight --all` CLI checked all 20 cases and both tiers (40
  finite forward/backward checks), with zero optimizer steps, new selected-model
  directories or official-test evaluations. Both dimensions reproduce the
  3,200/400/400/1,000 split. Maximum neural parameter-budget deviation is 0.297%.
- Support diagnostics retain all 27 positive target states. Twenty of the 108
  full visible states have no fitting examples; mean tuning target mass on absent
  fitting states is 0.325% at 256D and 0.208% at 768D (per-unit maxima 7.61% and
  10.31%). This is an extrapolation caveat, not a reason to discard probability.
- All 45 protected artifact files retain their checksums. Gaussian OT / OT-flow
  seeded predictions on fixed fitting inputs are bitwise equal before/after
  cleanup at both dimensions. The shared audit utility is byte-identical.

At the implementation handoff, the queue remained unstarted: these checks
established implementation and protocol readiness, not convergence or
counterfactual performance. The subsequent fitting and explicit evaluation are
documented in the linked run reports above. Those exploratory test results did
not trigger further fitting or checkpoint selection.

## Historical experiments and recovery

The original target-only C-sym experiments motivated `semantic_editor`. An old
`c_sym/none` checkpoint had no projection or preservation penalty but still used
a P/L/H/A guide in selection; it cannot establish a proxy-free comparison.
State-effect/correction/projection variants were then explored. The published
`c_sym_project` retained a state-effect map and explicit guide-based projection;
its preservation assumptions no longer match the intended visible-state-only
narrative. None of these weights is relabelled as a clean editor.

The earlier `ot_editor` was a different, source-and-target-conditioned distributional
OT algorithm: two convex potentials, alternating minimax updates and Sinkhorn
selection on factual state groups. It did not share the semantic-editor objective.
That implementation and its results are retired, not silently reinterpreted as
the new OT-structured editor. `latent_ot_v1` checkpoints naming `ot_editor` and
projected C-sym checkpoints are explicitly rejected by active loaders.

Earlier comparisons were mostly single-seed and official-test results have already
been inspected. New evaluation on those IDs remains exploratory, not an untouched
confirmatory benchmark. Historical metrics, histories and provenance are retained
unaltered in the recovery archive; they are not performance claims for these new maps.
Gaussian OT and OT flow keep their original selected weights, histories, audit
signatures and results. Their shared probe utility was relocated byte-for-byte to
`src/latent_intervention/audit_probes.py`; the new editor package never imports it.

The checksum-verified local archive is
`archives/editor-cleanup-20261016T155004Z/` (Git-ignored). `archive.json` inventories
303 pre-cleanup files including dirty/untracked OT work; `tracked.patch` records
the original Git changes. Original retired artifact directories are additionally
under `retired/`. `factual_parity.pt` records pre-cleanup Gaussian/flow predictions
on fitting IDs only. The earlier `archives/csym-experiments-2026-10-13/` is preserved.
Recovery means copying the desired archived files to a separate research checkout,
not re-enabling archived imports in the clean workflow. Share archives separately
if another machine needs historical experiments.
