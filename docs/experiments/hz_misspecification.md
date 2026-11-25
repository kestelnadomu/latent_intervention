# Five-method misspecification workflow

Implementation: `exp/benchmarks/misspecification/`; configuration:
`configs/hz_misspecification.yaml`; default run: `hz-misspec-v1`.
This implements the first regime of [the study design](wrong_symbolic_model.md),
not its label-budget, downstream-Y, cross-fitting or wording-generation extensions.
Preparation does not launch training or official-test evaluation.

The [consolidated results handoff](hz_misspecification_final_handoff.md) records
the completed 196-fit assembly and the official-test evaluation launched on
26 October 2026, including artifact locations, backups and resume commands.

The run was split across machines on 20 October 2026 and consolidated on
26 October. See the
[50-fit remote agent handoff](hz_misspecification_remote50_handoff.md) for the
historical assignment, portable bundle and shard-only launch/return commands.
All fitting is now complete; use the consolidated handoff for evaluation.

## Data and roles

Use the persisted `talent_motivation` partition and existing EmbeddingGemma 768D
embeddings: **2,800 factual fitting / 200 development / 1,000 official-test units**.
The guiding independent readers are the existing S1–S6 final checkpoints, seeds
42–46, from `g-independent-s1-s6-v1`. Each neural hZ seed uses its matching g seed.
True factual states are available throughout hZ fitting. This is **not** an
end-to-end 1,000-label experiment.

| Component | Fitting information | Selection |
|---|---|---|
| Both editors | factual Z, visible S, frozen g, assumed analytic hS | same factual-development objective |
| `state_flow` | factual Z and visible S | factual conditional NLL |
| `ot_flow` | factual Z and visible S | fixed-randomness factual flow-matching loss |
| `ot_gaussian` | factual Z and visible S | analytical fit; development NLL is diagnostic |
| Independent evaluation reader | reserved P: 1,000 factual units, nine heads | development joint NLL, 20 existing candidate settings |

The independent reader has heads X,T,D,U,M,P,L,H,A, excludes Y, and publishes
predeclared seed 42. P is disjoint from g's L; P may overlap the hZ factual pool,
as specified in the design. Its labels, weights and predictions never enter hZ
objectives or checkpoint selection. Calibration/accuracy are reported per head.
Disjoint reader labels alone do **not** establish wording robustness.

## Scenarios and reuse

S1=(X,T,D,U), S2=(X,D,U), S3=(X), S4=(X,T,U), S5=(X,U),
S6=(X,T,D,U,M). S1/S6 use correct kernels; S2–S5 restrict the mechanisms
without refitting remaining parameters or marginalizing omitted parents: the
existing parameter-preserving **unaware** construction.

| Operator | State space | Settings |
|---|---|---|
| O1 | S1 | remove T from D and U mechanisms |
| O2 | S1 | separately multiply X coefficients by 0.5, or noise SDs by 0.5 |
| O3 | S1 | separately remove X→D or X→U |
| O4 | S1 | block X→D |
| O5 | S6 | false X→M with relative strengths 0, 0.5, 1; matched root noise |

Only these kernels are constructed. No simulation/refitting sweeps are invoked.
Schemas plus all four transition matrices determine aliases:
`O4/block_X_D` = `O3/drop_X_D`; `O5/c0` = `S6/true`.
Reports retain all 15 labels; fitting uses 13 distinct combinations.

Both editors refit per canonical combination: 65 fits each. Each flow fits once
per state space and seed: 30 fits each. Gaussian fits once per state space: six
fits, reused with all five g seeds. **190 neural + six analytical fits**, plus
20 independent-reader search trials. No hZ hyperparameter search.

## Training and inference

The package reuses native models, losses, kernel builders and atomic artifact
utilities. It does not modify the main pipeline, existing trainers, global
registries or historical checkpoint contracts.

Editors minimize the unchanged objective
`semantic KL + 0.1 movement + 0.05 support + identity`, equally averaged over
all four `do(X=k)` actions. Four backward passes precede each optimizer step.
Each action uses its factual-X=k identity mask and full exact target mixture.
The support bank contains 128 fitting embeddings sampled with fixed seed 42;
normalization is fitting-only. g is frozen but differentiable with respect to
edited embeddings. No proxies, counterfactual targets or teachers enter fitting.

All neural models use tier 1, within ±7% of 331,136 instantiated trainable
parameters; state flow retains width 32. Gaussian statistics/storage are
reported separately. Similar capacity does not imply similar compute.
Defaults: max 500 epochs, LR 3e-4, clipping 5, patience 30, min improvement 1e-7,
plateau patience 10/factor 0.5/min LR 1e-6. Editors use batch 32, flows 64;
family optimizers and OT solver settings remain native. Best validation weights
are restored. Neural histories include initialization (epoch 0) as an eligible
checkpoint, so a non-improving run is reported honestly.

At inference, one shared operation draws
`s ~ g(s|z); s' ~ hS(s'|s,do(X=k))`. Every method uses the same 64 IID symbolic
pairs for a given g/operator/action/unit. Per-ID hashed random streams and
one-row g inference make draws independent of outer batches and worker order.
Flows/Gaussian use the coupled pair; editors use only its target, recovering
their target marginal. Flows reuse the residual abducted under that source.
Repeated deterministic transports are cached without changing multiplicities.

Empty actions return Z exactly. Transports also bypass unchanged source–target
pairs; editors deliberately do not. Uncertain inferred states mean factual-X=k
units need not be exactly unchanged. Outputs are neither projected nor
renormalized. OT flow retains RK4 step doubling 64→128→256 and tolerance 1e-4;
numerical failures raise, rather than publishing partial predictions.

## Explicit evaluation only

Evaluation requires a complete, checksum-verified frozen selection manifest.
Pilot runs cannot evaluate the official test set. Every scenario covers total
targets; additional symbolic references are S1: ps_D/ps_DU, S3: ps_DU, and
S4/S5/O3-drop-X→D/O4: ps_D. All four action values are evaluated.

Own-g semantic KL measures consistency with the **assumed command**.
Independent-reader accuracy/NLL measure recovery of simulated counterfactual
attributes, including omitted ones; command accuracy/NLL are separate.
Preservation accuracy and factual-to-edited accuracy drops cover unchanged
attributes, including T/M/P/L/H/A. D/U preservation/change is judged against the
chosen total/path-specific truth, not assumed automatically.

The factual confusion reference Gamma is computed only during explicit
evaluation, outside P (pool minus P plus test = 2,800 units), pooling each head
by true level. Empty levels are unavailable. It is never a selection signal.
Head probability vectors are used directly; the nine-head joint is not
enumerated. The existing 200 same-state text-variation pairs and no-edit outputs
serve as separate references, not new targets or training data.

Oracle embedding energy/L2/cosine are computed **only** on matching
`(id, do_value, total)` pairs: there is one encoded nonidentity action per test
unit. Path-specific embedding metrics are unavailable, never zero. Energy uses
the existing unbiased 64-draw estimator, fitting-only coordinate SD (floor
1e-6), and sqrt-dimension scaling. Primary scores use the complete predictive
distribution. Exact posterior mass involving unseen states (0 observations)
or rare states (1–4 observations) is retained and reported per unit; separate
transition-stratum command diagnostics do not redefine primary scores.

Per-unit metrics support paired comparisons. Reports average actions within a
unit and seeds separately; five-seed SD and 95% paired-unit bootstrap intervals
(2,000 resamples, seed 20260930) are distinguished. Gaussian seed variation is
reader/draw variation, not five independently trained Gaussian models.
Single-seed/pilot evidence is labeled accordingly. Results are not guarantees
of individual-counterfactual identification or attribute preservation.

## Commands and persistence

Run from the repository root using the existing environment:

```bash
# Preparation: no optimizer steps or official-test predictions.
.venv/bin/python -m exp.benchmarks.misspecification.run preflight
.venv/bin/python -m exp.benchmarks.misspecification.run status

# When authorized: detached, resumable four-worker queue (never auto-evaluates).
bash scripts/run_hz_misspecification.sh hz-misspec-v1

# Optional separate 50-epoch S1/S3/S6, seed-42 pilot (+ separate reader search).
bash scripts/run_hz_misspecification.sh hz-misspec-v1 pilot

# Individual subsets, using the same prepared protocol.
.venv/bin/python -m exp.benchmarks.misspecification.run train --method semantic_editor --state-space S3 --seed 42
.venv/bin/python -m exp.benchmarks.misspecification.run train --operator O4/block_X_D
.venv/bin/python -m exp.benchmarks.misspecification.run train-reader

# Only after all selections freeze, and with explicit evaluation authorization:
.venv/bin/python -m exp.benchmarks.misspecification.run evaluate
.venv/bin/python -m exp.benchmarks.misspecification.run report
```

All commands accept `--config`, `--run-id`, `--pilot`; fitting/evaluation support
repeatable `--method`, `--state-space`, `--operator`, `--seed` selectors.
`train --all` executes the full prepared queue and evaluation-reader search.
The launcher runs preflight first, then detaches this command in tmux; default
four CPU workers, one numerical thread each. Failed jobs preserve successful
ones and epoch snapshots; rerunning verifies completed jobs before skipping.

Models:
`models/talent_motivation/embeddinggemma_768/g-independent/S#/h-<method>/runs/<run>/<fitting-operator>/seed-<n>/selected/latent_intervention.pt`.
Flows/Gaussian use fitting operator `factual`; evaluations have separate
condition identities. The wrapper format `hz_misspecification_v1` includes
native architecture/weights/normalization, schema, supported actions and
provenance. Native legacy do(X=3) checkpoints are not widened or reinterpreted.

Protocol, input/source hashes, persisted split IDs, support, histories, frozen
selections, status and compact metrics live in
`reports/talent_motivation/hz_misspecification/<run>/`.
Atomic snapshots retain current/best weights, optimizer/scheduler/RNG state.
Exclusive locks reject concurrent writers; changed signatures require a new
run ID. Development-only preflight performs numerical forward/backward checks
but no optimizer step, Gaussian PCA/statistical fit, or official-test scoring.

Git exceptions publish only seed-42 hZ deployments, the evaluation reader and
compact records for all seeds. Other model seeds, optimizer snapshots, locks,
logs and per-unit tensors stay local/ignored; transfer them explicitly when
continuing analysis on another machine. Existing five-seed g inputs must also
be available there. Preparation does not stage, commit or push anything.
