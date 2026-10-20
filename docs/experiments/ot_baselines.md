# Standalone factual-only OT baselines

`ot_gaussian` and `ot_flow` remain independent non-oracle baselines for
EmbeddingGemma 256D/768D. Their models, histories and existing results are preserved.
They do not change the original nine-method benchmark membership or defaults.
The target-only `ot_editor` belongs to the separate
[clean semantic-editor workflow](semantic_editor.md), which also contains the
single account of retired methods and historical comparisons.

## Gaussian transport

`ot_gaussian` fits coordinate standardization and a shared **full-dimensional**
PCA basis on factual fitting embeddings; no dimensions are discarded. Within
that basis it estimates a diagonal Gaussian for each visible state. Sparse and
absent states pool toward categorical main/pairwise-effect ridge mean predictions
(ridge 1) and pooled residual variance. The shrinkage weight is `n/(n+32)` and
the variance floor is `1e-4`. The analytical transport in those coordinates is

$$T_{s,s'}(x)=\mu_{s'}+\frac{\sigma_{s'}}{\sigma_s}\odot(x-\mu_s).$$

This is an explicit restricted Gaussian approximation, not a full neural OT fit.
Its statistical dimensions and buffer storage are reported; it has no trainable
neural parameters. Fitting is analytical, with factual validation likelihood and
no fictitious epoch selection.

## Shared-reference OT flow

`ot_flow` learns one state-conditioned continuous-time flow $F_s$ through
conditional minibatch OT flow matching. Independent Gaussian reference samples
and factual embeddings are coupled **within each fixed state**, never across
mixed state labels. A two-hidden-layer SiLU velocity network uses state embeddings
and 16 fixed sinusoidal time features. Small Gaussian smoothing has standard
deviation 0.01 in standardized coordinates. The fitting objective does not use g,
h_S, auxiliary labels or paired counterfactual embeddings.

Prediction abducts and reuses the same reference variable:

$$u=F_s^{-1}(x),\qquad x'=F_{s'}(u).$$

It does not resample u for rendering. State-specific reference alignment is an
inductive assumption, not identification of true individual counterfactuals.
Conditional minibatch OT couplings do not guarantee globally optimal endpoint
transport after finite neural fitting or between every pair of states.

Log-domain Sinkhorn uses dimension-normalized squared distance, regularization
0.05 and marginal tolerance `1e-6`; the configured ceiling is 100,000 iterations
with early convergence. Failure to converge raises an error. PyTorch RK4 starts
at 64 integration steps and checks step doubling up to 256 at tolerance `1e-4`;
failure is explicit and function-evaluation counts are recorded. Outputs are not
projected or unit-renormalized, which would alter transport/invertibility.

## Inference, split and audit

Both methods use all positive coupled weights
$q(s,s'|z,a)=g(s|z)h_S(s'|s,a)$ in latent-only prediction. No top-k or independent
source/target draws are substituted. Empty actions and equal source/target states
return the embedding exactly; only `do(X=3)` and empty actions are supported.
Missing-state transitions remain available and are explicitly extrapolations.
Samples have shape `(samples, batch, dimension)`.

The official training IDs yield 3,200 fitting / 400 tuning / 400 audit units
(split seed 42, audit split seed 20261007); the 1,000 official test units are
excluded from fitting and selection. Factual fitting exposes only Z, S and IDs.
Separate optional audit readers use P/L/H/A **after selection**, never as OT
training inputs or gates. Existing reader signatures are unchanged by their
byte-identical relocation to `src/latent_intervention/audit_probes.py`.

Flow defaults are tier 1, seed 42, LR `3e-4`, batch 64, 500 maximum epochs,
patience 30, clipping 5 and a plateau scheduler. An epoch has `ceil(n_fit/64)`
outer steps. Fixed-randomness held-out flow-matching MSE selects/restores the
checkpoint. Tier targets are 199,552/660,992 parameters at 256D and
331,136/923,648 at 768D; instantiated flow counts must lie within ±7%.
Matching capacity does not imply matching compute. See `configs/hz_ot.yaml`.

## Commands and artifacts

```bash
# Read-only preflight for two methods at two dimensions.
.venv/bin/python -m src.latent_intervention.ot preflight --all --run-id NEW_OT_RUN

# Four sequential detached fits; does not evaluate the test set.
bash scripts/run_hz_ot.sh NEW_OT_RUN

# Individual fitting, separate optional audit, explicit test evaluation.
.venv/bin/python -m src.latent_intervention.ot train \
  --variant ot_flow --dimension 768 --run-id NEW_OT_RUN
.venv/bin/python -m src.latent_intervention.ot audit \
  --variant ot_flow --dimension 768 --run-id NEW_OT_RUN
.venv/bin/python -m src.latent_intervention.ot evaluate \
  --variant ot_flow --dimension 768 --run-id NEW_OT_RUN
```

Use a fresh run ID for changed source/protocol signatures. Existing `ot-gemma-v2`
deployments and records can still be loaded without retraining. Their historical
source hashes/settings remain historical, not rewritten to claim a new run.
Checkpoint format `latent_ot_v1` remains supported for these two variants.

Models live under
`models/talent/embeddinggemma_D/g-independent/h-V/runs/R/selected/`;
compact records live under the analogous `reports/` root. Selected models,
metadata and separate audit readers are trackable. Resume snapshots and logs
remain ignored. Atomic writes, exclusive run locks and signature checks prevent
conflicting writers or silent overwrites.

Explicit evaluation uses the shared common metrics, 64 draws, seed 20260930 and
fitting-only energy normalization. Report g-consistency, realized visible-state
accuracy and oracle-embedding recovery separately, for all/identity/nonidentity
units. The existing OT results are single-seed; the original benchmark has its own
tuning and multi-seed protocol. Previously viewed test results are exploratory.
No new fitting or test evaluation is part of editor consolidation.

The module split is `gaussian.py` / `flow.py` (models), `transport.py` (Sinkhorn/RK4),
`protocol.py` (factual boundary), `training.py` (fitting), `evaluation.py` (separate
audit/evaluation) and `workflow.py` / `__main__.py` (thin adapters). Active registry
and launcher membership contain only Gaussian OT and OT flow.
