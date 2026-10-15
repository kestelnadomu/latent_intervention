# Projected C-sym: supported method and experiment record

The supported variant is **`c_sym_project`**, formerly `state_effect/project`.
It is **not** the older target-only `c_sym/project` model. Both EmbeddingGemma
256D and 768D checkpoints are retained; 768D is our lead configuration. These
are the original seed-42 fits, repackaged on 2026-10-13, not retrained models.

## Method and guarantees

For a factual embedding Z, use the frozen semantic decoder and analytic SCM to
form the coupled weights

\[
q(s,s'\mid Z,a)=g(s\mid Z)h_S(s'\mid s,a).
\]

Every supported transition is retained: 252 edges for the four-node talent SCM
and `do(X=3)`. Source and target states are not sampled independently; no top-k
approximation or learned mixture-weight network is used. Conditional on an edge,

\[
\widehat Z'=Z+\Pi\{D_e[m_\phi(s')-m_\phi(s)]\}.
\]

The state-effect MLP m predicts standardized factual embeddings from categorical
states. D_e is the diagonal fit-only embedding standard deviation (`unbiased=False`,
floor 1e-4). The fixed projection acts in the preservation guide's standardized
coordinates: Pi(v) = D_g (I - B B^T) D_g^-1 v. B is an orthonormal basis spanning
the centered guide weight rows; both published bases have rank 8. The guide uses
its own fit-only scale (`unbiased=True`, floor 1e-4). No correction network is used.

- The guide's predicted probabilities for P/L/H/A remain unchanged to floating-point
  precision; audits enforce maximum absolute drift <= 2e-5.
- For an unchanged component S=S', the output is exactly Z. An empty intervention
  also returns Z exactly. A unit whose true X=3 can still move if g is uncertain
  about its factual state: component identity is not true-unit identity.
- This does not guarantee correct real attributes, exact decoded S', oracle Z'
  recovery, unit output norm, text decodability or preservation of every individual
  detail. State-effect differences learned from factual data are not automatically
  identified causal effects. The retained residual is a modeling assumption.
- Only the trained `do(X=3)` and the empty action are supported. Other interventions
  are rejected. g and h_S are required at inference; observed test states are not.

The runtime package separates model/persistence, probes, objective, training,
factual data/provenance, evaluation and the pipeline adapter. The existing nine
benchmark families and their defaults are unchanged. The main pipeline contains
only routing hooks; there is no dependency on the retired experiment packages.

## Use

Run from the repository root. Evaluation does not train or replace model weights:

```bash
.venv/bin/python -m src.latent_intervention.csym evaluate --dimension 768
.venv/bin/python -m src.latent_intervention.csym evaluate --all-dimensions
```

The existing pipeline also supports the new variant:

```bash
.venv/bin/python -m src.pipeline evaluate --encoder-variant embeddinggemma --embeddinggemma-dim 768 --decoder-variant independent --manipulator-variant c_sym_project
```

For future fitting, use a fresh output tag. Published migrated weights cannot be
overwritten by the trainer. The single configuration is `configs/hz_csym_project.yaml`.

```bash
.venv/bin/python -m src.latent_intervention.csym train --dimension 768 --tag csym-new-run
bash scripts/run_hz_csym_project.sh train csym-new-run
```

The detached launcher queues 768D then 256D with one CPU numerical thread. It
does not evaluate official-test targets during training. Logs are under
`reports/talent/csym_logs/`; an interruption in joint training resumes from an
atomic epoch snapshot, including optimizer, scheduler, RNG and best-model state.
Interrupted factual warmup and reader fitting restart deterministically. Reusing
a completed fit checks provenance rather than fitting again. Changed inputs,
configuration or source require a fresh tag. Fresh-tag outputs are local by default.

For Python inference, load with `src.latent_intervention.csym.ProjectedCSym.load`
or the shared `load_latent_intervention` helper, then use `sample_symbolic` or
`sample_counterfactual`, passing frozen g, h_S and `{"X": 3}`. Outputs have shape
`(samples, batch, dimension)`. Projection is internal in every public prediction
path. Loading and sampling do not need the audit-reader file or the local archive.
The supplied pipeline workflow is CPU float32; no encoder is loaded or rerun.

## Training and evaluation protocol

Training uses only factual embeddings and factual labels. Neither paired training
Z', paired S' labels nor a state-flow teacher enters the objective. Guide fitting
uses factual P/L/H/A labels; this additional supervision must be disclosed.

The unchanged split is 3,200 fit, 400 tuning and 400 independent-audit units from
the 4,000 official training units; 1,000 official test units are evaluation-only.
These splits can be recreated from the official IDs and seeds without pilot plans.
Frozen g previously used the wider validation pool, so the 400-unit audit is not
a fresh end-to-end holdout.

State effects receive up to 200 factual-warmup epochs (patience 20), followed by
up to 500 joint epochs (patience 30), batch size 32, AdamW lr 3e-4 and weight decay
1e-5. Joint training retains factual MSE (weight 1), exact-mixture semantic KL,
movement (0.1), factual-neighbor support (0.05), identity (1) and radial (0.1)
penalties. Entropy weight is zero. The scheduler halves learning rate after
patience 5, with floor 1e-5. Validation first requires semantic KL <= 0.15,
then minimizes guide drift + identity cost + 0.1 movement. Failed feasibility
is reported, not hidden. The best validation checkpoint is restored.

Independent per-attribute readers are fitted separately from the guide, with
their own linear warm starts and optional nonlinear corrections. A separate
nonlinear-trunk stress reader is retained as a lower-confidence diagnostic.
Neither reader contributes gradients or selection scores to the editor. Guide
and primary auditor require at least 0.60 tuning balanced accuracy per protected
attribute. Reader quality and linear fallbacks accompany the audit results.

Paired evaluation uses the original 1,000 ordered test IDs, 64 IID samples per
unit, batch size 32 and evaluation seed 20260930. Energy uses fit-only coordinate
scales (`unbiased=False`, floor 1e-6), sqrt(d) normalization and the off-diagonal
unbiased sample estimator. Apply g to each draw, then mix its probabilities.
Exact finite-mixture scores are supplementary and must not be compared directly
with sampled KL. Independent audit-reader KL measures its consistency with its
own SCM-updated factual posterior, not ground-truth S' accuracy.

## Published artifacts and migration

For each D in 256, 768:

```text
models/talent/embeddinggemma_D/g-independent/h-c_sym_project/
  latent_intervention.pt       # network, guide, projection, normalization, provenance
  latent_intervention.info.json
  audit_readers.pt             # independent readers; not needed for inference
reports/talent/embeddinggemma_D/g-independent/h-c_sym_project/
  protocol.json               # exact IDs, settings, hashes and original provenance
  training.json               # original training record, unchanged bytes
  probes.json                 # original reader training record, unchanged bytes
  audit.json                  # independent factual audit and no-edit references
  evaluation.json             # paired metrics, per-unit scores and exact diagnostics
  original_evaluation.json    # original selected-case evaluation, unchanged bytes
  migration.json              # source/destination hashes and parity checks
```

The embedded origin identifies `hz-csym-transition-v1`, original
`state_effect/project`, its probe source and `hz-csym-transition-paired-v1`.
Paths in original provenance are historical references, not live dependencies.
Migration checked all 1,000 test units at each dimension: component locations,
weights and seeded samples were bitwise identical; per-unit paired scores had
maximum difference 0, and independent audits reproduced. No embeddings, g models
or fitted editor weights were retrained. Published files are Git-trackable;
optimizer states, progress, logs, new-tag runs and archives remain ignored.

## Experiment history — retained here once

An initial preservation pilot compared target-only C-sym, particles and direct
semantic flow under `none`, `soft` and `project` controls at both dimensions.
Reader-gate preflights v1/v2 did not train editors. Pilot v3 completed; its budget
was 80 epochs and should not be confused with the later 500-epoch experiment.

The standalone follow-up compared three architectures: target-only `c_sym`,
factual `state_effect`, and `state_effect_correction`. Each used the three controls.
The following are the 18 follow-up fits on the common paired evaluation; lower
energy and KL are better. KL here is the **64-draw** estimate.

| Historical architecture | Control | Energy 256D | KL 256D | Energy 768D | KL 768D |
|---|---|---:|---:|---:|---:|
| c_sym | none | 0.49420 | 0.06938 | 0.51418 | 0.06761 |
| c_sym | soft | 0.49840 | 0.05181 | 0.51488 | 0.05281 |
| c_sym | project | 0.49013 | 0.06167 | 0.50392 | 0.05560 |
| state_effect | none | 0.45237 | 0.04535 | 0.46830 | 0.04194 |
| state_effect | soft | 0.45147 | 0.04410 | 0.46832 | 0.03937 |
| **state_effect** | **project (retained)** | **0.45151** | **0.04494** | **0.46829** | **0.03991** |
| state_effect_correction | none | 0.45221 | 0.04699 | 0.46786 | 0.04254 |
| state_effect_correction | soft | 0.45815 | 0.03518 | 0.47464 | 0.03315 |
| state_effect_correction | project | 0.45398 | 0.03643 | 0.47136 | 0.03368 |

Validation chose `soft` for target-only C-sym and `project` for both state-effect
architectures. We retained projected state effects for their explicit modeling
assumption, guide invariance and favorable recovery among validation-selected
variants. The correction improved KL but slightly worsened energy. Projection
itself barely changes energy, especially at 768D; it is not established as the
main cause of the recovery gain. The retained model is not the best test score
in every row or metric, and true individual preservation remains unresolved.

Compared within dimension, original distilled-flow energy is 0.41594 at 256D
and 0.48939 at 768D: the retained editor wins at 768D, not 256D. Direct semantic
flow has lower sampled semantic KL at both dimensions. Historical families had
different budgets, parameter counts and supervision. The comparison used seed
42, not a five-seed average. Checkpoints/modes were fixed before their paired
evaluation, but the subsequent decision to pursue this method was informed by
those test results. Treat this as exploratory; new confirmatory claims need fresh
data or a genuinely held-out protocol. No novelty or causal-identification claim
follows merely from these results.

All retired source, configurations, histories and model variants are recoverable
locally in `archives/csym-experiments-2026-10-13/`. Its `archive.json` lists 798
checksum-verified pre-cleanup files, including untracked files; `tracked.patch`
records the earlier tracked-file changes. The archive is deliberately not pushed,
and will not be available in a clean clone unless transferred separately. The
supported model, its training/evaluation code and published evidence do not need it.
The `conversion/` snapshot preserves the source used for initial checkpoint export;
`migration.json` separately records verification against the final cleaned source.
