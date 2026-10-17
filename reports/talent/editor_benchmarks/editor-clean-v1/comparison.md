# Clean editors: counterfactual evaluation and benchmark comparison

Completed on 2026-10-17. All 20 validation-selected clean-editor checkpoints were evaluated; no retraining, test-driven checkpoint selection, or preservation-reader fitting was performed.

## Assessment

- The flexible `semantic_editor` is the stronger of the two clean editors under this protocol: lower energy, semantic KL and realized-state NLL, higher joint-state accuracy, and substantially less identity drift at both dimensions. The convex-gradient restriction did not improve these outcomes; this does not establish that every possible OT-structured editor must fail.
- `semantic_editor` does not dominate the flow approaches. `direct_semantic_flow` retains the lowest semantic KL at both dimensions. At 256D, `distilled_flow` and the single-seed `ot_flow` have energy near 0.420, versus 0.49657 for `semantic_editor`. At 768D, single-seed `ot_flow` has energy 0.43475, versus 0.51162 for `semantic_editor`.
- Relative to `direct_semantic_flow`, the semantic editor has slightly lower **all-unit** energy and much smaller identity drift. The all-unit energy advantage is largely an identity effect: nonidentity energy is nearly equal at 256D (0.66153 versus 0.66330) and slightly worse at 768D (0.68202 versus 0.67645). This is not evidence of uniformly better edits on units that require a change.
- The 768D semantic editor has the highest observed mean joint-state accuracy in this comparison (61.56%), versus 60.00% for direct semantic flow. At 256D the ordering reverses (57.96% versus 58.54%). These are descriptive means on one shared test set, not significance claims or a universal dimensionality ranking.
- The new `ot_editor` has worse **all-unit** energy than the no-edit reference at both dimensions (0.62051 versus 0.58770; 0.63355 versus 0.60097), despite greatly improved symbolic consistency. Identity shifts around 0.11 contribute to this weakness. It should be presented as a negative result for the current matched design, not as an improved editor.

Practical conclusion: retain `semantic_editor` as the clean pipeline representative and `ot_editor` as its matched OT-structure ablation. The current evidence supports a recovery/consistency/identity trade-off, not a claim to beat every baseline or to solve individual preservation.

## Common evaluation protocol and fairness limits

- Same 1,000 official test IDs in the same order: 256 identity and 744 nonidentity units. Identity here is the dataset's oracle identity flag, not the empty-action API.
- Same 3,200 fitting IDs determine the energy scale: coordinate standard deviations with `unbiased=False`, floor `1e-6`, distances divided by `sqrt(d)`. The unbiased off-diagonal sample energy estimator is unchanged.
- Same frozen per-dimension independent g, analytic h_S, and `do(X=3)`. Latent-only inference only; the state-flow observed-state regime is deliberately excluded.
- 64 IID draws per stochastic model/unit, seed 20260930, batches of 32. Deterministic baseline, oracle regression and no-edit are point masses. A shared seed does not imply identical random variates across different sampling implementations.
- Original benchmark records use the same metric function: its parsed function body/decorators match the historical source snapshot. Existing data, g, checkpoint and signed record hashes were checked before reuse.
- Five training seeds (42–46) for both clean editors and each original benchmark. Gaussian OT and OT flow have one stored seed/run only. Means ± sample SD below describe training-seed variability, **not** test-population confidence intervals. The no-edit reference is deterministic, not one trained seed.
- Training conditions are not identical across all families. Original methods searched eight learning-rate/tier candidates and used 800 tuning units; clean editors used the predeclared tier-1 protocol with 400 tuning units and left 400 reserved-audit units unused. Gaussian OT/OT flow also use the 400/400 split. Some original winners are tier 2.
- Supervision differs: `baseline` and `dist` use paired counterfactual symbolic targets; `distilled_flow` uses a learned state-flow teacher; `oracle_regression` receives privileged paired training Z′. Clean editors use no proxy labels, teacher or paired counterfactual training targets. The two stored standalone OT methods use factual supervision, with auxiliary attributes confined to separate post-selection audits.
- Semantic KL measures agreement with g/h_S, not oracle recovery. Joint accuracy/NLL compare the g-decoded mixture to realized visible S′; these are still g-mediated measurements, not independent reconstruction or preservation audits. No P/L/H/A assessment was added for the clean editors.
- Earlier test results were already viewed during method development. These comparisons are exploratory; no new checkpoint or hyperparameter was selected using this evaluation.

## All test units: mean ± training-seed SD

Energy and semantic KL: lower is better. Joint S′ accuracy: higher is better. `n=1` rows have no estimated seed variability; `ref` is the unfitted no-edit reference.

### 256D

| Method | n | Selected tier | Energy ↓ | Semantic KL ↓ | Joint S′ accuracy (%) ↑ |
|---|---:|---:|---:|---:|---:|
| `semantic_editor` | 5 | 1 | 0.49657 ± 0.00243 | 0.03132 ± 0.00037 | 57.96 ± 0.11 |
| `ot_editor` | 5 | 1 | 0.62051 ± 0.00150 | 0.08403 ± 0.00196 | 57.40 ± 0.27 |
| `direct_semantic_flow` | 5 | 1 | 0.50754 ± 0.00675 | 0.02206 ± 0.00097 | 58.54 ± 0.27 |
| `state_flow` | 5 | 2 | 0.44591 ± 0.00145 | 0.31240 ± 0.02123 | 52.48 ± 0.40 |
| `distilled_flow` | 5 | 2 | 0.41983 ± 0.00408 | 0.82368 ± 0.06569 | 51.20 ± 0.84 |
| `ot_flow` | 1 | 1 | 0.41996 | 0.16054 | 55.10 |
| `ot_gaussian` | 1 | — | 0.44280 | 0.09236 | 56.10 |
| `particles` | 5 | 1 | 0.56522 ± 0.00668 | 0.07067 ± 0.00556 | 58.42 ± 0.36 |
| `dist` | 5 | 2 | 0.67602 ± 0.04496 | 0.14541 ± 0.04851 | 58.42 ± 0.58 |
| `pre_additive` | 5 | 2 | 0.78126 ± 0.11719 | 0.12121 ± 0.00557 | 55.92 ± 1.04 |
| `noise_token` | 5 | 1 | 0.75791 ± 0.04729 | 0.10756 ± 0.00569 | 58.00 ± 0.58 |
| `baseline` | 5 | 1 | 1.00171 ± 0.05455 | 0.37189 ± 0.00892 | 52.00 ± 0.83 |
| `oracle_regression` | 5 | 2 | 0.35570 ± 0.00101 | 0.95674 ± 0.00940 | 45.54 ± 0.93 |
| `no_edit` | ref | — | 0.58770 | 13.26688 | 20.00 |

### 768D

| Method | n | Selected tier | Energy ↓ | Semantic KL ↓ | Joint S′ accuracy (%) ↑ |
|---|---:|---:|---:|---:|---:|
| `semantic_editor` | 5 | 1 | 0.51162 ± 0.00531 | 0.02864 ± 0.00061 | 61.56 ± 0.19 |
| `ot_editor` | 5 | 1 | 0.63355 ± 0.00483 | 0.07724 ± 0.00284 | 60.42 ± 0.31 |
| `direct_semantic_flow` | 5 | 1 | 0.51541 ± 0.00345 | 0.02694 ± 0.00110 | 60.00 ± 0.31 |
| `state_flow` | 5 | 2 | 0.49372 ± 0.00700 | 0.75082 ± 0.48995 | 49.60 ± 6.70 |
| `distilled_flow` | 5 | 2 | 0.48413 ± 0.00771 | 0.93216 ± 0.48048 | 49.00 ± 6.37 |
| `ot_flow` | 1 | 1 | 0.43475 | 0.10718 | 61.00 |
| `ot_gaussian` | 1 | — | 0.45960 | 0.06736 | 60.60 |
| `particles` | 5 | 1 | 0.64688 ± 0.00606 | 0.10859 ± 0.00888 | 61.36 ± 0.34 |
| `dist` | 5 | 1 | 0.71826 ± 0.00701 | 0.18015 ± 0.00415 | 61.14 ± 0.39 |
| `pre_additive` | 5 | 1 | 1.58614 ± 0.55164 | 0.06279 ± 0.01166 | 60.56 ± 0.73 |
| `noise_token` | 5 | 1 | 1.85680 ± 0.80659 | 0.08457 ± 0.01437 | 60.76 ± 0.69 |
| `baseline` | 5 | 2 | 1.54818 ± 0.32612 | 0.29281 ± 0.01922 | 55.54 ± 1.06 |
| `oracle_regression` | 5 | 2 | 0.36851 ± 0.00038 | 0.80829 ± 0.02316 | 52.64 ± 0.63 |
| `no_edit` | ref | — | 0.60097 | 13.20760 | 21.50 |

## Identity versus nonidentity

The identity shift is the mean raw L2 displacement of individual draws from their factual embedding. It is not an auxiliary-attribute preservation score. Nonidentity energy is essential because 25.6% of test units require no change.

| Dimension | Method | Identity energy ↓ | Identity shift L2 ↓ | Nonidentity energy ↓ | Nonidentity semantic KL ↓ | Nonidentity joint accuracy (%) ↑ |
|---:|---|---:|---:|---:|---:|---:|
| 256 | `semantic_editor` | 0.01716 ± 0.00070 | 0.01071 ± 0.00040 | 0.66153 ± 0.00304 | 0.03909 ± 0.00052 | 51.10 ± 0.20 |
| 256 | `ot_editor` | 0.17924 ± 0.01378 | 0.10703 ± 0.00794 | 0.77234 ± 0.00426 | 0.09442 ± 0.00347 | 50.81 ± 0.25 |
| 256 | `direct_semantic_flow` | 0.05488 ± 0.00339 | 0.04457 ± 0.00532 | 0.66330 ± 0.00838 | 0.02542 ± 0.00126 | 52.18 ± 0.24 |
| 256 | `state_flow` | 0.00414 ± 0.00014 | 0.00242 ± 0.00007 | 0.59791 ± 0.00192 | 0.39897 ± 0.02748 | 43.79 ± 0.53 |
| 256 | `distilled_flow` | 0.00321 ± 0.00066 | 0.00251 ± 0.00041 | 0.56318 ± 0.00536 | 1.08004 ± 0.09132 | 42.04 ± 1.12 |
| 256 | `ot_flow` | 0.00278 | 0.00526 | 0.56351 | 0.20670 | 47.18 |
| 256 | `ot_gaussian` | 0.00289 | 0.00485 | 0.59417 | 0.11670 | 48.52 |
| 256 | `no_edit` | 0.00000 | 0.00000 | 0.78991 | 17.78664 | 0.27 |
| 768 | `semantic_editor` | 0.01640 ± 0.00039 | 0.01073 ± 0.00017 | 0.68202 ± 0.00717 | 0.03501 ± 0.00085 | 54.03 ± 0.19 |
| 768 | `ot_editor` | 0.18102 ± 0.00483 | 0.11546 ± 0.00322 | 0.78926 ± 0.00766 | 0.08690 ± 0.00342 | 52.80 ± 0.35 |
| 768 | `direct_semantic_flow` | 0.04739 ± 0.00137 | 0.03909 ± 0.00107 | 0.67645 ± 0.00421 | 0.03216 ± 0.00147 | 52.07 ± 0.48 |
| 768 | `state_flow` | 0.00344 ± 0.00035 | 0.00225 ± 0.00021 | 0.66242 ± 0.00950 | 0.99778 ± 0.65808 | 37.85 ± 8.96 |
| 768 | `distilled_flow` | 0.00250 ± 0.00045 | 0.00222 ± 0.00019 | 0.64985 ± 0.01044 | 1.23379 ± 0.64480 | 37.02 ± 8.54 |
| 768 | `ot_flow` | 0.00201 | 0.00376 | 0.58365 | 0.13664 | 53.23 |
| 768 | `ot_gaussian` | 0.00273 | 0.00484 | 0.61680 | 0.08370 | 52.69 |
| 768 | `no_edit` | 0.00000 | 0.00000 | 0.80775 | 17.71683 | 0.27 |

## Actual selected capacity

Counts exclude frozen g and any separate teacher. Comparable same-tier parameter budgets do not imply identical computation. Gaussian OT has fitted statistical buffers rather than trainable neural parameters; its reported zero is not zero model storage.

| Method | 256D parameters (tier) | 768D parameters (tier) |
|---|---:|---:|
| `semantic_editor` | 199,552 (1) | 331,136 (1) |
| `ot_editor` | 199,308 (1) | 330,154 (1) |
| `direct_semantic_flow` | 192,144 (1) | 310,672 (1) |
| `state_flow` | 648,144 (2) | 934,096 (2) |
| `distilled_flow` | 672,912 (2) | 902,928 (2) |
| `ot_flow` | 199,551 (1) | 330,559 (1) |
| `ot_gaussian` | 0 (analytical) | 0 (analytical) |
| `particles` | 201,729 (1) | 333,313 (1) |
| `dist` | 641,229 (2) | 332,773 (1) |
| `pre_additive` | 660,992 (2) | 331,136 (1) |
| `noise_token` | 200,192 (1) | 331,776 (1) |
| `baseline` | 199,552 (1) | 923,648 (2) |
| `oracle_regression` | 655,008 (2) | 929,696 (2) |

## Records and reproducibility

- [Clean-editor per-seed report and all metrics](summary.md).
- [Machine-readable aggregates and SHA-256 source manifest](comparison.json): all/identity/nonidentity metrics, per-group seeds/capacity, 114 referenced evaluation records, and verified source/checkpoint hashes.
- [Original nine-method report](../../hz_benchmarks/hz-gemma-v1/summary.md), reused without edits.
- [Method definition and commands](../../../../docs/experiments/semantic_editor.md).

Sources (paths relative to the repository root):

```text
reports/talent/embeddinggemma_D/g-independent/h-V/runs/editor-clean-v1/seed-N/evaluation.json
reports/talent/embeddinggemma_D/g-independent/h-V/runs/editor-clean-v1/seed-N/training.json
reports/talent/hz_benchmarks/hz-gemma-v1/evaluation/<case>/seed-N.json
reports/talent/hz_benchmarks/hz-gemma-v1/fits/<case>/final/seed-N/training.json
reports/talent/embeddinggemma_D/g-independent/h-ot_gaussian/runs/ot-gemma-v2/evaluation.json
reports/talent/embeddinggemma_D/g-independent/h-ot_flow/runs/ot-gemma-v2/evaluation.json
```

Twenty new evaluations and 94 existing evaluations are included. All 114 checkpoint files match their recorded hashes; signed evaluation records validate, subgroup means reproduce the all-unit aggregates, and all metrics are finite. Existing benchmark models/results were only read, not rewritten. Model weights and training records remain unchanged.

Aggregation: for each method/dimension/subset, average the stored per-seed unit-mean metrics, and report sample SD across the available training seeds. Do not pool seed×unit observations as independent test examples. Full-precision aggregates and input hashes are in `comparison.json`; displayed tables are rounded.

