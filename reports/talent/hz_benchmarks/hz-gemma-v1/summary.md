# h_Z benchmark

Status: prepared; no training or target encoding started.

18 cases; 8 search trials each; final seeds [42, 43, 44, 45, 46]. Up to 500 epochs, patience 30; best validation weights restored.

Split: 3200 fit / 800 validation / 1000 official test. g is frozen; h_S is analytic.

## Protocol

- Hyperparameters and early stopping use each family's declared validation metric. These unlike objectives are not ranked across families.
- Dist: fixed pretraining, then validation-controlled joint fitting within the total epoch ceiling.
- State-flow search winners supply search teachers; final distillation uses the same-seed final state-flow teacher. Teachers fit only fit IDs.
- All choices are frozen before test scoring. Five seeds measure training variability, not independent test datasets or confidence intervals.
- Stochastic test evaluation: 64 draws/unit, batches of 32; deterministic methods are point masses.
- Energy score uses fit-only per-coordinate standard deviations, divides distances by sqrt(d), and uses the unbiased off-diagonal sample estimator (it can be slightly negative).
- Raw MSE is not dimension-comparable by itself. Compare normalized energy/recovery, semantics, identity behavior, and the no-edit reference together.
- g-based metrics are descriptive and may favor models trained through g. A paired counterfactual sample cannot establish full distributional calibration.
- State-aware results are separate from latent-only results. The oracle is a privileged supervised reference, not a guaranteed upper bound.
- Frozen g was tuned on the same inner validation split; this is not independent nested end-to-end validation. The official test remains held out from h_Z selection.
- Trials restart after interruption; completed checkpoint/report pairs are verified and reused. No canonical artifacts are replaced.

## Cases

| Case | Validation criterion | Supervision |
|---|---|---|
| embeddinggemma_256/g-independent/h-baseline | counterfactual_state_nll | paired counterfactual structured S'; no training Z' |
| embeddinggemma_256/g-independent/h-pre_additive | semantic_forward_kl | frozen g and analytic h_S; no paired counterfactual targets |
| embeddinggemma_256/g-independent/h-noise_token | semantic_forward_kl | frozen g and analytic h_S; no paired counterfactual targets |
| embeddinggemma_256/g-independent/h-dist | semantic_forward_kl | paired counterfactual S' in pretraining; frozen g and h_S; no training Z' |
| embeddinggemma_256/g-independent/h-particles | semantic_forward_kl | frozen g and analytic h_S; no paired counterfactual targets |
| embeddinggemma_256/g-independent/h-state_flow | factual_conditional_nll_per_coordinate | factual (Z,S); observed-S and inferred-S inference reported separately |
| embeddinggemma_256/g-independent/h-distilled_flow | teacher_energy_distance | factual (Z,S), fitted state-flow teacher, analytic h_S |
| embeddinggemma_256/g-independent/h-direct_semantic_flow | semantic_forward_kl | frozen g and analytic h_S; no paired counterfactual targets |
| embeddinggemma_256/g-independent/h-oracle_regression | counterfactual_latent_mse | privileged paired training Z'; no g or h_S in fitting |
| embeddinggemma_768/g-independent/h-baseline | counterfactual_state_nll | paired counterfactual structured S'; no training Z' |
| embeddinggemma_768/g-independent/h-pre_additive | semantic_forward_kl | frozen g and analytic h_S; no paired counterfactual targets |
| embeddinggemma_768/g-independent/h-noise_token | semantic_forward_kl | frozen g and analytic h_S; no paired counterfactual targets |
| embeddinggemma_768/g-independent/h-dist | semantic_forward_kl | paired counterfactual S' in pretraining; frozen g and h_S; no training Z' |
| embeddinggemma_768/g-independent/h-particles | semantic_forward_kl | frozen g and analytic h_S; no paired counterfactual targets |
| embeddinggemma_768/g-independent/h-state_flow | factual_conditional_nll_per_coordinate | factual (Z,S); observed-S and inferred-S inference reported separately |
| embeddinggemma_768/g-independent/h-distilled_flow | teacher_energy_distance | factual (Z,S), fitted state-flow teacher, analytic h_S |
| embeddinggemma_768/g-independent/h-direct_semantic_flow | semantic_forward_kl | frozen g and analytic h_S; no paired counterfactual targets |
| embeddinggemma_768/g-independent/h-oracle_regression | counterfactual_latent_mse | privileged paired training Z'; no g or h_S in fitting |

## Model sizes

Actual trainable h_Z parameters, excluding frozen g and separate distillation teachers. Widths are explicitly set per dimension/family/tier in the benchmark YAML; all nine default models match the same-tier baseline transformer within 7%. Flows keep eight blocks, dist keeps both components, and the oracle keeps two hidden layers. This matches parameter budgets, not architecture, FLOPs, supervision, or guaranteed convergence. Exact constructor arguments are in `plan.json`.

| Case | Tier 1 | Tier 2 |
|---|---:|---:|
| embeddinggemma_256/g-independent/h-baseline | 199,552 | 660,992 |
| embeddinggemma_256/g-independent/h-pre_additive | 199,552 | 660,992 |
| embeddinggemma_256/g-independent/h-noise_token | 200,192 | 662,272 |
| embeddinggemma_256/g-independent/h-dist | 197,093 | 641,229 |
| embeddinggemma_256/g-independent/h-particles | 201,729 | 665,345 |
| embeddinggemma_256/g-independent/h-state_flow | 201,808 | 648,144 |
| embeddinggemma_256/g-independent/h-distilled_flow | 192,144 | 672,912 |
| embeddinggemma_256/g-independent/h-direct_semantic_flow | 192,144 | 672,912 |
| embeddinggemma_256/g-independent/h-oracle_regression | 197,376 | 655,008 |
| embeddinggemma_768/g-independent/h-baseline | 331,136 | 923,648 |
| embeddinggemma_768/g-independent/h-pre_additive | 331,136 | 923,648 |
| embeddinggemma_768/g-independent/h-noise_token | 331,776 | 924,928 |
| embeddinggemma_768/g-independent/h-dist | 332,773 | 912,077 |
| embeddinggemma_768/g-independent/h-particles | 333,313 | 928,001 |
| embeddinggemma_768/g-independent/h-state_flow | 334,672 | 934,096 |
| embeddinggemma_768/g-independent/h-distilled_flow | 310,672 | 902,928 |
| embeddinggemma_768/g-independent/h-direct_semantic_flow | 310,672 | 902,928 |
| embeddinggemma_768/g-independent/h-oracle_regression | 332,928 | 929,696 |

## Launch prerequisites

Oracle training targets: 256D verified; 768D verified.

## Files

`plan.json` records exact splits/configs/hashes; `selection.json` freezes choices before test; `fits/<case>/{search,final}/<trial-or-seed>/training.json` holds every epoch and best-epoch decision; `evaluation/<case>/seed-*.json` holds all metrics, identity/nonidentity subsets and no-edit references. `published.json` points to predeclared deployment-seed checkpoints under `models/talent/<case>/benchmarks/<run>/selected/`. Load these with `src.hz_training.load_model`. Selected weights plus all completed reports/histories are Git-trackable; trial/nondeployment weights, logs and live progress remain local.
