# h_Z benchmark

Status: complete.

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

## Latent Only: test mean ± seed SD

Descriptive comparisons, not a test-selected deployment winner. Lower energy/MSE/KL and higher cosine are better.

| Case | Seeds | Energy ↓ | Standardized mean MSE ↓ | Cosine ↑ | Semantic KL ↓ | Identity shift L2 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| embeddinggemma_256/g-independent/h-baseline | 5 | 1.00171 ± 0.05455 | 1.08988 ± 0.10217 | 0.86680 ± 0.01039 | 0.37189 ± 0.00892 | 0.30812 ± 0.04297 |
| embeddinggemma_256/g-independent/h-direct_semantic_flow | 5 | 0.50754 ± 0.00675 | 0.40783 ± 0.00234 | 0.94191 ± 0.00034 | 0.02206 ± 0.00097 | 0.04457 ± 0.00532 |
| embeddinggemma_256/g-independent/h-dist | 5 | 0.67602 ± 0.04496 | 0.64497 ± 0.03877 | 0.91155 ± 0.00526 | 0.14541 ± 0.04851 | 0.27698 ± 0.10601 |
| embeddinggemma_256/g-independent/h-distilled_flow | 5 | 0.41983 ± 0.00408 | 0.29681 ± 0.00233 | 0.95770 ± 0.00035 | 0.82368 ± 0.06569 | 0.00251 ± 0.00041 |
| embeddinggemma_256/g-independent/h-noise_token | 5 | 0.75791 ± 0.04729 | 0.72430 ± 0.08599 | 0.90462 ± 0.00901 | 0.10756 ± 0.00569 | 0.14488 ± 0.02576 |
| embeddinggemma_256/g-independent/h-oracle_regression | 5 | 0.35570 ± 0.00101 | 0.14072 ± 0.00068 | 0.98034 ± 0.00010 | 0.95674 ± 0.00940 | 0.10059 ± 0.00092 |
| embeddinggemma_256/g-independent/h-particles | 5 | 0.56522 ± 0.00668 | 0.54772 ± 0.01475 | 0.92427 ± 0.00204 | 0.07067 ± 0.00556 | 0.09880 ± 0.01631 |
| embeddinggemma_256/g-independent/h-pre_additive | 5 | 0.78126 ± 0.11719 | 0.80576 ± 0.21749 | 0.89449 ± 0.02388 | 0.12121 ± 0.00557 | 0.23582 ± 0.08363 |
| embeddinggemma_256/g-independent/h-state_flow | 5 | 0.44591 ± 0.00145 | 0.30842 ± 0.00172 | 0.95613 ± 0.00022 | 0.31240 ± 0.02123 | 0.00242 ± 0.00007 |
| embeddinggemma_768/g-independent/h-baseline | 5 | 1.54818 ± 0.32612 | 2.56863 ± 1.05897 | 0.71327 ± 0.07235 | 0.29281 ± 0.01922 | 0.67052 ± 0.17746 |
| embeddinggemma_768/g-independent/h-direct_semantic_flow | 5 | 0.51541 ± 0.00345 | 0.42982 ± 0.00205 | 0.92447 ± 0.00031 | 0.02694 ± 0.00110 | 0.03909 ± 0.00107 |
| embeddinggemma_768/g-independent/h-dist | 5 | 0.71826 ± 0.00701 | 0.68193 ± 0.01348 | 0.88765 ± 0.00192 | 0.18015 ± 0.00415 | 0.35975 ± 0.01235 |
| embeddinggemma_768/g-independent/h-distilled_flow | 5 | 0.48413 ± 0.00771 | 0.36312 ± 0.01069 | 0.93599 ± 0.00189 | 0.93216 ± 0.48048 | 0.00222 ± 0.00019 |
| embeddinggemma_768/g-independent/h-noise_token | 5 | 1.85680 ± 0.80659 | 4.47473 ± 3.98035 | 0.65269 ± 0.13987 | 0.08457 ± 0.01437 | 0.85281 ± 0.50553 |
| embeddinggemma_768/g-independent/h-oracle_regression | 5 | 0.36851 ± 0.00038 | 0.15210 ± 0.00034 | 0.97355 ± 0.00006 | 0.80829 ± 0.02316 | 0.10987 ± 0.00196 |
| embeddinggemma_768/g-independent/h-particles | 5 | 0.64688 ± 0.00606 | 0.65954 ± 0.01117 | 0.89108 ± 0.00171 | 0.10859 ± 0.00888 | 0.23916 ± 0.02020 |
| embeddinggemma_768/g-independent/h-pre_additive | 5 | 1.58614 ± 0.55164 | 3.43030 ± 2.13647 | 0.68860 ± 0.11613 | 0.06279 ± 0.01166 | 0.73065 ± 0.35238 |
| embeddinggemma_768/g-independent/h-state_flow | 5 | 0.49372 ± 0.00700 | 0.36710 ± 0.01081 | 0.93537 ± 0.00188 | 0.75082 ± 0.48995 | 0.00225 ± 0.00021 |

## Observed State: test mean ± seed SD

Descriptive comparisons, not a test-selected deployment winner. Lower energy/MSE/KL and higher cosine are better.

| Case | Seeds | Energy ↓ | Standardized mean MSE ↓ | Cosine ↑ | Semantic KL ↓ | Identity shift L2 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| embeddinggemma_256/g-independent/h-state_flow | 5 | 0.44461 ± 0.00138 | 0.30766 ± 0.00166 | 0.95624 ± 0.00022 | 0.34407 ± 0.02080 | 0.00000 ± 0.00000 |
| embeddinggemma_768/g-independent/h-state_flow | 5 | 0.49293 ± 0.00702 | 0.36681 ± 0.01078 | 0.93544 ± 0.00188 | 0.76768 ± 0.48857 | 0.00000 ± 0.00000 |

## Files

`plan.json` records exact splits/configs/hashes; `selection.json` freezes choices before test; `fits/<case>/{search,final}/<trial-or-seed>/training.json` holds every epoch and best-epoch decision; `evaluation/<case>/seed-*.json` holds all metrics, identity/nonidentity subsets and no-edit references. `published.json` points to predeclared deployment-seed checkpoints under `models/talent/<case>/benchmarks/<run>/selected/`. Load these with `src.hz_training.load_model`. Selected weights plus all completed reports/histories are Git-trackable; trial/nondeployment weights, logs and live progress remain local.
