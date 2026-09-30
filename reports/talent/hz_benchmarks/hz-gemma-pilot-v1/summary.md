# h_Z training pilot

Status: complete.

Fit/validation units: 3200/800; 26 fits, one seed 42, 50 epoch ceiling (10 pretraining for dist). No official-test scoring or deployment selection.

All nine families run at both dimensions. The separate pre-additive candidates use per-coordinate noise_std = noise_rms / sqrt(latent_dim), with the original noise_std=1 as a diagnostic. Noise levels are compared by that family's native validation semantic KL only. Exact inputs, source hashes and constructor settings are in `plan.json`; epoch histories are in `fits/.../pilot/.../training.json`.

## Noise calibration

| Dimension | RMS noise | Per-coordinate std | Best validation KL | No-edit KL | Shift L2 | Sample spread L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.100 | 0.006250 | 0.454877 | 13.441610 | 0.7886 | 0.0231 |
| 256 | 0.250 | 0.015625 | 0.303011 | 13.441610 | 0.7149 | 0.0514 |
| 256 | 0.500 | 0.031250 | 0.269941 | 13.441610 | 0.7483 | 0.0758 |
| 256 | 1.000 | 0.062500 | 0.418824 | 13.441610 | 0.8528 | 0.1224 |
| 256 | 16.000 | 1.000000 | 0.949295 | 13.441610 | 2.0932 | 0.1292 |
| 768 | 0.100 | 0.003608 | 0.301229 | 13.405993 | 1.7769 | 0.0250 |
| 768 | 0.250 | 0.009021 | 0.259780 | 13.405993 | 1.6758 | 0.0562 |
| 768 | 0.500 | 0.018042 | 0.192790 | 13.405993 | 1.2918 | 0.0794 |
| 768 | 1.000 | 0.036084 | 0.235472 | 13.405993 | 1.0572 | 0.1051 |
| 768 | 27.713 | 1.000000 | 0.993450 | 13.405993 | 3.5211 | 0.1563 |

Selected RMS noise by minimum pilot validation KL: 256D = 0.5; 768D = 0.5. This is a pilot choice at one learning rate and size tier, not an estimate of final test performance.

## Training diagnostics

The first/best/last validation values below use each family's own criterion; do not rank unlike criteria across families. Dist values start with its first joint epoch. Full epoch-level loss components and learning rates are in the saved training reports.

| Case | RMS noise | Epochs | Phases | Validation first / best / last | Best epoch | Final train loss components | Seconds |
|---|---:|---:|---|---|---:|---|---:|
| embeddinggemma_256/g-independent/h-baseline | — | 50 | train | 1.4638 / 0.49132 / 0.49132 | 50 | total=0.4618, consistency=0.4243, sparsity=0.03391, proximity=0.003595 | 34.7 |
| embeddinggemma_256/g-independent/h-direct_semantic_flow | — | 50 | train | 3.941 / 0.10613 / 0.10613 | 50 | total=0.1923, semantic_kl=0.07347, entropy=0.8112, proximity=0.05216, support=0.6381, identity=0.0006384, spread=1.625, log_det=11.4, grad_norm=1.435 | 171.6 |
| embeddinggemma_256/g-independent/h-dist | — | 50 | pretrain, joint | 0.56436 / 0.094165 / 0.16258 | 46 | phase=1, total=0.1355, kl=0.1269, sparsity=0.008305 | 323.5 |
| embeddinggemma_256/g-independent/h-distilled_flow | — | 50 | train | 0.34656 / 0.10406 / 0.10406 | 50 | total=0.08659, energy=0.08659, spread=0.9819, identity_fraction=0.1916, grad_norm=0.35 | 224.8 |
| embeddinggemma_256/g-independent/h-noise_token | — | 50 | train | 1.9567 / 0.22759 / 0.26831 | 48 | total=0.3512, kl=0.2514, entropy=0.8062, spread=0.09334, sparsity=0.01795, proximity=0.001175 | 276.2 |
| embeddinggemma_256/g-independent/h-oracle_regression | — | 50 | train | 0.00042094 / 0.0001638 / 0.0001638 | 50 | mse=0.0001678 | 10.7 |
| embeddinggemma_256/g-independent/h-particles | — | 50 | train | 1.247 / 0.19384 / 0.19384 | 50 | total=0.2972, kl=0.235, entropy=0.5254, eff_particles=12.71, distinct_states=4.366, sparsity=0.009401, proximity=0.0003138 | 160.7 |
| embeddinggemma_256/g-independent/h-pre_additive | 0.1 | 50 | train | 7.0036 / 0.45488 / 0.68956 | 47 | total=0.3819, kl=0.3534, sparsity=0.026, proximity=0.002459 | 236.6 |
| embeddinggemma_256/g-independent/h-pre_additive | 0.25 | 50 | train | 4.4408 / 0.30301 / 0.30301 | 50 | total=0.4397, kl=0.4141, sparsity=0.02353, proximity=0.002056 | 236.2 |
| embeddinggemma_256/g-independent/h-pre_additive | 0.5 | 50 | train | 8.3206 / 0.26994 / 0.31675 | 43 | total=0.4711, kl=0.4462, sparsity=0.02301, proximity=0.001957 | 236.8 |
| embeddinggemma_256/g-independent/h-pre_additive | 1.0 | 50 | train | 4.1527 / 0.41882 / 0.41882 | 50 | total=0.6555, kl=0.6243, sparsity=0.02827, proximity=0.002924 | 238.6 |
| embeddinggemma_256/g-independent/h-pre_additive | 16.0 | 50 | train | 1.8776 / 0.94929 / 0.99995 | 44 | total=1.323, kl=1.235, sparsity=0.07232, proximity=0.01663 | 249.9 |
| embeddinggemma_256/g-independent/h-state_flow | — | 50 | train | -2.0789 / -2.8495 / -2.8463 | 49 | total=-747.8, nll=-747.8, grad_norm=444.4 | 36.5 |
| embeddinggemma_768/g-independent/h-baseline | — | 50 | train | 1.0298 / 0.5194 / 0.54311 | 45 | total=0.3758, consistency=0.3496, sparsity=0.02404, proximity=0.002074 | 40.3 |
| embeddinggemma_768/g-independent/h-direct_semantic_flow | — | 50 | train | 4.0557 / 0.067235 / 0.067235 | 50 | total=0.1526, semantic_kl=0.04506, entropy=0.6842, proximity=0.05245, support=0.6609, identity=0.000805, spread=3.353, log_det=23.35, grad_norm=1.173 | 415.6 |
| embeddinggemma_768/g-independent/h-dist | — | 50 | pretrain, joint | 0.30945 / 0.074017 / 0.10044 | 45 | phase=1, total=0.07553, kl=0.06465, sparsity=0.0105 | 402.2 |
| embeddinggemma_768/g-independent/h-distilled_flow | — | 50 | train | 0.21524 / 0.068446 / 0.068463 | 47 | total=0.0585, energy=0.0585, spread=1.142, identity_fraction=0.1916, grad_norm=0.1554 | 584.4 |
| embeddinggemma_768/g-independent/h-noise_token | — | 50 | train | 2.2713 / 0.20921 / 0.23907 | 49 | total=0.3633, kl=0.2525, entropy=0.7714, spread=0.1585, sparsity=0.03101, proximity=0.002696 | 346.2 |
| embeddinggemma_768/g-independent/h-oracle_regression | — | 50 | train | 0.00018261 / 7.0209e-05 / 7.0313e-05 | 49 | mse=7.246e-05 | 15.3 |
| embeddinggemma_768/g-independent/h-particles | — | 50 | train | 1.6238 / 0.43676 / 0.50141 | 41 | total=0.5788, kl=0.5258, entropy=0.4542, eff_particles=15.64, distinct_states=4.597, sparsity=0.00746, proximity=0.0001446 | 219.9 |
| embeddinggemma_768/g-independent/h-pre_additive | 0.1 | 50 | train | 4.3127 / 0.30123 / 0.34548 | 46 | total=0.4334, kl=0.3911, sparsity=0.03839, proximity=0.00387 | 290.4 |
| embeddinggemma_768/g-independent/h-pre_additive | 0.25 | 50 | train | 4.4731 / 0.25978 / 0.34928 | 39 | total=0.3669, kl=0.3283, sparsity=0.03522, proximity=0.003344 | 287.2 |
| embeddinggemma_768/g-independent/h-pre_additive | 0.5 | 50 | train | 5.4172 / 0.19279 / 0.19279 | 50 | total=0.3546, kl=0.3243, sparsity=0.02812, proximity=0.002206 | 273.2 |
| embeddinggemma_768/g-independent/h-pre_additive | 1.0 | 50 | train | 8.8976 / 0.23547 / 0.36323 | 45 | total=0.4706, kl=0.4473, sparsity=0.02194, proximity=0.0014 | 268.8 |
| embeddinggemma_768/g-independent/h-pre_additive | 27.712812921102035 | 50 | train | 2.3242 / 0.99345 / 1.302 | 47 | total=1.345, kl=1.237, sparsity=0.08962, proximity=0.01836 | 248.5 |
| embeddinggemma_768/g-independent/h-state_flow | — | 50 | train | -2.5372 / -3.3796 / -3.3796 | 50 | total=-2638, nll=-2638, grad_norm=1434 | 54.2 |

Aggregate fitting time: 1.63 worker-hours across 26 fits; median 237.7 seconds/fit. This pilot establishes feasibility and scale; its short learning curves cannot certify convergence under the full 500-epoch ceiling.
