# Semantic decoder benchmark

Run: `g-all-encoders-v1`. Status: **complete**.

Trained cases: 36/36; evaluated cases: 36/36.

## Selection

Validation-selected recommendation: **`embeddinggemma_768/g-independent`**.

Selection was frozen before reading any official-test performance. The lowest mean full joint validation NLL across the five predeclared final seeds wins; ties use case-name order. Seed 42 is the active checkpoint, not the best-looking seed.

- Best `autoregressive`: `embeddinggemma_768/g-autoregressive`.
- Best `independent`: `embeddinggemma_768/g-independent`.

## Protocol

- 18 frozen latent spaces × two decoder variants; no encoders are retrained.
- Same official IDs throughout: 3200 decoder-fit / 800 validation / 1000 test units.
- 20 identical hyperparameter candidates per case; learning rate, weight decay, dropout and hidden width are tuned only on validation joint NLL.
- At most 500 epochs, validation early stopping (patience 30), plateau LR reduction, best-checkpoint restoration.
- Final initialization seeds: [42, 43, 44, 45, 46]. Hyperparameters are selected using search seed 42.
- Full joint NLL is in nats/example and comparable across independent and autoregressive decoders, despite their different optimization-loss scales.
- Mean ± sample SD summarizes initialization variability, not a confidence interval over test units. Validation is selection-biased; small ranking differences are not evidence of statistical significance.
- Higher accuracy is better; lower NLL, Brier and ECE are better. ECE is descriptive, uses fixed equal-width confidence bins, and no post-hoc calibrator is fitted.
- Test results are a final check of the frozen selection. Choosing another winner from this test table would reuse the test set for model selection.
- Parameter counts may increase with latent width/selected hidden width; the compute budget is equal per case, not equal per encoder family (families have different numbers of widths).

### LangVAE fine-tuning caveat

Of the decoder-validation texts, 730 were used for unsupervised LangVAE adaptation training and 70 for its validation/checkpoint selection. Decoder-validation structured labels were not used for fitting g, but these texts are not completely untouched end-to-end validation examples. All official-test IDs were excluded from adaptation. Account for this asymmetry when interpreting validation rankings.

## Main comparison (ordered by validation NLL)

Joint accuracy uses the exact joint MAP state, not greedy autoregressive decoding. Macro ECE is the unweighted mean across the structured-variable marginals. Accuracy is in percent; ECE is on [0, 1].

| Rank | Encoder | Dim | g | Validation NLL | Test NLL | Test joint accuracy (%) | Test macro ECE |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | embeddinggemma_768 | 768 | independent | 0.3788 ± 0.0023 | 0.4238 ± 0.0028 | 83.64 ± 0.38 | 0.0138 ± 0.0007 |
| 2 | embeddinggemma_768 | 768 | autoregressive | 0.3824 ± 0.0030 | 0.4405 ± 0.0038 | 83.22 ± 0.41 | 0.0135 ± 0.0016 |
| 3 | embeddinggemma_512 | 512 | independent | 0.4136 ± 0.0013 | 0.4493 ± 0.0021 | 83.50 ± 0.29 | 0.0143 ± 0.0013 |
| 4 | embeddinggemma_512 | 512 | autoregressive | 0.4150 ± 0.0047 | 0.4655 ± 0.0039 | 82.92 ± 0.30 | 0.0128 ± 0.0009 |
| 5 | embeddinggemma_256 | 256 | independent | 0.5150 ± 0.0042 | 0.5668 ± 0.0039 | 78.92 ± 0.41 | 0.0125 ± 0.0016 |
| 6 | embeddinggemma_256 | 256 | autoregressive | 0.5295 ± 0.0037 | 0.5746 ± 0.0050 | 78.90 ± 0.17 | 0.0136 ± 0.0020 |
| 7 | qwen3_1024 | 1024 | independent | 0.9128 ± 0.0048 | 0.9607 ± 0.0050 | 66.48 ± 0.54 | 0.0269 ± 0.0020 |
| 8 | embeddinggemma_128 | 128 | autoregressive | 0.9262 ± 0.0089 | 0.9467 ± 0.0054 | 64.94 ± 0.33 | 0.0158 ± 0.0014 |
| 9 | qwen3_1024 | 1024 | autoregressive | 0.9449 ± 0.0041 | 1.0236 ± 0.0030 | 63.38 ± 0.55 | 0.0259 ± 0.0059 |
| 10 | embeddinggemma_128 | 128 | independent | 0.9464 ± 0.0059 | 0.9742 ± 0.0095 | 63.30 ± 0.58 | 0.0185 ± 0.0023 |
| 11 | qwen3_768 | 768 | independent | 0.9494 ± 0.0059 | 1.0151 ± 0.0024 | 64.96 ± 1.19 | 0.0285 ± 0.0024 |
| 12 | qwen3_768 | 768 | autoregressive | 0.9777 ± 0.0132 | 1.0629 ± 0.0085 | 62.62 ± 0.52 | 0.0264 ± 0.0029 |
| 13 | qwen3_512 | 512 | autoregressive | 1.0713 ± 0.0050 | 1.1689 ± 0.0067 | 58.52 ± 0.40 | 0.0277 ± 0.0009 |
| 14 | qwen3_512 | 512 | independent | 1.0833 ± 0.0060 | 1.1414 ± 0.0046 | 60.28 ± 0.36 | 0.0277 ± 0.0012 |
| 15 | nomic_768 | 768 | autoregressive | 1.1972 ± 0.0266 | 1.1867 ± 0.0352 | 60.44 ± 2.07 | 0.0253 ± 0.0045 |
| 16 | nomic_768 | 768 | independent | 1.2216 ± 0.0140 | 1.1913 ± 0.0107 | 61.02 ± 1.04 | 0.0304 ± 0.0020 |
| 17 | nomic_512 | 512 | autoregressive | 1.3672 ± 0.0179 | 1.3591 ± 0.0230 | 56.20 ± 0.87 | 0.0265 ± 0.0027 |
| 18 | qwen3_256 | 256 | autoregressive | 1.3757 ± 0.0020 | 1.4156 ± 0.0027 | 50.20 ± 0.23 | 0.0268 ± 0.0017 |
| 19 | nomic_512 | 512 | independent | 1.3947 ± 0.0358 | 1.3486 ± 0.0437 | 55.14 ± 1.15 | 0.0222 ± 0.0022 |
| 20 | qwen3_256 | 256 | independent | 1.4270 ± 0.0079 | 1.4275 ± 0.0056 | 51.42 ± 0.56 | 0.0237 ± 0.0022 |
| 21 | nomic_256 | 256 | autoregressive | 1.7095 ± 0.0106 | 1.6763 ± 0.0139 | 44.80 ± 0.29 | 0.0237 ± 0.0018 |
| 22 | nomic_256 | 256 | independent | 1.8378 ± 0.0137 | 1.7563 ± 0.0155 | 44.24 ± 1.00 | 0.0265 ± 0.0027 |
| 23 | qwen3_128 | 128 | autoregressive | 1.8603 ± 0.0032 | 1.8963 ± 0.0051 | 39.86 ± 0.34 | 0.0258 ± 0.0021 |
| 24 | qwen3_128 | 128 | independent | 2.0075 ± 0.0033 | 1.9755 ± 0.0030 | 38.40 ± 0.12 | 0.0260 ± 0.0007 |
| 25 | nomic_128 | 128 | autoregressive | 2.2108 ± 0.0044 | 2.2088 ± 0.0103 | 31.64 ± 0.38 | 0.0338 ± 0.0034 |
| 26 | nomic_128 | 128 | independent | 2.3867 ± 0.0055 | 2.3312 ± 0.0055 | 29.22 ± 0.71 | 0.0320 ± 0.0032 |
| 27 | langvae_ft | 128 | autoregressive | 2.4165 ± 0.0084 | 2.4127 ± 0.0086 | 30.36 ± 0.44 | 0.0266 ± 0.0025 |
| 28 | qwen3_64 | 64 | autoregressive | 2.4503 ± 0.0100 | 2.4802 ± 0.0040 | 28.70 ± 0.52 | 0.0337 ± 0.0029 |
| 29 | langvae | 128 | autoregressive | 2.5228 ± 0.0163 | 2.5278 ± 0.0153 | 28.76 ± 0.67 | 0.0260 ± 0.0015 |
| 30 | langvae_ft | 128 | independent | 2.6030 ± 0.0099 | 2.5433 ± 0.0158 | 27.46 ± 0.34 | 0.0257 ± 0.0028 |
| 31 | nomic_64 | 64 | autoregressive | 2.6223 ± 0.0085 | 2.6044 ± 0.0051 | 23.92 ± 0.18 | 0.0428 ± 0.0045 |
| 32 | qwen3_64 | 64 | independent | 2.6304 ± 0.0119 | 2.6176 ± 0.0046 | 26.80 ± 0.70 | 0.0370 ± 0.0023 |
| 33 | langvae | 128 | independent | 2.7500 ± 0.0185 | 2.6920 ± 0.0165 | 24.76 ± 0.78 | 0.0262 ± 0.0026 |
| 34 | nomic_64 | 64 | independent | 2.8304 ± 0.0071 | 2.7474 ± 0.0092 | 22.40 ± 0.76 | 0.0449 ± 0.0023 |
| 35 | qwen3_32 | 32 | autoregressive | 3.0021 ± 0.0083 | 3.0073 ± 0.0103 | 18.68 ± 0.28 | 0.0292 ± 0.0056 |
| 36 | qwen3_32 | 32 | independent | 3.2881 ± 0.0079 | 3.2395 ± 0.0064 | 16.90 ± 0.76 | 0.0359 ± 0.0036 |

## Factual-test marginal accuracy (%)

These use each exact marginal's argmax, without true-prefix teacher forcing. T is hidden in the text and must be inferred from noisy proxies; 100% is not an appropriate expected target. Y is not a decoder target.

| Case | D | T | U | X | Joint Brier |
|---|---:|---:|---:|---:|---:|
| embeddinggemma_768/g-independent | 100.00 ± 0.00 | 84.46 ± 0.38 | 99.82 ± 0.08 | 99.18 ± 0.04 | 0.2388 ± 0.0018 |
| embeddinggemma_768/g-autoregressive | 99.98 ± 0.04 | 84.10 ± 0.41 | 99.74 ± 0.09 | 99.22 ± 0.04 | 0.2467 ± 0.0023 |
| embeddinggemma_512/g-independent | 99.98 ± 0.04 | 84.80 ± 0.27 | 99.36 ± 0.11 | 99.08 ± 0.04 | 0.2467 ± 0.0018 |
| embeddinggemma_512/g-autoregressive | 99.96 ± 0.05 | 84.26 ± 0.34 | 99.32 ± 0.11 | 99.14 ± 0.05 | 0.2531 ± 0.0013 |
| embeddinggemma_256/g-independent | 100.00 ± 0.00 | 82.38 ± 0.63 | 96.52 ± 0.27 | 99.10 ± 0.00 | 0.3061 ± 0.0017 |
| embeddinggemma_256/g-autoregressive | 99.96 ± 0.05 | 81.80 ± 0.43 | 96.62 ± 0.16 | 99.02 ± 0.04 | 0.3074 ± 0.0012 |
| qwen3_1024/g-independent | 99.26 ± 0.09 | 81.64 ± 0.48 | 82.82 ± 0.70 | 98.10 ± 0.16 | 0.4579 ± 0.0029 |
| embeddinggemma_128/g-autoregressive | 99.44 ± 0.09 | 76.50 ± 0.27 | 84.20 ± 0.38 | 99.04 ± 0.17 | 0.4755 ± 0.0035 |
| qwen3_1024/g-autoregressive | 98.66 ± 0.26 | 80.50 ± 0.47 | 80.42 ± 1.12 | 97.54 ± 0.30 | 0.4930 ± 0.0044 |
| embeddinggemma_128/g-independent | 99.46 ± 0.09 | 76.48 ± 0.33 | 83.22 ± 0.33 | 99.08 ± 0.04 | 0.4862 ± 0.0044 |
| qwen3_768/g-independent | 99.14 ± 0.13 | 81.64 ± 0.40 | 80.94 ± 1.03 | 97.80 ± 0.20 | 0.4754 ± 0.0038 |
| qwen3_768/g-autoregressive | 98.80 ± 0.20 | 80.52 ± 0.26 | 78.64 ± 0.64 | 97.60 ± 0.20 | 0.5048 ± 0.0023 |
| qwen3_512/g-autoregressive | 97.80 ± 0.21 | 78.94 ± 0.21 | 76.90 ± 0.23 | 96.76 ± 0.11 | 0.5416 ± 0.0008 |
| qwen3_512/g-independent | 98.14 ± 0.17 | 80.66 ± 0.36 | 77.40 ± 0.12 | 97.16 ± 0.11 | 0.5308 ± 0.0024 |
| nomic_768/g-autoregressive | 94.74 ± 0.31 | 76.36 ± 0.79 | 80.06 ± 2.36 | 98.96 ± 0.39 | 0.5356 ± 0.0169 |
| nomic_768/g-independent | 94.10 ± 0.16 | 76.84 ± 0.50 | 82.68 ± 1.10 | 98.72 ± 0.19 | 0.5303 ± 0.0060 |
| nomic_512/g-autoregressive | 92.02 ± 0.29 | 74.86 ± 0.55 | 76.06 ± 1.28 | 98.64 ± 0.17 | 0.5932 ± 0.0071 |
| qwen3_256/g-autoregressive | 94.84 ± 0.24 | 77.08 ± 0.11 | 72.50 ± 0.19 | 94.34 ± 0.18 | 0.6194 ± 0.0010 |
| nomic_512/g-independent | 91.24 ± 0.43 | 75.54 ± 0.51 | 77.74 ± 1.22 | 98.30 ± 0.16 | 0.5865 ± 0.0134 |
| qwen3_256/g-independent | 95.04 ± 0.15 | 76.70 ± 0.54 | 72.68 ± 0.31 | 94.76 ± 0.27 | 0.6255 ± 0.0017 |
| nomic_256/g-autoregressive | 85.54 ± 0.29 | 70.80 ± 0.45 | 70.48 ± 0.33 | 97.50 ± 0.12 | 0.6973 ± 0.0047 |
| nomic_256/g-independent | 85.30 ± 0.42 | 70.90 ± 0.78 | 69.90 ± 1.00 | 97.70 ± 0.16 | 0.7106 ± 0.0052 |
| qwen3_128/g-autoregressive | 85.24 ± 0.21 | 72.76 ± 0.44 | 66.00 ± 0.59 | 88.60 ± 0.25 | 0.7418 ± 0.0025 |
| qwen3_128/g-independent | 85.50 ± 0.25 | 73.12 ± 0.18 | 67.12 ± 0.33 | 88.20 ± 0.16 | 0.7516 ± 0.0006 |
| nomic_128/g-autoregressive | 74.38 ± 0.38 | 63.88 ± 0.77 | 63.20 ± 0.54 | 93.34 ± 0.29 | 0.8181 ± 0.0015 |
| nomic_128/g-independent | 73.68 ± 0.20 | 64.10 ± 0.58 | 63.02 ± 0.27 | 93.04 ± 0.27 | 0.8277 ± 0.0013 |
| langvae_ft/g-autoregressive | 65.90 ± 0.43 | 80.02 ± 0.15 | 69.68 ± 0.33 | 73.48 ± 0.39 | 0.8298 ± 0.0014 |
| qwen3_64/g-autoregressive | 79.44 ± 0.42 | 66.22 ± 0.59 | 63.52 ± 0.34 | 75.74 ± 0.43 | 0.8484 ± 0.0014 |
| langvae/g-autoregressive | 65.16 ± 0.09 | 78.10 ± 0.26 | 67.54 ± 0.23 | 72.42 ± 0.61 | 0.8441 ± 0.0029 |
| langvae_ft/g-independent | 65.90 ± 0.22 | 80.12 ± 0.24 | 67.86 ± 0.97 | 73.94 ± 0.34 | 0.8463 ± 0.0021 |
| nomic_64/g-autoregressive | 64.52 ± 0.66 | 60.28 ± 0.63 | 59.34 ± 0.34 | 84.86 ± 0.35 | 0.8718 ± 0.0011 |
| qwen3_64/g-independent | 79.90 ± 0.28 | 66.88 ± 0.26 | 63.58 ± 0.41 | 75.64 ± 0.57 | 0.8630 ± 0.0005 |
| langvae/g-independent | 64.24 ± 0.38 | 78.50 ± 0.29 | 65.26 ± 0.38 | 72.22 ± 0.79 | 0.8627 ± 0.0018 |
| nomic_64/g-independent | 65.60 ± 0.80 | 61.24 ± 0.48 | 59.58 ± 0.77 | 85.62 ± 0.48 | 0.8795 ± 0.0010 |
| qwen3_32/g-autoregressive | 72.48 ± 0.73 | 59.70 ± 0.67 | 58.88 ± 0.50 | 60.46 ± 0.92 | 0.9071 ± 0.0008 |
| qwen3_32/g-independent | 73.10 ± 0.40 | 59.64 ± 0.40 | 58.02 ± 0.82 | 60.02 ± 0.27 | 0.9216 ± 0.0010 |

## Counterfactual-test diagnostics

These are real encoded counterfactual texts, not outputs of a trained latent manipulator. They do not demonstrate h_Z performance or fairness. The non-identity subset removes copied identity pairs. Factual and counterfactual rows share unit IDs and must not be treated as independent samples.

| Case | All CF NLL | Non-identity CF NLL | Non-identity joint accuracy (%) | Non-identity macro ECE |
|---|---:|---:|---:|---:|
| embeddinggemma_768/g-independent | 0.4212 ± 0.0015 | 0.4220 ± 0.0017 | 84.17 ± 0.35 | 0.0119 ± 0.0012 |
| embeddinggemma_768/g-autoregressive | 0.4367 ± 0.0030 | 0.4368 ± 0.0027 | 83.87 ± 0.55 | 0.0112 ± 0.0018 |
| embeddinggemma_512/g-independent | 0.4550 ± 0.0020 | 0.4596 ± 0.0021 | 82.31 ± 0.24 | 0.0117 ± 0.0009 |
| embeddinggemma_512/g-autoregressive | 0.4683 ± 0.0057 | 0.4751 ± 0.0068 | 82.18 ± 0.15 | 0.0129 ± 0.0019 |
| embeddinggemma_256/g-independent | 0.6220 ± 0.0050 | 0.6320 ± 0.0051 | 76.45 ± 0.37 | 0.0133 ± 0.0015 |
| embeddinggemma_256/g-autoregressive | 0.6418 ± 0.0067 | 0.6536 ± 0.0065 | 75.99 ± 0.40 | 0.0167 ± 0.0007 |
| qwen3_1024/g-independent | 1.1591 ± 0.0101 | 1.1517 ± 0.0081 | 59.41 ± 0.69 | 0.0344 ± 0.0043 |
| embeddinggemma_128/g-autoregressive | 1.0772 ± 0.0057 | 1.1033 ± 0.0062 | 60.24 ± 0.44 | 0.0158 ± 0.0027 |
| qwen3_1024/g-autoregressive | 1.2335 ± 0.0250 | 1.2337 ± 0.0305 | 55.00 ± 2.09 | 0.0311 ± 0.0047 |
| embeddinggemma_128/g-independent | 1.0836 ± 0.0109 | 1.1121 ± 0.0097 | 58.60 ± 0.46 | 0.0215 ± 0.0038 |
| qwen3_768/g-independent | 1.2175 ± 0.0038 | 1.2093 ± 0.0045 | 58.15 ± 0.97 | 0.0299 ± 0.0050 |
| qwen3_768/g-autoregressive | 1.2568 ± 0.0080 | 1.2608 ± 0.0132 | 54.14 ± 1.38 | 0.0304 ± 0.0038 |
| qwen3_512/g-autoregressive | 1.3807 ± 0.0146 | 1.4039 ± 0.0169 | 50.13 ± 0.69 | 0.0357 ± 0.0030 |
| qwen3_512/g-independent | 1.3640 ± 0.0140 | 1.3777 ± 0.0144 | 51.83 ± 0.98 | 0.0332 ± 0.0030 |
| nomic_768/g-autoregressive | 1.2310 ± 0.0492 | 1.2405 ± 0.0569 | 55.11 ± 3.15 | 0.0202 ± 0.0020 |
| nomic_768/g-independent | 1.2079 ± 0.0174 | 1.2012 ± 0.0214 | 57.50 ± 1.36 | 0.0233 ± 0.0020 |
| nomic_512/g-autoregressive | 1.4013 ± 0.0181 | 1.4248 ± 0.0172 | 48.55 ± 1.80 | 0.0245 ± 0.0051 |
| qwen3_256/g-autoregressive | 1.5577 ± 0.0109 | 1.5807 ± 0.0103 | 43.68 ± 0.41 | 0.0307 ± 0.0017 |
| nomic_512/g-independent | 1.3892 ± 0.0475 | 1.4046 ± 0.0465 | 51.45 ± 2.34 | 0.0234 ± 0.0029 |
| qwen3_256/g-independent | 1.5914 ± 0.0142 | 1.6056 ± 0.0164 | 43.60 ± 0.28 | 0.0302 ± 0.0034 |
| nomic_256/g-autoregressive | 1.6993 ± 0.0103 | 1.7271 ± 0.0087 | 40.00 ± 1.16 | 0.0194 ± 0.0021 |
| nomic_256/g-independent | 1.7511 ± 0.0141 | 1.7658 ± 0.0108 | 38.12 ± 0.43 | 0.0262 ± 0.0035 |
| qwen3_128/g-autoregressive | 2.0421 ± 0.0089 | 2.0398 ± 0.0098 | 36.61 ± 0.75 | 0.0273 ± 0.0021 |
| qwen3_128/g-independent | 2.1301 ± 0.0094 | 2.1315 ± 0.0099 | 34.41 ± 0.81 | 0.0340 ± 0.0048 |
| nomic_128/g-autoregressive | 2.1580 ± 0.0127 | 2.1798 ± 0.0147 | 31.75 ± 0.87 | 0.0384 ± 0.0049 |
| nomic_128/g-independent | 2.2399 ± 0.0238 | 2.2517 ± 0.0233 | 29.73 ± 0.92 | 0.0361 ± 0.0010 |
| langvae_ft/g-autoregressive | 2.3990 ± 0.0065 | 2.4311 ± 0.0072 | 30.35 ± 0.53 | 0.0410 ± 0.0026 |
| qwen3_64/g-autoregressive | 2.5562 ± 0.0220 | 2.5659 ± 0.0229 | 25.30 ± 0.48 | 0.0470 ± 0.0038 |
| langvae/g-autoregressive | 2.5104 ± 0.0219 | 2.5369 ± 0.0236 | 28.12 ± 0.85 | 0.0416 ± 0.0018 |
| langvae_ft/g-independent | 2.5756 ± 0.0214 | 2.6177 ± 0.0241 | 28.41 ± 0.31 | 0.0390 ± 0.0035 |
| nomic_64/g-autoregressive | 2.5643 ± 0.0317 | 2.5838 ± 0.0345 | 25.67 ± 0.62 | 0.0391 ± 0.0019 |
| qwen3_64/g-independent | 2.7317 ± 0.0147 | 2.7440 ± 0.0156 | 23.74 ± 0.57 | 0.0532 ± 0.0031 |
| langvae/g-independent | 2.7510 ± 0.0237 | 2.7967 ± 0.0247 | 24.49 ± 0.83 | 0.0493 ± 0.0045 |
| nomic_64/g-independent | 2.7069 ± 0.0246 | 2.7354 ± 0.0261 | 22.72 ± 0.83 | 0.0406 ± 0.0033 |
| qwen3_32/g-autoregressive | 3.0483 ± 0.0273 | 3.0720 ± 0.0302 | 18.20 ± 0.86 | 0.0514 ± 0.0040 |
| qwen3_32/g-independent | 3.3735 ± 0.0155 | 3.4200 ± 0.0167 | 15.56 ± 0.44 | 0.0511 ± 0.0018 |

## Selected settings and checkpoints

These links point to the canonical active seed-42 checkpoints, which are eligible for Git tracking. Check their hashes against [published.json](published.json) if later training replaces them. Other final seeds and tuning trials remain retained locally.

| Case | Hidden width | LR | Weight decay | Dropout | Seed-42 best/run epochs | Parameters | Checkpoint |
|---|---:|---:|---:|---:|---:|---:|---|
| embeddinggemma_768/g-independent | 128 | 0.001 | 0.0001 | 0.3 | 160/190 | 116621 | [model](../../../../models/talent/embeddinggemma_768/g-independent/semantic_decoder.pt) |
| embeddinggemma_768/g-autoregressive | 128 | 0.001 | 0.0001 | 0.3 | 93/123 | 117117 | [model](../../../../models/talent/embeddinggemma_768/g-autoregressive/semantic_decoder.pt) |
| embeddinggemma_512/g-independent | 128 | 0.001 | 0.0001 | 0.3 | 125/155 | 83853 | [model](../../../../models/talent/embeddinggemma_512/g-independent/semantic_decoder.pt) |
| embeddinggemma_512/g-autoregressive | 128 | 0.001 | 0.0001 | 0.3 | 107/137 | 84349 | [model](../../../../models/talent/embeddinggemma_512/g-autoregressive/semantic_decoder.pt) |
| embeddinggemma_256/g-independent | 128 | 0.001 | 0.0001 | 0.3 | 267/297 | 51085 | [model](../../../../models/talent/embeddinggemma_256/g-independent/semantic_decoder.pt) |
| embeddinggemma_256/g-autoregressive | 128 | 0.001 | 0.0001 | 0.3 | 186/216 | 51581 | [model](../../../../models/talent/embeddinggemma_256/g-autoregressive/semantic_decoder.pt) |
| qwen3_1024/g-independent | 128 | 0.001 | 0.0001 | 0.3 | 129/159 | 149389 | [model](../../../../models/talent/qwen3_1024/g-independent/semantic_decoder.pt) |
| embeddinggemma_128/g-autoregressive | 128 | 0.001 | 0.0001 | 0.3 | 143/173 | 35197 | [model](../../../../models/talent/embeddinggemma_128/g-autoregressive/semantic_decoder.pt) |
| qwen3_1024/g-autoregressive | 128 | 0.001 | 0.0001 | 0.3 | 82/112 | 149885 | [model](../../../../models/talent/qwen3_1024/g-autoregressive/semantic_decoder.pt) |
| embeddinggemma_128/g-independent | 128 | 0.001 | 0.0001 | 0.3 | 198/228 | 34701 | [model](../../../../models/talent/embeddinggemma_128/g-independent/semantic_decoder.pt) |
| qwen3_768/g-independent | 128 | 0.001 | 0.0001 | 0.3 | 158/188 | 116621 | [model](../../../../models/talent/qwen3_768/g-independent/semantic_decoder.pt) |
| qwen3_768/g-autoregressive | 128 | 0.001 | 0.0001 | 0.3 | 111/141 | 117117 | [model](../../../../models/talent/qwen3_768/g-autoregressive/semantic_decoder.pt) |
| qwen3_512/g-autoregressive | 128 | 0.001 | 0.0001 | 0.3 | 124/154 | 84349 | [model](../../../../models/talent/qwen3_512/g-autoregressive/semantic_decoder.pt) |
| qwen3_512/g-independent | 128 | 0.001 | 0.0001 | 0.3 | 196/226 | 83853 | [model](../../../../models/talent/qwen3_512/g-independent/semantic_decoder.pt) |
| nomic_768/g-autoregressive | 128 | 0.001 | 0.0001 | 0.3 | 292/322 | 117117 | [model](../../../../models/talent/nomic_768/g-autoregressive/semantic_decoder.pt) |
| nomic_768/g-independent | 512 | 0.0003 | 1e-05 | 0.3 | 234/260 | 663053 | [model](../../../../models/talent/nomic_768/g-independent/semantic_decoder.pt) |
| nomic_512/g-autoregressive | 128 | 0.001 | 0.0001 | 0.3 | 210/240 | 84349 | [model](../../../../models/talent/nomic_512/g-autoregressive/semantic_decoder.pt) |
| qwen3_256/g-autoregressive | 512 | 0.003 | 0.001 | 0.3 | 186/216 | 401405 | [model](../../../../models/talent/qwen3_256/g-autoregressive/semantic_decoder.pt) |
| nomic_512/g-independent | 128 | 0.001 | 0.0001 | 0.3 | 354/384 | 83853 | [model](../../../../models/talent/nomic_512/g-independent/semantic_decoder.pt) |
| qwen3_256/g-independent | 128 | 0.001 | 0.0001 | 0.3 | 199/229 | 51085 | [model](../../../../models/talent/qwen3_256/g-independent/semantic_decoder.pt) |
| nomic_256/g-autoregressive | 128 | 0.001 | 0.0001 | 0.3 | 344/374 | 51581 | [model](../../../../models/talent/nomic_256/g-autoregressive/semantic_decoder.pt) |
| nomic_256/g-independent | 128 | 0.001 | 0.0001 | 0.3 | 377/407 | 51085 | [model](../../../../models/talent/nomic_256/g-independent/semantic_decoder.pt) |
| qwen3_128/g-autoregressive | 128 | 0.001 | 0.0001 | 0.3 | 161/191 | 35197 | [model](../../../../models/talent/qwen3_128/g-autoregressive/semantic_decoder.pt) |
| qwen3_128/g-independent | 128 | 0.001 | 0.0001 | 0.3 | 183/213 | 34701 | [model](../../../../models/talent/qwen3_128/g-independent/semantic_decoder.pt) |
| nomic_128/g-autoregressive | 256 | 0.001 | 0 | 0.1 | 108/138 | 102653 | [model](../../../../models/talent/nomic_128/g-autoregressive/semantic_decoder.pt) |
| nomic_128/g-independent | 512 | 0.001 | 1e-05 | 0.1 | 100/130 | 335373 | [model](../../../../models/talent/nomic_128/g-independent/semantic_decoder.pt) |
| langvae_ft/g-autoregressive | 256 | 0.0003 | 0.001 | 0.3 | 359/388 | 102653 | [model](../../../../models/talent/langvae_ft/g-autoregressive/semantic_decoder.pt) |
| qwen3_64/g-autoregressive | 512 | 0.0003 | 1e-05 | 0.3 | 313/343 | 303101 | [model](../../../../models/talent/qwen3_64/g-autoregressive/semantic_decoder.pt) |
| langvae/g-autoregressive | 256 | 0.0003 | 0.001 | 0.2 | 386/416 | 102653 | [model](../../../../models/talent/langvae/g-autoregressive/semantic_decoder.pt) |
| langvae_ft/g-independent | 512 | 0.001 | 0.001 | 0 | 267/269 | 335373 | [model](../../../../models/talent/langvae_ft/g-independent/semantic_decoder.pt) |
| nomic_64/g-autoregressive | 512 | 0.001 | 1e-05 | 0.1 | 176/206 | 303101 | [model](../../../../models/talent/nomic_64/g-autoregressive/semantic_decoder.pt) |
| qwen3_64/g-independent | 512 | 0.0003 | 1e-05 | 0.3 | 376/406 | 302605 | [model](../../../../models/talent/qwen3_64/g-independent/semantic_decoder.pt) |
| langvae/g-independent | 512 | 0.001 | 0.001 | 0 | 277/307 | 335373 | [model](../../../../models/talent/langvae/g-independent/semantic_decoder.pt) |
| nomic_64/g-independent | 512 | 0.001 | 1e-05 | 0.1 | 156/186 | 302605 | [model](../../../../models/talent/nomic_64/g-independent/semantic_decoder.pt) |
| qwen3_32/g-autoregressive | 128 | 0.001 | 0.0001 | 0.3 | 305/335 | 22909 | [model](../../../../models/talent/qwen3_32/g-autoregressive/semantic_decoder.pt) |
| qwen3_32/g-independent | 128 | 0.001 | 1e-06 | 0.2 | 261/291 | 22413 | [model](../../../../models/talent/qwen3_32/g-independent/semantic_decoder.pt) |

## Storage and reproducibility

```text
models/talent/<encoder>/g-<variant>/semantic_decoder.pt        # active seed 42
models/talent/<encoder>/g-<variant>/experiments/g-all-encoders-v1/
  search/trial-NNN/semantic_decoder.pt
  final/seed-NN/semantic_decoder.pt
models/talent/<encoder>/g-<variant>/archive/g-all-encoders-v1/       # previous active model
reports/talent/<encoder>/g-<variant>/                         # matching reports/history
reports/talent/decoder_benchmarks/g-all-encoders-v1/
  summary.md, results.json, evaluation.json, selection.json
  protocol.json, plan.json, status.json, published.json
  configs/<encoder>.yaml, source/, evaluation/<encoder>/g-<variant>/
```

The protocol records source/input hashes, all cases, common candidates, exact splits and package versions. Each training report includes learning curves, best epoch and checkpoint checksum; per-seed evaluation JSON includes reliability bins. Existing active models are archived before replacement. Original embeddings are never rewritten.

Canonical active g checkpoints and training reports, this summary, benchmark JSON, per-seed evaluation JSON, configs and the frozen source snapshot are eligible for Git tracking; add, commit and push them to transfer them. Experiment checkpoints/reports, archives, logs and temporary files remain ignored. Historical references to local-only files are intentional; a clone is not a full resumable copy of every fit. The separately distributed LangVAE-FT encoder checkpoint is still required by its provenance checks.

Use the saved per-encoder YAML with `src.pipeline` and the matching `--decoder-variant` for downstream work. Do not mix checkpoints from different latent spaces. No h_Z training has been started.

## Data behind the statistics

g predicts structured X, T, D, U labels from the saved `data/latents/talent/<encoder>/z_pairs.pt` embeddings. The factual and counterfactual ground-truth labels are ID-aligned from `data/sim_talent/sim_data_factual.csv` and `sim_data_counterfactual.csv`, not reconstructed CV text. `data/sim_talent/pair_index.csv` is the official split authority; the original CVs are in `data/text_talent/cv_factual.csv` and `cv_counterfactual.csv`.

| Saved record | Statistics or evidence |
|---|---|
| [results.json](results.json) | Each final seed's validation NLL, selected settings, validation mean and sample SD |
| [evaluation.json](evaluation.json) | Aggregated factual-test and counterfactual-test scores for all cases |
| [evaluation/](evaluation/) | Individual-seed metrics, per-attribute scores, reliability bins and input/model hashes |
| [protocol.json](protocol.json), [plan.json](plan.json) | Exact fit/validation/test IDs, input hashes, configurations and search candidates |
| [selection.json](selection.json) | Validation-only ranking frozen before test evaluation |
| [published.json](published.json) | Active checkpoint and training-report paths and checksums |

The ± values are sample standard deviations across training seeds on the same data split, not confidence intervals over CVs. Per-CV predictions are not saved here; the JSON records contain aggregate metrics and reliability bins. The original source snapshot and protocol remain unchanged when this Markdown is regenerated.
