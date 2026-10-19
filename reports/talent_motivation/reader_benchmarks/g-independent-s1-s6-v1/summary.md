# Independent readers: S1-S6

Run: `g-independent-s1-s6-v1`. Status: **complete**.

EmbeddingGemma 768D; factual L labels only; fixed development set. No official-test, held-out-wording, counterfactual or downstream evaluation.

Mean ± sample SD over seeds 42–46 (not confidence intervals). Development data select the models: these are not held-out performance estimates. NLLs across different state spaces are not comparable rankings.

| State | Heads | Development joint NLL | Selected trial |
|---|---|---:|---:|
| S1 | X, T, D, U | 0.546367 ± 0.012184 | 10 |
| S2 | X, D, U | 0.036647 ± 0.008902 | 16 |
| S3 | X | 0.018464 ± 0.005637 | 7 |
| S4 | X, T, U | 0.521972 ± 0.009139 | 10 |
| S5 | X, U | 0.050083 ± 0.007491 | 13 |
| S6 | X, T, D, U, M | 1.268115 ± 0.033527 | 10 |

## Development marginal accuracy

| State | Variable | Accuracy | ECE |
|---|---|---:|---:|
| S1 | X | 0.9950 ± 0.0000 | 0.0076 ± 0.0013 |
| S1 | T | 0.8570 ± 0.0091 | 0.0573 ± 0.0060 |
| S1 | D | 0.9980 ± 0.0045 | 0.0111 ± 0.0033 |
| S1 | U | 0.9670 ± 0.0084 | 0.0323 ± 0.0034 |
| S2 | X | 0.9950 ± 0.0000 | 0.0040 ± 0.0008 |
| S2 | D | 1.0000 ± 0.0000 | 0.0003 ± 0.0005 |
| S2 | U | 0.9970 ± 0.0027 | 0.0079 ± 0.0013 |
| S3 | X | 0.9950 ± 0.0000 | 0.0033 ± 0.0008 |
| S4 | X | 0.9950 ± 0.0000 | 0.0069 ± 0.0005 |
| S4 | T | 0.8500 ± 0.0061 | 0.0570 ± 0.0054 |
| S4 | U | 0.9570 ± 0.0027 | 0.0308 ± 0.0100 |
| S5 | X | 0.9950 ± 0.0000 | 0.0037 ± 0.0004 |
| S5 | U | 0.9960 ± 0.0042 | 0.0170 ± 0.0037 |
| S6 | X | 0.9950 ± 0.0000 | 0.0103 ± 0.0008 |
| S6 | T | 0.8480 ± 0.0067 | 0.0621 ± 0.0075 |
| S6 | D | 0.9980 ± 0.0027 | 0.0109 ± 0.0010 |
| S6 | U | 0.9830 ± 0.0076 | 0.0272 ± 0.0012 |
| S6 | M | 0.7060 ± 0.0207 | 0.0600 ± 0.0197 |

Each state's published checkpoint uses predeclared seed 42. All five selected seed checkpoints remain in its run directory. `protocol.json` records split IDs, input/code hashes, candidates and settings. Each fit's `training.json` contains its history and reload-verified development score.

Scope ends at g. Evaluation readers, cross-fitting, label-budget sweeps, symbolic operator fitting and h_Z training are not run.
