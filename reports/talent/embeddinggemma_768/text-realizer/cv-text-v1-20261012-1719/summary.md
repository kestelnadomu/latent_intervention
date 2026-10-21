# EmbeddingGemma text realization

No counterfactual data were used for development.

| Configuration | Eligible | Character similarity | NLL gap | Character gap |
|---|---:|---:|---:|---:|
| lora-compact | True | 0.3294 | 0.6464 | 0.0554 |
| lora-wide | True | 0.3280 | 0.6923 | 0.0563 |
| prefix-compact | False | 0.2743 | 0.9103 | 0.0173 |
| prefix-wide | False | 0.2764 | 0.9333 | 0.0173 |

Selected configuration: lora-compact.
Deployment seed: 42. Seeds 43 and 44 are repetitions, not selection candidates.

Audit improvement criterion passed: False.
Paired retrieval comparison: {'mean': 0.015053252925133457, 'lower_95': 0.008696940731243859, 'upper_95': 0.021316996989095973, 'samples': 2000, 'seed': 20261012}.
Exact inversion and counterfactual faithfulness are not guaranteed.
