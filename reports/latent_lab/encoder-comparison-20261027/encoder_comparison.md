# Encoder comparison on the extended-DAG corpus

Generated from `reports/latent_lab/run-20261027-225936.json` (SHA-256 `b3291d2e801cac8dcacb45800e4d038f8d4fe9127f326c8fd18643d6f961047a`). No model fitting or new evaluation is performed.

EmbeddingGemma-300m at 768D has the best mean pool out-of-fold accuracy, evaluation joint NLL and evaluation joint accuracy among the 17 tested spaces. It also leads the separately trained evaluation-reader columns.

| Encoder | d | Pool OOF joint acc. ψ (%) ↑ | Eval. joint NLL ψ ↓ | Eval. joint NLL ψ_eval ↓ | Eval. joint acc. ψ (%) ↑ | Eval. joint acc. ψ_eval (%) ↑ |
|---|---:|---:|---:|---:|---:|---:|
| LangVAE | 128 | 17.6 ± 0.7 | 3.154 ± 0.010 | 3.271 | 19.1 ± 0.6 | 18.6 |
| Nomic Embed v1.5 | 64 | 16.3 ± 0.4 | 3.192 ± 0.009 | 3.273 | 19.0 ± 0.7 | 17.0 |
| Nomic Embed v1.5 | 128 | 24.9 ± 0.6 | 2.556 ± 0.007 | 2.679 | 25.9 ± 0.9 | 22.7 |
| Nomic Embed v1.5 | 256 | 33.0 ± 0.9 | 2.115 ± 0.012 | 2.326 | 33.3 ± 0.8 | 30.3 |
| Nomic Embed v1.5 | 512 | 41.5 ± 0.9 | 1.733 ± 0.022 | 1.923 | 42.3 ± 1.0 | 39.9 |
| Nomic Embed v1.5 | 768 | 44.9 ± 0.5 | 1.598 ± 0.013 | 1.782 | 45.9 ± 0.8 | 42.3 |
| Qwen3-Embedding-0.6B | 32 | 13.0 ± 0.7 | 3.537 ± 0.027 | 3.624 | 14.7 ± 0.7 | 13.1 |
| Qwen3-Embedding-0.6B | 64 | 18.9 ± 0.5 | 3.072 ± 0.057 | 3.145 | 21.0 ± 1.3 | 19.3 |
| Qwen3-Embedding-0.6B | 128 | 31.3 ± 0.2 | 2.198 ± 0.012 | 2.355 | 35.0 ± 0.4 | 29.5 |
| Qwen3-Embedding-0.6B | 256 | 41.8 ± 0.3 | 1.625 ± 0.010 | 1.762 | 45.9 ± 1.1 | 42.5 |
| Qwen3-Embedding-0.6B | 512 | 49.1 ± 0.8 | 1.376 ± 0.013 | 1.478 | 54.9 ± 1.0 | 51.3 |
| Qwen3-Embedding-0.6B | 768 | 52.3 ± 0.6 | 1.261 ± 0.009 | 1.368 | 58.9 ± 0.6 | 53.2 |
| Qwen3-Embedding-0.6B | 1024 | 54.8 ± 0.6 | 1.189 ± 0.008 | 1.348 | 60.0 ± 0.6 | 54.2 |
| EmbeddingGemma-300m | 128 | 62.8 ± 0.6 | 1.000 ± 0.010 | 1.060 | 65.0 ± 0.8 | 61.5 |
| EmbeddingGemma-300m | 256 | 74.8 ± 0.7 | 0.613 ± 0.004 | 0.715 | 77.8 ± 0.6 | 74.3 |
| EmbeddingGemma-300m | 512 | 79.3 ± 0.4 | 0.478 ± 0.005 | 0.585 | 82.2 ± 0.5 | 78.0 |
| EmbeddingGemma-300m | 768 | **81.0** ± 0.4 | **0.454** ± 0.002 | **0.556** | **82.7** ± 0.3 | **79.2** |

## Interpretation and differences from the old table

- All readers have independent heads. ψ_eval is an evaluation role, **not an autoregressive model**.
- ψ is fitted on 2,000 labelled pool units in five folds per seed. Each fold supplies 1,600 units to Reader.fit: 1,280 for optimization and 320 for internal epoch selection. The best internal validation checkpoint is restored. There are 25 fold fits per encoder/dimension.
- ψ_eval uses the other 2,000 units (800 pool + 200 development + 1,000 evaluation), with five folds at seed 0 and the same internal split. A unit's evaluation reader excluded that unit from both optimization and internal validation. This is out-of-fold evaluation, not a reader trained entirely outside the evaluation set; it also uses one fold model per unit rather than ψ's five-model ensemble.
- Joint NLL and accuracy refer to X, T, D, U. The networks additionally learn M, whose per-head results remain in the source report. Accuracies in the CSV are fractions; the displayed tables convert them to percentages. NLL uses natural logarithms.
- ± is the sample standard deviation across five ψ seeds, not a confidence interval or variation across folds. The evaluation reader has one seed, so no seed deviation is reported.
- The saved internal validation losses cover all **five** heads and different within-fold validation subsets. They are not the same quantity as the four-variable NLL in this table; they are therefore not relabelled as the old table's validation column.
- Autoregressive readers and LangVAE-FT were not run on this corpus. Null and Bayes-optimal joint reference rows were not computed in this comparison. None of their old-data numbers is carried forward.
- These descriptive results concern factual CVs in the original wording, not downstream counterfactual quality. The source report separately contains held-out wording results. Best mean does not establish statistically significant superiority; the test comparison is exploratory, and the pool OOF column provides a training-pool-only diagnostic.

## Suggested LaTeX caption

```latex
Frozen-encoder comparison on the extended-DAG factual CVs.
All readers have independent categorical heads and two width-256 GELU hidden layers;
the two reader columns denote roles, not independent versus autoregressive architectures.
Joint metrics cover $S=(X,T,D,U)$; the additional fitted $M$ head is excluded from them.
For $\psi$, pool accuracy uses out-of-fold predictions on 2,800 units, while evaluation
on 1,000 separate units averages the five fold models' marginal probabilities.
Entries for $\psi$ are means $\pm$ sample standard deviations across seeds 0--4.
The evaluation reader $\psi_{\mathrm{eval}}$ uses a disjoint labelled half of the data,
including evaluation units, with five-fold cross-fitting at seed 0: each unit is
scored by the model that excluded that unit. Its entries have no across-seed deviation.
NLL is in nats; accuracies are percentages. Bold entries mark the best mean per column,
without asserting statistical significance. All results use the original wording
$\mathcal V_A$; paraphrase robustness is a separate analysis.
```

## Regenerate

```bash
python scripts/encoder_comparison_table.py
```

The `.tex` file contains only the tabular, ready for the paper's existing table float. The CSV stores full-precision values, and the metadata records the source and generator hashes.
