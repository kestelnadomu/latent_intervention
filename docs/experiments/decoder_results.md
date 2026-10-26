# Published CV-decoder results

Initial publication prepared on 2026-10-21 at the user's request, including full generated
CV text. This is an artifact-only publication: it does not retrain any model,
restore retired experiment code, or alter the encoder, `g`, or `h_Z`.

The initial publication contains **306 report/metadata files** (338.26 MiB
uncompressed). The largest file is 32.40 MiB. Every individual
file fits regular GitHub Git storage; no LFS or compression is required.
Existing tracked semantic-decoder and h_Z benchmark results are unchanged.

**2026-10-22 completion supplement:** the 1B factual/oracle comparison is now
finished. Its [completed result and interpretation](text_realizer_counterfactual_exposure_gemma1b.md#completed-result-and-decision-2026-10-22)
and [17-file manifest](../../reports/talent/embeddinggemma_768/text-decoder-results/publication-20261022-cf1b.json)
cover the final report, examples, training/selection histories, reload checks,
and completed statuses. Five plan/ETA files overlap the initial publication;
12 reports are new. The original dated manifest/checksums are preserved.
The supplement is prepared locally for the next commit, not automatically
committed or pushed by the documentation update.

## What is available after pulling

| Experiment | Report location | Interpretation |
|---|---|---|
| Original 270M campaign, seeds 42/43/44, factual audits, and human-review examples | `reports/talent/embeddinggemma_768/text-realizer/cv-text-v1-20261012-1719/` | Original factual-only decoder campaign; seed 42 remains the predeclared deployment seed. |
| All 270M architecture trials and replication histories | `.archive/artifacts/reports/talent/embeddinggemma_768/text-realizer/cv-text-v1-20261012-1719.*/` | Includes unsuccessful prefix-only/wide configurations, not just selected models. |
| Factual-only 1B pilot, compatibility checks, and its failed earlier launch | `.archive/artifacts/reports/talent/embeddinggemma_768/text-realizer/` under the `cv-text-1b-pilot-*` and `1b-pilot-compatibility-*` directories | Historical negative scaling result, not a promoted decoder. |
| 270M oracle-exposed A/B/C comparison | `reports/talent/embeddinggemma_768/text-realizer-cf/cv-cf-pilot-v1/` | Full per-ID comparison in `result.json`; oracle arm C did not meet the predeclared replication gate. |
| Factual continuation control B's training reports | `.archive/artifacts/reports/talent/embeddinggemma_768/text-realizer-cf/cv-cf-pilot-v1/factual/` | Retained equal-compute control, not a deployment decoder. |
| Factual draft-corrector, all epoch reports and final evaluation | `.archive/artifacts/reports/talent/embeddinggemma_768/text-corrector/cv-correction-pilot-20261016-1711/` | Negative pilot; publishing its results does not reactivate its code. |
| Completed 1B factual/oracle continuation comparison | `reports/talent/embeddinggemma_768/text-realizer-cf-1b/cv-cf-1b-pilot-20261021-0938/` | Completed 2026-10-22; `result.json` contains the full A/B/C evaluation. Predeclared advancement criteria failed. Reports are ready for the completion-supplement commit. |
| LangVAE FT smoke and selected training run | `models/langvae_ft/20260928T074231Z-smoke-8b2b824d/` and `models/langvae_ft/20260928T075426Z-train-c8d72b43/` | JSON audits, histories, selection, and generations only; no checkpoints. |
| Earlier LangVAE prototypes v1/v2/v3 | `models/langvae_posttrain/langvae_cv32_joint_v1/`, `langvae_cv_full_v2/`, and `langvae_cv_full_v3/` | Historical diagnostic/results JSONs, not changes to the retained `langvae_ft` baseline. |
| Finished preflight, reload, and timing diagnostics formerly stored with prepared data | `reports/talent/embeddinggemma_768/text-decoder-results/diagnostics/` | Byte-identical report copies; original data/cache locations are recorded in the manifest. |

Report JSONs and Markdown are kept at their existing locations, including archived
paths. The only copies are the eight small preflight/diagnostic exports from data
directories; source files were not modified. No model weights, optimizer states,
prepared tensors/training corpus, retired code, historical `source/` trees,
locks, logs, or redundant transfer tarballs are added to Git.

Historical plans contain source hashes and input identities. They do not imply
that the current code is the original run's source snapshot, or that the saved
result can be resumed without its separately retained weights and environment.

## The two previously requested files

The full 270M A/B/C comparison is
[`result.json`](../../reports/talent/embeddinggemma_768/text-realizer-cf/cv-cf-pilot-v1/result.json).

A's original factual-tune report is
[`factual_tune.json`](../../.archive/artifacts/reports/talent/embeddinggemma_768/text-realizer/cv-text-v1-20261012-1719.lora-compact.seed-42/factual_tune.json).
An identical copy already exists in the original campaign as
[`tune-seed-42.json`](../../reports/talent/embeddinggemma_768/text-realizer/cv-text-v1-20261012-1719/tune-seed-42.json).
The two copies have SHA-256
`745873bb24823846b6debe1e12aef22a2a60e1af522d0d0d92a2088772ad3473`.

Both reports identify the same original checkpoint:
`0d459c18343b1eb62573c7c0c19647b3eaf9b3b42e90a0412639220fd31adff7`.
They can be read and aggregated without a GPU or model download.

## Table construction: do not mix populations

In `result.json`, `worlds.<arm>.<world>` has two independent axes:

- Arms: `original` = A, `factual` = B, `oracle` = C.
- Worlds: `factual` evaluates Z against X; `oracle` evaluates **true oracle**
  Z' against X'. These are not predicted h_Z vectors.

Each model/world report contains the same **320 IDs**, with per-ID `rows`,
aggregate `summary`, and `facts.fields`. For example, A and C on oracle
targets are `worlds.original.oracle` and `worlds.oracle.oracle`.

The predeclared primary counterfactual comparison uses only **238 changed units**;
the other **82 are identities**. Its character/factual results appear in
`decision.arm_scores.<arm>.oracle_changed`. Do not combine this subset's scores
with all-320 chrF without explicitly labelling the different populations.

A's separate `factual_tune.json` covers **400 factual tune units**, including
the 80 later used for continuation checkpoint selection. For a matched 320-unit
factual comparison, use the factual-world entries in `result.json` instead.

Metric details:

- `summary.chrf` is the mean sentence-level chrF on a **0–1 scale**, not corpus
  chrF. The implementation is `src/text_realizer/metrics.py`.
- Each field's accuracy is
  `facts.fields.<field>.accuracy_on_scorable_references`. Preserve the report's
  scorable-reference counts and extraction coverage.
- The four-field macro uses country, degree, university tier, and project count.
  Region is reported separately and is not a fifth macro component.
- Retrieval and unedited-factual-CV reference rows must use the same IDs,
  target world, and original fit-only retrieval pool. Factual retrieval scores
  are not already oracle-target scores. CPU scoring requires the corresponding
  ID-aligned reference texts on the destination machine.

Saved results do not authorize a new audit or official-test evaluation. Read the
relevant experiment's protocol before interpreting development-set comparisons.

## Integrity and publication boundary

The [publication manifest](../../reports/talent/embeddinggemma_768/text-decoder-results/publication-20261021.json)
records every included report's path, byte count, SHA-256, experiment group, and
the source paths of diagnostic copies. Verify from the repository root:

```bash
sha256sum -c reports/talent/embeddinggemma_768/text-decoder-results/SHA256SUMS
```

All report JSONs were parsed, scanned for recognizable access-token/private-key
patterns, and checked against the 100 MiB per-file limit before staging.
Generated CV text has intentionally **not** been stripped.

The initial 2026-10-21 publication contained only the still-running 1B
experiment's preflight and immutable launch metadata. Its final comparison
completed on 2026-10-22 at 02:41 CEST. The supplement includes all 17 report
JSONs outside `source/`, while weights, optimizer states, and logs stay excluded.
No evaluation or training was rerun to prepare this documentation.

Verify the completed 1B report set separately:

```bash
sha256sum -c reports/talent/embeddinggemma_768/text-decoder-results/SHA256SUMS-cf1b-20261022
```

The 1B JSON uses the same `worlds.<arm>.<world>` schema and 320/238/82 population
boundaries described above. Its decision remains `worth_replication: false`;
this supplement does not open the audit, official test, or predicted-`h_Z`
evaluation, or promote a new decoder.
