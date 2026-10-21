# Retired CV text-decoder experiments

Archived on 2026-10-20. This folder preserves experiments that are **not active
decoders**. On 2026-10-21 the 1B pilot source/config/tests/launcher were restored
to the active code tree for Git transfer; its trained artifacts remain here.
Other retired experiment-specific source is in `code/`, and local checkpoints,
optimizer states, prepared caches, and reports are under `artifacts/`.
On 2026-10-21 the user approved publishing the full result reports, including
generated CV text. JSON/Markdown files under `artifacts/reports/` are now tracked
at their archived paths. See the [results index and checksums](../docs/experiments/decoder_results.md).
Weights, optimizer states, prepared caches, historical report `source/` snapshots,
and `code/` remain local; committing the reports does **not** transfer those
artifacts or reactivate a retired decoder.

## The two retained inference checkpoints

| Role | Active path | SHA-256 |
|---|---|---|
| Original factual-only Gemma 3 270M decoder (selected seed 42) | `models/talent/embeddinggemma_768/text-realizer/cv-text-v1-20261012-1719/selected/seed-42.pt` | `0d459c18343b1eb62573c7c0c19647b3eaf9b3b42e90a0412639220fd31adff7` |
| Oracle-exposed Gemma 3 270M decoder (selected update 600) | `models/talent/embeddinggemma_768/text-realizer-cf/cv-cf-pilot-v1/oracle/best.pt` | `fda6710871f3b67f681d5a42874dca8ab88e770ad1a528b6937692836e3f7b5b` |

Both use the shared active loader in `src/text_realizer/`. The oracle-exposed
checkpoint is an **exploratory, privileged-supervision comparator**: it was
trained on some aligned oracle `(Z', CV')` pairs, did not meet the predeclared
replication gate, and has not been tested on `h_Z` output vectors. It does not
replace the original decoder or change `g`, `h_Z`, or the encoder.

## What was archived

| Experiment | Archived material | Outcome / reason not retained |
|---|---|---|
| Original 270M architecture search and seed replications | Unselected `prefix-*` and `lora-wide` model/report directories; `lora-compact` trial best/last and seeds 43/44; selected seed-43/44 bundles, all under `artifacts/{models,reports}/talent/embeddinggemma_768/text-realizer/` | The predeclared deployment model is the selected seed-42 bundle above. The archived `last.pt` files contain optimizer/resume state. See [original workflow](../docs/experiments/text_realizer.md). |
| Gemma 3 1B pilot | Pilot-specific source, config, launcher, and tests restored to `exp/text_realizer/`, `configs/`, `scripts/`, and `tests/text_realizer/`; model best/last, reports, and pilot prepared caches remain under `artifacts/` | Lower free-running CV fidelity despite better teacher-forced NLL. See [1B result and caveats](../docs/experiments/text_realizer_gemma1b_results.md). |
| Factual draft-corrector | Full `src/text_corrector/`, `exp/text_corrector/`, tests, config, and launcher under `code/`; all model, report, and prepared-data directories under `artifacts/` | Failed its predeclared continuation gate. See [runbook](../docs/experiments/text_corrector.md) and [result](../docs/experiments/text_corrector_results.md). |
| Factual-only 600-update continuation (arm B) | Its best/last checkpoints and arm-specific training report under `artifacts/{models,reports}/talent/embeddinggemma_768/text-realizer-cf/cv-cf-pilot-v1/factual/` | This was the equal-compute control, not a deployment decoder. The combined A/B/C comparison remains in the active pilot `result.json`; see [protocol and result](../docs/experiments/text_realizer_counterfactual_exposure_plan.md). |
| Non-promotable counterfactual-pilot preflight weights | `artifacts/data/latents/talent/embeddinggemma_768/text-realizer-cf-prepared/<prepared-id>/diagnostics/` | Smoke/reload checks only, not trained inference bundles. |
| Redundant resume state for retained oracle model | `artifacts/models/talent/embeddinggemma_768/text-realizer-cf/cv-cf-pilot-v1/oracle/last.pt` | The active `best.pt` above is the selected inference bundle; `last.pt` exists only for training resume. |

The original campaign, 1B pilot, and oracle-comparison code share modules in
`src/text_realizer/`, `exp/text_realizer/`, and `exp/text_realizer_cf/`.
Those modules stay active because the retained checkpoints need their loader,
the encoder protocol, and comparison provenance. Support for multiple projector
variants inside the shared model file is not a separate runnable experiment;
only the two inference checkpoints listed above remain in the active model tree.

Other archived source and historical runbooks are **not runnable in place**.
The restored 1B pilot CLI is runnable, but its pinned historical comparison
requires the archived 270M development bundle and tune report at the paths in
its config. To reproduce a retired run, restore its source/config/artifacts to
the recorded original paths or adapt the paths explicitly. Historical report source
snapshots and hashes remain with each run. Canonical encodings, original CVs,
the frozen base-model cache, LangVAE, `g`, and `h_Z` were not moved.
