# Parallel factual-only Gemma 3 1B package decoder

Authorized and launched on 2026-10-27 as a larger-backbone comparison to the
[270M package decoder](text_realizer_package_256.md). This is **factual-only**,
with no oracle pairs or evaluation CVs used for fitting or model selection.
The frozen encoder, package readers and flows are unchanged.

## Controlled comparison

Configuration: `configs/text_realizer_talent_motivation_256_gemma1b.yaml`.
It is validated against the 270M configuration: only the pinned decoder base
and model/report output directories may differ. Both use exactly the same
prepared factual data, literal 2,800 fit / 200 development IDs, seed 42,
256-D vectors, complete CV targets, AdamW settings, effective batch 32,
30-epoch cap, patience 5, and lowest-development-NLL selection. The 1,000
package evaluation IDs remain reserved. Token limits are audited independently
using each pinned tokenizer on fit/development CVs only.

The 1B base is non-instruction-tuned `google/gemma-3-1b-pt`, revision
`fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`, with explicit eager attention.
Its prefix projector is `256 → 768 → 16 × 1152`, using the same GELU and fixed
`sqrt(256)` input scaling as the 270M recipe. The output width follows the
backbone's native embedding width. Train a fresh projector and LoRA adapters
(rank 16, alpha 32, dropout 0.05); keep base parameters, embeddings and output
head frozen. No old factual or oracle decoder supplies initialization.

This is a comparison under the same training recipe and epoch budget, not
equal GPU-hours or equal trainable-parameter count. Earlier 1B experiments on
the old 768-D `talent` corpus were not better overall; increased model size is
not a guarantee of more faithful reconstruction on the new corpus.

## Parallel execution and isolation

Initial run: **`package-256-1b-20261027-0958`**, physical **GPU 0**.
The concurrent 270M run **`package-256-20261027-0934-eager`** remains on
physical **GPU 2**. Each worker sees its selected physical GPU as `cuda:0`.
Each model fits and runs independently on one card; no distributed training or
GPU memory pooling is involved. The workers share only read-only base-model
caches, the immutable prepared factual bundle, and reusable source code.

The new `exp/text_realizer_package_1b/` entry point calls the existing package
preparation, preflight, numerical training and evaluation functions unchanged.
It has its own configuration loader, subprocess launcher and source checks.
No file included in the running 270M run's immutable source identity was
modified. Its source/configuration identity was checked before the 1B launch.
The additional 1B source hashes are included in the configuration, which is
already part of the shared immutable plan/checkpoint identity. They are also
snapshotted and checked at startup and before/after every phase. Changed
source or configuration requires a new run ID.

```bash
bash scripts/run_text_realizer_package_1b.sh UNIQUE_1B_RUN_ID 0
# After interruption, with unchanged code/configuration:
bash scripts/run_text_realizer_package_1b.sh EXISTING_1B_RUN_ID 0 --resume
```

The detached tmux session is
`text-realizer-package-1b-package-256-1b-20261027-0958` for the initial run.
The launcher runs the offline suite, verifies the pinned local snapshots,
acquires the selected GPU lease, then runs the full 1B preflight. Full training
starts only if preflight passes. Other GPU jobs are never terminated. If the
GPU becomes occupied before acquisition, the worker waits.

Preflight is repeated for this larger backbone: longest-target FP32 memory
checks with at least 15% headroom, native embedding/cached-logit correctness,
20 smoke updates, two-example overfitting, frozen-parameter and gradient checks,
reproducible resume, fresh-process reload, 256-D encoder parity, joint decoder
plus cycle-encoder evaluation memory, and a 100-update timing benchmark.
Microbatch is chosen independently (1 or 2) while effective batch stays 32.
Diagnostic weights are discarded before the fresh full training run.

The initial extension passed its eight focused offline tests, Ruff and shell
syntax checks. The launcher additionally runs the existing package and decoder
regression suites: **150 passed, three opt-in GPU tests deselected** at startup.
At approximately **09:59 Berlin time on 2026-10-27**, the 1B worker was running
preflight; the 270M worker had completed all preflight checks and started full
training on GPU 2. This record does not imply the 1B GPU preflight or training
has finished; inspect the live status files.

By **10:00 Berlin time**, 1B had passed the native cache check (maximum absolute
logit difference `3.24e-5`) and longest-target memory checks. Microbatch 2 was
selected, reserving **8.98 GB** of **11.54 GB** with the required headroom; 20
smoke updates had completed. Overfit/recovery, benchmark and final reload/cycle
checks were still pending. The old timing estimate below used microbatch 1 and
will be replaced by this run's measurements.

## Outputs and timing

```text
models/talent_motivation/embeddinggemma_256/text-realizer-1b/<run-id>/
reports/talent_motivation/embeddinggemma_256/text-realizer-1b/<run-id>/
```

The model tree retains `last.pt`, `best.pt`, and, after development diagnostics
and verified reload, the portable `inference/model.pt` and
`inference/manifest.json`. The report tree contains `status.json`,
`preflight/status.json`, `preflight/ready.json`, `eta.json`, `history.json`,
`development.json`, fixed-ID examples and the frozen manifest. The prepared
bundle is the same immutable artifact used by 270M; no corpus re-encoding is
needed. Both new output roots are locally ignored in `.git/info/exclude`, with
no shared ignore change or automatic commit/upload.

Allow provisionally **24–36 hours after GPU acquisition** for preflight, the
full 30-epoch budget and development diagnostics. The previous 1B preflight
on this GPU type measured roughly 23.6 seconds per effective-batch update at
microbatch 1, but the new benchmark is authoritative. Early stopping may
finish substantially sooner; generation length and the selected microbatch
can change runtime. `eta.json` appears during the current benchmark. This is
not a promise of a particular completion time.

Inference uses the existing `src.text_realizer.load_text_realizer` API or
`python -m exp.text_realizer_package generate`, supplied with the new 1B
checkpoint and finite ID-aligned `[batch, 256]` vectors. Edited vectors are not
silently normalized. Neither source CVs nor structured labels are needed.

Comparison uses the same development diagnostics as 270M: correct/shuffled/
constant/nearest conditioning, lexical reconstruction, conservative factual
extraction, memorization, fit-only retrieval and 256-D cycle metrics. These
are development results, not a held-out counterfactual success claim. The
five-CV illustration and official evaluation remain separate future tasks.
