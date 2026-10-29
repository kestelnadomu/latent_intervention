# Factual-only package CV decoders at 768 dimensions

This is the full-width counterpart of the [256-D package decoder](text_realizer_package_256.md)
and its [1B comparison](text_realizer_package_256_gemma1b.md). Both models learn
`D_text(Z) -> CV` on the current `talent_motivation` corpus. They are separate
from the historical 768-D decoders trained on `talent`.

The package update `18f09e3` (27 October 2026) uses
`Encoder('embeddinggemma', 768)` for the main downstream pipeline. These new
decoders match that representation. EmbeddingGemma, the cached vectors, package
readers and flows, and the existing 256-D decoder runs remain unchanged.

## Matched comparison

| Setting | 270M | 1B |
| --- | --- | --- |
| Base | `google/gemma-3-270m` | `google/gemma-3-1b-pt` |
| Revision | `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1` | `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29` |
| Projector | `768 -> 768 -> 16 x 640` | `768 -> 768 -> 16 x 1152` |
| Configuration | `configs/text_realizer_talent_motivation_768.yaml` | `configs/text_realizer_talent_motivation_768_gemma1b.yaml` |
| Model/report directory suffix | `text-realizer` | `text-realizer-1b` |

Initial detached runs were launched on **27 October 2026 at 15:51 Europe/Berlin**:

| Backbone | Run ID | Physical GPU |
| --- | --- | --- |
| 270M | `package-768-270m-20261027-1551` | 1 |
| 1B | `package-768-1b-20261027-1551` | 3 |

Both launchers passed 176 offline tests and entered GPU preflight. The existing
256-D jobs continue independently on GPUs 2 and 0. Each new run writes a
`launch_estimate.json` based explicitly on the corresponding 256-D measurements;
allow roughly 18 hours for 270M and 27 hours for 1B, including preflight and
diagnostics. These are provisional scheduling budgets, not measured 768-D
completion times. The run-specific `eta.json` supersedes them once benchmarking
has measurements. Early stopping can finish sooner.

At the initial check, both preflights passed native embedding scaling and
cached/full-logit parity across the 512-token sliding-window boundary (maximum
absolute differences `3.62e-5` for 270M and `4.20e-5` for 1B). Both selected
microbatch two with the required headroom: peak reserved memory was 5.54 GB and
8.95 GB, respectively, on 11.54 GB devices. Each completed its 20 smoke updates;
overfit, benchmark and final readiness checks were still in progress. This is
an interim launch record, not a claim of completed preflight or training.

The training recipe matches the 256-D runs: seed 42, two-layer GELU projector,
16 prefix tokens, fixed input scaling `sqrt(latent_dim)`, LoRA rank 16 / alpha
32 / dropout 0.05, frozen base parameters, FP32 and eager attention. Projector
LR is `1e-3`, LoRA LR `2e-4`, effective batch 32, weight decay 0.01, clipping
1.0, 5% warmup and cosine decay. Training allows 30 epochs with patience five;
the checkpoint with minimum development token NLL is retained. No previous
decoder weights initialize either run.

Relative to 256-D, the first projector layer gains 393,216 weights. Prefix
length, hidden width, LoRA capacity and language-model context are unchanged.
The larger vector does not triple language-model training cost. Better text
recovery remains an empirical question; additional dimensions or a larger base
model do not guarantee faithful inversion.

## Data boundaries and identity

- Use the existing package partition: 2,800 pool IDs for fitting and 200 dev
  IDs for checkpoint selection and diagnostics. The 1,000 eval IDs remain closed.
- Verify the same factual CSV, pair index, partition, latent artifact and
  encoder-provenance hashes as the 256-D runs. Sort and join strictly by ID.
- Use full cached FP32 768-D factual vectors unchanged. Validate unit norms,
  but do not truncate, pad or renormalize them. This matches the full-width
  package encoder loader exactly.
- Access only `ids` and `z` from the trusted paired tensor container. No
  counterfactual tensor, counterfactual text, or evaluation target enters
  preparation, training, selection, token-length decisions or diagnostics.
- Check that the prepared IDs and raw CV targets match the 256-D bundle and
  that normalizing their first 256 coordinates reproduces that bundle exactly.
- Train on complete CV targets plus EOS. BOS, prefix and padding labels are
  masked. Factual labels and template/persona fields are not training inputs.

Both backbones share one prepared factual bundle. Their run plans, training
states and outputs are separate. Source/configuration/environment identities
are immutable; changing them requires a new run ID.

## Implementation and execution

The additive entry point is `python -m exp.text_realizer_package_768`.
Dimension-specific preparation, stage orchestration and inference validation
live in `exp/text_realizer_package_768/`. It reuses the shared model, numerical
training primitives, metrics, GPU leases and existing preflight learning/memory
checks. The 256-D sources remain unchanged because running jobs bind their
hashes. Dedicated stage modules retain the same training and evaluation logic.

Use the existing isolated environment; no root dependency or environment
changes are needed:

```bash
exp/text_realizer/environment/.venv/bin/python -m exp.text_realizer_package_768 prepare
bash scripts/run_text_realizer_package_768.sh 270m UNIQUE_270M_RUN_ID 1
bash scripts/run_text_realizer_package_768.sh 1b UNIQUE_1B_RUN_ID 3
```

The GPU argument is a physical GPU index. Each launcher creates its own tmux
session, log, run lock and GPU lease. An occupied selected GPU is queued;
unrelated processes are never terminated. A new run does not overwrite an
existing run ID. To continue an interrupted run, repeat its exact command with
`--resume`; optimizer, scheduler, RNG, sampler and early-stopping state restore
from the atomic last checkpoint.

The detached workflow runs offline tests and verifies cached model files before
GPU work. Its preflight checks native/cached logits, memory with at least 15%
headroom, 20-update learning, a two-example overfit, nonzero trainable gradients,
frozen base parameters, reproducible continuation, fresh-process reload and
768-D encoder parity. It then benchmarks 100 optimizer updates and generation.
Training starts automatically only if preflight passes. A failed check records
the reason and stops the workflow.

The implementation checks passed 176 offline tests (three real-GPU tests
deselected), including protected factual-only access, full-width joins,
checkpoint reload and interrupted-versus-uninterrupted training. Real-model
preflight results are recorded separately for each run.

## Artifacts and monitoring

All generated files stay below these new package-corpus directories:

```text
data/latents/talent_motivation/embeddinggemma_768/text-realizer-prepared/<hash>/
models/talent_motivation/embeddinggemma_768/text-realizer[-1b]/<run-id>/
reports/talent_motivation/embeddinggemma_768/text-realizer[-1b]/<run-id>/
```

The machine-local `.git/info/exclude` excludes these generated subdirectories;
the shared repository ignore rules remain unchanged. Nothing is staged or
uploaded automatically. Base weights are shared through the existing cache.

Each report directory contains `launcher.log`, `status.json`, immutable plans,
`preflight/status.json`, readiness evidence and an `eta.json` once benchmarking
has measurements. ETA includes up to 30 epochs, validation and development
diagnostics with a 25% margin, but excludes GPU queue time. Early stopping can
finish sooner. The workflow ends only after diagnostics and export, not merely
after the last training epoch.

Final diagnostics use correct, shuffled, mean, zero and nearest-fit conditioning,
fit-only retrieval, lexical and independent factual checks, 768-D encoder cycle
metrics and grouped reports. Development results are selection-biased and are
not a held-out evaluation. Oracle evaluation remains closed.

## Inference after completion

The exported model is `inference/model.pt`, with a provenance/checksum manifest.
Its runtime requires only finite `[batch, 768]` vectors:

```python
from src.text_realizer import load_text_realizer

decoder = load_text_realizer("PATH_TO_RUN/inference/model.pt", device="cuda:0")
texts = decoder.decode(z_prime)
```

For ID-aligned files, use `python -m exp.text_realizer_package_768 generate` with
`--checkpoint`, `--latents`, `--output` and `--device`. The vector file must
contain exactly `ids`, `z`, `encoder` and `provenance`, matching the checkpoint's
encoder metadata. Edited vectors are not silently normalized. Preserve package
editor, intervention and sampling provenance. Dimensional compatibility alone
does not establish factual or counterfactual fidelity.
