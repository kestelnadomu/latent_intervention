# Preserved text decoders for package-generated latent vectors

Preservation check: **2026-10-26**, while merging `package` commit
`476b84245c6addee3894160dbcad0b87d82fdd23` into `decoder` at `b46b6ef`.
The package side was preferred for the sole merge conflict, in `.gitignore`.
The additive text-decoder modules, configurations, isolated environment, tests,
and historical reports were retained. No model was retrained or deleted.

**Inspection of the earlier fitted package run (2026-10-27):** the
`latent_lab` run present at the preservation check used
`Encoder("embeddinggemma", 256)`, and its saved readers and flows consume
256-D vectors. The 768-D cache described below is
the source from which those vectors are derived, not the fitted package model
dimension. These four historical decoders cannot directly decode its outputs.
See the separate [package 256-D decoder workflow](text_realizer_package_256.md)
for the new factual-only fit on `talent_motivation`.

**Subsequent package update (2026-10-27):** commit `18f09e3`, now merged into
`decoder`, fits the main pipeline at **768 dimensions** on the same package
corpus. The 256-D compatibility statements below describe the earlier snapshot.
The retained historical 768-D decoders match the new width but still have the
older `talent` training corpus. A separate [768-D package decoder comparison](text_realizer_package_768.md)
trains fresh factual-only 270M and 1B decoders on `talent_motivation`, preserving
all earlier checkpoints and the running 256-D jobs.

**Portable evaluation setup (2026-10-28):** the four new package-corpus
factual-only decoders have completed and exported their inference checkpoints.
The [package counterfactual evaluation runbook](text_realizer_package_evaluation.md)
provides `prepare`, `generate`, and `score` commands for all 1,000 test units,
from both oracle and predicted vectors. A verified portable bundle is ready
locally; full generation was launched on all four GPUs at 11:58 Europe/Berlin
on 28 October, with scoring scheduled after successful completion. Its manifest distinguishes
the earlier 256-D reader/flow protocol from the updated 768-D protocol.
Generation and scoring completed successfully at 22:42:52 that evening;
all 8,000 outputs and their checksums were verified on 29 October. The runbook
now includes the results and factual-fidelity limitations.

## Four retained inference checkpoints

Paths are relative to the repository root on this GPU machine. These are
trainable-only inference checkpoints: projector and LoRA weights, pinned base
model references, generation settings, provenance, and reload probes. They do
not contain optimizer state or a complete copy of Gemma's base weights.

| Decoder | Training exposure | Active checkpoint |
|---|---|---|
| Gemma 3 270M, original seed 42 | Factual `(Z, X)` only | `models/talent/embeddinggemma_768/text-realizer/cv-text-v1-20261012-1719/selected/seed-42.pt` |
| Gemma 3 270M, oracle continuation C | Factual and true oracle `(Z', X')` pairs | `models/talent/embeddinggemma_768/text-realizer-cf/cv-cf-pilot-v1/oracle/best.pt` |
| Gemma 3 1B, original seed 42 | Factual `(Z, X)` only | `models/talent/embeddinggemma_768/text-realizer/cv-text-1b-pilot-20261015-1026.lora-compact.seed-42/best.pt` |
| Gemma 3 1B, oracle continuation C | Factual and true oracle `(Z', X')` pairs | `models/talent/embeddinggemma_768/text-realizer-cf-1b/cv-cf-1b-pilot-20261021-0938/oracle/best.pt` |

The original factual 1B checkpoint was copied back into the active model tree;
its byte-identical original remains under `.archive/artifacts/` at the same
relative model path. Neither copy is added to Git. The equal-compute factual
continuation controls B are also untouched: 270M under
`.archive/artifacts/models/talent/embeddinggemma_768/text-realizer-cf/cv-cf-pilot-v1/factual/best.pt`
and 1B under
`models/talent/embeddinggemma_768/text-realizer-cf-1b/cv-cf-1b-pilot-20261021-0938/factual/best.pt`.
These controls are distinct from the original factual-only models above.

SHA-256 identities, in table order:

```text
270m-factual  0d459c18343b1eb62573c7c0c19647b3eaf9b3b42e90a0412639220fd31adff7
270m-oracle   fda6710871f3b67f681d5a42874dca8ab88e770ad1a528b6937692836e3f7b5b
1b-factual   59af957c6fdf29a86154efb40d66bbde4038ad82f5818a7a3afa0fcd133d84af
1b-oracle    32561b70c9fd26e31df84511273d5cf11e722cd3a6ad04befeba0b8c9390d2e2
```

All four passed `verify_checkpoint(..., device="cuda:0")` using the merged
working tree and cached base models. This validates checkpoint and base-file
hashes, reload logits, and exact greedy token agreement with the saved synthetic
probes. The focused offline suite passed **55 tests**, with three opt-in GPU
tests deselected; the four real-checkpoint checks were run separately.
No package CV, official-test target, or predicted counterfactual was decoded
during these preservation checks.

## Compatibility with package results

All four consume finite floating-point tensors shaped `[batch, 768]` from the
unchanged `google/embeddinggemma-300m` encoder at revision
`57c266a740f537b4dc058e1b0cda161fd15afa75`.

The new package manifest at
`data/latents/talent_motivation/embeddinggemma_768/manifest.json` has SHA-256
`33799b16597056f8f452fd6982df77f46ca404a04d50427f59d11a5eab101d90`.
Its encoder model, revision, dimension, maximum input length, complete encoding
protocol, and encoder/tokenizer file hashes match the preserved decoder's
training provenance. This includes the classification prompt, pooling,
projection, normalization, and FP32 inference protocol. Metadata field names
differ between the package manifest and decoder checkpoint; compare their
meaning and full protocol, not raw dictionary equality across schemas.

This establishes compatibility with the **full 768-D cache**, not with the
package's fitted 256-D reader/flow outputs, and **not reconstruction quality** on new
CVs or on predicted `h_Z` outputs. These decoders were trained on the older
`talent` CV corpus, not the new `talent_motivation` corpus. New templates,
attributes, interventions, and off-distribution edited vectors can affect
performance. The saved evaluation results are not measurements on the new
package data.

Use these historical checkpoints only with compatible **768-D** outputs; the
actual saved `latent_lab` run does not supply those outputs. Its 256-D vectors,
LangVAE, and Nomic outputs are not interchangeable. Do not pad 256-D vectors or
silently normalize edited vectors to make them fit. A stochastic editor's sample
or component must be chosen reproducibly and remain aligned with its candidate
ID and intervention; averaging latent modes is not a substitute for a sample.

## Existing inference entry points

Keep the isolated environment and cached model snapshots; pulling Git alone does
not transfer these local assets. The interpreter is:

```text
exp/text_realizer/environment/.venv/bin/python
```

The reusable API does not need the source CV, structured state, reader, or editor:

```python
from src.text_realizer import load_text_realizer

# checkpoint: one of the paths above.
# z_prime: ID-aligned [batch, 768] tensor from a provenance-checked package editor.
decoder = load_text_realizer(checkpoint, device="cuda:0")
texts = decoder.decode(z_prime)
```

The existing `python -m exp.text_realizer generate` command is also retained.
It accepts `--checkpoint`, `--latents`, `--device`, and `--output`. Its vector
file must contain exactly `ids`, `z`, `encoder`, and `provenance`; `z` holds the
predicted vectors. It intentionally rejects canonical factual/counterfactual
pair bundles. After checking the actual source encoder, the exporter must use
the decoder's encoder-metadata schema; do not relabel incompatible vectors just
to pass that check. Record the editor checkpoint, source artifact, sample seed,
and intervention in the export provenance. There is no automatic package-result
conversion or inference launch in this preservation change.

The cached base models live under `.cache/huggingface/hub`:

- `google/gemma-3-270m`, revision `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`.
- `google/gemma-3-1b-pt`, revision `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`.

The runtime remains in `src/text_realizer/`; the CLI is in `exp/text_realizer/`.
Neither depends on replacing the package's encoder, reader, or editor. Keep the
environment lockfile and the persistent interpreter under
`.cache/text-realizer-python` when cleaning local files. Git-ignore rules are
not backups: do not run `git clean -fdx` over these local assets. `.archive/`
also remains excluded locally through `.git/info/exclude`.

## Scientific labels remain unchanged

Use the factual-only models for a pipeline without counterfactual decoder
training pairs. Label the oracle-exposed models as privileged-supervision
comparisons, even when their input at inference is a predicted vector.
The oracle-exposed 270M had the better aggregate results in the historical
oracle-vector comparison, but none of these preservation checks selects a best
decoder for package-generated vectors or opens a new evaluation gate.

See [published results](decoder_results.md), the
[270M counterfactual-exposure study](text_realizer_counterfactual_exposure_plan.md),
and the [1B comparison](text_realizer_counterfactual_exposure_gemma1b.md).
