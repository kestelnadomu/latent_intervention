# Review map: lightweight entry points, unchanged results

This cleanup separates the additions made for multiple embedding dimensions and
validation-selected `g` training from the established modules. It changes code
organization, not the experiment. **Existing encodings do not need to be rerun.**

## Where to read what

| Established file | Responsibility retained | Extracted implementation |
| --- | --- | --- |
| [config.py](../src/config.py) | Load YAML, apply overrides, resolve artifact paths; preserve previous imports | [encoder_protocols.py](../src/encoder_protocols.py): supported dimensions, pinned defaults, protocol validation, encoder-specific metadata and lazy dispatch |
| [config.yaml](../src/config.yaml) | Explicit experiment settings, model pins, widths and training budget | Deliberately kept visible in one file; no hidden profile merging or changed values |
| [encoder.py](../src/encoder.py) | LangVAE/Nomic numerical encoding and the existing factory | [encoding_progress.py](../src/encoding_progress.py): shared console logging only; Qwen/Gemma remain in their dedicated encoder files |
| [pair_encoding.py](../src/pair_encoding.py) | Align IDs/splits, encode the same texts, preserve identity pairs, validate/load artifacts | [latent_writer.py](../src/latent_writer.py): no-overwrite publication, input hashes, metadata and atomic writes |
| [pipeline.py](../src/pipeline.py) | Select stages, align labels, choose the official split, call training/evaluation | [decoder_reporting.py](../src/decoder_reporting.py): training protocol, best-checkpoint callback, progress and final reports |
| [semantic_decoder.py](../src/semantic_decoder.py) | Both `g` architectures, loss definitions, metrics, checkpoint format and loading | [decoder_training.py](../src/decoder_training.py): unchanged optimization, validation selection, scheduling and early stopping |

Public commands and imports remain available, including
`src.semantic_decoder.train_semantic_decoder`, `src.config.encoder_dimension`,
`src.pair_encoding.write_latent_artifact`, and all existing CLI dimension flags.
The pipeline still calls `train_semantic_decoder`; the model module re-exports
that function. No compatibility wrapper changes its argument signature.

## What is unchanged

- Model revisions, tokenization, prompts, pooling, learned projections,
  normalization, widths and inference precision.
- Input text files, official pair IDs/splits and exact identity-vector copies.
- Artifact paths, tensors, sidecars, compatibility identities and saved model
  architecture/state-dict keys.
- Both `g` losses and initialization/shuffling behavior, the 500-epoch ceiling,
  validation selection, learning-rate schedule and early-stopping rules.
- Previous run logs, source snapshots, provenance manifests and checkpoint files.

The main YAML remains explicit because moving parameter values behind an implicit
merge would make scientific review harder. The remaining edits in established
scripts are dimension plumbing, small delegation calls and required atomic I/O.

## Verification without repeating expensive work

Before editing, [artifact_audit.py](../src/artifact_audit.py) recorded a reference at
`reports/talent/refactor/before.json`. The audit checks:

1. All 17 completed latent spaces: file and sidecar hashes, resolved configurations,
   encoder identities, IDs, splits, tensor validity, unit norms where applicable,
   identity pairs and cross-dimension consistency.
2. Four active `g` checkpoints: hashes, metadata, reports and predictions on a fixed
   32-row sample.
3. Both training procedures on a tiny synthetic fixture: exact losses, selection
   histories and weight fingerprints. This does not retrain any production model.

Repeat the comparison locally with:

```bash
.venv/bin/python -m src.artifact_audit --compare reports/talent/refactor/before.json
.venv/bin/python -m pytest -q
```

The reference report is Git-ignored, like other run reports; copy it explicitly if
you want the same before/after check on another server. The repository regression
test [test_refactor_compatibility.py](../tests/test_refactor_compatibility.py) also
pins all 17 encoder metadata fingerprints without needing model downloads or data.
No full encoder run or production training is performed by this verification.

Temporary-file round-trip checks also exercised the extracted writer for all 17
latent spaces in their original environments. Every tensor and non-container-hash
metadata field matched; the temporary copies were not used to replace any result.

## Historical provenance versus a future run

Old queue manifests correctly describe the source code that produced the existing
outputs. They are **not rewritten** to pretend that this refactor was the original
producer. Loading existing latents/checkpoints does not require matching today's
source-file hashes; it checks their recorded content and semantic compatibility.

A future queue or decoder experiment needs a **fresh run ID** after source changes.
Qwen/Gemma then need their bounded preflight for that new run ID. This is a guard
against mixing experiment revisions, not a requirement to regenerate existing
embeddings. Downstream `train-decoder` can use the existing files directly.
