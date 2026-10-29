# Factual-only CV decoder for the 256-D package representation

Implementation date: 2026-10-27. This is a new `D_text` for the actual
`latent_lab` package run, using `talent_motivation` and
`Encoder("embeddinggemma", 256)`. It preserves the four historical 768-D
decoders, the canonical embeddings, the reader, the renderers and their fitted
package caches. It neither refits the latent editor nor changes the SCM.

An independent [factual-only 1B comparison](text_realizer_package_256_gemma1b.md)
was added on 2026-10-27 to run on GPU 0 alongside this 270M job on GPU 2.
Its entry point reuses this numerical implementation without changing the
running experiment's source identity.

Initial launch (failed preflight): **`package-256-20261027-0054`**, physical **GPU 2**, in tmux
session `text-realizer-package-package-256-20261027-0054`. All four GPUs were
occupied by unrelated Blender jobs at launch, so the worker waits for its
selected GPU before attempting preflight. Consult the run's `status.json` and
`preflight/status.json` for the current state; this launch record is not a
claim that tuning finished. The implementation passed **140 offline tests**
(three opt-in GPU tests deselected), Ruff, shell syntax and Git whitespace
checks. SHA-256 checks confirmed all four retained decoder checkpoints were
unchanged.

### Preflight correction, 2026-10-27

The initial run acquired GPU 2 overnight, then stopped at **03:11 Berlin time**
before smoke updates or full training. Its cached/full logit check failed at
the 512-token local-attention boundary (maximum absolute difference **21.16**).
The new configuration had inadvertently inherited the historical 270M SDPA
default, despite the backend issue already documented in the
[original runbook](text_realizer.md#separate-gemma-3-1b-pilot).

The package decoder now explicitly uses **eager attention**, and configuration
validation rejects an absent backend or SDPA. This changes the numerical
attention implementation, not the architecture, data, loss, learning rates or
training budget. The original failed run and source snapshot remain intact;
the corrected run must use a new ID. Historical decoder configurations and
weights remain unchanged.

A separate real-GPU reproduction of the exact preflight sequence (longest-CV
memory checks, then native cached/full comparison) passed with the **original
tolerance** (`rtol=2e-5`, `atol=2e-4`): maximum absolute logit difference
**7.13e-5**, with deterministic generation and correct embedding scaling.
Microbatch 2 reserved **5.54 GB** of **11.54 GB**, satisfying the 15% headroom
gate. Evidence is in
`reports/talent_motivation/embeddinggemma_256/text-realizer/attention-backend-check-20261027/eager.json`.
An additional check before the diagnostic memory updates produced a much
smaller eager discrepancy of `4.30e-4` but exceeded the strict absolute
tolerance; it is not counted as a passing check. No tolerance was relaxed.
The complete preflight is repeated before training can start. Intermediate
memory, native-backend and learning checks are now persisted, and failure
states include the exception instead of remaining marked as in progress.

Corrected launch: **`package-256-20261027-0934-eager`**, GPU 2, on the morning
of 2026-10-27. Its startup suite passed **142 tests** (three opt-in GPU tests
deselected). At **09:34 Berlin time** the new run had passed memory and native
cache checks and was executing the remaining learning/recovery/benchmark
preflight. Full training had not started yet. Its live report directory is
`reports/talent_motivation/embeddinggemma_256/text-realizer/package-256-20261027-0934-eager/`;
the launcher automatically starts training only after complete preflight success.

At **09:59 Berlin time on 2026-10-27**, the corrected 270M run had passed the
complete preflight and started full training on GPU 2. Both diagnostic CVs
reconstructed exactly; resume and fresh reload passed. The benchmark measured
11.63 seconds per effective-batch optimizer update and 27.49 seconds per
generated CV. Its full 30-epoch training estimate was 8.53 hours, plus validation,
development diagnostics and the explicit 25% timing margin: about **17.15 hours
total**, provisionally **03:08 Berlin time on October 28**. Early stopping may
finish sooner. This measured estimate supersedes the initial historical
12–16-hour estimate below.

Until this run's benchmark is available, allow approximately **12–16 hours
after GPU availability**, including preflight, up to 30 training epochs and
development diagnostics. This provisional estimate uses the historical 270M
measurement (about 10.8 seconds per effective-batch update and 10.4 seconds per
generated CV on the same GPU type with similarly long targets); it is not a
new package benchmark. GPU queue time is additional and unknown.

## Why a separate fit is necessary

The package report `reports/latent_lab/run-20261024-164358.json` records a
**256-D** encoder. Its fitted readers and flows also use 256 dimensions, despite
the full cache being stored under `embeddinggemma_768`. The package derives
the actual input as:

```python
z_256 = torch.nn.functional.normalize(cached_z.float()[:, :256], dim=1)
```

The historical decoders consume 768-D vectors and were trained on the older
`talent` corpus. Their compatibility with the full encoder cache does not make
them compatible with the package's fitted 256-D models. Padding or relabelling
the new vectors would not fix this. The new decoder takes 256-D inputs directly.

## Fixed experiment

Configuration: `configs/text_realizer_talent_motivation_256.yaml`.

- Frozen encoder: `google/embeddinggemma-300m`, revision
  `57c266a740f537b4dc058e1b0cda161fd15afa75`, the existing classification protocol.
- Decoder base: `google/gemma-3-270m`, revision
  `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`, explicit eager attention.
- Prefix: `256 → 768 → 16 × 640`, GELU, fixed input scaling `sqrt(256)`.
  This keeps the historical compact projector's hidden width and decoder
  architecture. Only the projector input dimension changes.
- Fresh projector and LoRA initialization from the pretrained base; no old
  factual or oracle checkpoint is used as a warm start.
- LoRA rank 16, alpha 32, dropout 0.05, all attention and MLP projections.
  Base weights, token embeddings and output head remain frozen.
- Seed 42; AdamW with projector LR `1e-3`, LoRA LR `2e-4`, weight decay `0.01`
  excluding biases, gradient clipping 1.0, 5% warmup and cosine decay.
- Effective batch 32, FP32, gradient checkpointing; microbatch 1 or 2 chosen
  by a memory preflight requiring 15% headroom.
- Up to 30 epochs, development NLL every epoch, patience 5 with `1e-4`
  minimum improvement for patience. Retain the absolute lowest development
  token NLL checkpoint, independently of the patience threshold.
- Complete CV text with explicit BOS/EOS. Only CV tokens and EOS receive
  token-normalized cross-entropy supervision. No chunking, truncation, labels,
  structured-state supervision or synthetic targets.

The immutable package partition supplies **2,800 pool IDs for fitting**, **200
dev IDs for selection/diagnostics**, and **1,000 evaluation IDs reserved for a
later, explicitly authorized evaluation**. All 2,800 factual texts are valid
unsupervised reconstruction targets, regardless of the package reader's
labelled subset or out-of-fold allocation. The decoder's seed need not equal
the package's reader/editor seeds (0–4).

The preparer hashes factual CSV, pair index, partition, latent artifact and
encoder metadata/manifest, then joins strictly by ID. It extracts only `ids`
and factual `z` from the paired tensor container. It does not access `z_prime`
or read a counterfactual CSV. The container is necessarily deserialized; this
is not a claim that its serialized oracle bytes were never read. Evaluation
rows are filtered out before accessing CV text or computing any length/metric.
Only fit/development text and projected vectors are exported to training.

There are no counterfactual training CVs in this package corpus:
`include_train_counterfactual_texts: false`. The available counterfactual CVs
belong to the evaluation partition. This workflow therefore implements only
the factual-only approach; a future oracle-supervised comparison needs
separately generated, disjoint training pairs. It must not repurpose the
evaluation pairs for training.

## Isolation and execution

New experiment code lives in `exp/text_realizer_package/`, reusing the existing
`src/text_realizer/` model, generation, checkpoint and numerical-update code.
Historical experiment scripts, root dependencies/lockfile and `latent_lab`
code are unchanged. Use the existing isolated environment:

```bash
exp/text_realizer/environment/.venv/bin/python -m exp.text_realizer_package prepare
bash scripts/run_text_realizer_package.sh package-256-UNIQUE_RUN_ID 2
```

The launcher creates a detached tmux session and selects physical GPU 2 in
this example. It runs CPU tests and checks cached files first, then waits for
that GPU to become idle. It never terminates other GPU processes. Once leased,
it executes preflight, training and final development diagnostics in separate
processes. A failed prerequisite stops the chain; it does not silently relax
the recipe or switch to smaller targets. `prepare` alone never starts training.

Preflight includes native Gemma embedding/cache checks, longest-target memory
measurements, 20 smoke updates, a two-example overfit check (at least 0.95
character similarity for both examples, within 500 updates), nonzero trainable
gradients and frozen-base invariance, deterministic interrupted-update
continuation, fresh-process checkpoint reload, and 100 optimizer-update timing
measurements at the actual effective batch. It also verifies that the frozen
encoder reproduces the cached 256-D factual vectors and that decoder plus
encoder fit together during evaluation. Its diagnostic weights are never used
to initialize training.

The literal target budget uses only fit/dev: at least 1,024 tokens, or their
maximum target length rounded up to 128. Initial preparation found target
lengths **332–486 tokens**, median **405**, p95 **441**, with **zero tokenizer
round-trip discrepancies**. The resulting generation limit is **1,024**.
These are data/preparation checks, not evidence that GPU preflight or training
has completed.

Outputs are separate:

```text
data/latents/talent_motivation/embeddinggemma_256/text-realizer-prepared/<hash>/
models/talent_motivation/embeddinggemma_256/text-realizer/<run-id>/
reports/talent_motivation/embeddinggemma_256/text-realizer/<run-id>/
```

The model tree retains `last.pt` (optimizer, scheduler, RNG, sampler position,
patience/history and the committed best) and `best.pt` (inference weights).
Immutable plans bind source files, configuration, package versions, prepared
data and preflight. Changes require a new run ID. To continue after interruption:

```bash
bash scripts/run_text_realizer_package.sh EXISTING_RUN_ID 2 --resume
```

The report tree contains `launcher.log`, `status.json`, `preflight/status.json`,
`preflight/ready.json`, `eta.json`, `history.json` and final diagnostic reports.
ETA appears during the benchmark and excludes queue time. It includes the
30-epoch budget and development diagnostics with an explicit timing margin;
early stopping can finish sooner. No fixed completion time is credible while
the selected GPU is occupied.

Generated artifacts are locally excluded through `.git/info/exclude` on this
machine. The shared `.gitignore` is unchanged. No automatic staging, commit,
upload or weight deletion occurs. Local ignores are not backups.

## Evaluation and inference

At the best checkpoint, evaluate all 200 dev CVs with correct, shuffled,
mean-fit, zero and nearest-fit conditioning. Reuse lexical metrics, independent
conservative country/degree/university-tier/project-count extraction, fit-only
retrieval, memorization checks, output uniqueness, template/persona groups,
and cycle metrics using the exact 256-D encoder protocol. Motivation and
behavioral fidelity need human review; the four-field macro does not measure
them. Fixed-ID reference/reconstruction examples include that review context.

Report the existing operational conditioning thresholds (NLL gap at least
0.05 nats/token, character-similarity gap at least 0.02) and a paired bootstrap
comparison to fit-only retrieval. A dev character-similarity gain of 0.02 with
a positive lower 95% bound is a diagnostic, **not a held-out success claim**:
the same dev partition selected the checkpoint. No audit or official-test
score is produced automatically. Failure remains visible in the report and
manifest and does not trigger further architecture searches.

After development evaluation and reload verification, export
`inference/model.pt` and `inference/manifest.json`. This portable bundle contains
projector/adapter weights, references to pinned model files, generation
settings and provenance/checksums; no training corpus or optimizer state.
It still needs the cached base model/tokenizer.

```python
from src.text_realizer import load_text_realizer

decoder = load_text_realizer(
    "models/talent_motivation/embeddinggemma_256/text-realizer/RUN_ID/inference/model.pt",
    device="cuda:0",
)
texts = decoder.decode(z_prime)  # finite [batch, 256]; no source CV or labels needed
```

For reproducible ID-aligned file inference, use
`python -m exp.text_realizer_package generate --checkpoint ... --latents ...
--output ... --device cuda:0`. The input must contain exactly `ids`, `z`,
`encoder` (matching checkpoint metadata), and nonempty `provenance` identifying
the editor/checkpoint, intervention, sample seed and source artifact. This
command does not compute `Z'` or open canonical paired artifacts. It does not
normalize edited vectors. Matching shape/protocol enables inference but does
not establish factual faithfulness on off-distribution edited vectors.

The eventual five-example illustration and any official evaluation are
separate tasks after training. Existing package reader/flow caches need no
retraining for this decoder.
