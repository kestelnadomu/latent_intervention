# Factual-only 270M draft-correction pilot

This is an exploratory follow-up to the [270M/1B comparison](text_realizer_gemma1b_results.md),
not a replacement for the frozen 270M baseline. The 1B pilot improved teacher-forced
NLL but worsened generated text and several factual fields. Here the hypothesis is
that explicitly revising a draft using its embedding discrepancy may help the
270M model express information already present in the target vector. Success is
empirical; neither semantic similarity nor fluent text establishes exact inversion.

**Completed result (2026-10-17):** this pilot did not meet any predeclared
continuation gate. Mean tune character similarity changed by only +0.00372,
while four-field factual accuracy declined. The corrector was not promoted and
no counterfactual training began. See the [full results and decision record](text_corrector_results.md)
for the comparison, limitations, conditioning controls, and artifact hashes.

## Scope and isolation

The only external inference input remains a finite `[batch, 768]` tensor:

```text
target Z → existing frozen 270M inverter → draft CV
                                            ↓
                                   frozen EmbeddingGemma
                                            ↓
target Z + draft embedding + draft text → learned 270M corrector → corrected CV
```

The draft is generated internally; the original CV is **not** an inference input.
Original factual CVs are training targets only. The same API accepts an edited
vector without silently unit-normalizing it, but performance on edited vectors
has not been established by this factual pilot.

- New reusable code: `src/text_corrector/`.
- New experiment code: `exp/text_corrector/`.
- New configuration: `configs/text_corrector_embeddinggemma_768.yaml`.
- New tests: `tests/text_corrector/`.
- New launcher: `scripts/run_text_corrector.sh`.
- Reuses the existing isolated `exp/text_realizer/environment/.venv`; no root
  dependency, environment, encoder, `g`, `h_Z`, or baseline changes.
- Generated artifacts use new `text-corrector/` subdirectories under the existing
  EmbeddingGemma data/model/report trees. Existing ignore rules cover them.
- Old text-realizer source-hash enumeration does not include these sibling
  directories. New run identities include both the reused code and correction code.

Nothing is automatically staged, committed, uploaded, promoted, or removed.

Initial implementation validation (2026-10-16): all 24 correction tests passed;
the combined correction/decoder/configuration/EmbeddingGemma/LangVAE-FT checks
passed 129 tests with four opt-in integration tests skipped. Separately, 60 legacy
LangVAE/encoder/configuration checks passed using the existing legacy dependencies
with the persistent interpreter; no environment was modified. This includes an
interrupted full training-loop test whose resumed checkpoint matches the
uninterrupted run exactly.

The first preflight-gated pilot ran on GPU 0 as
`cv-correction-pilot-20261016-1711`, with `--train-after-preflight`. Preflight,
training, reload verification, and the factual tune comparison all completed;
the [results record](text_corrector_results.md) reports the negative outcome.
The other GPUs were left free. This run stopped after the factual tune comparison.

## What is trained

The original draft checkpoint is fixed:

```text
models/talent/embeddinggemma_768/text-realizer/
  cv-text-v1-20261012-1719/selected/seed-42.pt
SHA256: 0d459c18343b1eb62573c7c0c19647b3eaf9b3b42e90a0412639220fd31adff7
```

The corrector is a separate copy of `google/gemma-3-270m`, revision
`9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`. Its LoRA parameters and target-vector
projector are warm-started from that checkpoint. Frozen backbone/token embedding/
output-head parameters stay frozen. The original checkpoint is never updated.

Conditioning is:

```text
sqrt(768) × concat(Z, f(draft), Z − f(draft))
  → Linear(2304, 768) → GELU → Linear(768, 16 × 640)
  → 16 prefix embeddings
```

The target-vector block initially matches the original projector exactly; the
two new input blocks start at zero. They are trainable. The residual is redundant
with the two vectors and adds no information by itself. There is no input LayerNorm
or normalization of edited target vectors.

The causal context is `[prefix][BOS]Draft CV:\n{draft}\nCorrected CV:\n{original CV}[EOS]`.
Only original-CV tokens and EOS are supervised; prefix, BOS, draft, prompt, and
padding labels are masked. Targets are complete, unchunked original CVs, never
synthetic summaries or silently truncated text. The LM head is evaluated only for
the output region to save memory; the entire draft remains in the transformer context.

This pilot uses **one** correction pass, deterministic greedy generation, and the
existing 1,024-token generation budget, checked against all factual fit/tune target
lengths and the complete context limit. There is no reranking or decoding search.

The corrector explicitly uses eager attention. This avoids the long-context
cached/full mismatch observed with default SDPA on this machine. The initial draft
decoder preserves its historical backend and weights, making zero-correction and
one-correction comparisons use exactly the same draft. The corrector's cache
parity is tested beyond the sliding-window boundary.

## Scientific boundaries and limitations

Training can read only the already prepared factual development bundle, after
verifying its manifest, content hash, ID alignment, split sizes, finite unit vectors,
encoder provenance, and draft-checkpoint hash. It never calls the original canonical
preparer or paired-artifact loader. It never reads audit contents, official-test
texts/vectors, counterfactual targets, or `oracle_train/z_prime.pt`.

- Fit: all existing 3,200 factual IDs, used for updates and retrieval references.
- Checkpoint selection: a fixed random 80 of the existing 400 tune IDs, seed
  `20261016`, saved literally in the immutable run plan.
- Final development comparison: all 400 existing tune IDs.
- No new audit or official-test evaluation, and no counterfactual-exposed branch.

Fit drafts come from the already trained factual decoder, which saw those fit CVs.
They are **not out-of-fold**. This is a deliberate inexpensive pilot, not an unbiased
cross-fitted training design. Train-draft errors can differ from held-out-draft
errors. A positive result warrants a stronger follow-up; it does not settle this
distribution mismatch. No teacher-forced reference tokens are inserted into drafts.

The original audit and repeated tune comparisons have already informed this follow-up.
The 400-CV final comparison is therefore exploratory development evidence, not a fresh
confirmatory test. Do not relabel its descriptive bootstrap interval as an untouched
held-out result.

Only vectors, generated drafts, and original factual text enter reconstruction
updates. Country/degree/tier/project-count extraction is used for **selection and
evaluation**, not auxiliary training labels. It is independent of `g(f(text))`.
The checks are conservative, incomplete, and not a substitute for reviewing occupation,
behavioral episodes, or invented details.

## Training and selection

One seed, 42; FP32; microbatch one; effective batch 32; gradient checkpointing.
AdamW uses projector LR `3e-4`, LoRA LR `5e-5`, weight decay `0.01` except biases/
one-dimensional parameters, 5% warmup, cosine decay, and gradient clipping at 1.
LoRA rank 16, alpha 32, dropout 0.05 and q/k/v/o/gate/up/down targets are inherited.
The smaller rates are explicit warm-start settings, not a claim of an independently
validated optimum. No auxiliary loss, label smoothing, corruption training, or
encoder update is introduced.

Maximum 20 epochs, patience four. Each epoch generates the fixed 80 tune CVs.
Selection score is an equal-weight average of normalized character similarity and
macro accuracy over country, degree, university tier, and project count. NLL is
recorded and breaks selection-score ties. Missing generated facts count as failures
when the reference is scorable. This deliberately changes selection from NLL-only;
improvement cannot be attributed solely to architecture without further ablations.

The final report compares one correction against zero corrections on identical IDs.
Predeclared continuation criteria are all of:

- Mean character-similarity gain at least `0.02`.
- Positive lower bound of the descriptive paired-bootstrap 95% interval.
- Macro four-field factual gain at least `0.02`.
- No individual field regression greater than `0.02`.

These justify considering a follow-up, not automatic deployment or counterfactual
training. Results are retained even when the gates fail.

The report also includes token alignment, exact matches, ROUGE-L, chrF, termination,
lengths, frozen-encoder cycle metrics, fit-only retrieval, nearest-fit lexical copying,
output uniqueness, grouped template/persona results, and the first 20 tune examples
with a human-review rubric. It separately shuffles only the corrector's target vector
while holding its draft text/embedding fixed. This diagnoses **additional** target-vector
use; it is not a test of whole-pipeline Z dependence, since the initial decoder already
uses Z.

## Running and recovery

Use the existing persistent Python 3.12 environment, with its existing pinned Torch
2.6.0+cu124, Transformers 4.57.6, Sentence Transformers 5.1.2, and PEFT 0.17.1.
All production model loads are local-only; model access/licenses were handled by the
original workflow. This pilot does not accept licenses or download weights.

```bash
exp/text_realizer/environment/.venv/bin/python -m pytest -q tests/text_corrector

# Preflight only: no full draft preparation or full training.
bash scripts/run_text_corrector.sh correction-RUN_ID 0

# Explicitly continue through preparation, training, reload verification, and tune comparison.
bash scripts/run_text_corrector.sh correction-NEW_RUN_ID 0 --train-after-preflight

# Resume that same immutable run after interruption.
bash scripts/run_text_corrector.sh correction-NEW_RUN_ID 0 --train-after-preflight --resume
```

Use unique run IDs and an explicitly free GPU. The launcher detaches into tmux and
uses the same per-physical-GPU advisory leases as the original decoder. It never
kills other GPU jobs. No distributed training is used.

Preflight checks native embedding scaling, cached/full long-context parity,
deterministic generation, 20 smoke updates, frozen-parameter invariance/nonzero
prefix and LoRA gradients, exact interrupted-update continuation, two-example
greedy overfitting, checkpoint reload, and GPU memory with at least 15% reserved-memory
headroom. It benchmarks ten **effective-batch** updates (320 microsteps) and generation.
The numerical overfit check has its own explicitly faster learning rates. Synthetic
repeated-draft memory stress inputs are preflight-only; no training target is changed.
All preflight-trained weights are discarded before real training starts.

Check these under `reports/talent/embeddinggemma_768/text-corrector/RUN_ID/`:

- `plan.json`, `source/`: immutable configuration, literal IDs, code/environment hashes.
- `launcher.log`, `status.json`: logs, current phase, progress, heartbeat, errors.
- `preflight.json`, `preflight-overfit.json`: numerical and memory gates.
- `eta.json`: measured runtime scenarios after preflight, including draft generation.
- `history.json`, `epoch-NN.json`, `training.json`: selection and stopping history.
- `verify.json`, `result.json`: fresh reload verification and full tune comparison.

Draft generation saves each completed factual ID atomically, then writes a hash-bound
development cache. `models/.../text-corrector/RUN_ID/last.pt` restores optimizer,
scheduler, Python/Torch/CUDA RNG, epoch order/position, and stopping history at an
optimizer boundary. `best.pt` is the selected inference bundle. Configuration,
inputs, environment, or relevant source changes require a new run ID, never a silent
continuation of the old experiment. Final evaluation can be recomputed after interruption.

## Inference and artifacts

After successful training and reload verification:

```python
from src.text_corrector import load_text_corrector

decoder = load_text_corrector(
    "models/talent/embeddinggemma_768/text-corrector/RUN_ID/best.pt",
    device="cuda:0",
    cache_dir=".cache/huggingface/hub",
)
texts = decoder.decode(z)  # finite [batch, 768]; no originals or labels
```

The inference bundle contains correction weights, the immutable initial decoder's
trainable weights, pinned model references, encoder protocol, generation settings,
provenance, and checksums. It excludes optimizer state, the CV corpus, and duplicate
copies of pretrained backbone weights. The pinned decoder/encoder snapshots must
be cached on another machine before loading. Empty/overlong drafts fail explicitly;
there is no hidden reference-text lookup or truncation fallback.

## Counterfactual-exposed comparison remains a separate decision

This factual correction did **not** produce the predeclared worthwhile gain,
so no automatic counterfactual-trained twin of this corrector is warranted.
A separately authorized decoder comparison could still train with paired
`(Z', X')` in addition to factual `(Z, X)`, with all worlds of each unit kept
in one split. The encoder and `h_Z` can remain fixed, but this is privileged
decoder supervision and must be labeled separately from the factual-only method.
`Z'` without its matching `X'` is not a reconstruction training pair. No such
branch, data access, loss, or automatic follow-up is enabled here. See the
[results record](text_corrector_results.md) for the conditions under which this
would be a meaningful comparison.
