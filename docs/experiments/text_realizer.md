# EmbeddingGemma CV reconstruction workflow

This experiment trains `D_text(z)` to reconstruct complete CVs from the existing
768-dimensional EmbeddingGemma vectors. It is separate from semantic decoder
`g`, which predicts structured attributes. EmbeddingGemma, canonical vectors,
LangVAE baselines, and `h_Z` remain unchanged.

The decoder is pretrained Gemma 3 270M plus a learned 16-token prefix, optionally
with LoRA. It receives only `z` at inference. Exact recovery is an empirical goal,
not a guarantee: embeddings can omit distinctions in the original CV. Fluent
prose and high embedding-cycle similarity alone do not establish faithful
reconstruction or counterfactual fairness.

## Implementation boundaries

Reusable model, generation, numerical training, checkpoint, and lexical metric
functions live in `src/text_realizer`. CV data handling, protocol checks,
evaluation, and campaign orchestration live in `exp/text_realizer`. Settings are
in `configs/text_realizer_embeddinggemma_768.yaml`; dependencies have their own
`exp/text_realizer/environment/pyproject.toml` and `uv.lock`.

These are additive modules. Do not replace another decoder checkpoint or import
the text realizer into the existing pipeline. No root environment or existing
script needs modification. The [original handoff](embeddinggemma_text_inversion_handoff.md)
explains the scientific motivation. Full-model tuning, reranking, and iterative
correction are deliberately excluded from this first campaign.

## Separate Gemma 3 1B pilot

The pilot completed on 2026-10-16. Its [results and interpretation](text_realizer_gemma1b_results.md)
record the negative reconstruction result, conditioning diagnostics, numerical
caveats, and preserved checkpoints. The 270M decoder remains the reference.
The 1B-specific code, configuration, launcher, and tests were restored to the
active tree on 2026-10-21 for Git transfer. Checkpoints, prepared caches, and
reports remain in the [local decoder archive](../../.archive/README.md). The
commands in this section reproduce a historical, negative-result pilot; they
require its pinned 270M development bundle and tune report to be restored to
their original paths. Neither decoder's weights are included in Git.
Restoring the code changes the source identity, so the completed 2026-10-15
pilot is a historical result, not a run to resume under the new source tree.
Use a new run ID for any reproduction or follow-up.

`configs/text_realizer_embeddinggemma_768_gemma1b.yaml` defines an audit-informed
follow-up, **not** a continuation or replacement of the completed 270M campaign.
It pins `google/gemma-3-1b-pt` at
`fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`, trains only the compact projector and
LoRA adapters with seed 42, and preserves the encoder, canonical vectors, `g`,
and `h_Z`. Training settings and fit/tune IDs match the selected 270M baseline.

One recorded numerical-backend difference is necessary: the 1B pilot uses
Transformers' `eager` attention. On this RTX 2080 Ti / PyTorch 2.6 FP32 setup,
default SDPA produced large cached-versus-full logit discrepancies around the
512-token local-attention boundary. Both eager attention and math-only SDPA
restored agreement to approximately `4e-5` in the diagnostic. This changes the
attention implementation, not the model architecture or objective. The backend
is explicit in the pilot config and portable checkpoints; old 270M checkpoints
continue to load with their original SDPA default. Preflight repeats the parity
check and measures memory/runtime with the selected eager implementation.

**2026-10-15 numerical warning:** the unchanged, selected 270M checkpoint also
failed this longer cached/full parity check with SDPA and passed with eager
attention. A successful reload alone does not establish numerical correctness.
Its historical scores remain preserved, but any 1B comparison against them is
not a clean model-size ablation. The comparison report explicitly records this
caveat. Follow-up checks on 20 actual fit CVs found effectively identical losses
with SDPA/eager and microbatch 1/2; two complete tune reconstructions were also
identical. Fit/tune targets have at most 486 tokens (503 including prefix/BOS),
below the 512-token boundary. Thus the longer-forward discrepancy does **not**
by itself show that the historical reconstruction scores were wrong. Evidence
is preserved in
`.archive/artifacts/reports/talent/embeddinggemma_768/text-realizer/1b-pilot-compatibility-20261015/270m-attention-real-cvs.json`.
The corrected launch `cv-text-1b-pilot-20261015-1026` initially ran preflight only.
After explicit authorization, its single training run started on 2026-10-15 at
approximately 15:02 Berlin time and completed with tune evaluation and reload
verification on 2026-10-16 at 05:01. Early stopping ended training after 10 epochs;
epoch 5 supplied the selected checkpoint. It failed the predeclared improvement
criteria. Audit, oracle evaluation, and automatic replication remained closed.

The pilot copies only hash-verified development artifacts to a separate prepared
directory; audit tensors are deliberately absent. Its preflight adds real-model
embedding/cache parity, frozen-weight and gradient checks, and memory checks with
both the decoder and cycle encoder loaded. FP32 microbatch 1 retains effective
batch 32 and requires at least 15% GPU-memory headroom.

```bash
tr_python=exp/text_realizer/environment/.venv/bin/python
tr_config=configs/text_realizer_embeddinggemma_768_gemma1b.yaml
# Requires personally accepted model terms; never accepts them automatically.
"$tr_python" -m exp.text_realizer.setup --config "$tr_config" --download-models

# Optional real-1B numerical check after caching the pinned model and selecting a GPU.
TEXT_REALIZER_1B_GPU_TESTS=1 "$tr_python" -m pytest -q -s \
  tests/text_realizer/test_integration_1b.py

# Detached preflight only; one explicitly selected physical GPU.
bash scripts/run_text_realizer_pilot.sh cv-text-1b-pilot-v1 0

# Only after resolving the numerical-baseline concern and explicitly deciding to train:
# resume the same immutable pilot and authorize its single training run.
bash scripts/run_text_realizer_pilot.sh cv-text-1b-pilot-v1 0 --resume --train-after-preflight
# For a fresh run, --train-after-preflight can also be supplied on the first launch.
```

The pilot stops after best-checkpoint reload verification and **tune-only**
comparison. It never schedules other seeds, audit, oracle evaluation, or uploads.
The ordinary `campaign`, `freeze`, `audit`, `oracle-evaluate`, and canonical
`prepare` commands refuse this pilot configuration. The original 270M config and
launcher retain their original behavior.

Before training, its immutable plan binds baseline report hashes and operational
go/no-go thresholds: character similarity gain at least 0.02; mean factual
accuracy gain at least 0.02 across country, degree, university tier, and project
count; no field regression larger than 0.02; and the original shuffled-conditioning
thresholds. Region is excluded from the factual average to avoid double-counting
country evidence. Bootstrap intervals on tune are descriptive, not confirmatory.
These checks do not establish behavioral-episode fidelity or counterfactual fairness.

Each run writes `status.json`, phase logs, measured `eta.json` after preflight,
`pilot_result.json`, and the first 20 fixed-order `tune_examples.json`. Its child
run `<RUN_ID>.lora-compact.seed-42` retains `last.pt`, portable trainable-only
`best.pt`, and full tune diagnostics. Existing inference APIs load `best.pt`.
Changing the config, sources, or inputs requires a new run ID. Replication and a
fresh confirmatory evaluation require a separate decision; the previously viewed
audit is not an untouched holdout for this follow-up.

## Prepare the environment and data

Run commands from the repository root. Install `uv` if needed. Personally accept
both pinned models' licenses on Hugging Face and authenticate normally before
downloading. Never put tokens in YAML, command arguments, logs, or Git.

```bash
# Environment only; no downloads, training, or encoding.
bash scripts/setup_text_realizer.sh

# Explicitly cache the models after access has been granted.
bash scripts/setup_text_realizer.sh --download-models

tr_python=exp/text_realizer/environment/.venv/bin/python
"$tr_python" -m exp.text_realizer prepare
```

The isolated environment uses Python 3.12, PyTorch 2.6.0 with CUDA 12.4,
Transformers 4.57.6, Sentence Transformers 5.1.2, and PEFT 0.17.1. Its persistent
Python installation lives in the ignored project cache, not `/tmp`. Model
snapshots are shared between runs. Only explicit setup downloads access the Hub;
training and inference load the pinned local snapshots.

Preparation verifies source hashes and encoder provenance, joins by ID, and
writes `development.pt` and a separately gated `audit.pt`. It does not call the
existing paired-artifact loader. Deserializing the canonical container loads its
storage, but preparation accesses only factual `ids` and `z`, never oracle tensor
values. Actual training accepts only the derived development bundle.

| Partition | Rows | Permitted use |
|---|---:|---|
| Fit | 3,200 | Updates and retrieval reference |
| Tune | 400 | Early stopping and model selection |
| Audit | 400 | Evaluation after choices are frozen |
| Official test | 1,000 | Explicit later final evaluation |

The existing fit/validation lists are reused. Validation is permuted with seed
`20261007` and split equally into tune/audit. A missing or changed original split
plan is an error, not permission to generate a replacement split. Literal ID lists
and hashes are recorded. Template/persona metadata are reporting-only fields.

Only fit/tune texts determine tokenizer statistics and the generation budget.
Targets are complete CVs plus EOS. Prefix, BOS, and padding are excluded from the
loss. No target is silently truncated. Generation stops at EOS or its frozen
budget, at least 1,024 tokens. Longer audit/test references are still scored in
full; their outcomes cannot change the budget.

## Validate and launch

```bash
# Offline tests and relevant existing regression tests.
"$tr_python" -m pytest -q tests/text_realizer \
  tests/test_embeddinggemma_encoder.py tests/test_langvae_ft.py \
  tests/test_config_paths.py tests/test_config_propagation.py

# Optional bounded GPU checks, not a trained baseline.
TEXT_REALIZER_GPU_TESTS=1 "$tr_python" -m pytest -q -s \
  tests/text_realizer/test_integration.py

# Required readiness check; performs diagnostic training only.
"$tr_python" -m exp.text_realizer preflight --device cuda:0

# Explicit detached campaign on these physical GPUs.
bash scripts/run_text_realizer.sh cv-text-v1 0 1 2 3
```

Preflight runs offline tests, a 20-update loss-decrease check, bounded two-example
overfitting, resume/reload checks, canonical encoder parity, and a 100-update
benchmark. It tests microbatches 1 and 2 and requires 15% memory headroom.
Diagnostic weights cannot be promoted. Failed checks leave readiness false.

Precision is FP32, including encoder-cycle evaluation. One worker owns one GPU;
the cards are not pooled. Advisory leases and a busy-device check protect other
work. No process is killed and no driver setting is changed.

Measured timings are recorded in the prepared directory's `preflight.json`.
Validation and final free-running reconstruction controls add time beyond the
training benchmark. The launcher prints its tmux session, log, and status paths.
Status files contain heartbeat and progress information.

## Training and selection

Seed 42 trains prefix-only and prefix+LoRA models, each with a 768-wide or
`4 * hidden_size` projector. The bridge uses reversible scaling `sqrt(768) * z`
instead of the handoff's input LayerNorm, preserving the vector's mean before the
learned mapping. It introduces no additional fixed down-projection.

LoRA uses rank 16, alpha 32, dropout 0.05, and the seven attention/MLP projection
types. Token embeddings and the output head remain frozen. Defaults are projector
LR `1e-3`, LoRA LR `2e-4`, effective batch 32, five-percent warmup, cosine decay,
weight decay 0.01, clipping 1.0, at most 30 epochs, and early stopping after five
non-improving evaluations. The loss normalizes actual target-token sums over the
effective batch, not an unweighted mean of microbatch means.

Minimum tune NLL selects each run's checkpoint. Eligibility requires a gain over
shuffled conditioning of at least 0.05 nats/token and 0.02 character similarity.
Eligible models are ranked by character similarity, cycle cosine, and stable
variant ID. Mean-fit, zero, and nearest-fit-vector controls are also reported.
Retrieval copies only fit CVs and is labelled a non-generative reference.

The chosen configuration is repeated with seeds 43 and 44. Deployment remains
seed 42; repetitions do not change the configuration choice. If no model
qualifies, the campaign records `no_qualified_model` and stops without escalation.
Otherwise it freezes all choices, verifies exports in fresh processes, and runs
the factual audit. It ends with the oracle gate sealed.

To resume, repeat the launcher command after its old tmux session has exited.
Individual runs support the same configuration and explicit resume:

```bash
"$tr_python" -m exp.text_realizer train \
  --run-id cv-text-v1.lora-compact.seed-42 --variant lora-compact \
  --seed 42 --device cuda:0 --resume
```

Resume restores optimizer/scheduler, RNG, sample order/offset, and early stopping.
The rolling checkpoint also contains its committed best model, allowing recovery
between file replacements. Updates since the last checkpoint are replayed.
Changed code, configuration, data, or environment requires a new run ID and
preflight. Do not retune against an audit already opened.

## Fidelity reports

Reports include raw/whitespace-normalized exact match, character Levenshtein
similarity, aligned token accuracy, token error rate, word-level ROUGE-L F1,
chrF, lengths, and termination statistics. Token alignment uses unit-cost
Levenshtein edits; insertions affect token error rate separately. Premature EOS
means EOS before half the reference token length.

Cycle metrics use the exact frozen encoder and classification prompt.
Empty/overlong generations remain explicit failures, never truncated or silently
removed. Invalid cycles receive cosine -1 for selection; valid-only L2 and rank
summaries report coverage.

Independent text rules compare country/region, degree, university tier, and
project count. Missing predictions are failures on scorable references;
reference coverage is explicit. These rules do not infer hidden talent or use
`g` as an independent judge. Review the fixed first 20 audit IDs for occupation,
behavioral episodes, invented details, and readability.

Memorization checks report nearest fit texts, exact copies, output uniqueness,
and longest shared word spans with the nearest lexical fit. Template/persona and
seen/unseen template–persona combinations are reporting groups, never model inputs.

The primary improvement criterion is a deployment-model mean character-similarity
gain over retrieval of at least 0.02, with a positive lower bound on a paired 95%
bootstrap interval. All three seeds are reported. Failure is a negative result,
not permission to tune on audit/test.

## Load and transfer the decoder

Outputs stay ignored by Git:

- Prepared bundles: `data/latents/talent/embeddinggemma_768/text-realizer-prepared/`.
- Weights: `models/talent/embeddinggemma_768/text-realizer/<run-id>/`.
- Reports: `reports/talent/embeddinggemma_768/text-realizer/<run-id>/`.

Frozen inference files are `selected/seed-42.pt`, `seed-43.pt`, and `seed-44.pt`.
They contain projector/adapter weights, model/tokenizer references, generation
settings, hashes, provenance, and synthetic reload probes. They exclude the base
model, training corpus, and optimizer. Transfer selected weights and the frozen
manifest/report bundle privately. Uploads and Git staging are never automatic.

```python
from src.text_realizer import load_text_realizer

decoder = load_text_realizer(checkpoint, device="cuda:0")
texts = decoder.decode(z)  # finite [batch, 768], including edited vectors
```

Cache the pinned base model on the destination first. Without `cache_dir`, the
loader tries the project cache and then the standard Hugging Face cache. It does
not silently normalize edited vectors.

CLI input tensor files contain exactly `ids`, `z`, `encoder`, and `provenance`.
`encoder` must equal the settings recorded in the checkpoint, and `z` is an
ID-aligned `[N, 768]` tensor. Paired canonical artifacts are rejected.

```bash
"$tr_python" -m exp.text_realizer generate \
  --checkpoint models/talent/embeddinggemma_768/text-realizer/cv-text-v1-20261012-1719/selected/seed-42.pt \
  --latents edited_vectors.pt --device cuda:0 --output generated_cvs.json
```

## Final evaluation after freezing

The separate command requires a frozen manifest, verified reload, completed
factual audit, unchanged hashes, and explicit opt-in:

```bash
"$tr_python" -m exp.text_realizer oracle-evaluate \
  --run-id cv-text-v1 --device cuda:0 --allow-oracle-evaluation
```

Optionally add `--edited-latents edited_test_vectors.pt`, using the inference
format above with exactly the official-test IDs. Factual, oracle-counterfactual,
and learned-intervention results are separate. Identity/changed pairs are
stratified; identical factual/oracle vectors must yield identical token sequences.
This command never reads `oracle_train/z_prime.pt`, updates the decoder, or
invokes `h_Z` training.

These gates enforce a workflow, not an operating-system security boundary. Do
not bypass them by manually feeding audit/test vectors to the inference API
during development. Final test outcomes cannot influence choices presented as
part of this frozen experiment.
