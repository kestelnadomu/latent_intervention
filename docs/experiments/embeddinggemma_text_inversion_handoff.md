# GPU handoff: factual-only inversion of EmbeddingGemma CV embeddings

Status: implementation and training specification  
Prepared: 2026-10-12  
Immediate owner: agent operating on the GPU cluster

## 1. Immediate mission

Implement, train, and validate a new text realizer

\[
D_\phi: z \in \mathbb R^{768} \longrightarrow \widehat{x}_{\text{text}}
\]

that reconstructs a complete CV from its frozen 768-dimensional EmbeddingGemma
embedding. Train it **only** on factual pairs
\((z_i,x_{\text{text},i})\). The ideal is exact recovery of the original CV; in
practice, report exact, lexical, semantic, latent-cycle, and factual-attribute
fidelity because the embedding is a lossy many-to-one representation.

This experiment is intentionally not paired counterfactual editing. Do not give
the decoder the source CV, an intervention label, a factual/counterfactual pair,
or structured attributes. At inference it receives only one embedding.

The first deliverable is a frozen, reload-verified factual-only decoder plus a
sealed factual audit. Only after the architecture, checkpoint, generation method,
and all hyperparameters are frozen may a separate process open the oracle data
and evaluate

\[
D_\phi(z'_{\mathrm{oracle}}) \quad\text{against}\quad x'_{\text{oracle}}.
\]

Do not run that oracle evaluation during development.

## 2. Why this is part of the project

The repository studies counterfactual interventions in text latent space:

```text
factual CV x_text --frozen encoder f--> factual latent z
                                              |
                                              | learned latent intervention h_Z
                                              v
                                    predicted counterfactual z_hat'
                                              |
                                              | new text realizer D_phi
                                              v
                                    generated counterfactual CV x_hat'
```

The simulator also provides a true counterfactual CV and embedding
\((x'_{\text{text}},z')\). Those are an oracle ceiling for evaluating whether the
text-realization step works independently of errors in the learned intervention.
The eventual comparison is:

1. `D_phi(z)` on factual held-out embeddings: can the decoder invert the encoder?
2. `D_phi(z'_oracle)`: can the factual-only decoder transfer to a true
   counterfactual embedding on the same encoder manifold?
3. `D_phi(z_hat')`: can it realize the counterfactual latent predicted by the
   project's `h_Z` method?

If (1) fails, the problem is the text decoder. If (1) works but (2) fails, the
problem is counterfactual distribution shift or non-invertibility. If (2) works
but (3) fails, the predicted latent is likely inaccurate or off-manifold.

### Notation warning

In this document, `x_text` means CV text. Elsewhere in this repository, uppercase
`X` is the protected SCM variable (country-of-origin region). Keep those concepts
separate in code, reports, and paper text.

The new component should be named `text_realizer` or `D_text`, **not** `g`.
Existing `g` is the semantic decoder from a latent to the discrete structured
state `(X,T,D,U)` and is a different model.

## 3. Non-negotiable scientific boundary

During preparation, model fitting, hyperparameter selection, checkpoint
selection, and decoding-strategy selection:

- use factual CV text and factual `z` only;
- freeze the released EmbeddingGemma encoder and preserve every stored `z` value;
- never train on, validate on, or inspect counterfactual CV text `x_text'`;
- never train on, validate on, or inspect oracle `z'`;
- never open `data/latents/talent/embeddinggemma_768/oracle_train/z_prime.pt`;
- never select a model using the official 1,000 test IDs;
- never condition generation on template ID, persona ID, simulated state,
  intervention, factual source text, or any other sample metadata;
- do not fine-tune the EmbeddingGemma encoder;
- do not regenerate or replace the canonical embeddings.

“Oracle untouched” means oracle information is absent not only from gradient
updates but also from early stopping, architecture choice, decoding parameters,
candidate count, reranking policy, and qualitative debugging.

The oracle can be opened exactly once after a signed frozen manifest exists. If
the model is changed after viewing oracle results, label the later result as
test-informed rather than a clean held-out evaluation.

## 4. Existing inputs and verified state

The relevant existing files are:

| Purpose | Path |
|---|---|
| factual CVs | `data/text_talent/cv_factual.csv` |
| official split authority | `data/sim_talent/pair_index.csv` |
| canonical factual and official-test latent artifact | `data/latents/talent/embeddinggemma_768/z_pairs.pt` |
| latent provenance | `data/latents/talent/embeddinggemma_768/z_pairs.info.json` |
| frozen encoder implementation | `src/encoder/embeddinggemma.py` |
| latent validator | `src/pair_encoding.py::load_latent_artifact` |
| encoder protocol | `src/encoder/protocols.py` |

Current verified counts:

- 5,000 factual CVs and 5,000 unique IDs;
- official split: 4,000 train / 1,000 test;
- factual CV length: 301--432 whitespace-delimited words, median 350;
- factual latent tensor: `[5000, 768]`, `torch.float32`;
- official-test oracle latent tensor: `[1000, 768]`, `torch.float32`;
- official test: 256 identity and 744 changed counterfactual pairs;
- oracle training artifact: 4,000 rows, including 1,028 identity pairs; it is
  forbidden for this experiment.

Current hashes to verify after transfer:

```text
e2845bd8b16360ae9c92f51240af0942f92b26b94b1331cf0ffdb75253f67294  data/text_talent/cv_factual.csv
67616f395548dfffb7f063b554338b6b63731f5a77136134ffc3b2936b5d51ee  data/sim_talent/pair_index.csv
4722d665c2f7fc955925202eefeb49391f4fae51138beb718cd8ce0b38720a03  data/latents/talent/embeddinggemma_768/z_pairs.pt
```

The EmbeddingGemma artifact is pinned to:

```text
model: google/embeddinggemma-300m
revision: 57c266a740f537b4dc058e1b0cda161fd15afa75
dimension: 768
prompt: "task: classification | query: "
pooling: attention-mask mean, including prompt
projection: 768 -> 3072 -> 768, identity activations, no biases
normalization: L2 at native 768 dimensions
inference: float32, bidirectional attention, SDPA, no truncation
```

Use the full native 768D vector for this first inversion experiment. Smaller MRL
prefixes intentionally contain no more information and should be separate later
ablations, not development substitutes.

### Important artifact-isolation detail

`z_pairs.pt` physically contains both factual `z` and official-test `z_prime`.
The training program must therefore **not** consume that canonical file directly.
Implement a trusted `prepare` command that validates the canonical artifact and
then writes a factual-only derived bundle containing only:

```text
train factual IDs
train factual z[ID]             # [4000, 768]
train factual CV text[ID]
fit/tune/audit ID lists
input hashes and preparation metadata
```

The trusted preparation step may load the canonical payload solely to extract
its factual `z`; it must never print, summarize, score, copy, or expose `z_prime`.
All actual training and development commands must accept only the derived bundle.
After preparation, run training in a process whose input configuration has no
counterfactual paths.

## 5. Required split

Keep all official test IDs sealed. Reuse the exact split already established by
the semantic-decoder and preservation work:

| Partition | Count | Permitted use |
|---|---:|---|
| fit | 3,200 | gradient updates |
| tune | 400 | early stopping, architecture and decoding selection |
| audit | 400 | one final factual evaluation after freezing choices |
| official test | 1,000 | no development; later oracle evaluation only |

If the existing plans are present, reuse their exact ID lists:

```text
reports/talent/hz_benchmarks/hz-gemma-v1/plan.json
reports/talent/hz_preservation/hz-preservation-pilot-v3/plan.json
```

Expected SHA-256 values on the source system are:

```text
6698da61c789e3dad586d051e8bdad40157786e01c858212d89b95d5ee90b4be  hz-gemma-v1/plan.json
cb23cd8761eaac5e1db37c5ee8db74027d67e0abf8f9672e8e0ff73af4e8207c  hz-preservation-pilot-v3/plan.json
```

If those reports were not transferred, reproduce the split exactly:

1. Preserve canonical ID order from the latent artifact.
2. Remove official test IDs to obtain the 4,000 training IDs.
3. With `torch.Generator().manual_seed(42)`, apply `torch.randperm(4000)`.
4. The first 800 shuffled IDs are the old validation partition; the remaining
   3,200 are `fit`.
5. With `torch.Generator().manual_seed(20261007)`, permute those 800 validation
   IDs. The first 400 are `tune`; the remaining 400 are `audit`.
6. Save the literal lists and their canonical JSON SHA-256 in the derived bundle.

Assert that all four partitions are nonempty, unique, pairwise disjoint, and
their union equals all 5,000 IDs. The training process must assert that it has no
official-test row.

## 6. Decoder choice

### Recommended initialization

Use the pretrained, non-instruction-tuned `google/gemma-3-270m` causal language
model as the first decoder. Resolve its immutable Hugging Face commit on the GPU
cluster and pin that revision everywhere; never train from floating `main`.

This is a sensible family initialization because EmbeddingGemma began from an
approximately 300M-parameter Gemma 3 model through the T5Gemma adaptation recipe.
It is not, however, an existing inverse: EmbeddingGemma is bidirectional and
mean-pooled with learned projection and normalization, whereas the new decoder is
autoregressive. Do not attempt to attach the embedding directly as a normal token
or claim that the weights form a matched encoder-decoder pair.

Official references:

- <https://huggingface.co/google/gemma-3-270m>
- <https://developers.googleblog.com/en/gemma-explained-embeddinggemma-architecture-and-recipe/>
- embedding-inversion precedent: <https://aclanthology.org/2023.emnlp-main.765/>

Gemma weights require acceptance of Google's license and an authenticated
Hugging Face account. Never place a token in source, configuration, shell history,
logs, tmux commands, or reports.

### Required conditioning architecture

Use a learned latent prefix:

```text
z [B, 768]
   |
   v
LayerNorm -> MLP -> [B, K * d_model]
   |
   v
reshape to K virtual embeddings [B, K, d_model]
   |
   +-- concatenate -- [BOS + factual CV token embeddings]
   |
   v
Gemma causal language model -> factual CV tokens + EOS
```

Start with `K=16`. The MLP should have a nonlinear hidden layer and emit distinct
virtual-token vectors; derive `d_model` from the loaded Gemma configuration. A
reasonable initial projector is:

```text
LayerNorm(768)
Linear(768, 4 * d_model)
GELU
Linear(4 * d_model, K * d_model)
reshape(B, K, d_model)
```

The virtual tokens increase conditioning capacity but do not create information
not present in `z`.

For each training example:

1. tokenize raw factual CV text with Gemma's tokenizer;
2. form `[virtual prefix, BOS, CV tokens, EOS]`;
3. give Gemma an attention mask covering the prefix and non-padding target;
4. set labels for every prefix and padding position to `-100`;
5. compute teacher-forced next-token cross-entropy on CV and EOS tokens only.

Never silently truncate a CV. Run a tokenizer preflight over all factual texts,
record min/median/p95/max token counts, and set a sequence length that covers the
maximum plus prefix/BOS/EOS. Fail on overflow.

At generation time the model receives only `[latent prefix, BOS]` and generates
until EOS or the audited maximum. Implement and test cache/position handling for
`inputs_embeds`; do not assume `generate()` handles virtual input embeddings
correctly without a round-trip test. A small explicit cached autoregressive loop
is acceptable for greedy decoding. If using Transformers generation helpers,
test output slicing, attention masks, cache positions, EOS, and batch behavior.

## 7. Training ladder

Do not begin with a large uncontrolled search. Implement these variants in order,
using factual tune data only for every decision.

### A. Non-generative diagnostic baseline: nearest factual neighbor

For each tune embedding, retrieve the most cosine-similar **fit** embedding and
copy its CV. This is not the proposed decoder and must be labelled `retrieval`.
It quantifies duplicate/template structure and provides a lexical baseline.

### B. Prefix-only baseline

Freeze all Gemma parameters and train only the latent-prefix projector. This
tests whether the pretrained decoder can use the embedding through a small bridge.
Keep it as a paper ablation even if it is weak.

### C. Main model: prefix plus LoRA

Train the projector and LoRA adapters on Gemma attention and MLP linear layers.
Start with:

```text
prefix tokens K: 16
LoRA rank: 16
LoRA alpha: 32
LoRA dropout: 0.05
projector learning rate: 1e-3
LoRA learning rate: 2e-4
weight decay: 1e-2
warmup: 5% of optimizer updates
scheduler: cosine decay
gradient clipping: 1.0
effective batch size: 16 or 32 CVs via accumulation
maximum epochs: 30
early-stopping patience: 5 evaluations
seed: 42
precision: BF16 when the GPU supports it
```

These are starting settings, not immutable truths. Record any change before
viewing audit results. Evaluate at least once per epoch, save `last` and the true
best tune checkpoint atomically, and restore the best checkpoint at the end.

Thirty language-model epochs already expose the model to tens of millions of
target tokens; do not copy the 500-epoch MLP budget used elsewhere in the repo.
Let held-out factual reconstruction determine whether more training is useful.

### D. Conditional escalation: full 270M fine-tuning

Run full-model BF16 fine-tuning only if LoRA clearly underfits both fit and tune
sets and memory permits it. Use a much smaller model learning rate (start near
`2e-5`), gradient checkpointing, and the same sealed selection protocol. Do not
scale to a larger Gemma model merely because train reconstruction improves; require
an improvement on factual tune reconstruction and latent sensitivity.

An official pretrained Gemma 3 1B model is a secondary scale-up if the 270M model
has a demonstrated capacity ceiling. It is a new experimental variant and needs
its own pinned revision, run ID, manifest, and factual-only selection.

### E. Optional inversion refinement

After the single-pass decoder is stable, improve inversion without using oracle
text by generating a predeclared number of candidates, re-encoding each candidate
with the exact frozen EmbeddingGemma protocol, and choosing the candidate whose
embedding has highest cosine similarity to the input `z`.

Choose candidate count, beam/sampling settings, and reranking rule **only** on the
factual tune partition, then freeze them. Report single-pass and reranked results
separately. A more elaborate Vec2Text-style iterative corrector can also be trained
on factual pairs only, but it is a second-stage experiment, not a prerequisite for
the first valid decoder.

## 8. Objective and evidence that the model uses `z`

The primary optimization objective is exact target-token likelihood:

\[
\mathcal L_{\mathrm{CE}}
= -\frac{1}{\sum_i T_i}
  \sum_i\sum_{t=1}^{T_i}
  \log p_\phi(x_{i,t}\mid x_{i,<t},z_i).
\]

Include EOS, exclude virtual-prefix and padding positions, and report loss per
non-masked target token. Do not introduce label smoothing in the initial model,
because exact reconstruction is the ideal objective.

The narrow synthetic CV distribution makes it possible for a language model to
emit a plausible average CV while ignoring the latent. Every evaluation must
therefore include:

- correct `z`;
- deterministically batch-shuffled `z` from other IDs;
- mean fit embedding;
- zero vector;
- at least two different factual `z` values decoded with the same generation
  settings.

Correct conditioning should substantially improve target NLL and reconstruction
metrics over shuffled/mean/zero conditioning, and outputs should change with `z`.
If the gap is negligible, the run has failed even if the generated prose is
fluent.

If prefix+LoRA still ignores `z`, add a separately named factual-only variant
with a small mismatch-margin loss that requires correct-pair NLL to beat
shuffled-pair NLL. Do not silently change the primary objective mid-run.

## 9. Model and decoding selection

Predeclare selection before full training:

- within a training run, select the checkpoint with minimum factual-tune token
  NLL;
- require a positive, material correct-versus-shuffled conditioning gap;
- select architecture and decoding settings using factual tune only;
- because exact full-string match may be zero for every candidate, use normalized
  character edit similarity as the primary free-generation tie-breaker, followed
  by latent-cycle cosine and then checkpoint/run ID;
- run the factual audit only after all choices are frozen.

If compute permits, repeat the selected architecture with seeds 42, 43, and 44.
Do not choose the luckiest seed. Publish aggregate factual metrics and a
predeclared deployment seed, preferably 42.

## 10. Required factual evaluation

Evaluate free-running generation, not only teacher-forced likelihood.

### Exact and lexical reconstruction

Report at minimum:

- raw full-document exact match;
- whitespace-normalized exact match, reported separately;
- token accuracy under an explicitly documented alignment;
- normalized character Levenshtein similarity;
- ROUGE-L and chrF;
- generated/reference token and word lengths;
- premature-EOS and maximum-length termination rates.

Never present ROUGE or embedding similarity as exact reconstruction.

### Latent-cycle fidelity

Re-encode every generated CV using the exact frozen EmbeddingGemma implementation
and prompt, then report:

```text
cosine(f(x_hat), z_target)
L2(f(x_hat), z_target)
target retrieval rank among the evaluated z values
```

This is important but partly circular, so it must accompany rather than replace
text and attribute evaluation.

### Semantic and factual fidelity

Report recovery of the known factual state where possible, including country
region, degree, university tier, and the verbalized talent proxies. Prefer an
independent text extractor or human annotation. `g(f(x_hat))` is useful as a
cycle diagnostic but is not an independent judge because it reuses the encoder
under evaluation.

Provide at least 20 deterministic qualitative examples from the factual audit:
reference followed by reconstruction, chosen by fixed ID order rather than
hand-picking successful samples.

### Memorization checks

For every audit generation, report its closest fit CV by lexical similarity and
embedding cosine. Flag exact copies and long shared spans. Also report output
uniqueness. The decoder must beat the nearest-neighbor baseline without merely
copying training CVs.

Because templates and personas repeat across dataset rows, add a stress report
grouped by template and persona. Do not condition on those columns.

## 11. Oracle gate and final evaluation

The oracle-evaluation executable must be separate from training and must require:

1. a frozen checkpoint manifest;
2. a reload verification result;
3. a factual audit result;
4. a literal opt-in flag such as `--allow-oracle-evaluation`;
5. verification that model, checkpoint, tokenizer, candidate count, generation
   settings, and input hashes match the frozen manifest.

Only then may it read official-test `z_prime` from the canonical artifact and the
matching official-test rows from `cv_counterfactual.csv`. It must not read the
4,000-row `oracle_train/z_prime.pt` artifact.

Run and report these separately:

```text
factual ceiling:       D_phi(z_test)          vs x_text_test
oracle inversion:      D_phi(z'_oracle_test)  vs x'_text_test
learned intervention:  D_phi(z_hat'_test)     vs x'_text_test
```

For oracle and learned-intervention outputs, report all reconstruction metrics,
latent-cycle metrics, and causal-attribute fidelity. Stratify by the 256 identity
and 744 changed pairs. For identity pairs, `z'=z`, so deterministic decoding from
the same vector should be identical; assert this rather than merely averaging it.

Do not update the decoder after seeing these results.

## 12. Proposed repository layout

Keep generic model code separate from repo-specific experiment orchestration:

```text
src/text_realizer/
  __init__.py
  model.py              # latent projector + Gemma wrapper
  generation.py         # deterministic generation and optional reranking
  training.py           # numerical trainer, resume and checkpoint state
  metrics.py            # text, cycle and latent-use metrics
  artifacts.py          # manifests and strict reload

exp/text_realizer/
  __init__.py
  __main__.py            # prepare/preflight/train/audit/oracle-evaluate CLI
  data.py                # repo paths, ID joins, sealed split
  protocol.py            # hashes, plans and oracle gate
  report.py
  environment/
    pyproject.toml
    uv.lock

configs/text_realizer_embeddinggemma_768.yaml
scripts/run_text_realizer.sh
tests/test_text_realizer.py
tests/test_text_realizer_protocol.py
```

Use a dedicated pinned environment rather than modifying
`.venv-embeddinggemma`. It will need at least PyTorch, Transformers, Accelerate,
PEFT, pandas, PyYAML, and the exact Sentence Transformers dependency required to
run the existing EmbeddingGemma cycle evaluation. Quantization and
`bitsandbytes` are unnecessary for the 270M starting model unless hardware forces
a separately documented QLoRA variant.

Suggested local outputs:

```text
data/latents/talent/embeddinggemma_768/text-realizer-prepared/<split-hash>/
models/talent/embeddinggemma_768/text-realizer/<run-id>/
reports/talent/embeddinggemma_768/text-realizer/<run-id>/
```

Do not assume model/report paths are Git-trackable under the current `.gitignore`.
Keep large weights local or in an approved private artifact store. Decide later
which compact manifests/reports should be added to Git.

## 13. Target CLI contract to implement

The precise CLI spelling may change, but preserve these stage boundaries:

```sh
# 1. Create and hash the sealed factual-only bundle.
python -m exp.text_realizer prepare \
  --config configs/text_realizer_embeddinggemma_768.yaml

# 2. Verify tokenizer lengths, model access, GPU/dtype and one forward/generate pass.
python -m exp.text_realizer preflight \
  --config configs/text_realizer_embeddinggemma_768.yaml

# 3. Small end-to-end run: train, save, reload, resume and generate.
python -m exp.text_realizer train \
  --config configs/text_realizer_embeddinggemma_768.yaml \
  --run-id gemma270m-prefix-lora-smoke \
  --limit-fit 32 --limit-tune 16 --max-steps 20

# 4. Full factual-only run. Launch this through the detached script/tmux.
python -m exp.text_realizer train \
  --config configs/text_realizer_embeddinggemma_768.yaml \
  --run-id gemma270m-prefix-lora-v1

# 5. One factual audit after freezing all choices.
python -m exp.text_realizer audit \
  --run-id gemma270m-prefix-lora-v1

# 6. Separate later command; do not run during decoder development.
python -m exp.text_realizer oracle-evaluate \
  --run-id gemma270m-prefix-lora-v1 \
  --allow-oracle-evaluation
```

The detached launcher should create a uniquely named tmux session, refuse to
overwrite an existing run, log stdout/stderr, write a heartbeat/status JSON, and
print the attach command. Closing the terminal must not terminate training.

## 14. Checkpointing, resume, and provenance

Each run must write atomically:

- `plan.json` before training starts;
- `status.json` with state, heartbeat, epoch/step, ETA, best metric, and error;
- `history.jsonl` or equivalent complete training/validation history;
- `last/` with optimizer, scheduler, scaler, RNG, epoch, and sampler state;
- `best/` with projector, adapter/full model state, and tokenizer/config references;
- `factual_tune.json` and, only after freezing, `factual_audit.json`;
- deterministic reconstruction examples;
- `manifest.json` binding every artifact and input by SHA-256;
- a source/config snapshot or exact Git commit plus dirty-patch hash.

The manifest must include:

```text
EmbeddingGemma model and revision
EmbeddingGemma protocol and latent artifact SHA-256
Gemma decoder model and immutable revision
tokenizer revision and observed sequence-length statistics
projector architecture and K
LoRA/full-tune configuration
all optimizer/stopping/generation settings
literal split IDs and split hash
factual CSV and pair-index hashes
code/config/environment lock hashes
seed and numerical precision
best checkpoint hash
used_counterfactual_text: false
used_counterfactual_latent: false
used_official_test_for_selection: false
oracle_gate: sealed | opened-after-freeze
```

Resume must be bitwise/reproducible as far as the hardware permits and must never
reset early-stopping history. On interruption, resume from `last`; never overwrite
the already published `best` checkpoint without an atomic replacement.

Before declaring completion, start a fresh process, reload only the selected
artifact plus the pinned public base model, reproduce a fixed-logit probe and
fixed generation, and verify all hashes.

## 15. Required tests and preflight gates

Unit/integration tests must cover:

- ID join is exact and independent of CSV row order;
- no duplicate/missing IDs and no official-test ID in the factual bundle;
- no training code path accepts a counterfactual file or `z_prime` tensor;
- split reproduction and split hash;
- no tokenizer truncation;
- prefix, attention-mask, label-mask, and padding shapes;
- prefix and padding labels are `-100`, while CV/EOS targets are trained;
- changing `z` changes prefix embeddings and decoder logits;
- a finite forward/backward step on CPU or one GPU;
- tiny overfit test on one or two examples;
- save/reload preserves logits and generation;
- interruption/resume preserves step and early-stopping state;
- deterministic greedy generation in evaluation mode;
- shuffled/zero/mean conditioning controls;
- oracle command refuses to run without a frozen manifest and explicit opt-in;
- identity input vectors produce identical deterministic outputs;
- generated text can be re-encoded with the exact EmbeddingGemma protocol.

Do not launch the full run until:

1. all tests pass;
2. input hashes match;
3. the factual-only bundle is verified;
4. tokenizer length audit shows zero truncation;
5. a 20-step smoke run reduces loss;
6. save/reload and resume work;
7. GPU memory has at least 15% safety headroom;
8. generated smoke outputs depend measurably on `z`.

## 16. Hardware and operational requirements

Recommended starting resources:

- one CUDA GPU with BF16 support;
- 16 GB VRAM is likely sufficient for 270M LoRA with small microbatches;
- 24 GB or more is recommended for comfortable full-length training and possible
  full 270M fine-tuning;
- 32 GB host RAM;
- 15--25 GB free disk for model cache, environments, optimizer state,
  checkpoints, generated texts, and reports;
- network access once for gated Gemma download, followed by pinned/offline reuse.

CPU is sufficient for preparation, hashing, split validation, and some text
metrics, but not a sensible choice for full language-model training or generation.

Do not promise runtime before benchmarking the actual GPU. After preflight, time
at least 100 representative optimizer steps, include validation/generation
overhead, and write a measured ETA. A single 270M LoRA run should generally be an
hours-scale job rather than a multi-day job on a modern GPU; multiple variants,
seeds, and candidate reranking multiply that cost.

OOM fallback order:

1. lower microbatch size and increase gradient accumulation;
2. enable gradient checkpointing;
3. use memory-efficient SDPA/attention supported by the pinned environment;
4. shorten the virtual prefix only if it is a declared new variant;
5. use LoRA instead of full fine-tuning.

Never solve OOM by truncating CV targets or silently dropping examples.

## 17. Failure interpretation and decisions

| Observation | Likely interpretation | Next action using factual data only |
|---|---|---|
| fit and tune NLL remain high | optimization/interface bug or insufficient capacity | verify masks and conditioning; then LoRA/full tune |
| fit good, tune poor | memorization/overfitting | stronger regularization, earlier stopping, smaller model |
| fluent output but shuffled `z` is equally good | decoder ignores embedding | fix conditioning; test mismatch-margin variant |
| high latent cosine but low text fidelity | many-to-one/circular metric behavior | rely on lexical and independent factual metrics |
| retrieval beats decoder | decoder training or generation is weak | inspect target alignment and decoding before scaling |
| single pass weak, reranking improves cycle only | candidate search exploits encoder geometry | report both; do not call it exact recovery |
| factual audit strong, oracle later weak | counterfactual distribution shift | diagnose after freeze; do not train on oracle to repair this experiment |
| oracle strong, predicted `z_hat'` weak | latent intervention is off-manifold/inaccurate | return to `h_Z`, not the text decoder |

## 18. Definition of done for the GPU agent

The immediate factual-only task is complete only when the agent provides:

1. tested implementation and pinned environment;
2. verified factual-only bundle and exact split manifest;
3. retrieval, prefix-only, and prefix+LoRA results on factual tune;
4. one selected, frozen, reload-verified checkpoint;
5. one untouched factual-audit report with all required controls;
6. fixed-order reference/reconstruction examples;
7. a concise account of failures, conditioning sensitivity, memorization, GPU
   utilization, wall time, and peak memory;
8. exact commands and hashes required to reproduce the run;
9. an oracle-ready manifest whose gate is still marked `sealed`.

Stop and report rather than opening oracle data if any of the following holds:

- the decoder ignores `z`;
- validation texts were truncated;
- the selected checkpoint cannot be reloaded exactly;
- split or input hashes differ;
- the run used any official-test or counterfactual information;
- factual audit generation is mostly empty, duplicated, or copied from fit data.

The aspirational outcome is word-for-word reconstruction. The scientifically
defensible outcome is a decoder that is demonstrably conditioned on `z`, maximizes
held-out factual reconstruction under a sealed protocol, and provides a clean
upper-bound experiment for later counterfactual CV realization.
