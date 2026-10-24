# EmbeddingGemma CV reconstruction: 270M baseline and 1B pilot

Recorded: 2026-10-16. Status: completed, negative scaling result under the tested
configuration. All times below are Europe/Berlin (CEST, UTC+02:00).

This is the results and interpretation record for the isolated Gemma 3 1B pilot.
The [workflow runbook](text_realizer.md) describes implementation and operation;
the [original handoff](embeddinggemma_text_inversion_handoff.md) describes the
motivation. This document preserves the findings without changing the trained
models, historical reports, or downstream workflow.

## Conclusion and retained decision

Replacing the selected 270M text decoder with a 1B decoder, while retaining the
same factual embeddings and training recipe, **did not improve CV reconstruction**.
The 1B model achieved better teacher-forced token loss but worse free-running
text reconstruction and lower accuracy on three of four independently checked
factual fields. It failed the pilot's predeclared improvement criteria.

Retain the 270M decoder as the existing reference. Preserve the 1B checkpoint,
reports, and source snapshots as a completed experiment, not as its replacement.
Neither model has demonstrated reliable exact CV inversion or faithful
counterfactual text generation. In particular, high embedding-cycle similarity
is not sufficient evidence of factual fidelity.

Our leading explanation is **weaker use of person-specific conditioning relative
to the language model's ability to predict CV-like text**, compounded by an
earlier onset of overfitting. These are evidence-informed interpretations, not
an experimentally isolated causal explanation or proof that 1B models are
inherently worse.

## Question, scope, and comparison

The question was whether increasing the pretrained text decoder's capacity could
improve `D_text(Z)` without changing the frozen encoder or the project's latent
space. `D_text` generates text; it is not the structured semantic decoder `g`.

The frozen encoder remained `google/embeddinggemma-300m`, revision
`57c266a740f537b4dc058e1b0cda161fd15afa75`, producing the existing 768-D vectors.
No corpus-wide re-encoding, `g` training, `h_Z` training, or LangVAE modification
was performed. Only factual `(Z, CV text)` pairs supervised the text decoder.

| Setting | Existing reference | New pilot |
|---|---|---|
| Backbone | `google/gemma-3-270m` | `google/gemma-3-1b-pt` |
| Revision | `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1` | `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29` |
| Variant / comparison seed | `lora-compact` / 42 | `lora-compact` / 42 |
| Decoder hidden width | 640 | 1,152 |
| Prefix tokens / projector hidden width | 16 / 768 | 16 / 768 |
| Attention implementation | SDPA | Eager, after numerical checks |
| Microbatch / effective batch | 2 / 32 | 1 / 32 |
| Best checkpoint epoch | 8 | 5 |
| Epochs completed before early stopping | 13 | 10 |

Both models use `sqrt(768) * Z`, a two-layer GELU projector, and LoRA adapters
on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
LoRA rank is 16, alpha 32, and dropout 0.05. The base model, token embeddings,
and language-model output head remain frozen. Equal projector hidden widths and
LoRA ranks do not imply equal trainable parameter counts across backbones.

Shared optimization settings: FP32, AdamW, projector LR `1e-3`, LoRA LR `2e-4`,
weight decay 0.01, 5% warmup, cosine decay, gradient clipping at 1.0, gradient
checkpointing, at most 30 epochs, and early-stopping patience five. Each run's
checkpoint is selected by minimum tune token NLL, not by its best generated CVs.

The [1B configuration](../../configs/text_realizer_embeddinggemma_768_gemma1b.yaml)
and immutable run plans record the exact settings. This was a one-seed,
one-configuration pilot, not a 1B hyperparameter search or a clean parameter-count
ablation: the backbone, attention implementation, microbatching, and resulting
optimization trajectory differ.

## Data and scientific boundaries

- Fit: the existing 3,200 IDs, used for gradient updates and retrieval reference.
- Tune: the existing 400 IDs, used for stopping, diagnostics, and this comparison.
- Audit: 400 IDs, not opened by the 1B pilot. The earlier campaign's audit had
  already been viewed, so this follow-up is explicitly **audit-informed**.
- Official test: 1,000 IDs, not evaluated by the pilot.

The pilot copied only hash-verified development artifacts. It did not access
oracle training vectors, counterfactual targets, or edited `h_Z` outputs. The
audit and oracle gates remained closed. The results below are **tune-set results**,
not a fresh held-out audit, official-test result, or counterfactual fairness claim.

Targets were complete CVs with explicit BOS/EOS handling, no chunking, and no
silent truncation. Fit/tune targets including EOS contained 344–486 tokens,
median 405; tokenizer round trips introduced no raw-text discrepancies.
Generation used deterministic greedy decoding with the same 1,024-token budget.

## Execution and engineering findings

The corrected preflight completed on 2026-10-15 at 13:10. The authorized training
run began at approximately 15:02 that day. Training, all tune controls, and final
reload verification finished on 2026-10-16 at 05:01, about 14 hours after launch.
The original 27–32-hour planning estimate assumed all 30 epochs plus evaluation;
early stopping after 10 epochs shortened the run.

The pilot used one RTX 2080 Ti. Engineering checks established:

- 107 offline/regression tests passed; four opt-in GPU tests were skipped in
  ordinary collection. The dedicated 1B GPU cache test was run separately and passed.
- Cached/full logits agreed across the 512-token local-attention boundary with
  eager attention; the final preflight maximum absolute difference was about `2e-5`.
- Projector and LoRA gradients were nonzero; frozen backbone weights were unchanged.
- Resume continuation and fresh checkpoint reload checks passed.
- The diagnostic two-CV overfit check reached character similarity 1.0 for both
  examples. This demonstrates local fitting capability, not generalization.
- The longest-target memory check recorded approximately 6.23 GiB peak reserved
  memory; joint decoder/cycle-encoder evaluation recorded approximately 5.14 GiB.
  These are specific preflight measurements, not a claim that the complete
  training process always reserved only that amount of GPU memory.
- Benchmark timings were 23.59 seconds per effective-batch optimizer update,
  approximately 41.82 minutes per epoch including validation, and 17.27 seconds
  per generated diagnostic CV.

### Attention-backend issue and its limits

The initial SDPA preflight failed cached-versus-full logit parity on this
PyTorch 2.6 / RTX 2080 Ti FP32 stack. Eager attention and math-only SDPA restored
agreement in the diagnostic. The pilot explicitly records eager attention in
its configuration and portable checkpoint; the old 270M checkpoint was not
rewritten or silently switched to another backend.

The longer-forward parity test also failed for 270M under SDPA and passed under
eager attention. However, subsequent checks on 20 actual fit CVs gave effectively
identical NLL with SDPA/eager and microbatch 1/2. Two complete tune reconstructions
were identical between backends. Actual fit/tune targets, including prefix and
BOS, had maximum sequence length 503, below the 512-token local window.

Consequently, the diagnostic is a real numerical caveat, **not evidence that the
historical reconstruction scores were wrong**, and not an established explanation
for the 1B result. The limited real-CV checks do not replace a full matched-backend
replication. The immutable pilot report retains its conservative baseline warning;
this document records the follow-up evidence alongside it.

## Reconstruction and factual results

Both columns below evaluate the same 400 tune IDs, with each run's minimum-NLL
checkpoint. Arrows indicate the preferred direction.

| Reconstruction metric | 270M | 1B |
|---|---:|---:|
| Teacher-forced NLL, nats/token ↓ | 0.90786 | **0.77783** |
| Normalized character-edit similarity ↑ | **0.32939** | 0.30815 |
| Aligned word-token accuracy ↑ | **0.21433** | 0.18697 |
| ROUGE-L ↑ | **0.31233** | 0.28650 |
| chrF, scaled to 0–1 ↑ | **0.57035** | 0.56799 |
| Mean frozen-encoder cycle cosine ↑ | **0.95347** | 0.94953 |
| Mean cycle L2 distance ↓ | **0.29918** | 0.31286 |
| Raw / whitespace-normalized exact CV matches | 0 / 400 | 0 / 400 |
| EOS termination | 400 / 400 | 400 / 400 |
| Generation-budget termination | 0 / 400 | 0 / 400 |
| Mean generated words (reference mean: 350.69) | 338.00 | 357.45 |

Character similarity is whitespace-normalized Levenshtein similarity, not the
percentage of facts recovered. The aligned-token metric uses whitespace-separated
words, not the decoder's subword tokens. Both models produced 400 distinct outputs
and zero exact copies of fit CVs; this does not rule out partial memorization.

Fit-only nearest-neighbor retrieval, a **non-generative** reference, achieved
character similarity **0.31735** on these same IDs. The 270M decoder exceeded it
by 0.01205; the 1B decoder was below it by 0.00920. These tune comparisons do not
substitute for the original campaign's separately specified audit criterion.

### Independent factual checks

The evaluator applies conservative text rules to reference and generated CVs;
it does not use `g(f(text))` or tabular labels as a judge. Accuracy is conditional
on the reference being scorable. Missing or ambiguous generated values count as
failures, so a factual error score can reflect an incorrect value, omission, or
unparseable wording.

| Field | Scorable references | 270M accuracy | 1B accuracy | Change, percentage points |
|---|---:|---:|---:|---:|
| Country | 397 / 400 | 95.21% | **98.49%** | +3.27 |
| Degree | 377 / 400 | **93.10%** | 84.62% | -8.49 |
| University tier | 400 / 400 | **70.50%** | 57.25% | -13.25 |
| Project count | 313 / 400 | **15.02%** | 12.46% | -2.56 |
| Equal-weight mean of the four accuracies | Different denominators above | **68.46%** | 63.20% | -5.25 |

Region is excluded from that mean to avoid counting country/region evidence twice.
The number of unparseable generated values, restricted to scorable references,
was 4 → 1 for country, 18 → 22 for degree, 6 → 0 for university tier, and 57 → 110
for project count. Thus project-count deterioration partly reflects extraction
coverage, whereas the university-tier decline cannot be explained by parser
failure: all 400 generated 1B tiers were parseable.

In the fixed-order review examples, tune ID 1147 illustrates a concrete error:
the reference and 270M reconstruction specify a top-100 university; 1B specifies
top-500. Its reconstruction also changes the described tool-learning duration
from a few months to more than a year. This is an inspected example, not a
quantified behavioral-fidelity metric across all CVs. Full examples remain in
the local report, not duplicated into Git documentation.

## Conditioning controls and predeclared decision

All controls keep the reference CV fixed while changing only the decoder's input
vector. Mean-fit and zero conditioning generate a single common output per model;
nearest-fit-vector conditioning is still generation, unlike copying the nearest
fit text in the retrieval baseline above.

| Conditioning | 270M NLL ↓ | 1B NLL ↓ | 270M character similarity ↑ | 1B character similarity ↑ |
|---|---:|---:|---:|---:|
| Correct vector | 0.90786 | 0.77783 | 0.32939 | 0.30815 |
| Deterministically shuffled vector | 1.55428 | 0.97268 | 0.27399 | 0.26691 |
| Mean fit vector | 1.21308 | 0.89181 | 0.28357 | 0.28930 |
| Zero vector | 1.20254 | 0.86201 | 0.28901 | 0.25786 |
| Nearest fit vector | 0.94051 | 0.78675 | 0.31849 | 0.30489 |

Shuffling increases NLL by **0.64642** for 270M but only **0.19485** for 1B.
Correct conditioning improves character similarity over shuffled conditioning
by **0.05540** and **0.04124**, respectively. Both models use the embedding, but
the 1B model is less sensitive on these measures to its being the correct one.
These are intervention diagnostics on the conditioning input, not estimates of
mutual information or proof of a unique training mechanism.

| Pilot requirement | Observed 1B result | Decision |
|---|---:|---|
| Character-similarity gain over 270M ≥ 0.02 | -0.02124 | Fail |
| Mean factual-accuracy gain ≥ 0.02 | -0.05255 | Fail |
| No individual factual-field decline > 0.02 | Degree -0.08488; tier -0.13250; count -0.02556 | Fail |
| Shuffled-minus-correct NLL ≥ 0.05 | 0.19485 | Pass |
| Correct-minus-shuffled character similarity ≥ 0.02 | 0.04124 | Pass |

The paired character-gain bootstrap interval was **[-0.02706, -0.01548]**, using
2,000 resamples and seed `20261012`. Because these are tune data used for model
selection and the comparison is audit-informed, this interval is descriptive,
not a confirmatory population claim. The recorded decision is `promising: false`;
no extra seeds, final audit, or oracle evaluation were automatically launched.

## Why 1B was not better: evidence versus hypotheses

### 1. Token prediction and source-faithful generation are different objectives

**Observed:** teacher-forced NLL improved from 0.90786 to 0.77783 while generated
character similarity, aligned-token accuracy, ROUGE-L, and most factual checks
worsened.

During training and NLL evaluation the decoder sees the correct preceding words
when predicting the next token. During generation it sees its own preceding
predictions. A model can become better at completing a supplied reference prefix
without becoming better at recovering that entire reference from `Z` alone.
An early generated mistake can also affect later decisions. This general
training–generation distinction is discussed by
[Bengio et al., *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks*](https://arxiv.org/abs/1506.03099).
That paper motivates the distinction; we did not test scheduled sampling here.

The token-averaged objective gives every target token a contribution, but does
not give a university tier or project count special factual importance. Better
prediction of many ordinary words can outweigh worse prediction of a few
research-critical facts. Minimum-NLL checkpoint selection is therefore not
guaranteed to select the best free-running factual reconstruction checkpoint.
We followed the predeclared selection rule; we did not retrospectively select a
more favorable checkpoint or decoding policy.

### 2. The larger decoder may rely more on text context and learned CV patterns

**Observed:** its shuffled-conditioning penalties are smaller. Even with a zero
vector, 1B has lower teacher-forced NLL than correctly conditioned 270M
(0.86201 versus 0.90786), despite worse generated character similarity.

**Leading hypothesis:** the stronger language model can satisfy more of the
training objective through the supplied preceding text and familiar CV patterns,
reducing pressure to learn precise person-specific information from the vector.
The control results are consistent with this mechanism, but do not establish it
uniquely. They do not mean the decoder entirely ignores `Z`.

Related text-VAE work demonstrates that a powerful autoregressive decoder can
underuse global latent information:
[Bowman et al., *Generating Sentences from a Continuous Space*, Section 3.1](https://arxiv.org/html/1511.06349v4#S3.SS1).
This is a useful analogy, **not a diagnosis of VAE posterior collapse here**:
our encoder is frozen, there is no learned variational posterior or KL penalty
in this decoder training, and both conditioning gates passed.

### 3. Overfitting starts earlier, but is not a complete explanation

| Run / epoch | Fit NLL | Tune NLL |
|---|---:|---:|
| 270M, best epoch 8 | 0.71953 | 0.90786 |
| 270M, final epoch 13 | 0.45754 | 1.00699 |
| 1B, best epoch 5 | 0.65340 | 0.77783 |
| 1B, final epoch 10 | 0.31164 | 0.92898 |

Fit NLL is the average observed during that epoch's updates; tune NLL evaluates
the end-of-epoch model. The consistent direction across epochs nevertheless
shows the relevant pattern: training loss keeps decreasing while validation loss
increases. The 1B run reaches its validation minimum earlier on the same 3,200
fit CVs. Continuing this exact run is not supported by the observed trajectory.

However, the reported reconstructions use the **best epoch-5 checkpoint**, not
the overfitted epoch-10 checkpoint. Later overfitting explains why more epochs
are unpromising; it does not by itself explain why the selected 1B checkpoint
reconstructs worse than the selected 270M checkpoint. Zero exact fit copies also
does not exclude overfitting or partial memorization.

### 4. The copied optimization recipe may not suit the larger backbone

**Untested hypothesis:** projector and LoRA learning rates, the number of prefix
tokens, adaptation strength, or the balance between learning the projector and
adapting the language model may need different settings for 1B. The same nominal
recipe does not guarantee equally effective optimization across model sizes.

We have not run a learning-rate search, projector-first warmup comparison,
alternative checkpoint-selection study, or 1B seed replication. Consequently,
the experiment supports a negative result for this practical configuration,
not a claim that every possible 1B decoder would perform worse.

## Knowledge retained for the project

1. **More decoder capacity alone was insufficient.** It improved token likelihood,
   not the desired reconstruction outcome, in this pilot.
2. **Cycle similarity and recoverable facts are different.** Cosines near 0.95
   coexist with poor university-tier/project-count fidelity and zero exact CV
   matches. Re-encoding cannot serve as the sole independent factual judge.
3. **Conditioning controls are necessary.** A model can produce distinct outputs
   and pass a small overfit check while still using instance-specific vectors
   insufficiently for full-CV recovery.
4. **The result does not identify the encoder's information ceiling.** Both models
   receive exactly the same vectors. Poor decoding does not prove the missing
   facts are absent from those vectors; that requires a separate recoverability
   investigation. A shared information limitation would not, by itself, explain
   the relative 1B regression.
5. **Exact text, factual content, and readable prose must be evaluated separately.**
   Some lexical differences can be harmless paraphrases; the observed factual
   substitutions show that this result is not only a wording issue. No systematic
   human readability or behavioral-fidelity score was established by this pilot.
6. **Factual inversion does not validate `D_text(Z')`.** Oracle counterfactual
   vectors and learned `h_Z` outputs remain untested here. This run says nothing
   conclusive about counterfactual faithfulness or fairness, nor does it invalidate
   downstream `g`/`h_Z` models trained on the unchanged latent space.

The current decision is to retain both models and the negative result, with 270M
remaining the reference. Any further investigation should be separately
authorized and predeclared. Plausible follow-ups include a matched-backend check,
a small adaptation-strength/learning-rate study, or fact-recoverability probes;
none was run or selected as a result of writing this document. Do not use the
previously viewed audit as a fresh holdout or silently change the completed run.

## Artifact inventory and preservation

The following identifiers locate the complete local evidence:

| Role | Run ID |
|---|---|
| Historical campaign | `cv-text-v1-20261012-1719` |
| Compared 270M training run | `cv-text-v1-20261012-1719.lora-compact.seed-42` |
| 1B pilot orchestration | `cv-text-1b-pilot-20261015-1026` |
| 1B training/evaluation run | `cv-text-1b-pilot-20261015-1026.lora-compact.seed-42` |
| Backend compatibility diagnostics | `1b-pilot-compatibility-20261015` |

Reports are under `reports/talent/embeddinggemma_768/text-realizer/`; models are
under `models/talent/embeddinggemma_768/text-realizer/`. Relevant direct links:

- [Pilot decision and paired comparison](../../reports/talent/embeddinggemma_768/text-realizer/cv-text-1b-pilot-20261015-1026/pilot_result.json).
- [Immutable pilot plan](../../reports/talent/embeddinggemma_768/text-realizer/cv-text-1b-pilot-20261015-1026/plan.json), with configuration/input/source identity; the adjacent `source/` directory preserves implementation snapshots.
- [1B tune metrics, controls, and per-CV outputs](../../reports/talent/embeddinggemma_768/text-realizer/cv-text-1b-pilot-20261015-1026.lora-compact.seed-42/factual_tune.json) and [270M comparator](../../reports/talent/embeddinggemma_768/text-realizer/cv-text-v1-20261012-1719.lora-compact.seed-42/factual_tune.json).
- [1B learning curve](../../reports/talent/embeddinggemma_768/text-realizer/cv-text-1b-pilot-20261015-1026.lora-compact.seed-42/history.json) and [270M learning curve](../../reports/talent/embeddinggemma_768/text-realizer/cv-text-v1-20261012-1719.lora-compact.seed-42/history.json).
- [First 20 fixed-order tune comparisons](../../reports/talent/embeddinggemma_768/text-realizer/cv-text-1b-pilot-20261015-1026/tune_examples.json); these are not audit examples.
- [1B checkpoint reload verification](../../reports/talent/embeddinggemma_768/text-realizer/cv-text-1b-pilot-20261015-1026/verify.json).
- [270M real-CV backend checks](../../reports/talent/embeddinggemma_768/text-realizer/1b-pilot-compatibility-20261015/270m-attention-real-cvs.json) and [unchanged 270M reload verification](../../reports/talent/embeddinggemma_768/text-realizer/1b-pilot-compatibility-20261015/270m-reload.json).
- [1B preflight measurements](../../data/latents/talent/embeddinggemma_768/text-realizer-prepared/gemma1b-pilot/aca7014d6ceba4ea3b08674ebfdcd7567157452199d60cb1f7a6fa726b23eea8/preflight.json).
- [Selected 1B inference checkpoint](../../models/talent/embeddinggemma_768/text-realizer/cv-text-1b-pilot-20261015-1026.lora-compact.seed-42/best.pt) and [rolling training/resume checkpoint](../../models/talent/embeddinggemma_768/text-realizer/cv-text-1b-pilot-20261015-1026.lora-compact.seed-42/last.pt).
- [Preserved 270M deployment checkpoint](../../models/talent/embeddinggemma_768/text-realizer/cv-text-v1-20261012-1719/selected/seed-42.pt).

SHA-256 anchors, verified when this writeup was created:

| Artifact | SHA-256 |
|---|---|
| 1B `best.pt` | `59af957c6fdf29a86154efb40d66bbde4038ad82f5818a7a3afa0fcd133d84af` |
| 270M `selected/seed-42.pt` | `0d459c18343b1eb62573c7c0c19647b3eaf9b3b42e90a0412639220fd31adff7` |
| 1B `factual_tune.json` | `e724e6b55fc79bcda4f10488212b2bf07a14c4de8d587e0cf638da9bd346ef1c` |
| 270M `factual_tune.json` | `745873bb24823846b6debe1e12aef22a2a60e1af522d0d0d92a2088772ad3473` |
| Pilot `pilot_result.json` | `b5958d38d7aa25f6e70c02be2827d9afdbd9f612fa5e3699de44847ff13a5861` |
| Pilot `verify.json` | `c043ba84a426de033f8b165835b311aabcc1214e787997da1c6c3eda2cf5ebfb` |

The portable `best.pt` contains projector/adapter weights and pinned model
references, not the base-model weights or corpus. Loading it still requires the
cached pinned backbone; use the existing inference API described in the runbook.
It is an experimental checkpoint, not a promoted 1B deployment or a campaign
freeze that authorizes audit/oracle evaluation.

These generated model/report/prepared artifacts remain at their existing local
paths and are ignored by Git. This Markdown preserves the key numerical results,
interpretation, identifiers, and checksums in a lightweight, commit-eligible
form; **committing the documentation alone does not back up the checkpoints or
full per-CV reports**. A future transfer must separately copy the required
artifacts and pinned model references through an authorized channel. No files
were deleted, checkpoints overwritten, uploads made, or Git changes staged as
part of this writeup.
