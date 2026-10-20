# Counterfactual-exposed EmbeddingGemma-to-CV decoder: paired-data check and pilot plan

Recorded: 2026-10-17; updated 2026-10-20. Status: **paired-data readiness
verified; both 600-update arms and the tune-only comparison completed; the
predeclared replication gate failed**.
This is a separate,
privileged-supervision comparator for text reconstruction, not a continuation of
the [failed factual-only corrector](text_corrector_results.md) and not a change to
the main `h_Z` method. The selected [270M factual-only decoder](text_realizer.md)
remains the unchanged reference.

## Scientific question and boundary

Can a decoder trained with simulator-provided, aligned `(Z', CV')` pairs produce
more faithful counterfactual CVs from a vector than the existing decoder trained
only on factual `(Z, CV)` pairs? Here `Z' = f(CV')` is the **oracle encoded
counterfactual CV**. It is not an `h_Z` prediction. In the user's shorthand
`(Z', X')`, `X'` means counterfactual *text*; in the simulator's `do(X=3)`, `X`
instead means *country-of-origin region*. This document uses `CV` for text to
avoid that collision. Mediators and prose may also change, so accuracy must be
judged against `CV'`, not just by checking the country.

The project rule that `Z'` never enters a loss applies to the proposed
transferable **`h_Z` method**. This comparator deliberately violates that rule
*for `D_text` only* and must always be labelled **oracle-supervised** or
**counterfactual-exposed**. It does not retrain EmbeddingGemma, `g`, `h_Z`, the
flow, or the original decoder. A gain with oracle `Z'` cannot establish that
`D_text(h_Z(Z))` works: edited vectors may have a different distribution and
may fail to encode the target CV.

## Verified local pair availability

The checks below joined by literal candidate ID, not row number. Input bytes
matched their recorded SHA-256 hashes; the encoder revision and exact protocol
matched the factual development bundle. The oracle container is train-only and
has no official-test IDs. The counterfactual CSV contains all 5,000 unique IDs,
but the pair analysis retained only fit/tune rows. Integrity checks necessarily
read the full source files and the 4,000-row oracle tensor. The readiness checks
used only fit/tune CV contents and token lengths; they did not evaluate audit or
official-test examples. Earlier project analyses did view the factual audit,
so this new pilot must not be described as an untouched confirmatory study.

| Partition | Units | Identity `Z'=Z`, `CV'=CV` | Changed pairs | Proposed use |
|---|---:|---:|---:|---|
| Fit | 3,200 | 821 | 2,379 | Gradient updates |
| Tune | 400 | 100 | 300 | Predeclared pilot selection/evaluation split below |
| Audit | 400 | Not assessed here | Not assessed here | Gated post-freeze evaluation |
| Official test | 1,000 | Not assessed here | Not assessed here | Separate explicit final gate |

The 3,600 fit/tune `(Z', CV')` pairs are ID-aligned, nonempty, and unique as
counterfactual texts. All fit/tune identity texts equal their factual texts
exactly, and their oracle vectors equal the stored factual vectors bitwise.
Every changed fit/tune pair has `CV' != CV`. All oracle vectors are finite FP32,
768-dimensional and unit-normalized (maximum norm error over the train-only
source tensor: `1.20e-7`). As a direct alignment spot check, six fixed changed
fit/tune CVs (fit IDs 3050, 408, 1652; tune IDs 4204, 246, 739) were
re-encoded with the frozen, pinned EmbeddingGemma encoder;
cosine to their stored `Z'` was at least `0.99999994`, and L2 distance at most
`1.03e-6`. This is **not** a full re-encoding audit of all 2,679 changed
fit/tune pairs.

With the pinned Gemma 3 270M tokenizer, all 3,600 counterfactual fit/tune
texts round-trip without a raw-text discrepancy. Complete targets including
EOS are 348–483 tokens on fit and 353–462 on tune; none exceeds 1,024 tokens.
The longest observed fit/tune target plus the existing 16-token prefix and BOS
is 500 positions, within the pinned model's 32,768-position context. Do not
truncate targets or derive length settings from audit/test content. Retain the
existing 1,024-token generation limit and test cached/full-logit parity around
the 512-position local-attention boundary in preflight, as the prior 270M/1B
comparison identified an FP32 SDPA caveat there.

Immutable source references for a future run:

| Item | Path / SHA-256 |
|---|---|
| Oracle train vectors | `data/latents/talent/embeddinggemma_768/oracle_train/z_prime.pt` — `7a3e94f15d7343dcba38772553ce0d3aef9234a0aa091d5acdd5df64b8da02d4` |
| Counterfactual CV text | `data/text_talent/cv_counterfactual.csv` — `34b3a3ac0747bdd65ab8d7175c8582a9e9d38fcf06ec31f436b5c499ec9f993c` |
| Factual prepared bundle | `data/latents/talent/embeddinggemma_768/text-realizer-prepared/815a05b566bc35114b8c2a4fa03f1e5f372b26ce5e224a9fffb5b661f7590223/development.pt` — `bdb49b8ba94d2af2188b0ef205ff11689dd99d25ff32b1feb1262b781d0e2167` |
| Existing split plan | `reports/talent/hz_benchmarks/hz-gemma-v1/plan.json` — `6698da61c789e3dad586d051e8bdad40157786e01c858212d89b95d5ee90b4be` |
| Unchanged 270M reference | `models/talent/embeddinggemma_768/text-realizer/cv-text-v1-20261012-1719/selected/seed-42.pt` — `0d459c18343b1eb62573c7c0c19647b3eaf9b3b42e90a0412639220fd31adff7` |

The original vector artifact's metadata says `purpose: oracle_regression_only`.
That historical artifact must remain untouched. For this new task, a dedicated
preparer should validate it and emit a **new**, hash-bound, ID-filtered
`privileged_text_decoder` development bundle. It must not call the existing
oracle-regression loader, which has a different access scope, or the factual
loader as if it already contained counterfactual targets. The new training
entry point should accept only the derived fit/tune bundle, never the canonical
paired artifact or the complete oracle container.

## Predeclared pilot: three arms, one architecture

Use seed 42 and the existing `lora-compact` Gemma 3 270M architecture for
every arm: pinned base revision `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`,
16 prefix tokens, 768-wide two-layer projector, rank-16 LoRA on the same seven
attention/MLP projections, and the frozen EmbeddingGemma encoder revision
`57c266a740f537b4dc058e1b0cda161fd15afa75`. Full CVs plus EOS remain the
only target; no tabular label, template/persona ID, source CV, or intervention
code enters the model input. Inference continues to accept only a finite
`[batch, 768]` vector.

**Numerical-backend amendment, fixed before training (2026-10-17):** all three
arms use Transformers' `eager` attention when compared. The original A weights
remain unchanged; A is regenerated for this comparison on the common backend,
not assigned its historical SDPA score. The earlier 270M/1B investigation found
an FP32 cached/full-logit discrepancy with SDPA around the local-attention
boundary. Preflight must verify eager cached/full parity on this machine.

| Arm | Initial weights | Additional training | Purpose |
|---|---|---|---|
| `A_factual_original` | Selected checkpoint above | None | Unchanged factual-only reference |
| `B_factual_continuation` | Exact copy of A | Factual `(Z, CV)` only | Equal-update continuation control |
| `C_oracle_exposed` | Exact copy of A | Factual `(Z, CV)` plus changed `(Z', CV')` | Incremental value of paired counterfactual supervision |

Exclude the 821 identity fit counterfactuals from C's counterfactual stream;
they duplicate factual examples exactly. C draws 16 factual and 16 changed
counterfactual targets per effective batch of 32; B draws 32 factual targets.
Each stream independently reshuffles and cycles with a recorded RNG state.
Both B and C receive **600 optimizer updates** with the same number of target
sequences, optimizer settings, gradient accumulation, and checkpoint schedule.
This controls additional compute but not the fact that B receives more factual
repetitions, which is part of the data-mixture tradeoff. No synthetic `CV'`,
counterfactual tabular attributes, or `h_Z` outputs enter either training loss.

For both continuations, start a *new* AdamW optimizer from the copied selected
weights: projector LR `3e-4`, LoRA LR `5e-5`, weight decay `0.01` (excluding
bias/norm), 30-update warmup and cosine decay, gradient clipping at 1.0,
FP32, and gradient checkpointing. Retain the original effective batch of 32;
preflight chooses microbatch 1 or 2 with at least 15% GPU headroom. Save at
updates 100, 200, 300, 400, 500, and 600, plus atomic resumable `last` state.
Any necessary change after preflight requires a versioned amendment **before**
examining counterfactual output quality; never retune this plan on audit/test.

Use the first 80 literal tune IDs in the prepared manifest **only in their
factual world** for checkpoint selection. This subset has 18 identity and 62
changed-world units, but neither B nor C selection accesses those `CV'` texts.
For each continuation, choose the checkpoint maximizing
`(character_similarity + four_field_macro_accuracy) / 2` on these 80 factual
CVs; tie-break by higher character similarity, lower factual token NLL, then
earlier update. Both terms are on the 0–1 scale, and the fields are country,
degree, university tier, and project count, using the existing conservative
text extractor. This generated-text selection reflects the observed failure
of token NLL alone to predict free-running fidelity in the 1B pilot. A is
never reselected.

The remaining 320 tune IDs are the **pilot evaluation set**: 238 changed and
82 identity. They are disjoint from the new checkpoint-selection units. The
historical A checkpoint was selected using the original tune set, and earlier
project analysis has viewed factual audit results; therefore this comparison
is exploratory/developmental, not a pristine confirmatory test. Keep all worlds
of each candidate in the same original split. Do not evaluate or inspect the
400 audit counterfactual targets until both continuation checkpoints and this
comparison rule are frozen.

## Evaluation, decision rule, and limits

Generate deterministically from each arm on the *same* input vectors, with
the same tokenizer, 1,024-token budget, and eager numerical backend. Evaluate all
three arms on oracle `Z'` against `CV'`, separately for changed and identity
units; also evaluate factual `Z` against `CV` to detect lost factual capability.
The **primary pilot population is the 238 changed tune-evaluation units**;
identity units must not dilute its score. For identity units, identical `Z`
and `Z'` must produce identical deterministic text within each model.

Report normalized character-edit similarity, exact match, chrF, ROUGE-L,
aligned-token accuracy, output length/termination, fit-copy/long-span checks,
and frozen-encoder cycle cosine **separately**. Independent text extraction
reports country/region, degree, university tier, and project count, each with
reference scorable count and generated coverage. Missing or unparseable
generated values count as wrong where the reference is scorable. Report the
equal-weight four-field macro excluding region, template/persona strata, and
fixed-order human examples for occupation, behavioral episodes, unsupported
details, and readability. High cycle cosine or country correctness alone is
not evidence of faithful CV reconstruction.

Predeclared **pilot-to-replication gate** for C on changed tune units:

1. Mean character-similarity gain is at least `0.02` against **both** A and B,
   and each paired, unit-bootstrap 95% interval has lower bound above zero
   (2,000 resamples; seed `20261017`).
2. Four-field macro factual accuracy improves by at least `0.02` against A;
   no individual scorable field declines by more than `0.02` against A.
3. On the same 320 factual tune-evaluation units, C loses less than `0.02` in
   character similarity and macro factual accuracy against A. Report the B
   comparison too. No exact-copy or widespread length-limit failure is hidden
   by average scores.

These thresholds are operational advancement rules, not a guarantee of
recoverability or a formal significance claim. A positive one-seed tune pilot
would justify **separately authorized** seed-43/44 replication with the same
protocol, followed by a frozen, gated audit. A failed gate retains the negative
result and does not automatically trigger model scaling or a new corrector.
The official 1,000-unit test remains behind a distinct explicit authorization
gate after all choices are frozen. No checkpoint is automatically promoted.

Finally, evaluate supplied, ID-aligned `h_Z` output vectors only **after** the
oracle-input comparison is frozen, against their own `CV'` references. Report
oracle-`Z'` and `h_Z(Z)` results in separate columns; do not use `h_Z` outputs
for training or selection and do not retrain `h_Z`. Results are specific to
this one simulated intervention and cannot by themselves establish
counterfactual fairness or generalization to real CVs.

## Implementation boundary followed

The pilot puts its data gate, trainer/orchestrator, config, tests, model/reports,
and runbook in **new** counterfactual-exposure paths.
Reuse the read-only text-realizer model/evaluation primitives, but do not alter
the factual campaign's data loader, checkpoint, manifest, or oracle-evaluation
gate. The preparer must assert source hashes, split membership, exact ID joins,
identity equality, absence of audit/test IDs, and the unchanged encoder
protocol. Tests should prove wrong or missing IDs and attempts to pass the
whole oracle artifact into training are rejected. Record source/config hashes,
RNG and sampler state, target-length statistics, all generated diagnostics, and
reload verification. No upload, Git staging, or training should occur during
preparation.

## Implemented interface and operation

The additive implementation lives in `exp/text_realizer_cf/`, with its own
[configuration](../../configs/text_realizer_counterfactual_exposure.yaml),
offline tests in `tests/text_realizer_cf/`, and an explicit detached arm launcher
at `scripts/run_text_realizer_cf.sh`. It reuses the already-isolated
`exp/text_realizer/environment/.venv` and read-only `src/text_realizer`
primitives; no root dependencies, factual campaign source, canonical encoding,
or `h_Z` code were changed. The generated bundles, checkpoints, and reports
live only under separate `text-realizer-cf` data/model/report directories.

From the repository root:

```bash
cf_python=exp/text_realizer/environment/.venv/bin/python
"$cf_python" -m pytest -q tests/text_realizer_cf
"$cf_python" -m exp.text_realizer_cf prepare
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$cf_python" -m exp.text_realizer_cf preflight --device cuda:0
```

Preflight performs data/tokenizer and offline-test checks, eager cached/full
logit parity, frozen-parameter/gradient checks, 15%-headroom microbatch tests,
a 20-update two-world smoke, deterministic resume/reload, and a 10-update
*effective-batch-32* benchmark. It writes `preflight.json` with `ready: true`
only if every check passes. Diagnostic weights are marked non-promotable.
Neither `prepare` nor `preflight` starts the 600-update comparison.

Only after a passing preflight, explicitly start the two arms, each with its
own GPU lease and resumable checkpoints:

```bash
bash scripts/run_text_realizer_cf.sh cv-cf-pilot-v1 0 factual
bash scripts/run_text_realizer_cf.sh cv-cf-pilot-v1 1 oracle
"$cf_python" -m exp.text_realizer_cf status --run-id cv-cf-pilot-v1
# After both arms finish and reload-verify:
"$cf_python" -m exp.text_realizer_cf evaluate \
  --run-id cv-cf-pilot-v1 --device cuda:0
```

The `train` and `evaluate` entry points fail closed without their respective
preflight/completion records. Use `--resume` with the same arm and run ID after
an interruption; changed code, environment, config, or inputs require a new
run ID. The evaluation command only opens the separated 320-unit tune pilot
partition after both arm checkpoints have completed and verified. It does not
open audit, official test, or `h_Z` vectors. The detached launcher and commands
above describe how the run was started; completion evidence is recorded below.

### Preflight result (2026-10-17)

The dedicated preflight completed successfully on GPU 0 (RTX 2080 Ti, 11 GB).
Its exact record is in the ignored prepared directory's `preflight.json`; the
result is bound to the current config, source hashes, package versions, and
prepared-manifest hash. The earlier attempt inside the restricted sandbox
could not see CUDA; rerunning with GPU device access completed all checks.

| Check | Observed |
|---|---:|
| Offline focused tests inside preflight | 19 passed |
| Complete fit/selection targets; tokenizer round-trip discrepancies | 5,659; 0 |
| Longest complete target including EOS | 486 tokens |
| Eager cached/full-logit maximum absolute difference across position 512 | `9.20e-5` (passed native tolerance) |
| Microbatch 2 peak reserved memory | 5.54 GB of 11.54 GB; 15% headroom passed |
| Two-world 20-update NLL | 0.75720 → 0.02818 |
| Frozen base weights / projector / LoRA checks | Unchanged / nonzero / nonzero |
| Exact resume and fresh checkpoint reload | Passed |
| Ten effective-batch-32 benchmark updates | 107.51 seconds; 10.75 seconds/update |

The benchmark suggested about **1.8 hours of optimizer updates per arm**
(`600 × 10.75 s`), before six factual-selection generations, checkpoint saves,
startup, and evaluation. It was a measured extrapolation, not a completion
deadline. The preflight's diagnostic checkpoint remains non-promotable.

### Training and comparison status (2026-10-19)

Run `cv-cf-pilot-v1` started both arms on 2026-10-17 at 19:07 CEST, on separate
RTX 2080 Ti GPUs. Both completed all 600 updates and passed fresh-checkpoint
reload verification. Factual-only continuation selected update 400, with
checkpoint SHA-256 `8dab6756af94d7262024b36946ee0c7c4bf85b653570f6bc1f2a752c7b210f5f`.
Oracle-exposed continuation selected update 600, with checkpoint SHA-256
`fda6710871f3b67f681d5a42874dca8ab88e770ad1a528b6937692836e3f7b5b`.
The former used no counterfactual latent or text in training; the latter used
both, as predeclared. The unchanged original decoder remains arm A.

The 320-unit tune-only comparison of A, B, and C began on 2026-10-19 at
10:55 CEST and finished at 18:21 CEST. The completed arms' 80-unit factual
checkpoint-selection scores are not the counterfactual evaluation.

### Tune-only result and retained decision (2026-10-20)

The primary population was the 238 changed tune-evaluation CVs. All three
models received the same true oracle `Z'` vectors in this comparison; none
received an `h_Z` prediction. Character similarity and the independent
four-field macro factual accuracy were:

| Arm | Changed-CV character similarity | Changed-CV four-field macro | Factual-CV character similarity | Factual-CV four-field macro |
|---|---:|---:|---:|---:|
| A: original factual-only | 0.33033 | 0.66962 | 0.33075 | 0.68515 |
| B: factual-only continuation | 0.33115 | 0.68428 | 0.33102 | 0.73104 |
| C: oracle-exposed continuation | 0.34563 | 0.72379 | 0.34151 | 0.74744 |

C's paired mean character gain was `+0.01530` versus A (bootstrap 95% interval
`[+0.00775, +0.02286]`) and `+0.01448` versus B (`[+0.00719, +0.02193]`).
Both intervals exclude zero, but **both gains miss the predeclared `+0.02`
minimum**, so the two primary character gates fail. C passed the macro-factual
gain and factual-retention gates. Relative to A on changed CVs, university-tier
accuracy rose by `+0.12605` and project-count accuracy by `+0.11628`; country
fell by `-0.01702` and degree by `-0.00862`, both within the predeclared
`-0.02` field-regression tolerance. No arm hit the 1,024-token length limit
or produced an exact normalized fit-set copy in these subsets.

The combined decision is **`worth_replication: false`**. Do not launch seeds
43/44, open the audit or official test, or promote C as a faithful CV inverter
under this protocol. At the user's request, the selected A and C inference
checkpoints remain available as separate baselines, with C clearly labelled
oracle-supervised and exploratory. B's checkpoint, resume state, and arm-specific
report are now in the [local archive](../../.archive/README.md), while the
combined per-ID A/B/C comparison remains in the Git-ignored
`reports/talent/embeddinggemma_768/text-realizer-cf/cv-cf-pilot-v1/result.json`.
Audit, official test, and `h_Z` output vectors remain unopened by this pilot.
