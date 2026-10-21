# EmbeddingGemma CV reconstruction: factual-only 270M correction pilot

Recorded: 2026-10-17. Status: **completed; did not meet the predeclared continuation
criteria**. Times below are Europe/Berlin (CEST, UTC+02:00).

The [pilot runbook](text_corrector.md) documents the architecture, data boundary,
preflight, training recipe, and decision rule. This file records the measured result
without changing the original 270M decoder or promoting the corrector. It follows
the completed [270M/1B comparison](text_realizer_gemma1b_results.md).

**Archive update (2026-10-20):** all corrector-specific code, weights, optimizer
state, and local reports were moved to the [decoder archive](../../.archive/README.md).
The negative result and its hashes remain documented here. On 2026-10-21 the
user approved tracking the full archived JSON reports, including generated text;
see the [results index](decoder_results.md). Code, weights, optimizer states,
prepared data, and source snapshots still require separate transfer.

## Conclusion and retained decision

One learned correction pass produced a very small mean character-similarity gain
over the unchanged 270M draft decoder (**+0.00372**, against the required +0.02),
with a descriptive paired-bootstrap interval that includes zero. Equal-weight
accuracy across the four checked factual fields **fell by 0.80 percentage points**;
degree accuracy fell by 3.45 points. Every predeclared continuation gate failed.

Keep the original 270M decoder as the reference. Preserve this checkpoint and its
local reports as a negative pilot; do not promote it, present it as a factual
improvement, or use it as evidence that `D_text(Z')` works. No counterfactual
training or evaluation was started by this run.

This is a **development/tune-set** result. The 80 tune IDs used for checkpoint
selection are included in the final 400-ID comparison, and earlier project
analysis had already viewed the original audit. The interval below describes
this comparison; it is not a fresh confirmatory confidence interval.

## Question, data, and execution

The question was whether the existing frozen 270M inverter's draft CV could be
improved by one 270M correction model seeing the target vector `Z`, the draft
text, and the frozen EmbeddingGemma embedding of that draft. The original CV was
a training target only, never an inference input. No encoder, `g`, `h_Z`, or
existing decoder checkpoint was changed.

- Run ID: `cv-correction-pilot-20261016-1711`; one seed (42) and one tested
  correction configuration. It used the existing 3,200 factual fit IDs and 400
  factual tune IDs; 80 fixed tune IDs governed epoch selection.
- Preflight passed. All 3,600 factual fit/tune draft CVs were generated and
  cached. Six training epochs (600 effective-batch updates) completed; epoch 2
  had the best predeclared selection score, followed by four non-improving
  epochs and early stopping. The best checkpoint reloaded and verified.
- Final evaluation completed on 2026-10-17 at 13:15 CEST. It compared zero
  corrections with one correction on the **same 400 tune IDs and draft texts**,
  using deterministic greedy generation. Neither audit nor official-test
  content was opened. The result flags confirm no counterfactual vector or text
  was used in this pilot.
- Fit token NLL fell from 0.83109 at epoch 1 to 0.55239 at epoch 6, while
  tune NLL reached its minimum (0.92059) at epoch 2 and rose to 0.96592 at
  epoch 6. This is consistent with worsening validation generalization after
  epoch 2; it does not establish the cause of the final reconstruction result.

## Full tune comparison

| Reconstruction measure | Original 270M draft | One correction | Direction |
|---|---:|---:|---|
| Mean normalized character-edit similarity | 0.32939 | 0.33311 | +0.00372 |
| Aligned word-token accuracy | 0.21433 | 0.21784 | +0.00351 |
| ROUGE-L | 0.31233 | 0.31631 | +0.00397 |
| chrF (0–1 scale) | 0.57035 | 0.56735 | -0.00300 |
| Raw / whitespace-normalized exact CV matches | 0 / 400 | 0 / 400 | No change |
| EOS termination | 400 / 400 | 400 / 400 | No change |
| Generation-limit termination | 0 / 400 | 0 / 400 | No change |
| Mean generated words (reference: 350.69) | 338.01 | 336.54 | -1.46 |

Character similarity is a lexical measure, **not** the percentage of CV facts
recovered. Across the 400 pairs, the corrected text scored higher on that
measure for 50.75%, lower for 49.25%, and identically for none. All corrected
outputs were distinct and none was an exact fit-CV copy; those observations do
not rule out invented details or partial memorization. Fit-only nearest-neighbor
retrieval, a non-generative comparator, scored 0.31735 character similarity on
these IDs. The correction comparison's predeclared gate is against the original
decoder, not against retrieval.

### Conservative independent factual checks

The same text-extraction rules scored both outputs. Accuracy is conditional on
the reference being scorable; missing/unparseable generated values count as
wrong. The four-field macro is an **equal-weight mean of field accuracies**, not
the share of CVs with every fact correct. Region is excluded from the macro to
avoid counting country/region evidence twice.

| Field | Scorable references | Original | Corrected | Change, percentage points |
|---|---:|---:|---:|---:|
| Country | 397 / 400 | 95.21% | 94.46% | -0.76 |
| Degree | 377 / 400 | 93.10% | 89.66% | -3.45 |
| University tier | 400 / 400 | 70.50% | 71.50% | +1.00 |
| Project count | 313 / 400 | 15.02% | 15.02% | 0.00 |
| Four-field macro | Different denominators | 68.46% | 67.66% | -0.80 |

Region was 97.23% → 96.73% on 397 scorable references. Unparseable generated
values increased from 4 to 8 for country, 18 to 26 for degree, 6 to 7 for tier,
and 57 to 76 for project count. Project count remained particularly unreliable
even though its measured accuracy did not change. These four rules do not score
occupation, behavioral episodes, unsupported qualifications, or prose quality.
The report retains 20 fixed-order reference/reconstruction pairs and a rubric,
but no systematic human review was completed.

### Conditioning and cycle diagnostics

Shuffling **only the corrector's target vector** while holding each original
draft and its embedding fixed reduced character similarity from 0.33311 to
0.29022 and raised token NLL from 0.91606 to 1.21661. Thus this trained
corrector responds to the target vector (gaps +0.04289 similarity and +0.30056
nats/token), but the response did not translate into an adequate net improvement
over the already vector-conditioned original decoder. The control is not a test
of dependence of the whole pipeline on `Z`.

Re-encoding corrected outputs gave mean cycle cosine 0.95252 and valid fraction
1.0. High cycle similarity cannot establish exact text recovery or independent
factual fidelity; the low project-count accuracy and zero exact matches show
why those outcomes must be reported separately.

## Predeclared gates and interpretation

| Continuation requirement | Observed | Decision |
|---|---:|---|
| Mean character-similarity gain ≥ 0.02 | +0.00372 | Fail |
| Positive lower bound of paired-bootstrap 95% interval | -0.00167 (upper +0.00919) | Fail |
| Four-field macro factual gain ≥ 0.02 | -0.00801 | Fail |
| No factual field regresses by more than 0.02 | Degree -0.03448 | Fail |

The character interval used 2,000 paired resamples and seed `20261016`. It is
**descriptive** because model choices and earlier analyses used development
information; it must not be reported as an untouched-test significance result.
The run recorded `worth_followup: false`, `promoted: false`, and
`counterfactual_training_started: false`.

The selected epoch uses both text similarity and the four factual checks, yet
the final comparison still fails. The result rules out a useful gain for **this
one-seed, one-pass, factual-only correction configuration**, not every possible
corrector or the possibility that additional information can be extracted from
the vector. Fit drafts were generated by a decoder trained on those fit CVs,
whereas tune drafts were out of sample; that unmeasured draft-distribution gap
is a limitation, not an established explanation for failure. The observed
post-epoch-2 tune-loss rise is another warning against simply continuing this
run longer.

## Implication for a counterfactual-exposed decoder

The failed factual-only correction gate does **not** authorize an automatic
counterfactual-trained twin of this model. However, a separately designed
`D_text` comparison trained on aligned `(Z', X')` pairs remains scientifically
reasonable if the goal is to measure what *privileged counterfactual text
supervision* buys over the factual-only decoder. [LIBERTy](https://arxiv.org/abs/2601.10700)
provides structural counterfactual text pairs in the benchmark setting; paired
availability must still be verified for the exact local split and intervention.

That would be a different, explicitly labelled **oracle-supervised decoder
baseline**, not the deployable `Z`-only method and not a change to `h_Z` training.
Keep all worlds of each person in one split, hold the frozen encoder and existing
`h_Z` fixed, and compare on untouched units with independent factual and
intervention/preservation checks. Separate oracle `Z' = f(X')` from `h_Z`-generated
vectors at evaluation: success on the former does not guarantee robustness to
the latter or recover details not encoded in either vector. No such experiment
was initiated here. The appropriate next decision is to verify the availability
and ID alignment of paired training `(Z', X')`, then predeclare a small separate
pilot against the unchanged factual-only decoder. Merely repeating this failed
corrector with privileged inputs is not justified by its result.

**Follow-up (updated 2026-10-19):** the aligned-pair check passed, and the
separate [counterfactual-exposed decoder pilot](text_realizer_counterfactual_exposure_plan.md)
specifies the comparison and gates. Its two 600-update arms completed and
reload-verified on 2026-10-17; the tune-only comparison subsequently missed its
predeclared replication gate. No
counterfactual result should be inferred from this failed corrector or from the
new arms' training completion alone.

## Artifact provenance and preservation

The archived report directory is now
`.archive/artifacts/reports/talent/embeddinggemma_768/text-corrector/cv-correction-pilot-20261016-1711/`.
Its `plan.json` and `source/` preserve the immutable inputs/configuration and
source snapshot; `preflight.json`, `training.json`, `history.json`, `verify.json`,
and `result.json` preserve execution and evaluation. JSON reports, including
per-ID CV text, are now tracked at these paths; the `source/` snapshot remains
local. Publishing these historical results does not promote the corrector.

| Artifact | Local path / integrity marker |
|---|---|
| Run identity | `a9914996550716385b785690f6c1218916e14755b21a2044b6c2f8a2b5c5ddd1` |
| Selected corrector bundle | `.archive/artifacts/models/talent/embeddinggemma_768/text-corrector/cv-correction-pilot-20261016-1711/best.pt`; SHA256 `8ff75fd2c682904607147ff590cb156e3a1d90040f2b38d1a00efef426175bc1` |
| Full local report | `.archive/artifacts/reports/talent/embeddinggemma_768/text-corrector/cv-correction-pilot-20261016-1711/result.json`; SHA256 `f85eedf8bc7b3513665b3b43db5e23a3e7a306d908e6168e4ef05d2b4c3a46e8` |

This Markdown summary is the portable result; weights and the full report remain
local ignored artifacts unless intentionally transferred through a separate,
approved workflow. Archiving did not promote the corrector or upload its weights.
