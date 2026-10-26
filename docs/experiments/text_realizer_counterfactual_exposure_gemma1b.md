# Oracle exposure with the Gemma 3 1B text decoder

**Status: completed on 2026-10-22; the predeclared replication criteria failed.**
See [completed results, interpretation, and artifact checksums](#completed-result-and-decision-2026-10-22).
The protocol below records the original declaration; it is not a new launch request.

Declared on 2026-10-21, before the new continuations. This is a separate,
exploratory comparison using privileged oracle supervision for `D_text`.
The [270M protocol](text_realizer_counterfactual_exposure_plan.md) and its
completed result supply the recipe. The [factual-only 1B result](text_realizer_gemma1b_results.md)
supplies the starting checkpoint. Neither result establishes faithful inversion.

## Question and comparison

Does oracle exposure improve the 1B decoder more than an equal-budget factual
continuation? Three 1B arms are evaluated on the same tune IDs:

| Arm | Starting weights | New training |
|---|---|---|
| A / original | Selected factual-only 1B, seed 42, epoch 5 | None |
| B / factual | Same checkpoint | 600 factual-only updates |
| C / oracle | Same checkpoint | 600 updates, each with 16 factual and 16 changed oracle examples |

The frozen EmbeddingGemma encoder still produces 768-D vectors. No new encodings,
`g` training, or `h_Z` training is part of this experiment. All three arms receive
true oracle vectors during counterfactual evaluation. Performance on predicted
`h_Z` vectors remains an independent question.

The input checkpoint is
`.archive/artifacts/models/talent/embeddinggemma_768/text-realizer/cv-text-1b-pilot-20261015-1026.lora-compact.seed-42/best.pt`,
SHA-256 `59af957c6fdf29a86154efb40d66bbde4038ad82f5818a7a3afa0fcd133d84af`.
The base model is `google/gemma-3-1b-pt`, revision
`fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`, with eager attention. Both new arms
train the existing 16-token compact projector and rank-16 LoRA adapters. Base
weights remain frozen. Precision is FP32, microbatch 1 with effective batch 32,
and gradient checkpointing is enabled. Optimizer, rates, schedule, selection,
and decision thresholds match the 270M continuation protocol.

## Data, selection, and predeclared decision

The same 3,200 fit units supply factual examples and 2,379 changed counterfactuals.
The existing 400 tune units are divided exactly as in the 270M comparison:
the first 80 provide factual-only checkpoint selection every 100 updates; the
other 320 provide final factual and oracle evaluation, with the 238 changed units
as primary (82 units are identities). Identity counterfactuals are not duplicated in the oracle training
half. Unit IDs and all input hashes are pinned in the new configuration and
prepared manifest. Audit and official-test partitions stay closed.

Checkpoint selection uses the same average of factual character similarity and
four-field factual accuracy, followed by the existing deterministic tie breaks.
C must gain at least 0.02 character similarity over both A and B, with each
paired-bootstrap lower 95% bound positive. It must also gain at least 0.02
macro factual accuracy over A, pass each field's 0.02 regression tolerance, and
retain factual reconstruction within the original tolerances. A failure stops
this pilot; there is no automatic seed replication or expanded search.

This is development evaluation informed by earlier decoder results. A positive
result would justify a separately planned follow-up, not a fresh confirmatory
claim. Comparisons with the stored 270M results are descriptive; the primary
causal contrast here is C versus the equal-update 1B arm B.

## Implementation and operation

Configuration: [`text_realizer_counterfactual_exposure_gemma1b.yaml`](../../configs/text_realizer_counterfactual_exposure_gemma1b.yaml).
Shared code accepts the two explicitly pinned model profiles; the default
remains the existing 270M protocol. The 1B profile requires separate
`text-realizer-cf-1b` model/report and `text-realizer-cf-1b-prepared` data folders.
Old checkpoints and all original results are retained.

The controller verifies inputs and cached models, runs the GPU preflight, then
starts one independent worker per selected GPU. Preflight checks complete
targets, prefix/cache behavior, gradients, frozen weights, a 20-update smoke,
resume equivalence, ten full-batch timing updates, checkpoint reload, and joint
decoder/cycle-encoder memory with at least 15% headroom. Four fit examples supply
generation/NLL/cycle timings; no evaluation targets enter the benchmark.

```bash
# Preflight, both 600-update arms, and final comparison; detached under tmux.
bash scripts/run_text_realizer_cf_1b.sh RUN_ID 0 1 --train-after-preflight

# Without --train-after-preflight, stop after the readiness checks.
# To resume the same run after interruption, append --resume.
```

The controller's immutable plan records config, input/source hashes, environment,
and physical GPU assignments. Per-arm checkpoints preserve optimizer, scheduler,
RNG, sampler, and checkpoint-selection history. Each arm reload-verifies its
selected checkpoint before final comparison. The common plan publication is
serialized across the two workers. Each worker holds its own GPU lease; other
GPU jobs are never terminated.

`controller/status.json` and `controller/logs/` record orchestration;
`factual/status.json` and `oracle/status.json` record training. The final
`result.json` contains per-ID scores and the predeclared decision; `examples.json`
contains the first 20 comparison examples. `eta.json` includes optimizer work,
all six selection rounds, and all six final model/world evaluations, with a 25%
planning margin. Training uses 25-update recovery checkpoints.

No checkpoint, optimizer state, prepared CV text, or generated report is uploaded
by this workflow. A run with changed code/config/inputs needs a new run ID.

## Execution record: 2026-10-21

Run `cv-cf-1b-pilot-20261021-0938` was launched at approximately 09:38 CEST
in tmux session `text-realizer-cf1b-cv-cf-1b-pilot-20261021-0938`.
The fresh preflight passed, and both continuation workers started at about
09:47 CEST: factual on physical GPU 0 and oracle-exposed on physical GPU 1.
Both cards are RTX 2080 Ti with 11 GiB VRAM. At 09:49 CEST, both workers had
completed gradient updates with finite losses. That launch-time check was
operational evidence only; the completed scientific comparison is recorded below.

Validation before launch:

- The focused decoder, encoder, LangVAE, and configuration suite passed:
  125 tests passed, with three opt-in integration tests skipped. The actual
  GPU preflight then passed its model-specific checks separately.
- Input hashes, the factual-only starting checkpoint, and both offline model
  caches were verified. Prepared split IDs match the 270M protocol exactly.
- Complete targets contain 344–486 tokens including EOS, with no tokenizer
  round-trip discrepancies in the prepared training/selection targets.
- Smoke loss fell from 0.636 to 0.0071 nats/token on the tiny repeated training
  sample. This checks trainability only; it is not held-out performance.
- Checkpoint reload and interrupted-run continuation checks passed.
- Microbatch 1 passed the memory gate. Peak reserved memory was about 6.23 GiB
  for the training preflight and 6.48 GiB for joint decoder/cycle inference.

Ten full effective-batch updates averaged 24.35 seconds per update. Four fit
generations averaged 20.38 seconds per CV. Including all checkpoint selections
and the 1,920 final comparison generations, the measured estimate was about
18 hours, or 22.4 hours with the 25% planning margin. The recorded planning
completion is **2026-10-22 at 08:10 CEST**; checking around 10:00 CEST leaves
additional allowance. These estimates depend on actual generation lengths and
GPU load, and are not deadlines. Final evaluation starts automatically only
after both training workers finish and their checkpoints pass reload checks.

Machine-readable progress and the timing assumptions are under
`reports/talent/embeddinggemma_768/text-realizer-cf-1b/cv-cf-1b-pilot-20261021-0938/`:
`controller/status.json`, `eta.json`, and the per-arm `status.json` files.
Artifacts are separate from the retained 270M experiments and the original
1B checkpoint. No audit, official-test, or predicted-`h_Z` evaluation is opened
by this run.

## Completed result and decision: 2026-10-22

**Completed; do not advance under this pilot's predeclared rule.** Oracle
exposure improved the original 1B model's measured factual accuracy and character
similarity, but provided only a small, uncertain character-similarity gain over
the equal-update factual continuation. It did not overtake the earlier 270M
oracle-exposed comparator on the aggregate metrics. The saved decision is
`worth_replication: false`, not a training failure.

### Execution and checkpoint selection

Both workers completed all 600 updates. On 2026-10-21, arm C finished at
16:18 CEST and arm B at 16:22 CEST. Checkpoint selection used only the predeclared
80 factual tune units: B selected update **300**, and C selected update **400**.
The final evaluation therefore uses these selected checkpoints, not the
600-update last checkpoints. Both passed fresh reload verification, and their
file hashes were checked again while documenting the result.

All six model/world evaluations (three arms, two worlds, 320 units each)
completed by **2026-10-22 at 02:41 CEST**. The controller recorded completion
at 02:41:42 CEST. Total elapsed time from the approximately 09:38 CEST launch
was about **17 hours 4 minutes**, earlier than the conservative planning ETA
recorded above. The ETA file is historical benchmark output, not a current
prediction.

### Primary comparison: 238 changed tune units

All models receive true oracle Z' vectors and are scored against the aligned X'.
The remaining 82 identity units are excluded from this primary comparison.
The 1B and earlier 270M reports have identical 320-unit evaluation ID sequences;
both protocols identify 238 changed and 82 identity units.

| Decoder | Character-edit similarity | Four-field macro factual accuracy |
|---|---:|---:|
| A: original 1B | 0.31515 | 63.66% |
| B: factual-only 1B continuation | 0.32990 | 65.78% |
| C: oracle-exposed 1B continuation | 0.33440 | 69.55% |
| Earlier 270M oracle-exposed comparator | 0.34563 | 72.38% |

The four-field macro is an unweighted mean of conservative text-extraction
accuracies for country, degree, university tier, and project count. It is not
the fraction of wholly faithful CVs, and it is not a score from `g`.

### Predeclared gates

Paired-bootstrap intervals use the stored 2,000 resamples and seed 20261017:

| Character-similarity contrast | Mean gain | 95% interval | Predeclared character gate |
|---|---:|---|---|
| C minus A | +0.01925 | [0.01196, 0.02634] | Fail |
| C minus B | +0.00450 | [-0.00279, 0.01223] | Fail |

Each character gate requires a mean gain of at least **+0.02** and a positive
lower interval bound. C minus A has a positive interval but misses the minimum
mean improvement; C minus B misses the minimum and its interval includes zero.
The second result does **not** prove that oracle exposure has no benefit; it
does not establish the required extra lexical-reconstruction benefit here.

All four other gates passed: C's macro factual gain over A
(+0.05896),
no per-field regression beyond 0.02 versus A, factual character retention,
and factual macro retention. The observed macro gain over B was
+0.03771;
no confidence interval for that macro gain was predeclared or computed.
Because both character gates failed, the combined decision remains negative.

### Independent factual checks and extraction coverage

These accuracies use only scorable reference values. An unparseable generated
value is a failure when its reference is scorable; ambiguous reference values
are excluded. The denominators below are shared by all listed models.

| Field | Scorable reference coverage | 1B A | 1B B | 1B C | 270M oracle-exposed |
|---|---:|---:|---:|---:|---:|
| Country | 235/238 (98.74%) | 97.87% | 97.87% | 99.15% | 95.32% |
| Degree | 232/238 (97.48%) | 90.95% | 97.41% | 97.84% | 96.12% |
| University tier | 238/238 (100.00%) | 58.82% | 55.04% | 62.61% | 70.17% |
| Project count | 172/238 (72.27%) | 6.98% | 12.79% | 18.60% | 27.91% |

Project-count accuracy remains only **18.60%** for C on its 172 scorable
references (32 correct); 50 predictions in that scorable subset were
unparseable. University-tier accuracy is 62.61%. Although C improves both
relative to A, these remain substantial fidelity failures. Relative to 270M,
the 1B oracle-exposed model is better on country and degree but worse on
university tier and project count; it is not uniformly worse on every field.

### Factual retention: all 320 comparison units

This table uses factual Z and X, not the 238-unit changed-oracle subset:

| 1B decoder | Factual character-edit similarity | Factual four-field macro accuracy |
|---|---:|---:|
| A: original 1B | 0.30852 | 63.73% |
| B: factual-only continuation | 0.33123 | 66.17% |
| C: oracle-exposed continuation | 0.33618 | 66.80% |

C improves both factual aggregate metrics over A, so the observed oracle gains
did not require a regression in these predeclared factual-retention metrics.
That does not establish retention of every unmeasured CV detail.

### Additional generation diagnostics: all 320 oracle-world units

For C, mean sentence-level chrF is
0.58179, aligned token accuracy is
22.62%, and mean
frozen-encoder cycle cosine is
0.95655.
Its teacher-forced token NLL is 0.70915
nats/token, versus 0.78185 for A and
0.81460 for B.
These are **all-320** diagnostics and must not be relabelled as changed-only
scores. chrF is on a 0–1 scale and is an average of sentence scores, not corpus
chrF.

Across all three arms and both evaluation worlds, raw and whitespace-normalized
exact reconstruction rates were zero; every generated CV ended with EOS;
no generation hit the 1,024-token limit; output uniqueness was 100%; and no
normalized output exactly copied a fit CV. These checks show that the run
completed normally, not that its outputs were faithful or free of memorization.
For C's oracle outputs, 72.81%
share at least 20 consecutive words with the nearest lexical fit CV under the
implemented diagnostic. Common templates can explain shared spans; the report
does not identify their cause.

### Interpretation and limits

- Extra factual training already improves A's changed-oracle character score
  by +0.01475.
  The larger C-minus-A gain cannot all be attributed to counterfactual exposure.
  C-minus-B is the relevant equal-update contrast, and its lexical gain remains
  small and uncertain.
- Better token NLL and high cycle cosine coexist with zero exact recovery and
  poor project-count/tier fidelity. Fluent or embedding-consistent text is not
  sufficient evidence of supported, candidate-specific details.
- The 270M comparison is descriptive, not a clean causal estimate of parameter
  count: the models start from different checkpoints and training histories.
  This is one seed under one fixed adapter/projector recipe, not evidence that
  every larger decoder must perform worse.
- A limitation in expressing detailed information from the vector remains a
  plausible explanation, but this experiment does not isolate that mechanism
  or prove the missing facts are absent from the frozen embeddings.
- This is exploratory development evaluation, not fresh confirmatory evidence.
  The original 1B checkpoint had already been selected using the tune pool.
  True oracle Z' is used here; **no predicted h_Z vectors, new factual audit,
  or official-test evaluation were evaluated by this pilot**. No
  counterfactual-fairness claim follows from these decoder metrics.

Retain this as a completed, negative advancement result. The original factual
270M decoder and the separate oracle-exposed 270M comparator remain unchanged;
neither is thereby certified as a faithful inverter. Preserve the selected 1B
checkpoints and reports for provenance, but do not promote this run or
automatically launch seeds 43/44, longer tuning, or new evaluation gates.
EmbeddingGemma, canonical vectors, `g`, and `h_Z` remain unchanged.

### Saved evidence and integrity

The full [result JSON](../../reports/talent/embeddinggemma_768/text-realizer-cf-1b/cv-cf-1b-pilot-20261021-0938/result.json)
contains per-ID generations/metrics, factual checks, grouping diagnostics,
checkpoint identities, intervals, and decision flags.
The [first 20 comparison examples](../../reports/talent/embeddinggemma_768/text-realizer-cf-1b/cv-cf-1b-pilot-20261021-0938/examples.json)
contain aligned oracle references and A/B/C reconstructions.
The `factual/` and `oracle/` subfolders contain training histories,
selection records, and reload checks. Full generated CV text is preserved.

Selected checkpoint files are under
`models/talent/embeddinggemma_768/text-realizer-cf-1b/cv-cf-1b-pilot-20261021-0938/`:

| Artifact | SHA-256 |
|---|---|
| Final `result.json` | `9957c15902b101bcd13672054dc53f6cc08ff3f30bab75c7e8df7b7e9eac81a5` |
| B: `factual/best.pt` (update 300) | `568769bdaafe36ad4ed12e13bc53808e5b4da1c5e878cd85ee5e18c43647b275` |
| C: `oracle/best.pt` (update 400) | `32561b70c9fd26e31df84511273d5cf11e722cd3a6ad04befeba0b8c9390d2e2` |

The [2026-10-22 report manifest](../../reports/talent/embeddinggemma_768/text-decoder-results/publication-20261022-cf1b.json)
records the 17 finished report/metadata files, including the five plans/ETA
files already present in the initial publication. The 12 newly completed
reports are now eligible for Git tracking; weights, optimizer states, logs,
and source snapshots remain excluded. This documentation update does not
stage, commit, or push them automatically.

Verify the final report set from the repository root:

```bash
sha256sum -c reports/talent/embeddinggemma_768/text-decoder-results/SHA256SUMS-cf1b-20261022
```

