# Portable package counterfactual CV evaluation

This workflow evaluates the four **factual-only `talent_motivation` decoders**
on the same 1,000 official test units. For each of 256/768 dimensions and
Gemma 270M/1B, it generates a CV from the oracle counterfactual vector and
from the package editor's predicted counterfactual vector: **8,000 generated
CVs in total**. The reference is the dataset's counterfactual CV.

There are three commands: `prepare`, `generate`, `score`, under
`python -m exp.text_realizer_package_eval`. Preparation runs once on CPU;
generation workers can run on separate clusters; scoring runs on CPU after
collecting their outputs. Nothing is trained, re-encoded, submitted to a
scheduler, staged in Git or uploaded automatically.

## Fixed inputs and interpretation

`configs/text_realizer_package_eval.yaml` pins the input files, reader/flow
fits, their package reports and all four decoder exports by SHA-256. The
editor is S1 flow with predicted reader states, fixed seed 0, and argmax of
the mean probabilities from the five reader folds. The SCM and literal
partition order reproduce the package experiment's intervention draws.
Selection uses the existing package latent-distance benchmark, not generated
CV quality; it is not a new selection on an untouched test set.

| Dimension | Provided package pipeline | Decoder runs |
| --- | --- | --- |
| 256 | 24 October: 1,000 reader labels, no flow fidelity term | `package-256-20261027-0934-eager`, `package-256-1b-20261027-0958` |
| 768 | 27 October: 2,000 reader labels, flow fidelity 0.1 | `package-768-270m-20261027-1551`, `package-768-1b-20261027-1551` |

The data are the current package corpus at both dimensions. The 256-D fit
is the provided earlier protocol, **not a refit under the newer 768-D
settings**. Comparisons between dimensions therefore also include this
protocol difference. To study dimension alone, matched reader/flow fits
would be needed. The two decoder sizes within each dimension receive
identical vectors.

Only canonical 256-D embeddings are constructed by normalizing the first
256 coordinates of the cached 768-D vector, exactly as in the package.
Predicted editor outputs are passed to the decoder unchanged. The decoder
does not receive the original CV, reference text or SCM states.

## 1. Prepare once where the original artifacts exist

Run from the repository root. The setup scripts require `uv` on PATH or the
existing local `.cache/text-realizer-tools/uv`. They use a persistent Python
3.12 installation and frozen lockfiles. The supported setup is Linux
x86_64; GPU inference additionally needs a driver compatible with the pinned
PyTorch CUDA 12.4 build. CPU inference needs no GPU driver.

```bash
bash scripts/setup_text_realizer_package_eval.sh export
exp/text_realizer_package_eval/environment/.venv/bin/python \
  -m exp.text_realizer_package_eval prepare --allow-oracle-evaluation
```

The separate export environment supplies pandas 3 for the package reader
pickles and CPU PyTorch. It does not replace the root environment or the
decoder's pandas 2 environment. Preparation loads only the pinned trusted
package checkpoints; it never invokes a cache builder that could refit a model.

The explicit flag opens the official test references and oracle vectors for
this evaluation. Preparation checks ID joins, encoder protocols, hashes,
decoder reload manifests, matching training histories, and reproduction of
the package's reported seed-0 latent metrics. It refuses mismatches.

The resulting bundle is:

```text
models/talent_motivation/text-realizer-evaluation/package-v1/
  manifest.json
  references.jsonl
  vectors-256.pt
  vectors-768.pt
  states-256.json
  states-768.json
  checkpoints/{256-270m,256-1b,768-270m,768-1b}.pt
```

It contains IDs, original/reference CVs, both vector conditions, state and
intervention provenance, and the four inference checkpoints. It is about
331 MB, with no training corpus or optimizer states. It excludes the shared
pretrained Gemma bases. Repeating preparation verifies the existing bundle;
changed sources or inputs require a new `--bundle` directory.

## 2. Transfer and generate

On another cluster, use the same repository source and copy the whole bundle,
for example:

```bash
rsync -a models/talent_motivation/text-realizer-evaluation/package-v1/ \
  CLUSTER:/path/to/latent_intervention/models/talent_motivation/text-realizer-evaluation/package-v1/
```

Do not copy virtual environments. On the target, set up the existing pinned
decoder runtime:

```bash
bash scripts/setup_text_realizer_package_eval.sh decode
```

The account downloading the Gemma bases must already have access to the
models. If authentication is needed, use the local `hf auth login` command;
do not put tokens in scripts or Git. Cache both exact base revisions on a
node with network access:

```bash
bash scripts/setup_text_realizer_package_eval.sh decode --download-models
```

Alternatively, copy the existing cache directories
`.cache/huggingface/hub/models--google--gemma-3-270m/` and
`.cache/huggingface/hub/models--google--gemma-3-1b-pt/` with `rsync -a`.
Include their `blobs` and `snapshots` directories so symlinks remain valid.
Only these two bases and their tokenizers are needed; generation does not
load EmbeddingGemma, pandas 3, the reader/flow fits or the original CSV files.

Run one worker per allocated GPU. On a scheduler that exposes one GPU per
job, use `cuda:0` in every job. With four explicitly allocated visible GPUs
on one machine, launch each command in its own job/session:

```bash
bash scripts/run_text_realizer_package_eval.sh 256-270m cuda:0
bash scripts/run_text_realizer_package_eval.sh 256-1b   cuda:1
bash scripts/run_text_realizer_package_eval.sh 768-270m cuda:2
bash scripts/run_text_realizer_package_eval.sh 768-1b   cuda:3
```

These commands run in the foreground; the script does not select a free GPU
or detach itself. With fewer GPUs, schedule the commands sequentially as
resources become available. CPU execution is also supported:

```bash
bash scripts/run_text_realizer_package_eval.sh 256-270m cpu --threads 4
```

Both oracle and predicted conditions run by default. Each uses the checkpoint's
unchanged FP32, greedy, batch-one generation with at most 1,024 new tokens.
For a short initial timing check, add `--limit 5`: this generates five **new**
CVs per condition and stops. Repeat the command without `--limit` to continue.
Do not tune the decoder or generation settings using these test outputs.

Paths can be moved with `--bundle /path/to/bundle`, `--output /path/to/predictions`
and `--cache-dir /path/to/hub`. Workers may write to separate local directories;
merge their output trees before scoring. To divide a single model across two
workers, add `--num-shards 2 --shard-index 0` to one and `--num-shards 2
--shard-index 1` to the other. Keep the same shard count when resuming.
Use the same source, package versions and CPU/CUDA backend within a condition;
bitwise equality across different hardware is not promised.

Each complete CV is flushed to JSONL. Repeating the same command resumes at
the next ID, discarding only an interrupted final line. Each shard also has
metadata, a progress file with seconds/CV and estimated remaining seconds,
and a completion checksum. Stop/restart needs no optimizer or training state.

The default output tree is:

```text
reports/talent_motivation/text-realizer-evaluation/package-v1/
  predictions/<model>/<oracle|predicted>/part-000-of-001.*
```

Previous development runs on RTX 2080 Ti cards suggest roughly 14–18 hours
for all four workers in parallel, excluding queue/setup time. This is a
scheduling estimate, not a benchmark of the full test run; use the workers'
measured progress once generation starts. CPU execution is supported but
has not been timed for this workload.

## 3. Collect and score

Merge the workers' prediction directories, preserving model/condition names,
then run in the decoder environment:

```bash
exp/text_realizer/environment/.venv/bin/python \
  -m exp.text_realizer_package_eval score
```

Use `--bundle`, `--predictions` and `--output` for nondefault locations.
Scoring refuses missing/duplicate IDs, incomplete shards, changed prediction
files or mismatched provenance. It produces:

- `summary.csv` and `summary.json`: eight model/condition rows, text similarity,
  exact match, aligned token accuracy, ROUGE-L, chrF, lengths/termination, and
  independent text-based country/region, degree, university-tier and project-count
  checks with extraction coverage. Unparseable predictions fail where the
  reference is scorable.
- `comparisons.jsonl`: every ID's original CV, oracle counterfactual CV and
  all eight generated outputs.
- `examples.md`: the first five IDs in fixed package order, with a human-review
  rubric. They are not selected by output quality.

This minimal evaluation does not add cycle encoding or a learned factual
judge. Oracle-vector reconstruction measures decoder limitations;
predicted-vector reconstruction also includes reader/editor errors. Neither
text similarity nor these factual checks alone establish counterfactual fairness.

## Verified setup on 28 October 2026

The local bundle contains all 1,000 aligned units and all four frozen decoder
exports. Replaying the fixed package pipelines reproduces their seed-0 metrics:

| Dimension | Oracle cosine | Scaled latent distance |
| --- | ---: | ---: |
| 256 | 0.93874198 | 0.65117025 |
| 768 | 0.92162675 | 0.66275913 |

These are latent replay checks, not generated-text results. Repeated preparation
verified the bundle without recomputing it. All four real decoder exports loaded
on CPU and generated two tokens from synthetic vectors; no test CVs were
generated for that check. Evidence is in the local report tree's
`setup_validation.json`.

187 offline tests passed, including new alignment, oracle-gate, source-free
generation, edited-vector preservation, transfer/resume, sharding and scoring
tests, plus the existing package decoder, encoder, LangVAE and configuration
tests. Three real-GPU tests were deselected. Lint and shell syntax checks passed.
The complete inference run was not launched during setup; the subsequent
four-GPU launch is recorded below.

Generated bundle/report trees are excluded only in this checkout's
`.git/info/exclude`; the repository `.gitignore` is unchanged. Transfer model
artifacts separately from source code. The portable bundle includes CV texts.

## First full evaluation launch

On **28 October 2026 at 11:58 Europe/Berlin**, all four RTX 2080 Ti GPUs
were verified idle (1 MiB allocated each, 0% utilization, no compute processes),
and preparation reverified the existing bundle. The full 8,000-CV evaluation
was then launched with this allocation:

| Physical GPU | Decoder | Conditions |
| --- | --- | --- |
| 0 | 256-270m | Oracle and predicted vectors, 1,000 units each |
| 1 | 256-1b | Oracle and predicted vectors, 1,000 units each |
| 2 | 768-270m | Oracle and predicted vectors, 1,000 units each |
| 3 | 768-1b | Oracle and predicted vectors, 1,000 units each |

The local detached session is
`text-realizer-eval-launch-20261028-115728`. Its plan, controller, per-model
logs and live `status.json` are under
`reports/talent_motivation/text-realizer-evaluation/package-v1/launch-20261028-115728/`.
Workers select GPUs by UUID, hold the existing GPU leases and use the pinned
offline model cache. Generation resumes into the default prediction tree.
The controller runs CPU scoring automatically after all four workers succeed;
it records a failure instead if any worker exits unsuccessfully.

This is a launch record, not a completed-results claim. Each prediction shard's
`.progress.json` reports measured throughput; `scores/summary.json` is written
only after all eight model/condition cells are complete.

All four workers produced their first CVs successfully. Initial oracle timing
was approximately 14–16 seconds/CV for 270M and 18–22 seconds/CV for 1B.
Allow 14–18 hours for generation and scoring (approximately 02:00–06:00
Europe/Berlin on 29 October); this is preliminary, especially because later
predicted-vector outputs may have different lengths. The launch directory's
`initial_estimate.json` records the early measurements and scheduling budget.

## Completed results

The run completed successfully on **28 October 2026 at 22:42:52 Europe/Berlin**,
including scoring, after about 10 hours 45 minutes. All four workers and the
scorer exited with code zero. A subsequent check on 29 October verified all
8,000 generated CVs, all 1,000 ID-aligned comparison records, the bundle hash
and all 24 recorded prediction-artifact checksums. All GPUs were idle again.

The complete machine-readable results are in
`reports/talent_motivation/text-realizer-evaluation/package-v1/scores/`:
`summary.json`, `summary.csv`, `comparisons.jsonl`, and `examples.md`.
The examples are the first five IDs in the fixed package order.

| Decoder | Vector source | Character similarity | chrF (0–1) | Country accuracy | Degree accuracy | Project-count accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 256-270M | Oracle | 0.3209 | 0.5552 | 92.5% | 94.7% | 25.3% |
| 256-270M | Predicted | 0.3133 | 0.5466 | 22.6% | 64.7% | 25.4% |
| 256-1B | Oracle | 0.3075 | 0.5473 | 95.4% | 54.9% | 14.3% |
| 256-1B | Predicted | 0.3069 | 0.5449 | 24.9% | 51.3% | 15.4% |
| 768-270M | Oracle | 0.3185 | 0.5555 | 94.3% | 94.9% | 14.6% |
| 768-270M | Predicted | 0.3120 | 0.5482 | 20.9% | 65.3% | 15.4% |
| 768-1B | Oracle | 0.3104 | 0.5532 | 97.9% | 89.3% | 15.6% |
| 768-1B | Predicted | 0.3077 | 0.5494 | 18.9% | 62.3% | 15.7% |

Each row covers 1,000 test CVs. Fact accuracies use the conservative text
extractor and scorable references: 988 for country, 982 for degree and 870
for project count. Unparseable generated values count as failures. All outputs
terminated with EOS; none hit the length limit. Raw and whitespace-normalized
exact reconstruction rates were zero in all eight cells.

The 270M models have slightly higher character similarity than the 1B models
at both dimensions. Oracle-vector reconstruction already misses many details,
particularly project count. With predicted vectors, country accuracy falls
sharply while aggregate lexical similarity changes little. This is evidence
that lexical similarity alone is inadequate for judging these counterfactual
CVs. The comparison does not isolate reader error, flow error and decoder
response to edited vectors; it does not establish counterfactual fidelity.
The differing 256-D/768-D flow protocols also prevent a dimension-only conclusion.

## BERTScore follow-up

`python -m exp.text_realizer_bertscore` scores the saved `scores/comparisons.jsonl`.
It reads all eight decoder/condition outputs, preserves candidate IDs, and
adds semantic similarity without changing any decoder, embedding, existing
score, simulation file or frozen protocol. This is a supplementary analysis
specified after inspection of the earlier results.

The metric is the reference `bert-score==0.3.13` implementation, using
`FacebookAI/roberta-large` revision
`722cf37b1afa9454edce342e7895e588b6ff1d59`, layer 17, FP32, the slow tokenizer,
uniform token weights (`idf=False`) and unrescaled precision/recall/F1.
Whitespace is collapsed consistently in both texts. The script checks the
complete input with BERTScore's RoBERTa prefix-space convention and refuses
any CV over 512 tokens including special tokens. No text is truncated or
split, and the metric itself is not reimplemented.

The scope is one small scoring module, one four-worker shell launcher, an
isolated environment/lockfile and focused synthetic tests. Existing callers,
root dependencies, configuration schemas, training code and caches need no
changes. Sentiment, emotion, AlignScore, behavioral judges and additional
baselines are outside this run.

From the repository root, with `uv` installed (on this machine its executable
is `.cache/text-realizer-tools/uv`):

```bash
uv sync --frozen --project exp/text_realizer_bertscore/environment --python 3.12
exp/text_realizer_bertscore/environment/.venv/bin/python \
  -m exp.text_realizer_bertscore download
bash scripts/run_text_realizer_bertscore.sh 0 1 2 3
```

The final command runs in the foreground; detach it with tmux or submit it
through a scheduler. It maps physical GPU indices to UUIDs, takes the existing
GPU leases and launches one decoder's 2,000 comparisons on each GPU. Each
worker checkpoints its scores every 64 pairs, using model batches of 16.
After all four workers succeed, the launcher merges their outputs automatically.
Repeating the same command resumes completed chunks. Changes to the input,
model files, settings, evaluator source or environment lock refuse resume.

For a single allocated GPU or CPU, run an individual worker and then merge
once all four decoders are complete:

```bash
exp/text_realizer_bertscore/environment/.venv/bin/python \
  -m exp.text_realizer_bertscore score --decoder 256-270m --device cuda:0
exp/text_realizer_bertscore/environment/.venv/bin/python \
  -m exp.text_realizer_bertscore merge
```

The CLI accepts `--input`, `--output` and `--cache-dir` for another cluster.
The outputs are under
`reports/talent_motivation/text-realizer-evaluation/package-v1/bertscore-roberta-large/`:
four worker JSON files with progress/ETA, `per_cv.csv` (8,000 rows), and
`summary.csv`/`summary.json` (eight means). Every result binds its input,
model files, evaluator source and environment. Logs are in `logs/`.

BERTScore measures contextual semantic similarity. Its values are not a
percentage of facts recovered and do not establish counterfactual fairness;
report them alongside the existing field-level accuracies.

The first four-GPU run was launched on **29 October 2026 at 08:06:29
Europe/Berlin**, in tmux session `cv-bertscore-20261029-080629`. Every complete
CV fits the model context (maximum 500 tokens). Initial throughput was about
0.032–0.033 seconds per pair on each GPU, with approximately 2.7 GiB allocated
per GPU; the measured finish estimate was 08:09. This supersedes the earlier
unbenchmarked planning estimate of tens of minutes.

The run completed successfully at **08:08:03 Europe/Berlin**, approximately
94 seconds after launch, including model loading and merging. All 8,000
per-CV rows, worker checksums and aggregate means were verified. Each table
cell contains the same 1,000 evaluation IDs, compared against the oracle CV.

| Latent dimension / decoder | Oracle Z' BERTScore F1 | Predicted Z' BERTScore F1 |
| --- | ---: | ---: |
| 256 / 270M | 0.88954 | 0.88777 |
| 256 / 1B | 0.88721 | 0.88677 |
| 768 / 270M | 0.88981 | 0.88831 |
| 768 / 1B | 0.88822 | 0.88746 |

These are raw, unrescaled scores. The close means do not establish equivalent
factual quality or statistical significance; the existing factual diagnostics
remain necessary. Precision, recall and individual scores are retained in the
result files for subsequent analysis.

Validation: six new offline tests passed, 187 relevant existing regression
tests passed, three existing GPU integration tests were skipped, and
`ruff check src/ exp/` passed. A real-GPU check verified identical-text F1 of
1.0 and exact agreement between untruncated tokenization and the reference
library's tokenization. The full repository test command was also attempted;
collection stopped on existing missing `dotenv` and `langvae` dependencies
in the decoder environment. That environment was not modified. Raw check logs
and the launch provenance are saved beside the BERTScore outputs.

## VADER sentiment follow-up

`python -m exp.text_realizer_vader` adds a CPU-only sentiment comparison to
the same saved `scores/comparisons.jsonl`. It uses the unchanged reference
[`vaderSentiment==3.3.2` implementation](https://github.com/cjhutto/vaderSentiment)
and its bundled lexicons. No training, GPU, neural weights or external API is
required. This supplementary analysis was specified after the earlier results.

Each complete CV receives a compound sentiment score in [-1, 1]. Whitespace
is collapsed consistently; capitalization and punctuation are preserved.
There is no truncation or sentence splitting. The reference is always the
oracle counterfactual CV, including for generated text from predicted vectors.
For each candidate and each of the eight decoder/condition combinations:

- `reference_compound` and `generated_compound` retain both scores.
- `sentiment_delta = generated_compound - reference_compound`: positive means
  a shift towards more positive language, negative towards more negative.
- `absolute_sentiment_delta = abs(sentiment_delta)` measures the magnitude of
  the difference. Its mean is computed from the individual absolute differences,
  so opposite shifts cannot cancel as they can in the mean signed difference.

One new module reuses the existing comparison validator and atomic artifact
utilities. VADER loads lazily, and synthetic tests require no downloads.
The new dependency is installed in a temporary `uv --with` environment,
preserving the BERTScore environment and lockfile. No new launcher or training
configuration is needed. From the repository root, after setting up the
BERTScore environment above:

```bash
uv run --project exp/text_realizer_bertscore/environment --frozen \
  --with vaderSentiment==3.3.2 python -m exp.text_realizer_vader
```

On this machine, use `.cache/text-realizer-tools/uv` and set
`UV_CACHE_DIR=.cache/text-realizer-uv`. Optional `--input` and `--output` paths
support another cluster. The default output directory is
`reports/talent_motivation/text-realizer-evaluation/package-v1/vader/`, containing
`per_cv.csv` (8,000 pairs), `summary.csv` (eight means), and `summary.json`
(results, timings and checksums). `plan.json` binds the input, evaluator and
shared source files, Python/package version, lexicons and base environment.
Identical texts are scored once. Repeating a completed run verifies checksums
and returns its results; interrupted CPU runs restart, and changed inputs or
settings require a new output directory.

Interpret this as a coarse tone diagnostic. VADER was developed for social
media (Hutto and Gilbert, ICWSM 2014), and whole-CV scores may saturate on
predominantly positive descriptions. A small sentiment difference does not
establish factual fidelity, preservation of behavioral traits, employability
or counterfactual fairness. No sentiment classifier is fitted to these CVs,
and no significance claim follows from the reported means alone.

The first run completed on **29 October 2026 at 08:25:10 Europe/Berlin**,
after **28.94 seconds** of CPU scoring and artifact preparation. It produced
8,000 paired rows for the same 1,000 evaluation IDs; caching reduced 9,000
reference/generated text slots to 8,976 unique texts scored. Checksums, ID
counts, difference formulas and all summary means were independently checked.

| Latent dimension / decoder | Mean delta, oracle Z' | Mean delta, predicted Z' | Mean absolute delta, oracle Z' | Mean absolute delta, predicted Z' |
| --- | ---: | ---: | ---: | ---: |
| 256 / 270M | -0.000042 | -0.001289 | 0.007141 | 0.007959 |
| 256 / 1B | 0.000408 | 0.000518 | 0.006743 | 0.006922 |
| 768 / 270M | 0.000787 | 0.000624 | 0.006734 | 0.007087 |
| 768 / 1B | 0.000392 | 0.000821 | 0.007261 | 0.006806 |

The common reference mean is 0.988822. Scores concentrate near the positive
ceiling: 98.7% of the 1,000 reference CVs and 99.225% of the 8,000 generated
CVs have compound scores >= 0.95 (a descriptive threshold, not VADER's
positive/neutral classification boundary). This limits the diagnostic's
ability to distinguish these decoders. Small differences should not be read
as proof that tone, behavioral details or fairness were preserved.

Validation: five new synthetic tests and six existing BERTScore tests passed;
`uv run ruff check src/ exp/` passed. A real VADER smoke test checked positive,
negated and neutral text. The requested full `uv run pytest tests/ -q` command
was attempted in the existing decoder environment and stopped during collection
in seven modules requiring the absent `dotenv` or `langvae` dependencies.
Those dependencies and the root environment were left untouched. Validation
and run logs are saved in the VADER output directory. The prior BERTScore
input, scorer source, lockfile and worker artifact hashes remain unchanged.

## Changed and preserved factual values

`python -m exp.text_realizer_fact_changes` evaluates whether the generated CV
reproduces the oracle's changes while retaining its unchanged factual values.
This is a supplementary, post-hoc analysis of the same 1,000 IDs and eight
decoder/conditioning combinations. It reuses the unchanged conservative
`exp/text_realizer/facts.py` extractor for country, region, degree, university
tier and project count. No model download, training, GPU or new dependency is
required.

For each candidate and field, extract the original value `a`, oracle
counterfactual value `b`, and generated value `c`:

- If either `a` or `b` is missing or ambiguous, label the pair `unscorable`
  and exclude it from both accuracy denominators.
- If `a != b`, label it `changed`. Success requires `c == b`; producing a
  different but incorrect value, or retaining the original value, fails.
- If `a == b`, label it `preserved`. Success requires `c == b`.
- Missing or ambiguous generated values count as failures on scorable pairs.
  An empty group has null accuracy, not zero or perfect accuracy.

Each field has separate changed/preserved denominators, correct counts,
accuracy, original/oracle/joint extraction coverage and unparseable-output
counts. There is no pooled headline score: region is derived from country,
and field coverage and change frequency differ substantially. These are
changes observed in the supplied oracle text, not an independent verification
of what the SCM intervention should change. Occupation and behavioral episodes
are outside the existing extractor's scope.

From the repository root, with the existing BERTScore environment:

```bash
uv run --project exp/text_realizer_bertscore/environment --frozen \
  python -m exp.text_realizer_fact_changes
```

On this machine, use `.cache/text-realizer-tools/uv` and set
`UV_CACHE_DIR=.cache/text-realizer-uv`. Optional `--input` and `--output` paths
support another cluster. Results are written under
`reports/talent_motivation/text-realizer-evaluation/package-v1/fact-changes/`:

- `per_cv.csv`: 40,000 rows, one per candidate, decoder/condition and field;
  includes all three extracted values, status and correctness (blank when
  unscorable).
- `summary.csv`: 80 rows, one per decoder/condition, field and changed/preserved
  category, including categories with no observations.
- `summary.json`: the same summaries, provenance, timing and CSV checksums.
- `plan.json`: immutable input, codebook, extractor and scoring-source hashes.

Completed reruns verify the saved outputs. Changed inputs or code require a
new output directory; interrupted inexpensive CPU runs restart. Existing
CVs, scoring outputs, extractor rules, codebook and training artifacts are
unchanged.

The first run completed on **29 October 2026 at 08:53:16 Europe/Berlin** in
**8.46 seconds** on CPU. All 8,000 CV pairs, 40,000 field rows and 80 summary
groups were checked independently for IDs, classification, denominators,
correctness, aggregate accuracies and file checksums.

The following reference counts apply identically to every decoder/condition:

| Field | Original scorable | Oracle scorable | Changed | Preserved | Excluded pairs | Joint coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Country | 989 | 988 | 977 | 0 | 23 | 97.7% |
| Region | 989 | 988 | 977 | 0 | 23 | 97.7% |
| Degree | 983 | 982 | 420 | 549 | 31 | 96.9% |
| University tier | 1,000 | 1,000 | 479 | 521 | 0 | 100.0% |
| Project count | 870 | 870 | 1 | 798 | 201 | 79.9% |

Requiring both original and oracle values makes these denominators stricter
than those in the earlier oracle-only factual checks. Country and region
preservation accuracy is undefined because there are no scorable preserved
cases. Project-count change accuracy is based on one case (incorrect in all
eight variants) and should not support a general conclusion.

| Decoder | Vector source | Country changed | Degree changed | Degree preserved | Tier changed | Tier preserved | Project count preserved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256-270M | Oracle | 92.5% | 94.5% | 94.9% | 76.0% | 78.7% | 26.2% |
| 256-270M | Predicted | 22.7% | 59.0% | 68.9% | 55.3% | 70.1% | 25.8% |
| 256-1B | Oracle | 95.5% | 58.3% | 52.6% | 58.5% | 54.7% | 14.3% |
| 256-1B | Predicted | 24.6% | 52.6% | 50.3% | 56.4% | 54.3% | 15.3% |
| 768-270M | Oracle | 94.5% | 94.5% | 95.1% | 66.0% | 62.8% | 14.7% |
| 768-270M | Predicted | 21.0% | 57.4% | 71.4% | 60.5% | 58.3% | 15.8% |
| 768-1B | Oracle | 97.9% | 89.0% | 89.4% | 63.3% | 62.6% | 15.9% |
| 768-1B | Predicted | 18.9% | 54.3% | 68.5% | 59.3% | 59.3% | 16.0% |

Exact country changes deteriorate strongly when conditioning on predicted
vectors. Region change accuracy remains higher (64.3–83.2% for predicted
vectors); recovering the region does not ensure the exact country is correct.
Project-count preservation is poor even with oracle vectors, locating a
limitation in decoding/recoverability before introducing the predicted-vector
pipeline. For both 270M decoders, changed-degree accuracy falls from 94.5%
with oracle vectors to 57.4–59.0% with predicted vectors. These are descriptive
results on extractable fields, with no significance claim. Different 256-D and
768-D flow training protocols still prevent a dimension-only interpretation.

Validation: six new synthetic tests and 26 relevant existing tests passed;
`uv run ruff check src/ exp/` passed. The full `uv run pytest tests/ -q` was
attempted and remains blocked during collection in seven modules by the
existing missing `dotenv`/`langvae` dependencies. Check and run logs are saved
in the output directory. No new dependency or environment change was needed.

## Portable LaTeX tables for the paper

The historical `exp.paper_tables.decoder_table` still describes the earlier
320-unit development pilot. Use the separate
`exp.paper_tables.decoder_package_table` for this package evaluation: four
factual-only decoders (256/768 dimensions, Gemma 3 270M/1B), each evaluated
from oracle and predicted counterfactual vectors on the same 1,000 test IDs.
Here, "oracle" describes the inference vector, not oracle-exposed training.

The portable input is
[`results/decoder_package_v1.json`](results/decoder_package_v1.json), about
70 KiB. It contains the four aggregate reports, metric settings, source
checksums and pipeline provenance, with no CV texts, weights or individual
latent vectors. It is outside the machine-specific ignored result tree, so
include it together with the new script, test and this documentation when
committing the changes. No ignore-rule change or force-add is necessary.
Nothing is committed or pushed automatically.

On another machine, from the repository root, Python 3.12's standard library
is sufficient; neither the root ML environment nor a GPU is needed:

```bash
python -m exp.paper_tables.decoder_package_table \
  --output overleaf/tables/decoder.tex
```

The command creates four bare `booktabs` tabulars using the same formatting
helpers as the historical table (shown below with the stem `decoder_package`):

| File | Columns after dimension and Gemma size |
| --- | --- |
| `decoder_package.tex` | Character similarity, chrF, ROUGE-L, BERTScore F1, signed VADER difference, mean absolute VADER difference |
| `decoder_package_facts.tex` | Overall country, region, degree, university-tier and project-count accuracy |
| `decoder_package_changed.tex` | Accuracy on changed values of each field |
| `decoder_package_preserved.tex` | Accuracy on preserved values of each field |

Every table has separate oracle-vector and predicted-vector blocks. Fact
accuracies are percentages; text scores use their raw 0–1 scales, and VADER
differences retain their raw scale. Each factual column displays its own
scorable denominator. Undefined accuracy is `--`; the project-change column
explicitly has `n=1`. No winning values are bolded across the differing flow
protocols, and no confidence intervals are invented.

Each vector block also has a non-generative **retrieval** row per dimension: the
factual CV of the fit (pool) unit whose canonical latent is closest in cosine to
the oracle or predicted vector, as in the earlier pilot. A final **unedited x**
row offers the unit's own factual CV as its counterfactual. Both are scored with
the decoders' metrics against the same references and must share their fact
cohorts. Their portable aggregates are
[`results/decoder_package_baselines_v1.json`](results/decoder_package_baselines_v1.json),
made on CPU by `exp.text_realizer_package_eval.baselines` (RoBERTa-large
BERTScore with the decoders' settings, on CPU and the root environment's
torch/transformers; command in its docstring).

A last **same-state rewrite** row is the noise floor: the 200 floor texts (the
unit re-rendered with another country of the same region, so the same state)
scored against their own factual CVs with the same metrics. It shows how close
two correct CVs of one state get, i.e. roughly the best a decoder can reach
against x'. It uses other units and references than the 1,000 test pairs, the
country differs by construction (shown as `--`), and it has no changed/preserved
split. Aggregates: [`results/decoder_package_floor_v1.json`](results/decoder_package_floor_v1.json),
from `exp.text_realizer_package_eval.baselines --floor` (about 7 minutes on CPU).

For example, in a two-column paper (the float and caption remain under the
paper's control, as with the historical renderer):

```latex
% Preamble:
\usepackage{booktabs}

\begin{table*}[t]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \input{tables/decoder_package.tex}
  \caption{CV reconstruction from oracle and predicted counterfactual
    vectors on 1,000 test units. All decoders are factual-only.
    Text scores are unrescaled; VADER reports generated minus oracle
    sentiment and its mean absolute difference. The 256-D and 768-D
    reader/flow training protocols differ.}
  \label{tab:decoder-package}
\end{table*}
```

Use the companion files in separate floats, or stack them within one float
where page layout permits. Their captions should state that ambiguous
reference values are excluded, unparseable generated values fail on scorable
references, and changed/preserved scores additionally require an extractable
original value. The supplied JSON retains exact counts and coverage.

To regenerate the portable input on this GPU machine after an explicitly
updated evaluation, add `--reports` (it is not needed on the paper machine):

```bash
python -m exp.paper_tables.decoder_package_table \
  --reports reports/talent_motivation/text-realizer-evaluation/package-v1 \
  --output .cache/decoder-paper-preview/decoder.tex
```

`--summary PATH` can select a different portable JSON. Export checks the common
comparison hash, candidate IDs, all eight variants and all 80 factual groups.
Rendering checks the embedded report digest, unit counts, metric ranges,
accuracy denominators and reference-cohort consistency. These are validation
and rendering steps only; metric values are read from the saved results.

Validation on 29 October 2026: all 27 focused tests passed (including ten new
table tests), and `uv run ruff check src/ exp/` passed. The renderer was also
run with Python's site packages disabled, using only the portable JSON; all
four outputs reproduced exactly. The four real-data tabulars compiled with
`pdflatex` and `booktabs` without overflow warnings. A local preview is at
`.cache/decoder-paper-preview/preview.pdf`. The full repository pytest command
was attempted and still stops in seven existing modules requiring the absent
`dotenv`/`langvae` dependencies. The paper tables do not depend on either.

