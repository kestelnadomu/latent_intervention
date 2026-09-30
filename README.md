# Counterfactual Latent Representations: A Neurosymbolic Approach

Research code and publication materials for the *latent intervention* project.

For a student-facing walkthrough with diagrams, configuration maps and function
call trees, start with the [codebase guide](docs/codebase_guide.md).
For the behavior-preserving cleanup and a map of the smaller entry points, see the
[implementation review map](docs/implementation_refactor.md).

The pipeline builds counterfactual latent representations of text: an SCM simulates paired factual and counterfactual candidate attributes, an LLM verbalizes the approved worlds into CV personal statements, the configured frozen LangVAE, Nomic, Qwen3 or EmbeddingGemma encoder maps them into fixed-width latents, a semantic decoder grounds those latents in the SCM variables, and a latent manipulator learns to transform them analogously to the SCM's counterfactual operation. LangVAE uses 128 dimensions; the other encoders offer multiple embedding widths.

The synthetic SCM follows the CV Screening setup of the LIBERTy paper ([arXiv 2601.10700](https://arxiv.org/abs/2601.10700)). Text generation is a protocol-faithful adaptation of its Appendix D.3: seed personal statements are abstracted into narrative templates, personas are generated for sampled job titles, and each rendered world combines a fixed template, persona, and the unit's attribute values. This is not a literal replication of the released benchmark.

## Repository layout

| Path | Contents |
| --- | --- |
| `exp/sim/` | Data generation pipeline: Python SCM, codebook, generation prompts (`prompts/`), seed statements + job titles, LLM plumbing, stage runner, `config.yaml` (own README) |
| `exp/sim/R/` | Original SCM simulation in R, kept as the reference implementation (own README, renv) |
| `src/` | Encoder, semantic decoder, latent manipulator, training/eval pipeline, `config.yaml` (per-module hyperparameters) |
| `data/` | Generated simulation/text handoffs (tracked), latent artifacts (local by default), and older sampling pools |
| `poster/` | Quarto poster and slides |

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/). Run all commands from the repository root:

```bash
uv sync --extra dev --frozen
```

All billed stages—templates, personas, and CVs—use the single endpoint in `exp/sim/config.yaml`. The default configuration targets Azure OpenAI and reads `AZURE_OPENAI_API_KEY`. The key must grant access to that endpoint and to the deployment named `gpt-5.4` in `exp/sim/prompts/*.yaml`.

Provide the key locally, never in a committed file. Either create an ignored root `.env` file:

```dotenv
AZURE_OPENAI_API_KEY=your-key
```

or set it for the current shell:

```powershell
$env:AZURE_OPENAI_API_KEY = "your-key"
```

For another OpenAI-compatible provider, change `llm.base_url`, `llm.api_key_env`, and the prompt `model` deployment consistently before generating any text.

## Pipeline

The active talent experiment uses one fixed query, `do(X=3)`. Units already in region `X=3` are identity pairs; the other units receive the nontrivial intervention. It does not generate the opposite-value queries or an orbit of interventions.

The seeded `pair_index.csv` contains each integer unit ID, its 80/20 train/test split, and whether the structured intervention is an identity. Artifact coverage is:

| Object | Coverage | Artifact |
| --- | --- | --- |
| `S`, `S'` | All units | `data/sim_talent/sim_data_{factual,counterfactual}.csv` |
| `X` | All units | `data/text_talent/cv_factual.csv` |
| `X'` | All units by default; test only when configured | `data/text_talent/cv_counterfactual.csv` |
| `Z=f(X)` | All units | `z` in `data/latents/talent/{encoder}/z_pairs.pt` |
| `Z'=f(X')` | Test units only | `z_prime` in `data/latents/talent/{encoder}/z_pairs.pt` |

By default (`generation.include_train_counterfactual_texts: true` or omitted), `X'` is created for every unit; setting the option to `false` creates `X'` for test units only. Selected identity units copy `X'=X` without another LLM call, and identity test units use `Z'=Z` without another encoding. Nonidentity worlds reuse the same template, persona, and bin quantiles. Training `X'` is available for future work but does not enter the current training losses; `Z'` and counterfactual recovery evaluation remain test-only.

Data generation is configured by `exp/sim/config.yaml`. Every billed stage appends successful rows and resumes from missing IDs. Each active billed output CSV must remain together and be committed with its `*.generation.json` information file; the dated template CSV is only a legacy archive. `generation.limit` controls how many IDs are sent to the API per invocation; free identity copies do not count. Each selected ID permits at most `generation.max_attempts` API calls. Set `generation.limit: 1` for a small billed smoke test of at most three calls, inspect the result, then restore `null` and rerun the same stage to completion before advancing.

For a complete fresh run, execute these stages in order:

```bash
uv run python -m exp.sim.run simulate            # S/S', epsilon, pair_index.csv, simulation_info.json
uv run python -m exp.sim.run generate-templates  # seed statements -> data/text/templates.csv
uv run python -m exp.sim.run generate-personas   # job titles -> data/text/personas.csv
uv run python -m exp.sim.run generate-texts      # render_plan.csv and X for every unit
uv run python -m exp.sim.run generate-counterfactual-texts  # X' for configured train/test coverage
uv run python -m exp.sim.run validate-pairs      # validate complete S/S' and X/X' pairing
```

The completed run contains the configured template and persona pools, one factual CV per simulated unit, and one counterfactual CV per selected unit. With the default toggle, all units are selected; when it is disabled, only test units are selected. Identity rows are copied without an API call, so the billed total depends on the configured sample size, split, and number of nonidentity units.

The simulation-data pipeline is complete only when `validate-pairs` reports:

```text
validated <factual units> factual units and <selected units> counterfactual pairs
```

Stop here for the data-generation handoff. Encoding and model training below are separate work.

Paired latent encoding and the existing training/evaluation stages use `src/config.yaml`:

```bash
uv run python -m src.pipeline encode --encoder-variant langvae
uv run python -m src.pipeline encode --encoder-variant nomic

# Select one encoder/decoder combination per run; artifacts never overwrite another variant.
uv run python -m src.pipeline train-decoder --encoder-variant langvae --decoder-variant independent
uv run python -m src.pipeline train-decoder --encoder-variant langvae --decoder-variant autoregressive

uv run python -m src.pipeline train-manipulator --encoder-variant langvae --decoder-variant independent --manipulator-variant baseline
uv run python -m src.pipeline evaluate --encoder-variant langvae --decoder-variant independent --manipulator-variant baseline
```

Nomic defaults to `nomic_128`. Select another width with `--nomic-dim 64|128|256|512|768`
(one integer, not the whole list) or `encoder.nomic_latent_dim`. This selects dimension-stamped
latent/model/report directories automatically; LangVAE remains 128-D. Existing Nomic 128-D
artifacts and their model/report directories were relocated from `nomic/` to `nomic_128/`
without changing tensor/checkpoint contents. Historical experiment records retain original
path strings; the relocation manifest is `reports/talent/nomic_dimensions/nomic_128_migration.json`.
Canonical `nomic_<dimension>/z_pairs.{pt,info.json}` files are trackable by Git; queue logs,
pending outputs, models and reports remain ignored.

To prepare all five Nomic dimensions, then launch the queue separately:

```bash
.venv/bin/python -m src.nomic_encoding --prepare
# When ready to start (detached; survives closing this terminal):
bash scripts/run_nomic_dimensions.sh
```

Preparation does not load the encoder or produce new embeddings. The queue encodes 768-D
once, checks it against the existing 128-D artifact, then derives 64/256/512-D by prefix
truncation and L2 renormalization. It retains the same pinned model, prefix, token limit,
input hashes and official split. Completed artifacts are verified and reused, never
overwritten. Complete output directories are published atomically after validation; an
interrupted transformer pass restarts, while any incomplete staging files are retained.
Only test counterfactuals are encoded. A lock prevents concurrent queues. The launcher uses
the installed `.venv` and cached model offline; it does not install or update packages.

Logs, protocol, plan and runtime status are in
`reports/talent/nomic_dimensions/nomic-dimensions-v1/`. Inspect `run.log`, `launcher.log`,
and `status.json`; before launch, only `plan.json` and `protocol.json` exist. Re-run the
launcher to resume completed stages. After changing code/config/inputs, prepare and launch
with a fresh `--run-id` / script argument. No decoder training is part of this queue.

### Qwen3-Embedding-0.6B dimensions

Only the 0.6B checkpoint is supported by the `qwen3` variant. The embedding width, not
the parameter count, appears in the artifact directory name:
`data/latents/talent/qwen3_{32,64,128,256,512,768,1024}/`. Each completed directory
contains `z_pairs.pt` and its provenance sidecar `z_pairs.info.json`. The model name
and pinned revision are recorded in metadata; larger Qwen checkpoints cannot use
these names through this configuration.

Qwen needs a newer Transformers version than the LangVAE environment. Keep its
hash-locked dependencies in the separate `.venv-qwen3`; do not upgrade `.venv`:

```bash
# Setup on a new machine: installs the isolated environment and caches the pinned model.
bash scripts/setup_qwen3.sh

# Audit every input length and smoke-test the real model (not a full encoding run).
.venv-qwen3/bin/python -m src.qwen3_encoding --preflight
# Validate and save the seven-dimension plan, without inference.
.venv-qwen3/bin/python -m src.qwen3_encoding --prepare

# When ready to start (detached tmux session; survives closing this terminal):
bash scripts/run_qwen3_dimensions.sh
```

The baseline embeds plain CV text with no retrieval instruction. It uses the last
nonpadding token, then prefix truncation and L2 normalization: **no mean pooling or
Nomic-style extra layer norm**. Inference is frozen, float32, CPU by default, and
offline from the pinned local cache; remote model code is disabled. The 1024-token
budget is enforced with an error, never silent truncation. The current input audit
found a maximum of 491 Qwen tokens across 5,000 factual and 1,000 test-counterfactual
texts. Identity pairs reuse factual embeddings exactly.

One 1024-D transformer pass supplies all seven dimensions. Smaller outputs are
derived by normalizing their prefixes, not by rerunning the transformer. The shared
dimension queue validates artifacts before atomically publishing each complete
directory, refuses overwrites, locks concurrent writers, and resumes verified
completed stages. An interrupted transformer pass restarts. No downstream training
is started by this queue.

Reports live in `reports/talent/qwen3_dimensions/qwen3-dimensions-v1/`:
`preflight.json`, `protocol.json`, `plan.json`, then `run.log`, `launcher.log`, and
`status.json` after launch. Preparation and launch require a passing preflight for
the exact inputs, model files, code, settings and recorded environment versions.
After changing those, choose a fresh run ID, repeat preflight and preparation with
`--run-id NEW_ID`, then pass `NEW_ID` to the launcher. Generated reports and the model
cache are Git-ignored; copy reports explicitly when transferring results.

Once the embeddings exist, downstream commands can use the original `.venv` with
`--encoder-variant qwen3 --qwen3-dim 128` (or another configured width); they load the
saved latents, not the transformer. Train a separate `g` and `h_Z` for each latent
space. The four-combination `decoder_experiment` baseline runner below is not
automatically expanded to include Qwen or the additional Nomic dimensions.

### EmbeddingGemma-300m dimensions

The `embeddinggemma` variant uses the gated
[Google checkpoint](https://huggingface.co/google/embeddinggemma-300m), pinned to
`57c266a740f537b4dc058e1b0cda161fd15afa75`. Before setup on a new machine, personally
accept the model's Gemma terms on Hugging Face and authenticate with a read-capable
token using `hf auth login`. Never put credentials in the repository or logs.

```bash
# Installs .venv-embeddinggemma from its hash lock and downloads the approved model.
bash scripts/setup_embeddinggemma.sh

# Bounded validation, followed by plan preparation; neither starts full encoding.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv-embeddinggemma/bin/python -m src.embeddinggemma_encoding --preflight
.venv-embeddinggemma/bin/python -m src.embeddinggemma_encoding --prepare

# Only when ready: detached tmux worker, safe to close the terminal afterwards.
bash scripts/run_embeddinggemma_dimensions.sh
```

Outputs are `data/latents/talent/embeddinggemma_{128,256,512,768}/`, each with
`z_pairs.pt` and `z_pairs.info.json`. One native 768-D pass supplies the smaller
dimensions by prefix truncation and L2 normalization. These folders are created
only when verified artifacts are ready to publish; setup does not leave empty
production directories. Existing Nomic/Qwen/LangVAE artifacts are not replaced.

The fixed baseline uses Google's classification prefix (`task: classification | query: `)
for every factual and counterfactual CV. The full released SentenceTransformer
stack is retained: bidirectional Gemma3, masked mean pooling including the prompt,
both learned dense projections, then normalization. Float32 CPU inference avoids
float16, which this model does not support. Inputs above 2,048 tokens cause an error;
the audit includes prompts and special tokens. No text decoder or fine-tuning is added.

The shared queue preserves official splits and exact identity copies, hashes inputs
and model files, prevents concurrent writes, and atomically publishes verified
directories. Completed dimensions can resume; an interrupted transformer pass must
restart. Reports are under
`reports/talent/embeddinggemma_dimensions/embeddinggemma-dimensions-v1/`.
`--prepare` and `--run` require a matching passing `--preflight`. After source,
settings, inputs or environment changes, use a fresh run ID for preflight and
preparation and pass that ID to the launcher. Reports and weights stay Git-ignored;
only canonical embedding artifacts are eligible for tracking.

Later downstream training uses the original `.venv` and
`--encoder-variant embeddinggemma --embeddinggemma-dim 128` (or another supported
width). It reads stored embeddings and does not load Sentence Transformers.
Train separate `g`/`h_Z` models per latent space. The four-case decoder sweep is
unchanged; it does not automatically include this encoder.

### Downstream training

To train and compare both `g` variants across **all saved encoder dimensions**,
see the [all-encoder decoder benchmark](docs/experiments/decoder_benchmark.md).
It runs the common tuning budget in detached tmux and automatically writes a
Markdown comparison, with validation-based selection and separate held-out tests.
Canonical `g` checkpoints and their training reports, plus the benchmark summary,
per-seed metrics, configurations and provenance, are eligible for Git tracking.
They still need to be added, committed and pushed; tuning/extra-seed checkpoints,
archives and logs remain ignored. See the benchmark guide for the exact policy.

The same decoder commands work with `--encoder-variant nomic`. `pair_index.csv` is the sole train/test authority. `train-decoder` reserves a fixed validation holdout from the official training IDs. It trains for at most 500 epochs, halves the learning rate on a validation plateau, stops after 30 epochs without sufficient improvement, and restores the checkpoint with the lowest validation joint NLL. Per-epoch history, the exact split, hyperparameters, checkpoint checksum, and descriptive accuracy/ECE are saved below `reports/talent/{encoder}/g-{decoder}/`. The validation metrics are used for selection and are not test estimates; no temperature scaling is fitted. `train-manipulator` uses all official training IDs, while `evaluate` uses only official test IDs.

To tune and repeat all four baseline combinations with the same budget:

```bash
uv run python -m src.decoder_experiment --run-id cv-g500-v1 --workers 4 --threads-per-worker 1
```

This runs 20 common hyperparameter candidates per combination, selects each by validation joint NLL, then trains five initialization seeds with the split held fixed. Seed 42 is declared in advance as the active model; all five are retained and summarized. Trials and final seeds live in `experiments/{run-id}/` under each model/report directory. The overall protocol, source snapshot, input hashes, status, and publication manifest are under `reports/talent/decoder_experiments/{run-id}/`. The previous active checkpoint and report are archived before the new verified files replace them. Repeating the command resumes verified completed trials; an interrupted trial restarts. Use a new run ID after code, inputs, or configuration change. Canonical active checkpoints and their reports are eligible for Git tracking; the four-case experiment tree, archives and logs remain ignored and need explicit transfer to another server.

Latent, decoder, and manipulator artifacts are separated by encoder, decoder, and manipulator variant and bound to the configured encoder, source hashes, schema, and upstream artifact hashes. Loading incompatible or incomplete metadata fails with an instruction to re-encode or retrain instead of silently mixing latent spaces.

Choose $h_Z$ with `--manipulator-variant` or `latent_intervention.variant` in
`src/config.yaml`: `baseline`, `pre_additive`, `noise_token`, `dist`, `particles`,
`state_flow`, `distilled_flow`, `direct_semantic_flow`, or `oracle_regression`. All use separate
`models/talent/{encoder}/g-{decoder}/h-{variant}/` and matching report directories.
Distilled training creates or reuses a compatible `h-state_flow` teacher within
the same encoder/decoder combination; different embedding dimensions never share
that checkpoint. The state flow uses factual $S$ and $h_S$; distilled and direct
flows expose the standalone inference interface $(Z,\delta)\mapsto\Delta(\mathcal Z)$.

The established transformer stages and report format remain in `src/pipeline.py`; that file delegates only the three flow variants to `src/flow_workflow.py`. Flow architectures and objectives are isolated in `src/flow_intervention.py`, while flow evaluation adds paired-$Z'$ recovery and support diagnostics.

The active data supports the configured `do(X=3)` query; the deployable flows also
train a no-op. Other schema-valid flow interventions are accepted for exploratory
inference with a warning, but are outside training support. Downstream stages read
the existing 5,000-unit dataset and its official split; they do not regenerate the
data, re-encode CVs, or retrain the frozen semantic decoder. The small generation
example in `exp/sim/config.yaml` is not the size of those stored artifacts.

The separate `langvae_ft` baseline uses native LangVAE fine-tuned on CV passages,
with unchanged architecture. See the [checkpoint and downstream handoff](docs/experiments/langvae_ft_handoff.md)
for model transfer and opt-in encoding/training commands. Its checkpoints stay
outside Git; stock LangVAE, Nomic and the shared pipeline defaults are unchanged.

### Oracle-supervised latent regression

`oracle_regression` is the ninth editor: a residual MLP fitted to **training-pair
counterfactual embeddings**, using ordinary MSE, without loading `g` or `h_S`.
It is a privileged conditional-mean reference, not a guaranteed upper bound. It
supports only the fixed intervention recorded in its checkpoint (currently
`do(X=3)`); predictions are not projected onto the unit sphere.

The existing `z_pairs.pt` files remain unchanged and contain counterfactual
embeddings only for official test units. Prepare the separate training targets:

```bash
# Inspect inputs and save a plan only: no model inference or training.
.venv/bin/python -m src.oracle_encoding --prepare

# Only when ready: detached tmux encoding, using the existing pinned environment.
bash scripts/run_oracle_targets.sh
```

This encodes the 2,972 nonidentity training counterfactual texts once at 768D,
copies the 1,028 identity factual embeddings exactly, then derives 256D by prefix
truncation and L2 normalization. Each encoder directory receives a separate
`oracle_train/z_prime.pt` plus a provenance/checksum sidecar. Only the oracle
loader can consume this artifact; it requires exactly the official training IDs.
The queue verifies factual encoder probes, refuses incompatible outputs, and
reuses verified completed dimensions. An interrupted native pass restarts.
Logs and status are under `reports/talent/oracle_targets/oracle-targets-v1/`.
After source/config/input changes, choose a fresh `--run-id` (launcher argument).

After target encoding completes, train and evaluate one dimension at a time:

```bash
.venv/bin/python -m src.pipeline train-manipulator --encoder-variant embeddinggemma --embeddinggemma-dim 256 --decoder-variant independent --manipulator-variant oracle_regression
.venv/bin/python -m src.pipeline evaluate --encoder-variant embeddinggemma --embeddinggemma-dim 256 --decoder-variant independent --manipulator-variant oracle_regression
# Repeat with --embeddinggemma-dim 768 for the other selected candidate.
```

Defaults are two 512-wide hidden layers, AdamW, a 500-epoch ceiling, validation
MSE selection, plateau LR reduction, and early stopping. The seeded 3,200/800
fit/validation split uses only the 4,000 official training IDs. The 1,000 test
pairs are evaluation-only. The model, its sidecar, `training.json` (including the
exact split and epoch history), and `eval.json` live in the encoder/decoder's
`h-oracle_regression/` directories. Evaluation reports raw MSE, squared L2,
cosine recovery, identity/nonidentity groups, the no-edit reference, and the
existing flow-style distributional/semantic diagnostics. MSE is per coordinate;
do not rank embedding dimensions merely by that dimension-dependent scale.

Canonical oracle targets/checkpoints/reports are Git-trackable but must still be
added, committed, and pushed. Logs, progress, locks, and staging files remain
ignored. Training refuses to overwrite an existing run; use a fresh
`latent_intervention.tag` in a separate `--config` YAML for another seed/config.
Interrupted training restarts, rather than resuming optimizer state. This adds
the baseline, not the multi-variant tuning/benchmark runner.

### Queued nine-family h_Z benchmark (EmbeddingGemma 256D / 768D)

The isolated runner reuses the existing losses and frozen independent `g` models.
Its protocol is in `configs/hz_benchmark.yaml`; ordinary pipeline defaults are
unchanged. Preparation checks inputs and saves a manifest, but **does not encode,
train, evaluate on the test set, or launch tmux**:

```bash
.venv/bin/python -m src.hz_benchmark --prepare --run-id hz-gemma-v1
```

Defaults: four CPU worker processes, one numerical thread each; 18 cases;
eight learning-rate/width candidates per case; five final seeds (42–46);
500-epoch ceiling, early stopping with patience 30, learning-rate reduction with
patience 10, and best-validation checkpoint restoration. This is **234 fits**,
not 234 fits times an extra teacher budget. `dist` uses 100 pretraining epochs
plus at most 400 joint epochs; stopping/selection begin in the joint phase.
These settings are a reproducible starting search budget, not a guarantee of
publication-quality convergence or exhaustive tuning. The three flow variants,
`dist`, and `oracle_regression` use explicit per-dimension/per-family width
overrides (`model_widths`) from the benchmark YAML. All nine families' two size
tiers match the standard baseline transformer within 7% in trainable parameters.
Flows keep all eight blocks, `dist` keeps its weight and realiser components,
and the oracle keeps its two-hidden-layer residual MLP. Only their widths are
adjusted. This is parameter-budget matching, not equal compute or a convergence
guarantee. Actual counts and constructor settings are
saved in `plan.json`, with a count table in `summary.md` and counts in each
training report. Frozen `g` and separate distillation teachers are excluded.
Regularizer strengths and other architecture settings remain fixed at their
`src/config.yaml` values; the ordinary pipeline's model defaults are unchanged.

Before the full queue, run the isolated fit/validation pilot:

```bash
bash scripts/run_hz_pilot.sh
```

It fits one size and learning rate for all nine families at both dimensions,
with one seed and a 50-epoch ceiling. `dist` has 10 pretraining epochs plus
up to 40 joint epochs. For `pre_additive`, it also compares full-vector RMS
noise levels 0.1, 0.25, 0.5 and 1.0, plus the previous per-coordinate
`noise_std=1` as a control. The model uses `noise_std = RMS / sqrt(d)`.
These are 26 separate fits; each candidate's constructor, learning curve,
loss components, runtime and checksum are saved under
`reports/talent/hz_benchmarks/hz-gemma-pilot-v1/` and matching isolated model
paths. The pilot selects a noise level for each dimension by validation
semantic KL. Its report also shows the no-edit reference and output spread.
It does not score official-test targets or publish deployment models. Once
reviewed, enter the selected noise levels in `pre_additive_noise_rms` in the
benchmark YAML, refresh the unstarted full-run preparation, and check
readiness before launching the full queue. The pilot is a feasibility and
noise-scale check; its short histories do not certify full-run convergence.

The completed `hz-gemma-pilot-v1` run selected RMS 0.5 at both dimensions:
validation semantic KL was 0.269941 at 256D and 0.192790 at 768D, versus
0.949295 and 0.993450 with the previous per-coordinate `noise_std=1`.
Its [pilot report](reports/talent/hz_benchmarks/hz-gemma-pilot-v1/summary.md)
records all 26 fit/validation runs, diagnostics, and selection. The prepared
`hz-gemma-v1` full-run manifest uses those selected scales; it remains
unstarted until the explicit launch command below.

All h_Z fitting uses the same 3,200/800 split within the official 4,000 training
IDs. The 1,000 test units never select hyperparameters or stopping epochs.
Validation criteria differ by family: structured counterfactual NLL for the
baseline, factual conditional NLL for state flow, teacher energy distance for
distillation, true training-pair MSE for the oracle, and semantic forward KL for
the semantic distributional variants. These are **not** used for a combined
cross-family validation ranking. They are selection criteria; the established
training losses still include their original regularizers. Frozen `g` was tuned
on the same inner validation split, so this is not nested end-to-end validation.

The search queue waits for a dimension's state-flow search before its distilled
flow search. Final distillation waits for the same-dimension, same-seed final
state-flow teacher. Every teacher fits only the 3,200 fit IDs. All search choices
are frozen before test scoring begins. Evaluation uses 64 draws/unit (point
masses for deterministic models), batching, fit-only standardization, normalized
energy scores, latent recovery, structured/semantic diagnostics, a no-edit
reference, and identity/nonidentity breakdowns. Observed-state flow inference
is reported separately from latent-only inference. `baseline` and `dist` receive
paired S' labels; the oracle additionally receives paired training Z'. These
information-access differences are explicit in the report.

Before a future launch, oracle training targets must exist at both dimensions.
The runner reports missing targets and refuses to launch an incomplete nine-family
benchmark; it **never** launches their encoding implicitly. After explicitly
running the separate oracle-target queue above, check and launch with:

```bash
.venv/bin/python -m src.hz_benchmark --check-ready --run-id hz-gemma-v1
bash scripts/run_hz_benchmark.sh hz-gemma-v1
```

The launcher creates detached tmux session `hz-benchmark`. Closing the client
does not kill it (a server reboot still does). `status.json` and each fit's
`progress.json` expose progress; `run.log` contains completion/failure messages.
Relaunching the same command reuses hash-verified completed jobs; an interrupted
fit restarts from its fixed seed, **not** from a saved optimizer state. A failed
job blocks its dependents while unrelated jobs finish; failures are not silently
dropped from model selection. Changed source/config/input hashes require a new
run ID. Edit the protocol before preparing a run or use a fresh ID after edits.

Outputs are isolated, leaving existing encodings, `g`, and canonical h_Z untouched:

```text
reports/talent/hz_benchmarks/<run>/
  plan.json, readiness.json, selection.json, results.json, summary.md
  fits/embeddinggemma_<d>/g-independent/h-<variant>/{search,final}/<label>/training.json
  evaluation/embeddinggemma_<d>/g-independent/h-<variant>/seed-<n>.json
models/talent/embeddinggemma_<d>/g-independent/h-<variant>/benchmarks/<run>/
  search/trial-<n>/latent_intervention.pt
  final/seed-<n>/latent_intervention.pt
  selected/latent_intervention.pt
```

Selected deployment weights (predeclared seed 42, not the best test seed), their
sidecars, all completed training histories/evaluations, and the reproducibility
bundle are Git-trackable. Trial/other-seed weights, logs, live progress, locks,
and failures remain local. Nothing is automatically staged, committed, or pushed.
The benchmark checkpoints include their constructor settings and provenance;
reload them with `src.hz_training.load_model(path)`, not the legacy pipeline loader.
The modules `hz_benchmark_matrix`, `hz_training`/`hz_validation`, `hz_evaluation`,
and `hz_report` separate planning, fitting, evaluation, and reporting; only small
optional epoch-control hooks were added to the established trainers.

## Publish

From root directory:

```{bash}
quarto publish gh-pages poster/poster.qmd
```

## Props

* Poster and slide template: [mpimet](https://github.com/mpimet/quarto/)
