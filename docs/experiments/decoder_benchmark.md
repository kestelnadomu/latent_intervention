# All-encoder semantic-decoder benchmark

The benchmark trains both `independent` and `autoregressive` versions of g for
every completed canonical latent artifact in `data/latents/talent/`. The initial
matrix has 18 spaces and 36 cases: stock LangVAE, LangVAE-FT, five Nomic widths,
seven Qwen3-0.6B widths, and four EmbeddingGemma widths. Encoders are not rerun.

## Launch and inspect

Use the existing CPU `.venv`, from the repository root:

```bash
# Validate all artifacts, freeze the protocol, and create the initial report.
.venv/bin/python -m exp.benchmarks.semantic_decoder.run --prepare --run-id g-all-encoders-v1

# Four worker processes, one CPU thread each; survives closing the terminal.
bash scripts/run_decoder_benchmark.sh g-all-encoders-v1

# View progress without attaching to tmux.
tail -n 30 reports/talent/decoder_benchmarks/g-all-encoders-v1/run.log
```

The `decoder-benchmark` tmux session runs independently. `status.json` updates
about every 30 seconds, including completed fits and per-case epoch progress.
`summary.md` starts as an explicitly incomplete report, updates when cases finish,
and becomes the final comparison automatically. No h_Z training is launched.

## Fixed scientific protocol

- Same official split for all cases: 3,200 decoder-fit / 800 validation / 1,000 test.
- Same 20 hyperparameter candidates per case; select by validation full joint NLL.
- At most 500 epochs, existing early stopping, LR plateau scheduler, and best-epoch
  restoration. The established numerical trainer is unchanged.
- Refit the chosen settings with seeds 42–46. Rank encoder/decoder cases by their
  mean validation joint NLL. Fix the overall and per-decoder choices in
  `selection.json` **before** any official-test performance is computed.
- Evaluate every final seed on factual test CVs and counterfactual test CVs,
  including the non-identity subset separately. Report joint NLL, exact joint-MAP
  accuracy, joint Brier score, true marginal accuracy, ECE and reliability bins.
- Mean/sample SD over seeds is not a confidence interval over units. Do not pick
  a different winner using the test table and still call it an unbiased test.
- LangVAE-FT adaptation used 730 g-validation CVs for unsupervised training and
  70 for adaptation validation. Their g labels were not fitted, but validation
  texts were not untouched end-to-end. All official-test CVs were excluded from
  adaptation. This limitation appears in the generated report.

## Storage and recovery

Each case keeps all 20 trial checkpoints and five final seeds under
`models/talent/<encoder>/g-<variant>/experiments/<run-id>/`, with corresponding
reports under `reports/talent/<encoder>/g-<variant>/experiments/<run-id>/`.
After all cases pass validation and final evaluation, the predeclared seed-42
model is published to the canonical `semantic_decoder.pt`; any previous active
model/report is preserved in `archive/<run-id>/`. Publication is atomic per file,
not a single transaction spanning all 36 models. `published.json` records progress.

The root `reports/talent/decoder_benchmarks/<run-id>/` contains the protocol,
source snapshot, per-encoder downstream YAMLs, exact splits, training/evaluation
JSON, selection, status and Markdown report.

### Git tracking

The allowlist in `.gitignore` makes the following files eligible for committing:

- The 36 canonical `models/talent/<encoder>/g-<variant>/semantic_decoder.pt`
  checkpoints and their matching canonical `semantic_decoder.json` training
  reports. Seed 42 was chosen in advance; these are not the best test seeds.
- Benchmark `summary.md`, `results.json`, `evaluation.json`, `selection.json`,
  `protocol.json`, `plan.json`, `status.json`, `published.json` and `backups.json`.
- The 18 downstream configurations in `configs/`, all per-seed test metrics and
  case summaries in `evaluation/`, and the original code snapshot in `source/`.

Files still require `git add`, commit and push; changing the allowlist does not
perform those steps. Tuning and extra-seed checkpoints, per-fit experiment
reports, archives, logs, progress files, locks, h_Z models and encoder checkpoints
remain ignored. Historical JSON references to those local-only files are kept
as provenance; a clone is not a complete resumable copy of the 900-fit run.
The Markdown's completed-run model links use the canonical paths. Check their
hashes against `published.json`, especially if later training replaces them.

The LangVAE-FT encoder checkpoint is still a separate Hugging Face handoff, not
one of the 36 small g checkpoints. The existing latent loader requires it for
provenance verification; restore it at the recorded local path before using
LangVAE-FT downstream. This change does not weaken those checks.

### Where the statistics come from

g predicts the structured **X, T, D, U** labels from the already-saved
`data/latents/talent/<encoder>/z_pairs.pt` embeddings. Ground truth comes from
`data/sim_talent/sim_data_factual.csv` and `sim_data_counterfactual.csv`, aligned
by ID; it is not inferred from reconstructed CV text. The original CVs are in
`data/text_talent/cv_factual.csv` and `cv_counterfactual.csv`.

`data/sim_talent/pair_index.csv` defines the common 4,000/1,000 official
train/test split. Within the 4,000 training IDs, g uses 3,200 for fitting and a
fixed 800 for validation. Test performance is measured on 1,000 factual test
CVs. Separate diagnostics use the 1,000 counterfactual test CVs, with the 744
non-identity cases also reported separately; the other 256 are identity copies.
The two test populations share unit IDs and are not independent samples.

All paths in this table are relative to the benchmark run directory:

| Record | Contents |
|---|---|
| `results.json` | Each final seed's validation NLL, selected settings and validation mean/sample SD |
| `evaluation.json` | Test and counterfactual mean/sample SD across seeds 42–46, for all 36 cases |
| `evaluation/<encoder>/g-<variant>/seed-<seed>.json` | Individual-seed metrics, per-attribute scores, calibration bins and input/model hashes |
| `protocol.json`, `plan.json` | Exact fit/validation/test IDs, input hashes, configurations and common search candidates |
| `selection.json` | Validation-only ranking, frozen before test evaluation |
| `published.json` | Checksums and paths of the 36 active checkpoints and training reports |

Each seed is evaluated on the same test IDs. The reported ± is **sample standard
deviation across training seeds**, not a confidence interval and not variability
across different data splits. Per-CV predictions are not saved in this bundle;
the saved JSON contains aggregate metrics and reliability bins. Canonical
training reports additionally retain the active seed's learning curves,
validation calibration and exact split IDs.

The frozen `source/` snapshot and protocol are historical evidence; leave them
unchanged when updating the report renderer or Git policy after a completed run.

Re-running the launcher with the same run ID resumes only matching, verified
completed trials; an interrupted fit restarts from its seed. Independent cases
continue if another fails, but no final selection/publication happens until all
cases succeed. Source/config/input changes require a new run ID. Committing the
same source bytes does not invalidate the protocol. Do not edit training code,
configuration or input files while a run is active.

To regenerate the Markdown without training:

```bash
.venv/bin/python -m exp.benchmarks.semantic_decoder.run --report --run-id g-all-encoders-v1
```

The modules are separate from the established scripts: `decoder_benchmark_matrix`
handles discovery/splits, `decoder_benchmark` orchestration, `decoder_benchmark_evaluation`
held-out metrics, and `decoder_benchmark_report` Markdown. They reuse
`decoder_experiment.run_case`, `verify_run`, and `publish_results`.
