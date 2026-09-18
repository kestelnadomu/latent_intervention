# Counterfactual Latent Representations: A Neurosymbolic Approach

Research code and publication materials for the *latent intervention* project.

The pipeline builds counterfactual latent representations of text: an SCM simulates paired factual and counterfactual candidate attributes, an LLM verbalizes the approved worlds into CV personal statements, the configured frozen LangVAE or Nomic encoder maps them into 128-dimensional latents, a semantic decoder grounds those latents in the SCM variables, and a latent manipulator learns to transform them analogously to the SCM's counterfactual operation.

The synthetic SCM follows the CV Screening setup of the LIBERTy paper ([arXiv 2601.10700](https://arxiv.org/abs/2601.10700)). Text generation is a protocol-faithful adaptation of its Appendix D.3: seed personal statements are abstracted into narrative templates, personas are generated for sampled job titles, and each rendered world combines a fixed template, persona, and the unit's attribute values. This is not a literal replication of the released benchmark.

## Repository layout

| Path | Contents |
| --- | --- |
| `exp/sim/` | Data generation pipeline: Python SCM, codebook, generation prompts (`prompts/`), seed statements + job titles, LLM plumbing, stage runner, `config.yaml` (own README) |
| `exp/sim/R/` | Original SCM simulation in R, kept as the reference implementation (own README, renv) |
| `src/` | Encoder, semantic decoder, latent manipulator, training/eval pipeline, `config.yaml` (per-module hyperparameters) |
| `data/` | Generated artifacts (`sim/`, `text/`, `latents/` — only `text/` is tracked) and the older sampling pools |
| `poster/` | Quarto poster and slides |

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/). Run all commands from the repository root:

```bash
uv sync
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
| `Z=f(X)` | All units | `z` in `data/latents/talent/z_pairs.pt` |
| `Z'=f(X')` | Test units only | `z_prime` in `data/latents/talent/z_pairs.pt` |

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
uv run python -m src.pipeline encode             # configured frozen encoder: X plus test X' -> z_pairs.pt
uv run python -m src.pipeline train-decoder      # semantic decoder g: Z -> S + calibration report
uv run python -m src.pipeline train-manipulator  # selected h_Z variant on official training units
uv run python -m src.pipeline evaluate           # legacy report or flow recovery diagnostics
```

`pair_index.csv` is the sole train/test authority. `train-decoder` deterministically reserves a calibration holdout only from the official training IDs and writes per-column accuracy, ECE, and reliability bins to `reports/talent/semantic_decoder.json`; these measurements do not apply temperature scaling. `train-manipulator` uses all official training IDs, while `evaluate` uses only official test IDs.

Choose $h_Z$ with the single `latent_intervention.variant` value in `src/config.yaml`. The existing transformer variants remain available alongside `state_flow`, `distilled_flow`, and `direct_semantic_flow`. The state flow uses factual $S$ and $h_S$; the distilled and direct flows expose the standalone inference interface $(Z,\delta)\mapsto\Delta(\mathcal Z)$. Flow checkpoints and reports include the variant in their filename, and distilled training automatically creates or reuses a compatible state-flow teacher.

The established transformer stages and report format remain in `src/pipeline.py`; that file delegates only the three flow variants to `src/flow_workflow.py`. Flow architectures and objectives are isolated in `src/flow_intervention.py`, while flow evaluation adds paired-$Z'$ recovery and support diagnostics.

The active data currently supports only the configured `do(X=3)` query plus an explicitly trained no-op. Other schema-valid interventions are accepted for exploratory inference with a warning, but are outside training support. The configured `n: 10` is suitable only for an end-to-end smoke test of a 128-dimensional flow, not for a performance claim.

The pipeline binds latent, decoder, and manipulator artifacts to the configured encoder, source hashes, schema, and upstream artifact hashes. Pipeline loading rejects incompatible or incomplete metadata with an instruction to re-encode or retrain instead of silently mixing latent spaces.

When `encoder.variant: langvae` is selected, LangVAE may optionally be fine-tuned first (`uv run python -m src.finetune_vae`); point `encoder.local_checkpoint` in `src/config.yaml` at the resulting folder, then regenerate every downstream artifact.

## Publish

From root directory:

```{bash}
quarto publish gh-pages poster/poster.qmd
```

## Props

* Poster and slide template: [mpimet](https://github.com/mpimet/quarto/)
