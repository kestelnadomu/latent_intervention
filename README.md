# Counterfactual Latent Representations: A Neurosymbolic Approach

Research code and publication materials for the *latent intervention* project.

The pipeline builds counterfactual latent representations of text: an SCM simulates paired factual and counterfactual candidate attributes, an LLM verbalizes the approved worlds into CV personal statements, a frozen LangVAE encodes them into latents, a semantic decoder grounds the latents in the SCM variables, and a latent manipulator learns to transform latents analogously to the SCM's counterfactual operation.

The synthetic SCM follows the CV Screening setup of the LIBERTy paper ([arXiv 2601.10700](https://arxiv.org/abs/2601.10700)). Text generation is a protocol-faithful adaptation of its Appendix D.3: seed personal statements are abstracted into narrative templates, personas are generated for sampled job titles, and each rendered world combines a fixed template, persona, and the unit's attribute values. This is not a literal replication of the released benchmark.

## Repository layout

| Path | Contents |
| --- | --- |
| `exp/sim/` | Data generation pipeline: Python SCM, codebook, generation prompts (`prompts/`), seed statements + job titles, LLM plumbing, runner, `config.yaml` |
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

The experiment uses one fixed query, `do(G=1)`. Factual `G=0` units receive the nontrivial intervention; factual `G=1` units are identity pairs. It does not generate `do(G=0)`, an opposite-value query, or an orbit of interventions.

The seeded `pair_index.csv` contains each integer unit ID, its 80/20 train/test split, and whether the structured intervention is an identity. Artifact coverage is:

| Object | Coverage | Artifact |
| --- | --- | --- |
| `S`, `S'` | All units | `data/sim/sim_data_{factual,counterfactual}.csv` |
| `X` | All units | `data/text/cv_factual.csv` |
| `X'` | Test units only | `data/text/cv_counterfactual.csv` |
| `Z=f(X)` | All units | `z` in `data/latents/z_pairs.pt` |
| `Z'=f(X')` | Test units only | `z_prime` in `data/latents/z_pairs.pt` |

Identity test units copy `X'=X` without another LLM call and `Z'=Z` without another encoding. Nonidentity worlds reuse the same template, persona, and bin quantiles. `X'` and `Z'` are evaluation data and never enter the current training losses.

Data generation is configured by `exp/sim/config.yaml`. Every billed stage appends successful rows and resumes from missing IDs. Each active billed output CSV must remain together and be committed with its `*.generation.json` information file; the dated template CSV is only a legacy archive. `generation.limit` controls how many IDs are sent to the API per invocation; free identity copies do not count. Each selected ID permits at most `generation.max_attempts` API calls. Set `generation.limit: 1` for a small billed smoke test of at most three calls, inspect the result, then restore `null` and rerun the same stage to completion before advancing.

For a complete fresh run, execute these stages in order:

```bash
uv run python -m exp.sim.run simulate            # S/S', epsilon, pair_index.csv, simulation_info.json
uv run python -m exp.sim.run generate-templates  # seed statements -> data/text/templates.csv
uv run python -m exp.sim.run generate-personas   # job titles -> data/text/personas.csv
uv run python -m exp.sim.run generate-texts      # render_plan.csv and X for every unit
uv run python -m exp.sim.run generate-counterfactual-texts  # X' for test units only
uv run python -m exp.sim.run validate-pairs      # validate complete S/S' and X/X' pairing
```

With the current configuration and seeds, the completed run contains 50 templates, 100 personas, 100 factual CVs, and 20 counterfactual test CVs. Ten test counterfactuals are generated and ten identity cases are copied without an API call. This is 260 API calls if every request succeeds on its first attempt, and at most 780 for a completed run if failed or rejected attempts require all three tries.

The simulation-data pipeline is complete only when `validate-pairs` reports:

```text
validated 100 factual units and 20 counterfactual test pairs
```

Stop here for the data-generation handoff. Encoding and model training below are separate work.

Paired latent encoding and the existing training/evaluation stages use `src/config.yaml`:

```bash
uv run python -m src.pipeline encode             # X plus test X' -> one deterministic z_pairs.pt
uv run python -m src.pipeline train-decoder      # semantic decoder g: Z -> S
uv run python -m src.pipeline train-manipulator  # manipulator h_Z against frozen g and counterfactual targets
uv run python -m src.pipeline evaluate           # consistency accuracy + latent shift -> reports/eval.json
```

The model-training stages still use their pre-existing internal train/validation split; they do not yet consume `pair_index.csv`. Wiring the held-out pairing split into RQ1 training and evaluation remains separate model work, while the required `test_ids` and `z_prime` are already present in `z_pairs.pt`.

Optionally fine-tune the LangVAE on the generated CVs first (`uv run python -m src.finetune_vae`), then point `encoder.local_checkpoint` in `src/config.yaml` at the resulting folder.

## Publish

From root directory:

```{bash}
quarto publish gh-pages poster/poster.qmd
```

## Props

* Poster and slide template: [mpimet](https://github.com/mpimet/quarto/)
