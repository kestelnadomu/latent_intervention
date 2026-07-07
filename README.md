# Counterfactual Latent Representations: A Neurosymbolic Approach

Research code and publication materials for the *latent intervention* project.

The pipeline builds counterfactual latent representations of text: an SCM simulates factual and counterfactual candidate attributes, an LLM verbalizes the factual rows into CV personal statements, a frozen LangVAE encodes them into latents, a semantic decoder grounds the latents in the SCM variables, and a latent manipulator learns to transform latents analogously to the SCM's counterfactual operation.

The synthetic data reproduces the CV Screening dataset of the LIBERTy paper ([arXiv 2601.10700](https://arxiv.org/abs/2601.10700)): the SCM is its Table 9, and text generation follows its Appendix D.3 — seed personal statements are abstracted into narrative templates, personas are generated for sampled job titles, and each factual row is verbalized by combining a template, a persona, and the row's attribute values (with concrete age/years sampled inside their bins).

## Repository layout

| Path | Contents |
| --- | --- |
| `exp/sim/` | Data generation pipeline: Python SCM, codebook, generation prompts (`prompts/`), seed statements + job titles, LLM plumbing, runner, `config.yaml` |
| `exp/sim/R/` | Original SCM simulation in R, kept as the reference implementation (own README, renv) |
| `src/` | Encoder, semantic decoder, latent manipulator, training/eval pipeline, `config.yaml` (per-module hyperparameters) |
| `data/` | Generated artifacts (`sim/`, `text/`, `latents/` — only `text/` is tracked) and the older sampling pools |
| `poster/` | Quarto poster and slides |

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
export AZURE_OPENAI_API_KEY=...   # key for the endpoint configured in exp/sim/config.yaml (llm:)
```

Text generation talks to any OpenAI-compatible chat endpoint. Set `llm.base_url` in `exp/sim/config.yaml` (e.g. your Azure OpenAI resource's `/openai/v1/` endpoint, with the deployment name as `model` in the `exp/sim/prompts/*.yaml` metadata) and `llm.api_key_env` to the environment variable holding the key.

## Pipeline

Data generation (configured by `exp/sim/config.yaml`; the intervention defaults to do(G=1)). Every `generate-*` row is one billed API call; stages resume by skipping ids already in their output CSV and can be capped via `generation.limit`:

```bash
uv run python -m exp.sim.run simulate            # SCM -> data/sim/sim_data_{factual,counterfactual,epsilon}.csv
uv run python -m exp.sim.run generate-templates  # seed statements -> data/text/templates.csv
uv run python -m exp.sim.run generate-personas   # job titles -> data/text/personas.csv
uv run python -m exp.sim.run generate-texts      # template + candidate info + persona -> data/text/cv_factual.csv
```

Training and evaluation (hyperparameters in `src/config.yaml`):

```bash
uv run python -m src.pipeline encode             # texts -> latents (downloads the LangVAE checkpoint on first use)
uv run python -m src.pipeline train-decoder      # semantic decoder g: Z -> S
uv run python -m src.pipeline train-manipulator  # manipulator h_Z against frozen g and counterfactual targets
uv run python -m src.pipeline evaluate           # consistency accuracy + latent shift -> reports/eval.json
```

Optionally fine-tune the LangVAE on the generated CVs first (`uv run python -m src.finetune_vae`), then point `encoder.local_checkpoint` in `src/config.yaml` at the resulting folder.

## Publish

From root directory:

```{bash}
quarto publish gh-pages poster/poster.qmd
```

## Props

* Poster and slide template: [mpimet](https://github.com/mpimet/quarto/)
