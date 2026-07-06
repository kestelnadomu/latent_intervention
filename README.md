# Counterfactual Latent Representations: A Neurosymbolic Approach

Research code and publication materials for the *latent intervention* project.

The current pipeline generates synthetic CV personal statements: candidate attributes, persona details, and narrative templates are sampled from small pools and combined into a prompt, which is sent to an LLM through an OpenAI-compatible chat API (Mistral by default).

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | Python generation pipeline (`generate.py`: loading, sampling, API calls) |
| `src/R/` | SCM simulation in R: generates factual and counterfactual data (documented in its own README) |
| `data/` | Sampling pools (`personas.yaml`, `candidates.csv`, `templates.yaml`) and the chat prompt (`prompts.yaml`) |
| `poster/` | Quarto poster and slides |

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
export MISTRAL_API_KEY=...   # or OPENAI_API_KEY
```

## Usage

Sample one input combination and generate a statement:

```python
from src.generate import (load_personas, load_templates, load_samples, load_prompts,
                          sample_persona, sample_template, sample_candidate, generate_text)

sample = {
    "persona_details": sample_persona(load_personas("data/personas.yaml")),
    "candidate_info": sample_candidate(load_samples("data/candidates.csv")),
    "cv_template": sample_template(load_templates("data/templates.yaml")),
}
prompts = load_prompts("data/prompts.yaml")
text = generate_text(sample, prompts["prompts"][0], prompts["templates"])
```

Or run the batch CLI over a CSV (one generation per row × prompt; every prompt placeholder must be a CSV column):

```bash
uv run python src/generate.py data/samples_cv.csv data/prompts.yaml [output.csv]
```

Note that each generation is a billed API call.

## Simulation

The R code under `src/R/` simulates factual and counterfactual data from a structural causal model (entry point: `sim_main.R`, run from the repository root). See the README there for details.

## Publish

From root directory:

```{bash}
quarto publish gh-pages poster/poster.qmd
```

## Props

* Poster and slide template: [mpimet](https://github.com/mpimet/quarto/)
