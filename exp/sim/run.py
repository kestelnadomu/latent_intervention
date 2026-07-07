"""
Data generation pipeline: SCM simulation -> LLM verbalization.

Text generation follows Appendix D.3 of the LIBERTy paper (arXiv 2601.10700),
whose CV Screening SCM this simulation reproduces: seed personal statements are
abstracted into narrative templates, personas are generated for sampled job
titles, and each factual SCM row is turned into a CV Personal Statement by
combining a template, a persona, and the row's candidate information.

Stages (configured by exp/sim/config.yaml; every generate-* row is one billed
API call and resumes by skipping ids already present in its output CSV):
    simulate            write sim_data_{factual,counterfactual,epsilon}.csv
    generate-templates  seed statements -> narrative template pool
    generate-personas   job titles -> persona pool
    generate-texts      factual rows + pools -> CV personal statements

Run from the repository root:
    uv run python -m exp.sim.run simulate
    uv run python -m exp.sim.run generate-templates
    uv run python -m exp.sim.run generate-personas
    uv run python -m exp.sim.run generate-texts
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from exp.sim import scm
from exp.sim.generate_text import (
    DEFAULT_BASE_URL,
    candidate_info_from_row,
    generate_text,
    load_codebook_spec,
    load_prompts,
    load_samples,
)

CONFIG_PATH = Path(__file__).parent / "config.yaml"

# Seed-sequence namespaces keeping the per-id rng streams of the two sampling
# stages disjoint: default_rng([seed, namespace, id]).
RNG_PERSONAS, RNG_TEXTS = 1, 2


def load_sim_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the data generation config (exp/sim/config.yaml)."""
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stage_simulate(config: dict[str, Any]) -> None:
    """Generate factual/counterfactual/epsilon CSVs from the Python SCM."""
    factual, counterfactual, epsilon = scm.simulate(
        n=int(config["n"]),
        intervention=config["intervention"],
        seed=config.get("seed"),
    )
    sim_dir = Path(config["paths"]["sim_dir"])
    sim_dir.mkdir(parents=True, exist_ok=True)
    for name, df in [("factual", factual), ("counterfactual", counterfactual), ("epsilon", epsilon)]:
        out = sim_dir / f"sim_data_{name}.csv"
        df.to_csv(out, index=False)
        print(f"wrote {out} ({len(df)} rows)")
    changed = (factual.drop(columns="id") != counterfactual.drop(columns="id")).any(axis=1)
    print(f"intervention {config['intervention']}: {changed.mean():.1%} of rows changed")


def _load_string_list(yaml_path: str | Path, what: str) -> list[str]:
    """Load a YAML file that must contain a non-empty list of strings."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"{what} YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list) or not data or not all(isinstance(s, str) for s in data):
        raise ValueError(f"{what} YAML must be a non-empty list of strings: {path}")
    return data


def _resolve_llm(config: dict[str, Any]) -> tuple[str, str | None]:
    """Validate the llm: config section and return (base_url, api_key)."""
    llm = config.get("llm", {})
    base_url = llm.get("base_url", DEFAULT_BASE_URL)
    if "<resource-name>" in base_url:
        raise ValueError("Fill in llm.base_url in exp/sim/config.yaml (placeholder resource name).")
    api_key = None
    if llm.get("api_key_env"):
        api_key = os.getenv(llm["api_key_env"])
        if not api_key:
            raise ValueError(f"Environment variable {llm['api_key_env']} (llm.api_key_env) is not set.")
    return base_url, api_key


def _done_ids(out_path: Path, id_column: str) -> set[int]:
    """Ids already present in a (possibly nonexistent) output CSV."""
    if not out_path.exists():
        return set()
    return set(pd.read_csv(out_path, usecols=[id_column])[id_column].astype(int))


def _generate_rows(
    config: dict[str, Any],
    out_path: Path,
    header: list[str],
    todo: list[tuple[dict[str, Any], list[Any]]],
    prompt_path: str | Path,
    n_total: int,
    what: str,
) -> None:
    """
    Shared generate-* loop: one billed API call per (sample, prefix) item,
    appending `prefix + [text]` rows to out_path with flush after each row.
    """
    n_done = n_total - len(todo)
    limit = config.get("generation", {}).get("limit")
    if limit is not None:
        todo = todo[: int(limit)]
    if not todo:
        print(f"nothing to do: {n_done}/{n_total} {what} already in {out_path}")
        return
    print(f"generating {len(todo)} {what} ({n_done} already done) -> {out_path}")
    print("each row is one billed API call")

    prompts = load_prompts(prompt_path)
    prompt = prompts["prompts"][0]
    base_url, api_key = _resolve_llm(config)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with open(out_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        for i, (sample, prefix) in enumerate(todo, 1):
            text = generate_text(sample, prompt, prompts["templates"], api_key=api_key, base_url=base_url)
            writer.writerow([*prefix, text])
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")


def stage_generate_templates(config: dict[str, Any]) -> None:
    """Abstract seed statements into narrative templates (Box D.12)."""
    seeds = _load_string_list(config["pools"]["seed_statements"], "Seed statement")
    n_templates = int(config["generation"]["n_templates"])
    out_path = Path(config["paths"]["templates"])
    done = _done_ids(out_path, "template_id")
    todo = []
    for template_id in range(1, n_templates + 1):
        if template_id in done:
            continue
        seed_id = (template_id - 1) % len(seeds) + 1  # round-robin over seeds
        todo.append(({"sampled_statement": seeds[seed_id - 1]}, [template_id, seed_id]))
    _generate_rows(
        config, out_path, ["template_id", "seed_id", "text"], todo,
        config["prompts"]["templates"], n_templates, "templates",
    )


def stage_generate_personas(config: dict[str, Any]) -> None:
    """Generate personas for job titles sampled from the predefined list (Box D.13)."""
    titles = _load_string_list(config["pools"]["job_titles"], "Job title")
    n_personas = int(config["generation"]["n_personas"])
    seed = int(config["seed"])
    out_path = Path(config["paths"]["personas"])
    done = _done_ids(out_path, "persona_id")
    todo = []
    for persona_id in range(1, n_personas + 1):
        if persona_id in done:
            continue
        rng = np.random.default_rng([seed, RNG_PERSONAS, persona_id])
        job_title = titles[int(rng.integers(len(titles)))]
        todo.append(({"job_title": job_title}, [persona_id, job_title]))
    _generate_rows(
        config, out_path, ["persona_id", "job_title", "text"], todo,
        config["prompts"]["personas"], n_personas, "personas",
    )


def _load_pool(path: str | Path, id_column: str, stage: str) -> pd.DataFrame:
    """Load a generated pool CSV, failing with a pointer to the producing stage."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pool CSV not found: {path}. Run `uv run python -m exp.sim.run {stage}` first.")
    pool = pd.read_csv(path)
    if pool.empty:
        raise ValueError(f"Pool CSV is empty: {path}. Run `uv run python -m exp.sim.run {stage}` first.")
    return pool.sort_values(id_column).reset_index(drop=True)


def stage_generate_texts(config: dict[str, Any]) -> None:
    """Verbalize factual rows into CV statements from template + candidate info + persona (Box D.14)."""
    factual_csv = Path(config["paths"]["sim_dir"]) / "sim_data_factual.csv"
    rows = load_samples(factual_csv)
    spec = load_codebook_spec(config["codebook"])
    templates = _load_pool(config["paths"]["templates"], "template_id", "generate-templates")
    personas = _load_pool(config["paths"]["personas"], "persona_id", "generate-personas")
    seed = int(config["seed"])
    out_path = Path(config["paths"]["texts"])
    done = _done_ids(out_path, "id")

    todo = []
    for row in rows:
        row_id = int(row["id"])
        if row_id in done:
            continue
        # Per-row stream: pairing and concrete values are reproducible across
        # resumed runs and reusable for counterfactual text generation. Draw
        # order is fixed: template, persona, then bins (codebook column order).
        rng = np.random.default_rng([seed, RNG_TEXTS, row_id])
        template = templates.iloc[int(rng.integers(len(templates)))]
        persona = personas.iloc[int(rng.integers(len(personas)))]
        candidate_info, concrete = candidate_info_from_row(row, spec, rng)
        sample = {
            "cv_template": template["text"],
            "candidate_info": candidate_info,
            "persona_details": f"Job Title: {persona['job_title']}\n{persona['text']}",
        }
        prefix = [row_id, int(template["template_id"]), int(persona["persona_id"]),
                  concrete.get("A"), concrete.get("W")]
        todo.append((sample, prefix))
    _generate_rows(
        config, out_path, ["id", "template_id", "persona_id", "age", "years_experience", "text"],
        todo, config["prompts"]["cv"], len(rows), "texts",
    )


STAGES = {
    "simulate": stage_simulate,
    "generate-templates": stage_generate_templates,
    "generate-personas": stage_generate_personas,
    "generate-texts": stage_generate_texts,
}


def main() -> None:
    """CLI entry point for the data generation pipeline."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=STAGES, help="pipeline stage to run")
    parser.add_argument("--config", default=CONFIG_PATH, help="path to the sim config YAML")
    args = parser.parse_args()
    STAGES[args.stage](load_sim_config(args.config))


if __name__ == "__main__":
    main()
