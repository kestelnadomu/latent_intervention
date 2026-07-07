"""
Data generation pipeline: SCM simulation -> LLM verbalization.

Stages (configured by exp/sim/config.yaml):
    simulate        write sim_data_{factual,counterfactual,epsilon}.csv
    generate-texts  verbalize factual rows via the codebook and an LLM into
                    CV personal statements (one billed API call per row;
                    resumes: rows whose id is already in the output are skipped)

Run from the repository root:
    uv run python -m exp.sim.run simulate
    uv run python -m exp.sim.run generate-texts
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from exp.sim import scm
from exp.sim.generate_text import (
    DEFAULT_BASE_URL,
    generate_text,
    load_codebook,
    load_prompts,
    load_samples,
    verbalize_row,
)

CONFIG_PATH = Path(__file__).parent / "config.yaml"


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


def stage_generate_texts(config: dict[str, Any]) -> None:
    """Verbalize factual rows into CV statements via the LLM (billed API calls)."""
    factual_csv = Path(config["paths"]["sim_dir"]) / "sim_data_factual.csv"
    out_path = Path(config["paths"]["texts"])
    rows = load_samples(factual_csv)
    codebook = load_codebook(config["codebook"])
    prompts = load_prompts(config["prompts"])
    prompt = prompts["prompts"][0]

    llm = config.get("llm", {})
    base_url = llm.get("base_url", DEFAULT_BASE_URL)
    if "<resource-name>" in base_url:
        raise ValueError("Fill in llm.base_url in exp/sim/config.yaml (placeholder resource name).")
    api_key = None
    if llm.get("api_key_env"):
        api_key = os.getenv(llm["api_key_env"])
        if not api_key:
            raise ValueError(f"Environment variable {llm['api_key_env']} (llm.api_key_env) is not set.")

    done_ids: set[int] = set()
    if out_path.exists():
        done_ids = set(pd.read_csv(out_path, usecols=["id"])["id"].astype(int))
    todo = [row for row in rows if int(row["id"]) not in done_ids]
    limit = config.get("generation", {}).get("limit")
    if limit is not None:
        todo = todo[: int(limit)]
    if not todo:
        print(f"nothing to do: {len(done_ids)}/{len(rows)} texts already in {out_path}")
        return
    print(f"generating {len(todo)} texts ({len(done_ids)} already done) -> {out_path}")
    print("each row is one billed API call")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with open(out_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["id", "text"])
        for i, row in enumerate(todo, 1):
            sample = verbalize_row(row, codebook)
            text = generate_text(
                sample, prompt, prompts["templates"], api_key=api_key, base_url=base_url
            )
            writer.writerow([int(row["id"]), text])
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")


STAGES = {
    "simulate": stage_simulate,
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
