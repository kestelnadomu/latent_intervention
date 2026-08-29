"""
Text generation from tabular samples via an OpenAI-compatible chat API.

LLM plumbing for the text-generation pipeline (exp/sim/run.py):
1. Load inputs: load_samples (CSV rows), load_prompts (chat prompt YAML),
   load_codebook_spec (column phrases + labels + bins).
2. Verbalize: candidate_info_from_row (row -> paper-style candidate info list,
   concrete values sampled inside binned columns), verbalize_row (row ->
   {col}_text placeholders).
3. Generate: generate_text (one API call, placeholders in the prompt are
   filled from the sample dict).

load_personas / load_templates / sample_* / generate_batch belong to the older
pool-sampling workflow around data/prompts.yaml and are kept for compatibility.
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dotenv
import numpy as np
import pandas as pd
import yaml
from openai import OpenAI

Sample = dict[str, Any]
PromptTemplate = dict[str, Any]
Message = dict[str, str]


@dataclass(frozen=True)
class GenerationResult:
    """Generated text plus the response details useful for an audit."""

    text: str
    response_id: str | None = None
    model: str | None = None
    system_fingerprint: str | None = None
    finish_reason: str | None = None

# Any OpenAI-compatible chat endpoint works, e.g. Mistral (default) or Azure
# OpenAI's v1 endpoint (https://<resource>.openai.azure.com/openai/v1/ with an
# AZURE_OPENAI_API_KEY and the deployment name as `model`). The endpoint is
# configured in exp/sim/config.yaml (`llm:` section).
DEFAULT_MODEL = "mistral-medium-latest"
DEFAULT_BASE_URL = "https://api.mistral.ai/v1"
API_KEY_ENV_VARS = ("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY", "MISTRAL_API_KEY")
dotenv.load_dotenv()  # load .env file if present

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_yaml(yaml_path: str | Path) -> Any:
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_string_list(yaml_path: str | Path, what: str) -> list[str]:
    data = _load_yaml(yaml_path)
    if not isinstance(data, list) or not all(isinstance(p, str) for p in data):
        raise ValueError(f"{what} YAML must be a list of strings, got {type(data)}")
    return data


def load_samples(csv_path: str | Path, **kwargs: Any) -> list[Sample]:
    """Load samples from CSV into a list of dicts (one per row)."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(path, **kwargs).to_dict(orient="records")


def load_codebook(yaml_path: str | Path) -> dict[str, dict[int, str]]:
    """
    Load a codebook mapping tabular columns to {category index: phrase}.

    Accepts both a flat {column: {0: "phrase", ...}, ...} YAML and the nested
    layout of exp/sim/codebook.yaml (columns under a top-level `columns:` key);
    returns only the column-to-phrase mapping either way.
    """
    data = _load_yaml(yaml_path)
    if isinstance(data, dict) and "columns" in data:
        data = data["columns"]
    if not isinstance(data, dict) or not all(isinstance(v, dict) for v in data.values()):
        raise ValueError(f"Codebook YAML must map columns to {{index: phrase}} dicts, got {type(data)}")
    return {str(col): {int(k): str(v) for k, v in levels.items()} for col, levels in data.items()}


def load_codebook_spec(yaml_path: str | Path) -> dict[str, Any]:
    """
    Load the full codebook spec (exp/sim/codebook.yaml).

    Returns {"columns": {col: {index: phrase}}, "labels": {col: display name},
    "bins": {col: {index: (lo, hi)}}}; `labels` and `bins` are optional in the
    YAML and default to empty.
    """
    data = _load_yaml(yaml_path)
    if not isinstance(data, dict) or "columns" not in data:
        raise ValueError(f"Codebook YAML must have a top-level 'columns' key, got {type(data)}")
    columns = {str(col): {int(k): str(v) for k, v in levels.items()} for col, levels in data["columns"].items()}
    labels = {str(col): str(label) for col, label in data.get("labels", {}).items()}
    bins: dict[str, dict[int, tuple[int, int]]] = {}
    for col, ranges in data.get("bins", {}).items():
        if str(col) not in columns:
            raise ValueError(f"bins column '{col}' not in codebook columns {list(columns)}")
        bins[str(col)] = {int(k): (int(lo), int(hi)) for k, (lo, hi) in ranges.items()}
    return {"columns": columns, "labels": labels, "bins": bins}


def verbalize_row(row: Sample, codebook: dict[str, dict[int, str]]) -> dict[str, str]:
    """
    Map one integer-coded tabular row to phrases via the codebook.

    Returns {"<column>_text": phrase} for every codebook column, ready to fill
    prompt placeholders like {E_text}.
    """
    verbalized = {}
    for col, levels in codebook.items():
        if col not in row:
            raise ValueError(f"Row is missing column '{col}'. Available: {list(row)}")
        value = int(row[col])
        if value not in levels:
            raise ValueError(f"No codebook phrase for {col}={value}. Available: {sorted(levels)}")
        verbalized[f"{col}_text"] = levels[value]
    return verbalized


def candidate_info_from_row(
    row: Sample,
    spec: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[str, dict[str, int]]:
    """
    Build the candidate information list for one row (LIBERTy App. D.3 style).

    Every codebook column is included, e.g. "[Race: White, Gender: Female,
    Age: 41, ...]". Binned columns (see load_codebook_spec) are rendered as a
    concrete integer sampled uniformly inside the row's bin via `rng` — pass a
    per-row seeded generator for reproducibility; the draws happen in codebook
    column order. Returns (info string, {binned column: sampled value}).
    """
    parts = []
    concrete: dict[str, int] = {}
    for col, levels in spec["columns"].items():
        if col not in row:
            raise ValueError(f"Row is missing column '{col}'. Available: {list(row)}")
        value = int(row[col])
        if value not in levels:
            raise ValueError(f"No codebook phrase for {col}={value}. Available: {sorted(levels)}")
        label = spec["labels"].get(col, col)
        if col in spec["bins"]:
            lo, hi = spec["bins"][col][value]
            concrete[col] = int(rng.integers(lo, hi + 1))
            parts.append(f"{label}: {concrete[col]}")
        else:
            parts.append(f"{label}: {levels[value]}")
    return "[" + ", ".join(parts) + "]", concrete


def load_personas(yaml_path: str | Path) -> list[str]:
    """Load persona_details entries from a YAML file containing a list of strings."""
    return _load_string_list(yaml_path, "Persona")


def load_templates(yaml_path: str | Path) -> list[str]:
    """Load cv_template entries from a YAML file containing a list of strings."""
    return _load_string_list(yaml_path, "Template")


def _prompt_messages(pdict: Any) -> list[Message]:
    """Build chat messages from a {system: ..., user: ...} mapping."""
    msgs = []
    if isinstance(pdict, dict):
        for role in ("system", "user"):
            if role in pdict:
                msgs.append({"role": role, "content": pdict[role]})
    return msgs


def load_prompts(yaml_path: str | Path) -> dict[str, Any]:
    """
    Load chat prompts and global templates from YAML.
    Returns {'prompts': [{'messages': [...], 'metadata': {...}}, ...], 'templates': {...}}

    Handles two structures:
    - Dict: {prompt: {system: ..., user: ..., metadata: ...}, <name>: "<template>", ...}
    - List: [{prompt: {system: ..., user: ...}, metadata: ...}, {<name>: "<template>"}, "bare user prompt", ...]
    """
    data = _load_yaml(yaml_path)

    prompts: list[PromptTemplate] = []
    templates: dict[str, str] = {}

    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, str):
                prompts.append({"messages": [{"role": "user", "content": entry}], "metadata": {}})
            elif isinstance(entry, dict):
                if "prompt" in entry:
                    prompts.append({
                        "messages": _prompt_messages(entry["prompt"]),
                        "metadata": entry.get("metadata", {}),
                    })
                else:
                    templates.update({k: v for k, v in entry.items() if isinstance(v, str)})
            else:
                raise ValueError(f"Invalid YAML entry type: {type(entry)}")

    elif isinstance(data, dict):
        if "prompt" in data:
            pdict = data["prompt"]
            metadata = pdict.get("metadata", {}) if isinstance(pdict, dict) else {}
            prompts.append({"messages": _prompt_messages(pdict), "metadata": metadata})
        templates.update({k: v for k, v in data.items() if k != "prompt" and isinstance(v, str)})

    else:
        raise ValueError(f"YAML must be list or dict, got {type(data)}")

    return {"prompts": prompts, "templates": templates}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _choice(items: list[str], what: str, seed: int | None) -> str:
    if not items:
        raise ValueError(f"{what} list is empty.")
    return random.Random(seed).choice(items)


def sample_persona(personas: list[str], seed: int | None = None) -> str:
    """Draw one persona_details entry uniformly at random."""
    return _choice(personas, "Persona", seed)


def sample_template(templates: list[str], seed: int | None = None) -> str:
    """Draw one cv_template entry uniformly at random."""
    return _choice(templates, "Template", seed)


def sample_candidate(
    samples: list[Sample],
    index: int | None = None,
    seed: int | None = None,
) -> str:
    """
    Pick one row from tabular samples (see load_samples) and return it as a JSON string.

    Selects the row at `index` if given, otherwise draws uniformly at random
    (optionally seeded).
    """
    if not samples:
        raise ValueError("Sample list is empty.")
    if index is None:
        index = random.Random(seed).randrange(len(samples))
    if not 0 <= index < len(samples):
        raise ValueError(f"Index {index} out of range for {len(samples)} samples.")
    return json.dumps(samples[index], ensure_ascii=False)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def format_messages(
    msgs: list[Message],
    sample: Sample,
    templates: dict[str, str] | None = None,
) -> list[Message]:
    """Fill {placeholder} fields in messages from sample data, then global templates."""
    data = {**(templates or {}), **sample}
    result = []
    for msg in msgs:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        try:
            result.append({"role": role, "content": content.format(**data)})
        except KeyError as e:
            raise ValueError(f"Missing key '{e}'. Available: {list(data.keys())}") from e
    return result


def generate_text_result(
    sample: Sample,
    prompt_template: PromptTemplate,
    templates: dict[str, str] | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    max_completion_tokens: int = 500,
    temperature: float = 0.7,
    **kwargs: Any,
) -> GenerationResult:
    """
    Generate text via an OpenAI-compatible chat completions API
    (Mistral by default; Azure OpenAI via its /openai/v1/ base_url, where
    `model` is the deployment name).

    Per-prompt `metadata` (model, max_completion_tokens, temperature) overrides the
    function defaults. The API key falls back to the environment variables
    in API_KEY_ENV_VARS.
    """
    if api_key is None:
        for env_var in API_KEY_ENV_VARS:
            api_key = os.getenv(env_var)
            if api_key:
                break
    if not api_key:
        raise ValueError(f"API key required. Set one of {'/'.join(API_KEY_ENV_VARS)} or pass api_key.")

    msgs = prompt_template.get("messages", [])
    if not msgs:
        raise ValueError("Prompt template must contain 'messages' key.")

    formatted = format_messages(msgs, sample, templates)
    metadata = prompt_template.get("metadata", {})

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=metadata.get("model", model),
            messages=formatted,
            max_completion_tokens=metadata.get("max_completion_tokens", max_completion_tokens),
            temperature=metadata.get("temperature", temperature),
            **kwargs,
        )
        if response.choices:
            choice = response.choices[0]
            text = choice.message.content
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("API returned a choice without non-empty text content.")
            return GenerationResult(
                text=text,
                response_id=getattr(response, "id", None),
                model=getattr(response, "model", None),
                system_fingerprint=getattr(response, "system_fingerprint", None),
                finish_reason=getattr(choice, "finish_reason", None),
            )
        raise RuntimeError("No text generated.")
    except Exception as e:
        raise RuntimeError(f"API request failed: {e}") from e


def generate_text(
    sample: Sample,
    prompt_template: PromptTemplate,
    templates: dict[str, str] | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    max_completion_tokens: int = 500,
    temperature: float = 0.7,
    **kwargs: Any,
) -> str:
    """Backward-compatible text-only wrapper around ``generate_text_result``."""
    return generate_text_result(
        sample,
        prompt_template,
        templates,
        api_key,
        model,
        base_url,
        max_completion_tokens,
        temperature,
        **kwargs,
    ).text


def generate_batch(
    csv_path: str | Path,
    yaml_path: str | Path,
    api_key: str | None = None,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> list[str]:
    """Generate text for the cross product of all samples x prompts."""
    samples = load_samples(csv_path)
    data = load_prompts(yaml_path)
    results = [
        generate_text(sample, prompt, data["templates"], api_key, **kwargs)
        for sample in samples
        for prompt in data["prompts"]
    ]
    if output_path:
        pd.DataFrame({"generated_text": results}).to_csv(Path(output_path), index=False)
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python generate.py <samples.csv> <prompts.yaml> [output.csv]")
        sys.exit(1)
    try:
        results = generate_batch(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
        print(f"Generated {len(results)} results.")
        for i, r in enumerate(results):
            print(f"\n--- Result {i+1} ---\n{r}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
