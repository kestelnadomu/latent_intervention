"""
Text generation from tabular samples using LLM APIs.

Functions:
1. load_samples(csv_path) - Load samples from CSV
2. load_prompts(yaml_path) - Load chat prompts + template definitions from YAML
3. generate_text(sample, prompt, templates, api_key) - Generate text via OpenAI client
4. generate_batch(csv_path, yaml_path, api_key) - Batch processing convenience
5. load_personas(yaml_path) / sample_persona(personas) - Persona details sampling
6. load_templates(yaml_path) / sample_template(templates) - CV template sampling
7. sample_candidate(samples) - Pick a candidate row and return it as a JSON string

YAML can be a list of entries or a single mapping with 'prompt' and template keys.
Templates from YAML are merged with sample data during formatting.
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml

Sample = Dict[str, Any]
PromptTemplate = Dict[str, Any]
Message = Dict[str, str]


def load_samples(csv_path: Union[str, Path], **kwargs: Any) -> List[Sample]:
    """Load samples from CSV into list of dicts."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(path, **kwargs).to_dict(orient="records")


def load_prompts(yaml_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load prompts and templates from YAML.
    Returns {'prompts': [...], 'templates': {...}}
    
    Handles two structures:
    - List: [{prompt: {system: ..., user: ...}, metadata: ...}, cv_template: "..."]
    - Dict: {prompt: {system: ..., user: ...}, metadata: ..., cv_template: "..."}
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    prompts = []
    templates: Dict[str, str] = {}
    
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, str):
                prompts.append({"messages": [{"role": "user", "content": entry}], "metadata": {}})
            elif isinstance(entry, dict):
                if "prompt" in entry:
                    pdict = entry["prompt"]
                    msgs = []
                    if isinstance(pdict, dict):
                        if "system" in pdict:
                            msgs.append({"role": "system", "content": pdict["system"]})
                        if "user" in pdict:
                            msgs.append({"role": "user", "content": pdict["user"]})
                    prompts.append({"messages": msgs, "metadata": entry.get("metadata", {})})
                else:
                    for key, value in entry.items():
                        if isinstance(value, str):
                            templates[key] = value
            else:
                raise ValueError(f"Invalid YAML entry type: {type(entry)}")
    
    elif isinstance(data, dict):
        if "prompt" in data:
            pdict = data["prompt"]
            msgs = []
            metadata = {}
            if isinstance(pdict, dict):
                if "system" in pdict:
                    msgs.append({"role": "system", "content": pdict["system"]})
                if "user" in pdict:
                    msgs.append({"role": "user", "content": pdict["user"]})
                if "metadata" in pdict:
                    metadata = pdict["metadata"]
            prompts.append({"messages": msgs, "metadata": metadata})
        for key, value in data.items():
            if key != "prompt" and isinstance(value, str):
                templates[key] = value
    else:
        raise ValueError(f"YAML must be list or dict, got {type(data)}")
    
    return {"prompts": prompts, "templates": templates}


def _load_string_list(yaml_path: Union[str, Path], what: str) -> List[str]:
    """Load a YAML file expected to contain a list of strings."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list) or not all(isinstance(p, str) for p in data):
        raise ValueError(f"{what} YAML must be a list of strings, got {type(data)}")
    return data


def load_personas(yaml_path: Union[str, Path]) -> List[str]:
    """Load persona_details entries from a YAML file containing a list of strings."""
    return _load_string_list(yaml_path, "Persona")


def load_templates(yaml_path: Union[str, Path]) -> List[str]:
    """Load cv_template entries from a YAML file containing a list of strings."""
    return _load_string_list(yaml_path, "Template")


def sample_persona(personas: List[str], seed: Optional[int] = None) -> str:
    """Draw one persona_details entry uniformly at random."""
    if not personas:
        raise ValueError("Persona list is empty.")
    return random.Random(seed).choice(personas)


def sample_template(templates: List[str], seed: Optional[int] = None) -> str:
    """Draw one cv_template entry uniformly at random."""
    if not templates:
        raise ValueError("Template list is empty.")
    return random.Random(seed).choice(templates)


def sample_candidate(
    samples: List[Sample],
    index: Optional[int] = None,
    seed: Optional[int] = None,
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


def format_messages(msgs: List[Message], sample: Sample, templates: Optional[Dict[str, str]] = None) -> List[Message]:
    """Format message templates with sample data + global templates."""
    data = sample.copy()
    if templates:
        for k, v in templates.items():
            if k not in data:
                data[k] = v
    result = []
    for msg in msgs:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        try:
            result.append({"role": role, "content": content.format(**data)})
        except KeyError as e:
            raise ValueError(f"Missing key '{e}'. Available: {list(data.keys())}") from e
    return result


def generate_text(
    sample: Sample,
    prompt_template: PromptTemplate,
    templates: Optional[Dict[str, str]] = None,
    api_key: Optional[str] = None,
    model: str = "mistral-medium-latest",
    base_url: str = "https://api.mistral.ai/v1",
    max_tokens: int = 500,
    temperature: float = 0.7,
    **kwargs: Any,
) -> str:
    """Generate text via an OpenAI-compatible chat completions API."""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("MISTRAL_API_KEY")
    if api_key is None:
        raise ValueError("API key required. Set OPENAI_API_KEY / MISTRAL_API_KEY or pass api_key.")
    
    msgs = prompt_template.get("messages", [])
    if not msgs:
        raise ValueError("Prompt template must contain 'messages' key.")
    
    formatted = format_messages(msgs, sample, templates)
    metadata = prompt_template.get("metadata", {})
    
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        response = client.chat.completions.create(
            model=metadata.get("model", model),
            messages=formatted,
            max_tokens=metadata.get("max_tokens", max_tokens),
            temperature=metadata.get("temperature", temperature),
            **kwargs,
        )
        if response.choices:
            return response.choices[0].message.content
        raise RuntimeError("No text generated.")
    except Exception as e:
        raise RuntimeError(f"API request failed: {e}") from e


def generate_batch(
    csv_path: Union[str, Path],
    yaml_path: Union[str, Path],
    api_key: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> List[str]:
    """Batch generate text for all samples x prompts."""
    samples = load_samples(csv_path)
    data = load_prompts(yaml_path)
    results = []
    for sample in samples:
        for prompt in data["prompts"]:
            results.append(generate_text(sample, prompt, data["templates"], api_key, **kwargs))
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