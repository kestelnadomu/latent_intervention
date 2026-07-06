"""
Text generation from tabular samples using LLM APIs.

This module provides functions to:
1. Load samples from a CSV file
2. Load prompts from a YAML file
3. Generate text using an LLM API
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml

# Type aliases for better documentation
Sample = Dict[str, Any]
PromptTemplate = Dict[str, Any]


def load_samples(csv_path: Union[str, Path], **kwargs: Any) -> List[Sample]:
    """
    Load samples from a CSV file into a list of dictionaries.
    
    Args:
        csv_path: Path to the CSV file containing the samples.
        **kwargs: Additional keyword arguments passed to pandas.read_csv().
        
    Returns:
        List of dictionaries where each dictionary represents a row/sample.
        
    Raises:
        FileNotFoundError: If the CSV file does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
        
    Example:
        >>> samples = load_samples("data/samples.csv")
        >>> print(samples[0])
        {'text': 'Hello world', 'label': 1}
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(path, **kwargs)
    
    # Convert DataFrame to list of dictionaries
    samples = df.to_dict(orient="records")
    
    return samples


def load_prompts(yaml_path: Union[str, Path]) -> List[PromptTemplate]:
    """
    Load prompt templates from a YAML file.
    
    The YAML file should contain a list of prompt templates, where each template
    can be a string or a dictionary with 'template' and optional 'metadata' keys.
    
    Args:
        yaml_path: Path to the YAML file containing the prompts.
        
    Returns:
        List of prompt templates (as dictionaries).
        
    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the YAML file is malformed.
        
    Example YAML structure:
        ```yaml
        - template: "Generate text based on: {sample}"
          metadata:
            model: "mistral-medium"
            max_tokens: 500
        - template: "Summarize: {sample}"
          metadata:
            model: "mistral-small"
            max_tokens: 200
        ```
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    
    if prompts is None:
        prompts = []
    
    # Normalize prompts to list of dictionaries
    normalized_prompts = []
    for prompt in prompts:
        if isinstance(prompt, str):
            normalized_prompts.append({"template": prompt})
        elif isinstance(prompt, dict):
            normalized_prompts.append(prompt)
        else:
            raise ValueError(
                f"Invalid prompt type: {type(prompt)}. Expected str or dict."
            )
    
    return normalized_prompts


def generate_text(
    sample: Sample,
    prompt_template: PromptTemplate,
    api_key: Optional[str] = None,
    model: str = "mistral-medium-latest",
    base_url: str = "https://api.mistral.ai/v1",
    max_tokens: int = 500,
    temperature: float = 0.7,
    **kwargs: Any,
) -> str:
    """
    Generate text by prompting an LLM with a sample and prompt template.
    
    Args:
        sample: A dictionary containing the sample data to use for generation.
        prompt_template: A dictionary with at least a 'template' key containing
            the prompt template string. Can also contain 'metadata' with
            model-specific settings.
        api_key: The API key for the LLM provider. If None, will try to get
            from MISTRAL_API_KEY environment variable.
        model: The model identifier to use for generation.
        base_url: The base URL for the API endpoint.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 to 1.0).
        **kwargs: Additional keyword arguments passed to the API request.
        
    Returns:
        The generated text from the LLM.
        
    Raises:
        ValueError: If api_key is None and MISTRAL_API_KEY is not set.
        RuntimeError: If the API request fails.
        
    Example:
        >>> sample = {"text": "Hello world"}
        >>> prompt = {"template": "Expand on this: {text}"}
        >>> result = generate_text(sample, prompt, api_key="sk-...")
        >>> print(result)
    """
    # Get API key
    if api_key is None:
        api_key = os.getenv("MISTRAL_API_KEY")
    
    if api_key is None:
        raise ValueError(
            "API key is required. Either pass it as an argument or set "
            "the MISTRAL_API_KEY environment variable."
        )
    
    # Format the prompt with the sample data
    template_str = prompt_template.get("template", "{sample}")
    
    try:
        formatted_prompt = template_str.format(**sample)
    except KeyError as e:
        raise ValueError(
            f"Sample data is missing key required by prompt template: {e}"
        ) from e
    
    # Get model-specific settings from prompt metadata
    metadata = prompt_template.get("metadata", {})
    model = metadata.get("model", model)
    max_tokens = metadata.get("max_tokens", max_tokens)
    temperature = metadata.get("temperature", temperature)
    
    # Import mistralai client
    try:
        from mistralai.client import MistralClient
    except ImportError:
        raise ImportError(
            "The 'mistralai' package is required. Install it with: uv pip install mistralai"
        )
    
    # Create client and make request
    client = MistralClient(api_key=api_key, endpoint=base_url)
    
    try:
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": formatted_prompt,
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        
        # Extract the generated text
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            raise RuntimeError("No text was generated by the API.")
    
    except Exception as e:
        raise RuntimeError(f"API request failed: {e}") from e


def generate_batch(
    csv_path: Union[str, Path],
    yaml_path: Union[str, Path],
    api_key: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> List[str]:
    """
    Convenience function to generate text for all samples using all prompts.
    
    Args:
        csv_path: Path to the CSV file with samples.
        yaml_path: Path to the YAML file with prompt templates.
        api_key: API key for the LLM provider.
        output_path: Optional path to save results as CSV.
        **kwargs: Additional arguments passed to generate_text().
        
    Returns:
        List of generated texts (one per sample-prompt combination).
        
    Example:
        >>> results = generate_batch("data/samples.csv", "data/prompts.yaml")
    """
    samples = load_samples(csv_path)
    prompts = load_prompts(yaml_path)
    
    results = []
    for sample in samples:
        for prompt in prompts:
            text = generate_text(
                sample,
                prompt,
                api_key=api_key,
                **kwargs,
            )
            results.append(text)
    
    # Save to CSV if output_path is provided
    if output_path is not None:
        df = pd.DataFrame({"generated_text": results})
        df.to_csv(Path(output_path), index=False)
    
    return results


if __name__ == "__main__":
    # Example usage when run directly
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python generate.py <samples.csv> <prompts.yaml> [output.csv]")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    yaml_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        results = generate_batch(csv_path, yaml_path, output_path=output_path)
        print(f"Generated {len(results)} results.")
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(result)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
