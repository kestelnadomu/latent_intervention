"""Qwen3-Embedding-0.6B dimension queue: --preflight, --prepare, or --run.

Use .venv-qwen3/bin/python. Shares the validated serial artifact queue with Nomic.
"""

from src.encoders.nomic_encoding import main as queue_main


if __name__ == "__main__":
    queue_main(variant="qwen3")
