"""Prepare/run the isolated oracle-target queue: 768-D once, derive 256-D.

Preparation never loads an encoder. Run with .venv-embeddinggemma only when the
additional inference is requested. Existing canonical z_pairs files are read-only.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.artifact_io import sha256_file, write_json
from src.config import CONFIG_PATH, load_config
from src.oracle_regression import ORACLE_VARIANT
from src.oracle_targets import (
    derive_oracle_targets,
    encode_oracle_targets,
    load_oracle_targets,
    source_metadata,
    target_path,
    training_texts,
)
from src.pair_encoding import load_latent_artifact


def configurations(config_path=CONFIG_PATH):
    return {
        dim: load_config(
            path=config_path,
            encoder_variant="embeddinggemma",
            embeddinggemma_dim=dim,
            manipulator_variant=ORACLE_VARIANT,
        )
        for dim in (768, 256)
    }


def build_plan(configs):
    from src.nomic_encoding import compare_projection

    artifacts = {dim: load_latent_artifact(config) for dim, config in configs.items()}
    compare_projection(artifacts[768], artifacts[256])
    ids, identity, _, _ = training_texts(configs[768], artifacts[768])
    outputs = []
    for dimension, config in configs.items():
        if target_path(config).parent.exists():
            load_oracle_targets(config, artifacts[dimension])
        outputs.append(
            {
                "dimension": dimension,
                "path": str(target_path(config)),
                "sources": source_metadata(config, artifacts[dimension]),
            }
        )
    files = [
        "src/oracle_encoding.py",
        "src/oracle_targets.py",
        "src/oracle_regression.py",
        "src/pair_encoding.py",
        "src/latent_writer.py",
        "src/artifact_io.py",
        "src/encoder_protocols.py",
        "src/config.py",
        "src/embeddinggemma_encoder.py",
        "src/nomic_encoding.py",
    ]
    return {
        "format_version": 1,
        "purpose": "oracle_regression_only",
        "training_units": len(ids),
        "identity_copies": int(identity.sum()),
        "nonidentity_texts_to_encode": int((~identity).sum()),
        "official_test_units_excluded": len(artifacts[768].test_ids),
        "encoder": configs[768]["encoder"],
        "outputs": outputs,
        "source_sha256": {name: sha256_file(name) for name in files},
        "resume_policy": "Reuse verified completed dimensions; an interrupted native pass restarts.",
    }


def prepare(configs, root):
    plan = build_plan(configs)
    path = root / "plan.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError(
                "oracle encoding code/config/inputs changed; use a fresh --run-id"
            )
    else:
        write_json(path, plan)
    return plan


def run(configs, root, *, encoder_factory=None):
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / ".encoding.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(configs, root)
        completed = []

        def status(state, **extra):
            write_json(
                root / "status.json",
                {
                    "status": state,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "completed_dimensions": list(completed),
                    **extra,
                },
            )

        status("running")
        try:
            runtime_config = copy.deepcopy(configs[768])
            runtime_config["encoder"]["progress"] = True
            encode_oracle_targets(runtime_config, encoder_factory=encoder_factory)
            completed.append(768)
            status("running")
            derive_oracle_targets(configs[256], configs[768])
            completed.append(256)
            # Only the progress verbosity flag changed in memory; it is not an
            # encoder-protocol change. Recheck the immutable source plan on disk.
            for name, digest in plan["source_sha256"].items():
                if sha256_file(name) != digest:
                    raise ValueError(
                        "oracle encoding source code changed during the run"
                    )
            for config in configs.values():
                load_oracle_targets(config)
            status("complete")
        except BaseException as exc:
            status("failed", error=str(exc), error_type=type(exc).__name__)
            raise
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run", action="store_true")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--run-id", default="oracle-targets-v1")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id) or args.threads < 1:
        parser.error("run-id must be a simple name and threads must be positive")
    torch.set_num_threads(args.threads)
    root = Path("reports/talent/oracle_targets") / args.run_id
    configs = configurations(args.config)
    if args.prepare:
        print(json.dumps(prepare(configs, root), indent=2))
        print("Prepared only; no new embeddings or models were generated.")
    else:
        run(configs, root)
        print(f"Oracle targets complete. Status: {root / 'status.json'}")


if __name__ == "__main__":
    main()
