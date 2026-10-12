"""Snapshot/compare existing results across a refactor, without encoder inference.

python -m tools.artifact_audit --snapshot reports/talent/refactor/before.json
python -m tools.artifact_audit --compare reports/talent/refactor/before.json

Only an explicitly requested new snapshot is written. Production artifacts are read-only.
The small synthetic training check creates no model files and tests both g optimizers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.artifact_io import write_json
from src.config import load_config
from exp.encoding.queue import dimension_configs, verify_artifact
from src.encoder.matryoshka import DIMENSIONS, compare_projection
from src.pair_encoding import load_latent_artifact, sha256_file
from src.pipeline import _decoder_metadata
from src.schema import ColumnSpec, load_schema
from src.semantic_decoder.model import (
    load_semantic_decoder,
    make_semantic_decoder,
    train_semantic_decoder,
)


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def synthetic_training() -> dict:
    """Tiny deterministic numerical fingerprint, not retraining any production model."""
    z = torch.randn(16, 3, generator=torch.Generator().manual_seed(19))
    targets = {"x": torch.arange(16) % 2, "y": torch.arange(16) % 3}
    results = {}
    for variant in ("independent", "autoregressive"):
        model = make_semantic_decoder(
            variant,
            latent_dim=3,
            columns=[ColumnSpec("x", 2), ColumnSpec("y", 3)],
            hidden_dim=8,
            n_hidden=1,
            dropout=0.1,
            embed_dim=2,
        )
        history = train_semantic_decoder(
            model,
            z,
            targets,
            epochs=4,
            batch_size=4,
            seed=13,
            verbose=False,
            validation_latents=z,
            validation_targets=targets,
            weight_decay=1e-5,
        )
        summary = {
            key: value
            for key, value in model.training_summary.items()
            if key not in {"seconds", "history"}
        }
        records = [
            {key: value for key, value in row.items() if key != "seconds"}
            for row in model.training_summary["history"]
        ]
        results[variant] = {
            "losses": history,
            "summary": summary,
            "history": records,
            "weights": {
                key: tensor_hash(value) for key, value in model.state_dict().items()
            },
        }
    return results


def snapshot() -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    results = {
        "latents": {},
        "decoders": {},
        "synthetic_training": synthetic_training(),
    }
    configs = [load_config(encoder_variant="langvae")]
    for variant in DIMENSIONS:
        per_width = dimension_configs(variant=variant)
        source = verify_artifact(per_width[max(per_width)])
        for dim, config in per_width.items():
            artifact = verify_artifact(config)
            if dim != max(per_width):
                compare_projection(source, artifact)
        configs.extend(per_width.values())
    for config in configs:
        artifact = load_latent_artifact(config)
        path = Path(config["paths"]["latents"])
        results["latents"][str(path)] = {
            "sha256": artifact.artifact_sha256,
            "sidecar_sha256": sha256_file(path.with_suffix(".info.json")),
            "encoder_info": artifact.encoder_info,
            "resolved_config": load_config(
                encoder_variant=config["encoder"]["variant"],
                **(
                    {f"{config['encoder']['variant']}_dim": artifact.z.shape[1]}
                    if config["encoder"]["variant"] != "langvae"
                    else {}
                ),
            ),
            "factual_shape": list(artifact.z.shape),
            "test_shape": list(artifact.z_prime.shape),
        }
    for encoder in ("langvae", "nomic"):
        for decoder in ("independent", "autoregressive"):
            config = load_config(encoder_variant=encoder, decoder_variant=decoder)
            path = Path(config["paths"]["decoder_model"])
            artifact = load_latent_artifact(config)
            columns, _ = load_schema(config["sim_config"])
            model = load_semantic_decoder(
                path,
                expected_variant=decoder,
                expected_columns=columns,
                expected_metadata=_decoder_metadata(config, artifact),
            )
            model.eval()
            with torch.inference_mode():
                predictions = model.log_joint(artifact.z[:32])
            results["decoders"][str(path)] = {
                "sha256": sha256_file(path),
                "report_sha256": sha256_file(config["paths"]["decoder_report"]),
                "joint_predictions_sha256": tensor_hash(predictions),
                "metadata": model.checkpoint_metadata,
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--snapshot", type=Path)
    mode.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.snapshot and args.snapshot.exists():
        raise FileExistsError(
            f"refusing to overwrite reference snapshot: {args.snapshot}"
        )
    result = snapshot()
    if args.snapshot:
        write_json(args.snapshot, result)
        print(f"Saved read-only artifact baseline: {args.snapshot}")
    else:
        expected = json.loads(args.compare.read_text())
        for section in result:
            if result[section] != expected[section]:
                raise ValueError(
                    f"refactor changed {section}; inspect the reference snapshot"
                )
        print(
            "PASS: artifacts, sidecars, resolved settings, g predictions and deterministic training fingerprints are unchanged."
        )
    print(
        f"Checked {len(result['latents'])} latent spaces and {len(result['decoders'])} active decoders; no encoder inference or production training."
    )


if __name__ == "__main__":
    main()
