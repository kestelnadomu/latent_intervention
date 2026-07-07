"""
Training and evaluation pipeline: text latents -> semantic decoder -> manipulator.

Stages (hyperparameters from src/config.yaml; data from exp/sim/run.py):
    encode             encode generated CV texts into latents (data/latents/)
    train-decoder      train the semantic decoder g: Z -> S on factual pairs
    train-manipulator  train the latent manipulator h_Z against frozen g and
                       counterfactual targets S' from the SCM simulation
    evaluate           manipulator faithfulness on the validation split

Run from the repository root:
    uv run python -m src.pipeline encode
    uv run python -m src.pipeline train-decoder
    uv run python -m src.pipeline train-manipulator
    uv run python -m src.pipeline evaluate
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.config import load_config
from src.latent_intervention import (
    LatentIntervention,
    make_objective,
    train_latent_intervention,
)
from src.semantic_decoder import (
    SCM_COLUMNS,
    SemanticDecoder,
    accuracy,
    targets_from_dataframe,
    train_semantic_decoder,
)

SIM_CONFIG_PATH = Path(__file__).parent.parent / "exp" / "sim" / "config.yaml"


def _load_latents(config: dict[str, Any]) -> tuple[torch.Tensor, list[int]]:
    """Load the encoded latents and their row ids."""
    payload = torch.load(Path(config["paths"]["latents"]), weights_only=True)
    return payload["z"], payload["ids"]


def _aligned_targets(csv_path: str | Path, ids: list[int]) -> dict[str, torch.Tensor]:
    """Read a sim CSV and return per-column targets aligned to the latent ids."""
    df = pd.read_csv(csv_path).set_index("id").loc[ids].reset_index()
    return targets_from_dataframe(df)


def _split(n: int, val_split: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic train/val index split shared by all stages."""
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    n_val = max(1, int(n * val_split))
    return perm[n_val:], perm[:n_val]


def _intervention() -> dict[str, int]:
    """Read the do() spec the sim data was generated under (exp/sim/config.yaml)."""
    import yaml

    with open(SIM_CONFIG_PATH, "r", encoding="utf-8") as f:
        return {str(k): int(v) for k, v in yaml.safe_load(f)["intervention"].items()}


def stage_encode(config: dict[str, Any]) -> None:
    """Encode the generated CV texts into latent vectors."""
    from src.encoder import TextEncoder  # deferred: heavy import, downloads checkpoint

    df = pd.read_csv(config["paths"]["texts"])
    enc_cfg = config["encoder"]
    encoder = TextEncoder(
        model_name=enc_cfg["model_name"],
        device=enc_cfg["device"],
        max_len=enc_cfg["max_len"],
        local_checkpoint=enc_cfg.get("local_checkpoint"),
    )
    z = encoder.encode(df["text"].tolist(), batch_size=enc_cfg["batch_size"])
    out = Path(config["paths"]["latents"])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"z": z.cpu(), "ids": df["id"].astype(int).tolist()}, out)
    print(f"wrote {out}: {tuple(z.shape)} latents for {len(df)} texts")


def stage_train_decoder(config: dict[str, Any]) -> None:
    """Train the semantic decoder on (latent, factual tabular) pairs."""
    z, ids = _load_latents(config)
    targets = _aligned_targets(config["paths"]["sim_factual"], ids)
    cfg = config["semantic_decoder"]
    train_idx, val_idx = _split(len(ids), cfg["val_split"], config["seed"])

    decoder = SemanticDecoder(
        latent_dim=z.shape[1],
        hidden_dim=cfg["hidden_dim"],
        n_hidden=cfg["n_hidden"],
        dropout=cfg["dropout"],
    )
    train_semantic_decoder(
        decoder,
        z[train_idx],
        {k: v[train_idx] for k, v in targets.items()},
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        device=config["encoder"]["device"],
    )
    val_acc = accuracy(decoder, z[val_idx], {k: v[val_idx] for k, v in targets.items()})
    print("validation accuracy:", {k: round(v, 3) for k, v in val_acc.items()})
    decoder.save(config["paths"]["decoder_model"])
    print(f"wrote {config['paths']['decoder_model']}")


def stage_train_manipulator(config: dict[str, Any]) -> None:
    """Train the manipulator against the frozen decoder and counterfactual targets."""
    z, ids = _load_latents(config)
    s_prime = _aligned_targets(config["paths"]["sim_counterfactual"], ids)
    intervention = _intervention()
    cfg = config["latent_intervention"]
    train_idx, _ = _split(len(ids), config["semantic_decoder"]["val_split"], config["seed"])

    decoder = SemanticDecoder.load(config["paths"]["decoder_model"])
    model = LatentIntervention(
        latent_dim=z.shape[1],
        columns=decoder.columns,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
    )
    train_latent_intervention(
        model,
        decoder,
        z[train_idx],
        intervention,
        {k: v[train_idx] for k, v in s_prime.items()},
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        proximity_weight=cfg["proximity_weight"],
        sparsity_weight=cfg["sparsity_weight"],
        seed=config["seed"],
        device=config["encoder"]["device"],
    )
    model.save(config["paths"]["manipulator_model"])
    print(f"wrote {config['paths']['manipulator_model']} (intervention {intervention})")


def stage_evaluate(config: dict[str, Any]) -> None:
    """Evaluate manipulator faithfulness on the validation split."""
    z, ids = _load_latents(config)
    s_factual = _aligned_targets(config["paths"]["sim_factual"], ids)
    s_prime = _aligned_targets(config["paths"]["sim_counterfactual"], ids)
    intervention = _intervention()
    _, val_idx = _split(len(ids), config["semantic_decoder"]["val_split"], config["seed"])

    decoder = SemanticDecoder.load(config["paths"]["decoder_model"])
    model = LatentIntervention.load(config["paths"]["manipulator_model"])
    z_val = z[val_idx]
    values, mask = make_objective(intervention, decoder.columns, batch_size=len(val_idx))
    with torch.no_grad():
        z_prime = model(z_val, values, mask)
    preds = decoder.predict(z_prime)

    consistency = {
        col.name: (preds[col.name] == s_prime[col.name][val_idx]).float().mean().item()
        for col in SCM_COLUMNS
    }
    decoder_val_acc = accuracy(decoder, z_val, {k: v[val_idx] for k, v in s_factual.items()})
    delta = z_prime - z_val
    report = {
        "intervention": intervention,
        "n_val": len(val_idx),
        "decoder_factual_accuracy": decoder_val_acc,
        "consistency_accuracy": consistency,
        "consistency_accuracy_mean": sum(consistency.values()) / len(consistency),
        "latent_shift": {
            "l1_mean": delta.abs().sum(dim=1).mean().item(),
            "l2_mean": delta.norm(dim=1).mean().item(),
            "z_l2_mean": z_val.norm(dim=1).mean().item(),
        },
    }
    out = Path(config["paths"]["eval_report"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")


STAGES = {
    "encode": stage_encode,
    "train-decoder": stage_train_decoder,
    "train-manipulator": stage_train_manipulator,
    "evaluate": stage_evaluate,
}


def main() -> None:
    """CLI entry point for the training/eval pipeline."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=STAGES, help="pipeline stage to run")
    parser.add_argument("--config", default=None, help="path to an alternative config YAML")
    args = parser.parse_args()
    config = load_config(path=args.config) if args.config else load_config()
    STAGES[args.stage](config)


if __name__ == "__main__":
    main()
