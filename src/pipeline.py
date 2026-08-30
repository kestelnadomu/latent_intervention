"""
Training and evaluation pipeline: text latents -> semantic decoder -> manipulator.

Stages (hyperparameters from src/config.yaml; data + schema from exp/sim/):
    encode             encode generated input texts into latents (data/latents/)
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
    LatentInterventionPreAdditive,
    LatentInterventionDist,
    make_objective,
    train_latent_intervention,
    train_latent_intervention_dist,
    train_latent_intervention_preadditive,
)
from src.schema import load_intervention, load_schema
from src.semantic_decoder import (
    SemanticDecoder,
    SemanticAutoRegDecoder,
    accuracy,
    targets_from_dataframe,
    train_semantic_decoder,
)
from src.symbolic_intervention import load_symbolic_kernel

DECODER_VARIANTS = {
    "independent": SemanticDecoder,
    "autoregressive": SemanticAutoRegDecoder
}
INTERVENTION_VARIANTS = {
    "baseline": LatentIntervention,
    "pre_additive": LatentInterventionPreAdditive,
    "dist": LatentInterventionDist
}

def _load_latents(config: dict[str, Any]) -> tuple[torch.Tensor, list[int]]:
    """Load the encoded latents and their row ids."""
    payload = torch.load(Path(config["paths"]["latents"]), weights_only=True)
    return payload["z"], payload["ids"]


def _aligned_targets(csv_path: str | Path, ids: list[int], columns: list) -> dict[str, torch.Tensor]:
    """Read a sim CSV and return per-column targets aligned to the latent ids."""
    df = pd.read_csv(csv_path).set_index("id").loc[ids].reset_index()
    return targets_from_dataframe(df, columns)


def _split(n: int, val_split: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic train/val index split shared by all stages."""
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    n_val = max(1, int(n * val_split))
    return perm[n_val:], perm[:n_val]


def stage_encode(config: dict[str, Any]) -> None:
    """Encode the generated input texts into latent vectors."""
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
    columns, _ = load_schema()
    targets = _aligned_targets(config["paths"]["sim_factual"], ids, columns)
    cfg = config["semantic_decoder"]
    train_idx, val_idx = _split(len(ids), cfg["val_split"], config["seed"])

    dv = DECODER_VARIANTS[cfg["variant"]]
    decoder = dv(
        latent_dim=z.shape[1],
        columns=columns,
        hidden_dim=cfg["hidden_dim"],
        n_hidden=cfg["n_hidden"],
        dropout=cfg["dropout"],
        embed_dim=cfg["autoregressive"]["embed_dim"]
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
    intervention = load_intervention()
    cfg = config["latent_intervention"]
    train_idx, _ = _split(len(ids), config["semantic_decoder"]["val_split"], config["seed"])

    dv = DECODER_VARIANTS[config["semantic_decoder"]["variant"]]
    decoder = dv.load(config["paths"]["decoder_model"])
    s_prime = _aligned_targets(config["paths"]["sim_counterfactual"], ids, decoder.columns)
    h_s = load_symbolic_kernel()  # resolves objects.symbolic_kernel in exp/sim/config.yaml

    # shared kwargs for all manipulator variants
    model_kwargs = dict(
        latent_dim=z.shape[1],
        columns=decoder.columns,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
    )
    # kwargs common to every variant's trainer; each trainer ignores the ones it
    # does not need (baseline ignores h_s, pre_additive ignores s_prime).
    train_kwargs = dict(
        decoder=decoder,
        latents=z[train_idx],
        intervention=intervention,
        h_s=h_s,
        s_prime={k: v[train_idx] for k, v in s_prime.items()},
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        proximity_weight=cfg["proximity_weight"],
        sparsity_weight=cfg["sparsity_weight"],
        seed=config["seed"],
        device=config["encoder"]["device"],
    )

    # train the manipulator variant specified in the config
    if cfg["variant"] == "baseline":
        model = LatentIntervention(**model_kwargs)
        train_latent_intervention(model=model, epochs=cfg["epochs"], **train_kwargs)
    elif cfg["variant"] == "pre_additive":
        pa = cfg["pre_additive"]
        model = LatentInterventionPreAdditive(**model_kwargs, noise_std=pa["noise_std"])
        train_latent_intervention_preadditive(
            model=model, epochs=cfg["epochs"], n_samples=pa["n_samples"], **train_kwargs
        )
    elif cfg["variant"] == "dist":
        d = cfg["dist"]
        model = LatentInterventionDist(
            **model_kwargs, embed_dim=d["embed_dim"], top_k=d["top_k"]
        )
        train_latent_intervention_dist(
            model=model,
            pretrain_epochs=d["pretrain_epochs"],
            joint_epochs=d["joint_epochs"],
            realiser_l1=d["realiser_l1"],
            realiser_l2=d["realiser_l2"],
            **train_kwargs,
        )
    model.save(config["paths"]["manipulator_model"])
    print(f"wrote {config['paths']['manipulator_model']} (intervention {intervention})")


def stage_evaluate(config: dict[str, Any]) -> None:
    """Evaluate manipulator faithfulness on the validation split."""
    z, ids = _load_latents(config)
    intervention = load_intervention()
    _, val_idx = _split(len(ids), config["semantic_decoder"]["val_split"], config["seed"])

    dv = DECODER_VARIANTS[config["semantic_decoder"]["variant"]]
    decoder = dv.load(config["paths"]["decoder_model"])
    mv = INTERVENTION_VARIANTS[config["latent_intervention"]["variant"]]
    model = mv.load(config["paths"]["manipulator_model"])
    s_factual = _aligned_targets(config["paths"]["sim_factual"], ids, decoder.columns)
    s_prime = _aligned_targets(config["paths"]["sim_counterfactual"], ids, decoder.columns)
    z_val = z[val_idx]
    values, mask = make_objective(intervention, decoder.columns, batch_size=len(val_idx))
    with torch.no_grad():
        z_prime = model(z_val, values, mask)
    preds = decoder.predict(z_prime)

    consistency = {
        col.name: (preds[col.name] == s_prime[col.name][val_idx]).float().mean().item()
        for col in decoder.columns
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
