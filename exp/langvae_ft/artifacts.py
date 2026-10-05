"""Native immutable checkpoints, epoch resume and opt-in remote configuration."""

import copy
import importlib.metadata
import json
import os
import random
import uuid
from pathlib import Path

import torch
import yaml

from . import ARCHITECTURE, TAG
from .data import file_hash
from .model import load_native

ROOT = Path(__file__).resolve().parents[2]


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    torch.save(value, temporary)
    os.replace(temporary, path)


def source_hashes():
    files = sorted((ROOT / "exp/langvae_ft").glob("*.py"))
    files += [ROOT / "src/encoder/langvae.py", ROOT / "src/config.py"]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in files}


def packages():
    return {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "langvae")
    }


def trainable_schema(adapter):
    return {
        name: list(p.shape) for name, p in adapter.named_parameters() if p.requires_grad
    }


def save_resume(path, adapter, optimizer, **progress):
    device = next(adapter.projection.parameters()).device
    atomic_torch(
        path,
        {
            "projection": adapter.projection.state_dict(),
            "adapters": adapter.adapters.state_dict(),
            "optimizer": optimizer.state_dict(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device)
            if device.type == "cuda"
            else None,
            "progress": progress,
        },
    )


def restore_resume(path, adapter, optimizer):
    state = torch.load(path, map_location="cpu", weights_only=True)
    adapter.projection.load_state_dict(state["projection"])
    adapter.adapters.load_state_dict(state["adapters"])
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(
            state["cuda_rng"], next(adapter.projection.parameters()).device
        )
    return state["progress"]


@torch.no_grad()
def export_checkpoint(base, adapter, directory, dataset, run, epoch, validation):
    """Save native LangVAE files: loadable by unchanged src.encoder.langvae.TextEncoder."""
    name = f"checkpoint-{epoch:03d}-{uuid.uuid4().hex[:8]}"
    temporary = directory / ("." + name)
    base.save(str(temporary))
    adapter.eval()
    pooled = dataset["features"]["validation"][:2].to(
        next(adapter.projection.parameters()).device
    )
    mu, _ = adapter.posterior(pooled)
    atomic_torch(
        temporary / "reload_probe.pt",
        {
            "input_ids": [r["input_ids"] for r in dataset["records"]["validation"][:2]],
            "pooled": pooled.cpu(),
            "mu": mu.cpu(),
            "ids": adapter.generate(mu).cpu(),
        },
    )
    atomic_json(
        temporary / "langvae_ft.json",
        {
            "schema": 1,
            "architecture": ARCHITECTURE,
            "tag": TAG,
            "stage": run["stage"],
            "run_id": directory.name,
            "epoch": epoch,
            "encoder": run["encoder"],
            "recipe": run["recipe"],
            "source_sha256": run["source_sha256"],
            "packages": run["packages"],
            "data_manifest": run["data_manifest"],
            "validation": validation,
            "trainable_schema": trainable_schema(adapter),
            "frozen": ["BERT", "GPT-2"],
            "latent_dim": 128,
            "decoder_length": 32,
            "target_policy": "same short passage as input and target, plus EOS; all CV passages",
            "final_encoding": "full CV -> posterior mean; run only on destination cluster",
            "quality_claim": "domain-adapted baseline; not a full-CV reconstruction or fairness guarantee",
        },
    )
    hashes = {p.name: file_hash(p) for p in temporary.iterdir() if p.is_file()}
    atomic_json(temporary / "sha256.json", hashes)
    destination = directory / name
    os.replace(temporary, destination)
    selected = {
        "path": name,
        "epoch": epoch,
        "negative_elbo": validation["negative_elbo"],
        "sha256": hashes,
    }
    atomic_json(directory / "selected_checkpoint.json", selected)
    return selected


def read_checkpoint(checkpoint):
    checkpoint = Path(checkpoint)
    hashes = json.loads((checkpoint / "sha256.json").read_text())
    required = {
        "encoder.pt",
        "decoder.pt",
        "encoder_cfg.json",
        "decoder_cfg.json",
        "model_config.json",
        "environment.json",
        "langvae_ft.json",
        "reload_probe.pt",
    }
    if not required <= set(hashes):
        raise ValueError("incomplete checkpoint manifest")
    for name, digest in hashes.items():
        if Path(name).name != name or file_hash(checkpoint / name) != digest:
            raise ValueError(f"checkpoint checksum mismatch: {name}")
    metadata = json.loads((checkpoint / "langvae_ft.json").read_text())
    if metadata["architecture"] != ARCHITECTURE or metadata["tag"] != TAG:
        raise ValueError("wrong checkpoint architecture/tag")
    return metadata


@torch.no_grad()
def verify_checkpoint(checkpoint, device):
    """Verify features, posterior means, generation and original-decoder parity."""
    checkpoint = Path(checkpoint).resolve()
    metadata = read_checkpoint(checkpoint)
    base, adapter = load_native(
        {**metadata["encoder"], "local_checkpoint": str(checkpoint)}, device
    )
    adapter.eval()
    if trainable_schema(adapter) != metadata["trainable_schema"]:
        raise ValueError("native parameter shapes changed")
    probe = torch.load(
        checkpoint / "reload_probe.pt", map_location="cpu", weights_only=True
    )
    fresh = base.encoder.recode(probe["input_ids"])
    torch.testing.assert_close(fresh.cpu(), probe["pooled"], atol=2e-5, rtol=2e-5)
    mu, _ = adapter.posterior(probe["pooled"].to(device))
    torch.testing.assert_close(mu.cpu(), probe["mu"], atol=1e-6, rtol=1e-6)
    ids = adapter.generate(mu)
    torch.testing.assert_close(ids.cpu(), probe["ids"], atol=0, rtol=0)
    original = base.decoder(mu).reconstruction.argmax(-1)
    for i, row in enumerate(ids):
        eos = row.eq(adapter.eos_id).nonzero().flatten()
        length = int(eos[0]) + 1 if len(eos) else len(row)
        torch.testing.assert_close(original[i, :length], row[:length], atol=0, rtol=0)
    return {
        "passed": True,
        "tag": TAG,
        "checkpoint": str(checkpoint),
        "device": str(device),
        "offline": os.environ.get("HF_HUB_OFFLINE") == "1",
    }


def inference_config(raw_pipeline, checkpoint):
    """Opt in via --config; never rewrite the shared root config or resolve early."""
    checkpoint = Path(checkpoint).resolve()
    metadata = read_checkpoint(checkpoint)
    if metadata["stage"] != "train":
        raise ValueError(
            "diagnostic checkpoints cannot become a final encoding baseline"
        )
    config = copy.deepcopy(raw_pipeline)
    config.pop("finetune_vae", None)  # never copy historical experimental settings
    # Encoder identity comes from the checkpoint, not the receiving host's defaults.
    config["encoder"] = copy.deepcopy(metadata["encoder"])
    try:
        path = str(checkpoint.relative_to(ROOT))
    except ValueError:
        path = str(checkpoint)
    config["encoder"].update(
        variant="langvae", tag=TAG, local_checkpoint=path, deterministic=True
    )
    for key in (
        "latents",
        "decoder_model",
        "decoder_report",
        "manipulator_model",
        "eval_report",
    ):
        if "{encoder}" not in config["paths"][key]:
            raise ValueError(f"{key} must retain its encoder-specific path placeholder")
    return config


def write_inference_config(raw_pipeline, checkpoint, output):
    config = inference_config(raw_pipeline, checkpoint)
    # Refuse overwrite (especially src/config.yaml). Config lives OUTSIDE the
    # checksummed checkpoint: changing a local path must not alter its identity.
    output = Path(output)
    checkpoint = Path(checkpoint).resolve()
    if output.resolve().is_relative_to(checkpoint):
        raise ValueError(
            "write the encoding configuration outside the immutable checkpoint"
        )
    with output.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return config
