"""Discover saved latent spaces and validate a common decoder benchmark split."""

import copy
import json
import re
from pathlib import Path

import yaml

from src import pipeline
from src.config import CONFIG_PATH, encoder_tag, resolve_paths
from src.decoder_benchmark.decoder_experiment import candidate_settings
from src.encoder_protocols import configure_encoder
from src.encoders.langvae_ft.artifacts import inference_config, read_checkpoint
from src.pair_encoding import load_latent_artifact

ROOT = Path(__file__).resolve().parents[2]
DECODERS = ("independent", "autoregressive")


def case_name(config):
    return f"{encoder_tag(config['encoder'])}/g-{config['semantic_decoder']['variant']}"


def encoding_config(raw, folder):
    """Keep the original path placeholders for later decoder/manipulator choices."""
    config = copy.deepcopy(raw)
    enc = config["encoder"]
    enc.update(tag=None, local_checkpoint=None, device="cpu", deterministic=True)
    if folder.name in {"langvae", "langvae_ft"}:
        enc["variant"] = "langvae"
        if folder.name == "langvae_ft":
            info = json.loads((folder / "z_pairs.info.json").read_text())
            config = inference_config(config, info["local_checkpoint"])
            config["encoder"]["device"] = "cpu"
    else:
        match = re.fullmatch(r"(nomic|qwen3|embeddinggemma)_(\d+)", folder.name)
        if not match:
            raise ValueError(f"unrecognized canonical latent space: {folder}")
        variant, dimension = match.groups()
        enc.update(variant=variant, **{f"{variant}_latent_dim": int(dimension)})
    configure_encoder(config)
    config["semantic_decoder"]["tag"] = None
    if encoder_tag(config["encoder"]) != folder.name:
        raise ValueError(f"encoder tag does not match folder: {folder}")
    return config


def discover(config_path=CONFIG_PATH, latent_root=Path("data/latents/talent")):
    raw = yaml.safe_load(Path(config_path).read_text())
    folders = sorted(
        folder
        for folder in Path(latent_root).iterdir()
        if folder.is_dir()
        and not folder.name.startswith(".")
        and any(
            (folder / name).exists() for name in ("z_pairs.pt", "z_pairs.info.json")
        )
    )
    if not folders:
        raise ValueError("no canonical latent artifacts found")
    configs, inventory, templates = [], [], {}
    expected_split = None
    expected_candidates = None
    expected_inputs = None
    adaptation = None
    for folder in folders:
        template = encoding_config(raw, folder)
        templates[folder.name] = template
        resolved = resolve_paths(copy.deepcopy(template))
        if (
            Path(resolved["paths"]["latents"]).resolve()
            != (folder / "z_pairs.pt").resolve()
        ):
            raise ValueError(f"configuration does not select {folder}")
        artifact = load_latent_artifact(resolved)
        columns, _ = pipeline.load_schema(resolved["sim_config"])
        pipeline._aligned_targets(
            resolved["paths"]["sim_factual"], artifact.ids, columns
        )
        train, _ = pipeline._official_indices(artifact)
        cfg = resolved["semantic_decoder"]
        fit, val = pipeline._fit_calibration_split(
            train, float(cfg["calibration_split"]), int(cfg["split_seed"])
        )
        split = {
            "fit": [artifact.ids[i] for i in fit.tolist()],
            "validation": [artifact.ids[i] for i in val.tolist()],
            "official_test": artifact.test_ids,
        }
        sidecar = json.loads((folder / "z_pairs.info.json").read_text())
        candidates = candidate_settings(cfg)
        if expected_split is None:
            expected_split, expected_candidates = split, candidates
            expected_inputs = sidecar["input_sha256"]
        if split != expected_split or candidates != expected_candidates:
            raise ValueError(
                "latent spaces do not share identical splits/tuning candidates"
            )
        if sidecar["input_sha256"] != expected_inputs:
            raise ValueError(
                "latent spaces were generated from different text/pair inputs"
            )
        inventory.append(
            {
                "encoder_tag": folder.name,
                "dimension": int(artifact.z.shape[1]),
                "latents": str(folder / "z_pairs.pt"),
                "artifact_sha256": artifact.artifact_sha256,
                "factual_units": len(artifact.ids),
                "test_units": len(artifact.test_ids),
                "identity_test_units": int(artifact.is_identity.sum()),
            }
        )
        if folder.name == "langvae_ft":
            metadata = read_checkpoint(template["encoder"]["local_checkpoint"])
            manifest = metadata["data_manifest"]
            adaptation_train, adaptation_val = map(
                set, (manifest["train_ids"], manifest["validation_ids"])
            )
            if (adaptation_train | adaptation_val) & set(artifact.test_ids):
                raise ValueError("LangVAE adaptation contains official test IDs")
            if adaptation_train | adaptation_val != set(artifact.train_ids):
                raise ValueError(
                    "LangVAE adaptation does not match official training IDs"
                )
            if manifest["source_sha256"] != {
                "texts": expected_inputs["factual_text"],
                "pair_index": expected_inputs["pair_index"],
            }:
                raise ValueError("LangVAE adaptation used different source data")
            adaptation = {
                "g_validation_texts_used_in_adaptation_training": len(
                    set(split["validation"]) & adaptation_train
                ),
                "g_validation_texts_used_in_adaptation_validation": len(
                    set(split["validation"]) & adaptation_val
                ),
                "official_test_excluded": True,
                "note": "g validation labels are held out, but these CV texts were used by unsupervised LangVAE adaptation. Not a fully untouched end-to-end validation set; official test remains held out.",
            }
        for decoder in DECODERS:
            config = copy.deepcopy(template)
            config["semantic_decoder"]["variant"] = decoder
            configs.append(resolve_paths(config))
    return {
        "configs": configs,
        "templates": templates,
        "inventory": inventory,
        "split_ids": expected_split,
        "candidates": expected_candidates,
        "langvae_ft_validation_caveat": adaptation,
    }


def annotate_result(result):
    config = result["selected_config"]
    result["encoder_tag"] = encoder_tag(config["encoder"])
    result["case"] = case_name(config)
    # The actual checkpoint records the width too; this field is for reporting.
    from src.encoder_protocols import encoder_dimension

    result["latent_dimension"] = encoder_dimension(config["encoder"])
    return result


def freeze_selection(results, expected_cases):
    if len(results) != len(expected_cases) or {r["case"] for r in results} != set(
        expected_cases
    ):
        raise ValueError("cannot freeze selection before every case completes")
    ordered = sorted(
        results, key=lambda r: (r["summary"]["validation_joint_nll"]["mean"], r["case"])
    )
    return {
        "criterion": "Lowest mean validation full joint NLL over the five predeclared final seeds; exact ties broken by case name.",
        "frozen_before_test_evaluation": True,
        "recommended_case": ordered[0]["case"],
        "best_by_decoder": {
            decoder: next(r["case"] for r in ordered if r["decoder"] == decoder)
            for decoder in DECODERS
            if any(r["decoder"] == decoder for r in ordered)
        },
        "ranking": [
            {
                "case": r["case"],
                "validation_joint_nll": r["summary"]["validation_joint_nll"],
            }
            for r in ordered
        ],
        "deployment_seed": ordered[0]["deployment_seed"],
        "test_metrics_used_for_selection": False,
    }
