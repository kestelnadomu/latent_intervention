"""Immutable, training-free inventory and protocol for the nine-family h_Z study."""

from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import math
import re
from pathlib import Path

import yaml

from src import pipeline
from src.artifact_io import sha256_file
from src.config import CONFIG_PATH, load_config
from src.oracle_targets import load_oracle_targets, source_metadata, training_texts
from src.pair_encoding import load_latent_artifact
from src.schema import load_intervention, load_schema
from src.semantic_decoder import load_semantic_decoder
from src.symbolic_intervention import load_symbolic_kernel

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/hz_benchmark.yaml"
REPORTS = ROOT / "reports/talent/hz_benchmarks"
VARIANTS = (
    *pipeline.INTERVENTION_VARIANTS,
    *pipeline.FLOW_VARIANTS,
    pipeline.ORACLE_VARIANT,
)
MODEL_WIDTH_FIELDS = {
    **{variant: {"hidden_dim"} for variant in pipeline.FLOW_VARIANTS},
    "dist": {"d_model", "dim_feedforward"},
    "oracle_regression": {"hidden_dim"},
}
VALIDATION = {
    "baseline": "counterfactual_state_nll",
    "pre_additive": "semantic_forward_kl",
    "noise_token": "semantic_forward_kl",
    "dist": "semantic_forward_kl",
    "particles": "semantic_forward_kl",
    "state_flow": "factual_conditional_nll_per_coordinate",
    "distilled_flow": "teacher_energy_distance",
    "direct_semantic_flow": "semantic_forward_kl",
    "oracle_regression": "counterfactual_latent_mse",
}
SUPERVISION = {
    "baseline": "paired counterfactual structured S'; no training Z'",
    "pre_additive": "frozen g and analytic h_S; no paired counterfactual targets",
    "noise_token": "frozen g and analytic h_S; no paired counterfactual targets",
    "dist": "paired counterfactual S' in pretraining; frozen g and h_S; no training Z'",
    "particles": "frozen g and analytic h_S; no paired counterfactual targets",
    "state_flow": "factual (Z,S); observed-S and inferred-S inference reported separately",
    "distilled_flow": "factual (Z,S), fitted state-flow teacher, analytic h_S",
    "direct_semantic_flow": "frozen g and analytic h_S; no paired counterfactual targets",
    "oracle_regression": "privileged paired training Z'; no g or h_S in fitting",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def run_root(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run-id must be a simple directory name")
    return REPORTS / run_id


def read_settings(path=PROTOCOL):
    settings = yaml.safe_load(Path(path).read_text())
    for name in (
        "max_epochs",
        "batch_size",
        "patience",
        "scheduler_patience",
        "validation_samples",
        "evaluation_samples",
        "evaluation_batch_size",
        "workers",
        "threads_per_worker",
    ):
        value = settings[name]
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in (
        "dimensions",
        "variants",
        "final_seeds",
        "learning_rates",
        "width_multipliers",
    ):
        if not settings[name] or len(set(settings[name])) != len(settings[name]):
            raise ValueError(f"{name} must be nonempty and unique")
    if not set(settings["dimensions"]) <= {256, 768} or not set(
        settings["variants"]
    ) <= set(VARIANTS):
        raise ValueError("unsupported benchmark dimensions or variants")
    if (
        "distilled_flow" in settings["variants"]
        and "state_flow" not in settings["variants"]
    ):
        raise ValueError("distilled_flow requires state_flow in the same run")
    if settings["deployment_seed"] not in settings["final_seeds"]:
        raise ValueError("deployment seed must be a predeclared final seed")
    if (
        not 0 < settings["validation_fraction"] < 1
        or not 0 < settings["scheduler_factor"] < 1
    ):
        raise ValueError("invalid holdout fraction or scheduler factor")
    if (
        type(settings["dist_pretrain_epochs"]) is not int
        or not 0 <= settings["dist_pretrain_epochs"] < settings["max_epochs"]
    ):
        raise ValueError("dist pretraining must leave at least one joint epoch")
    for value in (
        *settings["final_seeds"],
        settings["split_seed"],
        settings["search_seed"],
        settings["evaluation_seed"],
    ):
        if type(value) is not int or not 0 <= value < 2**32:
            raise ValueError("seeds must be nonnegative 32-bit integers")
    for value in (
        *settings["learning_rates"],
        *settings["width_multipliers"],
        settings["grad_clip"],
        settings["min_lr"],
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("rates, widths and grad_clip must be positive and finite")
    if (
        settings["min_lr"] > min(settings["learning_rates"])
        or not math.isfinite(settings["min_delta"])
        or settings["min_delta"] < 0
        or any(type(x) is not int for x in settings["width_multipliers"])
    ):
        raise ValueError("invalid min_lr/min_delta")
    if settings["validation_samples"] < 2 or settings["evaluation_samples"] < 2:
        raise ValueError("stochastic evaluation requires multiple samples")
    for variant in set(settings["variants"]) & MODEL_WIDTH_FIELDS.keys():
        for dimension in settings["dimensions"]:
            for width in settings["width_multipliers"]:
                try:
                    overrides = settings["model_widths"][variant][str(dimension)][
                        str(width)
                    ]
                except (KeyError, TypeError) as exc:
                    raise ValueError(
                        f"model_widths needs {variant}/{dimension}/tier-{width}"
                    ) from exc
                if (
                    not isinstance(overrides, dict)
                    or set(overrides) != MODEL_WIDTH_FIELDS[variant]
                    or any(
                        type(value) is not int or value < 1
                        for value in overrides.values()
                    )
                ):
                    raise ValueError(
                        f"model_widths for {variant} must specify only "
                        f"{sorted(MODEL_WIDTH_FIELDS[variant])} as positive integers"
                    )
    if "pre_additive" in settings["variants"]:
        for dimension in settings["dimensions"]:
            rms = settings["pre_additive_noise_rms"].get(str(dimension))
            if type(rms) not in (int, float) or not math.isfinite(rms) or rms <= 0:
                raise ValueError("pre_additive_noise_rms must be positive and finite")
    return settings


def case_name(dimension, variant):
    return f"embeddinggemma_{dimension}/g-independent/h-{variant}"


def candidates(settings):
    return [
        dict(lr=lr, width_multiplier=width)
        for width, lr in itertools.product(
            settings["width_multipliers"], settings["learning_rates"]
        )
    ]


def split_ids(config, artifact, settings):
    train, _ = pipeline._official_indices(artifact)
    fit, val = pipeline._fit_calibration_split(
        train, settings["validation_fraction"], settings["split_seed"]
    )
    return {
        "fit": [artifact.ids[i] for i in fit.tolist()],
        "validation": [artifact.ids[i] for i in val.tolist()],
        "official_test": artifact.test_ids,
    }


def discover(config_path=CONFIG_PATH, protocol_path=PROTOCOL):
    """Read/validate inputs only; never instantiate an encoder or train any model."""
    settings = read_settings(protocol_path)
    cases, input_files, expected_split, oracle_sources = [], set(), None, []
    for dimension in settings["dimensions"]:
        config = load_config(
            path=config_path,
            encoder_variant="embeddinggemma",
            embeddinggemma_dim=dimension,
            decoder_variant="independent",
            manipulator_variant="oracle_regression",
        )
        if any(
            config[key].get("tag")
            for key in ("encoder", "semantic_decoder", "latent_intervention")
        ):
            raise ValueError("benchmark requires untagged canonical inputs")
        artifact = load_latent_artifact(config)
        columns, _ = load_schema(config["sim_config"])
        decoder = load_semantic_decoder(
            config["paths"]["decoder_model"],
            expected_variant="independent",
            expected_columns=columns,
            expected_metadata=pipeline._decoder_metadata(config, artifact),
        )
        if list(load_symbolic_kernel(config["sim_config"]).columns) != list(
            decoder.columns
        ):
            raise ValueError("g and h_S schemas differ")
        split = split_ids(config, artifact, settings)
        if expected_split is not None and split != expected_split:
            raise ValueError("encoders do not share the same fit/validation/test IDs")
        expected_split = split
        for key in ("sim_factual", "sim_counterfactual"):
            pipeline._aligned_targets(config["paths"][key], artifact.ids, columns)
        for key in (
            "latents",
            "decoder_model",
            "sim_factual",
            "sim_counterfactual",
            "pair_index",
            "texts",
            "texts_counterfactual",
        ):
            input_files.add(config["paths"][key])
        input_files.add(str(Path(config["paths"]["latents"]).with_suffix(".info.json")))
        if not isinstance(config["sim_config"], dict):
            input_files.add(str(config["sim_config"]))
        if "oracle_regression" in settings["variants"]:
            ids, identity, _, _ = training_texts(config, artifact)
            oracle_sources.append(
                dict(
                    dimension=dimension,
                    path=config["paths"]["oracle_targets"],
                    expected_metadata=source_metadata(config, artifact),
                    units=len(ids),
                    identity_copies=int(identity.sum()),
                    nonidentity_texts=int((~identity).sum()),
                )
            )
        for variant in settings["variants"]:
            item = load_config(
                path=config_path,
                encoder_variant="embeddinggemma",
                embeddinggemma_dim=dimension,
                decoder_variant="independent",
                manipulator_variant=variant,
            )
            cases.append(
                dict(
                    case=case_name(dimension, variant),
                    dimension=dimension,
                    variant=variant,
                    config=item,
                    validation_metric=VALIDATION[variant],
                    supervision=SUPERVISION[variant],
                )
            )
            if variant in MODEL_WIDTH_FIELDS:
                cases[-1]["model_widths"] = settings["model_widths"][variant][
                    str(dimension)
                ]
            if variant == "pre_additive":
                cases[-1]["noise_rms"] = settings["pre_additive_noise_rms"][
                    str(dimension)
                ]
    code_files = sorted(
        [
            *ROOT.glob("src/*.py"),
            *ROOT.glob("exp/sim/*.py"),
            ROOT / "scripts/run_hz_benchmark.sh",
            ROOT / "scripts/run_hz_pilot.sh",
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
        ]
    )
    source = {str(path.relative_to(ROOT)): sha256_file(path) for path in code_files}
    input_files.update((str(config_path), str(protocol_path)))

    # Canonical source/input paths remain portable when the repository is moved.
    def portable(path):
        try:
            return str(Path(path).resolve().relative_to(ROOT))
        except ValueError:
            return str(path)

    return dict(
        format_version=1,
        settings=settings,
        candidates=candidates(settings),
        cases=cases,
        split_ids=expected_split,
        oracle_sources=oracle_sources,
        intervention=load_intervention(cases[0]["config"]["sim_config"]),
        code_sha256=source,
        input_sha256={
            portable(path): sha256_file(path) for path in sorted(input_files)
        },
        packages={
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pandas", "pyyaml")
        },
        selection="Per-family validation metric; all choices frozen before any official-test scoring. No cross-family ranking by unlike validation objectives.",
        resume="Verified completed fits/evaluations are reused. An interrupted fit restarts from its fixed seed; no optimizer-state recovery.",
        device="cpu",
    )


def assert_unchanged(plan):
    for group in ("code_sha256", "input_sha256"):
        for name, expected in plan[group].items():
            if sha256_file(name) != expected:
                raise ValueError(f"{group} changed: {name}; use a new run ID")


def oracle_readiness(plan):
    """A missing target is a launch prerequisite, not permission to encode it."""
    result = []
    for source in plan["oracle_sources"]:
        case = next(
            c
            for c in plan["cases"]
            if c["dimension"] == source["dimension"]
            and c["variant"] == "oracle_regression"
        )
        path = Path(source["path"])
        if not path.exists():
            result.append(dict(**source, status="missing"))
            continue
        targets = load_oracle_targets(case["config"])
        if any(
            targets.metadata.get(k) != v for k, v in source["expected_metadata"].items()
        ):
            raise ValueError("oracle target source metadata changed")
        result.append(
            dict(
                **source,
                status="verified",
                sha256=targets.artifact_sha256,
                sidecar_sha256=sha256_file(path.with_suffix(".info.json")),
            )
        )
    return result
