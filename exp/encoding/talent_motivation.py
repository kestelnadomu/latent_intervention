"""One frozen encoder at its full width, three artifacts, no downstream fitting.

``--encoder`` picks the backend (default EmbeddingGemma, the original 768D run); narrower
widths are Matryoshka prefixes that latent_lab projects from the full-width bundle.
The established paired writer is unchanged. Completed artifacts resume after
validation; an interrupted encoder pass restarts that artifact only. The complete
bundle is published by one directory rename. Use the detached shell launcher.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import re
from time import perf_counter
from uuid import uuid4

import pandas as pd
import torch
import yaml

from exp.encoding.queue import now, protocol, verify_artifact
from exp.encoding.text_artifact import encode_text_artifact, load_text_artifact, read_texts
from src.artifact_io import sha256_file, write_json
from src.config import encoder_tag, load_config
from src.encoder import encoder_dimension, get_encoder_factory
from src.pair_encoding import _load_csv, _load_pair_index, encode_pairs, load_latent_artifact

CONFIG = "configs/encoding_talent_motivation.yaml"
# Each encoder is encoded once at its full width; LangVAE has a single 128D space.
FULL_WIDTH = {"embeddinggemma": 768, "nomic": 768, "qwen3": 1024, "langvae": 128}
AUXILIARY = (
    ("floor", "texts_floor", "floor_latents"),
    ("factual_vb", "texts_held_out", "held_out_latents"),
)
INPUT_KEYS = (
    "texts", "texts_counterfactual", "texts_floor", "texts_held_out", "pair_index",
    "sim_factual", "sim_counterfactual", "partition", "interventions",
)


def load_encoding_config(encoder, path=CONFIG):
    """``path`` resolved for one encoder at its full width."""
    width = FULL_WIDTH[encoder]
    dims = {f"{name}_dim": width if encoder == name else None for name in ("nomic", "qwen3", "embeddinggemma")}
    return load_config(path=path, encoder_variant=encoder, **dims)


def input_hashes(config):
    files = {key: Path(config["paths"][key]) for key in INPUT_KEYS}
    files["sim_config"] = Path(config["sim_config"])
    sim = yaml.safe_load(files["sim_config"].read_text())
    sim_dir = Path(sim["paths"]["sim_dir"])
    for name in ("simulation_info.json", "render_plan.csv", "sim_data_epsilon.csv",
                 "sim_data_counterfactual_states.csv"):
        files[name] = sim_dir / name
    for name in ("codebook", "codebook_held_out"):
        files[name] = Path(sim[name])
    for key in ("texts", "texts_counterfactual", "texts_floor", "texts_held_out"):
        files[f"{key}_generation"] = Path(config["paths"][key]).with_suffix(".generation.json")
    return {key: sha256_file(path) for key, path in files.items()}


def input_tables(config):
    paths = config["paths"]
    pairs = _load_pair_index(paths["pair_index"])
    test_ids = pairs.loc[pairs["split"].eq("test"), "id"].tolist()
    tables = {
        "factual": read_texts(paths["texts"], pairs["id"].tolist()),
        "counterfactual": read_texts(paths["texts_counterfactual"], test_ids),
    }
    for kind, text_key, _ in AUXILIARY:
        table = _load_csv(paths[text_key], kind, {"id", "text"})
        if not set(table["id"]) <= set(test_ids):
            raise ValueError(f"{kind} contains non-evaluation IDs")
        tables[kind] = read_texts(paths[text_key], table["id"].tolist())
    partition = _load_csv(paths["partition"], "partition", {"id", "partition", "floor"})
    if partition["id"].tolist() != pairs["id"].tolist():
        raise ValueError("partition IDs differ from pair IDs")
    if set(partition.loc[partition["partition"].eq("eval"), "id"]) != set(test_ids):
        raise ValueError("partition evaluation IDs differ from pair test IDs")
    counts = {kind: len(table) for kind, table in tables.items()}
    counts.update({str(key): int(value) for key, value in partition["partition"].value_counts().items()})
    if counts != config["expected_counts"]:
        raise ValueError(f"unexpected corpus/partition sizes: {counts}")
    return tables


def validate_simulation(config):
    # The simulation environment owns dotenv/openai; the isolated encoder does not.
    from exp.sim.helpers import load_sim_config
    from exp.sim.paired_data import load_pair_inputs
    from exp.sim.render import held_out_ids

    sim = load_sim_config(config["sim_config"])
    _, _, pairs = load_pair_inputs(sim)
    tables = input_tables(config)
    partition = pd.read_csv(config["paths"]["partition"])
    test_ids = set(pairs.loc[pairs["split"].eq("test"), "id"])
    if set(tables["floor"]["id"]) != set(partition.loc[partition["floor"], "id"]):
        raise ValueError("floor IDs differ from the recorded floor subset")
    if set(tables["factual_vb"]["id"]) != held_out_ids(sim, test_ids):
        raise ValueError("held-out wording IDs differ from the seeded subset")
    return {"status": "passed", "validated_at": now(), "input_sha256": input_hashes(config)}


def signature(config, config_path, threads, run_id):
    encoder = config["encoder"]
    width = FULL_WIDTH.get(encoder["variant"])
    if width is None or encoder_dimension(encoder) != width or encoder.get("local_checkpoint"):
        raise ValueError(f"encode a stock encoder once at its full width {FULL_WIDTH}")
    return {
        "run_id": run_id, "encoding": protocol({width: config}, threads),
        "config_sha256": sha256_file(config_path), "input_sha256": input_hashes(config),
        "source_sha256": {str(p): sha256_file(p) for p in (
            Path(__file__), Path("exp/encoding/text_artifact.py"),
            Path("scripts/run_talent_motivation_encoding.sh"),
            Path("exp/sim/paired_data.py"), Path("exp/sim/design.py"),
            Path("exp/sim/render.py"),
        )},
    }


def preflight(config, sig):
    if config["encoder"]["variant"] != "embeddinggemma":
        # Qwen3 refuses over-long inputs while encoding; Nomic and LangVAE truncate at max_len
        # exactly as in their original encodings. Nothing to audit up front.
        return {"status": "passed", "protocol": sig, "paired": None, "auxiliary": None}
    from exp.encoding.embeddinggemma_preflight import run_preflight
    from src.encoder.embeddinggemma import cached_snapshot, format_text
    from transformers import AutoTokenizer

    tables = input_tables(config)
    paired = run_preflight({768: config})
    tokenizer = AutoTokenizer.from_pretrained(
        cached_snapshot(config["encoder"]), local_files_only=True, trust_remote_code=False,
    )
    auxiliary = {}
    for kind, _, _ in AUXILIARY:
        texts = [format_text(text) for text in tables[kind]["text"]]
        lengths = []
        for start in range(0, len(texts), 128):
            tokens = tokenizer(texts[start:start + 128], padding=False, truncation=False,
                               add_special_tokens=True)
            lengths.extend(map(len, tokens["input_ids"]))
        if max(lengths) > config["encoder"]["max_len"]:
            raise ValueError(f"{kind} would exceed the encoder token budget")
        auxiliary[kind] = {"units": len(texts), "maximum_tokens": max(lengths), "truncated_texts": 0}
    if input_hashes(config) != sig["input_sha256"]:
        raise ValueError("inputs changed during preflight")
    return {"status": "passed", "protocol": sig, "paired": paired, "auxiliary": auxiliary}


@contextmanager
def output_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".encoding.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another worker holds the encoding output lock") from exc
        yield


def _retain_partial(path):
    """Keep a torn tensor/sidecar pair recoverable instead of overwriting it."""
    pair = (path, path.with_suffix(".info.json"))
    if sum(p.is_file() for p in pair) == 1:
        retained = path.parent / f".interrupted-{path.stem}-{uuid4().hex[:8]}"
        retained.mkdir()
        for p in pair:
            if p.exists():
                p.rename(retained / p.name)
        print(f"Retained partial output in {retained}", flush=True)


def run_queue(config, root, sig, *, encoder_factory=None):
    tables = input_tables(config)
    destination = Path(config["paths"]["latents"]).parent
    output_keys = ("latents", "floor_latents", "held_out_latents")
    if any(Path(config["paths"][key]).parent != destination for key in output_keys):
        raise ValueError("all three artifacts must share one output directory")
    if destination.name != encoder_tag(config["encoder"]):
        raise ValueError(f"output directory must be named {encoder_tag(config['encoder'])}")
    pending = destination.with_name(f".{destination.name}.{sig['run_id']}.pending")
    work = destination if destination.exists() else pending
    if work.exists():
        manifest = json.loads((work / "manifest.json").read_text())
        if manifest["protocol"] != sig:
            raise ValueError("existing embedding bundle has a conflicting run signature")
        if work == destination and not manifest.get("complete"):
            raise ValueError("published bundle is incomplete")
    else:
        work.mkdir(parents=True)
        write_json(work / "manifest.json", {"protocol": sig, "complete": False})
    staged = deepcopy(config)
    for key in output_keys:
        staged["paths"][key] = str(work / Path(config["paths"][key]).name)
    started = perf_counter()
    status = {"status": "running", "started_at": now(), "active_artifact": None,
              "completed": [], "run_id": sig["run_id"], "total_texts": sum(map(len, tables.values()))}
    root.mkdir(parents=True, exist_ok=True)
    model = None

    def encoder(_):
        nonlocal model
        if model is None:
            model = (encoder_factory or get_encoder_factory(config["encoder"]))(config["encoder"])
        return model

    def update():
        status["updated_at"] = now()
        status["elapsed_seconds"] = perf_counter() - started
        write_json(root / "status.json", status)

    try:
        for key in output_keys:
            if input_hashes(config) != sig["input_sha256"]:
                raise ValueError("source inputs changed; refusing stale encoding")
            path = Path(staged["paths"][key])
            status["active_artifact"] = path.name
            update()
            print(f"{now()} Starting/verifying {path.name}", flush=True)
            if work != destination:
                _retain_partial(path)
            if key == "latents":
                if not path.exists() and work != destination:
                    encode_pairs(staged, encoder_factory=encoder)
                # LangVAE's posterior means are not unit vectors; the others are L2-normalized.
                langvae = config["encoder"]["variant"] == "langvae"
                artifact = (load_latent_artifact if langvae else verify_artifact)(staged)
                shapes = {"z": list(artifact.z.shape), "z_prime": list(artifact.z_prime.shape)}
            else:
                kind, text_key, _ = next(item for item in AUXILIARY if item[2] == key)
                options = dict(texts=config["paths"][text_key], expected_ids=tables[kind]["id"].tolist(),
                               kind=kind, encoder_config=config["encoder"], provenance=sig["input_sha256"])
                if not path.exists() and work != destination:
                    encode_text_artifact(path, **options, encoder=encoder(config["encoder"]))
                payload = load_text_artifact(path, **options)
                shapes = {"z": list(payload["z"].shape)}
            status["completed"].append({"file": path.name, "shapes": shapes,
                                        "artifact_sha256": sha256_file(path)})
            update()
        if input_hashes(config) != sig["input_sha256"]:
            raise ValueError("source inputs changed before publication")
        if work != destination:
            write_json(work / "manifest.json", {
                "format": "talent_motivation_embeddings_v1", "protocol": sig, "complete": True,
                "completed_at": now(), "artifacts": status["completed"],
                "roles": {"z_pairs.z": "factual; use partition.csv", "z_pairs.z_prime": "evaluation_only",
                          "z_floor.z": "evaluation_only", "z_factual_vb.z": "evaluation_only"},
                "interventions": "per-unit do(X=k); see hashed interventions.csv, not fixed do(X=3)",
            })
            if destination.exists():
                raise FileExistsError(f"refusing to replace {destination}")
            pending.rename(destination)
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        status.update(status="complete", active_artifact=None, finished_at=now(), output=str(destination))
        update()
        print(f"{now()} Complete: {destination}", flush=True)
        return status
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        update()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "preflight", "run"))
    parser.add_argument("--encoder", choices=sorted(FULL_WIDTH), default="embeddinggemma")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--run-id", help="default: <encoder>-<width>-v1")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    config = load_encoding_config(args.encoder, args.config)
    args.run_id = args.run_id or f"{args.encoder}-{FULL_WIDTH[args.encoder]}-v1"
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", args.run_id) or args.threads < 1:
        parser.error("invalid run ID or thread count")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    root = Path(config["paths"]["encoding_reports"]) / args.run_id
    with output_lock(Path(config["paths"]["latents"]).parent.parent):
        if args.stage == "validate":
            result = validate_simulation(config)
            write_json(root / "data_validation.json", result)
            print("Simulation, partition, action and auxiliary-subset validation passed.")
            return
        sig = signature(config, args.config, args.threads, args.run_id)
        validation = json.loads((root / "data_validation.json").read_text())
        if validation["status"] != "passed" or validation["input_sha256"] != sig["input_sha256"]:
            raise ValueError("data validation is missing or stale; run validate first")
        manifest = root / "protocol.json"
        if manifest.exists() and json.loads(manifest.read_text()) != sig:
            raise ValueError("run signature changed; choose a new run ID")
        preflight_path = root / "preflight.json"
        if args.stage == "preflight":
            if preflight_path.exists():
                result = json.loads(preflight_path.read_text())
                if result.get("status") != "passed" or result.get("protocol") != sig:
                    raise ValueError("preflight signature changed; choose a new run ID")
                print("Reusing the matching passed preflight.")
            else:
                result = preflight(config, sig)
                write_json(preflight_path, result)
                write_json(manifest, sig)
                print(json.dumps({"paired": result["paired"], "auxiliary": result["auxiliary"]}, indent=2))
            return
        result = json.loads(preflight_path.read_text())
        if result.get("status") != "passed" or result.get("protocol") != sig:
            raise ValueError("a matching passed preflight is required before encoding")
        run_queue(config, root, sig)


if __name__ == "__main__":
    main()
