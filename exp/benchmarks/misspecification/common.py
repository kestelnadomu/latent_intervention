"""Small operational helpers shared only by this experiment."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re

import torch
import yaml

from src.artifact_io import sha256_file, write_json

CONFIG = "configs/hz_misspecification.yaml"


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(
        str((tuple(value.shape), value.dtype)).encode() + value.numpy().tobytes()
    ).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read_config(path=CONFIG, *, run_id=None, pilot=False):
    cfg = yaml.safe_load(Path(path).read_text())
    if run_id is not None:
        cfg["run_id"] = run_id
    if pilot:
        cfg["run_id"] += "-pilot"
        cfg["seeds"] = [cfg["pilot"]["seed"]]
        cfg["max_epochs"] = cfg["pilot"]["max_epochs"]
    cfg["is_pilot"] = pilot
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", cfg["run_id"]):
        raise ValueError("run ID must be a simple directory name")
    if (
        cfg["tier"] != 1
        or cfg["threads_per_worker"] != 1
        or cfg["evaluation_samples"] != 64
        or cfg["evaluation_batch_size"] != 32
        or cfg["evaluation_seed"] != 20260930
        or cfg["bootstrap_resamples"] != 2000
        or cfg["deployment_seed"] != 42
        or cfg["loss"] != dict(movement=0.1, support=0.05, identity=1.0)
        or cfg["seeds"] != ([42] if pilot else [42, 43, 44, 45, 46])
    ):
        raise ValueError("incompatible misspecification protocol")
    for key in (
        "workers",
        "max_epochs",
        "patience",
        "editor_batch_size",
        "flow_batch_size",
    ):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"invalid {key}")
    return cfg


def root(cfg):
    return Path(cfg["reports_root"]) / cfg["run_id"]


def settings_for(cfg, method, seed):
    result = dict(cfg, seed=seed)
    result["batch_size"] = (
        cfg["editor_batch_size"]
        if method.endswith("editor")
        else cfg["flow_batch_size"]
    )
    return result


def job_key(job):
    return f"{job['space']}/{job['method']}/{job['operator'].replace('/', '--')}/seed-{job['seed']}"


def paths(cfg, job):
    suffix = Path(job["operator"].replace("/", "--")) / f"seed-{job['seed']}"
    model = (
        Path(cfg["models_root"])
        / job["space"]
        / f"h-{job['method']}"
        / "runs"
        / cfg["run_id"]
        / suffix
        / "selected"
        / "latent_intervention.pt"
    )
    report = root(cfg) / "fits" / job_key(job)
    return model, report / "training.json", report / "resume.pt"


@contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another writer holds {path}") from error
        yield


def write_record(path, record):
    value = {k: v for k, v in record.items() if k != "record_sha256"}
    write_json(path, dict(value, record_sha256=digest(value)))


def read_record(path):
    value = json.loads(Path(path).read_text())
    checksum = value.pop("record_sha256")
    if checksum != digest(value):
        raise ValueError(f"record checksum mismatch: {path}")
    return dict(value, record_sha256=checksum)


def verify_files(signature):
    for group in ("inputs", "source"):
        for path, expected in signature[group].items():
            if sha256_file(path) != expected:
                raise ValueError(f"{group} changed since preparation: {path}")


def seeded(seed):
    """Caller owns the fork_rng context when global RNG isolation is required."""
    import random
    import numpy as np

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
