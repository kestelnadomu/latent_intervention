"""CPU encoding of the selected LangVAE checkpoint; no fine-tuning or g training.

Reuse the existing encoder, pair alignment, writer and loader unchanged. A forward
hook only validates/logs batches; it never replaces model outputs. Publish the
complete validated artifact directory together, without overwriting a baseline.
"""

import argparse
import copy
import fcntl
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch

from src.artifact_io import sha256_file, write_json
from src.config import load_config
from src.encoder import make_encoder
from exp.langvae_ft.artifacts import ROOT, packages, read_checkpoint, verify_checkpoint
from src.pair_encoding import encode_pairs, load_latent_artifact


class ProgressEncoder:
    """Observe native batches without changing tokenization, batching or tensors."""

    def __init__(self, config, update):
        self.encoder = make_encoder(config)
        self.latent_dim = self.encoder.latent_dim
        self.update = update

    def encode(self, texts, *, deterministic=True, batch_size=32):
        started = perf_counter()
        completed = 0
        batches = 0

        def observed(_module, _inputs, output):
            nonlocal completed, batches
            values = output.embedding
            if (
                values.ndim != 2
                or values.shape[1] != 128
                or not torch.isfinite(values).all()
            ):
                raise ValueError("invalid LangVAE posterior means")
            completed += len(values)
            batches += 1
            if batches == 1 or batches % 20 == 0 or completed == len(texts):
                elapsed = perf_counter() - started
                remaining = elapsed * (len(texts) - completed) / completed
                self.update(
                    phase="encoding",
                    completed_texts=completed,
                    total_texts=len(texts),
                    encoding_elapsed_seconds=elapsed,
                    eta_seconds=remaining,
                )
                print(
                    f"LangVAE-FT {completed}/{len(texts)} texts; "
                    f"elapsed {elapsed:.1f}s; ETA {remaining:.1f}s",
                    flush=True,
                )

        handle = self.encoder.model.encoder.register_forward_hook(observed)
        try:
            return self.encoder.encode(
                texts, deterministic=deterministic, batch_size=batch_size
            )
        finally:
            handle.remove()


def selected_checkpoint(config):
    scope = json.loads((ROOT / "configs/langvae_ft_scope.json").read_text())
    checkpoint = (ROOT / scope["selected_checkpoint"]).resolve()
    enc = config["encoder"]
    if (
        enc.get("variant") != "langvae"
        or enc.get("tag") != "langvae_ft"
        or not enc.get("local_checkpoint")
        or Path(enc["local_checkpoint"]).resolve() != checkpoint
        or enc.get("device") != "cpu"
        or enc.get("deterministic") is not True
        or enc.get("max_len") != 512
    ):
        raise ValueError(
            "use the selected checkpoint's full-CV CPU encoding configuration"
        )
    if sha256_file(checkpoint / "sha256.json") != scope["checkpoint_manifest_sha256"]:
        raise ValueError("checkpoint manifest differs from the Git-pinned selection")
    metadata = read_checkpoint(checkpoint)
    if metadata["stage"] != "train" or metadata["epoch"] != 24:
        raise ValueError("expected the selected 24-epoch training checkpoint")
    return checkpoint


def run(config_path, run_id, threads=4):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) or threads < 1:
        raise ValueError("invalid run ID or thread count")
    config_path = Path(config_path)
    config = load_config(path=config_path)
    output = Path(config["paths"]["latents"]).resolve()
    expected = ROOT / "data/latents/talent/langvae_ft/z_pairs.pt"
    if output != expected:
        raise ValueError(f"expected the isolated destination {expected}")
    reports = ROOT / "reports/talent/langvae_ft/encoding"
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = reports / run_id
        status_path = report / "status.json"
        pending = output.parent.with_name(f".langvae_ft-{run_id}.pending")
        if output.parent.exists() or pending.exists() or status_path.exists():
            raise FileExistsError(
                "output/run already exists; inspect it before starting a fresh run"
            )
        status = {
            "state": "running",
            "pid": os.getpid(),
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

        def update(**values):
            status.update(values, updated_at=datetime.now(timezone.utc).isoformat())
            write_json(status_path, status)

        try:
            torch.set_num_threads(threads)
            config_hash = sha256_file(config_path)
            checkpoint = selected_checkpoint(config)
            existing = {
                str(p): sha256_file(p) for p in output.parent.parent.glob("*/z_pairs.*")
            }
            write_json(
                report / "protocol.json",
                {
                    "config": config,
                    "config_sha256": config_hash,
                    "packages": packages(),
                    "threads": threads,
                    "git_revision": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"],
                        cwd=ROOT,
                        text=True,
                    ).strip(),
                    "source_sha256": {
                        str(p.relative_to(ROOT)): sha256_file(p)
                        for p in sorted(
                            [
                                *(ROOT / "src").rglob("*.py"),
                                *(ROOT / "exp/langvae_ft").glob("*.py"),
                                *(ROOT / "exp/encoding").glob("langvae_ft_queue.py"),
                            ]
                        )
                    },
                    "existing_artifacts_sha256": existing,
                },
            )
            update(phase="verifying_checkpoint")
            verification = verify_checkpoint(checkpoint, torch.device("cpu"))
            write_json(report / "checkpoint_verification.json", verification)
            print("Pinned checkpoint verified; starting full-CV encoding.", flush=True)
            staged = copy.deepcopy(config)
            staged["paths"]["latents"] = str(pending / "z_pairs.pt")
            update(phase="loading_encoder")
            encode_pairs(
                staged, encoder_factory=lambda enc: ProgressEncoder(enc, update)
            )
            update(phase="validating_artifact")
            artifact = load_latent_artifact(staged)
            selected_checkpoint(config)
            if sha256_file(config_path) != config_hash:
                raise ValueError("encoding configuration changed during this run")
            if any(sha256_file(path) != digest for path, digest in existing.items()):
                raise ValueError("an existing baseline changed during this run")
            if output.parent.exists():
                raise FileExistsError(f"refusing to overwrite {output.parent}")
            pending.rename(output.parent)
            load_latent_artifact(config)
            update(
                state="completed",
                phase="completed",
                output=str(output),
                artifact_sha256=artifact.artifact_sha256,
                factual_units=len(artifact.ids),
                test_units=len(artifact.test_ids),
                identity_test_units=int(artifact.is_identity.sum()),
                latent_dimension=int(artifact.z.shape[1]),
                existing_artifacts_unchanged=True,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            print(f"Completed and validated: {output}", flush=True)
        except BaseException as error:
            update(state="failed", error=f"{type(error).__name__}: {error}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="models/langvae_ft/encoding.yaml")
    parser.add_argument("--run-id", default="langvae-ft-epoch24-v1")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    os.chdir(ROOT)
    run(args.config, args.run_id, args.threads)


if __name__ == "__main__":
    main()
