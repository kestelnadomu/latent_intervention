"""Independent P-only measuring instrument, never imported by h_Z objectives."""

import copy
from pathlib import Path
from time import perf_counter

import torch
import yaml

from exp.benchmarks.semantic_decoder.experiment import candidate_settings
from src.artifact_io import copy_atomic, save_torch, sha256_file
from src.semantic_decoder.model import SemanticDecoder, joint_nll, load_semantic_decoder
from src.semantic_decoder.training import _reset_trainable_modules
from src.latent_intervention.editor.training import rng_state, restore_rng
from .common import (
    digest,
    exclusive,
    read_record,
    root,
    seeded,
    tensor_digest,
    write_record,
)
from .data import bundle, load_reader_data


def deployment(cfg):
    return (
        Path(cfg["models_root"])
        / "evaluation_reader"
        / "runs"
        / cfg["run_id"]
        / "seed-42"
        / "semantic_decoder.pt"
    )


def labels(part, columns):
    return {c.name: part.states[:, i] for i, c in enumerate(columns)}


@torch.no_grad()
def probabilities(model, z):
    # Independent heads: no need for the 26,244-cell nine-variable joint.
    return {k: v.softmax(-1) for k, v in model(z).items()}


@torch.no_grad()
def calibration(model, part, columns, bins=10):
    result = {}
    heads = probabilities(model, part.z)
    for i, col in enumerate(columns):
        p = heads[col.name]
        confidence, prediction = p.max(-1)
        correct = prediction.eq(part.states[:, i]).float()
        ece, reliability = 0.0, []
        for b in range(bins):
            mask = (confidence >= b / bins) & (
                (confidence <= 1) if b == bins - 1 else (confidence < (b + 1) / bins)
            )
            if mask.any():
                accuracy, certainty = (
                    float(correct[mask].mean()),
                    float(confidence[mask].mean()),
                )
                ece += float(mask.float().mean()) * abs(accuracy - certainty)
                reliability.append(
                    dict(
                        bin=b,
                        units=int(mask.sum()),
                        accuracy=accuracy,
                        confidence=certainty,
                    )
                )
        result[col.name] = dict(
            accuracy=float(correct.mean()), ece=ece, reliability=reliability
        )
    return result


def fit_trial(data, settings, checkpoint, report_path, metadata, *, on_epoch=None):
    """Same Adam/NLL/selection procedure as g; local loop adds full epoch resume."""
    if data.role != "evaluation_reader":
        raise TypeError("evaluation reader requires its P-only boundary")
    checkpoint, report_path = Path(checkpoint), Path(report_path)
    signature = digest(
        dict(
            settings=settings,
            metadata=metadata,
            fit_ids=data.fit.ids,
            tune_ids=data.tune.ids,
            factual=[
                tensor_digest(t) for p in (data.fit, data.tune) for t in (p.z, p.states)
            ],
        )
    )
    meta = dict(
        metadata,
        evaluation_reader_signature=signature,
        role="independent_evaluation_only",
        fit_ids=list(data.fit.ids),
        tune_ids=list(data.tune.ids),
    )
    targets, tuning = labels(data.fit, data.columns), labels(data.tune, data.columns)
    if report_path.exists():
        record = read_record(report_path)
        if record["signature"] != signature:
            raise ValueError("evaluation reader signature mismatch")
        if record["status"] == "complete":
            if sha256_file(checkpoint) != record["checkpoint_sha256"]:
                raise ValueError("evaluation reader checkpoint changed")
            model = load_semantic_decoder(
                checkpoint,
                expected_columns=data.columns,
                expected_metadata={"evaluation_reader_signature": signature},
            )
            score = joint_nll(model, data.tune.z, tuning)
            if abs(score - record["validation_joint_nll"]) > 1e-6:
                raise ValueError("evaluation reader selected score does not reproduce")
            return record
        if not report_path.with_name("resume.pt").exists():
            raise FileNotFoundError(
                "incomplete evaluation reader is missing its resume snapshot"
            )
    if checkpoint.exists():
        load_semantic_decoder(
            checkpoint,
            expected_columns=data.columns,
            expected_metadata={"evaluation_reader_signature": signature},
        )
    seeded(42)
    model = SemanticDecoder(
        data.fit.z.shape[1],
        data.columns,
        settings["hidden_dim"],
        settings["n_hidden"],
        settings["dropout"],
    )
    torch.manual_seed(42)
    model.apply(_reset_trainable_modules)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    stop, plateau = settings["early_stopping"], settings["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=plateau["patience"],
        factor=plateau["factor"],
        min_lr=plateau["min_lr"],
        threshold=stop["min_delta"],
        threshold_mode="abs",
    )
    best, reference = float("inf"), float("inf")
    best_state, best_epoch, stale, history = None, 0, 0, []
    started, elapsed, updates = perf_counter(), 0.0, 0
    resume = report_path.with_name("resume.pt")
    dummy = torch.Generator().manual_seed(42)
    best_training_state = None

    def snapshot():
        save_torch(
            resume,
            dict(
                format="misspec_reader_resume_v1",
                signature=signature,
                current=model.state_dict(),
                best_state=best_state,
                best_training_state=best_training_state,
                best=best,
                reference=reference,
                best_epoch=best_epoch,
                stale=stale,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                rng=rng_state(dummy),
                history=history,
                updates=updates,
                elapsed_seconds=elapsed + perf_counter() - started,
            ),
        )

    if resume.exists():
        saved = torch.load(resume, weights_only=True, map_location="cpu")
        if (
            saved.get("format") != "misspec_reader_resume_v1"
            or saved["signature"] != signature
        ):
            raise ValueError("evaluation reader resume mismatch")
        model.load_state_dict(saved["current"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"], dummy)
        best, reference, best_epoch, stale = (
            saved[k] for k in ("best", "reference", "best_epoch", "stale")
        )
        best_state, history, updates, elapsed = (
            saved[k] for k in ("best_state", "history", "updates", "elapsed_seconds")
        )
        best_training_state = saved["best_training_state"]
    else:
        snapshot()
    for epoch in range(len(history), settings["epochs"]):
        if stale >= stop["patience"]:
            break
        model.train()
        total = 0.0
        for indices in torch.randperm(len(data.fit.z)).split(settings["batch_size"]):
            loss = model.nll(
                data.fit.z[indices], {k: v[indices] for k, v in targets.items()}
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite evaluation reader loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            updates += 1
            total += float(loss.detach()) * len(indices)
        score = joint_nll(model, data.tune.z, tuning)
        if not torch.isfinite(torch.tensor(score)):
            raise FloatingPointError("non-finite evaluation reader development score")
        if score < best:
            best, best_epoch, best_state = (
                score,
                epoch + 1,
                copy.deepcopy(model.state_dict()),
            )
        if score < reference - stop["min_delta"]:
            reference, stale = score, 0
        else:
            stale += 1
        scheduler.step(score)
        if best_epoch == epoch + 1:
            best_training_state = dict(
                optimizer=copy.deepcopy(optimizer.state_dict()),
                scheduler=copy.deepcopy(scheduler.state_dict()),
                rng=rng_state(dummy),
                epoch=best_epoch,
            )
        history.append(
            dict(
                epoch=epoch + 1,
                training_joint_nll=total / len(data.fit.z) * len(data.columns),
                validation_joint_nll=score,
                best_epoch=best_epoch,
                learning_rate=optimizer.param_groups[0]["lr"],
            )
        )
        snapshot()
        write_record(
            report_path, dict(signature=signature, status="running", history=history)
        )
        if on_epoch is not None:
            on_epoch(epoch + 1)
    if best_state is None:
        raise ValueError("no evaluation-reader checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    restored = joint_nll(model, data.tune.z, tuning)
    if abs(restored - best) > 1e-6:
        raise ValueError("evaluation reader restoration failed")
    model.save(checkpoint, metadata=meta)
    write_record(
        report_path,
        dict(
            signature=signature,
            status="complete",
            checkpoint=str(checkpoint),
            checkpoint_sha256=sha256_file(checkpoint),
            settings=settings,
            history=history,
            selected_epoch=best_epoch,
            seed=42,
            validation_joint_nll=restored,
            calibration=calibration(model, data.tune, data.columns),
            fit_ids=list(data.fit.ids),
            tune_ids=list(data.tune.ids),
            elapsed_seconds=elapsed + perf_counter() - started,
            optimizer_updates=updates,
            used_official_test=False,
        ),
    )
    return read_record(report_path)


def train(cfg, signature):
    from .common import verify_files

    with exclusive(root(cfg) / "evaluation_reader" / "reader.lock"):
        verify_files(signature)
        loaded = bundle(cfg)
        data = load_reader_data(cfg, loaded)
        artifact = loaded[1]
        base = yaml.safe_load(Path(cfg["reader_protocol"]).read_text())["training"]
        if cfg.get("is_pilot"):
            base["epochs"] = min(base["epochs"], cfg["max_epochs"])
        metadata = dict(
            latent_artifact_sha256=artifact.artifact_sha256,
            encoder=artifact.encoder_info,
            protocol_sha256=digest(signature),
            role="independent_evaluation_only",
        )
        trials = []
        for i, candidate in enumerate(candidate_settings(base)):
            path = root(cfg) / "evaluation_reader" / "search" / f"trial-{i:03d}"
            trial = fit_trial(
                data,
                dict(base, **candidate),
                path / "semantic_decoder.pt",
                path / "training.json",
                dict(metadata, trial=i),
            )
            trials.append(dict(trial=i, report=trial))
        winner = min(
            trials, key=lambda t: (t["report"]["validation_joint_nll"], t["trial"])
        )
        destination = deployment(cfg)
        selected = winner["report"]
        if (
            destination.exists()
            and sha256_file(destination) != selected["checkpoint_sha256"]
        ):
            raise FileExistsError("conflicting published evaluation reader")
        copy_atomic(selected["checkpoint"], destination)
        verify_files(signature)
        record = dict(
            protocol_sha256=digest(signature),
            checkpoint=str(destination),
            checkpoint_sha256=sha256_file(destination),
            selected_trial=winner["trial"],
            selection="minimum development joint NLL; publish selected seed-42 weights without refitting",
            validation_joint_nll=selected["validation_joint_nll"],
            calibration=selected["calibration"],
            role="independent_evaluation_only",
            used_official_test=False,
            fit_ids=list(data.fit.ids),
            tune_ids=list(data.tune.ids),
        )
        write_record(root(cfg) / "evaluation_reader" / "selection.json", record)
        return record
