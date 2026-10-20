import pytest
import torch

from src.text_realizer import artifacts
from src.text_realizer.training import (
    optimizer_for,
    restore_resume,
    save_resume,
    scheduler_for,
    update,
)


def test_portable_checkpoint_and_corruption(tiny_factory, tmp_path, monkeypatch):
    model = tiny_factory(lora=True).eval()
    path = tmp_path / "model.pt"
    artifacts.save_checkpoint(
        path, model, encoder={"dimension": 768}, provenance={"stage": "test"}
    )
    monkeypatch.setattr(
        artifacts, "build_model", lambda *args, **kwargs: tiny_factory(lora=True)
    )
    assert artifacts.verify_checkpoint(path)["verified"]
    loaded = artifacts.load_text_realizer(path)
    z = torch.randn(2, 768)
    assert loaded.decode(z) == model.decode(z)
    payload = torch.load(path, weights_only=True)
    payload["weights"][next(iter(payload["weights"]))].flatten()[0] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="checksum"):
        artifacts.read_checkpoint(path)


def test_resume_restores_optimizer_scheduler_rng_sampler_and_early_stop(
    tiny_factory, numerical_settings, tmp_path
):
    model = tiny_factory(lora=True)
    optimizer = optimizer_for(model, numerical_settings)
    scheduler = scheduler_for(optimizer, 10, 0.1)
    z, targets = torch.randn(2, 768), [[4, 5, 2], [7, 8, 2]]
    update(model, optimizer, z, targets, [0, 1], microbatch=1, clip=1)
    scheduler.step()
    progress = {
        "epoch": 3,
        "offset": 2,
        "order": [1, 0],
        "bad_evaluations": 4,
        "step": 1,
    }
    path = tmp_path / "last.pt"
    save_resume(path, model, optimizer, scheduler, progress, identity="unchanged")
    update(model, optimizer, z, targets, [1, 0], microbatch=1, clip=1)
    scheduler.step()
    expected = model.trainable_state()
    assert (
        restore_resume(path, model, optimizer, scheduler, identity="unchanged")
        == progress
    )
    update(model, optimizer, z, targets, [1, 0], microbatch=1, clip=1)
    scheduler.step()
    for name, value in expected.items():
        torch.testing.assert_close(model.trainable_state()[name], value, atol=0, rtol=0)
    with pytest.raises(ValueError, match="identity"):
        restore_resume(path, model, optimizer, scheduler, identity="changed")


def test_changed_cached_tokenizer_is_rejected_before_model_load(
    tiny_factory, tmp_path, monkeypatch
):
    from src.artifact_io import sha256_file

    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("original pinned tokenizer")
    path = tmp_path / "model.pt"
    artifacts.save_checkpoint(
        path,
        tiny_factory(),
        encoder={"dimension": 768},
        provenance={
            "base_files_sha256": {"decoder": {"tokenizer.json": sha256_file(tokenizer)}}
        },
    )
    tokenizer.write_text("changed tokenizer")
    monkeypatch.setattr(artifacts, "cached_decoder", lambda *args: str(tmp_path))
    with pytest.raises(ValueError, match="tokenizer file changed"):
        artifacts.load_text_realizer(path)
