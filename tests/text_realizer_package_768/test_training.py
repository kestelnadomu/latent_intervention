import pytest
import torch

from exp.text_realizer_package_768 import training
from exp.text_realizer_package_768.data import prepare
from exp.text_realizer_package_768.settings import create_plan, run_paths
from src.artifact_io import write_json
from src.text_realizer.artifacts import read_checkpoint


def test_training_loop_resume_matches_uninterrupted(
    package_config, tiny_model, monkeypatch
):
    config = package_config
    config["training"].update(epochs=2, effective_batch=2, checkpoint_every=1)
    prepared = prepare(config)
    ready = {
        "hidden_size": 16,
        "generation": {"max_new_tokens": 12, "batch_size": 1},
        "microbatch": 1,
        "base_files_sha256": {},
    }
    monkeypatch.setattr(training, "readiness", lambda *args, **kwargs: ready)
    monkeypatch.setattr(training, "build_model", tiny_model)

    def setup(run_id):
        models, reports = run_paths(config, run_id)
        create_plan(config, prepared, reports)
        write_json(reports / "preflight/ready.json", ready)
        return models, reports

    full_models, _ = setup("uninterrupted")
    training.train(config, "uninterrupted", device="cpu")
    resumed_models, _ = setup("resumed")
    real_update = training.update
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
        return real_update(*args, **kwargs)

    monkeypatch.setattr(training, "update", interrupt)
    with pytest.raises(KeyboardInterrupt):
        training.train(config, "resumed", device="cpu")
    monkeypatch.setattr(training, "update", real_update)
    training.train(config, "resumed", device="cpu", resume=True)
    expected = torch.load(full_models / "last.pt", weights_only=True)
    actual = torch.load(resumed_models / "last.pt", weights_only=True)
    assert expected["progress"]["history"] == actual["progress"]["history"]
    for name, value in expected["weights"].items():
        torch.testing.assert_close(actual["weights"][name], value, atol=0, rtol=0)
    metadata = read_checkpoint(resumed_models / "best.pt")["metadata"]
    assert metadata["spec"]["latent_dim"] == 768
    assert metadata["provenance"]["used_counterfactual_latent"] is False
    with pytest.raises(ValueError, match="already started"):
        training.train(config, "resumed", device="cpu")


def test_768_loss_mask_and_checkpoint_reload(tiny_model, tmp_path, monkeypatch):
    from src.text_realizer import artifacts
    from src.text_realizer.training import optimizer_for, update

    torch.manual_seed(1)
    model = tiny_model()
    z = torch.randn(2, 768)
    targets = [[3, 4, 2], [5, 2]]
    inputs = model.teacher_inputs(z, targets)
    assert inputs["labels"].ne(-100).sum().item() == 5
    assert inputs["labels"][:, : model.prefix_tokens + 1].eq(-100).all()
    frozen = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if not p.requires_grad
    }
    optimizer = optimizer_for(
        model, {"projector_lr": 0.01, "lora_lr": 0.01, "weight_decay": 0.01}
    )
    update(model, optimizer, z, targets, [0, 1], microbatch=1, clip=1)
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.projector.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for name, p in model.named_parameters()
        if "lora_" in name
    )
    for name, p in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(p, frozen[name], atol=0, rtol=0)
    path = tmp_path / "model.pt"
    artifacts.save_checkpoint(path, model, encoder={"dimension": 768}, provenance={})

    def rebuild(*args, **kwargs):
        torch.manual_seed(1)
        return tiny_model()

    monkeypatch.setattr(artifacts, "build_model", rebuild)
    assert artifacts.verify_checkpoint(path)["verified"]
    loaded = artifacts.load_text_realizer(path)
    assert loaded.decode(z) == model.decode(z)
    with pytest.raises(ValueError, match="invalid latent batch"):
        loaded.decode(torch.randn(1, 256))
