"""Opt-in checks for the retained 270M decoder and frozen encoder."""

import gc
import os

import pytest
import torch

pytestmark = [
    pytest.mark.text_realizer_gpu,
    pytest.mark.skipif(
        os.environ.get("TEXT_REALIZER_GPU_TESTS") != "1",
        reason="set TEXT_REALIZER_GPU_TESTS=1 after caching the pinned models",
    ),
]


def test_real_complete_cv_update_and_reload(tmp_path):
    from exp.text_realizer.data import load_bundle, prepared_path
    from exp.text_realizer.experiment import enable_checkpointing
    from exp.text_realizer.preflight import token_audit
    from exp.text_realizer.runtime import gpu_lease
    from exp.text_realizer.settings import read_config, resolve
    from src.text_realizer.artifacts import (
        save_checkpoint,
        tensor_digest,
        verify_checkpoint,
    )
    from src.text_realizer.generation import generate
    from src.text_realizer.model import build_model, model_spec
    from src.text_realizer.training import optimizer_for, tokenize, update

    torch.set_num_threads(1)
    config = read_config()
    bundle = load_bundle(prepared_path(config))
    audited = token_audit(config, bundle)
    cache = str(resolve(config["paths"]["cache"]))
    with gpu_lease("cuda:0"):
        model = build_model(
            model_spec(config["model"], "lora-compact", audited["hidden_size"]),
            audited["generation"],
            device="cuda:0",
            cache_dir=cache,
        )
        enable_checkpointing(model, True)
        frozen = tensor_digest(
            {name: p for name, p in model.named_parameters() if not p.requires_grad}
        )
        targets = tokenize(model.tokenizer, bundle["fit"]["texts"][:1])
        optimizer = optimizer_for(model, config["training"])
        result = update(
            model, optimizer, bundle["fit"]["z"][:1], targets, [0], microbatch=1, clip=1
        )
        assert result["tokens"] == len(targets[0])
        assert (
            tensor_digest(
                {name: p for name, p in model.named_parameters() if not p.requires_grad}
            )
            == frozen
        )
        assert model.projector[0].weight.grad.abs().sum() > 0
        assert len(generate(model, bundle["fit"]["z"][:1], max_new_tokens=16)) == 1
        path = tmp_path / "diagnostic.pt"
        save_checkpoint(
            path, model, encoder=config["encoder"], provenance={"stage": "diagnostic"}
        )
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        assert verify_checkpoint(path, device="cuda:0", cache_dir=cache)["verified"]


def test_real_encoder_protocol_reproduces_factual_vector():
    from exp.text_realizer.campaign import make_cycle_encoder
    from exp.text_realizer.data import load_bundle, prepared_path
    from exp.text_realizer.runtime import gpu_lease
    from exp.text_realizer.settings import read_config

    torch.set_num_threads(1)
    config = read_config()
    fit = load_bundle(prepared_path(config))["fit"]
    with gpu_lease("cuda:0"):
        encoder = make_cycle_encoder(config, "cuda:0")
        z = encoder.encode(fit["texts"][:2], batch_size=1)
        torch.testing.assert_close(z, fit["z"][:2], atol=2e-5, rtol=2e-5)
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
