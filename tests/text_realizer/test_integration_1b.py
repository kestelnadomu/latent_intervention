"""Opt-in 1B backbone check across Gemma's local-attention window."""

import gc
import os

import pytest
import torch

pytestmark = [
    pytest.mark.text_realizer_gpu,
    pytest.mark.skipif(
        os.environ.get("TEXT_REALIZER_1B_GPU_TESTS") != "1",
        reason="set TEXT_REALIZER_1B_GPU_TESTS=1 after caching the pinned 1B model",
    ),
]


def test_real_1b_cache_across_local_attention_window():
    from exp.text_realizer.runtime import gpu_lease
    from exp.text_realizer.sanity import check_native_forward
    from exp.text_realizer.settings import PILOT_CONFIG, read_config, resolve
    from src.text_realizer.model import build_model, model_spec

    config = read_config(PILOT_CONFIG)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    with gpu_lease("cuda:0"):
        model = build_model(
            model_spec(config["model"], "lora-compact", 1152),
            {"max_new_tokens": 1024, "batch_size": 1},
            device="cuda:0",
            cache_dir=str(resolve(config["paths"]["cache"])),
        )
        z = torch.nn.functional.normalize(torch.randn(1, 768), dim=1)
        result = check_native_forward(model, z, [100, 200, 300, model.eos_id])
        assert result["attention_implementation"] == "eager"
        assert result["cache_test_sequence_length"] == 515
    del model
    gc.collect()
    torch.cuda.empty_cache()
