"""Equal-update comparison sampling and exact stream restoration."""

import pytest

from exp.text_realizer_cf.sampling import ArmSampler, CyclingStream
from exp.text_realizer_cf.selection import selection_key
from exp.text_realizer_cf.settings import read_config, run_paths


def test_both_arms_draw_32_and_oracle_draw_is_balanced():
    factual = ArmSampler(3200, 2379, "factual", 42)
    oracle = ArmSampler(3200, 2379, "oracle", 42)
    for _ in range(150):
        f = factual.take_batch()
        c = oracle.take_batch()
        assert len(f) == len(c) == 32
        assert all(0 <= index < 3200 for index in f)
        assert sum(index < 3200 for index in c) == 16
        assert sum(3200 <= index < 5579 for index in c) == 16


def test_sampler_resume_reproduces_future_batches():
    old = ArmSampler(7, 5, "oracle", 42)
    for _ in range(3):
        old.take_batch()
    saved = old.state_dict()
    expected = [old.take_batch() for _ in range(4)]
    restored = ArmSampler(7, 5, "oracle", 42)
    restored.load_state_dict(saved)
    assert [restored.take_batch() for _ in range(4)] == expected
    with pytest.raises(ValueError, match="sampler arm"):
        ArmSampler(7, 5, "factual", 42).load_state_dict(saved)
    with pytest.raises(ValueError, match="sampler state"):
        CyclingStream(3, 42).load_state_dict(saved["factual"])


def test_selection_tie_break_and_pinned_paths():
    record = {
        "score": 0.5,
        "character_similarity": 0.3,
        "factual_token_nll": 1.0,
        "step": 200,
    }
    earlier = {**record, "step": 100}
    assert selection_key(earlier) > selection_key(record)
    config = read_config()
    assert config["model"]["attention_implementation"] == "eager"
    assert config["training"]["updates"] == 600
    with pytest.raises(ValueError, match="unsafe run ID"):
        run_paths(config, "../outside")
    with pytest.raises(ValueError, match="unknown comparison arm"):
        run_paths(config, "pilot", "other")
