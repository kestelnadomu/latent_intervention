"""Synthetic, offline checks for BERTScore alignment, truncation guards and resume."""

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from exp.text_realizer_bertscore import __main__ as bs
from src.artifact_io import write_json


@pytest.fixture
def comparisons(tmp_path):
    path = tmp_path / "comparisons.jsonl"
    rows = [
        {
            "id": key,
            "reference": f"Reference  CV {key}",
            "generated": {
                f"{model}/{arm}": f"Generated CV {key}"
                for model in bs.DECODERS
                for arm in bs.CONDITIONS
            },
        }
        for key in (7, 2, 9)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.fixture
def fake_scorer(monkeypatch, tmp_path):
    state = {"calls": 0, "fail_on": None}

    class Tokenizer:
        def encode(self, text, **kwargs):
            assert kwargs == {
                "add_special_tokens": True,
                "add_prefix_space": True,
                "truncation": False,
            }
            return [0] * (len(text.split()) + 2)

    class Scorer:
        _tokenizer = Tokenizer()
        hash = "offline-fixture"

        def __init__(self, **kwargs):
            assert kwargs["num_layers"] == 17
            assert not kwargs["idf"] and not kwargs["rescale_with_baseline"]

        def score(self, candidates, references, **kwargs):
            state["calls"] += 1
            if state["calls"] == state["fail_on"]:
                raise RuntimeError("simulated interruption")
            assert len(candidates) == len(references)
            return tuple(
                torch.full((len(candidates),), value) for value in (0.8, 0.6, 0.7)
            )

    monkeypatch.setitem(sys.modules, "bert_score", SimpleNamespace(BERTScorer=Scorer))
    monkeypatch.setattr(bs, "snapshot", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(
        bs, "versions", lambda names: {name: "fixture" for name in names}
    )
    monkeypatch.setattr(bs, "CHUNK_SIZE", 2)
    state["tokenizer"] = Tokenizer()
    return state


def test_id_alignment_and_duplicate_rejection(comparisons):
    rows = bs.load_pairs(comparisons, "256-270m")
    assert [(row["id"], row["condition"]) for row in rows] == [
        (key, arm) for key in (7, 2, 9) for arm in bs.CONDITIONS
    ]
    assert rows[0]["reference"] == "Reference CV 7"
    with comparisons.open("a") as stream:
        stream.write(comparisons.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        bs.load_pairs(comparisons, "256-270m")


def test_missing_condition_and_empty_text_are_rejected(comparisons):
    row = json.loads(comparisons.read_text().splitlines()[0])
    row["generated"].pop("768-1b/oracle")
    comparisons.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="eight"):
        bs.load_pairs(comparisons, "256-270m")
    row["generated"]["768-1b/oracle"] = " "
    comparisons.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="empty"):
        bs.load_pairs(comparisons, "256-270m")


def test_full_text_length_guard(fake_scorer):
    pairs = [{"reference": "word " * 510, "prediction": "CV"}]
    assert bs.check_lengths(pairs, fake_scorer["tokenizer"]) == 512
    pairs[0]["reference"] += "extra"
    with pytest.raises(ValueError, match="truncation is forbidden"):
        bs.check_lengths(pairs, fake_scorer["tokenizer"])


def test_resume_and_changed_input_refusal(comparisons, fake_scorer, tmp_path):
    output = tmp_path / "scores"
    fake_scorer["fail_on"] = 2
    with pytest.raises(RuntimeError, match="interruption"):
        bs.score(comparisons, output, "256-270m", "cpu", tmp_path)
    assert bs.read_json(output / "256-270m.json")["completed"] == 2
    bs.score(comparisons, output, "256-270m", "cpu", tmp_path)
    saved = bs.read_json(output / "256-270m.json")
    assert saved["complete"] and saved["completed"] == 6
    assert fake_scorer["calls"] == 4
    bs.score(comparisons, output, "256-270m", "cpu", tmp_path)
    assert fake_scorer["calls"] == 4
    comparisons.write_text(comparisons.read_text().replace("Reference", "Changed"))
    with pytest.raises(ValueError, match="inputs/settings changed"):
        bs.score(comparisons, output, "256-270m", "cpu", tmp_path)


def test_complete_merge_and_corruption_guard(comparisons, fake_scorer, tmp_path):
    output = tmp_path / "scores"
    for decoder in bs.DECODERS:
        bs.score(comparisons, output, decoder, "cpu", tmp_path)
    bs.merge(comparisons, output, tmp_path)
    summary = bs.read_json(output / "summary.json")
    assert len(summary["metrics"]) == 8
    assert all(
        row["units"] == 3 and row["f1"] == pytest.approx(0.7)
        for row in summary["metrics"]
    )
    assert len((output / "per_cv.csv").read_text().splitlines()) == 25
    target = output / "768-1b.json"
    saved = bs.read_json(target)
    saved["rows"][0]["f1"] = 0.99
    write_json(target, saved)
    with pytest.raises(ValueError, match="corrupted"):
        bs.merge(comparisons, output, tmp_path)


def test_merge_refuses_incomplete_worker(comparisons, fake_scorer, tmp_path):
    output = tmp_path / "scores"
    for decoder in bs.DECODERS:
        bs.score(comparisons, output, decoder, "cpu", tmp_path)
    target = output / "768-1b.json"
    saved = bs.read_json(target)
    saved["complete"] = False
    write_json(target, saved)
    with pytest.raises(ValueError, match="incomplete"):
        bs.merge(comparisons, output, tmp_path)
