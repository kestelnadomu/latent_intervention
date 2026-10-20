"""Offline split and alignment tests for the CF-specific preparer."""

from __future__ import annotations

import copy

import pytest
import torch

from exp.text_realizer_cf.data import assemble


@pytest.fixture(scope="module")
def synthetic_case():
    split = {
        "fit": list(range(1, 3201)),
        "tune": list(range(3201, 3601)),
        "audit": list(range(3601, 4001)),
        "official_test": list(range(4001, 5001)),
    }
    factual = torch.zeros(4000, 768)
    factual[:, 0] = 1
    oracle = factual.clone()
    flags = torch.tensor([key % 4 == 0 for key in range(1, 4001)])
    oracle[~flags, 0] = 0
    oracle[~flags, 1] = 1
    parts = {
        name: {
            "ids": ids,
            "z": factual[[key - 1 for key in ids]],
            "texts": [f"Factual CV {key}" for key in ids],
        }
        for name, ids in (("fit", split["fit"]), ("tune", split["tune"]))
    }
    payload = {
        "train_ids": list(range(1, 4001)),
        "z_prime": oracle,
        "is_identity": flags,
    }
    pairs = {
        key: {
            "split": "train" if key <= 4000 else "test",
            "is_identity": str(key % 4 == 0),
        }
        for key in range(1, 5001)
    }
    cf = {
        key: {"text": f"Factual CV {key}" if key % 4 == 0 else f"Changed CV {key}"}
        for key in range(1, 3601)
    }
    return split, parts, payload, pairs, cf


def test_assemble_keeps_train_selection_evaluation_worlds_separate(synthetic_case):
    fit, selection, evaluation, splits = assemble(*synthetic_case)
    assert len(fit["factual"]["ids"]) == 3200
    assert len(fit["changed"]["ids"]) == 2400
    assert selection["factual"]["ids"] == splits["selection"]
    assert len(splits["selection"]) == 80
    assert len(splits["evaluation"]) == 320
    assert int(evaluation["is_identity"].sum()) == 80
    assert set(fit["changed"]["ids"]) <= set(splits["fit"])
    assert not set(splits["audit"] + splits["official_test"]) & set(
        fit["factual"]["ids"] + fit["changed"]["ids"] + selection["factual"]["ids"]
    )
    assert "oracle" not in selection
    assert "evaluation" not in fit


def test_wrong_identity_text_vector_and_test_id_are_rejected(synthetic_case):
    split, parts, payload, pairs, cf = synthetic_case
    wrong_text = dict(cf)
    wrong_text[4] = {"text": "not an identity copy"}
    with pytest.raises(ValueError, match="identity text/vector"):
        assemble(split, parts, payload, pairs, wrong_text)
    wrong_vector = {**payload, "z_prime": payload["z_prime"].clone()}
    wrong_vector["z_prime"][3, 0] = 0.5
    with pytest.raises(ValueError):
        assemble(split, parts, wrong_vector, pairs, cf)
    wrong_ids = {**payload, "train_ids": payload["train_ids"].copy()}
    wrong_ids["train_ids"][-1] = 4001
    with pytest.raises(ValueError, match="official-test"):
        assemble(split, parts, wrong_ids, pairs, cf)
    wrong_flag = {**payload, "is_identity": payload["is_identity"].clone()}
    wrong_flag["is_identity"][0] = True
    with pytest.raises(ValueError, match="identity flag"):
        assemble(split, parts, wrong_flag, pairs, cf)


def test_changed_text_equal_to_factual_is_rejected(synthetic_case):
    split, parts, payload, pairs, cf = synthetic_case
    changed = copy.deepcopy(cf)
    changed[1]["text"] = "Factual CV 1"
    with pytest.raises(ValueError, match="changed counterfactual"):
        assemble(split, parts, payload, pairs, changed)
