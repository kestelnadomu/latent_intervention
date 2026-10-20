import pytest
import torch

from exp.text_realizer.campaign import choose
from exp.text_realizer.evaluation import derangement, retrieval


def test_wrong_conditioning_never_maps_an_id_to_itself():
    for n in (2, 3, 16, 400):
        indices = derangement(n)
        assert (indices != torch.arange(n)).all()
        assert sorted(indices.tolist()) == list(range(n))
        assert torch.equal(indices, derangement(n))


def test_selection_thresholds_ties_and_no_qualified_model():
    def result(char, nll_gap=0.1, char_gap=0.03):
        return {
            "conditioning": {"nll_gap": nll_gap, "character_gap": char_gap},
            "conditions": {"correct": {"summary": {"character_similarity": char}}},
            "cycle": {"cosine_invalid_scored_minus_one": 0.7},
            "checkpoint_sha256": "abc",
        }

    thresholds = {"nll_gap": 0.05, "character_gap": 0.02}
    selected = choose(
        {"b": result(0.5), "a": result(0.5), "c": result(0.9, char_gap=0)}, thresholds
    )
    assert selected["selected"] == "a"
    assert selected["deployment_seed"] == 42
    assert choose({"a": result(0.8, nll_gap=0)}, thresholds)["selected"] is None


def test_lexical_metrics_bootstrap_and_retrieval_only_use_fit():
    pytest.importorskip("rapidfuzz")
    pytest.importorskip("sacrebleu")
    from src.text_realizer.metrics import paired_bootstrap, text_metrics

    scores = text_metrics("a  b", "a b")
    assert scores["exact_raw"] == 0 and scores["exact_normalized"] == 1
    assert scores["character_similarity"] == 1
    extra = text_metrics("a b", "a b c")
    assert extra["aligned_token_accuracy"] == 1 and extra["token_error_rate"] == 0.5
    fit = {"ids": [8, 9], "texts": ["fit one", "fit two"], "z": torch.eye(2)}
    part = {"ids": [10], "texts": ["held out"], "z": torch.tensor([[0.0, 1.0]])}
    assert retrieval(part, fit)["rows"][0]["fit_id"] == 9
    ci = paired_bootstrap([0.1, 0.2], samples=100, seed=1)
    assert ci["lower_95"] > 0


def test_independent_facts_do_not_fill_missing_predictions():
    from exp.text_realizer.facts import evaluate

    result = evaluate(
        [
            "I am from Germany, hold a Master's degree from a university ranked in the global top 100, and completed nine projects."
        ],
        ["I am from France and completed two projects."],
    )
    fields = result["fields"]
    assert fields["country"]["accuracy_on_scorable_references"] == 0
    assert fields["degree"]["unparseable_predictions"] == 1
    assert fields["degree"]["accuracy_on_scorable_references"] == 0
    assert fields["project_count"]["accuracy_on_scorable_references"] == 0
