"""Synthetic-only checks for the downstream predictor p: Z -> Y and its benchmark."""

import torch

from src.downstream import (
    OutcomePredictor,
    confusion_matrix,
    grouped_confusion,
    load_outcome_predictor,
    save_outcome_predictor,
    train_outcome_predictor,
)
from exp.benchmarks.downstream import report, run


def test_confusion_matrix_rows_are_true_columns_predicted():
    true = torch.tensor([0, 0, 1, 2, 2, 2])
    pred = torch.tensor([0, 1, 1, 2, 0, 2])
    assert confusion_matrix(pred, true, 3).tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 2]]


def test_grouped_confusion_partitions_the_overall_matrix():
    generator = torch.Generator().manual_seed(0)
    true, pred = torch.randint(0, 3, (2, 200), generator=generator)
    groups = torch.randint(0, 4, (200,), generator=generator)
    result = grouped_confusion(pred, true, groups, 3, group_name="X", n_groups=4)
    assert list(result) == ["all", "X=0", "X=1", "X=2", "X=3"]
    total = sum(torch.tensor(result[f"X={k}"]) for k in range(4))
    assert total.tolist() == result["all"]
    assert int(total.sum()) == 200


def _separable(n, seed):
    generator = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 3, (n,), generator=generator)
    centers = torch.eye(3, 8) * 4
    return centers[y] + torch.randn(n, 8, generator=generator), y


def test_predictor_learns_and_round_trips(tmp_path):
    z_fit, y_fit = _separable(300, 0)
    z_val, y_val = _separable(100, 1)
    model = OutcomePredictor(latent_dim=8, n_classes=3, hidden_dim=16, n_hidden=1)
    training = train_outcome_predictor(
        model, z_fit, y_fit, z_val, y_val, max_epochs=40, patience=5, seed=0
    )
    assert training["best_epoch"] >= 1
    accuracy = model.probabilities(z_val).argmax(-1).eq(y_val).float().mean()
    assert accuracy > 0.9
    path = tmp_path / "p.pt"
    save_outcome_predictor(model, path, dict(case="synthetic"))
    loaded = load_outcome_predictor(path)
    assert loaded.metadata == dict(case="synthetic")
    assert torch.equal(loaded.probabilities(z_val), model.probabilities(z_val))


def test_no_edit_draw_reproduces_p_of_z_and_report_renders(tmp_path):
    z, y = _separable(50, 2)
    groups = torch.arange(50) % 4
    predictor = OutcomePredictor(latent_dim=8, n_classes=3, hidden_dim=4, n_hidden=1)
    predictor.eval()
    settings = dict(evaluation_samples=4, evaluation_seed=0, evaluation_batch_size=16)
    edited = run.edited_probabilities(
        predictor, None, "no_edit", z,
        intervention={"X": 3}, decoder=None, h_s=None, settings=settings,
    )
    assert torch.allclose(edited, predictor.probabilities(z))

    class Spec:
        def __init__(self, name, n):
            self.name, self.n_categories = name, n

    data = dict(
        labels={"Y": y, "Y'": y}, groups=groups, outcome=Spec("Y", 3), group=Spec("X", 4)
    )
    factual = run.score(predictor.probabilities(z), data, ("Y",))
    assert run.score(edited, data, ("Y",)) == factual
    rows = [dict(row="p(Z)", confusion=factual)] + [
        dict(row="p(h_Z(Z))", family="baseline", seed=s, confusion=run.score(edited, data, ("Y", "Y'")))
        for s in (42, 43)
    ]
    summed = report.aggregate(rows)[1]["confusion"]["Y"]["all"]
    assert sum(map(sum, summed)) == 100
    report.render(
        tmp_path,
        dict(
            hz_run_id="synthetic",
            intervention={"X": 3},
            evaluation=settings,
            dimensions={
                "8": dict(
                    encoder="synthetic_8",
                    test_units=50,
                    predictor=dict(best_epoch=1, best_validation_ce=1.0),
                    rows=rows,
                )
            },
        ),
    )
    text = (tmp_path / "summary.md").read_text()
    assert "p(h_Z(Z)) baseline vs Y'" in text and "| X=3 |" in text
