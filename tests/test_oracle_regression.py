from __future__ import annotations

import pytest
import torch

from src.oracle_regression import (
    OracleRegression,
    regression_metrics,
    train_oracle_regression,
)


@pytest.mark.parametrize("dimension", [256, 768])
def test_residual_identity_initialization_and_portable_fixed_intervention(
    tmp_path, dimension
):
    model = OracleRegression(dimension, {"X": 3}, hidden_dim=8, n_hidden=2, dropout=0.0)
    z = torch.randn(3, dimension)
    assert torch.equal(model.predict(z), z)
    with torch.no_grad():
        model.residual[-1].bias.fill_(0.2)
    path = tmp_path / "oracle.pt"
    model.save(path, metadata={"seed": 42})
    restored = OracleRegression.load(
        path, expected_metadata={"seed": 42}, expected_intervention={"X": 3}
    )
    assert torch.equal(model.predict(z), restored.predict(z))
    assert not restored.training
    with pytest.raises(ValueError, match="trained intervention"):
        restored.predict(z, intervention={"X": 2})
    with pytest.raises(ValueError, match="intervention mismatch"):
        OracleRegression.load(path, expected_intervention={})
    with pytest.raises(ValueError, match="provenance"):
        OracleRegression.load(path, expected_metadata={"seed": 43})


def test_supervised_fit_is_reproducible_restores_best_and_does_not_normalize():
    generator = torch.Generator().manual_seed(5)
    z = torch.randn(40, 4, generator=generator)
    target = z + torch.tensor([0.3, -0.2, 0.1, 0.4])
    summaries, predictions = [], []
    rng_before = torch.random.get_rng_state()
    for _ in range(2):
        model = OracleRegression(4, {"X": 3}, hidden_dim=16, n_hidden=2, dropout=0.1)
        rng = torch.random.get_rng_state()
        summary = train_oracle_regression(
            model,
            z[:32],
            target[:32],
            validation_latents=z[32:],
            validation_targets=target[32:],
            epochs=100,
            batch_size=8,
            lr=1e-2,
            patience=20,
            seed=17,
        )
        assert torch.equal(torch.random.get_rng_state(), rng)
        prediction = model.predict(z[32:])
        metrics = regression_metrics(prediction, target[32:])
        assert metrics["mse"] < 0.005
        assert metrics["mse"] == pytest.approx(summary["best_validation_mse"], rel=1e-5)
        assert summary["best_validation_mse"] == min(
            row["validation_mse"] for row in summary["history"]
        )
        assert not torch.allclose(prediction.norm(dim=1), torch.ones(8))
        summaries.append(summary)
        predictions.append(prediction)
    assert torch.equal(predictions[0], predictions[1])
    assert summaries[0]["history"] == summaries[1]["history"]
    torch.random.set_rng_state(rng_before)


def test_early_stopping_uses_validation_and_restores_literal_best():
    z = torch.randn(8, 4, generator=torch.Generator().manual_seed(4))
    model = OracleRegression(4, {}, hidden_dim=8, n_hidden=1, dropout=0)
    summary = train_oracle_regression(
        model,
        z[:6],
        z[:6] + 0.5,
        validation_latents=z[6:],
        validation_targets=z[6:] + 0.5,
        epochs=20,
        batch_size=3,
        min_delta=100.0,
        patience=1,
    )
    assert summary["epochs_run"] == 2
    assert summary["stopped_early"]
    assert regression_metrics(model.predict(z[6:]), z[6:] + 0.5)[
        "mse"
    ] == pytest.approx(summary["best_validation_mse"])


@pytest.mark.parametrize("invalid", ["nan", "width", "rows", "empty"])
def test_fit_rejects_malformed_pairs(invalid):
    model = OracleRegression(4, {}, hidden_dim=8)
    z, target = torch.zeros(3, 4), torch.zeros(3, 4)
    if invalid == "nan":
        target[0, 0] = float("nan")
    elif invalid == "width":
        target = torch.zeros(3, 5)
    elif invalid == "rows":
        target = torch.zeros(2, 4)
    else:
        target = torch.zeros(0, 4)
    with pytest.raises(ValueError):
        train_oracle_regression(
            model, z, target, validation_latents=z, validation_targets=z, epochs=1
        )
