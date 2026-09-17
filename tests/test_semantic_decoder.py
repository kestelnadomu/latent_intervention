from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from src.schema import ColumnSpec, flat_state_index
from src.semantic_decoder import (
    SEMANTIC_DECODER_FORMAT_VERSION,
    SemanticAutoRegDecoder,
    SemanticDecoder,
    accuracy,
    calibration_metrics,
    load_semantic_decoder,
    make_semantic_decoder,
    targets_from_dataframe,
    train_semantic_decoder,
)


def _checkpoint_provenance() -> dict:
    return {
        "latent_artifact_sha256": "abc",
        "encoder": {"encoder_variant": "nomic", "encoder": "stub-model"},
    }


@pytest.fixture
def columns() -> list[ColumnSpec]:
    return [ColumnSpec("a", 2), ColumnSpec("b", 3)]


@pytest.fixture
def sample(
    columns: list[ColumnSpec],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    del columns
    latents = torch.tensor(
        [[-1.0, 0.5, 0.2], [0.0, -0.1, 0.7], [1.0, 0.3, -0.4], [0.2, 0.8, 0.0]]
    )
    targets = {
        "a": torch.tensor([0, 1, 1, 0]),
        "b": torch.tensor([2, 1, 0, 2]),
    }
    return latents, targets


@pytest.mark.parametrize("variant", ["independent", "autoregressive"])
def test_variants_expose_normalized_joint_and_marginals(
    variant: str,
    columns: list[ColumnSpec],
    sample: tuple[torch.Tensor, dict[str, torch.Tensor]],
) -> None:
    latents, targets = sample
    decoder = make_semantic_decoder(
        variant,
        latent_dim=latents.shape[1],
        columns=columns,
        hidden_dim=7,
        n_hidden=1,
        dropout=0.0,
        embed_dim=4,
    )

    loss = decoder.nll(latents, targets)
    joint = decoder.log_joint(latents)
    marginals = decoder.marginal_probabilities(latents)
    predictions = decoder.predict(latents)

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert joint.shape == (len(latents), 6)
    assert torch.allclose(joint.exp().sum(dim=-1), torch.ones(len(latents)))
    assert set(marginals) == {"a", "b"}
    assert torch.allclose(marginals["a"].sum(dim=-1), torch.ones(len(latents)))
    assert torch.allclose(marginals["b"].sum(dim=-1), torch.ones(len(latents)))
    assert {name: tuple(value.shape) for name, value in predictions.items()} == {
        "a": (len(latents),),
        "b": (len(latents),),
    }


def test_independent_joint_order_and_loss_scale(
    columns: list[ColumnSpec],
    sample: tuple[torch.Tensor, dict[str, torch.Tensor]],
) -> None:
    latents, targets = sample
    decoder = SemanticDecoder(3, columns, n_hidden=0, dropout=0.0)
    with torch.no_grad():
        decoder.heads["a"].weight.zero_()
        decoder.heads["a"].bias.copy_(torch.log(torch.tensor([0.25, 0.75])))
        decoder.heads["b"].weight.zero_()
        decoder.heads["b"].bias.copy_(torch.log(torch.tensor([0.2, 0.3, 0.5])))

    expected_joint = torch.tensor([0.05, 0.075, 0.125, 0.15, 0.225, 0.375])
    assert torch.allclose(decoder.log_joint(latents[:1]).exp()[0], expected_joint)

    logits = decoder(latents)
    expected_loss = torch.stack(
        [
            F.cross_entropy(logits[column.name], targets[column.name])
            for column in columns
        ]
    ).mean()
    assert torch.allclose(decoder.nll(latents, targets), expected_loss)


def test_autoregressive_teacher_forcing_matches_enumerated_joint(
    columns: list[ColumnSpec],
    sample: tuple[torch.Tensor, dict[str, torch.Tensor]],
) -> None:
    latents, targets = sample
    decoder = SemanticAutoRegDecoder(
        3, columns, hidden_dim=6, n_hidden=1, dropout=0.0, embed_dim=3
    )
    states = torch.stack([targets[column.name] for column in columns], dim=-1)
    flat = flat_state_index(states, columns)
    selected_joint = decoder.log_joint(latents).gather(1, flat[:, None]).squeeze(1)

    assert torch.allclose(decoder.log_prob(latents, targets), selected_joint, atol=1e-6)
    assert torch.allclose(decoder.nll(latents, targets), -selected_joint.mean())


def test_autoregressive_forward_requires_and_uses_targets(
    columns: list[ColumnSpec],
    sample: tuple[torch.Tensor, dict[str, torch.Tensor]],
) -> None:
    latents, targets = sample
    decoder = SemanticAutoRegDecoder(3, columns, hidden_dim=5, dropout=0.0, embed_dim=2)

    with pytest.raises(TypeError, match="requires targets"):
        decoder(latents)
    forwarded = decoder(latents, targets)
    conditional = decoder.conditional_logits(latents, targets)
    assert forwarded.keys() == conditional.keys()
    for column in columns:
        assert forwarded[column.name].shape == (len(latents), column.n_categories)
        assert torch.equal(forwarded[column.name], conditional[column.name])


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        (lambda: SemanticDecoder(0, [ColumnSpec("a", 2)]), "latent_dim"),
        (lambda: SemanticDecoder(2, []), "must not be empty"),
        (
            lambda: SemanticDecoder(2, [ColumnSpec("a", 2), ColumnSpec("a", 3)]),
            "duplicate",
        ),
        (lambda: SemanticDecoder(2, [ColumnSpec("a", 0)]), "cardinality"),
        (lambda: SemanticDecoder(2, [ColumnSpec("a", 2)], n_hidden=-1), "n_hidden"),
        (lambda: SemanticDecoder(2, [ColumnSpec("a", 2)], dropout=1.0), "dropout"),
        (
            lambda: SemanticAutoRegDecoder(2, [ColumnSpec("a", 2)], embed_dim=0),
            "embed_dim",
        ),
    ],
)
def test_constructor_validation(constructor, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        constructor()


def test_latent_and_target_validation(
    columns: list[ColumnSpec],
    sample: tuple[torch.Tensor, dict[str, torch.Tensor]],
) -> None:
    latents, targets = sample
    decoder = SemanticDecoder(3, columns)

    with pytest.raises(ValueError, match="shape"):
        decoder.log_joint(latents[:, :2])
    invalid_latents = latents.clone()
    invalid_latents[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        decoder.log_joint(invalid_latents)
    with pytest.raises(ValueError, match="target columns"):
        decoder.nll(latents, {"a": targets["a"]})
    bad_length = {**targets, "a": targets["a"][:-1]}
    with pytest.raises(ValueError, match="shape"):
        decoder.nll(latents, bad_length)
    bad_dtype = {**targets, "a": targets["a"].float()}
    with pytest.raises(ValueError, match="integer dtype"):
        decoder.nll(latents, bad_dtype)
    bad_range = {**targets, "b": torch.tensor([0, 1, 2, 3])}
    with pytest.raises(ValueError, match=r"\[0, 2\]"):
        decoder.nll(latents, bad_range)


def test_targets_from_dataframe_validates_integral_categories(
    columns: list[ColumnSpec],
) -> None:
    frame = pd.DataFrame({"a": ["0", "1"], "b": [2, 0], "ignored": [1, 1]})
    targets = targets_from_dataframe(frame, columns)
    assert torch.equal(targets["a"], torch.tensor([0, 1]))
    assert torch.equal(targets["b"], torch.tensor([2, 0]))

    with pytest.raises(ValueError, match="missing"):
        targets_from_dataframe(frame.drop(columns="a"), columns)
    with pytest.raises(ValueError, match="finite integers"):
        targets_from_dataframe(frame.assign(a=[0.5, 1]), columns)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        targets_from_dataframe(frame.assign(a=[0, 2]), columns)


def test_training_seed_controls_initialization_shuffle_and_dropout(
    columns: list[ColumnSpec],
    sample: tuple[torch.Tensor, dict[str, torch.Tensor]],
) -> None:
    latents, targets = sample
    torch.manual_seed(10)
    first = SemanticDecoder(3, columns, hidden_dim=8, n_hidden=2, dropout=0.4)
    torch.manual_seed(999)
    second = SemanticDecoder(3, columns, hidden_dim=8, n_hidden=2, dropout=0.4)

    first_history = train_semantic_decoder(
        first, latents, targets, epochs=3, batch_size=3, seed=17, verbose=False
    )
    second_history = train_semantic_decoder(
        second, latents, targets, epochs=3, batch_size=3, seed=17, verbose=False
    )

    assert first_history == pytest.approx(second_history)
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
    assert not first.training and not second.training


class _BatchSizeLossDecoder(SemanticDecoder):
    def nll(
        self,
        z: torch.Tensor,
        targets: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        del targets
        parameter_anchor = sum(parameter.sum() * 0 for parameter in self.parameters())
        return parameter_anchor + z.new_tensor(float(z.shape[0]))


def test_training_history_is_weighted_by_batch_size() -> None:
    columns = [ColumnSpec("a", 2)]
    decoder = _BatchSizeLossDecoder(1, columns, n_hidden=0)
    history = train_semantic_decoder(
        decoder,
        torch.zeros(5, 1),
        {"a": torch.zeros(5, dtype=torch.long)},
        epochs=1,
        batch_size=2,
        seed=0,
        verbose=False,
    )
    assert history == pytest.approx([1.8])


def test_training_rejects_empty_data() -> None:
    decoder = SemanticDecoder(2, [ColumnSpec("a", 2)])
    with pytest.raises(ValueError, match="must not be empty"):
        train_semantic_decoder(
            decoder,
            torch.empty(0, 2),
            {"a": torch.empty(0, dtype=torch.long)},
            verbose=False,
        )


@pytest.mark.parametrize(
    ("variant", "expected_type"),
    [("independent", SemanticDecoder), ("autoregressive", SemanticAutoRegDecoder)],
)
def test_versioned_save_load_round_trip_and_compatibility_checks(
    tmp_path,
    variant: str,
    expected_type: type,
    columns: list[ColumnSpec],
    sample: tuple[torch.Tensor, dict[str, torch.Tensor]],
) -> None:
    latents, _ = sample
    decoder = make_semantic_decoder(
        variant,
        latent_dim=3,
        columns=columns,
        hidden_dim=5,
        dropout=0.0,
        embed_dim=2,
    ).eval()
    expected_joint = decoder.log_joint(latents)
    path = tmp_path / f"{variant}.pt"
    metadata = _checkpoint_provenance()
    decoder.save(path, metadata=metadata)

    payload = torch.load(path, weights_only=True)
    assert payload["format_version"] == SEMANTIC_DECODER_FORMAT_VERSION
    loaded = load_semantic_decoder(
        path,
        device="cpu",
        expected_variant=variant,
        expected_columns=columns,
        expected_metadata={"latent_artifact_sha256": "abc"},
    )
    assert isinstance(loaded, expected_type)
    assert not loaded.training
    assert next(loaded.parameters()).device.type == "cpu"
    assert loaded.checkpoint_metadata["latent_artifact_sha256"] == "abc"
    assert loaded.checkpoint_metadata["encoder"] == metadata["encoder"]
    assert len(loaded.checkpoint_metadata["schema_signature"]) == 64
    assert torch.equal(loaded.log_joint(latents), expected_joint)

    with pytest.raises(ValueError, match="re-encode/retrain"):
        load_semantic_decoder(path, expected_variant="wrong")
    with pytest.raises(ValueError, match="re-encode/retrain"):
        load_semantic_decoder(path, expected_columns=[ColumnSpec("a", 2)])
    with pytest.raises(ValueError, match="re-encode/retrain"):
        load_semantic_decoder(
            path, expected_metadata={"latent_artifact_sha256": "different"}
        )


def test_save_and_load_reject_missing_provenance(
    tmp_path, columns: list[ColumnSpec]
) -> None:
    decoder = SemanticDecoder(2, columns)
    path = tmp_path / "decoder.pt"
    with pytest.raises(ValueError, match="latent and encoder provenance"):
        decoder.save(path)

    decoder.save(path, metadata=_checkpoint_provenance())
    payload = torch.load(path, weights_only=True)
    del payload["metadata"]["encoder"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="re-encode/retrain"):
        load_semantic_decoder(path)


def test_class_load_rejects_other_variant(
    tmp_path,
    columns: list[ColumnSpec],
) -> None:
    path = tmp_path / "autoregressive.pt"
    SemanticAutoRegDecoder(2, columns).save(path, metadata=_checkpoint_provenance())
    with pytest.raises(ValueError, match="variant mismatch"):
        SemanticDecoder.load(path)


def test_accuracy_and_calibration_restore_training_mode() -> None:
    column = ColumnSpec("a", 2)
    decoder = SemanticDecoder(1, [column], n_hidden=0, dropout=0.0)
    with torch.no_grad():
        decoder.heads["a"].weight.zero_()
        decoder.heads["a"].bias.copy_(torch.log(torch.tensor([0.3, 0.7])))
    decoder.train()
    latents = torch.zeros(4, 1)
    targets = {"a": torch.tensor([1, 1, 0, 0])}

    greedy_accuracy = accuracy(decoder, latents, targets)
    metrics = calibration_metrics(decoder, latents, targets, n_bins=5)

    assert decoder.training
    assert greedy_accuracy == {"a": 0.5}
    assert metrics["a"]["accuracy"] == pytest.approx(0.5)
    assert metrics["a"]["ece"] == pytest.approx(0.2)
    reliability = metrics["a"]["reliability"]
    assert len(reliability) == 1
    assert reliability[0]["lower"] == pytest.approx(0.6)
    assert reliability[0]["upper"] == pytest.approx(0.8)
    assert reliability[0]["count"] == 4
    assert reliability[0]["mean_confidence"] == pytest.approx(0.7)
    assert reliability[0]["accuracy"] == pytest.approx(0.5)


def test_factory_rejects_unknown_variant(columns: list[ColumnSpec]) -> None:
    with pytest.raises(ValueError, match="unknown semantic decoder variant"):
        make_semantic_decoder("flat", latent_dim=2, columns=columns)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_load_and_evaluation_colocate_cpu_inputs_on_cuda(tmp_path) -> None:
    columns = [ColumnSpec("a", 2)]
    path = tmp_path / "decoder.pt"
    SemanticDecoder(2, columns).save(path, metadata=_checkpoint_provenance())
    decoder = load_semantic_decoder(path, device="cuda")

    assert next(decoder.parameters()).is_cuda
    result = accuracy(
        decoder,
        torch.zeros(2, 2),
        {"a": torch.zeros(2, dtype=torch.long)},
    )
    assert set(result) == {"a"}
