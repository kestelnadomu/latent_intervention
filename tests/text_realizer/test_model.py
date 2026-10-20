import copy

import pytest
import torch

from src.text_realizer.generation import generate
from src.text_realizer.training import evaluate_nll, optimizer_for, update


def test_masks_eos_and_native_embedding_scaling(tiny_factory):
    model = tiny_factory()
    z = torch.randn(2, 768)
    targets = [[4, 5, 2], [8, 2]]
    inputs = model.teacher_inputs(z, targets)
    assert inputs["labels"].tolist() == [
        [-100, -100, -100, 4, 5, 2],
        [-100, -100, -100, 8, 2, -100],
    ]
    assert inputs["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]]
    expected = model.language_model.get_input_embeddings()(
        torch.tensor([[1, 4, 5, 2], [1, 8, 2, 0]])
    )
    torch.testing.assert_close(inputs["inputs_embeds"][:, 2:], expected)
    losses, counts = model.loss_sums(z, targets)
    assert counts.tolist() == [3, 2]
    assert torch.isfinite(losses).all()
    with pytest.raises(ValueError, match="context"):
        model.loss_sums(z[:1], [[4] * 128 + [2]])
    with pytest.raises(ValueError, match="finite"):
        model.prefix(torch.full((1, 768), float("nan")))


def test_causal_teacher_and_cached_generation_parity_beyond_sliding_window(
    tiny_factory,
):
    model = tiny_factory().eval()
    z = torch.randn(2, 768)
    cached = generate(model, z, max_new_tokens=24, batch_size=2)
    repeated = generate(model, z, max_new_tokens=24, batch_size=1)
    assert cached == repeated == generate(model, z, max_new_tokens=24, batch_size=2)
    for i, row in enumerate(cached):
        previous = [model.bos_id]
        cached_output = None
        for token in row["token_ids"]:
            prefix = model.prefix(z[i : i + 1])
            embeds = model.language_model.get_input_embeddings()(
                torch.tensor([previous])
            )
            inputs = torch.cat((prefix, embeds), dim=1)
            logits = model.language_model(inputs_embeds=inputs, use_cache=False).logits[
                :, -1
            ]
            if cached_output is None:
                cached_output = model.language_model(
                    inputs_embeds=inputs, use_cache=True
                )
            else:
                position = torch.tensor([inputs.shape[1] - 1])
                cached_output = model.language_model(
                    input_ids=torch.tensor([[previous[-1]]]),
                    attention_mask=torch.ones(1, inputs.shape[1], dtype=torch.long),
                    past_key_values=cached_output.past_key_values,
                    cache_position=position,
                    position_ids=position[None, :],
                    use_cache=True,
                )
            torch.testing.assert_close(
                cached_output.logits[:, -1], logits, atol=1e-6, rtol=1e-5
            )
            assert token == int(logits.argmax(-1))
            previous.append(token)
    inputs = model.teacher_inputs(z, [[4, 5, 2], [7, 8, 2]])
    inputs.pop("labels")
    original = model.language_model(**inputs, use_cache=False).logits.detach()
    inputs["inputs_embeds"][:, -1] += 100
    changed = model.language_model(**inputs, use_cache=False).logits
    torch.testing.assert_close(original[:, :-1], changed[:, :-1])


@pytest.mark.parametrize("lora", [False, True])
def test_only_projector_and_lora_update_and_use_z(
    tiny_factory, numerical_settings, lora
):
    model = tiny_factory(lora=lora)
    frozen = {
        name: value.clone()
        for name, value in model.named_parameters()
        if not value.requires_grad
    }
    z = torch.nn.functional.normalize(torch.randn(2, 768), dim=1)
    assert not torch.allclose(model.prefix(z)[:1], model.prefix(z)[1:])
    targets = [[4, 5, 2], [7, 8, 2]]
    optimizer = optimizer_for(model, numerical_settings)
    before = evaluate_nll(model, z, targets)["nll"]
    for _ in range(25):
        update(model, optimizer, z, targets, [0, 1], microbatch=1, clip=1)
    assert evaluate_nll(model, z, targets)["nll"] < before
    for name, value in model.named_parameters():
        if name in frozen:
            assert torch.equal(value, frozen[name])
        else:
            assert name.startswith("projector.") or "lora_" in name
    assert model.projector[0].weight.grad.abs().sum() > 0
    if lora:
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for name, p in model.named_parameters()
            if "lora_" in name
        )


def test_gradient_accumulation_weights_tokens_not_examples(
    tiny_factory, numerical_settings
):
    model = tiny_factory()
    other = copy.deepcopy(model)
    z = torch.randn(2, 768)
    targets = [[4, 2], [5, 6, 7, 8, 2]]
    # SGD exposes gradient weighting directly; Adam's near-zero denominator can
    # amplify harmless GEMM rounding differences in almost-zero coordinates.
    left = update(
        model,
        torch.optim.SGD(model.projector.parameters(), lr=0.01),
        z,
        targets,
        [0, 1],
        microbatch=1,
        clip=1,
    )
    right = update(
        other,
        torch.optim.SGD(other.projector.parameters(), lr=0.01),
        z,
        targets,
        [0, 1],
        microbatch=2,
        clip=1,
    )
    assert left["nll"] == pytest.approx(right["nll"], abs=1e-6)
    for name, value in model.trainable_state().items():
        torch.testing.assert_close(
            value, other.trainable_state()[name], atol=2e-6, rtol=2e-5
        )


def test_identical_vectors_give_identical_text_and_off_manifold_is_not_normalized(
    tiny_factory,
):
    model = tiny_factory().eval()
    z = torch.ones(1, 768)
    rows = model.decode(z.repeat(2, 1))
    assert rows[0] == rows[1]
    assert not torch.allclose(model.prefix(z), model.prefix(z * 2))
    assert model.decode(torch.empty(0, 768)) == []


def test_finished_rows_are_masked_without_stopping_other_rows(
    tiny_factory, monkeypatch
):
    from types import SimpleNamespace

    model = tiny_factory()
    masks = []

    def forward(**kwargs):
        masks.append(kwargs["attention_mask"].clone())
        logits = torch.full((2, 1, 32), -100.0)
        tokens = [2, 7] if len(masks) == 1 else [9, 2]
        logits[torch.arange(2), 0, tokens] = 100.0
        return SimpleNamespace(logits=logits, past_key_values=None)

    monkeypatch.setattr(model.language_model, "forward", forward)
    rows = generate(model, torch.zeros(2, 768), batch_size=2)
    assert [row["token_ids"] for row in rows] == [[2], [7, 2]]
    assert all(row["termination"] == "eos" for row in rows)
    assert masks[1][:, -1].tolist() == [0, 1]


def test_tiny_two_example_greedy_overfit(tiny_factory):
    model = tiny_factory(lora=True)
    settings = {"projector_lr": 0.01, "lora_lr": 0.01, "weight_decay": 0}
    optimizer = optimizer_for(model, settings)
    z = torch.nn.functional.normalize(torch.randn(2, 768), dim=1)
    targets = [[4, 2], [7, 2]]
    for step in range(80):
        update(model, optimizer, z, targets, [0, 1], microbatch=2, clip=1)
        if (step + 1) % 10 == 0:
            observed = [
                row["token_ids"] for row in generate(model, z, max_new_tokens=4)
            ]
            if observed == targets:
                break
    assert observed == targets
