"""Deterministic cycling streams with exact optimizer-boundary resume."""

from __future__ import annotations

import torch


class CyclingStream:
    def __init__(self, size: int, seed: int):
        if size < 1:
            raise ValueError("cannot sample an empty training stream")
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator).tolist()
        self.offset = 0

    def take(self, count: int) -> list[int]:
        if count < 1:
            raise ValueError("sample count must be positive")
        result = []
        while len(result) < count:
            if self.offset == self.size:
                self.order = torch.randperm(
                    self.size, generator=self.generator
                ).tolist()
                self.offset = 0
            amount = min(count - len(result), self.size - self.offset)
            result.extend(self.order[self.offset : self.offset + amount])
            self.offset += amount
        return result

    def state_dict(self) -> dict:
        return {
            "size": self.size,
            "order": list(self.order),
            "offset": self.offset,
            "generator": self.generator.get_state().clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        if (
            state["size"] != self.size
            or sorted(state["order"]) != list(range(self.size))
            or not 0 <= state["offset"] <= self.size
        ):
            raise ValueError("sampler state does not match prepared training data")
        self.order = list(state["order"])
        self.offset = int(state["offset"])
        self.generator.set_state(state["generator"])


class ArmSampler:
    def __init__(self, factual_count: int, changed_count: int, arm: str, seed: int):
        if arm not in {"factual", "oracle"}:
            raise ValueError("unknown comparison arm")
        self.arm = arm
        self.factual_count = factual_count
        self.factual = CyclingStream(factual_count, seed)
        self.changed = (
            CyclingStream(changed_count, seed + 10_000) if arm == "oracle" else None
        )

    def take_batch(self) -> list[int]:
        if self.changed is None:
            return self.factual.take(32)
        factual = self.factual.take(16)
        changed = [self.factual_count + i for i in self.changed.take(16)]
        return [index for pair in zip(factual, changed, strict=True) for index in pair]

    def state_dict(self) -> dict:
        return {
            "arm": self.arm,
            "factual_count": self.factual_count,
            "factual": self.factual.state_dict(),
            "changed": None if self.changed is None else self.changed.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state["arm"] != self.arm or state["factual_count"] != self.factual_count:
            raise ValueError("sampler arm changed")
        self.factual.load_state_dict(state["factual"])
        if (state["changed"] is None) != (self.changed is None):
            raise ValueError("sampler counterfactual stream changed")
        if self.changed is not None:
            self.changed.load_state_dict(state["changed"])
