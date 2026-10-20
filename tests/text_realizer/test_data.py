import json
from pathlib import Path

import pytest
import torch

from exp.text_realizer.data import (
    load_bundle,
    load_groups,
    make_splits,
    prepare,
    verified_inputs,
)
from exp.text_realizer.protocol import oracle_gate
from exp.text_realizer.settings import digest


def test_preparation_joins_by_id_never_accesses_oracle_and_seals_audit(
    factual_config, monkeypatch
):
    real_load = torch.load
    accessed = []

    class FactualOnly(dict):
        def __getitem__(self, key):
            accessed.append(key)
            assert key in {"ids", "z"}, "oracle key was touched"
            return super().__getitem__(key)

    def guarded(path, *args, **kwargs):
        payload = real_load(path, *args, **kwargs)
        return (
            FactualOnly(payload)
            if Path(path) == Path(factual_config["inputs"]["latents"])
            else payload
        )

    monkeypatch.setattr(torch, "load", guarded)
    root = prepare(factual_config)
    assert accessed == ["ids", "z"]
    bundle = load_bundle(root)
    assert set(bundle) == {"format_version", "split_sha256", "fit", "tune"}
    for part in (bundle["fit"], bundle["tune"]):
        assert part["texts"] == [f"CV number {key}" for key in part["ids"]]
        assert not set(part["ids"]) & {10, 11}
    manifest = json.loads((root / "manifest.json").read_text())
    assert digest(manifest["split_ids"]) == bundle["split_sha256"]
    assert len(manifest["split_ids"]["audit"]) == 2
    with pytest.raises(PermissionError, match="frozen"):
        load_bundle(root, audit=True)
    with pytest.raises(PermissionError, match="frozen"):
        load_groups(root, audit=True)
    assert prepare(factual_config) == root
    splits = make_splits(
        list(range(12)),
        {i: {"split": "test" if i >= 10 else "train"} for i in range(12)},
        factual_config["split"],
        json.loads(Path(factual_config["inputs"]["split_plan"]).read_text()),
    )
    assert splits == manifest["split_ids"]


def test_missing_split_plan_cannot_silently_create_new_splits(factual_config):
    Path(factual_config["inputs"]["split_plan"]).rename(
        Path(factual_config["inputs"]["split_plan"]).with_suffix(".sealed")
    )
    with pytest.raises(FileNotFoundError):
        prepare(factual_config)


def test_input_and_bundle_hash_mismatches_are_rejected(factual_config):
    root = prepare(factual_config)
    with (root / "development.pt").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="hash"):
        load_bundle(root)
    with Path(factual_config["inputs"]["factual"]).open("a") as stream:
        stream.write("changed")
    with pytest.raises(ValueError, match="hash"):
        verified_inputs(factual_config)


def test_invalid_provenance_and_split_rejected(factual_config):
    info_path = Path(factual_config["inputs"]["info"])
    info = json.loads(info_path.read_text())
    info["model_revision"] = "a" * 40
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="provenance"):
        prepare(factual_config)
    pairs = {i: {"split": "test" if i >= 10 else "train"} for i in range(12)}
    plan = {
        "split_ids": {
            "fit": list(range(6)),
            "validation": [5, 7, 8, 9],
            "official_test": [10, 11],
        }
    }
    with pytest.raises(ValueError, match="splits"):
        make_splits(list(pairs), pairs, factual_config["split"], plan)


def test_oracle_opt_in_is_checked_before_reading_any_files(tmp_path):
    with pytest.raises(PermissionError, match="allow-oracle"):
        oracle_gate(tmp_path / "does-not-exist", allowed=False)
    with pytest.raises(FileNotFoundError):
        oracle_gate(tmp_path, allowed=True)
