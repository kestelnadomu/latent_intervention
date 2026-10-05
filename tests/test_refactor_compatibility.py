"""Pinned metadata fingerprints captured before the behavior-preserving extraction.

These tests need no downloaded models or generated corpus. Deliberate future changes
to a latent-space protocol should update its experiment identity, not silently make
old artifacts look compatible.
"""

import hashlib
import json

import pytest

from src import config
from src import encoder as encoder_protocols
from src.semantic_decoder import model as semantic_decoder
from src.semantic_decoder import training as decoder_training
from src.pair_encoding import _active_encoder_info

METADATA_SHA256 = {
    "embeddinggemma_128": "a7b50f804916d275cfde5793655239b6da02dc76a34aee8d5b92bdedb40d4153",
    "embeddinggemma_256": "d121b034abc75a57f9393356c83120d2e34a339b915deabb8f3d1a9d8dbd734c",
    "embeddinggemma_512": "87415c622b53eb377efd90f66ad87c2d3041eb0ed1305b9a75d5334002fe4516",
    "embeddinggemma_768": "21c9ceb6da78d23597a5e1667c47cb20ac7eb19469eda8773f6b22a9d735536b",
    "langvae": "a72d22755109da564459446414bc026003f73654f7951c337d26235594cedcae",
    "nomic_128": "298a297744a531b5c1e5a93adb5837de0f48d94f797ec2281567fb3a5e84b53d",
    "nomic_256": "4934a214ab3d9e4790954a66dee54be8dc5e8e290bd935a0b28a41b249454dbf",
    "nomic_512": "cf439b7522b66726f002436bdbd9ff5223c258c99076c7aaecfbb5dc1eaeca2e",
    "nomic_64": "cb7adf516c9edf4f7ba190b00098b1b9b5185b0fc8d9a885abd756c894c93c5a",
    "nomic_768": "db73c3a97af42d5de136463fa3de85cba9e27c71aa74c5e6995d1ef3132a6546",
    "qwen3_1024": "a0577d9ea11418709ba6ed31470cca0b7d6c269768bc4d1b1cdc08f5825f4df4",
    "qwen3_128": "e714743b8119c8326111b57df634e59feecf1d1af674926277ab84125b45ceb8",
    "qwen3_256": "dbe75097c7aca27841c2f25a64d6f8f3e7fac8d464afff0b63280e8cf6aabc35",
    "qwen3_32": "59b2e5f55f44a73ffcf969b4ce9dfcb7f115842302ee2cce87dacb2cc8e1056c",
    "qwen3_512": "947fcb0a35d094a7f9fe54ddbe92331cbddb53513b3ad0fb490e32259e3116d2",
    "qwen3_64": "27cdecbbad1cb47cf1412b2314c75cc255044385fb461f93348f7273a7ff9248",
    "qwen3_768": "1ed0dff95b0a94f398ad614f0f7d709f6d99484ab903a418f4f00d731f35df80",
}


@pytest.mark.parametrize("tag,expected", METADATA_SHA256.items())
def test_encoder_identity_matches_pre_refactor_fingerprint(tag, expected):
    if tag == "langvae":
        cfg = config.load_config(encoder_variant=tag)
    else:
        variant, dimension = tag.rsplit("_", 1)
        cfg = config.load_config(
            encoder_variant=variant, **{f"{variant}_dim": int(dimension)}
        )
    info = _active_encoder_info(cfg["encoder"])
    observed = hashlib.sha256(
        json.dumps(info, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed == expected


def test_historical_imports_remain_available():
    assert (
        semantic_decoder.train_semantic_decoder
        is decoder_training.train_semantic_decoder
    )
    for name in (
        "NOMIC_DIMENSIONS",
        "QWEN3_DIMENSIONS",
        "EMBEDDINGGEMMA_DIMENSIONS",
        "encoder_dimension",
        "qwen3_protocol",
        "embeddinggemma_protocol",
    ):
        assert getattr(config, name) is getattr(encoder_protocols, name)


def test_progress_reporting_keeps_original_intervals_and_text(monkeypatch, capsys):
    import src.encoder.progress as progress

    monkeypatch.setattr(progress, "perf_counter", lambda: 20.0)
    progress.log_encoding_progress("Nomic", 0, 8, 200, 10.0, enabled=True)
    assert capsys.readouterr().out == "Nomic 8/200 texts; elapsed 10.0s; ETA 240.0s\n"
    progress.log_encoding_progress("Nomic", 8, 8, 200, 10.0, enabled=True)
    assert capsys.readouterr().out == ""
    progress.log_encoding_progress("Nomic", 160, 8, 200, 10.0, enabled=True)
    assert "168/200" in capsys.readouterr().out
    progress.log_encoding_progress("Nomic", 192, 8, 200, 10.0, enabled=True)
    assert "200/200" in capsys.readouterr().out
    progress.log_encoding_progress("Nomic", 0, 8, 200, 10.0, enabled=False)
    assert capsys.readouterr().out == ""
