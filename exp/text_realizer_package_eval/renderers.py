"""Decode the edits of the five other renderers with the package evaluation's decoder.

The package evaluation (package-v1) decoded the state flow's edits only. This decodes the
editor, OT editor, OT flow, state effect and group OT on the same 1,000 test units, in the
same deployment setting (``configs/text_realizer_renderers_eval.yaml``), and phi(x) itself:

1. ``prepare`` (CPU, package-eval environment, next to package-v1 and the latent_lab cache)
   replays every renderer's edits, each checked against the run report, into a new bundle
   with package-v1's decoder export and references, one vector set per renderer and the
   unedited phi(x) as the set ``factual``;
2. ``python -m exp.text_realizer_package_eval generate`` on that bundle, one shard per GPU;
3. ``score`` (one GPU) scores every renderer's texts against x', and phi^-1(phi(x)) against
   x, with the paper tables' metrics: text similarity, VADER, overall fact accuracy and
   RoBERTa-large BERTScore.

Steps 2 and 3 on four GPUs: ``scripts/run_text_realizer_renderers.sh``. Outputs: the bundle
``models/.../renderers-v1``, the predictions and ``scores/comparisons.jsonl`` (every text)
in ``reports/.../renderers-v1`` and the portable aggregates
``docs/experiments/results/decoder_package_renderers_v1.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.artifact_io import (
    atomic_output,
    copy_atomic,
    save_torch,
    sha256_file,
    write_json,
)

from .common import (
    ROOT,
    checked_file,
    checked_source,
    csv_records,
    digest,
    file_record,
    immutable_json,
    jsonl_records,
    load_manifest,
    lock,
    read_config,
    source_hashes,
)
from .prepare import aligned_vectors, replay

CONFIG = ROOT / "configs/text_realizer_renderers_eval.yaml"
BUNDLE = ROOT / "models/talent_motivation/text-realizer-evaluation/renderers-v1"
OUTPUT = ROOT / "reports/talent_motivation/text-realizer-evaluation/renderers-v1"
RESULTS = ROOT / "docs/experiments/results/decoder_package_renderers_v1.json"


def prepare(spec: dict, bundle: Path, *, allow_oracle: bool = False) -> dict:
    """The renderers' edits of the test units, with package-v1's decoder and references."""
    if not allow_oracle:
        raise PermissionError("official test export requires --allow-oracle-evaluation")
    config = read_config(ROOT / spec["base"])
    source = ROOT / spec["source_bundle"]
    base = load_manifest(source)
    key, dim = spec["decoder"], spec["width"]
    model = base["models"][key]
    if model["latent_dim"] != dim:
        raise ValueError("decoder and renderer widths differ")
    provenance = {
        "spec": spec,
        "source_bundle_sha256": sha256_file(source / "manifest.json"),
        "source_sha256": source_hashes(),
    }
    with lock(bundle / ".prepare.lock"):
        if (bundle / "manifest.json").exists():
            saved = load_manifest(bundle)
            if saved["provenance"] != provenance:
                raise ValueError(
                    "bundle recipe/source changed; choose a new output directory"
                )
            for item in saved["files"].values():
                checked_file(bundle, item)
            return saved
        inputs = {name: checked_source(item) for name, item in config["inputs"].items()}
        interventions = csv_records(inputs["interventions"])
        # package-v1's IDs, in the partition order that reproduces the operator's draws
        ids = base["ids"]
        do_values = [int(interventions[unit]["do_value"]) for unit in ids]
        payload = torch.load(inputs["latents"], map_location="cpu", weights_only=True)
        pipeline = {**config["pipelines"][str(dim)], **spec["renderers"]}
        vectors, replays = {"ids": ids, "encoder": model["encoder"]}, {}
        for kind in spec["renderers"]:
            _, vectors[kind], states = replay(
                config, pipeline, dim, payload, ids, do_values, kind
            )
            replays[kind] = states["reproduced_package_metrics"]
            print(
                f"{kind}: replay reproduces the run report {replays[kind]}", flush=True
            )
        # the renderers' input phi(x), decoded as a reconstruction of x
        vectors["factual"] = aligned_vectors(payload, ids, dim, oracle=False)
        files = {}
        for name in ("references", key):
            destination = bundle / base["files"][name]["file"]
            copy_atomic(checked_file(source, base["files"][name]), destination)
            files[name] = file_record(bundle, destination)
        path = bundle / f"vectors-{dim}.pt"
        save_torch(path, vectors)
        files[f"vectors-{dim}"] = file_record(bundle, path)
        manifest = {
            "format_version": 1,
            "complete": True,
            "ids": ids,
            "conditions": [*spec["renderers"], "factual"],
            "models": {key: {**model, **files[key]}},
            "files": files,
            "provenance": provenance,
            "replays": replays,
            "scope": "official test: decoded edits of the renderers other than the state flow; "
            "S1, rho trained and tested with psi(z), seed 0; and the unedited phi(x)",
        }
        immutable_json(bundle / "manifest.json", manifest)
        return manifest


def score(bundle: Path, predictions: Path, output: Path, device: str) -> dict:
    """Every renderer's texts against x', phi^-1(phi(x)) against x, with the decoder tables' metrics."""
    from src.text_realizer.metrics import normalize

    from exp.text_realizer_bertscore.__main__ import check_lengths
    from exp.text_realizer_vader import load_analyzer

    from .baselines import bertscore, bertscorer, provenance, text_scores
    from .score import collect_predictions

    manifest = load_manifest(bundle)
    ((key, model),) = manifest["models"].items()
    ids = manifest["ids"]
    reference_rows = jsonl_records(
        checked_file(bundle, manifest["files"]["references"])
    )
    if [row["id"] for row in reference_rows] != ids:
        raise ValueError("reference IDs differ from the bundle")
    bundle_hash = sha256_file(bundle / "manifest.json")
    analyzer, vader = load_analyzer()
    scorer = bertscorer(
        device, download=False
    )  # the cached model of the BERTScore runs
    rows, outputs, compound = [], {}, {}
    for kind in manifest["conditions"]:
        generated = collect_predictions(
            predictions,
            bundle_hash,
            key,
            kind,
            ids,
            model["sha256"],
            model["generation"],
        )
        texts = [generated[i]["text"] for i in ids]
        against = "x" if kind == "factual" else "x'"
        field = "original" if kind == "factual" else "reference"
        references = [row[field] for row in reference_rows]
        pairs = [
            {"id": i, "reference": normalize(a), "prediction": normalize(b)}
            for i, a, b in zip(ids, references, texts, strict=True)
        ]
        check_lengths(pairs, scorer._tokenizer)
        row = {"method": kind, "decoder": key, "against": against, "units": len(ids)}
        row |= text_scores(references, texts, pairs, analyzer, compound)
        row |= bertscore(scorer, pairs)
        rows.append(row)
        outputs[f"{key}/{kind}"] = texts
        print(
            f"{kind} against {against}: chrF {row['chrf']:.3f}, BERTScore F1 {row['f1']:.4f}",
            flush=True,
        )
    output.mkdir(parents=True, exist_ok=True)
    path = output / "comparisons.jsonl"
    with atomic_output(path) as temporary:
        with temporary.open("w", encoding="utf-8") as stream:
            for i, row in enumerate(reference_rows):
                generated = {name: texts[i] for name, texts in outputs.items()}
                stream.write(
                    json.dumps({**row, "generated": generated}, ensure_ascii=False)
                    + "\n"
                )
    result = {
        "format_version": 1,
        "units": len(ids),
        "unit_ids_sha256": digest(ids),
        "decoder": key,
        "bundle_sha256": bundle_hash,
        "comparisons_sha256": sha256_file(path),
        "replays": manifest["replays"],
        **provenance(vader, device),
        "rows": rows,
    }
    write_json(RESULTS, result)
    print(f"wrote {path} and {RESULTS}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    step = commands.add_parser(
        "prepare", help="replay the renderers' edits into the bundle"
    )
    step.add_argument("--config", type=Path, default=CONFIG)
    step.add_argument("--bundle", type=Path, default=BUNDLE)
    step.add_argument("--allow-oracle-evaluation", action="store_true")
    step = commands.add_parser(
        "score", help="score all decoded texts after generation"
    )
    step.add_argument("--bundle", type=Path, default=BUNDLE)
    step.add_argument("--predictions", type=Path, default=OUTPUT / "predictions")
    step.add_argument("--output", type=Path, default=OUTPUT / "scores")
    step.add_argument(
        "--device", default="cpu", help="cpu or a visible device such as cuda:0"
    )
    args = parser.parse_args()
    if args.command == "prepare":
        spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        manifest = prepare(spec, args.bundle, allow_oracle=args.allow_oracle_evaluation)
        print(
            f"Prepared {len(manifest['ids'])} units x {len(manifest['conditions'])} vector sets at {args.bundle}"
        )
    else:
        score(args.bundle, args.predictions, args.output, args.device)


if __name__ == "__main__":
    main()
