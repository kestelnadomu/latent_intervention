"""Non-generative reference rows for the package decoder evaluation, on its 1,000 test units.

- ``retrieval``: per latent width and input vector (oracle Z' or predicted Z'-hat), the
  factual CV of the fit (pool) unit, i.e. a decoder training CV, whose canonical latent is
  closest in cosine; as the earlier pilot's nearest-fit retrieval.
- ``unedited``: the unit's own factual CV offered as its counterfactual.

Both are scored against the same oracle counterfactual CVs with the decoders' metrics: text
similarity, overall/changed/preserved fact accuracy, VADER and RoBERTa-large BERTScore.
Writes the portable ``docs/experiments/results/decoder_package_baselines_v1.json``
(aggregates only, no CV text). CPU only; the first run downloads RoBERTa-large:

    uv run --with rapidfuzz==3.14.1 --with sacrebleu==2.5.1 --with vaderSentiment==3.3.2 \
        --with bert-score==0.3.13 python -m exp.text_realizer_package_eval.baselines

With ``--floor``, only the noise floor instead: the 200 floor texts (the unit re-rendered
with another country of the same region, same state) scored against their factual CVs,
i.e. how close two correct CVs of one state get; ``decoder_package_floor_v1.json``.
"""

from __future__ import annotations

import argparse
import math
from statistics import mean

import torch
import torch.nn.functional as F
import yaml

from exp.text_realizer import facts
from exp.text_realizer_bertscore.__main__ import (
    BATCH_SIZE,
    CACHE,
    CHUNK_SIZE,
    MODEL,
    REVISION,
    check_lengths,
    snapshot,
)
from exp.text_realizer_fact_changes import CODEBOOK
from exp.text_realizer_fact_changes import evaluate as fact_changes
from exp.text_realizer_vader import load_analyzer
from src.artifact_io import sha256_file, write_json
from src.text_realizer.metrics import aggregate, normalize, text_metrics

from .common import (
    CONDITIONS,
    ROOT,
    checked_source,
    csv_records,
    digest,
    read_config,
    read_json,
    versions,
)
from .prepare import aligned_vectors, replay, select_units

SUMMARY = ROOT / "docs/experiments/results/decoder_package_v1.json"
OUTPUT = ROOT / "docs/experiments/results/decoder_package_baselines_v1.json"
FLOOR = ROOT / "docs/experiments/results/decoder_package_floor_v1.json"
FLOOR_TEXTS = ROOT / "data/text_talent_motivation/cv_floor.csv"
BERTSCORE = {  # the decoders' settings (exp.text_realizer_bertscore)
    "num_layers": 17,
    "idf": False,
    "rescale_with_baseline": False,
}


def bertscorer(device: str = "cpu", download: bool = True):
    from bert_score import BERTScorer

    return BERTScorer(
        model_type=str(snapshot(CACHE, download=download)),
        lang="en",
        batch_size=BATCH_SIZE,
        use_fast_tokenizer=False,
        device=device,
        **BERTSCORE,
    )


def bertscore(scorer, pairs: list[dict]) -> dict:
    """Mean BERTScore precision, recall and F1 of the pairs' predictions."""
    bert = [[], [], []]
    for start in range(0, len(pairs), CHUNK_SIZE):
        chunk = pairs[start : start + CHUNK_SIZE]
        values = scorer.score(
            [p["prediction"] for p in chunk],
            [p["reference"] for p in chunk],
            batch_size=BATCH_SIZE,
        )
        for column, value in zip(bert, values, strict=True):
            column.extend(value.tolist())
    return {name: mean(v) for name, v in zip(("precision", "recall", "f1"), bert)}


def text_scores(
    references: list[str],
    generated: list[str],
    pairs: list[dict],
    analyzer,
    compound: dict,
) -> dict:
    """Text similarity, VADER difference and overall fact accuracy against the references."""
    metrics = aggregate([text_metrics(a, b) for a, b in zip(references, generated)])
    row = {k: metrics[k] for k in ("character_similarity", "chrf", "rouge_l")}
    for p in pairs:
        for text in (p["reference"], p["prediction"]):
            if text not in compound:
                compound[text] = analyzer.polarity_scores(text)["compound"]
    deltas = [compound[p["prediction"]] - compound[p["reference"]] for p in pairs]
    row["mean_sentiment_delta"] = mean(deltas)
    row["mean_absolute_sentiment_delta"] = mean(map(abs, deltas))
    overall = facts.evaluate(references, generated)["fields"]
    row["facts"] = {
        field: {
            "accuracy": values["accuracy_on_scorable_references"],
            "reference_scorable": values["reference_scorable"],
        }
        for field, values in overall.items()
    }
    return row


def baseline_texts(config: dict) -> tuple[list[int], dict, dict, dict]:
    """Test IDs, their factual and oracle CVs, and the texts of every baseline variant."""
    inputs = {name: checked_source(item) for name, item in config["inputs"].items()}
    partition = csv_records(inputs["partition"])
    factual = csv_records(inputs["factual"])
    reference = csv_records(inputs["counterfactual"])
    ids = select_units(
        partition, csv_records(inputs["pairs"]), reference, config["unit_count"]
    )
    if digest(ids) != read_json(SUMMARY)["unit_ids_sha256"]:
        raise ValueError("test IDs differ from the decoder evaluation")
    interventions = csv_records(inputs["interventions"])
    do_values = [int(interventions[key]["do_value"]) for key in ids]
    payload = torch.load(inputs["latents"], map_location="cpu", weights_only=True)
    fit = [key for key, row in partition.items() if row["partition"] == "pool"]
    variants, replays = (
        {("unedited", None, None): [factual[key]["text"] for key in ids]},
        {},
    )
    for width, pipeline in config["pipelines"].items():
        dim = int(width)
        oracle, predicted, states = replay(
            config, pipeline, dim, payload, ids, do_values
        )
        replays[width] = states["reproduced_package_metrics"]
        candidates = aligned_vectors(payload, fit, dim, oracle=False)  # unit norm
        for condition, query in zip(CONDITIONS, (oracle, predicted), strict=True):
            nearest = (F.normalize(query, dim=1) @ candidates.T).argmax(dim=1)
            variants["retrieval", dim, condition] = [
                factual[fit[i]]["text"] for i in nearest.tolist()
            ]
    texts = {
        "original": {key: factual[key]["text"] for key in ids},
        "reference": [reference[key]["text"] for key in ids],
    }
    return ids, texts, variants, replays


def provenance(vader: dict, device: str = "cpu") -> dict:
    return {
        "bertscore": {
            "model": MODEL,
            "revision": REVISION,
            **BERTSCORE,
            "device": device,
        },
        "vader": vader["versions"],
        "versions": versions(
            ("torch", "transformers", "bert-score", "rapidfuzz", "sacrebleu")
        ),
    }


def floor() -> None:
    """The noise floor: each floor text scored against its unit's factual CV."""
    config = read_config()
    factual = csv_records(checked_source(config["inputs"]["factual"]))
    rewrites = csv_records(FLOOR_TEXTS)
    ids = sorted(rewrites)
    references = [factual[key]["text"] for key in ids]
    generated = [rewrites[key]["text"] for key in ids]
    pairs = [
        {"id": key, "reference": normalize(a), "prediction": normalize(b)}
        for key, a, b in zip(ids, references, generated, strict=True)
    ]
    analyzer, vader = load_analyzer()
    scorer = bertscorer()
    check_lengths(pairs, scorer._tokenizer)
    row = {"method": "same_state_rewrite", "units": len(ids)}
    row |= text_scores(references, generated, pairs, analyzer, {})
    row |= bertscore(scorer, pairs)
    if not all(math.isfinite(v) for v in row.values() if isinstance(v, float)):
        raise RuntimeError("non-finite noise-floor metric")
    write_json(
        FLOOR,
        {
            "format_version": 1,
            "units": len(ids),
            "unit_ids_sha256": digest(ids),
            "definition": "floor text (unit re-rendered with another country of the same "
            "region, same state) scored against the unit's factual CV",
            "inputs_sha256": {
                "factual": config["inputs"]["factual"]["sha256"],
                "floor": sha256_file(FLOOR_TEXTS),
            },
            **provenance(vader),
            "row": row,
        },
    )
    print(f"chrF {row['chrf']:.3f}, F1 {row['f1']:.4f}; wrote {FLOOR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--floor", action="store_true", help="score the noise floor only"
    )
    if parser.parse_args().floor:
        return floor()
    config = read_config()
    ids, texts, variants, replays = baseline_texts(config)
    references = texts["reference"]
    codebook = yaml.safe_load(CODEBOOK.read_text(encoding="utf-8"))
    analyzer, vader = load_analyzer()
    compound = {}
    scorer = bertscorer()
    rows, all_pairs = [], []
    for (method, dim, condition), generated in variants.items():
        pairs = [
            {
                "id": key,
                "decoder": method,
                "condition": condition,
                "reference": normalize(ref),
                "prediction": normalize(text),
            }
            for key, ref, text in zip(ids, references, generated, strict=True)
        ]
        check_lengths(pairs, scorer._tokenizer)
        row = {"method": method, "dim": dim, "condition": condition, "units": len(ids)}
        row |= text_scores(references, generated, pairs, analyzer, compound)
        for summary in fact_changes(pairs, texts["original"], codebook)[1]:
            row.setdefault(summary["status"], {})[summary["field"]] = {
                key: summary[key] for key in ("units", "correct", "accuracy")
            }
        rows.append(row)
        all_pairs.append(pairs)
    # BERTScore last: it takes hours on CPU, so the cheap metrics above fail fast.
    print("text, VADER and fact metrics done; BERTScore next", flush=True)
    for row, pairs in zip(rows, all_pairs, strict=True):
        row |= bertscore(scorer, pairs)
        label = f"{row['method']} {row['dim']} {row['condition']}"
        if not all(
            math.isfinite(value) for value in row.values() if isinstance(value, float)
        ):
            raise RuntimeError(f"non-finite baseline metric: {label}")
        print(f"{label}: chrF {row['chrf']:.3f}, F1 {row['f1']:.4f}", flush=True)
    write_json(
        OUTPUT,
        {
            "format_version": 1,
            "units": len(ids),
            "unit_ids_sha256": digest(ids),
            "retrieval": "factual CV of the fit (pool) unit with the highest cosine "
            "between its canonical latent and the oracle or predicted vector",
            "unedited": "the unit's own factual CV",
            "reproduced_package_metrics": replays,
            "inputs_sha256": {k: v["sha256"] for k, v in config["inputs"].items()},
            "pipelines": config["pipelines"],
            **provenance(vader),
            "rows": rows,
        },
    )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
