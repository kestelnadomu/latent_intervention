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

With ``--decoded-unedited``, only the decoded do-nothing instead: phi^-1(phi(x)) from the
published renderers-v1 texts, scored against x', the edits' zero point under the same
decoder; ``decoder_package_decoded_unedited_v1.json``.

With ``--joint``, only the joint fact accuracy over O (region), D and U of every row of the
decoder table (decoders from the package-v1 texts, retrieval, unedited x, noise floor) and
of the renderer table (the renderers-v1 texts), checked against the published per-fact
accuracies; for the renderer table also the decoded facts against the commanded state (the
true s' or the operator's draw) and, where the command changes a fact, how often a CV
states the new value or still the value of x; ``decoder_package_joint_v1.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
DECODED_UNEDITED = (
    ROOT / "docs/experiments/results/decoder_package_decoded_unedited_v1.json"
)
RENDERERS = ROOT / "docs/experiments/results/decoder_package_renderers_v1.json"
COMPARISONS = (
    ROOT
    / "reports/talent_motivation/text-realizer-evaluation/renderers-v1/scores/comparisons.jsonl"
)
JOINT = ROOT / "docs/experiments/results/decoder_package_joint_v1.json"
PACKAGE_COMPARISONS = (
    ROOT
    / "reports/talent_motivation/text-realizer-evaluation/package-v1/scores/comparisons.jsonl"
)
# O, D and U as the CVs state them: fact field -> (state column, the stated value per level)
JOINT_FIELDS = {
    "region": ("X", {level: level for level in range(4)}),
    "degree": ("D", {0: "bachelor", 1: "master", 2: "doctorate"}),
    "university_tier": ("U", {0: "unranked", 1: "top500", 2: "top100"}),
}
RENDERERS_CONFIG = ROOT / "configs/text_realizer_renderers_eval.yaml"
FACTUAL_STATES = ROOT / "data/sim_talent_motivation/sim_data_factual.csv"
KINDS = ("editor", "ot_editor", "flow", "ot_flow", "state_effect", "group_ot")
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
    """Test IDs, their factual and oracle CVs, the texts of every baseline variant, and the
    replayed states (source, target, reproduced metrics) per latent width."""
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
        replays[width] = states
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


def decoded_unedited() -> None:
    """phi^-1(phi(x)) against x': the decoded unedited CV offered as its counterfactual."""
    renderers = read_json(RENDERERS)
    raw = COMPARISONS.read_bytes().replace(
        b"\r\n", b"\n"
    )  # a CRLF checkout verifies too
    if hashlib.sha256(raw).hexdigest() != renderers["comparisons_sha256"]:
        raise ValueError("comparisons differ from the published renderer results")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    ids = [row["id"] for row in rows]
    if digest(ids) != renderers["unit_ids_sha256"]:
        raise ValueError("comparisons cover other test units")
    key = renderers["decoder"]
    references = [row["reference"] for row in rows]
    generated = [row["generated"][f"{key}/factual"] for row in rows]
    pairs = [
        {
            "id": unit,
            "decoder": key,
            "condition": "decoded_unedited",
            "reference": normalize(ref),
            "prediction": normalize(text),
        }
        for unit, ref, text in zip(ids, references, generated, strict=True)
    ]
    # the renderer rows' BERTScore cohort (no truncation: one edit exceeds 512 tokens)
    excluded = set(renderers["bertscore_cohort"]["excluded_ids"])
    bert_pairs = [pair for pair in pairs if pair["id"] not in excluded]
    analyzer, vader = load_analyzer()
    scorer = bertscorer()
    check_lengths(bert_pairs, scorer._tokenizer)
    row = {
        "method": "decoded_unedited",
        "decoder": key,
        "against": "x'",
        "units": len(ids),
    }
    row |= text_scores(references, generated, pairs, analyzer, {})
    originals = {row["id"]: row["original"] for row in rows}
    codebook = yaml.safe_load(CODEBOOK.read_text(encoding="utf-8"))
    for summary in fact_changes(pairs, originals, codebook)[1]:
        row.setdefault(summary["status"], {})[summary["field"]] = {
            name: summary[name] for name in ("units", "correct", "accuracy")
        }
    # BERTScore last: about 50 minutes on CPU
    print(f"text, VADER and fact metrics done (chrF {row['chrf']:.3f}); BERTScore next")
    row |= bertscore(scorer, bert_pairs)
    row["bertscore_units"] = len(bert_pairs)
    write_json(
        DECODED_UNEDITED,
        {
            "format_version": 1,
            "units": len(ids),
            "unit_ids_sha256": digest(ids),
            "definition": "phi^-1(phi(x)) of renderers-v1 scored against x'",
            "comparisons_sha256": renderers["comparisons_sha256"],
            "bertscore_excluded_ids": sorted(excluded),
            **provenance(vader),
            "row": row,
        },
    )
    print(f"F1 {row['f1']:.4f}; wrote {DECODED_UNEDITED}")


def joint_accuracy(references: list[str], generated: list[str], codebook: dict) -> dict:
    """Per-fact accuracy of O, D and U, and the share of units with all three right, over
    the units whose reference states all three."""
    expected = [facts.extract(text, codebook) for text in references]
    observed = [facts.extract(text, codebook) for text in generated]
    row = {"facts": {}}
    for field in JOINT_FIELDS:
        valid = [i for i, e in enumerate(expected) if e[field] is not None]
        row["facts"][field] = mean(
            observed[i][field] == expected[i][field] for i in valid
        )
    hits = [
        all(o[f] == e[f] for f in JOINT_FIELDS)
        for o, e in zip(observed, expected)
        if all(e[f] is not None for f in JOINT_FIELDS)
    ]
    row["joint"] = {"accuracy": mean(hits), "reference_scorable": len(hits)}
    return row


def command_scores(
    generated: list[str], commands: list[dict], sources: list[dict], codebook: dict
) -> dict:
    """Decoded O, D and U against the commanded states (each fact and all three), and,
    where the command changes a fact, the shares of CVs that state the commanded value
    (realized) or still the source value (kept); the rest states another or no value.
    Joint: all three facts at once, where the command changes the state."""
    observed = [facts.extract(text, codebook) for text in generated]
    row, right, values = {"facts": {}, "changed": {}}, {}, {}
    for field, (column, stated) in JOINT_FIELDS.items():
        target = [stated[int(state[column])] for state in commands]
        before = [stated[int(state[column])] for state in sources]
        values[field] = [(o[field], t, b) for o, t, b in zip(observed, target, before)]
        right[field] = [o[field] == t for o, t in zip(observed, target, strict=True)]
        row["facts"][field] = mean(right[field])
    values["joint"] = [tuple(zip(*unit)) for unit in zip(*values.values())]
    for field, units in values.items():
        changed = [(o, t, b) for o, t, b in units if t != b]
        row["changed"][field] = {
            "units": len(changed),
            "realized": mean(o == t for o, t, _ in changed),
            "kept": mean(o == b for o, _, b in changed),
        }
    row["joint"] = mean(map(all, zip(*right.values())))
    return row


def joint() -> None:
    """Joint fact accuracy of every row of the decoder and the renderer fact tables."""
    config = read_config()
    summary = read_json(SUMMARY)
    raw = PACKAGE_COMPARISONS.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(raw).hexdigest() != summary["input_sha256"]:
        raise ValueError("comparisons differ from the published decoder results")
    comparisons = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    ids, texts, variants, replays = baseline_texts(config)
    if [row["id"] for row in comparisons] != ids:
        raise ValueError("comparisons cover other test units")
    published = {
        (row["method"], row["dim"], row["condition"]): row["facts"]
        for row in read_json(OUTPUT)["rows"]
    }
    for row in summary["reports"]["scores"]["metrics"]:
        key = row["model"], None, row["condition"]
        variants[key] = [
            unit["generated"][f"{key[0]}/{key[2]}"] for unit in comparisons
        ]
        published[key] = {
            f: {"accuracy": row[f"{f}_accuracy_on_scorable_references"]}
            for f in JOINT_FIELDS
        }
    factual = csv_records(checked_source(config["inputs"]["factual"]))
    rewrites = csv_records(FLOOR_TEXTS)
    floor_ids = sorted(rewrites)
    floor_row = read_json(FLOOR)
    if digest(floor_ids) != floor_row["unit_ids_sha256"]:
        raise ValueError("floor texts differ from the published noise floor")
    published["same_state_rewrite", None, None] = floor_row["row"]["facts"]
    codebook = yaml.safe_load(CODEBOOK.read_text(encoding="utf-8"))

    def scored(label: str, references: list, generated: list, published: dict) -> dict:
        row = joint_accuracy(references, generated, codebook)
        # same texts and rules as the published per-fact accuracies
        if any(
            not math.isclose(row["facts"][f], published[f]["accuracy"])
            for f in JOINT_FIELDS
        ):
            raise ValueError(f"per-fact accuracy differs from the published: {label}")
        print(f"{label}: joint {row['joint']['accuracy']:.3f}")
        return row

    pairs = {
        key: (texts["reference"], generated) for key, generated in variants.items()
    }
    pairs["same_state_rewrite", None, None] = (
        [factual[key]["text"] for key in floor_ids],
        [rewrites[key]["text"] for key in floor_ids],
    )
    rows = [
        {
            "method": method,
            "dim": dim,
            "condition": condition,
            **scored(method, *texts_pair, published[method, dim, condition]),
        }
        for (method, dim, condition), texts_pair in pairs.items()
    ]
    # the renderer table: every renderer's decoded edits, phi^-1(phi(x)) against x and x'
    renderers = read_json(RENDERERS)
    raw = COMPARISONS.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(raw).hexdigest() != renderers["comparisons_sha256"]:
        raise ValueError("comparisons differ from the published renderer results")
    decoded = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if digest([row["id"] for row in decoded]) != renderers["unit_ids_sha256"]:
        raise ValueError("renderer comparisons cover other test units")
    published = {row["method"]: row for row in renderers["rows"]}
    published["decoded_unedited"] = read_json(DECODED_UNEDITED)["row"]
    renderer_rows = []
    for method, row in published.items():
        source = "factual" if method == "decoded_unedited" else method
        reference = "original" if row["against"] == "x" else "reference"
        result = scored(
            method,
            [unit[reference] for unit in decoded],
            [unit["generated"][f"{renderers['decoder']}/{source}"] for unit in decoded],
            row["facts"],
        )
        renderer_rows.append({"method": method, "against": row["against"], **result})
    # the commanded states: the true s' and the operator's draw omega_S(psi(z)), the same
    # for every renderer (S1, seed 0), here from the state flow's verified replay
    spec = yaml.safe_load(RENDERERS_CONFIG.read_text(encoding="utf-8"))
    true_target = csv_records(checked_source(spec["target_states"]))
    factual_states = csv_records(FACTUAL_STATES)
    commands = {"s'": [true_target[k] for k in ids], "omega": replays["768"]["target"]}
    if [int(c["X"]) for c in commands["omega"]] != [
        int(c["X"]) for c in commands["s'"]
    ]:
        raise ValueError("the operator's draw does not follow each unit's do(O = k)")
    by_id = {unit["id"]: unit["generated"] for unit in decoded}
    key = renderers["decoder"]
    sets = {
        (kind, "s'"): [by_id[k][f"{key}/{kind}-true-target"] for k in ids]
        for kind in KINDS
    }
    for kind in KINDS:  # package-v1 decoded the state flow's draw
        sets[kind, "omega"] = (
            [unit["generated"][f"{key}/predicted"] for unit in comparisons]
            if kind == "flow"
            else [by_id[k][f"{key}/{kind}"] for k in ids]
        )
    sets["oracle", "s'"] = [unit["generated"][f"{key}/oracle"] for unit in comparisons]
    sets["factual", "s'"] = [by_id[k][f"{key}/factual"] for k in ids]
    sources = [factual_states[k] for k in ids]
    # how often the operator's draw is the true s', per fact and for all three
    equal = {
        field: [
            int(o[c]) == int(t[c]) for o, t in zip(commands["omega"], commands["s'"])
        ]
        for field, (c, _) in JOINT_FIELDS.items()
    }
    operator = {field: mean(v) for field, v in equal.items()}
    operator["joint"] = mean(map(all, zip(*equal.values())))
    command_rows = []
    for (method, command), generated in sets.items():
        row = command_scores(generated, commands[command], sources, codebook)
        command_rows.append({"method": method, "command": command, **row})
        print(f"{method} {command}: commanded joint {row['joint']:.3f}")
    write_json(
        JOINT,
        {
            "format_version": 1,
            "units": len(ids),
            "unit_ids_sha256": digest(ids),
            "comparisons_sha256": summary["input_sha256"],
            "floor_unit_ids_sha256": floor_row["unit_ids_sha256"],
            "definition": "share of units whose CV states O (region), D and U all as the "
            "reference does, over the units whose reference states all three",
            "rows": rows,
            "renderers": {
                "comparisons_sha256": renderers["comparisons_sha256"],
                "unit_ids_sha256": renderers["unit_ids_sha256"],
                "rows": renderer_rows,
            },
            "commands": {
                "definition": "decoded O, D and U against the commanded state (the true "
                "s' or the operator's draw omega_S(psi(z))); changed: where the command "
                "changes a fact, the shares of CVs that state the commanded value "
                "(realized) or the value of x (kept); joint: all three facts at once, "
                "where the command changes the state",
                "target_states_sha256": spec["target_states"]["sha256"],
                "factual_states_sha256": sha256_file(FACTUAL_STATES),
                "operator_equals_true": operator,
                "rows": command_rows,
            },
        },
    )
    print(f"wrote {JOINT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--floor", action="store_true", help="score the noise floor only"
    )
    parser.add_argument(
        "--decoded-unedited",
        action="store_true",
        help="score phi^-1(phi(x)) against x' only",
    )
    parser.add_argument(
        "--joint",
        action="store_true",
        help="joint fact accuracy over O, D and U of the decoder fact table only",
    )
    args = parser.parse_args()
    if args.floor:
        return floor()
    if args.decoded_unedited:
        return decoded_unedited()
    if args.joint:
        return joint()
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
            "reproduced_package_metrics": {
                width: states["reproduced_package_metrics"]
                for width, states in replays.items()
            },
            "inputs_sha256": {k: v["sha256"] for k, v in config["inputs"].items()},
            "pipelines": config["pipelines"],
            **provenance(vader),
            "rows": rows,
        },
    )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
