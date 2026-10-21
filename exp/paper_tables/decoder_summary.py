"""Aggregates for the paper's text-decoder table; writes no CV text.

Two tasks on the pilot's 320 tune-evaluation units (the original campaign had
already selected decoder A on these, so they are development, not test, data):

- reconstruction: decode the factual latent z into the factual CV x (320 units);
- counterfactual: decode the oracle latent phi(x') into the counterfactual CV x'
  (the 238 units whose CV changes under do(X=3); identity units are excluded).

Decoder rows are the pilot's factual-only (A, "original") and oracle-exposed (C)
arms, recomputed from their per-CV outputs with the pilot's own metric code and
checked against its stored scores. References, scored the same way:
nearest-fit retrieval (copy the fit CV whose latent is closest; non-generative),
the decoder fed a shuffled latent (campaign control, reconstruction only), and the
unedited factual CV offered as x' (counterfactual only). Facts are extracted from
text by the campaign's conservative rules (``exp.text_realizer.facts``).

Needs rapidfuzz and sacrebleu, pinned as in the pilot's environment:

    uv run --with rapidfuzz==3.14.1 --with sacrebleu==2.5.1 \
        python -m exp.paper_tables.decoder_summary
"""

import csv
import hashlib
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from exp.text_realizer import facts
from src.text_realizer.metrics import text_metrics

PILOT = Path("reports/talent/embeddinggemma_768/text-realizer-cf/cv-cf-pilot-v1")
CAMPAIGN = Path("reports/talent/embeddinggemma_768/text-realizer/cv-text-v1-20261012-1719")
SPLIT_PLAN = Path("reports/talent/hz_benchmarks/hz-gemma-v1/plan.json")
PAIR_INDEX = Path("data/sim_talent/pair_index.csv")
CV = {"factual": Path("data/text_talent/cv_factual.csv"), "counterfactual": Path("data/text_talent/cv_counterfactual.csv")}
LATENTS = Path("data/latents/talent/embeddinggemma_768/z_pairs.pt")
ORACLE = Path("data/latents/talent/embeddinggemma_768/oracle_train/z_prime.pt")
ARMS = ("original", "oracle")
FIELDS = ("region", "degree", "university_tier", "project_count")
TEXT_METRICS = ("character_similarity", "chrf", "rouge_l")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _texts(path):
    csv.field_size_limit(10**9)
    with open(path, encoding="utf-8", newline="") as stream:
        return {int(row["id"]): row["text"] for row in csv.DictReader(stream)}


def _summary(references, predictions, rows=None):
    """Mean text metrics (stored per-CV rows if given) and conservative fact accuracy."""
    if rows is None:
        rows = [text_metrics(r, p) for r, p in zip(references, predictions, strict=True)]
    fields = facts.evaluate(references, predictions)["fields"]
    return {
        "units": len(references),
        **{key: statistics.mean(row[key] for row in rows) for key in TEXT_METRICS},
        "facts": {
            field: {
                "accuracy": fields[field]["accuracy_on_scorable_references"],
                "scorable": fields[field]["reference_scorable"],
                "unparseable": fields[field]["unparseable_predictions"],
            }
            for field in FIELDS
        },
    }


def _check(name, ours, stored, tolerance=1e-9):
    if abs(ours - stored) > tolerance:
        raise ValueError(f"{name}: recomputed {ours} differs from stored {stored}")


def compute():
    result = _read_json(PILOT / "result.json")
    texts = {world: _texts(path) for world, path in CV.items()}
    with open(PAIR_INDEX, encoding="utf-8", newline="") as stream:
        identity = {int(r["id"]): r["is_identity"].strip().lower() in ("true", "1") for r in csv.DictReader(stream)}

    ids = result["worlds"]["original"]["factual"]["ids"]
    for arm in ARMS:
        for world in ("factual", "oracle"):
            if result["worlds"][arm][world]["ids"] != ids:
                raise ValueError(f"{arm}/{world} uses a different unit order")
    changed = [i for i, key in enumerate(ids) if not identity[key]]
    if len(changed) != result["decision"]["changed_units"]:
        raise ValueError("changed-unit count differs from the pilot's decision record")

    factual_refs = [texts["factual"][key] for key in ids]
    counterfactual_refs = [texts["counterfactual"][ids[i]] for i in changed]
    rows = {"reconstruction": {}, "counterfactual": {}}

    # Decoders: the pilot's own per-CV outputs and metrics.
    for arm in ARMS:
        world = result["worlds"][arm]
        factual_rows = world["factual"]["rows"]
        oracle_rows = [world["oracle"]["rows"][i] for i in changed]
        for reference, row in zip(factual_refs[:5], factual_rows[:5]):
            _check(f"{arm} reference text", text_metrics(reference, row["text"])["character_similarity"], row["character_similarity"])
        rows["reconstruction"][arm] = _summary(factual_refs, [r["text"] for r in factual_rows], factual_rows)
        rows["counterfactual"][arm] = _summary(counterfactual_refs, [r["text"] for r in oracle_rows], oracle_rows)
        stored = result["decision"]["arm_scores"][arm]
        _check(f"{arm} reconstruction", rows["reconstruction"][arm]["character_similarity"], stored["factual"]["character_similarity"])
        _check(f"{arm} counterfactual", rows["counterfactual"][arm]["character_similarity"], stored["oracle_changed"]["character_similarity"])

    # Reconstruction references from the campaign's tune report of decoder A (seed 42).
    tune = _read_json(CAMPAIGN / "tune-seed-42.json")
    for name, source in (("retrieval", tune["retrieval"]["rows"]), ("shuffled", tune["conditions"]["shuffled"]["rows"])):
        by_id = {row["id"]: row for row in source}
        subset = [by_id[key] for key in ids]
        rows["reconstruction"][name] = _summary(factual_refs, [r["text"] for r in subset], subset)

    # Nearest-fit retrieval with the campaign's rule; reproduce its factual choices first.
    latents = torch.load(LATENTS, weights_only=False)
    oracle = torch.load(ORACLE, weights_only=False)
    position = {key: i for i, key in enumerate(latents["ids"])}
    fit_ids = _read_json(SPLIT_PLAN)["split_ids"]["fit"]
    fit_z = F.normalize(latents["z"][[position[key] for key in fit_ids]].float(), dim=1)

    def nearest(z):
        return [fit_ids[i] for i in (F.normalize(z.float(), dim=1) @ fit_z.T).argmax(1).tolist()]

    factual_z = latents["z"][[position[key] for key in ids]]
    campaign_choice = {row["id"]: row["fit_id"] for row in tune["retrieval"]["rows"]}
    if nearest(factual_z) != [campaign_choice[key] for key in ids]:
        raise ValueError("fit set or retrieval rule differs from the campaign")
    oracle_position = {key: i for i, key in enumerate(oracle["train_ids"])}
    oracle_z = oracle["z_prime"][[oracle_position[ids[i]] for i in changed]]
    retrieved = [texts["factual"][key] for key in nearest(oracle_z)]
    rows["counterfactual"]["retrieval"] = _summary(counterfactual_refs, retrieved)
    rows["counterfactual"]["unedited"] = _summary(counterfactual_refs, [texts["factual"][ids[i]] for i in changed])

    # Held-out audit of decoder A over its three seeds, for the caption.
    audit = [_read_json(CAMPAIGN / f"audit-seed-{seed}.json") for seed in (42, 43, 44)]
    similarity = [a["conditions"]["correct"]["summary"]["character_similarity"] for a in audit]
    return {
        "run": PILOT.name,
        "scope": "development tune units; decoder A was selected on them; one training seed per decoder",
        "units": {"reconstruction": len(ids), "counterfactual": len(changed), "identity_excluded": len(ids) - len(changed)},
        "rows": rows,
        "audit_original": {
            "units": len(audit[0]["conditions"]["correct"]["rows"]),
            "seeds": [42, 43, 44],
            "character_similarity_mean": statistics.mean(similarity),
            "character_similarity_sd": statistics.stdev(similarity),
            "retrieval_character_similarity": audit[0]["retrieval"]["summary"]["character_similarity"],
        },
        "sources_sha256": {
            str(p): _sha256(p)
            for p in (PILOT / "result.json", CAMPAIGN / "tune-seed-42.json", PAIR_INDEX, *CV.values(), LATENTS, ORACLE)
        },
    }


def main():
    summary = compute()
    path = PILOT / "paper_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for task, entries in summary["rows"].items():
        for name, entry in entries.items():
            accuracy = " ".join(f"{f[:6]} {100 * entry['facts'][f]['accuracy']:5.1f}" for f in FIELDS)
            print(f"{task:14s} {name:9s} n={entry['units']} char {entry['character_similarity']:.4f} "
                  f"chrF {entry['chrf']:.4f}  {accuracy}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
