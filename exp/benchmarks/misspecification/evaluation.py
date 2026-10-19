"""Explicit held-out evaluation. Never imported by fitting or selection code."""

import json
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

from exp.encoding.text_artifact import load_text_artifact
from src.artifact_io import save_torch, sha256_file
from src.schema import ColumnSpec
from src.semantic_decoder.model import load_semantic_decoder
from . import AUDIT_COLUMNS, EDITORS
from .common import (
    digest,
    exclusive,
    job_key,
    read_record,
    root,
    verify_files,
    write_record,
)
from .data import FactualPartition, bundle, counts, guide, guide_path, load_factual
from .metrics import (
    aggregate,
    embedding_match,
    embedding_metrics,
    factual_confusion,
    symbolic_metrics,
)
from .models import load
from .reader import calibration
from .sampling import apply_pairs, draw_pairs, posterior_rows, support_mass
from .scenarios import build


def frozen(cfg, signature):
    selection = read_record(root(cfg) / "selection.json")
    if (
        selection["protocol_sha256"] != digest(signature)
        or selection["official_test_used_for_selection"]
    ):
        raise ValueError(
            "official-test evaluation requires frozen, development-only selections"
        )
    if set(selection["models"]) != {job_key(j) for j in signature["jobs"]}:
        raise ValueError("selection manifest does not cover the prepared fitting queue")
    for item in [*selection["models"].values(), selection["evaluation_reader"]]:
        if sha256_file(item["checkpoint"]) != item["checkpoint_sha256"]:
            raise ValueError("frozen deployment checkpoint changed")
        if "report" in item and sha256_file(item["report"]) != item["report_sha256"]:
            raise ValueError("frozen training report changed")
    verify_files(signature)
    return selection


def evaluation_data(cfg, signature, selection):
    """This is the only boundary exposing oracle/auxiliary evaluation labels."""
    encoding, artifact, splits = bundle(cfg)
    ids = splits["official_test"]
    index = {v: i for i, v in enumerate(artifact.ids)}
    if artifact.test_ids != ids:
        raise ValueError("official-test embedding order mismatch")
    factual = pd.read_csv(
        encoding["paths"]["sim_factual"], usecols=["id", *AUDIT_COLUMNS]
    ).set_index("id")
    path = Path(encoding["paths"]["sim_factual"]).with_name(
        "sim_data_counterfactual_states.csv"
    )
    cf = pd.read_csv(path, usecols=["id", "do_value", "target", *AUDIT_COLUMNS])
    if cf.duplicated(["id", "do_value", "target"]).any():
        raise ValueError("duplicate counterfactual-state keys")
    cf = cf.set_index(["id", "do_value", "target"])
    expected = pd.MultiIndex.from_product([ids, range(4), ["total", "ps_D", "ps_DU"]])
    if not expected.isin(cf.index).all():
        raise ValueError("missing official-test symbolic target")
    interventions = pd.read_csv(encoding["paths"]["interventions"])
    if interventions.id.duplicated().any():
        raise ValueError("duplicate oracle intervention IDs")
    available = dict(zip(interventions.id, interventions.do_value))
    manifest = json.loads(
        Path(encoding["paths"]["latents"]).with_name("manifest.json").read_text()
    )
    # The available actions must be those used when the actual texts were encoded.
    if (
        sha256_file(encoding["paths"]["interventions"])
        != manifest["protocol"]["input_sha256"]["interventions"]
    ):
        raise ValueError("oracle actions differ from encoded text provenance")
    columns = tuple(ColumnSpec(n, 4 if n == "X" else 3) for n in AUDIT_COLUMNS)
    reader = load_semantic_decoder(
        selection["evaluation_reader"]["checkpoint"],
        expected_variant="independent",
        expected_columns=columns,
        expected_metadata={
            "role": "independent_evaluation_only",
            "protocol_sha256": digest(signature),
            "latent_artifact_sha256": artifact.artifact_sha256,
        },
    )
    reader.eval().requires_grad_(False)
    outside_p = [
        i for i in [*splits["fit"], *ids] if i not in set(splits["reader_fit"])
    ]
    ref_z = artifact.z[[index[i] for i in outside_p]]
    ref_states = torch.tensor(
        factual.loc[outside_p, list(AUDIT_COLUMNS)].to_numpy(), dtype=torch.long
    )
    confusion = factual_confusion(reader, ref_z, ref_states, columns)
    partition = pd.read_csv(encoding["paths"]["partition"])
    floor_ids = partition.loc[partition["floor"].astype(bool), "id"].tolist()
    floor = load_text_artifact(
        encoding["paths"]["floor_latents"],
        texts=encoding["paths"]["texts_floor"],
        expected_ids=floor_ids,
        kind="floor",
        encoder_config=encoding["encoder"],
        provenance=manifest["protocol"]["input_sha256"],
    )
    factual_floor = artifact.z[[index[i] for i in floor["ids"]]]
    scale = (
        artifact.z[[index[i] for i in splits["fit"]]]
        .std(0, unbiased=False)
        .clamp_min(1e-6)
    )
    floor_values = embedding_metrics(factual_floor[None], floor["z"], scale)
    floor_record = dict(
        ids=floor["ids"],
        metrics={k: v.tolist() for k, v in floor_values.items()},
        description="same-state text variation, not a paired counterfactual target",
    )
    test_z = artifact.z[[index[i] for i in ids]]
    test_states = torch.tensor(
        factual.loc[ids, list(AUDIT_COLUMNS)].to_numpy(), dtype=torch.long
    )
    reference = dict(
        protocol_sha256=digest(signature),
        reader_sha256=selection["evaluation_reader"]["checkpoint_sha256"],
        confusion_ids=outside_p,
        confusion_pooling="per head and true level; outside P; evaluation only",
        confusion=confusion,
        same_state_text_variation=floor_record,
        factual_test_calibration=calibration(
            reader, FactualPartition(test_z, test_states, tuple(ids)), columns
        ),
    )
    write_record(root(cfg) / "evaluation" / "references.json", reference)
    return dict(
        ids=ids,
        z=test_z,
        oracle=artifact.z_prime,
        factual=test_states,
        cf=cf,
        available_actions=available,
        columns=columns,
        auditor=reader,
        scale=scale,
        confusion=confusion,
        loaded=(encoding, artifact, splits),
    )


def condition_path(cfg, case, method, seed, action, target):
    return (
        root(cfg)
        / "evaluation"
        / case.replace("/", "--")
        / method
        / f"seed-{seed}"
        / f"action-{action}"
        / target
        / "metrics.json"
    )


def evaluate_condition(
    cfg, signature, selection, data, case, method, seed, action_value, target
):
    if target not in case.targets:
        raise ValueError("unregistered symbolic evaluation target")
    job = dict(
        space=case.space,
        method=method,
        operator=case.name if method in EDITORS else "factual",
        seed=42 if method == "ot_gaussian" else seed,
    )
    item = None if method == "no_edit" else selection["models"][job_key(job)]
    context = digest(
        dict(
            guide=sha256_file(guide_path(cfg, case.space, seed)),
            operator=case.fingerprint,
        )
    )
    identity = dict(
        protocol=digest(signature),
        selection=digest(selection),
        method=method,
        checkpoint=None if item is None else item["checkpoint_sha256"],
        context=context,
        action=action_value,
        target=target,
        reader_seed=seed,
    )
    path = condition_path(cfg, case.name, method, seed, action_value, target)
    array_path = path.with_name("per_unit.pt")
    with exclusive(path.with_name("evaluation.lock")):
        if path.exists():
            record = read_record(path)
            if (
                record["identity"] != identity
                or sha256_file(array_path) != record["per_unit_sha256"]
            ):
                raise ValueError("stale or changed evaluation")
            return record
        model = (
            None
            if item is None
            else load(item["checkpoint"], signature=item["fit_signature"])
        )
        decoder = guide(cfg, case.space, seed, data["loaded"][1])
        factual_data = load_factual(cfg, case.space, data["loaded"])
        fit_counts = counts(factual_data)
        action, matrix = {"X": action_value}, case.matrices[action_value]
        ids, z = data["ids"], data["z"]
        truth = torch.tensor(
            data["cf"]
            .loc[[(i, action_value, target) for i in ids], list(AUDIT_COLUMNS)]
            .to_numpy(),
            dtype=torch.long,
        )
        matched = embedding_match(ids, action, target, data["available_actions"])
        collected, strata, cost = (
            {},
            {},
            dict(nfe=0, velocity_point_evaluations=0, sampled_pairs=0, unique_pairs=0),
        )
        started = perf_counter()
        for start in range(0, len(ids), cfg["evaluation_batch_size"]):
            sl = slice(start, start + cfg["evaluation_batch_size"])
            posterior = posterior_rows(decoder, z[sl])
            pairs = draw_pairs(
                posterior,
                matrix,
                ids[sl],
                decoder.columns,
                action=action,
                context=context,
                seed=cfg["evaluation_seed"],
                n_samples=cfg["evaluation_samples"],
            )
            if model is None:
                samples = z[sl][None].expand(cfg["evaluation_samples"], -1, -1)
                diagnostics = {}
            else:
                samples, diagnostics = apply_pairs(model, z[sl], pairs)
                empty_pairs = draw_pairs(
                    posterior,
                    matrix,
                    ids[sl],
                    decoder.columns,
                    action={},
                    context=context,
                    seed=cfg["evaluation_seed"],
                    n_samples=cfg["evaluation_samples"],
                )
                empty, _ = apply_pairs(model, z[sl], empty_pairs)
                if not torch.equal(empty, z[sl][None].expand_as(empty)):
                    raise ValueError("empty-action identity failed")
            for k, value in diagnostics.items():
                cost[k] = cost.get(k, 0) + value
            values, transition = symbolic_metrics(
                samples,
                z[sl],
                posterior,
                matrix,
                pairs,
                data["factual"][sl],
                truth[sl],
                decoder,
                data["auditor"],
                data["columns"],
                fit_counts,
                data["confusion"],
            )
            values.update(
                {
                    f"support_{k}": v
                    for k, v in support_mass(posterior, matrix, fit_counts).items()
                }
            )
            # Deliberately never evaluate a distance to a mismatched oracle vector.
            local_match = matched[sl]
            for k in ("energy_score", "mean_l2", "mean_mse", "mean_cosine"):
                values[k] = torch.full((len(samples[0]),), float("nan"))
            if local_match.any():
                recovery = embedding_metrics(
                    samples[:, local_match],
                    data["oracle"][sl][local_match],
                    data["scale"],
                )
                for k, value in recovery.items():
                    values[k][local_match] = value
            for key, value in values.items():
                if torch.isinf(value).any():
                    raise FloatingPointError(f"infinite evaluation metric: {key}")
                collected.setdefault(key, []).append(value)
            for key, values in transition.items():
                for metric, value in values.items():
                    strata.setdefault(key, {}).setdefault(metric, 0)
                    strata[key][metric] += value
        collected = {k: torch.cat(v).cpu() for k, v in collected.items()}
        actual_identity = data["factual"][:, 0] == action_value
        save_torch(
            array_path,
            dict(
                ids=ids,
                values=collected,
                factual_identity=actual_identity,
                embedding_available=matched,
                target=target,
                do_value=action_value,
            ),
        )
        record = dict(
            identity=identity,
            scenario=case.name,
            state_space=case.space,
            method=method,
            reader_seed=seed,
            model_seed=None if model is None else job["seed"],
            action=action_value,
            target=target,
            summary=aggregate(collected, actual_identity),
            transition_strata=strata,
            primary_distribution="complete; no support-mass filtering",
            empty_action_exact=True,
            embedding_unavailable_reason="only matching (id, do_value, total) embeddings are available",
            per_unit=str(array_path),
            per_unit_sha256=sha256_file(array_path),
            elapsed_seconds=perf_counter() - started,
            inference_cost=cost,
        )
        write_record(path, record)
        return record


def evaluate(cfg, signature, *, methods, spaces, operators, seeds):
    torch.set_num_threads(1)
    selection = frozen(cfg, signature)
    with exclusive(root(cfg) / "evaluation" / "run.lock"):
        data = evaluation_data(cfg, signature, selection)
        by_name = {c.name: c for c in build(data["loaded"][0]["sim_config"])}
        canonical = {
            r["canonical"]
            for r in signature["scenarios"]
            if (not spaces or r["space"] in spaces)
            and (not operators or r["name"] in operators)
        }
        for name in sorted(canonical):
            case = by_name[name]
            for seed in seeds:
                for method in [*methods, "no_edit"]:
                    for k in range(4):
                        for target in case.targets:
                            evaluate_condition(
                                cfg,
                                signature,
                                selection,
                                data,
                                case,
                                method,
                                seed,
                                k,
                                target,
                            )
        verify_files(signature)
