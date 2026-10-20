"""Preparation, numerical preflight and frozen-selection manifests."""

import importlib.metadata
import json
from pathlib import Path

import torch
import yaml

from src.artifact_io import sha256_file
from src.schema import flat_state_index
from src.latent_intervention.editor.objective import terms_for, total_loss
from src.latent_intervention.ot.training import flow_batch
from . import EDITORS, METHODS
from .common import (
    CONFIG,
    digest,
    exclusive,
    job_key,
    now,
    paths,
    read_record,
    root,
    settings_for,
    verify_files,
    write_record,
)
from .data import (
    bundle,
    counts,
    guide,
    guide_path,
    load_factual,
    load_reader_data,
    support_summary,
)
from .models import capacity, construct, load
from .sampling import posterior_rows, support_mass
from .scenarios import build, jobs, registry


def prepare(cfg, config_path=CONFIG):
    loaded = bundle(cfg)
    encoding, artifact, ids = loaded
    cases = build(encoding["sim_config"])
    rows = registry(cases)
    fitting_jobs = jobs(cfg, rows)
    inputs = {
        Path(config_path),
        Path(cfg["encoding_config"]),
        Path(cfg["reader_protocol"]),
        Path(encoding["sim_config"]),
        Path("src/config.yaml"),
    }
    for key in (
        "latents",
        "partition",
        "sim_factual",
        "interventions",
        "pair_index",
        "floor_latents",
        "texts_floor",
        "texts",
        "texts_counterfactual",
    ):
        inputs.add(Path(encoding["paths"][key]))
    latent = Path(encoding["paths"]["latents"])
    inputs.update(
        [
            latent.with_suffix(".info.json"),
            latent.with_name("manifest.json"),
            Path(encoding["paths"]["floor_latents"]).with_suffix(".info.json"),
            Path(encoding["paths"]["sim_factual"]).with_name(
                "sim_data_counterfactual_states.csv"
            ),
        ]
    )
    for space in dict.fromkeys(j["space"] for j in fitting_jobs):
        for seed in cfg["seeds"]:
            guiding = guide(cfg, space, seed, artifact)
            guiding_split = dict(
                fit=ids["reader_L"],
                validation=ids["tune"],
                evaluation_reader_reserved=ids["reader_fit"],
                official_test=ids["official_test"],
                unused_pool=[
                    i
                    for i in ids["fit"]
                    if i not in set(ids["reader_L"] + ids["reader_fit"])
                ],
            )
            expected = dict(
                split_sha256=digest(guiding_split),
                label_policy="only L and development labels of the named state space",
                training_inputs=dict(
                    sim_factual_sha256=sha256_file(encoding["paths"]["sim_factual"]),
                    sim_config_sha256=sha256_file(encoding["sim_config"]),
                ),
            )
            if any(
                guiding.checkpoint_metadata.get(k) != v for k, v in expected.items()
            ):
                raise ValueError(
                    "guiding reader factual provenance or L/P split mismatch"
                )
            inputs.add(guide_path(cfg, space, seed))
    source = {
        *Path("exp/benchmarks/misspecification").glob("*.py"),
        *Path("src/latent_intervention").rglob("*.py"),
        *Path("src/semantic_decoder").glob("*.py"),
        *Path("src/encoder").rglob("*.py"),
        *Path("exp/sim").glob("*.py"),
        *Path("exp/sim_talent").glob("*.py"),
        *Path("exp/sim_talent_motivation").glob("*.py"),
        Path("exp/benchmarks/reader/protocol.py"),
        Path("exp/benchmarks/semantic_decoder/experiment.py"),
        Path("src/schema.py"),
        Path("src/symbolic_intervention.py"),
        Path("src/pair_encoding.py"),
        Path("src/artifact_io.py"),
        Path("src/config.py"),
        Path("exp/encoding/text_artifact.py"),
        Path("scripts/run_hz_misspecification.sh"),
        Path("pyproject.toml"),
        Path("uv.lock"),
    }
    signature = dict(
        format="misspecification_protocol_v1",
        config=cfg,
        split_ids=ids,
        scenarios=rows,
        jobs=fitting_jobs,
        inputs={str(p): sha256_file(p) for p in sorted(inputs)},
        source={str(p): sha256_file(p) for p in sorted(source)},
        packages={
            p: importlib.metadata.version(p)
            for p in ("torch", "numpy", "pandas", "pyyaml")
        },
        label_regime="true factual states available; not a limited-label end-to-end claim",
        selection="factual development only; evaluation reader never selects h_Z",
    )
    signature = json.loads(
        json.dumps(signature)
    )  # Same tuple/list representation on resume.
    path = root(cfg) / "protocol.json"
    with exclusive(root(cfg) / "protocol.lock"):
        if path.exists():
            if read_record(path)["signature"] != signature:
                raise ValueError("protocol changed: choose a new run ID")
        else:
            write_record(
                path,
                dict(
                    signature=signature,
                    signature_sha256=digest(signature),
                    created_at=now(),
                ),
            )
    return signature, loaded, cases


def numerical_check(cfg, method, data, decoder, matrices):
    settings = settings_for(cfg, method, 42)
    model = construct(method, data.columns, dimension=data.fit.z.shape[-1], config=cfg)
    size = capacity(model)
    if method in EDITORS:
        model.fit_normalization(data.fit.z)
        indices = torch.randperm(
            len(data.fit.z),
            generator=torch.Generator().manual_seed(cfg["support_seed"]),
        )[: cfg["support_bank_size"]]
        bank = ((data.fit.z[indices] - model.mean) / model.scale).detach()
        z = data.tune.z[:2]
        posterior = decoder.log_joint(z).exp().detach()
        loss_value = 0.0
        for k, matrix in enumerate(matrices):
            terms = terms_for(
                model,
                z,
                posterior @ matrix,
                data.tune.states[:2, 0] == k,
                decoder,
                bank,
                create_graph=True,
            )
            loss = total_loss(terms, settings) / 4
            loss.backward()
            loss_value += float(loss.detach())
    elif method == "state_flow":
        model.set_latent_statistics(data.fit.z)
        loss = -model.log_prob(data.tune.z[:2], data.tune.states[:2]).mean()
        loss.backward()
        loss_value = float(loss.detach())
    else:
        model.fit_normalization(data.fit.z, data.fit.states)
        if method == "ot_gaussian":
            # Exercise buffer/transport validation, never fit PCA/state statistics.
            value = model.transport(
                data.tune.z[:2], data.tune.states[:2], data.fit.states[:2]
            )
            loss_value = float(value.square().mean())
        else:
            state = int(model.fit_counts.argmax())
            indices = (
                (flat_state_index(data.fit.states, data.columns) == state)
                .nonzero()
                .flatten()
            )
            x = ((data.fit.z - model.mean) / model.scale).detach()
            batch = flow_batch(
                x,
                indices,
                model.states[state],
                dict(settings, batch_size=8),
                torch.Generator().manual_seed(42),
            )
            loss = (
                (
                    model.velocity(batch["x"], batch["time"], batch["states"])
                    - batch["velocity"]
                )
                .square()
                .mean()
            )
            loss.backward()
            loss_value = float(loss.detach())
    if not torch.isfinite(torch.tensor(loss_value)) or any(
        p.grad is not None and not torch.isfinite(p.grad).all()
        for p in model.parameters()
    ):
        raise FloatingPointError("preflight found non-finite forward/backward values")
    return dict(
        capacity=size,
        finite_objective=loss_value,
        optimizer_steps=0,
        analytic_fit_performed=False,
    )


def preflight(cfg, config_path=CONFIG):
    from exp.benchmarks.semantic_decoder.experiment import candidate_settings

    torch.set_num_threads(1)
    signature, loaded, cases = prepare(cfg, config_path)
    lookup = {c.name: c for c in cases}
    spaces = dict.fromkeys(j["space"] for j in signature["jobs"])
    numerical, support = {}, {}
    for space in spaces:
        data = load_factual(cfg, space, loaded)
        decoder = guide(cfg, space, 42, loaded[1])
        support[space] = support_summary(data)
        reference = next(c for c in cases if c.name.startswith(space + "/"))
        for method in METHODS:
            numerical[f"{space}/{method}"] = numerical_check(
                cfg, method, data, decoder, reference.matrices
            )
    transition_support = {}
    for row in signature["scenarios"]:
        if row["space"] not in spaces:
            continue
        data = load_factual(cfg, row["space"], loaded)
        decoder = guide(cfg, row["space"], 42, loaded[1])
        posterior = posterior_rows(decoder, data.tune.z)
        case = lookup[row["name"]]
        fitting_counts = counts(data)
        tuning_counts = torch.bincount(
            flat_state_index(data.tune.states, data.columns),
            minlength=len(fitting_counts),
        )
        transition_support[row["name"]] = {
            str(k): dict(
                posterior_mass={
                    name: float(value.mean())
                    for name, value in support_mass(
                        posterior, matrix, fitting_counts
                    ).items()
                },
                positive_transitions=int((matrix > 0).sum()),
                fitting_observed_transitions=int(
                    (
                        (matrix > 0)
                        & (fitting_counts[:, None] > 0)
                        & (fitting_counts[None, :] > 0)
                    ).sum()
                ),
                development_observed_transitions=int(
                    (
                        (matrix > 0)
                        & (tuning_counts[:, None] > 0)
                        & (tuning_counts[None, :] > 0)
                    ).sum()
                ),
            )
            for k, matrix in enumerate(case.matrices)
        }
        if row["name"] == row["canonical"] and not row["name"].startswith(
            row["space"] + "/"
        ):
            for method in EDITORS:
                numerical[f"{row['name']}/{method}"] = numerical_check(
                    cfg, method, data, decoder, case.matrices
                )
    audit = load_reader_data(cfg, loaded)
    if set(audit.fit.ids) & set(signature["split_ids"]["reader_L"]):
        raise ValueError("evaluation reader shares L training labels")
    reader_settings = yaml.safe_load(Path(cfg["reader_protocol"]).read_text())[
        "training"
    ]
    if len(candidate_settings(reader_settings)) != 20:
        raise ValueError(
            "evaluation reader requires the predeclared 20-candidate search"
        )
    verify_files(signature)
    report = dict(
        status="passed",
        checked_at=now(),
        signature_sha256=digest(signature),
        neural_fits=sum(j["method"] != "ot_gaussian" for j in signature["jobs"]),
        analytical_fits=sum(j["method"] == "ot_gaussian" for j in signature["jobs"]),
        evaluation_reader_search_fits=20,
        numerical=numerical,
        support=support,
        development_transition_support=transition_support,
        evaluation_reader=dict(
            fit_units=len(audit.fit.ids),
            development_units=len(audit.tune.ids),
            columns=[c.name for c in audit.columns],
            independent_of_L=True,
        ),
        optimizer_steps=0,
        real_data_fitting=False,
        official_test_evaluation=False,
    )
    write_record(root(cfg) / "preflight.json", report)
    return report


def fit_job(cfg, signature, job):
    from .training import fit

    torch.set_num_threads(1)
    verify_files(signature)
    loaded = bundle(cfg)
    data = load_factual(cfg, job["space"], loaded)
    decoder, matrices = None, None
    if job["method"] in EDITORS:
        decoder = guide(cfg, job["space"], job["seed"], loaded[1])
        matrices = next(
            c for c in build(loaded[0]["sim_config"]) if c.name == job["operator"]
        ).matrices
    model = construct(job["method"], data.columns, seed=job["seed"], config=cfg)
    metadata = dict(
        protocol_sha256=digest(signature),
        job=job,
        input_hashes=signature["inputs"],
        source_hashes=signature["source"],
        config_sha256=digest(signature["config"]),
        fitting_operator=job["operator"],
        architecture_tier=1,
    )
    record = fit(
        model,
        data,
        settings_for(cfg, job["method"], job["seed"]),
        *paths(cfg, job),
        metadata,
        decoder=decoder,
        matrices=matrices,
    )
    verify_files(signature)
    return record


def freeze(cfg, signature):
    from .reader import deployment

    selected = {}
    for job in signature["jobs"]:
        checkpoint, report, _ = paths(cfg, job)
        if not report.exists():
            return None
        fit = read_record(report)
        if fit.get("status") != "complete":
            return None
        if (
            fit.get("metadata", {}).get("protocol_sha256") != digest(signature)
            or fit["metadata"].get("job") != job
        ):
            raise ValueError("completed fit belongs to a different protocol/job")
        if sha256_file(checkpoint) != fit["checkpoint_sha256"]:
            raise ValueError("completed model changed before selection freeze")
        load(checkpoint, signature=fit["signature"])
        selected[job_key(job)] = dict(
            checkpoint=str(checkpoint),
            checkpoint_sha256=fit["checkpoint_sha256"],
            report=str(report),
            report_sha256=sha256_file(report),
            fit_signature=fit["signature"],
        )
    reader_path = root(cfg) / "evaluation_reader" / "selection.json"
    if not reader_path.exists():
        return None
    reader = read_record(reader_path)
    if reader["protocol_sha256"] != digest(signature) or reader[
        "checkpoint_sha256"
    ] != sha256_file(deployment(cfg)):
        raise ValueError("evaluation reader does not match the frozen protocol")
    record = dict(
        protocol_sha256=digest(signature),
        models=selected,
        evaluation_reader=reader,
        official_test_used_for_selection=False,
    )
    path = root(cfg) / "selection.json"
    if path.exists() and read_record(path) != dict(
        record, record_sha256=digest(record)
    ):
        raise ValueError("frozen selections cannot be changed")
    write_record(path, record)
    return record


def read_protocol(cfg):
    record = read_record(root(cfg) / "protocol.json")
    signature = record["signature"]
    if cfg != signature["config"] or record["signature_sha256"] != digest(signature):
        raise ValueError("configuration differs from prepared protocol")
    verify_files(signature)
    return signature
