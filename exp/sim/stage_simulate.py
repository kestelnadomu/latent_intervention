"""Stage ``simulate``: aligned S/S' for every unit plus the seeded pair index."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from exp.sim.design import (
    build_partition,
    counterfactual_states,
    draw_interventions,
    load_design,
    noise_from_epsilon,
    text_counterfactual,
)
from exp.sim.helpers import generation_info_path, write_json
from exp.sim.paired_data import (
    simulation_paths,
    simulation_record,
    split_settings,
    state_columns,
    validate_query,
)
from exp.sim.pairing import PairingError, build_pair_index, checked_ids


def _archive_existing(paths: Iterable[str | Path]) -> None:
    """Move existing run artifacts aside using one date-only suffix."""
    existing = list(dict.fromkeys(Path(path) for path in paths if Path(path).exists()))
    if not existing:
        return
    stamp = datetime.now().strftime("%y-%m-%d")
    destinations = {}
    for path in existing:
        suffixes = "".join(path.suffixes)
        basename = path.name[: -len(suffixes)] if suffixes else path.name
        destinations[path] = path.with_name(f"{basename}_{stamp}{suffixes}")
    collisions = [destination for destination in destinations.values() if destination.exists()]
    if collisions:
        raise FileExistsError(f"dated archive already exists: {collisions[0]}")
    for path, destination in destinations.items():
        path.rename(destination)
        print(f"archived {path} -> {destination}")


def stage_simulate(config: dict[str, Any]) -> None:
    """Generate aligned S/S' for all units and a seeded test index."""
    from src.schema import load_object

    factory = load_object(config["objects"]["scm"])
    scm = factory(config)
    factual, counterfactual, epsilon = scm.simulate(
        n=int(config["n"]),
        intervention=config["intervention"],
        seed=config.get("seed"),
    )
    factual = checked_ids(factual, "factual simulation")
    counterfactual = checked_ids(counterfactual, "counterfactual simulation")
    epsilon = checked_ids(epsilon, "simulation noise")
    expected_ids = factual["id"].tolist()
    if (
        len(expected_ids) != int(config["n"])
        or counterfactual["id"].tolist() != expected_ids
        or epsilon["id"].tolist() != expected_ids
    ):
        raise PairingError("configured SCM must return n aligned factual/counterfactual/noise IDs")
    design_outputs = _per_unit_design(config, scm, factual, epsilon)
    if design_outputs:
        counterfactual = design_outputs.pop("counterfactual")
    validate_query(factual, counterfactual, config, design_outputs.get("interventions"))
    split_seed, test_fraction = split_settings(config)
    pair_index = build_pair_index(
        factual,
        counterfactual,
        state_columns(config),
        seed=split_seed,
        test_fraction=test_fraction,
    )
    spec = load_design(config)
    if spec is not None:
        design_outputs["partition"] = build_partition(pair_index, spec, split_seed)

    sim_dir = Path(config["paths"]["sim_dir"])
    text_outputs = [
        Path(config["paths"]["texts"]),
        Path(config["paths"]["texts_counterfactual"]),
    ]
    archive_paths = [
        *simulation_paths(config).values(),
        sim_dir / "simulation_info.json",
        Path(config["paths"]["render_plan"]),
    ]
    for output in text_outputs:
        archive_paths.extend([output, generation_info_path(output)])
    _archive_existing(archive_paths)

    sim_dir.mkdir(parents=True, exist_ok=True)
    outputs = simulation_paths(config)
    for name, frame in (
        ("factual", factual),
        ("counterfactual", counterfactual),
        ("epsilon", epsilon),
        ("pair_index", pair_index),
        *design_outputs.items(),
    ):
        outputs[name].parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(outputs[name], index=False)
        print(f"wrote {outputs[name]} ({len(frame)} rows)")

    write_json(sim_dir / "simulation_info.json", simulation_record(config))
    changed = ~pair_index["is_identity"]
    if spec is None:
        print(f"intervention {config['intervention']}: {changed.mean():.1%} of units changed")
    else:
        partition = design_outputs["partition"]
        sizes = partition["partition"].value_counts().to_dict()
        print(
            f"per-unit do({spec.node}): {changed.mean():.1%} of units changed; partition {sizes}, "
            f"L={int((partition['L_rank'] > 0).sum())}, P={int(partition['in_P'].sum())}, "
            f"floor={int(partition['floor'].sum())}"
        )


def _per_unit_design(
    config: dict[str, Any], scm: Any, factual: pd.DataFrame, epsilon: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Interventions, counterfactual states and the text counterfactual of a per-unit design."""
    spec = load_design(config)
    if spec is None:
        return {}
    noise = noise_from_epsilon(epsilon)
    regenerated = scm.generate(dict(noise))
    if not regenerated.equals(factual.drop(columns="id").reset_index(drop=True)):
        raise PairingError("regenerating the factual world from the stored noise must be exact")
    cardinality = int(config["schema"]["columns"][spec.node])
    interventions = draw_interventions(factual, spec.node, cardinality, int(config["seed"]))
    states = counterfactual_states(scm, noise, factual, spec, cardinality)
    columns = [column for column in factual.columns if column != "id"]
    return {
        "interventions": interventions,
        "counterfactual_states": states,
        "counterfactual": text_counterfactual(states, interventions, spec, columns),
    }
