"""Deterministic CV rendering: what every generated text is grounded in.

One persisted render plan (template, persona, and one quantile per binned
field) fixes the context of a unit's factual and counterfactual CV, so the only
difference between X and X' is the intervened structured state. ``RenderContext``
bundles everything the CV and validation stages need; ``validate_grounding``
replays the plan to check a written CSV against it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from exp.sim.generate_text import DEFAULT_BASE_URL, GenerationResult, load_codebook_spec
from exp.sim.helpers import (
    input_digest,
    load_pool,
    read_csv,
    slug,
)
from exp.sim.paired_data import load_pair_inputs
from exp.sim.pairing import build_render_plan, materialize_binned_values

RENDERER_VERSION = "liberty-fixed-context-v1"
RESPONSE_COLUMNS = [
    "response_id",
    "model",
    "system_fingerprint",
    "finish_reason",
    "generation_mode",
]


def load_render_spec(config: Mapping[str, Any]) -> dict[str, Any]:
    """Load the codebook and check it covers exactly the configured schema."""
    spec = load_codebook_spec(config["codebook"])
    cardinalities = {str(k): int(v) for k, v in config["schema"]["columns"].items()}
    if set(spec["columns"]) != set(cardinalities):
        raise ValueError("codebook columns must exactly match schema.columns")
    for column, cardinality in cardinalities.items():
        levels = set(range(cardinality))
        if set(spec["columns"][column]) != levels:
            raise ValueError(f"codebook levels for {column} must be 0..{cardinality - 1}")
        if column in spec["bins"] and set(spec["bins"][column]) != levels:
            raise ValueError(f"bins for {column} must cover every category")
    return spec


def cv_schema(spec: Mapping[str, Any]) -> tuple[list[str], dict[str, str]]:
    """Column layout of a generated CV CSV and the header of each binned field."""
    headers = {column: slug(spec["labels"].get(column, column)) for column in spec["bins"]}
    return [
        "id",
        "template_id",
        "persona_id",
        *headers.values(),
        "text",
        *RESPONSE_COLUMNS,
    ], headers


def ensure_render_plan(
    config: Mapping[str, Any],
    pairs: pd.DataFrame,
    templates: pd.DataFrame,
    personas: pd.DataFrame,
    binned: list[str],
) -> pd.DataFrame:
    """Write the render plan on first use, or check the persisted one still matches."""
    expected = build_render_plan(
        pairs["id"],
        templates["template_id"],
        personas["persona_id"],
        binned,
        seed=int(config["seed"]),
    )
    path = Path(config["paths"]["render_plan"])
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        expected.to_csv(path, index=False)
        print(f"wrote {path} ({len(expected)} rows)")
        return expected
    observed = read_csv(path, "render plan", expected.columns.drop("id"))
    try:
        pd.testing.assert_frame_equal(observed, expected, check_dtype=False, atol=0, rtol=0)
    except AssertionError as exc:
        raise RuntimeError("render_plan.csv is incompatible with current inputs") from exc
    return observed


def generation_digest(config: Mapping[str, Any], counterfactual: bool) -> str:
    """Digest of every input a factual or counterfactual CV file depends on."""
    sim_dir = Path(config["paths"]["sim_dir"])
    states = "sim_data_counterfactual.csv" if counterfactual else "sim_data_factual.csv"
    files = {
        "states": sim_dir / states,
        "render_plan": Path(config["paths"]["render_plan"]),
        "templates": Path(config["paths"]["templates"]),
        "personas": Path(config["paths"]["personas"]),
        "prompt": Path(config["prompts"]["cv"]),
        "codebook": Path(config["codebook"]),
    }
    settings: dict[str, Any] = {
        "renderer": RENDERER_VERSION,
        "world": "counterfactual" if counterfactual else "factual",
        "endpoint": config.get("llm", {}).get("base_url", DEFAULT_BASE_URL),
    }
    if counterfactual:
        files["pair_index"] = Path(config["paths"]["pair_index"])
        files["factual_text"] = Path(config["paths"]["texts"])
        settings["intervention"] = config["intervention"]
    return input_digest(settings, files)


def _candidate_info(
    state: Mapping[str, Any],
    spec: Mapping[str, Any],
    concrete: Mapping[str, int],
) -> str:
    parts = []
    for column, levels in spec["columns"].items():
        category = int(state[column])
        label = spec["labels"].get(column, column)
        value = concrete[column] if column in concrete else levels[category]
        parts.append(f"{label}: {value}")
    return "[" + ", ".join(parts) + "]"


# --- the shared rendering context ----------------------------------------------------


@dataclass
class RenderContext:
    """Everything needed to render, or re-check, one unit's CV."""

    config: Mapping[str, Any]
    spec: dict[str, Any]
    binned: list[str]
    schema: list[str]
    headers: dict[str, str]
    templates: pd.DataFrame  # indexed by template_id
    personas: pd.DataFrame  # indexed by persona_id
    factual: pd.DataFrame
    counterfactual: pd.DataFrame
    pairs: pd.DataFrame
    plan: pd.DataFrame
    factual_by_id: pd.DataFrame
    counterfactual_by_id: pd.DataFrame
    pairs_by_id: pd.DataFrame
    plan_by_id: pd.DataFrame

    @property
    def all_ids(self) -> set[int]:
        return set(self.pairs["id"])

    @property
    def test_ids(self) -> set[int]:
        return set(self.pairs.loc[self.pairs["split"] == "test", "id"])

    def states(self, counterfactual: bool) -> pd.DataFrame:
        return self.counterfactual if counterfactual else self.factual

    def states_by_id(self, counterfactual: bool) -> pd.DataFrame:
        return self.counterfactual_by_id if counterfactual else self.factual_by_id

    def output_path(self, counterfactual: bool) -> Path:
        key = "texts_counterfactual" if counterfactual else "texts"
        return Path(self.config["paths"][key])

    def render_inputs(
        self, row_id: int, counterfactual: bool
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """Prompt sample and concrete binned values for one unit."""
        state = self.states_by_id(counterfactual).loc[row_id]
        plan = self.plan_by_id.loc[row_id]
        template = self.templates.loc[int(plan["template_id"])]
        persona = self.personas.loc[int(plan["persona_id"])]
        concrete = materialize_binned_values(state, self.spec["bins"], plan)
        sample = {
            "cv_template": template["text"],
            "candidate_info": _candidate_info(state, self.spec, concrete),
            "persona_details": f"Job Title: {persona['job_title']}\n{persona['text']}",
        }
        return sample, concrete

    def cv_row(
        self,
        row_id: int,
        concrete: Mapping[str, int],
        text: str,
        result: GenerationResult | None,
        mode: str,
    ) -> list[Any]:
        """One CSV row in ``self.schema`` order."""
        plan = self.plan_by_id.loc[row_id]
        return [
            row_id,
            int(plan["template_id"]),
            int(plan["persona_id"]),
            *(concrete[column] for column in self.binned),
            text,
            result.response_id if result else None,
            result.model if result else None,
            result.system_fingerprint if result else None,
            result.finish_reason if result else None,
            mode,
        ]


def render_context(config: Mapping[str, Any]) -> RenderContext:
    """Load S/S', the pools, and the render plan into one checked context."""
    factual, counterfactual, pairs = load_pair_inputs(config)
    spec = load_render_spec(config)
    binned = list(spec["bins"])
    schema, headers = cv_schema(spec)
    templates = load_pool(config, "templates")
    personas = load_pool(config, "personas")
    plan = ensure_render_plan(config, pairs, templates, personas, binned)
    return RenderContext(
        config=config,
        spec=spec,
        binned=binned,
        schema=schema,
        headers=headers,
        templates=templates.set_index("template_id"),
        personas=personas.set_index("persona_id"),
        factual=factual,
        counterfactual=counterfactual,
        pairs=pairs,
        plan=plan,
        factual_by_id=factual.set_index("id"),
        counterfactual_by_id=counterfactual.set_index("id"),
        pairs_by_id=pairs.set_index("id"),
        plan_by_id=plan.set_index("id"),
    )


def validate_grounding(
    ctx: RenderContext,
    output: pd.DataFrame,
    *,
    counterfactual: bool,
    factual_text: pd.DataFrame | None = None,
) -> None:
    """Replay the render plan and check a written CV file matches it row by row."""
    if output.empty:
        return
    factual_by_id = factual_text.set_index("id") if factual_text is not None else None
    for row in output.to_dict(orient="records"):
        row_id = int(row["id"])
        plan_row = ctx.plan_by_id.loc[row_id]
        _, concrete = ctx.render_inputs(row_id, counterfactual)
        if int(row["template_id"]) != int(plan_row["template_id"]):
            raise ValueError(f"incorrect template for id={row_id}")
        if int(row["persona_id"]) != int(plan_row["persona_id"]):
            raise ValueError(f"incorrect persona for id={row_id}")
        for column, value in concrete.items():
            if int(row[ctx.headers[column]]) != value:
                raise ValueError(f"incorrect concrete {column} for id={row_id}")
        identity = bool(ctx.pairs_by_id.at[row_id, "is_identity"])
        expected_mode = "identity_copy" if counterfactual and identity else "generated"
        if row["generation_mode"] != expected_mode:
            raise ValueError(f"incorrect generation mode for id={row_id}")
        if expected_mode == "generated":
            if row["finish_reason"] != "stop":
                raise ValueError(f"generated text did not finish normally for id={row_id}")
            if pd.isna(row["model"]) or not str(row["model"]).strip():
                raise ValueError(f"generated text has no response model for id={row_id}")
        else:
            if factual_by_id is None or row["text"] != factual_by_id.at[row_id, "text"]:
                raise ValueError(f"identity counterfactual text differs for id={row_id}")
            if any(pd.notna(row[column]) for column in RESPONSE_COLUMNS[:-1]):
                raise ValueError(f"identity counterfactual has response metadata for id={row_id}")
