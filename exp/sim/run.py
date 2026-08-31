"""SCM simulation and LIBERTy-style factual/counterfactual CV generation.

Stages:
    simulate                       S and S' for every unit, plus a train/test index
    generate-templates             narrative template pool
    generate-personas              persona pool
    generate-texts                 factual X for every unit
    generate-counterfactual-texts  X' for test units only
    validate-pairs                 validate the complete paired-data contract

Each stage lives in its own module; this one only dispatches. The shared layers
are ``helpers`` (resumable billed outputs), ``paired_data`` (structured S/S'),
and ``render`` (the deterministic CV render plan and its grounding checks).
"""

from __future__ import annotations

import argparse

from exp.sim.helpers import CONFIG_PATH, load_sim_config
from exp.sim.stage_cv import stage_generate_counterfactual_texts, stage_generate_texts
from exp.sim.stage_pools import stage_generate_personas, stage_generate_templates
from exp.sim.stage_simulate import stage_simulate
from exp.sim.stage_validate import stage_validate_pairs

STAGES = {
    "simulate": stage_simulate,
    "generate-templates": stage_generate_templates,
    "generate-personas": stage_generate_personas,
    "generate-texts": stage_generate_texts,
    "generate-counterfactual-texts": stage_generate_counterfactual_texts,
    "validate-pairs": stage_validate_pairs,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", default=CONFIG_PATH)
    args = parser.parse_args()
    STAGES[args.stage](load_sim_config(args.config))


if __name__ == "__main__":
    main()
