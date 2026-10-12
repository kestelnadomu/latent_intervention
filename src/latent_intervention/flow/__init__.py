"""Normalizing-flow latent interventions.

Three flow-based variants beside the base transformer plans in
:mod:`src.latent_intervention.base`:

``state_flow`` (:mod:`.state_conditional`)
    A conditional density model ``Z = F(S, U)`` fitted only to factual ``(S, Z)``.
    Counterfactual transport abducts ``U`` under the factual state and renders the
    same noise under a target state.
``distilled_flow`` (:mod:`.distilled`)
    A deployable conditional flow ``q(Z' | Z, delta)`` fitted to samples from a
    frozen ``state_flow`` teacher with an energy-distance objective.
``direct_semantic_flow`` (:mod:`.direct_semantic`)
    The same deployable architecture fitted directly against the frozen semantic
    decoder and symbolic transition kernel.

The two direct variants share :mod:`.direct` and need only ``Z`` and ``delta`` after
training. Construction, persistence, training and sampling of any variant go through
:mod:`src.latent_intervention.dispatch`; the pipeline adapter is :mod:`.workflow`.
"""

from src.latent_intervention.flow.common import (
    FLOW_CHECKPOINT_FORMAT_VERSION,
    FLOW_VARIANTS,
    multivariate_energy_distance,
    sample_symbolic_states,
)
from src.latent_intervention.flow.direct_semantic import (
    DirectSemanticFlowIntervention,
    train_direct_semantic_flow,
)
from src.latent_intervention.flow.distilled import (
    DistilledFlowIntervention,
    train_distilled_flow,
)
from src.latent_intervention.flow.state_conditional import (
    StateConditionalFlow,
    train_state_conditional_flow,
)

__all__ = [
    "FLOW_CHECKPOINT_FORMAT_VERSION",
    "FLOW_VARIANTS",
    "StateConditionalFlow",
    "DistilledFlowIntervention",
    "DirectSemanticFlowIntervention",
    "train_state_conditional_flow",
    "train_distilled_flow",
    "train_direct_semantic_flow",
    "sample_symbolic_states",
    "multivariate_energy_distance",
]
