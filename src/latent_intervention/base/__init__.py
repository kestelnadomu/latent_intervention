"""
Latent editor h_Z(z' | z, delta): Z -> Delta(Z).

Freeze f, g, h_S; train h_Z so the two paths from z to a distribution over counterfactual
states agree (h_S . g == g . h_Z). See docs/architecture/latent_intervention.md, which holds
the overview table, benefits and caveats of every plan.

Five parameterisations (config `latent_intervention.variant`):

- LatentIntervention (`baseline`) -- Plan 0: deterministic z' = z + Delta_theta(z, delta).
  Trained by per-column CE against the simulated S'; learns the per-column marginals of the
  counterfactual and cannot represent its multimodality.
- LatentInterventionPreAdditive (`pre_additive`) -- Plan A (engression):
  z' = z + Delta_theta(z + eps, delta), eps ~ N(0, sigma^2 I). Monte-Carlo composition.
- LatentInterventionNoiseToken (`noise_token`) -- Plan B (outsourced noise):
  z' = z + Delta_theta(z, eps, delta), eps ~ N(0, I_r) as its own token, so z reaches
  Delta intact. Monte-Carlo composition + per-sample entropy term.
- LatentInterventionDist (`dist`) -- Plan C (discrete mixture over states):
  h_Z = sum_{s' in top-k} w_theta(s' | z, delta) * dirac_{z + Delta_phi(z, s')}.
  w_theta is an autoregressive head distilled from h_S . g; Delta_phi is a do()-agnostic
  realiser. Exact composition; trained pretrain-then-joint.
- LatentInterventionParticles (`particles`) -- Plan D (particle set):
  h_Z = sum_j w_j(z, delta) * dirac_{z + Delta_j(z, delta)}, j = 1..d, from d learned
  query tokens. Exact composition + weighted per-particle entropy term.

The consistency objective (Plans A-D) is
    L = E_z[ D_KL( (h_S . g)(. | z, delta) || (g . h_Z)(. | z, delta) ) ]
        + lambda * E_{z' ~ h_Z}[ H(g(. | z')) ]                  (Plans B, D)
        + alpha * mean|z' - z| + beta * mean (z' - z)^2
with forward (mass-covering) KL over the dense |S| = prod(cardinalities) vector. The
penalties average over latent dimensions and over samples / components / particles.

Common scaffolding (config + persistence, the training loop, the latent-space penalties)
lives in `common.py`; each plan module supplies only its model body and its per-batch loss.
"""

from src.latent_intervention.base.baseline import LatentIntervention, train_latent_intervention
from src.latent_intervention.base.pre_additive import LatentInterventionPreAdditive, train_latent_intervention_preadditive
from src.latent_intervention.base.noise_token import LatentInterventionNoiseToken, train_latent_intervention_noise_token
from src.latent_intervention.base.dist import LatentInterventionDist, train_latent_intervention_dist
from src.latent_intervention.base.particles import LatentInterventionParticles, train_latent_intervention_particles
from src.latent_intervention.base.common import consistency_target, make_objective

__all__ = [
    "LatentIntervention",
    "train_latent_intervention",
    "LatentInterventionPreAdditive",
    "train_latent_intervention_preadditive",
    "LatentInterventionNoiseToken",
    "train_latent_intervention_noise_token",
    "LatentInterventionDist",
    "train_latent_intervention_dist",
    "LatentInterventionParticles",
    "train_latent_intervention_particles",
    "make_objective",
    "consistency_target",
]
