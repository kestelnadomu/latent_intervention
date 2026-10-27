"""latent_lab: modular latent operation (MLO), read -> operate -> render, for notebooks.

    latents  = Encoder("nomic", 128).latents("talent")          # phi (cached encodings)
    reader   = Reader(["X", "T", "D", "U"]).fit(z, states)        # psi
    operator = LinearSCM(levels, parents, noise).do(X=3)          # omega_S: or any function
    renderer = FlowRenderer(["X", "T", "D", "U"]).fit(z, states)   # rho
    result   = Pipeline(reader, operator, renderer)(z_test)   # omega_Z, the MLO

Data loading and operators are defined in the notebook; see experiment.ipynb in the repo root.
"""

from latent_lab.encoder import ENCODERS, Encoder, Latents
from latent_lab.pipeline import Pipeline, Result, evaluate, evaluate_samples, factual_score
from latent_lab.reader import OutOfFold, Reader, cross_fit, mean_predict, mean_proba, mean_sample, state_metrics
from latent_lab.renderer import (
    RENDERERS,
    ConstantRenderer,
    EditorRenderer,
    FlowRenderer,
    GroupOTRenderer,
    IdentityRenderer,
    OTEditorRenderer,
    OTFlowRenderer,
    Renderer,
    StateEffectRenderer,
)
from latent_lab.scm import LinearSCM

__all__ = [
    "ENCODERS",
    "RENDERERS",
    "ConstantRenderer",
    "EditorRenderer",
    "Encoder",
    "FlowRenderer",
    "GroupOTRenderer",
    "IdentityRenderer",
    "Latents",
    "LinearSCM",
    "OTEditorRenderer",
    "OTFlowRenderer",
    "OutOfFold",
    "Pipeline",
    "Reader",
    "Renderer",
    "Result",
    "StateEffectRenderer",
    "cross_fit",
    "evaluate",
    "evaluate_samples",
    "factual_score",
    "mean_predict",
    "mean_proba",
    "mean_sample",
    "state_metrics",
]
