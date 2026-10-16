"""latent_lab: the read -> operate -> render pipeline, for notebooks.

    latents  = Encoder("nomic", 128).latents("talent")          # phi (cached encodings)
    reader   = Reader(["X", "T", "D", "U"]).fit(z, states)        # psi
    scm      = LinearSCM(levels, parents, noise)                   # a causal model of S
    operator = scm.do(X=3)                                        # omega_S: or any function
    renderer = FlowRenderer(["X", "T", "D", "U"]).fit(z, states)   # rho
    result   = Pipeline(reader, operator, renderer)(z_test)

Data loading and operators are defined in the notebook; see latent_lab/example.ipynb.
The full library (benchmarks, all h_Z families) is ``src/``.
"""

from latent_lab.encoder import ENCODERS, Encoder, Latents
from latent_lab.pipeline import Pipeline, Result, evaluate
from latent_lab.reader import Reader
from latent_lab.renderer import (
    RENDERERS,
    ConstantRenderer,
    FlowRenderer,
    GroupOTRenderer,
    IdentityRenderer,
    Renderer,
    StateEffectRenderer,
)
from latent_lab.scm import LinearSCM

__all__ = [
    "ENCODERS",
    "RENDERERS",
    "ConstantRenderer",
    "Encoder",
    "FlowRenderer",
    "GroupOTRenderer",
    "IdentityRenderer",
    "Latents",
    "LinearSCM",
    "Pipeline",
    "Reader",
    "Renderer",
    "Result",
    "StateEffectRenderer",
    "evaluate",
]
