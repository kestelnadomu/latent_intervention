# Contribution and positioning

Notation as in [problem_formulation.md](problem_formulation.md).

## Contribution statement (this project: SCM counterfactuals)

> Given a frozen encoder $f$ and a semantic decoder $g$ that grounds its latents in a
> structured state space $\mathcal S$, we learn a latent editor $h_Z$ that reproduces an SCM's
> counterfactual $h_S$ (abduction–action–prediction) in latent space:
> $g \circ h_Z = h_S \circ g$ in distribution. Among the many latents that decode to the
> counterfactual state, it selects a minimal, on-manifold one. Training needs only factual
> latents and $h_S$, and no paired counterfactual data. At inference $h_Z$ needs only $z$ and
> $\delta$: no SCM and no counterfactual source input. The editor must *inject* the
> intervention's downstream (mediator) effects, because no computation in the representation
> propagates them.

The components that carry the claim:

- **High-level model = SCM of the data-generating process.** $h_S$ is a rung-3 counterfactual
  with exogenous noise, not an algorithm the network computes. Hence mediator injection:
  under $do(\texttt X)$, $h_Z$ must also move $\texttt D, \texttt U$. Swapping an
  $\texttt X$-subspace alone gives a controlled direct effect, not the total-effect
  counterfactual.
- **Source-free and amortised.** The SCM is internalised in $h_Z$'s weights.
- **Distributional.** Forward KL against $h_S \circ g$ covers $g$'s grounding uncertainty and
  multimodal counterfactuals (see [architecture/latent_intervention.md](architecture/latent_intervention.md)).
- **No paired training data.** The target $M_\delta^\top g(\cdot \mid z)$ needs factual latents
  only. Paired counterfactual texts serve evaluation (recovery of $Z'$) only.

## General form (next project: arbitrary symbolic operators)

Nothing in the construction requires $h_S$ to be an SCM counterfactual. The general statement:

> Given a frozen encoder $f$, a semantic decoder $g$, and any (possibly stochastic,
> non-invertible) symbolic operator $h_S: \mathcal S \to \Delta(\mathcal S)$, learn a latent
> module $h_Z$ that **lifts $h_S$ through $g$**, i.e. $g \circ h_Z = h_S \circ g$, choosing
> a minimal, on-manifold lift. SCM counterfactuals are one instance.

What changes when we generalise:

- **Mediator injection** becomes: the representation contains no computation implementing
  $h_S$, so the module *is* the implementation, not a patch that downstream circuitry
  propagates.
- **Lift selection** becomes the central identifiability question: every $z' \in g^{-1}(\mathbf s')$
  satisfies the square.
- **Non-invertibility** matters. $do(\cdot)$ is idempotent, so interventions form a monoid,
  not a group, and group-equivariant frameworks do not apply.
- **Evidence needs several $h_S$ instances**: e.g. a deterministic rule (attribute swap,
  ordinal increment), a stochastic non-causal kernel, and the SCM counterfactual. The kernel
  is pluggable via `objects.symbolic_kernel`, so this is mostly data work.

## Neighbouring work

| Line of work | Same | Different |
|---|---|---|
| Causal abstraction, interchange interventions, IIT (Geiger et al. 2021, 2022) | commuting square between a symbolic model and a neural representation | high-level model is an algorithm the network *computes*; the network's own layers do the propagating |
| DAS / Boundless DAS (Geiger et al. 2023; Wu et al. 2023) | learned intervention subspace, counterfactual objective | needs a **source input** at intervention time; point-wise (IIA), not distributional |
| ReFT / LoReFT (Wu et al. 2024) | source-free learned edit of a frozen representation | trained on task labels, not on a symbolic model's counterfactuals; candidate **parameterisation** for $h_Z$ on the residual stream |
| RAVEL (Huang et al. 2024) | evaluating targeted edits (Cause / Isolate) | evaluation benchmark, not a method; adopt its scores |
| Causal Proxy Models (Wu et al. 2022), CEBaB, LIBERTy | counterfactual text supervision via causal models | explain/mimic a black box; need counterfactual inputs |
| Deep SCM \cite{pawlowski_dscm_2020}, CausalVAE, VACA \cite{sanchezmartin_vaca_2022} | SCM counterfactuals in a learned latent space | SCM built *into* the representation; ours is post hoc on a frozen $f$ |
| MDP homomorphisms (van der Pol et al. 2020), neural algorithmic reasoning (Veličković et al.) | lifting an abstract transition/algorithm into latent space | encoder trained *jointly* so the square commutes; ours fits the operator into an existing representation |
| Equivariant / symmetry-based representations (Higgins et al. 2018) | latent operator mirroring a symbolic one | assumes invertible group actions |

The shared differentiator across the table: **post hoc** (frozen, pretrained $f$), **source-free**,
**distributional**, with **a symbolic model of the world** rather than of the network's computation.

## Constraints that follow from the positioning

- **$g$ is a probe, which is a weakness.** A probe can read information the model does not use,
  and it can be fooled by off-manifold edits. Defend with the VAE prior, an ensemble of $g$,
  the identity check ($\mathbf S' = \mathbf S \Rightarrow z' \approx z$), and $Z'$ recovery.
- **Restrict the editor.** With an unconstrained nonlinear map the square becomes trivially
  satisfiable (the *non-linear representation dilemma*, Sutter et al. 2025). Low rank
  (LoReFT-style), minimal shift, and the prior are what make a successful lift meaningful.
- **Beware interpretability illusions** (Makelov et al. 2023): high consistency may come from
  activating dormant pathways in $g$ rather than from moving $z$ to a real counterfactual.
- **Baselines**: a DAS-style subspace swap built from the paired counterfactual latents
  (needs a source, so it is an upper-bound baseline, not a competitor), and a constant
  direction $z' = z + v_\delta$.

## References to add to `references.bib`

Not yet in the bib; verify titles and venues before citing:
Geiger et al. 2021 (causal abstractions of neural networks), Geiger et al. 2022 (IIT),
Geiger, Wu, Potts, Icard & Goodman 2023 (DAS), Wu et al. 2023 (Boundless DAS),
Wu et al. 2024 (ReFT), Huang et al. 2024 (RAVEL), Wu et al. 2022 (Causal Proxy Models),
Makelov et al. 2023 (interpretability illusions), Sutter et al. 2025 (non-linear
representation dilemma), van der Pol et al. 2020 (MDP homomorphisms), Higgins et al. 2018
(definition of disentangled representations), Veličković et al. (neural algorithmic reasoning),
Yang et al. 2021 (CausalVAE).
