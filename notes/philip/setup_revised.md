# Revised normalizing-flow setup for $h_Z$

Status: implementation note. This records the agreed target, terminology, and first
implementation; the statistical assumptions still require empirical evaluation.

## What changed

The earlier term **LIBERTy-free** is misleading. We may still use LIBERTy data to
benchmark the methods, especially its observed structured state $S$. The actual restriction is:

> Neither normalizing-flow method may use the true counterfactual latent $Z'$ during training.

Access to $S$ is a separate experimental choice. In an **oracle-$S$ benchmark**, the observed
factual state is supplied so that uncertainty and errors from the semantic decoder $g(S\mid Z)$
do not obscure the quality of the latent intervention method itself. An **inferred-$S$ setting**
may later use $g$, but this is a different information regime rather than a different flow
objective.

The two previously described supervised flows that fit observed pairs containing $Z'$ are no
longer part of the target setup. True $Z'$ should be reserved for held-out evaluation, subject to
final confirmation below.

The implementation should remain schema-generic. In the active talent experiment,
$Z\in\mathbb R^{128}$ and $S=(X,T,D,U)$ has cardinalities $(4,3,3,3)$ and 108 joint states;
the older eight-column/3,456-state example is not the active schema.

## Declared target

We want three separately usable normalizing-flow approaches:

1. a **state-conditioned counterfactual flow** that uses $S$ and a symbolic transition to
   construct a counterfactual latent while preserving inferred unit-level flow noise; and
2. a **direct intervention-conditioned flow** that needs only $(Z,\delta)$ at inference and is
   distilled from the first approach without observing true $Z'$; and
3. a **direct semantic flow** with the same $(Z,\delta)$ inference interface, trained from
   frozen $g$ and $h_S$ without using the state flow as a teacher.

The first setup provides a structured benchmark in which the contribution of $h_Z$ can be
studied without conflating it with estimation error in $g$. The second and third provide the
intended deployment interface for users who have an encoded text and an intervention but no
semantic state or semantic decoder at inference time. Comparing them isolates the value of the
state-conditioned teacher from the value of the flow architecture itself.

## Setup A: state-conditioned counterfactual flow

### Factual flow

Train a conditional bijection $F_\theta$ only on factual training pairs $(S_i,Z_i)$:

$$
Z = F_\theta(S,U),
\qquad U\sim\mathcal N(0,I_{128}),
$$

where $F_\theta(S,\cdot)$ is invertible for every fixed $S$. For
$u_i=F_\theta^{-1}(Z_i;S_i)$, training uses the ordinary conditional-flow likelihood

$$
\mathcal L_{\mathrm{state}}
=-
\sum_i
\left[
\log p_U(u_i)
+
\log\left|
\det\frac{\partial F_\theta^{-1}(Z_i;S_i)}{\partial Z_i}
\right|
\right].
$$

No factual/counterfactual latent pair $(Z_i,Z'_i)$ enters this loss.

### Counterfactual transport

For a factual unit $(Z_i,S_i)$ and intervention $\delta$:

$$
u_i=F_\theta^{-1}(Z_i;S_i),
\qquad
S'_i\sim h_S(\cdot\mid S_i,\delta),
\qquad
\widehat Z'_i=F_\theta(S'_i,u_i).
$$

The same abducted noise $u_i$ is kept across worlds. Counterfactual randomness comes from the
symbolic transition when $h_S$ is stochastic, not from replacing the unit's inferred noise.

Information regimes:

- **Oracle-$S$ benchmark:** use observed factual $S$ and the known $h_S$. This deliberately
  removes $g$'s estimation uncertainty from the comparison.
- **Inferred-$S$ setting:** obtain or sample $S$ from $g(S\mid Z)$ before applying $h_S$. This
  measures the combined semantic-decoding and latent-transport pipeline and must be reported
  separately.

Setup A requires a structured state at inference. Its purpose is both to provide the structured
flow method and to act as the teacher for Setup B.

## Setup B: direct intervention-conditioned flow

Setup B must not require $S$, $S'$, $g$, or $h_S$ at inference. Its public inference interface is

$$
\widehat Z'=D_\phi(Z,\delta,E),
\qquad E\sim\mathcal N(0,I),
$$

where $D_\phi(Z,\delta,\cdot)$ is a conditional normalizing flow. For a fixed $(Z,\delta)$ it
represents a distribution over counterfactual latents.

Because true $Z'$ is unavailable during training, train this model by distillation. For each
factual training unit, Setup A produces pseudo-targets:

$$
S_i \text{ is observed or inferred},
\qquad
S'_i\sim h_S(\cdot\mid S_i,\delta),
$$

$$
u_i=F_\theta^{-1}(Z_i;S_i),
\qquad
\widetilde Z'_i=F_\theta(S'_i,u_i).
$$

The direct flow is fitted by a multi-sample energy distance between teacher and student draws.
This is deliberate: for fixed $(Z_i,S_i,\delta)$, the teacher distribution is a finite mixture
over target states, so unconstrained continuous-flow likelihood can collapse its scale. Multiple
symbolic transitions and pseudo-targets are drawn in memory for each factual unit.

This student removes semantic inputs only at inference. Its training still inherits the factual
flow, $h_S$, the chosen source of $S$, and all associated assumptions. It therefore approximates
the teacher-induced counterfactual distribution; it is not directly supervised by the true
counterfactual latent.

## Setup C: direct semantic flow

The third model has the same standalone inference interface as Setup B,

$$
\widehat Z'=D_\psi(Z,\delta,E),
$$

but receives no explicit $S$ or $S'$ and is not trained on state-flow pseudo-targets. During
training only, frozen $g$ and $h_S$ define the semantic target $(h_S\circ g)(\cdot\mid Z,\delta)$.
The loss compares this target with the average of $g(\cdot\mid\widehat Z')$ over several flow
draws, with small entropy, proximity, identity, and factual-support regularisers. This objective
identifies semantic agreement under $g$, not the unique true counterfactual latent; therefore it
is a direct baseline rather than an identified estimator.

## Terminology

Avoid **LIBERTy-free** for both methods. Working names:

| role | preferred working name | inference inputs |
|---|---|---|
| Setup A | state-conditioned counterfactual flow | $Z,S,\delta$ (and $h_S$) |
| Setup B | direct intervention-conditioned flow / distilled flow | $Z,\delta$ |
| Setup C | direct semantic flow | $Z,\delta$ |

Use **oracle-$S$** and **inferred-$S$** for the benchmark regimes. Reserve $Z'$ for the real,
unavailable counterfactual latent, $\widehat Z'$ for a model output, and $\widetilde Z'$ for a
teacher-generated pseudo-target. The teacher $F_\theta$ and student $D_\phi$ should have distinct
names and checkpoints rather than both being called $h_Z$ internally.

## Shared assumptions and boundaries

- The encoder is frozen and produces 128-dimensional factual latents.
- Only official training units may fit the factual flow or create student pseudo-targets.
- True $Z'$ is never a training target for any of the three flows.
- Setup A relies on an invariant, bijective conditional generation mechanism between $S$, $U$,
  and $Z$. A normalizing-flow architecture enables this assumption but does not prove it.
- Setup B inherits Setup A's assumptions and any uncertainty or misspecification in $h_S$ and
  the source of $S$.
- Setup C inherits errors in $g$ and $h_S$ and is constrained only in directions visible to $g$.
- The existing transformer-based latent-intervention variants remain separate baselines. Their
  modular configuration, artifact, split, and evaluation conventions should be reused where
  suitable, but this note does not yet prescribe a code integration.

## Decisions for the first implementation

- Use a torch-only eight-block affine-coupling flow with fixed permutations and bounded scales.
- Train Setup A by factual conditional likelihood only; no proximity or semantic loss selects
  the cross-state alignment.
- Default Setup A to sampling $S'$ through $h_S$, while retaining explicit $S'$ as a labelled
  oracle-target evaluation mode.
- Train Setup B with an energy distance on eight teacher and eight student samples; cache and reuse
  a compatible state-flow teacher.
- Train Setup C with semantic forward KL after averaging probabilities across eight samples.
- Train only the configured intervention plus an empty identity action. Schema-valid unseen
  interventions remain callable but must emit a warning and carry no generalisation claim.
- Use true paired $Z'$ only on official-test IDs for evaluation.
- Use the same semantic decoder for training and semantic evaluation; consequently paired-$Z'$
  recovery and latent-support diagnostics are the primary checks against decoder gaming.

## Deferred questions

1. Which experiments should use observed $S$, sampled $S\sim g(\cdot\mid Z)$, or both? The
   oracle-$S$ and inferred-$S$ results must not be mixed.
2. Is the estimand an individual counterfactual obtained by preserving abducted $u$, or the full
   conditional distribution $p(Z'\mid Z,\delta)$? What output should Setup B expose: one sample,
   several samples, log density, or a deterministic representative?
3. Should later experiments replace warning-only extrapolation with an explicit multi-intervention
   training curriculum?
