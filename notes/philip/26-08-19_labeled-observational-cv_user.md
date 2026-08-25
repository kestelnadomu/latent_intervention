# Action brief: labeled factual CVs without counterfactual pairs

> **Setup B — keep separate from the LIBERTy full-orbit experiment.**  
> Detailed handoff: [agent note](./26-08-19_labeled-observational-cv_agent.md)  
> Setup A: [LIBERTy-orbit user brief](./26-08-07_stochastic-kernel_user.md)

## Decision

Human labels make one part of the pipeline straightforward:

$$
X\longrightarrow Z^a\longrightarrow \mathbf S^a.
$$

They do **not** show what the same person would look like under the requested intervention. Two bridges are still missing:

$$
\mathbf S^a\dashrightarrow\mathbf S^b
\qquad\text{and}\qquad
(Z^a,\mathbf S^a,\mathbf S^b)\dashrightarrow Z^b.
$$

Therefore a true same-person $Q_e$ or direct $H_e$ is **not identifiable from factual $(X,\mathbf S^a)$ data alone**. In this project, $K_S$ is instead identified *relative to a prespecified SCM family*: we assume the DAG, structural-equation/noise family, and conditions needed to estimate its parameters. No loss function can create the still-unobserved latent pairing.

## What is actually trainable?

| Component | Training input | Observed target and loss | Status |
|---|---|---|---|
| Encoder $e$ | CV $X_i$ | None; freeze and cache $Z_i=e(X_i)$ | Available |
| Semantic model $G_e$ | $Z_i$, source value/context | Human $\mathbf S_i^a$; cross-entropy or another proper supervised score | **Identified and trainable** |
| Structured transition $K_S$ | For each node in the known DAG: its observed parents and regime | Observed node value; nodewise likelihood, cross-entropy, or MSE | **Assumed estimable** in the prespecified SCM/noise family; abduction–action–prediction then induces $K_S$ |
| Target-conditioned editor $Q_e$ | Paired $(Z_i^a,\mathbf S_i^a,\mathbf S_i^b)$, or factual $(Z_i,\mathbf S_i,C_i)$ for a structural renderer | Paired $Z_i^b$ loss if available; otherwise factual renderer likelihood/reconstruction | Without pairs, a shared-noise renderer can induce only a **model-implied** $Q_e$ |
| Direct editor $H_e$ | $(Z_i^a,a,b)$ | Paired $Z_i^b$, or pseudo-targets from the structural teacher; proper paired/distillation score | **Optional**; useful only when one-step symbolic-free inference is required |

### Assumption adopted for the structured transition

We assume the DAG is known and that a correctly specified, identifiable structural-equation/noise family has been chosen. We fit each node mechanism to factual parent–child observations, then calculate $K_S$ by **abduction–action–prediction**: infer the factual exogenous variables, intervene, and reuse those variables in the target world. Thus $\mathbf S_i^b$ is not a supervised label. The DAG alone would not determine this individual cross-world coupling; it is supplied by the SCM family and its shared-noise semantics.

### $Q_e$ and $H_e$ are interfaces, not necessarily two networks

- $Q_e$ receives the requested target state: $(Z^a,\mathbf S^a,\mathbf S^b,a,b)$.
- $H_e$ receives only $(Z^a,a,b)$ and internally averages over or imitates the symbolic route.

If the original one-model editor receives only $(Z^a,a,b)$, it is already $H_e$ in the present notation, even if it was called $Q_e$ before; no second network is required. A modular $Q_e$ can instead be retained for explicit symbolic control, with an optional $H_e$ student for symbolic-free deployment. Both modes may share one backbone, but each desired input contract needs an appropriate training objective.

If verified $\mathbf S^a$ labels are also supplied for a new CV, use them directly at inference and bypass $G_e$. The semantic model is required only when future CVs arrive without labels.

## Why target-group examples are not enough

People observed with the target attribute tell us what the **target population** looks like. They do not tell us which target CV belongs to the same factual person.

For example, two binary worlds may both be 50/50. “Every person stays the same” and “every person flips” produce the same two marginals but opposite individual transitions. The missing matching is the causal information.

The same issue exists in latent space. Requiring $G_e(\widetilde Z^b)=\mathbf S^b$ finds an embedding with the right labels, but many embeddings have those labels. Minimal distance, cycle consistency, optimal transport, or a teacher chooses one matching; it does not prove that matching is the person's counterfactual. Hence the frozen-$G_e$ constraint is a useful semantic auxiliary, but not the loss that identifies $Q_e$ or $H_e$.

## Three defensible ways forward

### 1. Collect controlled paired rewrites — recommended for individual claims

Define the target state with a defensible SCM or explicit edit policy, then create one or more audited rewrites of each source CV while preserving declared invariants:

$$
(X_i^a,\mathbf S_i^a)
\longrightarrow
(X_i^b,\mathbf S_i^b)
\longrightarrow
(Z_i^a,Z_i^b).
$$

Now $Q_e$ and direct $H_e$ can use a proper paired loss against the observed rewrite embedding $Z_i^b$, exactly as in the LIBERTy setup. The claim is relative to the human/LLM editing protocol, which must be documented and validated.

### 2. Declare a causal model and an invariant-residual model

An implementable factual-only baseline is

$$
Z=m(\mathbf S,C)+U_Z,
$$

where $C$ is observed invariant context and $U_Z$ is **assumed** to represent the same person's residual information across worlds.

Training uses factual $(\mathbf S_i,C_i,Z_i)$ and MSE or likelihood. At inference:

1. obtain $\widetilde{\mathbf S}^b$ from the declared causal model;
2. infer $u_i=z_i-m(\mathbf S_i^a,C_i)$;
3. output $\widetilde z_i^b=m(\widetilde{\mathbf S}_i^b,C_i)+u_i$.

This is trainable, but the shared residual is a cross-world assumption, not something the factual data prove. Results must be called **model-implied counterfactuals** and compared across several plausible residual/coupling choices.

#### A more flexible option: a bijective generation mechanism

A BGM posits

$$
Z=\Phi_e(\mathbf S,C,U_Z),
\qquad \Phi_e(\mathbf s,c,\cdot)\text{ bijective},
$$

and assumes the same person-specific $U_Z$ persists across worlds. Train $\Phi_e$ only on factual triples $(\mathbf S_i,C_i,Z_i)$ using conditional likelihood (for example, a structured conditional flow). At inference,

$$
\widehat u_i=\Phi_e^{-1}(\mathbf S_i^a,C_i,Z_i^a),
\qquad
\widetilde Z_i^b=\Phi_e(\mathbf S_i^b,C_i,\widehat u_i).
$$

This directly induces a model-implied $Q_e$; a separate $Q_e$ network is unnecessary unless we want amortization or a shared editor architecture. With exact inputs and an exact bijection it is deterministic; uncertainty in $\mathbf S^b$, missing context, parameters, or residuals makes the resulting kernel stochastic.

However, **bijectivity and a known DAG are not enough**. [Nasr-Esfahany, Alizadeh, and Shah (2023)](https://proceedings.mlr.press/v202/nasr-esfahany23a.html) require additional monotonicity/independence, valid-IV, or strong backdoor-variability conditions, depending on the setting. Our encoder latent is 128-dimensional, so their simple scalar theorem does not apply; the relevant vector result needs a defensible adjustment variable and strong variability conditions. A BGM is therefore a reasonable explicit renderer and sensitivity baseline, but should be called **BGM-implied** unless those conditions are justified.

### 3. Target a population interventional analogue

If same-person preservation is unnecessary, estimate a target distribution conditional on measured pre-treatment context:

$$
\mathbf S^b\sim P(\mathbf S(b)\mid C_{\mathrm{pre}}),
\qquad
\widetilde Z^b\sim P_{\mathrm{obs}}(Z\mid\mathbf S^b,C_{\mathrm{pre}},T=b).
$$

This needs a well-defined intervention, no unmeasured confounding, positivity, and stable mechanisms. It produces a plausible target-population CV embedding, not the same person's latent counterfactual.

If none of these assumptions is defensible, report bounds/sensitivity across possible couplings or focus on a robust downstream predictor instead of claiming recovery of $Z^b$.

## Recommended adapted training

1. **Ingest and split factual data by person/source.** Store the CV, human labels, intervention value, pre-treatment context, label provenance, and a persistent split.

2. **Freeze the encoder.** Input $X_i$, output $Z_i$, no loss.

3. **Train $G_e$.** Input $(Z_i,T_i,C_i)$, target human $\mathbf S_i^a$, proper supervised loss. This is the only bridge directly supervised without a structural cross-world assumption.

4. **Fit the prespecified structured SCM.** Use the known DAG and factual parent–child tuples to estimate the declared node mechanisms/noise laws with nodewise proper losses. Then obtain $K_S$ by abduction–action–prediction. Audit fit, support, causal closure, and sensitivity; there is still no supervised $\mathbf S_i^b$ loss.

5. **Choose how $Z^b$ is defined.** Prefer paired rewrites. Otherwise fit an additive/location-scale renderer or a structured BGM to factual $(\mathbf S,C,Z)$ and explicitly state the shared-noise coupling. A semantic-only editor remains a baseline, not the identified target method.

6. **Optionally train a direct mode $H_e$.** If symbolic-free inference is required, train the original $(Z^a,a,b)\mapsto Z^b$ interface on real paired targets or on pseudo-targets from the $K_S$ plus latent-renderer teacher. It may share a backbone with $Q_e$; “separate” means a separate input-conditioned objective, not necessarily a second network. Distillation adds no new identification.

7. **Evaluate honestly.** $G_e$ can be evaluated against held-out human labels. Without paired targets, evaluate causal-model sensitivity, support, target semantics with an independent check, invariant retention, plausibility, and downstream utility—but do not report same-person counterfactual accuracy or coverage.

## Inference

When factual labels are supplied:

$$
z=e(x),
\qquad
\widetilde{\mathbf S}^b_m\sim\widehat K_S(\cdot\mid\mathbf S^a,a,b,C),
\qquad
\widetilde Z^b_m\sim\text{selected realizer}(z,\mathbf S^a,\widetilde{\mathbf S}^b_m,C).
$$

When labels are unavailable, first sample

$$
\widetilde{\mathbf S}^a_m\sim\widehat G_e(\cdot\mid z,a,C)
$$

and propagate that uncertainty. A distilled direct student instead takes only $(z,a,b)$, but its outputs remain teacher/model-implied unless trained on genuine pairs.

Control and record:

- the exact intervention and allowed causal paths;
- confounders, pre-treatment context, and declared invariants;
- whether labels are observed or predicted;
- the SCM and residual/transport assumption;
- encoder checkpoint and preprocessing;
- target-state support and overlap;
- target provenance: paired, model-implied, population-resampled, or semantic edit.

## Actionable recommendation

If the paper wants to sell **recovery of an individual latent counterfactual**, collect a small but rigorous paired-rewrite dataset and preferably use it for both training and independent testing.

If that is impossible, keep Setup A as the identification experiment and position Setup B explicitly as either:

- a model-implied real-data extension with SCM and shared-residual sensitivity analysis; or
- a weaker population interventional application.

The next decision is therefore not the neural architecture. It is which of those estimands the real-data study is allowed to claim.
