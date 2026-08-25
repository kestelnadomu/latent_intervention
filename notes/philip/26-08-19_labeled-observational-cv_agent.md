# Agent handoff: labeled factual CVs without counterfactual pairs

> **Setup B — observational, human-labeled CVs.** This is deliberately separate from the LIBERTy full-orbit setup.
>
> Companion: [short user brief](./26-08-19_labeled-observational-cv_user.md)  
> Setup A: [LIBERTy-orbit agent handoff](./26-08-07_stochastic-kernel_agent.md) and [user brief](./26-08-07_stochastic-kernel_user.md)

## 0. Executive verdict

The available data are factual CVs with human-assigned factual structured labels. They make the semantic bridge and the factual node mechanisms of a prespecified SCM supervised, but they do **not** directly reveal either cross-world bridge:

1. the causal bridge from the observed structured state to the same unit's state under the requested intervention;
2. the realization bridge from the observed CV embedding to the same unit's counterfactual CV embedding.

Consequently, from factual records alone:

| Object | Direct training target available? | Identified from the observed data? | Correct interpretation |
|---|---:|---:|---|
| Frozen encoder $e:X\mapsto Z$ | No target needed | Yes, as a fixed transformation | Cache $Z_i=e(X_i)$ |
| Semantic kernel $G_e(d\mathbf s\mid z,t)$ | **Yes:** human $\mathbf S_i$ | **Yes**, on factual support | Ordinary supervised semantic analysis |
| Structured counterfactual kernel $K_S^{a\to b}(d\mathbf s^b\mid \mathbf s^a)$ | No paired $\mathbf S_i^b$; **yes** for factual node values given parents | **Yes only relative to the adopted identifiable SCM/noise family** | Fit factual node mechanisms, then calculate $K_S$ by abduction–action–prediction |
| Target-conditioned latent kernel $Q_e^{a\to b}(dz^b\mid z^a,\mathbf s^a,\mathbf s^b)$ | No paired $Z_i^b$; **yes** for factual $Z_i$ in a structural renderer | **No nonparametrically**; model-implied under an explicit cross-world renderer/residual assumption | Paired rewrites, or a declared shared-noise model such as an additive renderer or suitably restricted BGM |
| Direct kernel $H_e^{\mathrm{dir},a\to b}(dz^b\mid z^a)$ | **No:** $Z_i^b$ is absent | **No nonparametrically** | Optional symbolic-free interface; train on pairs or distill the assumption-defined modular teacher |

This is an information limitation, not an architecture limitation. A semantic loss, cycle loss, optimal-transport cost, or teacher can choose a coupling, but none can discover the unobserved coupling from the same factual distribution.

### Project assumption adopted for $K_S$

For this study, treat the structured DAG as known and prespecify a correctly specified structural-equation/noise family whose parameters are identifiable from the factual records. Assume that the observed state contains the required causal variables, relevant confounding is handled as declared, positivity holds for the requested states, and mechanisms are stable. Estimate the node mechanisms on factual parent–child tuples and compute individual transitions by abduction–action–prediction with the fitted shared exogenous variables.

This makes $K_S$ **SCM-identified/model-relative**, not pair-supervised. A known DAG alone would generally identify neither the structural equations nor the same-unit cross-world noise coupling; those are supplied by the adopted SCM family. This distinction must remain explicit in claims and evaluation.

The recommended scientific positioning is:

- keep Setup A as the controlled setting in which complete paired orbits identify simulator-relative $K_S,Q_e,H_e$;
- treat Setup B as a separate real-data extension;
- if individual latent counterfactuals are essential, acquire controlled paired rewrites;
- if pairs cannot be acquired, choose and name either a **model-implied individual edit**, a **population interventional analogue**, a **controlled semantic edit**, or a **partially identified target**. Do not call any of these an empirically identified same-person latent counterfactual without the corresponding assumptions.

## 1. Exact problem and notation

### 1.1 Observed data

Let $T$ denote the sensitive/intervention variable and let $a,b$ denote source and requested target values. Use $T$, not $A$, in new code and notes because `A` already means age in the repository's CV SCM.

The factual dataset is

\[
\mathcal D_{\mathrm{obs}}
=
\{(i,X_i,T_i,\mathbf S_i,C_i)\}_{i=1}^n,
\]

where:

- $X_i$ is the observed CV;
- $\mathbf S_i$ is the human-labeled structured factual state;
- $C_i$ is optional observed pre-treatment or intervention-invariant context;
- $Z_i=e(X_i)$ is produced by one frozen encoder.

The human labels are to be treated as ground truth for the coordinates they actually annotate. If the annotation covers only a subset of the causal state, call it $L_i$, not $\mathbf S_i$. Writing $L_i=\mathbf S_i$ requires the substantive assumption that the labels contain the counterfactual closure of the query: the intervention variable, relevant parents/confounders, affected descendants or mediators, and declared invariants.

The data do **not** contain matched

\[
\mathbf S_i(b),\qquad X_i(b),\qquad Z_i(b)=e(X_i(b))
\]

for the same person under $do(T=b)$.

### 1.2 Observed is not automatically interventional

Potential-outcome notation is

\[
\mathbf S_i(t),\quad X_i(t),\quad Z_i(t)=e(X_i(t)).
\]

Consistency says that, for a unit observed with $T_i=a$,

\[
(\mathbf S_i,X_i,Z_i)
=
(\mathbf S_i(a),X_i(a),Z_i(a)).
\]

The natural observation is nevertheless not a sample from a randomized $do(T=a)$ regime. Observing people with $T=0$ and $T=1$ gives two observational groups; it does not by itself give paired potential outcomes or remove confounding.

For clarity, use $Z^{\mathrm{obs}},\mathbf S^{\mathrm{obs}}$ when discussing raw input data. The shorter $Z^a,\mathbf S^a$ is acceptable only after stating consistency and setting $a=T_i$.

### 1.3 Deployment variants must be distinguished

There are two different runtime contracts:

1. **Labels available at inference.** The user supplies $(x,\mathbf s^{\mathrm{obs}},a,b)$. Use the ground-truth $\mathbf s^{\mathrm{obs}}$ directly and bypass $G_e$.
2. **Labels available only during training.** The user supplies $(x,a,b)$. Estimate or sample $\mathbf S^{\mathrm{obs}}\sim G_e(\cdot\mid e(x),a)$ and propagate its uncertainty.

Do not force every deployment through $G_e$ when verified labels are already available.

The symbols $Q_e$ and $H_e$ name different **input contracts**, not necessarily different neural networks:

- $Q_e$ is target-conditioned and receives $(Z^a,\mathbf S^a,\mathbf S^b,a,b,C)$;
- $H_e^{\mathrm{dir}}$ receives only $(Z^a,a,b)$ and must internally marginalize over, or imitate, the symbolic route.

Therefore the original proposal's one-step editor with input $(Z^a,a,b)$ is already $H_e^{\mathrm{dir}}$ in this handoff, even if it was originally named $Q_e$. It does not need a second direct model. Conversely, when the method retains an explicit $K_S\to Q_e$ pipeline for state-level control, a separate **direct training mode or student objective** is needed only if deployment must avoid $\mathbf S^a,\mathbf S^b$. The two modes may share a backbone and parameters.

## 2. Formal identification boundary

### 2.1 What the observed law identifies

For a fixed encoder, the data identify the factual joint law

\[
P_{\mathrm{obs}}(dz,d\mathbf s,dt,dc)
\]

and therefore regular conditional laws such as

\[
G_e^*(d\mathbf s\mid z,t,c)
=
P_{\mathrm{obs}}(d\mathbf s\mid Z=z,T=t,C=c)
\]

almost surely on observed support. A sufficiently well-specified model fitted with a proper supervised score can consistently estimate this target under ordinary sampling, regularization, and optimization conditions.

The data can also identify descriptive conditionals such as $P_{\mathrm{obs}}(Z\mid \mathbf S,C,T)$. A descriptive conditional is not automatically a counterfactual conditional.

### 2.2 Short nonidentification argument

Let $L$ be **any** kernel for the missing variables:

\[
L(d\mathbf s^b,dz^b\mid z,\mathbf s,t,c,b).
\]

Then

\[
P_{\mathrm{obs}}(dz,d\mathbf s,dt,dc)
L(d\mathbf s^b,dz^b\mid z,\mathbf s,t,c,b)
\]

has exactly the same distribution for every observed record, regardless of the choice of $L$. Two different choices of $L$ can imply different $K_S,Q_e,$ and $H_e$, yet fit the factual data equally well. Thus those kernels are not functionals of the observed law without additional restrictions.

An intuitive binary example is enough. Suppose the factual and target marginals of a binary variable are both 50/50. The coupling “everyone stays the same” and the coupling “everyone flips” have identical marginals but opposite individual counterfactuals. Unpaired groups—even randomized groups—do not reveal which coupling applies to a person.

### 2.3 The semantic-fiber problem remains

Even if an external model provides a target state $\mathbf s^b$, the constraint

\[
G_e(\mathbf s^b\mid \widetilde z^b)\text{ is large}
\]

only places $\widetilde z^b$ in a target semantic fiber. Many embeddings can express the same labeled state while differing in person, wording, style, omitted attributes, and downstream information. $L_1/L_2$ proximity, cycle consistency, adversarial marginal matching, and optimal transport select one convention within or between fibers; they do not reveal the missing same-person target.

This is precisely why the paired $Z^b$ objective in Setup A cannot simply be copied into Setup B: its target does not exist in the factual dataset.

### 2.4 Trainability is not identification

A model can always be optimized against a self-generated or regularized objective. That does not make its output data-identified.

- A teacher converts assumptions into pseudo-labels; it cannot manufacture counterfactual ground truth.
- Agreement between a distilled student and its own teacher measures fidelity, not causal validity.
- Matching the target group's latent distribution identifies a marginal fit, not the individual pairing.
- Applying the same $G_e$ for both training and evaluation can reward probe gaming and is not independent semantic validation.

## 3. Four defensible routes

The estimand must be selected before implementation. These routes should not be mixed in claims or tables.

### 3.1 Route B1 — controlled paired rewrites

**Recommended when the paper needs an individual latent counterfactual claim.**

For each source CV:

1. obtain $\mathbf S_i^b$ from a defensible causal model or a precisely declared controlled-edit policy;
2. ask a human editor, or an LLM followed by independent human validation, to produce $X_i^b$ while preserving declared invariants;
3. preferably collect multiple admissible rewrites to represent renderer variation;
4. encode $Z_i^b=e(X_i^b)$.

This creates training tuples

\[
(Z_i^a,\mathbf S_i^a,\mathbf S_i^b,Z_i^b,a,b).
\]

The paired $Q_e$ and direct $H_e$ objectives from Setup A then become available. The resulting law is identified relative to the declared editing protocol and population, not as a metaphysical real-life counterfactual. That narrower claim is still substantially stronger than a semantic-only edit.

Use a held-out editor or rewrite protocol where possible. If all targets are generated by one LLM, the learned coupling is that LLM's editing policy.

### 3.2 Route B2 — model-implied individual counterfactual

Specify both a functional causal model for the structured state and a structural model for the frozen embedding. One explicit formulation is

\[
\mathbf S^r=F_r(C,U_S),
\qquad
Z^r=\Phi_e(\mathbf S^r,C,U_Z),
\]

with invariant $C$ and the same $(U_S,U_Z)$ reused across regimes.

Then

\[
K_S^{a\to b}(d\mathbf s'\mid \mathbf s,c)
=
\int \delta_{F_b(c,u)}(d\mathbf s')
P(du\mid \mathbf S^a=\mathbf s,C=c),
\]

and

\[
Q_e^{a\to b}(dz'\mid z,\mathbf s,\mathbf s',c)
=
\int \delta_{\Phi_e(\mathbf s',c,v)}(dz')
P(dv\mid Z^a=z,\mathbf S^a=\mathbf s,C=c).
\]

This makes the complete pipeline implementable, but only **relative to the declared shared-noise structural model**. Factual data do not identify the cross-regime reuse of the same noise.

A transparent first latent model is additive residual transport:

\[
Z=m_\theta(\mathbf S,C)+U_Z.
\]

Fit $m_\theta$ on factual records, abduct

\[
u_i=z_i-m_\theta(\mathbf s_i,c_i),
\]

and define

\[
\widetilde z_i^b
=
m_\theta(\widetilde{\mathbf s}_i^b,c_i)+u_i.
\]

A location-scale version uses

\[
Z=\mu_\theta(\mathbf S,C)+L_\theta(\mathbf S,C)U_Z
\]

and preserves the standardized residual. A conditional invertible flow can implement a more flexible $\Phi_e$: invert $U_Z=\Phi_e^{-1}(Z^a;\mathbf S^a,C)$, replace the state, and decode with the same $U_Z$.

#### BGM specialization

A **bijective generation mechanism** (BGM) is a principled way to make this missing coupling explicit. Specify

\[
Z=\Phi_{e,\theta}(\mathbf S,C,U_Z),
\qquad
u\mapsto\Phi_{e,\theta}(\mathbf s,c,u)
\text{ bijective for every }(\mathbf s,c).
\]

Train only on factual rows $(\mathbf S_i,C_i,Z_i)$ by conditional density estimation, for example

\[
\mathcal L_{\mathrm{BGM}}(\theta)
=
-\sum_i\log p_\theta(z_i\mid\mathbf s_i,c_i),
\]

with the independence, graph, and transformation restrictions required by the chosen identification theorem built into the structural generator. An ordinary unconstrained conditional flow fitted by likelihood is not enough. At inference,

\[
\widehat u_i
=
\Phi_{e,\theta}^{-1}(\mathbf s_i^a,c_i,z_i^a),
\qquad
\widetilde z_i^b
=
\Phi_{e,\theta}(\mathbf s_i^b,c_i,\widehat u_i).
\]

Thus the BGM itself is the realizer of $Q_e$; there is no need to train another $Q_e$ network against an unavailable $Z_i^b$ unless amortization, uncertainty modeling, or a common editor architecture is desired. Conditional on exact $(z^a,\mathbf s^a,\mathbf s^b,c)$ and an exact complete bijection, the induced $Q_e$ is a Dirac kernel. It becomes stochastic after integrating uncertainty in the target state, context, parameters, measurement, or residual posterior.

This construction is **not identified by bijectivity plus observational likelihood alone**. Nasr-Esfahany, Alizadeh, and Shah (ICML 2023) obtain counterfactual identification only under one of several additional regimes:

1. a scalar Markovian setting with exogenous-parent independence and a common strict monotonic orientation;
2. a scalar instrumental-variable setting with IV validity, positivity, differentiability/monotonicity, and a rank/variability condition; or
3. a multidimensional backdoor setting with a valid observed adjustment variable, the required conditional independence, differentiable Jacobians, and a strong full-rank variability condition across at least $d+1$ contexts.

The repository currently uses the `l128` LangVAE encoder (`src/config.yaml`) and the intervention code assumes a 128-dimensional latent in its smoke test (`src/latent_intervention.py`). The easy scalar theorem therefore does not apply. A vector backdoor result is a possible route in principle, but it needs a scientifically defensible $C$ and a demanding variability condition; human $\mathbf S$ labels and knowledge of the DAG do not supply these automatically. A frozen deterministic encoder may also concentrate embeddings on a lower-dimensional support, which needs care in any continuous-flow likelihood.

Accordingly, position the BGM as the main **model-implied renderer and sensitivity model** unless the exact theorem conditions can be mapped to the CV process and defended. Compare it with additive, location-scale, and other admissible couplings, and validate against paired rewrites if the paper makes individual-counterfactual claims.

The caveat is fundamental. State-dependent measure-preserving transformations of $U_Z$ can fit the same factual $P(Z\mid \mathbf S,C)$ but generate different individual $Z^b$. Additivity, shared ranks, monotonicity, a fixed triangular flow ordering, or a Brenier/optimal-transport map are cross-world assumptions, not facts learned from factual likelihood.

Therefore label outputs as **model-implied under assumption set $\mathcal A$** and perform sensitivity analysis across plausible residual couplings.

### 3.3 Route B3 — population interventional analogue

If same-person preservation is not essential, target a weaker and more identifiable population object. Let $C_{\mathrm{pre}}$ contain measured pre-treatment context. Under a well-defined intervention, consistency, conditional exchangeability/no unmeasured confounding, positivity, and stable mechanisms, estimate

\[
P(\mathbf S(b)\mid C_{\mathrm{pre}})
\]

using a g-formula, inverse weighting, or another appropriate causal estimator.

If target-group factual data exist and one additionally assumes

\[
P(Z(b)\mid \mathbf S(b),C_{\mathrm{pre}})
=
P_{\mathrm{obs}}(Z\mid \mathbf S,C_{\mathrm{pre}},T=b),
\]

plus positivity, train a conditional realizer

\[
R_e(dz\mid \mathbf s,c,b)
\approx
P_{\mathrm{obs}}(dz\mid \mathbf S=\mathbf s,C_{\mathrm{pre}}=c,T=b).
\]

Inference samples a plausible target-population embedding. It does not identify which target CV belongs to the same source person. In the strongest resampling version,

\[
Z^b\perp Z^a\mid \mathbf S^b,C_{\mathrm{pre}},
\]

so $Z^a$ is used only to recover $C_{\mathrm{pre}}$. This makes the loss estimable but deliberately gives up residual person/style preservation.

Do not condition a nonparametric realizer on the complete, nearly unique $Z^a$ and claim positivity: the dataset never observes that same embedding under another state.

### 3.4 Route B4 — controlled semantic edit or partial identification

Two honest fallbacks are available.

**Controlled semantic edit.** A user or policy defines a target transformation $\tau(\mathbf S^a,a,b)$. Train an editor to produce an embedding with those target labels, plus support and preservation regularization. This is semantic recourse or controlled editing, not a total causal counterfactual unless $\tau$ comes from a defensible SCM.

**Partial identification.** Retain the set of causal/cross-world couplings compatible with observed marginals and assumptions. Propagate Fréchet, optimal-transport, or sensitivity bounds to downstream predictions and fairness measures. This may be preferable to one arbitrary point model when neither pairs nor a credible shared-noise model are available.

If the actual goal is only bias mitigation, a robust or invariant downstream predictor may be scientifically cleaner than pretending to recover $Z^b$.

## 4. Assumption ledger

Every Setup-B experiment must state which of the following it uses.

| Assumption | Why it is needed | What the factual data can check |
|---|---|---|
| Well-defined intervention $do(T=b)$ | Fixes what is changed and whether the query is total, direct, or path-specific | Only clarity/feasibility, not truth |
| Consistency and no interference | Connect observations to potential outcomes | Partly a design judgment |
| Correct temporal order and causal graph | Determines adjustment and propagation | Conditional-independence checks can falsify, not prove |
| Prespecified identifiable SCM/noise family on the known DAG | Lets factual nodewise losses estimate the mechanisms used by $K_S$ | Observational fit is testable; correct functional form and noise semantics are assumptions |
| Causal sufficiency / measured confounding or sequential exchangeability | Identifies population intervention effects | Sensitivity and negative controls only |
| Positivity/overlap | Ensures target states occur for relevant context | Empirically auditable on measured support |
| Counterfactual closure of $\mathbf S$ | Prevents omitted causal information from invalidating $K_S(\mathbf S^b\mid\mathbf S^a)$ | Label coverage can be audited; sufficiency remains substantive |
| Functional/shared-noise coupling for individual $K_S$ | Selects one joint law of $(\mathbf S^a,\mathbf S^b)$ | Not identified from single-world facts |
| Invariant context/residual or paired renderer for $Q_e$ | Selects which target embedding belongs to the same unit | Not identified without pairs/multiple views/anchors |
| If using a BGM: bijectivity plus the selected theorem's independence, monotonicity/IV/backdoor, differentiability, and variability conditions | Rules out state-dependent noise relabelings that preserve factual likelihood but alter counterfactuals | Some implications/support can be falsified; the full cross-world restriction is not established by fit |
| Stable text/embedding mechanism | Allows factual target-group conditionals to realize intervention worlds | External/multi-environment validation is needed |
| Target support | Prevents extrapolation to unseen state-context cells | Auditable with counts/density diagnostics |
| Valid human labels | Makes semantic supervision meaningful | Agreement, guidelines, adjudication, missingness audits |
| Frozen encoder and preprocessing | Defines a stable latent estimand | Fully controlled by implementation |
| Selection/transport to deployment population | Extends claims beyond the labeled corpus | Requires sampling information or external validation |

Path-specific interventions require additional mediation and cross-world assumptions and may fail standard identification criteria. Setup B should begin with a total intervention or a clearly controlled direct edit.

Human ground-truth labels make $\mathbf S^a$ measured; they do not make it counterfactually sufficient. If $Z^a$ contains omitted baseline information predictive of $\mathbf S^b$, either enlarge $\mathbf S/C$ or condition the causal model on that information and abandon the claim of a purely symbolic bridge.

## 5. Adapted training pipeline

### 5.1 Stage 0 — ingest, validate, and split

Minimum factual manifest:

- `unit_id` and, if needed, `person_id`, `author_id`, or source-group identifiers;
- `text`;
- observed `source_value` for $T$;
- one field per structured label;
- explicit `pre_treatment_context` and declared invariants;
- annotation source, confidence, missingness, and adjudication metadata;
- persisted train/validation/test split.

Split by person/author/source before model fitting. If several CV versions exist for a person, keep them in one split.

Any later target or pseudo-pair manifest must additionally store:

- `pair_id`, source and target query;
- $\mathbf S^b$ and its provenance (`expert_scm`, `estimated_scm`, `controlled_policy`, and model version);
- $X^b,Z^b$, if present, and provenance (`observed_intervention`, `human_rewrite`, `llm_rewrite`, `structural_generator`);
- preservation instructions, editor identity/model, prompt/settings, and audit status.

Never merge genuine and pseudo targets without an explicit target-type field.

### 5.2 Stage 1 — freeze and cache the encoder

**Input:** factual CV $X_i$.  
**Output:** $Z_i=e(X_i)$.  
**Training target/loss:** none; the encoder remains fixed.

Record encoder checkpoint, tokenizer, preprocessing, pooling, truncation, and artifact hash. If domain fine-tuning is used as an ablation, finish it once, freeze it, recompute every embedding, and retrain all downstream components.

### 5.3 Stage 2 — train the semantic kernel $G_e$

**Input:** $(Z_i,T_i,C_i)$, with the regime/context terms included only when scientifically needed.  
**Observed target:** human $\mathbf S_i$.  
**Output:** calibrated $G_{e,\eta}(d\mathbf s\mid z,t,c)$.  
**Primary loss:** a joint proper supervised score, for example

\[
\mathcal L_G
=
-\frac1n\sum_i\log g_\eta(\mathbf s_i\mid z_i,t_i,c_i).
\]

For mixed labels, use masked categorical cross-entropy, binary cross-entropy, and continuous likelihood terms as appropriate. Prefer a joint/autoregressive structured model or constrain impossible combinations; independent heads can generate invalid states.

Evaluate held-out NLL/Brier score, calibration, accuracy or macro-F1, subgroup calibration, and annotator agreement. If labels are supplied at runtime, $G_e$ is a diagnostic/optional convenience rather than a required bridge.

### 5.4 Stage 3 — obtain a structured transition $K_S$

**Selected project route:** the DAG is known, and the structural-equation/noise family is prespecified and assumed correctly specified and identifiable. For each non-intervened node $S_j$, train on factual parent–child tuples.

**Input:** $(\operatorname{pa}_j(\mathbf S_i),C_i,T_i)$ as required by the known graph.  
**Observed target:** the factual node value $S_{ij}$.  
**Loss:** a nodewise proper likelihood or the corresponding cross-entropy/MSE,

\[
\mathcal L_{\mathrm{SCM}}(\phi)
=
-\sum_{i=1}^n\sum_{j=1}^p
\log p_{\phi_j}
\!\left(s_{ij}\mid\operatorname{pa}_j(\mathbf s_i),c_i,t_i\right),
\]

with irrelevant conditioning terms omitted according to the DAG. The exact likelihood must match the declared structural equation and exogenous-noise law. At inference, abduct $U_S$ from $(\mathbf S_i^a,C_i)$, replace the intervention equation by $do(T=b)$, and predict with the same $U_S$. Integrating an uncertain exogenous posterior induces $K_S$.

There is still no supervised counterfactual loss of the form

\[
-\log k_\phi(\mathbf S_i^b\mid \mathbf S_i^a,a,b)
\]

because $\mathbf S_i^b$ is unobserved.

The project is choosing the following source of $K_S$:

1. **Known-DAG structural mechanism estimation:** fit the prespecified node mechanisms and exogenous laws as above. The adopted structural-noise semantics fix the individual coupling, and abduction–action–prediction calculates $K_S$. Validate factual fit and perform functional/noise sensitivity analysis.

Other defensible alternatives, if this assumption is relaxed, are:

2. **Expert-fixed functional SCM:** specify all equations and exogenous laws externally and estimate only declared nuisance parameters.
3. **External interventional/longitudinal data:** use the design-specific causal estimator; paired repeated units are especially valuable for transition information.
4. **Controlled edit policy:** set $\mathbf S^b=\tau(\mathbf S^a,a,b)$ by definition. Call it a policy/semantic edit, not a learned causal counterfactual.
5. **Partial-identification set:** keep several compatible kernels rather than selecting one.

Causal discovery is not required here because the graph is assumed known. Nevertheless, graph knowledge alone is not a substitute for the chosen functional/noise semantics: without them, the individual cross-world response coupling would remain undetermined even if population interventions were identifiable.

### 5.5 Stage 4A — paired latent training, if rewrites are acquired

**Input:** $(Z_i^a,\mathbf S_i^a,\mathbf S_i^b,a,b,C_i)$.  
**Observed protocol-relative target:** $Z_i^b=e(X_i^b)$.  
**Output:** $Q_{e,\theta}(dz^b\mid z^a,\mathbf s^a,\mathbf s^b,a,b,c)$.  
**Primary loss:** a proper paired score,

\[
\mathcal L_Q
=
\frac1n\sum_i
\mathcal S_{\mathrm{dist}}
\left(
Q_{e,\theta}(\cdot\mid z_i^a,\mathbf s_i^a,\mathbf s_i^b,a,b,c_i),
z_i^b
\right).
\]

Use conditional NLL only for an explicitly dominated/smoothed target family; otherwise use a suitable sample score such as an energy score. Semantic, identity, and preservation losses are auxiliaries.

This is the only Setup-B route that restores genuine paired supervision without assuming a latent residual coupling.

### 5.6 Stage 4B — factual structural latent model, if no rewrites exist

The most transparent first assumption-defined baseline is

\[
Z=m_\theta(\mathbf S,C)+U_Z,
\qquad
U_Z\perp \mathbf S\mid C
\]

with $U_Z$ declared invariant across the intervention.

**Input:** factual $(\mathbf S_i,C_i)$.  
**Observed target:** factual $Z_i$, not $Z_i^b$.  
**Loss:** MSE for a deterministic location model or Gaussian NLL for a location/scale model,

\[
\mathcal L_{\mathrm{lat,obs}}
=
-\frac1n\sum_i
\log r_\theta(z_i\mid \mathbf s_i,c_i).
\]

**Pseudo-counterfactual construction:** abduct $u_i$, obtain $\widetilde{\mathbf s}_i^b$ from the selected $K_S$, and reuse the same $u_i$ when decoding the target state.

A BGM is the preferred flexible version of this structural route if its assumptions are accepted:

- **factual training input/target:** $(\mathbf S_i,C_i)\mapsto Z_i$;
- **factual loss:** exact conditional flow NLL, including the Jacobian term, subject to the selected BGM structural restrictions;
- **inference:** $u_i=\Phi_e^{-1}(\mathbf S_i^a,C_i,Z_i^a)$ and $\widetilde Z_i^b=\Phi_e(\widetilde{\mathbf S}_i^b,C_i,u_i)$;
- **induced object:** the pushforward law is $Q_e$ itself. It is not supervised against an unobserved $Z_i^b$.

Do not use the frozen semantic constraint $-\log G_e(\widetilde{\mathbf S}^b\mid\widetilde Z^b)$ as the identifying BGM objective. It may be an auxiliary diagnostic or regularizer, but it only enforces membership in a semantic fiber. Similarly, a generic conditional flow, VAE, or diffusion model can replace the additive model as a sensitivity model, but reconstruction/likelihood alone identifies only factual $P(Z\mid\mathbf S,C)$. The alignment of residuals between states comes from the declared architecture/order/cost and cross-world assumptions.

Useful factual-only diagnostics are:

- held-out reconstruction or likelihood;
- residual predictability of $(T,\mathbf S)$ conditional on $C$ as a misspecification check;
- identity-query behavior;
- reversible-query cycle checks where scientifically meaningful;
- target-cell support and overlap;
- conditional distribution matching to held-out factual target-group embeddings;
- sensitivity across additive, location-scale, monotone-flow, and conditional-OT couplings.

These diagnostics can falsify bad models; they cannot validate the true individual $Z^b$.

### 5.7 Stage 4C — population realizer

**Input:** factual target-domain $(\mathbf S_i,C_i,T_i=b)$.  
**Observed target:** factual $Z_i$.  
**Output:** $R_e(dz\mid\mathbf s,c,b)$.  
**Loss:** conditional NLL or another proper score against $Z_i$.

At inference, sample $\widetilde{\mathbf S}^b$ from the selected population intervention model and then sample

\[
\widetilde Z^b\sim R_e(\cdot\mid\widetilde{\mathbf S}^b,C,b).
\]

This is a target-population resample. It should not be named $Q_e(Z^b\mid Z^a,\ldots)$ unless the extra independence/invariance assumptions connecting it to the potential target law are explicitly stated.

### 5.8 Stage 4D — original semantic editor as a baseline

Once some external $K_S$ or controlled policy supplies $\widetilde{\mathbf S}_i^b$, the existing constraint-based editor can be trained.

**Input:** $Z_i^a$, query $(a,b)$; the target state may be passed as an input for an oracle-state version.  
**Target used in the loss:** $\widetilde{\mathbf S}_i^b$, not $Z_i^b$.  
**Loss:** frozen-$G_e$ semantic consistency plus declared proximity/support/preservation penalties.

Call this the **semantic projection/editor baseline**. It is not a consistent estimator of a paired latent counterfactual, because no paired latent target appears.

If its deployed input is only $(Z^a,a,b)$, then this original one-model editor has the runtime contract of $H_e^{\mathrm{dir}}$ in the present notation—not of target-conditioned $Q_e$. Calling it $Q_e$ in the initial proposal does not change what is statistically trained. The constraint

\[
-\log G_e(\widetilde{\mathbf S}^b\mid \widetilde Z^b)
\]

can make the output semantically valid, but cannot choose the same person's $Z^b$ from all embeddings with those semantics. It therefore cannot turn this one model into a paired-counterfactual estimator without real paired $Z_i^b$ targets or an explicit structural teacher.

Do not evaluate semantic success solely through the same $G_e$ used in its loss. Use an independent probe, human labels on decoded/realized outputs, support diagnostics, or paired rewrites if available.

### 5.9 Stage 5 — optional direct symbolic-free student

This stage is optional and does **not** imply that $Q_e$ and $H_e$ must be separate neural networks. It is needed only when the chosen main method is modular/target-conditioned but deployment must accept no symbolic inputs. A shared backbone can expose a target-conditioned $Q$ mode and a masked/direct $H$ mode, provided both modes receive their corresponding training objectives.

If paired rewrites exist, direct $H_e^{\mathrm{dir}}$ can be trained exactly as in Setup A using $(Z_i^a,a,b)\mapsto Z_i^b$. If the original editor already has precisely this input contract, reuse it as $H_e$ rather than adding a duplicate model. The paired proper score is the primary objective; the $G_e$ semantic constraint is auxiliary.

Without pairs, first define an assumption-based teacher $T_{\mathcal A}$ using the chosen $K_S$ and structural latent model or population realizer:

\[
\widetilde Z_{im}^b
\sim
T_{\mathcal A}(\cdot\mid Z_i^a,\mathbf S_i^a,a,b,C_i).
\]

Then train

\[
\widehat H_{e,\omega}^{\mathrm{distill}}(dz\mid z^a,a,b)
\]

with a proper distributional score, KL where tractable, energy distance, or sample matching against teacher samples.

This supplies the desired symbolic-free inference interface

\[
X\to Z^a\to\widetilde Z^b,
\]

but the target is the teacher-induced marginal, not the unknown true direct kernel. Report teacher fidelity separately from causal/editor validity. A semantic $G_e$ loss is an auxiliary; by itself it returns to the fiber problem.

Equivalently, the teacher-implied direct law is the marginal composition

\[
H_{e,\mathcal A}^{\mathrm{comp}}(dz^b\mid z^a,a,b,c)
=
\int
G_e(d\mathbf s^a\mid z^a,a,c)
K_S(d\mathbf s^b\mid\mathbf s^a,a,b,c)
Q_e(dz^b\mid z^a,\mathbf s^a,\mathbf s^b,a,b,c),
\]

with observed $\mathbf S^a$ substituted for $G_e$ whenever available. If $C$ is not supplied, it must also be marginalized under an explicitly chosen deployment law. Distillation compresses this composition; it adds convenience, not causal information.

If neither pairs nor a declared teacher/generator exist, direct $H_e$ has no valid training target and training should hard-fail.

## 6. Inference

### 6.1 Modular inference when factual labels are supplied

Runtime input:

\[
(x,\mathbf s^{\mathrm{obs}},a,b,C).
\]

Compute

\[
z=e(x),
\qquad
\widetilde{\mathbf S}^b_m\sim\widehat K_S^{a\to b}(\cdot\mid\mathbf s^{\mathrm{obs}},C),
\]

then either

\[
\widetilde Z^b_m
=
\widehat\Phi_e(\widetilde{\mathbf S}^b_m,C,\widehat U_Z(z,\mathbf s^{\mathrm{obs}},C))
\]

for the model-implied individual route, or

\[
\widetilde Z^b_m\sim\widehat R_e(\cdot\mid\widetilde{\mathbf S}^b_m,C,b)
\]

for the population-resampling route.

### 6.2 Modular inference when factual labels are unavailable

Add the semantic stage:

\[
\widetilde{\mathbf S}^{\mathrm{obs}}_m
\sim
\widehat G_e(\cdot\mid z,a,C),
\]

and propagate each sample through $K_S$ and the selected realizer. Do not plug in a hard $\arg\max$ without quantifying the lost uncertainty.

### 6.3 Direct distilled inference

The student takes only

\[
(x,a,b),\quad z=e(x),
\]

and returns

\[
\widetilde Z^b_m
\sim
\widehat H_e^{\mathrm{distill}}(\cdot\mid z,a,b).
\]

This is convenient but hides which $\mathbf S^b$ was realized and inherits the teacher's SCM, residual, support, and estimation assumptions.

### 6.4 Required runtime controls

Record or fix:

- encoder checkpoint, tokenizer, preprocessing, and latent standardization;
- exact causal query, including total versus controlled/path-specific semantics;
- whether factual labels are supplied or predicted;
- causal graph/SCM version and adjustment set;
- invariant context and residual-coupling convention;
- support status for $(a,b,\mathbf S,C)$;
- number of Monte Carlo samples and random seed;
- target provenance and whether the output is paired, model-implied, population-resampled, or controlled-semantic.

The output is a distribution of latent embeddings. Producing counterfactual text requires a separately trained and validated text decoder/editor.

## 7. Evaluation and falsification

### 7.1 What can be evaluated from factual data

**Semantic kernel $G_e$:** held-out proper scores, calibration, accuracy/F1, subgroup calibration, missingness, annotation agreement.

**Causal model $K_S$:** observational mechanism likelihood, posterior predictive checks, conditional-independence implications, negative controls, balance/overlap, known external effects, and sensitivity across graphs/noise couplings.

**Structural latent model or population realizer:** held-out factual likelihood/reconstruction, residual diagnostics, conditional target-group distribution match, support/OOD rate, identity edit, declared invariant retention, and sensitivity across couplings.

**Direct student:** distributional fidelity to its teacher and runtime efficiency.

**Generated outputs:** independent semantic checks, human plausibility, non-target preservation, downstream utility/fairness, and robustness across plausible $K_S$/renderer models.

### 7.2 What cannot be evaluated without new data

Factual-only data cannot evaluate:

- same-person $\mathbf S_i^b$ transition accuracy;
- same-person $Z_i^b$ recovery error or coverage;
- calibration of an individual counterfactual law;
- whether preserved latent residuals really remain invariant across the intervention;
- whether direct $H_e$ recovers a real paired kernel rather than its teacher.

Fairness or downstream consistency on generated samples is evidence about the selected model, not proof that the model recovered the true missing counterfactual.

### 7.3 Strongest validation upgrade

Reserve a small, independently produced paired-rewrite set even if most training is factual-only. It can test:

- structured target validity;
- preservation protocol;
- latent paired error/proper score;
- whether the residual/OT coupling is plausible;
- whether a symbolic-free student generalizes beyond its own teacher.

Multiple valid rewrites per source are more informative than one because they expose renderer variation.

## 8. Repository-grounded implementation blueprint

### 8.1 Reusable components

- `src/encoder.py:47-69` already exposes a frozen text encoder and deterministic posterior mean. Reuse it with provenance and truncation audits.
- `src/semantic_decoder.py:30-93,120-170` is a useful first supervised $G_e$, but its hard-coded simulator columns and independent categorical heads must become schema-driven and calibrated.
- `src/latent_intervention.py:140-168,171-243` can remain as the semantic-projection/historical baseline once an external target $\mathbf S^b$ exists.
- The encoder and decoder parts of `src/pipeline.py` are adaptable after replacing simulator-specific data loading.
- `exp/estim/R` may provide a known-DAG/additive-model proof of concept, but it is not currently a sampler for real-data $K_S$ or $Z^b$.

### 8.2 Components that do not transfer automatically

- `exp/sim/scm.py` is the synthetic LIBERTy-style SCM, not a learned model of the real domain.
- `src/pipeline.py:118-140` reads the simulator's counterfactual state table. Setup B has no such target.
- The current text generator produces factual synthetic CVs; it is not a controlled paired editor for real CVs.
- The row-random split in `src/pipeline.py` must become person/source-group aware.
- No real-data ingestion, generic label schema, fitted known-DAG abduction module, invariant-residual/BGM realizer, pseudo-pair provenance, direct-mode distillation, or counterfactual sensitivity evaluator currently exists. In particular, the present `LatentIntervention` network is not a BGM: it has no invertible $U_Z\leftrightarrow Z$ map or BGM identification restrictions.

Simulator pretraining may later become a transport experiment, but it cannot silently supply real counterfactual identification. It needs explicit support and mechanism-invariance assumptions and should remain outside Setup B's primary claim for now.

### 8.3 Recommended separate implementation surface

Do not overload the Setup-A orbit pipeline. Add a separate observational pipeline with stages conceptually like:

1. `ingest-labeled-cvs`;
2. `make-group-splits`;
3. `encode-factual`;
4. `train-semantic`;
5. `fit-known-dag-scm` and audit abduction–action–prediction;
6. `fit-latent-realizer` (additive/location-scale/BGM) or `ingest-paired-rewrites`;
7. `build-target-manifest` with provenance;
8. optionally `train-paired-editor` or `distill-direct-mode` when the structural realizer itself is not the deployment interface;
9. `evaluate-observed` and `sensitivity-analysis`.

The configuration needs an explicit `estimand` enum, for example:

- `factual_semantics`;
- `controlled_semantic_edit`;
- `population_intervention`;
- `model_implied_counterfactual`;
- `paired_rewrite_counterfactual`;
- `partially_identified`.

Hard requirements should follow the selected estimand. In particular, paired $Q/H$ training must refuse to run unless a valid $Z^b$ target exists; distilled $H$ must record its teacher and assumption set.

### 8.4 Required tests

Add tests for:

- schema and label coverage;
- no person/source leakage;
- encoder artifact provenance;
- masked/mixed-label semantic loss and calibration;
- graph ordering and intervention semantics;
- abduction/action/prediction under toy SCMs;
- identity interventions;
- target support checks;
- residual inversion and round trips;
- pseudo-pair provenance and hard failures;
- direct-student teacher fidelity;
- sensitivity results changing when the coupling assumption changes.

### 8.5 Existing paper-plan inconsistency

`paper/experiments.md:198-222` currently says that real-data Studies 2/3 lack a known SCM and that $h_S$ “must be learned” with the same interface. The present study decision supersedes that description: it assumes a known DAG and a prespecified identifiable structural-equation/noise family, fits its node parameters to factual data, and calculates $K_S$ by abduction–action–prediction. The paper plan should state this assumption and the factual nodewise objective explicitly; it must not imply supervision against an unavailable $\mathbf S_i^b$.

The plan must separately choose the latent route: paired rewrites or a declared model-implied coupling such as the additive renderer/BGM. It should also state that no observed $X^b/Z^b$ means true latent counterfactual accuracy cannot be validated, and that the existing one-step editor is already the direct-$H$ interface if it takes only $(Z^a,a,b)$.

## 9. Identification and consistency statements that are safe

### 9.1 Factual semantic identification

Under a fixed encoder and iid or appropriately clustered factual sampling, $G_e^*=P(\mathbf S\mid Z,T,C)$ exists and is identified almost surely on factual support. A correctly specified/sieve estimator trained with a strictly proper score can consistently estimate it.

### 9.2 Population intervention identification

Under consistency, no interference, a correct causal graph, measured confounding/sequential exchangeability, positivity, and stable mechanisms, selected interventional marginals or conditionals for $\mathbf S(b)$ may be identified by the relevant g-formula or causal estimator. This does not generally identify $P(\mathbf S^b\mid\mathbf S^a)$ for the same unit.

### 9.3 Model-relative individual identification

If a functional SCM $F_r$, latent structural mechanism $\Phi_e$, exogenous laws, and shared-noise coupling are known or identifiable within a declared structural family, the induced $K_S,Q_e,H_e$ are identified **relative to that model family and coupling**. Single-world factual data do not nonparametrically establish those assumptions.

### 9.4 Paired-rewrite identification

Controlled paired rewrites identify the conditional laws induced by the source population and rewrite protocol on supported queries. They do not prove that the protocol equals an unobservable real-life causal text mechanism.

### 9.5 Estimation conditions

Any consistency claim additionally needs independent units, stable sampling, sufficient support, proper scoring, appropriate model/sieve capacity, controlled regularization and optimization error, and honest separation of generated pseudo-targets from observations.

## 10. Claims permitted and forbidden

### Permitted

- Human labels enable supervised estimation and calibration of $G_e$.
- A specified causal model or policy induces a structured target distribution.
- An explicit shared-residual/transport model produces model-implied individual edits.
- Conditional resampling produces population target analogues under stated exchangeability/invariance/support assumptions.
- A direct student can amortize a declared modular teacher for symbolic-free inference.
- Results are sensitive or robust to declared SCM and coupling alternatives.
- Paired rewrites identify an editor relative to their audited protocol.

### Forbidden without additional evidence

- Factual $(X,\mathbf S^a)$ labels identify $\mathbf S^b$ or $Z^b$.
- Natural observations from both $T$ groups are paired intervention worlds.
- Observational likelihood identifies a unique individual SCM/noise coupling.
- $G_e(\widetilde Z^b)=\mathbf S^b$ identifies the correct target embedding.
- Minimal latent distance, cycle consistency, OT, or adversarial matching recovers the true same-person coupling.
- Teacher distillation creates independent counterfactual evidence.
- A factual-only experiment validates individual latent-counterfactual coverage or error.
- Simulation-to-real transfer holds without an explicit transport argument.

## 11. Recommended decision gate

Before coding Setup B, answer in order:

1. Is the target an individual same-person counterfactual, a population intervention distribution, a controlled semantic edit, or a robust downstream predictor?
2. Which variable is intervened on, and which descendants/paths may change?
3. Do the human labels contain the causal closure and pre-treatment adjustment variables?
4. Are labels also supplied at inference?
5. Are the known DAG, structural equations/noise family, identification conditions, and abduction procedure for $K_S$ fully specified and auditable?
6. If individual $Z^b$ is claimed, what supplies the renderer/residual coupling: paired rewrites, multiple views, anchors, or an explicit structural assumption?
7. Does the data have support for every requested target state and context?
8. Can a small paired audit set be collected?

Recommended choices:

- **If individual latent counterfactual recovery is central:** collect paired rewrites and use paired $Q/H$ losses.
- **If pairs are impossible but an SCM and residual convention are defensible:** implement the additive/location-scale model first; add a BGM only after mapping its multidimensional identification assumptions, call both model-implied, and report coupling sensitivity.
- **If same-person preservation is unnecessary:** target a population interventional analogue and say so plainly.
- **If neither causal assumptions nor pairs are defensible:** report partial bounds or focus on downstream robustness rather than $Z^b$ recovery.

## 12. References and local context

- [Setup A agent handoff](./26-08-07_stochastic-kernel_agent.md), especially its semantic-fiber diagnosis and paired-orbit identification results. Those results do not transfer to factual-only Setup B.
- [Current manuscript](../../paper/ecaf26-template.tex): line 186 assumes access to an SCM; lines 227–234 already acknowledge that real-data causal relations and full structured observability are restrictive.
- [Current experiment plan](../../paper/experiments.md): Study 2 correctly flags the missing real-domain SCM but currently understates the impossibility of learning $h_S/K_S$ and $Q/H$ from factual labels alone.
- [LIBERTy paper extraction](../../literature/Toker%20et%20al.%20-%202026%20-%20LIBERTy%20A%20Causal%20Framework%20for%20Benchmarking%20Concept-Based%20Explanations%20of%20LLMs%20with%20Structural%20Coun.md): contrast only. Its declared shared-noise and renderer coupling supplies information absent here.
- [Nasr-Esfahany, Alizadeh, and Shah, “Counterfactual Identifiability of Bijective Causal Models,” ICML 2023](https://proceedings.mlr.press/v202/nasr-esfahany23a.html): BGM definition, theorem-specific scalar Markovian/IV and multidimensional backdoor identification conditions, and counterfactual equivalence up to a parent-independent noise relabeling.
- [Nasr-Esfahany and Kiciman, “Counterfactual (Non-)identifiability of Learned Structural Causal Models,” 2023](https://arxiv.org/abs/2301.09031): why a known DAG, observational fit, and flexible high-dimensional mechanisms do not generally identify individual counterfactuals without additional structural restrictions.
- Pearl, *Causality*, 2nd ed.: SCM intervention and abduction-action-prediction semantics.
- Hernán and Robins, *Causal Inference: What If*: consistency, exchangeability, positivity, and g-formula identification.
- Manski, *Partial Identification of Probability Distributions*: honest set-valued inference when the cross-world coupling is not point identified.
- The general impossibility of unsupervised disentanglement is relevant to any claim that factual reconstruction alone discovers an invariant person/style residual; supervision of $\mathbf S$ helps semantic prediction but does not by itself align residuals across unobserved worlds.

## 13. One-sentence handoff

In Setup B, factual labels supervise $Z^a\to\mathbf S^a$ and the adopted known-DAG SCM's node parameters; $K_S$ is then obtained by model-relative abduction–action–prediction, while a same-unit $Z^b$ still requires paired rewrites or an explicit renderer coupling (with BGM as a strong-assumption option), and a separate $H_e$ is needed only for an additional symbolic-free interface.
