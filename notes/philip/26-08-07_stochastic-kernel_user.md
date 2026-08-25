# Action brief: rerun LIBERTy as complete counterfactual orbits

## Decision

Yes: rerunning the LIBERTy CV generator from scratch can give us the right training data for the revised method.

The important condition is that “generate all counterfactual queries” must mean **complete matched text-and-embedding orbits**, not only additional counterfactual tables. For every simulated candidate, every world needed by the paper must be generated from the same underlying candidate and the same fixed rendering context.

The current repository is not there yet. It already creates correct shared-noise factual and counterfactual **tabular states**, but only for one configured intervention, and it verbalizes only the factual state. Therefore it has no matched counterfactual text $X^b$ or target embedding $Z^b$.

## What one complete orbit looks like

For one simulated candidate:

$$
\text{same SCM noise, persona, template, renderer context}
\longrightarrow
\begin{cases}
\mathbf S^a\to X^a\to Z^a,\\
\mathbf S^b\to X^b\to Z^b,\\
\ldots
\end{cases}
$$

Only the intervention regime changes. The SCM noise, persona, template, and every renderer choice intended to represent the same person or style remain coupled across worlds.

This pairing is the source of identification. The old method only finds a nearby latent point with the right decoded attributes. Many points can satisfy that requirement. The paired target $Z^b$ shows which embeddings the simulator actually produces for the same unit under the intervention.

## What must be implemented before the full paid run

1. **Lock the query set.** Specify every source and target regime the paper will claim, including whether the intervention is total or path-specific. The current SCM implements total node interventions; path-specific claims require separate edge/path interventions.

2. **Generate one long orbit per unit.** Sample the SCM noise once and evaluate the complete configured regime grid from it. Store one row per $(\text{unit},\text{regime},\text{renderer replicate})$.

3. **Couple and audit the renderer.** Reuse persona and template, explicitly couple concrete values such as age and work experience, reuse the exact text for identity worlds, and record prompt, model, settings, and response metadata. Temperature zero and equal persona IDs alone do not prove that person and style were preserved.

4. **Use compound identifiers and resumable generation.** A single integer ID cannot distinguish several worlds. The generator needs at least `unit_id`, `regime_id`, and `renderer_id`.

5. **Split by unit before making pairs.** All worlds and renderer replicates belonging to the same SCM unit must stay in the same train, validation, or test split.

6. **Run a small end-to-end pilot first.** Before generating the full dataset, verify on roughly 20–50 units that every orbit is complete, the intended attributes occur in the text, invariant content is preserved, identity worlds are exact, and the resulting $Z^a,Z^b$ pairs align correctly.

7. **Choose the sample size from transition support.** More interventions for one candidate do not create more independent candidates. Use a large, cheap tabular simulation to count rare transitions before deciding how many expensive CV orbits to render.

## How the revised training works

1. **Freeze and cache the encoder.**

   - **Input:** every generated text $X_i^r$, where $i$ is the unit and $r$ is its intervention regime.
   - **Output:** the fixed embedding $Z_i^r=e(X_i^r)$, keyed by unit, regime, and renderer replicate.
   - **Loss/objective:** none. The encoder is not trained. Use one fixed representation—initially the LangVAE posterior mean—and record the checkpoint and preprocessing.

2. **Train the regime-aware semantic kernel $G_e$.**

   - **Input:** $Z_i^r$ and regime label $r$.
   - **Training target:** the simulator state $\mathbf S_i^r$.
   - **Output:** $G_e(\mathbf S\mid Z,r)$, a calibrated distribution over the structured state in regime $r$.
   - **Loss/objective:** a proper supervised loss, initially joint or per-variable categorical cross-entropy:

     $$
     \mathcal L_G
     =
     -\mathbb E\log g_\eta(\mathbf S_i^r\mid Z_i^r,r).
     $$

3. **Fit or calculate the shared SCM kernel $K_S$.**

   - **Input:** factual state $\mathbf S_i^a$, source regime $a$, and requested target regime $b$.
   - **Training target:** the matched structured counterfactual $\mathbf S_i^b$.
   - **Output:** $K_S^{a\to b}(\mathbf S^b\mid\mathbf S^a)$, which represents uncertainty about the target state when the unit’s hidden SCM noise is unavailable.
   - **Loss/objective:** the stored-noise oracle requires no training loss. A deployable table or model minimizes a proper transition loss such as

     $$
     \mathcal L_K
     =
     -\mathbb E\log k_\phi(\mathbf S_i^b\mid\mathbf S_i^a,a,b),
     $$

     and is validated against held-out stored-noise oracle transitions.

4. **Train the paired latent editor $Q_e$.**

   - **Input:** factual embedding $Z_i^a$, factual state $\mathbf S_i^a$, target state $\mathbf S_i^b$, and regime labels $a,b$.
   - **Training target:** the matched target embedding $Z_i^b=e(X_i^b)$.
   - **Output:** $Q_e(Z^b\mid Z^a,\mathbf S^a,\mathbf S^b,a,b)$, a conditional distribution over target embeddings. The version omitting $\mathbf S^a$ is an ablation testing whether $Z^a$ already contains all useful factual semantics.
   - **Loss/objective:** a strictly proper conditional distributional score:

     $$
     \mathcal L_Q
     =
     \mathbb E\,
     \mathcal S_{\mathrm{dist}}
     \left(
       Q_\theta(\cdot\mid Z_i^a,\mathbf S_i^a,\mathbf S_i^b,a,b),
       Z_i^b
     \right).
     $$

     Conditional negative log-likelihood is appropriate only for an explicit density or smoothed target; an energy score can be used when the target law is singular. Semantic and identity losses are auxiliaries, not the identifying objective.

5. **Train the direct symbolic-free editor $H_e^{\mathrm{dir}}$.**

   - **Input:** factual embedding $Z_i^a$ and regime query $(a,b)$—no structured state enters the model.
   - **Training target:** the genuine paired target embedding $Z_i^b$.
   - **Output:** a fitted $\widehat H_{e,\psi}^{\mathrm{dir}}(Z^b\mid Z^a,a,b)$, which can be sampled without $G_e$ or $K_S$ at inference.
   - **Loss/objective:** a proper paired score against $Z_i^b$:

     $$
     \mathcal L_H
     =
     \mathbb E\,
     \mathcal S_{\mathrm{dist}}
     \left(
       \widehat H_{e,\psi}^{\mathrm{dir}}(\cdot\mid Z_i^a,a,b),
       Z_i^b
     \right).
     $$

     A frozen-$G_e$ semantic loss using $S_i^b$ and distillation from the modular model are optional auxiliaries; neither may replace paired $Z_i^b$ supervision.

6. **Retain the original deterministic editor as a baseline.**

   - **Input:** $Z_i^a$ and the intervention encoding for $a\to b$.
   - **Training target:** $\mathbf S_i^b$, used by the frozen semantic model to score the edited point; it is supervision, not a direct input to the current editor.
   - **Output:** one edited point $\widehat Z_i^b$.
   - **Loss/objective:** the original semantic-consistency loss plus latent displacement penalties,

     $$
     \mathcal L_{\mathrm{old}}
     =
     \mathcal L_{\mathrm{sem}}
     +\alpha\lVert\widehat Z_i^b-Z_i^a\rVert_1
     +\beta\lVert\widehat Z_i^b-Z_i^a\rVert_2^2.
     $$

7. **Evaluate the modules and their composition.**

   - **Input:** held-out complete orbits grouped by unit.
   - **Output:** separate results for $G_e$, deployable versus oracle $K_S$, true-state $Q_e$, the complete $G\to K\to Q$ pipeline, and direct symbolic-free $H_e^{\mathrm{dir}}$.
   - **Loss/objective:** no parameter updates. Report proper scores and calibration, semantic and support validity, paired target accuracy, persona/style preservation, downstream utility, and the declared fairness measure.

## Inference: input, output, and required controls

### Interpretable modular inference

For a new CV, inference receives:

- the factual text $x$;
- the known or declared source regime $a$;
- the requested target regime $b$;
- optionally, explicit persona/renderer context if the method is designed to control it separately.

It then performs

$$
z=e(x),\qquad
\mathbf S^a_m\sim G_e(\cdot\mid z,a),\qquad
\mathbf S^b_m\sim K_S^{a\to b}(\cdot\mid\mathbf S^a_m),\qquad
Z^b_m\sim Q_e(\cdot\mid z,\mathbf S^a_m,\mathbf S^b_m,a,b).
$$

The output is a sample $\{Z^b_m\}_{m=1}^M$ approximating the composed counterfactual latent distribution $H_e^{\mathrm{comp},a\to b}(\cdot\mid z)$. A downstream predictor can turn these into a predictive distribution or fairness statistic. The core method does **not** automatically output a counterfactual text; that would require a separately validated text decoder.

For component evaluation only, an oracle mode may replace $G_e$ and $K_S$ with the simulator’s true $\mathbf S^a,\mathbf S^b$. That isolates $Q_e$, but it is not the deployable inference procedure.

The following must be controlled:

- **Encoder:** use exactly the frozen checkpoint and preprocessing used during training.
- **Causal query:** $a$, $b$, and total-versus-path-specific semantics must match a supported training query.
- **Causal closure:** $K_S$ determines which structured variables and descendants are allowed to change; $Q_e$ should realize that target state rather than independently choosing causal changes.
- **Same-unit information:** retain the complete factual embedding $z$ as an input to $Q_e$; it carries observable person and presentation information not contained in $\mathbf S$.
- **Renderer invariants:** persona, template, and causally irrelevant renderer choices must have been coupled correctly in the training orbits. If one of them must be explicitly controlled at inference, it must be recorded and added as a model input; an unobserved variable cannot be guaranteed fixed.
- **Unknown SCM noise:** do not use the stored-noise oracle for a new CV. The deployable $K_S$ must average over SCM noises compatible with the inferred factual state.
- **Support:** reject or flag source states and target transitions outside the training support.
- **Monte Carlo:** fix and report the number of samples and random seed for reproducible evaluation, while retaining the full sample distribution rather than only its mean.

A stochastic kernel may turn out to be almost deterministic. We should test that rather than force artificial variance. If deterministic texts and embeddings make the target distribution singular, use a proper distributional score that supports such laws, or explicitly state that a Gaussian model estimates a smoothed approximation.

## Symbolic-free deployment path: learn $H_e^{\mathrm{dir}}$ directly

If deployment should require only the factual embedding and intervention query, train a separate direct model

$$
\widehat H_{e,\psi}^{\mathrm{dir}}(dZ^b\mid Z^a,a,b)
\approx
P(Z^b\in dZ^b\mid Z^a,a,b).
$$

Its **input** is $(Z_i^a,a,b)$ and its **primary target** is the genuine paired $Z_i^b$. It therefore needs no $S^a$, $S^b$, $G_e$, or $K_S$ at inference:

$$
z=e(x),
\qquad
Z_m^b\sim\widehat H_{e,\psi}^{\mathrm{dir}}(\cdot\mid z,a,b).
$$

The simulator states remain useful as privileged training information. With frozen $G_e$, add the auxiliary check

$$
\mathcal L_{\mathrm{sem}}^H
=
-\mathbb E\log G_e^b(S_i^b\mid\widetilde Z_i^b),
\qquad
\widetilde Z_i^b\sim\widehat H_{e,\psi}^{\mathrm{dir}}(\cdot\mid Z_i^a,a,b).
$$

This semantic loss must not replace the paired loss against $Z_i^b$; by itself it recreates the original semantic-fiber ambiguity. Optionally, the composed $G\to K\to Q$ model can act as a teacher and provide extra samples for distributional distillation, but genuine paired $Z^b$ remains the identifying target so the student does not merely inherit teacher errors.

During this auxiliary loss, freeze $G_e$ itself but let the gradient pass through $G_e$ back into the direct editor.

The trade-off is simple: direct $H_e^{\mathrm{dir}}$ gives convenient symbolic-free inference but marginalizes over the possible $S^a,S^b$. It cannot expose or explicitly select the realized target state, and it needs direct training support for every claimed $a\to b$ query.

## Which model plays which role?

The editor variants are trained **separately on the same unit splits**:

| Editor | What it learns from | Role |
|---|---|---|
| Original deterministic editor | Target semantics plus $L_1/L_2$ proximity; it never sees the matched $Z^b$ | Historical baseline representing the original proposal |
| Deterministic modular regressor $m_Q$ | Paired $Z^b$, conditioned on $Z^a,S^a,S^b,a,b$, but predicts one point | Falsification baseline for stochastic $Q_e$ |
| Stochastic paired editor $Q_e$ | Paired $Z^b$, conditioned on $Z^a,S^a,S^b,a,b$ | Interpretable modular editor inside the scientific $G\to K\to Q$ model |
| Deterministic direct regressor $m_H$ | Paired $Z^b$, conditioned only on $Z^a,a,b$, but predicts one point | Falsification baseline for stochastic direct $H_e^{\mathrm{dir}}$ |
| Direct stochastic $\widehat H_e^{\mathrm{dir}}$ | Paired $Z^b$, conditioned only on $Z^a,a,b$ | Symbolic-free deployment student and direct-kernel baseline |

The comparisons answer four different questions:

1. **Original versus deterministic modular:** does direct supervision by the true paired $Z^b$ improve on semantic proximity?
2. **Deterministic modular versus stochastic $Q_e$:** is latent synthesis still ambiguous after $(S^a,S^b)$ are known?
3. **Deterministic direct versus stochastic direct $H_e^{\mathrm{dir}}$:** is the complete symbolic-free counterfactual law nondegenerate?
4. **Composed $G\to K\to Q$ versus direct $\widehat H_e^{\mathrm{dir}}$:** what is gained by explicit symbolic control, and what is lost when it is marginalized for simpler deployment?

The two population targets are

$$
\begin{aligned}
Q_e^*
&=P(Z^b\mid Z^a,\mathbf S^a,\mathbf S^b,a,b),\\
H_e^{\mathrm{dir},*}
&=P(Z^b\mid Z^a,a,b).
\end{aligned}
$$

Test point-mass feasibility separately. $Q_e^*$ can be deterministic after $(S^a,S^b)$ are given while direct $H_e^{\mathrm{dir},*}$ remains stochastic because it marginalizes uncertainty over those states. Otherwise, paired squared-error regression approaches a conditional mean, which may average incompatible targets. The original $L_1/L_2$ editor only approaches a minimizer of its semantic-distance objective; that point is not identified as the paired counterfactual.

The recommended paper design is to treat $G\to K\to Q$ as the interpretable scientific/reference model and direct $\widehat H_e^{\mathrm{dir}}$ as the practical symbolic-free student. Their matching deterministic regressors are the safety/falsification baselines, and the old editor is the historical baseline. Claim important stochasticity for either model only if it beats its deterministic counterpart on a common held-out proper score and produces calibrated, supported variation. Otherwise, report that the corresponding kernel is effectively deterministic for this simulator and encoder.

## What this lets us claim

Complete population orbits directly identify

$$
H_e^{\mathrm{dir},*}(dz'\mid z,a,b)
=
P(Z^b\in dz'\mid Z^a=z,a,b),
$$

as well as the regime-conditioned semantic, structured-counterfactual, and latent-editor distributions **relative to our fixed LIBERTy-style SCM, renderer, and frozen encoder, and only on supported queries**. Direct $H_e^{\mathrm{dir}}$ does not require the structured-state sufficiency assumption.

For the modular composition $G\to K\to Q$ to equal the directly paired simulator counterfactual law, the structured state must contain all factual causal information relevant to the target state. If the factual embedding reveals additional relevant hidden information, enlarge the structured state or let $K_S$ condition on that context.

This establishes the method in a controlled synthetic world. It does not automatically identify real-person CV counterfactuals or justify simulation-to-real transfer.

## Immediate next action

Do not start the full billed generation yet. First implement and validate the query manifest, multi-regime orbit schema, renderer coupling, compound resume keys, unit-level split, and small end-to-end pilot. Once that pilot produces correctly aligned paired $Z^a,Z^b$, the full rerun is aligned with the revised problem.

## Intuition: why $G_e$ and $K_S$ can still be stochastic

- **Deterministic encoder does not imply deterministic semantics.** The encoder fixes $Z=e(X)$ for a given text, but it may discard information. If several structured states are compatible with the same $Z$, then $G_e(S\mid Z,a)$ is a distribution. It collapses to a point only when the relevant state is recoverable from $Z$.
- **Deterministic SCM does not imply deterministic abduction from an observed state.** The structural equations are deterministic given the hidden SCM noise $U$, but a new CV does not reveal $U$. Several noise values can produce the same observed $S^a$ and different $S^b$, so $K_S(S^b\mid S^a,a,b)$ averages over them. With stored $U$ in synthetic oracle evaluation, it collapses to one target.
