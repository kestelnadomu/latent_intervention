# Action brief: rerun LIBERTy as complete counterfactual orbits

## Decision

Yes: rerunning the LIBERTy CV generator from scratch can give us the right training data for the revised method.

The important condition is that “generate all counterfactual queries” must mean **complete matched text-and-embedding orbits**, not only additional counterfactual tables. For every simulated candidate, every world needed by the paper must be generated from the same underlying candidate and the same fixed rendering context.

The current repository is not there yet. It already creates correct shared-noise factual and counterfactual **tabular states**, but only for one configured intervention, and it verbalizes only the factual state. Therefore it has no matched counterfactual text \(X^b\) or target embedding \(Z^b\).

## What one complete orbit looks like

For one simulated candidate:

\[
\text{same SCM noise, persona, template, renderer context}
\longrightarrow
\begin{cases}
\mathbf S^a\to X^a\to Z^a,\\
\mathbf S^b\to X^b\to Z^b,\\
\ldots
\end{cases}
\]

Only the intervention regime changes. The SCM noise, persona, template, and every renderer choice intended to represent the same person or style remain coupled across worlds.

This pairing is the source of identification. The old method only finds a nearby latent point with the right decoded attributes. Many points can satisfy that requirement. The paired target \(Z^b\) shows which embeddings the simulator actually produces for the same unit under the intervention.

## What must be implemented before the full paid run

1. **Lock the query set.** Specify every source and target regime the paper will claim, including whether the intervention is total or path-specific. The current SCM implements total node interventions; path-specific claims require separate edge/path interventions.

2. **Generate one long orbit per unit.** Sample the SCM noise once and evaluate the complete configured regime grid from it. Store one row per
   \[
   (\text{unit},\text{regime},\text{renderer replicate}).
   \]

3. **Couple and audit the renderer.** Reuse persona and template, explicitly couple concrete values such as age and work experience, reuse the exact text for identity worlds, and record prompt, model, settings, and response metadata. Temperature zero and equal persona IDs alone do not prove that person and style were preserved.

4. **Use compound identifiers and resumable generation.** A single integer ID cannot distinguish several worlds. The generator needs at least `unit_id`, `regime_id`, and `renderer_id`.

5. **Split by unit before making pairs.** All worlds and renderer replicates belonging to the same SCM unit must stay in the same train, validation, or test split.

6. **Run a small end-to-end pilot first.** Before generating the full dataset, verify on roughly 20–50 units that every orbit is complete, the intended attributes occur in the text, invariant content is preserved, identity worlds are exact, and the resulting \(Z^a,Z^b\) pairs align correctly.

7. **Choose the sample size from transition support.** More interventions for one candidate do not create more independent candidates. Use a large, cheap tabular simulation to count rare transitions before deciding how many expensive CV orbits to render.

## How the revised training works

1. Freeze the encoder and cache the embedding of every orbit world.
2. Train the regime-aware semantic kernel
   \[
   G_e^a(\mathbf S\mid Z)
   \]
   on genuine embeddings from all supported regimes.
3. Implement the shared SCM kernel
   \[
   K_S^{a\to b}(\mathbf S^b\mid\mathbf S^a).
   \]
   Stored simulator noise gives an oracle version. A new CV requires a deployable version that averages over unknown noise compatible with its observed state.
4. Train the latent editor
   \[
   Q_e(Z^b\mid Z^a,\mathbf S^b,a,b)
   \]
   directly against the matched target \(Z^b\).
5. Keep the existing semantic-consistency+\(L_1/L_2\) editor as the original-method baseline.
6. Evaluate \(G_e\), \(K_S\), and \(Q_e\) separately before evaluating the complete \(G\to K\to Q\) pipeline and downstream fairness.

A stochastic kernel may turn out to be almost deterministic. We should test that rather than force artificial variance. If deterministic texts and embeddings make the target distribution singular, use a proper distributional score that supports such laws, or explicitly state that a Gaussian model estimates a smoothed approximation.

## What this lets us claim

Complete population orbits identify the regime-conditioned semantic, structured-counterfactual, and latent-editor distributions **relative to our fixed LIBERTy-style SCM, renderer, and frozen encoder, and only on supported queries**.

For the modular composition \(G\to K\to Q\) to equal the directly paired simulator counterfactual law, the structured state must contain all factual causal information relevant to the target state. If the factual embedding reveals additional relevant hidden information, enlarge the structured state or let \(K_S\) condition on that context.

This establishes the method in a controlled synthetic world. It does not automatically identify real-person CV counterfactuals or justify simulation-to-real transfer.

## Immediate next action

Do not start the full billed generation yet. First implement and validate the query manifest, multi-regime orbit schema, renderer coupling, compound resume keys, unit-level split, and small end-to-end pilot. Once that pilot produces correctly aligned paired \(Z^a,Z^b\), the full rerun is aligned with the revised problem.
