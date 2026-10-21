# Closest-work verification for the short related-work draft

Checked 2026-10-19. Supports [the manuscript-shaped draft](../manuscript-draft.md); follows the [broader field selection](../distilled-review.md).

## Scope and conclusions

This is a targeted comparison check, not a full review of every cited paper. Nine entries below were checked against selected full-text method/theory passages; the Makkuva entry uses its primary proceedings abstract for a narrow architectural/objective claim. Proofs and reported experimental results were not comprehensively audited. Other citations in the short draft retain the bounded evidence from the preceding library passes. No superiority claim follows from this check.

The resulting editorial choices are:

- Keep **read–operate–render over frozen document embeddings** as the framing, with semantic-target versus distribution-based rendering as the main comparison.
- Describe the semantic editors as **learned, reusable maps**. Do not claim that probe consistency plus proximity is new.
- Acknowledge **shared-reference inverse–forward editing** as an existing construction. It is not unique to our state flow or evidence of identification.
- Include **MUtE as a closely related preprint**, rather than leaving it solely as a reading lead. Its idealized routing and distribution assumptions are not adopted as guarantees for our methods.
- Keep the distinction between **a convex-gradient map**, fitting a specified distribution, and the correct individual causal correspondence.

## Source-level checks

The middle column reports what the cited source states. The last column is our project comparison and editorial judgment, not a claim made by those authors. Section references identify what was checked; they do not imply review of every section or proof.

| Work and inspected source | Source-backed observation | Consequence for our comparison |
| --- | --- | --- |
| [Beyond CBMs / Laguna et al.](../papers/laguna2024beyond-cbm.md): [v3, §3.1 Eq. (1), §3.3](https://arxiv.org/html/2401.13544v3) | A concept-labelled validation set fits a probe. An intervention optimizes the representation itself against concept loss and distance. The paper separately proposes model fine-tuning for intervenability; that is not required by its basic editing procedure. | Closest objective precedent. Our two editors amortize editing into target-conditioned maps and use an externally supplied target-state mixture. Avoid contrasting our frozen components with a claim that their basic procedure necessarily retrains the backbone. |
| [Representation Surgery / MiMiC](../papers/singh2024representation-surgery.md): [v7, §4.1–4.2, Propositions 4.1–4.2](https://arxiv.org/html/2402.09631v7) | Mean matching yields a translation under its least-squares steering criterion. MiMiC adds second-moment matching, with full-rank class covariances in the stated result, and connects to Gaussian OT. | This is not merely erasure or a generic direction-vector heuristic. It is relevant both to semantic steering and our Gaussian transport baseline; moment matching does not establish the benchmark's individual pairing. |
| [MUtE](../papers/saillenfest2026mute.md): [v1, §2.1–2.4 Eqs. (5)–(7), §3.2–3.3](https://arxiv.org/html/2609.11253v1) | Class-conditional invertible erasure maps share a common output law; composing one map with another's inverse yields a target-class transport. Theory idealizes source-class routing as exact; implementation uses labelled density estimates and predicted routing, whose errors limit alignment. | Directly relevant to both frozen-representation editing and shared-reference transport. Include as a preprint, not a proven solution to our multivariate symbolic intervention problem. Matching class distributions does not select our simulator's true counterpart. |
| [StyleFlow](../papers/abdal2021styleflow-images.md): [§5.1, §5.3, §6.2](https://arxiv.org/html/2008.02401) | Fits attribute-conditional likelihood on StyleGAN codes with inferred image attributes. Editing reverses under source attributes and forwards with changed attributes while fixing the reference code. The image system also includes edit-specific selection of StyleGAN layers. | A close construction and training precedent for state flow, not merely a vague flow citation. Our external symbolic operation and document embeddings distinguish the setting; they do not make inverse–forward editing new. |
| [DSCM](../papers/pawlowski-dscm-2020.md): [§2.2–2.3](https://arxiv.org/html/2006.06485) | Places neural mechanisms inside an SCM. Conditional flows support explicit noise inversion; noninvertible mechanisms can use variational abduction. Counterfactual prediction uses the abducted noise after changing structural assignments. | Our fitted conditional embedding model adopts the abduction/rendering pattern but does not learn or identify the complete text-generating SCM. The known symbolic kernel and fitted latent renderer have different roles. |
| [Transport-based Counterfactual Models](../papers/delara2024transport-counterfactuals.md): [§4.3, §5.1.2, Theorem 12](https://www.jmlr.org/papers/volume25/21-1440/21-1440.pdf) | Defines counterfactual models using couplings with marginal, identity and symmetry requirements. Its OT/structural equivalence result assumes an acyclic model, relative exogeneity, injective mechanisms, density/moment conditions, and a structural counterfactual map that is itself a convex gradient. | Cite the formulation and conditional connection, not a blanket equation of geometric and causal transport. Those assumptions have not been established for our embedding/state construction. |
| [OT via ICNNs](../papers/makkuva2020ot-icnn.md): [proceedings abstract](https://proceedings.mlr.press/v119/makkuva20a.html) | Learns transport between sampled distributions using two convex functions and a minimax objective. | Our current `ot_editor` uses one target-conditioned convex potential and semantic loss, not this distribution-fitting algorithm. Cite the convex-map connection without calling our editor a reproduction of Makkuva et al. |
| [Minibatch OT-CFM](../papers/tong2024ot-cfm.md): [§3.2.3, Proposition 3.4 and minibatch discussion](https://arxiv.org/html/2302.00482) | Uses coupled endpoints for conditional flow-matching regression. The exact dynamic-OT statement has coupling, regularity and vanishing-smoothing requirements; minibatches approximate the full coupling. | Supports our training ingredient. Regularized minibatch fitting and reusing a reference across states do not by themselves guarantee exact cross-state OT or the true unit-level counterfactual. |
| [Counterfactual Cocycles](../papers/dance2024counterfactual-cocycles.md): [v4, §3.1–3.2, Theorem 5, §4.2–5.1](https://arxiv.org/html/2405.13844v4) | Separates identity, path independence and distribution adaptedness. Factorized invertible maps enforce the first two; estimation targets conditional-distribution discrepancy without requiring a specified noise law. Identification additionally depends on the admissible model class and causal assumptions. | This is a direct transport-modeling neighbor, not only a cautionary reference. Our source-to-reference-to-target construction has a related algebra, but density fitting and outer state inference differ. Coherence of fixed-state maps is not proof of correct pairing or unseen-operation generalization. |
| [Dynamic-OT identification](../papers/desousaribeiro2025dynamic-ot.md): [v2, §4.2, §5, Appendix B.4](https://arxiv.org/html/2510.08294v2) | Studies identification using structured continuous transports; stated conditions include causal exogeneity, exact OT structure and density/support regularity. §5 uses reverse–forward flow inference. | Cite as investigating identification, not as a theorem validating our implementations. The multivariate monotonicity implication also needs scrutiny; see the concrete check below. |

## Specific theorem caveat: monotonicity under composition

**Independent algebraic check, not an audit of the whole dynamic-OT paper.** Its v2 Proposition 4.4 states that monotone mechanisms induce a monotone counterfactual composition. This implication does not hold without further restrictions.

Let \(f_0(u)=Au\), \(f_1(u)=Bu\), with
\(A=\operatorname{diag}(10,1)\) and \(B=\begin{pmatrix}1&2\\2&5\end{pmatrix}\).
Both matrices are symmetric positive definite, hence both maps are monotone. But
\(f_1\circ f_0^{-1}(x)=BA^{-1}x\), where
\(BA^{-1}=\begin{pmatrix}0.1&2\\0.2&5\end{pmatrix}\).
For \(v=(10,-1)\), \(v^\top BA^{-1}v=-7<0\).

Thus we should not rely on that proposition's unqualified composition claim. This does not assess every theorem in the paper; the short draft only notes its research direction. A stronger theoretical use would require checking for a correction or additional hypotheses.

## Current-project checks

Comparison targets were rechecked against the current [semantic-editor guide](../../experiments/semantic_editor.md), [OT guide](../../experiments/ot_baselines.md), and [state-flow architecture](../../architecture/latent_intervention.md#f1-state-conditioned-flow), not inferred from the older manuscript introduction.

- `semantic_editor` and `ot_editor` share a decoded-mixture KL plus movement, support and identity terms. They fit no paired counterfactual latent targets. Only their map classes differ.
- `state_flow` fits factual conditional likelihood; the documented latent-only path uses a source-state point estimate. Do not claim it propagates reader uncertainty identically to the editors or OT baselines.
- `ot_flow` couples reference samples and factual latents within a state, not factual and true counterfactual examples. `ot_gaussian` is diagonal in a shared full-dimensional PCA basis.
- The five-method no-paired-target statement concerns latent-renderer fitting. It must not be generalized to privileged oracle baselines or every text-decoder experiment in the repository.
- The documented current editor/OT interfaces restrict supported actions. Modular construction alone is not evidence of arbitrary-intervention generalization.

## Handoff boundaries

The short draft is ready for author review and later LaTeX conversion, not certified as an exhaustive survey or proof-checked theoretical contribution. Final bibliography versions and venue metadata still need reconciliation. Baseline implementation choices, theorem-level reliance, and empirical comparisons remain separate decisions.

No paper cards were deleted, reclassified, or marked as fully reviewed. This note is the canonical home of this focused comparison pass; the complete 207-record index and earlier 48-paper selection remain available.
