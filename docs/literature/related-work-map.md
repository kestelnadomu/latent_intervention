# Working narrative and method-to-literature map

Reorganized on 2026-10-19 around the owner's planned methods, then supplemented by a [targeted web search](imports/2026-10-19-web-search.md) informed by the revised Overleaf draft. This is an organizing proposal, not manuscript prose or a final citation shortlist. All 207 papers remain in the [complete topic index](index.md); examples here are navigation links. The subsequent [distilled review draft](distilled-review.md) provides the owner-requested provisional selection and field-organized paragraphs.

## Updated draft and newly found neighbors

The inspected draft (`overleaf` commit `a99392e`, 2026-10-19) now uses read–operate–render. Its encoder `phi` corresponds to the frozen text encoder; its reader `psi` to semantic decoder `g`; its symbolic operation `omega_S` to `h_S`; and its renderer `rho` to the latent editing/transport module. This renderer is **not** the separate text decoder `D_text`.

The [web-search report](imports/2026-10-19-web-search.md#closest-additions-to-read-first) adds close post-hoc concept-editing and invertible-interpretation comparisons, plus state-flow/image-editing and counterfactual-flow-matching neighbors. It also adds probability-calibration, scoring and identification references. Existing topic clusters remain intact.

Positioning implication (agent assessment): novelty should not rest on frozen-backbone semantic editing or invertibility alone. Examine the specific combination of an external symbolic operation, frozen document embeddings, competing renderers and structural-ground-truth evaluation. Architectural modularity is not evidence of generalization to unseen operations, and population-distribution guarantees are not individual-counterfactual identification. These are reading questions for the next pass, not edits to the manuscript.

## Narrative to build toward

Working framing: **given a frozen text representation and an explicit symbolic intervention, compare ways to construct its latent counterpart, while assessing semantic correctness, counterfactual recovery and text realization separately.**

The benchmark enables these comparisons; it does not make a fitted transport causally identified by default. Encoder and decoder choices are part of the experimental question, not interchangeable implementation details.

A possible reading/writing sequence is:

1. **Why this problem and this benchmark?** Use [benchmark and causal-fairness literature](index.md#01-benchmarks-and-fairness) to motivate hiring and distinguish structural counterfactuals, concept-edited text and population-representative evaluation. Connect to [fair representations](index.md#02-fair-representations): suppressing sensitive information, enforcing invariance and making an explicit causal edit are different goals. LIBERTy is the experimental anchor, not one of our editing algorithms.
2. **What is being edited and what counts as the intended change?** Connect [frozen encoders](index.md#03-text-encoders), [semantic decoding](index.md#04-semantic-decoding) and [causal abstraction](index.md#05-causal-abstraction). The decoder maps a vector to a state distribution; the symbolic kernel specifies how it should change. Recoverable concepts, calibrated probabilities and faithful symbolic alignment are distinct requirements.
3. **Semantic-target editing: flexible versus OT-structured maps.** Place `semantic_editor` and `ot_editor` next to [representation editing](index.md#06-semantic-editing), with the latter also linked to [convex OT map structure](index.md#08-optimal-transport). Both pursue the same semantic target; their map classes differ. Activation steering and source-text editing are neighboring tasks, not necessarily direct baselines.
4. **Distribution-based transport: likelihood, flow matching and an analytic approximation.** Place `state_flow`, `ot_flow` and `ot_gaussian` between [causal generative models](index.md#07-causal-flows) and [OT](index.md#08-optimal-transport). Explain what selects the cross-state correspondence: shared noise/reference coordinates, geometric preferences or a Gaussian approximation. Good factual distribution fit does not establish the correct individual pairing.
5. **Can the result be rendered as text?** Use [inversion and text realization](index.md#09-text-realization) to separate factual reconstruction, decoding an oracle counterfactual embedding and rendering a predicted edited embedding. Fluent or plausible output alone does not establish correct counterfactual content.
6. **What evidence and guarantees are possible?** Carry [identification assumptions](index.md#10-identification) and [evaluation validity](index.md#11-evaluation) through the comparison. Paired latent recovery, semantic scores, independent checks, support diagnostics and rendered-text fidelity provide different evidence.

These are narrative blocks for later condensation, not six fixed subsections. The eleven library clusters are deliberately broader so no collected paper has to be excluded now.

## The five methods and their literature roles

This table summarizes the current project design from the repository guides below. It does not attribute our implementation to a cited paper or assert that it performs better.

| Method | What defines the edit? | Literature connections | Boundary to keep explicit |
| --- | --- | --- | --- |
| `semantic_editor` | A target distribution from frozen `g` composed with `h_S` weights target-conditioned residual transformer maps; training compares their decoded mixture with that target. | [Editing](index.md#06-semantic-editing), [abstraction](index.md#05-causal-abstraction), [semantic decoding](index.md#04-semantic-decoding) | Mixture agreement through `g` does not certify each component, draw or individual counterpart. |
| `ot_editor` | The same target, objective and supervision, but each fixed-target map is an input-convex potential's gradient. | [OT/ICNN geometry](index.md#08-optimal-transport), [editing](index.md#06-semantic-editing) | OT-structured editing, without factual target-density matching. A convex gradient does not establish transport to the desired state distribution. |
| `state_flow` | Fit factual state-conditional densities by likelihood; invert under the source state and render under the target state with the same abducted noise. | [Causal flows](index.md#07-causal-flows), [identification](index.md#10-identification) | Cross-state noise alignment is a modeling assumption, not a consequence of likelihood fit alone. |
| `ot_flow` | Fit a state-conditioned continuous-time flow from a Gaussian reference by within-state minibatch OT flow matching; invert and reuse the reference across states. | [OT dynamics](index.md#08-optimal-transport), [causal flows](index.md#07-causal-flows) | Minibatch OT does not guarantee globally optimal learned endpoint transport or correct individual pairing. |
| `ot_gaussian` | Estimate state means and diagonal variances in a shared full-dimensional PCA basis after standardization; use analytic location/scale transport. | [OT foundations and restricted transport models](index.md#08-optimal-transport) | A restricted Gaussian approximation, not full-covariance Gaussian fitting, dimensionality reduction or a neural editor. |

Project sources: [clean semantic editors](../experiments/semantic_editor.md), [OT baselines](../experiments/ot_baselines.md), and [state-flow architecture](../architecture/latent_intervention.md).

Across these five methods, paired counterfactual embeddings are not training targets. Their inference protocols must still be stated separately: target-only editors use a target-state mixture; the OT baselines use coupled source/target-state weights; state flow can use an observed or inferred source state. Do not silently assume identical uncertainty propagation or oracle access.

`ot_editor` therefore bridges semantic editing and OT geometry; it is not interchangeable with the distribution-based OT baselines. Likewise, discrete affine-coupling flows and continuous-time flow matching share a transport viewpoint but not a fitting objective.

Earlier projected C-sym variants, direct semantic flows, teacher–student distillation and legacy transformer variants remain historical context, not additional central methods in this five-method narrative. Papers encountered in those discussions stay eligible for later selection.

## Benchmark, encoder and decoder coverage

| Component | Literature and experiment context | Comparison to make later |
| --- | --- | --- |
| **LIBERTy and CVs** | [LIBERTy](papers/toker-liberty-2026.md), [CEBaB](papers/abraham2022cebab.md), [Bias in Bios](papers/de2019bias.md), [PopResume](papers/yu2026popresume.md); [benchmark/fairness cluster](index.md#01-benchmarks-and-fairness) | What supplies states, structural mechanisms, paired texts, population coverage and semantic validation? These benchmark properties are not equivalent. |
| **Frozen encoder `f`** | LangVAE and its CV-adapted version; Nomic, Qwen3 Embedding and EmbeddingGemma; [encoder/VAE cluster](index.md#03-text-encoders) | Training domain/objective, length, retained information, dimension and geometry. Jina and Sentence-BERT are context, not automatically active experiment families. |
| **Semantic decoder `g`** | Independent versus autoregressive heads; [concept/probe literature](index.md#04-semantic-decoding) and [evaluation controls](index.md#11-evaluation) | Marginal versus joint state recovery, dependency assumptions, calibration and editor exploitation. Concept-bottleneck papers are neighboring work, not exact implementations of these heads. |
| **Text decoder `D_text`** | Native/adapted LangVAE decoding; EmbeddingGemma-to-Gemma prefix/projector and prefix-plus-LoRA experiments, including the 270M reference and 1B pilot; [realization/inversion literature](index.md#09-text-realization) | Native generative latent versus learned embedding inversion; factual-only versus counterfactual-exposed training; backbone, adapter and generation choices. BART/T5 are architectural leads, not completed inversion baselines merely because they are indexed. |

Current project context: [encoders](../architecture/text_encoder.md), [semantic decoder](../architecture/semantic_decoder.md), [text-realizer variants](../experiments/text_realizer.md), [Gemma 1B pilot](../experiments/text_realizer_gemma1b_results.md), and [counterfactual-exposure pilot](../experiments/text_realizer_counterfactual_exposure_plan.md). These guides, rather than the outdated manuscript, anchor this snapshot. Recheck them before drafting implementation claims.

Keep two decoder boundaries explicit:

- `g(Z)` predicts a structured-state distribution; `D_text(Z)` generates text. Improving one does not establish the other's quality.
- The counterfactual-exposed `D_text` pilot uses oracle `(Z', CV')` supervision and is a separate privileged comparator. It does **not** change the factual-only training claim for the five `h_Z` methods. Improvement on oracle vectors does not establish performance on estimated edited vectors.

The collected [Gemma 3 technical report](papers/gemma-team2025gemma3.md) covers the 1B–27B family, not the later 270M release. [Prefix-tuning](papers/li2021prefix-tuning.md) and [LoRA](papers/hu2021lora.md) are conditioning/adaptation context, not evidence of a pretrained inverse for our CV embeddings. Keep model-specific source gaps visible in the [organization review](reviews/2026-10-19-organization.md#unresolved-leads-and-reading-gaps).

## Connections to preserve when reading

These bridges keep the primary-home classification from hiding overlaps:

- **Fairness ↔ editing ↔ OT:** fairadapt, fairness-oriented OT and affine/concept interventions may compare closely while optimizing different targets. Connect clusters [02](index.md#02-fair-representations), [06](index.md#06-semantic-editing) and [08](index.md#08-optimal-transport).
- **Flow architecture ↔ causal interpretation:** Real NVP and flow reviews describe model classes; DSCM, CAREFL, cGNF and bijective models have distinct causal assumptions. BERT-flow belongs primarily with encoder geometry; FairFlow/StyleFlow with text applications. Connect [03](index.md#03-text-encoders), [07](index.md#07-causal-flows) and [10](index.md#10-identification).
- **OT structure ↔ identification:** ICNN/convex-map papers, CellOT, transport-based counterfactuals, cocycles and dynamic-OT identification address different questions. Check assumptions before transferring any guarantee to this factual-only setting. Connect [08](index.md#08-optimal-transport) and [10](index.md#10-identification).
- **Abstraction ↔ semantic evaluation:** IIT/DAS/CPM, concept bottlenecks, probe controls, RAVEL and abstraction critiques help test whether semantic scores reflect the intended mechanism. Connect [04](index.md#04-semantic-decoding), [05](index.md#05-causal-abstraction) and [11](index.md#11-evaluation).
- **Text editing ↔ latent-only inversion:** source-guided methods such as Polyjuice and inversion methods have different source-text, encoder-query and supervision access. Neither automatically validates decoding an off-support edit. Connect [06](index.md#06-semantic-editing) and [09](index.md#09-text-realization).
- **Evaluation ↔ all components:** scores through the training decoder are non-independent. Independent readers, held-out paired `Z'`, support/identity checks and text fidelity probe different failures. Model selection and repeated experiment inspection also belong in the protocol description.

## Questions for later paper-by-paper comparison

For each potentially cited work, record:

1. **Object and target:** source text, activations, frozen embeddings, learned generative latents or a decision; invariance, behavior control, population matching or individual counterfactuals?
2. **Training access:** factual data, states, paired counterfactuals, environments/interventions, encoder queries, pretrained generators or privileged simulator information?
3. **Inference access:** only `Z` and an action, or also text, states, a semantic decoder, a symbolic kernel or an oracle target?
4. **Correspondence assumption:** what fixes the cross-state pairing, and what is actually identifiable?
5. **Validation:** same-decoder consistency, independent semantics, paired recovery, support, preserved attributes, text fidelity and selection protocol?
6. **Project relationship:** shared ingredient, neighboring task, direct comparator or cautionary result—supported by a specific passage, not a title or chat summary?

No novelty, unbiasedness or causal-recovery claim follows from this organizing pass. Provisional selection is now recorded separately in [the distilled review draft](distilled-review.md); substantive reading, assumption checks and manuscript integration remain distinct steps.
