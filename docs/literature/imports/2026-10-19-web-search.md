# Targeted web search for the revised paper

Date: 2026-10-19. Scope: additional relevant literature absent from the existing 182-card library. Result: **25 new, metadata-verified records; 207 records in total**. No earlier record was removed or replaced. This is a targeted discovery pass and reading-priority assessment, not an exhaustive systematic review or a final bibliography.

## What guided the search

Read the updated nested Overleaf checkout at commit `a99392e` (main abstract, introduction, related-work outline, method, experiments and discussion). The method now foregrounds **read–operate–render**, with a frozen encoder, semantic reader, separately specified symbolic operation, latent renderer and frozen downstream predictor. The draft remains incomplete and does not yet describe every planned method.

The search also retained the owner's five-method scope: `semantic_editor`, `ot_editor`, `state_flow`, `ot_flow`, `ot_gaussian`, with encoder comparisons and separate semantic/text decoders. See the [working map](../related-work-map.md) for the repository terminology. Draft aspirations such as exchanging arbitrary operations are not treated as experimentally established capabilities.

Search families: post-hoc concept intervention in frozen representations; invertible semantic interpretation and latent translation; conditional-flow attribute editing; counterfactual/OT flow matching; dependent concept uncertainty; counterfactual identification; controlled text generation and fairness; calibration and distributional scoring. Follow-up searches checked publication identities and selected method sections using proceedings, author pages and arXiv. No new encoder family was selected.

## Closest additions to read first

The priorities below are agent assessment of overlap with our framing, not a claim that these papers solve the same problem.

| Work | Why it matters for us | Difference to establish in the paper |
| --- | --- | --- |
| [Laguna et al. (2024), Beyond CBMs](../papers/laguna2024beyond-cbm.md) | Frozen-probe latent intervention with concept loss plus movement cost; especially close to `semantic_editor`. | Per-input optimization versus amortized editing; corrected concepts versus a symbolic target distribution. |
| [Esser et al. (2020), IIN](../papers/esser2020iin.md) | Invertible translation into editable semantic factors and back, on top of existing networks. | External causal operations and document embeddings versus learned interpretable image factors. |
| [Kulkarni et al. (2025), post-hoc generative CBMs](../papers/kulkarni2025posthoc-generative-cbm.md) | Concept editing and reconstruction with a frozen generative backbone. | Arbitrary document embeddings versus a generator's own latent space. |
| [Abdal et al. (2021), image StyleFlow](../papers/abdal2021styleflow-images.md) | Conditional-flow inversion followed by rendering with changed attributes; close to `state_flow`. | Attribute control versus a structural intervention and verified individual recovery. |
| [Cao et al. (2025), LeapFactual](../papers/cao2025leapfactual.md) | Conditional flow matching for visual counterfactual explanations. | Prediction-changing explanations versus SCM-defined counterfactual counterparts. |
| [Rombach et al. (2020), Net2Net](../papers/rombach2020net2net.md) | Connects frozen representations with an invertible adapter. | Cross-network translation versus our semantic operator and text reconstruction. |
| [Nangi et al. (2021), latent text style transfer](../papers/nangi2021counterfactual-text-style.md) | Explicit counterfactual modification of text latents. | Learned style/content spaces versus frozen general-purpose encoders. |
| [Chen and Du (2025), exogenous isomorphism](../papers/chen2025exogenous-isomorphism.md) | Addresses when exogenous-coordinate agreement supports counterfactual identification. | Whether our actual model satisfies the required restrictions; not just invertibility. |

Primary-source details and the extent of screening are in each linked card. Particularly useful checks were Laguna et al., Section 3.1 Eq. (1), and StyleFlow's reverse/forward inference description. These support close comparisons, not a conclusion that our complete method is identical.

## Important recent preprints

- [Dance et al. (May 2026), deconfounding flows](../papers/dance2026deconfounding-flows.md): observational-to-population-counterfactual transport. Its estimand and assumptions differ from recovering the same person's latent counterfactual.
- [Guan et al. (October 2026), coupling-sensitive rates](../papers/guan2026counterfactual-flow-rates.md): distributional guarantees depend on coupling and nuisance estimation. The sampler and assumptions differ from our current flow.
- [Hoffmann et al. (2025; revised March 2026), post-hoc stochastic CBMs](../papers/hoffmann2025posthoc-stochastic-cbm.md): a lightweight dependent-uncertainty extension for a pretrained concept bottleneck. Venue not verified.

These are worth monitoring, but their results are not validated guarantees for this project. Abstracts and selected problem/assumption sections were screened, not the proofs.

## Additional foundations and adjacent comparisons

- **Semantic reader:** [Koh et al. (2020)](../papers/koh2020concept-bottleneck.md), [Yuksekgonul et al. (2023)](../papers/yuksekgonul2023posthoc-cbm.md), [Vandenhirtz et al. (2024)](../papers/vandenhirtz2024stochastic-cbm.md) and [Dominici et al. (2025)](../papers/dominici2025counterfactual-cbm.md). Distinguish concept prediction, uncertainty, model editing and joint counterfactual-generator training.
- **Causal interpretation:** [Rubenstein et al. (2017)](../papers/rubenstein2017causal-consistency.md) and [Xia et al. (2023)](../papers/xia2023neural-counterfactual-identification.md). Formal abstraction and identification are stronger than satisfying a learned consistency loss.
- **OT-flow training:** [Lipman et al. (2023)](../papers/lipman2023flow-matching.md) and [Tong et al. (2024)](../papers/tong2024ot-cfm.md). These are essential ingredient citations if the paper presents `ot_flow`, not competitors with the same causal estimand.
- **Analytic baseline:** [Takatsu (2011)](../papers/takatsu2011gaussian-wasserstein.md) provides Gaussian Wasserstein background; our diagonal/common-coordinate restriction must remain explicit.
- **Text and fairness:** [Wang et al. (2023), COFFEE](../papers/wang2023coffee.md), [Liu et al. (2024), MAGIC](../papers/liu2024magic.md) and [Ravfogel et al. (2020), INLP](../papers/ravfogel2020inlp.md). These connect personalized generation, correlated-attribute control and erasure, respectively; none should be described as the same objective.
- **Evaluation:** [Guo et al. (2017)](../papers/guo2017calibration.md) and [Gneiting and Raftery (2007)](../papers/gneiting2007proper-scoring.md). Support calibration and distributional scoring without claiming those scores alone establish causal validity.

## Implications for positioning — assessment, not manuscript text

1. **Do not pitch read–operate–render, frozen-backbone concept editing, or invertible attribute editing as new in isolation.** The closest additions supply concrete prior implementations of those ingredients.
2. **A more defensible contribution to investigate is the combination:** an explicit symbolic/causal operation translated into frozen text representations; a controlled comparison of semantic-target and distribution-based renderers; and evaluation against structural ground truth across encoder/decoder choices. This is a positioning hypothesis, not a verified first-of-its-kind claim.
3. **Keep estimands separate.** Concept correction, classifier recourse, attribute control, population intervention distributions and unit-level structural counterfactuals are related but not interchangeable.
4. **State what selects the correspondence.** Density fit, a chosen geometric coupling and semantic agreement impose different constraints. None alone proves the correct individual pairing. The new identification papers complement the existing non-identifiability and cocycle records.
5. **Keep architectural flexibility distinct from demonstrated generalization.** A modular interface does not establish successful unseen-operation composition. The paper should connect each claim to its training support and actual experiments.

Existing key neighbors remain important: [DSCM](../papers/pawlowski-dscm-2020.md), [counterfactual cocycles](../papers/dance2024counterfactual-cocycles.md), the OT/CellOT papers and the existing [editing shelf](../index.md#06-semantic-editing). This search adds gaps; it does not replace that collection.

## Coverage and verification

- Deduplicated against all 182 existing cards by normalized title, recorded arXiv/DOI identifiers and aliases; all 25 additions have distinct records.
- Kept image StyleFlow distinct from the existing text StyleFlow. Kept publication years separate from preprint/revision dates; Takatsu's 2011 article is not silently merged with the similarly titled 2010 chapter.
- Used primary-source metadata. Some OpenReview forum pages were unavailable behind browser verification; accessible paper/proceedings/arXiv sources were used instead.
- Every new card remains `reading_status: not-reviewed`, with its actual screening extent recorded. No full-paper or theorem audit was performed.
- The index retains one primary entry per record. Updated counts are 207 total: 109 metadata-verified, 96 recovered, 2 needs-resolution. Earlier import/organization counts remain historical snapshots.
- Validation: all 207 cards have exactly one primary index entry; no dangling entries or missing cards. All 425 local file links checked across the new cards and updated navigation/report files resolve; `git diff --check` passed.
- Only `docs/literature/` was changed. No code, manuscript, original bibliography, branch, staging or commit changes.

## Suggested next reading pass

Read the first four architectural/objective neighbors in full, then compare their supervision, frozen components, conditioning, losses and guarantees against our five implementations. Next examine exogenous-isomorphism assumptions and the two recent causal-flow preprints. Select citations only after that comparison; no collected paper needs to be dropped now.
