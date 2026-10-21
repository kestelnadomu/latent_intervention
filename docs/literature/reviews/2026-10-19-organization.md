# Organization review — 2026-10-19

## Scope and sources

Reviewed the 182 existing paper cards, the three import reports, and the literature changes from initial-import commit `51dca46` to the current merged checkout `5968d04`. Read the current encoder, semantic-decoder, clean-editor, OT and text-realizer guides to organize the collection around the owner's planned paper, rather than the outdated Overleaf draft.

This is a **collection and classification review**, not a new search, full-paper review or novelty assessment. No paper was added, removed, merged, renamed or selected for exclusion. The existing 182 cards, their metadata/reading statuses and provenance, all earlier import reports and [unresolved leads](../unresolved.md) remain unchanged.

## What the other devices contributed

- The [Linux supplement](../imports/2026-10-16-linux.md) added **27 papers**: ten OT/convex-map/counterfactual-transport records, eight semantic/text-editing records, four flow/application records, two encoder records, two protocol records and one inversion record under the new primary classification.
- The [decoder-chat supplement](../imports/2026-10-16-current-device.md) added **13 papers**: four encoder/VAE records, eight text-realization records and Polyjuice under semantic/text editing.
- The Linux pass also updated **eight existing cards**: [vec2text](../papers/morris2023vec2text.md) received corrected, verified metadata; [cGNF](../papers/balgi2024cgnf.md), [RAVEL](../papers/huang2024ravel.md), [CAREFL](../papers/khemakhem2021carefl.md), [CITRIS](../papers/lippe2022citris.md), [counterfactual non-identifiability](../papers/nasr-esfahany2023nonidentifiability.md), [DSCM](../papers/pawlowski-dscm-2020.md) and [high-fidelity counterfactuals](../papers/sousaribeiro-fidelty-2023.md) gained provenance, links or caveats. These are not eight additional papers.

The main organizational change is to stop burying OT under generic causal inference and text generation under encoders. The additions also strengthen representation-editing comparisons and experimental-protocol cautions. They do not imply that every collected generator, editor or flow is an implemented project baseline.

## No-drop accounting

Counts below refer to primary homes, not cross-links. The original 142 records are also fully classified; this is not just a supplement index.

| Primary cluster | Initial Windows | Linux addition | Decoder addition | Total |
| --- | ---: | ---: | ---: | ---: |
| [01. Benchmarks, hiring and causal fairness](../index.md#01-benchmarks-and-fairness) | 15 | 0 | 0 | 15 |
| [02. Fair and invariant representations](../index.md#02-fair-representations) | 39 | 0 | 0 | 39 |
| [03. Frozen text encoders and language VAEs](../index.md#03-text-encoders) | 12 | 2 | 4 | 18 |
| [04. Semantic decoding, concepts and probing](../index.md#04-semantic-decoding) | 9 | 0 | 0 | 9 |
| [05. Causal abstraction and neurosymbolic alignment](../index.md#05-causal-abstraction) | 16 | 0 | 0 | 16 |
| [06. Semantic editing and controllable text](../index.md#06-semantic-editing) | 11 | 8 | 1 | 20 |
| [07. Causal generative models and state-conditioned flows](../index.md#07-causal-flows) | 10 | 4 | 0 | 14 |
| [08. Optimal transport and geometric counterfactual maps](../index.md#08-optimal-transport) | 3 | 10 | 0 | 13 |
| [09. Text realization, reconstruction and embedding inversion](../index.md#09-text-realization) | 2 | 1 | 8 | 11 |
| [10. Identification and causal representation learning](../index.md#10-identification) | 20 | 0 | 0 | 20 |
| [11. Evaluation validity and experimental protocol](../index.md#11-evaluation) | 5 | 2 | 0 | 7 |
| **Total** | **142** | **27** | **13** | **182** |

Metadata status remains **84 metadata-verified / 96 recovered / 2 needs-resolution**. All 182 retain `reading_status: not-reviewed`. In particular, being central to the proposed narrative does not promote a record's evidence status: even the LIBERTy card still needs its recorded metadata check.

## Important placement decisions

- **LIBERTy, CEBaB, Bias in Bios and PopResume:** benchmark/application home, with links toward concepts and evaluation. Do not conflate structural counterfactual supervision, human concept edits, demographic bias data and population representativeness.
- **OT has several roles:** ICNN and convex-map references support the geometry discussion for `ot_editor`; transport/coupling and dynamic references connect to `ot_flow`; Gaussian transport needs its own assumption/source check. Fairness transports remain connected to the broader fairness field. A shared OT label does not make these methods equivalent.
- **Architecture versus causal meaning:** BERT-flow is primarily encoder geometry, not evidence of causal identification. FairFlow/StyleFlow are in flow applications; causal text-generation papers including MATTE remain cross-linked from editing. CITRIS-NF stays within its existing CITRIS record.
- **Two decoders:** independent/autoregressive `g` belongs with concept recovery and probing; prefix/projector, LoRA, generative backbones and inversion belong with `D_text`. Existing records previously tagged `encoders` need not be rewritten to express this distinction.
- **Limits and controls remain visible:** abstraction critiques stay next to alignment methods; probe controls, concept leakage, adversarial counterexamples and selection bias have an evaluation home. Identification results require a separate assumptions check, not automatic transfer to frozen embeddings.
- **Historical directions remain represented:** distillation, direct flows, projected editors and alternative fairness objectives are not removed merely because the owner currently centers five other variants.

These decisions are editorial triage. Use each card's source links and caveats before asserting technical equivalence or a literature result.

## Unresolved leads and reading gaps

All original rows in [unresolved.md](../unresolved.md) are retained. Route the existing leads as follows; do not manufacture paper identities from a model name:

| Existing unresolved lead | Reading destination |
| --- | --- |
| Anonymous VAE manipulation; Lample multiple-attribute rewriting | [Semantic editing](../index.md#06-semantic-editing) |
| Tan/C3M identity | [Semantic decoding](../index.md#04-semantic-decoding) |
| Squires; von Kügelgen 2023; Zhang/Yao/Xu 2024 CRL references | [Identification](../index.md#10-identification) |
| DELLA/LD4LG; Sentence-T5; GPT-2/generic VAEs; EntailmentBank | [Encoders](../index.md#03-text-encoders), with [text realization](../index.md#09-text-realization) where relevant |
| Blog Authorship; TrustPilot; FYelp/Amazon, Yelp and GYAFC datasets | [Benchmark context](../index.md#01-benchmarks-and-fairness) and [text editing](../index.md#06-semantic-editing) |
| HSIC; energy score/distance | [Evaluation](../index.md#11-evaluation), with fairness/distributional-objective cross-links once the intended sources are identified |

The planned methods expose additional **coverage questions**, not new verified citations:

- **OT-flow foundations:** identify the exact conditional flow-matching, minibatch-OT and entropic/Sinkhorn sources appropriate to the implemented objective. Dynamic-OT counterfactual papers do not automatically supply all of those references.
- **Analytic Gaussian transport:** resolve a primary source for the restricted diagonal/shared-basis transport and its assumptions before using a general Gaussian-OT result as justification.
- **Semantic probability quality:** the concept/probe collection does not replace direct references for calibration, proper scoring rules and joint/autoregressive probability models.
- **Exact model releases and training domains:** the Gemma 3 report on file covers 1B–27B, not 270M. Match backbone/checkpoint sources to the versions actually used; the EntailmentBank lead and LangVAE checkpoint evidence serve different purposes.
- **Metrics and claims:** resolve the existing energy-score/distance lead if it is used in the paper. Keep same-decoder semantic agreement, independent semantic checks, latent recovery, support and text reconstruction separate.

No new candidate papers were invented to fill these gaps. The retrieval limitations in the original import reports also remain in force.

## Integrity checks

Validated: **182 unique card IDs and 182 primary entries**, exact coverage of all 40 additions, cluster/count totals, and unchanged index citation/status lines. All **267 distinct local link references** checked in the six changed/new navigation documents resolve, including their anchors. `git diff --check` passes. The Git diff confirms that paper cards, unresolved leads and original import reports are unchanged, and all six changed/new files are inside `docs/literature/`. No training, downloads, staging, commits or manuscript edits were performed. This does not re-certify external URLs or historical device-specific links inside unchanged cards.

Next phase, only when requested: read and compare candidates using the common questions in [the narrative map](../related-work-map.md), then select and condense citations for the paper. This pass intentionally leaves that choice open.
