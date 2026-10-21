---
id: koh2020concept-bottleneck
topics: ["concepts"]
metadata_status: metadata-verified
reading_status: not-reviewed
aliases: ["CBM"]
added: 2026-10-19
updated: 2026-10-19
---

# Concept Bottleneck Models

Pang Wei Koh; Thao Nguyen; Yew Siang Tang; Stephen Mussmann; Emma Pierson; Been Kim; Percy Liang. 2020. ICML, PMLR 119, pp. 5338–5348.

- [Proceedings](https://proceedings.mlr.press/v119/koh20a.html)

## Project relevance (triage)

Foundational context for predicting human-readable concepts and correcting them at inference before predicting an outcome. Introduces the reader/intervention interface that later post-hoc methods generalize.

## Notes and caveats

A concept bottleneck used for prediction need not render an edited vector back into the original embedding space. Correcting a concept prediction is not automatically a causal intervention on its descendants.

## Evidence status

ICML metadata and abstract checked. Checked on 2026-10-19. Targeted screening only, not a full-paper review; the comparison with our project is agent inference.

## Encountered in

- [2026-10-19 web search](../imports/2026-10-19-web-search.md) prompted by the updated Overleaf draft (commit `a99392e`). Newly discovered candidate, not a recovered prior chat citation.
