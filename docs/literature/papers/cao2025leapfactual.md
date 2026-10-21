---
id: cao2025leapfactual
topics: ["editing","causal"]
metadata_status: metadata-verified
reading_status: not-reviewed
aliases: ["LeapFactual","2510.14623"]
added: 2026-10-19
updated: 2026-10-19
---

# LeapFactual: Reliable Visual Counterfactual Explanation Using Conditional Flow Matching

Zhuo Cao; Xuan Zhao; Lena Krieger; Hanno Scharr; Ira Assent. 2025. NeurIPS.

- [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ac1af21780d24a1345774c4a2e383972-Abstract-Conference.html)
- [Preprint](https://arxiv.org/abs/2510.14623)

## Project relevance (triage)

A recent direct neighbor for flow-based counterfactual generation, using conditional flow matching for visual explanations. Its treatment of disagreement between learned and ground-truth class boundaries is relevant to decoder-based evaluation.

## Notes and caveats

Its target is a prediction-changing visual explanation, not necessarily the same individual's outcome under a structural intervention. Compare against `ot_flow` at the mechanism and evaluation level without equating the estimands.

## Evidence status

NeurIPS publication, arXiv metadata and abstract screened; method PDF inspected selectively. Checked on 2026-10-19. Targeted screening only, not a full-paper review; the comparison with our project is agent inference.

## Encountered in

- [2026-10-19 web search](../imports/2026-10-19-web-search.md) prompted by the updated Overleaf draft (commit `a99392e`). Newly discovered candidate, not a recovered prior chat citation.
