---
id: dance2026deconfounding-flows
topics: ["causal","editing"]
metadata_status: metadata-verified
reading_status: not-reviewed
aliases: ["DecFM","2605.07665"]
added: 2026-10-19
updated: 2026-10-19
---

# Debiased Counterfactual Generation via Flow Matching from Observations

Hugh Dance; Johnny Xi; Peter Orbanz; Benjamin Bloem-Reddy. 2026. arXiv preprint, first posted 2026-05-08; v1 screened.

- [Preprint](https://arxiv.org/abs/2605.07665)
- [Paper (version screened)](https://arxiv.org/html/2605.07665v1)

## Project relevance (triage)

Learns a deconfounding transport from an observational conditional outcome distribution to a population counterfactual distribution, using influence-function-corrected flow matching. A timely comparator for causal distribution transport.

## Notes and caveats

Section 2.1 targets a population regime under consistency, conditional exchangeability and positivity; it also distinguishes a noncausal rebalancing interpretation. These results do not identify the true paired counterfactual for each CV.

## Evidence status

Metadata, abstract, introduction and Section 2.1 screened. Preprint; estimator proofs and empirical claims not independently validated. Checked on 2026-10-19. Targeted screening only, not a full-paper review; the comparison with our project is agent inference.

## Encountered in

- [2026-10-19 web search](../imports/2026-10-19-web-search.md) prompted by the updated Overleaf draft (commit `a99392e`). Newly discovered candidate, not a recovered prior chat citation.
