---
id: ravfogel2020inlp
topics: ["fairness","editing"]
metadata_status: metadata-verified
reading_status: not-reviewed
aliases: ["INLP","2004.07667","10.18653/v1/2020.acl-main.647"]
added: 2026-10-19
updated: 2026-10-19
---

# Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection

Shauli Ravfogel; Yanai Elazar; Hila Gonen; Michael Twiton; Yoav Goldberg. 2020. ACL.

- [Proceedings](https://aclanthology.org/2020.acl-main.647/)
- [Preprint](https://arxiv.org/abs/2004.07667)

## Project relevance (triage)

A key missing predecessor to the existing LEACE/concept-erasure shelf: iterative nullspace projection removes information accessible to linear attribute classifiers from an existing representation.

## Notes and caveats

Erasure differs from changing an attribute to a specified target while retaining its permitted downstream consequences. Linear removal is not a guarantee against arbitrary nonlinear recovery or a counterfactual-fairness theorem.

## Evidence status

Author/arXiv metadata and abstract checked against the ACL record. Checked on 2026-10-19. Targeted screening only, not a full-paper review; the comparison with our project is agent inference.

## Encountered in

- [2026-10-19 web search](../imports/2026-10-19-web-search.md) prompted by the updated Overleaf draft (commit `a99392e`). Newly discovered candidate, not a recovered prior chat citation.
