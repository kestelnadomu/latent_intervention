---
id: laguna2024beyond-cbm
topics: ["concepts","editing"]
metadata_status: metadata-verified
reading_status: not-reviewed
aliases: ["Beyond CBMs","2401.13544"]
added: 2026-10-19
updated: 2026-10-19
---

# Beyond Concept Bottleneck Models: How to Make Black Boxes Intervenable?

Sonia Laguna; Ričards Marcinkevičs; Moritz Vandenhirtz; Julia E. Vogt. 2024. NeurIPS.

- [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9a439efaa34fe37177eba00737624824-Abstract-Conference.html)
- [Paper (version screened)](https://arxiv.org/html/2401.13544v3)

## Project relevance (triage)

A particularly close precedent for `semantic_editor`: Section 3.1, Eq. (1), edits an existing latent vector by balancing a concept-probe loss against distance from the original vector. The edited representation is passed to the existing predictor.

## Notes and caveats

Our comparison should distinguish per-example optimization from an amortized editor and corrected concepts from an externally specified symbolic target distribution. The paper also studies fine-tuning for intervenability; do not describe every variant as requiring encoder retraining. Probe agreement is not identification of a structural counterfactual.

## Evidence status

Proceedings metadata and Sections 3.1/3.3 of arXiv v3 screened. Checked on 2026-10-19. Targeted screening only, not a full-paper review; the comparison with our project is agent inference.

## Encountered in

- [2026-10-19 web search](../imports/2026-10-19-web-search.md) prompted by the updated Overleaf draft (commit `a99392e`). Newly discovered candidate, not a recovered prior chat citation.
