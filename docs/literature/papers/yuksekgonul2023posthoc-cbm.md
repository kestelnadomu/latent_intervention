---
id: yuksekgonul2023posthoc-cbm
topics: ["concepts"]
metadata_status: metadata-verified
reading_status: not-reviewed
aliases: ["PCBM","2205.15480"]
added: 2026-10-19
updated: 2026-10-19
---

# Post-hoc Concept Bottleneck Models

Mert Yuksekgonul; Maggie Wang; James Zou. 2023. ICLR; 2022 preprint.

- [Paper](https://openreview.net/pdf?id=nA5AZ8CEyow)
- [Preprint](https://arxiv.org/abs/2205.15480)

## Project relevance (triage)

Adds concept-based interpretation and model editing to pretrained representations. Important context for attaching a semantic reader to a frozen encoder rather than learning an interpretable backbone from scratch.

## Notes and caveats

Concept-based predictor editing and residual prediction are not the same as reconstructing an edited document embedding for an unchanged downstream model. Use together with the more directly comparable latent-editing papers.

## Evidence status

Paper/arXiv identity, publication information and abstract screened. Checked on 2026-10-19. Targeted screening only, not a full-paper review; the comparison with our project is agent inference.

## Encountered in

- [2026-10-19 web search](../imports/2026-10-19-web-search.md) prompted by the updated Overleaf draft (commit `a99392e`). Newly discovered candidate, not a recovered prior chat citation.
