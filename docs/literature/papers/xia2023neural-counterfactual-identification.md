---
id: xia2023neural-counterfactual-identification
topics: ["causal","identifiability"]
metadata_status: metadata-verified
reading_status: not-reviewed
aliases: ["2210.00035"]
added: 2026-10-19
updated: 2026-10-19
---

# Neural Causal Models for Counterfactual Identification and Estimation

Kevin Xia; Yushu Pan; Elias Bareinboim. 2023. ICLR; 2022 preprint.

- [Paper](https://openreview.net/pdf?id=vouQcZS8KfW)
- [Preprint](https://arxiv.org/abs/2210.00035)

## Project relevance (triage)

Separates counterfactual identification from numerical estimation with neural causal models. Useful for explaining why fitting a flexible latent model is not sufficient to justify a particular counterfactual answer.

## Notes and caveats

Identification depends on the causal assumptions and available observational/experimental distributions. Do not transfer its guarantees to our unstructured conditional flow without establishing the required model and query conditions.

## Evidence status

arXiv/paper metadata and abstract screened; identification results not fully reviewed. Checked on 2026-10-19. Targeted screening only, not a full-paper review; the comparison with our project is agent inference.

## Encountered in

- [2026-10-19 web search](../imports/2026-10-19-web-search.md) prompted by the updated Overleaf draft (commit `a99392e`). Newly discovered candidate, not a recovered prior chat citation.
