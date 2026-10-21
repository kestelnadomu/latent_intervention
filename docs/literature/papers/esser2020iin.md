---
id: esser2020iin
topics: ["concepts","editing"]
metadata_status: metadata-verified
reading_status: not-reviewed
aliases: ["IIN","Invertible Interpretation Network"]
added: 2026-10-19
updated: 2026-10-19
---

# A Disentangling Invertible Interpretation Network for Explaining Latent Representations

Patrick Esser; Robin Rombach; Björn Ommer. 2020. CVPR, pp. 9223–9232.

- [Proceedings](https://openaccess.thecvf.com/content_CVPR_2020/html/Esser_A_Disentangling_Invertible_Interpretation_Network_for_Explaining_Latent_Representations_CVPR_2020_paper.html)
- [Author project and method overview](https://compvis.github.io/iin/)

## Project relevance (triage)

A strong architectural precedent for read–operate–render: translate a frozen network's hidden representation into an interpretable factorization, modify a semantic factor, and invert the translation. The author examples retain a residual factor during semantic swaps.

## Notes and caveats

An invertible semantic interface is therefore not new by itself. Our external symbolic kernel, frozen document embeddings and structural-counterfactual benchmark require a more specific comparison. Invertibility alone does not establish that the retained residual equals the true exogenous state.

## Evidence status

CVPR metadata and the author's method/examples screened; no complete training-objective review. Checked on 2026-10-19. Targeted screening only, not a full-paper review; the comparison with our project is agent inference.

## Encountered in

- [2026-10-19 web search](../imports/2026-10-19-web-search.md) prompted by the updated Overleaf draft (commit `a99392e`). Newly discovered candidate, not a recovered prior chat citation.
