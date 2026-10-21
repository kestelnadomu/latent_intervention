---
id: gneiting2007proper-scoring
topics: ["benchmarks"]
metadata_status: metadata-verified
reading_status: not-reviewed
aliases: ["10.1198/016214506000001437","Energy score"]
added: 2026-10-19
updated: 2026-10-19
---

# Strictly Proper Scoring Rules, Prediction, and Estimation

Tilmann Gneiting; Adrian E. Raftery. 2007. Journal of the American Statistical Association 102(477), pp. 359–378.

- [Publisher](https://doi.org/10.1198/016214506000001437)
- [Author-hosted paper](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)

## Project relevance (triage)

Foundational source for proper probabilistic scoring, including the multivariate energy score. Relevant when evaluating a distribution of generated latents rather than only its mean.

## Notes and caveats

Propriety concerns expected scores under the appropriate distribution and moment conditions; a finite benchmark score does not prove causal identification. This resolves an energy-score source, not every energy-distance or HSIC objective mentioned in the notes.

## Evidence status

Publisher/author metadata and abstract checked; complete scoring-rule derivations not reviewed. Checked on 2026-10-19. Targeted screening only, not a full-paper review; the comparison with our project is agent inference.

## Encountered in

- [2026-10-19 web search](../imports/2026-10-19-web-search.md) prompted by the updated Overleaf draft (commit `a99392e`). Newly discovered candidate, not a recovered prior chat citation.
