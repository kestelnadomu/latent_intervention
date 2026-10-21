---
id: dominici2025counterfactual-cbm
topics: ["concepts","editing"]
metadata_status: metadata-verified
reading_status: not-reviewed
aliases: ["CF-CBM"]
added: 2026-10-19
updated: 2026-10-19
---

# Counterfactual Concept Bottleneck Models

Gabriele Dominici; Pietro Barbiero; Francesco Giannini; Martin Gjoreski; Giuseppe Marra; Marc Langheinrich. 2025. ICLR.

- [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0ff54b4ec4f70b3ae12c8621ca8a49f4-Abstract-Conference.html)

## Project relevance (triage)

Combines concept-based prediction with a learned counterfactual generator rather than a post-hoc search. Useful comparator for amortized concept-level counterfactual generation.

## Notes and caveats

The generator is trained jointly with the CBM and can change the prediction mechanism. That differs from preserving the encoder/downstream predictor while translating a separately specified symbolic operation.

## Evidence status

ICLR metadata and abstract checked. Checked on 2026-10-19. Targeted screening only, not a full-paper review; the comparison with our project is agent inference.

## Encountered in

- [2026-10-19 web search](../imports/2026-10-19-web-search.md) prompted by the updated Overleaf draft (commit `a99392e`). Newly discovered candidate, not a recovered prior chat citation.
