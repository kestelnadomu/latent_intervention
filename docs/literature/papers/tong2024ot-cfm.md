---
id: tong2024ot-cfm
topics: ["editing"]
metadata_status: metadata-verified
reading_status: not-reviewed
aliases: ["OT-CFM","2302.00482"]
added: 2026-10-19
updated: 2026-10-19
---

# Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport

Alexander Tong; Kilian Fatras; Nikolay Malkin; Guillaume Huguet; Yanlei Zhang; Jarrid Rector-Brooks; Guy Wolf; Yoshua Bengio. 2024. TMLR; 2023 preprint.

- [Preprint and publication metadata](https://arxiv.org/abs/2302.00482)
- [Published paper in author institution repository](https://www.pure.ed.ac.uk/ws/portalfiles/portal/472344876/TongEtalTMLR2024ImprovingAndGeneralizingFlow-based.pdf)

## Project relevance (triage)

The most directly relevant training reference for minibatch OT conditional flow matching. Explains how endpoint couplings enter the regression objective used to learn transport dynamics.

## Notes and caveats

The dynamic-OT connection assumes access to the appropriate OT plan. Finite minibatches, a learned approximation, and composition through a shared reference do not automatically give a globally optimal cross-state map.

## Evidence status

arXiv v4 and published-paper metadata/abstract checked; use the final author order rather than an earlier submission. Checked on 2026-10-19. Targeted screening only, not a full-paper review; the comparison with our project is agent inference.

## Encountered in

- [2026-10-19 web search](../imports/2026-10-19-web-search.md) prompted by the updated Overleaf draft (commit `a99392e`). Newly discovered candidate, not a recovered prior chat citation.
