---
id: lipman2023flow-matching
topics: ["editing"]
metadata_status: metadata-verified
reading_status: not-reviewed
aliases: ["Flow Matching","2210.02747"]
added: 2026-10-19
updated: 2026-10-19
---

# Flow Matching for Generative Modeling

Yaron Lipman; Ricky T. Q. Chen; Heli Ben-Hamu; Maximilian Nickel; Matt Le. 2023. ICLR; 2022 preprint.

- [Paper (version screened)](https://arxiv.org/html/2210.02747v2)
- [Preprint metadata](https://arxiv.org/abs/2210.02747)
- [Conference listing](https://iclr.cc/Downloads/2023)

## Project relevance (triage)

Core training foundation for `ot_flow`: learn a continuous flow by regressing conditional probability-path vector fields, without simulating the ODE during training.

## Notes and caveats

The conditioning used to derive flow-matching objectives is not itself semantic-state conditioning. Straight conditional paths do not guarantee a globally optimal source–target coupling or causal individual correspondence.

## Evidence status

arXiv metadata, conference listing and Sections 3–4 screened. Checked on 2026-10-19. Targeted screening only, not a full-paper review; the comparison with our project is agent inference.

## Encountered in

- [2026-10-19 web search](../imports/2026-10-19-web-search.md) prompted by the updated Overleaf draft (commit `a99392e`). Newly discovered candidate, not a recovered prior chat citation.
