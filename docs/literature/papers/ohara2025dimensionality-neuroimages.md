---
id: ohara2025dimensionality-neuroimages
topics: ["causal", "encoders"]
metadata_status: metadata-verified
reading_status: not-reviewed
aliases: []
added: 2026-10-16
updated: 2026-10-16
---

# Dimensionality reduction in 3D causal deep learning for neuroimage generation: an evaluation study

Erik Y. Ohara; Vibujithan Vigneshwaran; Raissa Souza; Finn G. Vamosi; Matthias Wilms; Nils D. Forkert. 2025. Journal of Medical Imaging 12(2):024506.

- [Primary metadata and abstract](https://pubmed.ncbi.nlm.nih.gov/40276097/)
- [Open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12014944/)

## Project relevance (triage)

Cross-domain example of compressing MRI images before flow-based causal modeling and generating counterfactuals. It informs the role of representation quality, but its experiment evaluates neuroimage generation rather than establishing our text-embedding counterfactuals.

## Notes and caveats

An earlier chat shorthand treated this as a general flow-identification precedent. The linked work is specifically a dimensionality-reduction comparison for neuroimages; do not import its abstract's “true counterfactual” wording as a guarantee in our setting.

## Evidence status

Title, authors, journal/year and abstract checked against PubMed on 2026-10-16; not a full-paper review.

## Encountered in

- Local Codex `chat:01a1159d-d5b0-7ca0-99e0-337529ac542e:msg_0bb445122f277c20016ac60c78ab0c81918a97ced4494386fc` — cross-domain flow comparison.
