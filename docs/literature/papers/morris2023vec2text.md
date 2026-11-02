---
id: morris2023vec2text
topics: ["encoders"]
metadata_status: metadata-verified
reading_status: not-reviewed
aliases: ["vec2text"]
added: 2026-10-16
updated: 2026-10-16
---

# Text Embeddings Reveal (Almost) As Much As Text

John Morris; Volodymyr Kuleshov; Vitaly Shmatikov; Alexander Rush. 2023. EMNLP.

- [Publisher / author source](https://aclanthology.org/2023.emnlp-main.765/)

## Project relevance (triage)

Background for encoder choices, controllability and information retained in text embeddings.

## Notes and caveats

vec2text precedent. Inversion of factual embeddings is different from reliable decoding of edited, potentially off-support latents.

## Evidence status

Title, authors, year and venue checked against ACL Anthology on 2026-10-16; not a full-paper review.

## Encountered in

- [docs/experiments/embeddinggemma_text_inversion_handoff.md](<../../../docs/experiments/embeddinggemma_text_inversion_handoff.md>).
- Local Codex `chat:01a0f28d-f6ea-7232-ab07-862e7335569f:msg_09fb1a8140bd96f0016acca28cb4f08191a73bdb41173c2584` — embedding-inversion feasibility discussion.
