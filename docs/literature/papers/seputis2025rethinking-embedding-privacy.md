---
id: seputis2025rethinking-embedding-privacy
topics: ["encoders"]
metadata_status: metadata-verified
reading_status: not-reviewed
aliases: []
added: 2026-10-16
updated: 2026-10-16
---

# Rethinking the Privacy of Text Embeddings: A Reproducibility Study of "Text Embeddings Reveal (Almost) As Much As Text"

Dominykas Seputis, Yongkang Li, Karsten Langerak and Serghei Mihailov. 2025. arXiv preprint.

- [Primary source](https://arxiv.org/abs/2507.07700)

## Project relevance (triage)

Reproduction and sensitivity study of [vec2text](morris2023vec2text.md), relevant to expectations about embedding inversion and longer CV sequences.

## Notes and caveats

The authors report reproducing key Vec2Text results under their tested conditions and identify length sensitivity. Their experiments do not establish that edited CV latents can be faithfully inverted.

## Evidence status

Title, authors, year and abstract checked against arXiv on 2026-10-16; not a full-paper review.

## Encountered in

- Local Codex thread `01a0c8f8-810f-7c82-baeb-0070132e8408`, message `msg_0f7071987f5f6bc5016ad1d03671708191bb9ef79f55ec1745` — possible improvements to factual inversion.
