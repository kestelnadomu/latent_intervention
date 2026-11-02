---
id: devlin2019bert
topics: ["encoders"]
metadata_status: metadata-verified
reading_status: not-reviewed
aliases: []
added: 2026-10-16
updated: 2026-10-16
---

# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. 2019. NAACL.

- [Primary source](https://aclanthology.org/N19-1423/)

## Project relevance (triage)

Encoder background for the earlier discussion of BERT-derived 768-dimensional representations and whether a sentence embedding has an associated text-generation decoder.

## Notes and caveats

BERT's pretraining objective is not document reconstruction from a single pooled vector. This record does not describe the specific LangVAE checkpoint or the project's frozen EmbeddingGemma encoder.

## Evidence status

Title, authors and venue year checked against ACL Anthology on 2026-10-16; not a full-paper review.

## Encountered in

- Local Codex thread `01a0c8f8-810f-7c82-baeb-0070132e8408`, message `msg_09af1393376ec8a3016ab389c53ea087d2b7fa37fdfbe96989` — discussion of standalone encoders and decoders.
