---
id: reimers2019sentence-bert
topics: ["encoders"]
metadata_status: metadata-verified
reading_status: not-reviewed
aliases: ["SBERT"]
added: 2026-10-16
updated: 2026-10-16
---

# Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

Nils Reimers and Iryna Gurevych. 2019. EMNLP-IJCNLP.

- [Primary source](https://aclanthology.org/D19-1410/)

## Project relevance (triage)

Example of an encoder trained for useful sentence representations without a document-reconstruction decoder, discussed when clarifying the role of BERT features in the project.

## Notes and caveats

Sentence-BERT is not the project's EmbeddingGemma or LangVAE encoder. Similarity utility does not imply text invertibility.

## Evidence status

Title, authors and venue year checked against ACL Anthology on 2026-10-16; not a full-paper review.

## Encountered in

- Local Codex thread `01a0c8f8-810f-7c82-baeb-0070132e8408`, message `msg_09af1393376ec8a3016ab389c53ea087d2b7fa37fdfbe96989` — encoder-only models.
