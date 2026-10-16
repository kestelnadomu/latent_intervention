---
id: raffel2020t5
topics: ["encoders"]
metadata_status: metadata-verified
reading_status: not-reviewed
aliases: []
added: 2026-10-16
updated: 2026-10-16
---

# Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

Colin Raffel and colleagues. 2020. Journal of Machine Learning Research. Introduces T5.

- [Primary source](https://www.jmlr.org/papers/v21/20-074.html)

## Project relevance (triage)

Pretrained sequence-to-sequence alternative raised in the discussion of decoders that could be conditioned on a frozen BERT-style vector.

## Notes and caveats

T5's encoder–decoder pretraining does not supply a pretrained inverse for this project's chosen single-vector encoder.

## Evidence status

Title, lead author and publication year checked against JMLR on 2026-10-16; not a full-paper review.

## Encountered in

- Local Codex thread `01a0c8f8-810f-7c82-baeb-0070132e8408`, message `msg_09af1393376ec8a3016ab389c53ea087d2b7fa37fdfbe96989` — decoder alternatives.
