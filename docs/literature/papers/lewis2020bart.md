---
id: lewis2020bart
topics: ["encoders"]
metadata_status: metadata-verified
reading_status: not-reviewed
aliases: []
added: 2026-10-16
updated: 2026-10-16
---

# BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension

Mike Lewis and colleagues. 2020. ACL.

- [Primary source](https://aclanthology.org/2020.acl-main.703/)

## Project relevance (triage)

Encoder–decoder model background for the discussion of alternatives to GPT-2 when mapping a frozen text embedding into a CV generator.

## Notes and caveats

Pretrained BART reconstructs corrupted text supplied to its own encoder; it is not pretrained to invert this project's standalone 768-dimensional embedding.

## Evidence status

Title, lead author and venue year checked against ACL Anthology on 2026-10-16; not a full-paper review.

## Encountered in

- Local Codex thread `01a0c8f8-810f-7c82-baeb-0070132e8408`, message `msg_09af1393376ec8a3016ab389c53ea087d2b7fa37fdfbe96989` — decoder alternatives.
