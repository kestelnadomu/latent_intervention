---
id: hu2021lora
topics: ["encoders"]
metadata_status: metadata-verified
reading_status: not-reviewed
aliases: []
added: 2026-10-16
updated: 2026-10-16
---

# LoRA: Low-Rank Adaptation of Large Language Models

Edward J. Hu and colleagues. 2021. arXiv preprint.

- [Primary source](https://arxiv.org/abs/2106.09685)

## Project relevance (triage)

Parameter-efficient decoder adaptation used in the project's frozen-encoder text-realizer experiments and considered for BERT-vector-to-generator reconstruction.

## Notes and caveats

LoRA adapts the generator while leaving the encoder's embedding space fixed if that encoder is frozen. Its general results do not guarantee CV fact recovery.

## Evidence status

Title, lead author and preprint year checked against arXiv on 2026-10-16; not a full-paper review.

## Encountered in

- Local Codex thread `01a0c8f8-810f-7c82-baeb-0070132e8408`, message `msg_07103927c15d8fce016ab2c19fa0d087d29e81906aa232b1cb` — decoder fine-tuning options.
