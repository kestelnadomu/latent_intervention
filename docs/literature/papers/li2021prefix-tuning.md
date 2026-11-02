---
id: li2021prefix-tuning
topics: ["encoders"]
metadata_status: metadata-verified
reading_status: not-reviewed
aliases: []
added: 2026-10-16
updated: 2026-10-16
---

# Prefix-Tuning: Optimizing Continuous Prompts for Generation

Xiang Lisa Li and Percy Liang. 2021. ACL.

- [Primary source](https://aclanthology.org/2021.acl-long.353/)

## Project relevance (triage)

Precedent for conditioning a frozen language model through learned continuous prompts in the project's embedding-to-CV decoder design.

## Notes and caveats

The paper's prefix-tuning implementation is not automatically equivalent to this project's vector-to-input-embedding virtual-prefix projector. Either way, a learned prefix cannot recreate information absent from its conditioning vector.

## Evidence status

Title, authors and venue year checked against ACL Anthology on 2026-10-16; not a full-paper review.

## Encountered in

- Local Codex thread `01a0c8f8-810f-7c82-baeb-0070132e8408`, item `item-36` — full-CV decoder proposal.
