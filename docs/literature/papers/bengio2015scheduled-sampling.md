---
id: bengio2015scheduled-sampling
topics: ["encoders"]
metadata_status: metadata-verified
reading_status: not-reviewed
aliases: []
added: 2026-10-16
updated: 2026-10-16
---

# Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks

Samy Bengio, Oriol Vinyals, Navdeep Jaitly and Noam Shazeer. 2015. Preprint.

- [Primary source](https://arxiv.org/abs/1506.03099)

## Project relevance (triage)

Background for the mismatch between teacher-forced training and autoregressive CV generation. It was raised when diagnosing why factual decoder losses need not predict long-form reconstruction quality.

## Notes and caveats

The paper studies recurrent sequence prediction; it is not evidence that scheduled sampling would fix this project's Gemma decoder or factual inaccuracies.

## Evidence status

Title, authors and preprint year checked against arXiv on 2026-10-16; not a full-paper review.

## Encountered in

- Local Codex thread `01a0c8f8-810f-7c82-baeb-0070132e8408`, message `msg_0f7071987f5f6bc5016ad1c9430e6c81919950363366908377` — text-realizer failure analysis.
