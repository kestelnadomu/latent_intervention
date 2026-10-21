---
id: rombach2020net2net
topics: ["editing","encoders"]
metadata_status: metadata-verified
reading_status: not-reviewed
aliases: ["Net2Net","2005.13580"]
added: 2026-10-19
updated: 2026-10-19
---

# Network-to-Network Translation with Conditional Invertible Neural Networks

Robin Rombach; Patrick Esser; Björn Ommer. 2020. NeurIPS.

- [Proceedings](https://papers.neurips.cc/paper_files/paper/2020/hash/1cfa81af29c6f2d8cacb44921722e753-Abstract.html)
- [Preprint](https://arxiv.org/abs/2005.13580)

## Project relevance (triage)

Conditional invertible translation connects fixed pretrained representations without retraining their expert networks, including a BERT-to-BigGAN example. Useful precedent for modular latent adapters and a separately trained rendering component.

## Notes and caveats

Cross-network translation is not reconstruction of the unique original text, nor an identified causal intervention. Position this beside text realization while making the cross-modal setting explicit.

## Evidence status

NeurIPS/arXiv metadata and author abstract/method overview screened. Checked on 2026-10-19. Targeted screening only, not a full-paper review; the comparison with our project is agent inference.

## Encountered in

- [2026-10-19 web search](../imports/2026-10-19-web-search.md) prompted by the updated Overleaf draft (commit `a99392e`). Newly discovered candidate, not a recovered prior chat citation.
