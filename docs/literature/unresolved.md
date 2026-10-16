# Unresolved references and citation cautions

This file preserves leads without inventing a citation. Records marked `recovered` in [the index](index.md) also need metadata checking before manuscript use; they are not all repeated here.

## Identity or attribution needs resolution

| Lead | Encountered in | Next step |
| --- | --- | --- |
| Anonymous language-VAE manipulation submission | [Record](papers/vechtomova2020unsupervised.md); notes/leo/literature.md | The OpenReview PDF confirms the title but is anonymous. Establish authors, year and publication status; the existing Vechtomova key is not evidence. |
| Tan et al. (2024), “C3M” | [Candidate record](papers/tan2023concept-bottlenecks.md); notes/leo/semantic_decoder.md:47 | Confirm that the acronym maps to the 2023 preprint / later concept-bottleneck publication, and select the correct version. |
| Lample et al., multiple-attribute rewriting | notes/leo/semantic_decoder.md:81 | Resolve exact title, authors, year and primary source. |
| Squires et al. (2023) | notes/leo/literature.md:91 | Identify the specific multi-distribution CRL paper from the cited Ng paper or the original discussion. |
| von Kügelgen et al. (2023) | notes/leo/literature.md:91 | Do not conflate with the already recorded 2021 content/style paper. |
| Zhang et al. (2024) | notes/leo/literature.md:91 | Identify the intended CRL work; do not conflate with the two text-VAE papers. |
| Yao et al. (2024) | notes/leo/literature.md:91 | Resolve the multiple-view CRL citation. |
| Xu et al. (2024) | notes/leo/literature.md:91 | Resolve the multiple-view CRL citation; not the recorded Xu 2020 text-VAE paper. |

## Model and dataset names, not yet specific paper citations

These were discussed in project material but do not uniquely specify a bibliographic work. Keep them as leads rather than padding the paper catalog with guessed references.

| Name | Encountered in | What remains |
| --- | --- | --- |
| DELLA / LD4LG | notes/leo/vae.md:39 | Resolve each architecture's paper separately; verify the notes' grouping and claims. |
| Sentence-T5 / Sentence-BERT | notes/leo/vae.md:40 | Resolve the relevant sentence-embedding papers if used in the motivation; a model name alone does not establish which publication was meant. |
| BERT, GPT-2, BART, T5, generic VAEs | encoder and VAE notes | Optional implementation-background citations; exact variants/reports were not consistently specified. |
| EntailmentBank | docs/architecture/text_encoder.md:112; encoder chats | Resolve the dataset publication if discussing the LangVAE training domain. Do not substitute the dataset paper for checkpoint-specific training evidence. |
| Blog Authorship Corpus | notes/leo/semantic_decoder.md:74 | Resolve dataset origin and accompanying publication. |
| TrustPilot demographic corpus | notes/leo/semantic_decoder.md:75 | Determine which dataset/release and paper are intended. |
| FYelp/Amazon, Yelp, GYAFC | notes/leo/semantic_decoder.md:85 | Resolve dataset versions if selected for experiments; not each generic dataset name is a paper. |
| Generic HSIC, energy score/distance, optimal transport | implementation/theory discussions | Architecture/objective names were often used without explicit bibliographic citations. Add foundational sources later as new literature, not as recovered historical citations unless evidence is found. |

## Corrections already recorded, originals intentionally unchanged

- [GAN-based counterfactual fairness](papers/ma2023counterfactualgan.md): the legacy `grari2023counterfactual` authors are wrong for arXiv:2310.17687. Preserve its 2023 and renamed 2025 versions; use the correct Ma et al. author list for the chosen version.
- [Joo et al.](papers/joo2024constructing.md): the legacy author list disagrees with arXiv:2412.17523.
- [RegDiff](papers/zhou2025regdiff.md): the paper is by Zhou, Tian and Van de Cruys, not the Vaeth attribution in the notes/key.
- [Bias in Bios](papers/de2019bias.md): two existing BibTeX keys refer to one work.
- [LatentOps](papers/liu2022composable.md), [CPM](papers/wu2023cpm.md), [DAS](papers/geiger2023das.md), [CB-LLM](papers/sun2025cbllm.md), [LIBERTy](papers/toker-liberty-2026.md): distinguish preprint year, venue year, acronym and title.
- [Heilmann](papers/heilmann2026neurosymbolic.md): the 2025 proceedings and recovered 2026 journal entry have different author lists; do not merge their citation metadata blindly.
- [Conneau probing](papers/conneau2018probing.md): the notes' arXiv ID identifies the linguistic probing paper, not the separate SentEval toolkit paper.
- [Kernel representations](papers/li-learning-2020.md): the misleading legacy key is not the first author's name.

## Retrieval gaps

The [import report](imports/2026-10-16-windows.md) identifies 23 project subagent threads with metadata but no locally accessible messages, plus an imported ChatGPT discussion whose old internal citation markers do not resolve to public sources. Another device may recover those references.

Primary-source pages were checked selectively for identities and conflicts, not for every record. No exhaustive full-text review, citation-network expansion, novelty search or systematic-review protocol has been completed. Every entry still needs substantive reading before it supports a scientific claim.
