# Current-device decoder-chat supplement — 2026-10-16

## Scope and result

This supplement follows the [Windows import](2026-10-16-windows.md) and [earlier Linux-device supplement](2026-10-16-linux.md). It focuses on the locally accessible project conversation about LangVAE fine-tuning, BERT vectors, EmbeddingGemma inversion, Gemma decoders and CV counterfactual generation. It adds **13 metadata-verified paper records** after deduplication, bringing the library from 169 to **182** records (84 verified, 96 recovered, 2 unresolved). The cards document encounters and bibliographic identity, not full-paper reviews.

The working branch was already `literature`. Existing decoder code, experiments, model artifacts, manuscript files and Git staging were not changed for this import.

## Sources checked

- Local Codex session catalog: 17 indexed entries and 36 rollout JSONL files at the scan snapshot. Many rollouts were project guardian/review captures, not independent literature discussions; repeated quoted transcripts and tool-result pages were not treated as separate encounters.
- Local `thread_history_1.sqlite`: 13 thread IDs and more than 1,800 items at the scan snapshot, overlapping the rollouts. The high-yield source was project thread `01a0c8f8-810f-7c82-baeb-0070132e8408`; new cards carry its specific message/item IDs.
- Project Markdown, BibTeX and TeX sources with scholarly links, excluding this library, to check whether a cited work was already represented. The earlier imports had already covered the existing bibliographies. The current text-realizer experiment note independently cites scheduled sampling.
- Official ACL Anthology, JMLR, arXiv and Google DeepMind report pages for title/author/year identity and short relevance triage. Private reasoning, tool output dumps, unrelated conversations and embedded guardian transcripts were not used as provenance.

The [earlier Linux supplement](2026-10-16-linux.md) reports a larger 124-rollout/50-thread catalog; this scan used the **locally available** 36-rollout/13-thread snapshot. Its wider-coverage claim was not assumed to describe the same accessible history. Overlap was deduplicated by paper identity, not added as separate records.

## New records

- **Embedding and generator foundations:** [BERT](../papers/devlin2019bert.md), [Sentence-BERT](../papers/reimers2019sentence-bert.md), [BART](../papers/lewis2020bart.md), [T5](../papers/raffel2020t5.md), and the [Gemma 3 report](../papers/gemma-team2025gemma3.md).
- **Decoder and language-VAE methods:** [prefix-tuning](../papers/li2021prefix-tuning.md), [LoRA](../papers/hu2021lora.md), [cyclical KL annealing](../papers/fu2019cyclical-annealing.md), [structured self-attentive sentence embeddings](../papers/lin2017structured-self-attention.md), [scheduled sampling](../papers/bengio2015scheduled-sampling.md), and [neural text degeneration](../papers/holtzman2020neural-text-degeneration.md).
- **Inversion and counterfactual generation:** [Vec2Text reproducibility](../papers/seputis2025rethinking-embedding-privacy.md) and [Polyjuice](../papers/wu2021polyjuice.md).

Already indexed works encountered again—including Vec2Text, LangVAE, Optimus, EmbeddingGemma, LIBERTy, A Causal Lens and Beyond What If—were not duplicated. Exact chat citations resolved the previously generic Sentence-BERT, BERT, BART and T5 leads in [unresolved.md](../unresolved.md). Generic GPT-2/Sentence-T5 mentions, academic URLs present only in search/tool arguments, and references inside another paper without direct project discussion were not promoted to paper cards.

## Limits and validation

This is a scan of accessible project material on this device, not a search across other machines, deleted chats, inaccessible attachments or the complete reference lists of recovered papers. A cited pretrained generator's architecture does not imply a pretrained inverse for our 768-dimensional vectors. The Gemma 3 technical report describes the 1B–27B family, not the later 270M release.

Validated record count, status distribution, unique IDs, one topic-index entry per paper, and the new cards' internal links and encounter markers. The import remains unstaged, uncommitted and unpushed.
