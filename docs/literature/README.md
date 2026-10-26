# Project literature library

This is the shared, Markdown-based literature library for **latent_intervention**. It supports the planned paper across fairness, frozen text representations, semantic editing, causal and OT transport, and text reconstruction. It is designed for humans and agents contributing from different devices. The compact review was integrated into the [main paper](../../overleaf/sections/2_related.tex) on 2026-10-19; broader selection notes and source records remain here. The full collection remains intact.

Start with [the manuscript-shaped draft](manuscript-draft.md) and its [closest-work checks](reviews/2026-10-19-closest-work.md). The [broader distilled review](distilled-review.md), [complete topic index](index.md), [related-work map](related-work-map.md), and [unresolved leads](unresolved.md) retain the wider context. The [initial Windows import](imports/2026-10-16-windows.md), [Linux-device supplement](imports/2026-10-16-linux.md), and [current-device decoder-chat supplement](imports/2026-10-16-current-device.md) explain what was searched and what could not be recovered.

## What is here

- [reading-guide.md](reading-guide.md): all 56 papers in the current related-work section, in reading order, with PDF citation numbers and clickable article links (snapshot: 2026-10-23).
- [manuscript-draft.md](manuscript-draft.md): the approved eight topic paragraphs with run-in headings, 33 references and selective grouped citations; integrated into the main paper, with editorial follow-ups retained here.
- [Closest-work checks](reviews/2026-10-19-closest-work.md): ten focused comparisons, inspected source locations, project distinctions and one explicit theorem-level caveat.
- [distilled-review.md](distilled-review.md): provisional 48-paper core across all 11 clusters, organized into 13 field-focused paragraphs, with selection rationales and three conditional reading leads; not manuscript text yet.
- `papers/`: one small, independently editable record per work or explicitly linked version family.
- [index.md](index.md): all 207 records in 11 primary topic clusters, with subgroups and cross-links; not a ranking or final bibliography.
- [related-work-map.md](related-work-map.md): proposed narrative, the five-method comparison, encoder/decoder roles and later reading questions.
- [2026-10-19 organization review](reviews/2026-10-19-organization.md): what the other devices added, coverage counts and remaining literature gaps.
- [2026-10-19 web search](imports/2026-10-19-web-search.md): 25 additional papers screened against the revised draft, with reading priorities and comparison caveats.
- [unresolved.md](unresolved.md): incomplete citations and method/dataset names that still need resolving.
- `imports/`: dated coverage reports for each device/import.
- [template.md](template.md): the record format.
- [log.md](log.md): short, append-only change history.

The library currently has **207 records**: **109 metadata-verified**, **96 recovered**, and **2 needs-resolution**. The initial Windows import contributed 142; the Linux-device supplement added 27 and verified one existing record; the current-device decoder-chat supplement added 13. The 2026-10-19 targeted web search added 25. All 75 unique keys from the three existing project bibliographies are represented; two Bias in Bios keys share one record. These are collection counts, not counts of papers read.

## Project context for the next agent

The planned paper uses **LIBERTy** as the benchmark setting, compares frozen encoders (LangVAE, Nomic, Qwen3 Embedding and EmbeddingGemma), and centers five methods: **`semantic_editor`, `ot_editor`, `state_flow`, `ot_flow`, `ot_gaussian`**. This is the owner's current narrative direction, not a claim that every method/encoder combination has already been evaluated. The updated Overleaf draft (commit `a99392e`, inspected 2026-10-19) foregrounds read–operate–render but remains incomplete. It informs the web-search priorities without replacing the owner's five-method scope.

A semantic decoder `g` maps embeddings to structured-state distributions; `h_S` specifies the intended semantic change. The two target-only editors share that target but use different map classes. State flow fits factual conditional densities and reuses abducted noise; OT flow uses a shared reference learned through flow matching; Gaussian OT supplies a restricted analytic transport. A separate text decoder `D_text` renders vectors as CVs. Keep **semantic decoding** and **text reconstruction** distinct, including their different supervision and evaluation needs.

Read the [clean-editor guide](../experiments/semantic_editor.md), [OT guide](../experiments/ot_baselines.md), [flow architecture](../architecture/latent_intervention.md), and the component sources linked in [the narrative map](related-work-map.md) before making implementation claims. Earlier projected editors, distilled/direct flows and legacy transformers remain historical context; no papers from those discussions have been discarded. Distinguish semantic success, density fit, paired latent recovery, text fidelity and causal identification: none establishes all the others.

## Organization rules

- Each paper has **one primary home** in the index. Use cluster cross-links for overlaps; do not duplicate, rename or move paper cards just to reorganize them.
- Primary clusters and subgroups are editorial navigation, not verified claims or relevance rankings. Their sizes do not prescribe manuscript space, and eleven clusters need not become eleven related-work subsections.
- Keep the existing card `topics` as broad discovery tags. The index is the single primary-classification record; no synchronized second taxonomy or new database is needed.
- Preserve every paper, alias, provenance note and unresolved lead. The broad selection lives in [distilled-review.md](distilled-review.md), and its shorter successor in [manuscript-draft.md](manuscript-draft.md). Revise these drafts without pruning the collection or silently changing the manuscript.

## Evidence conventions

- `recovered`: a reference was found locally; the complete bibliographic identity still needs checking.
- `metadata-verified`: basic title/author/year identity was checked against a primary source; full author lists may be abbreviated. This does **not** mean the paper was read or its claims validated.
- `needs-resolution`: a specific identity, attribution or version ambiguity prevents confident citation.
- `reading_status: not-reviewed`: no full-paper review is recorded in this library. It says nothing about what a human read previously. Use `reviewed` only after adding a sourced reading note.

“Project relevance (triage)” is the importing agent's assessment, not a theorem or an author claim. Chat suggestions and older project notes may contain errors, outdated plans or exaggerated novelty claims. Treat them as discovery provenance, not research evidence.

## Instructions for agents on other devices

1. Inspect Git status and preserve local work. This library belongs to the **main project repository**; `overleaf/` is a separate repository. Do not switch or edit that nested repository merely to add literature.
2. Read this README, the index and the latest import reports. Search this device's project notes and accessible project chats, including relevant archived/subagent records where available. Do not search unrelated private conversations.
3. Extract paper identities, method-name leads and the context in which they mattered. Do not commit raw chat exports, account databases, credentials, private absolute paths or irrelevant conversation content.
4. **Deduplicate before adding:** compare DOI, arXiv ID, normalized title and aliases. Preprint/venue years and renamed revisions often refer to one work. Add encounter provenance to an existing record rather than creating a device-specific duplicate.
5. Use [the template](template.md). Keep a stable lowercase filename/ID. Preserve old citation keys in `aliases`, even when they are wrong. Distinguish a historical reference from a newly found candidate resolution; put uncertain mappings in `unresolved.md`.
6. Verify metadata through the original paper, publisher, author repository or official proceedings. Never invent missing authors, titles, years or links. Record corrections and version-specific differences explicitly; do not silently “repair” the existing `.bib` files.
7. For a full reading note, add a brief contribution, assumptions, relevant result with page/section, limitations, and a careful project comparison. Keep author claims separate from your inference. Do not copy whole papers into these cards.
8. Assign each new paper to one primary subgroup in the index, update its cluster and overall counts, add useful cluster cross-links, and update relevant unresolved items. Add an `imports/YYYY-MM-DD-device.md` coverage report and append one line to the log. Report inaccessible chats and missing attachments. Do not describe an incomplete local scan as exhaustive across devices.
9. Check that local Markdown links resolve and IDs are unique. Keep changes confined to this folder unless the user asks otherwise. Do not stage, commit, push or change the manuscript bibliography without permission.

Example handoff request:

> Read docs/literature/README.md. Add papers discussed in this project's chats and notes on this device, deduplicate against the existing records, retain provenance, and document coverage and unresolved references. Do not edit the manuscript or existing bibliographies.

## Existing sources stay where they are

The root [literature/](../../literature/) directory contains existing full-text assets, including LIBERTy Markdown/PDF and the Causal Fairness Analysis PDF. This folder indexes those assets; it does not duplicate them or automatically import every item in their reference lists. Some local PDFs, notes and the nested Overleaf checkout may not be present on another device; use the primary-source links in each record.

The library is plain Markdown: no database, plugin, model download or generated bibliography is required.
