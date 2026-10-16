# Project literature library

This is the shared, Markdown-based literature library for **latent_intervention**. It supports motivating the paper and writing related work across fairness, representation editing, neurosymbolic modelling and causal inference. It is designed for humans and agents contributing from different devices.

Start with [the topic index](index.md), [the related-work map](related-work-map.md), and [unresolved leads](unresolved.md). The [initial Windows import](imports/2026-10-16-windows.md) and [Linux-device supplement](imports/2026-10-16-linux.md) explain what was searched and what could not be recovered.

## What is here

- `papers/`: one small, independently editable record per work or explicitly linked version family.
- [index.md](index.md): topic-grouped navigation; not a ranking or final bibliography.
- [related-work-map.md](related-work-map.md): provisional reading priorities and comparison questions.
- [unresolved.md](unresolved.md): incomplete citations and method/dataset names that still need resolving.
- `imports/`: dated coverage reports for each device/import.
- [template.md](template.md): the record format.
- [log.md](log.md): short, append-only change history.

The library currently has **169 records**: **71 metadata-verified**, **96 recovered**, and **2 needs-resolution**. The initial Windows import contributed 142; the Linux-device supplement added 27 and verified one existing record. All 75 unique keys from the three existing project bibliographies are represented; two Bias in Bios keys share one record. These are collection counts, not counts of papers read.

## Project context for the next agent

The project studies interventions in a frozen text encoder's latent space. A semantic decoder `g` relates embeddings `Z` to structured states `S`; a symbolic intervention mechanism `h_S` supplies the intended semantic change. Current design discussions compare a target-only **semantic editor**, an **OT-structured semantic editor**, and **state flow**. Their implementation and paper framing are still evolving; check the current branch and method documentation before describing any as finalized.

The semantic editor is motivated by learning latent edits consistent with a symbolic target; its OT-structured counterpart constrains the map to a convex-potential gradient while keeping the semantic task. State flow uses a state-conditioned renderer and preserved abducted noise to define transport. Earlier chats also discuss projected C-sym, direct semantic flows, distributional OT, teacher–student distillation and several transformer benchmarks. These are historical context, not evidence that every discussed variant remains active.

Read current [architecture documentation](../architecture/latent_intervention.md), [encoder documentation](../architecture/text_encoder.md), and [the manuscript method section](../../overleaf/sections/3_method.tex) before making implementation claims. Distinguish semantic success, factual density fit, held-out latent recovery and causal identification. None implies all the others automatically.

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
8. Update the index, relevant unresolved items, and an `imports/YYYY-MM-DD-device.md` coverage report; append one line to the log. Report inaccessible chats and missing attachments. Do not describe an incomplete local scan as exhaustive across devices.
9. Check that local Markdown links resolve and IDs are unique. Keep changes confined to this folder unless the user asks otherwise. Do not stage, commit, push or change the manuscript bibliography without permission.

Example handoff request:

> Read docs/literature/README.md. Add papers discussed in this project's chats and notes on this device, deduplicate against the existing records, retain provenance, and document coverage and unresolved references. Do not edit the manuscript or existing bibliographies.

## Existing sources stay where they are

The root [literature/](../../literature/) directory contains existing full-text assets, including LIBERTy Markdown/PDF and the Causal Fairness Analysis PDF. This folder indexes those assets; it does not duplicate them or automatically import every item in their reference lists. Some local PDFs, notes and the nested Overleaf checkout may not be present on another device; use the primary-source links in each record.

The library is plain Markdown: no database, plugin, model download or generated bibliography is required.
