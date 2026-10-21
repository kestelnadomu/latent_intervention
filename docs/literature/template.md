# Paper-record template

Copy the block below into `papers/<stable-id>.md`. Keep unknown fields explicit and remove unused optional sections. Record an actual repository-relative source or local thread/message ID; a web discovery alone is not evidence that this project previously discussed the paper.

```markdown
---
id: author-year-short-title
topics: ["causal"]
metadata_status: recovered
reading_status: not-reviewed
aliases: []
added: YYYY-MM-DD
updated: YYYY-MM-DD
---

# Full paper title

Authors (or unresolved). Year (identify preprint versus publication). Venue, if known.

- [Primary source](https://example.org/paper)

## Project relevance (triage)

One or two sentences explaining which part of our project motivates reading this.

## Notes and caveats

Optional: version differences, corrected attribution, uncertainty, related records.

## Evidence status

What was checked, against which source and when. Metadata verification is not a paper review.

## Encountered in

- [Project source](../../../notes/example.md) — line or heading.
- Local Codex thread ID and message ID, with a brief topic label if useful.
```

Broad discovery tags: `fairness`, `causal`, `editing`, `abstraction`, `concepts`, `identifiability`, `encoders`, `benchmarks`. Use multiple tags when helpful; they are not the primary taxonomy. Assign the record exactly once to a subgroup in [the topic index](index.md), which also provides the dedicated OT, text-realization and evaluation clusters. Use cluster cross-links for overlap and update the counts; no new card field is required.

After a full review, optionally add “Contribution”, “Assumptions and limitations”, “Relevant result” (page/section/theorem), and “Comparison with our method”. Separate direct evidence from interpretation; use short paraphrases and source links. Add cross-links using `[label](other-record.md)`.
