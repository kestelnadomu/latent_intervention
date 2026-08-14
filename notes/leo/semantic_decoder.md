# Semantic Decoder $g: Z \to S$ — Related Work

Framing: $g$ is a **probe that classifies several n-ary attributes at once from a frozen
latent**. That combination (multi-attribute, categorical-not-binary, frozen encoder) is
covered by four literatures that barely cite each other. Sorted by distance to our setup.
See also `literature.md` §4 for the causal-inference-flavoured predecessors (CEVAE etc.).

---

## 1. Concept-annotated counterfactual text — closest analog

- **Abraham et al. (2022) — CEBaB** (arXiv 2205.14140, NeurIPS).
  Our setup built by hand instead of by SCM. 2,299 OpenTable reviews expanded to ~15,089
  texts; four aspect concepts (food, service, ambiance, noise), each **ternary**
  (positive / negative / unknown), plus a review-level outcome rating. The expansion is
  *counterfactual*: annotators edit an original review to flip one aspect → approximate
  $(x, x')$ pairs with $(S, S')$ labels. Real-text equivalent of our shared-noise
  factual/counterfactual rows.

- **Wu et al. (2023) — Causal Proxy Models** (ICML).
  The method paper on top of CEBaB, and the single most relevant read. Trains a model to
  mimic counterfactual behaviour; the interchange-intervention variant localises each
  concept in a fixed slice of the hidden representation. Structurally our consistency
  constraint $h_S(g(z)) = g(h_Z(z))$, minus the explicit SCM.

## 2. Causal abstraction / distributed alignment

- **Huang et al. (2024) — RAVEL** (ACL; code: github.com/explanare/ravel).
  Benchmark for exactly the failure mode $g$ will hit. Entities (cities, Nobel laureates,
  physical objects, verbs, occupations) each carry *several* n-ary attributes — a city has
  country, continent, language, timezone — and the question is whether a method moves one
  without dragging the others. Scores **cause** (target attribute moves) and **iso**
  (others don't). Much sharper than our current per-column consistency accuracy and
  **directly importable as an eval**. Their method **MDAS** (Multi-task Distributed
  Alignment Search) learns a subspace satisfying multiple causal criteria at once — a
  supervised-rotation alternative to our MLP probe, worth a baseline comparison.
  Data: templated prompts over entity–attribute tables, i.e. cheap and fully controlled —
  same philosophy as our SCM route.

## 3. Concept Bottleneck Models for text — the multi-head probe as architecture

This family *is* a multi-attribute probe, just trained end-to-end: a bottleneck predicts
$k$ concepts, a linear layer maps concepts → label.

- **Ludan et al. (2024) — Text Bottleneck Models** (arXiv 2310.19660): GPT-4 infers the
  concepts, no human concept labels.
- **Tan et al. (2024) — C3M**: mixed human- + ChatGPT-generated concepts, sentiment tasks.
- **Sun et al. (2024) — CB-LLM** (arXiv 2412.07992): ChatGPT-generated concepts scored via
  a text-embedding model; competitive with black-box baselines on classification.
- **(2025) — Concept completeness for textual CBMs** (arXiv 2502.11100): does the concept
  set span the task.

LLM-annotated concepts = precisely our `generate-texts` pipeline, so this is the family
whose *data-generation* story we already implement.

### Two warnings from this literature (affect the consistency constraint directly)

1. **Concept leakage** (Mahinpei et al. 2021): soft concept scores smuggle information
   beyond the concept label, so the downstream layer isn't really using the concepts.
   If $g$ leaks, $h_Z$ can satisfy $h_S(g(z)) = g(h_Z(z))$ by exploiting the leak instead
   of moving the semantics.
2. **Probe selectivity** (Hewitt & Liang 2019, control tasks): a high-capacity $g$ fits
   labels that are not linearly present in $z$.

→ Report per-head selectivity, and evaluate $h_Z$ against a **retrained** decoder. Cheap
to do, and pre-empts the obvious referee objection.

## 4. Classic probing & attribute inference — evidence, not method

- **Conneau et al. (2018) — SentEval probing** (arXiv 1805.01070) and **X-Probe**
  (arXiv 1906.05061): 10 linguistic properties, several n-ary, but one independent MLP per
  property on a frozen sentence vector. Methodologically thin for us; it is the citation
  for "MLP on frozen embedding" as a probe.
- Demographic-attribute data with naturally multiple n-ary labels: **BiosBias**
  (occupation + gender), **Blog Authorship Corpus** ((age-bracket, gender, industry) for
  19,320 authors), **TrustPilot** (age, gender, location, sentiment). Plus the adversarial
  -removal line (Elazar & Goldberg) showing these attributes survive erasure attempts.

## 5. Multi-attribute control in latent spaces — the $h_Z$ side

Lample et al. (multiple-attribute rewriting), LatentOps, attribute-regularised latent
diffusion, and *Editing Entangled Latent Representations* (arXiv 1905.12926) all train
attribute classifiers **on the latent** and use their gradients to move it. That is $h_Z$
trained through $g$, with a decoder-fluency objective instead of an SCM.
Data: FYelp/Amazon with (sentiment, gender, category), Yelp, GYAFC. (Overlaps
`literature.md` §3.)

## 6. How to *represent* n-ary, not just classify it

- **Park et al. (ICLR 2025) — The Geometry of Categorical and Hierarchical Concepts**
  (arXiv 2406.01506): categorical concepts are represented as **simplices**, hierarchical
  relations as orthogonality; validated on 900+ WordNet concepts in Gemma and Llama-3.
  Relevant because our $S$ columns are ordinal/categorical with structure (education
  levels, wage bins). A softmax head throws that structure away; a simplex-geometry head
  gives $h_Z$ a smoother target and may make the ordinal columns much easier to hit.

---

## Where the gap is

No one has (a) an SCM giving **exact** counterfactual pairs, (b) n-ary concepts, and
(c) a **frozen** encoder all at once:

| Line | exact CF pairs | n-ary | frozen encoder |
|---|---|---|---|
| CEBaB / CPM | approximate, human-written | yes (ternary) | yes |
| RAVEL / MDAS | template-level ground truth | yes | yes |
| Textual CBMs | no | yes | usually not |
| Latent control (§5) | no | partly | no |

The SCM-generated exact pairs are our actual novelty.

## TODO / next actions

- [ ] Import RAVEL's **cause / iso** metric pair into `src/pipeline.py evaluate` — replaces
      bare per-column consistency accuracy with a disentanglement measure.
- [ ] Add a **control-task selectivity** number per decoder head (Hewitt & Liang).
- [ ] Evaluate $h_Z$ against a **retrained** $g$ to rule out concept leakage.
- [ ] Cheapest credibility win: run the pipeline on **CEBaB** (frozen LangVAE → $g$ →
      their human counterfactual pairs). Turns "does this work on real text?" from a
      question into a table.
- [ ] Consider a simplex/ordinal-aware head for A, E, W instead of plain softmax.
- [ ] Read CPM (Wu et al. 2023) properly and write a two-pager in `literature/`.
