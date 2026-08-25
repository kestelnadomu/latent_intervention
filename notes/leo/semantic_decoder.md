# Related Work

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

---

# Autoregressive $g$: Worked Example

Concrete numbers from a run with $\mathbf S = (\texttt B, \texttt T, \texttt F)$, cardinalities $2, 3, 5$, so $\lvert\mathcal S\rvert = 30$, and $z \in \mathbb R^{10}$.

## Architecture

$$p(\mathbf s \mid z) = p(\texttt b \mid z)\; p(\texttt t \mid z, \texttt b)\; p(\texttt f \mid z, \texttt b, \texttt t)$$

One shared trunk $h = \tanh(W_2\tanh(W_1 z + b_1) + b_2) \in \mathbb R^4$, then three heads whose input grows by a 2-d embedding of each already-decoded value:

| head | input | shape | out |
|---|---|---|---|
| $\texttt B$ | $h$ | $2\times4$ | 2 logits |
| $\texttt T$ | $[h;\ e_{\texttt B}(\texttt b)]$ | $3\times6$ | 3 logits |
| $\texttt F$ | $[h;\ e_{\texttt B}(\texttt b);\ e_{\texttt T}(\texttt t)]$ | $5\times8$ | 5 logits |

**10 output units, not 30.** That gap is the whole reason to prefer this over Approach A (flat softmax).

## The run

$z = [0.00,\ 0.30,\ -0.27,\ -0.89,\ -0.45,\ -0.99,\ 0.06,\ 1.34,\ -0.49,\ -0.62]$  →  $h = [0.659,\ 0.658,\ -0.082,\ 0.879]$

### Step 1: Predict $\texttt B$

$$W_{\texttt B} = \begin{psmallmatrix}-0.34 & -0.27 & -1.24 & -0.73\\ 1.49 & -0.60 & -0.95 & 0.30\end{psmallmatrix}, \quad b_{\texttt B} = [0.42, -0.44]$$

$$W_{\texttt B}h + b_{\texttt B} = [-0.514,\ 0.492] \xrightarrow{\text{softmax}} p(\texttt B \mid z) = [0.268,\ 0.732]$$

### Step 2: Predict $\texttt T$ given $\texttt B$

Feed the embedding of each $\texttt b$ back in:

```
p(T | z, B=0) = [0.758, 0.020, 0.222]     logits [ 0.469, -3.165, -0.757]
p(T | z, B=1) = [0.348, 0.242, 0.410]     logits [ 0.386,  0.025,  0.550]
```

This is the coupling. Notice: $\texttt T{=}1$ has probability $0.020$ under $\texttt b{=}0$ and $0.242$ under $\texttt b{=}1$ — a factor of 12. No independent head can express that.

### Step 3: Predict $\texttt F$ given $\texttt B, \texttt T$

Six conditionals, one per prefix:

```
p(F | z, B=0, T=0) = [0.008, 0.525, 0.464, 0.001, 0.002]
p(F | z, B=0, T=1) = [0.018, 0.932, 0.049, 0.001, 0.001]
p(F | z, B=0, T=2) = [0.011, 0.518, 0.469, 0.001, 0.002]
p(F | z, B=1, T=0) = [0.128, 0.347, 0.457, 0.011, 0.057]
p(F | z, B=1, T=1) = [0.281, 0.614, 0.048, 0.005, 0.051]
p(F | z, B=1, T=2) = [0.163, 0.327, 0.440, 0.010, 0.059]
```

### Enumeration

$2 + 2{\cdot}3 = 8$ head evaluations give all 30 joint entries; they sum to $1.0$ exactly.

## The 30-vector: AR vs independent

This is the exact joint from the autoregressive factorisation:

| $(\texttt b,\texttt t)$ | $\texttt f{=}0$ | $\texttt f{=}1$ | $\texttt f{=}2$ | $\texttt f{=}3$ | $\texttt f{=}4$ |
|---|---|---|---|---|---|
| 0,0 | .0016 | **.1065** | .0941 | .0003 | .0003 |
| 0,1 | .0001 | .0050 | .0003 | .0000 | .0000 |
| 0,2 | .0006 | .0308 | .0279 | .0001 | .0001 |
| 1,0 | .0325 | .0885 | **.1164** | .0027 | .0146 |
| 1,1 | .0500 | **.1090** | .0085 | .0009 | .0091 |
| 1,2 | .0490 | **.0980** | **.1321** | .0031 | .0178 |

Now fit independent softmaxes to its **exact marginals**:
- $\texttt B: [0.268, 0.732]$
- $\texttt T: [0.458, 0.183, 0.359]$
- $\texttt F: [0.134, 0.438, 0.379, 0.007, 0.042]$

Result:

| $(\texttt b,\texttt t)$ | $\texttt f{=}0$ | $\texttt f{=}1$ | $\texttt f{=}2$ | $\texttt f{=}3$ | $\texttt f{=}4$ |
|---|---|---|---|---|---|
| 0,0 | .0164 | .0536 | .0465 | .0009 | .0051 |
| 0,1 | .0066 | .0214 | .0186 | .0003 | .0020 |
| 0,2 | .0129 | .0421 | .0365 | .0007 | .0040 |
| 1,0 | .0448 | **.1468** | **.1271** | .0024 | .0140 |
| 1,1 | .0179 | .0586 | .0508 | .0010 | .0056 |
| 1,2 | .0352 | **.1153** | .0999 | .0019 | .0110 |

Every per-column accuracy is identical for these two models. The joints are not.

## The damage

$$D_{\mathrm{KL}}(\text{AR} \parallel \text{indep}) = \mathbf{0.197} \text{ nats}$$

This is exactly the **total correlation** — the information the independent form discards. Check: $H(\text{indep}) - H(\text{AR}) = 2.784 - 2.587 = 0.197$ ✓

**The mode flips:**
- AR argmax: $(1,2,2)$ at $0.132$
- Independent argmax: $(1,0,1)$ at $0.147$
- The independent model's mode ranks *sixth* in the AR model.

**Entry-wise errors:**
- $(0,0,1)$: AR says $0.107$, indep says $0.054$ — off by 2×
- $(1,0,2)$: AR says $0.116$, indep says $0.127$ — also 2×
- $(1,0,1)$: AR says $0.089$, indep says $0.147$ — nearly 1.7× in the other direction

The independent form is strictly more diffuse (entropy increases by 0.197). It cannot help but smear — the product of marginals is the unique max-entropy distribution with those marginals.

## Why this matters

Your objective consumes the 30-vector:

$$(h_S \circ g)(\mathbf s' \mid z,\delta) = \sum_{\mathbf s} h_S(\mathbf s'\mid\mathbf s,\delta)\,g(\mathbf s\mid z)$$

$h_S$ propagates a *full assignment* through the SCM, so feeding it a factorised $g$ mixes over states it never should have weighted at full strength. The 0.197 nats is not a reporting metric — it enters the loss $h_Z$ is trained against.

This is why the autoregressive design is necessary: the KL your objective actually sees is exact, not an approximation degraded by marginal factorisation.

## Scaling to your schema

Your real decoder:
- 8 nodes: R, G, A, E, S, W, V, C
- Total cardinality: $4 \times 2 \times 3 \times 4 \times 3 \times 3 \times 2 \times 2 = 3456$ (or 2263 supported states)
- Flat approach: $\sum = 3456$ output units
- Autoregressive approach: $\sum = 4 + 2 + 3 + 4 + 3 + 3 + 2 + 2 = 23$ output units

Enumeration cost: $4 + 4\cdot2 + 4\cdot2\cdot3 + \cdots = 4 + 8 + 24 + 96 + 288 + 864 + 1728 + 3456 = 6468$ head evaluations for the full 3456-vector. Still tractable.

## Implementation notes

Changes to `src/semantic_decoder.py`:

1. Add per-column embedding tables to `__init__`
2. Change each head's input dimensionality to `hidden_dim + emb_dim * position`
3. Implement `forward()` in two modes:
   - **Train**: teacher-force the true prefix, evaluate each head once
   - **Inference/Enumerate**: iterate through $\lvert\mathcal S\rvert$, building each state's prefix incrementally
4. Cache embeddings of each value to avoid recomputation

Cost: $\sim$50 lines, no change to the training loop.

