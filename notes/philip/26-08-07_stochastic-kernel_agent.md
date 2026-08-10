# Agent Handoff: Frozen-Backbone Stochastic Counterfactual Latent Editing

**Project:** *Counterfactual Latent Representations: A Neurosymbolic Approach*  
**Status:** Conceptual and theoretical redesign agreed; a fresh full-orbit LIBERTy-style DGP is judged sufficient for simulator-relative identification, but the required orbit data contract and \(G_e/K_S/Q_e\) pipeline are not yet implemented  
**Audience:** Next research/writing/coding agent  
**Primary design decision:** Preserve **frozen-backbone modularity**. A downloaded encoder is never fine-tuned, but small encoder-specific post-hoc modules are trained from simulator-generated paired counterfactual data.

> **Repository-grounding note (2026-08-10).** The conceptual handoff has now been cross-checked against the live manuscript source in `paper/ecaf26-template.tex`, the LIBERTy source in `literature/Toker et al. - 2026 - LIBERTy A Causal Framework for Benchmarking Concept-Based Explanations of LLMs with Structural Coun.md`, the Python simulator and text generator in `exp/sim/`, the current encoder/semantic-decoder/manipulator pipeline in `src/`, the R fairness estimators in `exp/estim/R/`, and the current configuration and artifact state. The repository-specific findings and implementation cautions below supersede any older implication that matched counterfactual texts or the stochastic editor already exist in code.

> **Companion file.** `26-08-07_stochastic-kernel_user.md` is the short, non-implementation-facing explanation. This file is the source of truth for an agent continuing the research, manuscript, or code work.

---

## 0. Executive state of the project

The original proposal used a frozen text encoder

\[
e:\mathcal X\to\mathcal Z_e,
\]

a learned semantic decoder

\[
g_e:\mathcal Z_e\to\mathcal S,
\]

a symbolic counterfactual operation

\[
h_S:\mathcal S\to\mathcal S,
\]

and a deterministic neural latent manipulator

\[
h_Z:\mathcal Z_e\to\mathcal Z_e.
\]

The manipulator was trained mainly by the semantic consistency equation

\[
g_e(h_Z(z))=h_S(g_e(z)),
\]

plus latent-space \(L_1\) and \(L_2\) penalties intended to make the edit sparse and small.

The central diagnosis is that this objective generally identifies only a **semantic fiber**, not a unique counterfactual latent point. The Euclidean penalties choose one point in that fiber according to arbitrary coordinates of the frozen encoder. They do not identify the simulator-paired individual counterfactual. Moreover, if the frozen representation loses person- or style-specific information, an exact deterministic counterfactual may not exist as a function of \(z\).

The minimal viable redesign is:

1. **Keep the encoder completely frozen.**
2. Use the simulator to generate **matched counterfactual CV pairs/orbits** with persona and rendering style fixed.
3. Train an encoder-specific regime-conditioned semantic analysis model \(G_e^a\) on the fixed embeddings.
4. Replace the deterministic map \(h_Z\) by an encoder-specific **stochastic conditional editor**
   \[
   Q_e(dz'\mid z,s',a,b),
   \]
   trained directly on paired targets
   \[
   Z_e^b=e(X^b)
   \quad\text{given}\quad
   (Z_e^a,S^b,a,b).
   \]
5. Compose semantic analysis, the symbolic SCM counterfactual, and latent synthesis into the counterfactual kernel
   \[
   \boxed{
   H_{Z,e}^{a\to b}(dz'\mid z)
   =
   \int G_e^a(ds\mid z)
   \int K_S^{a\to b}(ds'\mid s)
   Q_e(dz'\mid z,s',a,b).
   }
   \]

The final object is therefore not necessarily one point \(z'\). It is a simulator-induced conditional law over plausible paired counterfactual embeddings. A deterministic editor is a special case in which this law is almost surely a Dirac measure.

Fresh complete LIBERTy-style orbits are sufficient at the **population DGP level**: they identify the regular conditional laws \(G_e^{a,*}\), \(K_S^{a\to b,*}\), \(Q_e^{a\to b,*}\), and the direct oracle kernel \(H_e^{a\to b,*}\), almost surely on the support of the actual simulator coupling. This positive result does not mean that the current repository already creates those orbits, that a finite neural estimator is automatically consistent, or that the synthetic kernels transport to real CVs.

This preserves the intended modularity: replacing the encoder requires recomputing simulator embeddings and fitting new lightweight modules \((G_e,Q_e)\), but it does **not** require changing the encoder weights or the SCM.

---

## 0A. Repository-grounded state of the project

This section records what is actually present in the repository. It is deliberately explicit because the final architecture described later in this handoff is a **design target**, whereas the checked-in Python pipeline still implements the original deterministic proposal.

### 0A.1 Audit snapshot and version-control caveats

The repository was audited on **2026-08-10** on branch `dev_philip` at commit `319fc24`.

- `paper/ecaf26-template.tex` is the substantive manuscript source. `paper/template.tex` is conference boilerplate.
- The local `paper/` directory is excluded through `.git/info/exclude`, so edits there may not appear in ordinary `git status` output.
- At audit time, `notes/` was untracked. These handoff files therefore need to be explicitly added if they are meant to become versioned project artifacts.
- Generated `data/sim/`, `data/latents/`, `models/`, and `reports/` are ignored as reproducible artifacts.
- `data/text/` is intentionally tracked because LLM generations cost API credits. At audit time it contained `templates.csv`, but not the configured `personas.csv` or `cv_factual.csv`.
- Do not overwrite paid text generations casually. New orbit generation needs resumable compound identifiers and a migration plan for existing artifacts.

### 0A.2 What each relevant component currently does

| Component | Current repository behavior | Consequence for the redesign |
|---|---|---|
| `paper/ecaf26-template.tex` | Describes the deterministic \(h_Z\), semantic consistency, and \(L_1/L_2\) loss. | The manuscript has not yet been revised to the kernel formulation. |
| `poster/poster.qmd` and `poster/slides.qmd` | Still communicate the original deterministic analogy and loss. | Update only after the manuscript notation and claims stabilize. |
| `exp/sim/scm.py` | Generates factual and one configured counterfactual tabular row per ID by reusing the exact exogenous noise. Default intervention is `do(G=1)`. | The tabular simulator already supplies a valid paired coupling for that one regime, but it does not yet produce full multi-regime orbits. |
| `exp/sim/run.py` | Generates templates, personas, and **factual texts only**. It records `template_id`, `persona_id`, concrete age/work values, and text. | Matched counterfactual texts \(X^b\) do not exist yet; this is the first major implementation gap. |
| `exp/sim/prompts/cv_generation.yaml` | Uses one LLM call per CV with temperature 0 and combines a template, candidate-information list, and persona. | Reusing inputs across two independent calls does not guarantee identical renderer randomness. Prefer a paired/orbit response in one call or a deterministic renderer for the core identification experiment. |
| `src/encoder.py` | Wraps LangVAE. `encode(..., deterministic=True)` returns `.embedding`, documented as the posterior mean; stochastic sampling is optional. It also exposes text decoding for diagnostics. | The default already matches the recommended fixed posterior-mean representation, but encoder/checkpoint and preprocessing fingerprints must be stored with caches. |
| `src/semantic_decoder.py` | MLP with independent categorical heads for eight discrete SCM variables; mean cross-entropy loss. Outcome column `Q` is excluded. | This is a reusable first \(G_e\), but its softmax heads are only a factored approximation to a joint structured-state kernel and currently lack calibration diagnostics. |
| `src/latent_intervention.py` | Deterministic residual transformer conditioned on \(z\) and intervention-value tokens. It is trained through frozen semantic-decoder cross-entropy plus \(L_1/L_2\). | Preserve it as the exact original-method baseline; add a separate probabilistic editor rather than silently changing its semantics. |
| `src/pipeline.py` | Encodes only factual texts, performs a row-wise random split, trains the semantic decoder on factual latents, trains the old manipulator against tabular \(S^b\), and evaluates semantic consistency plus latent shift. | It needs orbit-aware records, grouped splits, paired \(Z^b\), staged \(G_e/Q_e\) training, sampled inference, and distributional evaluation. |
| `src/finetune_vae.py` | Optionally fine-tunes LangVAE on factual CV texts. | Fine-tuning is outside the locked primary method. If used as an upper bound, finish it first, freeze the resulting checkpoint, and fit fresh \((G_e,Q_e)\). Never jointly tune it with the editor in the main experiment. |
| `exp/estim/R/` | Implements separate structured-data counterfactually fair predictors (Level One, Fair Add, Fair K). | These are not currently connected to the latent editor. They may supply downstream baselines, but should not be presented as the implemented distributional latent-fairness stage. |

The manuscript mentions transfer to Bias in Bios, but no real-data ingestion, semantic calibration, SCM-abduction, editor, or transport-evaluation pipeline for that dataset exists in this checkout. Treat real-data use as a later transport study, not part of the implemented prototype.

The current CV prompt requires every codebook attribute to be mentioned explicitly in the text. Consequently, the first semantic-decoder experiment is closer to controlled information extraction than to recovering subtle latent semantics. This is useful for a proof of concept but must be acknowledged when interpreting \(G_e\)'s accuracy and sim-to-real relevance.

### 0A.3 Current end-to-end behavior

The implemented Python route is currently:

```text
paired tabular SCM rows (S^a, S^b, known shared noise)
              |
              +--> verbalize S^a only --> X^a
                                       --> frozen LangVAE --> Z^a
                                                               |
                         train g_e(Z^a) against S^a             |
                                                               v
                          deterministic h_Z(Z^a, do-token)
                                                               |
                          frozen g_e(h_Z(...)) against S^b
                          + L1/L2 displacement penalties
```

There is no \(X^b\), no \(Z^b=e(X^b)\), no conditional density \(q_e(Z^b\mid Z^a,S^b)\), no sampled counterfactual inference, and no distributional fairness objective in the current Python code.

### 0A.4 What is already reusable

The redesign is not a restart. Reuse:

- the Python SCM and its shared-noise factual/counterfactual pairing;
- the R SCM as the reference implementation that must remain synchronized;
- the codebook and prompt assets;
- deterministic per-ID selection of templates/personas as the starting point for orbit coupling;
- the frozen LangVAE wrapper and default deterministic embedding;
- the categorical semantic-decoder architecture as a baseline \(G_e\);
- the current manipulator as the original loss baseline;
- configuration loading, checkpoint serialization, and CLI-stage patterns;
- the structured-data fairness estimators as separate downstream comparators where scientifically appropriate.

### 0A.5 Repository-specific notation collisions to avoid

The original handoff used \(C_i\) for persona and \(V_{ij}\) for renderer style. In this repository, `C` already means **professional certificates** and `V` means **volunteering**. Other collisions are equally easy to miss.

Use the following notation consistently in future revisions:

| Concept | Recommended notation | Repository collision avoided |
|---|---|---|
| Complete structured SCM state | \(\mathbf S\in\mathcal S\) | `S` is also the socioeconomic-status column |
| Persona/person context | \(\Pi_i\) | `C` is professional certificates |
| Causally irrelevant renderer state | \(\Omega_{ij}\) | `V` is volunteering; `R` is race |
| SCM exogenous state | \(U_i\) | no material collision |
| Outcome | \(Y\) in the manuscript | the code calls the simulated outcome `Q`, while \(Q_e\) denotes the stochastic editor |
| Source/target regimes | lowercase \(a,b\) or explicit labels | uppercase `A` is age |

In code, prefer descriptive class and field names such as `StochasticLatentEditor`, `person_id`, `renderer_id`, `source_regime`, and `target_regime`. Mathematical \(Q_e\) can remain in theory, but should not be confused with the simulated outcome column `Q`.

### 0A.6 A missing SCM component that must not be overlooked

`exp/sim/scm.py` obtains \(S^b\) by retaining the true exogenous noise \(U\) and rerunning the structural equations under intervention. That solves counterfactual generation **inside the simulator**, but it does not implement the inference-time kernel

\[
K_S^{a\to b}(d\mathbf s'\mid \mathbf s)
\]

needed for a new real or held-out text, where \(U\) is unknown and only an estimated \(\mathbf S\) is available. Because the current SCM is discrete, clipped, and rounded, abduction from \(\mathbf S\) to \(U\) is generally nonunique.

The implementation therefore needs one explicit choice:

1. sample from the posterior \(P(U\mid \mathbf S,a)\) and propagate each draw under \(b\);
2. estimate an empirical/learned conditional kernel from simulated \((\mathbf S^a,\mathbf S^b)\) pairs;
3. augment the observed structured state until the target is deterministic, if scientifically defensible; or
4. restrict the first experiment to an oracle-\(U\) synthetic setting and label it clearly as such, while separately implementing the deployable kernel.

Do not claim that the current `simulate()` function already supplies deployable abduction. It supplies training pairs and an oracle synthetic benchmark.

### 0A.7 Prior internal assessment that should remain in view

`pab-notes/26-07-21_assessment.md` predates the final stochastic-kernel decision but contains useful evidence and alternative designs:

- It independently diagnosed semantic-fiber nonidentification, probe gaming, coordinate-dependent sparsity, and off-support edits.
- It reports a preliminary 300,000-row analysis of the current clipped/rounded SCM: among factual `G=0` cases under `do(G=1)`, 93.9% of factual structured states were compatible with multiple counterfactual states; the reported Bayes ceiling for the exact counterfactual vector was about 62.5%, and for counterfactual education about 70.3%. These numbers strongly motivate a stochastic \(K_S\), but they must be reproduced by a checked-in script/test before entering the manuscript.
- It notes that the one-sided `do(G=1)` design yields many identity cases and few examples for some descendant changes at \(n=1000\). Full `do(G=0)`/`do(G=1)` orbits and a power/effective-transition-count audit are required before model comparison.
- It proposed a learned causal-slot-plus-residual bottleneck and stronger frozen embedding models. The final decision does not use that factorization as the primary method because it weakens frozen-backbone modularity, but it remains a valuable upper-bound/ablation.
- It flags a possible domain and sequence-length mismatch between the configured EntailmentBank LangVAE checkpoint and generated CVs. Measure tokenizer truncation, actual token-length distributions, semantic recovery, and decoding quality in this repository rather than relying on model reputation. LangVAE should be a baseline alongside at least one stronger fixed embedding encoder.

This earlier assessment is complementary rather than controlling: paired stochastic editing is the locked main route; factorized/tuned representations and deterministic invertible SCMs remain comparisons or fallbacks.

---

## 0B. Fresh LIBERTy rerun: exact sufficiency verdict

This section records the conclusion reached after comparing the proposed kernel architecture, the live repository, and the full LIBERTy CV-generation description. It should be read before starting another paid text-generation run.

### 0B.1 Bottom line

The **LIBERTy generation mechanism is aligned with the revised problem**. A fresh run is sufficient to define and identify the intended kernels on simulator support if it is converted into a complete, matched, unit-level orbit generator.

The following three claims must remain separate:

1. **Population identification under the synthetic DGP: yes.** Complete matched orbits identify the simulator-induced conditional laws almost surely on supported regimes.
2. **Consistency of a fitted finite neural model: conditional.** This additionally needs independent units, adequate support, an appropriate model or sieve, a strictly proper score, regularization, and controlled optimization error.
3. **Identification of real-person CV counterfactuals: no.** That requires a separate simulator-to-real transport assumption and validation.

Simply looping the current `intervention` configuration is not sufficient if the pipeline continues to verbalize only the factual row. “Generate all counterfactual queries” must mean: generate, audit, encode, and retain every structured and textual world needed by the paper.

### 0B.2 What is being reused from LIBERTy

The local source is:

`literature/Toker et al. - 2026 - LIBERTy A Causal Framework for Benchmarking Concept-Based Explanations of LLMs with Structural Coun.md`

Relevant source anchors are:

- lines 330–348 and 424–448: endogenous concepts/text and exogenous SCM/persona/template variables;
- lines 512–570: factual generation and abduction–action–prediction with shared exogenous state, persona, and template;
- lines 574–590: published dataset subsets, with counterfactual pairs reserved for the interventional test set;
- lines 547–560: approximately 998 retained CV pairs and sparse intervention sampling;
- lines 1962–1978: the original CV template/persona pools and renderer;
- lines 2008–2036: the CV SCM and noise laws mirrored by this repository;
- lines 2082–2106: the CV-rendering prompt, including permission to adjust persona/template details for coherence;
- lines 1314–1328: human validation, including only 84.7% concept agreement for CV texts;
- lines 1161–1183: the explicit limitation that the benchmark SCM is synthetic rather than a recovered real-world mechanism.

The useful LIBERTy principle is:

\[
U_i,\Pi_i,\Omega_{ij}
\longmapsto
\left\{
(\mathbf S_i^r,X_{ij}^r,Z_{e,ij}^r):r\in\mathcal R
\right\},
\]

where:

- \(U_i\) is one SCM exogenous state;
- \(\Pi_i\) is persona/context;
- \(\Omega_{ij}\) is one declared renderer realization or replicate;
- \(r\) is a named intervention regime;
- \(\mathbf S_i^r=F_r(U_i)\);
- \(X_{ij}^r=R(\mathbf S_i^r,\Pi_i,\Omega_{ij})\);
- \(Z_{e,ij}^r=e(X_{ij}^r)\).

The same \((U_i,\Pi_i,\Omega_{ij})\) is held fixed while \(r\) changes. This cross-world coupling, not latent proximity, defines which target text/embedding belongs to the same simulated unit.

Use the **generation mechanism**, not the published benchmark split. LIBERTy’s released counterfactuals were primarily test data, use only sparse random changes, and are too small to serve simultaneously as unrestricted high-dimensional editor training data and independent evaluation. The planned project run must create new train, validation, and test orbits for the exact paper queries.

The repository is an adaptation rather than a bitwise reproduction: it currently configures GPT-5.4, 50 templates, and 100 personas, whereas LIBERTy reports GPT-4o, 1,235 templates, and 990 personas. This is scientifically permissible, but it defines a different simulator population and must be labeled and versioned as such.

### 0B.3 Exact orbit data contract

For every independent `unit_id`, sample \(U_i\) once. For each `renderer_id`, sample or declare \((\Pi_i,\Omega_{ij})\) once. Evaluate every regime \(r\) claimed by the paper from that same \(U_i\), and render every resulting state under the coupled renderer context.

Persist a long-format source table with at least:

`unit_id, regime_id, renderer_id, split,`  
`all structured columns and the simulated outcome,`  
`all SCM exogenous variables,`  
`persona_id, template_id, and their realized contents,`  
`shared renderer noise/common uniforms and concrete rendered values,`  
`text, generation-audit status,`  
`prompt hash, model/deployment, parameters, response metadata, and code/config revision.`

Derive transition records only after the unit split:

\[
(Z_e^a,Z_e^b,\mathbf S^a,\mathbf S^b,a,b,\text{metadata}).
\]

Include every source–target pair actually claimed at inference. If the paper claims only natural-to-`do(G=0)` and natural-to-`do(G=1)`, that is the necessary support. If it claims a generic regime-to-regime editor, generate the complete configured regime grid, both directions, and identity transitions. Ordinary node interventions identify total-intervention kernels only; path-specific claims require actual edge/path interventions and corresponding orbit labels.

### 0B.4 What the complete orbit law identifies

For fixed source and target regimes \(a,b\), the full population orbit law identifies the regular conditional probabilities

\[
G_e^{a,*}(d\mathbf s\mid z)
=
P(\mathbf S^a\in d\mathbf s\mid Z_e^a=z,a),
\]

\[
K_S^{a\to b,*}(d\mathbf s'\mid\mathbf s)
=
P(\mathbf S^b\in d\mathbf s'\mid
\mathbf S^a=\mathbf s,a,b),
\]

\[
Q_e^{a\to b,*}(dz'\mid z,\mathbf s')
=
P(Z_e^b\in dz'\mid
Z_e^a=z,\mathbf S^b=\mathbf s',a,b),
\]

and the direct oracle latent law

\[
H_e^{a\to b,*}(dz'\mid z)
=
P(Z_e^b\in dz'\mid Z_e^a=z,a,b).
\]

These laws are unique almost surely on their respective simulator supports. Repeated observations at exactly the same continuous \(z\) are not required for population identification. Finite estimation nevertheless requires structural smoothness/model assumptions and enough independent units.

Source/target regime labels are canonical conditions in a multi-regime experiment. Suppress them only when a regime is globally fixed or an explicit invariance has been established.

The direct \(H_e^{a\to b,*}\) is identified by complete paired orbits without a factorization assumption. The neurosymbolic composition

\[
H_e^{a\to b,*}
=
Q_e^{a\to b,*}\circ
K_S^{a\to b,*}\circ
G_e^{a,*}
\]

equals that direct oracle law only if the structured state is counterfactually sufficient:

\[
P(\mathbf S^b\in d\mathbf s'
\mid \mathbf S^a=\mathbf s,Z_e^a=z,a,b)
=
P(\mathbf S^b\in d\mathbf s'
\mid \mathbf S^a=\mathbf s,a,b).
\]

Ideal LIBERTy makes this plausible when the text renderer receives \(U_i\) only through \(\mathbf S_i^a\), and persona/template/renderer state is independent of \(U_i\). If \(Z^a\) reveals hidden causal information relevant to \(\mathbf S^b\) beyond \(\mathbf S^a\), the direct \(H\) remains identified but the displayed modular factorization fails. Enlarge \(\mathbf S\), include the missing context, or use a kernel such as \(K_S(d\mathbf s'\mid\mathbf s,z,\text{context},a,b)\), then retype the composition.

### 0B.5 Oracle and deployable SCM kernels are different

The stored \(U_i\) gives the exact pathwise target \(\mathbf S_i^b\) for a simulator unit. It is an oracle benchmark, not deployable inference for a new CV.

Because the current structural equations clip and round continuous disturbances, many \(U\) values can yield the same observed \(\mathbf S^a\) but different \(\mathbf S^b\). Nonunique abduction does not make \(K_S\) unidentified; it makes \(K_S\) stochastic.

Implement and report both:

1. **Oracle \(K_S\):** use stored \(U_i\) to isolate \(Q_e\).
2. **Deployable \(K_S\):** calculate/sample \(P(U\mid\mathbf S^a,a)\) and propagate under \(b\), or fit a smoothed regime-conditioned transition table from training-unit structured orbits.

Structured simulation is cheap. Generate far more tabular orbits for fitting and validating \(K_S\) than expensive rendered CV orbits if needed.

### 0B.6 Renderer coupling is part of the estimand

Fixed persona/template identifiers are necessary but do not by themselves prove that the realized person and style are fixed.

- Remote temperature-zero calls may still have unrecorded provider/model variation.
- The current prompt permits persona/template details to be changed for coherence.
- Concrete age/work values are sampled within bins and need an explicit shared-uniform coupling across regimes.
- An identity world must reuse the exact cached text; a second API call would create artificial counterfactual variation.
- Prompt, model, deployment, parameters, response metadata, and outputs must be versioned.
- Every generated world must pass automated structured-feature checks and persona/style/invariant-field audits; failures must be regenerated or explicitly modeled as renderer noise.

The strongest core experiment uses a version-pinned deterministic renderer or one structured orbit-level response. If remote randomness is intentionally marginalized, declare it as part of \(\Omega\) and generate adequate replicates.

Multiple renderer replicates are **not required for population identification** when independent complete orbits sample from the declared renderer law. They are strongly useful for diagnosing renderer uncertainty, preservation, multimodality, and finite-sample calibration.

### 0B.7 Statistical conditions that “all queries” does not solve

- Independent sample size is the number of SCM units, not the number of derived transition rows or Monte Carlo samples.
- Complete regime coverage does not create support at impossible or extremely rare structured states.
- \(G_e\circ K_S\) must not feed \(Q_e\) unsupported target-state conditions.
- All worlds and renderer replicates for a unit must remain in one persisted split.
- A finite set of roughly 1,000 units cannot nonparametrically determine an unrestricted 128-dimensional conditional distribution. Use restricted baselines, support counts, learning curves, and regularization.
- Deterministic text plus deterministic posterior-mean embeddings can make \(Q_e^*\) atomic, Dirac, or lower-dimensional. A Gaussian Lebesgue NLL then estimates an explicitly smoothed or best-in-family approximation, not the exact law. Use a proper score admitting singular laws, or define the smoothing/reference measure and variance floor explicitly.
- Predicted kernel variance is aleatoric under the fitted simulator law; parameter and finite-data uncertainty require unit-level bootstrap, ensembles, or another explicit procedure.
- Any weighting or oversampling of rare regimes/states changes the training law unless corrected when estimating the original target population.

### 0B.8 Go/no-go checklist before paid CV generation

Do not launch the full billed renderer run until all of the following exist and pass on a small pilot:

1. a machine-readable query manifest defining every regime, direction, total/path semantics, and downstream fairness query;
2. one multi-regime shared-\(U\) orbit generator rather than disconnected two-table runs;
3. a compound-key long schema and resumable writer keyed by `(unit_id, regime_id, renderer_id)`;
4. an explicit coupling for persona, template, concrete bin values, and renderer randomness;
5. a persisted `unit_id` train/validation/test manifest created before pair expansion;
6. automatic checks for complete orbits, shared-noise invariants, identity-text reuse, structured-feature realization, persona/style drift, and duplicate keys;
7. a 20–50-unit end-to-end pilot that can be encoded into correctly aligned paired \(Z^a,Z^b\) artifacts;
8. effective transition/support counts from a large cheap tabular simulation, used to choose the number of independent rendered units.

After the pilot passes, run the complete generator, freeze/cache every encoder world, fit and certify \(G_e^a\), fit/validate deployable \(K_S^{a\to b}\), train \(Q_e^{a\to b}\) against paired \(Z^b\), and compare direct-oracle \(H\) with the composed \(G\to K\to Q\) pipeline.

The defensible final claim is:

> For a fixed encoder and a fixed, audited LIBERTy-style simulator coupling, complete population orbits identify the regime-conditioned semantic, structured-counterfactual, latent-editor, and direct latent-counterfactual kernels almost surely on supported regimes. Consistency of a chosen estimator requires additional unit-level sampling, support, model, scoring, regularization, and optimization conditions.

---

## 1. Locked interpretation of “modularity”

### 1.1 What is required

The project now targets **frozen-backbone modularity**:

- A pretrained encoder can be downloaded and used as-is.
- Its weights are never updated.
- The causal model and simulator protocol are shared across encoders.
- For each encoder \(e\), small post-hoc modules \(G_e\) and \(Q_e\) are trained.
- Encoder-specific compatibility is measured rather than assumed.

### 1.2 What is not required

The project does **not** target universal zero-shot modularity. It does not claim that one fixed editor can be applied to arbitrary encoders without retraining any bridge. Such a claim would generally be impossible because encoders differ in:

- output dimension;
- coordinate system and scale;
- information retained or discarded;
- pooling and preprocessing;
- invariances;
- stochasticity;
- whether their outputs lie in a decodable latent space.

The shared object is the **method and causal interface**, not necessarily the weights of the post-hoc editor.

### 1.3 Interface contract

For each encoder \(e\), the method expects:

1. a fixed measurable representation \(z=e(x)\);
2. simulator-generated factual/counterfactual text pairs;
3. an encoder-specific semantic probe \(G_e\);
4. an encoder-specific conditional editor \(Q_e\);
5. encoder admissibility diagnostics.

A concise description is:

\[
\boxed{
\text{frozen encoder}
+
\text{small encoder-specific probabilistic bridge}
+
\text{shared SCM}
}
\]

---

## 2. The original manuscript proposal

The initial proposal can be summarized as follows.

### 2.1 Representation

A pretrained VAE-like text encoder maps a CV to a latent representation:

\[
Z=e(X).
\]

The encoder is frozen so that representation learning is separated from causal intervention.

### 2.2 Semantic grounding

A semantic decoder is trained:

\[
g_e:Z\mapsto S,
\]

where \(S\) contains selected structured variables participating in an explicit SCM. The manuscript explicitly allows \(g_e\) to decode only part of the information in \(Z\).

### 2.3 Symbolic counterfactual

The SCM defines a counterfactual operation

\[
h_S:S\mapsto S'.
\]

This was described through abduction-action-prediction, potentially for a path-specific intervention on a sensitive attribute.

### 2.4 Deterministic latent analogue

A neural manipulator was proposed:

\[
h_Z:Z\mapsto Z'.
\]

It was trained to make the following diagram commute:

\[
h_S(g_e(z))=g_e(h_Z(z)).
\]

The manuscript’s latent objective was conceptually

\[
\mathcal L_{\mathrm{old}}
=
 d_S\!\left(h_S(g_e(Z)),g_e(h_Z(Z))\right)
+
\alpha\lVert h_Z(Z)-Z\rVert_1
+
\beta\lVert h_Z(Z)-Z\rVert_2^2.
\]

The first term enforces semantic consistency. The last two terms choose a sparse, small latent displacement.

**Manuscript-versus-code distinction.** The current Python implementation is slightly stronger than the displayed manuscript objective: it feeds intervention tokens to the deterministic editor and uses the simulator's true paired tabular target \(\mathbf S^b\), rather than literally calculating \(h_S(g_e(z))\), as the frozen-decoder consistency target. This is useful teacher forcing and should be retained. It does not resolve the identification problem because the code still has no paired counterfactual text/embedding \(X^b,Z^b\); it selects a point only through semantic compatibility and latent displacement.

### 2.5 Intended downstream use

The single generated counterfactual embedding \(Z'=h_Z(Z)\) would be used in fairness training or evaluation, for example by penalizing discrepancies between \(p(Z)\) and \(p(Z')\).

---

## 3. Why the original construction is not theoretically sufficient

The original proposal is not useless, but its strongest interpretation is unsupported. Its loss can produce a semantically compatible edit, yet it does not generally identify the individual simulator-paired counterfactual latent representation.

### 3.1 Semantic consistency identifies a fiber, not a point

For any target semantic state \(s'\), define the semantic fiber

\[
F_{s'}
=
\{z'\in\mathcal Z_e:g_e(z')=s'\}.
\]

The equation

\[
g_e(h_Z(z))=h_S(g_e(z))
\]

only implies

\[
h_Z(z)\in F_{h_S(g_e(z))}.
\]

If \(g_e\) decodes only part of the embedding, that fiber will generally contain many points. Therefore, semantic consistency alone does not say which point is the counterfactual counterpart of the observed person and rendering style.

A useful negative statement is:

> A deterministic latent point is identified by semantic consistency alone if and only if the relevant target semantic fiber is a singleton. This is not expected when \(g_e\) is partial.

### 3.2 The \(L_1/L_2\) terms do not solve identification

The old regularizers choose the point with a small displacement in the chosen encoder coordinates. This is a modeling convention, not a causal fact.

For an invertible latent reparameterization \(\phi\), define

\[
\widetilde e=\phi\circ e,
\qquad
\widetilde g=g_e\circ\phi^{-1}.
\]

The semantic content is unchanged, but

\[
\lVert \phi(z')-\phi(z)\rVert_p
\]

need not preserve the ordering induced by

\[
\lVert z'-z\rVert_p.
\]

Thus “minimal latent change” is coordinate-dependent. Freezing an encoder fixes a gauge operationally, but it does not turn Euclidean minimality into an identified counterfactual principle.

### 3.3 The deterministic map can be misspecified

Even if the simulator holds persona and style fixed, the map from an observed frozen embedding \(Z^a\) to its paired target \(Z^b\) may be nonunique.

Two different underlying simulator states can satisfy

\[
Z^a(w_1)=Z^a(w_2)
\]

while producing

\[
Z^b(w_1)\neq Z^b(w_2).
\]

This occurs whenever the frozen encoder discards information needed to distinguish the paired counterfactuals. No deterministic post-hoc function can reconstruct information that is absent from \(Z^a\).

The correct object is then

\[
P(Z^b\in\cdot\mid Z^a=z,S^b=s'),
\]

not one arbitrary point estimate.

### 3.4 Structured counterfactuals can also be stochastic

Abduction need not be unique. If the factual structured state is compatible with multiple exogenous states, then the symbolic counterfactual is naturally a kernel

\[
K_S^{a\to b}(ds'\mid s)
\]

rather than a deterministic function \(h_S(s)\).

No rank-preservation or full-rank Jacobian assumption is required when the paired simulator directly specifies the joint cross-world law. Regular conditional probabilities are sufficient on standard Borel spaces.

### 3.5 The old training design wastes the strongest supervision

The simulator can generate paired CVs

\[
(X^a,X^b)
\]

for the same persona and style. After applying the frozen encoder, this directly supplies

\[
(Z^a,Z^b).
\]

The old objective does not use \(Z^b\) as the primary target. It instead asks the model to produce *some* point decoded as \(S^b\), then chooses among possible points with \(L_1/L_2\) proximity. This throws away the identifying information available in the paired simulator.

### 3.6 Semantic-only training can produce off-support edits

A flexible manipulator may learn points that fool \(g_e\) into outputting the target semantic state while lying outside the support of real encoder outputs. Paired target embeddings mitigate this problem because the oracle conditional law is supported on embeddings actually generated by the frozen encoder from simulator texts.

A fitted ambient Gaussian can still place some mass outside the empirical support, so support diagnostics remain necessary, but the target itself is no longer defined only through a potentially exploitable probe.

---

## 4. An alternative route considered and why it is no longer the main design

A theoretically clean route was explored in which the encoder itself would be tuned to produce a factorized representation

\[
E(X)=(S,R_{\mathrm{person}},V_{\mathrm{style}}),
\]

with a decoder or inverse map reconstructing the original latent or text. Paired counterfactual orbits could enforce:

- semantic accuracy of \(S\);
- invariance of person and style blocks;
- cross-reconstruction under interventions;
- a residual-preserving lift.

Under strong assumptions, this could yield a clean product or reachable-support decomposition and a unique residual-preserving intervention.

That route was rejected as the **primary** method because it conflicts with the desired frozen-backbone modularity. It remains useful as:

- an optional upper-bound model;
- a fallback when a frozen encoder is empirically inadmissible;
- an ablation showing how much performance is lost by preserving modularity.

The final method therefore avoids requiring a product decomposition of an arbitrary frozen representation.

---

## 5. Final architecture

### 5.1 Notation

Let:

- \(e:\mathcal X\to\mathcal Z_e\) be a fixed encoder;
- \(a\) be a source intervention regime;
- \(b\) be a target intervention regime;
- \(S^a,S^b\in\mathcal S\) be structured potential states;
- \(X^a,X^b\in\mathcal X\) be matched CV potential outcomes;
- \(Z^a=e(X^a)\), \(Z^b=e(X^b)\).

For a VAE encoder, use a deterministic representation such as the posterior mean for the first implementation. If encoder sampling is retained, its randomness must be included in the probability model and simulator coupling.

### 5.2 Semantic analysis kernel

For source regime \(a\), train

\[
G_{e,\eta}^{a}(ds\mid z)
\approx
P(S^a\in ds\mid Z^a=z,a).
\]

A deterministic semantic decoder \(g_e(z)\) is the special case

\[
G_{e,\eta}^{a}(ds\mid z)=\delta_{g_e^a(z)}(ds).
\]

In a shared multi-regime implementation, write this equivalently as \(G_{e,\eta}(ds\mid z,a)\). Suppress \(a\) only when the source regime is globally fixed or regime invariance \(P(S\mid Z,a)=P(S\mid Z)\) has been established.

The structured state must contain the variables needed to answer the chosen SCM counterfactual query. It should include the **counterfactual closure** of the intervention: all modeled variables whose values or textual manifestations may change along the selected causal pathways, plus any structured context needed for abduction.

### 5.3 Symbolic counterfactual kernel

The deployable structured counterfactual kernel is

\[
K_S^{a\to b}(ds'\mid s)
:=
P(S^b\in ds'\mid S^a=s,a,b).
\]

Complete structured orbits identify this regular conditional law on supported cells, and the known SCM can calculate or approximate it by integrating over \(P(U\mid S^a=s,a)\). Stored row-specific \(U_i\) instead supplies an oracle pathwise target. Do not conflate that oracle lookup with deployable inference for a new unit.

If abduction is unique, the kernel reduces to

\[
K_S^{a\to b}(ds'\mid s)
=
\delta_{h_S^{a\to b}(s)}(ds').
\]

If additional observed structured context \(c\) is required, replace \(s\) by the augmented structured state \((s,c)\). Avoid hiding required tabular information inside an unmodeled latent condition.

### 5.4 Encoder-specific stochastic latent editor

Train the regime-explicit conditional editor

\[
Q_{e,\theta}(dz'\mid z,s',a,b)
\approx
P_e(Z^b\in dz'\mid Z^a=z,S^b=s',a,b).
\]

A family notation is equivalent:

\[
Q_{e,\theta}^{a\to b}(dz'\mid z,s').
\]

Conditioning additionally on the factual state \(s\) may improve optimization,

\[
Q_{e,\theta}(dz'\mid z,s,s',a,b),
\]

but is not essential at the conceptual level because \(z\) already contains factual information. The regime-explicit notation is canonical whenever several queries share a model.

The full factual embedding \(z\) acts as an **implicit residual carrier**. There is no assumption that persona and style occupy identifiable coordinates.

### 5.5 Overall latent counterfactual kernel

The final method is

\[
\boxed{
H_{Z,e}^{a\to b}(B\mid z)
=
\int_{\mathcal S}G_e^{a}(ds\mid z)
\int_{\mathcal S}K_S^{a\to b}(ds'\mid s)
Q_e(B\mid z,s',a,b)
}
\]

for measurable \(B\subseteq\mathcal Z_e\).

This defines the proposed modular inference kernel. It equals the directly paired simulator law \(P(Z^b\in B\mid Z^a=z,a,b)\) only under the counterfactual-sufficiency condition stated in Section 9.5. Complete orbits identify the direct law even when the modular factorization fails, so direct-\(H\) versus composed-\(H\) evaluation is required.

In kernel-composition notation:

\[
\boxed{
H_{Z,e}^{a\to b}
=
Q_e^{a\to b}
\circ
K_S^{a\to b}
\circ
G_e^{a},
}
\]

with the factual latent \(z\) retained as side information by \(Q_e\).

### 5.6 Why the mapping is stochastic

The simulator may be deterministic conditional on its full hidden state, persona, style, and random seed. Nevertheless, the inference-time relation conditional only on \(Z^a=z\) is stochastic when multiple hidden states are compatible with the same embedding.

Sources of uncertainty include:

- semantic decoding uncertainty \(S^a\mid Z^a\);
- SCM abduction uncertainty \(S^b\mid S^a\);
- information discarded by the frozen encoder;
- remaining text/rendering variability;
- stochastic encoder output, if retained.

These are sources of **aleatoric counterfactual variation under the simulator-relative estimand**. Finite-data, model-selection, and parameter uncertainty are epistemic and are not automatically represented by one fitted \(Q_e\). Use orbit-level bootstrap, ensembles, or another explicit procedure when uncertainty about the estimated kernel itself matters; never interpret the editor's predicted variance as containing both kinds by default.

### 5.7 Deterministic special case

If

\[
G_e^{a}(\cdot\mid z)=\delta_{g_e^a(z)},
\]

\[
K_S^{a\to b}(\cdot\mid s)=\delta_{h_S^{a\to b}(s)},
\]

and

\[
Q_e^{a\to b}(\cdot\mid z,s')=\delta_{q_e^{a\to b}(z,s')},
\]

then

\[
H_{Z,e}^{a\to b}(\cdot\mid z)
=
\delta_{q_e^{a\to b}(z,h_S^{a\to b}(g_e(z)))}.
\]

Thus the original deterministic architecture is nested inside the new formulation, but it is no longer imposed without evidence.

---

## 6. Simulator and data design

The simulator is the central source of identification.

### 6.1 Controlled variables

For simulated person \(i\), let:

- \(\Pi_i\): fixed persona/person-level context;
- \(\Omega_{ij}\): fixed causally irrelevant renderer state for replicate \(j\);
- \(U_i\): tabular SCM exogenous state;
- \(a\in\mathcal A\): intervention regime.

Generate

\[
\mathbf S_i^a=F_a(U_i)
\]

and

\[
X_{ij}^a\sim P_X(\cdot\mid \mathbf S_i^a,\Pi_i,\Omega_{ij}).
\]

In the LIBERTy CV DGP, persona is a renderer input rather than a parent of the tabular SCM. Fold every random realization intended to be preserved into \(\Omega_{ij}\), then hold \((U_i,\Pi_i,\Omega_{ij})\) fixed while varying \(a\). The symbols deliberately avoid collisions with the repository's SCM columns `C` (professional certificates), `V` (volunteering), and `R` (race).

### 6.2 Style terminology must be precise

Separate:

- **causally affected presentation variables**, which belong in \(S\); and
- **renderer style/noise**, which is held fixed across paired worlds.

For example, if gender causally affects the presentation of skills and that path is being intervened upon, “presentation of skills” cannot be put in the invariant style residual. It is part of \(S\). Only causally irrelevant lexical/layout variation should be fixed as renderer style.

### 6.3 Paired training unit

For source \(a\) and target \(b\), a training unit is

\[
\left(
X_{ij}^a,
X_{ij}^b,
\mathbf S_i^a,
\mathbf S_i^b,
\Pi_i,
\Omega_{ij},
a,b
\right).
\]

Apply the frozen encoder:

\[
Z_{e,ij}^a=e(X_{ij}^a),
\qquad
Z_{e,ij}^b=e(X_{ij}^b).
\]

The practical post-hoc training record is

\[
\boxed{
(Z_e^a,Z_e^b,S^a,S^b,a,b)
}
\]

plus persona/style labels for auxiliary evaluation or regularization.

### 6.4 Split by SCM unit, not by text instance

Train/validation/test splits must be made by `unit_id`, meaning one SCM exogenous state \(U_i\), before orbit rows are expanded into transition pairs. Otherwise, the editor can memorize nearly identical worlds and appear to preserve identity/style on leaked counterfactuals. A repeated `persona_id` is renderer/context metadata, not automatically a unique person; group by persona as an additional restriction only if the scientific interpretation makes it identity-bearing.

### 6.5 Intervention coverage

Every intervention pair used at inference should have direct support in training, or the paper must state an explicit compositional/generalization assumption. Arbitrary extrapolation to unseen semantic states is not identified by paired data alone.

---

## 7. Training with a frozen encoder

### 7.1 What is frozen and what is trained

Frozen:

- the downloaded encoder \(e\);
- preferably its preprocessing and pooling;
- the SCM/simulator definition;
- after pretraining, the semantic probe during editor calibration.

Trained for each encoder:

- semantic analysis model \(G_{e,\eta}\);
- stochastic editor \(Q_{e,\theta}\);
- optional frozen-latent persona/style probes used for diagnostics;
- optional intervention embeddings or small shared heads.

Fitted or calculated once for the shared SCM:

- regime-conditioned deployable \(K_S^{a\to b}\), with stored-\(U\) oracle transitions retained separately for evaluation.

No gradient enters \(e\).

### 7.2 Precompute embeddings

Compute and store all supported orbit embeddings

\[
Z_e^r=e(X^r),\qquad r\in\mathcal R
\]

once. This makes training cheap, reproducible, and unambiguously post-hoc.

For a VAE, initially use

\[
Z_e=\mu_e(X)
\]

rather than a newly sampled posterior latent on every epoch.

### 7.3 Train the semantic probe

Use a proper supervised loss:

\[
\boxed{
\mathcal L_G
=
-\mathbb E\log g_{e,\eta}(S^a\mid Z_e^a,a).
}
\]

For mixed tabular data:

\[
\mathcal L_G
=
\sum_{k\in\mathcal C}
\operatorname{CE}(g_{e,k}(Z),S_k)
+
\sum_{k\in\mathcal N}
\ell_k(g_{e,k}(Z),S_k),
\]

where \(\mathcal C\) and \(\mathcal N\) index categorical and numerical coordinates.

Train \(G_e^a\) across all supported source regimes and freeze it before using the target-regime kernel \(G_e^b\) to calibrate \(Q_e\). This prevents the probe and editor from jointly redefining semantics.

### 7.4 Main paired counterfactual objective

The primary identifying objective is a strictly proper conditional distributional score:

\[
\boxed{
\mathcal L_{\mathrm{CF}}
=
\mathbb E\,
\mathcal S_{\mathrm{dist}}
\left(
Q_{e,\theta}(\cdot\mid Z_e^a,S^b,a,b),
Z_e^b
\right).
}
\]

When the conditional law is dominated by a declared reference measure and \(Q_e\) has density \(q_e\), conditional log loss is the special case

\[
\mathcal L_{\mathrm{CF}}
=
-\mathbb E
\log q_{e,\theta}
\left(
Z_e^b
\mid
Z_e^a,S^b,a,b
\right).
\]

The interpretation is direct:

> Given a frozen factual embedding and the target structured counterfactual state, assign high probability to the frozen embedding of the matched counterfactual CV.

This loss uses the simulator’s cross-world coupling and replaces the old attempt to identify \(Z'\) through semantic consistency plus distance.

For atomic, Dirac, or lower-dimensional targets, use a strictly proper score that admits singular laws, such as a conditional energy score, or explicitly define a smoothed target/reference measure. A Gaussian variance floor prevents collapse but by itself targets a smoothed or best-in-family approximation rather than the exact singular \(Q_e^*\).

### 7.5 Identity objective

When no intervention is requested, anchor the model at the factual embedding:

\[
\boxed{
\mathcal L_{\mathrm{id}}
=
-\mathbb E
\log q_{e,\theta}^{a\to a}
\left(
Z_e^a\mid Z_e^a,S^a
\right).
}
\]

This is a useful anchor and tests whether the editor unnecessarily perturbs representations. For a continuous density model, this NLL encourages concentration near \(Z_e^a\) but does **not** literally guarantee an exact identity sample. If exact identity is part of the interface contract, implement an explicit `a == b` identity/skip branch or a mixed distribution with a Dirac component.

### 7.6 Semantic calibration of generated embeddings

Draw

\[
\widetilde Z^b
\sim
Q_{e,\theta}^{a\to b}(\cdot\mid Z_e^a,S^b)
\]

and use the already trained, frozen semantic probe:

\[
\boxed{
\mathcal L_{\mathrm{sem}}
=
-\mathbb E
\log g_{e,\eta}(S^b\mid\widetilde Z^b).
}
\]

This encourages the distributional analogue of semantic consistency, but it is now auxiliary to direct paired likelihood rather than the sole training signal.

### 7.7 Optional persona/style preservation losses

If the renderer coupling truly preserves persona/context and causally irrelevant style, the paired conditional proper loss targets that preservation law at the population level. It cannot repair a flawed coupling or individually recover information absent from \(Z^a\). Auxiliary losses and audits may improve finite-sample training and reveal violations.

Train frozen diagnostic probes \(r_{\Pi}(\pi\mid z)\) and \(r_{\Omega}(\omega\mid z)\) on genuine embeddings, then penalize generated counterfactuals that do not preserve known labels:

\[
\mathcal L_{\mathrm{res}}
=
-\mathbb E\log r_{\Pi}(\Pi_i\mid\widetilde Z^b)
-
\mathbb E\log r_{\Omega}(\Omega_{ij}\mid\widetilde Z^b).
\]

These probes must be trained on real frozen embeddings and then frozen to avoid collusion.

Retrieval or contrastive losses using the matched target \(Z_e^b\) are also possible, but they should not collapse the stochastic distribution to one point when true conditional variation remains.

### 7.8 Optional composition consistency

When the simulator’s coupling is path-compositional, enforce

\[
Q_e^{b\to c}
\circ
Q_e^{a\to b}
\approx
Q_e^{a\to c}.
\]

Operationally, compare

\[
\widetilde Z^c_{\mathrm{seq}}
\sim
Q_e^{b\to c}(\cdot\mid\widetilde Z^b,S^c)
\]

with

\[
\widetilde Z^c_{\mathrm{direct}}
\sim
Q_e^{a\to c}(\cdot\mid Z^a,S^c)
\]

using a distributional discrepancy.

This is optional. It should be imposed only if the simulator defines the same preserved residual state under sequential and direct interventions.

### 7.9 Full objective

A compact way to list all available terms is

\[
\boxed{
\begin{aligned}
\mathcal L
={}&
\lambda_G\mathcal L_G
+
\lambda_{\mathrm{CF}}\mathcal L_{\mathrm{CF}}
+
\lambda_{\mathrm{id}}\mathcal L_{\mathrm{id}}
\\
&+
\lambda_{\mathrm{sem}}\mathcal L_{\mathrm{sem}}
+
\lambda_{\mathrm{res}}\mathcal L_{\mathrm{res}}
+
\lambda_{\mathrm{comp}}\mathcal L_{\mathrm{comp}}.
\end{aligned}
}
\]

Recommended priority:

1. \(\mathcal L_G\) to establish semantic recoverability;
2. \(\mathcal L_{\mathrm{CF}}\) as the primary editor loss;
3. \(\mathcal L_{\mathrm{id}}\) and \(\mathcal L_{\mathrm{sem}}\);
4. residual and composition objectives only as supported auxiliaries.

The old \(L_1/L_2\) terms may be retained as weak numerical regularizers, but they must not be presented as identifying the true counterfactual. Their sensitivity should be ablated.

The displayed expression is **not** a recommendation to optimize all modules jointly. Training is staged: optimize \(\mathcal L_G\) first, freeze \(G_e\), and then optimize the editor terms with \(\lambda_G=0\). Frozen probes may transmit gradients with respect to their input \(\widetilde Z\) into \(Q_e\), but their own parameters must not update.

### 7.10 Recommended training schedule

**Stage A: Generate paired orbits.**  
Generate every configured intervention world under shared \(U\), persona, and declared renderer state; split by `unit_id` before pair expansion.

**Stage B: Freeze and embed.**  
Precompute \(Z_e^r=e(X^r)\) for every world and encoder under study.

**Stage C: Train and certify \(G_e\).**  
Fit the regime-conditioned semantic kernel on genuine frozen embeddings from all regimes. If semantic recoverability or joint-state validity is inadequate, mark the encoder as unsuitable for the intended intervention.

**Stage D: Fit and validate \(K_S\).**  
Fit or calculate the deployable regime-conditioned SCM kernel from training-unit structured orbits and validate it against held-out stored-\(U\) oracle transitions. For the current finite state space, a smoothed conditional-frequency table is the transparent first baseline; posterior sampling from the known SCM is the reference.

**Stage E: Train \(Q_e\) with teacher-forced targets.**  
Use true \(S^b\) and minimize a strictly proper paired distributional score; use conditional NLL only when the dominated or explicitly smoothed density model is part of the estimand.

**Stage F: Add identity and semantic calibration.**  
Freeze \(G_e\), sample from \(Q_e\), and apply auxiliary losses.

**Stage G: Evaluate the complete inference pipeline.**  
At evaluation, infer \(S^a\) through \(G_e\), obtain \(S^b\) through the SCM kernel, and sample \(Z^b\) through \(Q_e\).

Report both an **oracle-structured** editor evaluation, which feeds true simulator \(\mathbf S^b\) to \(Q_e\), and a **full-pipeline** evaluation, which feeds samples or predictions from \(G_e\) and \(K_S\). The gap measures exposure to semantic/SCM error. Because editor training is teacher-forced on exact \(\mathbf S^b\), consider calibrated perturbations or sampled structured inputs during a later robustness phase if the full-pipeline gap is large; do not blur the clean paired objective in the first baseline.

### 7.11 One minibatch iteration

For each matched pair:

1. Load
   \[
   (Z_e^a,Z_e^b,S^a,S^b,a,b).
   \]
2. Update \(G_e\) using \(S^a\) and \(Z_e^a\), unless the probe is already frozen.
3. Evaluate
   \[
   -\log q_e(Z_e^b\mid Z_e^a,S^b,a,b).
   \]
4. Include identity examples with \(a=b\).
5. Sample
   \[
   \widetilde Z^b\sim Q_e(\cdot\mid Z_e^a,S^b,a,b).
   \]
6. Evaluate semantic and optional residual-preservation probes.
7. Update only the post-hoc modules.

For a Gaussian mixture, gradients from auxiliary losses on sampled \(\widetilde Z^b\) require an explicit estimator: component enumeration, a relaxed/reparameterized mixture assignment, or a score-function estimator. A normalizing flow or single Gaussian has simpler pathwise gradients. Conditional NLL itself does not require differentiating through the sampled component.

### 7.12 Model choice for \(Q_e\)

Start with the simplest model that can represent the observed conditional distribution.

- **Conditional diagonal or low-rank Gaussian:** strong baseline; may be unimodal and over-smooth.
- **Conditional Gaussian mixture / mixture density network:** useful after a single Gaussian and deterministic paired regressor show evidence of multimodality.
- **Conditional normalizing flow:** tractable conditional likelihood and flexible density; attractive for low- or moderate-dimensional VAE latents.
- **Conditional diffusion/score model:** useful for high-dimensional or multimodal embeddings, but complicates likelihood-based exposition.

For the current default scale (nominally \(n=1000\), 128-dimensional LangVAE latents), begin with a standardized residual target \(\Delta Z=Z^b-Z^a\), a deterministic paired regressor, and a conditional diagonal or low-rank Gaussian with a strict variance floor. Move to a mixture or flow only if renderer replicates and held-out proper scores demonstrate that unimodality is inadequate. Mixtures/flows remain easier to explain and evaluate than diffusion once the data justify them.

Two cautions are essential for the first model:

1. With a deterministic encoder and a deterministic paired renderer, the oracle conditional law may contain Dirac or lower-dimensional components and need not have a Lebesgue density. An unconstrained Gaussian likelihood can then shrink variances toward zero. Use a variance floor/explicit smoothing model, a mixed deterministic-stochastic family, or a proper sample score such as the energy score, and state which estimand is being fitted.
2. One unique \(Z^b\) for every nearly unique continuous \(Z^a\) is weak finite-sample evidence for multimodality. Generate renderer replicates and evaluate held-out proper scores against a deterministic paired regressor before claiming that stochasticity is empirically necessary.

---

## 8. Inference

For a new CV \(x\):

1. Freeze-encode:
   \[
   z=e(x).
   \]
2. Sample or estimate the factual structured state:
   \[
   s_m\sim G_e^a(\cdot\mid z)
   =
   G_e(\cdot\mid z,a).
   \]
3. Perform symbolic counterfactual inference:
   \[
   s'_m\sim K_S^{a\to b}(\cdot\mid s_m).
   \]
4. Generate the latent counterfactual:
   \[
   z'_m\sim Q_e(\cdot\mid z,s'_m,a,b).
   \]

The samples

\[
z'_1,\ldots,z'_M
\sim
H_{Z,e}^{a\to b}(\cdot\mid z)
\]

represent uncertainty about the counterfactual embedding.

A deterministic summary such as the conditional mean may be reported, but it should not replace the distribution when the fitted kernel is visibly nondegenerate.

---

## 9. Theory that is now available

The final design supports clean, modest, and defensible theoretical statements. The theory is **simulator-relative and encoder-relative**, not an assertion that a universal true latent counterfactual is identified from observational real-world text.

### 9.1 Standing assumptions

A useful assumption package is:

**A1. Standard Borel spaces.**  
\(\mathcal X,\mathcal Z_e,\mathcal S\) are standard Borel spaces. This covers finite, Euclidean, mixed tabular, and ordinary neural representation spaces.

**A2. Frozen measurable encoder.**  
\(e:\mathcal X\to\mathcal Z_e\) is fixed throughout training and inference.

**A3. Paired simulator coupling.**  
The simulator supplies complete joint orbits \((U,\Pi,\Omega,\{S^r,X^r,Z^r\}_{r\in\mathcal R})\) under the intended shared-\(U\) and renderer coupling. The coupling actually executed by the generator—not merely equal persona/template identifiers—is the target population law.

**A4. Counterfactual sufficiency of the structured state.**  
For every supported \(a,b\),
\[
P(S^b\in ds'\mid S^a=s,Z^a=z,a,b)
=
P(S^b\in ds'\mid S^a=s,a,b)
=
K_S^{a\to b}(ds'\mid s).
\]
If this is false, the direct orbit kernel \(H_e^{a\to b}\) remains identified, but the proposed modular factorization does not. Enlarge \(S\) or condition \(K_S\) on the required \(z\)/observed context and retype the composition.

**A5. Intervention support.**  
The source-target regimes and structured states evaluated at inference have adequate simulator support. In particular, outputs of \(G_e^a\circ K_S^{a\to b}\) lie in the conditioning support on which \(Q_e^{a\to b}\) was trained. Full regime enumeration does not create support for impossible or negligibly rare state combinations.

**A6. Conditional model consistency/realizability.**  
The chosen estimators for \(G_e\), deployable \(K_S\), and \(Q_e\) are correctly specified or form suitable consistent sieves in the probability metrics used in the theorem. Their scoring rules, regularization, and optimization procedures satisfy the corresponding consistency conditions.

**A7. Independent orbit sampling.**  
Independent SCM units/orbits are sampled from a stable declared DGP, \(N_{\mathrm{units}}\to\infty\), and within-orbit transition rows are treated as dependent clusters. The sampling or weighting of regimes defines the target regime mixture. More intervention rows or renderer replicates for one \(U_i\) do not replace independent units.

**A8. Transport for real deployment.**  
Claims on real CVs require a separate sim-to-real assumption or bound relating the simulator and real conditional laws. Unlimited simulation alone does not establish this.

### 9.2 Proposition 1: existence and simulator-relative identification

For a fixed encoder \(e\), complete paired orbits induce a joint law

\[
P_e(U,\Pi,\Omega,\{Z^r,S^r\}_{r\in\mathcal R}).
\]

Under A1, for every supported \(a,b\), the following regular conditional distributions exist and are unique almost surely on their respective conditioning supports:

\[
\begin{aligned}
G_e^{a,*}(ds\mid z)
&=
P_e(S^a\in ds\mid Z^a=z,a),\\
K_S^{a\to b,*}(ds'\mid s)
&=
P_e(S^b\in ds'\mid S^a=s,a,b),\\
Q_e^{a\to b,*}(dz'\mid z,s')
&=
P_e(Z^b\in dz'\mid Z^a=z,S^b=s',a,b),\\
H_e^{a\to b,*}(dz'\mid z)
&=
P_e(Z^b\in dz'\mid Z^a=z,a,b).
\end{aligned}
\]

This is the key identification statement:

> Complete paired simulator orbits identify the regime-conditioned semantic, structured-counterfactual, latent-editor, and direct latent-counterfactual laws on simulator support, without requiring an injective encoder, a disentangled latent space, a full-rank Jacobian, rank preservation, or unique SCM-noise abduction.

Nonunique abduction makes \(K_S\) stochastic; it does not make its population conditional law unidentified. The usual almost-sure qualification is important, especially for continuous \(Z^a\).

Here “identify” refers to the **population joint law induced by the simulator coupling**. A finite paired sample does not nonparametrically determine the kernel at every continuous \((z,\mathbf s')\); estimation still requires smoothness, model-class, regularization, and optimization assumptions, and evaluation is meaningful only on supported regions.

### 9.3 Proposition 2: proper-score recovery

For any of the identified conditional laws above, a strictly proper conditional scoring rule is uniquely minimized at the true regular conditional law, subject to its integrability conditions. This is the primary recovery statement.

As the dominated special case, suppose \(Q_e^{a\to b,*}\) admits a density \(q_e^*\) relative to a declared reference measure and the model class contains it. Then

\[
\begin{aligned}
\mathbb E[-\log q_\theta(Z^b\mid Z^a,S^b,a,b)]
={}&
\mathbb E[-\log q_e^*(Z^b\mid Z^a,S^b,a,b)]
\\
&+
\mathbb E\!
\left[
\operatorname{KL}
\left(
Q_e^{a\to b,*}(\cdot\mid Z^a,S^b)
\Vert
Q_{e,\theta}(\cdot\mid Z^a,S^b,a,b)
\right)
\right].
\end{aligned}
\]

Therefore, every population minimizer recovers \(Q_e^{a\to b,*}\) almost surely, subject to the stated sampling, model, regularization, and optimization assumptions. Analogous proper-score statements apply to probabilistic \(G_e^a\) and an estimated \(K_S^{a\to b}\).

The density assumption is substantive. Existence of a regular conditional probability does **not** imply existence of a Lebesgue density. With deterministic posterior-mean embeddings and a tightly coupled renderer, \(Q_e^*\) may be Dirac or supported on a lower-dimensional set. In that case, either formulate the theorem and estimator with a proper distributional score that permits singular laws, or explicitly define a smoothed observation model/reference measure and a variance floor. Do not describe unconstrained Gaussian NLL as “exact recovery” of a singular target.

### 9.4 Proposition 3: deterministic feasibility criterion

There exists a measurable deterministic editor \(m_e\) satisfying

\[
Z^b=m_e(Z^a,S^b,a,b)
\quad\text{almost surely}
\]

if and only if

\[
Q_e^{a\to b,*}(\cdot\mid Z^a,S^b)
\]

is almost surely a Dirac measure.

For Euclidean \(Z_e\) with finite second moments,

\[
\boxed{
\inf_m
\mathbb E
\left[
\lVert Z^b-m(Z^a,S^b,a,b)\rVert_2^2
\right]
=
\mathbb E
\left[
\operatorname{tr}
\operatorname{Var}(Z^b\mid Z^a,S^b,a,b)
\right].
}
\]

This quantity is an **irreducible counterfactual ambiguity score** for the frozen encoder. A deterministic \(L_2\) model converges to the conditional mean, which can average incompatible latent modes.

More precisely, it is irreducible **for the chosen simulator coupling, conditioning set \((Z^a,\mathbf S^b,a,b)\), coordinate system, and squared loss**. It can include renderer variability and is not an encoder-intrinsic or coordinate-free scalar. Compare it to standardized baselines within an encoder, not as an absolute score across unrelated latent spaces.

This proposition concerns a deterministic **conditional synthesizer** \(m_e(Z^a,\mathbf S^b,a,b)\). A deterministic end-to-end editor depending only on \(Z^a\) for fixed \(a,b\) exists only if the complete composed law

\[
H_{Z,e}^{a\to b}(\cdot\mid Z^a)
\]

is almost surely Dirac, including uncertainty contributed by \(G_e\) and \(K_S\). Degeneracy of \(Q_e(\cdot\mid Z^a,\mathbf S^b)\) alone is not enough.

For intuition, the law of total variance separates two sources of ambiguity:

\[
\operatorname{Var}(Z^b\mid Z^a,a,b)
=
\mathbb E\!\left[\operatorname{Var}(Z^b\mid Z^a,\mathbf S^b,a,b)\mid Z^a,a,b\right]
+
\operatorname{Var}\!\left(\mathbb E[Z^b\mid Z^a,\mathbf S^b,a,b]\mid Z^a,a,b\right).
\]

The first term is residual latent-synthesis ambiguity after the target state is known; the second is ambiguity about which target structured state the semantic/SCM stages produce.

### 9.5 Proposition 4: oracle factorization of the complete latent kernel

Assume, for fixed supported \(a,b\),

\[
P(S^b\in ds'\mid S^a=s,Z^a=z,a,b)
=
P(S^b\in ds'\mid S^a=s,a,b)
=
K_S^{a\to b,*}(ds'\mid s),
\]

which is the counterfactual-sufficiency condition for \(S\). Then, by iterated conditioning,

\[
\boxed{
P(Z^b\in B\mid Z^a=z,a,b)
=
\int G_e^{a,*}(ds\mid z)
\int K_S^{a\to b,*}(ds'\mid s)
Q_e^{a\to b,*}(B\mid z,s').
}
\]

Thus the proposed composition equals the simulator’s directly identified oracle latent counterfactual kernel \(H_e^{a\to b,*}\). The direct \(H_e^{a\to b,*}\) exists from the paired orbit law even if this factorization assumption fails; direct-versus-composed \(H\) is therefore an important diagnostic.

If the conditional independence above is not plausible, the remedy is not a rank assumption. The remedy is to enlarge the structured state or let the SCM kernel condition on the missing observed context.

For rigorous proofs, \(Q_e\circ K_S\circ G_e\) is shorthand because \(Q_e\) retains the original factual \(z\). Lift the intermediate kernels to the augmented state \((z,\mathbf s)\) so that ordinary Markov-kernel composition is well typed. Also condition every kernel explicitly on source/target regimes when those are not fixed globally.

### 9.6 Proposition 5: stochastic commutation

Let \(G_e^b\) be the target-regime semantic measurement kernel on generated embeddings. Suppose the editor is semantically calibrated:

\[
(G_e^b\circ Q_e^{a\to b})(A\mid z,s')
=
\mathbf 1\{s'\in A\}
\]

for measurable \(A\subseteq\mathcal S\). Then

\[
\boxed{
G_e^b\circ H_{Z,e}^{a\to b}
=
K_S^{a\to b}\circ G_e^a.
}
\]

This is the correct stochastic replacement for

\[
g_e(h_Z(z))=h_S(g_e(z)).
\]

It says that analyzing after latent editing gives the same structured distribution as analyzing first and then applying the symbolic counterfactual kernel.

In practice, exact equality is replaced by a calibrated discrepancy measured on held-out simulator data.

The displayed equality is a **strong special case**, not an automatic consequence of fitting paired data. If class-conditional latent laws overlap, even the Bayes-optimal semantic posterior need not return \(\delta_{s'}\) on every generated embedding. Present exact commutation only under explicit deterministic semantic recoverability/separability assumptions; otherwise state and measure an approximate distributional calibration criterion with an independent held-out probe.

### 9.7 Proposition 6: modular error propagation

Let

\[
H=Q\circ K\circ G,
\qquad
\widehat H=\widehat Q\circ\widehat K\circ\widehat G.
\]

If

\[
\sup_z d_{\mathrm{TV}}(\widehat G(\cdot\mid z),G(\cdot\mid z))\leq\varepsilon_G,
\]

\[
\sup_s d_{\mathrm{TV}}(\widehat K(\cdot\mid s),K(\cdot\mid s))\leq\varepsilon_K,
\]

and

\[
\sup_{z,s'} d_{\mathrm{TV}}(\widehat Q(\cdot\mid z,s'),Q(\cdot\mid z,s'))\leq\varepsilon_Q,
\]

then Markov-kernel contraction and the triangle inequality yield

\[
\boxed{
\sup_z d_{\mathrm{TV}}(\widehat H(\cdot\mid z),H(\cdot\mid z))
\leq
\varepsilon_G+\varepsilon_K+\varepsilon_Q.
}
\]

For bounded downstream \(p\),

\[
\left|
\mathbb E_{\widehat H(\cdot\mid z)}p(Z')
-
\mathbb E_{H(\cdot\mid z)}p(Z')
\right|
\leq
2\lVert p\rVert_\infty
(\varepsilon_G+\varepsilon_K+\varepsilon_Q),
\]

under the conventional definition of total variation. Check the factor-of-two convention in the final manuscript.

Total variation is often too strong for continuous or singular latent laws: for example, a narrow Gaussian and a Dirac measure can have maximal TV despite close downstream behavior. Keep the TV result as an abstract lemma if useful, but add Wasserstein or bounded-Lipschitz error propagation for Lipschitz downstream predictors and use metrics compatible with the fitted model in experiments.

### 9.8 What theory is deliberately not claimed

The project should not claim:

- identification of a unique “true” latent point;
- identification of the encoder’s latent coordinates;
- a global product decomposition \(Z\simeq S\times R\);
- exact preservation for every arbitrary encoder;
- validity outside simulator support without transport assumptions;
- causal identification of the SCM from observational CV data;
- coordinate-invariant meaning of \(L_1/L_2\) latent distance.

---

## 10. Encoder admissibility and certification

The abstract method can be instantiated for any compatible frozen representation satisfying the interface; the present implementation target is a stable finite-dimensional vector encoder. Not every encoder will be informative enough. Applicability should therefore be an empirical certification problem.

### 10.1 Semantic recoverability

Evaluate held-out proper loss or calibration for

\[
G_e^a(S^a\mid Z_e^a)
=
G_e(S^a\mid Z_e^a,a).
\]

If the intervention-relevant structured state cannot be recovered from \(Z_e\), the semantic bridge is unreliable.

### 10.2 Counterfactual predictability

Evaluate held-out conditional NLL, energy score, calibration, or coverage for

\[
Q_e(Z_e^b\mid Z_e^a,S^b,a,b).
\]

Compare against baselines that ignore \(Z^a\), ignore \(S^b\), or use a deterministic regressor.

### 10.3 Deterministic ambiguity

Estimate the residual conditional variance or compare deterministic and stochastic predictive scores. A large irreducible variance indicates that the frozen encoder does not support exact pathwise editing from \(Z^a\).

### 10.4 Persona/style preservation

Use known simulator labels to evaluate whether generated \(Z'\):

- retrieves the correct person;
- preserves fixed renderer style;
- changes only the intended structured state;
- matches the paired target distribution.

These are evaluation properties, not evidence that persona/style coordinates are identifiable.

### 10.5 Support and realism

Evaluate whether generated embeddings lie near the support of genuine encoder outputs, for example through:

- held-out conditional likelihood or density ratio diagnostics;
- nearest-neighbor distances after whitening;
- two-sample tests between generated and true \(Z^b\);
- downstream decoder quality when a compatible decoder exists.

### 10.6 Admissibility conclusion

A defensible claim is:

> The method is architecturally compatible with any fixed vector encoder satisfying the interface, while encoder-specific retraining and empirical certification determine whether it retains sufficient information for the chosen counterfactual task.

Do not claim equal performance or exact preservation for all encoders.

---

## 11. Distributional downstream fairness

The original single-point penalty should become distributional.

For a predictor \(p\), a natural objective is

\[
\boxed{
\mathcal L_{\mathrm{fair}}
=
\mathbb E_{Z^a}
\mathbb E_{Z'\sim H_{Z,e}^{a\to b}(\cdot\mid Z^a)}
\left[
\ell\bigl(p(Z^a),p(Z')\bigr)
\right].
}
\]

Depending on the application, one may compare:

- expected predictions;
- full predictive distributions;
- quantiles or worst-case deviations;
- paired Monte Carlo predictions using common residual draws where available.

The paper must state whether fairness is defined in expectation, in distribution, or almost surely. These are different requirements.

The displayed expected pairwise loss is a candidate training/evaluation functional; it is not automatically equivalent to the formal counterfactual-fairness definition. State what is conditioned on, whether renderer randomness is coupled or marginalized, and whether equality is required for each individual/context or only after averaging over the simulator-induced kernel.

---

## 12. Original proposal versus final proposal

| Dimension | Original proposal | Final frozen-backbone kernel proposal |
|---|---|---|
| Encoder | Frozen pretrained VAE | Any fixed encoder; no parameter updates |
| Encoder-specific training | Semantic decoder and deterministic manipulator | Semantic probe and stochastic conditional editor |
| Latent counterfactual object | One point \(Z'=h_Z(Z)\) | Conditional law \(H_{Z,e}^{a\to b}(dZ'\mid Z)\) |
| Main supervision | Semantic consistency through \(g\) | Paired simulator target \(Z_e^b\) |
| Main editor loss | Consistency + \(L_1/L_2\) displacement | Conditional NLL or other proper score |
| Role of \(g/G\) | Sole bridge supervising \(h_Z\) | Semantic analysis and calibration component |
| Selection within semantic fiber | Euclidean sparsity/minimality | Learned paired conditional distribution |
| Persona/style preservation | Hoped for through small displacement | Learned from matched pairs by conditioning on full factual \(z\); tested explicitly |
| Product decomposition | Implicitly desirable but absent | Not required |
| Abduction | Mostly deterministic \(h_S\) | Kernel \(K_S\), deterministic as special case |
| Identifiability claim | Implicit point counterfactual | Simulator-relative kernel identification on support |
| Arbitrary encoder information loss | Hidden failure mode | Represented as conditional uncertainty and measured |
| Commuting diagram | Equality of functions | Equality/calibration of Markov kernels |
| Downstream fairness | Compare \(p(z)\) with one \(p(z')\) | Average or compare distributions over \(z'\) |
| Modularity | Frozen encoder, but weakly supervised editor | Frozen encoder plus lightweight encoder-specific probabilistic bridge |

---

## 13. The minimum conceptual change necessary

The proposal does not need to abandon its original neurosymbolic modularity. The minimum necessary change is:

### Old object

\[
h_Z(z)
\]

trained to satisfy

\[
g(h_Z(z))=h_S(g(z))
\]

with Euclidean minimality.

### New object

\[
Q_e(dz'\mid z,s',a,b)
\]

trained from matched simulator pairs, then composed with semantic and symbolic kernels:

\[
H_{Z,e}^{a\to b}
=
Q_e^{a\to b}\circ K_S^{a\to b}\circ G_e^a.
\]

This one conceptual replacement resolves several problems simultaneously:

- nonunique semantic fibers;
- arbitrary coordinate-dependent selection;
- hidden encoder information loss;
- stochastic abduction;
- stochastic synthesis;
- failure to exploit paired simulator supervision.

The rest of the original architecture can remain recognizable:

- a frozen encoder still produces \(Z\);
- a semantic module still grounds \(Z\) in \(S\);
- an explicit SCM still defines the symbolic operation;
- a learned module still constructs latent counterfactuals;
- downstream fairness still compares factual and counterfactual predictions.

---

## 14. Required manuscript revisions

### 14.1 Abstract and contribution

Replace “learns a latent transformation analogous to a symbolic counterfactual” with language such as:

> For each frozen encoder, we learn a stochastic conditional editor from simulator-generated matched counterfactual pairs. The editor estimates the encoder-specific conditional distribution of counterfactual embeddings under an explicit symbolic intervention.

Emphasize simulator-relative identification and frozen-backbone modularity.

### 14.2 Problem formulation

Define potential structured states, texts, and embeddings:

\[
(S^a,X^a,Z^a),
\qquad
Z^a=e(X^a).
\]

Define the target as

\[
H_{Z,e}^{a\to b}(\cdot\mid z)
=
P_e(Z^b\in\cdot\mid Z^a=z),
\]

or its conditional-editor factorization through \(S^b\).

### 14.3 Procedure

Retain two broad phases but redefine them:

1. train a semantic analysis kernel \(G_e\) on frozen embeddings;
2. train a stochastic conditional editor \(Q_e\) on matched frozen embedding pairs.

The SCM kernel sits between them at inference.

### 14.4 Replace the old latent loss

The primary loss should be

\[
-\log q_e(Z_e^b\mid Z_e^a,S^b,a,b).
\]

Semantic consistency becomes an auxiliary calibration term. \(L_1/L_2\) penalties become optional regularizers or baselines.

### 14.5 Rewrite the data section

The simulator must generate **counterfactual orbits**, not one independent text per tabular row. State explicitly that persona and causally irrelevant renderer style are fixed across interventions.

### 14.6 Rewrite the consistency equation

Use

\[
G_e\circ H_{Z,e}^{a\to b}
=
K_S^{a\to b}\circ G_e
\]

as the ideal stochastic commutation relation.

### 14.7 Add theorem section

Include at minimum:

1. joint-orbit existence/identification of \(G_e^{a,*}\), \(K_S^{a\to b,*}\), \(Q_e^{a\to b,*}\), and direct \(H_e^{a\to b,*}\);
2. population recovery under a proper scoring rule;
3. deterministic-if-and-only-if-Dirac criterion;
4. the counterfactual-sufficiency condition under which \(G\to K\to Q\) equals direct \(H\);
5. regime-correct stochastic commutation;
6. modular error decomposition.

### 14.8 Update downstream fairness

Use Monte Carlo integration over \(H_{Z,e}^{a\to b}\), not one arbitrary counterfactual point.

### 14.9 Update limitations

Explicitly discuss:

- encoder admissibility;
- simulator-relative identification;
- sim-to-real transport;
- intervention support;
- semantic-state sufficiency;
- conditional-density model misspecification.

---

## 15. Recommended experiments

### 15.1 Core synthetic experiment

For multiple held-out people and styles:

1. generate full intervention orbits;
2. encode with several frozen encoders;
3. train separate \((G_e,Q_e)\) modules;
4. evaluate paired conditional prediction and semantic calibration.

### 15.2 Baselines

Compare:

- original deterministic consistency + \(L_1/L_2\) manipulator;
- deterministic paired regressor \(m(z,s')\);
- stochastic paired editor \(Q_e(z'\mid z,s')\);
- editor ignoring factual \(z\): \(Q_e(z'\mid s')\);
- editor ignoring target semantics: \(Q_e(z'\mid z)\);
- optional tuned-encoder factorized model as an upper bound.

### 15.3 Metrics

Use:

- semantic prediction/calibration for \(G_e\);
- conditional NLL or proper score for \(Q_e\);
- coverage of predictive regions;
- two-sample distance between generated and true target embeddings;
- identity error;
- persona/style preservation;
- composition consistency where justified;
- downstream fairness and utility;
- sensitivity to encoder choice;
- uncertainty versus deterministic error.

### 15.4 Demonstrate the fiber problem

Construct several embeddings that decode to the same \(S^b\) but differ substantially in paired-target likelihood, persona/style probes, or downstream prediction. Show that semantic consistency alone cannot select correctly.

### 15.5 Demonstrate coordinate sensitivity

Apply controlled invertible transformations or whitening/scaling to embeddings and show that the old \(L_1/L_2\) solution changes, while paired conditional targets remain defined relative to the chosen fixed representation.

### 15.6 Sim-to-real evaluation

Treat real-data use as transport, not automatic identification. Evaluate probe calibration and uncertainty shift. The theoretical guarantees remain conditional on a simulator-to-real approximation assumption.

---

## 16. Claims the paper can make

A strong but defensible formulation is:

> We introduce a frozen-backbone neurosymbolic method that learns, for any compatible fixed vector encoder, an encoder-specific stochastic counterfactual editing kernel from paired simulator-generated CVs. The SCM controls the structured intervention, while the post-hoc editor learns the conditional law of the corresponding frozen embeddings. Under standard regularity, support, and estimation conditions, this kernel is identified relative to the simulator coupling and can be estimated by a suitable conditional proper-loss procedure. A deterministic editor is justified only when the relevant composed conditional law is degenerate.

Additional useful claims:

- The method does not require encoder fine-tuning.
- It does not require an identifiable or disentangled latent coordinate system.
- The same SCM and training recipe can be reused across encoders.
- Uncertainty caused by encoder information loss is represented rather than hidden.
- Paired simulator supervision replaces arbitrary latent minimality as the main identifying signal.

---

## 17. Claims the paper should avoid

Avoid statements implying that:

- every arbitrary encoder is guaranteed to work;
- the method recovers the unique true latent counterfactual;
- \(L_1/L_2\) minimality has causal meaning;
- semantic consistency alone identifies person-preserving edits;
- all causal assumptions live only in the tabular graph;
- simulator validity automatically transfers to real CVs;
- fixed persona/style labels imply identifiable numerical residual coordinates.

The cross-world simulator coupling and tabular-to-text renderer are themselves substantive assumptions.

---

## 18. Compact algorithmic specification

### Training

**Input:** paired simulator records \((X^a,X^b,S^a,S^b,a,b)\), fixed encoder \(e\).

```text
1. Freeze encoder e.
2. Precompute z_a = e(x_a), z_b = e(x_b) for every paired record.
3. Train semantic kernel G_e(s | z, a) on all supported orbit worlds.
4. Freeze G_e.
5. Fit/calculate deployable K_S(s_b | s_a, a, b) from training-unit
   structured orbits; validate it against held-out stored-U oracle transitions.
6. Train editor Q_e(z_b | z_a, s_b, a, b) using a strictly proper
   distributional score; use NLL only for an explicit dominated/smoothed law.
7. Add identity examples a -> a.
8. Optionally add:
      - semantic calibration through frozen G_e,
      - persona/style preservation through frozen auxiliary probes,
      - composition consistency when the simulator coupling supports it.
9. Evaluate on held-out SCM units and secondary held-out renderer contexts.
10. Certify or reject the encoder for the intended counterfactual task.
```

### Inference

```text
Input: CV x, source regime a, target regime b.
1. z = e(x).
2. Draw/estimate s ~ G_e(. | z, a).
3. Draw s' ~ K_S^{a->b}(. | s).
4. Draw z' ~ Q_e(. | z, s', a, b).
5. Repeat to approximate H_{Z,e}^{a->b}(. | z).
6. Aggregate downstream predictions according to the declared fairness target.
```

---

## 19. Repository-grounded implementation blueprint

This is the recommended order of work against the current repository. Later stages depend on the data contract established earlier; do not begin by replacing the neural network in `src/latent_intervention.py` while the orbit and SCM-inference semantics remain undefined.

### 19.1 Lock the scientific query and identifiers

Before generating more paid text, write a small machine-readable intervention specification that fixes:

- the protected/intervened node;
- source and target regimes;
- total versus path-specific intervention semantics;
- the counterfactual closure included in \(\mathbf S\);
- which renderer variables are invariant across worlds;
- whether source regime \(a\) is observed, decoded from \(\mathbf S\), or marginalized;
- which inference-time version of \(K_S^{a\to b}\) is used;
- the declared downstream fairness functional.

The current code implements a **total** `do(G=1)` and propagates it to all descendants. The manuscript may discuss path-specific fairness as motivation, but the implementation must not claim path-specific editing until edge/path interventions and allowed versus forbidden paths are explicitly represented.

Define identifiers with separate meanings:

- `unit_id`: one SCM exogenous state \(U_i\); this is the primary split group;
- `regime_id`: a named intervention world such as `do_G_0` or `do_G_1`;
- `renderer_id`: one invariant renderer replicate for that unit;
- `persona_id` and `template_id`: renderer/context metadata, not automatically unique people;
- `pair_id`: derived from `(unit_id, renderer_id, source_regime, target_regime)`.

The current `persona_id` is sampled from a pool and may repeat across SCM units. Decide whether persona text is identity-bearing. If it is, either generate unique profiles or group-split by persona as well; otherwise describe it as shared renderer context rather than person identity.

### 19.2 Replace the two-table convention by a long orbit dataset

Extend the simulator runner to materialize a long table with at least:

```text
unit_id, regime_id, renderer_id,
all structured-state columns,
persona_id, template_id,
renderer_seed/noise, concrete rendered values,
text, prompt/model metadata
```

Generate the complete configured regime grid for every query claimed in the paper. For binary gender this includes at least `do(G=0)` and `do(G=1)` for every \(U_i\), not merely an observational row and `do(G=1)`. Derive every claimed direction and identity transition from the orbit. If the scientific scope is deliberately smaller, encode that restriction in the query manifest and claims rather than relying on unseen-transition generalization.

Preserve the exact shared-noise SCM coupling already implemented in `exp/sim/scm.py`. Keep the Python and reference R structural equations synchronized.

### 19.3 Make text pairing auditable

The renderer coupling is part of the causal estimand, not cosmetic preprocessing.

- Reuse the same persona and narrative template across every world in an orbit replicate.
- The current prompt explicitly allows adjusting persona or template details "for coherence". Remove or constrain that permission for paired generation; fixed IDs alone do not guarantee that the realized persona/style stays fixed.
- For an identity world with unchanged \(\mathbf S\), reuse the exact same text; a second remote LLM call would manufacture irrelevant variation.
- For changed worlds, prefer generating the complete orbit in one structured LLM response or use a deterministic renderer for the core theorem experiment. Temperature 0 across independent remote calls is not an exact common-random-number guarantee.
- Store prompt version/hash, model/deployment, generation settings, and raw/parsed response metadata.
- Replace `_done_ids` with compound-key resume logic. A single integer ID cannot distinguish regimes or renderer replicates.
- Treat concrete values within binned age/work categories as renderer variables. Draw a common base uniform/noise value once and deterministically map it into each regime's bin, rather than relying implicitly on repeated `rng.integers` calls with different bounds.
- Audit that only intended structured changes and unavoidable surface realization changes occur across paired texts.

Multiple renderer replicates are not required for population identification when independent complete orbits sample from the declared renderer law. Generate them for at least a diagnostic subset because they are strongly useful for separating renderer variation from finite-sample/model uncertainty and for testing multimodality and preservation.

### 19.4 Split first, then expand pairs

Create and persist train/validation/test manifests **by `unit_id` before constructing transition pairs**. Every regime and renderer replicate belonging to a unit must remain in the same split.

Useful secondary evaluations are:

- unseen persona/context IDs;
- unseen templates/renderer styles;
- seen versus unseen intervention transitions, if a compositional claim is eventually made.

All normalization, whitening, calibration, density fitting, and hyperparameter selection must use training/validation splits only. The optional VAE fine-tuning code is also text-row split and would need group-safe splits if retained as an upper-bound ablation.

### 19.5 Cache all encoder worlds with provenance

For every frozen encoder, cache \(Z_e^{a}=e(X^a)\) for all orbit rows, then construct paired records:

\[
(Z_e^a,Z_e^b,\mathbf S^a,\mathbf S^b,a,b,\text{metadata}).
\]

Each cache should record:

- encoder name and checkpoint hash;
- latent dimension;
- deterministic posterior mean versus sampled representation;
- tokenizer/preprocessing and maximum length;
- source-data hash and split-manifest hash;
- code/config revision.

Namespace all models and reports by encoder. The current generic artifact paths would overwrite results when modularity is tested across several encoders.

### 19.6 Upgrade and certify the semantic kernel \(G_e\)

Reuse the existing categorical-head MLP as the baseline, but:

- train on genuine embeddings from all regimes;
- expose categorical probabilities, not only argmax predictions;
- report per-variable NLL, Brier score, calibration, class-conditional accuracy, and joint-state validity;
- handle imbalance explicitly;
- test whether independent head samples yield impossible or unsupported combinations of \(\mathbf S\).

Independent softmax heads define a factored approximation, not necessarily the true joint \(G_e(d\mathbf s\mid z,a)\). A deterministic joint point estimate is an acceptable first baseline. If uncertainty propagation matters empirically, consider an autoregressive structured head or constrained joint sampler.

Reject or qualify an encoder when the intervention-relevant state is not recoverable. This is an empirical admissibility result, not a training failure to hide.

### 19.7 Implement and evaluate the symbolic kernel \(K_S\)

Provide two clearly named paths:

1. **Oracle synthetic kernel:** use the stored \(U_i\) to produce the exactly paired \(\mathbf S_i^b\). This isolates editor quality.
2. **Deployable inferred kernel:** given only \(\mathbf S^a\) and source regime, infer/sample the compatible exogenous state and propagate under \(b\), or estimate the corresponding conditional kernel from simulator pairs.

The transparent first baseline for the current finite discrete state space is a smoothed regime-conditioned transition table fitted from a large cheap structured-orbit sample. Stronger/reference implementations include rejection or importance sampling from the known noise priors, analytic interval/truncated-noise calculations where feasible, or an amortized posterior. Validate every inferred kernel against held-out oracle paired simulator targets before placing it inside the full latent pipeline.

If an inferred \(K_S\) must be deferred, a direct regime-conditioned model \(H_e(dz^b\mid z^a,a,b)\) can be run as a reduced baseline. It does not implement the claimed semantic-SCM-editor factorization and must be labeled accordingly, not used to imply that the neurosymbolic inference path is operational.

### 19.8 Add \(Q_e\) as a new model, keep \(h_Z\) as a baseline

Do not rename the current class and change its loss. Retain it as `deterministic_consistency_l1_l2`. Add a separate `StochasticLatentEditor` with explicit distribution methods such as:

```text
log_prob(z_target, z_source, target_state, source_regime, target_regime)
sample(..., n_samples)
mean_or_summary(...)  # diagnostic only
```

Recommended first comparison:

- deterministic paired regressor \(m_e(Z^a,\mathbf S^b,a,b)\);
- conditional diagonal Gaussian;
- conditional diagonal-Gaussian mixture with variance floor;
- optional conditional flow only if the mixture is inadequate.

A Gaussian mixture can be implemented in PyTorch without adding a dependency. A flow requires an explicit dependency and reproducibility review.

Train on the paired proper objective, use true \(\mathbf S^b\) first, and add auxiliaries only after the primary likelihood/score is working. Evaluate variance collapse, component use, scale sensitivity, and whether samples—not just means—lie near genuine target embeddings.

### 19.9 Evaluate at three distinct levels

Do not collapse all failures into one end-to-end score.

**Level 1: semantic analysis**

- held-out recoverability and calibration of \(G_e\);
- supported/joint structured states;
- sensitivity by encoder and regime.

**Level 2: oracle editor**

- conditional NLL when the density assumptions are appropriate;
- energy score or another proper score for non-dominated targets;
- calibration/coverage and sharpness;
- deterministic paired error;
- generated-versus-true target two-sample distances;
- semantic validity through both the training-time frozen \(G_e\) and an independent held-out probe/audit;
- identity, persona/context, renderer preservation;
- nearest-neighbor/support and decoded-text diagnostics;
- ablations omitting \(Z^a\), \(\mathbf S^b\), or regime labels.

**Level 3: composed inference and fairness**

- oracle versus inferred \(G_e/K_S\) gap;
- Monte Carlo stability of \(H_{Z,e}^{a\to b}\);
- downstream predictor utility for the simulated outcome `Q`/manuscript \(Y\);
- the declared expectation-, distribution-, quantile-, or worst-case fairness functional;
- sim-to-real shift/calibration separately from simulator-relative validity.

Quantify sampling uncertainty by resampling **orbit/unit IDs**, not individual transition rows or Monte Carlo draws from a fitted editor. Generated draws are not independent evidence about the fitted model's population performance.

Scores involving raw densities or Euclidean distances are generally not comparable across encoders with different dimensions/scales. Compare models within an encoder, report whitening/normalization, and use downstream or calibrated distributional metrics for cross-encoder conclusions.

### 19.10 Reproducibility, configuration, and tests

Extend configuration with:

- regime grid and renderer replicates;
- compound artifact schema and split manifests;
- encoder registry;
- \(G_e\) calibration settings;
- \(Q_e\) family, mixture components, scale parameterization, and variance floor;
- staged auxiliary weights;
- SCM-kernel mode (`oracle` versus `inferred`);
- Monte Carlo sample counts and fairness definition;
- explicit old-method baseline configuration.

Seed Python, NumPy, Torch initialization, minibatch shuffling, and generation streams consistently. Save resolved configuration and hashes with every artifact. Note that NumPy is used directly but is not currently an explicit dependency in `pyproject.toml`; add it rather than relying on transitive installation.

Add Python tests for:

- shared exogenous-state orbit invariants;
- exact identity text reuse;
- compound-key uniqueness and resumability;
- renderer coupling and bin-value mapping;
- zero unit leakage across splits;
- encoder cache alignment/provenance;
- frozen encoder and frozen-probe gradients;
- mixture `log_prob`, sampling, numerical stability, and save/load;
- explicit identity behavior;
- oracle and inferred \(K_S\) validation;
- composed-inference shapes/reproducibility;
- metric and end-to-end smoke tests.

The current module `__main__` checks are useful smoke tests but not a sufficient regression suite.

### 19.11 Minimal completion criteria for the first revised experiment

The first implementation is complete only when all of the following are true:

1. a group-safe, versioned orbit dataset contains paired texts and embeddings in both directions;
2. the frozen encoder is never updated and its provenance is stored;
3. \(G_e\) is trained on all regimes, calibrated, frozen, and reported separately;
4. deployable \(K_S^{a\to b}\) is fitted or calculated from training-unit structured orbits and validated against held-out stored-\(U\) oracle transitions;
5. the old deterministic editor, a deterministic paired regressor, and a stochastic paired editor all run on the same splits;
6. the stochastic editor is scored against held-out paired \(Z^b\), not merely through \(G_e\);
7. direct-oracle \(H\), oracle-structured \(Q_e\), and full composed inference are reported separately;
8. uncertainty is empirically assessed rather than assumed;
9. distributional fairness and utility are computed for a declared functional;
10. tests protect pairing, leakage, freezing, support, and probabilistic-model behavior;
11. manuscript claims match the actually executed intervention, SCM kernel, encoder, renderer coupling, and simulator support.

---

## 20. Important implementation choices still open

The next agent should not reopen the frozen-versus-tuned encoder decision unless new evidence demands it. The following choices remain genuinely open:

1. **Semantic-kernel family:** a probabilistic, calibrated \(G_e^a\) is required for the claimed stochastic composition; deterministic point prediction remains a special-case baseline.
2. **Deployable SCM estimator:** choose among a smoothed discrete transition table, known-SCM posterior sampling, analytic truncated-noise calculation, or an amortized posterior. Retain stored-\(U\) oracle evaluation regardless of this choice.
3. **Editor family:** Gaussian mixture, normalizing flow, or diffusion.
4. **Conditioning:** canonical \((z,s',a,b)\) versus augmentation by factual state/context \((z,s,s',a,b)\).
5. **Intervention parameterization:** one shared editor with intervention tokens versus a family of smaller editors.
6. **Representation from VAE:** posterior mean versus explicitly modeled encoder samples.
7. **Fairness notion:** equality in expectation, distributional equality, or a risk/quantile criterion.
8. **Support metric:** Euclidean after whitening, learned metric, likelihood, or downstream-task metric.
9. **Real-data transport:** explicit domain-adaptation assumption, calibration set, or bounds under distribution shift.

Recommended first prototype:

- deterministic/frozen encoder output;
- probabilistic semantic probe if feasible;
- oracle SCM target for editor isolation plus a separately validated inferred SCM kernel;
- standardized residual prediction with a deterministic paired baseline and a conditional diagonal/low-rank Gaussian with a variance floor, conditioned on \((z,\mathbf S^b,a,b)\); add a mixture only if held-out evidence supports it;
- paired conditional NLL when the dominated/smoothed density assumption is explicit, otherwise a strictly proper sample score;
- semantic and identity auxiliaries;
- multiple frozen encoders for the modularity experiment.

---

## 21. Decision history for the next agent

This is a concise rationale, not a hidden reasoning transcript.

1. **Initial concern:** Neither an arbitrary VAE coordinate system nor a partial semantic decoder identifies a unique latent counterfactual.
2. **First theoretical response:** Interpret semantic consistency as identifying a target fiber; consider a residual-preserving product decomposition to select a unique lift.
3. **Simulator opportunity:** The repository's SCM already supplies exact shared-noise tabular pairs, and the renderer has deterministic per-ID hooks for reusing persona/template choices. The text generator still needs to be extended to generate and audit matched counterfactual CVs. Once implemented, those pairs provide much stronger supervision than semantic consistency.
4. **First positive design:** Tune an encoder into semantic/person/style blocks and use cross-reconstruction. This would support a clean residual-preserving lift theorem.
5. **Modularity constraint:** The collaborator wants the encoder to remain untouched so the method can attach to downloaded models.
6. **Resolution:** Do not force a product decomposition. Retain the entire factual embedding as the carrier of preserved information and directly estimate
   \[
   P(Z^b\mid Z^a,S^b,a,b).
   \]
7. **Final conceptual shift:** Replace deterministic point editing and coordinate-dependent minimality with an encoder-specific stochastic conditional kernel trained by paired proper loss.
8. **Resulting theory:** Identification is now of a simulator-induced conditional law on frozen-encoder support. Determinism becomes a testable degeneracy condition, and modular errors can be decomposed across semantic analysis, SCM inference, and latent synthesis.

---

## 22. Immediate next work products

A next agent can continue efficiently by producing, in this order:

1. a revised formal problem statement using potential texts and embeddings;
2. polished theorem statements and complete proofs for Propositions 1-6;
3. a revised architecture figure showing \(e\), \(G_e\), \(K_S\), and \(Q_e\);
4. a revised training section with complete paired orbit generation and a strictly proper conditional distributional score;
5. a simulation protocol and encoder-admissibility evaluation plan;
6. a revised abstract, contributions list, discussion, and limitations;
7. pseudocode and a minimal implementation using deterministic paired residual regression and a conditional diagonal/low-rank Gaussian, with mixtures/flows reserved for evidence of multimodality;
8. an ablation comparing the original deterministic \(L_1/L_2\) editor with the paired stochastic editor.

---

## 23. One-sentence handoff

The project should remain modular by freezing any compatible vector encoder, but it should stop treating semantic consistency plus latent distance as identification of one counterfactual point; instead, use audited complete LIBERTy-style orbits to identify and fit regime-conditioned \(G_e^a\), deployable \(K_S^{a\to b}\), and encoder-specific \(Q_e(dz'\mid z,s',a,b)\), verify when their composition equals the directly paired oracle law, and retain determinism as a testable special case.
