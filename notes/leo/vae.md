# Variational Autoencoders for Text

## Architecture: Four-Phase Pipeline for Counterfactual Fairness

- **Phase 0** — Pretrained text VAE (encoder + decoder). The key design choice is producing a **single, fixed-dimensional continuous latent vector** per input (not token-level representations). This is non-trivial: standard models like BERT/Sentence-BERT don't have this structure, making their latents unsuitable for manipulation.
- **Phase 1** — Semantic decoder: a small MLP from frozen latent z → tabular metadata. Acts as a consistency oracle and interpretability handle.
- **Phase 2** — Neural manipulator: trained to mimic a *symbolic manipulator* acting on metadata. Loss = discrepancy between `semantic_decoder(manipulated_z)` and `symbolic_manipulator(metadata)`. This is the direct LatPlan analogue.
- **Phase 3/4** — Predictor on manipulated latents, enforcing counterfactual fairness by requiring predictions to be consistent across real and counterfactual inputs.

---

## VAE Design Reasoning

The core challenge is that text VAEs are harder to manipulate than image VAEs. Two critical failure modes from the literature:

1. **Posterior collapse** — the encoder ignores the latent and the decoder learns to operate without it (KL term → 0). Requires careful KL annealing or free-bits techniques.
2. **Latent holes** (Xu et al., 2020) — when you manipulate a latent vector, you can land in regions not covered by the aggregate posterior, causing the decoder to produce nonsense. This is a fundamental obstacle to Phase 2 working at all.

This is why model choice matters: the latent space must be structured enough that arithmetic/manipulation stays on the manifold. Optimus was the prior state-of-the-art for this, but **LangVAE** (2025) supersedes it via KV cache injection — the latent is linearly projected into the decoder's Key-Value cache, keeping base decoder weights frozen, reducing parameters by >95% while improving latent structure.

| | Optimus (2020) | LangVAE (2025) |
|---|---|---|
| Backbone | BERT + GPT-2 (fixed) | Any HF encoder + decoder (modular) |
| Latent injection | All-layer embedding injection | KV cache injection; decoder frozen |
| Param efficiency | Full fine-tuning | >95% reduction vs. Optimus |
| Analysis tooling | Community notebooks | LangSpace: vector arithmetic, interpolation, disentanglement metrics (MIG, DCI, z-diff) |
| Status | Unmaintained | EMNLP 2025, actively maintained |

**T5VQVAE** is an alternative with discrete latents — closer to LatPlan's original binary latent spirit, where the manipulator becomes a genuinely symbolic discrete operator rather than regression in continuous space.

---

## Alternatives Ranked

| Option | Fit | Why |
|---|---|---|
| **LangVAE** | ★★★★★ | Fixed-dim continuous latent + modular + demonstrated manipulability + LangSpace tooling |
| **T5VQVAE** | ★★★★ | Discrete latents; more LatPlan-aligned; outperforms Optimus on controllability |
| **DELLA / LD4LG** | ★★★★ | Good latent structure, but adds diffusion complexity |
| **Sentence-T5 or Sentence-BERT + learned decoder** | ★★★ | Pragmatic but latent isn't KL-regularized; may hurt manipulability |
| **β-VAE on transformer backbone** | ★★★ | Disentanglement helps manipulator; harder to train |
| **Optimus** | ★★★ | Superseded by LangVAE; dated codebase |
| **Token-level BERT / plain RNN-VAE (Bowman)** | ★★ | Token-level latents too high-dimensional and unstructured |

---

## Literature Review

### LatPlan Lineage

- **Asai & Fukunaga (2018/2022)** — original LatPlan; SAE learns propositional states, AMA learns symbolic action models. Your manipulator is an AMA analogue for text. JAIR version explicitly notes the architecture doesn't assume images and could apply to text.
- **Takata & Fukunaga (2023)** — latent plans can be "invalid" w.r.t. the ground-truth domain; your semantic decoder loss is the direct fix for this failure mode.
- **Asai (2019)** — extension to first-order logic; relevant if metadata has relational structure.

### Counterfactual Fairness

- **Kusner et al. (2017)** — foundational definition: a decision is fair if it matches the counterfactual world with the sensitive attribute flipped.
- **Ma et al., CLAIRE (2023)** — structurally nearest to your full pipeline: maps observed variables to latent space and enforces counterfactual fairness as a constraint on embeddings.
- **Joo et al. (2024)** — module on frozen pretrained generator that constructs a fair latent space via disentanglement; remarkably close to Phase 0+2 structure.
- **Chiappa (2019)** — path-specific counterfactual fairness; relevant if the manipulator should only neutralize certain causal pathways.
- **Grari et al., GCFN (2023)** — critical caveat: argues VAE-based approaches can still be correlated with sensitive attributes; VAEs have "weak capability in constructing latent representations."

### Causal Representation Learning

The theoretical foundation underlying both the HSIC argument and the manipulator's non-identifiability concerns. If Phase 2 is framed as "learning to perform a causal intervention in latent space," this is the literature that tells you when that is even possible.

- **Schölkopf et al. (2021)** — *Towards Causal Representation Learning*. The field-defining survey; connects disentanglement, ICA, and causality, and argues that robust/transferable representations require causal structure rather than purely statistical dependencies. Frames the problem your manipulator is solving.
- **Locatello et al. (2019, ICML best paper)** — *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations*. **The impossibility result.** Proves that unsupervised disentanglement is fundamentally impossible without inductive biases on both models and data. Every disentangled representation can be transformed into an entangled one with the same marginal distribution. This is why your manipulator needs the supervision signal from the semantic decoder — without it, there is no ground truth about what "the right direction" in latent space is.
- **Hyvärinen & Morioka (2016, 2017)** — nonlinear ICA using temporal structure / time-contrastive learning. First positive identifiability results for nonlinear ICA, showing auxiliary information breaks the impossibility.
- **Hyvärinen, Sasaki & Turner (2019)** — *Nonlinear ICA Using Auxiliary Variables*. Generalizes the above: if you have an auxiliary variable u (e.g., metadata, environment label) that the latent factors depend on, identifiability can be recovered. **Your metadata is exactly such an auxiliary variable** — this paper is the theoretical justification for why Phase 1's semantic decoder is more than just a consistency check.
- **Khemakhem et al., iVAE (2020)** — *Variational Autoencoders and Nonlinear ICA: A Unifying Framework*. Practical iVAE construction: conditioning the prior on auxiliary variables makes the VAE identifiable up to a simple transformation. Directly applicable to your pipeline — the metadata can serve as the iVAE auxiliary input.
- **von Kügelgen et al. (2021, NeurIPS)** — *Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style*. Shows that paired views (original + augmentation) identify content variables while leaving style unconstrained. The "counterfactual pairs" framing closest to how your manipulator training data is structured: (x, x_counterfactual) tied via metadata edits.
- **Locatello et al. (2020)** — *Weakly-Supervised Disentanglement Without Compromises*. Shows paired samples sharing some factors is sufficient for disentanglement — a practical middle ground between fully supervised and unsupervised. Your metadata-paired training regime falls into this category.
- **Brehmer et al. (2022, NeurIPS)** — *Weakly Supervised Causal Representation Learning*. Identifies causal variables from paired pre-/post-intervention observations. **This is structurally identical to Phase 2**: the symbolic manipulator is the intervention, the original and manipulated latents are the pre/post pair.
- **Ahuja et al. (2022, 2023)** — *Interventional Causal Representation Learning* / *Weakly Supervised Representation Learning with Sparse Perturbations*. Formalizes identifiability under sparse interventions; quantifies how many intervention types are needed to identify the latent causal structure. Relevant for deciding how many distinct metadata edits you need in training.
- **Lippe et al., CITRIS (2022) / BISCUIT (2023)** — temporal causal representation learning with known/weakly-known interventions. Useful if your metadata edits have a compositional structure over time.
- **Wang & Jordan (2021)** — *Desiderata for Representation Learning: A Causal Perspective*. Argues that good representations should support counterfactual reasoning, which requires structural assumptions beyond statistical independence. Philosophical grounding for the whole project.

**Why this matters for Phase 2:** The manipulator's training signal — paired (original latent, manipulated latent) tied to a symbolic metadata edit — is precisely the weakly supervised CRL setting. Reading Brehmer et al. (2022) and von Kügelgen et al. (2021) gives you the vocabulary and identifiability results to defend the manipulator-based approach against the "no identifiability guarantee" objection. The honest position: you get identifiability *conditional on* the symbolic manipulator being a true intervention on a real latent factor, not free identifiability the way HSIC claims.

### Text VAE Manipulation

- **Xu et al. (2020)** — identifies the latent hole problem; proposes constraining posterior mean to a probability simplex. Must-read before Phase 2.
- **Hu et al. (2017)** — VAE + attribute discriminators for controlled text generation; semantic decoder plays an analogous disentangling role.
- **Liu et al., LatentOps (2022)** — ODE-based composable operators in GPT-2 latent space; their "plug-in operators" are conceptually identical to the neurosymbolic manipulator.
- **Li et al., Optimus (2020)** — prior backbone candidate; superseded by LangVAE.
- **Vaeth et al., RegDiff (2025)** — diffusion in VAE latent space for controllable style manipulation.
- **Anonymous (ICLR)** — unsupervised discovery of interpretable latent directions in language VAEs; useful baseline your manipulator should outperform.

### Manchester/Freitas Group (LangVAE lineage)

- **Mercatali & Freitas (2021)** — discrete VAEs for text with explicit generative factors; starting point of the group.
- **Carvalho et al. (2023, EACL)** — disentangled representations for NL definitions using syntactic/semantic regularities.
- **Zhang et al., LlaMaVAE (2023)** — beats Optimus on LM, STS, and definition modeling; increased semantic clustering and geometric consistency.
- **Zhang et al., T5VQVAE (2024, EACL)** — discrete/VQ variant; outperforms Optimus on controllability. The LatPlan-aligned option.
- **Zhang et al. (2024, NAACL)** — graph-induced syntactic injection into transformer VAEs; useful if metadata is relational.
- **Carvalho et al. (2025, EMNLP)** — LangVAE paper itself.
- **Zhang/Carvalho/Freitas (survey)** — compares VAE, VQ-VAE, Sparse AE through lens of compositional semantics; theoretical framing directly aligned with this project's motivation. Read early.

[Medium article](https://medium.com/@rodolfojaamorim/variational-auto-encoders-meet-llms-fe4286b9177d)

### Semantic Decoders / Property Predictors

- **Louizos et al., CEVAE (2017)** — VAE with auxiliary predictors for causal inference; structural precursor to Phase 1.
- **Tu et al., PCAE (2022)** — plug-in conditional AE framework; analogous Phase 1 setup with a broadcasting net for injecting control signals.

---

## Manipulator (Phase 2) vs. HSIC

### Case FOR Phase 2 Manipulator

**Encoder reuse and compositionality.** HSIC bakes a specific notion of bias into the encoder at training time. If you later discover a new sensitive attribute, you retrain from scratch. A single general-purpose VAE paired with many lightweight manipulators lets you add new interventions cheaply. Compositionality also holds: you can chain manipulators (remove gender bias, then age bias), whereas HSIC encoders trained to enforce independence from multiple attributes simultaneously degrade reconstruction quality quickly as the number of attributes grows.

**Interpretability and verifiability.** The HSIC-debiased encoder is a black box — you can't point at any dimension and say "this encodes attribute A." A linear Phase 2 manipulator is a literal concept vector: inspectable, steerable, visualizable. The semantic decoder provides a second interpretability handle — you can verify the manipulator moved the gender field while leaving other metadata unchanged. This maps directly onto LatPlan's strength: the latent is inspectable as propositions, actions are inspectable as STRIPS operators.

**Counterfactual explanations.** HSIC produces a fair predictor but no explanations — there is no operation "show me what this input would look like if the sensitive attribute were flipped." The manipulator produces this for free. Joo et al. (2024) argue this is the primary justification for their module-on-frozen-generator design: fair *prediction* and counterfactual *explanation* are distinct goals, and only the manipulator approach delivers both.

**Testability of the neurosymbolic bridge.** With HSIC, you trust the penalty worked; failure manifests only downstream when the predictor is still unfair. Phase 1+2 has a built-in check: if the semantic decoder loss stays high, the manipulator is not working, and you know before running the predictor.

**Weaker assumptions.** Theorem 1 (identifiability result) requires condition A1: sufficient variability of S over the latent space. These conditions are non-trivial and generally unverifiable in the wild. The manipulator-based approach shifts the burden to a weaker, empirically testable assumption: that there exists a smooth mapping in latent space corresponding to the metadata edit, visible directly from semantic decoder loss convergence.

### Case AGAINST (Where HSIC is Stronger)

**No identifiability guarantee.** Theorem 1 gives a formal guarantee that the HSIC-debiased subspace is an invertible transformation of the true bias-free latent. A learned manipulator has no such guarantee — non-linear ICA results show VAE latents are generically non-identifiable, and non-identifiability of latents implies non-identifiability of counterfactual queries. The manipulator might produce metadata-consistent decoded output while changing the latent in a direction that isn't the "true" counterfactual, leaving residual bias in correlated latent dimensions the predictor still exploits.

**Counterfactual leakage.** If the bias-free and spurious latents are correlated in training data (which they almost always are), a manipulator that only edits the "gender-decodable" part of the latent leaves the correlation channels intact. The decoder reconstructs text that still carries gender signals through unmanipulated but correlated dimensions. HSIC severs these correlations structurally; Phase 2 manipulation doesn't, unless you explicitly penalize them — at which point you're reinventing HSIC with extra steps.

**Brittle out-of-distribution latents.** Combined with the latent hole problem (Xu et al., 2020), a manipulator trained on observed pairs may push latents into unpopulated regions, producing decoded text that is nonsense or implausible as a counterfactual.

### Synthesis

| Priority | Approach |
|---|---|
| Formal guarantee; single known bias; encoder retraining OK | HSIC |
| Encoder reuse; interpretability; counterfactual explanations; LatPlan story | Phase 2 manipulator |
| Maximum robustness | Hybrid: HSIC at Phase 0 for known bias axes + Phase 2 for post-hoc interventions |

**Recommended framing:** *"HSIC-based identifiability requires bias-specific encoder training. We instead propose a modular approach where a general-purpose encoder is paired with learned symbolic-consistency-constrained manipulators, trading identifiability guarantees for compositionality, interpretability, and encoder reuse — following the LatPlan architectural paradigm."*

The HSIC approach, as an established technique, represents an engineering contribution when applied to text. The LatPlan-style manipulator — where the semantic decoder acts as a neurosymbolic bridge enforcing consistency between latent edits and symbolic metadata operations — is the genuinely novel contribution.

---

## Suggested Reading Order

1. **Asai et al. (2022) JAIR** — north star for the neurosymbolic framing
2. **Kusner et al. (2017)** — ground the fairness framing
3. **Locatello et al. (2019)** — the impossibility result; explains why supervision is needed
4. **Khemakhem et al. (2020, iVAE)** — how auxiliary variables (metadata) restore identifiability
5. **Brehmer et al. (2022)** — weakly supervised CRL from paired interventions; theoretical backbone for Phase 2
6. **Ma et al. (2023, CLAIRE)** — structurally closest to the full pipeline
7. **Xu et al. (2020)** — understand the latent vacancy problem before hitting it
8. **Zhang/Carvalho/Freitas survey** — theoretical framing for why structured latents work
9. **Liu et al. (2022, LatentOps)** — manipulator framework
10. **Joo et al. (2024)** — fair latent space paper most aligned with the phase structure
11. **Carvalho et al. (2025, LangVAE)** — the backbone
