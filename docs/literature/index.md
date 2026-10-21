# Literature index

**207 records; 11 primary clusters.** Local collection through 2026-10-16, reorganized and extended by a [targeted web search](imports/2026-10-19-web-search.md) on 2026-10-19 using the updated draft. All earlier records are retained; search priorities are not a final citation shortlist.

The [distilled review draft](distilled-review.md) selects a provisional 48-paper core and three conditional leads without changing this complete catalog. The compact [manuscript-shaped draft](manuscript-draft.md) uses 33 references in eight topic paragraphs with run-in headings, supported by [focused closest-work checks](reviews/2026-10-19-closest-work.md). It was integrated into the [main paper's related-work section](../../overleaf/sections/2_related.tex) on 2026-10-19; the complete catalog remains unchanged.

Each paper has exactly one primary entry below. Cross-links between clusters expose overlaps without duplicating records. The cluster/subgroup assignment is organizational triage, not a verified statement about a paper's assumptions or results. These are library shelves, **not eleven proposed manuscript subsections**.

Status: **V** = basic metadata verified; **R** = recovered, verify before citing; **?** = unresolved identity/version issue. New web candidates have primary-source-checked metadata and explicit screening limits; none of the records has a full-paper reading note yet. See the [README](README.md), [narrative and component map](related-work-map.md), [organization review](reviews/2026-10-19-organization.md), and [unresolved leads](unresolved.md).

| Primary cluster | Records | Connection to the planned paper |
| --- | ---: | --- |
| [01. Benchmarks, hiring and causal fairness](#01-benchmarks-and-fairness) | 15 | LIBERTy and application motivation |
| [02. Fair and invariant representations](#02-fair-representations) | 39 | Alternative fairness objectives |
| [03. Frozen text encoders and language VAEs](#03-text-encoders) | 18 | LangVAE / Nomic / Qwen3 / EmbeddingGemma |
| [04. Semantic decoding, concepts and probing](#04-semantic-decoding) | 14 | Structured semantic decoder g |
| [05. Causal abstraction and neurosymbolic alignment](#05-causal-abstraction) | 17 | Symbolic target and latent alignment |
| [06. Semantic editing and controllable text](#06-semantic-editing) | 27 | semantic_editor and neighboring editing tasks |
| [07. Causal generative models and state-conditioned flows](#07-causal-flows) | 16 | state_flow and shared-noise transport |
| [08. Optimal transport and geometric counterfactual maps](#08-optimal-transport) | 18 | ot_editor / ot_flow / ot_gaussian |
| [09. Text realization, reconstruction and embedding inversion](#09-text-realization) | 12 | Text-decoder variants and reconstruction |
| [10. Identification and causal representation learning](#10-identification) | 22 | Limits of counterfactual guarantees |
| [11. Evaluation validity and experimental protocol](#11-evaluation) | 9 | Cross-method and decoder evaluation |
| **Total** | **207** | All collected records retained |

<a id="01-benchmarks-and-fairness"></a>

## 01. Benchmarks, hiring and causal fairness

Application motivation, the meaning of fairness, and benchmark supervision. LIBERTy belongs here as the experimental setting, not as an intervention algorithm.

Comparison question: What ground truth and causal assumptions does the benchmark supply, and what would a successful edit establish about fairness?

### Hiring, discrimination and bias measurement

- [Are Emily and Greg More Employable Than Lakisha and Jamal? A Field Experiment on Labor Market Discrimination](papers/intro-race.md) — 2004; R.
- [Gender and the labor market: What have we learned from field and lab experiments?](papers/intro-gender.md) — 2014; R.
- [Algorithmic Unfairness through the Lens of EU Non-Discrimination Law: Or Why the Law is not a Decision Tree](papers/weerts2023.md) — 2023; R.
- [Exploring the Jungle of Bias: Political Bias Attribution in Language Models via Dependency Analysis](papers/jenny2024jungle-bias.md) — 2024; R.
- [Fairness and Bias in Algorithmic Hiring: A Multidisciplinary Survey](papers/fabris-fairness-2025.md) — 2025; R.

### Causal fairness targets and decision-making

- [A continuous framework for fairness](papers/hacker-continuous-2017.md) — 2017; R.
- [Counterfactual Fairness](papers/kusner2017counterfactual.md) — 2017; R.
- [Path-Specific Counterfactual Fairness](papers/chiappa-path-specific-2019.md) — 2019; R.
- [Causal Fairness Analysis](papers/plecko2024causal-fairness.md) — 2024; V.
- [Mind the Gap: A Causal Perspective on Bias Amplification in Prediction & Decision-Making](papers/plecko2024mind-gap.md) — 2024; V.
- [Causal Bias Detection in Generative Artificial Intelligence](papers/plecko2026generative-bias.md) — 2026; V.

### Text and resume benchmarks

- [Bias in Bios: A Case Study of Semantic Representation Bias in a High-Stakes Setting](papers/de2019bias.md) — 2019; R.
- [CEBaB: Estimating the Causal Effects of Real-World Concepts on NLP Model Behavior](papers/abraham2022cebab.md) — 2022; V.
- [LIBERTy: A Causal Framework for Benchmarking Concept-Based Explanations of LLMs with Structural Counterfactuals](papers/toker-liberty-2026.md) — 2026; R.
- [PopResume: Causal Fairness Evaluation of LLM/VLM Resume Screeners with Population-Representative Dataset](papers/yu2026popresume.md) — 2026; V.

Related clusters: [02. Fair and invariant representations](#02-fair-representations); [04. Semantic decoding, concepts and probing](#04-semantic-decoding); [11. Evaluation validity and experimental protocol](#11-evaluation).

<a id="02-fair-representations"></a>

## 02. Fair and invariant representations

Alternative ways to remove, preserve or control sensitive information, including pre-processing and learned representations. This large inherited field is retained in full; its size is not a proposed citation budget.

Comparison question: How do invariance, group or individual fairness, and encoder retraining differ from an explicit intervention on a frozen embedding?

### Adversarial, information-based and controllable invariance

- [Learning Fair Representations](papers/zemel-learning-2013.md) — 2013; R.
- [Censoring Representations with an Adversary](papers/edwards-censoring-2016.md) — 2016; R.
- [Optimized Pre-Processing for Discrimination Prevention](papers/calmon-optimized-2017.md) — 2017; R.
- [Controllable Invariance through Adversarial Feature Learning](papers/xie-controllable-2017.md) — 2017; R.
- [Data Pre-Processing for Discrimination Prevention: Information-Theoretic Optimization and Analysis](papers/calmon-data-2018.md) — 2018; R.
- [Adversarial Removal of Demographic Attributes from Text Data](papers/elazar2018removal.md) — 2018; V.
- [Learning Adversarially Fair and Transferable Representations](papers/madras-learning-2018.md) — 2018; R.
- [Invariant Representations without Adversarial Training](papers/moyer-invariant-2018.md) — 2018; R.
- [Learning Fair Representations via an Adversarial Framework](papers/feng-learning-2019.md) — 2019; R.
- [Learning Controllable Fair Representations](papers/song-learning-2019.md) — 2019; R.
- [Invariant Representations through Adversarial Forgetting](papers/jaiswal-invariant-2020.md) — 2020; R.
- [Fair Representation for Safe Artificial Intelligence via Adversarial Learning of Unbiased Information Bottleneck](papers/kim-fair-2020.md) — 2020; R.
- [Conditional Learning of Fair Representations](papers/zhao-conditional-2020.md) — 2020; R.
- [Learning Unbiased Representations via Renyi Minimization](papers/grari-learning-2021.md) — 2021; R.
- [Counterfactual Invariance to Spurious Correlations in Text Classification](papers/veitch2021counterfactual.md) — 2021; R.
- [Learning Bias-Invariant Representation by Cross-Sample Mutual Information Minimization](papers/zhu-learning-2021.md) — 2021; R.
- [Generating Fair Universal Representations Using Adversarial Models](papers/kairouz-generating-2022.md) — 2022; R.

### Disentanglement, latent debiasing and counterfactual fairness models

- [The Variational Fair Autoencoder](papers/louizos-variational-2016.md) — 2016; R.
- [Flexibly Fair Representation Learning by Disentanglement](papers/creager-flexibly-2019.md) — 2019; R.
- [Fair Attribute Classification Through Latent Space De-Biasing](papers/ramaswamy2021fair.md) — 2021; R.
- [Learning Fair Representation via Distributional Contrastive Disentanglement](papers/oh-learning-2022.md) — 2022; R.
- [Fair Representation Learning through Implicit Path Alignment](papers/shui-fair-2022.md) — 2022; R.
- [Learning for Counterfactual Fairness from Observational Data](papers/ma2023clairelearning.md) — 2023; R.
- [Counterfactual Fairness for Predictions using Generative Adversarial Networks](papers/ma2023counterfactualgan.md) — 2023; V.
- [Constructing Fair Latent Space for Intersection of Fairness and Explainability](papers/joo2024constructing.md) — 2024; V.

### Individual fairness, guarantees, kernel and linear representations

- [Provably Fair Representations](papers/mcnamara-provably-2017.md) — 2017; R.
- [Fair Kernel Learning](papers/perez-fair-2017.md) — 2017; R.
- [The Price of Fair PCA: One Extra dimension](papers/samadi-price-2018.md) — 2018; R.
- [iFair: Learning Individually Fair Data Representations for Algorithmic Decision Making](papers/lahoti-ifair-2019.md) — 2019; R.
- [Operationalizing Individual Fairness with Pairwise Fair Representations](papers/lahoti-operationalizing-2019.md) — 2019; R.
- [Learning Fair Representations for Kernel Models](papers/li-learning-2020.md) — 2020; R.
- [Learning Certified Individually Fair Representations](papers/ruoss-learning-2020.md) — 2020; R.
- [Efficient fair principal component analysis](papers/kamani-efficient-2022.md) — 2022; R.

### Limited labels, transfer, federated and reward-model settings

- [FairVFL: A Fair Vertical Federated Learning Framework with Contrastive Adversarial Learning](papers/qi-fairvfl-2022.md) — 2022; R.
- [Don't Throw it Away! The Utility of Unlabeled Data in Fair Decision Making](papers/rateike-dont-2022.md) — 2022; R.
- [Semi-FairVAE: Semi-supervised Fair Representation Learning with Adversarial Variational Autoencoder](papers/wu-semi-2022.md) — 2022; R.
- [Fair Transfer Learning with Factor Variational Auto-Encoder](papers/liu-fair-2023.md) — 2023; R.
- [Eliminating Inductive Bias in Reward Models with Information-Theoretic Guidance](papers/li2025eliminating.md) — 2025; R.
- [Debiasing Reward Models by Representation Learning with Guarantees](papers/ng2025debiasing.md) — 2025; R.

Related clusters: [01. Benchmarks, hiring and causal fairness](#01-benchmarks-and-fairness); [06. Semantic editing and controllable text](#06-semantic-editing); [08. Optimal transport and geometric counterfactual maps](#08-optimal-transport).

<a id="03-text-encoders"></a>

## 03. Frozen text encoders and language VAEs

The representation being edited: language-VAE latents, sentence/document embeddings, dimension choices and training geometry. Encoder quality alone is not evidence of causal editability or invertibility.

Comparison question: What information and geometry does each frozen encoder make available, at what input length and latent dimension?

### Embedding architectures, pooling and geometry

- [A Structured Self-attentive Sentence Embedding](papers/lin2017structured-self-attention.md) — 2017; V.
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](papers/devlin2019bert.md) — 2019; V.
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](papers/reimers2019sentence-bert.md) — 2019; V.
- [On the Sentence Embeddings from Pre-trained Language Models (BERT-flow)](papers/li2020bert-flow.md) — 2020; V.
- [Matryoshka Representation Learning](papers/kusupati2022matryoshka.md) — 2022; R.
- [Jina Embeddings: A Novel Set of High-Performance Sentence Embedding Models](papers/gunther2023jina-embeddings.md) — 2023; V.

### Encoder families in the project comparison

- [Nomic Embed: Training a Reproducible Long Context Text Embedder](papers/nussbaum2024nomic.md) — 2024; V.
- [LangVAE and LangSpace: Building and Probing for Language Model VAEs](papers/carvalho2025langvae.md) — 2025; R.
- [Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models](papers/qwen2025embedding.md) — 2025; R.
- [EmbeddingGemma: Powerful and Lightweight Text Representations](papers/vera2025embeddinggemma.md) — 2025; V.

### Language VAEs, semantic control and posterior training

- [Generating Sentences from a Continuous Space](papers/bowman2016generating.md) — 2016; R.
- [Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing](papers/fu2019cyclical-annealing.md) — 2019; V.
- [Optimus: Organizing Sentences via Pre-trained Modeling of a Latent Space](papers/li2020optimus.md) — 2020; R.
- [On Variational Learning of Controllable Representations for Text without Supervision](papers/xu2020variational.md) — 2020; R.
- [Learning Disentangled Representations for Natural Language Definitions](papers/carvalho2023definitions.md) — 2023; R.
- [LlaMaVAE: Guiding Large Language Model Generation via Continuous Latent Sentence Spaces](papers/zhang2023llamavae.md) — 2023; R.
- [Graph-Induced Syntactic-Semantic Spaces in Transformer-Based Variational AutoEncoders](papers/zhang2024graph.md) — 2024; R.
- [Improving Semantic Control in Discrete Latent Spaces with Transformer Quantized Variational Autoencoders](papers/zhang2024improving.md) — 2024; R.

Related clusters: [04. Semantic decoding, concepts and probing](#04-semantic-decoding); [07. Causal generative models and state-conditioned flows](#07-causal-flows); [09. Text realization, reconstruction and embedding inversion](#09-text-realization).

<a id="04-semantic-decoding"></a>

## 04. Semantic decoding, concepts and probing

The semantic decoder g: recovering structured state information and uncertainty from embeddings. This is distinct from reconstructing text with D_text.

Comparison question: What is recoverable about S, how are joint dependencies represented, and what can semantic prediction alone certify?

### Probing, mediation and concept geometry

- [What you can cram into a single vector: Probing sentence embeddings for linguistic properties](papers/conneau2018probing.md) — 2018; R.
- [Probing Multilingual Sentence Representations With X-Probe](papers/xprobe2019.md) — 2019; R.
- [Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias](papers/vig2020mediation.md) — 2020; R.
- [Explore Spurious Correlations at the Concept Level in Language Models for Text Classification](papers/zhou2024explore.md) — 2024; R.
- [The Geometry of Categorical and Hierarchical Concepts in Large Language Models](papers/park2025concept-geometry.md) — 2025; R.

### Concept bottlenecks and completeness

- [Interpreting Pretrained Language Models via Concept Bottlenecks](papers/tan2023concept-bottlenecks.md) — 2023; ?.
- [Interpretable-by-Design Text Understanding with Iteratively Generated Concept Bottleneck](papers/ludan2024text-bottleneck.md) — 2024; V.
- [Towards Achieving Concept Completeness for Textual Concept Bottleneck Models](papers/bhan2025concept-completeness.md) — 2025; V.
- [Concept Bottleneck Large Language Models](papers/sun2025cbllm.md) — 2025; V.
- [Concept Bottleneck Models](papers/koh2020concept-bottleneck.md) — 2020; V.
- [Post-hoc Concept Bottleneck Models](papers/yuksekgonul2023posthoc-cbm.md) — 2023 (2022 preprint); V.
- [Stochastic Concept Bottleneck Models](papers/vandenhirtz2024stochastic-cbm.md) — 2024; V.
- [Post-hoc Stochastic Concept Bottleneck Models](papers/hoffmann2025posthoc-stochastic-cbm.md) — 2025/2026 preprint; V.
- [Counterfactual Concept Bottleneck Models](papers/dominici2025counterfactual-cbm.md) — 2025; V.

Related clusters: [05. Causal abstraction and neurosymbolic alignment](#05-causal-abstraction); [09. Text realization, reconstruction and embedding inversion](#09-text-realization); [11. Evaluation validity and experimental protocol](#11-evaluation).

<a id="05-causal-abstraction"></a>

## 05. Causal abstraction and neurosymbolic alignment

Connecting a specified symbolic intervention to neural representations or computation. Planning and abstraction are neighboring problem settings, not automatically equivalent to editing a frozen CV embedding.

Comparison question: What must align between the symbolic model and latent intervention, and what supervision or deployment access is required?

### Causal alignment, intervention training and proxy models

- [Causal Abstractions of Neural Networks](papers/geiger2021abstractions.md) — 2021; V.
- [Inducing Causal Structure for Interpretable Neural Networks](papers/geiger2022iit.md) — 2022; R.
- [Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations](papers/geiger2023das.md) — 2023; V.
- [Interpretability at Scale: Identifying Causal Mechanisms in Alpaca](papers/wu2023boundless-das.md) — 2023; R.
- [Causal Proxy Models for Concept-based Model Explanations](papers/wu2023cpm.md) — 2023; V.
- [Causal Consistency of Structural Equation Models](papers/rubenstein2017causal-consistency.md) — 2017; V.

### Symbol grounding, latent planning and algorithmic structure

- [Classical Planning in Deep Latent Space: Bridging the Subsymbolic-Symbolic Boundary](papers/asai2018classical.md) — 2018; R.
- [Unsupervised Grounding of Plannable First-Order Logic Representation from Images](papers/asai2019unsupervised.md) — 2019; R.
- [MDP Homomorphic Networks: Group Symmetries in Reinforcement Learning](papers/vanderpol2020mdp.md) — 2020; V.
- [Neural Algorithmic Reasoning](papers/velickovic2021algorithmic.md) — 2021; V.
- [Classical Planning in Deep Latent Space](papers/asai2022classical.md) — 2022; R.
- [Plausibility-Based Heuristics for Latent Space Classical Planning](papers/takata2023plausibility.md) — 2023; R.

### Neurosymbolic fairness

- [Neural-Symbolic Integration for Fairness in AI](papers/wagner-neural-2021.md) — 2021; R.
- [ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias](papers/adriaensen-problog4fairness-2026.md) — 2026; R.
- [A neurosymbolic approach to counterfactual fairness](papers/heilmann2026neurosymbolic.md) — 2026; R.

### Limits of interpreting an alignment

- [Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching](papers/makelov2023illusion.md) — 2023; V.
- [The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?](papers/sutter2025dilemma.md) — 2025; V.

Related clusters: [04. Semantic decoding, concepts and probing](#04-semantic-decoding); [06. Semantic editing and controllable text](#06-semantic-editing); [10. Identification and causal representation learning](#10-identification).

<a id="06-semantic-editing"></a>

## 06. Semantic editing and controllable text

Editing concepts, attributes or behavior in neural representations and generated text. Organize by the edited object and source of the target, not by a shared use of the word counterfactual.

Comparison question: Does the method edit a frozen document embedding, a model activation or source text, and what fixes the intended change and preserved content?

### Post-hoc concept editing and invertible interpretation

- [Beyond Concept Bottleneck Models: How to Make Black Boxes Intervenable?](papers/laguna2024beyond-cbm.md) — 2024; V.
- [A Disentangling Invertible Interpretation Network for Explaining Latent Representations](papers/esser2020iin.md) — 2020; V.
- [Interpretable Generative Models through Post-hoc Concept Bottlenecks](papers/kulkarni2025posthoc-generative-cbm.md) — 2025; V.

### Concept erasure, affine steering and representation editing

- [LEACE: Perfect linear concept erasure in closed form](papers/belrose2023leace.md) — 2023; V.
- [Steering Llama 2 via Contrastive Activation Addition](papers/panickssery2023caa.md) — 2023; V.
- [Representation Engineering: A Top-Down Approach to AI Transparency](papers/zou2023repe.md) — 2023; V.
- [Explaining Text Classifiers with Counterfactual Representations](papers/lemberger2024counterfactual-text-representations.md) — 2024; V.
- [Representation Surgery: Theory and Practice of Affine Steering](papers/singh2024representation-surgery.md) — 2024; V.
- [ReFT: Representation Finetuning for Language Models](papers/wu2024reft.md) — 2024; V.
- [MUtE: A Dual Framework for Concept Erasure and Counterfactual Interventions](papers/saillenfest2026mute.md) — 2026; V.
- [Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection](papers/ravfogel2020inlp.md) — 2020; V.

### Latent controls and attribute-conditioned generation

- [Toward Controlled Generation of Text](papers/hu2017toward.md) — 2017; R.
- [Plug and Play Language Models: A Simple Approach to Controlled Text Generation](papers/dathathri2019pplm.md) — 2019; V.
- [Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation](papers/entangled2019attribute-transfer.md) — 2019; R.
- [Unsupervised Discovery of Interpretable Latent Manipulations in Language VAEs](papers/vechtomova2020unsupervised.md) — 2020; ?.
- [PCAE: A Framework of Plug-in Conditional Auto-Encoder for Controllable Text Generation](papers/tu2022pcae.md) — 2022; V.
- [Composable Text Controls in Latent Space with ODEs](papers/liu2022composable.md) — 2023; R.
- [Controllable Stylistic Text Generation with Train-Time Attribute-Regularized Diffusion](papers/zhou2025regdiff.md) — 2025; V.
- [Counterfactuals to Control Latent Disentangled Text Representations for Style Transfer](papers/nangi2021counterfactual-text-style.md) — 2021; V.
- [Multi-Aspect Controllable Text Generation with Disentangled Counterfactual Augmentation](papers/liu2024magic.md) — 2024; V.

### Causal and source-guided text counterfactuals

- [CausaLM: Causal Model Explanation Through Counterfactual Language Models](papers/feder2021causalm.md) — 2021; V.
- [A Causal Lens for Controllable Text Generation](papers/hu2021causal-lens-text.md) — 2021; V.
- [Polyjuice: Generating Counterfactuals for Explaining, Evaluating, and Improving Models](papers/wu2021polyjuice.md) — 2021; V.
- [Counterfactual Generation with Identifiability Guarantees (MATTE)](papers/yan2023matte.md) — 2023; V.
- [Beyond What If: Advancing Counterfactual Text Generation with Structural Causal Modeling](papers/wang2024beyond-what-if.md) — 2024; V.
- [COFFEE: Counterfactual Fairness for Personalized Text Generation in Explainable Recommendation](papers/wang2023coffee.md) — 2023; V.

### Feasibility and actionable counterfactuals

- [FACE: Feasible and Actionable Counterfactual Explanations](papers/poyiadzi2020face.md) — 2020; V.

Related clusters: [05. Causal abstraction and neurosymbolic alignment](#05-causal-abstraction); [07. Causal generative models and state-conditioned flows](#07-causal-flows); [08. Optimal transport and geometric counterfactual maps](#08-optimal-transport); [11. Evaluation validity and experimental protocol](#11-evaluation).

<a id="07-causal-flows"></a>

## 07. Causal generative models and state-conditioned flows

Factual conditional-density models, abduction and shared-noise transport. Flow-based text transformations are kept alongside cross-domain precedents, with their differing supervision explicit.

Comparison question: What makes the reused noise correspond to the same unit across states, beyond fitting each factual conditional distribution?

### Normalizing-flow foundations

- [Density estimation using Real NVP](papers/dinh2016realnvp.md) — 2016; V.
- [Normalizing Flows for Probabilistic Modeling and Inference](papers/papamakarios2021flows.md) — 2021; V.

### Structural causal, bijective and conditional-distribution models

- [Causal Effect Inference with Deep Latent-Variable Models](papers/louizos2017cevae.md) — 2017; R.
- [Deep Structural Causal Models for Tractable Counterfactual Inference](papers/pawlowski-dscm-2020.md) — 2020; R.
- [Causal Autoregressive Flows](papers/khemakhem2021carefl.md) — 2021; V.
- [VACA: Designing Variational Graph Autoencoders for Causal Queries](papers/sanchezmartin-vaca-2022.md) — 2022; R.
- [Counterfactual Identifiability of Bijective Causal Models](papers/nasr-esfahany2023bgm.md) — 2023; V.
- [High Fidelity Image Counterfactuals with Probabilistic Causal Models](papers/sousaribeiro-fidelty-2023.md) — 2023; R.
- [Counterfactually-Equivalent Structural Causal Modelling Using Causal Graphical Normalizing Flows](papers/balgi2024cgnf.md) — 2024; V.
- [Engression: extrapolation through the lens of distributional regression](papers/shen2025engression.md) — 2025; V.

### Flow-based text editing and cross-domain applications

- [Harmonization with Flow-based Causal Inference](papers/wang2021flow-harmonization.md) — 2021; V.
- [FairFlow: Model-based Counterfactual Data Augmentation For NLP](papers/tokpo2024fairflow.md) — 2024; V.
- [StyleFlow: Disentangle Latent Representations via Normalizing Flow](papers/zhu2024styleflow.md) — 2024; V.
- [Dimensionality reduction in 3D causal deep learning for neuroimage generation](papers/ohara2025dimensionality-neuroimages.md) — 2025; V.
- [StyleFlow: Attribute-conditioned Exploration of StyleGAN-Generated Images using Conditional Continuous Normalizing Flows](papers/abdal2021styleflow-images.md) — 2021 (2020 preprint); V.
- [LeapFactual: Reliable Visual Counterfactual Explanation Using Conditional Flow Matching](papers/cao2025leapfactual.md) — 2025; V.

Related clusters: [03. Frozen text encoders and language VAEs](#03-text-encoders); [06. Semantic editing and controllable text](#06-semantic-editing); [08. Optimal transport and geometric counterfactual maps](#08-optimal-transport); [10. Identification and causal representation learning](#10-identification).

<a id="08-optimal-transport"></a>

## 08. Optimal transport and geometric counterfactual maps

Geometric map constraints, distribution matching and transport-based counterfactual assumptions. Keep ot_editor, ot_flow and ot_gaussian distinct even though they share OT motivation.

Comparison question: Is OT being used to constrain a map, fit a coupling between distributions, or justify an individual counterfactual under additional assumptions?

### OT foundations, convex potentials and map regularization

- [Input Convex Neural Networks](papers/amos2017icnn.md) — 2017; V.
- [Computational Optimal Transport](papers/peyre2019computational-ot.md) — 2019; V.
- [Optimal transport mapping via input convex neural networks](papers/makkuva2020ot-icnn.md) — 2020; V.
- [The Monge Gap: A Regularizer to Learn All Transport Maps](papers/uscidda2023monge-gap.md) — 2023; V.
- [Flow Matching for Generative Modeling](papers/lipman2023flow-matching.md) — 2023 (2022 preprint); V.
- [Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport](papers/tong2024ot-cfm.md) — 2024 (2023 preprint); V.
- [Wasserstein geometry of Gaussian measures](papers/takatsu2011gaussian-wasserstein.md) — 2011; V.

### Counterfactual transports, dynamics and coherence

- [Counterfactual Cocycles](papers/dance2024counterfactual-cocycles.md) — 2024; V.
- [Transport-based Counterfactual Models](papers/delara2024transport-counterfactuals.md) — 2024; V.
- [Sequential Conditional Transport on Probabilistic Graphs for Interpretable Counterfactual Fairness](papers/fernandesmachado2024sequential-transport.md) — 2024; V.
- [Counterfactual Identifiability via Dynamic Optimal Transport](papers/desousaribeiro2025dynamic-ot.md) — 2025; V.
- [Debiased Counterfactual Generation via Flow Matching from Observations](papers/dance2026deconfounding-flows.md) — 2026 preprint; V.
- [Counterfactual Generation via Flow Matching: Coupling-Sensitive End-to-End Rates](papers/guan2026counterfactual-flow-rates.md) — 2026 preprint; V.

### Perturbation-response and representation-alignment applications

- [Learning single-cell perturbation responses using neural optimal transport (CellOT)](papers/bunne2023cellot.md) — 2023; V.
- [CounterPC: Counterfactual Feature Realignment for Unsupervised Domain Adaptation on Point Clouds](papers/yang2025counterpc.md) — 2025; V.

### Fairness-oriented transport and causal pre-processing

- [Obtaining Fairness using Optimal Transport Theory](papers/gordaliza-obtaining-2019.md) — 2019; R.
- [Matching code and law: achieving algorithmic fairness with optimal transport](papers/zehlike-matching-2020.md) — 2020; R.
- [fairadapt: Causal Reasoning for Fair Data Pre-processing](papers/plecko2021fairadapt.md) — 2021; V.

Related clusters: [02. Fair and invariant representations](#02-fair-representations); [06. Semantic editing and controllable text](#06-semantic-editing); [07. Causal generative models and state-conditioned flows](#07-causal-flows); [10. Identification and causal representation learning](#10-identification).

<a id="09-text-realization"></a>

## 09. Text realization, reconstruction and embedding inversion

The text decoder D_text: producing a CV from a latent vector. Recovering a factual source, decoding an oracle counterfactual vector and rendering an estimated edited vector are separate tests.

Comparison question: Which inference inputs and training pairs are available, and does fluent text faithfully express the information in the edited embedding?

### Embedding inversion, information leakage and reproducibility

- [Sentence Embedding Leaks More Information than You Expect: Generative Embedding Inversion Attack to Recover the Whole Sentence](papers/li2023embedding-inversion.md) — 2023; R.
- [Text Embeddings Reveal (Almost) As Much As Text](papers/morris2023vec2text.md) — 2023; V.
- [Rethinking the Privacy of Text Embeddings: A Reproducibility Study of "Text Embeddings Reveal (Almost) As Much As Text"](papers/seputis2025rethinking-embedding-privacy.md) — 2025; V.
- [Embedding Inversion via Conditional Masked Diffusion Language Models](papers/xiao2026embedding-inversion.md) — 2026; V.

### Generative backbones and lightweight conditioning

- [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](papers/lewis2020bart.md) — 2020; V.
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)](papers/raffel2020t5.md) — 2020; V.
- [LoRA: Low-Rank Adaptation of Large Language Models](papers/hu2021lora.md) — 2021; V.
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](papers/li2021prefix-tuning.md) — 2021; V.
- [Gemma 3 Technical Report](papers/gemma-team2025gemma3.md) — 2025; V.
- [Network-to-Network Translation with Conditional Invertible Neural Networks](papers/rombach2020net2net.md) — 2020; V.

### Training exposure and sequence-decoding behavior

- [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](papers/bengio2015scheduled-sampling.md) — 2015; V.
- [The Curious Case of Neural Text Degeneration](papers/holtzman2020neural-text-degeneration.md) — 2020; V.

Related clusters: [03. Frozen text encoders and language VAEs](#03-text-encoders); [04. Semantic decoding, concepts and probing](#04-semantic-decoding); [06. Semantic editing and controllable text](#06-semantic-editing); [11. Evaluation validity and experimental protocol](#11-evaluation).

<a id="10-identification"></a>

## 10. Identification and causal representation learning

Conditions under which latent factors or cross-world counterparts are identifiable. These are related but different identification problems; a result for a learned encoder does not automatically apply to a frozen embedding.

Comparison question: What observations, interventions, invertibility and noise assumptions are actually needed by each guarantee?

### Nonlinear ICA and auxiliary information

- [Nonlinear independent component analysis: Existence and uniqueness results](papers/hyvarinen1999nonlinear.md) — 1999; R.
- [Unsupervised Feature Extraction by Time-Contrastive Learning and Nonlinear ICA](papers/hyvarinen2016temporal.md) — 2016; V.
- [Nonlinear ICA of temporally dependent stationary sources](papers/hyvarinen2017stationary.md) — 2017; R.
- [Nonlinear ICA Using Auxiliary Variables](papers/hyvarinen2019auxiliary.md) — 2019; R.
- [Variational Autoencoders and Nonlinear ICA: A Unifying Framework](papers/khemakhem2020ivae.md) — 2020; R.
- [Nonlinear independent component analysis for principled disentanglement in unsupervised deep learning](papers/hyvarinen2023nonlinear.md) — 2023; R.

### Disentanglement, causal desiderata and observational limits

- [Towards a Definition of Disentangled Representations](papers/higgins2018disentangled.md) — 2018; V.
- [Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](papers/locatello2019challenging.md) — 2019; R.
- [Towards Causal Representation Learning](papers/scholkopf2021crl.md) — 2021; R.
- [Desiderata for Representation Learning: A Causal Perspective](papers/wang2021desiderata.md) — 2021; V.
- [CausalVAE: Disentangled Representation Learning via Neural Structural Causal Models](papers/yang2021causalvae.md) — 2021; R.
- [A Causal Ordering Prior for Unsupervised Representation Learning](papers/kori2023causal.md) — 2023; R.

### Weak supervision, interventions, temporal structure and augmentations

- [Weakly-Supervised Disentanglement Without Compromises](papers/locatello2020weak-supervision.md) — 2020; R.
- [Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style](papers/kugelgen2021content-style.md) — 2021; R.
- [Weakly Supervised Representation Learning with Sparse Perturbations](papers/ahuja2022sparse.md) — 2022; V.
- [Weakly Supervised Causal Representation Learning](papers/brehmer2022crl.md) — 2022; R.
- [CITRIS: Causal Identifiability from Temporal Intervened Sequences](papers/lippe2022citris.md) — 2022; V.
- [Interventional Causal Representation Learning](papers/ahuja2023interventional.md) — 2023; R.
- [BISCUIT: Causal Representation Learning from Binary Interactions](papers/lippe2023biscuit.md) — 2023; V.

### Counterfactual ambiguity after model fitting

- [Counterfactual (Non-)identifiability of Learned Structural Causal Models](papers/nasr-esfahany2023nonidentifiability.md) — 2023; V.
- [Neural Causal Models for Counterfactual Identification and Estimation](papers/xia2023neural-counterfactual-identification.md) — 2023 (2022 preprint); V.
- [Exogenous Isomorphism for Counterfactual Identifiability](papers/chen2025exogenous-isomorphism.md) — 2025; V.

Related clusters: [05. Causal abstraction and neurosymbolic alignment](#05-causal-abstraction); [07. Causal generative models and state-conditioned flows](#07-causal-flows); [08. Optimal transport and geometric counterfactual maps](#08-optimal-transport).

<a id="11-evaluation"></a>

## 11. Evaluation validity and experimental protocol

Cross-cutting checks on probes, editor objectives and benchmark selection. These works support evaluation design, not additional intervention variants.

Comparison question: Could a good score reflect decoder exploitation, concept leakage or selection bias rather than faithful counterfactual recovery?

### Probability calibration and distributional scores

- [On Calibration of Modern Neural Networks](papers/guo2017calibration.md) — 2017; V.
- [Strictly Proper Scoring Rules, Prediction, and Estimation](papers/gneiting2007proper-scoring.md) — 2007; V.

### Probe, concept and edit validity

- [Designing and Interpreting Probes with Control Tasks](papers/hewitt2019control-tasks.md) — 2019; R.
- [Adversarial Examples Are Not Bugs, They Are Features](papers/ilyas2019adversarial.md) — 2019; R.
- [Promises and Pitfalls of Black-Box Concept Learning Models](papers/mahinpei2021concept-leakage.md) — 2021; V.
- [Exploring Counterfactual Explanations Through the Lens of Adversarial Examples: A Theoretical and Empirical Analysis](papers/pawelczyk2022adversarial.md) — 2022; R.
- [RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations](papers/huang2024ravel.md) — 2024; V.

### Model selection and training-protocol context

- [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](papers/cawley2010model-selection.md) — 2010; V.
- [Revisiting Deep Learning Models for Tabular Data](papers/gorishniy2021tabular-dl.md) — 2021; V.

Related clusters: [01. Benchmarks, hiring and causal fairness](#01-benchmarks-and-fairness); [04. Semantic decoding, concepts and probing](#04-semantic-decoding); [09. Text realization, reconstruction and embedding inversion](#09-text-realization); [10. Identification and causal representation learning](#10-identification).

## Maintenance

Keep the primary assignment in this index; keep bibliographic identity, provenance and reading notes in the existing `papers/<id>.md` card. Card `topics` remain broad discovery tags and are not a second, synchronized classification. Do not rename or duplicate a card to place it in another field. When adding a record, choose one primary subgroup, add appropriate cluster cross-links, and update the counts above and in the README.

Ambiguous citations and model/dataset names without an identified paper remain in [unresolved.md](unresolved.md); they have not been discarded or converted into guessed records. The [review](reviews/2026-10-19-organization.md#unresolved-leads-and-reading-gaps) routes those leads to the relevant fields.
