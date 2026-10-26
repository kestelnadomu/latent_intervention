# Background and Related Work: literature overview

One row per paper: what it contributes and how it relates to this project.

**Scope.** The contribution is a *workflow for controlling text representations through a
user-defined symbolic operator*: a semantic decoder $g$ reads a structured state from the latent
of a frozen encoder, any operator $h_S$ on states (an SCM counterfactual, a rule, an erasure, a
stochastic kernel) transforms it, and a renderer $\rho$, learned once from factual data, writes
the target state back into the latent. Counterfactual fairness with an SCM-derived $h_S$ is the
running application, not the scope. See [contribution.md](contribution.md).

**Key column:** keys without a mark exist in `references.bib`. Keys marked † are proposed and still
need an entry. The † entries were collected from memory: verify titles, venues, and the stated
contributions before citing.

**Relevance (Rel.)**, rows sorted by it within each table:

- ●●● **core**: closest work; cite and position against explicitly.
- ●●○ **supporting**: cite for a specific claim, method, or contrast.
- ●○○ **context**: cite in a group (e.g. "adversarial approaches [a, b, c]") or optional.

## Core reading (all ●●●, by importance)

| # | Paper | § | Why |
|---|---|---|---|
| 1 | Geiger et al. 2021, causal abstraction | 2 | Aligns a symbolic model with a representation; their symbolic model is what the network computes, ours is any operator we choose |
| 2 | Wu et al. 2024, ReFT | 1 | Learned, source-free edits of frozen representations; candidate parameterisation of $h_Z$ |
| 3 | Geiger et al. 2023, DAS | 2 | Learned alignment of representation subspaces with symbolic variables |
| 4 | Sutter et al. 2025 | 2 | Non-linear alignment can be trivial; constrains what a successful lift means |
| 5 | Zou et al. 2023, RepE | 1 | "Representation control" as a research programme; concepts read and steered top-down |
| 6 | Rimsky et al. 2024, CAA | 1 | Additive steering; our constant-direction baseline |
| 7 | Belrose et al. 2023, LEACE | 1 | Provable representation control (erasure); erasure is one possible $h_S$ |
| 8 | Pawlowski et al. 2020, DSCM | 3 | The state-conditioned flow (F1) is its recipe; observed parents vs. inferred $\mathbf s$ |
| 9 | Abdal et al. 2021, StyleFlow | 1 | Conditional flow over a frozen latent space; nearest architectural relative of F1 |

## 1. Controlling representations

Editing or steering the representation of a frozen model without retraining it: the same
operation as $h_Z$. Recurring contrasts with our workflow:

- **Specification:** the target is a direction or behaviour defined by examples (contrast pairs,
  probes, task labels). Ours is an explicit operator $h_S$ on a symbolic state space.
- **Edit form:** mostly additive and linear, one concept at a time. $h_Z$ is nonlinear and
  changes several state variables jointly, including dependent ones.
- **Verification:** output behaviour or a probe, versus $g$-consistency plus recovery of known
  target latents where available.

### Activation steering and representation engineering

| Paper | Rel. | Key | Contribution | Relation to us |
|---|---|---|---|---|
| Zou et al. 2023, RepE | ●●● | `zou_repe_2023`† | Representation Engineering: read and control high-level concepts from population-level representations | Shares the goal of representation control; concepts defined by stimuli, not by a symbolic model |
| Rimsky et al. 2024, CAA | ●●● | `rimsky_caa_2024`† | Contrastive Activation Addition: mean difference over many contrast pairs | Additive baseline; analogous to a constant $z' = z + v_\delta$ |
| Turner et al. 2023, ActAdd | ●●○ | `turner_actadd_2023`† | Steering vector from the activation difference of a contrastive prompt pair, added at inference | Additive, single-concept, no learning |
| Li et al. 2023, ITI | ●●○ | `li_iti_2023`† | Inference-Time Intervention: shift attention-head activations along linear-probe directions | Probe-defined direction, like steering by $g$ |
| Park et al. 2024 | ●●○ | `park_lrh_2024`† | Formalises the linear representation hypothesis; causal inner product | Theory behind additive steering; our operators need not be linear |

### Learned interventions

| Paper | Rel. | Key | Contribution | Relation to us |
|---|---|---|---|---|
| Wu et al. 2024, ReFT / LoReFT | ●●● | `wu_reft_2024`† | Low-rank, source-free learned edits of a frozen model's hidden representations | Candidate parameterisation for $h_Z$; trained on task labels, not on a symbolic operator |
| Wu et al. 2022, Causal Proxy Models | ●●○ | `wu_cpm_2022`† | Proxy models trained with counterfactual supervision to mimic and explain a black box | Counterfactual text supervision from a causal model; needs counterfactual inputs |

### Concept erasure

| Paper | Rel. | Key | Contribution | Relation to us |
|---|---|---|---|---|
| Belrose et al. 2023, LEACE | ●●● | `belrose_leace_2023`† | Closed-form, provably minimal linear concept erasure | Erasure is one $h_S$ (constant or marginalised state); strongest linear baseline |
| Ravfogel et al. 2020, INLP | ●●○ | `ravfogel_inlp_2020`† | Iterative nullspace projection removes linearly decodable attribute information | Linear, single attribute |
| Ravfogel et al. 2021, AlterRep | ●●○ | `ravfogel_alterrep_2021`† | Counterfactual interventions on representations to test a concept's causal role | Edits toward a counterfactual value; one concept, linear |
| Elazar & Goldberg 2018 | ●●○ | `elazar_adversarial_2018`† | Adversarial removal of demographic attributes from text representations; attributes remain recoverable | Erasure leaks; verification must not rely on the training probe |
| Gonen & Goldberg 2019, Lipstick on a Pig | ●○○ | `gonen_lipstick_2019`† | Word-embedding debiasing hides bias rather than removing it | Same argument for static embeddings |
| Bolukbasi et al. 2016 | ●○○ | `bolukbasi_man_2016`† | Hard debiasing of word embeddings by projecting out a gender direction | Origin of representation-level text debiasing |

### Controlling text generation through latents

| Paper | Rel. | Key | Contribution | Relation to us |
|---|---|---|---|---|
| Liu et al. 2023, LatentOps | ●●○ | `liu2022composable` | Composable text attribute control in a text VAE latent space via classifier-guided ODEs | Composable operators in a text latent space; attributes independent, VAE-specific |
| Dathathri et al. 2020, PPLM | ●●○ | `dathathri_pplm_2020`† | Gradients of a small attribute model shift an LM's hidden activations during decoding | Classifier-guided control, like optimising against $g$ |
| Subramani et al. 2022 | ●●○ | `subramani_steering_2022`† | Steering vectors that make a frozen LM generate a target sentence; vector arithmetic for style transfer | Editable sentence-level vectors of a frozen LM |
| Xu et al. 2020 | ●○○ | `xu2020variational` | Controllable text representations in VAEs; constrained posteriors for attribute manipulation (ICML) | Attribute edits in a text-VAE latent |
| Vechtomova et al. 2020 | ●○○ | `vechtomova2020unsupervised` | Unsupervised discovery of interpretable latent manipulations in language VAEs | Latent directions; bib author field says *Anonymous*, fix it |
| Hu et al. 2017 | ●○○ | `hu2017toward` | Controlled text generation with VAE + attribute discriminators | Early attribute control |
| Vaeth et al. 2025, RegDiff | ●○○ | `vaeth2025regdiff` | Controllable stylistic text generation with train-time attribute regularisation (CoRR) | Attribute control at training time |

### Latent editing of frozen generators

| Paper | Rel. | Key | Contribution | Relation to us |
|---|---|---|---|---|
| Abdal et al. 2021, StyleFlow | ●●● | `abdal_styleflow_2021`† | Conditional CNF over frozen StyleGAN latents given classifier-predicted attributes; edit = invert under source, regenerate under target | Nearest architectural relative of F1; attributes independent, no operator on them |
| Wołczyk et al. 2022, PluGeN | ●●○ | `wolczyk_plugen_2022`† | Flow on a pretrained model's latent space separating attribute coordinates | Plug-in edits without retraining the generator |
| Shen et al. 2020, InterFaceGAN | ●●○ | `shen_interfacegan_2020`† | Linear classifiers in GAN latent space; normal vectors as edit directions | Latent-space classifier, like $g$ |

## 2. Aligning representations with symbolic models

Reading symbolic states from representations, and executing symbolic operators on states or
latents.

### Causal abstraction

| Paper | Rel. | Key | Contribution | Relation to us |
|---|---|---|---|---|
| Geiger et al. 2021 | ●●● | `geiger_abstraction_2021`† | Causal abstraction: interchange interventions test whether a network implements a high-level causal model | Their high-level model is the network's own computation, so downstream layers propagate edits. Ours is an external operator, executed symbolically and rendered back |
| Geiger et al. 2023, DAS | ●●● | `geiger_das_2023`† | Distributed alignment search: a learned rotation finds the subspace aligned with a causal variable | Needs a source input at intervention time; point-wise, not distributional |
| Geiger et al. 2022, IIT | ●●○ | `geiger_iit_2022`† | Interchange intervention training induces a causal structure during training | Trains the network itself; ours is post hoc on frozen $f$ |
| Wu et al. 2023, Boundless DAS | ●○○ | `wu_boundless_2023`† | Scales DAS to LLMs with learned subspace boundaries | Same as DAS |

### Grounding: probes and concept models

| Paper | Rel. | Key | Contribution | Relation to us |
|---|---|---|---|---|
| Belinkov 2022 | ●●○ | `belinkov_probing_2022`† | Survey of probing classifiers: what probes show and what they do not | $g$ is a probe; a probe can read information the model does not use |
| Hewitt & Liang 2019 | ●●○ | `hewitt_control_2019`† | Control tasks: probe accuracy partly reflects probe capacity | Caveat for interpreting $g$-accuracy |
| Koh et al. 2020, Concept Bottleneck Models | ●●○ | `koh_cbm_2020`† | Predict through a layer of human-defined concepts; interventions on concepts at test time | Symbolic state between input and output, editable; but trained end-to-end, edits on $\mathbf s$ not on $z$ |
| Kim et al. 2018, TCAV | ●○○ | `kim_tcav_2018`† | Concept activation vectors: concept directions from examples | Concept directions defined by examples |

### Structure-preserving latent operators

| Paper | Rel. | Key | Contribution | Relation to us |
|---|---|---|---|---|
| van der Pol et al. 2020, MDP homomorphisms | ●●○ | `vanderpol_mdp_2020`† | Lift abstract transitions into a learned latent space so the square commutes | Encoder trained jointly; ours fits the operator into an existing representation |
| Higgins et al. 2018 | ●●○ | `higgins_disentangled_2018`† | Disentanglement defined through group actions on the representation | Assumes invertible group actions; $do(\cdot)$ and erasures are not invertible |
| Veličković & Blundell 2021, Neural algorithmic reasoning | ●○○ | `velickovic_nar_2021`† | Neural processors aligned with classical algorithms in latent space | Algorithm executed in latent space; trained jointly with encoders |

### Identifiability of representations

| Paper | Rel. | Key | Contribution | Relation to us |
|---|---|---|---|---|
| Locatello et al. 2019 | ●●○ | `locatello2019challenging` | Unsupervised disentanglement is impossible without inductive biases or supervision (ICML) | Why $g$ is supervised by symbolic variables instead of reading disentangled latents |
| Hyvärinen et al. 2023 | ●○○ | `hyvarinen2023nonlinear` | Nonlinear ICA as principled disentanglement (Patterns) | Identifiability background |
| Kori et al. 2023 | ●○○ | `kori2023causal` | Causal ordering prior for unsupervised representation learning (CoRR) | Causal structure as inductive bias in the encoder; ours keeps $f$ frozen |

### Evaluation and pitfalls of alignment

| Paper | Rel. | Key | Contribution | Relation to us |
|---|---|---|---|---|
| Sutter et al. 2025 | ●●● | `sutter_nonlinear_2025`† | Non-linear representation dilemma: unconstrained nonlinear alignment makes causal abstraction trivial | Motivates restricting $h_Z$ (proximity, support, low rank) |
| Makelov et al. 2023 | ●●○ | `makelov_illusion_2023`† | Interpretability illusions: subspace patching can succeed by activating dormant pathways | Same risk for $g$-consistency; motivates checks against known target latents |
| Huang et al. 2024, RAVEL | ●●○ | `huang_ravel_2024`† | Benchmark for disentangling entity attributes: *Cause* and *Isolate* scores | Adopt Cause/Isolate as edit metrics |
| Abraham et al. 2022, CEBaB | ●○○ | `abraham_cebab_2022`† | Benchmark of concept-based explanations with human-written counterfactual texts | Counterfactual text pairs for evaluation |

### Neurosymbolic approaches

| Paper | Rel. | Key | Contribution | Relation to us |
|---|---|---|---|---|
| Asai & Fukunaga 2018; Asai et al. 2022, LatPlan | ●●○ | `asai2018classical`, `asai2022classical` | Neural encoder to a discrete latent space where a symbolic planner reasons | Origin of our modular split: neural encoding + symbolic operator; their encoder is trained for it |
| Adriaensen et al. 2026, ProbLog4Fairness | ●●○ | `adriaensen_problog4fairness_2026` | Neurosymbolic modelling and mitigation of bias with probabilistic logic programming (AAAI) | Symbolic bias model; acts on predictions, not on representations |
| Heilmann et al. 2026 | ●●○ | `heilmann2026neurosymbolic` | A neurosymbolic approach to counterfactual fairness (Neurosymbolic AI journal) | Own prior work with a different approach; cite as related, not as predecessor |
| Wagner & d'Avila Garcez 2021 | ●○○ | `wagner_neural_2021` | Neural-symbolic integration for fairness in AI (AAAI Spring Symposium) | Neurosymbolic fairness |
| Asai 2019 | ●○○ | `asai2019unsupervised` | Unsupervised grounding of first-order logic representations from images (ICAPS) | LatPlan extension: symbolic grounding of latents |
| Takata et al. 2023 | ●○○ | `takata2023plausibility` | Plausibility-based heuristics for latent-space classical planning (ICAPS workshop) | LatPlan follow-up |

## 3. Counterfactual inference: one instance of $h_S$

Background for the running application, where $h_S$ is an SCM counterfactual, and for the
state-conditioned flow (F1).

### Normalizing flows

| Paper | Rel. | Key | Contribution | Relation to us |
|---|---|---|---|---|
| Dinh et al. 2017, RealNVP | ●●○ | `dinh_realnvp_2017`† | Affine coupling layers: exact likelihood and cheap inverse | Our flow core (affine coupling) |
| Winkler et al. 2019 | ●●○ | `winkler_cnf_2019`† | Conditional normalizing flows $p(x \mid c)$ | F1 is a conditional flow over $z$ given $\mathbf s$ |
| Rezende & Mohamed 2015 | ●○○ | `rezende_flows_2015`† | Normalizing flows as flexible variational posteriors | Origin of flows |

### Deep structural causal models

| Paper | Rel. | Key | Contribution | Relation to us |
|---|---|---|---|---|
| Pawlowski et al. 2020, DSCM | ●●● | `pawlowski_dscm_2020` | Mechanisms as conditional flows / amortised VAEs; abduction by inversion, prediction with shared $u$ | F1 is DSCM's recipe applied to $z$; DSCM's parents are observed labels |
| Javaloy et al. 2023, Causal NFs | ●●○ | `javaloy_causalflows_2023`† | Theory and practice of flows as SCMs; when they identify interventions and counterfactuals | Identifiability frame for F1 |
| De Sousa Ribeiro et al. 2023 | ●●○ | `sousaribeiro_fidelty_2023` | HVAE mechanisms for high-fidelity image counterfactuals; counterfactual training with predictors | Learned $\mathbf s$-predictors in training, like $g$ (images, observed parents) |
| Sánchez-Martín et al. 2022, VACA | ●●○ | `sanchezmartin_vaca_2022` | Graph VAEs for interventional and counterfactual queries; fairness application | Counterfactual fairness with deep SCMs, tabular |
| Khemakhem et al. 2021, CAREFL | ●○○ | `khemakhem_carefl_2021`† | Autoregressive flows as SCMs: causal discovery, interventions, counterfactuals | Flow-based SCM over observed variables |
| Sanchez & Tsaftaris 2022, Diff-SCM | ●○○ | `sanchez_diffscm_2022`† | Diffusion counterfactuals guided by anti-causal classifiers | Classifier in the generation loop |

### Identifiability and evaluation of counterfactuals

| Paper | Rel. | Key | Contribution | Relation to us |
|---|---|---|---|---|
| Nasr-Esfahany et al. 2023 | ●●○ | `nasresfahany_identifiability_2023`† | Conditions under which bijective causal models identify counterfactuals | Shared-$u$ in F1 is an assumption, not identified by factual likelihood |
| Monteiro et al. 2023 | ●●○ | `monteiro_soundness_2023`† | Axioms for counterfactual models: composition, effectiveness, reversibility; pseudo-oracle predictors | Composition = identity at $\mathbf s'=\mathbf s$; warns against reusing the training predictor for evaluation (our $g$) |
| Nasr-Esfahany & Kıcıman 2023 | ●○○ | `nasresfahany_nonidentifiability_2023`† | Learned SCMs can fit observational data and still disagree on counterfactuals | Same gap; motivates testing on paired $Z'$ |

### Transport-based counterfactuals

| Paper | Rel. | Key | Contribution | Relation to us |
|---|---|---|---|---|
| De Lara et al. 2024 | ●●○ | `delara_transport_2024`† | Counterfactuals as transport maps; when they coincide with causal counterfactuals | F1 = flow-induced transport $p(z\mid\mathbf s)\to p(z\mid\mathbf s')$ |
| Plečko & Meinshausen 2020, fairadapt | ●●○ | `plecko_fairadapt_2020`† | Counterfactual descendants of $A$ by quantile preservation (quantile regression forests) | 1-D analogue of F1's shared-noise transport |
| Black et al. 2020, FlipTest | ●●○ | `black_fliptest_2020`† | Audits classifiers by OT-mapping individuals across groups | Structurally our downstream benchmark |
| Gordaliza et al. 2019 | ●○○ | `gordaliza_obtaining_2019` | Fairness by optimal-transport repair of the feature distributions (ICML) | Group-level transport, non-causal |
| Zehlike et al. 2020 | ●○○ | `zehlike_matching_2020` | Matching code and law: fairness with optimal transport (DMKD) | Transport-based repair |

## 4. Application: fairness of text representations

### Fairness in text

| Paper | Rel. | Key | Contribution | Relation to us |
|---|---|---|---|---|
| De-Arteaga et al. 2019, Bias in Bios | ●●○ | `de2019bias` / `arteaga_biasbios_2019` | Occupation classification from biographies; gender stays predictive after removing explicit indicators | Real-world target dataset |
| Toker et al. 2026, LIBERTy | ●●○ | `toker_liberty_2026` | Causal benchmark of LLM-generated texts from SCMs with counterfactual texts | Recipe for our data generation |
| Veitch et al. 2021 | ●●○ | `veitch2021counterfactual` | Counterfactual invariance of text classifiers to spurious correlations; regularisers depend on the causal structure | Training-level; no representation edits |
| Zhou et al. 2024 | ●○○ | `zhou2024explore` | Spurious correlations in language models at the concept level | Sensitive attributes act through concepts, not tokens |

(Representation-level text debiasing, i.e. erasure, is in §1.)

### Counterfactual and causal fairness

| Paper | Rel. | Key | Contribution | Relation to us |
|---|---|---|---|---|
| Kusner et al. 2017 | ●●○ | `kusner2017counterfactual` | Counterfactual fairness: prediction unchanged under $do(A=a')$ via abduction–action–prediction | Criterion of the running application |
| Plečko & Bareinboim 2024, Causal Fairness Analysis | ●●○ | `plecko_causalfairness_2024`† | Standard fairness model (SFM); decomposes disparity into direct, indirect, spurious effects | Our experiment's SCM is an SFM; path-specific states are other $h_S$ instances |
| Chiappa 2019 | ●●○ | `chiappa_path-specific_2019` | Path-specific counterfactual fairness; only unfair pathways are corrected | Path-specific $h_S$ |
| Zuo et al. 2023 | ●●○ | `zuo_counterfactually_2023`† | Counterfactually fair representations: a representation-based criterion and construction (NeurIPS) | Fair representation defined counterfactually; verify contribution first |
| Kilbertus et al. 2017 | ●○○ | `kilbertus_avoiding_2017`† | Causal criteria for discrimination (proxy, unresolved discrimination) | Causal view of what must change under $do(A)$ |
| Kim et al. 2021, DCEVAE | ●○○ | `kim_dcevae_2021`† | VAE separating exogenous noise into parts affected and unaffected by $A$ | Generative counterfactuals, tabular |
| Xu et al. 2019, CFGAN | ●○○ | `xu_cfgan_2019`† | GAN generator structured by a causal graph for causally fair data | Generative counterfactuals, tabular |
| Grari et al. 2023 | ●○○ | `grari2023counterfactual` | Counterfactual fairness for predictions using generative adversarial networks | Generative counterfactuals with an adversary |
| Ma et al. 2023, CLAIRE | ●○○ | `ma2023clairelearning` | Counterfactual fairness from observational data without a known causal model | No explicit SCM |

### Fair representation learning

All of these train the encoder for one fixed target, usually invariance to $A$. In our workflow,
invariance is one choice of $h_S$ applied to a frozen encoder.

| Paper | Rel. | Key | Contribution | Relation to us |
|---|---|---|---|---|
| Zemel et al. 2013, LFR | ●●○ | `zemel_learning_2013` | First fair representation learning: prototypes that hide group membership while preserving utility | Origin of the line |
| Creager et al. 2019, FFVAE | ●●○ | `creager_flexibly_2019` | Disentangled VAE latents per sensitive attribute; fairness criterion chosen after training | Modular: criterion chosen after encoding, like a swappable $h_S$ |
| Song et al. 2019 | ●●○ | `song_learning_2019` | Controllable fair representations: user-specified trade-offs via mutual-information bounds (AISTATS) | Control at use time, within invariance |
| Balunović et al. 2022, Fair Normalizing Flows | ●●○ | `balunovic_fnf_2022`† | One flow per sensitive group into a shared latent; exact densities bound any adversary | Flows for fair representations; group-level invariance, no counterfactuals |
| Zhao & Gordon 2019 | ●●○ | `zhao_tradeoffs_2019`† | Inherent trade-off between fairness and utility in learned representations (NeurIPS) | Cost of invariance; editing keeps the rest of the representation |
| Madras et al. 2018, LAFTR | ●●○ | `madras_learning_2018` | Adversarial objectives tied to group fairness criteria; transferable representations | Representative adversarial method |
| Locatello et al. 2019, fairness of disentangled representations | ●○○ | `locatello_fairness_2019`† | Disentanglement scores correlate with fairness of downstream predictors when $A$ is unobserved | Differs from `locatello2019challenging` |
| Oh et al. 2022 | ●○○ | `oh_learning_2022` | Fair representations via distributional contrastive disentanglement | Modular design |
| Louizos et al. 2016, VFAE | ●○○ | `louizos_variational_2016` | VAE with $A$ as a separate latent and an MMD penalty | VAE-based invariance |
| Ng et al. 2025 | ●○○ | `ng2025debiasing` | VAE text representations with an independence constraint (reward models) | Text setting; encoder retrained for invariance |
| Li et al. 2025 | ●○○ | `li2025eliminating` | Information-theoretic debiasing of reward models (CoRR) | Text setting; companion to Ng et al. 2025 |
| Ramaswamy et al. 2021 | ●○○ | `ramaswamy2021fair` | Fair attribute classification by de-biasing GAN latent space (latent perturbations for augmentation) | Latent-space edits for fairness, attribute in isolation |
| Ruoss et al. 2020 | ●○○ | `ruoss_learning_2020` | Certified individually fair representations | Individual fairness by similarity |
| Edwards & Storkey 2016; Xie et al. 2017; Feng et al. 2019; Jaiswal et al. 2020; Kim et al. 2020; Kairouz et al. 2022 | ●○○ | `edwards_censoring_2016`, `xie_controllable_2017`, `feng_learning_2019`, `jaiswal_invariant_2020`, `kim_fair_2020`, `kairouz_generating_2022` | Adversarial invariance | Group citation |
| Moyer et al. 2018; Grari et al. 2021; Zhu et al. 2021; McNamara et al. 2017 | ●○○ | `moyer_invariant_2018`, `grari_learning_2021`, `zhu_learning_2021`, `mcnamara_provably_2017` | Non-adversarial (information-theoretic, Rényi) invariance and guarantees | Group citation |
| Zhao et al. 2020; Shui et al. 2022 | ●○○ | `zhao_conditional_2020`, `shui_fair_2022` | Criterion-specific invariance (equalised odds; implicit path alignment) | Group citation |
| Liu et al. 2023; Wu et al. 2022; Rateike et al. 2022 | ●○○ | `liu_fair_2023`, `wu_semi_2022`, `rateike_dont_2022` | VAE-based fair representations (factor VAE transfer, semi-supervised, unlabelled data) | Group citation |
| Calmon et al. 2017, 2018; Hacker & Wiedemann 2017 | ●○○ | `calmon_optimized_2017`, `calmon_data_2018`, `hacker_continuous_2017` | Pre-processing / data transformation for discrimination prevention | Group citation |
| Lahoti et al. 2019 (iFair, PFR) | ●○○ | `lahoti_ifair_2019`, `lahoti_operationalizing_2019` | Individually fair representations | Group citation |
| Samadi et al. 2018; Kamani et al. 2022; Pérez-Suay et al. 2017; Tan et al. 2020 | ●○○ | `samadi_price_2018`, `kamani_efficient_2022`, `perez_fair_2017`, `li_learning_2020` | Fair PCA and kernel methods | Group citation; `li_learning_2020` has first author Tan |
| Hong et al. 2024 | ●○○ | `joo2024constructing` | Fair latent space for the intersection of fairness and explainability (CoRR) | Bib key says *joo* but first author is Hong |
| Qi et al. 2022, FairVFL | ●○○ | `qi_fairvfl_2022` | Fair vertical federated learning | Off-topic; probably drop |

### Steering for fairness (to fill)

Placeholder: needs a literature search before rows go in. Directions to cover: bias or gender
steering vectors in LLMs, concept erasure used for debiasing, and steering evaluated with
counterfactual fairness criteria.

| Paper | Rel. | Key | Contribution | Relation to us |
|---|---|---|---|---|
| | | | | |

## 5. Text encoders ($f$)

The workflow takes any frozen text encoder. Decoding $z'$ back to text needs a generative
encoder (VAE); control of the representation itself does not.

| Paper | Rel. | Key | Contribution | Relation to us |
|---|---|---|---|---|
| Reimers & Gurevych 2019, Sentence-BERT | ●●○ | `reimers_sbert_2019`† | Siamese BERT for semantically meaningful sentence embeddings | Origin of general-purpose sentence encoders |
| Kusupati et al. 2022, Matryoshka | ●●○ | `kusupati_matryoshka_2022`† | Nested embeddings whose prefixes are valid lower-dimensional embeddings | Our dimension sweeps (128–768) use Matryoshka prefixes |
| Nussbaum et al. 2024, Nomic Embed | ●●○ | `nussbaum_nomic_2024`† | Open, long-context text embedding model | Encoder backend (`nomic-embed-text-v1.5`) |
| Zhang et al. 2025, Qwen3 Embedding | ●●○ | `zhang_qwen3emb_2025`† | LLM-based text embedding models | Encoder backend (`Qwen3-Embedding-0.6B`) |
| Vera et al. 2025, EmbeddingGemma | ●●○ | `vera_embeddinggemma_2025`† | Small open text embedding model | Encoder backend (`embeddinggemma-300m`) |
| Carvalho et al. 2025, LangVAE | ●●○ | `carvalho2025langvae` | Pretrained text VAEs with structured latent spaces, plus probing (EMNLP) | VAE backend; allows decoding $z'$, but small context window |
| Li et al. 2020, Optimus | ●○○ | `li2020optimus` | Large pretrained text VAE (BERT encoder, GPT-2 decoder) (EMNLP) | Architecture LangVAE builds on |
| Bowman et al. 2016 | ●○○ | `bowman2016generating` | Sentence VAE: generating sentences from a continuous space (CoNLL) | Origin of text VAEs |
| Zhang et al. 2023, LlaMaVAE | ●○○ | `zhang2023llamavae` | Guiding LLM generation through a continuous VAE latent space (CoRR) | Text-VAE latent controlling an LLM |
| Zhang et al. 2024 (EACL Findings) | ●○○ | `zhang2024improving` | Semantic control in discrete latent spaces of transformer VAEs | Text-VAE latent structure |
| Zhang et al. 2024 (NAACL Findings) | ●○○ | `zhang2024graph` | Graph-induced syntactic-semantic spaces in transformer VAEs | Text-VAE latent structure |

## Gaps

1. **Specification.** Representation control is specified by examples (contrast pairs, probes,
   task labels). There is no way to specify the intended change as an operator on explicit
   semantics.
2. **Structured, dependent edits.** Edits target one concept at a time. Operators that change
   several variables jointly, with dependencies between them (mediators under an intervention,
   rule-based updates), are not covered.
3. **External operators on frozen representations.** Causal abstraction aligns a representation
   with the network's *own* computation, and structure-preserving approaches train the encoder
   jointly. Fitting an externally defined operator into a frozen representation is open.
4. **Inferred semantics.** Deep SCMs and concept bottlenecks assume labelled variables. In text
   they must be inferred, and some (talent $\texttt T$) are never stated.
5. **Verification.** Probe-based success can be illusory (erasure leaks, interpretability
   illusions, trivial nonlinear alignment). Checks against known target representations are
   needed.

**How we address them:** $h_S$ is an explicit operator on a symbolic state space (1, 2).
$g$ and the renderer $\rho$ are fitted post hoc to a frozen encoder from factual data only, and
$h_S$ enters only at inference, so operators can be exchanged without retraining (3). $g$ infers
the state, including hidden variables (4). Simulated paired data gives target latents $f(x')$ to
verify fidelity and preservation beyond $g$ (5). Counterfactual fairness with an SCM-derived
$h_S$ is the running instance; the state-conditioned flow (F1) is the main renderer.

## Working notes

- **Scope:** [contribution.md](contribution.md) now states the general claim (read → operate →
  render, any $h_S$); the SCM counterfactual is the running instance.
- **Several $h_S$ instances needed as evidence** for the general claim (also noted in
  contribution.md): e.g. the SCM counterfactual, an erasure, a deterministic rule.
- **Evaluation consequence.** With observed $\mathbf S$, F1 is essentially DSCM and has
  privileged information. The decoder-inferred-$\mathbf s$ regime carries gap 4.
- **Architectures carried forward:** F1 (state-conditioned flow) and Plan C (as C-sym). Both
  are state-indexed renderers (read → operate → render); see "Consistency by construction" in
  [architecture/latent_intervention.md](architecture/latent_intervention.md). Ablations: oracle
  regression (reference); baseline residual transformer (exposes adversarial edits against $g$);
  F2 (cost of amortising F1); A, B, D, F3 (appendix / negative result).
- **Inference needs $g$ and $h_S$** for F1 and C-sym. Under the general claim this is the
  control interface, not a drawback; F2-style distillation is optional amortisation for one
  fixed $h_S$.
- **Open: C-sym is not implemented.** Only C-distilled exists, and its $w_\theta$ is fitted to one
  $h_S$, so it cannot swap $h_S$. Needed: weights taken from $h_S\circ g$ directly, and realiser
  pretraining on states beyond each unit's true $\mathbf s'$ (sampled from $h_S\circ g$ or
  uniformly from $\mathcal S$).
- **Open: F1's support warning is too strict.** F1 records the configured do() as its training
  support (`src/latent_intervention/dispatch.py`, `train_latent_intervention_model`) and warns on
  any other intervention, although its training never sees $\delta$. The relevant check is
  whether the *target states* appeared in training. Matters as soon as $h_S$ is swapped.
- **Open: F1 abducts under $\arg\max g$.** Sample $\hat{\mathbf s}\sim g(\cdot\mid z)$ per draw
  instead, so $g$'s uncertainty (mostly about $\texttt T$) reaches $h_S$.
- **Open: semantic metric against real counterfactuals.** `_semantic_sample_metrics`
  (`src/latent_intervention/flow/workflow.py`) compares only against $h_S\circ g$. Add the KL
  against $g(\cdot\mid f(x'))$ on paired data, which is what an honest $h_Z$ can attain.
- **Done: [problem_formulation.md](archive/problem_formulation.md) is archived.** Its notation
  table described the archived CV-screening state ($\lvert\mathcal S\rvert = 3456$) and called
  $h_Z(z'\mid z,\delta)$ "the method". Current notation: [contribution.md](contribution.md).
- **Possible method extension:** SCM invariances (under $do(\texttt X)$, $\texttt T$ and proxies
  fixed) as a cross-state alignment constraint on $u$, evaluated on paired $Z'$.
- **Elsewhere:** engression (Shen & Meinshausen 2023) for F2's energy-distance training.
- **Bib hygiene:** `joo2024constructing` (first author Hong) and `li_learning_2020` (first author
  Tan) have misleading keys; `vechtomova2020unsupervised` has author *Anonymous*;
  `de2019bias` and `arteaga_biasbios_2019` are duplicates (FAccT vs. CoRR).
