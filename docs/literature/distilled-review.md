# Distilled related-work draft

Working version 1 — 2026-10-19. For discussion in this library only; **not inserted into the manuscript**.

Next iteration: [the manuscript-shaped draft](manuscript-draft.md) condenses this material into eight paragraphs, with [focused closest-work checks](reviews/2026-10-19-closest-work.md). This broader selection is preserved; MUtE is included in the shorter draft after that targeted check.

This draft selects **48 core works from 207 collected records**, covering **all 11 clusters in 13 field-focused paragraphs**. Semantic editing and OT each receive two paragraphs because they contain distinct comparisons central to the paper. The identification paragraph also draws on two works whose primary library home is OT. Three additional recent preprints remain a conditional reading list below. Nothing has been deleted from the [complete index](index.md).

Selection favors the closest task, supervision and inference setting; a necessary foundational reference; or a result that materially limits our claims. It does not favor recency or citation count alone. The 48 works are a working citation pool, not a commitment to cite all of them in the final related-work section.

The paragraphs are draft prose. The italicized selection notes and local record links are editorial scaffolding for our next iteration. Project comparisons follow the [method-to-literature map](related-work-map.md): a frozen encoder, semantic reader `g`, external symbolic operation `h_S`, and five latent editors/transports. The latent renderer is distinct from the text decoder.

## 01. Causal fairness and counterfactual text benchmarks

Causal fairness asks which changes in a decision should follow from changing a protected attribute, rather than treating every statistical association as equivalent. [Counterfactual Fairness](https://proceedings.neurips.cc/paper_files/paper/2017/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf) and [Causal Fairness Analysis](https://arxiv.org/abs/2207.11385) provide complementary foundations for specifying such comparisons and distinguishing causal pathways. In text, [Bias in Bios](https://arxiv.org/abs/1901.09451) demonstrates that occupational gender disparities can persist after explicit gender indicators are removed. Counterfactual benchmarks then make different kinds of intervention evidence available: [CEBaB](https://arxiv.org/abs/2205.14140) uses human-edited concept changes in reviews, whereas [LIBERTy](https://aclanthology.org/2026.findings-acl.399/) propagates interventions through an explicit structural model before generating text. LIBERTy therefore supplies the structural benchmark for our comparison of latent interventions. Its simulator defines the reference counterfactuals; it does not by itself establish real-world causal validity or perfect faithfulness of generated CVs.

*Selection rationale:* Keep the causal definition, pathway perspective, hiring-related text motivation, and two contrasting benchmark designs. Other hiring datasets and fairness definitions remain available if the introduction needs a more application-specific argument.

Records: [Kusner et al.](papers/kusner2017counterfactual.md); [Plečko and Bareinboim](papers/plecko2024causal-fairness.md); [De-Arteaga et al.](papers/de2019bias.md); [Abraham et al.](papers/abraham2022cebab.md); [Toker et al.](papers/toker-liberty-2026.md).

## 02. Fair and transferable representations

Fair representation learning seeks useful representations while controlling information or dependence associated with protected attributes. [Zemel et al.](https://proceedings.mlr.press/v28/zemel13.html) establish this utility–fairness formulation, and [Madras et al.](https://proceedings.mlr.press/v80/madras18a.html) study adversarially learned representations intended to transfer to downstream tasks. Importantly, [Creager et al.](https://proceedings.mlr.press/v97/creager19a.html) already allow learned representations to be modified at test time for different fairness requirements. Our distinction is therefore not simply that an intervention can be applied after training. The intended comparison concerns explicit, state-dependent edits of a frozen text representation under a separately specified symbolic operation, including changes that should propagate to other semantic attributes. Suppressing protected information, satisfying a group-level criterion, and constructing an individual's structural counterfactual are related but different objectives.

*Selection rationale:* These three works cover the foundational objective, downstream transfer, and flexible test-time use. They prevent a misleading contrast with an allegedly uniform, fixed-policy fair-representation literature. Specialized variants can be restored if they become experimental comparators.

Records: [Zemel et al.](papers/zemel-learning-2013.md); [Madras et al.](papers/madras-learning-2018.md); [Creager et al.](papers/creager-flexibly-2019.md).

## 03. Frozen text representations and generative latent spaces

The representation being edited is itself a substantive modeling choice. [LangVAE and LangSpace](https://aclanthology.org/2025.emnlp-demos.57/) provide a modular framework for language VAEs and analysis of their latent spaces. By contrast, [Nomic Embed](https://arxiv.org/abs/2402.01613), [Qwen3 Embedding](https://arxiv.org/abs/2506.05176), and [EmbeddingGemma](https://arxiv.org/abs/2509.20354) represent pretrained text-embedding approaches whose objectives and intended uses differ from VAE reconstruction. Their inclusion motivates testing whether latent intervention depends on a particular encoder family rather than assuming a universal editable geometry. Neither a generative prior nor strong embedding-task performance establishes that semantic interventions are recoverable, that long CVs are represented faithfully, or that edited vectors can be decoded. Those properties require evidence for the exact checkpoint, input-length policy, and adaptation used in our experiments.

*Selection rationale:* Cite the four encoder families actually in scope. Generic BERT, GPT-2, sentence-embedding and VAE references belong in implementation details where needed, rather than expanding this paragraph into a model history. This paragraph may ultimately move to experimental setup.

Records: [LangVAE](papers/carvalho2025langvae.md); [Nomic](papers/nussbaum2024nomic.md); [Qwen3](papers/qwen2025embedding.md); [EmbeddingGemma](papers/vera2025embeddinggemma.md).

## 04. Concept-based semantic interfaces

Concept-based models connect otherwise opaque representations to variables that humans can inspect or modify. [Concept Bottleneck Models](https://proceedings.mlr.press/v119/koh20a.html) make concept predictions an explicit intermediate representation, while [Post-hoc Concept Bottleneck Models](https://arxiv.org/abs/2205.15480) adapt this idea to pretrained models. [Stochastic Concept Bottleneck Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5c7894ac8788555f1cecf536f1e0fd35-Abstract-Conference.html) further emphasize uncertainty and dependence among concepts. These works motivate a semantic reader that exposes a structured state distribution rather than only hard labels. Our reader does not replace the downstream predictor with a concept-only bottleneck: its output specifies a target for a separate latent renderer. Consequently, predicting concepts, representing their joint uncertainty, and realizing a requested change in the original embedding are separate tasks. Correcting a concept prediction also need not implement a structural intervention with downstream semantic consequences.

*Selection rationale:* Keep the original interface, its post-hoc extension, and a dependency-aware stochastic formulation. These motivate the independent/autoregressive reader comparison without presenting either implementation as a reproduction of SCBM.

Records: [Koh et al.](papers/koh2020concept-bottleneck.md); [Yuksekgonul et al.](papers/yuksekgonul2023posthoc-cbm.md); [Vandenhirtz et al.](papers/vandenhirtz2024stochastic-cbm.md).

## 05. Causal abstraction and symbolic–neural alignment

Causal abstraction concerns whether interventions at one descriptive level correspond to interventions at another. [Rubenstein et al.](https://proceedings.mlr.press/r15/rubenstein-17a.html) formalize consistency between structural models at different levels, while [Distributed Alignment Search](https://arxiv.org/abs/2303.02536) learns alignments between interpretable causal variables and distributed neural representations. [Causal Proxy Models](https://proceedings.mlr.press/v202/wu23b.html) instead learn interpretable proxies of model behavior using factual and approximate counterfactual supervision. These perspectives motivate read–operate–render: read a semantic state, apply an external symbolic operation, and realize the result in latent space. They also delimit the claim. Agreement through a learned reader is weaker than establishing a faithful causal abstraction, and architectural separation alone does not demonstrate composition or generalization to unseen operations. Unlike CPM's training setup, the five planned latent methods do not use paired counterfactual embeddings as training targets.

*Selection rationale:* Keep one formal foundation and two close neural/text connections. Broader neurosymbolic planning work is useful historical context but less direct than these comparisons. The final supervision comparison must distinguish approximate counterfactual text from true paired latent targets.

Records: [Rubenstein et al.](papers/rubenstein2017causal-consistency.md); [DAS](papers/geiger2023das.md); [CPM](papers/wu2023cpm.md).

## 06a. Post-hoc concept erasure and representation editing

Post-hoc representation editing already spans several notions of semantic control. [LEACE](https://arxiv.org/abs/2306.03819) removes linearly accessible concept information with a minimal-distortion criterion, while [Representation Surgery / MiMiC](https://arxiv.org/abs/2402.09631) develops constrained affine steering maps. Closer to semantic-target optimization, [Laguna et al.](https://arxiv.org/html/2401.13544v3) make black-box representations intervenable using concept predictions and a representation-distance penalty. [Invertible Interpretation Networks](https://compvis.github.io/iin/) and [post-hoc generative concept bottlenecks](https://openaccess.thecvf.com/content/CVPR2025/html/Kulkarni_Interpretable_Generative_Models_through_Post-hoc_Concept_Bottlenecks_CVPR_2025_paper.html) provide additional precedents for interpretable editing interfaces around pretrained representations or generators. Thus, neither frozen-model editing nor semantic consistency plus proximity is a sufficient novelty claim. The relevant comparison for `semantic_editor` and `ot_editor` is their externally specified structured target distribution, map class, training access, and counterfactual evaluation—not merely their ability to change a decoded concept.

*Selection rationale:* Keep linear erasure, affine steering, the closest semantic-loss precedent, and two modular generative interfaces. These are complementary comparisons, not five interchangeable baselines. Erasure guarantees do not establish causal replacement; the vision-based interfaces are architectural neighbors rather than directly equivalent CV methods.

Records: [LEACE](papers/belrose2023leace.md); [MiMiC](papers/singh2024representation-surgery.md); [Beyond CBMs](papers/laguna2024beyond-cbm.md); [IIN](papers/esser2020iin.md); [Kulkarni et al.](papers/kulkarni2025posthoc-generative-cbm.md).

## 06b. Latent text control and counterfactual representations

Text-specific work provides an equally important comparison to vision-based latent editing. [LatentOps](https://aclanthology.org/2023.emnlp-main.1030/) composes text controls through classifier-guided latent ODEs. [MATTE](https://proceedings.neurips.cc/paper_files/paper/2023/hash/afda6bf3fb086eabbaf161ba1cec5a9a-Abstract-Conference.html) studies counterfactual text generation with dependent content and style factors, deriving identifiability under a relative-sparsity condition. [Lemberger and Saillenfest](https://arxiv.org/abs/2402.00711) directly investigate counterfactual text representations for explaining classifiers. This literature rules out a broad claim that prior text editing is merely noncausal or always requires paired counterfactual training data. Our intended distinction is the comparison of several renderers over fixed document embeddings, with a separately supplied multivariate symbolic operation and structural benchmark targets. Controllable generation, representation-based explanation, and recovery of an individual benchmark counterfactual still impose different requirements, even when all operate in a latent space.

*Selection rationale:* Retain composable control, assumption-dependent identification, and direct classifier-facing counterfactual representations. Source-guided rewriting tools remain in reserve unless text-level editing becomes a main baseline.

Records: [LatentOps](papers/liu2022composable.md); [MATTE](papers/yan2023matte.md); [Lemberger and Saillenfest](papers/lemberger2024counterfactual-text-representations.md).

## 07. State-conditioned flows and shared-noise editing

Generative counterfactual models provide a natural template for inferring latent noise from a factual observation and reusing it after an intervention. [Deep Structural Causal Models](https://arxiv.org/abs/2006.06485) develop tractable abduction using expressive generative components. [StyleFlow](https://rameenabdal.github.io/StyleFlow/) offers a close editing precedent: invert an image latent under its original attributes and transform it under changed attributes with a conditional flow. In NLP, [FairFlow](https://arxiv.org/html/2407.16431v1) uses invertible attribute manipulation of word representations to help construct parallel texts and train a counterfactual text generator. Our `state_flow` instead fits factual document embeddings conditional on semantic state and transports them by retaining abducted reference coordinates. This resembles shared-noise editing, but a conditional density model is not automatically an identified structural model: factual likelihood does not uniquely determine which individuals correspond across states.

*Selection rationale:* Keep the causal-inference template, the closest inverse–forward attribute-editing precedent, and a text/fairness flow method. FairFlow is not a whole-document conditional-likelihood equivalent. Real NVP and general flow surveys remain implementation citations.

Records: [DSCM](papers/pawlowski-dscm-2020.md); [StyleFlow (images)](papers/abdal2021styleflow-images.md); [FairFlow](papers/tokpo2024fairflow.md).

## 08a. Optimal transport as a correspondence assumption

Optimal transport selects correspondences using both endpoint distributions and a cost. [Makkuva et al.](https://proceedings.mlr.press/v119/makkuva20a.html) learn transport maps through input-convex neural potentials, and [CellOT](https://www.nature.com/articles/s41592-023-01969-x) applies neural OT to unpaired control and perturbed cell populations. [De Lara et al.](https://www.jmlr.org/papers/v25/21-1440.html) develop transport-based counterfactual models and examine conditions relating their couplings to causal counterparts. Complementarily, [fairadapt](https://www.jstatsoft.org/article/view/v110i04) uses causal-graph information for fair data transformation, including allowances for nondiscriminatory pathways. These works connect geometric correspondence and causal preprocessing without equating the two. In particular, `ot_editor` adopts a convex-gradient map class while optimizing a semantic target; it does not fit that target state's empirical density. Its distinction from `semantic_editor` is therefore map geometry, whereas the distribution-based transports introduce an additional correspondence assumption. A low-cost transformation alone does not establish a causal intervention.

*Selection rationale:* Keep the practical convex-map construction, a major unpaired intervention application, the direct counterfactual formulation, and the causal-fairness preprocessing connection. CellOT has observed perturbation populations, not the same observational-state supervision as our setting; fairadapt acts on causally specified variables rather than directly solving our latent-rendering problem.

Records: [Makkuva et al.](papers/makkuva2020ot-icnn.md); [CellOT](papers/bunne2023cellot.md); [De Lara et al.](papers/delara2024transport-counterfactuals.md); [fairadapt](papers/plecko2021fairadapt.md).

## 08b. Flow matching and Gaussian transport

Distribution-based transport can be learned dynamically or approximated analytically. [Flow Matching](https://arxiv.org/abs/2210.02747) trains continuous generative dynamics through vector-field regression, and [Tong et al.](https://arxiv.org/abs/2302.00482) incorporate minibatch OT couplings into conditional flow matching. These provide the methodological basis for `ot_flow`, which learns state-conditioned paths from a shared reference distribution and reuses reference coordinates across states. [Gaussian Wasserstein geometry](https://ir.library.osaka-u.ac.jp/repo/ouka/all/4973/) supplies the analytic counterpart motivating `ot_gaussian`: location/scale transport under a restricted Gaussian model. Our version uses diagonal state covariances in a shared PCA basis, not unrestricted covariance transport. This comparison separates modeling capacity from the chosen geometry. Neither minibatch OT training nor composing two reference-to-state maps automatically establishes globally optimal cross-state transport or the true individual counterfactual coupling.

*Selection rationale:* Keep the two training ingredients and one analytic foundation. General OT textbooks, numerical solvers and alternative dynamic formulations can be cited in methods where their specific constructions are used.

Records: [Lipman et al.](papers/lipman2023flow-matching.md); [Tong et al.](papers/tong2024ot-cfm.md); [Takatsu](papers/takatsu2011gaussian-wasserstein.md).

## 09. Embedding inversion and text realization

Rendering an edited embedding as readable text is a separate problem from recovering its semantic state. [Vec2Text](https://aclanthology.org/2023.emnlp-main.765/) shows that substantial textual information can be recovered from embeddings using learned inversion and iterative correction. A [reproducibility study by Seputis et al.](https://arxiv.org/abs/2507.07700) examines sensitivity to sequence length and domain, cautioning against treating reconstruction results as universal. [Xiao's conditional masked-diffusion approach](https://arxiv.org/abs/2602.11047), a 2026 preprint, explores inversion without encoder queries at inference, with experiments on short sequences. These works motivate evaluating text realization rather than assuming it follows from embedding quality. For our comparison, factual reconstruction, reconstruction from an oracle counterfactual embedding, and decoding a predicted edit must remain separate. Success on in-distribution short texts does not establish faithful long-CV reconstruction or successful decoding of off-support latent edits.

*Selection rationale:* Keep the central inversion method, evidence about its limits, and an alternative inference-access regime. Prefix-tuning, LoRA and language-model backbone references belong with the actual decoder implementation, not as evidence that inversion works.

Records: [Vec2Text](papers/morris2023vec2text.md); [Seputis et al.](papers/seputis2025rethinking-embedding-privacy.md); [Xiao](papers/xiao2026embedding-inversion.md).

## 10. Identification and coherence of counterfactual maps

A learned representation or transport needs assumptions beyond predictive fit to justify a causal interpretation. [Locatello et al.](https://arxiv.org/abs/1811.12359) establish limits of unsupervised disentanglement without inductive biases; the distinct problem of learned counterfactual mechanisms is addressed by [Nasr-Esfahany and Kiciman](https://arxiv.org/abs/2301.09031), who show non-identifiability in general expressive settings, including higher-dimensional noise. Positive results must be read with their restrictions: [Exogenous Isomorphism](https://proceedings.mlr.press/v267/chen25p.html) studies conditions for counterfactual identifiability, while [dynamic-OT identification](https://arxiv.org/abs/2510.08294) develops guarantees for suitably constrained continuous flows. [Counterfactual Cocycles](https://arxiv.org/abs/2405.13844) further emphasizes coherence across successive transports, not just pairwise distribution fit. Together these works frame shared noise, minimal displacement and Gaussian structure as substantive modeling assumptions in our methods. Benchmark recovery can compare their usefulness; it cannot establish identification outside the benchmark's assumptions.

*Selection rationale:* Keep the distinction between representation identification and counterfactual coupling, two relevant positive directions, and composition/coherence. Dynamic OT and cocycles are cross-links from cluster 08, not additional library entries. Do not transfer their guarantees before checking the hypotheses against our exact models.

Records: [Locatello et al.](papers/locatello2019challenging.md); [Nasr-Esfahany and Kiciman](papers/nasr-esfahany2023nonidentifiability.md); [Chen and Du](papers/chen2025exogenous-isomorphism.md); [De Sousa Ribeiro et al.](papers/desousaribeiro2025dynamic-ot.md); [Dance and Bloem-Reddy](papers/dance2024counterfactual-cocycles.md).

## 11. Evaluating semantic interventions and predictive distributions

Semantic success should be assessed with more than the same reader used to train an editor. [Mahinpei et al.](https://arxiv.org/abs/2106.13314) show that learned concept representations can carry information beyond their assigned meanings, while [RAVEL](https://aclanthology.org/2024.acl-long.470/) evaluates whether representation interventions change a targeted attribute while preserving others. Probability quality is another axis: [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html) distinguish neural predictive confidence from calibration, and [Gneiting and Raftery](https://doi.org/10.1198/016214506000001437) provide the proper-scoring framework underlying distributional measures such as the energy score. These perspectives motivate separate checks of semantic change, preservation of attributes that the symbolic intervention should leave unchanged, calibration, paired latent recovery, sample support and text fidelity. Scores from the training reader remain useful consistency measures, but they are not independent evidence that an edited embedding represents the intended counterfactual.

*Selection rationale:* Keep one concept-validity warning, one targeted-intervention benchmark, and foundations for probability and distributional evaluation. Calibration is descriptive unless a correction is actually fitted; it does not certify causal correctness. Independent semantic evaluation is a recommendation, not a claim that it is already implemented.

Records: [Mahinpei et al.](papers/mahinpei2021concept-leakage.md); [RAVEL](papers/huang2024ravel.md); [Guo et al.](papers/guo2017calibration.md); [Gneiting and Raftery](papers/gneiting2007proper-scoring.md).

## Recent work to check before fixing novelty or guarantees

These three works are outside the 48-paper core, not dismissed as irrelevant. Their roles need a closer assumption-level comparison before expanding the draft.

- **[MUtE](https://arxiv.org/abs/2609.11253)** — September 2026 preprint connecting concept erasure and counterfactual interventions. Especially relevant to whether our editing contribution differs from existing preservation/translation constructions. Check alongside 06a before making novelty claims. [Record](papers/saillenfest2026mute.md).
- **[Debiased Counterfactual Generation via Flow Matching from Observations](https://arxiv.org/abs/2605.07665)** — 2026 preprint on population counterfactual distributions under causal adjustment assumptions. Add to 08b/10 if the narrative develops observational deconfounding; do not conflate its target with individual paired recovery. [Record](papers/dance2026deconfounding-flows.md).
- **[Counterfactual Generation via Flow Matching: Coupling-Sensitive End-to-End Rates](https://arxiv.org/abs/2610.01193)** — October 2026 preprint relevant to coupling-sensitive distributional guarantees. Its smoothing, corrected stochastic sampler and statistical assumptions need comparison with our implementation; it is not a ready-made theorem for our deterministic transports. [Record](papers/guan2026counterfactual-flow-rates.md).

## How to condense this further

For the eventual manuscript, 01–02 can motivate the problem; 04–08 contain the closest methodological comparisons; and 10 supplies the qualifications that should accompany causal claims. Paragraphs 03, 09 and 11 may be shortened into experimental setup or evaluation rather than becoming separate related-work subsections. This is a field-organized working draft, not a proposal for thirteen paper subsections.

Next iteration should settle three editorial choices:

1. Which precise fairness operation the paper foregrounds, so the motivation does not promise a stronger fairness claim than the experiments support.
2. How much text reconstruction belongs in the main contribution versus a separate evaluation of the latent outputs.
3. Which close comparators need detailed contrasts or empirical baselines—especially semantic-loss editing, affine steering, shared-noise editing and transport-based counterfactuals.

The other **156 records** remain in the full library outside this 48-paper core and three-paper watchlist. They include alternative foundations, specialized variants, implementation ingredients and neighboring tasks. Nonselection is an editorial decision for this narrative, not a judgment of quality.

## Evidence and provenance

This is a user-requested distillation of the existing collection, informed by the [organization review](reviews/2026-10-19-organization.md), [targeted search](imports/2026-10-19-web-search.md), current [project-method map](related-work-map.md), and additional primary-source checks on 2026-10-19. Linked abstracts and selected method passages support the bounded descriptions above; this is **not a full-paper or proof review of all selected works**. Comparisons with our project and selection rationales are editorial assessments.

The canonical cards and their metadata/reading statuses are unchanged. Primary links in this draft also resolve details not yet reconciled in older cards: LatentOps has an EMNLP 2023 proceedings page; LIBERTy has a Findings of ACL 2026 page; fairadapt has a 2024 Journal of Statistical Software version; and the Seputis preprint now reports acceptance in the RecSys 2025 reproducibility track. Stable filenames can retain a preprint year. Check final venue metadata and theorem assumptions before exporting a manuscript bibliography. No quantitative superiority, exhaustive-coverage or first-method claim is made here.

