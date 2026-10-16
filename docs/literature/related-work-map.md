# Provisional related-work map

This is a navigation aid for writing the paper, not a finished literature review or a novelty claim. It is based on the recovered project discussions and the current outline in [the manuscript](../../overleaf/sections/2_related.tex). Read the primary papers before turning a comparison question into prose.

## 1. Fairness of text representations

Start with [Counterfactual Fairness](papers/kusner2017counterfactual.md), [Path-Specific Counterfactual Fairness](papers/chiappa-path-specific-2019.md), [Causal Fairness Analysis](papers/plecko2024causal-fairness.md), [Bias in Bios](papers/de2019bias.md), and [the algorithmic-hiring survey](papers/fabris-fairness-2025.md).

Questions for the motivation: Why is hiring a meaningful test case? Which fairness notion do we target? Which causal pathways, decision rules and social assumptions are being fixed? A useful representation edit is not itself a fairness guarantee.

Contrast the project with [Learning Fair Representations](papers/zemel-learning-2013.md), [adversarially fair representations](papers/madras-learning-2018.md), [the Variational Fair Autoencoder](papers/louizos-variational-2016.md), [Flexibly Fair](papers/creager-flexibly-2019.md), [distributional contrastive disentanglement](papers/oh-learning-2022.md), [CLAIRE](papers/ma2023clairelearning.md), [Joo et al.](papers/joo2024constructing.md), [Ng et al.](papers/ng2025debiasing.md), and [fairadapt](papers/plecko2021fairadapt.md).

Comparison axes: retraining an encoder versus editing a frozen representation; invariance versus intervention; available causal structure; factual versus paired counterfactual supervision; and the actual fairness target. The index retains the broader fairness bibliography for later selection.

## 2. Editing latent representations

Read [RepE](papers/zou2023repe.md), [CAA](papers/panickssery2023caa.md), [ReFT/LoReFT](papers/wu2024reft.md), [LEACE](papers/belrose2023leace.md), [LatentOps](papers/liu2022composable.md), [PPLM](papers/dathathri2019pplm.md), [controlled generation with VAEs](papers/hu2017toward.md), and [RegDiff](papers/zhou2025regdiff.md).

For C-sym, ask what supplies the target edit: a behavior label, a concept direction, a counterfactual example, or a symbolic causal kernel? Check whether a method edits token activations, a sentence/document embedding, or a generative latent. Compare inference-time dependencies, preservation objectives and support diagnostics rather than just the word “intervention”.

[Controllable representations without supervision](papers/xu2020variational.md), [counterfactual explanations versus adversarial examples](papers/pawelczyk2022adversarial.md), and [adversarial features](papers/ilyas2019adversarial.md) motivate checking plausibility and independent semantics, not merely the same decoder used by the training objective.

## 3. Causal abstraction and neurosymbolic learning

Core leads: [LatPlan](papers/asai2022classical.md), [causal abstractions](papers/geiger2021abstractions.md), [IIT](papers/geiger2022iit.md), [DAS](papers/geiger2023das.md), [Boundless DAS](papers/wu2023boundless-das.md), [Causal Proxy Models](papers/wu2023cpm.md), and [neurosymbolic counterfactual fairness](papers/heilmann2026neurosymbolic.md).

Compare what is aligned: an internal computation, a learned proxy's behavior, or a latent edit constrained by an external symbolic intervention. Check requirements for source examples, paired counterfactuals, joint encoder training and symbolic access at deployment. Older notes sometimes call methods “identical” or claim no precedent; these claims have not been validated.

Read [Sutter et al.](papers/sutter2025dilemma.md) and [Makelov et al.](papers/makelov2023illusion.md) before treating a commuting diagram or a successful activation edit as sufficient evidence of a faithful causal explanation. This is a live discussion; assess responses as well, not just the criticism.

## 4. Deep causal inference and state flow

Start with [Deep Structural Causal Models](papers/pawlowski-dscm-2020.md), [high-fidelity image counterfactuals](papers/sousaribeiro-fidelty-2023.md), [VACA](papers/sanchezmartin-vaca-2022.md), [CAREFL](papers/khemakhem2021carefl.md), [cGNF](papers/balgi2024cgnf.md), and [bijective causal models](papers/nasr-esfahany2023bgm.md).

For state flow, compare the meaning of a preserved noise variable and the assumptions under which it represents the same unit across states. Separate a fitted conditional density from an identified cross-state coupling. [Counterfactual (Non-)identifiability](papers/nasr-esfahany2023nonidentifiability.md) is central to that distinction.

[The flow review](papers/papamakarios2021flows.md) and [Real NVP](papers/dinh2016realnvp.md) motivate the architecture, not its causal interpretation. [Engression](papers/shen2025engression.md) belongs with conditional-distribution learning and sample-based objectives; historical discussions also considered it for distillation.

## 5. Semantic decoders and evaluation

Read [CEBaB](papers/abraham2022cebab.md), [CPM](papers/wu2023cpm.md), [RAVEL](papers/huang2024ravel.md), [Text Bottleneck Models](papers/ludan2024text-bottleneck.md), [CB-LLM](papers/sun2025cbllm.md), [concept completeness](papers/bhan2025concept-completeness.md), [concept leakage](papers/mahinpei2021concept-leakage.md), and [probe control tasks](papers/hewitt2019control-tasks.md).

Questions: Can the decoder recover the intended concepts? Does it preserve joint dependencies and uncertainty? Can an editor exploit it? Do evaluation concepts measure both desired change and preservation? Probe accuracy, semantic consistency and counterfactual latent recovery answer different questions.

## 6. Identification and representation learning

Use [Locatello et al.](papers/locatello2019challenging.md), [the CRL survey](papers/scholkopf2021crl.md), [iVAE](papers/khemakhem2020ivae.md), [auxiliary-variable ICA](papers/hyvarinen2019auxiliary.md), [paired causal representation learning](papers/brehmer2022crl.md), [content/style isolation](papers/kugelgen2021content-style.md), and [sparse perturbations](papers/ahuja2022sparse.md).

Build an assumption checklist before citing a positive guarantee: what is observed, what is intervened on, which noise is shared, what varies across environments, what is invertible, and which equivalence class is identified? The present factual-only flow setup must not inherit claims from papers whose results need paired observations or stronger intervention coverage.

## 7. Encoders and data generation

Separate encoder engineering from the main intervention contribution. [LangVAE](papers/carvalho2025langvae.md), [Optimus](papers/li2020optimus.md), [Nomic](papers/nussbaum2024nomic.md), [Qwen3 Embedding](papers/qwen2025embedding.md), [EmbeddingGemma](papers/vera2025embeddinggemma.md), and [Matryoshka representations](papers/kusupati2022matryoshka.md) provide background. [Embedding inversion](papers/morris2023vec2text.md) does not by itself validate inversion of edited latents.

For benchmark motivation, compare [LIBERTy](papers/toker-liberty-2026.md), [CEBaB](papers/abraham2022cebab.md), [PopResume](papers/yu2026popresume.md), and [causal bias detection in generative AI](papers/plecko2026generative-bias.md). Record which provide semantic labels, a causal mechanism, paired counterfactual texts, population representativeness, and independent evaluation.

## Practical first reading pass

A manageable first pass is: LIBERTy; Kusner; Causal Fairness Analysis; the hiring survey; CPM; LatentOps; ReFT; DSCM; the BGM identification paper; counterfactual non-identifiability; and the closest fairness comparator among CLAIRE/Joo/Ng. Then fill gaps from the topic index.

For each, write down only: target problem; training supervision; intervention mechanism; inference inputs; guarantee and assumptions; evaluation; and the precise difference from our method. Do not claim “first”, “only”, unbiased recovery, or causal equivalence from this collection alone.
