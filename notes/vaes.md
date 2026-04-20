# Variational Autoencoders for Text

## Workflow

- Four-phase pipeline aimed at **counterfactual fairness**
- Essentially a LatPlan-flavored architecture applied to a downstream prediction task (predicting job counts from text about jobs, it seems):

  - **Phase 0**: Pretrained VAE (encoder + decoder on text)
  - **Phase 1**: Train a *semantic decoder* from the frozen latent → tabular metadata (analogous to LatPlan's "state decoder" that maps the latent to propositional facts)
  - **Phase 2**: Train a *neural manipulator* that mimics the effect of a *symbolic manipulator* acting on metadata. Loss = discrepancy between (semantic_decoder(manipulated_z)) and (symbolic_manipulator(metadata)). This is the LatPlan-inspired core.
  - **Phase 3**: Predictor operates on manipulated latents
  - **Phase 4**: Counterfactual fairness via consistency between real and manipulated predictions

- LatPlan analogy:
  - Their State AutoEncoder learns a discrete latent, and action models learn to transition between latent states in a way consistent with symbolic PDDL operators. 
  - Here: same thing, but for text, with continuous latents, and with fairness/debiasing as the downstream objective rather than planning.

---

## Framework Recommendation

For the **text encoder/VAE part (Phase 0)**, my strong recommendation is:

### **Optimus** — the best fit for your use case

Reasons it fits your specific needs:

1. **It is the text VAE with the most "LatPlan-like" properties.** It produces a single, fixed-dimensional continuous latent vector per input — unlike token-level models. This is exactly what you need for Phase 2, where the manipulator has to act on a bounded, well-defined representation.

2. **The latent space is structured enough for arithmetic/manipulation.** Optimus was explicitly demonstrated to support latent-space interpolation, sentence transfer, and attribute manipulation (e.g., tense, sentiment). This is the *critical* property for your manipulator to be learnable — if moving in latent space doesn't correspond to meaningful semantic change, Phase 2 won't work. This is where plain token-level encoders (including raw BERT/Sentence-BERT) fail.

3. **Pretrained checkpoints exist** (BERT encoder + GPT-2 decoder), so Phase 0 can genuinely be "pretrained" as your diagram suggests. You can then fine-tune on your domain corpus.

4. **Your semantic decoder (Phase 1) becomes straightforward** — a small MLP from z → tabular metadata works well because z is a compact fixed vector.

### Honest caveats about Optimus

- The original codebase is from 2020 and is somewhat dated (older HuggingFace APIs). You may need to port it to modern `transformers`, or find a community reimplementation.
- Posterior collapse is still a risk; budget some time for KL annealing / free-bits tuning.
- If your texts are long (>512 tokens), you'll need chunking or a different backbone.

---

## Alternatives Ranked for Your Use Case

| Option | Fit | Why |
|---|---|---|
| **Optimus** | ★★★★★ | Fixed-dim continuous latent + pretrained + demonstrated manipulability |
| **DELLA / LD4LG (latent diffusion for text)** | ★★★★ | Good latent structure, but adds complexity; the diffusion in latent space may actually *help* your manipulator, since training a diffusion-like manipulator has precedent |
| **Sentence-T5 or Sentence-BERT + learned decoder** | ★★★ | Pragmatic: use a frozen sentence encoder, train only a decoder to make it VAE-like. Less principled but much simpler. The latent isn't KL-regularized though, which may hurt manipulability |
| **β-VAE on transformer backbone** | ★★★ | Disentanglement helps the manipulator, but harder to train |
| **VQ-VAE for text** | ★★★ | Discrete codes are *very* LatPlan-like (LatPlan uses binary latents), but the manipulator becomes a discrete operator — different problem formulation. Worth considering if you want to stay closer to LatPlan's spirit |
| **Token-level BERT / plain RNN-VAE (Bowman)** | ★★ | Token-level latents are too high-dimensional and unstructured for your manipulator; Bowman is too weak for real text |

---

## A Suggestion You Might Consider

Given how close your architecture is to LatPlan, a **VQ-VAE-based text encoder** is worth serious consideration as a parallel track. LatPlan's whole trick is that *binary* latents let the symbolic and neural worlds meet cleanly — the manipulator becomes a genuinely symbolic operator over discrete bits, not a regression in continuous space. If interpretability and the neurosymbolic bridge matter more than generation quality, VQ-VAE (or even a **Binary AutoEncoder** like LatPlan's own SAE, but adapted to text) aligns more tightly with the original LatPlan thesis.

Optimus gives you better text modeling; a binary/VQ encoder gives you a tighter neurosymbolic story. Which matters more depends on whether Phase 2's "symbolic manipulator" is actually symbolic (PDDL-style rules) or just "tabular metadata edits" (which is what your diagram literally shows, and which Optimus handles fine).

---

Would you like me to sketch out a concrete implementation plan for Phase 0 → Phase 2 with Optimus, or go deeper on the VQ-VAE / binary-latent alternative?



Let me search for recent work in this specific direction — neurosymbolic manipulation of latent representations with auxiliary metadata decoders, especially for fairness.Good, I have solid grounding on LatPlan follow-ups. Now let me search for the counterfactual fairness angle, which is equally central to your workflow.Excellent finds. Let me do one more search for the specific "steering" / latent editing direction, which is the closest technical cousin to your manipulator.I have enough grounded material to give you a comprehensive, well-sourced answer. Let me organize this around the four distinct research threads that your workflow touches.


## 1. The LatPlan Lineage (closest conceptual ancestor)

These are the papers most directly in the spirit of your architecture — learning a latent space and a *manipulator* (action model) that acts on it in a way consistent with symbolic semantics.

- **Asai & Fukunaga (2018/2022)** — *Classical Planning in Deep Latent Space* The original LatPlan paper; proposes the State AutoEncoder (SAE) which finds a propositional state representation using a VAE, and the Action Autoencoder/Discriminator which jointly finds action symbols and implicit action models. Your "Neurosymbolic Manipulator" is essentially an AMA analogue.

- **Asai et al. (2022)** — *Classical Planning in Deep Latent Space* (JAIR version, the one you cited). Contains the extended discussion of symbol stability and improvements to make the AMA more robust. Importantly, this paper explicitly notes: "While we present an image-based implementation (data = raw images), the architecture itself does not make such assumptions and could be applied to other types of data such as audio/text" — a direct endorsement of your plan to port it to text.

- **Takata & Fukunaga (2023)** — *Plausibility-Based Heuristics for Latent Space Classical Planning*. Addresses a critical weakness relevant to you: latent plans produced by LatPlan can be "invalid with respect to the underlying, ground-truth domain" — i.e., the latent manipulation may not correspond to a semantically valid change. This is exactly the failure mode your "semantic decoder loss" tries to guard against.

- **Asai (2019)** — *Unsupervised Grounding of Plannable First-Order Logic Representation from Images*. Extension to first-order logic — relevant if your metadata has relational structure (e.g., dependencies between attributes).

---

## 2. Counterfactual Fairness via Latent Manipulation (closest to your Phase 4)

This is where your architecture's *purpose* lives, and there's a very active literature here.

- **Kusner et al. (2017)** — *Counterfactual Fairness* (the foundational paper). Introduces the framework: a decision is fair if it coincides with the one that would have been taken in a counterfactual world where the sensitive attribute were different. Your Phase 4 is a direct operationalization of this.

- **Ma et al. (2023)** — *CLAIRE: Learning for Counterfactual Fairness from Observational Data*. Generates counterfactuals for each individual with different sensitive attribute values, learning fair representations by minimizing the difference between predictions made on original data and on its counterfactuals. It maps observed variables to a latent representation space and uses a counterfactual fairness constraint on the embeddings. This is **structurally nearly identical to your Phase 4**.

- **Grari et al. (2023) / GCFN** — *Counterfactual Fairness for Predictions using GANs*. Critiques VAE-based approaches: "the learned representation can be potentially correlated with the sensitive attributes, which thus leads to bias, and VAEs have weak capability in constructing latent representations". Important caveat worth knowing about before you commit.

- **Chiappa (2019)** — *Path-Specific Counterfactual Fairness*. Generalizes abduction-action-prediction for counterfactual reasoning, allowing some paths from the sensitive attribute to remain active. Useful if your manipulator should only neutralize *some* causal pathways (your diagram note "Manipulator might remove a dependency" hints at this).

- **Joo et al. (2024)** — *Constructing Fair Latent Space for Intersection of Fairness and Explainability*. Proposes a module that constructs a fair latent space by disentangling and redistributing labels and sensitive attributes, attaching to a pretrained generative model to transform its biased latent space into a fair one. Counterfactual explanations are generated by adjusting values along label/sensitive-attribute dimensions. **This is remarkably close to your workflow** — frozen pretrained encoder, lightweight module on top, counterfactual manipulation in latent space.

- **Ramaswamy et al. (2021)** — *Fair Attribute Classification through Latent Space De-biasing*. Proposes perturbing vectors in GAN latent space to de-correlate target and protected attributes. GAN-based analogue.

---

## 3. Latent Manipulation in Text VAEs (the technical foundation for your Phase 0–2)

These papers address the specific challenges of making latent manipulation *work* for text — which, as I noted earlier, is much harder than for images.

- **Hu et al. (2017)** — *Toward Controlled Generation of Text*. Combines VAEs and attribute discriminators, where the VAE trains the generator to reconstruct sentences while discriminators enforce attribute-coherent output. Learns disentangled representations with "designated semantic structure" for generating sentences with dynamically specified attributes. Your semantic decoder plays a similar disentangling role, although with regression rather than classification.

- **Xu et al. (2020)** — *On Variational Learning of Controllable Representations for Text without Supervision*. A critical paper you should read early: "sequence VAEs trained on text fail to properly decode when the latent codes are manipulated, because the modified codes often land in holes or vacant regions in the aggregated posterior latent space". Proposes constraining the posterior mean to a learned probability simplex to enable manipulation. **This is a foundational problem your manipulator will hit.**

- **Li et al. (2020)** — *Optimus: Organizing Sentences via Pre-trained Modeling of a Latent Space*. The one I recommended earlier. "The FIRST VAE with big pre-trained models (BERT and GPT-2)"; downstream tasks include controlled text generation.

- **Liu et al. (2022)** — *Composable Text Control Operations in Latent Space with ODEs (LatentOps)*. Uses the latent space of an adapted GPT-2 for composable text manipulation: "The low-dimensionality and differentiability of the text latent vector allow us to develop an efficient sampler based on ODEs given arbitrary plug-in operators". **A very strong candidate framework** for your manipulator — their "plug-in operators" are conceptually identical to your neurosymbolic manipulator.

- **Vaeth et al. (2025) — RegDiff** — *Controllable Stylistic Text Generation with Train-Time Attribute-Regularized Diffusion*. Integrates diffusion into the latent space of a VAE for interpretable and controllable style manipulation. Recent (2025) and relevant if you want latent-diffusion-based manipulation.

- **Anonymous (ICLR submission)** — *Unsupervised Discovery of Interpretable Latent Manipulations in Language VAEs*. Makes the key observation relevant to your Phase 2: "State-of-the-art NLG models like BART/T5/GPT-3 do not have latent spaces, therefore there is no natural way to perform controlled generation. In contrast, less popular models with explicit latent spaces have the innate ability to manipulate text attributes by moving along latent directions". Finds that PCA on VAE representations discovers meaningful directions like sentence length, subject age, and verb tense. Useful baseline — your learned manipulator should outperform simple linear directions.

---

## 4. Semantic Decoders / Property Predictors from Latents (your Phase 1)

- **Louizos et al. (2017)** — *Causal Effect Inference with Deep Latent-Variable Models (CEVAE)*. Predecessor to many of the fairness VAEs above; uses a VAE with auxiliary predictors for causal inference. Your semantic decoder is structurally similar.

- **PCAE (Tu et al., 2022)** — *A Framework of Plug-in Conditional Auto-Encoder for Controllable Text Generation*. "Invents Broadcasting Net to repeatedly add control signals into latent space to create a concentrate and manipulable latent space in VAE. Experienced on both RNN and BART VAE models". Framework for plugging auxiliary heads into a pretrained VAE, similar to your Phase 1 setup.

---

## A Word of Warning from the Literature

There's one important caveat worth surfacing. Non-linear ICA results show that VAE latent variables are non-identifiable, meaning that VAE-based methods can estimate the latent variable but it is not guaranteed to be correctly identified, and non-identifiability of latent variables implies non-identifiability of counterfactual queries. In plain terms: your Phase 2 manipulator may learn to change *something* in latent space that decodes to the right metadata change, but it's not guaranteed to be the *causally correct* change. GCFN (Grari et al.) raises exactly this point. Your architecture partially sidesteps it by using the semantic decoder as a consistency check, but it's worth being explicit about.

---

## Suggested Reading Order

If I had to prioritize, I'd read in this order:

1. **Asai et al. (2022) JAIR version** — your north star
2. **Kusner et al. (2017)** — to ground the fairness framing
3. **Ma et al. (2023, CLAIRE)** — because it's structurally the closest to your full pipeline
4. **Xu et al. (2020)** — to understand the latent vacancy problem before you hit it
5. **Li et al. (2020, Optimus)** — as the text VAE backbone candidate
6. **Liu et al. (2022, LatentOps)** — for the manipulator framework
7. **Joo et al. (2024)** — the "construct a fair latent space" paper most aligned with your phase structure

Want me to dig deeper into any one of these, or help you map specific design decisions (e.g., discrete vs. continuous latents, choice of manipulator architecture) against what these papers found?