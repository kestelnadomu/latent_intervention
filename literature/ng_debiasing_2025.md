# Debiasing Reward Models by Representation Learning with Guarantees (Ng et al, 2025)
- [arXiv](https://arxiv.org/pdf/2510.23751)

- **Causal Representation Learning** as preliminary:
  - aims to uncover latent variables and their causal relations from low-level observations such as images or text (Schölkopf et al., 2021).
  - The task is notoriously difficult, because the latent variables generally cannot be identified even when the they areindependent (i.e., the nonlinear independent component analysis problem) (Hyvärinen and Pajunen, 1999; Hyvärinen et al., 2023).
  - To achieve identifiability, existing works therefore relies on further assumptions, such as access to multiple distributions (Squires et al., 2023; von Kügelgen et al., 2023; Zhang et al., 2024) or multiple views (Yao et al., 2024; Xu et al., 2024)
- **Identifiability**: Observed data --uniquely determine--> latent variables
- **Theorem 1**: If you train a VAE that reconstructs observable data, uses $S$ (surrogate for spurious variable) in the prior $p(Ẑ | S)$, and enforces $Ẑ_C ⊥ S$ (via the HSIC regularizer), the $Ẑ_C$ you recover will be a faithful encoding of $Z_C$ (correctness-related content). A reward model trained on $Ẑ_C$ will therefore generalize across all p_test values.
- **Theorem 2**: Harder and more interesting theorem. You don't have S. You don't even have a name for the bias. But you have something else: multiple human labelers, each giving you a reward R_k, and each labeler implicitly attends to a different subset of the latent factors.
- **Relevance**: 
  - If your metadata contains a column that is the sensitive attribute (or a clean surrogate for it), you're in the Theorem 1 world. Their ELBO + HSIC loss would let you train Phase 0 in a way that makes a sensitive-free sub-latent $Ẑ_C$ explicit from the start, rather than hoping your Phase 2 manipulator can "undo" the bias after the fact.
  - If your metadata doesn't include a clean sensitive attribute but you have annotations from multiple sources that differ in what biases they reflect, Theorem 2 tells you how to exploit that diversity