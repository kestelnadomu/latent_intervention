# Notation

| Symbol | Meaning |
|---|---|
| $\mathcal X, \mathcal Z, \mathcal S$ | text space, latent space ($\mathbb R^{128}$), structured state space |
| $\mathbf S = (\texttt{R},\texttt{G},\texttt{A},\texttt{E},\texttt{S},\texttt{W},\texttt{V},\texttt{C})$ | structured state; $\lvert\mathcal S\rvert = 4\cdot2\cdot3\cdot4\cdot3\cdot3\cdot2\cdot2 = 3456$ |
| $\texttt{Q}$ | downstream outcome, excluded from $\mathbf S$ |
| $\delta$ | intervention spec, e.g. $\delta = do(\texttt{G}=1)$ |
| $'$ | counterfactual under $\delta$ (so $\mathbf S'$, $X'$, $Z'$) |
| $f: \mathcal X \to \mathcal Z$ | encoder, **frozen** |
| $g(\mathbf s \mid z)$ | semantic kernel, $\mathcal Z \to \Delta(\mathcal S)$ |
| $h_S(\mathbf s' \mid \mathbf s, \delta)$ | symbolic counterfactual kernel, $\mathcal S \to \Delta(\mathcal S)$ |
| $h_Z(z' \mid z, \delta)$ | latent editor, $\mathcal Z \to \Delta(\mathcal Z)$ — **the method** |

Conventions:

- **$f$, not $e$**, for the encoder (matches the extended abstract). $h_Z$ consumes a latent, so
  the composition is $h_Z(f(X))$ — never $f(h_Z(X))$.
- **Kernels are written as conditionals, not as arrows into a density.** Write $g(\mathbf s \mid z)$,
  not $g: Z \to p(S \mid Z)$; the arrow form names *spaces* ($\mathcal Z \to \Delta(\mathcal S)$),
  not elements.
- **Bold $\mathbf S$ for the state vector**, upright $\texttt{S}$ for the SCM node of the same
  name. Different objects; the collision is otherwise unreadable.
- Noise that induces a distribution is **marginalized, never conditioned on**. Write the map, or
  write the marginal — $p(\cdot \mid z, \varepsilon)$ is a point estimate.

# Problem Formulation

Let
- $X$ denote a text input (e.g., a CV)
- $A$ a sensitive attribute (e.g., gender)
- $Z$ be a latent representation produced by a pre-trained encoder.

We assume access to
- a structural causal model $M$
- over a set of observable variables $S$
- that captures the causal effect of $A$ on relevant features of $X$.

Given a counterfactual value $a'$ for $A$, our goal is to learn a latent manipulator that produces $Z'_{a'}$, the latent representation $X$ would have had under $do(A = a')$.