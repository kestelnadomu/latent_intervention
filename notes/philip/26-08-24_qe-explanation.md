## Original proposal for $h_Z$

The original method can work as a **semantic editor**, but it does not consistently estimate the same-person counterfactual $Z'$.

The original loss is essentially

$$
\widehat Z'
=
\arg\min_{\widetilde z}
\left[
-\log g(S'\mid \widetilde z)
+\alpha\|\widetilde z-Z\|_1
+\beta\|\widetilde z-Z\|_2^2
\right],
\qquad
\widetilde z=h_Z(Z,\delta).
$$

It says:

1. Find an embedding that $g$ recognizes as having semantics $S'$.
2. Among all such embeddings, choose one close to the factual $Z$.

The problem is that $g$ decodes only a few coarse attributes from a 128-dimensional embedding. Consequently, there can be many different embeddings $z'$ for which

$$
g(S'\mid z')
$$

is large. These embeddings can differ in wording, candidate-specific information, omitted attributes, and downstream meaning. The semantic constraint therefore tells us only that the result must belong to the large set of embeddings that $g$ labels as $S'$. It does not tell us which one is the counterfactual embedding of this particular candidate.

The $L_1/L_2$ penalty selects one embedding from this set, but “closest in the encoder coordinates” is not a causal principle. Rotating or rescaling the latent representation could change which embedding is closest without changing the information represented. Making $h_Z$ stochastic does not solve this: many different distributions of $Z'$ can produce exactly the same distribution after being decoded by $g$.

Statistically, the $g$-constraint defines a set of admissible answers, while the proximity regularizer arbitrarily chooses within that set. The model may also find unnatural embeddings that are classified correctly by the frozen $g$.

## Recommended alternative

### Training with conditioning on $S'$

The revised training uses the simulator’s complete matched observations

$$
(Z_i,S_i,S'_i,Z'_i),
$$

where factual and counterfactual CVs share the same candidate, SCM noise, persona, and rendering context. We then train a target-conditioned version of the latent editor,

$$
h_Z^{\mathrm{cond}}
\bigl(dz'\mid z,s,s',\delta\bigr),
$$

against the actually observed simulated $Z'_i$, using a proper paired loss such as conditional likelihood or an energy score. The semantic loss

$$
-\log g(S'_i\mid \widetilde Z'_i)
$$

can remain an auxiliary constraint, but it is no longer used as a substitute for the missing $Z'_i$.

Conditioning on $S'$ separates two questions:

- $h_S(S'\mid S,\delta)$: **Which symbolic counterfactual occurs?**
- $h_Z^{\mathrm{cond}}(Z'\mid Z,S,S',\delta)$: **How is that symbolic state realized for this candidate in latent space?**

### Inference without conditioning on $S'$

At inference, $S'$ does not have to be supplied. The original symbolic-free editor is recovered by marginalizing over $S$ and $S'$:

$$
h_Z^{\mathrm{dir}}(dz'\mid z,\delta)
=
\sum_{s,s'}
g(ds\mid z)\,
h_S(ds'\mid s,\delta)\,
h_Z^{\mathrm{cond}}(dz'\mid z,s,s',\delta).
$$

This is a mixture over the possible symbolic counterfactuals—not an average latent embedding. If only one network should be used at deployment, this composition can be distilled into a direct $h_Z^{\mathrm{dir}}(Z'\mid Z,\delta)$. Because the simulator supplies paired $Z'$, that direct model should preferably also be trained against the true paired $Z'$; distillation is mainly a way to compress the modular procedure.

Thus, the decisive improvement is not merely adding $S'$ as an input. It is:

> Use $S'$ to separate the symbolic transition from its latent realization, and train that realization against the paired $Z'$ produced by the simulator.

Without paired $Z'$, or an explicit structural assumption defining how candidate-specific latent information is preserved, neither version identifies the true same-person counterfactual.
