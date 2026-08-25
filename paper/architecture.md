# Architecture

Source of truth for model definitions and notation. Study design, data, baselines and
evaluation live in `experiments.md`.

## Notation

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

---

# Semantic kernel $g$

$g(\mathbf s \mid z): \mathcal Z \to \Delta(\mathcal S)$. Trained on factual pairs
$(f(X), \mathbf S)$ only.

## Definition — autoregressive over the topological order

$$p(\mathbf s \mid z) = \prod_{i} p\big(s_i \mid s_{<i},\ z\big), \qquad
\texttt{R},\texttt{G},\texttt{A},\texttt{E},\texttt{S},\texttt{W},\texttt{V},\texttt{C}$$

Shared MLP trunk, one small categorical head per column, each head fed embeddings of the
already-decoded prefix. Exact joint, no independence assumption, $\sum_i \lvert\mathcal S_i\rvert = 23$
output units. Impossible states receive zero mass structurally rather than by being learned.

**Condition each head on the full prefix $s_{<i}$, not on the node's SCM parents.** Given $z$ the
variables are all coupled — the posterior does *not* inherit the SCM's factorisation, so
restricting to $\mathbf{pa}(i)$ would be a second, subtler independence error.

### Training

$$\mathcal L_g = \mathbb E\big[\mathrm{CE}\big(g(\cdot \mid z),\ \mathbf s\big)\big]$$

Cross-entropy is a strictly proper scoring rule for categorical targets, so its population
minimiser is the true $p(\mathbf s \mid z)$ — no distributional noise model is needed at this
stage. If calibration is a concern, fix it post-hoc (temperature scaling on a held-out split)
and measure it (per-column ECE, reliability curves); do not reach for input-noise ensembling,
which under CE training is explicitly trained to be noise-invariant and collapses.

### Rejected alternatives

**Independent per-column softmaxes** (`src/semantic_decoder.py` as of writing) assert
$\texttt{E},\texttt{S},\texttt{W},\texttt{V},\texttt{C}$ conditionally independent given $z$,
which is false. The objective compares distributions *over $\mathcal S$*, so a wrong joint
corrupts the loss itself, not merely a reported metric.

**Flat joint softmax** over all 3456 states is trivially a correct joint and composes directly
with a transition matrix, which is its appeal. But at `hidden_dim: 256` the final layer is
$256 \times 3456 = 884{,}736$ parameters against `n: 100` in `exp/sim/config.yaml`; even at
$n=10^4$ that is ~3 examples per class. Worse, the SCM is *concentrated* — E, S, W, V, C are
near-deterministic in the roots plus small Gaussians — so realized support is a few hundred
states, while $h_S \circ g$ pushes mass onto states that were rare or absent factually. The
consistency loss would then compare against untrained logits precisely where it matters most.
The autoregressive form generalizes compositionally: it never needs to have seen a tuple, only
each conditional.

Note this is not a trade-off against the flat vector: **the flat 3456-vector is a *view*, not a
parameterization.** The autoregressive product materializes to it exactly via one `einsum` over
a $(4,2,3,4,3,3,2,2)$ tensor (~28k flops/example), so anything downstream that wants a dense
vector still gets one.

---

# Symbolic counterfactual kernel $h_S$

$h_S(\mathbf s' \mid \mathbf s, \delta)$. **Available in closed form** — no sampling error, no
fitting. Empirical transition counts are a validation check, not the source.

Validity of $h_S$ acting on $g$'s output requires $\mathbf S' \perp Z \mid \mathbf S$: the latent
carries no information about the counterfactual beyond what $\mathbf s$ contains. This holds in
the simulator — $\mathbf s$ is the complete set of endogenous non-outcome nodes, so
$p(\varepsilon \mid \mathbf s)$ is determined by $\mathbf s$, and everything else feeding the
text (template, persona, sub-bin age/experience, LLM sampling) is drawn from independent rngs.
It is a real assumption and would break if the text leaked $\texttt{Q}$.

## Abduction

Given the full factual state $\mathbf s$, every parent of every node is observed, so the noise
posterior factorises:

$$p(\varepsilon \mid \mathbf s) = \prod_i \mathrm{TruncNormal}\big(\varepsilon_i;\ \mu_i,\ \sigma_i,\ [l_i, u_i)\big)$$

with the interval obtained by inverting `clip_round` against the node's mechanism
$m_i = m_i(\mathbf{pa}_i)$:

- interior, $0 < s_i < k_i - 1$: $\varepsilon_i \in [\,s_i - m_i - 0.5,\ s_i - m_i + 0.5\,)$
- floor, $s_i = 0$: $\varepsilon_i \in (-\infty,\ 0.5 - m_i)$
- ceiling, $s_i = k_i - 1$: $\varepsilon_i \in [\,k_i - 1.5 - m_i,\ \infty)$

Clipping is what makes the boundary categories carry unbounded noise mass, which is why
$\texttt{E}=0$ and $\texttt{E}=3$ rows behave differently from interior ones.

## Action and prediction

$\mathbf s' = F(\varepsilon, \delta)$ is deterministic, so push the factorised posterior forward
in topological order by exact enumeration. With $\lvert\mathcal S\rvert = 3456$ the whole thing
precomputes as a **sparse $3456 \times 3456$ transition matrix per $\delta$** — mean 5.8
nonzeros per row, max 14, 2263 supported states, a few MB — and $h_S$ becomes a lookup at train
time. Applied to a distribution it is a pushforward through a Markov kernel:

$$(h_S \circ g)(\mathbf s' \mid z, \delta) = \sum_{\mathbf s \in \mathcal S} h_S(\mathbf s' \mid \mathbf s, \delta)\, g(\mathbf s \mid z)$$

## Composition of interventions

Composing two actions marginalizes over the intermediate state:

$$p(\mathbf s' \mid \mathbf s, \delta_a, \delta_b) = \mathbb E_{\mathbf s^* \sim p(\cdot \mid \mathbf s, \delta_b)}\big[\,p(\mathbf s' \mid \mathbf s^*, \delta_a)\,\big]$$

Intersectionality does **not** blow up exponentially, because the descendant closure saturates:
$do(\texttt{G}{=}1)$ touches $\{\texttt{E},\texttt{S},\texttt{W},\texttt{V},\texttt{C}\}$, and
adding $do(\texttt{R}{=}2)$ touches the same set. Composition is "set both nodes, propagate
once," not a product of dense matrices. If a matrix form is wanted for a compound $\delta$,
restrict it to the closure and condition on the non-descendants: for $do(\texttt{G}{=}1)$ that
is $\mathbb R^{12 \times 288 \times 144}$ (context $(\texttt{R},\texttt{A})$, source
$(\texttt{G},\texttt{E},\texttt{S},\texttt{W},\texttt{V},\texttt{C})$, target
$(\texttt{E}',\texttt{S}',\texttt{W}',\texttt{V}',\texttt{C}')$) — 498k entries, ~2 MB.

For Studies 2/3 there is no known SCM and $h_S$ must be learned. Design the interface so the
analytic and learned versions are interchangeable: same signature,
`h_S(s, delta) -> sparse vector over S`.

---

# Latent editor $h_Z$

A kernel $h_Z(z' \mid z, \delta): \mathcal Z \to \Delta(\mathcal Z)$. Freeze $f, g, h_S$; train
$h_Z$ on the objective below. Noise input lets $h_Z$ internalise $h_S$'s ambiguity, so **no SCM
is needed at inference**.

The failure to design against: discrete ambiguity (is $\texttt{E}'$ 1 or 2?) induces distinct,
separated regions of latent space. A **post**-additive residual Gaussian
$z' = z + \mu_\theta + \sigma_\theta \odot u$ is a location family and must place mass *between*
the modes — the "averaging incompatible targets" failure.

## A — Pre-additive noise (engression)

$$z' = z + \Delta_\theta(z + \varepsilon,\ \delta), \qquad \varepsilon \sim \mathcal N(0, \sigma^2 I)$$

$h_Z(\cdot\mid z,\delta)$ is the pushforward of $\varepsilon$ through this map. Being
pre-additive, it is *not* subject to the failure above: a nonlinear $\Delta_\theta$ can fold a
unimodal $\varepsilon$ onto separated modes. Composition needs Monte Carlo:

$$(g\circ h_Z)(\cdot\mid z,\delta) \approx \tfrac1M\textstyle\sum_m g\big(\cdot\mid z+\Delta_\theta(z+\varepsilon_m,\delta)\big)$$

By convexity of $D_{\mathrm{KL}}$ in its second argument this is an *upper bound* on the true
objective — a valid surrogate, but $M$ forward passes of $g$ per example and gradient variance
that is worst in exactly the ambiguous strata the study is built to test.

## B — Discrete mixture over counterfactual states (preferred)

$$h_Z(\cdot \mid z,\delta) = \sum_{\mathbf s' \in \mathcal S} w_\theta(\mathbf s' \mid z,\delta)\ \ \delta_{\,z + \Delta_\phi(z, \mathbf s')}$$

- $w_\theta$ is an internal head with the same autoregressive architecture as $g$.
- The realiser $\Delta_\phi$ is **deterministic**, so each component is a point mass; $h_Z$ has
  no Lebesgue density and "sample $z'$" means sample $\mathbf s'$, then evaluate.
- At inference only $z$ and $\delta$ are inputs — $h_S$ is internalised in $w_\theta$'s weights.

Because the components are Dirac, the composition is exact with no inner expectation:

$$(g\circ h_Z)(\cdot\mid z,\delta) = \sum_{\mathbf s'} w_\theta(\mathbf s'\mid z,\delta)\ g\big(\cdot\mid z+\Delta_\phi(z,\mathbf s')\big)$$

If the realiser is good, $g(\cdot \mid z + \Delta_\phi(z,\mathbf s')) \approx \delta_{\mathbf s'}$
and the KL collapses to $D_{\mathrm{KL}}\big((h_S\circ g)\,\|\,w_\theta\big)$. That splits one
hard bilevel problem into two ordinary supervised ones:

- **$w_\theta$**: distil the exact 3456-vector $(h_S\circ g)(\cdot \mid z,\delta)$. Dense target,
  no sampling, no gradient-through-samples variance.
- **$\Delta_\phi$**: minimise $-\log g(\mathbf s' \mid z + \Delta_\phi(z,\mathbf s')) + \alpha\lVert\Delta\rVert_1 + \beta\lVert\Delta\rVert_2^2$
  against frozen $g$.

Then fine-tune end-to-end on the true KL to absorb the $\approx$. Pretrain-then-joint, not either
alone. **The advantage is in the pretraining** — evaluating the sum above still costs one $g$
pass per retained $\mathbf s'$, comparable to $M$-sample Monte Carlo under A.

**Truncation is not free.** $(h_S\circ g)$ is a $g$-weighted mixture over all $\mathbf s$, so its
support is the union of the rows $h_S(\cdot\mid\mathbf s,\delta)$ over $\{\mathbf s : g(\mathbf s\mid z) > 0\}$
— up to $14K$ for $g$-support $K$, not 14. The max-14 figure is a property of a *single row* of
$M_\delta$. Truncation collapses to lossless only where $g$ is near one-hot, i.e. it is lossy
exactly in the ambiguous rows of interest. Measure the composed support empirically before
fixing $k$.

**Design consequence**, to state openly: routing everything through $\mathbf s'$ makes $h_Z$ an
information bottleneck — it can only produce counterfactuals expressible in $\mathcal S$. A gives
up less, but that extra freedom is *unsupervised* (the objective constrains $z'$ only in
directions $g$ can read), so the loss could not have taught it either way.

Structurally this is close to GPT's $G \to K \to Q$, with $w_\theta$ playing $K_S$'s role. The two
differences that matter both survive: **$Z'$ never enters a loss**, and **inference needs no SCM**.

## Shared risk: adversarial edits against a frozen probe

Both designs optimise a latent to maximise a frozen classifier's confidence — the textbook
adversarial-example construction. $\Delta$ will find off-manifold directions that convince $g$ of
$\mathbf s'$ while $z'$ sits nowhere near a real counterfactual latent, producing excellent RQ1a
numbers with worthless RQ1b recovery. Defenses:

- **Use the VAE prior.** LangVAE gives $p(z)\approx\mathcal N(0,I)$ for free; penalise $z'$
  implausible under the aggregate posterior. This is the only term that specifically punishes
  going off-manifold — L1/L2 proximity to $z$ merely limits step size.
- Keep dropout live in $g$ at edit time, or ensemble $g$; fooling an ensemble is much harder.
- Treat the identity check ($\mathbf S' = \mathbf S \Rightarrow z' \approx z$) as a first-class
  diagnostic — cheap, and it catches adversarial drift immediately.

---

# Consistency constraint and objective

The constraint is that the two paths from $z$ to a distribution over counterfactual states agree:

$$h_S \circ g \;=\; g \circ h_Z$$

Both sides live on $\mathcal S$, which is why they are directly comparable — and
$\lvert\mathcal S\rvert = 3456$ is small enough to **enumerate exactly**, so no sampling
approximation and no per-column independence assumption is needed.

$$\mathcal L = \mathbb E_{z}\Big[\, D_{\mathrm{KL}}\big(\,(h_S \circ g)(\cdot \mid z,\delta)\ \big\|\ (g \circ h_Z)(\cdot \mid z,\delta)\,\big) \,\Big] + \alpha \lVert z' - z\rVert_1 + \beta \lVert z' - z \rVert_2^2$$

Direction matters: **forward** KL is mass-covering, which is what we want — $h_Z$ must not
collapse onto one mode when the symbolic counterfactual is genuinely ambiguous.

## Guiding principle

$Z'$ **never enters a loss.** It is the evaluation target only. $g$ exists precisely because $Z'$
is unavailable outside the simulator; supervising on $Z'$ would make the method untransferable
and $g$ redundant. See `notes/leo/idea.md`.
