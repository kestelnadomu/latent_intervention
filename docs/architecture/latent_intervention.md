# Latent editor $h_Z$

$$h_Z(z' \mid z, \delta): \mathcal Z \to \Delta(\mathcal Z)$$

* Freeze $f, g, h_S$
* train $h_Z$ on the objective below
* Noise input lets $h_Z$ internalise $h_S$'s ambiguity, so **no SCM is needed at inference**.

## Why a kernel, not a plain transformer $z \mapsto z'$?

* naive baseline: deterministic net $z' = T_\theta(z, \delta)$ (a residual transformer over $[z\text{-token},\ \delta\text{-tokens}]$)
* why a distributional approach: target is a distribution:
  * $(h_S\circ g)(\cdot\mid z,\delta)$ is genuinely multimodal in the ambiguous strata (is $\texttt{E}'$ 1 or 2?)
  * distinct, separated regions of $\mathcal Z$ decode to each mode
  * Under the mass-covering **forward** KL, a single $z'$ is dragged to a point *between* the modes that decodes to neither — the   "averaging incompatible targets" failure.
  * A location family ($z' = z + \mu_\theta + \sigma_\theta\odot u$, post-additive Gaussian) fails the same way for the same reason

## Open question: unit or distributional counterfactual?

To be argued openly in the paper, not settled by fiat. Both readings are defensible and the
plans commit to different ones.

**For the unit counterfactual.** A text is one unit. Pearl's rung-3 counterfactual of a unit
is *deterministic* once its exogenous noise is known: one text, one counterfactual. The spread
in $(h_S\circ g)(\cdot\mid z,\delta)$ has two sources, and neither belongs to the unit:

* **coarseness of $\mathcal S$**: $T$ is hidden, and where $D$ and $U$ sit inside their bins
  is unknown, so $h_S$ maps a discretised state to a spread of $\mathbf s'$;
* **information discarded by $g$**: $z$ may carry evidence about the exogenous noise (the
  proxies $P, L, H, A$; wording) that $g(z)$ drops.

If $z$ carries that evidence, $p(\mathbf S'\mid z)$ is *sharper* than $h_S\circ g$. The
consistency KL then forces $h_Z$ to be less informative than the text allows. Plan 0 is in
effect trained on this reading: its per-column CE uses the simulated $\mathbf S'$ with the
true unit noise, so it estimates the marginals of $p(\mathbf S'_j\mid z)$.

**For the distributional counterfactual.** The only counterfactual semantics we can *check* is
the symbolic one. $h_S$ is the ground truth we own, and $g$ is the only lens on $z$. A target
sharper than $h_S\circ g$ is an unverifiable claim about what the encoder "knows". It also
makes the result depend on how much noise information the LLM happened to leak into the text.
Under abduction uncertainty an honest manipulator should return a *distribution* of
counterfactual latents, each one a plausible text. That is the neurosymbolic promise:
$h_Z$ inherits the SCM's uncertainty.

**What the experiment can say.** The talent SFM makes the gap measurable:

* `talent_posterior_accuracy()` gives the Bayes-optimal recovery of $T$ from the verbalised
  proxies;
* compare $\mathrm{KL}\big(p(\mathbf S'\mid\text{true noise})\,\|\,g\circ h_Z\big)$ for Plan 0
  (unit target) against Plans A/B/C (symbolic target), per stratum of $g$'s confidence.

Where $g$ is confident the two targets coincide. In the ambiguous strata the difference
between them measures how much information beyond $\mathcal S$ the latent carries.

## Plan A — Pre-additive noise (engression)

$$z' = z + \Delta_\theta(z + \varepsilon,\ \delta), \qquad \varepsilon \sim \mathcal N(0, \sigma^2 I)$$

* $h_Z(\cdot\mid z,\delta)$ is the pushforward of $\varepsilon$ through this map
* nonlinear $\Delta_\theta$ can fold a unimodal $\varepsilon$ onto separated modes
* Composition needs Monte Carlo:

$$(g\circ h_Z)(\cdot\mid z,\delta) \approx \tfrac1M\textstyle\sum_m g\big(\cdot\mid z+\Delta_\theta(z+\varepsilon_m,\delta)\big)$$

* convexity of $D_{\mathrm{KL}}$ in its second argument this is an *upper bound* on the true objective — a valid surrogate, but $M$ forward passes of $g$ per example and gradient variance
that is worst in exactly the ambiguous strata the study is built to test.

```python
class LatentInterventionA(nn.Module):
    """z' = z + Delta_theta(z + eps, delta), eps ~ N(0, noise_std^2 I)."""

    def forward(self, z, values, mask, generator=None):
        eps = self.noise_std * torch.randn(z.shape, device=z.device, generator=generator)
        return z + self.delta(z + eps, values, mask)          # Delta_theta: zero-init residual

    def composed_log_joint(self, z, values, mask, decoder, n_samples, generator=None):
        """MC estimate of log (g . h_Z)(. | z, delta): log-mean-exp of g's dense joint."""
        zs = torch.stack([self.forward(z, values, mask, generator) for _ in range(n_samples)])
        log_joints = torch.stack([decoder.log_joint(z_m) for z_m in zs])   # (M, batch, |S|)
        composed = torch.logsumexp(log_joints, dim=0) - math.log(n_samples)
        return composed, zs - z                                            # (batch, |S|), shifts

# training: loss = D_KL(target || composed) + alpha||z'-z||_1 + beta||z'-z||_2^2
#   target = (h_S . g)(. | z, delta), precomputed once against the frozen decoder + h_S
# src/latent_intervention.py :: LatentInterventionA, train_latent_intervention_a
```

## Plan C — Outsourced noise token

$$z' = z + \Delta_\theta(z,\ \varepsilon,\ \delta), \qquad \varepsilon \sim \mathcal N(0, I_k)$$

* $\varepsilon$ enters $\Delta_\theta$ as its **own token**: $[z\text{-token},\ \varepsilon\text{-token},\ \delta\text{-tokens}]$.
  Unlike Plan A, the latent reaches the network intact. Plan A's full-dimensional
  $z+\varepsilon$ mixes noise into the conditioning, so part of its effect is smoothing.
* Noise outsourcing: any conditional law can be written as $F(z, U)$. The counterfactual
  uncertainty here is low-dimensional ($T$, plus where $D$ and $U$ sit inside their bins),
  so $k \approx 2\text{–}8$.
* **Identifiability of the mechanism.** Forward KL in $\mathcal S$ cannot tell apart
  (a) ignoring $\varepsilon$ and parking $z'$ where $g$ is ambiguous (the Plan 0 solution) from
  (b) using $\varepsilon$ to scatter $z'$ over points where $g$ is confident. Both give the
  same $g\circ h_Z$. A per-sample entropy term prefers (b):

$$\mathcal L_C = D_{\mathrm{KL}}\big((h_S\circ g)\,\|\,\widehat{g\circ h_Z}\big) + \lambda\,\mathbb E_\varepsilon\big[H\big(g(\cdot\mid z'_\varepsilon)\big)\big] + \alpha\lVert z'-z\rVert_1 + \beta\lVert z'-z\rVert_2^2$$

* Diagnostic `spread`: the mean norm of the per-dimension std of $z'$ across $\varepsilon$.
  A value near 0 means $\varepsilon$ is ignored.
* **Bridge to Plan B:** make $\varepsilon$ discrete with a learned prior $w_\theta$ and the
  construction becomes Plan B's mixture.
* The same entropy term applies to Plan A, which has the same ambiguity.

```python
# src/latent_intervention.py :: LatentInterventionNoiseToken, train_latent_intervention_noise_token
# config: latent_intervention.variant: noise_token  (noise_dim, n_samples, entropy_weight)
```

## Plan B — Discrete mixture over counterfactual states (preferred)

$$h_Z(\cdot \mid z,\delta) = \sum_{\mathbf s' \in \mathcal S} w_\theta(\mathbf s' \mid z,\delta)\ \ \delta_{\,z + \Delta_\phi(z, \mathbf s')}$$

* $w_\theta$: internal head with the same (autoregressive) architecture as $g$.
* $\Delta_\phi$ is **deterministic realiser**
  * so each component is a point mass
  * $h_Z$ has no Lebesgue density
  * "sample $z'$" means sample $\mathbf s'$, then evaluate.
* At inference only $z$ and $\delta$ are inputs — $h_S$ is internalised in $w_\theta$'s weights.

* *components are Dirac --> composition is exact with no inner expectation:

$$(g\circ h_Z)(\cdot\mid z,\delta) = \sum_{\mathbf s'} w_\theta(\mathbf s'\mid z,\delta)\ g\big(\cdot\mid z+\Delta_\phi(z,\mathbf s')\big)$$

* good realiser --> $g(\cdot \mid z + \Delta_\phi(z,\mathbf s')) \approx \delta_{\mathbf s'}$
* --> KL collapses to $D_{\mathrm{KL}}\big((h_S\circ g)\,\|\,w_\theta\big)$. That splits one
hard bilevel problem into two ordinary supervised ones:

* **$w_\theta$**: distil the exact 3456-vector $(h_S\circ g)(\cdot \mid z,\delta)$.
  * Dense target
  * no sampling, no gradient-through-samples variance
* **$\Delta_\phi$**: minimise $-\log g(\mathbf s' \mid z + \Delta_\phi(z,\mathbf s')) + \alpha\lVert\Delta\rVert_1 + \beta\lVert\Delta\rVert_2^2$ against frozen $g$.

* Then fine-tune end-to-end on the true KL to absorb the $\approx$. Pretrain-then-joint, not either
alone. **The advantage is in the pretraining** — evaluating the sum above still costs one $g$
pass per retained $\mathbf s'$, comparable to $M$-sample Monte Carlo under A.

```python
class LatentInterventionB(nn.Module):
    """h_Z(. | z, delta) = sum_s' w_theta(s' | z, delta) dirac_{z + Delta_phi(z, s')}."""

    def __init__(self, latent_dim, columns=SCM_COLUMNS, top_k=16, ...):
        self.w_theta  = _AutoregWeights(latent_dim, columns, ...)   # same arch as g, + delta
        self.realiser = _DeltaNet(latent_dim, columns, ...)         # Delta_phi(z, s'), deterministic

    def realise(self, z, s_prime):                                  # s_prime: (batch, n_cols)
        mask = torch.ones_like(s_prime, dtype=torch.bool)
        return z + self.realiser(z, s_prime, mask)

    def forward(self, z, values, mask):                             # inference: argmax w_theta
        idx, _ = self.w_theta.top_k(z, values, mask, 1)
        return self.realise(z, unflatten_state_index(idx[:, 0], self.columns))

    def composed_log_joint(self, z, values, mask, decoder):
        """Exact log (g . h_Z) over the retained top-k s' -- no inner expectation."""
        idx, logw = self.w_theta.top_k(z, values, mask, self.top_k)          # (batch, k)
        comps = [logw[:, j:j+1] + decoder.log_joint(self.realise(z, unflatten_state_index(idx[:, j], self.columns)))
                 for j in range(idx.shape[1])]
        return torch.logsumexp(torch.stack(comps), dim=0)

# pretrain:  w_theta  -> D_KL( (h_S . g)(. | z, delta) || w_theta )        (dense target)
#            Delta_phi -> -log g(s' | z + Delta_phi(z, s')) + l1||Delta||_1 + l2||Delta||_2^2   (true s')
# joint:     D_KL(target || composed_log_joint) + alpha/beta proximity on the realised shifts
# src/latent_intervention.py :: LatentInterventionB, train_latent_intervention_b
```

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

* two paths from $z$ to a distribution over counterfactual states agree:

$$h_S \circ g \;=\; g \circ h_Z$$

$$\mathcal L = \mathbb E_{z}\Big[\, D_{\mathrm{KL}}\big(\,(h_S \circ g)(\cdot \mid z,\delta)\ \big\|\ (g \circ h_Z)(\cdot \mid z,\delta)\,\big) \,\Big] + \alpha \lVert z' - z\rVert_1 + \beta \lVert z' - z \rVert_2^2$$

Direction matters: **forward** KL is mass-covering, which is what we want — $h_Z$ must not collapse onto one mode when the symbolic counterfactual is genuinely ambiguous.

```python
@torch.no_grad()
def consistency_target(decoder, h_s, latents, intervention):
    """(n, |S|) target (h_S . g)(. | z, delta) = M_delta^T @ g(. | z), frozen -> precompute once."""
    m_t = h_s.transition_matrix(intervention).t().coalesce()
    g_probs = decoder.log_joint(latents).exp()
    return torch.sparse.mm(m_t, g_probs.t()).t()

def _forward_kl(pred_log, target):                     # D_KL(target || pred), batch mean
    return F.kl_div(pred_log, target, reduction="batchmean")
```

Both Plan A and Plan B share this target and this direction; only `pred_log` — the estimate of
$\log(g\circ h_Z)$ — differs (Monte Carlo vs. exact top-$k$ sum). The `src/latent_intervention.py`
baseline `LatentIntervention` (Plan 0) skips the KL entirely and trains per-column CE against a
single $\mathbf s'$ — the deterministic map the section head argues against.