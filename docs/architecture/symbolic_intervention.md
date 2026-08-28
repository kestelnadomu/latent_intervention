# Symbolic counterfactual kernel $h_S$

$$h_S(\mathbf s' \mid \mathbf s, \delta)$$

* Available in closed form (given the SCM mechanism + noise parameters)
* Assumption: $\mathbf S' \perp Z \mid \mathbf S$:
  * the latent carries no information about the counterfactual beyond what $\mathbf s$ contains.
  * This holds in the simulator — $\mathbf s$ is the complete set of endogenous non-outcome nodes, so $p(\varepsilon \mid \mathbf s)$ is determined by $\mathbf s$
  * everything else feeding the text (template, persona, sub-bin age/experience, LLM sampling) is drawn from independent rngs.
  * It is a real assumption and would break if the text leaked $\texttt{Q}$

## Abduction

* Given full factual state $\mathbf s$:
  * every parent of every node is observed, so the noise posterior factorises:

$$p(\varepsilon \mid \mathbf s) = \prod_i \mathrm{TruncNormal}\big(\varepsilon_i;\ \mu_i,\ \sigma_i,\ [l_i, u_i)\big)$$

with the interval obtained by inverting `clip_round` against the node's mechanism $m_i = m_i(\mathbf{pa}_i)$:

- interior, $0 < s_i < k_i - 1$: $\varepsilon_i \in [\,s_i - m_i - 0.5,\ s_i - m_i + 0.5\,)$
- floor, $s_i = 0$: $\varepsilon_i \in (-\infty,\ 0.5 - m_i)$
- ceiling, $s_i = k_i - 1$: $\varepsilon_i \in [\,k_i - 1.5 - m_i,\ \infty)$

Clipping is what makes the boundary categories carry unbounded noise mass, which is why
$\texttt{E}=0$ and $\texttt{E}=3$ rows behave differently from interior ones.

## Unknown noise parameters

The closed form needs $(\mu_i, \sigma_i)$ — and the mechanism $m_i$. Three regimes:

* **Simulator available (Study 1).** Take $m_i$ and $(\mu_i,\sigma_i)$ from `exp/sim/scm.py`.
  Exact; `SymbolicIntervention.from_scm()`.
* **Mechanism known, noise unknown.** Estimate $(\mu_i,\sigma_i)$ from a factual sample: the
  residual $r = s_i - m_i(\mathbf{pa}_i)$ is $\varepsilon_i$ plus uniform rounding jitter, so
  method-of-moments on the *interior* (non-censored) rows gives
  $\hat\mu_i = \bar r$, $\hat\sigma_i^2 = \widehat{\mathrm{Var}}(r) - \tfrac1{12}$. Principled
  version: MLE of the rounded-censored-normal likelihood. `SymbolicIntervention.fit(df)`.
  Concentrated nodes (E, S sit near a boundary) have few interior rows and the estimate is
  biased low — check against empirical transition counts.
* **Nothing known (Studies 2/3).** No SCM; $h_S$ is learned. Keep the interface identical —
  `h_S(s, delta) -> sparse vector over S` — so the analytic and learned kernels are drop-in.

Empirical transition counts from paired $(\mathbf s, \mathbf s')$ data are always the
validation check, never the source, whenever a closed form exists.

## Action and prediction

* $\mathbf s' = F(\varepsilon, \delta)$ is deterministic
* --> push the factorised posterior forward in topological order by exact enumeration over the
  descendant closure of $\delta$ (non-descendants keep their factual value).
* e.g. with $\lvert\mathcal S\rvert = 3456$ the whole thing precomputes as a **sparse $3456 \times 3456$
  transition matrix per $\delta$** — for $do(\texttt{G}{=}1)$: ~11.6k nonzeros, mean 3.3 / max 17
  per row, 1728 reachable states (G pinned), a few MB
* $h_S$ becomes a lookup at train time. Applied to a distribution it is a pushforward through a Markov kernel:

$$(h_S \circ g)(\mathbf s' \mid z, \delta) = \sum_{\mathbf s \in \mathcal S} h_S(\mathbf s' \mid \mathbf s, \delta)\, g(\mathbf s \mid z)$$

```python
import math
import numpy as np
import torch


def normal_cdf(x, mu, sigma):
    if x in (math.inf, -math.inf):
        return float(x > 0)
    return 0.5 * math.erfc(-(x - mu) / (sigma * math.sqrt(2)))


def abduction_interval(s_i, m_fac, k):
    """Invert clip_round(m_fac + eps) = s_i for the noise interval [l, u)."""
    if s_i <= 0:
        return (-math.inf, 0.5 - m_fac)
    if s_i >= k - 1:
        return (k - 1.5 - m_fac, math.inf)
    return (s_i - m_fac - 0.5, s_i - m_fac + 0.5)


def category_probs(m_prime, lo, hi, mu, sigma, k):
    """P(clip_round(m_prime + eps) = c) for eps ~ TruncNormal(mu, sigma, [lo, hi))."""
    p = np.zeros(k)
    for c in range(k):
        pre_lo = -math.inf if c == 0 else c - 0.5 - m_prime
        pre_hi = math.inf if c == k - 1 else c + 0.5 - m_prime
        a, b = max(lo, pre_lo), min(hi, pre_hi)
        if b > a:
            p[c] = normal_cdf(b, mu, sigma) - normal_cdf(a, mu, sigma)
    return p / p.sum()


# Full three-step recipe (abduct -> act -> predict by exact enumeration over the
# descendant closure) and the |S| x |S| sparse M_delta live in
# src/symbolic_intervention.py::SymbolicIntervention.
#   h_S = SymbolicIntervention.from_scm()
#   M = h_S.transition_matrix({"G": 1})          # sparse (3456, 3456), rows sum to 1
#   hs_g = h_S.compose(g_probs, {"G": 1})        # (h_S . g): (batch, 3456) -> (batch, 3456)
```

## Composition of interventions

Composing two actions marginalizes over the intermediate state:

$$p(\mathbf s' \mid \mathbf s, \delta_a, \delta_b) = \mathbb E_{\mathbf s^* \sim p(\cdot \mid \mathbf s, \delta_b)}\big[\,p(\mathbf s' \mid \mathbf s^*, \delta_a)\,\big]$$

* Intersectionality does **not** blow up exponentially, because the descendant closure saturates:
  * e.g. $do(\texttt{G}{=}1)$ touches $\{\texttt{E},\texttt{S},\texttt{W},\texttt{V},\texttt{C}\}$, and adding $do(\texttt{R}{=}2)$ touches the same set
  * Composition is "set both nodes, propagate once," not a product of dense matrices.