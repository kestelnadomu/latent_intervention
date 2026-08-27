# Semantic kernel $g$

$$g(\mathbf s \mid z): \mathcal Z \to \Delta(\mathcal S)$$

* $\Delta(\mathcal S)$: probability simplex over the $\lvert\mathcal S\rvert = 3456$ joint states
  * set of vectors $q \in \mathbb R^{3456}_{\ge 0}$ with $\sum_{\mathbf s} q_{\mathbf s} = 1$.
  * $g$ maps a latent to a full distribution over structured states, not a point estimate; the consistency constraint later compares two such distributions.

## Plan A — independent per-column heads

* Shared MLP trunk
* one categorical head per column
* columns predicted independently given $z$:

$$p(\mathbf s \mid z) = \prod_i p(s_i \mid z)$$

* asserts features in $\mathcal{S}$ conditionally independent given $z$
* --> Technically false
* Start here anyway:
  * if per-column marginal accuracy and downstream consistency loss are already good enough, the joint error does not bite
  * Escalate to Plan B only if it does.

```python
import torch
from torch import nn


class SemanticDecoderA(nn.Module):
    """Plan A: MLP trunk + one independent categorical head per column."""

    def __init__(self, latent_dim: int, cardinalities: dict[str, int],
                 hidden_dim: int = 256, n_hidden: int = 2, dropout: float = 0.1):
        super().__init__()
        layers, d = [], latent_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            d = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({c: nn.Linear(d, k) for c, k in cardinalities.items()})

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(z)
        return {c: head(h) for c, head in self.heads.items()}  # per-column logits

    def log_joint(self, z: torch.Tensor) -> torch.Tensor:
        """Dense (batch, 3456) log-probabilities via outer sum of per-column log-softmaxes."""
        logps = [torch.log_softmax(l, dim=-1) for l in self.forward(z).values()]
        acc = logps[0]
        for lp in logps[1:]:
            acc = acc[..., None] + lp[:, None, :]
            acc = acc.reshape(acc.shape[0], -1)
        return acc
```

## Plan B — autoregressive over the topological order

$$p(\mathbf s \mid z) = \prod_{i} p\big(s_i \mid s_{<i},\ z\big)$$

* Shared MLP trunk
* one small categorical head per column
* each head additionally fed embeddings of the already-decoded prefix
* --> Exact joint, no independence assumption, $\sum_i \lvert\mathcal S_i\rvert = 23$
output units. Impossible states receive zero mass structurally rather than by being learned.

* Condition each head on the full prefix $s_{<i}$
* not only on node's SCM parents $\mathbf{pa}(i)$ 

```python
class SemanticDecoderB(nn.Module):
    """Plan B: same trunk, heads conditioned on an embedding of the decoded prefix."""

    def __init__(self, latent_dim: int, order: list[tuple[str, int]],
                 hidden_dim: int = 256, embed_dim: int = 16):
        super().__init__()
        self.order = order
        self.trunk = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU())
        self.embed = nn.ModuleDict({c: nn.Embedding(k, embed_dim) for c, k in order})
        self.heads = nn.ModuleDict()
        for i, (c, k) in enumerate(order):
            self.heads[c] = nn.Linear(hidden_dim + i * embed_dim, k)

    def log_prob(self, z: torch.Tensor, s: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sum of per-column log p(s_i | s_<i, z) for observed rows s."""
        h, prefix, total = self.trunk(z), [], 0.0
        for c, _ in self.order:
            logits = self.heads[c](torch.cat([h, *prefix], dim=-1))
            total = total + torch.log_softmax(logits, dim=-1).gather(-1, s[c][:, None]).squeeze(-1)
            prefix.append(self.embed[c](s[c]))
        return total
```

## Training

$$\mathcal L_g = \mathbb E\big[\mathrm{CE}\big(g(\cdot \mid z),\ \mathbf s\big)\big]$$

- Cross-entropy is a strictly proper scoring rule for categorical targets
- its population minimiser is the true $p(\mathbf s \mid z)$
- no distributional noise model is needed
- post-hoc calibration if needed (temperature scaling on a held-out split) and measure it (per-column ECE, reliability curves)


## Rejected alternative

* Flat joint softmax** over all 3456 states
* trivially a correct joint and composes directly with a transition matrix
* But costly: at `hidden_dim: 256` the final layer is $256 \times 3456 = 884{,}736$ parameters against `n: 100` in `exp/sim/config.yaml`; even at $n=10^4$ that is ~3 examples per class
* Also -> not sparse enough:
  * SCM is *concentrated* — E, S, W, V, C are near-deterministic in the roots plus small Gaussians
  * --> so realized support is a few hundred states
  * $h_S \circ g$ pushes mass onto states that were rare or absent factually

Note this is not a trade-off against the flat vector: **the flat 3456-vector is a *view*, not a
parameterization.** The autoregressive product materializes to it exactly via one `einsum` over
a $(4,2,3,4,3,3,2,2)$ tensor (~28k flops/example), so anything downstream that wants a dense
vector still gets one.