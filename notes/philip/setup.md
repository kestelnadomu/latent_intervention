### Setup

+ $S$ has 8 categorical variables with cardinalities $(4,2,3,4,3,3,2,2)$: 23 logits and $3{,}456$ joint states.
+ $Z\in\mathbb R^{128}$.

### Semantic decoder $g$
+ Model: 
    + autoregressive categorical model
$$
g(S\mid Z)=\prod_{j=1}^{8}g_j(S_j\mid Z,S_{<j}).
$$
    

+ Architecture: 
    + MLP $128\to256\to256$ (GELU, dropout $0.1$) and 8 prefix-conditioned softmax heads.
+ Training:
    + For an observed pairs $(Z_i,S_i)$, use categorical negative log-likelihood
$$
\mathcal L_g=-\sum_{i,j}\log g_j(S_{ij}\mid Z_i,S_{i,<j}).
$$

### Symbolic intervention $h_S$
+ Model:
    + (SCM) transition kernel (matrix)
$$
S'\sim h_S(\cdot\mid S,\delta).
$$
+ Input: 
    + factual $S$ and intervention $\delta$; output: a distribution over $S'$.
+ LIBERTy: 
    + use the known SCM directly; no training, compute transition matrix.
+ LIBERTy-free: 
    + Assume bijective generation mechanism (Nasr-Esfahany, 2023), fit all node mechanisms, then do abduction-intervention-prediction.

### Latent renderer $h_Z$

#### $h_Z$ using $(S,S')$
+ **LIBERTy setup:** 
    + $(Z_i,S_i,S'_i,Z'_i)$ is observed. 
    + Model: conditional normalizing flow
    $$
        Z'=h_Z(Z,S,S',\varepsilon),
        \qquad \varepsilon\sim\mathcal N(0,I_{128}).
    $$  
    where, for fixed $(Z,S,S')$,
    $$
        h_Z(Z,S,S',\cdot):\mathbb R^{128}_{\varepsilon}
        \longrightarrow\mathbb R^{128}_{Z'}
    $$
    is bijective.
    + Architecture:
        + (stacked) conditional affine or spline flows; each conditioner receives $(Z,S,S')$.
    + Training: 
        + For $\varepsilon_i=h_Z^{-1}(Z'_i;Z_i,S_i,S'_i)$, minimize
        $$
        \mathcal L_{h_Z}
        =
        -\sum_i
        \left[
        \log p_\varepsilon(\varepsilon_i)
        +
        \log\left|
        \det
        \frac{\partial h_Z^{-1}(Z'_i;Z_i,S_i,S'_i)}
        {\partial Z'_i}
        \right|
        \right].
        $$

+ **LIBERTy-free setup**:
    + $(S'_i,Z'_i)$ is unobserved
    + Model:
        + conditional normalizing flow
    $$
    Z=h_Z(S,U),
    \qquad U\sim\mathcal N(0,I_{128}),
    $$
    where, for fixed $S$,
    $$
    h_Z(S,\cdot):\mathbb R^{128}_U\longrightarrow\mathbb R^{128}_Z.
    $$
    + Architecture: 
        + (stacked) conditional or spline flows.
    + Training: 
        + observed pairs $(S_i,Z_i)$, let $u_i=h_Z^{-1}(Z_i;S_i)$, then optimize the conditional flow negative log-likelihood
    $$
    \mathcal L_{h_Z}
    =-\sum_i\left[
    \log p_U(u_i)+
    \log\left|\det\frac{\partial h_Z^{-1}(Z_i;S_i)}{\partial Z_i}\right|
    \right].
    $$
    + Assumption: 
        + Assume bijective generation mechanism (Nasr-Esfahany, 2023).
    + Abduction-intervention-prediction:
        + Given factual $(Z_i,S_i)$, first obtain $S'_i\sim h_S(\cdot\mid S_i,\delta)$ and then preserve the inferred flow noise:
$$
u_i=h_Z^{-1}(Z_i;S_i),
\qquad
\widetilde Z'_i=h_Z(S'_i,u_i).
$$


#### $h_Z$ without $(S,S')$

+ **LIBERTy setup:**
    + $(Z_i,\delta,Z'_i)$ is observed.
    + Model: conditional normalizing flow
    $$
    Z'=h_Z(Z,\delta,\varepsilon),
    \qquad \varepsilon\sim\mathcal N(0,I_{128}),
    $$
    where, for fixed $(Z,\delta)$,
    $$
    h_Z(Z,\delta,\cdot):\mathbb R^{128}_{\varepsilon}
    \longrightarrow\mathbb R^{128}_{Z'}
    $$
    is bijective.
    + Architecture:
        + (stacked) conditional affine or spline flows; each conditioner receives $(Z,\delta)$.
    + Training:
        + For $\varepsilon_i=h_Z^{-1}(Z'_i;Z_i,\delta)$, minimize
    $$
    \mathcal L_{h_Z}
    =-\sum_i\left[
    \log p_\varepsilon(\varepsilon_i)+
    \log\left|\det\frac{\partial h_Z^{-1}(Z'_i;Z_i,\delta)}{\partial Z'_i}\right|
    \right].
    $$
    + Alternatively use distillation (optional):
        + use the $(S,S')$-conditioned model (from above) as a teacher for $h_Z^{\mathrm{dist}}(Z'\mid Z,\delta)$.

+ **LIBERTy-free setup:**
    + $Z'_i$ is unobserved.
    + Distillation: 
        + use pseudo-targets from the $(S,S')$-conditioned model to train $h_Z^{\mathrm{dist}}(Z'\mid Z,\delta)$ (the student receives the teacher's assumptions).
    + Procedure:
        + sample multiple pseudo-targets:
    $$
    S_i\sim g(\cdot\mid Z_i),
    \qquad S'_i\sim h_S(\cdot\mid S_i,\delta),
    $$
    $$
    u_i=h_Z^{-1}(Z_i;S_i),
    \qquad \widetilde Z'_i=h_Z(S'_i,u_i).
    $$
    + Model: 
        + direct conditional normalizing flow
    $$
    Z'=h_Z^{\mathrm{dist}}(Z,\delta,\varepsilon),
    \qquad \varepsilon\sim\mathcal N(0,I_{128}).
    $$
    + Architecture:
        + same as in the LIBERTy setup; each conditioner receives $(Z,\delta)$.
    + Training:
        + use the same conditional flow loss as above with $\widetilde Z'_i$ in place of $Z'_i$.
