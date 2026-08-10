# R Counterfactually Fair Estimation

This folder contains reusable source functions for the three constructions in
[Kusner et al. (2017)](https://proceedings.neurips.cc/paper_files/paper/2017/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf):
Level One, Fair Add, and Fair K. The layout mirrors the R simulation code:
`estim_main.R` is the entry point and implementation helpers live under
`utils/`.

## Assumptions

- `adj_mat` is a square, binary, named DAG adjacency matrix. Parents are rows
  and children are columns, matching `exp/sim/R`.
- Every DAG node is a complete, finite numeric column in the training data.
  Integer/ordinal scores are modeled as continuous in this first version.
- `pa` may name one or several protected attributes; `target` names one numeric
  outcome.
- Models return an expected numeric target. They do not threshold or round
  predictions back to ordinal categories.

The methods use progressively stronger assumptions:

- `level_one`: regress the target on observable non-descendants of `pa` only.
- `fair_add`: additionally recover additive structural errors for nodes on
  directed `pa`-to-`target` paths by regressing every node on all of its DAG
  parents.
- `fair_k`: add one shared standard-normal latent factor K as a parent of all
  nodes on the protected-to-target paths. A linear-Gaussian EM algorithm learns
  the node equations. Prediction infers K from pre-target nodes only, so an
  observed target in `newdata` is never used.

Counterfactual invariance is model-based. It is exact for Fair Add and Fair K
under their fitted linear additive/linear-Gaussian SCMs, but only approximate
when the data-generating process violates those assumptions. In particular, the
project's clipped and rounded ordinal simulator is intentionally a useful
misspecification check, so the runnable example reports the observed prediction
shift instead of asserting that it is zero.

## In-memory use

Run from the repository root:

```r
source("exp/estim/R/estim_main.R")

fit <- estim_fit(
  data = train_data,
  adj_mat = adj_mat,
  pa = "G",
  target = "Q",
  method = "fair_add"
)

y_fair <- predict(fit, newdata = test_data)
fair_inputs <- predict(fit, newdata = test_data, type = "features")
```

The direct method-specific fitters are also available:

```r
fit_l1 <- estim_fit_level_one(train_data, adj_mat, pa = "G", target = "Q")
fit_add <- estim_fit_fair_add(train_data, adj_mat, pa = "G", target = "Q")
fit_k <- estim_fit_fair_k(
  train_data,
  adj_mat,
  pa = "G",
  target = "Q",
  control = list(max_iter = 300L, tol = 1e-3, verbose = FALSE)
)
```

All fitted objects retain the DAG, protected attribute, target, selected fair
features, structural/latent models, and final outcome model for auditing.

## Runnable example and tests

`estim_main.R` runs a small train/test example on the default simulation SCM
when invoked directly:

```bash
Rscript exp/estim/R/estim_main.R
```

Run the dependency-free regression tests with:

```bash
Rscript exp/estim/R/tests/test_estim.R
```
