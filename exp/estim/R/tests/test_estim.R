# Regression tests for counterfactually fair estimation -----------------------

wd <- getwd()
source(file.path(wd, "exp", "estim", "R", "estim_main.R"))

expect_error <- function(expr, pattern = NULL) {
  error <- tryCatch(
    {
      force(expr)
      NULL
    },
    error = identity
  )
  if (is.null(error)) stop("Expected an error, but none was raised.")
  if (!is.null(pattern) && !grepl(pattern, conditionMessage(error))) {
    stop("Unexpected error: ", conditionMessage(error))
  }
  invisible(error)
}

make_adj_mat <- function() {
  nodes <- c("A", "X", "M1", "M2", "Y")
  out <- matrix(0L, length(nodes), length(nodes), dimnames = list(nodes, nodes))
  out["A", "M1"] <- 1L
  out["X", "M1"] <- 1L
  out["M1", "M2"] <- 1L
  out["X", "M2"] <- 1L
  out["A", "Y"] <- 1L
  out["M2", "Y"] <- 1L
  out["X", "Y"] <- 1L
  out
}

simulate_linear_dag <- function(n, seed) {
  set.seed(seed)
  A <- stats::rbinom(n, 1L, 0.5)
  X <- stats::rnorm(n)
  K <- stats::rnorm(n)
  M1 <- 1.2 * A + 0.4 * X + 1.1 * K + stats::rnorm(n, sd = 0.25)
  M2 <- 0.7 * M1 - 0.3 * X + 0.8 * K + stats::rnorm(n, sd = 0.25)
  Y <- 0.5 * A + 0.9 * M2 + 0.6 * X + K + stats::rnorm(n, sd = 0.3)
  data.frame(A = A, X = X, M1 = M1, M2 = M2, Y = Y)
}

adj_mat <- make_adj_mat()
train <- simulate_linear_dag(600L, seed = 1L)
test <- simulate_linear_dag(100L, seed = 2L)

# Graph-driven feature selection.
stopifnot(identical(estim_nodes_on_paths(adj_mat, "A", "Y"), c("M1", "M2")))
stopifnot(identical(estim_level_one_features(adj_mat, "A", "Y"), "X"))

fits <- list(
  level_one = estim_fit(train, adj_mat, pa = "A", target = "Y",
                        method = "level_one"),
  fair_add = estim_fit(train, adj_mat, pa = "A", target = "Y",
                       method = "fair_add"),
  fair_k = estim_fit(
    train,
    adj_mat,
    pa = "A",
    target = "Y",
    method = "fair_k",
    control = list(max_iter = 300L, tol = 1e-3)
  )
)

stopifnot(identical(fits$level_one$fair_features, "X"))
stopifnot(identical(fits$fair_add$mediator_nodes, c("M1", "M2")))
stopifnot(identical(fits$fair_k$evidence_nodes, c("M1", "M2")))
stopifnot(isTRUE(fits$fair_k$measurement_model$converged))

# Every method returns finite numeric predictions and never reads newdata$Y.
for (fit in fits) {
  prediction <- predict(fit, test)
  stopifnot(is.numeric(prediction), length(prediction) == nrow(test))
  stopifnot(all(is.finite(prediction)))

  changed_target <- test
  changed_target$Y <- changed_target$Y + 1e6
  stopifnot(isTRUE(all.equal(
    prediction,
    predict(fit, changed_target),
    tolerance = 0
  )))

  target_removed <- test[, setdiff(names(test), "Y"), drop = FALSE]
  stopifnot(isTRUE(all.equal(
    prediction,
    predict(fit, target_removed),
    tolerance = 0
  )))
}

# Fair Add is invariant when an intervention is propagated with the fitted
# structural equations while holding each estimated error fixed.
add_cf <- test
add_cf$A <- 1 - add_cf$A
for (node in fits$fair_add$mediator_nodes) {
  node_model <- fits$fair_add$structural_models[[node]]
  factual_base <- estim_predict_linear(
    node_model$model,
    estim_feature_matrix(test, node_model$parents)
  )
  structural_error <- test[[node]] - factual_base
  cf_base <- estim_predict_linear(
    node_model$model,
    estim_feature_matrix(add_cf, node_model$parents)
  )
  add_cf[[node]] <- cf_base + structural_error
}
stopifnot(isTRUE(all.equal(
  predict(fits$fair_add, test),
  predict(fits$fair_add, add_cf),
  tolerance = 1e-10
)))

# Fair K is invariant under the analogous fitted-model intervention with K and
# conditional errors fixed. K inference uses only M1/M2, never Y.
k_factual <- estim_fair_features(fits$fair_k, test)[[".fair_K"]]
k_cf <- test
k_cf$A <- 1 - k_cf$A
for (node in fits$fair_k$evidence_nodes) {
  equation <- fits$fair_k$measurement_model$equations[[node]]
  factual_base <- estim_predict_equation_base(equation, test)
  structural_error <- test[[node]] - factual_base - equation$loading * k_factual
  cf_base <- estim_predict_equation_base(equation, k_cf)
  k_cf[[node]] <- cf_base + equation$loading * k_factual + structural_error
}
stopifnot(isTRUE(all.equal(
  predict(fits$fair_k, test),
  predict(fits$fair_k, k_cf),
  tolerance = 1e-10
)))

# Validation catches malformed graphs and unsupported data.
cyclic <- adj_mat
cyclic["M2", "M1"] <- 1L
expect_error(
  estim_fit(train, cyclic, pa = "A", target = "Y", method = "level_one"),
  "acyclic"
)
with_na <- train
with_na$M1[[1L]] <- NA_real_
expect_error(
  estim_fit(with_na, adj_mat, pa = "A", target = "Y", method = "fair_add"),
  "missing or non-finite"
)

# Multiple protected attributes and a direct-effect-only DAG reduce safely to
# intercept-only fair prediction when no pre-target fair information exists.
direct_nodes <- c("A", "B", "Y")
direct_adj <- matrix(
  0L,
  nrow = length(direct_nodes),
  ncol = length(direct_nodes),
  dimnames = list(direct_nodes, direct_nodes)
)
direct_adj["A", "Y"] <- 1L
direct_adj["B", "Y"] <- 1L
direct_data <- data.frame(
  A = rep(0:1, 20L),
  B = rep(rep(0:1, each = 2L), 10L),
  Y = seq_len(40L) / 10
)
for (method in c("level_one", "fair_add", "fair_k")) {
  direct_fit <- estim_fit(
    direct_data,
    direct_adj,
    pa = c("A", "B"),
    target = "Y",
    method = method
  )
  direct_prediction <- predict(
    direct_fit,
    direct_data[, c("A", "B"), drop = FALSE]
  )
  stopifnot(length(direct_prediction) == nrow(direct_data))
  stopifnot(length(unique(round(direct_prediction, 12L))) == 1L)
}

message("All counterfactually fair estimation tests passed.")
