# Shared estimation helpers ---------------------------------------------------

#' Validate an adjacency matrix
#'
#' The estimation code uses the same convention as the simulation code: parent
#' nodes are rows, child nodes are columns, and an entry of one denotes an edge.
#'
#' @param adj_mat Square, named adjacency matrix.
#'
#' @return The validated adjacency matrix as an integer matrix.
#'
#' @noRd
estim_validate_adj_mat <- function(adj_mat) {
  if (!is.matrix(adj_mat)) {
    stop("`adj_mat` must be a matrix.", call. = FALSE)
  }
  if (nrow(adj_mat) != ncol(adj_mat)) {
    stop("`adj_mat` must be square.", call. = FALSE)
  }

  rn <- rownames(adj_mat)
  cn <- colnames(adj_mat)
  if (is.null(rn) || is.null(cn)) {
    stop("`adj_mat` must have row and column names.", call. = FALSE)
  }
  if (any(is.na(rn)) || any(is.na(cn)) ||
      any(!nzchar(rn)) || any(!nzchar(cn))) {
    stop("`adj_mat` row and column names must be non-empty.", call. = FALSE)
  }
  if (anyDuplicated(rn) || anyDuplicated(cn)) {
    stop("`adj_mat` row and column names must be unique.", call. = FALSE)
  }
  if (!identical(rn, cn)) {
    stop("`adj_mat` row and column names must be identical and in the same order.",
         call. = FALSE)
  }

  values <- as.vector(adj_mat)
  if (any(is.na(values)) || any(!values %in% c(0, 1, 0L, 1L, FALSE, TRUE))) {
    stop("`adj_mat` must contain only binary values 0/1.", call. = FALSE)
  }
  if (any(diag(adj_mat) != 0L)) {
    stop("`adj_mat` must not contain self-loops.", call. = FALSE)
  }

  out <- matrix(
    as.integer(adj_mat),
    nrow = nrow(adj_mat),
    ncol = ncol(adj_mat),
    dimnames = dimnames(adj_mat)
  )
  estim_topological_order(out)
  out
}

#' Compute a deterministic topological order
#'
#' @param adj_mat Adjacency matrix with parent rows and child columns.
#'
#' @return Character vector of graph nodes in topological order.
#'
#' @noRd
estim_topological_order <- function(adj_mat) {
  nodes <- colnames(adj_mat)
  incoming <- colSums(adj_mat != 0L)
  queue <- nodes[incoming == 0L]
  node_order <- character()

  while (length(queue) > 0L) {
    node <- queue[[1L]]
    queue <- queue[-1L]
    node_order <- c(node_order, node)

    children <- nodes[adj_mat[node, ] != 0L]
    for (child in children) {
      incoming[[child]] <- incoming[[child]] - 1L
      if (incoming[[child]] == 0L) queue <- c(queue, child)
    }
  }

  if (length(node_order) != length(nodes)) {
    stop("`adj_mat` must describe an acyclic graph.", call. = FALSE)
  }
  node_order
}

#' Return the parents of a graph node
#'
#' @param adj_mat Validated adjacency matrix.
#' @param node Character scalar naming a node.
#'
#' @return Character vector of parent nodes.
#'
#' @noRd
estim_parents <- function(adj_mat, node) {
  rownames(adj_mat)[adj_mat[, node] != 0L]
}

#' Traverse a graph away from a set of starting nodes
#'
#' @param adj_mat Validated adjacency matrix.
#' @param starts Character vector of starting nodes.
#' @param direction Either `"down"` for descendants or `"up"` for ancestors.
#'
#' @return Character vector of reached nodes, excluding `starts`.
#'
#' @noRd
estim_traverse <- function(adj_mat, starts, direction = c("down", "up")) {
  direction <- match.arg(direction)
  if (length(starts) == 0L) return(character())

  nodes <- colnames(adj_mat)
  frontier <- as.character(starts)
  seen <- character()

  while (length(frontier) > 0L) {
    node <- frontier[[1L]]
    frontier <- frontier[-1L]
    neighbours <- if (direction == "down") {
      nodes[adj_mat[node, ] != 0L]
    } else {
      nodes[adj_mat[, node] != 0L]
    }
    new_nodes <- setdiff(neighbours, c(seen, starts))
    seen <- unique(c(seen, new_nodes))
    frontier <- c(frontier, new_nodes)
  }

  nodes[nodes %in% seen]
}

#' Return descendants of one or more nodes
#'
#' @param adj_mat Validated adjacency matrix.
#' @param nodes Character vector of nodes.
#'
#' @return Character vector of descendants.
#'
#' @noRd
estim_descendants <- function(adj_mat, nodes) {
  estim_traverse(adj_mat, nodes, direction = "down")
}

#' Return ancestors of one or more nodes
#'
#' @param adj_mat Validated adjacency matrix.
#' @param nodes Character vector of nodes.
#'
#' @return Character vector of ancestors.
#'
#' @noRd
estim_ancestors <- function(adj_mat, nodes) {
  estim_traverse(adj_mat, nodes, direction = "up")
}

#' Find nodes lying on protected-attribute-to-target paths
#'
#' @param adj_mat Validated adjacency matrix.
#' @param pa Character vector naming protected attributes.
#' @param target Character scalar naming the outcome.
#' @param include_target Logical; whether to include the target.
#'
#' @return Character vector in topological order.
#'
#' @noRd
estim_nodes_on_paths <- function(adj_mat,
                                pa,
                                target,
                                include_target = FALSE) {
  descendants <- estim_descendants(adj_mat, pa)
  ancestors <- estim_ancestors(adj_mat, target)
  out <- intersect(descendants, c(ancestors, target))
  if (!include_target) out <- setdiff(out, target)
  estim_topological_order(adj_mat)[estim_topological_order(adj_mat) %in% out]
}

#' Select Level 1 fair predictors
#'
#' Selects observable non-descendants of the protected attributes. The target
#' and its descendants are also excluded to avoid outcome or post-outcome
#' leakage when the protected attribute does not cause the target.
#'
#' @param adj_mat Validated adjacency matrix.
#' @param pa Character vector naming protected attributes.
#' @param target Character scalar naming the outcome.
#'
#' @return Character vector of eligible observed features.
#'
#' @noRd
estim_level_one_features <- function(adj_mat, pa, target) {
  nodes <- colnames(adj_mat)
  excluded <- unique(c(
    pa,
    target,
    estim_descendants(adj_mat, pa),
    estim_descendants(adj_mat, target)
  ))
  nodes[!nodes %in% excluded]
}

#' Validate a fair-prediction training problem
#'
#' @param data Training data frame.
#' @param adj_mat Named adjacency matrix.
#' @param pa Character vector naming protected attributes.
#' @param target Character scalar naming the outcome.
#'
#' @return Normalized problem inputs.
#'
#' @noRd
estim_validate_problem <- function(data, adj_mat, pa, target) {
  if (!is.data.frame(data) || nrow(data) < 1L) {
    stop("`data` must be a non-empty data frame.", call. = FALSE)
  }
  if (is.null(names(data)) || anyDuplicated(names(data)) ||
      any(is.na(names(data))) || any(!nzchar(names(data)))) {
    stop("`data` must have unique, non-empty column names.", call. = FALSE)
  }

  adj_mat <- estim_validate_adj_mat(adj_mat)
  nodes <- colnames(adj_mat)
  missing_nodes <- setdiff(nodes, names(data))
  if (length(missing_nodes) > 0L) {
    stop("`data` is missing DAG node(s): ",
         paste(missing_nodes, collapse = ", "), call. = FALSE)
  }

  pa <- as.character(pa)
  if (length(pa) < 1L || any(is.na(pa)) || any(!nzchar(pa)) ||
      anyDuplicated(pa)) {
    stop("`pa` must contain one or more unique, non-empty node names.",
         call. = FALSE)
  }
  target <- as.character(target)
  if (length(target) != 1L || is.na(target) || !nzchar(target)) {
    stop("`target` must be one non-empty node name.", call. = FALSE)
  }
  unknown <- setdiff(c(pa, target), nodes)
  if (length(unknown) > 0L) {
    stop("Unknown DAG node(s): ", paste(unknown, collapse = ", "),
         call. = FALSE)
  }
  if (target %in% pa) {
    stop("`target` cannot also be a protected attribute.", call. = FALSE)
  }

  non_numeric <- nodes[!vapply(data[nodes], is.numeric, logical(1))]
  if (length(non_numeric) > 0L) {
    stop("All DAG columns must be numeric. Non-numeric column(s): ",
         paste(non_numeric, collapse = ", "), call. = FALSE)
  }
  invalid <- nodes[vapply(data[nodes], function(x) {
    anyNA(x) || any(!is.finite(x))
  }, logical(1))]
  if (length(invalid) > 0L) {
    stop("DAG columns must not contain missing or non-finite values: ",
         paste(invalid, collapse = ", "), call. = FALSE)
  }

  list(data = data, adj_mat = adj_mat, pa = pa, target = target)
}

#' Validate new data columns used by a fitted estimator
#'
#' @param newdata New data frame.
#' @param required Character vector of required columns.
#'
#' @return `newdata`, invisibly validated.
#'
#' @noRd
estim_validate_newdata <- function(newdata, required) {
  if (!is.data.frame(newdata) || nrow(newdata) < 1L) {
    stop("`newdata` must be a non-empty data frame.", call. = FALSE)
  }
  missing_columns <- setdiff(required, names(newdata))
  if (length(missing_columns) > 0L) {
    stop("`newdata` is missing required column(s): ",
         paste(missing_columns, collapse = ", "), call. = FALSE)
  }
  non_numeric <- required[!vapply(newdata[required], is.numeric, logical(1))]
  if (length(non_numeric) > 0L) {
    stop("Required `newdata` columns must be numeric: ",
         paste(non_numeric, collapse = ", "), call. = FALSE)
  }
  invalid <- required[vapply(newdata[required], function(x) {
    anyNA(x) || any(!is.finite(x))
  }, logical(1))]
  if (length(invalid) > 0L) {
    stop("Required `newdata` columns contain missing or non-finite values: ",
         paste(invalid, collapse = ", "), call. = FALSE)
  }
  invisible(newdata)
}

#' Convert selected data-frame columns to a numeric matrix
#'
#' @param data Data frame.
#' @param features Character vector of columns.
#'
#' @return Numeric matrix with one row per observation.
#'
#' @noRd
estim_feature_matrix <- function(data, features) {
  if (length(features) == 0L) {
    return(matrix(numeric(), nrow = nrow(data), ncol = 0L,
                  dimnames = list(NULL, character())))
  }
  out <- as.matrix(data[, features, drop = FALSE])
  storage.mode(out) <- "double"
  out
}

#' Fit a stable linear model to a numeric feature matrix
#'
#' @param x Numeric feature matrix.
#' @param y Numeric response vector.
#'
#' @return Lightweight linear-model list.
#'
#' @noRd
estim_fit_linear <- function(x, y) {
  x <- as.matrix(x)
  y <- as.numeric(y)
  design <- cbind(`(Intercept)` = 1, x)
  fit <- stats::lm.fit(x = design, y = y)
  coefficients <- fit$coefficients
  coefficients[is.na(coefficients)] <- 0
  fitted <- as.numeric(design %*% coefficients)

  list(
    coefficients = coefficients,
    features = colnames(x),
    fitted.values = fitted,
    residuals = y - fitted,
    rank = fit$rank
  )
}

#' Predict from a lightweight linear model
#'
#' @param model Model returned by `estim_fit_linear()`.
#' @param x Numeric feature matrix in the training feature order.
#'
#' @return Numeric prediction vector.
#'
#' @noRd
estim_predict_linear <- function(model, x) {
  x <- as.matrix(x)
  if (ncol(x) != length(model$features)) {
    stop("Prediction matrix does not match the fitted feature layout.",
         call. = FALSE)
  }
  if (length(model$features) > 0L) colnames(x) <- model$features
  design <- cbind(`(Intercept)` = 1, x)
  as.numeric(design %*% model$coefficients)
}

#' Solve a regularized symmetric linear system
#'
#' @param lhs Square numeric matrix.
#' @param rhs Numeric vector.
#' @param ridge Non-negative ridge penalty; the intercept is not penalized.
#'
#' @return Numeric solution vector.
#'
#' @noRd
estim_solve_system <- function(lhs, rhs, ridge) {
  penalty <- rep(ridge, nrow(lhs))
  penalty[[1L]] <- 0
  lhs <- lhs + diag(penalty, nrow = nrow(lhs))
  tryCatch(
    as.numeric(solve(lhs, rhs)),
    error = function(e) as.numeric(qr.solve(lhs, rhs, tol = 1e-10))
  )
}

#' Fit one conditional linear-Gaussian equation with latent K
#'
#' Uses posterior moments of K in the M-step of the Fair K EM algorithm.
#'
#' @param parents Numeric parent matrix.
#' @param y Numeric node values.
#' @param k_mean Posterior means of K.
#' @param k_var Posterior variances of K.
#' @param ridge Small ridge penalty for numerical stability.
#' @param variance_floor Positive lower bound for residual variance.
#'
#' @return Fitted equation parameters.
#'
#' @noRd
estim_fit_latent_equation <- function(parents,
                                     y,
                                     k_mean,
                                     k_var,
                                     ridge,
                                     variance_floor) {
  parents <- as.matrix(parents)
  w <- cbind(`(Intercept)` = 1, parents)
  w_names <- colnames(w)

  lhs <- rbind(
    cbind(crossprod(w), crossprod(w, k_mean)),
    c(crossprod(k_mean, w), sum(k_mean^2 + k_var))
  )
  rhs <- c(crossprod(w, y), sum(k_mean * y))
  coefficients <- estim_solve_system(lhs, rhs, ridge = ridge)
  names(coefficients) <- c(w_names, ".fair_K")

  beta <- coefficients[seq_len(ncol(w))]
  loading <- coefficients[[length(coefficients)]]
  mean_y <- as.numeric(w %*% beta + loading * k_mean)
  expected_sse <- sum((y - mean_y)^2 + loading^2 * k_var)

  list(
    parents = colnames(parents),
    coefficients = beta,
    loading = loading,
    variance = max(expected_sse / length(y), variance_floor)
  )
}

#' Predict the non-latent part of a Fair K node equation
#'
#' @param equation Fitted latent equation.
#' @param data Numeric data frame.
#'
#' @return Numeric conditional mean excluding the K contribution.
#'
#' @noRd
estim_predict_equation_base <- function(equation, data) {
  parents <- estim_feature_matrix(data, equation$parents)
  design <- cbind(`(Intercept)` = 1, parents)
  as.numeric(design %*% equation$coefficients)
}
