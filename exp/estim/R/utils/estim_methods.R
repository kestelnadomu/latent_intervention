# Counterfactually fair estimation methods ------------------------------------

#' Build a data frame from selected numeric features
#'
#' @param data Source data frame.
#' @param features Character vector of feature names.
#'
#' @return Numeric data frame preserving exact feature names.
#'
#' @noRd
estim_feature_frame <- function(data, features) {
  if (length(features) == 0L) {
    return(data.frame(row.names = seq_len(nrow(data))))
  }
  out <- as.data.frame(data[, features, drop = FALSE], check.names = FALSE)
  names(out) <- features
  out
}

#' Construct a fitted fair-estimation object
#'
#' @param method Estimation method identifier.
#' @param problem Validated problem inputs.
#' @param fields Method-specific object fields.
#'
#' @return An object inheriting from `estim_fair_model`.
#'
#' @noRd
estim_new_model <- function(method, problem, fields) {
  structure(
    c(
      list(
        method = method,
        adj_mat = problem$adj_mat,
        pa = problem$pa,
        target = problem$target
      ),
      fields
    ),
    class = c(paste0("estim_", method), "estim_fair_model")
  )
}

# Level One -------------------------------------------------------------------

#' Fit a Level One counterfactually fair predictor
#'
#' Implements Kusner et al.'s Level 1 construction by fitting the target only
#' on observable non-descendants of the protected attribute. Target descendants
#' are excluded as a safeguard against post-outcome leakage.
#'
#' @param data Numeric training data frame containing all DAG nodes.
#' @param adj_mat Square named adjacency matrix; parents are rows and children
#'   are columns.
#' @param pa Character vector naming one or more protected attributes.
#' @param target Character scalar naming the prediction target.
#'
#' @return A fitted object accepted by `predict()` and `estim_fair_features()`.
#'
#' @export
estim_fit_level_one <- function(data, adj_mat, pa, target) {
  problem <- estim_validate_problem(data, adj_mat, pa, target)
  features <- estim_level_one_features(
    problem$adj_mat,
    problem$pa,
    problem$target
  )
  x <- estim_feature_matrix(problem$data, features)
  outcome_model <- estim_fit_linear(x, problem$data[[problem$target]])

  estim_new_model(
    method = "level_one",
    problem = problem,
    fields = list(
      fair_features = features,
      required_newdata = features,
      outcome_model = outcome_model
    )
  )
}

#' Transform new observations into Level One fair features
#'
#' @param object Fitted Level One model.
#' @param newdata New data frame.
#'
#' @return Data frame containing the fair features.
#'
#' @noRd
estim_transform_level_one <- function(object, newdata) {
  estim_validate_newdata(newdata, object$required_newdata)
  estim_feature_frame(newdata, object$fair_features)
}

# Fair Add --------------------------------------------------------------------

#' Fit an additive-error counterfactually fair predictor
#'
#' Implements the Level 3 / Fair Add construction. Each observed node on a
#' protected-attribute-to-target path is fitted on all of its observed DAG
#' parents. Its additive residual estimates the structural error term. The
#' outcome model uses those residuals together with Level One features.
#'
#' @param data Numeric training data frame containing all DAG nodes.
#' @param adj_mat Square named adjacency matrix; parents are rows and children
#'   are columns.
#' @param pa Character vector naming one or more protected attributes.
#' @param target Character scalar naming the prediction target.
#'
#' @return A fitted object accepted by `predict()` and `estim_fair_features()`.
#'
#' @export
estim_fit_fair_add <- function(data, adj_mat, pa, target) {
  problem <- estim_validate_problem(data, adj_mat, pa, target)
  level_one <- estim_level_one_features(
    problem$adj_mat,
    problem$pa,
    problem$target
  )
  mediators <- estim_nodes_on_paths(
    problem$adj_mat,
    problem$pa,
    problem$target,
    include_target = FALSE
  )
  residual_names <- make.unique(paste0(".fair_eps_", mediators))
  names(residual_names) <- mediators

  structural_models <- setNames(vector("list", length(mediators)), mediators)
  for (node in mediators) {
    parents <- estim_parents(problem$adj_mat, node)
    model <- estim_fit_linear(
      estim_feature_matrix(problem$data, parents),
      problem$data[[node]]
    )
    structural_models[[node]] <- list(parents = parents, model = model)
  }

  fair_train <- estim_feature_frame(problem$data, level_one)
  for (node in mediators) {
    node_model <- structural_models[[node]]
    fitted <- estim_predict_linear(
      node_model$model,
      estim_feature_matrix(problem$data, node_model$parents)
    )
    fair_train[[residual_names[[node]]]] <- problem$data[[node]] - fitted
  }

  outcome_model <- estim_fit_linear(
    estim_feature_matrix(fair_train, names(fair_train)),
    problem$data[[problem$target]]
  )
  required <- unique(c(
    level_one,
    mediators,
    unlist(lapply(structural_models, `[[`, "parents"), use.names = FALSE)
  ))
  required <- setdiff(required, problem$target)

  estim_new_model(
    method = "fair_add",
    problem = problem,
    fields = list(
      level_one_features = level_one,
      mediator_nodes = mediators,
      residual_names = residual_names,
      structural_models = structural_models,
      fair_features = names(fair_train),
      required_newdata = required,
      outcome_model = outcome_model
    )
  )
}

#' Transform new observations into Fair Add structural-error features
#'
#' @param object Fitted Fair Add model.
#' @param newdata New data frame.
#'
#' @return Data frame containing Level One variables and estimated errors.
#'
#' @noRd
estim_transform_fair_add <- function(object, newdata) {
  estim_validate_newdata(newdata, object$required_newdata)
  out <- estim_feature_frame(newdata, object$level_one_features)

  for (node in object$mediator_nodes) {
    node_model <- object$structural_models[[node]]
    fitted <- estim_predict_linear(
      node_model$model,
      estim_feature_matrix(newdata, node_model$parents)
    )
    out[[object$residual_names[[node]]]] <- newdata[[node]] - fitted
  }
  out
}

# Fair K ----------------------------------------------------------------------

#' Normalize and validate Fair K controls
#'
#' @param control Named list of optional controls.
#'
#' @return Complete validated control list.
#'
#' @noRd
estim_validate_fair_k_control <- function(control) {
  if (is.null(control)) control <- list()
  if (!is.list(control) || (length(control) > 0L && is.null(names(control)))) {
    stop("`control` must be a named list.", call. = FALSE)
  }
  defaults <- list(
    max_iter = 200L,
    tol = 1e-3,
    ridge = 1e-6,
    variance_floor = 1e-6,
    verbose = FALSE
  )
  unknown <- setdiff(names(control), names(defaults))
  if (length(unknown) > 0L) {
    stop("Unknown Fair K control(s): ", paste(unknown, collapse = ", "),
         call. = FALSE)
  }
  out <- utils::modifyList(defaults, control)

  out$max_iter <- as.integer(out$max_iter)
  if (length(out$max_iter) != 1L || is.na(out$max_iter) || out$max_iter < 1L) {
    stop("`control$max_iter` must be a positive integer.", call. = FALSE)
  }
  for (name in c("tol", "ridge", "variance_floor")) {
    value <- as.numeric(out[[name]])
    if (length(value) != 1L || !is.finite(value) || value <= 0) {
      stop("`control$", name, "` must be a positive finite number.",
           call. = FALSE)
    }
    out[[name]] <- value
  }
  if (!is.logical(out$verbose) || length(out$verbose) != 1L ||
      is.na(out$verbose)) {
    stop("`control$verbose` must be TRUE or FALSE.", call. = FALSE)
  }
  out
}

#' Initialize row-level latent K values from conditional residuals
#'
#' @param data Training data.
#' @param adj_mat Validated adjacency matrix.
#' @param latent_nodes Nodes in the latent measurement model.
#'
#' @return Numeric vector of initial K scores.
#'
#' @noRd
estim_initialize_k <- function(data, adj_mat, latent_nodes) {
  residual_matrix <- vapply(latent_nodes, function(node) {
    parents <- estim_parents(adj_mat, node)
    model <- estim_fit_linear(
      estim_feature_matrix(data, parents),
      data[[node]]
    )
    model$residuals
  }, numeric(nrow(data)))
  if (is.null(dim(residual_matrix))) {
    residual_matrix <- matrix(residual_matrix, ncol = 1L)
  }

  scales <- apply(residual_matrix, 2L, stats::sd)
  usable <- is.finite(scales) & scales > sqrt(.Machine$double.eps)
  if (!any(usable)) return(rep(0, nrow(data)))

  standardized <- sweep(
    residual_matrix[, usable, drop = FALSE],
    2L,
    scales[usable],
    "/"
  )
  decomposition <- svd(standardized, nu = 1L, nv = 0L)
  scores <- as.numeric(decomposition$u[, 1L] * decomposition$d[[1L]])
  scores <- as.numeric(scale(scores))
  if (any(!is.finite(scores))) return(rep(0, nrow(data)))

  first_residual <- residual_matrix[, which(usable)[[1L]]]
  if (stats::cor(scores, first_residual) < 0) scores <- -scores
  scores
}

#' Fit the shared-K linear-Gaussian measurement model
#'
#' @param data Training data.
#' @param adj_mat Validated adjacency matrix.
#' @param latent_nodes Path nodes, including the target.
#' @param control Validated Fair K controls.
#'
#' @return Fitted latent model and EM diagnostics.
#'
#' @noRd
estim_fit_k_measurement <- function(data, adj_mat, latent_nodes, control) {
  n <- nrow(data)
  k_mean <- estim_initialize_k(data, adj_mat, latent_nodes)
  k_var <- rep(0.25, n)
  equations <- setNames(vector("list", length(latent_nodes)), latent_nodes)
  converged <- FALSE
  change <- Inf

  for (iteration in seq_len(control$max_iter)) {
    for (node in latent_nodes) {
      parents <- estim_parents(adj_mat, node)
      equations[[node]] <- estim_fit_latent_equation(
        parents = estim_feature_matrix(data, parents),
        y = data[[node]],
        k_mean = k_mean,
        k_var = k_var,
        ridge = control$ridge,
        variance_floor = control$variance_floor
      )
    }

    precision <- rep(1, n)
    numerator <- rep(0, n)
    for (node in latent_nodes) {
      equation <- equations[[node]]
      base_mean <- estim_predict_equation_base(equation, data)
      precision <- precision + equation$loading^2 / equation$variance
      numerator <- numerator +
        equation$loading * (data[[node]] - base_mean) / equation$variance
    }
    new_var <- 1 / precision
    new_mean <- numerator * new_var
    change <- max(abs(new_mean - k_mean), abs(new_var - k_var))
    k_mean <- new_mean
    k_var <- new_var

    if (isTRUE(control$verbose) &&
        (iteration == 1L || iteration %% 10L == 0L || change < control$tol)) {
      message("Fair K EM iteration ", iteration, ": change = ",
              format(change, digits = 4L))
    }
    if (change < control$tol) {
      converged <- TRUE
      break
    }
  }

  # Align equation parameters with the final posterior moments.
  for (node in latent_nodes) {
    parents <- estim_parents(adj_mat, node)
    equations[[node]] <- estim_fit_latent_equation(
      parents = estim_feature_matrix(data, parents),
      y = data[[node]],
      k_mean = k_mean,
      k_var = k_var,
      ridge = control$ridge,
      variance_floor = control$variance_floor
    )
  }

  list(
    equations = equations,
    nodes = latent_nodes,
    iterations = iteration,
    converged = converged,
    final_change = change,
    control = control
  )
}

#' Infer posterior moments of K without using the target value
#'
#' @param measurement Fitted shared-K measurement model.
#' @param data Data used as evidence.
#' @param evidence_nodes Path nodes available before the target.
#'
#' @return List containing posterior `mean` and `variance` vectors.
#'
#' @noRd
estim_infer_k <- function(measurement, data, evidence_nodes) {
  n <- nrow(data)
  precision <- rep(1, n)
  numerator <- rep(0, n)

  for (node in evidence_nodes) {
    equation <- measurement$equations[[node]]
    base_mean <- estim_predict_equation_base(equation, data)
    precision <- precision + equation$loading^2 / equation$variance
    numerator <- numerator +
      equation$loading * (data[[node]] - base_mean) / equation$variance
  }

  list(mean = numerator / precision, variance = 1 / precision)
}

#' Fit a shared-latent-factor counterfactually fair predictor
#'
#' Implements a numeric, linear-Gaussian generalization of Kusner et al.'s
#' Level 2 Fair K model. One standard-normal latent K is added as a parent of
#' every observed node on a protected-attribute-to-target path. EM estimates
#' the measurement equations. Row-level K is inferred at prediction time only
#' from pre-target path nodes, never from the observed target.
#'
#' @param data Numeric training data frame containing all DAG nodes.
#' @param adj_mat Square named adjacency matrix; parents are rows and children
#'   are columns.
#' @param pa Character vector naming one or more protected attributes.
#' @param target Character scalar naming the prediction target.
#' @param control Optional named list with `max_iter`, `tol`, `ridge`,
#'   `variance_floor`, and `verbose`.
#'
#' @return A fitted object accepted by `predict()` and `estim_fair_features()`.
#'
#' @export
estim_fit_fair_k <- function(data,
                             adj_mat,
                             pa,
                             target,
                             control = list()) {
  problem <- estim_validate_problem(data, adj_mat, pa, target)
  control <- estim_validate_fair_k_control(control)
  level_one <- estim_level_one_features(
    problem$adj_mat,
    problem$pa,
    problem$target
  )
  latent_nodes <- estim_nodes_on_paths(
    problem$adj_mat,
    problem$pa,
    problem$target,
    include_target = TRUE
  )
  evidence_nodes <- setdiff(latent_nodes, problem$target)

  if (length(evidence_nodes) == 0L) {
    measurement <- list(
      equations = list(),
      nodes = character(),
      iterations = 0L,
      converged = TRUE,
      final_change = 0,
      control = control
    )
    k_train <- rep(0, nrow(problem$data))
  } else {
    measurement <- estim_fit_k_measurement(
      data = problem$data,
      adj_mat = problem$adj_mat,
      latent_nodes = latent_nodes,
      control = control
    )
    if (!isTRUE(measurement$converged)) {
      warning(
        "Fair K EM reached `control$max_iter` before convergence; final change = ",
        format(measurement$final_change, digits = 4L),
        ". Inspect `measurement_model` or relax `control$tol`.",
        call. = FALSE
      )
    }
    k_train <- estim_infer_k(
      measurement,
      problem$data,
      evidence_nodes
    )$mean
  }

  fair_train <- estim_feature_frame(problem$data, level_one)
  fair_train[[".fair_K"]] <- k_train
  outcome_model <- estim_fit_linear(
    estim_feature_matrix(fair_train, names(fair_train)),
    problem$data[[problem$target]]
  )

  evidence_parents <- if (length(evidence_nodes) == 0L) {
    character()
  } else {
    unlist(lapply(measurement$equations[evidence_nodes], `[[`, "parents"),
           use.names = FALSE)
  }
  required <- unique(c(level_one, evidence_nodes, evidence_parents))
  required <- setdiff(required, problem$target)

  estim_new_model(
    method = "fair_k",
    problem = problem,
    fields = list(
      level_one_features = level_one,
      latent_nodes = latent_nodes,
      evidence_nodes = evidence_nodes,
      measurement_model = measurement,
      fair_features = names(fair_train),
      required_newdata = required,
      outcome_model = outcome_model
    )
  )
}

#' Transform new observations into Fair K features
#'
#' @param object Fitted Fair K model.
#' @param newdata New data frame.
#'
#' @return Data frame containing Level One variables and posterior mean K.
#'
#' @noRd
estim_transform_fair_k <- function(object, newdata) {
  estim_validate_newdata(newdata, object$required_newdata)
  out <- estim_feature_frame(newdata, object$level_one_features)
  if (length(object$evidence_nodes) == 0L) {
    out[[".fair_K"]] <- rep(0, nrow(newdata))
  } else {
    out[[".fair_K"]] <- estim_infer_k(
      object$measurement_model,
      newdata,
      object$evidence_nodes
    )$mean
  }
  out
}
