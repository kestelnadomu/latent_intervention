# Public fit, transform, and prediction interface -----------------------------

#' Fit a counterfactually fair prediction model
#'
#' Dispatches to Level One, Fair Add, or Fair K while keeping one common public
#' signature for arbitrary named DAG adjacency matrices.
#'
#' @param data Numeric training data frame containing every DAG node.
#' @param adj_mat Square named adjacency matrix with parents in rows and children
#'   in columns.
#' @param pa Character vector naming one or more protected attributes.
#' @param target Character scalar naming the prediction target.
#' @param method One of `"level_one"`, `"fair_add"`, or `"fair_k"`.
#' @param control Optional method controls. Currently used only by Fair K.
#'
#' @return A fitted object with class `estim_fair_model`.
#'
#' @export
estim_fit <- function(data,
                      adj_mat,
                      pa,
                      target,
                      method = c("level_one", "fair_add", "fair_k"),
                      control = list()) {
  method <- match.arg(method)
  switch(
    method,
    level_one = estim_fit_level_one(data, adj_mat, pa, target),
    fair_add = estim_fit_fair_add(data, adj_mat, pa, target),
    fair_k = estim_fit_fair_k(data, adj_mat, pa, target, control = control)
  )
}

#' Extract the fair representation used for prediction
#'
#' @param object Fitted `estim_fair_model` object.
#' @param newdata New numeric data frame. The target column is never required or
#'   consulted.
#'
#' @return Data frame of method-specific counterfactually fair features.
#'
#' @export
estim_fair_features <- function(object, newdata) {
  if (!inherits(object, "estim_fair_model")) {
    stop("`object` must inherit from `estim_fair_model`.", call. = FALSE)
  }
  switch(
    object$method,
    level_one = estim_transform_level_one(object, newdata),
    fair_add = estim_transform_fair_add(object, newdata),
    fair_k = estim_transform_fair_k(object, newdata),
    stop("Unknown fitted estimation method: ", object$method, call. = FALSE)
  )
}

#' Predict a counterfactually fair target
#'
#' @param object Fitted `estim_fair_model` object.
#' @param newdata New numeric data frame.
#' @param type Either `"response"` for numeric expected-target predictions or
#'   `"features"` for the fair representation.
#' @param ... Reserved for compatibility with `stats::predict()`.
#'
#' @return Numeric prediction vector or fair-feature data frame.
#'
#' @export
predict.estim_fair_model <- function(object,
                                     newdata,
                                     type = c("response", "features"),
                                     ...) {
  type <- match.arg(type)
  features <- estim_fair_features(object, newdata)
  if (type == "features") return(features)

  estim_predict_linear(
    object$outcome_model,
    estim_feature_matrix(features, object$fair_features)
  )
}

#' Print a fitted fair-estimation model
#'
#' @param x Fitted `estim_fair_model` object.
#' @param ... Unused.
#'
#' @return `x`, invisibly.
#'
#' @export
print.estim_fair_model <- function(x, ...) {
  cat("Counterfactually fair estimation model\n")
  cat("  method: ", x$method, "\n", sep = "")
  cat("  protected attribute(s): ", paste(x$pa, collapse = ", "), "\n",
      sep = "")
  cat("  target: ", x$target, "\n", sep = "")
  cat("  fair features: ",
      if (length(x$fair_features) == 0L) "(intercept only)" else
        paste(x$fair_features, collapse = ", "),
      "\n", sep = "")
  if (identical(x$method, "fair_k")) {
    cat("  EM converged: ", x$measurement_model$converged,
        " (", x$measurement_model$iterations, " iteration(s))\n", sep = "")
  }
  invisible(x)
}
