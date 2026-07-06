#' Return a fallback value when an object is `NULL`
#'
#' @param x Any R object.
#' @param y Fallback object to return when `x` is `NULL`.
#'
#' @return `x` if it is not `NULL`; otherwise `y`.
#'
#' @noRd
`%||%` <- function(x, y) if (!is.null(x)) x else y

#' Parse command-line key-value arguments
#'
#' Parses arguments of the form `key=value` and overlays them on a defaults
#' list. Arguments without `=` and arguments with empty keys are ignored.
#'
#' @param args Character vector of command-line arguments.
#' @param defaults Named list of default values.
#'
#' @return A named list containing `defaults` updated by parsed CLI values.
#'
#' @noRd
sim_parse_cli_args <- function(args = commandArgs(trailingOnly = TRUE),
                               defaults = list()) {
  out <- defaults
  if (length(args) == 0L) return(out)

  for (arg in args) {
    eq_pos <- regexpr("=", arg, fixed = TRUE)
    if (eq_pos < 1L) next

    key <- trimws(substr(arg, 1L, eq_pos - 1L))
    value <- trimws(substr(arg, eq_pos + 1L, nchar(arg)))
    if (!nzchar(key)) next

    out[[key]] <- value
  }

  out
}

#' Coerce a scalar value to integer
#'
#' Converts a scalar integer-like input to an R integer and raises an error when
#' the value is missing, non-finite, fractional, or outside the integer range.
#'
#' @param x Scalar value to coerce.
#' @param name Character scalar used in error messages.
#' @param default Optional fallback returned when `x` is missing or blank.
#'
#' @return An integer scalar.
#'
#' @noRd
sim_as_integer <- function(x, name = "value", default = NULL) {
  if (is.null(x) || length(x) == 0L || is.na(x[[1]]) ||
      !nzchar(trimws(as.character(x[[1]])))) {
    if (!is.null(default)) return(default)
    stop("`", name, "` must be an integer.", call. = FALSE)
  }

  value <- suppressWarnings(as.numeric(x[[1]]))
  if (!is.finite(value) || value != floor(value)) {
    stop("`", name, "` must be an integer.", call. = FALSE)
  }

  out <- as.integer(value)
  if (is.na(out)) stop("`", name, "` is outside the integer range.", call. = FALSE)
  out
}

#' Coerce a scalar value to nullable integer
#'
#' Converts a scalar integer-like input to an R integer. Blank values and the
#' strings `"NULL"`, `"None"`, and `"NA"` are treated as `NULL`.
#'
#' @param x Scalar value to coerce.
#' @param name Character scalar used in error messages.
#'
#' @return An integer scalar or `NULL`.
#'
#' @noRd
sim_as_nullable_integer <- function(x, name = "value") {
  if (is.null(x) || length(x) == 0L) return(NULL)

  value_chr <- trimws(as.character(x[[1]]))
  if (!nzchar(value_chr) || tolower(value_chr) %in% c("null", "none", "na")) {
    return(NULL)
  }

  sim_as_integer(value_chr, name = name)
}

#' Coerce a scalar value to logical
#'
#' Converts common logical encodings to `TRUE` or `FALSE`, including `TRUE/FALSE`,
#' `T/F`, `1/0`, and `yes/no`.
#'
#' @param x Scalar value to coerce.
#' @param name Character scalar used in error messages.
#' @param default Optional fallback returned when `x` is missing or blank.
#'
#' @return A logical scalar.
#'
#' @noRd
sim_as_logical <- function(x, name = "value", default = NULL) {
  if (is.null(x) || length(x) == 0L || is.na(x[[1]]) ||
      !nzchar(trimws(as.character(x[[1]])))) {
    if (!is.null(default)) return(default)
    stop("`", name, "` must be TRUE or FALSE.", call. = FALSE)
  }

  if (is.logical(x)) return(isTRUE(x[[1]]))

  value_chr <- tolower(trimws(as.character(x[[1]])))
  if (value_chr %in% c("true", "t", "1", "yes", "y")) return(TRUE)
  if (value_chr %in% c("false", "f", "0", "no", "n")) return(FALSE)

  stop("`", name, "` must be TRUE or FALSE.", call. = FALSE)
}

#' Validate a positive integer scalar
#'
#' Coerces an input to integer and verifies that it is at least one.
#'
#' @param x Scalar value to validate.
#' @param name Character scalar used in error messages.
#'
#' @return A positive integer scalar.
#'
#' @noRd
sim_validate_positive_integer <- function(x, name = "value") {
  x <- sim_as_integer(x, name = name)
  if (x < 1L) stop("`", name, "` must be a positive integer.", call. = FALSE)
  x
}

#' Round and clip numeric values to integer bounds
#'
#' Applies base R `round()` to `x`, clips the result to inclusive integer bounds,
#' and returns an integer vector.
#'
#' @param x Numeric vector to round and clip.
#' @param lower Integer-like lower bound.
#' @param upper Integer-like upper bound.
#'
#' @return Integer vector with values in `[lower, upper]`.
#'
#' @noRd
sim_clip_round <- function(x, lower, upper) {
  lower <- sim_as_integer(lower, name = "lower")
  upper <- sim_as_integer(upper, name = "upper")
  if (lower > upper) stop("`lower` must be <= `upper`.", call. = FALSE)

  as.integer(pmin(upper, pmax(lower, round(x))))
}

#' Ensure that an output file's parent directory exists
#'
#' Creates the parent directory of `path` recursively when needed.
#'
#' @param path Character scalar file path.
#'
#' @return Invisibly returns `TRUE`.
#'
#' @noRd
sim_ensure_parent_dir <- function(path) {
  path <- as.character(path)[[1]]
  if (!nzchar(path)) stop("`path` must be non-empty.", call. = FALSE)

  parent <- dirname(path)
  if (!nzchar(parent) || parent == ".") return(invisible(TRUE))

  dir.create(parent, recursive = TRUE, showWarnings = FALSE)
  if (!dir.exists(parent)) {
    stop("Could not create output directory: ", parent, call. = FALSE)
  }

  invisible(TRUE)
}

#' Validate an adjacency matrix
#'
#' Checks that an adjacency matrix is square, has identical non-empty row and
#' column node names, and contains only binary edge indicators.
#'
#' @param adj_mat Matrix with parent nodes in rows and child nodes in columns.
#'
#' @return The validated adjacency matrix as an integer matrix.
#'
#' @noRd
sim_validate_adj_mat <- function(adj_mat) {
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

  out <- matrix(as.integer(adj_mat), nrow = nrow(adj_mat), ncol = ncol(adj_mat),
                dimnames = dimnames(adj_mat))
  out
}

#' Get root nodes from an adjacency matrix
#'
#' Root nodes are nodes with no incoming edges.
#'
#' @param adj_mat Matrix with parent nodes in rows and child nodes in columns.
#'
#' @return Character vector of root node names.
#'
#' @noRd
sim_root_nodes <- function(adj_mat) {
  adj_mat <- sim_validate_adj_mat(adj_mat)
  colnames(adj_mat)[colSums(adj_mat != 0L) == 0L]
}

#' Compute a topological node order
#'
#' Computes a deterministic topological ordering from an adjacency matrix with
#' parent rows and child columns.
#'
#' @param adj_mat Matrix with parent nodes in rows and child nodes in columns.
#'
#' @return Character vector of node names in topological order.
#'
#' @noRd
sim_topological_order <- function(adj_mat) {
  adj_mat <- sim_validate_adj_mat(adj_mat)

  nodes <- colnames(adj_mat)
  incoming <- colSums(adj_mat != 0L)
  queue <- nodes[incoming == 0L]
  node_order <- character()

  while (length(queue) > 0L) {
    node <- queue[[1]]
    queue <- queue[-1]
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

#' Validate SCM noise generators
#'
#' Checks that every graph node has a named noise-generator function and that the
#' noise specification does not contain unknown node names.
#'
#' @param noise_spec Named list of noise-generator functions.
#' @param nodes Character vector of graph node names.
#'
#' @return Invisibly returns `TRUE`.
#'
#' @noRd
sim_validate_noise_spec <- function(noise_spec, nodes) {
  if (!is.list(noise_spec) || is.null(names(noise_spec))) {
    stop("`noise_spec` must be a named list of functions.", call. = FALSE)
  }

  spec_names <- names(noise_spec)
  if (any(is.na(spec_names)) || any(!nzchar(spec_names))) {
    stop("`noise_spec` names must be non-empty.", call. = FALSE)
  }
  if (anyDuplicated(spec_names)) {
    stop("`noise_spec` names must be unique.", call. = FALSE)
  }

  missing_nodes <- setdiff(nodes, spec_names)
  unknown_nodes <- setdiff(spec_names, nodes)
  if (length(missing_nodes) > 0L) {
    stop("`noise_spec` is missing generators for: ",
         paste(missing_nodes, collapse = ", "), call. = FALSE)
  }
  if (length(unknown_nodes) > 0L) {
    stop("`noise_spec` contains unknown nodes: ",
         paste(unknown_nodes, collapse = ", "), call. = FALSE)
  }

  not_fun <- spec_names[!vapply(noise_spec[spec_names], is.function, logical(1))]
  if (length(not_fun) > 0L) {
    stop("`noise_spec` entries must be functions. Invalid entries: ",
         paste(not_fun, collapse = ", "), call. = FALSE)
  }

  invisible(TRUE)
}

#' Validate structural SCM functions
#'
#' Checks that every non-root node has a structural function and that the SCM
#' function list does not contain unknown or root-node entries.
#'
#' @param scm Named list of structural functions.
#' @param nodes Character vector of graph node names.
#' @param root_nodes Character vector of root node names.
#'
#' @return Invisibly returns `TRUE`.
#'
#' @noRd
sim_validate_scm <- function(scm, nodes, root_nodes) {
  if (!is.list(scm)) {
    stop("`scm` must be a named list of functions.", call. = FALSE)
  }

  if (length(scm) == 0L) {
    scm_names <- character()
  } else {
    if (is.null(names(scm))) {
      stop("`scm` must be a named list of functions.", call. = FALSE)
    }
    scm_names <- names(scm)
    if (any(is.na(scm_names)) || any(!nzchar(scm_names))) {
      stop("`scm` names must be non-empty.", call. = FALSE)
    }
    if (anyDuplicated(scm_names)) {
      stop("`scm` names must be unique.", call. = FALSE)
    }
  }

  non_root_nodes <- setdiff(nodes, root_nodes)
  missing_nodes <- setdiff(non_root_nodes, scm_names)
  unknown_nodes <- setdiff(scm_names, nodes)
  root_entries <- intersect(scm_names, root_nodes)
  if (length(missing_nodes) > 0L) {
    stop("`scm` is missing structural functions for: ",
         paste(missing_nodes, collapse = ", "), call. = FALSE)
  }
  if (length(unknown_nodes) > 0L) {
    stop("`scm` contains unknown nodes: ",
         paste(unknown_nodes, collapse = ", "), call. = FALSE)
  }
  if (length(root_entries) > 0L) {
    stop("`scm` should not contain root-node functions: ",
         paste(root_entries, collapse = ", "), call. = FALSE)
  }

  not_fun <- scm_names[!vapply(scm[scm_names], is.function, logical(1))]
  if (length(not_fun) > 0L) {
    stop("`scm` entries must be functions. Invalid entries: ",
         paste(not_fun, collapse = ", "), call. = FALSE)
  }

  invisible(TRUE)
}

#' Validate a generated simulation vector
#'
#' Checks that a generated root, noise, or structural output is a vector-like
#' object with exactly `n` entries.
#'
#' @param x Generated vector-like object.
#' @param node Character scalar node name used in error messages.
#' @param n Expected vector length.
#' @param role Character scalar describing the generated object.
#'
#' @return The generated object.
#'
#' @noRd
sim_validate_generated_vector <- function(x, node, n, role = "generated value") {
  if (is.null(x) || length(dim(x)) > 1L || length(x) != n) {
    stop("`", role, "` for node `", node, "` must be a vector of length ",
         n, ".", call. = FALSE)
  }

  x
}

#' Validate intervention node names
#'
#' Checks that requested intervention targets are valid graph nodes.
#'
#' @param intervention_nodes Character vector of intervention node names, or
#'   `NULL`.
#' @param nodes Character vector of graph node names.
#'
#' @return Character vector of validated intervention node names.
#'
#' @noRd
sim_validate_intervention_nodes <- function(intervention_nodes, nodes) {
  if (is.null(intervention_nodes) || length(intervention_nodes) == 0L) {
    return(character())
  }

  intervention_nodes <- as.character(intervention_nodes)
  if (any(is.na(intervention_nodes)) || any(!nzchar(intervention_nodes))) {
    stop("`intervention_nodes` must contain non-empty node names.",
         call. = FALSE)
  }
  if (anyDuplicated(intervention_nodes)) {
    stop("`intervention_nodes` must not contain duplicates.", call. = FALSE)
  }

  unknown_nodes <- setdiff(intervention_nodes, nodes)
  if (length(unknown_nodes) > 0L) {
    stop("Unknown intervention node(s): ",
         paste(unknown_nodes, collapse = ", "), call. = FALSE)
  }

  intervention_nodes
}

#' Validate explicit intervention values
#'
#' Checks that explicit intervention values are named scalar values and that all
#' names are included in the requested intervention nodes.
#'
#' @param intervention_values Named list or named atomic vector of scalar values,
#'   or `NULL`.
#' @param intervention_nodes Character vector of intervention node names.
#'
#' @return Named list of validated scalar intervention values.
#'
#' @noRd
sim_validate_intervention_values <- function(intervention_values,
                                             intervention_nodes) {
  if (is.null(intervention_values)) return(list())
  if (length(intervention_values) == 0L) return(list())

  if (is.atomic(intervention_values) && !is.null(names(intervention_values))) {
    intervention_values <- as.list(intervention_values)
  }
  if (!is.list(intervention_values) || is.null(names(intervention_values))) {
    stop("`intervention_values` must be a named list or named atomic vector.",
         call. = FALSE)
  }

  value_names <- names(intervention_values)
  if (any(is.na(value_names)) || any(!nzchar(value_names))) {
    stop("`intervention_values` names must be non-empty.", call. = FALSE)
  }
  if (anyDuplicated(value_names)) {
    stop("`intervention_values` names must be unique.", call. = FALSE)
  }

  unknown_values <- setdiff(value_names, intervention_nodes)
  if (length(unknown_values) > 0L) {
    stop("`intervention_values` contains value(s) for non-intervened node(s): ",
         paste(unknown_values, collapse = ", "), call. = FALSE)
  }

  bad_values <- value_names[vapply(intervention_values, function(value) {
    is.list(value) || length(dim(value)) > 1L || length(value) != 1L
  }, logical(1))]
  if (length(bad_values) > 0L) {
    stop("`intervention_values` entries must be scalar. Invalid entries: ",
         paste(bad_values, collapse = ", "), call. = FALSE)
  }

  intervention_values
}

#' Compute descendants of graph nodes
#'
#' Finds all downstream descendants of one or more nodes in an adjacency matrix
#' with parent rows and child columns.
#'
#' @param adj_mat Matrix with parent nodes in rows and child nodes in columns.
#' @param intervention_nodes Character vector of starting node names.
#'
#' @return Character vector of descendant node names.
#'
#' @noRd
sim_descendants <- function(adj_mat, intervention_nodes) {
  adj_mat <- sim_validate_adj_mat(adj_mat)
  if (length(intervention_nodes) == 0L) return(character())

  nodes <- colnames(adj_mat)
  frontier <- intervention_nodes
  seen <- character()

  while (length(frontier) > 0L) {
    node <- frontier[[1]]
    frontier <- frontier[-1]

    children <- nodes[adj_mat[node, ] != 0L]
    new_children <- setdiff(children, seen)
    seen <- unique(c(seen, children))
    frontier <- c(frontier, new_children)
  }

  seen
}

#' Compute a deterministic mode
#'
#' Finds the most frequent value in a vector. Ties are resolved by sorting the
#' tied values and taking the smallest.
#'
#' @param x Vector of factual node values.
#'
#' @return Scalar mode value.
#'
#' @noRd
sim_mode_value <- function(x) {
  if (length(x) == 0L) stop("Cannot compute a mode for an empty vector.",
                            call. = FALSE)

  x_non_missing <- x[!is.na(x)]
  if (length(x_non_missing) == 0L) return(x[NA_integer_])

  candidates <- sort(unique(x_non_missing))
  counts <- vapply(candidates, function(value) {
    sum(x_non_missing == value)
  }, integer(1))

  candidates[which.max(counts)]
}

#' Complete intervention values
#'
#' Fills missing explicit intervention values with the factual mode of each
#' intervened node.
#'
#' @param intervention_nodes Character vector of intervention node names.
#' @param intervention_values Named list of explicit scalar intervention values.
#' @param factual_data Named list of factual node vectors.
#'
#' @return Named list with one scalar value for every intervention node.
#'
#' @noRd
sim_complete_intervention_values <- function(intervention_nodes,
                                             intervention_values,
                                             factual_data) {
  values <- vector("list", length(intervention_nodes))
  names(values) <- intervention_nodes

  for (node in intervention_nodes) {
    if (node %in% names(intervention_values)) {
      values[[node]] <- intervention_values[[node]]
    } else {
      values[[node]] <- sim_mode_value(factual_data[[node]])
    }
  }

  values
}

#' Generate factual node values
#'
#' Evaluates an SCM in topological order from fixed exogenous noise.
#'
#' @param noise Named list of exogenous noise vectors.
#' @param node_order Character vector of nodes in topological order.
#' @param root_nodes Character vector of root node names.
#' @param scm Named list of structural functions for non-root nodes.
#' @param n Expected number of observations.
#'
#' @return Named list of generated node vectors.
#'
#' @noRd
sim_generate_factual_nodes <- function(noise, node_order, root_nodes, scm, n) {
  data <- list()
  for (node in node_order) {
    if (node %in% root_nodes) {
      data[[node]] <- noise[[node]]
    } else {
      out <- scm[[node]](data = data, noise = noise, n = n)
      data[[node]] <- sim_validate_generated_vector(out, node = node, n = n,
                                                    role = "structural output")
    }
  }

  data
}

#' Generate counterfactual node values
#'
#' Applies simultaneous interventions to factual data, keeps exogenous noise
#' fixed, and recomputes downstream descendants in topological order.
#'
#' @param factual_data Named list of factual node vectors.
#' @param noise Named list of exogenous noise vectors.
#' @param adj_mat Matrix with parent nodes in rows and child nodes in columns.
#' @param node_order Character vector of nodes in topological order.
#' @param scm Named list of structural functions for non-root nodes.
#' @param intervention_nodes Character vector of intervention node names.
#' @param intervention_values Named list of scalar intervention values.
#' @param n Expected number of observations.
#'
#' @return Named list of counterfactual node vectors.
#'
#' @noRd
sim_generate_counterfactual_nodes <- function(factual_data,
                                              noise,
                                              adj_mat,
                                              node_order,
                                              scm,
                                              intervention_nodes,
                                              intervention_values,
                                              n) {
  counterfactual_data <- factual_data
  if (length(intervention_nodes) == 0L) return(counterfactual_data)

  for (node in intervention_nodes) {
    counterfactual_data[[node]] <- rep(intervention_values[[node]], n)
  }

  descendants <- sim_descendants(adj_mat, intervention_nodes)
  recompute_nodes <- node_order[node_order %in% setdiff(descendants,
                                                        intervention_nodes)]

  for (node in recompute_nodes) {
    out <- scm[[node]](data = counterfactual_data, noise = noise, n = n)
    counterfactual_data[[node]] <- sim_validate_generated_vector(
      out,
      node = node,
      n = n,
      role = "counterfactual structural output"
    )
  }

  counterfactual_data
}

#' Format generated SCM data
#'
#' Converts named node and noise lists into a data frame with optional `id` and
#' `eps_<node>` columns.
#'
#' @param data Named list of node vectors.
#' @param noise Named list of exogenous noise vectors.
#' @param nodes Character vector of node names in output order.
#' @param include_noise Logical. Whether to include exogenous noise columns.
#' @param include_id Logical. Whether to include an `id` column.
#' @param n Number of observations.
#'
#' @return A `data.frame`.
#'
#' @noRd
sim_format_dataset <- function(data,
                               noise,
                               nodes,
                               include_noise,
                               include_id,
                               n) {
  sim_df <- as.data.frame(data[nodes], optional = FALSE)

  if (include_id) {
    sim_df <- data.frame(id = seq_len(n), sim_df)
  }

  if (include_noise) {
    noise_df <- as.data.frame(noise[nodes], optional = FALSE)
    names(noise_df) <- paste0("eps_", nodes)
    sim_df <- cbind(sim_df, noise_df)
  }

  sim_df
}
