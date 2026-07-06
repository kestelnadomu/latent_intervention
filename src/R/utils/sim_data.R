#' Simulate data from a discrete SCM
#'
#' Generates exogenous noise and observed variables from an SCM defined by an
#' adjacency matrix, a named list of structural functions, and a named list of
#' noise-generator functions.
#'
#' By default, this function uses the latent intervention SCM:
#' \deqn{R = \epsilon_R,}
#' \deqn{G = \epsilon_G,}
#' \deqn{A = \epsilon_A,}
#' \deqn{E = \min(3, \max(0, \mathrm{round}(0.4(R + A + G) + \epsilon_E))),}
#' \deqn{S = \min(2, \max(0, \mathrm{round}(0.45E + 0.25A + \epsilon_S))),}
#' \deqn{W = \min(2, \max(0, \mathrm{round}(0.5A + 0.3E + \epsilon_W))),}
#' \deqn{V = \min(1, \max(0, \mathrm{round}(0.2E + 0.3S + \epsilon_V))),}
#' \deqn{C = \min(1, \max(0, \mathrm{round}(0.15(E + W) + \epsilon_C))),}
#' \deqn{Q = \min(2, \max(0, \mathrm{round}(0.3(E + V + C + W) + \epsilon_Q))).}
#'
#' @param n Positive integer. Number of observations to simulate.
#' @param seed Optional integer seed for reproducible simulation. Use `NULL` to
#'   leave the current RNG state unchanged.
#' @param adj_mat Square adjacency matrix with parent nodes in rows and child
#'   nodes in columns. Row and column names define the output node order.
#' @param scm Named list of structural functions for non-root nodes. Each
#'   function must have the contract `function(data, noise, n)` and return one
#'   vector of length `n`.
#' @param noise_spec Named list of exogenous noise-generator functions for all
#'   nodes. Each function must accept `n` and return one vector of length `n`.
#' @param intervention_nodes Optional character vector of node names to
#'   intervene on. Interventions are simultaneous `do()` interventions.
#' @param intervention_values Optional named list or named atomic vector of
#'   scalar intervention values. Any node in `intervention_nodes` without an
#'   explicit value is set to the factual mode of that node; ties use the
#'   smallest tied mode after sorting.
#' @param include_noise Logical. If `TRUE`, include the exogenous/noise columns
#'   named `eps_<node>`.
#' @param include_id Logical. If `TRUE`, include an integer row identifier column
#'   named `id`.
#'
#' @return A named list with two `data.frame`s: `factual` and `counterfactual`.
#'   Each data frame contains observed SCM variables in adjacency node order and
#'   optionally an `id` column and exogenous/noise columns.
#'
#' @export
#'
#' @examples
#' result <- sim_data(n = 10, seed = 1)
#' names(result)
#'
#' sim_data(n = 10, seed = 1, intervention_nodes = "A")
#' sim_data(n = 10, seed = 1, intervention_nodes = "A",
#'          intervention_values = list(A = 0))
#'
#' adj_mat <- matrix(c(0, 1, 0, 0), nrow = 2, byrow = TRUE,
#'                   dimnames = list(c("X", "Y"), c("X", "Y")))
#' noise_spec <- list(X = function(n) seq_len(n), Y = function(n) rep(1, n))
#' scm <- list(Y = function(data, noise, n) data$X + noise$Y)
#' sim_data(n = 3, adj_mat = adj_mat, scm = scm, noise_spec = noise_spec,
#'          intervention_nodes = "X", intervention_values = list(X = 0))
sim_data <- function(n = 1000L,
                     seed = NULL,
                     adj_mat = sim_default_adj_mat(),
                     scm = sim_default_scm(),
                     noise_spec = sim_default_noise_spec(),
                     intervention_nodes = NULL,
                     intervention_values = NULL,
                     include_noise = TRUE,
                     include_id = FALSE) {
  n <- sim_validate_positive_integer(n, name = "n")
  seed <- sim_as_nullable_integer(seed, name = "seed")
  adj_mat <- sim_validate_adj_mat(adj_mat)
  include_noise <- sim_as_logical(include_noise, name = "include_noise")
  include_id <- sim_as_logical(include_id, name = "include_id")

  nodes <- colnames(adj_mat)
  root_nodes <- sim_root_nodes(adj_mat)
  node_order <- sim_topological_order(adj_mat)
  sim_validate_noise_spec(noise_spec, nodes)
  sim_validate_scm(scm, nodes, root_nodes)
  intervention_nodes <- sim_validate_intervention_nodes(intervention_nodes,
                                                        nodes)
  intervention_values <- sim_validate_intervention_values(
    intervention_values,
    intervention_nodes
  )

  if (!is.null(seed)) set.seed(seed)

  noise <- lapply(nodes, function(node) {
    out <- noise_spec[[node]](n = n)
    sim_validate_generated_vector(out, node = node, n = n, role = "noise")
  })
  names(noise) <- nodes

  factual_data <- sim_generate_factual_nodes(
    noise = noise,
    node_order = node_order,
    root_nodes = root_nodes,
    scm = scm,
    n = n
  )
  intervention_values <- sim_complete_intervention_values(
    intervention_nodes = intervention_nodes,
    intervention_values = intervention_values,
    factual_data = factual_data
  )
  counterfactual_data <- sim_generate_counterfactual_nodes(
    factual_data = factual_data,
    noise = noise,
    adj_mat = adj_mat,
    node_order = node_order,
    scm = scm,
    intervention_nodes = intervention_nodes,
    intervention_values = intervention_values,
    n = n
  )

  list(
    factual = sim_format_dataset(
      data = factual_data,
      noise = noise,
      nodes = nodes,
      include_noise = include_noise,
      include_id = include_id,
      n = n
    ),
    counterfactual = sim_format_dataset(
      data = counterfactual_data,
      noise = noise,
      nodes = nodes,
      include_noise = include_noise,
      include_id = include_id,
      n = n
    )
  )
}

