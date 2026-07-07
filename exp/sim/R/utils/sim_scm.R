#' Return the default SCM adjacency matrix
#'
#' Builds the default graph for the latent intervention simulation. Rows are
#' parent nodes and columns are child nodes.
#'
#' @return A square integer adjacency matrix.
#'
#' @export
#'
#' @examples
#' sim_default_adj_mat()
sim_default_adj_mat <- function() {
  nodes <- c("R", "G", "A", "E", "S", "W", "V", "C", "Q")
  adj_mat <- matrix(0L, nrow = length(nodes), ncol = length(nodes),
                    dimnames = list(nodes, nodes))

  adj_mat["R", "E"] <- 1L
  adj_mat["G", "E"] <- 1L
  adj_mat["A", "E"] <- 1L
  adj_mat["E", "S"] <- 1L
  adj_mat["A", "S"] <- 1L
  adj_mat["A", "W"] <- 1L
  adj_mat["E", "W"] <- 1L
  adj_mat["E", "V"] <- 1L
  adj_mat["S", "V"] <- 1L
  adj_mat["E", "C"] <- 1L
  adj_mat["W", "C"] <- 1L
  adj_mat["E", "Q"] <- 1L
  adj_mat["V", "Q"] <- 1L
  adj_mat["C", "Q"] <- 1L
  adj_mat["W", "Q"] <- 1L

  adj_mat
}

#' Return the default SCM noise specification
#'
#' Builds named exogenous noise-generator functions for every node in the
#' default SCM. Each function accepts `n` and returns one vector of length `n`.
#'
#' @return A named list of noise-generator functions.
#'
#' @export
#'
#' @examples
#' noise_spec <- sim_default_noise_spec()
#' noise_spec$R(5)
sim_default_noise_spec <- function() {
  list(
    R = function(n) sample(0:3, size = n, replace = TRUE),
    G = function(n) sample(0:1, size = n, replace = TRUE),
    A = function(n) sample(0:2, size = n, replace = TRUE,
                           prob = c(0.25, 0.50, 0.25)),
    E = function(n) stats::rnorm(n, mean = 0.35, sd = 0.50),
    S = function(n) stats::rnorm(n, mean = 0.25, sd = 0.35),
    W = function(n) stats::rnorm(n, mean = 0.00, sd = 0.50),
    V = function(n) stats::rnorm(n, mean = -0.35, sd = 0.20),
    C = function(n) stats::rnorm(n, mean = 0.00, sd = 0.30),
    Q = function(n) stats::rnorm(n, mean = 0.00, sd = 0.30)
  )
}

#' Return the default SCM structural functions
#'
#' Builds named structural functions for every non-root node in the default SCM.
#' Each function follows the contract `function(data, noise, n)` and returns one
#' vector of length `n`.
#'
#' @return A named list of structural functions.
#'
#' @export
#'
#' @examples
#' sim_default_scm()
sim_default_scm <- function() {
  list(
    E = function(data, noise, n) {
      sim_clip_round(0.4 * (data$R + data$A + data$G) + noise$E,
                     lower = 0L, upper = 3L)
    },
    S = function(data, noise, n) {
      sim_clip_round(0.45 * data$E + 0.25 * data$A + noise$S,
                     lower = 0L, upper = 2L)
    },
    W = function(data, noise, n) {
      sim_clip_round(0.5 * data$A + 0.3 * data$E + noise$W,
                     lower = 0L, upper = 2L)
    },
    V = function(data, noise, n) {
      sim_clip_round(0.2 * data$E + 0.3 * data$S + noise$V,
                     lower = 0L, upper = 1L)
    },
    C = function(data, noise, n) {
      sim_clip_round(0.15 * (data$E + data$W) + noise$C,
                     lower = 0L, upper = 1L)
    },
    Q = function(data, noise, n) {
      sim_clip_round(0.3 * (data$E + data$V + data$C + data$W) + noise$Q,
                     lower = 0L, upper = 2L)
    }
  )
}
