# SCM simulation runner --------------------------------------------------------
#
# This script generates factual and counterfactual data from the configured SCM
# and optionally writes both datasets to CSV files. Supporting inputs and helper
# functions are defined in the scripts stored under `src/R/utils/`.

# Set project working directory ------------------------------------------------
wd <- getwd()

# Load simulation utilities ----------------------------------------------------
source(file.path(wd, "src", "R", "utils", "libraries.R"))
source(file.path(wd, "src", "R", "utils", "sim_func.R"))
source(file.path(wd, "src", "R", "utils", "sim_scm.R"))
source(file.path(wd, "src", "R", "utils", "sim_data.R"))

# Define and parse runtime configuration --------------------------------------
sim_defaults <- list(
  n = 1000L,
  seed = NULL,
  save = TRUE,
  out_path = file.path("data", "sim", "sim_data.csv"),
  intervention_node = "G",
  intervention_value = 1L,
  include_noise = TRUE,
  include_id = FALSE
)
sim_args <- sim_parse_cli_args(defaults = sim_defaults)

# Validate runtime configuration ----------------------------------------------
n <- sim_validate_positive_integer(sim_args$n, name = "n")
seed <- sim_as_nullable_integer(sim_args$seed, name = "seed")
save <- sim_as_logical(sim_args$save, name = "save")
out_path <- as.character(sim_args$out_path)[[1]]
intervention_node <- as.character(sim_args$intervention_node)[[1]]
intervention_value <- sim_as_integer(sim_args$intervention_value,
                                     name = "intervention_value")
include_noise <- sim_as_logical(sim_args$include_noise, name = "include_noise")
include_id <- sim_as_logical(sim_args$include_id, name = "include_id")

if (!nzchar(out_path)) stop("`out_path` must be non-empty.", call. = FALSE)
if (!isTRUE(grepl("^([A-Za-z]:)?[\\\\/]", out_path))) {
  out_path <- file.path(wd, out_path)
}

# Generate factual and counterfactual data ------------------------------------
sim_df <- sim_data(
  n = n,
  seed = seed,
  intervention_nodes = intervention_node,
  intervention_values = setNames(list(intervention_value), intervention_node),
  include_noise = include_noise,
  include_id = include_id
)

# Save datasets or keep them in memory ----------------------------------------
if (save) {
  out_files <- paste0(tools::file_path_sans_ext(out_path), "_",
                      names(sim_df), ".csv")
  invisible(lapply(seq_along(sim_df), function(i) {
    sim_ensure_parent_dir(out_files[[i]])
    utils::write.csv(sim_df[[i]], file = out_files[[i]], row.names = FALSE)
  }))
  message("Saved SCM simulation data to ",
          paste(normalizePath(out_files, winslash = "/", mustWork = FALSE),
                collapse = " and "))
} else {
  message("Generated SCM simulation result in memory only.")
}

# Return simulation result for interactive use --------------------------------
invisible(sim_df)
