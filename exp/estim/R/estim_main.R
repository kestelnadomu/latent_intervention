# Counterfactually fair estimation runner -------------------------------------
#
# This script exposes reusable estimation functions. When run directly from the
# repository root, it also fits all three methods to a small sample from the
# project's default SCM and reports predictive error and counterfactual shift.

# Set project working directory ------------------------------------------------
wd <- getwd()

# Load estimation utilities ---------------------------------------------------
source(file.path(wd, "exp", "estim", "R", "utils", "libraries.R"))
source(file.path(wd, "exp", "estim", "R", "utils", "estim_func.R"))
source(file.path(wd, "exp", "estim", "R", "utils", "estim_methods.R"))
source(file.path(wd, "exp", "estim", "R", "utils", "estim_data.R"))

# Run a small end-to-end example only under Rscript ---------------------------
if (sys.nframe() == 0L) {
  source(file.path(wd, "exp", "sim", "R", "utils", "sim_func.R"))
  source(file.path(wd, "exp", "sim", "R", "utils", "sim_scm.R"))
  source(file.path(wd, "exp", "sim", "R", "utils", "sim_data.R"))

  example_data <- sim_data(
    n = 800L,
    seed = 42L,
    intervention_nodes = "G",
    intervention_values = list(G = 1L),
    include_noise = FALSE
  )
  train_rows <- seq_len(600L)
  test_rows <- 601:800
  train_data <- example_data$factual[train_rows, , drop = FALSE]
  test_factual <- example_data$factual[test_rows, , drop = FALSE]
  test_counterfactual <- example_data$counterfactual[test_rows, , drop = FALSE]
  adj_mat <- sim_default_adj_mat()

  methods <- c("level_one", "fair_add", "fair_k")
  example_models <- setNames(vector("list", length(methods)), methods)
  summary_rows <- setNames(vector("list", length(methods)), methods)
  for (method in methods) {
    model <- estim_fit(
      data = train_data,
      adj_mat = adj_mat,
      pa = "G",
      target = "Q",
      method = method
    )
    example_models[[method]] <- model
    pred_factual <- predict(model, test_factual)
    pred_counterfactual <- predict(model, test_counterfactual)

    summary_rows[[method]] <- data.frame(
      method = method,
      rmse = sqrt(mean((pred_factual - test_factual$Q)^2)),
      mean_absolute_counterfactual_shift = mean(abs(
        pred_factual - pred_counterfactual
      )),
      row.names = NULL
    )
  }
  example_summary <- do.call(rbind, summary_rows)
  rownames(example_summary) <- NULL

  print(example_summary, row.names = FALSE)
  invisible(list(models = example_models, summary = example_summary))
}
