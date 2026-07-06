# R Simulation Runner

This folder contains the R entry point for generating simulated SCM data. The
main script is `sim_main.R`; supporting libraries, helper functions, SCM
definitions, and the data-generation function are stored in `src/R/utils/`.

## Before Running

This project uses `renv` for the R environment. From a fresh checkout, restore
the locked R environment once from the repository root:

```bash
Rscript -e "renv::restore(prompt = FALSE)"
```

After that, regular `Rscript` calls from the repository root automatically use
the project `renv` environment through `.Rprofile`. The Python environment is
managed separately and is not needed for the R simulation runner.

## Run From The Console

Run all commands from the repository root:

```bash
Rscript src/R/sim_main.R
```

By default, this generates factual and counterfactual datasets with `n = 1000`.
The default counterfactual intervention is `do(G = 1)`, where `G` is the gender
node and `1` represents male. The script writes three CSV files:

```text
data/sim/sim_data_factual.csv
data/sim/sim_data_counterfactual.csv
data/sim/sim_data_espilon.csv
```

Generated simulation outputs under `data/sim/` are ignored by Git.

## Runtime Options

The script accepts simple `key=value` arguments:

```bash
Rscript src/R/sim_main.R n=5000 seed=123 save=TRUE intervention_node=G intervention_value=1 include_noise=TRUE include_id=FALSE
```

Available options:

- `n`: number of observations to simulate. Default: `1000`.
- `seed`: optional integer seed. Default: `NULL`.
- `save`: whether to write CSV files. Default: `TRUE`.
- `out_path`: base CSV path used to derive output filenames. Default:
  `data/sim/sim_data.csv`.
- `intervention_node`: node intervened on in the counterfactual data. Default:
  `G`.
- `intervention_value`: scalar value assigned to `intervention_node`. Default:
  `1`.
- `include_noise`: whether to write/return a separate `espilon` dataset with
  `eps_<node>` columns. Default: `TRUE`.
- `include_id`: whether to include an `id` column. Default: `FALSE`.

If `out_path=data/sim/example.csv`, the script writes:

```text
data/sim/example_factual.csv
data/sim/example_counterfactual.csv
data/sim/example_espilon.csv
```

## In-Memory Use

To run the same simulation from an interactive R session:

```r
source("src/R/sim_main.R")
```

The script invisibly returns `sim_df`, a named list with:

```r
sim_df$factual
sim_df$counterfactual
sim_df$espilon
```

For custom adjacency matrices, SCM functions, interventions, or noise
generators, call `sim_data()` directly after sourcing the utility scripts.
