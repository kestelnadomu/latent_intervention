# latent_lab

A small entry point to the project: the read → operate → render pipeline, meant to be driven from one notebook. Start with [example.ipynb](example.ipynb).

| piece | symbol | in latent_lab |
|---|---|---|
| encoder | φ | `Encoder(name, dim)`: `nomic` 64–768, `qwen3` 32–1024, `embeddinggemma` 128–768; `.latents("talent")` loads cached encodings, `.encode(texts)` runs the model |
| reader | ψ | `Reader(columns, hidden=0)`: linear (`hidden=0`) or MLP probe; `cross_fit` reads each fold with readers fitted on the others, `mean_predict` averages them |
| operator | ω_S | any function `states DataFrame -> target states DataFrame`. `LinearSCM(levels, parents, noise)` gives causal ones: `.do(X=3, blocked=None)` for SCM and path-specific counterfactuals, `print(scm)` shows the equations. The example also defines a wrong SCM, an erasure and a rule |
| renderer | ρ | subclasses of `Renderer`: `IdentityRenderer`, `ConstantRenderer`, `GroupOTRenderer` (N2), `StateEffectRenderer` (C-sym; `guard=` for projected C-sym), `FlowRenderer` (F1), `EditorRenderer` (minimal edit trained on random targets s' ~ Uniform(S)), `OTEditorRenderer` (the same with an input-convex gradient map), `OTFlowRenderer` (OT flow matching within each state, shared reference). New ones implement `_fit` and `_render` |
| pipeline | ω_Z | `Pipeline(reader, operator, renderer)` and `evaluate`: gap closed to φ(x'), judge accuracy |

Data are loaded in the notebook with `pd.read_csv`: any table of discrete states (levels
0..k-1) indexed by unit id works.

Notation and the renderer options follow [docs/contribution.md](../docs/contribution.md)
and [docs/experiments/renderer_benchmark.md](../docs/experiments/renderer_benchmark.md).

Notes:

- Cached latents exist for `talent` (`data/latents/talent/<encoder>_<dim>/`) and `talent_motivation` (`embeddinggemma_768` only). A width without its own cache is projected from the full-width one (Matryoshka prefix, renormalized), e.g. `Encoder("embeddinggemma", 256).latents("talent_motivation")`.
- In pandas, use `df["T"]` for the talent column: `df.T` is the transpose.
- Qwen3 and EmbeddingGemma need their isolated environments to encode live; their
  cached latents load in the main environment.
- To run the notebook: `uv run --with ipykernel jupyter lab` (or pick `.venv` as the
  kernel in VS Code).
- `src/` remains the full library (benchmarks, all nine h_Z families, artifact checks);
  `latent_lab` reuses its encoders and semantic decoder.
