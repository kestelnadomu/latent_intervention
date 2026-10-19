# Talent-motivation CV encodings

Encode the active extended-DAG corpus with the validation-selected frozen
`embeddinggemma_768` recipe from `g-all-encoders-v1`. The model revision, full
SentenceTransformer stack, classification prefix, float32 inference and L2
normalization are unchanged. No encoder is fitted. Archived `pilot_v0` CVs are excluded.

Configuration: `configs/encoding_talent_motivation.yaml`. Implementation:
`exp/encoding/talent_motivation.py`, reusing `src/pair_encoding.py` unchanged;
`exp/encoding/text_artifact.py` handles the two auxiliary evaluation-only sets.

| Input in `data/text_talent_motivation/` | Rows | Output |
|---|---:|---|
| `cv_factual.csv` | 4,000 | `z_pairs.pt`: `ids`, `z` |
| `cv_counterfactual.csv` | 1,000 | `z_pairs.pt`: `test_ids`, `z_prime`, `is_identity` |
| `cv_floor.csv` | 200 | `z_floor.pt`: `ids`, `z`, `text_variant` |
| `cv_factual_vb.csv` | 500 | `z_factual_vb.pt`: `ids`, `z`, `text_variant` |

All 5,700 texts are encoded. Same-state floor/wording variants are **not** copied
from factual embeddings. IDs are corpus-local, not interchangeable with the original
`talent` experiment. Every output is accompanied by a checksum/provenance `.info.json`.

The complete bundle is published to
`data/latents/talent_motivation/embeddinggemma_768/`, with a `manifest.json` recording
input/config/source/model hashes, protocol and output checksums. The manifest binds
the embeddings to `partition.csv`, `interventions.csv`, the simulation records,
codebooks and generation journals. The counterfactual action is **per-unit** `do(X=k)`;
it must not be replaced by the config's legacy reference `do(X=3)`.

Use `partition.csv`: 2,800 pool / 200 dev / 1,000 evaluation. Counterfactual, floor
and held-out-wording embeddings are evaluation-only. Encoding these texts through
the frozen encoder is not permission to use them for fitting or model selection.
Adapting downstream models/splits is a separate task.

## Launch and resume

```bash
bash scripts/run_talent_motivation_encoding.sh embeddinggemma-768-v1
```

The launcher validates the simulation in `.venv`, then checks tokenizer lengths and
bounded inference in `.venv-embeddinggemma`. These environments stay separate; no
dependencies are added. It launches `encode-talent-motivation-768` in detached tmux
with four numerical threads and batch size eight. Closing the client does not stop it.

```bash
tmux attach -t encode-talent-motivation-768
tail -f reports/talent_motivation/encoding/embeddinggemma-768-v1/run.log
```

Status is in the same directory's `status.json`. Matching completed artifacts are
verified and reused. An interrupted transformer pass restarts that artifact, not
already completed artifacts. Torn tensor/sidecar pairs are retained in hidden
`.interrupted-*` directories. Changes to inputs/config/source are rejected for an
existing run ID; existing published embeddings are never silently replaced.

While running, outputs live in an ignored sibling `.embeddinggemma_768.<run>.pending`
directory. Only after all three artifacts pass validation is the whole directory
renamed into its final location. A filesystem lock excludes concurrent writers.
Published artifacts and compact reports are Git-trackable; staging, locks and logs
remain ignored. Nothing is staged or committed automatically.
