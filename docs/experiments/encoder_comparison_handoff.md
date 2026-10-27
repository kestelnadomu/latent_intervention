# Handoff: encoder comparison on the extended corpus

**Goal.** Rerun the encoder comparison of the paper's appendix on the extended-DAG corpus
(5,700 texts in `data/text_talent_motivation/`):

1. Encode the corpus with three frozen encoders at full width. No encoder is trained.
2. Fit the readers ψ and the judge ψ_eval on all 17 latent spaces with `experiment.ipynb`.

Branch `package-encoder`. Run one job at a time. Do not change code or configs: the
encoding run signatures hash them, and the reader cache keys hash `latent_lab/`. Do not
re-encode EmbeddingGemma; its encodings are in git. Fit nothing beyond what is listed here.

| Encoder | Full width | Latent spaces (Matryoshka prefixes) | Encoding run ID |
|---|---:|---|---|
| Nomic Embed v1.5 | 768 | 64, 128, 256, 512, 768 | `nomic-768-v1` |
| Qwen3-Embedding-0.6B | 1024 | 32, 64, 128, 256, 512, 768, 1024 | `qwen3-1024-v1` |
| LangVAE (stock) | 128 | 128 | `langvae-128-v1` |
| EmbeddingGemma-300m | 768 | 128, 256, 512, 768 | already done |

## 0. Setup

- **Platform.** Linux with `tmux` and `uv`; the encoding queue needs `fcntl`.
- **Python 3.12** (see `.python-version`). The cache keys hash bytecode, so another
  version misses the cache.
- **Line endings.** Check out with LF (`git config core.autocrlf` must not be `true`).
  The cache keys hash the raw bytes of `latent_lab/*.py`.
- **`.venv`:** `uv sync --extra dev`. It runs Nomic, LangVAE, the validation and the
  notebook.
- **`.venv-qwen3`:** `bash scripts/setup_qwen3.sh`. It installs the locked dependencies
  and fetches the pinned model.
- **Nomic and LangVAE weights.** The launcher runs offline, so fetch them once online:

  ```bash
  for e in nomic langvae; do .venv/bin/python -c "
  from exp.encoding.talent_motivation import load_encoding_config
  from src.encoder import get_encoder_factory
  c = load_encoding_config('$e')['encoder']
  print('$e', get_encoder_factory(c)(c).encode(['test'], deterministic=True, batch_size=1).shape)"; done
  ```

- **Tests.** `.venv/bin/python -m pytest tests/test_talent_motivation_encoding.py -q`
  should pass except `test_only_published_encoding_artifacts_are_trackable`. That failure
  is pre-existing and comes from `data/latents/**` being unignored.

## 1. Encode (about 2 h)

```bash
for e in nomic qwen3 langvae; do
    bash scripts/run_talent_motivation_encoding.sh "$e"
    while tmux has-session -t =encode-talent-motivation 2>/dev/null; do sleep 60; done
done
```

The launcher validates the inputs, runs a preflight and starts the encoding in the
detached tmux session `encode-talent-motivation`. Rough durations: Nomic about 20 min,
Qwen3 about 1–1.5 h, LangVAE about 15 min.

**How to watch it.** `tail -f reports/talent_motivation/encoding/<run-id>/run.log`.

**Done when** each `reports/talent_motivation/encoding/<run-id>/status.json` reports
`complete` and these three folders hold a `manifest.json`:

- `data/latents/talent_motivation/nomic_768/`
- `data/latents/talent_motivation/qwen3_1024/`
- `data/latents/talent_motivation/langvae/`

**After an interruption** rerun the same command; verified artifacts are reused. Context
for the encodings: `docs/experiments/talent_motivation_encoding.md`.

## 2. Fit the readers (about 1–1.5 h)

Start this only after all three encodings are complete. The notebook runs headless with
the plain interpreter, with no Jupyter needed. Its functions must live in `__main__`,
otherwise the cache keys differ.

```bash
cat > /tmp/run_notebook.py <<'EOF'
import json
ns = {"__name__": "__main__"}
for i, cell in enumerate(json.load(open("experiment.ipynb"))["cells"]):
    if cell["cell_type"] == "code":
        print(f"--- cell {i} ({cell.get('id')})", flush=True)
        exec(compile("".join(cell["source"]), f"cell-{i}", "exec"), ns)
EOF
tmux new-session -d -s latent-lab-readers \
    "PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -u /tmp/run_notebook.py > /tmp/latent_lab_readers.log 2>&1"
```

**What gets fitted.** Per latent space: ψ (5 seeds × 5 folds) and the judge (5 folds),
30 reader fits each. The 13 new latent spaces add 390 fits.

**What loads from the cache** (`models/latent_lab/`):

- EmbeddingGemma-768 and everything after Section 3: search, renderers ρ, π and results.
  These take seconds.
- EmbeddingGemma 128/256/512 may refit (90 fits). Their projected latents can differ in
  the last bit between CPUs.

**Stop and report** if `psi embeddinggemma-768` takes minutes per seed instead of
seconds. That means the cache keys don't match, and the whole pipeline would be refit,
which takes about 3.5 h and changes the paper's main results.

**Output.** A new `reports/latent_lab/run-<timestamp>.json`.

**Check** that the main results are unchanged:

```bash
.venv/bin/python - <<'EOF'
import glob, json
load = lambda p: json.load(open(p, encoding="utf-8"))
old, new = load("reports/latent_lab/run-20261027-075057.json"), load(sorted(glob.glob("reports/latent_lab/run-*.json"))[-1])
print("results", "identical" if json.dumps(old["results"]) == json.dumps(new["results"]) else "DIFFERENT")
print(sorted({c.split(" | ")[0] for row in new["psi"] for c in row if " | " in c}))
EOF
```

This should print `results identical`, followed by the 17 latent-space keys.

- `results` comes straight from the cache, so it must match bit for bit.
- The metrics recomputed outside the cache can differ in the last digits between machines.
  These are `summary`, `outcome`, and `realism` in `search` and `energy` in `wrong_dag`.
  That is expected.

## 3. Deliver

Check `git status` first. Never add `.pending` directories, `.lock` files or `run.log`.

```bash
git add data/latents/talent_motivation/{nomic_768,qwen3_1024,langvae} \
    reports/talent_motivation/encoding/{nomic-768-v1,qwen3-1024-v1,langvae-128-v1}/*.json \
    reports/latent_lab/run-*.json models/latent_lab/*.pt
git commit -m "Encoder comparison on the extended corpus: encodings and readers"
git push origin package-encoder
```

**Report back:**

- the runtimes
- any errors
- the output of the check above
- the joint-accuracy rows of the `psi` table in the new run JSON, per latent space for ψ
  and the judge
