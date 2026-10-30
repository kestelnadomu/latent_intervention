# Handoff: decode the renderers, with sampled and true target states, and phi(x) (renderers-v1)

**Goal.** The package evaluation (package-v1) decoded only the state flow's edits. This run
decodes, with the **768-D Gemma 3 270M decoder** (`768-270m`), on the same 1,000 test units:

1. **The other five renderers' edits** (editor, OT editor, OT flow, state effect, group OT)
   in package-v1's setting: S1, rho trained and tested with psi(z), seed 0, each unit under
   its own do(O = k), target state s' drawn by the operator.
2. **All six renderers (the state flow too) with the true target state:** the same fits and
   psi(z) as source state, but conditioned on the simulation's true s' of each unit
   (`data/sim_talent_motivation/sim_data_counterfactual.csv`) instead of the operator's draw.
   Vector sets `<renderer>-true-target`.
3. **The unedited phi(x)** of the same units (vector set `factual`).

The group OT reads only O, which do(O = k) sets exactly, so its true-target edits equal its
sampled ones. `prepare` detects this and does not decode them twice; `score` reuses the
texts. That leaves 11 vector sets, 11,000 CVs. Nothing is retrained. Score everything against
x', except phi^-1(phi(x)), which is scored against x, with the paper tables' metrics,
BERTScore included. Full procedure: section "The other renderers (renderers-v1)" of
[text_realizer_package_evaluation.md](text_realizer_package_evaluation.md).

## Before running, check on this machine

1. `git switch decoder && git pull`. Needed: `exp/text_realizer_package_eval/renderers.py`,
   `configs/text_realizer_renderers_eval.yaml`, `scripts/run_text_realizer_renderers.sh`.
2. **Inputs present.**
   - `models/talent_motivation/text-realizer-evaluation/package-v1/manifest.json`, the bundle
     of the first run. Its `checkpoints/768-270m.pt` and `references.jsonl` are reused.
   - The six renderer fits in `models/latent_lab/`, pinned with SHA-256 (the state flow in
     `configs/text_realizer_package_eval.yaml`, the others in the renderers config);
     `prepare` verifies them.
   - `data/latents/talent_motivation/embeddinggemma_768/`, `data/sim_talent_motivation/`
     (including `sim_data_counterfactual.csv`) and
     `reports/latent_lab/run-20261027-075057.json`.
3. **Environments.**
   - `exp/text_realizer_package_eval/environment/.venv` (prepare, CPU).
   - `exp/text_realizer/environment/.venv` (generation and scoring).
   - `.cache/text-realizer-tools/uv`.
4. **Offline model caches** in `.cache/huggingface/hub`:
   - `google/gemma-3-270m` at revision `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`
   - `FacebookAI/roberta-large` at revision `722cf37b1afa9454edce342e7895e588b6ff1d59`

   The launcher runs with `HF_HUB_OFFLINE=1`.
5. **Tests.**

   ```bash
   exp/text_realizer/environment/.venv/bin/python -m pytest tests/text_realizer_package_eval -q
   ```

   All 13 tests should pass.
6. **Capacity.**
   - Four GPUs, idle (`nvidia-smi`), for about 12 hours. The launcher takes physical indices.
   - Disk: under 100 MB in total. The bundle is about 34 MB of vectors plus the decoder
     export; the outputs are about 11,000 texts.

## Run

```bash
# 1. CPU, a few minutes: writes models/.../text-realizer-evaluation/renderers-v1
exp/text_realizer_package_eval/environment/.venv/bin/python \
  -m exp.text_realizer_package_eval.renderers prepare --allow-oracle-evaluation
# 2. the scoring packages must resolve before the long run
UV_CACHE_DIR=.cache/text-realizer-uv .cache/text-realizer-tools/uv run \
  --project exp/text_realizer/environment --frozen \
  --with bert-score==0.3.13 --with vaderSentiment==3.3.2 \
  python -c "import bert_score, vaderSentiment"
# 3. inside tmux: generation on four GPUs, then scoring on the first
UV=.cache/text-realizer-tools/uv UV_CACHE_DIR=.cache/text-realizer-uv \
  bash scripts/run_text_realizer_renderers.sh 0 1 2 3
```

**Expected behaviour.**

- **`prepare`:** per renderer, two lines.
  - "replay reproduces the run report", with cosine and distance. At seed 0 the distances
    are about 0.663 (flow), 0.719 (editor), 0.702 (OT editor), 0.711 (OT flow), 0.675
    (state effect) and 0.679 (group OT).
  - `<renderer>-true-target` with gap closed, distance and cosine. Gap closed about 0.197
    (flow), 0.128 (editor), 0.153 (OT editor), 0.121 (OT flow), 0.186 (state effect) and
    0.160 (group OT).
  - Then "group_ot-true-target: identical to group_ot, not decoded again", and last
    "Prepared 1000 units x 11 vector sets".
- **Generation:** each GPU decodes 250 units for all 11 vector sets, about 15 s per CV, so
  about 11.5 hours.
  - Logs and `.progress.json` files are under
    `reports/talent_motivation/text-realizer-evaluation/renderers-v1/`.
  - Rerunning the launcher resumes.
- **Scoring:** takes minutes and prints one line per decoded vector set (11) with chrF and
  BERTScore F1: the renderers "against x'", "factual against x".

## Stop and report instead of working around

- **`prepare` refuses:** a checksum mismatch, "replay differs" from the run report, or "true
  target states do not follow each unit's do(O = k)". The pinned fits or inputs are then not
  the evaluated ones.
- **Step 2 fails:** the packages don't resolve, offline or locked.
- **A worker fails or a GPU is busy.**

Do not edit the config, the pinned files or the code to make a step pass. Do not pull while
the run is in progress: the run pins the code's hashes and would refuse to resume.

## Deliver

On `decoder`, commit and push:

- `docs/experiments/results/decoder_package_renderers_v1.json`
- `reports/talent_motivation/text-realizer-evaluation/renderers-v1/scores/comparisons.jsonl`
  (add it with `git add -f`; the report tree is excluded locally)

Report back:

- the prepare lines
- the start and end time
- the eleven scoring lines
- the commit hash
