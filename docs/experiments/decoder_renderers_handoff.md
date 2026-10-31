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
x', except phi^-1(phi(x)), which is scored against x, with text similarity, BERTScore,
VADER and overall/changed/preserved fact accuracy. The group-OT alias gives 12 reported
conditions from 11 decoding jobs; sampled state-flow results remain in package-v1.
The true-target conditions still use predicted psi(z) as the source state: they are
oracle-assisted diagnostics, not a deployable setting or a removal of all reader error.
Full procedure: section "The other renderers (renderers-v1)" of
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

   All 19 tests should pass.
6. **Capacity.**
   - Four GPUs, idle (`nvidia-smi`), for about 12 hours. The launcher takes physical indices.
   - Disk: budget 0.3 GB for the bundle, predictions and scores, plus several GB if scoring
     dependencies/model caches need downloading. The prepared bundle alone is about 87 MB.

## Run

```bash
# 1. CPU, a few minutes: writes models/.../text-realizer-evaluation/renderers-v1
exp/text_realizer_package_eval/environment/.venv/bin/python \
  -m exp.text_realizer_package_eval.renderers prepare --allow-oracle-evaluation
# 2. resolve scoring packages and verify GPU execution before the long run
UV_CACHE_DIR=.cache/text-realizer-uv .cache/text-realizer-tools/uv run \
  --project exp/text_realizer/environment --frozen \
  --index https://download.pytorch.org/whl/cu124 --with torch==2.6.0+cu124 \
  --with bert-score==0.3.13 --with vaderSentiment==3.3.2 \
  python -c 'import bert_score, torch, vaderSentiment; print("Scoring GPU ready:", torch.__version__, torch.ones(1, device="cuda:0").item())'
# 3. inside tmux: generation on four GPUs, then scoring on the first
UV=.cache/text-realizer-tools/uv UV_CACHE_DIR=.cache/text-realizer-uv \
  bash scripts/run_text_realizer_renderers.sh 0 1 2 3
```

`--frozen` locks the project environment but not the extra `--with` packages. The explicit
PyTorch CUDA 12.4 pin keeps scoring compatible with this machine's driver. The launcher
uses the prepared dependency cache offline and repeats the GPU check before starting any
generation worker; imports alone cannot verify CUDA compatibility. It holds the existing
text-realizer GPU leases and refuses duplicate, reserved or busy devices.

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
  The portable result also contains per-field `changed` and `preserved` counts/accuracies.
  Both original and target facts must be extractable; unparseable generated values count
  as errors. For factual reconstruction, the target is x, so changed groups are empty
  (`accuracy: null`) and every scorable fact belongs to the preserved group.

## Stop and report instead of working around

- **`prepare` refuses:** a checksum mismatch, "replay differs" from the run report, or "true
  target states do not follow each unit's do(O = k)". The pinned fits or inputs are then not
  the evaluated ones.
- **Step 2 fails:** the packages don't resolve, or the scoring environment cannot execute on the GPU.
- **A worker fails or a GPU is busy.**

Do not edit the config, the pinned files or the code to make a step pass. Do not pull while
the run is in progress: the run pins the code's hashes and would refuse to resume.

## Local readiness verification (30 October 2026)

- The production bundle is prepared for all 1,000 IDs and 11 distinct vector sets. All
  pinned hashes and six sampled renderer replays passed.
- The first four IDs (18, 19, 21, 22) were decoded across all four GPUs: 44 CVs in 177
  seconds. End-to-end scoring produced all 12 condition rows, including the group-OT
  alias and changed/preserved-fact metrics. This is a functional smoke test, not a
  performance estimate for the paper. Its artifacts are in `.cache/renderers-ready-smoke/`.
- BERTScore passed at batch size 16 using the longest reference (502 tokens), with
  PyTorch 2.6.0+cu124. Both model caches and scoring dependencies are available offline.
- All 33 focused pipeline/metric tests, Ruff and shell syntax checks passed. The eight
  encoder tests and four LangVAE-FT encoding tests passed in a separate compatible
  environment; LangVAE requires Transformers 4.48 rather than the decoder's 4.57.
- The wider repository sweep still has six pre-existing Git tracking-policy failures:
  those tests expect blanket artifact exclusions that the package branch disabled.
  The preferred package policy and local exclusions are preserved. These failures do
  not affect generation or scoring. Logs: `.cache/renderers-ready-pytest.log` and
  `.cache/renderers-ready-langvae*-tests.log`.
- The full 11,000-CV run started on 30 October at 23:10 CET in tmux session
  `decoder-renderers-v1-20261030-231004`. Generation finished on 31 October at 09:52 CET;
  all 44 shards and 11,000 CVs passed completion, ID and checksum verification.
- Scoring stopped at 09:59 CET after seven conditions: the `editor-true-target` CV for
  ID 2333 has 518 RoBERTa tokens, exceeding the 512-token limit. It terminated normally
  with EOS; its reference has 397 tokens. A complete scan found exactly one overlength
  prediction among 11,000; all other pairs fit. The no-truncation guard remains intact.
  This interrupted the final aggregate/comparisons export. The completed decoding does
  not need repeating; the scoring-only follow-up is described below.
  Launch logs, prediction shards and `length-diagnostic.json` are in
  `reports/talent_motivation/text-realizer-evaluation/renderers-v1/`.

## Scoring-only follow-up (31 October 2026)

The agreed policy excludes ID 2333 from **BERTScore only, in every condition**, giving
one common cohort of 999/1,000 IDs (99.9% coverage). All other metrics and every exported
CV retain all 1,000 IDs. CVs are never truncated. The result records `bertscore_cohort`
(literal IDs, hash, excluded IDs and coverage) and `bertscore_units` on each row.
Unlisted overlength inputs still cause an error. This BERTScore cohort differs from the
original package-v1 results on 1,000 IDs; tables must identify the denominator explicitly.

On an allocated idle GPU, use the existing bundle and saved predictions directly:

```bash
UV_CACHE_DIR=.cache/text-realizer-uv .cache/text-realizer-tools/uv run \
  --project exp/text_realizer/environment --frozen --offline \
  --index https://download.pytorch.org/whl/cu124 --with torch==2.6.0+cu124 \
  --with bert-score==0.3.13 --with vaderSentiment==3.3.2 \
  python -m exp.text_realizer_package_eval.renderers score \
    --device cuda:0 --bertscore-exclude-id 2333
```

Do not repeat preparation or the generation launcher for this completed run: its
generation hashes describe the original code, while this follow-up changes only scoring.
Scoring verifies the saved bundle/checkpoints and every prediction shard, and records the
updated scoring source hashes separately. Allow about 10–15 minutes for scoring on one GPU.

The scoring-only job started at 11:11 CET on GPU 0 in tmux session
`decoder-renderers-score-20261031-111059` and finished successfully at **11:21:54 CET**
(exit code 0). All 12 condition rows and the complete 1,000-ID CV comparison export
are written. Bundle, comparison, cohort and scoring-source hashes were verified;
the original references and reused group-OT texts match. All four GPUs are idle.
Its command, cohort and log path are recorded in `renderers-v1/score-launch.json`.
All 36 focused pipeline/metric checks and Ruff passed before launch; the wider sweep
still reports only the same six pre-existing Git tracking-policy failures.

## Deliver

On `decoder`, publish the code/tests/documentation changes and these completed results:

- `docs/experiments/results/decoder_package_renderers_v1.json`
- `reports/talent_motivation/text-realizer-evaluation/renderers-v1/scores/comparisons.jsonl`
  (add it with `git add -f`; the report tree is excluded locally)
- The run's prediction JSONL shards, metadata/completion/progress JSONs, launch records,
  length diagnostic and logs. Exclude transient `.lock` files.
- `reports/talent_motivation/text-realizer-evaluation/renderers-v1/bundle-manifest.json`,
  an exact copy of the scored bundle's manifest, verified against the result's
  `bundle_sha256`. This preserves input/checkpoint provenance without transferring weights.

The run outputs total about 78 MB before Git compression; the largest file is the
29.18 MiB comparison JSONL. They contain every CV, including ID 2333 in all conditions.
Only BERTScore excludes that ID. On the other cluster, `git pull origin decoder`
while on `decoder` retrieves the results for analysis; no GPU or model weights are needed
to read the JSON/JSONL files. Decoder checkpoints and environment/model caches are separate
from this results transfer.

Report back:

- the prepare lines
- the start and end time
- the eleven scoring lines
- the commit hash
