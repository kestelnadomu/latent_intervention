# Shipping `langvae_ft` for downstream analysis

The user selected the completed **24-epoch** model on 2026-09-29. Do not extend
training or regenerate the model as part of this handoff. The original LangVAE
and Nomic baselines, shared defaults, and downstream implementation are unchanged.

## Selected model and evidence

- Checkpoint: `models/langvae_ft/20260928T075426Z-train-c8d72b43/checkpoint-024-974eea2e/`.
- Native 128-D LangVAE; unchanged 32-token decoder heads. Jointly tuned existing
  encoder projection and decoder KV adapters, with frozen BERT and GPT-2.
- Factual training CVs only: 3,600 training / 400 validation CVs, split before
  constructing 49,720 / 5,529 self-reconstruction passages from throughout CVs.
  Official test CVs and counterfactual texts were not adaptation examples.
- Finished 2026-09-29 at 00:59 CEST: 24 epochs, 74,592 updates, 15 h 5 min.
  Epoch 24 had the best validation objective; offline reload and integrity passed.
- Validation negative ELBO: 448.36 → 113.06; posterior-mean reconstruction NLL
  per token: 10.88 → 2.38. Generated word-overlap F1 on 64 validation passages:
  13.3% → 39.9% (not factual accuracy).
- Generation still repeats text and invents facts. The model is selected for
  downstream **analysis**, not certified as a faithful CV generator or fair
  representation. Full-CV latent quality and downstream fairness remain to test.
  Training ended at the epoch limit, not a demonstrated convergence plateau.

[Architecture and reproducible recipe](../architecture/langvae_ft.md).

## Two separate deliveries

1. **Git:** only the explicit files in `configs/langvae_ft_scope.json`: the
   native fine-tuning/verification/configuration code, recipe, tests, concise
   documentation and pinned environment descriptors. No changes to encoder,
   pipeline or downstream training code are needed.
2. **Private artifact transfer:** `langvae_ft-epoch24-20260929.tar.gz` (about
   279 MiB compressed; 301 MiB extracted). It contains **only** the selected
   immutable checkpoint, including native weights, provenance and reload probe.
   The archive was prepared locally at
   `models/langvae_ft/langvae_ft-epoch24-20260929.tar.gz`; it is not in Git.

The archive and checkpoint-manifest SHA-256 digests are pinned in
`configs/langvae_ft_scope.json`. Do not send other epoch checkpoints, smoke
runs, optimizer state, prepared features, old experiments, virtual environments,
or pretrained-model caches. The receiving cluster needs the same pinned public
BERT/GPT-2 models in its own cache; the native checkpoint references them rather
than bundling their weights. The probe contains two CV passages as token IDs:
treat this archive as project data, not a public source artifact.

## On the destination cluster only

Pull the code commit and copy the archive to the repository root using the
project's private transfer channel. From that root:

```sh
# Compare this digest with artifact_archive_sha256 in the committed scope file.
sha256sum langvae_ft-epoch24-20260929.tar.gz
# Refuse to overwrite any existing checkpoint files.
tar --keep-old-files -xzf langvae_ft-epoch24-20260929.tar.gz

uv sync --project environments/langvae-cu124 --frozen
FT_PYTHON=environments/langvae-cu124/.venv/bin/python
FT_CHECKPOINT=models/langvae_ft/20260928T075426Z-train-c8d72b43/checkpoint-024-974eea2e
FT_CONFIG=models/langvae_ft/encoding.yaml
export HF_HOME="$PWD/models/hf_cache"

"$FT_PYTHON" -m src.langvae_ft verify --checkpoint "$FT_CHECKPOINT" --device cpu
"$FT_PYTHON" -m src.langvae_ft configure --checkpoint "$FT_CHECKPOINT" --output "$FT_CONFIG"
"$FT_PYTHON" -m src.pipeline encode --config "$FT_CONFIG"
```

Stop if a checksum/reload check fails. Verification can download the pinned
public backbones if absent; set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` when
the cache is already complete. Use the pinned environment to avoid decoder-cache
API differences. Tiny device-dependent numerical differences can affect strict
generation checks; investigate rather than bypassing a failure.

The generated full YAML is opt-in and lives **outside** the immutable checkpoint.
It uses `variant: langvae`, `tag: langvae_ft`, the local checkpoint, and the
existing full-CV encoding procedure (128-D posterior mean, configured input
limit 512). There is no `--encoder-variant langvae_ft` CLI choice: select this
baseline through `--config`. Encoding writes
`data/latents/talent/langvae_ft/z_pairs.pt`, not the stock/Nomic folders.

Then train fresh downstream models using that same configuration, for example:

```sh
"$FT_PYTHON" -m src.pipeline train-decoder --config "$FT_CONFIG" --decoder-variant independent
"$FT_PYTHON" -m src.pipeline train-manipulator --config "$FT_CONFIG" --decoder-variant independent --manipulator-variant baseline
"$FT_PYTHON" -m src.pipeline evaluate --config "$FT_CONFIG" --decoder-variant independent --manipulator-variant baseline
```

Use the existing alternative decoder/manipulator flags if required by the
experiment plan. Always retain `--config "$FT_CONFIG"`; artifacts then remain
under `models/talent/langvae_ft/` and `reports/talent/langvae_ft/`. Never reuse
the original LangVAE's trained `g` or `h_Z`: the tuned encoder changes coordinates.
No production encoding or downstream training was performed on the tuning host.

## Local cleanup boundary

The earlier experimental modifications to the four tracked files were removed.
Only a short `langvae_ft` README pointer remains as a new tracked-file change;
the shared config, original fine-tuner and encoder architecture document were
restored to their committed versions. A recovery patch is retained locally at
`_notes/langvae_ft_shipping_20260929/prior-tracked-changes.patch` (not shipped).
The 20 untracked V1/V2/V3 experiment source/configuration/documentation/test
files were removed from the working tree after verifying a recovery archive at
`_notes/langvae_ft_shipping_20260929/legacy-untracked-experiments.tar.gz` (not
shipped). All models, data, pretrained caches and the working environment remain
local and untouched. No old experiment files are dependencies of this baseline.
