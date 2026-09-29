# `langvae_ft`: native LangVAE adapted to CV passages

This is the third encoder baseline, alongside unchanged `langvae` and `nomic`.
It is opt-in and does not change the default pipeline, existing latents, semantic
decoders `g`, or intervention models. The implementation is in `src/langvae_ft/`;
it imports none of the earlier V1/V2/V3 training modules.

## What is fine-tuned

Start from the same pinned pretrained
`neuro-symbolic-ai/eb-langvae-bert-base-cased-gpt2-l128` checkpoint. Train its
existing BERT-to-posterior linear projection and existing positional key/value
decoder adapters jointly. BERT and GPT-2 remain frozen, as in native LangVAE
adapter training. This is **not** end-to-end updating of the transformer
backbones. The resulting posterior means, and therefore the latent coordinates,
will change; downstream models fitted to the original coordinates are not
interchangeable with this baseline.

Keep the 128-D latent, original parameter shapes, original 32-token decoder,
dropout, tokenizers, and rolling positional-KV decoding convention. There are
no enlarged heads, prefix-model replacement, LoRA, auxiliary predictors,
ranking losses, or lexical losses.

## Data and objective

1. Use only factual CVs from the official training partition. With the current
   data and seed 42, split these 4,000 CV IDs into 3,600 adaptation-training and
   400 validation CVs **before** making passages. The 1,000 official test CVs
   never supply training/validation features. Their text is read only for an
   exact normalized-document duplicate check. Counterfactual texts and structured
   labels are not read by this workflow.
2. Pack consecutive, non-overlapping passages at whitespace boundaries, each
   fitting at most 31 GPT-2 content tokens plus supervised EOS. Include passages
   from the beginning, middle and end of every CV. Character spans cover the
   entire stripped CV; an over-budget single word fails instead of being silently
   truncated. Tokenizer normalization remains native to the checkpoint.
3. Encode each **passage**, then reconstruct that same passage. In particular,
   do not encode a complete CV and target only its opening. Cache frozen 768-D
   BERT features locally for efficiency; never cache the trainable projection
   during optimization. Passages have equal weight, so longer CVs contribute
   more examples. Full-document duplicate checks do not exclude naturally
   recurring phrases across different CVs.
4. Minimize token-summed teacher-forced reconstruction cross-entropy plus the
   standard Gaussian KL, averaged over passages. Padding alone is ignored;
   EOS remains supervised even though its token ID equals the padding ID.
   Linearly warm the KL coefficient from zero to one over the first epoch.
   There are no free bits or additional losses.

The wrapper corrects training mechanics, not the model architecture: frozen
transformers stay in evaluation mode, gradients still flow through GPT-2 to
the existing adapters, teacher forcing is causal, and gradient checks cover
both trainable components. Generation is checked against the original native
decoder. Training uses FP32, gradient clipping and AdamW without weight decay.

## Selection and interpretation

`configs/langvae_ft.yaml` is the standalone recipe. Defaults are encoder LR
1e-5, decoder-adapter LR 1e-4, batch size 16, at most 24 epochs, minimum 8 epochs
before patience-4 early stopping, and a 36-hour elapsed-run budget checked at
epoch boundaries. The budget is **not a completion-time estimate or hard
deadline**: the current epoch and final reload check may overrun it.

Select among trained epoch checkpoints using validation negative ELBO, always
with KL coefficient one and fixed-seed Monte Carlo sampling. Report posterior-
mean token NLL separately. Compare against an untuned validation measurement
from the same run. Correct-latent, wrong-CV-latent and zero-latent generation/NLL
controls on one passage per sampled validation CV are diagnostics only, not
extra training objectives. Also record active latent dimensions.

Before accepting a checkpoint, review validation improvement, latent-use gaps,
and held-out generated passages. The workflow deliberately does not declare
quality success automatically. A lower teacher-forced loss alone is not proof
of faithful generation, useful downstream features, or counterfactual fairness.

The unchanged decoder can generate only short passages, **not complete CVs**.
This baseline measures how much ordinary, architecture-preserving domain
adaptation helps. Final research encodings still use the existing full-CV
encoding procedure (configured input limit 512, deterministic 128-D posterior
mean), not an average of passage encodings. The passage/full-CV distribution
difference is an explicit limitation; downstream quality must be measured.

## Running on this GPU host

From the repository root, use the tested isolated Python 3.12 environment:

```sh
export HF_HOME="$PWD/models/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
environments/langvae-cu124/.venv/bin/python -m src.langvae_ft smoke --device cuda:0
environments/langvae-cu124/.venv/bin/python -u -m src.langvae_ft train --device cuda:0
```

These are separate commands: `smoke` runs only two updates on four CVs and
produces a diagnostic checkpoint, never a final encoding configuration. `train`
starts the main run. Choose an idle GPU first. Offline mode assumes the pinned
pretrained weights/tokenizers already exist in the cache; omit the offline
flags during an authorized first download on a new host.

To recreate this environment, retain its `pyproject.toml` and `uv.lock` and run
`uv sync --project environments/langvae-cu124 --frozen`. They pin Torch 2.6.0
with CUDA 12.4, Transformers 4.48.0 and LangVAE 0.6.13 without editing the root
dependencies or root lockfile. The installed `.venv` is not a Git artifact.

Each command prints a unique `RUN_DIR` under `models/langvae_ft/`. It contains
configuration/source/data provenance, prepared passage features, baseline and
trained validation reports, generated examples, epoch history, status, native
checkpoints, and epoch-boundary optimizer/RNG resume state. No production paired
latents are generated. Resume an interrupted run with the **same** recipe,
source, environment and device setting:

```sh
environments/langvae-cu124/.venv/bin/python -u -m src.langvae_ft train \
  --device cuda:0 --resume models/langvae_ft/RUN_ID
```

An interrupted partial epoch is replayed from the previous completed epoch.
There is no resume state until the first epoch completes; start a new run if
preparation/that first epoch failed. Completed runs cannot be resumed in place.
For a memory problem, lower batch size and increase accumulation in a separate
recipe, then start a fresh run rather than changing an existing run's identity.

After training, the selected native checkpoint must reload in a fresh offline
process and reproduce probe features, posterior means and generated token IDs.
Frozen-backbone hashes and parameter shapes must remain unchanged, and both
trainable components must have updated. A main run then writes a separate
`encoding.yaml`; it does not execute it. Smoke checkpoints cannot be promoted
through the configuration command.

## Transfer and destination-cluster encoding

See [the selected checkpoint and downstream handoff](../experiments/langvae_ft_handoff.md).
Weights, prepared features, optimizer states and outputs stay outside Git in
the already-ignored `models/` and `data/latents/` trees. The selected checkpoint
archive and checksum are hosted in the private Hugging Face model repository
`latent-intervention/latent_intervention`. Follow the
[access and pinned-download instructions](../experiments/langvae_ft_handoff.md#access-and-download-from-hugging-face)
for organization membership, repository-scoped read access, checksum verification
and extraction on the destination cluster.
The checkpoint saves native trainable weights and references pinned backbone
models; it does not bundle BERT/GPT-2 weights. Its reload probe contains CV token
IDs and should be treated as project data, not a public source artifact.

On the destination, keep the checkpoint immutable and write a full opt-in
pipeline YAML **outside** that directory:

```sh
python -m src.langvae_ft verify --checkpoint models/langvae_ft/RUN_ID/CHECKPOINT
python -m src.langvae_ft configure \
  --checkpoint models/langvae_ft/RUN_ID/CHECKPOINT \
  --output models/langvae_ft/RUN_ID/encoding-destination.yaml
python -m src.pipeline encode \
  --config models/langvae_ft/RUN_ID/encoding-destination.yaml
```

Use the compatible environment and pinned model cache for these commands. On a
different device, tiny numeric differences may affect the strict generation
probe; investigate a failed check rather than silently bypassing it. The encoder
uses `variant: langvae`, `tag: langvae_ft` and the local checkpoint. No new
pipeline variant or downstream-code change is required. The full generated
configuration preserves other pipeline settings and removes the old experimental
`finetune_vae` section. It defaults to CPU encoding, as the existing baseline
does; a destination-specific change should be made only in this opt-in YAML.

With the existing talent paths, paired latents will be written to
`data/latents/talent/langvae_ft/z_pairs.pt`. Existing `langvae/` and `nomic/`
artifacts remain untouched. A future downstream run would likewise get its own
`models/talent/langvae_ft/` and `reports/talent/langvae_ft/` namespaces. Do not
reuse the original baseline's `g` or intervention weights for the new space.
