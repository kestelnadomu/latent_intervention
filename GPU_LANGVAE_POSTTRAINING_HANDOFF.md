# GPU handoff: LangVAE post-training for the CV domain

Last updated: 2026-09-22

Repository: `latent_intervention`

Current branch: `traing_g`
Current implementation commit: `5e6371dd29ca0f1286bef7fa5c169454f30df63c`

## 0. Mission for the receiving agent

Continue the work on a GPU server by making the LangVAE post-training pipeline scientifically valid and operational, then run a staged CV-domain adaptation experiment.

The immediate objective is to determine whether post-training the pinned LangVAE on CV-domain text can produce:

1. more specific and coherent CV-like reconstructions;
2. a useful 128-dimensional latent representation for the downstream semantic decoder `g(S|Z)`; and
3. a separate, reproducible fine-tuned latent space that can be compared with stock LangVAE and Nomic.

Do **not** launch `python -m src.finetune_vae` unchanged. The current file is a prototype with known correctness, leakage, device-placement, and sequence-length problems. Fix and test those issues first.

The recommended first experiment is an explicit **32-token CV-domain adaptation pilot**. It is a correctness and feasibility gate, not a claim of full-CV reconstruction. Genuine reconstruction of approximately 410-token CVs requires a redesigned decoder; it cannot be obtained by merely changing a YAML length.

## 1. What the project is about

This repository studies **counterfactual latent representations of text**. A known structural causal model (SCM) generates paired factual and counterfactual candidate attributes. Those structured worlds are verbalized as matched CV personal statements and embedded in a 128-dimensional latent space.

The main objects are:

- `X`: factual CV text;
- `X'`: matched counterfactual CV text;
- `f: X -> Z`: text encoder, frozen after any optional domain adaptation;
- `g(S|Z)`: learned semantic decoder/probe grounding a latent in the structured SCM state;
- `h_S(S'|S, delta)`: exact symbolic counterfactual kernel supplied by the simulator;
- `h_Z(Z'|Z, delta)`: learned latent editor/manipulator.

The intended commuting relationship is:

```text
g o h_Z  approximately equals  h_S o g
```

The ultimate goal is a **post-hoc, source-free, distributional latent editor** that internalizes the symbolic counterfactual transformation. At inference, it should require only a factual latent `z` and intervention `delta`, not the SCM, a structured target `S'`, or a counterfactual source text.

An important scientific constraint is that paired counterfactual latent `Z'=f(X')` is an **evaluation target only** for the principal latent-editor experiment. It must not become direct supervision for the principal `h_Z` objective. Text-only, unsupervised LangVAE domain adaptation is a separate representation-learning stage; after adaptation, the encoder must be frozen again.

Relevant project documents:

- `README.md`
- `docs/problem_formulation.md`
- `docs/contribution.md`
- `docs/experiments.md`
- `docs/architecture/text_encoder.md`
- `docs/architecture/semantic_decoder.md`
- `docs/architecture/latent_intervention.md`
- `exp/sim/README.md`

## 2. Active talent-SFM experiment

The active simulation is configured in `exp/sim/config.yaml` and implemented in `exp/sim/talent_sfm.py`.

The structured state is `S=(X,T,D,U)` with `4 * 3 * 3 * 3 = 108` states:

- `X`: country-of-origin region, four levels;
- `T`: hidden talent, three levels; decoded by `g` but never stated explicitly;
- `D`: degree, three levels;
- `U`: university-rank tier, three levels;
- `P, L, H, A`: simulated and verbalized behavioral proxies for talent, but not part of `S`;
- `Y`: qualification outcome, simulated but excluded from `S` and the current consistency loss.

The fixed intervention is `do(X=3)`. Factual and counterfactual units share exogenous noise. The render plan holds template, persona, and within-level quantiles fixed, so paired texts should differ through the intervention and its SCM descendants.

Current canonical data:

- 5,000 units;
- 4,000 official training IDs;
- 1,000 official test IDs;
- training split: 1,028 identity and 2,972 non-identity interventions;
- test split: 256 identity and 744 non-identity interventions;
- 5,000 factual CV texts;
- 5,000 counterfactual CV texts;
- `data/sim_talent/pair_index.csv` is the sole split and identity authority.

The validation command:

```bash
uv run python -m exp.sim.run validate-pairs
```

passed complete coverage and pairing. No CV exceeds the configured 512-token encoder budget.

Known data caveat: 63 factual and 65 counterfactual texts contain a forbidden talent-related word such as “talent”, “exceptional”, or “outstanding”. This affects about 1.3% of each corpus and could leak information about hidden `T`. It is not a technical blocker, but it must remain documented and should eventually be included in a sensitivity analysis.

## 3. Conversation history and work already completed

### 3.1 Earlier conversation: setup and missing data

The earlier conversation was titled “Plan training g implementations.” Work initially stopped because the paired-data contract was incomplete. A colleague subsequently supplied the missing files under `data/sim_talent/`:

- `pair_index.csv`
- `sim_data_factual.csv`
- `sim_data_counterfactual.csv`
- `sim_data_epsilon.csv`
- `render_plan.csv`
- `simulation_info.json`

These files are now present and validated.

The encoder setup was then made reproducible and variant-safe:

- LangVAE and Nomic are selectable independently;
- model revisions are pinned;
- LangVAE’s reconstructed BERT and GPT-2 bases are pinned separately;
- encoder-, decoder-, and manipulator-dependent paths are isolated by variant;
- latent metadata records model/config/source hashes;
- incompatible artifacts fail validation rather than mixing latent spaces silently;
- configuration supports encoder, semantic-decoder, and manipulator variants.

The last full preflight had 77 passing tests and one CUDA-dependent skip. Ruff, compile, and diff checks also passed.

No semantic-decoder `g` or latent-editor `h_Z` checkpoint has been trained yet. The representation decision comes first.

### 3.2 Baseline LangVAE encoding

The stock LangVAE encoding completed successfully:

- start: 2026-09-22 08:59:44 CEST;
- end: 2026-09-22 09:23:24 CEST;
- runtime: 23 minutes 40 seconds on CPU;
- process exit code: 0;
- output: `data/latents/talent/langvae/z_pairs.pt`;
- sidecar: `data/latents/talent/langvae/z_pairs.info.json`;
- factual `z` shape: `(5000, 128)`, float32, finite;
- official-test `z_prime` shape: `(1000, 128)`, float32, finite;
- 256 identity test latents copied exactly;
- all 744 non-identity test latents changed;
- artifact SHA-256:
  `43d44f3ab55e3db5c1b7f202f1755bc2c2cb3078986cd8a888003dbcffce4547`.

The only runtime message was the known nonfatal PyTorch sparse-invariant warning.

### 3.3 Baseline Nomic encoding

The Nomic encoding also completed successfully:

- start: 2026-09-22 09:55:41 CEST;
- end: 2026-09-22 10:27:44 CEST;
- runtime: 32 minutes 03 seconds on CPU;
- process exit code: 0;
- output: `data/latents/talent/nomic/z_pairs.pt`;
- sidecar: `data/latents/talent/nomic/z_pairs.info.json`;
- factual `z` shape: `(5000, 128)`, float32, finite;
- official-test `z_prime` shape: `(1000, 128)`, float32, finite;
- vectors are unit normalized within float32 tolerance;
- 256 identity test rows are exact;
- all 744 non-identity test rows changed;
- artifact SHA-256:
  `465fdcea75628720b1006c802092bf6955e7921975e806f68c506c07e83bd407`.

Nomic has no text decoder. It is a representation baseline and must keep separate downstream `g` and `h_Z` checkpoints.

### 3.4 Stock LangVAE reconstruction audit

The colleague asked how decoded stock-LangVAE encodings look under the known domain mismatch. A deterministic, offline reconstruction audit was run without altering the production artifacts.

Audit design:

- 32 factual CVs stratified by source length and region;
- 12 non-identity official-test pairs, decoding both factual and counterfactual worlds;
- 8 length-matched, 32-token CV-prefix controls;
- 8 short science-style controls;
- 72 total decodes, of which 56 were production CV latents.

The most important discovery is that the checkpoint’s decoder configuration has `max_len=32`. The production CVs contain 343–499 GPT-2 tokens and average about 410.4 tokens. Standard decoding therefore cannot reconstruct a complete CV even in principle.

Main audit results:

| Metric | Stock LangVAE result |
|---|---:|
| Production full-text token F1 | 0.087 |
| Production prefix token F1 | 0.159 |
| LangVAE decode/re-encode cosine | 0.377 |
| Independent Nomic semantic cosine, matched | 0.553 |
| Independent Nomic semantic cosine, shuffled | 0.536 |
| Nomic matched-minus-shuffled margin | 0.017 |
| Production Nomic top-1 retrieval | 5.4% |
| Degenerate production reconstruction rate | 26.8% |
| Country recovery | 0% |
| Degree recovery | 0% |
| University-tier recovery | 0% |
| Project-count recovery | 0% |
| Science-control token F1 | 0.636 |
| Science-control Nomic cosine | 0.943 |

Typical CV reconstruction:

```text
photos a a apour of the a to make coal for a work and energy in a a long and a a of a
```

Short controls were substantially better:

```text
Coal is a fossil fuel. -> coal is a kind of fossil fuel
Iron is a metal.       -> iron is a kind of metal
```

This shows severe **length and domain mismatch** in the frozen encoder-decoder round trip. It does not prove that the posterior mean lacks all useful structured signal; held-out `g(S|Z)` accuracy and calibration are the representation-level tests.

Audit files:

- `reports/talent/langvae/reconstruction_audit/report.md`
- `reports/talent/langvae/reconstruction_audit/summary.json`
- `reports/talent/langvae/reconstruction_audit/reconstructions.csv`
- `reports/talent/langvae/reconstruction_audit/paired_reconstructions.csv`
- `reports/talent/langvae/reconstruction_audit/ten_original_reconstruction_pairs.md`
- `reports/talent/langvae/reconstruction_audit/run_audit.py`

## 4. Reproducibility pins and hashes

Pinned models:

| Component | Repository | Revision |
|---|---|---|
| LangVAE checkpoint | `neuro-symbolic-ai/eb-langvae-bert-base-cased-gpt2-l128` | `7f277f7c42e3a2ff94a4ce7299e1c944f872a8e6` |
| LangVAE BERT | `bert-base-cased` | `cd5ef92a9fb2f889e972770a36d4ed042daf221e` |
| LangVAE GPT-2 | `gpt2` | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| Nomic model | `nomic-ai/nomic-embed-text-v1.5` | `e5cf08aadaa33385f5990def41f7a23405aec398` |
| Nomic remote code | — | `7710840340a098cfb869c4f65e87cf2b1b70caca` |

Relevant package versions from the CPU environment:

- Python 3.12.8;
- uv 0.8.22;
- torch 2.12.1, CUDA 13.0 build;
- transformers 4.48.0;
- LangVAE 0.6.13;
- huggingface-hub 0.36.2;
- accelerate 1.14.0;
- pandas 3.0.3;
- NumPy 2.5.1.

Canonical input hashes:

| File | SHA-256 |
|---|---|
| `data/text_talent/cv_factual.csv` | `e2845bd8b16360ae9c92f51240af0942f92b26b94b1331cf0ffdb75253f67294` |
| `data/text_talent/cv_counterfactual.csv` | `34b3a3ac0747bdd65ab8d7175c8582a9e9d38fcf06ec31f436b5c499ec9f993c` |
| `data/sim_talent/pair_index.csv` | `67616f395548dfffb7f063b554338b6b63731f5a77136134ffc3b2936b5d51ee` |
| `data/sim_talent/sim_data_factual.csv` | `98de23e96b27135c36d181cb51ce52e02d526c4c1f3c81a7a5f835c347e69150` |
| `data/sim_talent/sim_data_counterfactual.csv` | `e6e2f2ee1f4be2938abff051ea6fa9c875811de1c7fc3f1a7c73bb069609dd4c` |
| `data/sim_talent/sim_data_epsilon.csv` | `811ac2d6d3fcbe8bbdad0cc16e7551af05b9736b612b137d0b6b7a3bea62a4dd` |
| `data/sim_talent/render_plan.csv` | `1175d44adce2064c1e1f961b8cc46544b0410bda6f4c8aaed50fe035649ca441` |

Audit hashes:

| File | SHA-256 |
|---|---|
| Reconstruction report | `7abc1877f7eabfc03d71162ba1bd32922c9cbda7cd87488531f9505f430944f9` |
| Reconstruction summary | `fb79f0b6c4240066678cd00b70fc4eb8d6d18b86248b7aedb2e7fdc8f1e1eae5` |
| Ten-pair Markdown | `2ceb68fa8779c765db75976d109501b16a0601acd0e4d7cdaff4af047b982744` |

## 5. Transfer state and warning

At the time this handoff was prepared, the CPU branch contained these required commits beyond the previous remote base:

```text
dc1a5d6a335fd0a599dd569d57ef220f631ff0b6  langvae u. nomic encodings
5e6371dd29ca0f1286bef7fa5c169454f30df63c  reproduzierbare encoder pipeline
```

This handoff and the associated documentation updates are intended to be committed immediately above them. After all three commits are pushed, a normal checkout of the tip of `traing_g` will include the source changes, tests, documentation, handoff, and both baseline encoding artifacts. On the GPU server, verify that the history contains both hashes above before doing any work.

Although `data/latents/` is ignored by default, these four baseline files were explicitly added in commit `dc1a5d6` and are therefore tracked:

- `data/latents/talent/langvae/z_pairs.pt`
- `data/latents/talent/langvae/z_pairs.info.json`
- `data/latents/talent/nomic/z_pairs.pt`
- `data/latents/talent/nomic/z_pairs.info.json`

The following generated paths remain ignored by Git and require explicit transfer if they are needed:

- `models/`
- `reports/`
- `.venv/`

Do not copy `.venv` between servers; recreate it from `uv.lock`. For an exact before/after reconstruction comparison, separately copy or archive:

- `reports/talent/langvae/reconstruction_audit/`

No existing model checkpoint is required: post-training starts from the pinned Hugging Face LangVAE checkpoint. The committed Nomic artifacts are useful as a comparison but are not required to run LangVAE post-training.

If the GPU server has internet access, download the pinned HF snapshots normally. If it is offline, copy the complete cache repositories, including blobs and snapshot symlinks:

- `~/.cache/huggingface/hub/models--neuro-symbolic-ai--eb-langvae-bert-base-cased-gpt2-l128`
- `~/.cache/huggingface/hub/models--bert-base-cased`
- `~/.cache/huggingface/hub/models--gpt2`

Do not copy snapshot symlinks without their blob directories.

## 6. What “post-training” means here

The stock checkpoint contains frozen BERT and GPT-2 models plus trainable mappings:

- frozen BERT: approximately 108.3M parameters;
- frozen GPT-2: approximately 124.4M parameters;
- trainable encoder posterior projection `W`: 196,608 parameters;
- trainable decoder latent-to-key/value adapters: 78,465,024 parameters;
- total registered/trainable parameters: 78,661,632.

This is **not full language-model fine-tuning**. In the current LangVAE package, BERT and GPT-2 remain frozen. Post-training refits the posterior projection and latent-to-decoder conditioning.

The desired experiment is:

1. adapt the trainable mappings to the CV domain;
2. keep the latent dimension at 128;
3. preserve a VAE-style KL objective;
4. select checkpoints only on a held-out adaptation-validation set;
5. rerun the reconstruction audit on untouched official-test examples;
6. freeze the adapted encoder;
7. create a separately tagged latent artifact;
8. retrain all downstream components in that new latent space.

The first pilot must use a target of no more than 32 tokens because that is what the existing decoder adapter was built to condition directly. It may improve short CV-like snippets. It will not produce a complete 400-token CV.

## 7. Why `src/finetune_vae.py` must not be run unchanged

### 7.1 Data leakage and incomplete corpus use

The current script:

- reads only `paths.texts`, i.e. 5,000 factual CVs;
- ignores all 5,000 counterfactual CVs;
- randomly chooses 4,500 training and 500 validation rows;
- disregards `pair_index.csv`;
- consequently allows official-test IDs into post-training.

This invalidates later held-out claims.

### 7.2 Sequence-length mismatch

The script tokenizes to `encoder.max_len=512`, but the checkpoint decoder adapter was constructed with `decoder.max_len=32`.

LangVAE can technically loop for the longer batch target length, but after position 32 it no longer inserts a new latent-conditioned key/value slot. Standard `decode_sentences` still defaults to 32 output tokens. A 512-token run is therefore not a clean full-length decoder adaptation.

Simply rebuilding the current per-position adapter for 512 positions would scale it to about 1.22B trainable parameters:

- approximately 4.88 GB for FP32 weights alone;
- approximately 19.5 GB for weights, gradients, and Adam states;
- more memory for frozen bases, logits, activations, and the trainer’s best-model copy.

An 80 GB GPU would be the safe class for that literal expansion, and the design would remain inefficient. Do not pursue it.

### 7.3 Incorrect reconstruction-loss reduction

LangVAE 0.6.13 currently sums flattened token NLL to a scalar before multiplying by the `(B,S)` mask. The result scales approximately quadratically with sequence length and includes padded-token loss incorrectly.

The loss must be corrected locally so token loss is reshaped/masked before sequence and batch reduction.

### 7.4 Excessive dense target memory

The package materializes sparse one-hot text as dense `B x S x 50,257` int64 targets inside the loss. At batch 8 and roughly 410 tokens, the target alone is about 1.23 GiB. Probability/log tensors and autograd consume substantially more.

Use token IDs directly. Do not densify the vocabulary axis.

### 7.5 Gradient clipping is in the wrong order

The installed trainer calls `clip_grad_norm_` before backward. The actual zero-grad/backward/step occurs afterward, so clipping is ineffective.

Use a local training step with this order:

1. zero gradients;
2. forward;
3. backward;
4. finite-gradient check;
5. gradient clipping;
6. optimizer step.

### 7.6 GPU placement is unsafe

The package stores frozen BERT and GPT-2 in ordinary Python lists. They are not registered in the outer module tree, so an outer `model.to("cuda")` does not reliably move them or update their internal device fields.

Move and assert every component explicitly:

- `model.encoder._encoder[0]`;
- `model.decoder._decoder[0]`;
- encoder projection `W`;
- all decoder adapter layers;
- inputs, masks, targets, and latent tensors.

Do not silently fall back to CPU.

### 7.7 AMP is not safely implemented

Pythae’s current `amp=True` path uses autocast without a GradScaler. Begin in FP32. Enable mixed precision only after adding a correct FP16 scaler or an explicitly tested BF16 path.

### 7.8 Weak checkpoint/recovery behavior

The current configuration does not request periodic checkpoints. The final path is timestamp-nested under `models/langvae_cv/.../final_model` and should not be assumed to be `models/langvae_cv/final_model`.

Add versioned output paths, per-epoch checkpoints, best-validation selection, resume support, and an atomic run manifest/status.

## 8. Data contract for post-training

The user states that CVs were generated specifically for encoder post-training. In the current checkout, the only visible text corpora are the canonical factual and counterfactual files. The receiving agent must identify whether a genuinely separate adaptation corpus exists on the GPU server or whether the canonical corpus was intended.

Do not guess.

### Preferred case: separate adaptation corpus

If a genuinely separate CV corpus exists:

- record its path, row count, schema, generator provenance, and SHA-256;
- guarantee no duplicate or near-duplicate official-test CVs;
- split at source-unit ID before creating windows;
- reserve 10–20% as adaptation validation;
- preserve a completely untouched reconstruction test.

### Fallback case: use canonical official-training IDs only

If no separate corpus exists:

- never train on the 1,000 official-test IDs;
- start from the 4,000 IDs marked `train` in `pair_index.csv`;
- deterministically reserve 400 IDs for adaptation validation using seed 42;
- train on the remaining 3,600 IDs;
- keep factual and counterfactual variants from the same ID in the same partition;
- it is reasonable to include factual plus nonduplicate counterfactual text;
- deduplicate identity counterfactuals rather than overweighting those units.

There are 6,972 unique factual/non-identity-counterfactual texts among official-training IDs.

Persist:

- training IDs;
- validation IDs;
- any untouched audit IDs;
- window construction policy;
- seed;
- exact source hashes;
- duplicate-removal rule.

## 9. Required implementation before any GPU run

### 9.1 Make training length explicit

Add a fine-tuning setting such as:

```yaml
finetune_vae:
  mode: cv_window
  sequence_length: 32
  epochs: 3
  max_epochs: 5
  batch_size: 4
  gradient_accumulation_steps: 1
  learning_rate: 1.0e-4
  max_beta: 1.0
  n_cycles: 1
  target_kl: 2.0
  grad_clip: 1.0
  val_split: 0.1
  checkpoint_every_epochs: 1
  output_dir: models/langvae_posttrain/langvae_cv32_v1
```

Keep `encoder.max_len=512` for later production encoding. Do not reuse it as the post-training target length.

### 9.2 Construct deterministic CV windows

For the first pilot:

- tokenize with the pinned GPT-2 tokenizer;
- create explicit <=32-token windows;
- split by unit ID **before** expanding into windows;
- keep validation windows fixed across epochs;
- begin with one or a small fixed set of beginning/middle/end windows per document;
- do not immediately multiply every CV into all approximately 13 windows;
- if resampling train windows by epoch, seed it deterministically and persist the policy.

This pilot tests domain adaptation to CV-like snippets. It does not test full-CV generation.

### 9.3 Replace the reconstruction loss

A valid outline is:

```python
# recon_prob: (B, S, V)
# target_ids: (B, S), pad positions equal pad_id
logp = recon_prob.clamp_min(torch.finfo(recon_prob.dtype).tiny).log()
token_nll = F.nll_loss(
    logp.transpose(1, 2),
    target_ids,
    ignore_index=pad_id,
    reduction="none",
)
valid = target_ids.ne(pad_id)
recon_per_example = (token_nll * valid).sum(dim=1)
raw_kl = -0.5 * (
    1 + log_var - mu.square() - log_var.exp()
).sum(dim=1)
weighted_kl = beta * torch.where(
    raw_kl > target_kl,
    raw_kl,
    torch.zeros_like(raw_kl),
)
loss = (recon_per_example + weighted_kl).mean()
```

Validate BOS/target alignment through a tiny overfit test. Do not assume it.

Log separately:

- total loss;
- reconstruction NLL;
- per-token NLL;
- raw KL;
- weighted KL;
- beta;
- active latent dimensions;
- gradient norm.

### 9.4 Fix the training loop

The local loop must support:

- explicit device selection and CUDA fail-fast;
- deterministic seeds;
- correct backward/clip/step ordering;
- finite loss and gradient checks;
- optional gradient accumulation;
- checkpoint each epoch;
- best validation checkpoint;
- resume;
- atomic status file;
- hardware and package manifest;
- fixed decoded validation examples after every epoch.

### 9.5 Add tests

At minimum:

1. official-test IDs never enter training or adaptation validation;
2. factual/counterfactual variants and windows remain grouped by ID;
3. every window is <=32 GPT-2 tokens;
4. padding does not change the loss of a sequence;
5. loss is finite and scales sensibly with duplicated/batched examples;
6. one CPU and one CUDA forward/backward step are finite;
7. BERT and GPT-2 remain frozen and unchanged;
8. `W` and decoder adapters receive nonzero finite gradients;
9. clipping happens after backward;
10. every model component is on the selected device;
11. save/reload reproduces deterministic encodings and decodes;
12. seeded window construction is repeatable;
13. output paths do not collide with stock LangVAE.

## 10. GPU environment and resource expectations

Recommended for the 32-token pilot:

- one CUDA GPU;
- 16 GB VRAM minimum;
- 24 GB VRAM preferred;
- at least 32 GB system RAM;
- at least 10 GB free disk;
- FP32 for the first correctness runs;
- initial batch size 2 or 4;
- raise toward 8 only after measuring peak allocated/reserved VRAM.

Current CPU-machine measurements:

- 32-token batch-1 training step: about 1.93 seconds after warm-up;
- 32-token training peak process RSS: about 3.15 GiB;
- 398-token batch-1 no-grad forward: about 9.43–11.57 seconds;
- 455-token batch-8 no-grad forward: about 23.61 seconds;
- unchanged five-epoch, factual-only full-length script: estimated 50–75 CPU hours if batch 8 backward fits, otherwise 4–8 days;
- 32-token CPU adaptation: roughly 4–10 hours.

Estimated 32-token GPU pilot time is roughly 30 minutes to two hours on a modern 16–24 GB GPU, but this is not a promise. Benchmark 100 optimizer steps on the actual GPU and extrapolate from the actual number of windows and epochs.

The current lock resolves a CUDA 13.0 PyTorch build. Verify driver compatibility:

```bash
nvidia-smi
uv run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("compiled CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available()
print("device:", torch.cuda.get_device_name(0))
print(
    "VRAM GiB:",
    torch.cuda.get_device_properties(0).total_memory / 2**30,
)
PY
```

If the driver cannot support the locked CUDA build, install a compatible PyTorch build and record the deviation. Do not silently change the environment.

No OpenAI/Azure API key or billed text generation is needed for post-training.

## 11. Staged execution protocol

Do not jump directly to a multi-epoch full-corpus run.

### Stage 0: synchronize and verify

1. Pull the tip of `traing_g` and verify that commits `dc1a5d6` and `5e6371d` are ancestors of `HEAD`.
2. Copy the ignored reconstruction audit if an exact baseline comparison is needed.
3. Recreate the environment:

   ```bash
   uv sync --extra dev --frozen
   ```

4. Verify branch, commit, diff, input hashes, CUDA, and model caches.
5. Run:

   ```bash
   uv run pytest -q
   uv run ruff check .
   uv run python -m exp.sim.run validate-pairs
   ```

### Stage 1: one-batch CUDA smoke

Run exactly one forward/backward/optimizer step and assert:

- finite total/reconstruction/KL loss;
- finite gradient norm;
- nonzero gradients only for intended parameters;
- frozen BERT and GPT-2 unchanged;
- all devices are CUDA where expected;
- peak VRAM is comfortably below the GPU limit.

Save/reload the checkpoint in a fresh process.

### Stage 2: tiny overfit gate

Use 50–200 training snippets and 10–20 validation snippets for 100–500 optimizer steps.

Require:

- clear reconstruction-NLL reduction;
- input-specific decoded snippets;
- no NaN/Inf;
- no generic duplicate collapse;
- reload reproduces deterministic output.

If the model cannot overfit this tiny set, stop and fix alignment/objective code.

### Stage 3: throughput pilot

Run 100 representative optimizer steps at the intended batch size.

Record:

- seconds per optimizer step;
- examples or windows per second;
- maximum allocated and reserved VRAM;
- CPU RAM;
- loss components;
- gradient norm;
- projected total runtime.

Only then choose batch size, gradient accumulation, and whether mixed precision is necessary.

### Stage 4: one real epoch

Run one epoch on the real adaptation-training partition. Save a checkpoint and fixed held-out reconstructions.

Continue only if:

- held-out per-token NLL improves;
- decoded text becomes more CV-like and source-specific;
- KL does not remain collapsed near zero;
- there is no duplicate/repetitive collapse;
- save/reload and resume work.

### Stage 5: full pilot

Run three epochs initially, with a maximum of five. Checkpoint each epoch and select the lowest validation objective, not automatically the final epoch. Do not select a checkpoint using official-test metrics.

Use tmux and durable logging, for example:

```bash
tmux new -s langvae_cv32_v1
```

The repaired script should support a command resembling:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
TOKENIZERS_PARALLELISM=false \
uv run python -u -m src.finetune_vae \
  --config path/to/versioned_config.yaml
```

Do not assume this CLI exists before implementing and testing it.

Tee logs to a versioned report directory and maintain a status file containing start time, end time, exit code, selected checkpoint, and SHA-256.

Monitor:

- `nvidia-smi`;
- allocated/reserved VRAM;
- system RAM and disk;
- total, reconstruction, KL, and per-token losses;
- beta and active units;
- gradient norm;
- step time and throughput;
- NaN/Inf;
- fixed validation decodes each epoch.

## 12. Acceptance criteria

### Engineering gates

- clean process exit;
- no OOM, NaN, or Inf;
- full configuration and hardware manifest saved;
- deterministic seed and split/window manifests saved;
- per-epoch and best checkpoints saved;
- checkpoint reload succeeds offline in a fresh process;
- 128-dimensional deterministic encodings are finite;
- BERT/GPT-2 are still frozen unless the user explicitly approves otherwise;
- exact source and checkpoint hashes recorded.

Expected LangVAE checkpoint contents include:

- `environment.json`
- `model_config.json`
- `encoder.pt`
- `encoder_cfg.json`
- `decoder.pt`
- `decoder_cfg.json`
- `training_config.json` or an equivalent complete run manifest.

### Optimization gates

- tiny set can be overfit;
- held-out per-token reconstruction NLL improves;
- KL does not remain collapsed near zero;
- decoded outputs become source-specific;
- duplicate and degeneration rates do not worsen;
- results reproduce after reload.

### Scientific gates

Rerun the same reconstruction audit on an untouched official-test sample and compare with the stock metrics listed in Section 3.4.

Since output remains capped at 32 tokens for this pilot, prioritize:

- 32-token target/prefix token F1;
- independent Nomic matched-minus-shuffled margin;
- retrieval accuracy/rank;
- LangVAE cycle similarity;
- country/degree/university/project fact recovery;
- duplicate and degeneration rates;
- qualitative source specificity.

Full-text F1 is secondary because 32 tokens cannot reproduce a 410-token source.

Most importantly, later train and evaluate `g(S|Z)` on the official split. Reconstruction improvement alone does not establish a better causal representation.

## 13. Artifact naming and isolation

Never overwrite:

- `data/latents/talent/langvae/`;
- `data/latents/talent/nomic/`;
- the stock LangVAE checkpoint;
- Nomic outputs.

Use an immutable experiment tag, for example `langvae_cv32_v1`:

- post-training root:
  `models/langvae_posttrain/langvae_cv32_v1/<run-id>/`;
- selected model:
  `models/langvae_posttrain/langvae_cv32_v1/<run-id>/final_model/`
  or an explicit best-checkpoint pointer/manifest;
- reports:
  `reports/talent/langvae_cv32_v1/posttrain/`.

After selecting a checkpoint, configure:

```yaml
encoder:
  variant: langvae
  local_checkpoint: models/langvae_posttrain/langvae_cv32_v1/<run-id>/final_model
  tag: langvae_cv32_v1
  max_len: 512
  device: cuda

finetune_vae:
  sequence_length: 32
```

Here `encoder.max_len=512` is the later full-CV **encoder input** limit. It is not the post-training reconstruction-target length.

The strict latent sidecar records the resolved local checkpoint path and directory hash. Preserve the checkpoint directory and its manifest.

## 14. Re-encoding and downstream rebuild

Any successful post-training changes the posterior projection and therefore defines a new latent coordinate system.

After checkpoint selection:

1. smoke-load the exact checkpoint offline;
2. verify deterministic 128-dimensional encode/decode;
3. rerun the reconstruction audit with the fine-tuned checkpoint;
4. re-encode all factual and official-test counterfactual texts from scratch;
5. validate the new artifact and metadata;
6. train fresh semantic decoders;
7. train fresh latent manipulators;
8. evaluate only under the matching encoder tag.

With the YAML retaining `local_checkpoint` and `tag`, run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run python -m src.pipeline encode --encoder-variant langvae
```

The expected output is:

```text
data/latents/talent/langvae_cv32_v1/z_pairs.pt
data/latents/talent/langvae_cv32_v1/z_pairs.info.json
```

Validate:

- `z` is `(5000,128)`, finite;
- `z_prime` is `(1000,128)`, finite;
- 256 identity test rows are exact copies;
- all IDs and splits align with `pair_index.csv`;
- sidecar contains source hashes, local-checkpoint path, and checkpoint hash.

The CLI `--encoder-variant langvae` changes the variant only; the YAML must retain the intended `local_checkpoint` and explicit tag.

Train both semantic-decoder variants if they remain in the experiment matrix:

```bash
uv run python -m src.pipeline train-decoder \
  --encoder-variant langvae \
  --decoder-variant independent

uv run python -m src.pipeline train-decoder \
  --encoder-variant langvae \
  --decoder-variant autoregressive
```

Then train/evaluate the required manipulator variants under the same encoder and decoder tag:

```bash
uv run python -m src.pipeline train-manipulator \
  --encoder-variant langvae \
  --decoder-variant independent \
  --manipulator-variant baseline

uv run python -m src.pipeline evaluate \
  --encoder-variant langvae \
  --decoder-variant independent \
  --manipulator-variant baseline
```

Repeat for the chosen `pre_additive` and `dist` matrix only after the baseline is validated. Never reuse stock-LangVAE or Nomic `g`/`h_Z` checkpoints; provenance guards should reject them.

Final comparisons should include:

- stock LangVAE;
- fine-tuned LangVAE;
- Nomic;
- held-out semantic-decoder accuracy and calibration;
- reconstruction quality;
- counterfactual semantic recovery;
- latent-shift magnitude and sparsity;
- manipulator faithfulness on the official test split.

## 15. Full-CV reconstruction is a separate design

If the actual requirement is readable reconstruction of the entire 343–499-token CV, stop after the 32-token pilot and discuss architecture before spending compute.

Recommended direction:

- replace the per-output-position 32-slot adapter with a **fixed-size latent prefix**;
- map `z` to a small number `K` of GPT-2 key/value prefix positions;
- use teacher forcing over token-ID targets;
- compute memory-efficient next-token cross-entropy with `ignore_index=pad_id`;
- keep BERT and GPT-2 frozen for the first experiment;
- train the posterior projection and prefix mapper;
- optionally consider LoRA only after the frozen-LM version works;
- generate with EOS-aware stopping and a documented maximum length.

For example, `K=8` or `K=16` keeps parameter count independent of the 512-token output length. This is preferable to expanding the current position-specific adapter to approximately 1.22B parameters.

Suggested staged optimization:

1. initialize from the pinned encoder projection;
2. train the new prefix mapper while initially freezing or using a lower learning rate for `W`;
3. unfreeze `W` with a smaller learning rate only after the decoder begins learning;
4. retain the KL schedule and monitor active units/posterior collapse;
5. use a 24 GB GPU with batch 1–2 plus gradient accumulation as the initial target;
6. benchmark before estimating total runtime.

Changing to this decoder is a research/engineering decision, not a routine continuation of the current checkpoint. Ask the user before making it the main experimental branch.

## 16. Arrival checklist for the GPU agent

- [ ] Confirm this handoff and all modified source files are present.
- [ ] Run `git rev-parse HEAD` and `git status --short`.
- [ ] Confirm commits `dc1a5d6` and `5e6371d` are present in the branch history.
- [ ] Verify canonical input hashes.
- [ ] Locate the claimed dedicated post-training CV corpus, if separate.
- [ ] Recreate the environment; do not copy `.venv`.
- [ ] Verify NVIDIA driver and PyTorch CUDA compatibility.
- [ ] Confirm pinned HF checkpoints load.
- [ ] Copy the stock reconstruction audit and baseline latent artifact if needed.
- [ ] Read and repair `src/finetune_vae.py` before launch.
- [ ] Add the required tests.
- [ ] Pass one-batch CUDA smoke.
- [ ] Pass tiny-overfit and save/reload gates.
- [ ] Benchmark 100 steps and report exact VRAM/time.
- [ ] Run one real epoch and inspect held-out reconstructions.
- [ ] Continue to 3–5 epochs only if the gates pass.
- [ ] Record all hashes, manifests, metrics, and checkpoint paths.
- [ ] Do not overwrite baseline artifacts.
- [ ] Do not train downstream `g` or `h_Z` until the fine-tuned encoder is selected and frozen.

## 17. Stop and ask the user before

Stop and obtain direction before:

- training on any official-test text;
- choosing a different target than explicit <=32-token windows/snippets for the pilot;
- unfreezing BERT or GPT-2;
- changing the latent dimension;
- replacing the decoder architecture;
- attempting literal 512-position adapter expansion;
- using counterfactual latents as direct supervision for the principal latent editor;
- overwriting a stock artifact;
- selecting a checkpoint based on official-test results.

## 18. Expected handback from the GPU agent

Return:

1. code changes and tests;
2. exact data/split/window manifests;
3. GPU model, driver, PyTorch/CUDA versions;
4. selected hyperparameters;
5. peak allocated/reserved VRAM;
6. measured step throughput and total runtime;
7. per-epoch train/validation loss components and KL;
8. checkpoint paths and SHA-256 values;
9. fresh-process save/reload proof;
10. fixed qualitative reconstructions;
11. the complete before/after reconstruction audit;
12. a recommendation to accept/reject the adapted encoder;
13. if accepted, the validated `langvae_cv32_v1` latent artifact and a reminder that all downstream models must be retrained.

Do not report success merely because training exits with code 0. Success requires correctness gates, held-out improvement, reproducible artifacts, and preserved experimental separation.
