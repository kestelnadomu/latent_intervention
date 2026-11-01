# GPU compute and wall-clock handoff for the paper

Collected on **1 November 2026**, on the GPU server, from the completed runs
behind [package-v1](text_realizer_package_evaluation.md) and
[renderers-v1](decoder_renderers_handoff.md). Source checkout: `7f16f1c`.
This supplements the other machine's encoder/renderer compute audit; it does
not replace that audit or rerun any experiment.

## What this resolves

- The four final decoder fits used approximately **17.24 GPU job-hours for
  the training stages**, including validation and checkpoint I/O.
- Including startup/preflight, development diagnostics and export, their
  workflows used approximately **31.96 GPU job-hours** and finished within
  a **15.71-hour elapsed window**, with staggered starts on separate GPUs.
  Replace the earlier bounds of 26 hours and 93 GPU-hours with these values.
- The two test-generation campaigns produced **19,000 CVs**, with **79.03
  aggregate GPU generation-loop hours**. Their elapsed durations were about
  **10 h 45 min** and **10 h 42 min**, respectively.
- These successful decoder workflows, generation campaigns and GPU scoring
  amount to approximately **111 GPU job-hours**. The sum of their elapsed
  phase windows is approximately **37.4 hours**, excluding gaps between
  launches. Neither figure is a total for all project development work.

**Precision:** these are defensible rounded runtime figures, not a hardware
utilization or energy measurement. Final completion times are logged in JSON.
Training-stage boundaries must be reconstructed from the original local file
modification times: the launcher logs have epoch numbers but no timestamps,
and `status.json` overwrites earlier heartbeats. Consequently, do not describe
training durations as exact instrumented measurements. The timestamp evidence
is copied below because Git does not preserve file modification times.

## Hardware and runtime

Observed using `hostname`, `lscpu`, `free -b`, `nvidia-smi`, `/etc/os-release`
and `uname -r` on 1 November:

| Item | GPU server |
| --- | --- |
| Host | `gpuserver.stat.uni-muenchen.de` |
| CPU | **2 × Intel Xeon Gold 6226R @ 2.90 GHz** |
| CPU topology | 16 physical cores/socket; 32 physical cores and 64 logical CPUs total |
| System memory visible to Linux | 134,545,379,328 bytes = **125.3 GiB**; installed DIMM capacity was not queried |
| GPUs | **4 × NVIDIA GeForce RTX 2080 Ti**, each 11,264 MiB = **11 GiB VRAM** |
| NVIDIA driver | `550.163.01` |
| OS / kernel | CentOS Linux 8 / `4.18.0-193.19.1.el8_2.x86_64` |
| Recorded training runtime | Python 3.12.12, PyTorch `2.6.0+cu124`, Transformers 4.57.6, PEFT 0.17.1 |
| Numerical settings | FP32, eager attention, gradient checkpointing; microbatch 2, effective batch 32 |
| CPU threads for training | `torch.set_num_threads(1)`; launchers set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` |

The GPU type and runtime versions are also recorded in the historical run
plans/preflights. CPU model, RAM, OS and current driver were read now, rather
than recovered from a historical hardware inventory. These are the GPU
server's specifications; **do not attribute them to `cpuserver17`, Leo's
machine, or the laptop used for baseline scoring**.

## Four decoder training workflows

The four checkpoints and export manifests were verified against
[`configs/text_realizer_package_eval.yaml`](../../configs/text_realizer_package_eval.yaml).
All are **factual-only `talent_motivation` fits**, seed 42, using 2,800 factual
fit CVs and 200 development CVs. The 1,000 official test CVs did not enter
training. They are not the earlier oracle-exposed or old-corpus models.
Only the prefix projector and LoRA parameters were trained; the base decoder
and EmbeddingGemma encoder remained frozen.

Each row was one independent single-GPU job. Durations below are rounded to
the nearest second; small discrepancies in sums of displayed values are rounding.

| Decoder | Physical GPU | Epochs run / selected | Startup + preflight | Training stage | Diagnostics + export | Whole workflow |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256-D / 270M | 2 | 14 / 9 | 00:25:47 | 03:52:52 | 02:29:34 | 06:48:14 |
| 256-D / 1B | 0 | 11 / 6 | 00:36:52 | 04:51:10 | 03:37:13 | 09:05:15 |
| 768-D / 270M | 1 | 13 / 8 | 00:22:02 | 03:38:20 | 02:39:12 | 06:39:34 |
| 768-D / 1B | 3 | 11 / 6 | 00:34:15 | 04:52:19 | 03:57:50 | 09:24:24 |
| Aggregate job-hours | | | **1.98** | **17.24** | **12.73** | **31.96** |

The 30-epoch setting was a maximum, not the executed budget. Early stopping
ended the fits after 11–14 epochs. Reporting just the time to the selected
epoch would omit the subsequent epochs needed to establish early stopping.

The full workflows overlap from **27 October 09:33:17 to 28 October 01:15:48
Europe/Berlin (CET, UTC+1)**: approximately **15 h 43 min**. The 768-D jobs
started around 15:51, approximately six hours after the 256-D jobs. Therefore
neither adding all four job durations nor dividing that sum by four gives
the actual elapsed schedule. The largest individual workflow took 9 h 24 min.

Definitions used for this reconstruction:

- **Startup + preflight:** `mtime(training_plan.json) - mtime(plan.json)`.
  This includes tests/cache checks and the successful run's readiness checks;
  it is an approximate GPU job occupancy allowance, since some startup is CPU-only.
- **Training stage:** `mtime(training.json) - mtime(training_plan.json)`.
  This includes model loading, optimization, per-epoch development NLL and
  saving. The immutable training plan is created before model loading, and
  `training.json` is written after the final epoch. It is not pure GPU kernel time.
- **Diagnostics + export:** final `status.json.heartbeat` minus
  `mtime(training.json)`. This includes reload verification, five conditioning
  controls on 200 dev CVs, cycle encoding, retrieval/other metrics and export.
- **Whole workflow:** final heartbeat minus `mtime(plan.json)`. Initial data
  preparation before the plan, dependency downloads, and earlier failed runs
  are outside this interval.

The boundaries are consistent with each run's saved preflight-ready heartbeat,
last epoch history, launcher log and reload/export files. No interrupted
continuation is evident in these four successful logs. There is no separate
cumulative per-update timing ledger to recover pure optimization time.

### Portable boundary evidence

All timestamps in this table are **UTC**, copied from the original GPU-server
files on 1 November. The first three columns are filesystem modification
times; the final column is a timestamp stored inside JSON. Use these saved
values when working on another machine, not the modification times of a new
Git checkout or a copy that did not preserve timestamps.

| Decoder | `plan.json` mtime | `training_plan.json` mtime | `training.json` mtime | Final `status.json.heartbeat` |
| --- | --- | --- | --- | --- |
| 256-270M | 2026-10-27T08:33:17.032702Z | 2026-10-27T08:59:04.410810Z | 2026-10-27T12:51:56.324070Z | 2026-10-27T15:21:30.592727Z |
| 256-1B | 2026-10-27T08:58:43.680063Z | 2026-10-27T09:35:35.529063Z | 2026-10-27T14:26:45.467180Z | 2026-10-27T18:03:58.813616Z |
| 768-270M | 2026-10-27T14:51:11.719211Z | 2026-10-27T15:13:13.532110Z | 2026-10-27T18:51:33.396344Z | 2026-10-27T21:30:45.493242Z |
| 768-1B | 2026-10-27T14:51:24.761051Z | 2026-10-27T15:25:39.754011Z | 2026-10-27T20:17:58.454021Z | 2026-10-28T00:15:48.465962Z |

The report directories, relative to `reports/talent_motivation/`, are:

| Decoder | Report directory |
| --- | --- |
| 256-270M | `embeddinggemma_256/text-realizer/package-256-20261027-0934-eager/` |
| 256-1B | `embeddinggemma_256/text-realizer-1b/package-256-1b-20261027-0958/` |
| 768-270M | `embeddinggemma_768/text-realizer/package-768-270m-20261027-1551/` |
| 768-1B | `embeddinggemma_768/text-realizer-1b/package-768-1b-20261027-1551/` |

Each contains `plan.json`, `preflight/status.json`, `preflight/ready.json`,
`training_plan.json`, `training.json`, `history.json`, `launcher.log`,
`reload.json`, `frozen.json` and final `status.json`. The corresponding model
directory under `models/` contains `inference/model.pt` and `manifest.json`.
All four model/export hashes, the frozen manifests' report hashes, and epoch
counts were checked. Full checkpoint/export hashes are already pinned in the
linked evaluation config; this handoff requires no weights or CV text transfer.

## Test generation and scoring

The source root for the following records is
`reports/talent_motivation/text-realizer-evaluation/`.

| Phase | Work | Elapsed wall-clock | Aggregate GPU job/loop hours |
| --- | --- | ---: | ---: |
| `package-v1` generation + initial CPU scoring | Four decoders × oracle/predicted vectors × 1,000 IDs; four GPU workers | **10:45:25** | **36.78** generation-loop hours |
| `package-v1` BERTScore follow-up | 8,000 pairs; RoBERTa-large; four GPU workers | **about 1 min 28 s** | about **0.10** job-hours including startup |
| `package-v1` VADER | 8,000 pairs, CPU | **28.94 s** | None |
| `package-v1` changed/preserved facts | 8,000 pairs, CPU | **8.46 s** | None |
| `renderers-v1` generation | 11 distinct vector conditions × 1,000 IDs; four GPU shards | **about 10:42:00** | **42.25** generation-loop hours |
| `renderers-v1` successful scoring retry | Lexical/factual/sentiment scores plus BERTScore; one GPU | **10 min 55 s** | about **0.18** job-hours |

Generation-loop hours are computed as
`sum(progress.total * progress.seconds_per_cv) / 3600` over completed shards.
The saved source starts its timer after model loading and includes output I/O.
The eight package shards and 44 renderer shards each ran from their first ID
to completion without a recorded resume; their logs and file-time spans agree
with this interpretation. This formula would not recover full runtime after
a partial resume, because `seconds_per_cv` describes only the latest invocation.
All **52 completion records and prediction-file checksums** were reverified.

For cross-checking the sums:

| Campaign | Individual worker generation-loop hours | Sum |
| --- | --- | ---: |
| `package-v1` | 256-270M: 7.566936; 256-1B: 10.594638; 768-270M: 7.896190; 768-1B: 10.726445 | **36.784209 h** |
| `renderers-v1` | Shards 0–3: 10.695970, 10.623927, 10.637946, 10.289482 | **42.247325 h** |

Timing sources and boundaries:

- `package-v1/launch-20261028-115728/plan.json`: `created_at` is
  **28 October 11:57:28 CET**. Final `status.json.heartbeat` is
  **22:42:52.740080 CET**, with all four workers and CPU scorer exiting zero.
  Last generation `.done.json` mtime is 22:41:51.690840 CET; initial CPU
  scoring and controller overhead account for approximately the last minute.
- `package-v1/predictions/*/*/*.progress.json`: eight completed 1,000-CV
  cells, supplying the first generation sum.
- `package-v1/bertscore-roberta-large/launch.json`: requested at
  **29 October 08:06:29 CET**. Latest worker JSON heartbeat is
  **08:07:57.304114 CET**. The four workers' loop timers sum to 0.07275 GPU-h;
  approximately 0.10 GPU-h allows for their parallel startup. Aggregation
  adds a small uninstrumented overhead.
- `package-v1/vader/summary.json` and `fact-changes/summary.json` contain
  explicit `elapsed_seconds` for the CPU follow-ups.
- `renderers-v1/launch.json`: launched **30 October 23:10:04.272648 CET**.
  Last generation `.done.json` mtime is **31 October 09:52:04.472915 CET**.
  Its 44 `predictions/768-270m/*/*.progress.json` files supply the second sum.
- `renderers-v1/score-launch.json` and
  `logs/score-common-cohort-20261031-111059.log`: successful scoring ran
  **31 October 11:10:59–11:21:54 CET**, exit zero.

The 12 reported renderer conditions require only 11 sets of generated texts:
group OT's true-target condition aliases its identical sampled condition.
Do not count that alias as another 1,000 decoding operations. Sampled state
flow was already generated in `package-v1`, and is not rerun in `renderers-v1`.

The initial renderer scoring attempt failed at **09:59:10 CET**, roughly
seven minutes after generation, because one generated CV exceeded RoBERTa's
512-token limit. This is additional failed-attempt overhead, excluded from
the successful-scoring row. The retry did not regenerate CVs. It used a common
999-ID BERTScore cohort, excluding ID 2333; all other metrics and CV exports
retain all 1,000 IDs. Idle time before the retry is not compute time.

## What to say in the paper

Suggested decoder paragraph, with deliberately rounded values:

> Decoder experiments ran on a server with two Intel Xeon Gold 6226R CPUs
> (32 physical cores in total), 125 GiB of OS-visible RAM, and four NVIDIA
> RTX 2080 Ti GPUs with 11 GiB VRAM each. We trained four factual-only
> projector-plus-LoRA decoders, combining 256- and 768-dimensional embeddings
> with Gemma 3 270M and 1B backbones. Each used one GPU in FP32, gradient
> checkpointing and an effective batch size of 32. Training with validation
> took approximately 3.6–4.9 hours per model (17.2 GPU-hours in total).
> Including preflight and development diagnostics, the four workflows used
> approximately 32 GPU-hours and completed within 15.7 elapsed hours with
> staggered starts. The subsequent evaluations generated 8,000 and 11,000
> CVs in two four-GPU campaigns taking approximately 10.8 and 10.7 hours,
> respectively, or 79 aggregate GPU generation-hours. GPU-based scoring
> added approximately 1.5 minutes on four GPUs and 11 minutes on one GPU;
> the separate CPU sentiment and fact-change analyses each took under a
> minute. Training durations were reconstructed from saved artifact
> timestamps; generation totals use recorded per-worker timings.

If one rounded decoder budget is needed, use **approximately 111 GPU
job-hours for these successful runs**, including their readiness and
development diagnostics, with the accounting qualifications above. Summing
the elapsed windows of the training campaign, two generation campaigns and
successful scoring yields **about 37.4 hours**. This is neither 37.4 hours
with all four GPUs continuously busy nor the calendar interval from the
first launch to the last score. Do not multiply it by four.

## Remaining gaps and exclusions

- **Encoder server:** `cpuserver17` CPU model/RAM and independently confirmed
  host identity are not established by this GPU inspection. Obtain `hostname`,
  `lscpu` and `free -b` on that machine. The encoder durations in the supplied
  cross-branch audit were not re-audited here.
- **CPU renderer fits:** Leo's machine/CPU and missing operator-sampled
  evaluation/fairness-analysis timings still require his records. Do not
  assign this server's hardware to those fits. CPU wall-hours with four
  threads are not the same quantity as four CPU core-hours per wall-hour
  unless utilization is measured or an explicit allocation convention is used.
- **Laptop baselines:** the supplied approximately 4 h 16 min is outside the
  GPU budget and was not independently verified here. Keep its hardware and
  elapsed time separate in the paper's compute account.
- **Development:** old `talent`-corpus pilots, LangVAE fine-tuning, oracle-exposed
  prototypes, failed preflights (including `package-256-20261027-0054`), smoke
  runs, and debugging are not included in the successful-run totals. This
  handoff does not establish the total development compute budget.
- **Other costs:** LLM-API corpus generation, downloads, environment setup,
  queue/idle gaps, and unrecorded preparation overhead are excluded. No
  energy, FLOP or hardware-utilization measurements are available.

Only this Markdown handoff is added. Original logs, model weights, generated
CVs, result values and experiment code are unchanged; nothing is staged,
committed or pushed by this documentation task.
