# Consolidated handoff: hz-misspec-v1

Prepared on **26 October 2026**, updated after completion on **27 October 2026**, on **cpuserver17**, in
`/home/boustanip/latent_intervention/latent_intervention`, branch `extended-dag`
(base commit `76c7c0d`). This is the current handoff after assembling both
machines' results. It supersedes the transfer instructions in the earlier
remote50, return12 and keep38 handoffs; those remain provenance records.

**All 196 hZ fits, their selected models and training records, all 30 guiding
readers, and the independent evaluation reader are available locally.**
Selections are frozen and verified. The user authorized official-test
evaluation on 26 October; the detached evaluation was launched at approximately
**14:16 CET**. It finished successfully on **27 October at 01:00:23 CET**:
**2,280/2,280 evaluation conditions, zero failures**, in approximately
**10 hours 44 minutes**. All per-unit artifact checksums and complete report
coverage were reverified on 27 October. Nothing was staged, committed or pushed
during assembly, evaluation launch or completion verification.

## Scientific scope

The project studies counterfactual editing of text embeddings:
`Z → g(S|Z) → hS(S'|S,a) → hZ → predicted Z'`.
The question is how symbolic counterfactual commands can produce embeddings
that recover the desired attributes while preserving other information.
This run tests sensitivity to state-space and symbolic-operator misspecification
on the extended CV simulation. It uses **EmbeddingGemma 768D only**, with the
independent S1–S6 guiding readers; it is not a dimension sweep.

The protocol and equations are in the
[workflow document](hz_misspecification.md), based on the broader
[study design](wrong_symbolic_model.md). Implementation is isolated in
`exp/benchmarks/misspecification/`; configuration is
`configs/hz_misspecification.yaml`. Do not substitute models from the older
`talent` experiment or the separate `package` branch: their data, objectives,
training and inference protocols differ.

### Fitting grid and completed results

| Method | Fitting dependencies | Completed fits | Last training epoch range | Reached 500 epochs |
|---|---|---:|---:|---:|
| `semantic_editor` | 13 distinct operators/state spaces × five matching g seeds | 65 | 70–417 | 0 |
| `ot_editor` | same 13 combinations × five matching g seeds | 65 | 204–500 | 14 |
| `state_flow` | six state spaces × five seeds; reused across operators | 30 | 279–395 | 0 |
| `ot_flow` | six state spaces × five seeds; reused across operators | 30 | 452–500 | 28 |
| `ot_gaussian` | six state spaces; analytical, reused across five g seeds | 6 | not applicable | not applicable |
| **Total** | **190 neural + six analytical** | **196** | | |

The independent evaluation reader is additional to these 196 fits. It used the
existing 20-candidate development-selected search and publishes seed 42.
Training completion does not establish convergence at the global optimum.
The selected weights are the best factual-development checkpoints, which may
precede the final training epoch. Preserve each run's history and selected epoch.

S1=(X,T,D,U), S2=(X,D,U), S3=(X), S4=(X,T,U), S5=(X,U),
S6=(X,T,D,U,M). S1/S6 use correct kernels; S2–S5 use the
parameter-preserving restricted **unaware** construction.

| Operator | State space | Evaluated settings |
|---|---|---|
| O1 | S1 | remove T from D/U mechanisms |
| O2 | S1 | X coefficients ×0.5; noise SDs ×0.5, separately |
| O3 | S1 | drop X→D; drop X→U, separately |
| O4 | S1 | block X→D; exact alias of O3/drop-X→D |
| O5 | S6 | false X→M strengths 0, 0.5, 1; zero aliases S6/true |

There are **15 scenario labels but 13 distinct schema/kernel combinations**.
Aliases retain separate report labels without redundant fitting/evaluation.
All four `do(X=k)`, k=0..3, are supported by this experiment's checkpoints.

### Data and information boundaries

The persisted split is **2,800 factual fitting / 200 development / 1,000
official test**. The guiding-reader L subset and independent-reader P subset
contain 1,000 factual fitting units each and are disjoint. P can overlap hZ's
2,800-unit factual pool. Exact IDs are in `protocol.json` and training records;
do not regenerate a split from the older 3,200/400/400 protocol.

True factual states are available in this regime. The editors use inferred
g/hS target mixtures for their semantic loss and factual X for the identity
mask; flows/Gaussian fit factual state-conditioned distributions. Counterfactual
targets, the independent evaluation reader and P/L/H/A preservation labels
never enter hZ fitting or checkpoint selection. This is **not** an end-to-end
1,000-label or cross-fitted experiment.

Inputs and encodings:

- Simulator: `exp/sim_talent_motivation/config.yaml` and
  `data/sim_talent_motivation/`.
- Texts: `data/text_talent_motivation/`.
- Encoder configuration: `configs/encoding_talent_motivation.yaml`, pinned
  `google/embeddinggemma-300m` revision
  `57c266a740f537b4dc058e1b0cda161fd15afa75`.
- Embeddings: `data/latents/talent_motivation/embeddinggemma_768/z_pairs.pt`
  (4,000 factual and 1,000 paired total-effect counterfactual embeddings),
  `z_floor.pt` (200 same-state text-variation pairs), and
  `z_factual_vb.pt` (500 held-out-wording examples).
- Complete frozen input inventory and hashes: `protocol.json` below.

## Consolidation and integrity

The original allocation was **146 local + 50 remote**. Twelve untouched remote
fits were returned to cpuserver17, leaving **146 + 12 + 38 = 196**. The 38
remaining fits completed on cpuserver16 on 25 October and were imported on
26 October without replacing differing local artifacts.

Assembly verified the archive checksums, safe member paths, every manifest
payload hash, assignment/allocation identities, and existing-file compatibility.
The merge kit was byte-identical to this checkout. All 196 model/report
checksums were checked and all 196 selected checkpoints successfully loaded.
The 38 remote selected state dictionaries also match their saved best resume
weights; histories, signatures and selected epochs agree. The current
evaluator rechecks the frozen selection and source/input hashes before and
after evaluation.

Let **R** denote
`reports/talent_motivation/hz_misspecification/hz-misspec-v1`.

| Record | Location under R | Identity |
|---|---|---|
| Frozen scientific protocol | `protocol.json` | signature SHA-256 `142a08922146ede419ac0a9d25b1a61bc68d54f0cda39b6d528f44c7ee8e2a1d` |
| Complete frozen deployment selection | `selection.json` | record SHA-256 `7ddcffe6565bd4516bc77fbeb40834918852dfa68bde3be601eb4457e4b7cdca` |
| Remote import receipt | `shards/remote50-20261020/return12-20261022/merge-keep38.json` | record SHA-256 `97a206b8353004e183fa25834a9e27b08cbc3b0e59b0fdfc4ebb41a9806d3825` |

These are signature/record digests, not interchangeable with SHA-256 of the
JSON file bytes. Use the existing `read_record`/`digest` utilities to verify.
The protocol covers 48 input files and 103 source files. Its recorded package
versions are torch 2.12.1, NumPy 2.5.1, pandas 3.0.3 and PyYAML 6.0.3.
Continue with the existing `.venv`; avoid upgrading it mid-evaluation.

### Selected models and compact records

Model paths follow:

```text
models/talent_motivation/embeddinggemma_768/g-independent/
  S#/h-<method>/runs/hz-misspec-v1/<fitting-operator>/seed-N/selected/latent_intervention.pt
```

Fitting operators use `O2--coef_X_x0.5`-style directory names for editors,
and `factual` for flows/Gaussian. The authoritative exact paths, hashes and fit
signatures are in `R/selection.json`. Use the experiment's
`exp.benchmarks.misspecification.models.load`, not a legacy model loader.

Training records live at
`R/fits/S#/<method>/<fitting-operator>/seed-N/training.json`.
They contain architecture/capacity, fitting/development IDs, input/source
signatures, complete histories, runtime and checkpoint selection information.

The 30 guiding readers are at
`models/talent_motivation/embeddinggemma_768/g-independent/S#/runs/g-independent-s1-s6-v1/final/seed-N/semantic_decoder.pt`.
The independent nine-head reader is at
`models/talent_motivation/embeddinggemma_768/g-independent/evaluation_reader/runs/hz-misspec-v1/seed-42/semantic_decoder.pt`;
its search/selection/calibration records are under `R/evaluation_reader/`.

### Backups and resumability

The received bundles and full remote training-record backup remain at:

```text
/home/boustanip/research/hz-misspec-keep38-return-20261026/
  hz-misspec-keep38-results.tar.gz
  hz-misspec-keep38-results.tar.gz.sha256
  hz-misspec-keep38-merge-kit.tar.gz
  hz-misspec-keep38-merge-kit.tar.gz.sha256
  remote-report-backup/
```

Results archive SHA-256:
`b67418f1731ed1f10fd85356591c46444441ac5331892e23c8b94a14ca41f319`.
Its 81 manifest-listed payload files include all 38 models and training records.
The separate backup contains the 38 optimizer/RNG resume snapshots and logs;
they were intentionally not included in the small deployment bundle.

The verified extracted staging copy is Git-ignored under
`R/shards/remote50-20261020/return12-20261022/transfer/keep38-stage.hisIMC/`
(`incoming/` and `merge-kit/`). Preserve it and the original bundles.
The canonical report tree has 152 local neural resume snapshots; the other
38 neural snapshots are in the remote backup. Thus all 190 neural training
resumes are retained, but are not all in one directory. Six analytical fits
have no optimizer snapshot. All 196 selected deployments and compact training
records are in their canonical paths and sufficient for evaluation.

Historical top-level/shard status files may still say "delegated". The complete
frozen selection and verified per-fit records are the current source of truth.
Do not relaunch an old training queue because of these historical labels.

## Official-test evaluation completed

The detached tmux session **`hz-misspec-v1-evaluation`** exited normally after
completion. It used four CPU workers, one numerical thread each, on cpuserver17.
No evaluation process remains running and no restart is needed.

The first full OT-flow condition took **405 seconds** for 1,000 units; the
other methods/reference initially took approximately **2–6 seconds** each.
The complete evaluation and report generation took **38,632 seconds**
(approximately **10 hours 44 minutes**), finishing earlier than the initial
12–18 hour estimate. All five methods and no-edit completed 380 conditions each.
Short timing-only checks on eight S1/S6 test units also returned finite outputs;
their diagnostics are in the local `evaluation/runtime_probe.json`. They did
not select models or change the evaluation protocol.

The added runner `scripts/hz_misspecification_evaluate.py` schedules the existing
`evaluate_condition` unchanged. It holds the same global evaluation lock and
retains native per-condition locks, cache identities and checksums. It does not
change any frozen scientific source/configuration or fitted weights. Its own
hash, host, progress and timings are recorded in `R/evaluation/queue.json`.

The full queue is **19 canonical scenario/target pairs × five seeds × four
actions × six methods/references = 2,280 conditions**. The sixth is `no_edit`.
Each condition uses all 1,000 official-test units and the fixed 64 IID coupled
draws, evaluation seed 20260930, batch size 32. OT flow keeps the existing
RK4 step-doubling/tolerance checks. No samples, states or difficult cases are
dropped to shorten evaluation. Failed conditions are recorded with tracebacks;
other conditions continue and successful results remain resumable.

Total targets are evaluated everywhere. Extra symbolic references are S1:
`ps_D`, `ps_DU`; S3: `ps_DU`; S4, S5 and O3/drop-X→D (also O4): `ps_D`.
Paired embedding scores exist only for matching `(id, do_value, total)`
targets—one encoded nonidentity action per test unit. Path-specific embedding
scores are unavailable, not zero.

Interpret own-g KL as assumed-command consistency. Interpret independent-reader
attribute recovery/preservation together with that reader's per-head factual
accuracy/calibration. Its heads are X,T,D,U,M,P,L,H,A, excluding Y.
Neither score alone establishes true individual counterfactual identification.
Primary predictions retain all rare/unseen posterior mass. Reports distinguish
all/identity/nonidentity groups, unsupported-state mass, no-edit performance,
and same-state text variation. Five-seed SD and 2,000-resample paired-unit
95% intervals represent different sources of variation.

### Monitoring, resume and outputs

Run from the repository root:

```bash
tmux list-sessions
tail -n 30 reports/talent_motivation/hz_misspecification/hz-misspec-v1/evaluation/run.log
.venv/bin/python -m json.tool reports/talent_motivation/hz_misspecification/hz-misspec-v1/evaluation/queue.json
```

`R/evaluation/workers/<pid>.json` records each worker's current condition and
start time. While a long condition is running, the completed count can remain
unchanged. The tmux session exits once the runner finishes. Completion requires
`queue.json` status **complete**, **2,280 completed**, **zero failures**, and
the consolidated report. A disappearing session alone does not prove success.

After an interruption, and only once the previous session/processes have
stopped, rerun:

```bash
bash scripts/run_hz_misspecification_evaluation.sh hz-misspec-v1 4
```

The same command verifies and reuses completed conditions. Do not run the
native sequential evaluator concurrently; the shared lock rejects this.
No training command is needed.

Outputs under R:

| Artifact | Contents |
|---|---|
| `evaluation/references.json` | factual test reader calibration, outside-P confusion reference, same-state text-variation reference |
| `evaluation/<scenario>/<method>/seed-N/action-k/<target>/metrics.json` | compact condition metrics, matching rules, runtime and diagnostics |
| adjacent `per_unit.pt` | exact IDs, per-unit metrics and identity/availability masks for paired comparisons |
| `evaluation/queue.json`, `evaluation/run.log` | operational progress, failures and timings |
| `results.json` | final numerical summaries, seed SDs, paired intervals and method contrasts |
| `summary.md` | final human-readable comparison, including scenario aliases |

The wrapper automatically generates `results.json`/`summary.md` after the
condition queue. If any conditions fail, those reports are partial and the
queue exits unsuccessfully. Individual summary groups should contain **20/20**
action/seed conditions. To regenerate a report after successful evaluation:

```bash
.venv/bin/python -m exp.benchmarks.misspecification.run report
```

### Launch and completion verification

- Frozen protocol, all 196 selected model/report hashes and evaluation reader
  verified before launch; no fitting or checkpoint reselection performed.
- Scheduling test confirms exact native condition coverage, deduplication of
  scenario aliases and no duplicate output keys.
- Existing misspecification regression suite plus scheduling, cache-resume and
  failure-recording tests: **56 passed** across the targeted test runs.
- Ruff checks and launcher Bash syntax check passed.
- All **2,280 expected condition records** are present, with no missing or extra
  conditions, and every per-unit artifact checksum matches its record.
- All **114 canonical summary groups** contain **20/20** action/seed conditions,
  covering seeds 42–46. The final report includes **285 paired method contrasts**.
- Frozen scientific source, inputs, all selected models/training records and
  the independent reader remain unchanged after evaluation. The queue reports
  **complete**, with **zero failures**; `results.json` and `summary.md` are saved.

## What remains outside this run

This is a completed **fitting and official-test evaluation** experiment for the explicitly approved first
regime, not the whole paper study. Remaining study extensions include:

- Additional encoder dimensions, cross-fitted inferred-state hZ training and
  reader label-budget sweeps. Those require new fitting protocols/results;
  this run cannot be relabeled as such retrospectively.
- The held-out-wording reader test using the existing 500 V_B embeddings.
  Their data are present; this queue does not execute that separate gate.
- More operator grids (estimated/refitted parameters or label errors),
  downstream Y prediction, additional reference methods and residual/diagnostic
  analyses from the broader study design.
- An operator-agnostic-editor experiment, if the paper requires it. These
  editors were deliberately trained separately per assumed operator; transport
  models were reused across operators.

Do not tune these frozen fits using the newly exposed official-test scores.
New scientific variants need a separate protocol/run and an explicit account
of prior test exposure. Previously studied projected C-sym and old distributional
OT-editor results are historical experiments, not the clean editors in this run.

## Git and transferring the consolidated result

Git policy allows seed-42 hZ selected models, the independent evaluation reader,
and compact records for all seeds. Other hZ seed checkpoints, optimizer
snapshots, logs and per-unit arrays remain ignored. **A Git push alone is not
a complete backup or enough to reproduce five-seed evaluation elsewhere.**

For another machine, transfer all files named by `selection.json` and
`protocol.json` (including the five-seed guiding readers and input artifacts),
plus the complete evaluation tree after completion. Preserve the local/remote
resume backups separately if future training continuation matters. Verify
hashes before and after transfer; do not overwrite differing artifacts silently.
The current work is intentionally unstaged and uncommitted for joint review.
