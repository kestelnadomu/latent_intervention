# Agent handoff: 50 remote misspecification fits

Prepared 20 October 2026. This is a **training-only shard of an already-running
experiment**, not a new experiment or a request to redesign the methods.

## Outcome required from the receiving agent

Verify the supplied bundle, launch exactly its 50 assigned jobs in detached tmux
on this machine, verify initial stability, and tell the user the estimated finish
time. When training finishes, verify completion and export the selected
checkpoints and compact training records for return to the original machine.
Do not launch official-test evaluation. Do not stage, commit or push.

The original machine continues its complementary jobs. Never use the old
`train --all` command or `scripts/run_hz_misspecification.sh` on this remote
machine: those would duplicate work.

## Project and scientific context

The project studies counterfactual editing of text embeddings: given a factual
CV embedding Z, construct a distribution over edited embeddings Z' consistent
with an intervention on symbolic attributes. A frozen independent-head reader
g maps embeddings to symbolic state probabilities; analytic h_S supplies the
assumed symbolic intervention kernel. The synthetic CV data provide truth for
later evaluation, but paired counterfactual targets are not editor supervision.

This experiment uses the new `talent_motivation` corpus, existing
**EmbeddingGemma 768D** encodings, six state spaces S1–S6, and operator
misspecifications O1–O5. It compares `semantic_editor`, `ot_editor`,
`state_flow`, `ot_flow` and `ot_gaussian`. All flows/Gaussians and the independent
evaluation reader have already finished on the original machine. The remaining
expensive jobs are editors. See [the experiment workflow](hz_misspecification.md).

Both editors use target-only conditioning and the complete target marginal
q(s'|z,a) = sum_s g(s|z) h_S(s'|s,a). `semantic_editor` is a flexible residual
map; `ot_editor` is a target-conditioned convex-gradient map, **not** the retired
distributional minimax OT editor and not projected C-sym. They minimize the
same objective: semantic KL + 0.1 movement + 0.05 support + identity.

The protocol is frozen:

- Run ID: `hz-misspec-v1`; tier 1, approximately 331,136 trainable parameters.
- Persisted 2,800 factual fitting / 200 development / 1,000 official-test IDs.
  True factual states are available; this is not a limited-label end-to-end claim.
- Seeds 42–46, paired with the matching already-trained g checkpoints.
- Every editor fits all four `do(X=k)` actions with equally weighted losses;
  four backward passes precede each optimizer step.
- Factual-only fitting/selection: no P/L/H/A supervision, audit readers,
  counterfactual states/embeddings, teacher or test-set selection.
- AdamW, LR 3e-4, weight decay 1e-5, batch 32, clipping 5, max 500 epochs.
  Early stopping: patience 30, minimum improvement 1e-7. Plateau scheduler:
  patience 10, factor 0.5, minimum LR 1e-6. Restore the best development checkpoint.
- CPU execution, one numerical thread per worker. A GPU is not needed or used.
  Do not convert this shard to GPU fitting or modify precision/batch sizes.

The bundle contains some counterfactual files because the existing provenance
checks hash them. Their presence is not authorization to use them for fitting,
selection, inspection of predictive performance, or test evaluation.

## Exact assignment

Authoritative machine-readable assignment:

`reports/talent_motivation/hz_misspecification/hz-misspec-v1/shards/remote50-20261020/assignment.json`

Its `assignments.remote` list contains **50 jobs**, ten groups of five seeds:

| State space | Method | Fitting operator | Seeds |
|---|---|---|---|
| S6 | ot_editor | O5/c0.5 | 42–46 |
| S6 | ot_editor | O5/c1 | 42–46 |
| S6 | semantic_editor | O5/c0.5 | 42–46 |
| S6 | semantic_editor | O5/c1 | 42–46 |
| S1 | ot_editor | O1/unaware | 42–46 |
| S1 | ot_editor | O2/coef_X_x0.5 | 42–46 |
| S1 | ot_editor | O2/noise_sigma_x0.5 | 42–46 |
| S1 | ot_editor | O3/drop_X_D | 42–46 |
| S1 | ot_editor | O3/drop_X_U | 42–46 |
| S1 | semantic_editor | O2/noise_sigma_x0.5 | 42–46 |

S1 observes X,T,D,U; S6 additionally observes M. O1 removes T from the D/U
mechanisms. O2 separately halves X coefficients or noise standard deviations.
O3 removes one indicated direct X edge. O5 introduces a false X→M edge with the
documented matched-noise construction. O4 is already an alias of O3/drop_X_D;
O5/c0 is already an alias of S6/true. Do not add duplicate fits for aliases.

All 50 jobs were verified **unstarted at assignment time**. The original
machine owns all other 146 jobs, including previously completed fits and any
interrupted fits resumed from their epoch snapshots. The assignment is exhaustive
and disjoint. Do not run `split` again or change the assignment.

## Transfer and environment

Two files are supplied from the original machine:

- `hz-misspec-remote50-inputs.tar.gz`
- `hz-misspec-remote50-inputs.tar.gz.sha256`

They are stored there under
`reports/talent_motivation/hz_misspecification/hz-misspec-v1/shards/remote50-20261020/transfer/`.

The archive includes the exact working source (including work not yet committed),
configurations, encodings, corpus/provenance files, **all 30 guiding-reader
checkpoints including Git-ignored seeds**, the original protocol/preflight,
assignment, queue wrapper and this handoff. It excludes credentials, `.git`,
virtual environments, encoder model weights, prior completed h_Z fits and
optimizer snapshots. A plain Git clone/pull alone is therefore not sufficient.

Use a **new, separate directory**, not a dirty checkout or a directory shared
with the original running queue. From the directory containing the archive:

```bash
sha256sum -c hz-misspec-remote50-inputs.tar.gz.sha256
mkdir hz-misspec-remote50
tar -xzf hz-misspec-remote50-inputs.tar.gz -C hz-misspec-remote50
cd hz-misspec-remote50

# Verify all extracted files, including the portable dependency sources.
python3 - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path("TRANSFER_MANIFEST.json").read_text())
expected = manifest.pop("record_sha256")
actual = hashlib.sha256(json.dumps(
    manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
).encode()).hexdigest()
assert actual == expected, "transfer manifest checksum mismatch"
for name, expected in manifest["files"].items():
    path = Path(name)
    assert not path.is_absolute() and ".." not in path.parts
    assert path.resolve().is_relative_to(Path.cwd().resolve())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
assert manifest["kind"] == "inputs"
print("Transfer verified:", len(manifest["files"]), "files")
PY

# Install from the supplied lockfile; do not regenerate it.
uv sync --frozen --python 3.12
```

The originating interpreter is Python 3.12.8. The immutable protocol requires
these package distribution versions, which the shard runner verifies:
`torch==2.12.1`, `numpy==2.5.1`, `pandas==3.0.3`, `pyyaml==6.0.3`.
Use the supplied lockfile for other dependencies. If the environment cannot
provide the required versions, stop and report the mismatch rather than editing
the protocol or upgrading its hashes. Network access may be needed for package
installation; fitting itself is offline and needs no API/Hugging Face token.

## Verify and launch

Check allocated CPU cores and RAM, including scheduler/cgroup limits. Use one
numerical thread per worker; choose workers based on **allocated physical CPU
cores**, not the GPU count. Budget approximately 2 GiB per worker plus overhead.
For example, use eight workers if at least eight physical cores and sufficient
memory are allocated; otherwise choose a smaller value.

```bash
# Inspect CPU allocation without starting training.
lscpu
.venv/bin/python scripts/hz_misspecification_shards.py status --shard remote

# Example for an eight-core allocation. Preflight runs before tmux launch.
bash scripts/run_hz_misspecification_shard.sh remote 8

tmux ls
.venv/bin/python scripts/hz_misspecification_shards.py status --shard remote
tail -n 20 reports/talent_motivation/hz_misspecification/hz-misspec-v1/shards/remote50-20261020/remote/run.log
```

The tmux session is `hz-misspec-v1-remote`. The launcher checks the frozen
protocol/input/source/package hashes, validates the assignment, and runs
forward/backward numerical checks for its ten method/operator combinations
without optimizer steps before launching actual training.

The original YAML's `workers: 4` is deliberately **not edited**. The new wrapper's
worker argument is an execution-only override, recorded with hostname, CPU
affinity, platform information and orchestration hashes in
`shards/remote50-20261020/remote/executions/`. The scientific fitting settings,
metadata and signatures remain unchanged. Different CPU hardware may produce
small floating-point differences; do not promise cross-host bitwise identity.

Inspect the first epoch histories and process CPU usage before reporting that
the queue is healthy. The coordinator's `run.log` prints completed jobs;
per-epoch progress is in the existing per-job logs:

`reports/talent_motivation/hz_misspecification/hz-misspec-v1/logs/S#/METHOD/OPERATOR--SETTING/seed-N.log`

Status reports `started/resumable` from artifacts, not proof of a live process;
also inspect tmux/processes. Do not start a second competing queue. Per-host
queue locks and per-fit locks reject concurrent writers, but **separate hosts
are protected by the explicit assignment**, not by a distributed lock.

To resume after a stopped session, rerun the same launcher with the same shard.
It verifies completed records/checkpoint hashes, skips those jobs, and resumes
incomplete fits with model/optimizer/scheduler/RNG snapshots. At most the
unfinished epoch is repeated. Do not erase snapshots, alter sources/configs or
change run ID to bypass a failed consistency check.

If tmux is unsuitable on a batch-scheduled cluster, use the scheduler's durable
job allocation instead. In the allocated directory, run:

```bash
.venv/bin/python scripts/hz_misspecification_shards.py check --shard remote --numerical
.venv/bin/python -u scripts/hz_misspecification_shards.py train --shard remote --workers 8
```

Request resources matching the worker count. Do not fit on a cluster login node.
Closing a tmux window does not bypass a scheduler job time limit.

## Files produced and return procedure

Models use unchanged paths:

`models/talent_motivation/embeddinggemma_768/g-independent/S#/h-METHOD/runs/hz-misspec-v1/OPERATOR--SETTING/seed-N/selected/latent_intervention.pt`

Per-job `training.json` and `resume.pt` live under:

`reports/talent_motivation/hz_misspecification/hz-misspec-v1/fits/S#/METHOD/OPERATOR--SETTING/seed-N/`

The training JSON includes full histories, selected epochs/development scores,
split IDs, source/input/config provenance, runtime and checkpoint hashes.
The remote shard does **not** train the independent evaluation reader or freeze
the complete 196-model selection on its own.

After all 50 jobs complete:

```bash
.venv/bin/python scripts/hz_misspecification_shards.py status --shard remote
.venv/bin/python scripts/hz_misspecification_shards.py pack-results \
  --output hz-misspec-remote50-results.tar.gz
sha256sum -c hz-misspec-remote50-results.tar.gz.sha256
```

Export refuses incomplete jobs and verifies the 50 selected checkpoint hashes.
The archive includes **all five seeds**, not just Git-trackable seed 42, plus
their training records and compact remote execution/check records. Logs and
optimizer snapshots remain remote; retain that directory until the original
machine confirms a successful merge.

Return the result archive and its `.sha256` file to the user. **Do not copy
the whole remote reports tree over the local one**: that could overwrite the
local queue status, selection, protocol or unrelated results.

On the original machine, after its local shard has finished:

```bash
# From the directory containing the returned result archive:
sha256sum -c hz-misspec-remote50-results.tar.gz.sha256
mkdir hz-misspec-remote50-return
tar -xzf hz-misspec-remote50-results.tar.gz -C hz-misspec-remote50-return

# From the original project root, supply the absolute extracted return directory:
.venv/bin/python scripts/hz_misspecification_shards.py merge-results \
  --from-directory /absolute/path/to/hz-misspec-remote50-return
```

The merge validates assignment, protocol, every checksum, job metadata and
checkpoint loading, and rejects differing existing files before importing.
It holds the local queue lock: do not terminate still-running local fits merely
to import results. Once all 196 fits plus the existing evaluation reader are
available, it freezes the complete selection manifest. It does **not** evaluate
the official test set; ask the user before that next step.

## Non-negotiable safeguards

Do not modify existing training sources, YAML, data, g checkpoints, assignment,
seed lists, epochs, objectives, model capacity, splits, operators or precision.
Do not re-encode texts, retrain g, train the evaluation reader, introduce proxy
labels or inspect test performance. Do not run unassigned jobs, infer that
`--all` is safe, or silently drop failed fits. If a guard fails, report the exact
error and preserve artifacts; do not rewrite historical signatures.
