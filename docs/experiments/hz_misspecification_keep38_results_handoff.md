# Results handoff: assemble hz-misspec-v1 on the original cluster

Prepared 26 October 2026 for the original agent on **cpuserver17**.

## Requested outcome

Import the **38 completed fits from cpuserver16** into the original experiment
checkout, verify the other fits, and confirm that all **196 fits** plus the
existing evaluation reader are assembled with a valid frozen selection manifest.
This handoff is for results assembly. Do not launch official-test evaluation,
retrain completed fits, change the frozen protocol, or stage/commit/push files.

All 38 assigned fits on cpuserver16 completed successfully. The last finished
on **25 October 2026 at 07:13:52 Europe/Berlin (CET)**. Their records and selected
checkpoint checksums were verified, and the training queue has exited.

| Original experiment component | Fits | Where to verify |
|---|---:|---|
| Original `local` shard | 146 | Your original checkout; previously reported complete |
| `return12` reassigned from cpuserver16 | 12 | Your checkout or your separate return12 run directory |
| `keep38` completed on cpuserver16 | 38 | The supplied result archive |
| **Total** | **196** | Original checkout after assembly |

The 12 were returned because CPU fitting on cpuserver16 was slow. This 38-fit
archive is their exact complement. Their current completion status must be
verified on your host; the sender has not checked it remotely.

## Hosts, paths and supplied files

- Receiving/original host: **cpuserver17**, `boustanip@192.168.9.145`.
- Destination checkout: `/home/boustanip/latent_intervention/latent_intervention`.
- Sending host: **cpuserver16**, `boustanip@192.168.9.144`.
- Sender's transfer directory:
  `/home/boustanip/research/latent_intervention/reports/talent_motivation/hz_misspecification/hz-misspec-v1/shards/remote50-20261020/return12-20261022/transfer`.
- Use the existing destination environment and data. No training inputs need
  transferring or re-encoding for this import.

Fetch these four files from the sender's transfer directory:

1. `hz-misspec-keep38-results.tar.gz`
2. `hz-misspec-keep38-results.tar.gz.sha256`
3. `hz-misspec-keep38-merge-kit.tar.gz`
4. `hz-misspec-keep38-merge-kit.tar.gz.sha256`

The **51,846,265-byte result archive** contains 38 selected model checkpoints,
38 complete training records, the original assignment, the supplementary
allocation, and three execution/check/status records. All five seeds are
included where assigned. Its SHA-256 is:

`b67418f1731ed1f10fd85356591c46444441ac5331892e23c8b94a14ca41f319`

The small **merge kit** contains the exact supplementary runner and launcher,
the original `allocation.json` and `cutover.json`, and this handoff. These
metadata records are supplied directly; do not regenerate them. The kit does
not contain model results or training data.

Expected identities checked below:

| Record | Checksum |
|---|---|
| Frozen protocol signature | `142a08922146ede419ac0a9d25b1a61bc68d54f0cda39b6d528f44c7ee8e2a1d` |
| Original assignment record | `26381c03db12fbdba49689cecf26f80b3e2ab8c41fedea8d9179b59b36dfe5b2` |
| Supplementary allocation record | `a291ada4ca4628d07c3ec6bda6d5627c89192c2a0b20e523c8fe37f1ac217cfc` |
| Cutover record | `6fa008f86db1a5b248aeb18ae4262ed0e3d5a9641b7b0bd14269820f4504728c` |
| Result transfer manifest record | `997f1e4dff0b18debf01aed90e81aedf69137742a83ba120bbfffd54fae7ddc4` |

## 1. Copy and extract on cpuserver17

Run the following in **Bash on cpuserver17**. Extraction is into a separate
staging directory, not over the original experiment. If already received, reuse
the verified staging directories and continue at step 2 instead of extracting
again. SSH may require the user's password; the sending agent could not push
files noninteractively.

```bash
set -euo pipefail
merge_receive='/home/boustanip/research/hz-misspec-keep38-return-20261026'
merge_source='/home/boustanip/research/latent_intervention/reports/talent_motivation/hz_misspecification/hz-misspec-v1/shards/remote50-20261020/return12-20261022/transfer'
mkdir -p "$merge_receive"
cd "$merge_receive"

# This prefix selects the two archives and their checksum files.
scp "boustanip@192.168.9.144:$merge_source/hz-misspec-keep38-*.tar.gz*" .
sha256sum -c hz-misspec-keep38-results.tar.gz.sha256
sha256sum -c hz-misspec-keep38-merge-kit.tar.gz.sha256
printf '%s  %s\n' \
    'b67418f1731ed1f10fd85356591c46444441ac5331892e23c8b94a14ca41f319' \
    'hz-misspec-keep38-results.tar.gz' | sha256sum -c -

mkdir incoming merge-kit
tar -xzf hz-misspec-keep38-results.tar.gz -C incoming
tar -xzf hz-misspec-keep38-merge-kit.tar.gz -C merge-kit
```

## 2. Verify both bundles and install only missing merge-kit files

The following verifies every transferred file and both ownership records before
installing the small merge kit. Existing files must have identical bytes; it
does not overwrite differing files or copy any model results at this step.

```bash
python3 - <<'PY'
import hashlib
import json
import shutil
from pathlib import Path

original = Path('/home/boustanip/latent_intervention/latent_intervention').resolve()
received = Path('/home/boustanip/research/hz-misspec-keep38-return-20261026').resolve()
incoming, kit = received / 'incoming', received / 'merge-kit'
sub = Path('reports/talent_motivation/hz_misspecification/hz-misspec-v1/shards/remote50-20261020/return12-20261022')
protocol_hash = '142a08922146ede419ac0a9d25b1a61bc68d54f0cda39b6d528f44c7ee8e2a1d'
assignment_hash = '26381c03db12fbdba49689cecf26f80b3e2ab8c41fedea8d9179b59b36dfe5b2'
allocation_hash = 'a291ada4ca4628d07c3ec6bda6d5627c89192c2a0b20e523c8fe37f1ac217cfc'

def record(path):
    value = json.loads(path.read_text())
    expected = value.pop('record_sha256')
    actual = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if actual != expected:
        raise ValueError(f'Record checksum mismatch: {path}')
    return dict(value, record_sha256=expected)

def safe(base, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or str(relative) in ('', '.'):
        raise ValueError(f'Unsafe path: {name}')
    path = base / relative
    if not path.resolve().is_relative_to(base):
        raise ValueError(f'Path escapes root: {name}')
    return path

for directory, kind in ((incoming, 'return12-keep38-results'), (kit, 'return12-keep38-merge-kit')):
    manifest = record(directory / 'TRANSFER_MANIFEST.json')
    assert manifest['kind'] == kind and manifest['shard'] == 'remote'
    assert manifest['protocol_sha256'] == protocol_hash
    assert manifest['assignment_sha256'] == assignment_hash
    for name, expected in manifest['files'].items():
        assert hashlib.sha256(safe(directory, name).read_bytes()).hexdigest() == expected, name
    if directory == incoming:
        assert len(manifest['files']) == 81
        assert manifest['record_sha256'] == '997f1e4dff0b18debf01aed90e81aedf69137742a83ba120bbfffd54fae7ddc4'

allocation = record(kit / sub / 'allocation.json')
assert allocation['record_sha256'] == allocation_hash
assert allocation == record(incoming / sub / 'allocation.json')
assert allocation['parent_assignment_sha256'] == assignment_hash
assert allocation['protocol_sha256'] == protocol_hash
assert {role: len(jobs) for role, jobs in allocation['assignments'].items()} == {'keep38': 38, 'return12': 12}
cutover = record(kit / sub / 'cutover.json')
assert cutover['record_sha256'] == '6fa008f86db1a5b248aeb18ae4262ed0e3d5a9641b7b0bd14269820f4504728c'
assert cutover['allocation_sha256'] == allocation_hash and cutover['returned_jobs_unstarted']
for name, expected in allocation['orchestration_sha256'].items():
    assert hashlib.sha256(safe(kit, name).read_bytes()).hexdigest() == expected, name

kit_manifest = record(kit / 'TRANSFER_MANIFEST.json')
allowed = {
    'scripts/hz_misspecification_return12.py',
    'scripts/run_hz_misspecification_return12.sh',
    'docs/experiments/hz_misspecification_keep38_results_handoff.md',
    str(sub / 'allocation.json'), str(sub / 'cutover.json'),
}
assert set(kit_manifest['files']) == allowed
for name in allowed:
    destination = safe(original, name)
    if destination.exists() and destination.read_bytes() != safe(kit, name).read_bytes():
        raise FileExistsError(f'Preserve and inspect the differing destination file: {destination}')
for name in allowed:
    destination = safe(original, name)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(safe(kit, name), destination)
print('All transferred files verified; original allocation/cutover and merge tools installed.')
PY
```

If a runner/launcher differs, preserve the local version and investigate the
difference. The allocation deliberately checks those scripts' exact bytes;
do not edit its hashes to accept a different version. The original scientific
sources, YAML, inputs, reader checkpoints and parent assignment are also checked
by the importer. A Git pull alone does not supply all the result checkpoints.

## 3. Inspect the original experiment and merge keep38

Use the **existing original checkout and environment**. The other agent
previously confirmed its original 146-job queue had finished. Check that no
return12 writer is still running there; the importer's exclusive queue lock
will refuse a concurrent writer. Wait for it to finish rather than stopping
training to import results.

```bash
set -euo pipefail
cd /home/boustanip/latent_intervention/latent_intervention
.venv/bin/python scripts/hz_misspecification_shards.py status --shard local
.venv/bin/python scripts/hz_misspecification_return12.py status --role return12
.venv/bin/python scripts/hz_misspecification_return12.py merge-results \
    --role keep38 \
    --from-directory /home/boustanip/research/hz-misspec-keep38-return-20261026/incoming
```

Use this **role-specific importer**, not the historical full-50 importer. It
validates checksums, identities, checkpoint loading and every existing-file
collision before copying results. It accepts identical existing files and
refuses differing bytes. Do not resolve collisions by deleting existing fits
or copying the whole returned reports tree over the original one.

If the other 158 fits and the existing evaluation reader are present, expect:

```text
Imported 38 fits (keep38); full selection frozen: True. No test evaluation.
```

The merge receipt is written to:

`reports/talent_motivation/hz_misspecification/hz-misspec-v1/shards/remote50-20261020/return12-20261022/merge-keep38.json`

If `full selection frozen: False`, the 38-fit import can still have succeeded.
Some other required fits or the evaluation reader are missing from this
checkout. Do not report all 196 assembled until the final audit below passes.

### If return12 was trained in a separate directory

If those 12 were trained directly in the original checkout, they are already
in place and need no separate import. Otherwise, export and import that
completed `return12` run using the role-specific procedure in
[the return12 handoff](hz_misspecification_return12_handoff.md#finish-export-and-import-the-12-results):

1. In the actual return12 working directory, verify all 12 complete and use
   `pack-results --role return12 --output ...` to export them.
2. Verify its `.sha256` file and extract into its own staging directory.
3. In the original checkout, use `merge-results --role return12 --from-directory ...`.

Both import orders work; the last successful import freezes the selection once
all 196 fits and the existing evaluation reader are present. Rerunning the
verified keep38 import after the other results arrive is also supported.
Preserve incomplete runs and report their exact missing job keys if assembly
cannot yet finish. Do not retrain fits merely because they have not been copied.

## 4. Verify all 196 and the frozen selection

Run this in the original checkout after the imports. It checks the complete
job inventory, selected checkpoint/report hashes and the evaluation-reader
selection. It performs no fitting or official-test evaluation.

```bash
cd /home/boustanip/latent_intervention/latent_intervention
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
import json
from scripts import hz_misspecification_return12 as sub
from exp.benchmarks.misspecification.reader import deployment

p = sub.parent
cfg, signature, assignment, allocation = sub.load_allocation()
groups = {
    'original_local': assignment['assignments']['local'],
    'return12': allocation['assignments']['return12'],
    'keep38': allocation['assignments']['keep38'],
}
expected = {p.job_key(j): j for j in signature['jobs']}
assembled = [j for jobs in groups.values() for j in jobs]
assert len(assembled) == len(expected) == 196
assert len({p.job_key(j) for j in assembled}) == 196
assert {p.job_key(j): j for j in assembled} == expected

completed, missing, records = {}, {}, {}
for name, jobs in groups.items():
    absent = []
    for job in jobs:
        fit = p.completed_record(cfg, signature, job)
        if fit is None:
            absent.append(p.job_key(job))
        else:
            records[p.job_key(job)] = fit
    completed[name] = len(jobs) - len(absent)
    if absent:
        missing[name] = absent
print(json.dumps({'completed': completed, 'missing': missing}, indent=2))
if missing:
    raise SystemExit('Assembly incomplete: locate the missing original results before proceeding.')
assert completed == {'original_local': 146, 'return12': 12, 'keep38': 38}

selection_path = p.root(cfg) / 'selection.json'
if not selection_path.exists():
    raise SystemExit('All fits are present but selection.json is missing. Check the existing evaluation reader, then rerun the verified merge.')
selection = p.read_record(selection_path)
assert selection['protocol_sha256'] == p.digest(signature)
assert selection['official_test_used_for_selection'] is False
assert set(selection['models']) == set(expected)
for key, job in expected.items():
    checkpoint, report, _ = p.paths(cfg, job)
    entry = selection['models'][key]
    assert entry['checkpoint'] == str(checkpoint) and entry['report'] == str(report), key
    assert entry['checkpoint_sha256'] == p.sha256_file(checkpoint), key
    assert entry['report_sha256'] == p.sha256_file(report), key
    assert entry['fit_signature'] == records[key]['signature'], key
reader = p.read_record(p.root(cfg) / 'evaluation_reader/selection.json')
assert selection['evaluation_reader'] == reader
assert reader['protocol_sha256'] == p.digest(signature)
assert reader['checkpoint_sha256'] == p.sha256_file(deployment(cfg))
print(json.dumps({
    'verified_fits': 196,
    'evaluation_reader_verified': True,
    'selection': str(selection_path),
    'selection_record_sha256': selection['record_sha256'],
    'official_test_used_for_selection': False,
}, indent=2))
PY
```

The top-level `status.json` may retain older scheduling labels. The validated
per-fit records and `selection.json` are the evidence of complete assembly.

## Report back and retain the source results

Tell the user the verified counts **146 + 12 + 38 = 196**, the destination
checkout, the merge receipt path, and the final selection record checksum.
Report any incomplete jobs or conflicting files explicitly. Do not describe
this handoff as merged until the import and final audit have actually passed
on cpuserver17.

Retain the sender's completed run and both transfer archives until the user
confirms successful assembly on the original machine. Any later official-test
evaluation is a separate step requiring the user's instruction.
