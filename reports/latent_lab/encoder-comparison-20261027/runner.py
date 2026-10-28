"""Operational handoff runner; the repository's scientific code is unchanged."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path('/home/boustanip/latent_intervention/latent_intervention')
RUN = REPO / 'reports/latent_lab/encoder-comparison-20261027'
BASELINE = REPO / 'reports/latent_lab/run-20261027-075057.json'
sys.path.insert(0, str(REPO))
os.chdir(REPO)


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def verify_sources():
    protocol = json.loads((RUN / 'launch.json').read_text())
    for name, expected in protocol['sources'].items():
        if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Source changed during the queue: {name}')


def readers():
    import torch
    from latent_lab import ENCODERS
    from latent_lab.cache import CACHE_DIR, KEY_VERSION, digest, cached

    torch.set_num_threads(4)
    allowed = set()
    for name, dimensions in ENCODERS.items():
        for dim in dimensions:
            key = f'{name}-{dim}'
            if key != 'embeddinggemma-768':
                allowed.add(f'judge-{key}')
                allowed.update(f'psi-{key}-seed{s}' for s in range(5))
    events = []

    def guarded_cache(name, fn, folder=None):
        verify_sources()
        path = Path(folder or CACHE_DIR) / f'{name}-{digest((KEY_VERSION, fn))[:16]}.pt'
        hit = path.is_file()
        if not hit and name not in allowed:
            raise RuntimeError(f'Refusing an unplanned refit: {name}; missing cache {path}')
        print(now(), 'CACHE HIT' if hit else 'FIT READER', name, flush=True)
        start = time.perf_counter()
        result = cached(name, fn, folder)
        events.append(dict(name=name, cache_hit=hit, path=str(path), seconds=time.perf_counter()-start))
        write(RUN / 'reader-cache.json', dict(updated_at=now(), events=events))
        return result

    notebook = json.loads((REPO / 'experiment.ipynb').read_text())
    ns = {'__name__': '__main__'}
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            verify_sources()
            print(now(), 'NOTEBOOK CELL', i, flush=True)
            exec(compile(''.join(cell['source']), f'cell-{i}', 'exec'), ns)
            # Cell 1 imports cached; preserve all notebook cells and function identities.
            ns['cached'] = guarded_cache
    old = json.loads(BASELINE.read_text())
    new = json.loads(ns['path'].read_text())
    expected = {f'{n}-{d}' for n, dims in ENCODERS.items() for d in dims}
    actual = {c.split(' | ')[0] for row in new['psi'] for c in row if ' | ' in c}
    if json.dumps(old['results']) != json.dumps(new['results']):
        raise RuntimeError('Main results differ from the baseline; inspect the new report')
    if actual != expected or len(actual) != 17:
        raise RuntimeError(f'Incomplete encoder comparison: {actual}')
    verification = dict(
        finished_at=now(), report=str(ns['path'].relative_to(REPO)),
        baseline=str(BASELINE.relative_to(REPO)), results_identical=True,
        latent_spaces=sorted(actual), caches_loaded=sum(e['cache_hit'] for e in events),
        reader_cache_groups_fitted=sum(not e['cache_hit'] for e in events),
    )
    write(RUN / 'verification.json', verification)
    print('VERIFIED', json.dumps(verification), flush=True)


def queue():
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(status='running', started_at=now(), stage='initializing', completed=[])
        clock = time.perf_counter()

        def update(**changes):
            state.update(changes, updated_at=now(), elapsed_seconds=time.perf_counter()-clock)
            write(RUN / 'status.json', state)

        try:
            verify_sources()
            for encoder, run_id in (
                ('nomic', 'nomic-768-v1'), ('qwen3', 'qwen3-1024-v1'),
                ('langvae', 'langvae-128-v1'),
            ):
                verify_sources()
                update(stage=f'encoding:{encoder}')
                print(now(), 'LAUNCH ENCODING', encoder, flush=True)
                subprocess.run(['bash', 'scripts/run_talent_motivation_encoding.sh', encoder, run_id], check=True)
                while subprocess.run(['tmux', 'has-session', '-t', '=encode-talent-motivation'],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                    time.sleep(15)
                report = json.loads((REPO / 'reports/talent_motivation/encoding' / run_id / 'status.json').read_text())
                if report['status'] != 'complete':
                    raise RuntimeError(f'{encoder} failed: {report}')
                manifest = json.loads((REPO / report['output'] / 'manifest.json').read_text())
                if not manifest['complete']:
                    raise RuntimeError(f'{encoder} bundle is incomplete')
                state['completed'].append(dict(stage=f'encoding:{encoder}', report=report))
                update()
            verify_sources()
            update(stage='readers')
            print(now(), 'LAUNCH READERS', flush=True)
            with (RUN / 'readers.log').open('a') as log:
                subprocess.run([str(REPO / '.venv/bin/python'), '-u', __file__, '--readers'],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            verify_sources()
            verification = json.loads((RUN / 'verification.json').read_text())
            update(status='complete', stage='complete', finished_at=now(), verification=verification)
            print(now(), 'COMPLETE', verification['report'], flush=True)
        except BaseException:
            update(status='failed', error=traceback.format_exc())
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--readers', action='store_true')
    args = p.parse_args()
    readers() if args.readers else queue()
