"""CPU source and unchanged-control checks for cx8t; no simulation or submission.

    python scripts/cx8_transfer_verify.py --runs out/cx8t --reference out/cx8r --commit 036e512

The primary reducer separately checks all 36 protocols and reconstructs the trace metrics. This
check ties every frozen and recorded source hash to the submitted commit, and compares the six H0
controls exactly with the earlier HG metrics and arrays. It does not replace an independent skeptic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flyverse.interp.common import resolve_commit  # noqa: E402


def verify(runs, reference, commit):
    runs, reference = Path(runs), Path(reference)
    frozen = json.loads((runs / 'predeclared.json').read_text(encoding='utf-8'))
    hashes = {}
    # A submitted id predating the 2026-09-17 rewrite is read through the map (docs/INTERP.md 10.4 rule 30).
    resolved = resolve_commit(commit)
    submitted = resolved if resolved == str(commit).strip() else f'{resolved} (recorded {str(commit).strip()})'

    def at_commit(name):
        if name not in hashes:
            blob = subprocess.check_output(['git', 'show', f'{resolved}:{name}'], cwd=ROOT)
            hashes[name] = hashlib.sha256(blob.replace(b'\r\n', b'\n')).hexdigest()
        return hashes[name]

    for name, value in frozen['source_sha256_lf'].items():
        assert value == at_commit(name), name

    baselines, joint, sources, seen = {}, {}, set(), set()
    for path in sorted(runs.glob('*_s*.json')):
        records = json.loads(path.read_text(encoding='utf-8'))
        assert len(records) == 1, path
        row = records[0]
        arm, seed = row['arm'], row['seed']
        assert (arm, seed) not in seen
        seen.add((arm, seed))
        stamp = row['provenance']['source_fingerprint']
        for mapping in (stamp.get('files_loaded', {}), stamp.get('files_lf', stamp.get('files', {}))):
            for name, value in mapping.items():
                assert value == at_commit(name), (path.name, name)
                sources.add(name)
        joint[arm] = joint.get(arm, 0) + all(x['status'] == 'PASS' for x in row['ledger'].values())
        if arm != 'H0':
            continue
        previous = json.loads((reference / f'HG_s{seed}.json').read_text(encoding='utf-8'))[0]
        for key, value in previous['metrics'].items():
            np.testing.assert_equal(row['metrics'][key], value, err_msg=key)
        with np.load(runs / Path(row['ledger_npz']).name) as actual, \
                np.load(reference / Path(previous['ledger_npz']).name) as old:
            for key in old.files:
                np.testing.assert_equal(actual[key], old[key], err_msg=key)
            baselines[str(seed)] = dict(metrics=len(previous['metrics']), arrays=len(old.files))

    assert seen == {(a, s) for a in ('H0', 'HL', 'HR', 'C0', 'CL', 'CR') for s in range(6)}
    assert len(baselines) == 6
    report = dict(submitted_commit=submitted, n_runs=len(seen), frozen_sources=len(frozen['source_sha256_lf']),
                  recorded_sources=len(sources), recorded_sources_not_in_freeze=sorted(sources-set(frozen['source_sha256_lf'])),
                  all_sources_match_commit=True, h0_matches_hg=baselines, joint_ledger=joint)
    dest = runs / 'verification'
    dest.mkdir(exist_ok=True)
    (dest / 'sources_and_control.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', default='out/cx8t')
    ap.add_argument('--reference', default='out/cx8r')
    ap.add_argument('--commit', required=True, help='commit submitted to house, not the current analysis commit')
    args = ap.parse_args()
    verify(args.runs, args.reference, args.commit)
