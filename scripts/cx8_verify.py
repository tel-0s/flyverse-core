"""CPU verification of cx8's saved traces, separate from its simulator and primary reducer.

    python scripts/cx8_verify.py --runs out/cx8 --out out/cx8/verification

No simulation or submission. Reconstructs the declared moving four-wedge confinement rule, circular centre,
least-squares velocity, and group L-R means from every NPZ. Checks saved JSONs and frozen batch/source hashes.
The constants below are the pre-existing ledger rule, not fitted to cx8. Float32 rate reductions may differ
by a few ulps from double precision; the absolute comparison tolerance is 5e-5 Hz (relative 1e-6).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct(z):
    rates = z['epg'].astype(float)
    wedges, times = z['wedge_of'].astype(int), z['t'].astype(float)
    assert rates.shape == (800, 46) and times.shape == (800,)
    np.testing.assert_allclose(times, np.arange(800) / 100, atol=1e-12, rtol=0)
    turn = (times >= 3.5) & (times < 6.5)
    post = times >= 3
    np.testing.assert_array_equal(z['yaw_deg_s'], np.where(turn, 90, 0))
    assert turn.sum() == 300
    phase = np.exp(2j * np.pi * wedges / 16)
    moment = rates @ phase / np.maximum(rates.sum(axis=1), 1e-9)
    centres = np.mod(np.angle(moment), 2*np.pi) * 16 / (2*np.pi)
    starts = np.rint(centres - 1.5).astype(int) % 16
    # For each frame choose the four consecutive wedges around its circular centre.
    in_block = ((wedges[None, :] - starts[:, None]) % 16) < 4
    above = rates > 22
    confined = ((np.abs(moment) > .6) &
                ((above & in_block).sum(axis=1) / in_block.sum(axis=1) >= 8/11 - 1e-9) &
                ((above & ~in_block).sum(axis=1) <= 3))
    x = times[turn]
    y = np.unwrap(centres[turn] * 2*np.pi/16) * 16/(2*np.pi)
    velocity = np.dot(x-x.mean(), y-y.mean()) / np.dot(x-x.mean(), x-x.mean())
    last = np.flatnonzero(post & confined)
    m = dict(bump_follow_wedges_per_s=float(velocity), bump_follow_confined_frac=float(confined[turn].mean()),
             frac_confined_post=float(confined[post].mean()), survival_s=float(times[last[-1]] + .01 - 3) if len(last) else 0.)
    for g in ('GLNO', 'PEN', 'DNa02', 'PS196b', 'AFF'):
        if f'g__{g}_L' not in z:
            continue
        left, right = z[f'g__{g}_L'].astype(float), z[f'g__{g}_R'].astype(float)
        m[f'{g}_LR_hz'] = float((left-right)[turn].mean())
        m[f'{g}_L_hz_turn'] = float(left[turn].mean())
        m[f'{g}_R_hz_turn'] = float(right[turn].mean())
    profiles = np.column_stack([rates[:, wedges == w].mean(axis=1) for w in range(16)])
    widths = (profiles > .5*profiles.max(axis=1)[:, None]).sum(axis=1)
    bump_rate = (rates*in_block).sum(axis=1) / in_block.sum(axis=1)
    good = post & confined
    m['width_half_post'] = float(widths[good].mean()) if good.any() else float('nan')
    m['bump_hz_post'] = float(bump_rate[good].mean()) if good.any() else float('nan')
    return m


def verify(runs, out):
    runs, out = Path(runs), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((runs/'predeclared.json').read_text(encoding='utf-8'))
    issues, rows, metrics, checked_sources = [], [], [], set()
    for file, key in [('batch.sh','batch_sha256'), ('arms.json','arms_sha256')]:
        if digest(runs/file) != frozen[key]: issues.append(f'{file}: differs from declaration')
    if (runs/'predeclared_archive.json').read_bytes() != (runs/'predeclared.json').read_bytes():
        issues.append('declaration changed after archiving')
    for path in sorted(runs.glob('*_s*.json')):
        raw = json.loads(path.read_text(encoding='utf-8'))
        for i, row in enumerate(raw if isinstance(raw,list) else [raw]):
            rid = f'{path.name}#{i}'
            if row['provenance']['model']['lif'] != frozen['resolved_lif_by_arm'].get(row['arm']):
                issues.append(f'{rid}: resolved model differs from declaration')
            record = row['provenance']['source_fingerprint']
            fp = {**record.get('files_loaded', {}), **(record.get('files_lf') or record.get('files', {}))}
            for file, wanted in frozen['source_sha256_lf'].items():
                # The simulator fingerprints probe scripts and the package, not the analysis-only generator.
                if file in fp:
                    checked_sources.add(file)
                    if fp[file] != wanted: issues.append(f'{rid}: source mismatch {file}')
            if not all(f in fp for f in ('scripts/cx_wedge.py','scripts/probe_compass_room.py','flyverse/instruments.py')):
                issues.append(f'{rid}: missing core source hashes')
            try:
                with np.load(path.parent / Path(row['ledger_npz']).name, allow_pickle=False) as z:
                    values = reconstruct(z)
                for key, value in values.items():
                    saved = row['metrics'].get(key, float('nan'))
                    ok = bool(np.isclose(value, saved, rtol=1e-6, atol=5e-5, equal_nan=True))
                    metrics.append(dict(run_id=rid, key=key, recomputed=value, saved=saved, matches=ok))
                    if not ok: issues.append(f'{rid}: metric mismatch {key}')
                rows.append(dict(run_id=rid, arm=row['arm'], seed=row['seed'],
                                 eligible=values['bump_follow_confined_frac'] >= .5, **values))
            except (AssertionError, ValueError, KeyError, OSError) as e:
                issues.append(f'{rid}: {type(e).__name__}: {e}')
    if len(rows) != 48: issues.append(f'expected 48 trace records, got {len(rows)}')
    pd.DataFrame(rows).to_csv(out/'traces.csv', index=False)
    pd.DataFrame(metrics).to_csv(out/'metric_checks.csv', index=False)
    report = dict(n_runs=len(rows), n_metric_checks=len(metrics), n_source_files_checked=len(checked_sources),
                  source_files_checked=sorted(checked_sources), issues=issues,
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
                  predeclaration_sha256=digest(runs/'predeclared.json'))
    (out/'verification.json').write_text(json.dumps(report, indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='source_files_checked'},indent=2))
    return not issues


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs',default='out/cx8')
    ap.add_argument('--out',default='out/cx8/verification')
    a=ap.parse_args()
    raise SystemExit(0 if verify(a.runs,a.out) else 2)
