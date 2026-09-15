"""The single round-7 GLNO-to-PEN follow-up: a fixed direct neural challenge, not a fitted mechanism.

    python scripts/cx8_transfer.py plan --out out/cx8t
    python scripts/cx8_transfer.py freeze --out out/cx8t  # after review, before submission
    python scripts/cx8_transfer.py run --seed 0 --block fam_s0 --out out/cx8t
    python scripts/cx8_transfer.py analyse --out out/cx8t

Six fresh brains per seed/job, six seeds: H0/HL/HR on the HG configuration, C0/CL/CR with GLNO->PEN
held additionally. L/R is 90 Hz Poisson forcing on the two GLNO cells on that somaSide for [3.5,6.5) s.
This is an unverified diagnostic challenge. The common nominal +90 deg/s clock is recorded but never fed
to proprioception; no sided-turn instrument or body is present. No new receptor row or gain is adopted.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from flyverse import brain, connectome
from flyverse.interp import common
import cx_wedge as cw
import cx_velocity_route as cvr
from cx8_verify import reconstruct

ARMS={'H0':(False,None),'HL':(False,'L'),'HR':(False,'R'),
      'C0':(True,None),'CL':(True,'L'),'CR':(True,'R')}
CUT=(r'^GLNO$',r'^PEN_',0.)
LEVEL=90.
FAMILY=[('1_engagement','GLNO_LR_hz','HL','HR'),
        ('2_transfer','PEN_LR_hz','HL','HR'),
        ('3_left_edge_dependency','PEN_LR_hz','HL','CL'),
        ('4_right_edge_dependency','PEN_LR_hz','HR','CR')]
LAW='unverified diagnostic level; a direct neural challenge, not a physiological transfer law'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def holds(arm):
    cut,_=ARMS[arm]
    return [(*cvr.HOLD.split(':',1),0.)]+([CUT] if cut else [])


def events(c,arm,start=3.5,duration=3.):
    side=ARMS[arm][1]
    if side is None:return []
    idx=cw.side_groups(c,{'GLNO':'GLNO'})[f'GLNO_{side}']
    if len(idx)!=2:raise ValueError(f'expected two GLNO_{side} cells')
    return [dict(name=f'direct_GLNO_{side}',idx=idx,poisson_hz=LEVEL,start_s=start,duration_s=duration,law=LAW)]


def expected(c,arm):
    lif=cvr.resolved_lif_by_arm()['HG']
    if ARMS[arm][0]:lif['type_path_gain'].append(list(CUT))
    return dict(lif=lif,hold_edges=[list(h) for h in holds(arm)],hold_edges_resolved=cw.hold_edge_counts(c,holds(arm)),
                instruments=['ring_dc_hold']+(['glno_pen_hold'] if ARMS[arm][0] else [])+['glno_sign'],
                instrument_records=[i.describe() for i in cw.build_instruments(c,[], 'instrumented',
                                    hold_edges=holds(arm),nt_override={'GLNO':'glutamate'})],
                neural_stimuli=cw.prepare_neural_stimuli(c,events(c,arm),8.),
                preset='instrumented',nt_override={'GLNO':'glutamate'},cache_md5=cvr.CACHE_MD5[True])


def source_hashes():
    files=cvr.source_hashes()
    for name in ('cx8_transfer.py','cx8_verify.py'):
        p=ROOT/'scripts'/name
        files[f'scripts/{name}']=hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    return files


def plan(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'predeclared.json').exists() or (out/'submitted_at.txt').exists():raise ValueError('batch already frozen')
    c,_,_=cw.load_connectome({'GLNO':'glutamate'},root=ROOT/'out')
    rel=f'out/{out.name}'
    commands=[]
    for seed in range(6):
        stem=f'{rel}/seed{seed}'
        cmd=f'python scripts/cx8_transfer.py run --seed {seed} --block fam_s{seed} --out {rel} > {stem}.txt 2>&1'
        commands.append(cvr.job_line(cmd,stem,rel))
    shell=['#!/bin/bash','set -o pipefail','# DRAFT: freeze the reviewed declaration before submission.',
           '# One house submission: six jobs, each with all six conditions on fresh brains; 36 runs.',
           'python scripts/cluster_run.py --target house --name cx8t --minutes 30 --arm-block fam '+
           ' '.join('"'+s+'"' for s in commands)+f' --fetch {rel}/ 2>&1 | tee {rel}/client_stdout.txt || exit $?']
    (out/'batch.sh').write_text('\n'.join(shell)+'\n',encoding='utf-8',newline='\n')
    record=dict(status='DRAFT',batch='cx8t',seeds=list(range(6)),arms={a:expected(c,a) for a in ARMS},family=FAMILY,
                protocol=cvr.PROTOCOL,level_hz=LEVEL,law=LAW,turn_clock_only=True,
                recording_s=8.,input_window_s=[3.5,6.5],replicate_unit='fresh-brain runs, six independent seeds per arm',
                decision='engagement and intact transfer require positive results; held-edge contrasts describe network dependence, not a unique receptor mechanism',
                multiplicity='one Holm family m=4, common.compare, n=6 per arm; floor 4*2/C(12,6)=0.00866',
                falsifiers='no GLNO side engagement leaves transfer undetermined; engagement with null PEN side contrast detects no transfer at this challenge level',
                limits='HL-CL/HR-CR include the changed operating state caused by removing GLNO-PEN feedback; H0/C0 are descriptive baselines. Result versus null is not an interaction test. No adoption or second follow-up is authorized.')
    (out/'arms.json').write_text(json.dumps(common.to_jsonable(record),indent=2)+'\n',encoding='utf-8')
    print(f'planned six jobs / 36 runs in {out}')


def freeze(out):
    out=Path(out)
    if (out/'submitted_at.txt').exists() or list(out.glob('*_s*.json')):raise ValueError('already submitted or results exist')
    record=json.loads((out/'arms.json').read_text(encoding='utf-8'))
    assert record['seeds']==list(range(6)) and record['family']==[list(f) for f in FAMILY]
    record.update(status='PREDECLARED, not submitted',stamped_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  source_sha256_lf=source_hashes(),batch_sha256=sha(out/'batch.sh'),arms_sha256=sha(out/'arms.json'))
    with (out/'predeclared.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(record,f,indent=2);f.write('\n')
    print('froze',out/'predeclared.json')


def run(out,seed,block,arms=None,device=None,no_graphs=False,smoke=False):
    if block!=f'fam_s{seed}':raise ValueError('seed block mismatch')
    if smoke and device!='cpu':raise ValueError('the shortened implementation smoke is CPU-only')
    if arms is not None and not smoke:raise ValueError('arm subsets are only for the CPU implementation smoke')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    selected=list(ARMS) if arms is None else arms
    if not selected or any(a not in ARMS for a in selected):raise ValueError('unknown/empty arm selection')
    c,_,_=cw.load_connectome({'GLNO':'glutamate'},root=ROOT/'out')
    cells=cw.compass_cells(c)
    params=brain.LIFParams()
    records=[]
    for arm in selected:
        stem=out/f'{arm}_s{seed}'
        if any(stem.with_suffix(ext).exists() for ext in ('.json','.txt','.npz')):raise FileExistsError(stem)
        timing=dict(settle_s=.1,pulse_s=.1,seconds=.6,turn_window=(.1,.4)) if smoke else {}
        ev=events(c,arm,start=.3 if smoke else 3.5,duration=.3 if smoke else 3.)
        with stem.with_suffix('.txt').open('x',encoding='utf-8') as log,contextlib.redirect_stdout(log):
            rows=cw.simulate(c,cells,[(1.,1.)],seed=seed,ledger=True,arm=arm,block=block,
                             device=device,cuda_graphs=not no_graphs,ledger_npz=str(stem.with_suffix('.npz')),
                             hold_edges=holds(arm),nt_override={'GLNO':'glutamate'},preset='instrumented',
                             receptor_model=params.receptor_model,receptor_net_rule=params.receptor_net_rule,
                             turn_deg_s=90.,neural_stimuli=ev,**timing)
        for row in rows:
            row['transfer_probe']=dict(version=1,cut=ARMS[arm][0],side=ARMS[arm][1],level_hz=LEVEL,
                                       law=LAW,nominal_turn_clock_only=True,smoke=smoke)
        with stem.with_suffix('.json').open('x',encoding='utf-8') as f:json.dump(rows,f,indent=1)
        records.extend(rows)
        print(f'{arm} seed {seed}: completed, device {rows[0]["device"]}',flush=True)
    return records


def compare_family(df,problems=()):
    comps=[]
    for name,key,a,b in FAMILY:
        x,y=df[df.arm==a],df[df.arm==b]
        rec=dict(test=name,key=key,stim_arm=a,null_arm=b,stim_run_ids=x.run_id.tolist(),null_run_ids=y.run_id.tolist(),
                 **common.compare(x[key].to_numpy(float),y[key].to_numpy(float)))
        comps.append(rec)
    adjusted=cvr.holm({x['test']:x['p'] for x in comps},m=4)
    for r in comps:
        r['p_holm']=adjusted[r['test']];r['m']=4
        r['verdict_holm']='null' if r['verdict']=='result' and r['p_holm']>.05 else r['verdict']
        if problems:r['verdict']=r['verdict_holm']='undetermined'
    return comps


def analyse(out):
    out=Path(out);dest=out/'analysis';dest.mkdir(exist_ok=True)
    frozen=json.loads((out/'predeclared.json').read_text(encoding='utf-8'))
    problems=[];seen=set();rows=[];metric_checks=[]
    if frozen['family']!=[list(f) for f in FAMILY] or frozen['protocol']!=cvr.PROTOCOL or frozen['level_hz']!=LEVEL:
        problems.append('analysis differs from frozen family/protocol/level')
    for file,key in [('batch.sh','batch_sha256'),('arms.json','arms_sha256')]:
        if sha(out/file)!=frozen[key]:problems.append(f'{file}: frozen hash differs')
    if (out/'predeclared_archive.json').read_bytes()!=(out/'predeclared.json').read_bytes():problems.append('declaration changed')
    for path in sorted(out.glob('*_s*.json')):
        for index,r in enumerate(json.loads(path.read_text(encoding='utf-8'))):
            rid=f'{path.name}#{index}';arm=r.get('arm');seed=r.get('seed');bad=[]
            if arm not in ARMS or seed not in range(6):problems.append(f'{rid}: unexpected arm/seed');continue
            if (arm,seed) in seen:bad.append('duplicate arm/seed')
            seen.add((arm,seed));e=frozen['arms'][arm];p=r['provenance'];m=r['metrics']
            if p['model']['lif']!=e['lif']:bad.append('resolved LIF')
            for k in ('preset','instruments','nt_override','hold_edges','hold_edges_resolved','neural_stimuli'):
                if r.get(k)!=e[k]:bad.append(k)
            if p['preset']!=e['preset'] or [d['name'] for d in p['instruments']]!=e['instruments']:bad.append('instrument provenance')
            if p['instruments']!=e['instrument_records']:bad.append('instrument descriptions')
            probe=dict(version=1,cut=ARMS[arm][0],side=ARMS[arm][1],level_hz=LEVEL,law=LAW,nominal_turn_clock_only=True,smoke=False)
            if r.get('transfer_probe')!=probe or r.get('glno_nt')!=['glutamate']:bad.append('probe/relabel metadata')
            if p['compiled_connectome']['md5']!=e['cache_md5']:bad.append('compiled graph')
            if p['stimulus']['params'].get('neural_stimuli')!=e['neural_stimuli']:bad.append('stimulus provenance')
            applied=[dict(name=x['name'],start_s=x['start_s']) for x in e['neural_stimuli']]
            if r.get('neural_stimuli_applied')!=applied:bad.append('applied pulses')
            if r.get('block')!=f'fam_s{seed}' or r.get('transfer_probe',{}).get('smoke') is not False:bad.append('block/smoke')
            if not r.get('device','').startswith('cuda') or 'device cuda' not in path.with_suffix('.txt').read_text(encoding='utf-8'):bad.append('CUDA console')
            for k,value in frozen['protocol'].items():
                if p['stimulus']['params'].get(k)!=value:bad.append(f'protocol {k}')
            if m.get('turn_fed') or r.get('instrument_specs') or r.get('edge_gains') or r.get('lif_overrides'):bad.append('unexpected input/override')
            for key in ('GLNO_LR_hz','PEN_LR_hz','DNa02_LR_hz','frac_confined_post'):
                if not isinstance(m.get(key),(float,int)) or not np.isfinite(m[key]):bad.append(f'nonfinite {key}')
            fp=p['source_fingerprint'];actual={**fp.get('files_loaded',{}),**fp.get('files_lf',fp.get('files',{}))}
            for name in ('scripts/cx8_transfer.py','scripts/cx_wedge.py','scripts/probe_compass_room.py'):
                if name not in actual:bad.append(f'missing source {name}')
            for name,h in actual.items():
                if name in frozen['source_sha256_lf'] and h!=frozen['source_sha256_lf'][name]:bad.append(f'source {name}')
            try:
                with np.load(path.parent/Path(r['ledger_npz']).name) as z:
                    values=reconstruct(z)
                    command=np.zeros((800,len(e['neural_stimuli'])),np.float32)
                    if command.shape[1]:command[350:650,:]=LEVEL
                    np.testing.assert_array_equal(z['neural_stimulus_hz'],command)
                    for key,value in values.items():
                        ok=bool(np.isclose(value,m[key],atol=5e-5,rtol=1e-6,equal_nan=True))
                        metric_checks.append(dict(run_id=rid,key=key,matches=ok,recomputed=value,saved=m[key]))
                        if not ok:bad.append(f'trace {key}')
            except (ValueError,KeyError,AssertionError,OSError) as err:bad.append(f'trace: {err}')
            record=dict(run_id=rid,file=path.name,arm=arm,seed=seed,checks=';'.join(bad),**m)
            rows.append(record);problems.extend(f'{rid}: {x}' for x in bad)
    if seen!={(a,s) for a in ARMS for s in range(6)} or len(rows)!=36:problems.append('expected 36 unique arm/seed runs')
    df=pd.DataFrame(rows);df.to_csv(dest/'runs.csv',index=False)
    pd.DataFrame(metric_checks).to_csv(dest/'metric_checks.csv',index=False)
    per=[]
    keys=['GLNO_LR_hz','PEN_LR_hz','DNa02_LR_hz','GLNO_L_hz_turn','GLNO_R_hz_turn','PEN_L_hz_turn','PEN_R_hz_turn',
          'frac_confined_post','bump_follow_confined_frac','bump_follow_wedges_per_s','survival_s','bump_hz_post','width_half_post',
          'GLNO_mean_pre','GLNO_mean_during','GLNO_mean_post','PEN_mean_pre','PEN_mean_during','PEN_mean_post']
    for arm in ARMS:
        group=df[df.arm==arm].sort_values('seed')
        for key in keys:
            a=group[key].to_numpy(float)
            per.append(dict(arm=arm,key=key,seeds=','.join(map(str,group.seed)),files=','.join(group.file),run_ids=','.join(group.run_id),
                            values=','.join(f'{x:.4f}' for x in a),mean=float(np.nanmean(a)) if np.isfinite(a).any() else None,
                            sd=float(np.nanstd(a,ddof=1)) if np.isfinite(a).sum()>1 else None,n=int(np.isfinite(a).sum())))
    pd.DataFrame(per).to_csv(dest/'per_seed.csv',index=False)
    comps=compare_family(df,problems)
    pd.DataFrame(comps).to_csv(dest/'compare.csv',index=False)
    report=dict(n_runs=len(rows),problems=problems,family=comps,n_metric_checks=len(metric_checks),
                valid_batch=not problems,analysis_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
                declaration_sha256=sha(out/'predeclared.json'))
    (dest/'analysis.json').write_text(json.dumps(common.to_jsonable(report),indent=2)+'\n',encoding='utf-8')
    print(f'{len(rows)} runs; {len(problems)} problems; {len(metric_checks)} trace checks')
    common.print_table(pd.DataFrame(comps)[['test','verdict_holm','diff','z','p','p_holm']])
    if problems:print('\n'.join(problems[:20]))
    return not problems


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['plan','freeze','run','analyse'])
    ap.add_argument('--out',default='out/cx8t');ap.add_argument('--seed',type=int,default=0);ap.add_argument('--block')
    ap.add_argument('--arms',help='implementation smoke only; default all six conditions')
    ap.add_argument('--device');ap.add_argument('--no-graphs',action='store_true');ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    if a.action=='plan':plan(a.out)
    elif a.action=='freeze':freeze(a.out)
    elif a.action=='run':run(a.out,a.seed,a.block,arms=a.arms.split(',') if a.arms else None,device=a.device,no_graphs=a.no_graphs,smoke=a.smoke)
    else:raise SystemExit(0 if analyse(a.out) else 2)
