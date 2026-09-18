"""Recompute the frozen compass engineering gates and emit every run/row, never filter eligibility.

Analysis runs on CPU after fetching the single house submission. Inputs are immutable; outputs live in analysis/.
The profile can be analysed separately with --profile-only. This script makes no default-adoption decision.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import subprocess

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from probe_compass_driver import measure
from flyverse.interp.common import resolve_commit


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def csv_write(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)


def input_hashes(root):
    paths=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.npz','.json','.png')
           and 'analysis' not in p.relative_to(root).parts]
    hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    (root/'analysis/input_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')


def profile(path):
    records=read(path)['records'];rows=[]
    for batch in (1,8,32):
        subset=[r for r in records if r['batch']==batch]
        raw=[r['wall_ms_per_frame'] for r in subset if r['mode']=='raw']
        inst=[r['wall_ms_per_frame'] for r in subset if r['mode']=='instrumented']
        assert len(raw)==len(inst)==4
        overhead=100*(np.median(inst)/np.median(raw)-1)
        rows.append(dict(batch=batch,raw_ms=float(np.median(raw)),instrumented_ms=float(np.median(inst)),
                         overhead_percent=float(overhead),target_pass=bool(overhead<=10),
                         module_cuda_us=float(next(r['cuda_ms_per_frame'] for r in subset if r['mode']=='module_only')*1000),
                         raw_repeats_ms=raw,instrumented_repeats_ms=inst))
    return rows


def gates(r):
    speed=r['speed_deg_s'];limits=max(22.5,.1*abs(speed)+15),max(5,.15*abs(speed))
    checks=dict(strength=r['mean_strength']>=.7,weak=r['weak_fraction']<=.05,
                stationary=r['stationary_error_p95_deg']<=15,
                moving=speed==0 or r['moving_error_p95_deg'] is not None and r['moving_error_p95_deg']<=limits[0],
                first=r['first_slope_deg_s'] is not None and abs(r['first_slope_deg_s']-speed)<=limits[1],
                reverse=r['reverse_slope_deg_s'] is not None and abs(r['reverse_slope_deg_s']+speed)<=limits[1])
    return ';'.join(k for k,v in checks.items() if not v)


def analyse(root):
    root=Path(root);out=root/'analysis';out.mkdir(exist_ok=True)
    frozen=read(root/'predeclared.json');records=[];verified=[];suite_rows=[];room_rows=[]
    def verify(p,mode,label):
        assert p['compiled_connectome']['md5']=='ef23cc27bea13be7f6a96f3c04fd3737',label
        assert p['preset']==mode,label
        assert [i['name'] for i in p['instruments']]==(['compass'] if mode=='instrumented' else []),label
        assert p['execution']['device']=='cuda',label
        if mode=='instrumented':
            inst=p['instruments'][0]
            assert inst['law']=='unverified' and inst['parameters']==frozen['parameters'],label
            assert len(inst['body_ids'])==46 and len(p['model']['modules'])==1,label
        fp=p['source_fingerprint'];assert fp['computed'],label
        files=fp['files_lf'];checks={k:v for k,v in frozen['source_sha256_lf'].items() if k in files}
        assert len(checks)>30 and 'flyverse/compass.py' in files,label
        assert all(files[k]==v for k,v in checks.items()),label
        verified.append(dict(record=label,matched_source_files=len(checks)))

    for mode in ('raw','instrumented'):
        for seed in frozen['contract_seeds']:
            stem=f'{mode}_s{seed}';data=read(root/(stem+'.json'))
            assert (data['batch'],data['seconds'],data['seed'])==(8,12.,seed)
            assert data['graphs']>0,stem
            verify(data['provenance'],mode,stem)
            recomputed=measure(root/(stem+'.npz'))
            for a,b in zip(recomputed,data['records'],strict=True):
                assert a.keys()==b.keys()
                for k,v in a.items():
                    assert v is None and b[k] is None or v is not None and np.isclose(v,b[k],rtol=1e-10,atol=1e-10),(stem,k)
                records.append(dict(mode=mode,seed=seed,**a,failed_gates=gates(a) if mode=='instrumented' else 'reference only'))
    for seed in frozen['suite_seeds']:
        pair={mode:read(root/f'suite_{mode}_s{seed}.json') for mode in ('raw','instrumented')}
        for mode,data in pair.items():
            assert data['config']['draw_seed']==seed
            assert len(data['controllers'])>8
            assert all('error' not in r for r in data['sections'].values())
            for i,p in enumerate(data['controllers']):verify(p,mode,f'suite_{mode}_s{seed}:{i}')
        by_mode={mode:{r['key']:r for r in data['checks']} for mode,data in pair.items()}
        assert by_mode['raw'].keys()==by_mode['instrumented'].keys()
        for key,raw in by_mode['raw'].items():
            inst=by_mode['instrumented'][key]
            suite_rows.append(dict(seed=seed,key=key,raw_value=raw['measured'],instrumented_value=inst['measured'],
                                   raw_status=raw['status'],instrumented_status=inst['status'],
                                   regressed=raw['status'].startswith('PASS') and not inst['status'].startswith('PASS')))
    for mode in ('raw','instrumented'):
        data=read(root/f'room_{mode}.json');verify(data['provenance'],mode,'room_'+mode)
        assert data['seconds']==60 and data['batch']==6 and data['graphs']>0
        assert data['provenance']['stimulus']['program']=='none'
        room_rows.extend(dict(mode=mode,**r) for r in data['rows'])
    for i,p in enumerate(read(root/'profile.json')['controllers']):verify(p,p['preset'],f'profile:{i}')
    timing=profile(root/'profile.json')
    driven=[r for r in records if r['mode']=='instrumented']
    summary=dict(functional_rows=len(driven),functional_pass=sum(not r['failed_gates'] for r in driven),
                 functional_failures=[r for r in driven if r['failed_gates']],
                 suite_pairs=len(suite_rows),suite_regressions=[r for r in suite_rows if r['regressed']],
                 suite_status_changes=[r for r in suite_rows if r['raw_status']!=r['instrumented_status']],
                 suite_missing=[r for r in suite_rows if 'MISSING' in (r['raw_status'],r['instrumented_status'])],
                 profile=timing,verified_records=len(verified),
                 ui_screenshot_exists=(root/'ui.png').exists())
    csv_write(out/'contract_rows.csv',records);csv_write(out/'suite_rows.csv',suite_rows)
    csv_write(out/'room_rows.csv',room_rows);csv_write(out/'source_checks.csv',verified)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    lines=['Ordered by seed 10..15, then batch row 0..7. No sorting by outcome or dropping rows.',
           'Source: contract_rows.csv; instrumented rows; one list per run. Rows share a brain RNG stream.']
    for seed in frozen['contract_seeds']:
        subset=[r for r in driven if r['seed']==seed]
        lines.append(f'seed {seed}:')
        for key in ('mean_strength','weak_fraction','stationary_error_p95_deg','moving_error_p95_deg',
                    'first_slope_deg_s','reverse_slope_deg_s','mean_peak_hz','mean_width_wedges','failed_gates'):
            lines.append(f'  {key}: '+json.dumps([r[key] if not isinstance(r[key],float) else round(r[key],4) for r in subset]))
    (out/'per_seed_lists.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    input_hashes(root)
    print(json.dumps(summary,indent=2))
    plot(root,out)


def plot(root,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for col,mode in enumerate(('raw','instrumented')):
        z=np.load(root/f'{mode}_s10.npz');r=z['rates'][:,6];w=z['wedges'];t=z['t']
        p=np.stack([r[:,w==i].mean(1) for i in range(16)],axis=1)
        moment=p@np.exp(1j*np.arange(16)*2*np.pi/16);strength=abs(moment)/np.maximum(p.sum(1),1e-9)
        phase=np.unwrap(np.angle(moment))*180/np.pi;phase[strength<.6]=np.nan
        expected=np.unwrap(z['expected'][:,6])*180/np.pi
        image=axes[0,col].imshow(p.T,origin='lower',aspect='auto',extent=(t[0],t[-1],-.5,15.5),vmin=0,vmax=70,cmap='magma')
        axes[0,col].set_title(mode+(' / imposed angular memory' if col else ' / plain brain'))
        axes[0,col].set_ylabel('EPG wedge');axes[1,col].plot(t,expected,'k--',label='imposed heading')
        axes[1,col].plot(t,phase,color='#168c9b',label='neural phase (strength >=0.6)')
        axes[1,col].set_xlabel('Time (s)');axes[1,col].set_ylabel('Unwrapped heading (deg)')
        axes[1,col].set_ylim(-20,560);axes[1,col].legend(fontsize=8,loc='lower right')
    fig.suptitle('Full brain, seed 10 row 6: stationary / +180 deg/s / stationary / reverse / stationary')
    fig.subplots_adjust(top=.88,hspace=.22,bottom=.09,right=.87)
    fig.colorbar(image,cax=fig.add_axes((.89,.535,.015,.345)),label='Actual EPG spike rate (Hz)')
    fig.savefig(out/'turn_trace.png',dpi=160);plt.close(fig)


def scheduler(original, updated):
    original=Path(original);updated=Path(updated);out=updated/'analysis';out.mkdir(exist_ok=True)
    frozen=read(updated/'predeclared.json');exact=[];source_checks=[]
    def verify(p,label):
        assert p['execution']['device']=='cuda',label
        fp=p['source_fingerprint'];assert fp['computed'],label
        files=fp['files_lf'];want=frozen['source_sha256_lf']
        matched=[k for k in want if k in files]
        assert len(matched)>30 and all(files[k]==want[k] for k in matched),label
        source_checks.append(dict(record=label,matched_files=len(matched)))
    def compare(stem):
        old=np.load(original/(stem+'.npz'));new=np.load(updated/(stem+'.npz'))
        assert old.files==new.files,stem
        arrays={k:np.array_equal(old[k],new[k]) for k in old.files}
        exact.append(dict(run=stem,arrays=arrays,passed=all(arrays.values())))
        verify(read(updated/(stem+'.json'))['provenance'],stem)
    for seed in range(10,16):compare(f'instrumented_s{seed}')
    compare('room_instrumented')
    timings={};identities={}
    for label in ('profile','profile_native'):
        p=read(updated/(label+'.json'))
        identities[label]=p['identity']
        timings[label]=profile(updated/(label+'.json'))
        for i,c in enumerate(p['controllers']):verify(c,f'{label}:{i}')
    check=read(updated/'cuda_checks.json');assert len(check['checks'])==9
    verify(check['provenance'],'cuda_checks')
    rooms={}
    for mode in ('raw','instrumented'):
        r=read(updated/f'room_native_{mode}.json');verify(r['provenance'],'native room '+mode)
        rooms[mode]={k:r[k] for k in ('frame_ms_median','frame_ms_p95','rows')}
    data=dict(exact=exact,exact_pass=all(r['passed'] for r in exact),timings=timings,
              matched_input_identity=identities,
              performance_pass=all(r['target_pass'] for rows in timings.values() for r in rows),
              native_rooms=rooms,cuda_checks=check['checks'],source_checks=source_checks)
    (out/'summary.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(data,indent=2))


def difference(left,right):
    a,b=np.load(left),np.load(right);assert a.files==b.files
    rows={}
    for key in a.files:
        mismatch=a[key]!=b[key]
        rows[key]=dict(exact=bool(np.array_equal(a[key],b[key])),different_values=int(np.count_nonzero(mismatch)),
                       max_abs=float(np.max(np.abs(a[key].astype(float)-b[key].astype(float)))) if a[key].size else 0.,
                       first_index=np.argwhere(mismatch)[0].tolist() if np.any(mismatch) else None)
    return rows


def captured(original,updated):
    original=Path(original);updated=Path(updated);out=updated/'analysis';out.mkdir(exist_ok=True)
    frozen=read(updated/'predeclared.json');checks=[];rows=[];exact={}
    def verify(p,label):
        assert p['execution']['device']=='cuda',label
        assert p['compiled_connectome']['md5']=='ef23cc27bea13be7f6a96f3c04fd3737',label
        fp=p['source_fingerprint'];assert fp['computed'],label
        files=fp['files_lf'];wanted=frozen['source_sha256_lf'];shared=[k for k in wanted if k in files]
        assert len(shared)>30 and all(files[k]==wanted[k] for k in shared),label
        checks.append(dict(record=label,matched_files=len(shared)))
    for seed in range(10,16):
        stem=f'instrumented_s{seed}';record=read(updated/(stem+'.json'));verify(record['provenance'],stem)
        assert record['module_graphs']>0 and record['batch']==8 and record['seconds']==12,stem
        exact[stem]=difference(original/(stem+'.npz'),updated/(stem+'.npz'))
        for r in measure(updated/(stem+'.npz')):rows.append(dict(seed=seed,**r,failed_gates=gates(r)))
    timing={};identities={}
    for name in ('profile_native','profile_eager'):
        r=read(updated/(name+'.json'));timing[name]=profile(updated/(name+'.json'));identities[name]=r['identity']
        for i,p in enumerate(r['controllers']):verify(p,f'{name}:{i}')
        for step in r['records']:
            if step['mode']=='instrumented':assert (step['module_graphs']>0)==(name=='profile_native')
    fixture=read(updated/'cuda_checks.json');assert len(fixture['checks'])==13;verify(fixture['provenance'],'cuda fixture')
    traces=read(updated/'trace/summary.json')['records'];trace_rows=[]
    for r in traces:
        verify(r['provenance'],'trace '+r['mode'])
        assert r['local_scalar_dense']==(40 if r['mode']=='checked' else 0),r['mode']
        trace_rows.append({k:r[k] for k in ('mode','frames','module_graphs','local_scalar_dense')})
    rooms={}
    for name in ('room_eager_a','room_eager_b','room_instrumented','room_native_raw','room_native_instrumented'):
        r=read(updated/(name+'.json'));verify(r['provenance'],name)
        assert r['seconds']==60 and r['batch']==6
        if name in ('room_instrumented','room_native_instrumented'):assert r['module_graphs']>0
        rooms[name]={k:r[k] for k in ('rows','frame_ms_median','frame_ms_p95','module_graphs')}
    room_compare={}
    for a,b in (('room_eager_a','room_eager_b'),('room_eager_a','room_instrumented')):
        room_compare[a+' vs '+b]=difference(updated/(a+'.npz'),updated/(b+'.npz'))
    data=dict(functional_rows=len(rows),functional_pass=sum(not r['failed_gates'] for r in rows),
              failed_rows=[r for r in rows if r['failed_gates']],contract_arrays=exact,timings=timing,
              matched_input_identity=identities,cuda_checks=fixture['checks'],trace=trace_rows,
              rooms=rooms,room_comparisons=room_compare,verified_records=len(checks))
    csv_write(out/'contract_rows.csv',rows);csv_write(out/'source_checks.csv',checks)
    (out/'summary.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    input_hashes(updated)
    print(json.dumps(data,indent=2))


def publication(root, capture_dir, lifetime_dir):
    """Emit the exact tables pasted into the audit; verify the complete frozen source snapshots."""
    root,capture_dir,lifetime_dir=map(Path,(root,capture_dir,lifetime_dir))
    out=root/'analysis';capture=read(capture_dir/'analysis/summary.json')
    final=read(lifetime_dir/'profile_native.json');fixture=read(lifetime_dir/'cuda_checks.json')
    assert len(fixture['checks'])==15
    frozen=read(lifetime_dir/'predeclared.json');verified=[]
    for i,p in enumerate([fixture['provenance'],*final['controllers']]):
        assert p['execution']['device']=='cuda'
        files=p['source_fingerprint']['files_lf'];shared=set(files)&set(frozen['source_sha256_lf'])
        assert 'flyverse/compass.py' in shared and len(shared)==51
        assert all(files[k]==frozen['source_sha256_lf'][k] for k in shared)
        verified.append(dict(record=i,matched_files=len(shared)))
    revisions=('497589d','89bf14b','07cf966','3aa676c','17205bb','3742a57')
    snapshots=[]
    for n,commit in enumerate(revisions,1):
        folder=root if n==1 else root.with_name(root.name+f'_r{n}')
        wanted=read(folder/'predeclared.json')['source_sha256_lf']
        # An id from before the 2026-09-17 rewrite is read through the map (docs/INTERP.md 10.4 rule 30).
        resolved=resolve_commit(commit)
        for path,h in wanted.items():
            data=subprocess.check_output(['git','show',f'{resolved}:{path}'],cwd=ROOT)
            assert hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()==h,(commit,path)
        stamped=resolved if resolved==commit else f'{resolved} (recorded {commit})'
        snapshots.append(dict(batch=folder.name,archive_commit=stamped,matched_files=len(wanted)))
    data=dict(source_snapshots=snapshots,final_provenance=verified,final_cuda_checks=fixture['checks'],
              final_profile=profile(lifetime_dir/'profile_native.json'),final_matched_input_identity=final['identity'])
    (out/'final_checks.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    (lifetime_dir/'analysis').mkdir(exist_ok=True)
    input_hashes(lifetime_dir)
    input_hashes(root.with_name(root.name+'_r2'))
    with (out/'suite_rows.csv').open() as f:suite=list(csv.DictReader(f))
    lines=['### Suite values by draw', '',
           'Source: `out/compass_standin/analysis/suite_rows.csv`, columns `raw_value` and `instrumented_value`.',
           'Each list is ordered by draw seed [0,1,2], unsorted by outcome. P=PASS, F=FAIL, G=KNOWN GAP.', '',
           '| check | raw [0,1,2] | instrumented [0,1,2] | raw / instrumented status |',
           '|---|---|---|---|']
    fmt=lambda v: str(v) if not isinstance(v,(float,int)) else f'{v:.6g}'
    for key in dict.fromkeys(r['key'] for r in suite):
        rows=[r for r in suite if r['key']==key];assert [int(r['seed']) for r in rows]==[0,1,2]
        values=['['+', '.join(fmt(float(r[k+'_value'])) for r in rows)+']' for k in ('raw','instrumented')]
        status=[''.join({'PASS':'P','FAIL':'F','KNOWN GAP':'G'}[r[k+'_status']] for r in rows) for k in ('raw','instrumented')]
        lines.append(f'| {key} | {values[0]} | {values[1]} | {status[0]} / {status[1]} |')
    lines+=['','### Room values by row','',
            'Sources: initial `room_rows.csv`; r5 `summary.json:rooms`. Lists follow environment seeds [10,11,12,13,14,15],',
            'all rows retained. Distance is XY path length between 0.1 s samples, not displacement toward fruit.', '', '```text']
    rooms={}
    with (out/'room_rows.csv').open() as f:initial=list(csv.DictReader(f))
    for mode in ('raw','instrumented'):rooms['initial '+mode]=[r for r in initial if r['mode']==mode]
    rooms.update({k:v['rows'] for k,v in capture['rooms'].items()})
    for name,rows in rooms.items():
        lines.append(name+':')
        for key in ('hops','distance_m','mean_abs_yaw_deg_s','mean_epg_hz'):
            lines.append('  '+key+': ['+', '.join(fmt(float(r[key])) for r in rows)+']')
    lines+=['```','','### Performance repeats','',
            'Sources: r5 and r6 `profile_native.json:records`, synchronized wall ms per 10 ms frame.',
            'Four alternating repeats in recorded order; repetitions on one device are not independent animals.', '', '```text']
    for folder in (capture_dir,lifetime_dir):
        for row in profile(folder/'profile_native.json'):
            lines.append(f'{folder.name} B={row["batch"]}: raw {row["raw_repeats_ms"]}; instrumented {row["instrumented_repeats_ms"]}')
    lines+=['```','']
    (out/'report_tables.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(data,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--runs',default='out/compass_standin')
    ap.add_argument('--profile-only',action='store_true');ap.add_argument('--scheduler',help='directory of scheduler rerun')
    ap.add_argument('--captured',help='directory of successful capture validation')
    ap.add_argument('--publication',action='store_true',help='verify final r6 and emit audit tables after both batches finish')
    a=ap.parse_args()
    if a.publication:publication(a.runs,'out/compass_standin_r5','out/compass_standin_r6')
    elif a.captured:captured(a.runs,a.captured)
    elif a.scheduler:scheduler(a.runs,a.scheduler)
    elif a.profile_only:print(json.dumps(profile(Path(a.runs)/'profile.json'),indent=2))
    else:analyse(a.runs)
