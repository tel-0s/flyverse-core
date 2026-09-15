"""CPU checks for the fixed direct-GLNO challenge and its declared inputs. No submission."""
import copy
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import cx8_transfer as tr
from flyverse import connectome


def test_primary_family_keeps_run_ids_signs_and_invalid_batch_guard():
    rows=[]
    for arm,(cut,side) in tr.ARMS.items():
        sign=1 if side=='L' else -1 if side=='R' else 0
        for seed in range(6):
            rows.append(dict(arm=arm,run_id=f'{arm}_s{seed}.json#0',GLNO_LR_hz=sign*10+seed*.01,
                             PEN_LR_hz=(0 if cut else sign*5)+seed*.01))
    df=pd.DataFrame(rows);family=tr.compare_family(df)
    assert len(family)==4 and all(r['verdict_holm']=='result' and r['m']==4 for r in family)
    assert [np.sign(r['diff']) for r in family]==[1,1,1,-1]
    assert all(len(r['stim_run_ids'])==len(r['null_run_ids'])==6 for r in family)
    assert all(r['p_holm']<=.05 for r in family)
    assert all(r['verdict_holm']=='undetermined' for r in tr.compare_family(df,['missing artifact']))


def test_scheduled_neural_stimulus_resolves_ids_and_leaves_the_graph_alone():
    c=SimpleNamespace(n=3,neurons=pd.DataFrame({'bodyId':[10,20,30]}))
    e=dict(name='test',idx=[2],poisson_hz=90.,start_s=.3,duration_s=.3,law='unverified')
    original=copy.deepcopy(e)
    row=tr.cw.prepare_neural_stimuli(c,[e],.8)[0]
    assert row['body_ids']==['30'] and row['idx']==[2]
    assert e==original and tr.cw.prepare_neural_stimuli(c,None,.8)==[]
    for change in [dict(idx=[3]),dict(idx=[-1]),dict(idx=[1,1]),dict(idx=[.5]),dict(start_s=.301),
                   dict(duration_s=0),dict(duration_s=.6),dict(poisson_hz=-1),dict(poisson_hz=float('nan'))]:
        with pytest.raises(ValueError):tr.cw.prepare_neural_stimuli(c,[dict(e,**change)],.8)


@pytest.mark.skipif(not (connectome.CACHE_DIR/'W_post_pre.npz').exists(),reason='needs MaleCNS cache')
def test_one_submission_contains_every_condition_per_seed_and_freezes(tmp_path):
    tr.plan(tmp_path)
    manifest=json.loads((tmp_path/'arms.json').read_text())
    shell=(tmp_path/'batch.sh').read_text()
    assert shell.startswith('#!/bin/bash\nset -o pipefail\n') and shell.count('cluster_run.py')==1
    assert '--target house' in shell and '--arm-block fam' in shell and shell.rstrip().endswith('|| exit $?')
    jobs=[x for x in shlex.split(shell.splitlines()[-1]) if x.startswith('mkdir -p')]
    assert len(jobs)==6
    for seed,job in enumerate(jobs):
        assert f'--seed {seed} --block fam_s{seed}' in job and '--arms' not in job and '--smoke' not in job
    assert list(manifest['arms'])==list(tr.ARMS) and len(manifest['family'])==4
    for arm,e in manifest['arms'].items():
        cut,side=tr.ARMS[arm]
        assert e['lif']['receptor_net_rule']=='abs' and e['lif']['same_type_gain']==.1
        if cut:
            edge=e['hold_edges_resolved'][-1]
            assert edge['n_entries']==84 and edge['synapses']==16371 and edge['n_post_cells']==42
        assert len(e['neural_stimuli'])==bool(side)
        if side:assert len(e['neural_stimuli'][0]['body_ids'])==2 and e['neural_stimuli'][0]['poisson_hz']==90
    tr.freeze(tmp_path)
    frozen=(tmp_path/'predeclared.json').read_bytes()
    with pytest.raises(FileExistsError):tr.freeze(tmp_path)
    with pytest.raises(ValueError):tr.plan(tmp_path)
    assert (tmp_path/'predeclared.json').read_bytes()==frozen


@pytest.mark.skipif(not (connectome.CACHE_DIR/'W_post_pre.npz').exists(),reason='needs MaleCNS cache')
def test_actual_direct_glno_smoke_has_the_declared_model_and_side(tmp_path):
    rows=tr.run(tmp_path,0,'fam_s0',arms=['HL','HR','CL'],device='cpu',no_graphs=True,smoke=True)
    c,_,_=tr.cw.load_connectome({'GLNO':'glutamate'},root=ROOT/'out')
    for r in rows:
        side=tr.ARMS[r['arm']][1];other='R' if side=='L' else 'L';e=tr.expected(c,r['arm'])
        assert r['provenance']['model']['lif']==e['lif']
        assert r['provenance']['instruments']==e['instrument_records']
        assert r['neural_stimuli'][0]['body_ids']==e['neural_stimuli'][0]['body_ids']
        assert r['neural_stimuli_applied']==[dict(name=f'direct_GLNO_{side}',start_s=.3)]
        assert r['metrics'][f'GLNO_{side}_hz_turn']>r['metrics'][f'GLNO_{other}_hz_turn']+5
        assert not r['metrics']['turn_fed'] and r['transfer_probe']['smoke']
        with np.load(r['ledger_npz']) as z:
            expected=np.zeros((80,1),np.float32);expected[30:60]=90
            np.testing.assert_array_equal(z['neural_stimulus_hz'],expected)
