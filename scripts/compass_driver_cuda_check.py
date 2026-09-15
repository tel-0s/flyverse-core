"""House-only GPU equivalence of async compass plumbing and the original checked module path.

This is a small EPG-subgraph engineering fixture, not a compass physiology probe.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flyverse import connectome
from flyverse.fly import FlyBrain
from flyverse.brain import LIFParams
from flyverse.interp.common import provenance,to_jsonable


def run(out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    if not torch.cuda.is_available():raise RuntimeError('house CUDA only')
    parent=connectome.load(verbose=False);c=parent.subset(parent.indices({'type':'EPG'}))
    params=LIFParams(receptor_model=None)
    a,b=[FlyBrain(c,batch=3,device='cuda',optic=None,preset='instrumented',instruments=['compass'],
                  seed=31,lif_params=params,cuda_graphs=True) for _ in range(2)]
    b.instruments['compass'].cuda_async_validation=False
    b.instruments['compass'].poisson_always_on=False
    checks=[]
    def equal(label):
        for name in FlyBrain.BRAIN_TENSORS:
            torch.testing.assert_close(getattr(a.brain,name),getattr(b.brain,name),rtol=0,atol=0,msg=label+':'+name)
        torch.testing.assert_close(a.instruments['compass'].phase,b.instruments['compass'].phase,rtol=0,atol=0)
        checks.append(label)
    for frame in range(80):
        yaw=np.array([1.,-2.,.5])*(1 if frame<40 else -1)
        for fb in (a,b):
            fb.proprioception(0,0,0,False,yaw_rate=yaw)
            if frame==20:fb.stimulate(np.arange(8),100.,3.)
            fb.step(10.)
        if frame in (0,20,40,79):equal(f'frame {frame}')
    for fb in (a,b):fb.reset(rows=[1])
    for fb in (a,b):fb.step(10.)
    equal('partial reset')
    state=a.state_dict();a.step(50.);expected={k:getattr(a.brain,k).clone() for k in FlyBrain.BRAIN_TENSORS}
    a.load_state_dict(state);a.step(50.)
    for k,v in expected.items():torch.testing.assert_close(getattr(a.brain,k),v,rtol=0,atol=0)
    checks.append('checkpoint replay')
    b.step(50.);equal('checkpoint paired reference')
    for fb in (a,b):fb.reset(rows=[0,1,2]);assert not fb.brain._poisson_on
    checks.append('partial reset of all rows disables zero Poisson')
    p=provenance(c,fb=a,seeds=[31],stimulus={'name':'async versus checked compass scheduler','fixture':'EPG subgraph',
                  'control':'same module math; async validation disabled and activity-proof hint disabled'})
    for fb in (a,b):fb.detach('compass');fb.step(10.)
    for k in FlyBrain.BRAIN_TENSORS:torch.testing.assert_close(getattr(a.brain,k),getattr(b.brain,k),rtol=0,atol=0)
    checks.append('detach')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(to_jsonable(dict(device='cuda',checks=checks,provenance=p)),indent=2)+'\n',encoding='utf-8')
    print('device cuda; all',len(checks),'exact-state checks passed',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True);a=ap.parse_args();run(a.out)
