"""Batched room comparison and timing for the optional compass. No body program is installed.

This is an engineering/side-effect observation, not the 300-second adoption rate-half or a food-finding claim.
Default six independently evolving rooms for 60 s; writes every trajectory and controller provenance.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flyverse import BatchSim
from flyverse.compass import epg_columns
from flyverse.interp.common import provenance,to_jsonable


def run(a):
    dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.with_suffix('.json').exists():raise FileExistsError(dest)
    sim=BatchSim(a.batch,seed=a.seed,seeds=range(a.seed,a.seed+a.batch),device=a.device,
                 cuda_graphs=a.device=='cuda',program='none',fruit_set='apple',fence=True,
                 start=(-.15,.15,.75),preset=a.mode,instruments=['compass'] if a.mode=='instrumented' else [],
                 cuda_kernels=True if a.native else None,event_driven=True if a.native else None)
    if a.scheduler=='eager' and sim.fb.instruments:
        sim.fb.instruments['compass'].cuda_graph_safe=False
    idx,w=epg_columns(sim.c);rec=[];rates=[];timings=[]
    hops=np.zeros(a.batch,int);previous=np.array([f.airborne for f in sim.flies])
    for frame in range(round(a.seconds*100)):
        start=time.perf_counter();sim.step();timings.append(time.perf_counter()-start)
        airborne=np.array([f.airborne for f in sim.flies]);hops+=(airborne&~previous);previous=airborne
        if frame%10==9:
            rec.append([[f.x,f.y,f.z,f.heading,f.yaw_rate,f.speed,float(f.airborne)] for f in sim.flies])
            rates.append(sim.brain.rate[:,sim.brain._idx(idx)].clone())
    if a.device=='cuda':torch.cuda.synchronize()
    values=np.asarray(rec);rt=torch.stack(rates).cpu().numpy()
    np.savez_compressed(dest.with_suffix('.npz'),body=values,epg=rt,wedges=w,dt_record_s=.1)
    distance=np.hypot(np.diff(values[:,:,0],axis=0),np.diff(values[:,:,1],axis=0)).sum(axis=0)
    prov=provenance(sim.c,fb=sim.fb,seeds=[a.seed],env_seeds=list(sim.seeds),batch=a.batch,
                    stimulus={'name':'compass room observation','params':vars(a),'program':'none'})
    result=dict(mode=a.mode,seconds=a.seconds,batch=a.batch,seed=a.seed,device=str(sim.fb.device),provenance=prov,
                rows=[dict(row=i,env_seed=sim.seeds[i],hops=int(hops[i]),distance_m=float(distance[i]),
                           mean_abs_yaw_deg_s=float(np.rad2deg(abs(values[:,i,4])).mean()),
                           mean_epg_hz=float(rt[:,i].mean())) for i in range(a.batch)],
                frame_ms_median=float(np.median(timings[100:])*1000),frame_ms_p95=float(np.percentile(timings[100:],95)*1000),
                graphs=len(sim.fb._graphs),module_graphs=sum(isinstance(g,tuple) for g in sim.fb._graphs.values()))
    dest.with_suffix('.json').write_text(json.dumps(to_jsonable(result),indent=2)+'\n',encoding='utf-8')
    print('device',sim.fb.device,'graphs',len(sim.fb._graphs),'median ms',result['frame_ms_median'],flush=True)
    print(result['rows'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['raw','instrumented'],default='instrumented')
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--batch',type=int,default=6)
    ap.add_argument('--seconds',type=float,default=60.);ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    ap.add_argument('--native',action='store_true')
    ap.add_argument('--scheduler',choices=['captured','eager'],default='captured',help='eager is the previous module scheduler for reproducibility controls')
    ap.add_argument('--out',required=True);a=ap.parse_args()
    if a.seconds<2:ap.error('seconds must be >=2')
    run(a)
