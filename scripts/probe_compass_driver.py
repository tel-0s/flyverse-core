"""Engineering contract for the imposed compass, including the actual neural readout.

CPU smoke: --device cpu --seconds 1 --batch 1. Full declaration: six seeds, raw/compass,
12 s each, B=8 with velocities [-180,-90,-45,0,45,90,180,90] deg/s. Stop [0,2), turn
[2,5), stop [5,7), reverse [7,10), stop [10,12). No sensory cue, edge hold or NT relabel.
The last row starts at a different phase to check rotation equivariance descriptively.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flyverse import connectome
from flyverse.compass import epg_columns
from flyverse.fly import FlyBrain
from flyverse.interp.common import provenance, to_jsonable


SPEEDS=np.array([-180.,-90.,-45.,0.,45.,90.,180.,90.])


def measure(path):
    with np.load(path) as z:
        t=z['t']; rates=z['rates']; w=z['wedges']; expected=z['expected']; yaw=z['yaw_deg_s']
        profile=np.stack([rates[:,:,w==i].mean(axis=2) for i in range(16)],axis=2)
        moment=profile@np.exp(1j*np.arange(16)*2*np.pi/16)
        strength=np.abs(moment)/np.maximum(profile.sum(axis=2),1e-9)
        phase=np.angle(moment)
        error=np.abs(np.angle(np.exp(1j*(phase-expected))))*180/np.pi
        out=[]
        for row in range(rates.shape[1]):
            valid=t>=.5
            moving=valid&(yaw[:,row]!=0)
            stationary=valid&(yaw[:,row]==0)&~(((t>=5)&(t<5.5))|((t>=10)&(t<10.5)))
            slopes=[]
            for lo,hi in ((2.5,5.),(7.5,10.)):
                window=(t>=lo)&(t<hi)
                slopes.append(float(np.polyfit(t[window],np.unwrap(phase[window,row]),1)[0]*180/np.pi)
                              if window.sum()>2 and strength[window,row].mean()>=.6 else None)
            width=(profile[:,row]>.5*profile[:,row].max(axis=1,keepdims=True)).sum(axis=1)
            out.append(dict(row=row,speed_deg_s=float(SPEEDS[row]),mean_strength=float(strength[valid,row].mean()),
                            weak_fraction=float((strength[valid,row]<.6).mean()),
                            stationary_error_p95_deg=float(np.percentile(error[stationary,row],95)) if stationary.any() else None,
                            moving_error_p95_deg=float(np.percentile(error[moving,row],95)) if moving.any() else None,
                            first_slope_deg_s=slopes[0],reverse_slope_deg_s=slopes[1],mean_width_wedges=float(width[valid].mean()),
                            mean_peak_hz=float(profile[valid,row].max(axis=1).mean()),
                            mean_epg_hz=float(rates[valid,row].mean())))
        return out


def run(args):
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.with_suffix('.json').exists():raise FileExistsError(out)
    c=connectome.load(verbose=False)
    fb=FlyBrain(c,batch=args.batch,device=args.device,seed=args.seed,optic=None,
                preset=args.mode,instruments=['compass'] if args.mode=='instrumented' else [],
                cuda_graphs=args.device=='cuda')
    idx,w=epg_columns(c)
    driver=fb.instruments.get('compass')
    if driver is not None and args.batch==8:
        driver.phase[7]=np.pi/2
    offsets=np.zeros(args.batch);offsets[7:]=np.pi/2
    phases=offsets.copy(); rates=[]; expected=[]; yaws=[]; times=[]
    if args.device=='cuda':torch.cuda.synchronize()
    start=time.perf_counter()
    for frame in range(round(args.seconds*100)):
        t=frame/100
        sign=1 if 2<=t<5 else -1 if 7<=t<10 else 0
        yaw=SPEEDS[:args.batch]*sign
        phases=np.remainder(phases+np.deg2rad(yaw)*.01,2*np.pi)
        if driver is not None:fb.proprioception(0,0,0,False,yaw_rate=np.deg2rad(yaw))
        fb.step(10.)
        rates.append(fb.brain.rate[:,fb.brain._idx(idx)].clone())
        expected.append(phases.copy()); yaws.append(yaw.copy()); times.append((frame+1)/100)
    if args.device=='cuda':torch.cuda.synchronize()
    wall=time.perf_counter()-start
    np.savez_compressed(out.with_suffix('.npz'),rates=torch.stack(rates).cpu().numpy(),wedges=w,t=times,
                        expected=expected,yaw_deg_s=yaws)
    records=measure(out.with_suffix('.npz'))
    prov=provenance(c,fb=fb,seeds=[args.seed],stimulus={'name':'compass_driver_contract','params':vars(args),
                    'initial_phase_rad':offsets.tolist(),'speeds_deg_s':SPEEDS[:args.batch].tolist()})
    result=dict(mode=args.mode,seed=args.seed,batch=args.batch,seconds=args.seconds,device=str(fb.device),
                wall_s=wall,records=records,provenance=prov,graphs=len(fb._graphs),
                module_graphs=sum(isinstance(g,tuple) for g in fb._graphs.values()),
                source_sha256_lf=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest())
    out.with_suffix('.json').write_text(json.dumps(to_jsonable(result),indent=2)+'\n',encoding='utf-8')
    print('device',fb.device,'graphs',len(fb._graphs),'wall',wall,flush=True)
    for record in records:print(record,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['raw','instrumented'],default='instrumented')
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--seconds',type=float,default=12.)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    if a.batch not in range(1,9) or a.seconds<.6:ap.error('batch 1-8 and seconds >=0.6 required')
    run(a)
