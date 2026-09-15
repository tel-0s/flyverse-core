"""Profile module overhead independently of body work; fixed B=1,8,32, alternating raw/instrumented.

CUDA only, house only. Both end-to-end synchronized wall time and CUDA elapsed time are reported.
Shared-machine timings are observations, not idle-device guarantees. No brain state is adopted.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flyverse import connectome
from flyverse.fly import FlyBrain
from flyverse.interp.common import provenance,to_jsonable


def run(out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    if not torch.cuda.is_available():raise RuntimeError('run this profile on house CUDA')
    c=connectome.load(verbose=False);records=[];controllers=[]
    for batch in (1,8,32):
        brains={mode:FlyBrain(c,batch=batch,device='cuda',optic=None,cuda_graphs=True,seed=0,
                              preset=mode,instruments=['compass'] if mode=='instrumented' else [])
                for mode in ('raw','instrumented')}
        # Same external sensory forcing establishes comparable Poisson work in both controllers.
        for fb in brains.values():
            controllers.append(provenance(c,fb=fb,seeds=[0],batch=batch,
                               stimulus={'name':'matched tonic profile','EPG_hz':50.,'yaw_rate':0.}))
            fb.stimulate({'type':'EPG'},50.,100000.)
            for _ in range(50):fb.step(10.)
        for repeat in range(4):
            for mode in (('raw','instrumented') if repeat%2==0 else ('instrumented','raw')):
                fb=brains[mode]
                torch.cuda.synchronize();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start=time.perf_counter();begin.record()
                for _ in range(100):fb.step(10.)
                end.record();end.synchronize();wall=(time.perf_counter()-start)*10
                records.append(dict(batch=batch,mode=mode,repeat=repeat,wall_ms_per_frame=wall,cuda_ms_per_frame=begin.elapsed_time(end)/100))
        for fb in brains.values():
            if fb.instruments:
                module=fb.instruments['compass'];start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(1000):module.step(10.,{})
                end.record();end.synchronize()
                records.append(dict(batch=batch,mode='module_only',cuda_ms_per_frame=start.elapsed_time(end)/1000))
        del brains,fb;torch.cuda.empty_cache()
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(to_jsonable(dict(records=records,controllers=controllers,
                   protocol='four alternating repeats of 100 frames after 50 warmup frames; identical tonic EPG external forcing in both arms; module-only timing is separate')),indent=2)+'\n',encoding='utf-8')
    print('device cuda',records,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True);a=ap.parse_args();run(a.out)
