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


def run(out, native=False, module_eager=False):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    if not torch.cuda.is_available():raise RuntimeError('run this profile on house CUDA')
    c=connectome.load(verbose=False);records=[];controllers=[];identity=[]
    for batch in (1,8,32):
        flags=dict(cuda_kernels=True,cuda_sparse='warp' if batch==1 else 'torch') if native else {}
        from flyverse.brain import LIFParams
        params=LIFParams(event_driven=True) if native else None
        brains={mode:FlyBrain(c,batch=batch,device='cuda',optic=None,cuda_graphs=True,seed=0,
                              preset=mode,instruments=['compass'] if mode=='instrumented' else [],lif_params=params,**flags)
                for mode in ('raw','instrumented')}
        # Same external sensory forcing establishes comparable Poisson work in both controllers.
        for fb in brains.values():
            if module_eager and fb.instruments:
                fb.instruments['compass'].cuda_graph_safe=False
            controllers.append(provenance(c,fb=fb,seeds=[0],batch=batch,
                               stimulus={'name':'matched tonic profile','EPG_hz':50.,'yaw_rate':0.}))
            fb.brain.set_poisson(c.indices({'type':'EPG'}),50.)
            for _ in range(50):fb.step(10.)
        for repeat in range(4):
            for mode in (('raw','instrumented') if repeat%2==0 else ('instrumented','raw')):
                fb=brains[mode]
                torch.cuda.synchronize();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start=time.perf_counter();begin.record()
                for _ in range(100):fb.step(10.)
                end.record();end.synchronize();wall=(time.perf_counter()-start)*10
                records.append(dict(batch=batch,mode=mode,repeat=repeat,wall_ms_per_frame=wall,cuda_ms_per_frame=begin.elapsed_time(end)/100,
                                    module_graphs=sum(isinstance(g,tuple) for g in fb._graphs.values())))
        for fb in brains.values():
            if fb.instruments:
                module=fb.instruments['compass'];start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(1000):module.step(10.,{})
                end.record();end.synchronize()
                records.append(dict(batch=batch,mode='module_only',cuda_ms_per_frame=start.elapsed_time(end)/1000))
        a,b=brains['raw'].brain,brains['instrumented'].brain
        identity.append(dict(batch=batch,tensors={name:torch.equal(getattr(a,name),getattr(b,name)) for name in FlyBrain.BRAIN_TENSORS},
                             max_abs={name:float((getattr(a,name).float()-getattr(b,name).float()).abs().max()) if getattr(a,name).numel() else 0. for name in FlyBrain.BRAIN_TENSORS}))
        del brains,fb;torch.cuda.empty_cache()
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(to_jsonable(dict(records=records,controllers=controllers,identity=identity,native=native,module_eager=module_eager,
                   protocol='four alternating repeats of 100 frames after 50 warmup frames; identical held-setter tonic EPG forcing in both arms (no timed pulse); module-only timing is separate')),indent=2)+'\n',encoding='utf-8')
    print('device cuda',records,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True)
    ap.add_argument('--native',action='store_true');ap.add_argument('--module-eager',action='store_true')
    a=ap.parse_args();run(a.out,a.native,a.module_eager)
