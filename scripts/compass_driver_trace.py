"""House CUDA trace: raw, synchronous-check control, eager-module and captured-module frames."""
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
    if (out/'summary.json').exists():raise FileExistsError(out)
    if not torch.cuda.is_available():raise RuntimeError('house CUDA only')
    out.mkdir(parents=True,exist_ok=True);c=connectome.load(verbose=False);records=[]
    for mode in ('raw','checked','eager','captured'):
        fb=FlyBrain(c,device='cuda',optic=None,cuda_graphs=True,cuda_kernels=True,cuda_sparse='warp',
                    lif_params=LIFParams(event_driven=True),preset='raw' if mode=='raw' else 'instrumented',
                    instruments=[] if mode=='raw' else ['compass'])
        driver=fb.instruments.get('compass')
        if mode in ('checked','eager'):driver.cuda_graph_safe=False
        if mode=='checked':
            driver.cuda_async_validation=False;driver.poisson_always_on=False
        fb.brain.set_poisson(c.indices({'type':'EPG'}),50.)
        def frame():
            if driver:
                if mode=='checked':
                    # The original blocking upload, with the same held zero-yaw input.
                    driver.yaw_rate.copy_(torch.as_tensor(np.zeros((1,1),np.float32),device=fb.device))
                else:driver.observe_turn(0.)
            fb.step(10.)
        for _ in range(50):frame()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(20):frame()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(out/(mode+'.json')))
        averages=prof.key_averages()
        records.append(dict(mode=mode,frames=20,module_graphs=sum(isinstance(g,tuple) for g in fb._graphs.values()),
                            local_scalar_dense=sum(e.count for e in averages if e.key=='aten::_local_scalar_dense'),
                            cpu_ops=[dict(name=e.key,count=e.count,self_cpu_us=e.self_cpu_time_total,
                                          self_device_us=e.self_device_time_total) for e in averages],
                            provenance=provenance(c,fb=fb,seeds=[0],stimulus={'name':'scheduler trace','params':{'mode':mode,'held_EPG_hz':50,'yaw':0}})))
        fb=None;torch.cuda.empty_cache()   # not `del`: frame() closes over fb and ruff reads `del` as unbinding it (F821)
    (out/'summary.json').write_text(json.dumps(to_jsonable(dict(records=records)),indent=2)+'\n',encoding='utf-8')
    print('device cuda',[(r['mode'],r['local_scalar_dense'],r['module_graphs']) for r in records],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True);a=ap.parse_args();run(a.out)
