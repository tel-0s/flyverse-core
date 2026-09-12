"""Compare complete batched room frames and a scalar demo baseline in one process.

Timing is descriptive on a shared GPU. Work counts and scalar-equivalence tests establish
what was batched independently of contention. All configs retain the full connectome.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from collections import Counter
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flyverse import BatchSim
from batch_sustain import add_options,sim_options


def measure(sim, warmup, frames, batch, trace=None):
    cuda = sim.fb.device.type=='cuda'
    for _ in range(warmup): sim.step()
    if cuda: torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    times,events = [],[]
    for _ in range(frames):
        if cuda:
            start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
        before = time.perf_counter(); sim.step()
        if cuda: end.record(); end.synchronize(); events.append(start.elapsed_time(end))
        times.append((time.perf_counter()-before)*1000)
    result = dict(batch=batch,median_ms=float(np.median(times)),p90_ms=float(np.percentile(times,90)),
                  fly_frames_per_s=batch*1000/float(np.median(times)),fly_s_per_wall_s=batch*10/float(np.median(times)),
                  cuda_span_ms=float(np.median(events)) if cuda else None,
                  peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20 if cuda else None,
                  neurons=sim.c.n,brain_batch=sim.fb.B)
    # Time host-side environment stages separately using the real current poses/motors.
    if isinstance(sim,BatchSim):
        from flyverse.batch_body import frames as pose_frames
        motor = sim.fb.motor()
        snapshots = sim.state_dict()
        before = time.perf_counter()
        for _ in range(frames):
            eye,f,left,_ = pose_frames(sim.flies)
            sim.air.step(.01); sim.air.antennae(eye,left,f); sim.air.deflections(f,left); sim.nearest_fruit()
            cmd,wcmd = sim.body.readout(motor,.01); sim.body.step(cmd,wcmd,sim.tasting,.01)
        result['body_and_physical_senses_ms'] = (time.perf_counter()-before)*1000/frames
        sim.load_state_dict(snapshots)
        result['work_per_frame'] = dict(brain_frames=1,ray_batches=1 if sim.optic is not None else 0,
                                      sensory_encodings=len(sim.fb.available_senses),motor_readouts=1)
    if trace:
        for _ in range(warmup): sim.step()  # restore above cleared captured neural frames
        activities = [torch.profiler.ProfilerActivity.CPU]+([torch.profiler.ProfilerActivity.CUDA] if cuda else [])
        with torch.profiler.profile(activities=activities,record_shapes=True) as profile:
            for _ in range(3): sim.step()
            if cuda: torch.cuda.synchronize()
        profile.export_chrome_trace(trace)
        gpu = [e for e in profile.events() if e.device_type==torch.autograd.DeviceType.CUDA]
        result['profile'] = dict(frames=3,gpu_events=len(gpu),gpu_event_us=sum(e.self_device_time_total for e in gpu),
                                 most_frequent_gpu_events=Counter(e.name for e in gpu).most_common(12))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__); add_options(ap)
    ap.add_argument('--batches',default='1,4,8,16')
    ap.add_argument('--frames',type=int,default=50)
    ap.add_argument('--warmup',type=int,default=10)
    ap.add_argument('--scalar',action='store_true',help='also measure the existing room_demo.Sim (B=1)')
    ap.add_argument('--json',default='out/profile_batch.json')
    ap.add_argument('--trace',default='',help='directory for three-frame CPU/CUDA traces per configuration')
    args = ap.parse_args()
    if args.device: os.environ['FLYVERSE_DEVICE'] = args.device
    try: batches=[int(b) for b in args.batches.split(',')]
    except ValueError: ap.error('--batches must be comma-separated integers')
    if not batches or min(batches)<1 or args.frames<1 or args.warmup<0: ap.error('positive batches/frames and nonnegative warmup required')
    if args.trace: Path(args.trace).mkdir(parents=True,exist_ok=True)
    report = dict(options=vars(args),device=torch.cuda.get_device_name() if torch.cuda.is_available() and args.device!='cpu' else args.device,
                  timing_note='CUDA spans include GPU idle gaps while Python prepares work. A shared GPU can inflate both spans and wall time.',results=[])
    configs = [('scalar',1)] if args.scalar else []
    configs += [('batch',b) for b in batches]
    for kind,b in configs:
        print(f'constructing {kind} B={b}',flush=True)
        options = sim_options(args)
        if kind=='scalar':
            from room_demo import Sim
            options.pop('device')
            sim = Sim(**options,trail_seconds=0)
        else: sim = BatchSim(b,**options)
        trace = str(Path(args.trace)/f'{kind}_{b}.json') if args.trace else None
        result = measure(sim,args.warmup,args.frames,b,trace)
        result['kind']=kind; report['results'].append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='profile'}),flush=True)
        if 'profile' in result:
            print('profile:',json.dumps({k:v for k,v in result['profile'].items() if k!='most_frequent_gpu_events'}),flush=True)
        Path(args.json).parent.mkdir(parents=True,exist_ok=True)
        Path(args.json).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        del sim; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__': main()
