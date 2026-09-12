"""Run independent full room rollouts through one batched brain, without a UI.

python scripts/batch_sustain.py --batch 8 --minutes 5 --program cx --fruit apple --fence --cuda-graphs --cuda-kernels --event-driven --json out/batch.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flyverse import BatchSim


def add_options(ap):
    ap.add_argument('--seed',type=int,default=0,help='batched brain RNG seed; first environment seed')
    ap.add_argument('--program',default='none')
    ap.add_argument('--fruit',choices=('all','apple'),default='all')
    ap.add_argument('--fence',action='store_true')
    ap.add_argument('--escape-gating',action='store_true')
    ap.add_argument('--brain-dt',type=float,default=.5)
    ap.add_argument('--optic-dt',type=float,default=1.)
    ap.add_argument('--cuda-graphs',action='store_true')
    ap.add_argument('--cuda-kernels',action=argparse.BooleanOptionalAction,default=None)
    ap.add_argument('--event-driven',action=argparse.BooleanOptionalAction,default=None)
    ap.add_argument('--cuda-sparse',choices=('torch','warp'),default='torch')
    ap.add_argument('--weight-dtype',choices=('float32','float16'),default='float32')
    ap.add_argument('--device',default=None)


def sim_options(args):
    return dict(seed=args.seed,program=args.program,fruit_set=args.fruit,fence=args.fence,
                escape_gating=args.escape_gating,brain_dt=args.brain_dt,optic_dt=args.optic_dt,
                cuda_graphs=args.cuda_graphs,cuda_kernels=args.cuda_kernels,event_driven=args.event_driven,
                cuda_sparse=args.cuda_sparse,weight_dtype=args.weight_dtype,device=args.device)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_options(ap)
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--seeds',default='',help='comma-separated environment seeds; length must equal batch')
    ap.add_argument('--minutes',type=float,default=5.)
    ap.add_argument('--start',default='-0.15,0.15',help='x,y or x,y,z, shared initial position')
    ap.add_argument('--energy',type=float,default=.4)
    ap.add_argument('--log-every',type=float,default=10.,help='simulated seconds between progress lines')
    ap.add_argument('--json',default='out/batch_sustain.json')
    ap.add_argument('--save',default='',help='save the final batched checkpoint')
    ap.add_argument('--load',default='',help='resume a matching batched checkpoint')
    args = ap.parse_args()
    if not np.isfinite([args.minutes,args.energy,args.log_every]).all() or args.minutes<=0 or args.log_every<=0 or not 0<=args.energy<=1:
        ap.error('minutes/log-every must be positive; energy must be in [0,1]')
    try:
        start = tuple(float(v) for v in args.start.split(','))
        if len(start)==2: start += (.75,)
        if len(start)!=3: raise ValueError()
        seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else None
    except ValueError: ap.error('invalid --start or --seeds')
    sim = BatchSim(args.batch,seeds=seeds,start=start,**sim_options(args))
    if args.load: sim.load_state(args.load)
    else:
        for seed,fly,m in zip(sim.seeds,sim.flies,sim.metabolisms):
            fly.heading = np.random.default_rng(seed).uniform(-np.pi,np.pi)
            m.energy = args.energy
    print(f'BatchSim B={sim.B} neurons={sim.c.n:,} device={sim.fb.device} environment seeds={sim.seeds}',flush=True)
    print('Environment seeds select scenes/plumes/initial headings; the batched brain has its own shared RNG stream.',flush=True)
    count = max(1,round(args.minutes*60_000/sim.FRAME_MS))
    interval = max(1,round(args.log_every*1000/sim.FRAME_MS))
    modes = [{} for _ in range(sim.B)]
    hops = np.zeros(sim.B,dtype=int); previous_air = np.array([f.airborne for f in sim.flies])
    path = np.zeros(sim.B); previous_xy = np.array([[f.x,f.y] for f in sim.flies])
    minimum = np.array([m.energy for m in sim.metabolisms])
    started = time.perf_counter()
    for k in range(count):
        sim.step()
        current_air = np.array([f.airborne for f in sim.flies]); hops += current_air&~previous_air; previous_air=current_air
        xy = np.array([[f.x,f.y] for f in sim.flies]); path += np.linalg.norm(xy-previous_xy,axis=1); previous_xy=xy
        energy = np.array([m.energy for m in sim.metabolisms]); minimum = np.minimum(minimum,energy)
        for i,cmd in enumerate(sim.commands):
            mode = 'feeding' if sim.feeding[i] else cmd.get('mode','plain')
            modes[i][mode] = modes[i].get(mode,0)+1
        if (k+1)%interval==0 or k+1==count:
            print(f't={sim.fb.t/1000:.2f}s energy={np.round(energy,3).tolist()} meals={[m.meals for m in sim.metabolisms]} hops={hops.tolist()}',flush=True)
    elapsed = time.perf_counter()-started
    _,distance = sim.nearest_fruit()
    rows = [dict(row=i,environment_seed=sim.seeds[i],energy=m.energy,min_energy=float(minimum[i]),meals=m.meals,
                 hops=int(hops[i]),path_m=float(path[i]),distance_cm=float(distance[i]*100),
                 modes={k:v/count for k,v in modes[i].items()},position=sim.flies[i].pos.tolist()) for i,m in enumerate(sim.metabolisms)]
    result = dict(batch=sim.B,brain_seed=args.seed,frames=count,simulated_s=count*sim.FRAME_MS/1000,
                  wall_s=elapsed,aggregate_fly_s_per_wall_s=sim.B*count*sim.FRAME_MS/1000/elapsed,
                  options=vars(args),rows=rows,
                  rng_note='Rows are independent; changing batch size changes brain RNG draw layout. These are not bit-identical replays of single-seed processes.')
    Path(args.json).parent.mkdir(parents=True,exist_ok=True)
    Path(args.json).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    if args.save: sim.save_state(args.save)
    print(f'{sim.B} rollouts saved to {args.json}; {result["aggregate_fly_s_per_wall_s"]:.2f} aggregate fly-seconds/wall-second',flush=True)


if __name__ == '__main__': main()
