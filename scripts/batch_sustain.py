"""Run independent full room rollouts through one batched brain, without a UI.

python scripts/batch_sustain.py --batch 8 --minutes 5 --program cx --fruit apple --fence --cuda-graphs --cuda-kernels --event-driven --json out/batch.json

Every take-off is attributed to its route by flyverse.batch_body (escape: GF >= Flight.gf_hz; voluntary: wing power >=
takeoff_power_hz held takeoff_hold_s): each row carries hops (airborne transitions), hops_escape, hops_voluntary and
the walking-GF maximum; --gf-hz 1e9 disables the escape route for a voluntary-only run.
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
    ap.add_argument('--receptor-model',choices=('default','off','sign'),default='default',
                    help="LIFParams.receptor_model: 'default' leaves the LIFParams default (sign / abs since round 3), "
                         "'off' selects the presynaptic-sign rule (receptor_model=None), 'sign' the receptor lookup")
    ap.add_argument('--receptor-net-rule',choices=('class','abs','nonmda'),default='abs',help="with --receptor-model sign")


def patch_receptor(model,net_rule,table=None):
    """Make every brain.LIFParams built from here on (BatchSim's included) carry the requested receptor model
    (LIFParams.receptor_model; docs/NT_INTEGRATION.md). 'default' with no table leaves the class untouched.
    `table` is a LIFParams.receptor_table path (a receptors_by_type.csv other than the shipped one, e.g. the hold
    tables of scripts/build_hold_tables.py); it is applied whenever the receptor stage is on, so --receptor-table
    works with --receptor-model default as well as sign."""
    if model=='default' and table is None: return
    from flyverse import brain
    L = brain.LIFParams
    def make(**kw):
        p = L(**kw)
        if model!='default':
            p.receptor_model = None if model=='off' else model
            p.receptor_net_rule = net_rule
        if table is not None and p.receptor_model is not None: p.receptor_table = table
        return p
    brain.LIFParams = make


def sim_options(args):
    return dict(seed=args.seed,program=args.program,fruit_set=args.fruit,fence=args.fence,
                escape_gating=args.escape_gating,brain_dt=args.brain_dt,optic_dt=args.optic_dt,
                cuda_graphs=args.cuda_graphs,cuda_kernels=args.cuda_kernels,event_driven=args.event_driven,
                cuda_sparse=args.cuda_sparse,weight_dtype=args.weight_dtype,device=args.device)


def _lif_dump(lp):
    """The resolved LIFParams (every field, JSON-safe) so an arm's identity lives in the JSON rather than in prose."""
    import dataclasses
    out={}
    for f in dataclasses.fields(lp):
        v=getattr(lp,f.name)
        try: json.dumps(v); out[f.name]=v
        except TypeError: out[f.name]=repr(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_options(ap)
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--seeds',default='',help='comma-separated environment seeds; length must equal batch')
    ap.add_argument('--minutes',type=float,default=5.)
    ap.add_argument('--start',default='-0.15,0.15',help='x,y or x,y,z, shared initial position')
    ap.add_argument('--energy',type=float,default=.4)
    ap.add_argument('--log-every',type=float,default=10.,help='simulated seconds between progress lines')
    ap.add_argument('--gf-hz',type=float,default=None,
                    help="override every row's body.Flight.gf_hz before the loop (1e9 disables the GF escape route, leaving the "
                         "voluntary wing-power route as the only take-off; default: leave Flight.gf_hz, 33 Hz)")
    ap.add_argument('--receptor-table',default=None,metavar='PATH',
                    help='LIFParams.receptor_table: a receptors_by_type.csv other than flyverse/data/receptors_by_type.csv '
                         '(e.g. out/receptors_holdBrain.csv from scripts/build_hold_tables.py); no effect with '
                         '--receptor-model off, which switches the receptor stage off')
    ap.add_argument('--json',default='out/batch_sustain.json')
    ap.add_argument('--save',default='',help='save the final batched checkpoint')
    ap.add_argument('--load',default='',help='resume a matching batched checkpoint')
    args = ap.parse_args()
    if not np.isfinite([args.minutes,args.energy,args.log_every]).all() or args.minutes<=0 or args.log_every<=0 or not 0<=args.energy<=1:
        ap.error('minutes/log-every must be positive; energy must be in [0,1]')
    if args.gf_hz is not None and not args.gf_hz>0: ap.error('--gf-hz must be positive')
    if args.receptor_table is not None:
        if not Path(args.receptor_table).is_file(): ap.error(f'--receptor-table {args.receptor_table}: no such file')
        if args.receptor_model=='off': ap.error('--receptor-table has no effect with --receptor-model off (the receptor stage is off)')
    try:
        start = tuple(float(v) for v in args.start.split(','))
        if len(start)==2: start += (.75,)
        if len(start)!=3: raise ValueError()
        seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else None
    except ValueError: ap.error('invalid --start or --seeds')
    patch_receptor(args.receptor_model,args.receptor_net_rule,args.receptor_table)
    sim = BatchSim(args.batch,seeds=seeds,start=start,**sim_options(args))
    lp = sim.fb.brain.p
    receptor_info = dict(model=lp.receptor_model,net_rule=lp.receptor_net_rule if lp.receptor_model else None,
                         table=lp.receptor_table if lp.receptor_model else None,
                         fast_sign_changed_entries=int((sim.fb.receptor.fast_sign!=np.sign(sim.c.W.data)).sum()) if sim.fb.receptor is not None else 0)
    print(f'receptor model {receptor_info["model"]} ({receptor_info["net_rule"]}) table {receptor_info["table"] or "flyverse/data/receptors_by_type.csv"}; '
          f'fast sign changed on {receptor_info["fast_sign_changed_entries"]:,} of {sim.c.W.nnz:,} entries',flush=True)
    if args.load: sim.load_state(args.load)
    else:
        for seed,fly,m in zip(sim.seeds,sim.flies,sim.metabolisms):
            fly.heading = np.random.default_rng(seed).uniform(-np.pi,np.pi)
            m.energy = args.energy
    if args.gf_hz is not None:
        for f in sim.flights: f.gf_hz = float(args.gf_hz)
    flight = sim.flights[0]
    print(f'BatchSim B={sim.B} neurons={sim.c.n:,} device={sim.fb.device} environment seeds={sim.seeds}',flush=True)
    print('Environment seeds select scenes/plumes/initial headings; the batched brain has its own shared RNG stream.',flush=True)
    print(f'take-off routes: escape at GF >= {flight.gf_hz:g} Hz (landing refractory {flight.landing_refractory_s:g} s); '
          f'voluntary at wing power >= {flight.takeoff_power_hz:g} Hz held {flight.takeoff_hold_s:g} s',flush=True)
    count = max(1,round(args.minutes*60_000/sim.FRAME_MS))
    interval = max(1,round(args.log_every*1000/sim.FRAME_MS))
    modes = [{} for _ in range(sim.B)]
    hops = np.zeros(sim.B,dtype=int); previous_air = np.array([f.airborne for f in sim.flies])
    path = np.zeros(sim.B); previous_xy = np.array([[f.x,f.y] for f in sim.flies])
    minimum = np.array([m.energy for m in sim.metabolisms])
    # the GF the flight model compares with gf_hz, sampled while the fly is on the ground (the frames a launch can start from)
    gf_max_walk = np.zeros(sim.B); gf_max = np.zeros(sim.B); power_max = np.zeros(sim.B); air_frames = np.zeros(sim.B,dtype=int)
    started = time.perf_counter()
    for k in range(count):
        sim.step()
        gf = np.array([w['gf'] for w in sim.wcommands]); power = np.array([w['power'] for w in sim.wcommands])
        gf_max_walk = np.where(previous_air,gf_max_walk,np.maximum(gf_max_walk,gf)); gf_max = np.maximum(gf_max,gf); power_max = np.maximum(power_max,power)
        current_air = np.array([f.airborne for f in sim.flies]); hops += current_air&~previous_air; previous_air=current_air
        air_frames += current_air
        xy = np.array([[f.x,f.y] for f in sim.flies]); path += np.linalg.norm(xy-previous_xy,axis=1); previous_xy=xy
        energy = np.array([m.energy for m in sim.metabolisms]); minimum = np.minimum(minimum,energy)
        for i,cmd in enumerate(sim.commands):
            mode = 'feeding' if sim.feeding[i] else cmd.get('mode','plain')
            modes[i][mode] = modes[i].get(mode,0)+1
        if (k+1)%interval==0 or k+1==count:
            print(f't={sim.fb.t/1000:.2f}s energy={np.round(energy,3).tolist()} meals={[m.meals for m in sim.metabolisms]} hops={hops.tolist()} '
                  f'escape={int(sim.hops_escape.sum())} voluntary={int(sim.hops_voluntary.sum())}',flush=True)
    elapsed = time.perf_counter()-started
    escape,voluntary = sim.hops_escape.copy(),sim.hops_voluntary.copy()
    if not np.array_equal(hops,escape+voluntary):
        print(f'WARNING: airborne transitions {hops.tolist()} != escape {escape.tolist()} + voluntary {voluntary.tolist()}',flush=True)
    _,distance = sim.nearest_fruit()
    rows = [dict(row=i,environment_seed=sim.seeds[i],energy=m.energy,min_energy=float(minimum[i]),meals=m.meals,
                 hops=int(hops[i]),hops_escape=int(escape[i]),hops_voluntary=int(voluntary[i]),
                 gf_max_walk_hz=float(gf_max_walk[i]),gf_max_hz=float(gf_max[i]),power_max_hz=float(power_max[i]),
                 airborne_frac=float(air_frames[i]/count),path_m=float(path[i]),distance_cm=float(distance[i]*100),
                 modes={k:v/count for k,v in modes[i].items()},position=sim.flies[i].pos.tolist()) for i,m in enumerate(sim.metabolisms)]
    fly_s = sim.B*count*sim.FRAME_MS/1000
    result = dict(batch=sim.B,brain_seed=args.seed,frames=count,simulated_s=count*sim.FRAME_MS/1000,fly_s=fly_s,
                  wall_s=elapsed,aggregate_fly_s_per_wall_s=fly_s/elapsed,
                  options=vars(args),receptor=receptor_info,
                  lif=_lif_dump(lp),device=str(sim.fb.device),cache_sum_abs_W=float(abs(sim.c.W).sum()),
                  flight=dict(gf_hz=float(flight.gf_hz),takeoff_power_hz=float(flight.takeoff_power_hz),takeoff_hold_s=float(flight.takeoff_hold_s),
                              landing_refractory_s=float(flight.landing_refractory_s)),
                  hops_total=int(hops.sum()),hops_escape_total=int(escape.sum()),hops_voluntary_total=int(voluntary.sum()),
                  hops_escape_per_1000_fly_s=float(escape.sum()/fly_s*1000),hops_voluntary_per_1000_fly_s=float(voluntary.sum()/fly_s*1000),
                  gf_max_walk_median_hz=float(np.median(gf_max_walk)),rows_gf_at_threshold=int((gf_max_walk>=33.0).sum()),
                  rows=rows,
                  rng_note='Rows are independent; changing batch size changes brain RNG draw layout. These are not bit-identical replays of single-seed processes.')
    Path(args.json).parent.mkdir(parents=True,exist_ok=True)
    Path(args.json).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    if args.save: sim.save_state(args.save)
    print(f'hops {result["hops_total"]} = escape {result["hops_escape_total"]} + voluntary {result["hops_voluntary_total"]} over {fly_s:,.0f} fly-s '
          f'({result["hops_escape_per_1000_fly_s"]:.2f} / {result["hops_voluntary_per_1000_fly_s"]:.2f} per 1,000 fly-s); '
          f'walking GF max per row median {result["gf_max_walk_median_hz"]:.2f} Hz, {result["rows_gf_at_threshold"]}/{sim.B} rows at or above {33.0:g} Hz',flush=True)
    print(f'{sim.B} rollouts saved to {args.json}; {result["aggregate_fly_s_per_wall_s"]:.2f} aggregate fly-seconds/wall-second',flush=True)


if __name__ == '__main__': main()
