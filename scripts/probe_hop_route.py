"""Batched room rollout that attributes every take-off to its route: GF escape or voluntary wing power.

scripts/batch_sustain.py counts `hops` as false->true transitions of FlyState.airborne, which
flyverse/batch_body.py step() produces by EITHER route -- `escape = gf >= gf_threshold` (a giant-fibre
jump, body.Flight.launch(escape=True)) or `voluntary = power >= takeoff_power_hz held takeoff_hold_s`
(body.Flight.launch(escape=False)).  The round-4 sustain report calls the excess "spontaneous take-offs"
and proposes to score it with benchmark.py's sec_walk_gf counter, which sets `sim.flight.gf_hz = 1e9`
and so counts the voluntary route only.  This script decides which route the room hops actually use.

It reproduces batch_sustain.py's setup and loop exactly (same options, same BatchSim construction, same
initial headings/energy, same per-frame bookkeeping) and adds two things that do not touch the dynamics:

  * a wrapper on body.Flight.launch that records (row, t, escape flag) for every take-off, and
  * `--gf-hz X`, which sets every row's flight.gf_hz before the loop -- 1e9 disables the escape route
    entirely, the same trick benchmark.sec_walk_gf uses, so a run with it isolates voluntary take-offs.

    python scripts/probe_hop_route.py --batch 16 --seeds 0,..,15 --program cx --fruit apple --fence \
        --minutes 5 --energy 0.9 --cuda-graphs --cuda-kernels --event-driven --cuda-sparse torch \
        --json out/x.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyverse import BatchSim, body  # noqa: E402
from batch_sustain import add_options, patch_receptor, sim_options  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_options(ap)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--seeds', default='', help='comma-separated environment seeds; length must equal batch')
    ap.add_argument('--minutes', type=float, default=5.)
    ap.add_argument('--start', default='-0.15,0.15')
    ap.add_argument('--energy', type=float, default=.4)
    ap.add_argument('--log-every', type=float, default=10.)
    ap.add_argument('--gf-hz', type=float, default=None,
                    help='override every row flight.gf_hz (1e9 disables the GF escape route; default: leave 33 Hz)')
    ap.add_argument('--json', default='out/hop_route.json')
    args = ap.parse_args()
    if not np.isfinite([args.minutes, args.energy, args.log_every]).all() or args.minutes <= 0 \
            or args.log_every <= 0 or not 0 <= args.energy <= 1:
        ap.error('minutes/log-every must be positive; energy must be in [0,1]')
    try:
        start = tuple(float(v) for v in args.start.split(','))
        if len(start) == 2: start += (.75,)
        if len(start) != 3: raise ValueError()
        seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else None
    except ValueError:
        ap.error('invalid --start or --seeds')

    patch_receptor(args.receptor_model, args.receptor_net_rule)
    sim = BatchSim(args.batch, seeds=seeds, start=start, **sim_options(args))
    lp = sim.fb.brain.p
    receptor_info = dict(model=lp.receptor_model, net_rule=lp.receptor_net_rule if lp.receptor_model else None,
                         fast_sign_changed_entries=int((sim.fb.receptor.fast_sign != np.sign(sim.c.W.data)).sum())
                         if sim.fb.receptor is not None else 0)
    print(f'receptor model {receptor_info["model"]} ({receptor_info["net_rule"]}); fast sign changed on '
          f'{receptor_info["fast_sign_changed_entries"]:,} of {sim.c.W.nnz:,} entries', flush=True)

    for seed, fly, m in zip(sim.seeds, sim.flies, sim.metabolisms):
        fly.heading = np.random.default_rng(seed).uniform(-np.pi, np.pi)
        m.energy = args.energy
    if args.gf_hz is not None:
        for f in sim.flights: f.gf_hz = float(args.gf_hz)
    print(f'BatchSim B={sim.B} neurons={sim.c.n:,} device={sim.fb.device} environment seeds={sim.seeds}', flush=True)
    print(f'gf_hz per row {[f.gf_hz for f in sim.flights][:4]}... takeoff_power_hz '
          f'{sim.flights[0].takeoff_power_hz} hold {sim.flights[0].takeoff_hold_s} s', flush=True)

    # Route attribution: record every launch and its escape flag without changing the dynamics.
    row_of = {id(f): i for i, f in enumerate(sim.flies)}
    launches = []
    original = body.Flight.launch

    def launch(self, fly, escape):
        launches.append(dict(row=row_of.get(id(fly), -1), t_s=sim.fb.t / 1000., escape=bool(escape)))
        return original(self, fly, escape)

    body.Flight.launch = launch
    try:
        count = max(1, round(args.minutes * 60_000 / sim.FRAME_MS))
        interval = max(1, round(args.log_every * 1000 / sim.FRAME_MS))
        modes = [{} for _ in range(sim.B)]
        hops = np.zeros(sim.B, dtype=int); previous_air = np.array([f.airborne for f in sim.flies])
        air_frames = np.zeros(sim.B, dtype=int)
        path = np.zeros(sim.B); previous_xy = np.array([[f.x, f.y] for f in sim.flies])
        minimum = np.array([m.energy for m in sim.metabolisms])
        gf_max = np.zeros(sim.B); gf_sum = np.zeros(sim.B); power_max = np.zeros(sim.B)
        started = time.perf_counter()
        for k in range(count):
            sim.step()
            current_air = np.array([f.airborne for f in sim.flies])
            hops += current_air & ~previous_air; previous_air = current_air
            air_frames += current_air
            gf = np.array([w['gf'] for w in sim.wcommands]); pw = np.array([w['power'] for w in sim.wcommands])
            gf_max = np.maximum(gf_max, gf); gf_sum += gf; power_max = np.maximum(power_max, pw)
            xy = np.array([[f.x, f.y] for f in sim.flies]); path += np.linalg.norm(xy - previous_xy, axis=1)
            previous_xy = xy
            energy = np.array([m.energy for m in sim.metabolisms]); minimum = np.minimum(minimum, energy)
            for i, cmd in enumerate(sim.commands):
                mode = 'feeding' if sim.feeding[i] else cmd.get('mode', 'plain')
                modes[i][mode] = modes[i].get(mode, 0) + 1
            if (k + 1) % interval == 0 or k + 1 == count:
                esc = sum(1 for L in launches if L['escape']); vol = len(launches) - esc
                print(f't={sim.fb.t/1000:.2f}s energy={np.round(energy,3).tolist()} '
                      f'meals={[m.meals for m in sim.metabolisms]} hops={hops.tolist()} '
                      f'escape={esc} voluntary={vol}', flush=True)
        elapsed = time.perf_counter() - started
    finally:
        body.Flight.launch = original

    esc_per_row = np.zeros(sim.B, dtype=int); vol_per_row = np.zeros(sim.B, dtype=int)
    for L in launches:
        (esc_per_row if L['escape'] else vol_per_row)[L['row']] += 1
    _, distance = sim.nearest_fruit()
    rows = [dict(row=i, environment_seed=sim.seeds[i], energy=m.energy, min_energy=float(minimum[i]),
                 meals=m.meals, hops=int(hops[i]), escape_hops=int(esc_per_row[i]),
                 voluntary_hops=int(vol_per_row[i]), airborne_frac=float(air_frames[i] / count),
                 gf_max_hz=float(gf_max[i]), gf_mean_hz=float(gf_sum[i] / count),
                 power_max_hz=float(power_max[i]), path_m=float(path[i]),
                 distance_cm=float(distance[i] * 100), modes={k: v / count for k, v in modes[i].items()},
                 position=sim.flies[i].pos.tolist()) for i, m in enumerate(sim.metabolisms)]
    total_e, total_v = int(esc_per_row.sum()), int(vol_per_row.sum())
    result = dict(batch=sim.B, brain_seed=args.seed, frames=count, simulated_s=count * sim.FRAME_MS / 1000,
                  wall_s=elapsed, aggregate_fly_s_per_wall_s=sim.B * count * sim.FRAME_MS / 1000 / elapsed,
                  options=vars(args), receptor=receptor_info, gf_hz=float(sim.flights[0].gf_hz),
                  takeoff_power_hz=float(sim.flights[0].takeoff_power_hz),
                  hops_total=int(hops.sum()), escape_total=total_e, voluntary_total=total_v,
                  launches=launches, rows=rows,
                  rng_note='Rows are independent; the batched rollout is not bit-reproducible on CUDA.')
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(f'hops {int(hops.sum())} = escape {total_e} + voluntary {total_v}; '
          f'gf max over rows {gf_max.max():.2f} Hz (threshold {sim.flights[0].gf_hz:g}); '
          f'power max {power_max.max():.2f} Hz (threshold {sim.flights[0].takeoff_power_hz:g}); '
          f'airborne frac mean {air_frames.mean()/count:.4f}')
    print(f'wall {elapsed:.1f} s, {result["aggregate_fly_s_per_wall_s"]:.2f} fly-s per wall-s -> {args.json}')


if __name__ == '__main__':
    main()
