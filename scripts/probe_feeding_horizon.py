"""Feeding-capable protocol: what horizon / metabolic drain makes meals a countable measure?

    python scripts/probe_feeding_horizon.py --batch 16 --minutes 10 --energy 0.9 --drain-scale 1.0 \
        --program cx --fruit apple --fence --cuda-graphs --cuda-kernels --event-driven --cuda-sparse torch \
        --seed 0 --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --json out/feedh_default_d1.0_10min_s0.json

    python scripts/probe_feeding_horizon.py --report out/feedh_*.json            # aggregate / compare arms

Why this script exists (docs/audits/receptor_verification.md round 5, handover item 3): under the shipped
`body.Metabolism` a 0.9 tank is empty by t = 170-180 s and 13-16 of 16 flies sit at energy 0 by 300 s, so meals
(0.17-0.23 per fly per 5 min) and final energy (a floor at 0) cannot rank models.  `flyverse/body.py` is a shared
file and the drain default is its owner's decision, so this probe changes NOTHING in the model: it builds a normal
`BatchSim` and then writes `drain_per_s` / `walk_drain_per_m` on each fly's own `Metabolism` object before the
loop.  `batch_body._metabolism` reads both parameters off those scalar objects every frame
(docs/BATCH_SIM.md, "Changing a body's parameters affects the next batch update"), so `--drain-scale 0.5` is
exactly the experiment "what would half the drain give?" with no edit to `body.py`.

It is an experiment, not a proposed default (THE PROJECT RULE): the arms below apply a metabolic gain to ask a
question.  Everything else -- room, brain, programs, senses, take-off routes -- is the shipped default.

Recorded per fly: meals and their times, first meal, the energy trajectory, the time the tank first empties and
the fraction of the run spent at 0, distance to the fruit (final, minimum and when), fruit contacts (the
`tasting` frames that a meal needs), mode fractions, path length and the escape / voluntary hop split.  Counts
are also split at the moment the tank empties (before / after), because `Metabolism` has no death and no motor
consequence at energy 0 -- hunger merely saturates at 1 -- so the before / after contact rate is the direct test
of whether starvation limits food-finding at all.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ZERO = 1e-9                     # Metabolism clips energy at 0.0; this is the "empty tank" test


# ---------------------------------------------------------------------------------------------- options
# The simulator / receptor options mirror scripts/batch_sustain.py's (deliberately duplicated rather than
# imported: that file is under concurrent edit, and a probe that ships to the cluster should not break with it).

def add_sim_options(ap):
    ap.add_argument('--seed', type=int, default=0, help='batched brain RNG seed; first environment seed')
    ap.add_argument('--seeds', default='', help='comma-separated environment seeds; length must equal --batch')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--program', default='cx')
    ap.add_argument('--fruit', choices=('all', 'apple'), default='apple')
    ap.add_argument('--fence', action='store_true')
    ap.add_argument('--escape-gating', action='store_true')
    ap.add_argument('--brain-dt', type=float, default=.5)
    ap.add_argument('--optic-dt', type=float, default=1.)
    ap.add_argument('--cuda-graphs', action='store_true')
    ap.add_argument('--cuda-kernels', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--event-driven', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--cuda-sparse', choices=('torch', 'warp'), default='torch')
    ap.add_argument('--weight-dtype', choices=('float32', 'float16'), default='float32')
    ap.add_argument('--device', default=None)
    ap.add_argument('--receptor-model', choices=('default', 'off', 'sign'), default='default',
                    help="LIFParams.receptor_model: 'default' leaves the shipped default (sign / abs since round 3), "
                         "'off' selects the presynaptic-sign rule (receptor_model=None), 'sign' the receptor lookup")
    ap.add_argument('--receptor-net-rule', choices=('class', 'abs', 'nonmda'), default='abs',
                    help='with --receptor-model sign')


def patch_receptor(model, net_rule):
    """Make every brain.LIFParams built from here on (BatchSim's included) carry the requested receptor model
    (docs/NT_INTEGRATION.md). 'default' leaves the class untouched."""
    if model == 'default': return
    from flyverse import brain
    L = brain.LIFParams

    def make(**kw):
        p = L(**kw)
        p.receptor_model = None if model == 'off' else model
        p.receptor_net_rule = net_rule
        return p
    brain.LIFParams = make


def sim_options(args):
    return dict(seed=args.seed, program=args.program, fruit_set=args.fruit, fence=args.fence,
                escape_gating=args.escape_gating, brain_dt=args.brain_dt, optic_dt=args.optic_dt,
                cuda_graphs=args.cuda_graphs, cuda_kernels=args.cuda_kernels, event_driven=args.event_driven,
                cuda_sparse=args.cuda_sparse, weight_dtype=args.weight_dtype, device=args.device)


# ---------------------------------------------------------------------------------------------- the run

def run(args):
    from flyverse import BatchSim
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else None
    start = tuple(float(v) for v in args.start.split(','))
    if len(start) == 2: start += (.75,)
    patch_receptor(args.receptor_model, args.receptor_net_rule)
    sim = BatchSim(args.batch, seeds=seeds, start=start, **sim_options(args))
    B, dt = sim.B, sim.FRAME_MS / 1000
    lp = sim.fb.brain.p
    receptor = dict(model=lp.receptor_model, net_rule=lp.receptor_net_rule if lp.receptor_model else None,
                    fast_sign_changed_entries=int((sim.fb.receptor.fast_sign != np.sign(sim.c.W.data)).sum())
                    if sim.fb.receptor is not None else 0)
    print(f'receptor model {receptor["model"]} ({receptor["net_rule"]}); fast sign changed on '
          f'{receptor["fast_sign_changed_entries"]:,} of {sim.c.W.nnz:,} entries', flush=True)

    # initial state: the same convention as scripts/batch_sustain.py, so the drain-scale 1.0 arm is comparable
    # with the round-4 / round-5 sustain batches (heading from the environment seed, shared energy).
    for seed, fly, m in zip(sim.seeds, sim.flies, sim.metabolisms):
        fly.heading = np.random.default_rng(seed).uniform(-np.pi, np.pi)
        m.energy = args.energy

    # THE ONE PARAMETER CHANGE, on the per-fly Metabolism objects, not in body.py.
    base = (sim.metabolisms[0].drain_per_s, sim.metabolisms[0].walk_drain_per_m)
    rest = base[0] if args.drain_per_s is None else args.drain_per_s
    walk = base[1] if args.walk_drain_per_m is None else args.walk_drain_per_m
    rest, walk = rest * args.drain_scale, walk * args.drain_scale
    for m in sim.metabolisms:
        m.drain_per_s, m.walk_drain_per_m = float(rest), float(walk)
    metabolism = dict(energy0=args.energy, drain_scale=args.drain_scale,
                      drain_per_s=float(rest), walk_drain_per_m=float(walk),
                      shipped_drain_per_s=float(base[0]), shipped_walk_drain_per_m=float(base[1]),
                      feed_per_s=float(sim.metabolisms[0].feed_per_s), satiety=float(sim.metabolisms[0].satiety),
                      resume_below=float(sim.metabolisms[0].resume_below))
    flight = sim.flights[0]
    count = max(1, round(args.minutes * 60_000 / sim.FRAME_MS))
    print(f'BatchSim B={B} neurons={sim.c.n:,} device={sim.fb.device} environment seeds={sim.seeds}', flush=True)
    print(f'metabolism per fly: energy0 {args.energy:g}, drain {rest:.6g}/s + {walk:.6g}/m '
          f'(drain-scale {args.drain_scale:g} on the shipped {base[0]:.6g}/s + {base[1]:.6g}/m), '
          f'feed {metabolism["feed_per_s"]:.4g}/s, satiety {metabolism["satiety"]:g}, resume {metabolism["resume_below"]:g}',
          flush=True)
    print(f'horizon {args.minutes:g} min = {count} frames of {sim.FRAME_MS:g} ms; '
          f'take-off routes: escape at GF >= {flight.gf_hz:g} Hz, voluntary at power >= {flight.takeoff_power_hz:g} Hz '
          f'held {flight.takeoff_hold_s:g} s', flush=True)

    # accumulators
    meals = np.zeros(B, dtype=int); meal_times = [[] for _ in range(B)]
    energy_sum = np.zeros(B); min_energy = np.full(B, np.inf); frames_zero = np.zeros(B, dtype=int)
    t_zero = np.full(B, np.nan)
    contact_frames = np.zeros(B, dtype=int); contact_episodes = np.zeros(B, dtype=int)
    first_contact = np.full(B, np.nan); prev_taste = np.zeros(B, dtype=bool)
    feeding_frames = np.zeros(B, dtype=int)
    min_distance = np.full(B, np.inf); t_min_distance = np.full(B, np.nan)
    near = {cm: np.zeros(B, dtype=int) for cm in (2., 5., 10.)}
    hops = np.zeros(B, dtype=int); previous_air = np.array([f.airborne for f in sim.flies])
    air_frames = np.zeros(B, dtype=int)
    path = np.zeros(B); previous_xy = np.array([[f.x, f.y] for f in sim.flies])
    gf_max_walk = np.zeros(B); power_max = np.zeros(B)
    modes = [{} for _ in range(B)]; modes_after = [{} for _ in range(B)]
    # split at the frame the tank first empties: Metabolism has no death, so this tests whether an empty tank
    # changes anything the fly does.
    split = dict(frames=np.zeros(B, dtype=int), path=np.zeros(B), meals=np.zeros(B, dtype=int),
                 contacts=np.zeros(B, dtype=int))       # the "after energy 0" halves
    traj_every = max(1, round(args.trajectory_s * 1000 / sim.FRAME_MS))
    traj_t, traj_energy, traj_distance = [], [], []
    checkpoints = []
    cp_every = max(1, round(args.checkpoint_s * 1000 / sim.FRAME_MS)) if args.checkpoint_s else 0

    started = time.perf_counter()
    for k in range(count):
        sim.step()
        t = (k + 1) * dt
        _, distance = sim.nearest_fruit()
        distance_cm = distance * 100
        taste = np.asarray(sim.tasting, dtype=bool)
        energy = np.array([m.energy for m in sim.metabolisms])
        now_meals = np.array([m.meals for m in sim.metabolisms])
        new_meal = now_meals > meals
        for i in np.flatnonzero(new_meal): meal_times[i].append(round(t, 3))
        meals = now_meals
        energy_sum += energy; min_energy = np.minimum(min_energy, energy)
        empty = energy <= ZERO
        frames_zero += empty
        first_empty = empty & np.isnan(t_zero); t_zero[first_empty] = t
        was_empty = ~np.isnan(t_zero)                       # already empty before / at this frame
        contact_frames += taste
        rise = taste & ~prev_taste; contact_episodes += rise
        newly = rise & np.isnan(first_contact); first_contact[newly] = t
        prev_taste = taste
        feeding_frames += np.asarray(sim.feeding, dtype=bool)
        better = distance_cm < min_distance
        t_min_distance[better] = t; min_distance = np.minimum(min_distance, distance_cm)
        for cm, acc in near.items(): acc += distance_cm <= cm
        current_air = np.array([f.airborne for f in sim.flies])
        hops += current_air & ~previous_air; previous_air = current_air; air_frames += current_air
        xy = np.array([[f.x, f.y] for f in sim.flies]); step_m = np.linalg.norm(xy - previous_xy, axis=1)
        path += step_m; previous_xy = xy
        gf = np.array([w['gf'] for w in sim.wcommands]); power = np.array([w['power'] for w in sim.wcommands])
        gf_max_walk = np.where(current_air, gf_max_walk, np.maximum(gf_max_walk, gf))
        power_max = np.maximum(power_max, power)
        split['frames'] += was_empty; split['path'] += np.where(was_empty, step_m, 0.)
        split['meals'] += was_empty & new_meal; split['contacts'] += was_empty & rise
        for i, cmd in enumerate(sim.commands):
            mode = 'feeding' if sim.feeding[i] else cmd.get('mode', 'plain')
            modes[i][mode] = modes[i].get(mode, 0) + 1
            if was_empty[i]: modes_after[i][mode] = modes_after[i].get(mode, 0) + 1
        if (k + 1) % traj_every == 0 or k + 1 == count:
            traj_t.append(round(t, 3)); traj_energy.append([round(float(v), 5) for v in energy])
            traj_distance.append([round(float(v), 3) for v in distance_cm])
        if cp_every and ((k + 1) % cp_every == 0 or k + 1 == count):
            checkpoints.append(dict(t_s=round(t, 3), meals=meals.tolist(), meals_per_fly=float(meals.mean()),
                                    flies_with_meal=int((meals > 0).sum()), contacts=contact_episodes.tolist(),
                                    flies_with_contact=int((contact_episodes > 0).sum()),
                                    energy_mean=float(energy.mean()), rows_at_zero=int(empty.sum()),
                                    min_distance_cm_median=float(np.median(min_distance)),
                                    hops=int(hops.sum()), path_m_mean=float(path.mean())))
        if (k + 1) % max(1, round(args.log_every * 1000 / sim.FRAME_MS)) == 0 or k + 1 == count:
            print(f't={t:.1f}s energy_mean={energy.mean():.3f} rows_at_zero={int(empty.sum())} '
                  f'meals={meals.tolist()} contacts={contact_episodes.tolist()} '
                  f'min_dist_cm={np.round(min_distance, 1).tolist()} hops={int(hops.sum())}', flush=True)
    elapsed = time.perf_counter() - started
    escape, voluntary = sim.hops_escape.copy(), sim.hops_voluntary.copy()
    if not np.array_equal(hops, escape + voluntary):
        print(f'WARNING: airborne transitions {hops.tolist()} != escape {escape.tolist()} + voluntary {voluntary.tolist()}',
              flush=True)
    _, distance = sim.nearest_fruit()
    before_frames = count - split['frames']
    rows = []
    for i, m in enumerate(sim.metabolisms):
        rows.append(dict(
            row=i, environment_seed=sim.seeds[i],
            energy=float(m.energy), min_energy=float(min_energy[i]), mean_energy=float(energy_sum[i] / count),
            meals=int(m.meals), meal_times_s=meal_times[i],
            first_meal_s=(meal_times[i][0] if meal_times[i] else None),
            t_zero_s=(None if np.isnan(t_zero[i]) else float(t_zero[i])),
            frac_at_zero=float(frames_zero[i] / count), s_at_zero=float(frames_zero[i] * dt),
            contacts=int(contact_episodes[i]), contact_s=float(contact_frames[i] * dt),
            first_contact_s=(None if np.isnan(first_contact[i]) else float(first_contact[i])),
            feeding_s=float(feeding_frames[i] * dt),
            distance_cm=float(distance[i] * 100), min_distance_cm=float(min_distance[i]),
            t_min_distance_s=(None if np.isnan(t_min_distance[i]) else float(t_min_distance[i])),
            near_frac={f'{cm:g}cm': float(acc[i] / count) for cm, acc in near.items()},
            hops=int(hops[i]), hops_escape=int(escape[i]), hops_voluntary=int(voluntary[i]),
            gf_max_walk_hz=float(gf_max_walk[i]), power_max_hz=float(power_max[i]),
            airborne_frac=float(air_frames[i] / count), path_m=float(path[i]),
            modes={key: v / count for key, v in modes[i].items()},
            # before / after the tank empties (an empty tank has no motor consequence in body.Metabolism)
            before_zero=dict(s=float(before_frames[i] * dt), path_m=float(path[i] - split['path'][i]),
                             meals=int(m.meals - split['meals'][i]),
                             contacts=int(contact_episodes[i] - split['contacts'][i])),
            after_zero=dict(s=float(split['frames'][i] * dt), path_m=float(split['path'][i]),
                            meals=int(split['meals'][i]), contacts=int(split['contacts'][i]),
                            modes={key: v / max(1, int(split['frames'][i])) for key, v in modes_after[i].items()}),
            position=sim.flies[i].pos.tolist()))
    fly_s = B * count * sim.FRAME_MS / 1000
    result = dict(
        probe='feeding_horizon', batch=B, brain_seed=args.seed, frames=count,
        simulated_s=count * sim.FRAME_MS / 1000, fly_s=fly_s, wall_s=elapsed,
        aggregate_fly_s_per_wall_s=fly_s / elapsed, options=vars(args), receptor=receptor,
        metabolism=metabolism,
        flight=dict(gf_hz=float(flight.gf_hz), takeoff_power_hz=float(flight.takeoff_power_hz),
                    takeoff_hold_s=float(flight.takeoff_hold_s),
                    landing_refractory_s=float(flight.landing_refractory_s)),
        summary=summarise(rows, count, dt, fly_s),
        trajectory=dict(t_s=traj_t, energy=traj_energy, distance_cm=traj_distance),
        checkpoints=checkpoints, rows=rows,
        rng_note='Rows are independent; changing batch size changes the brain RNG draw layout. Not bit-identical '
                 'replays of single-seed processes.')
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    s = result['summary']
    print(f"meals {s['meals_total']} = {s['meals_per_fly']:.3f} per fly over {args.minutes:g} min "
          f"({s['flies_with_meal']}/{B} flies fed); contacts {s['contacts_total']} "
          f"({s['flies_with_contact']}/{B} flies reached the fruit); first meal median "
          f"{fmt(s['first_meal_s_median'])} s; tank empty in {s['flies_at_zero']}/{B} at median "
          f"{fmt(s['t_zero_s_median'])} s, {s['frac_at_zero_mean']*100:.1f} % of the run at 0", flush=True)
    print(f"min distance to fruit median {s['min_distance_cm_median']:.1f} cm (best {s['min_distance_cm_min']:.1f}); "
          f"path {s['path_m_mean']:.2f} m; contact rate before / after the tank empties "
          f"{fmt(s['contacts_per_1000_s_before_zero'])} / {fmt(s['contacts_per_1000_s_after_zero'])} per 1,000 fly-s; "
          f"hops {s['hops_total']} = {s['hops_escape_total']} escape + {s['hops_voluntary_total']} voluntary", flush=True)
    print(f'{B} rollouts saved to {args.json}; {result["aggregate_fly_s_per_wall_s"]:.2f} aggregate fly-s/wall-s',
          flush=True)
    return result


def fmt(x):
    return 'n/a' if x is None else f'{x:.1f}'


def summarise(rows, count, dt, fly_s):
    def col(k): return np.array([r[k] for r in rows], dtype=float)
    firsts = [r['first_meal_s'] for r in rows if r['first_meal_s'] is not None]
    contacts_first = [r['first_contact_s'] for r in rows if r['first_contact_s'] is not None]
    zeros = [r['t_zero_s'] for r in rows if r['t_zero_s'] is not None]
    before_s = sum(r['before_zero']['s'] for r in rows); after_s = sum(r['after_zero']['s'] for r in rows)
    before_c = sum(r['before_zero']['contacts'] for r in rows); after_c = sum(r['after_zero']['contacts'] for r in rows)
    before_m = sum(r['before_zero']['meals'] for r in rows); after_m = sum(r['after_zero']['meals'] for r in rows)
    before_p = sum(r['before_zero']['path_m'] for r in rows); after_p = sum(r['after_zero']['path_m'] for r in rows)
    return dict(
        meals_total=int(col('meals').sum()), meals_per_fly=float(col('meals').mean()),
        flies_with_meal=int((col('meals') > 0).sum()),
        meals_countable=bool(col('meals').mean() >= 1.0),
        contacts_total=int(col('contacts').sum()), contacts_per_fly=float(col('contacts').mean()),
        flies_with_contact=int((col('contacts') > 0).sum()),
        first_meal_s_median=(float(np.median(firsts)) if firsts else None),
        first_contact_s_median=(float(np.median(contacts_first)) if contacts_first else None),
        energy_final_mean=float(col('energy').mean()), energy_mean_over_run=float(col('mean_energy').mean()),
        flies_at_zero=len(zeros), t_zero_s_median=(float(np.median(zeros)) if zeros else None),
        frac_at_zero_mean=float(col('frac_at_zero').mean()),
        min_distance_cm_median=float(np.median(col('min_distance_cm'))),
        min_distance_cm_min=float(col('min_distance_cm').min()),
        distance_cm_mean=float(col('distance_cm').mean()),
        near_frac_5cm_mean=float(np.mean([r['near_frac']['5cm'] for r in rows])),
        path_m_mean=float(col('path_m').mean()), speed_cm_s=float(col('path_m').mean() / (count * dt) * 100),
        feeding_s_total=float(col('feeding_s').sum()),
        hops_total=int(col('hops').sum()), hops_escape_total=int(col('hops_escape').sum()),
        hops_voluntary_total=int(col('hops_voluntary').sum()),
        hops_escape_per_1000_fly_s=float(col('hops_escape').sum() / fly_s * 1000),
        hops_voluntary_per_1000_fly_s=float(col('hops_voluntary').sum() / fly_s * 1000),
        gf_max_walk_median_hz=float(np.median(col('gf_max_walk_hz'))),
        fly_s_before_zero=before_s, fly_s_after_zero=after_s,
        contacts_before_zero=before_c, contacts_after_zero=after_c,
        meals_before_zero=before_m, meals_after_zero=after_m,
        contacts_per_1000_s_before_zero=(before_c / before_s * 1000 if before_s else None),
        contacts_per_1000_s_after_zero=(after_c / after_s * 1000 if after_s else None),
        speed_cm_s_before_zero=(before_p / before_s * 100 if before_s else None),
        speed_cm_s_after_zero=(after_p / after_s * 100 if after_s else None))


# ---------------------------------------------------------------------------------------------- geometry

def geometry(fruit_set, minutes, speed_cm_s, taste_m=.015):
    """The assay's geometric ceiling: how often an IDEAL (never-revisiting) random searcher would touch fruit.

    A fly walks in the plane z = table_top; `BatchSim.step` tastes at surface distance < 0.015 m
    (flyverse/batch_sim.py:174) using `|p - centre| - radius` with radius = max(s.radii) (batch_sim.py:116-117,
    world.make_room's `info['fruit']`). For a sphere of radius R whose centre sits h above that plane the taste
    condition is sqrt(r^2 + h^2) < R + taste, i.e. an in-plane capture disc of radius sqrt((R+taste)^2 - h^2).
    Expected crossings of a target of width 2r by a path of length L over a walkable area A is L * 2r / A."""
    from flyverse.world import make_room
    _, info = make_room(0, fruit_set)
    x0, x1, y0, y1 = info['table_extent']; z = info['table_top_z']; area = (x1 - x0) * (y1 - y0)
    rows, swath, disc = [], 0., 0.
    for name, centre, radius in info['fruit']:
        h = float(centre[2]) - z
        reach = radius + taste_m
        r = float(np.sqrt(max(reach ** 2 - h ** 2, 0.))) if reach > abs(h) else 0.
        rows.append((name, radius, h, r)); swath += 2 * r; disc += np.pi * r ** 2
    print(f"fruit set {fruit_set!r}: {len(rows)} source(s) on a {x1-x0:g} x {y1-y0:g} m fenced table "
          f"(area {area:.3f} m^2, top z {z:g}), taste threshold {taste_m*100:g} cm")
    print(f"{'fruit':10s} {'radius cm':>9s} {'centre above table cm':>22s} {'in-plane capture radius cm':>27s}")
    for name, radius, h, r in rows:
        print(f'{name:10s} {radius*100:>9.2f} {h*100:>22.2f} {r*100:>27.2f}')
    # union of the capture discs (exact area by a fine grid) and, for the encounter width, one equivalent disc per
    # connected group of overlapping discs (single-linkage on centre distance < r_i + r_j): overlapping grapes are
    # one target to a walking fly, not nine.
    centres = np.array([[float(c[0]), float(c[1])] for _, c, _ in info['fruit']])
    radii = np.array([r for _, _, _, r in rows])
    keep = radii > 0
    gx, gy = np.meshgrid(np.linspace(x0, x1, 1201), np.linspace(y0, y1, 801))
    inside = np.zeros(gx.shape, dtype=bool)
    for c, r in zip(centres[keep], radii[keep]):
        inside |= (gx - c[0]) ** 2 + (gy - c[1]) ** 2 <= r ** 2
    union = inside.mean() * area
    parent = list(range(int(keep.sum())))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    cc, rr = centres[keep], radii[keep]
    for i in range(len(rr)):
        for j in range(i + 1, len(rr)):
            if np.linalg.norm(cc[i] - cc[j]) < rr[i] + rr[j]:
                a, b = find(i), find(j)
                if a != b: parent[a] = b
    groups = {}
    for i in range(len(rr)): groups.setdefault(find(i), []).append(i)
    width = 0.
    for members in groups.values():
        sub = np.zeros(gx.shape, dtype=bool)
        for i in members: sub |= (gx - cc[i][0]) ** 2 + (gy - cc[i][1]) ** 2 <= rr[i] ** 2
        width += 2 * np.sqrt(sub.mean() * area / np.pi)        # equivalent disc of the group's union area
    print(f'summed capture discs {disc*1e4:.1f} cm^2; union {union*1e4:.1f} cm^2 = {union/area*100:.2f} % of the table; '
          f'{len(groups)} separate target group(s), summed capture width {width*100:.1f} cm '
          f'(summed per disc, ignoring overlap: {swath*100:.1f} cm)')
    L = speed_cm_s / 100 * minutes * 60
    print(f'ideal non-revisiting searcher at {speed_cm_s:g} cm/s for {minutes:g} min (path {L:.2f} m): '
          f'{L*width/area:.3f} expected encounters per fly')
    print(f'path for 1.0 expected encounter: {area/width:.1f} m = {area/width/(speed_cm_s/100)/60:.1f} min at that speed')


def odour_profile(fruit_set='apple', offsets_cm=(-60, -40, -25, -15, -10, -5, -4.5, 0, 4, 5, 10, 15, 20, 30)):
    """The shipped odour field on the plume axis through the first fruit, at the fly's walking height.

    `Air`'s Gaussian plume term exists only DOWNWIND of a source (flyverse/air.py: np.where(x > 0, ...)); upwind
    the only odour is the isotropic near field near_gain / (1 + (r/near_d0)^2). That asymmetry is what a fly which
    has overshot the source is left with, so print both halves. `puff_depth` is set to 0 for the mean field."""
    from flyverse import air
    from flyverse.world import make_room
    _, info = make_room(0, fruit_set)
    srcs = [(n, c, r / .02, r) for n, c, r in info['fruit']]
    name, centre, strength, radius = srcs[0]
    centre = np.asarray(centre, float); z = info['table_top_z'] + .0012
    def field(plume_gain):
        a = air.Air(srcs, air.WindParams(speed=.3, direction_deg=180.), seed=0)
        a.plume.puff_depth = 0.; a.plume.plume_gain = plume_gain
        return a
    full, nearonly = field(air.PlumeParams().plume_gain), field(0.)
    print(f"{name} at {np.round(centre,3).tolist()} radius {radius:g} m, strength {strength:g} (= radius / 0.02); "
          f"wind towards -x (WindParams.direction_deg 180), so x < {centre[0]:g} is downwind; "
          f"fly height z = {z:.4f}")
    print(f"{'offset cm':>10s} {'side':>9s} {'surface dist cm':>16s} {'total conc':>11s} {'near field':>11s} {'plume':>9s}")
    for dx in offsets_cm:
        p = np.array([centre[0] + dx / 100, centre[1], z])
        t = sum(full.concentration(p).values()); nf = sum(nearonly.concentration(p).values())
        print(f'{dx:>10.1f} {("downwind" if dx < 0 else "UPWIND"):>9s} '
              f'{(np.linalg.norm(p-centre)-radius)*100:>16.1f} {t:>11.4f} {nf:>11.4f} {t-nf:>9.4f}')


def prior(files, fruit=(.25, .15, .79), radius=.04, extent=(-.6, .6, -.4, .4), top_z=.75):
    """Pool finished `batch_sustain.py` JSONs (the round-4 / round-5 room batches): meals, energy, mode fractions,
    and the enrichment of the final positions in shells around the fruit against a uniform point on the table."""
    rows, names = [], []
    for pattern in files:
        for f in sorted(glob.glob(pattern)) or [pattern]:
            d = json.loads(Path(f).read_text(encoding='utf-8'))
            rows += d['rows']; names.append((Path(f).name, d['receptor']['model'] or 'off',
                                             float(np.mean([r['meals'] for r in d['rows']]))))
    meals = np.array([r['meals'] for r in rows]); energy = np.array([r['energy'] for r in rows])
    print(f'{len(names)} batches, {len(rows)} fly-rollouts: {int(meals.sum())} meals ({meals.mean():.3f} per fly), '
          f'{int((meals>0).sum())} flies fed ({(meals>0).mean()*100:.1f} %), max {int(meals.max())} per fly; '
          f'{int((energy<=ZERO).sum())} of {len(rows)} ended at energy 0 ({(energy<=ZERO).mean()*100:.1f} %)')
    print('per-batch meals/fly: ' + ', '.join(f'{n} [{m}] {v:.3f}' for n, m, v in names))
    M = {}
    for r in rows:
        for k, v in r.get('modes', {}).items(): M[k] = M.get(k, 0.) + v / len(rows)
    print('mean mode fractions: ' + ', '.join(f'{k} {v*100:.2f} %' for k, v in sorted(M.items(), key=lambda kv: -kv[1])))
    pos = np.array([r['position'] for r in rows]); centre = np.asarray(fruit, float)
    d = (np.linalg.norm(pos - centre, axis=1) - radius) * 100
    x0, x1, y0, y1 = extent
    rng = np.random.default_rng(0); N = 2_000_000
    ux, uy = rng.uniform(x0, x1, N), rng.uniform(y0, y1, N)
    ud = (np.linalg.norm(np.stack([ux, uy, np.full(N, top_z)], 1) - centre, axis=1) - radius) * 100
    print(f"\nfinal positions vs a uniform point on the {x1-x0:g} x {y1-y0:g} m table:")
    print(f"{'shell cm':>12s} {'flies':>6s} {'%':>7s} {'uniform %':>10s} {'enrichment':>11s}")
    edges = [0, 4, 8, 12, 16, 20, 25, 30, 40, 70]
    for a, b in zip(edges, edges[1:]):
        n = int(((d > a) & (d <= b)).sum()); p = n / len(d) * 100; u = float(((ud > a) & (ud <= b)).mean() * 100)
        print(f"{f'{a}-{b}':>12s} {n:>6d} {p:>7.1f} {u:>10.1f} {(p/u if u else np.nan):>11.2f}")
    edge = np.minimum(x1 - np.abs(pos[:, 0]), y1 - np.abs(pos[:, 1]))
    ue = np.minimum(x1 - np.abs(ux), y1 - np.abs(uy))
    print(f'<= 20 cm from the fruit: {(d<=20).mean()*100:.1f} % vs uniform {(ud<=20).mean()*100:.1f} % '
          f'(enrichment {(d<=20).mean()/(ud<=20).mean():.2f}); <= 4 cm: {(d<=4).mean()*100:.2f} % vs '
          f'{(ud<=4).mean()*100:.2f} % (enrichment {(d<=4).mean()/max((ud<=4).mean(),1e-12):.2f})')
    print(f'upwind of the fruit (x > {centre[0]:g}): {(pos[:,0]>centre[0]).mean()*100:.1f} %; median final x '
          f'{np.median(pos[:,0]):.3f}; within 2 cm of a fence edge {(edge<=.02).mean()*100:.1f} % vs uniform '
          f'{(ue<=.02).mean()*100:.1f} %; within 10 cm {(edge<=.10).mean()*100:.1f} % vs {(ue<=.10).mean()*100:.1f} %')
    # what each measure can resolve at the sample sizes this project runs (80 % power, two-sided 0.05: z = 2.80)
    dist = np.array([r['distance_cm'] for r in rows]); sd = dist.std(ddof=1); rate = meals.mean()
    print(f'\nresolving power (80 %, two-sided 0.05) at meals {rate:.3f} per fly and a final-distance spread of '
          f'{dist.mean():.2f} +- {sd:.2f} cm:')
    print(f"{'flies/arm':>10s} {'expected meals':>15s} {'meal-rate ratio':>16s} {'distance shift cm':>18s}")
    for n in (16, 48, 96, 240):
        lam = rate * n
        print(f'{n:>10d} {lam:>15.1f} {np.exp(2.8*np.sqrt(2/lam)):>15.2f}x {2.8*sd*np.sqrt(2/n):>18.2f}')


# ---------------------------------------------------------------------------------------------- report

def arm(d):
    o = d['options']; m = d['metabolism']
    model = d['receptor']['model'] or 'off'
    return f"{model}/d{m['drain_scale']:g}/{o['minutes']:g}min"


def arm_key(key):
    """Sort arms so that the default / off pair of one (drain-scale, horizon) cell is adjacent: 'sign/d1/10min'."""
    model, drain, minutes = key.split('/')
    return (-float(drain[1:]), float(minutes[:-3]), model != 'sign', model)


def mwu(a, b):
    from scipy import stats
    if not len(a) or not len(b): return None
    r = stats.mannwhitneyu(a, b, alternative='two-sided')
    return dict(U=float(r.statistic), p=float(r.pvalue))


def report(files):
    runs = []
    for pattern in files:
        for f in sorted(glob.glob(pattern)) or [pattern]:
            runs.append((f, json.loads(Path(f).read_text(encoding='utf-8'))))
    print(f'{len(runs)} run(s)\n')
    head = (f"{'file':44s} {'arm':22s} {'seed':>4s} {'meals':>6s} {'/fly':>6s} {'fed':>5s} {'cont':>5s} "
            f"{'reach':>5s} {'1stmeal':>8s} {'t0':>6s} {'@0%':>5s} {'minD':>6s} {'Efin':>6s} {'hops':>10s}")
    print(head); print('-' * len(head))
    for f, d in runs:
        s = d['summary']; B = d['batch']
        print(f"{Path(f).name:44s} {arm(d):22s} {d['brain_seed']:>4d} {s['meals_total']:>6d} "
              f"{s['meals_per_fly']:>6.3f} {s['flies_with_meal']:>3d}/{B:<2d} {s['contacts_total']:>5d} "
              f"{s['flies_with_contact']:>3d}/{B:<2d} {fmt(s['first_meal_s_median']):>8s} "
              f"{fmt(s['t_zero_s_median']):>6s} {s['frac_at_zero_mean']*100:>5.1f} "
              f"{s['min_distance_cm_median']:>6.1f} {s['energy_final_mean']:>6.3f} "
              f"{s['hops_total']:>3d}={s['hops_escape_total']:>2d}e+{s['hops_voluntary_total']:<2d}v")
    cells = {}
    for f, d in runs: cells.setdefault(arm(d), []).append(d)
    print('\npooled per arm (all rows of all replicates)')
    head2 = (f"{'arm':22s} {'runs':>4s} {'flies':>5s} {'meals/fly':>10s} {'fed':>7s} {'contacts/fly':>13s} "
             f"{'reached':>8s} {'minD med':>9s} {'@0 %':>6s} {'E over run':>11s} {'cont/1000s b|a 0':>18s}")
    print(head2); print('-' * len(head2))
    pooled = {}
    for key, ds in sorted(cells.items(), key=lambda kv: arm_key(kv[0])):
        rows = [r for d in ds for r in d['rows']]
        pooled[key] = rows
        meals = np.array([r['meals'] for r in rows]); cont = np.array([r['contacts'] for r in rows])
        md = np.array([r['min_distance_cm'] for r in rows])
        fz = np.array([r['frac_at_zero'] for r in rows]); me = np.array([r['mean_energy'] for r in rows])
        bs = sum(r['before_zero']['s'] for r in rows); as_ = sum(r['after_zero']['s'] for r in rows)
        bc = sum(r['before_zero']['contacts'] for r in rows); ac = sum(r['after_zero']['contacts'] for r in rows)
        per = lambda c, s: f'{c/s*1000:.2f}' if s else 'n/a'
        print(f"{key:22s} {len(ds):>4d} {len(rows):>5d} {meals.mean():>10.3f} "
              f"{int((meals>0).sum()):>3d}/{len(rows):<3d} {cont.mean():>13.3f} "
              f"{int((cont>0).sum()):>4d}/{len(rows):<3d} {np.median(md):>9.1f} {fz.mean()*100:>6.1f} "
              f"{me.mean():>11.4f} {per(bc,bs):>8s}|{per(ac,as_):<9s}")
    print('\nsearch / locomotion per arm')
    head3 = (f"{'arm':22s} {'path m':>7s} {'cm/s':>6s} {'near 5cm %':>11s} {'near 2cm %':>11s} {'feeding s':>10s} "
             f"{'1st contact med':>16s} {'top modes':>40s}")
    print(head3); print('-' * len(head3))
    for key, ds in sorted(cells.items(), key=lambda kv: arm_key(kv[0])):
        rows = pooled[key]
        pm = np.array([r['path_m'] for r in rows]); secs = ds[0]['simulated_s']
        n5 = np.mean([r['near_frac']['5cm'] for r in rows]); n2 = np.mean([r['near_frac']['2cm'] for r in rows])
        fc = [r['first_contact_s'] for r in rows if r['first_contact_s'] is not None]
        modes = {}
        for r in rows:
            for k, v in r['modes'].items(): modes[k] = modes.get(k, 0.) + v / len(rows)
        top = ' '.join(f'{k} {v*100:.0f}%' for k, v in sorted(modes.items(), key=lambda kv: -kv[1])[:4])
        print(f"{key:22s} {pm.mean():>7.2f} {pm.mean()/secs*100:>6.3f} {n5*100:>11.2f} {n2*100:>11.2f} "
              f"{np.mean([r['feeding_s'] for r in rows]):>10.2f} {(f'{np.median(fc):.0f}' if fc else 'n/a'):>16s} {top:>40s}")
    if any(d.get('checkpoints') for _, d in runs):
        print('\nhorizon: cumulative meals / contacts per fly at each checkpoint (mean over replicates of the arm)')
        for key, ds in sorted(cells.items(), key=lambda kv: arm_key(kv[0])):
            series = [d['checkpoints'] for d in ds if d.get('checkpoints')]
            if not series: continue
            n = min(len(s) for s in series)
            parts = []
            for i in range(n):
                t = series[0][i]['t_s']
                m = np.mean([s[i]['meals_per_fly'] for s in series]); c = np.mean([np.mean(s[i]['contacts']) for s in series])
                parts.append(f'{t:.0f}s {m:.2f}/{c:.2f}')
            print(f'  {key:22s} ' + '  '.join(parts))
    # instrument check: for a fly that never fed, never took off and never hit the energy floor, the drain is an
    # exact identity -- energy = energy0 - drain_per_s * T - walk_drain_per_m * path (batch_body._metabolism uses
    # speed 0 while airborne and skips launch frames, and the fence can clamp a walking fly so that its commanded
    # speed exceeds its displacement, which is the one-sided residual below).
    print('\ninstrument check: energy0 - drain_per_s * T - walk_drain_per_m * path_m vs the recorded energy')
    for f, d in runs:
        m = d['metabolism']; T = d['simulated_s']
        good = [r for r in d['rows'] if r['meals'] == 0 and r['hops'] == 0 and r['airborne_frac'] == 0 and r['energy'] > ZERO]
        if not good:
            print(f'  {Path(f).name}: no fly qualifies (all fed, hopped or hit the floor)'); continue
        res = np.array([r['energy'] - (m['energy0'] - m['drain_per_s'] * T - m['walk_drain_per_m'] * r['path_m'])
                        for r in good])
        print(f'  {Path(f).name}: {len(good)} flies, residual median {np.median(res):+.2e} '
              f'(min {res.min():+.2e}, max {res.max():+.2e})')
    print('\nper-run scatter (meals/fly, contacts/fly, min-distance median) -- quote this before calling a difference a result')
    for key, ds in sorted(cells.items(), key=lambda kv: arm_key(kv[0])):
        print(f"  {key:22s} meals/fly {[round(d['summary']['meals_per_fly'],3) for d in ds]} "
              f"contacts/fly {[round(d['summary']['contacts_per_fly'],3) for d in ds]} "
              f"minD {[round(d['summary']['min_distance_cm_median'],1) for d in ds]}")
    print('\ndefault vs off within each (drain-scale, horizon) cell, two-sided Mann-Whitney over pooled flies')
    conditions = sorted({k.split('/', 1)[1] for k in cells})
    for cond in conditions:
        a = [k for k in cells if k.startswith('sign/') and k.endswith(cond)]
        b = [k for k in cells if k.startswith('off/') and k.endswith(cond)]
        if not a or not b:
            print(f'  {cond}: only {sorted(cells)} present -- no pair'); continue
        ra, rb = pooled[a[0]], pooled[b[0]]
        for metric in ('meals', 'contacts', 'min_distance_cm', 'mean_energy', 'path_m', 'hops'):
            xa = np.array([r[metric] for r in ra], dtype=float); xb = np.array([r[metric] for r in rb], dtype=float)
            t = mwu(xa, xb)
            print(f'  {cond:10s} {metric:16s} default {xa.mean():8.3f} (n {len(xa)})  off {xb.mean():8.3f} '
                  f"(n {len(xb)})  U {t['U']:8.1f}  p {t['p']:.3g}")


# ---------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_sim_options(ap)
    ap.add_argument('--minutes', type=float, default=10., help='rollout horizon in simulated minutes')
    ap.add_argument('--energy', type=float, default=.9, help='initial tank on every fly')
    ap.add_argument('--drain-scale', type=float, default=1.,
                    help='multiply BOTH Metabolism drain terms (resting per s and per metre walked) on every '
                         "fly's own Metabolism object after construction; 1.0 = the shipped drain, no change")
    ap.add_argument('--drain-per-s', type=float, default=None,
                    help='set the resting drain explicitly before --drain-scale (default: the shipped 1/300)')
    ap.add_argument('--walk-drain-per-m', type=float, default=None,
                    help='set the walking drain explicitly before --drain-scale (default: the shipped 0.15)')
    ap.add_argument('--start', default='-0.15,0.15', help='x,y or x,y,z shared initial position')
    ap.add_argument('--trajectory-s', type=float, default=5., help='simulated seconds between trajectory samples')
    ap.add_argument('--checkpoint-s', type=float, default=60.,
                    help='simulated seconds between cumulative checkpoints (0 disables); a 10-min run can be '
                         'read at the 5-min horizon from these')
    ap.add_argument('--log-every', type=float, default=30., help='simulated seconds between progress lines')
    ap.add_argument('--json', default='out/feeding_horizon.json')
    ap.add_argument('--report', nargs='+', default=None, metavar='JSON',
                    help='aggregate/compare finished probe JSONs instead of running (no GPU needed)')
    ap.add_argument('--geometry', action='store_true',
                    help="print the assay's geometric ceiling for --fruit / --minutes and exit (CPU, no brain): "
                         'per-fruit in-plane capture radii and the encounters an ideal non-revisiting searcher '
                         'would make at --search-speed')
    ap.add_argument('--search-speed', type=float, default=1.223, metavar='CM_S',
                    help='walking speed for --geometry (default 1.223 cm/s, the measured room value: 3.67 m / 300 s)')
    ap.add_argument('--odour-profile', action='store_true',
                    help="print the shipped odour field on the plume axis through --fruit's first source and exit "
                         '(CPU, no brain): the downwind plume against the upwind near field')
    ap.add_argument('--prior', nargs='+', default=None, metavar='JSON',
                    help='pool finished batch_sustain.py JSONs (meals, energy, mode fractions and the enrichment of '
                         'the final positions in shells around the fruit) and exit (CPU)')
    args = ap.parse_args()
    if args.report:
        report(args.report); return
    if args.prior:
        prior(args.prior); return
    if args.geometry:
        geometry(args.fruit, args.minutes, args.search_speed); return
    if args.odour_profile:
        odour_profile(args.fruit); return
    if not np.isfinite([args.minutes, args.energy, args.drain_scale, args.trajectory_s, args.log_every]).all():
        ap.error('non-finite --minutes/--energy/--drain-scale/--trajectory-s/--log-every')
    if args.minutes <= 0 or args.trajectory_s <= 0 or args.log_every <= 0: ap.error('minutes/trajectory-s/log-every must be positive')
    if not 0 <= args.energy <= 1: ap.error('--energy must be in [0,1]')
    if args.drain_scale < 0: ap.error('--drain-scale must be >= 0')
    if args.checkpoint_s < 0: ap.error('--checkpoint-s must be >= 0')
    if args.seeds and len([s for s in args.seeds.split(',')]) != args.batch: ap.error('--seeds length must equal --batch')
    run(args)


if __name__ == '__main__': main()
