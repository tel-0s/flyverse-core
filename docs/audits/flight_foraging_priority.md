# Flight foraging priority (engineering correction)

## 1. Problem and scope

The original optional `flight` instrument requested 100 Hz wing-power activity whenever energy
exceeded 0.05, except while feeding or sated. `hunger` scaled navigation but never interrupted
flight. In the original six 60 s room observations every fly spent 59.67 s airborne and none
fed (see `navigation_instruments.md` section 4). The owner independently observed flight
continuing during starvation. This is an instrument policy defect, not evidence for a missing
biological transmitter, receptor or energy cost.

The correction stays inside `FlightDrive`, through its existing neural extension boundary.
No parent synapse, body constant, raw path, preset default or metabolism law changes.
The new policy is **unverified engineering**, declared before the assays below:

- Search on the ground for 20 s before requesting voluntary flight. Feeding, satiety, low
  reserves and strong odor interrupt this interval. Actual touchdown starts a new interval.
- Power each bout for at most 8 s, including a failed takeoff attempt. A spontaneous native
  takeoff can be assisted for one bounded bout if no interruption applies.
- At energy <= 0.35 withdraw artificial wing drive. Re-arm only at energy >= 0.50.
  These are policy thresholds, not measurements of Drosophila hunger or endurance.
- Read the same biological LH berry/apple populations and normalization as `plume`, through
  this module's own frame-boundary inputs. Low-pass for 1 s, interrupt at normalized odor >=
  0.60 and clear at <= 0.25. Missing LH groups are explicitly absent in provenance; reduced
  graphs retain reserve and duration limits without a fabricated odor signal.
- Once interruption begins in the air, hold the artificial drive at zero until actual
  touchdown, even if the initiating cue disappears. Feeding also zeros drive immediately.

This withdraws the extra power and steering Poisson input, allowing the shipped physics to
bring the fly down. It is not a precision landing controller or a choice of landing site.
Native activity, including GF escape hops, remains possible. Odor is a neural cue, not food
distance; this policy neither locates fruit nor establishes successful plume navigation.
The existing 100 Hz servo target and gains are unchanged while a bout is active. The 100 Hz
equilibrium is from the body equations, not physiology. Namiki et al. 2022 (the original
flight source) does not supply the new bout, reserve or landing rules. No new literature
claim or physiological fit is made. Retire this policy when validated neural flight-state,
feeding-priority and descending/VNC mechanisms supply the same behavior.

All new per-row state stays on the module device, advances without host scalar reads and is
included in checkpoint, full reset and row reset. No mutable `plume` state is read, so
attachment order cannot alter the decisions. Direct users must continue supplying
`interoception`; the room and BatchSim already do so.

## 2. Protocol frozen before submission

One sequential house submission, generated with
`python scripts/navigation_batch.py --flight-priority --out out/flight_priority_v1`.
The generator hashes the source and this declaration before submission. GPU work is house
only; CPU regression tests are local. No parameter sweep, food-finding fit or preset admission.

1. Existing `navigation_probe.lifecycle`: 13 exact scheduler/checkpoint/reset/detach checks
   for each of `compass` and `compass_ring`, with plume/hunger/flight attached.
2. `flight_priority_probe.transitions`: 31 s, biological selected graph, B=3, seed 31,
   torch sparse, no optic or receptor model. Captured and checked eager paths compared
   exactly at 17 declared frames, all brain and instrument state. Body observations and
   LH frame-boundary rates are prescribed (0 or 40 Hz), not behavioral results. Rows are
   a healthy ground wait and timed bout, an odor-interrupted airborne bout, and depleted
   airborne reserves. Gates: capture actually used; no launch before 20 s; healthy bout
   begins and expires by 28.03 s; odor stops artificial power by 2 s and remains off until
   touchdown at 10 s; depleted row never requests power.
3. Two 60 s full MaleCNS B=6 rooms, neural seed 0, environment seeds 0-5, apple fruit,
   fenced table, no program, `compass plume hunger flight`, native CUDA kernels/events,
   automatic optic and captured sensory/brain frames. First arm starts on the table at
   (-0.15, 0.15, 0.75), energy 0.6. Second starts airborne at z=1 m, zero velocity,
   energy 0.1; otherwise identical settings. Gates: no artificial power at energy <=
   0.35, each commanded bout <= 8.02 s (float32 accumulation/frame tolerance), and all
   initially depleted airborne rows touch down within 5 s. Record airtime, powered time,
   feeding, first landing, native/voluntary hops, energy, trajectory and policy states.
   Food finding is descriptive, with no pass threshold. These are not independent
   physiological replicates, an adoption suite or a statistical causal comparison with
   the prior room runs. Endogenous escape hops do not fail the controller gates.

No performance speedup is predicted from timings on this shared machine. The controller
adds only per-row tensor arithmetic and small LH reductions; capture validity is tested.
No end-to-end UI performance claim follows from headless rooms.

## 3. Results

All frozen functional gates pass on B200. Implementation and frozen protocol:
`266d2741671fc79013cea30ee861f04f5c345791`. One house job, four sequential assays, no
retry or parameter change. The exact remote source archive matches all 79 frozen LF-normalized
hashes and that commit; each of the seven provenance records agrees on 52 shared runtime
source files. Whole MaleCNS compiled CSR MD5: `ef23cc27bea13be7f6a96f3c04fd3737`.

- All 26 existing exact multi-module lifecycle checks pass (13 per heading provider).
- All 17 additional captured/eager transition checkpoints agree exactly, with six policy
  gates passing. The prescribed healthy row waits, powers a bout and times out; the odor
  row interrupts and stays off until touchdown; the depleted row never powers flight.
- All six initially depleted airborne room rows land in 0.47-0.52 s. No artificial power is
  delivered at low reserves. Two later native escape hops occur, one in each of seeds 1
  and 5; there are no voluntary launches. Two rows feed for 15 s each; four exhaust their
  energy. This is not a successful starvation-recovery policy for every fly.
- In the normal-energy room arm, all six rows request zero artificial wing power; the neural
  odor interruption prevents the ground interval completing. One native escape hop lasts
  0.38 s. Seed 0 feeds for 8.06 s; the other five do not feed. This differs from the original
  sustained-flight observation, but it is not a statistical food-finding improvement claim.
  The bounded-bout room gate is vacuous here because there are no powered bouts; the
  non-vacuous timeout test is the controlled transition assay and CPU regression.

All per-environment rows, script-emitted without selection, are in
[`data/flight_priority/room_table.md`](data/flight_priority/room_table.md). Exact values, gates,
raw artifact hashes and source hashes are in [`summary.json`](data/flight_priority/summary.json);
the original declaration is [`predeclared.json`](data/flight_priority/predeclared.json).
The original navigation results remain unchanged in their audit.

Reproduce the derived files with:

```sh
python scripts/flight_priority_analyse.py --source out/flight_priority_v1 --out docs/audits/data/flight_priority
```

The ignored raw directory retains all four JSONs, the original declaration, console/launcher
logs and exact remote `source.tar.gz` (extracted under `source/`). Public derived files exclude
host/path-bearing provenance. No gain, timing, threshold or protocol was changed after results.

## 4. Review status

Author self-review: all state transitions use held internal observations and frame-boundary
neural rates; there are no world-position/fruit-distance inputs, body commands or synapse writes.
The odor population read avoids module-order dependence; the CPU regression crosses the odor
threshold with reversed attachment order. Capture keeps persistent state buffers, and the house
fixture exercises both decision boundaries and the existing checkpoint/row-reset/detach paths.
Failed takeoffs time out and touchdown resets the ground interval. The old flight checkpoint
schema is deliberately rejected rather than silently omitting the new state.

CPU: **500 passed / 19 skipped / 220 subtests**. Targeted navigation plus original golden:
24 passed / 10 subtests. Ruff and diff whitespace checks pass. Both working copies retain
cache MD5s `c50c598a708b5b373cbaffca7d6a9d82`, `ac131529cebf98decde58d0c227b7954`, and
`bf01d724acf2a1fec8fdb60ef8a9e066`. No full-brain eager/captured equality, runtime improvement,
precision landing or reliable food-finding claim follows from these checks.

Independent skeptic pending (Fable, when accounts reset). Nothing adopted into raw or made a
preset default. The energy/bout/odor policy remains an explicit unverified stand-in.
