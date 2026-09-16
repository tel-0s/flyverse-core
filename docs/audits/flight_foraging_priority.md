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

Pending the frozen submission. No results have been used to choose these parameters.

## 4. Review status

Self-review and full CPU/golden validation pending. Independent skeptic pending (Fable,
when accounts reset). Nothing adopted into raw or made a preset default.
