# Navigation instruments: recurrent memory, goals, metabolic gain, flight

2026-09-15, Astra. Experimental opt-ins; self-review only. No change to the raw default,
parent synapses, receptors, neurotransmitter labels, or body constants. Independent review
is pending Fable. This is implementation and engineering validation, not preset admission.

## 1. Scope and sources

The owner requested a more direct implementation of Wang's ideas and their primary sources,
plus plume tracking, flight, and an optional metabolic gain. Five names compose under
`--preset instrumented --instruments ...`: the existing `compass`, its alternative
`compass_ring`, `plume`, `hunger`, and `flight`. The two heading providers are incompatible.
Plume requires one heading provider. Hunger requires plume or flight. Duplicate names and
overlapping neural writes are errors. Missing target populations are errors, not silent no-ops.

| source | implementation and limit |
|---|---|
| [Wang, pinned 80b94e6](https://github.com/pwang724/fly-circuit-exploration/tree/80b94e6608cf927ca2c9f2bbce0577c5a998684c), `simulations/eb_epg_pen_loop.py`, finding 4 | Independently expressed 16 EPG / 16 PEN rate dynamics. No external code is imported or executed. Noise is zero instead of Wang's 0.02; gains and units below are explicit. |
| [Turner-Evans et al. 2017, eLife 23496](https://elifesciences.org/articles/23496), model section | Shifted PEN recurrence supplies angular memory. Our reduced model is Wang's version, not the paper's full 54 EPG / 18 PEN model or a recovered native circuit. |
| [Mussells Pires et al. 2024](https://doi.org/10.1038/s41586-023-07006-3), Methods, "Full PFL3 model", PDF pp. 17-18 | The published 12-cell-per-LAL heading/goal comparator and preferred-angle arrays are implemented directly. Its aggregate output is translated to biological PFL3 sides by an engineering bridge, not a cell-by-cell anatomical correspondence. |
| [Matheson et al. 2022](https://doi.org/10.1038/s41467-022-32247-7), Fig. 7, and [2024 addendum](https://doi.org/10.1038/s41467-024-46225-8) | Odor-gated allocentric wind goals motivate the input combination. The VT062617 physiology/behavior cannot be assigned confidently to hDeltaC alone: the addendum identifies hDeltaK labeling. This implementation assigns no native hDelta identity. |
| [Siliciano et al. 2026](https://doi.org/10.1038/s41586-026-10827-7), published July 22, goal-memory/model sections | Entry-bearing memory motivates walking returns after odor loss. This is a reduced hybrid policy, not the full stochastic leaving/returning model; it stores direction, not displacement or source coordinates. |
| [van Breugel and Dickinson 2014](https://doi.org/10.1016/j.cub.2013.12.023), odor-loss casting results | Horizontal casting after loss, with a 0.45 s delay. The measured delay was 450 +/- 165 ms; our clock begins after an engineering odor filter/hysteresis, so total latency is not that measurement. No vertical casting or visual wind estimation is implemented. |
| [Root et al. 2011](https://doi.org/10.1016/j.cell.2011.02.008) | Insulin/sNPF modulation of specific ORNs supports state-dependent food search. Our instantaneous global navigation gain is explicitly an engineering alternative, not Or42b/sNPFR1 kinetics or a claimed hunger-neuron identity. |
| [Namiki et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9206711/), DNg02 population flight-power results | Supports investigating descending wing-power control during flight; does not establish takeoff sufficiency. Our flight controller bypasses that unsolved circuit and drives biological wing MNs. |

The published Siliciano PDF in the ignored research checkout has SHA-256
`bc4cd16a77681a7e90d96464c456f04999c84aa69009b725ab345a8369f3558b`.
The newer connectome-fitting leads remain research leads: [Duan, Dong and Fiete 2025](https://doi.org/10.1101/2025.05.26.655406)
and [Hulse et al. 2026](https://www.janelia.org/publication/hidden-symmetries-in-network-connectivity-support-ring-attractor-dynamics-in-the-flys).
The latter uses a hidden-symmetry construction; it is not simply a transmitter sign correction.
No fitted parameters from either are claimed here. Wang's EB-brake diagnostic does not show
that biological EB contacts must be removed: contact counts are not effective synaptic gains.

## 2. Laws and boundaries

**compass_ring.** Memory is recurrent rates `e[16]`, `p[16]`, not an integrated angle variable.
With 2 ms substeps and tau = 50 ms, PEN is updated first using rectified EPG input and sided
gains `1-v`, `1+v`; EPG then uses local excitation plus shifted PEN return minus `4.8*mean(e)`.
Local excitation is `0.6*exp(-distance_in_wedges^2/2)`. PB input = 1, PEN return = 0.8 at
each of two target wedges, EPG cap = 5. EB input is `eb_ratio/2` at each target wedge.
The default EB/PB ratio 0 is a **textbook counterfactual**, never a parent edge deletion.
The diagnostic ratio 2.7 is a hemibrain count ratio, not measured efficacy in MaleCNS.
`v = clip(yaw_rad_s * (0.3/(pi/2)), -0.8, 0.8)` is **uncalibrated**; 90 deg/s input does
not imply 90 deg/s bump velocity. Output to existing EPGs is `0.0001 + 10*e[wedge]` Poisson Hz.
No visual anchoring, tilt compensation, or inference of the native operating state is supplied.

**plume.** Reads biological EPG rates, occupancy-balanced across 16 wedges, and the existing
berry/apple LH channels. Uses antennal deflections through `FlyBrain.wind`: `dL+dR` and `dL-dR`
give incoming-wind forward/left components in the shipped sensor geometry. It receives neither
world heading nor wind angle, source coordinates, distance, or motor commands. The goal lives
in an ideal FC2-like memory, not biological FC2 cells. In-odor goal = heading + body-relative
upwind angle. Walking reentry goal = exponentially averaged entry-heading vector (learning 0.5).
Flight outside odor alternates +/-90 degrees around upwind every 1.5 s. This policy can miss
the plume; it has no source-localization or landing policy.

Odor uses the preexisting model LH calibration: berry `(Hz-10)/12`, apple `(Hz-5)/8`, maximum
clipped to [0,1], tau 1 s, enter >0.6 / leave <0.25. Return memory decays over 30 s, weight 0.35;
loss clock saturates at one hour. These are **unverified engineering choices**, not new fits.
Unconfined heading (strength <0.6) suppresses navigation; feeding suppresses it too.

The comparator is `r = 29.23*softplus(2.17*(cos(H-Hpref)+0.63*cos(G-Gpref)-0.7))` Hz,
with both complete preferred-angle arrays from the cited Methods. Those arrays are irregular,
so the population readout has small heading-dependent variation. R-L follows positive goal
error in this declared coordinate convention, checked across a heading grid. The difference
divided by 40 Hz, multiplied by odor/memory and optional hunger gain, is clipped to [-1,1].
Positive values drive PFL3 soma-R, negative drive soma-L, up to 80 Hz; DNp09 gets up to 60 Hz.
These injection gains and the soma-side bridge are engineering parameters. MaleCNS's C1-C9
annotations are not silently treated as the paper's twelve ideal populations. Biological
PFL3 activity still goes through the parent circuit and ordinary motor readout.

**hunger.** `gain = (1-energy)*(not sated)`. Default before internal input is 1. No neural
read/write targets, receptor row or hormone kinetics are fabricated. Consumers bind to this
explicit input transducer; attachment order does not alter its value. Without this instrument,
plume has gain 1 regardless of energy. `FlyBrain.interoception(energy, sated=, airborne=, feeding=)`
accepts normalized energy and Boolean flags, scalar or `(B,)`; room and BatchSim supply these
from current body/metabolism state before stepping. Raw has no receiver and rejects the call.

**flight.** An opt-in request for flight whenever energy >0.05, not sated and not feeding,
after an explicit internal-state input. It reads biological wing-power/PFL3 rates and writes
Poisson input to biological power/steering MNs. Target mean power = 100 Hz because the shipped
body has `az = (1.5/40)*(power-20)-3-drag*vz`. This is a **body-equation equilibrium**, not a
physiological recording. PI input = clip(100 + 0.25*error + integral, 0,250) Hz, integral gain
0.5/s with bounds [-100,150]. Steering inputs = 10 +/- clip(0.1*(PFL3_R-PFL3_L),-10,10) Hz.
Off request zeros those inputs and integral. The controller has no altitude feedback, collision
avoidance, landing policy or added flight metabolic cost. It can sustain, climb, oscillate,
or fail through the current body/circuit; the assay reports which. Native DNg02 selection would
need all released suffix variants, not the empty literal type `DNg02`.

All modules checkpoint and reset per batch row. Runtime neural reads are gathered before any
module writes. Parent CSR and raw behavior remain unchanged. Positive-Poisson proofs permit
multiple explicitly graph-safe modules to run within a captured CUDA frame; arbitrary hooks
and modules retain the checked eager scheduler. Composition order must not affect results.

## 3. Predeclared engineering checks

Freeze source SHA-256s and the batch before house submission. No gains are fitted to these
results. Failures may motivate separately labelled later work; they do not authorize adoption.

1. CPU functional tests: heading/goal sign, recurrent motion versus EB brake, odor/wind/memory,
   hunger/feeding controls, invalid combinations, both construction orders, checkpoint and row
   reset, and actual BatchSim internal-state delivery. Full CPU suite and MaleCNS golden gate.
2. House CUDA lifecycle fixture: same biological subgraph and seed 31, batch 3, captured versus
   checked/eager modules, for each heading provider with plume+hunger+flight. Exact brain tensors,
   module state and held inputs through changed yaw/internal inputs, pulses, drive changes,
   5/10 ms frames, partial/full reset, checkpoint, queued reset and detach. Failure blocks a
   performance claim and must be fixed without changing module laws.
3. Full MaleCNS neural forcing control: six rows, 30 s, last 10 s summarized. LH Poisson 40 Hz;
   wind left/right, sated, feeding, airborne left/right. No optic input or body commands.
   Expect signed plume output where heading is confined; no instrument drive when sated/feeding.
   Assess whether wing power approaches the body's 100 Hz equilibrium; report saturation/failure.
4. Three 60 s room observations, six environment seeds 0-5 each: compass alone; compass+plume+hunger;
   compass+plume+hunger+flight. Identical apple table start (-0.15,0.15,0.75), fence on, no body program.
   Record trajectories, wing power, path length, closest fruit distance, feeding and airborne time.
   These are exploratory engineering observations, **not** the 300 s adoption rate-half or proof of
   improved food finding. Stochastic trajectories need not match frame for frame across arms.
5. CUDA performance: B=1,8,32, four alternating repeats of 100 frames after 50 warmup frames;
   compare compass, compass+plume+hunger, all four, and compass_ring+plume+hunger. Matched external
   250 Hz forcing on the union of output targets dominates instrument inputs and must make brain
   tensors identical across arms. Report CUDA event and synchronized wall time, graph capture,
   and overhead over compass; shared-machine measurements are not idle-device guarantees or UI FPS.
6. One headless UI launch with plural names and brain map. No claim of measured interactive FPS.

Neither the three-draw 29-check admission suite nor the full room rate-half is replaced by these
checks. The existing compass already fails strict admission by moving a taste row. These new
names remain experimental even if their functional assays pass. Raw remains the default.

## 4. Results

Pending the frozen house batch. CPU development established recurrent motion at roughly
+132/-133 deg/s for +/-90 deg/s input with EB=0, and near-zero rotation with EB=2.7.
Those are diagnostic observations at the declared gains, not a velocity calibration.
