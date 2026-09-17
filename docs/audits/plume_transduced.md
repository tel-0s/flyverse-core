# A transduced-contrast plume law (Astra, 2026-09-17)

## 0. Answer and status

Implemented as an opt-in computation stand-in: `plume:bilateral=orn`. The bilateral cue
comes from the brain's antennal ORN population rates. No physical concentration receiver
is installed on this variant. The default `plume`, `raw`, sensory transduction, weights
and body constants are unchanged. Nothing adopted; independent skeptic pending Fable.

The CPU characterization is unfavorable for reliable instantaneous lateralization. At
the historical median total concentration and 0.57% physical contrast, four explicit
odor-mixture scenarios give **0.022-0.056 Hz** weighted L-R signal, against **0.224-0.273 Hz**
SD in a 250 ms counting window. Including the existing 100 ms rate filter and the new
250 ms contrast filter improves the approximate signal/noise ratio to **0.16-0.34**,
still below one. This is an independent sensory-input model, not observed full-brain
ORN activity. More gain amplifies noise as well as signal.

Eighteen independent room runs are prepared, **NOT submitted**. Fable must first release
the GPU pool. No GPU work, behavioral result, performance claim or default adoption is
part of this delivery. `docs/audits/determinism_gate.md` was not present at this branch's
base; no bit-reproducibility exception is assumed for the eventual rooms.

## 1. Population choice and side bias

Use `senses.Smell`'s assignments, including its reference-connectome laterality lookup,
not soma side. MaleCNS has 2,639 olfactory-class cells: **708 left, 1,396 right, 535
unsided**. All 54 transducer labels have both sides (53 named labels plus an empty label
containing two cells per side). Unsided cells receive averaged physical input and are
excluded from the directional cue. No active glomerulus is selected from the room's
physical smell dictionary.

Two unweighted antenna means are unsuitable: unequal glomerulus composition creates a
1.6-3.8 Hz false L-R difference under exactly equal odor input in the four median-total
scenarios, larger than the intended signal. For each glomerulus g with both sides, set:

```
h_g = n_Lg * n_Rg / (n_Lg + n_Rg)
q_g = h_g / sum_g(h_g)
r_L = sum_g(q_g * mean(brain.rate[ORN_Lg]))
r_R = sum_g(q_g * mean(brain.rate[ORN_Rg]))
```

Both antennas receive the same glomerulus mass. This removes the count-composition bias
in the independent-input model; full-circuit asymmetries can remain. The fixed weights
minimize difference variance if independent cells have equal rates. Actual glomeruli
have different rates, so this is an **unverified pooling rule**, not an optimal decoder
for this scene. Effective cell counts (`1/sum(cell_weight^2)`) are about 656 left and
1,127 right. Counts, per-label weights and exact target IDs are recorded in `describe()`.

ORNs provide a large directly stimulated population with the transducer's explicit side
assignment. Switching to PNs would introduce a downstream circuit transfer and correlations;
no evidence here establishes improved signal/noise there. This experiment therefore does
not choose PNs or fit a decoder to scene responses. Static CPU evaluations of the actual
room generator (six environment seeds, fixed default pose, times 0/2/4/6 s) produce expected
weighted sensory forcing rates of **3.9-5.5 Hz per side**, L-R **-0.016 to +0.033 Hz** and
cascade SNR **0.016-0.27**. These are sensory-field evaluations, not neural room runs.

The old room traces log antenna concentration sums but **not** glomerular mixtures or ORN/PN
rates. They cannot establish historical `brain.rate` values. The planned rooms explicitly
record ORN rates; full-brain firing rates, correlations and possible side bias remain open.

## 2. Law and boundary

At each 10 ms module frame, the ordinary extension scheduler supplies `brain.rate` at
the frame boundary. Compute matched rL/rR above, then:

```
d = (rL-rR)/(rL+rR)                   # zero if both rates are zero
b = exp(-dt/0.25)*b + (1-exp(-dt/0.25))*d
goal = heading + atan(200*atanh(clamp(b, -0.999, 0.999)))
```

`2*atanh(d) = log(rL/rR)`, so this is a response to the log *rate* ratio. The old
concentration law had small-contrast slope `2*0.1/0.001 = 200`; this experiment explicitly
retains that dimensionless slope on the new rate contrast. **The gain 200 is unverified.**
It is not an inversion of the ORN transducer and no distance is inferred from neural rates.
For one homogeneous concentration c, small contrasts obey
`d_rate ~= [c*f'(c)/f(c)]*d_concentration`, where `f(c)=1+150*c/(c+0.5)`;
saturation attenuates contrast. A heterogeneous population does not admit a single c
inverse. Neither the concentration gain nor this retained rate gain is physiological.

The walking goal is local whenever weighted rates are nonzero. Airborne or zero-rate rows
use the existing wind/entry-memory fallback. Basal spikes can therefore produce a noisy
walking goal even without physical odor; the existing LH/confidence/feeding/hunger gate
still controls output. No hidden physical availability flag suppresses this noise.

Existing EPG heading, LH gate, PFL3 comparator, DNp09 output and DNa02 feedback are retained.
Only the bilateral cue changes in the transduced arm. Wind and internal-state receivers
remain explicit; "rate-only" refers to this bilateral cue. Output remains Poisson drive
on biological PFL3 and DNp09, with no weight, receptor, body or sensory-law edits.

The opt-in constructor is `PlumeNavigation(c, bilateral="orn")`; the named syntax is
`plume:bilateral=orn`. `describe()` records `replaces="computation"`, `law="unverified"`,
the input, pooling, gain, target IDs, audit and inherited removal condition. Per-row
weighted rates reuse the checkpointed `odor_L/R` buffers (Hz in this mode); `bilateral`
holds the filtered rate contrast. Configuration identity prevents loading a physical-cue
checkpoint into the rate-cue configuration. The new mode has no `observe_smell` callback.
The default path keeps its prior arithmetic, state fields and description.

Runtime work is two fixed weighted reductions over 2,104 ORNs, followed by existing
tensor arithmetic. No per-frame CPU readback, extra Poisson sampler or dense neural
matrix is added. CUDA replay compatibility follows the existing module structure but
is **not tested in this CPU-only delivery**; no timing or capture-equality claim.

## 3. CPU signal/noise, before any rooms

Generator: `scripts/plume_transduced_characterise.py`. All conditions, counts, hashes,
assumptions and Monte Carlo seeds are in
[characterisation.json](data/plume_transduced/characterisation.json); the generated
[table](data/plume_transduced/characterisation.md) lists the median-total cases.

The 3,600 historical compass samples yield total antenna concentrations (mean of L/R)
at p05/p50/p95 of about **0.64 / 1.80 / 5.15**. We combine those totals with apple, lime,
banana and an explicitly equal-source mixture of the six fruit odor profiles. These are
bracketing scenarios, **not recovered historical glomerular inputs**. Each uses
`cL_g=c_g*(1+d)` and `cR_g=c_g*(1-d)` for d = 0, 0.0026, 0.0057, 0.007, 0.011, matching
the review's typical and p95 contrast ranges. No sensory response or goal gain is fitted.

For independent sensory spikes with per-cell rates lambda and normalized weights w:

```
V = sum_L(w_i^2 * lambda_i) + sum_R(w_i^2 * lambda_i)
SD_250ms_box(L-R) = sqrt(V/0.25)
SD_100ms_plus_250ms_EMAs(L-R) ~= sqrt(V/[2*(0.1+0.25)])
```

The second formula is the squared-impulse integral for two first-order filters. Contrast
noise uses the delta-method derivatives `2*rR/(rL+rR)^2` and `-2*rL/(rL+rR)^2`;
filtering the nonlinear ratio is only approximately a linear cascade. The shipped
forcing is Bernoulli per 0.5 ms step; its variance correction is `lambda*(1-lambda*dt)`.
Both the Poisson estimate and that correction are recorded. Forced events can bypass
the LIF refractory state (`brain.py`); native synaptic spikes and correlations are not
included in this analytical input model.

The exact isolated-forcing check draws grouped Bernoulli counts at 0.5 ms, updates the
100 ms rate EMA, reads the contrast every 10 ms and applies its 250 ms EMA. Eight CPU
seeds, 30 s each, discarding the first 2 s; glomeruli and antennae are independent. At
the median-total equal-source mixture and 0.57% contrast, the filtered contrast's temporal
SD is **0.0104 +/- 0.0010**, and correct turn sign occupies **0.667 +/- 0.069** of samples.
Here +/- is across-draw sample SD (eight independent streams); samples within a stream
are autocorrelated. No binomial significance calculation treats them as replicates.

The inherited gain yields mean absolute goal offsets **50.7 +/- 3.1 deg** in this signal
condition, versus **48.3 +/- 3.1 deg** in the zero-contrast control. The latter is noise
alone; no steering or brain circuit ran. This is the most direct warning against
interpreting a high-gain rate cue as accurate lateralization. Gains and filters were
left fixed after this result. An eventual room improvement could involve integration,
exploration or other loop dynamics; it would not retroactively validate the sensory law.

## 4. Prepared comparison, not submitted

Generator: `scripts/plume_transduced_batch.py plan --out out/plume_transduced_rooms_v1`.
The frozen JSON and guarded `batch.sh` are copied beside the CPU artifacts. The user
confirmed the goal-only definition on 2026-09-17:

| Arm | Bilateral goal | DNa02 feedback | Instruments |
|---|---|---|---|
| full | existing physical contrast | on | compass plume hunger flight |
| transduced | matched ORN rate contrast | on | compass plume:bilateral=orn hunger flight |
| goal_only | existing physical contrast | off | compass plume:feedback=off hunger flight |

Six **independent processes/runs per arm**, B=1; matched environment seeds 0-5 and neural
seeds 31-36 across arms. Seed-drawn starts cover the tabletop with a 3 cm edge margin,
rejecting positions already within 1.5 cm of fruit. Start RNG seeds 20260917-20260922;
heading uniform [-180,180). All six exact positions/headings are frozen before results.
This fixture construction uses geometry; no controller receives it. Arm order rotates
across seeds. This changes the prior B=6/shared-start diagnostic to independent draws
with broader starting poses; it does not promise old contact times will reproduce.

Otherwise use the steering configuration: full MaleCNS, compass/hunger/flight, 60 s,
initial energy 0.1, all fruit, no fence/program/escape gating, base GF threshold 33 Hz,
0.5 ms LIF, 1 ms optic, native CUDA/event-driven/captured frames, torch sparse backend.
No local GPU or cluster room was run. The shell has `set -euo pipefail`, house target,
family blocking, preserved per-job exit codes and a named fetch directory. The shell
and room entry point refuse execution without the explicit pool-release environment
flag; it must not be set until Fable authorizes submission. Source hashes must match
the declaration or the room runner refuses.

Report six feeding durations and contact indicators per arm (>=1 s feeding), censored
first-contact times, energy, distance, airborne duration, neural rates, goal offsets and
target/measured DNa02. Each metric is a run-level summary; report mean plus across-run SD,
not trace-sample pseudo-replication. Keep all failures and exact machine records, but quote
no unsupported decimal precision. No gain search, pass/fail adoption gate, physiological
claim or significance claim is predeclared. Full/transduced tests cue replacement;
full/goal-only isolates the feedback bridge. Analysis rejects mismatched provenance,
non-CUDA or incomplete runs. Native-path determinism remains unestablished here.

## 5. Author self-review and pending work

Eight new CPU tests cover default versus explicit legacy equality, unequal populations,
signed rate responses, actual frame-start `brain.rate` reads, physical-input isolation,
checkpoint replay/configuration/partial reset, input validation, feedback-off behavior,
draft generation/guards and run-level analysis. A separate before/after replay against
the pre-edit branch snapshot matches **all 40 state arrays byte for byte** after 120
frames. Full CPU suite, including the original MaleCNS golden: **516 passed / 19 skipped /
220 subtests**. The final metadata/assay checks also pass all eight targeted tests.
MaleCNS cache MD5s remain `c50c598a708b5b373cbaffca7d6a9d82` (neurons),
`ac131529cebf98decde58d0c227b7954` (W) and `bf01d724acf2a1fec8fdb60ef8a9e066` (sign-0 counts).

This is the author's self-review. Independent skeptic pending Fable. Pending experimental
work: GPU lifecycle/capture checks and the authorized room comparison after pool release;
measure actual neural rates and correlations, then assess whether any approach improvement
survives the low sensory SNR. No GPU result is implied by a prepared harness.
