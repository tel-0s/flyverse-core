# What the connectome cannot tell us: the unknowns behind each behaviour, and the experiments that would settle them

flyverse-core, release document, 2026-09-19. Owner-written; every number below is taken from a named audit in
`docs/audits/` or from `docs/NOTES.md` as a skeptic-verified figure. The round audits carry an independent skeptic pass
recorded verbatim in `docs/audits/receptor_verification.md`; two source documents used here do not -- the anatomy
survey `flywire_banc_survey.md` (not a predeclared round; its counts are recomputed from the release tables) and
`exr6_evidence.md` (an evidence memo) -- and the backend audits are reviewed in their own `*_review.md` files; literature claims are sourced at page / figure level or marked
*unverified* (`docs/INTERP.md` 10.4 rule 29). Nothing here is a claim about real flies; each "prediction" is a
statement of what the wired model does, offered as something an experiment can contradict.

## 0. The shape of the result

Fourteen sessions of predeclared rounds, each closed by an independent skeptic (the anatomy survey and the ExR6
evidence memo excepted, as above), converged on one picture.
The MaleCNS v1.0 graph, driven through a rate optic lobe and one leaky integrate-and-fire cell model with
transmitter-signed synapses (`docs/OVERVIEW.md`), produces unprompted what is feedforward: loom escape through the
giant fibre, direction selectivity in T4/T5, wind-direction descending neurons, taste to the proboscis motor
neurons, sustained walking. It does not produce anything that needs an internal state: a heading bump, a signed
steering command that leads the yaw, a flight state, plume tracking. The 29-check ledger's `raw` column is 27 pass /
0 fail / 2 known gap in two of three draws and 26 / 1 / 2 in the third (`docs/audits/instrumented_suite.md`
section 2), and the two known gaps are the compass row (`compass.wedge_cells_persisting`, no attractor) and the small-object row
(`object.LC10a_flip_hz`, no small-object signal in the rate optic lobe).

The blockers are not simulator defects; each traces to a small set of physiological facts the connectome does not
carry. They are listed below with the behaviour they gate, what the graph does fix about them, the quantitative
statement the model makes, and the measurement that would decide it. The last section is the table.

## 1. Turning: a signed steering command that leads the yaw

**What the graph fixes.** DNa02, the descending steering neuron, sits under a large sign-correct inhibitory budget
in all three public connectomes: excitation : inhibition on DNa02 by presynaptic transmitter is ACh 31.7 k vs
GABA + Glu 16.1 k in MaleCNS (2.0 : 1), 18.2 k vs 9.3 k in FAFB (1.96 : 1), 7.7 k vs 4.6 k in BANC (1.7 : 1); its
top inhibitory rows (PS049, PS059, VES051, LAL126, AOTU019 in the survey's selection; LAL083 and VES052 are of the same size) and its lateralised cholinergic excitation (AN04B003) and lateralised glutamatergic inhibition (LT51) are the
same rows in all three, and the contralateral IN12B014 pair is symmetric in MaleCNS and BANC
(`docs/audits/flywire_banc_survey.md` section 3; the survey's row table has no FAFB column). The net-inhibited resting state flyverse finds is therefore a
rate statement about the circuit, not an artefact of one reconstruction.

**What the model does.** In `raw` the fly walks and does not turn: left and right DNa02 fire in step; no wired
source produces a DNa02 L-R that leads the yaw (`docs/audits/round3_integration.md` section 9;
`docs/audits/deficit_turning.md`). A leg-cycle proprioceptive input, matched on all three leg channels, raises
AN04B003 by +2.67 / +3.88 Hz (left / right) and DNa02-left by a `result`-level +0.13 Hz over the matched control, and
nothing sided -- DNa02-right, the clean yaw SD, straightness and DNa02 L-R are all null, and the amplitude / turn
law adds nothing detectable beyond the per-phase modulation -- a null on all seven primaries, not an equivalence bound
(`docs/audits/level_fixed_point.md`; rounds 4-4d). The one route
that would carry a signed self-turn report upward breaks at AN04B003 -> PS196_b: PS196_b's L-R moves the same way
in both turn directions under a labelled haltere stop-gap -- a 3 v 3 descriptive figure the audit labels
`underpowered`, not a `result` (`docs/audits/vnc_drive.md` section 6; NOTES compass round 2). Given a signed input, the descending stage itself works: PFL3-left at 80 Hz drives DNa02-right at 22.6 Hz
through the wired PFL3 -> DNa02 -> leg path (`flyverse/cx.py`), and under the plume instrument a commanded DNa02 L-R
of +-8.9 Hz is met at +9.4 / -8.9 Hz (`docs/audits/plume_steering.md` section 6).

**The unknowns.** (a) The sign of the ascending turn report at PS196_b -- two cells, cholinergic, 1,801 synapses
onto GLNO (19-21 % of GLNO's input), fed by AN07B037_a/_b (419 / 52 synapses) and by CB0675 / GNG580 / PS047_b, and
never recorded during a turn (`docs/NOTES.md` "literature note", 2026-09-15; Wang 2026 finding 3 -- an LLM-driven mining of the same graph whose author reports
its first pass stated wrong facts, and whose counts we reproduced ourselves -- names the same cell from the compass
side). (b) DNa02's own baseline and intrinsic gain, which one LIF for every cell cannot set. (c)
Whether the PS049 / PS059 inhibition is modulated by state (section 5).

**Prediction and experiment.** The model predicts that PS196_b carries a *signed* L-R during self-generated turns
if and only if its ascending afferents are sided; with sided afferents the model routes a left turn to PS196_b-right
-> GLNO-left -> PEN-right, every hop contralateral (AN07B037_a L->R 202 / R->L 214; PS196_b -> GLNO L->R 832 / R->L
966; GLNO -> PEN_a L->R 5,024 / R->L 4,782, ipsilateral 0; `docs/audits/instruments_review.md` section 3, `docs/INSTRUMENTS.md`). A
recording of PS196_b (or AN07B037) in a walking fly turning both ways decides it: a signed L-R that leads yaw makes
the route real and puts the compass's velocity input on the ascending side; the same sign in both directions, as
the model's haltere stop-gap gives, means the velocity input into the compass is not proprioceptive, and the model
has no other candidate (there is no visual route to PEN: HS / VS / H2 / DNp20 / DNp15 / JO make no synapse onto PEN,
and the eye's contribution to GLNO is 8 synapses; NOTES compass round 2).

## 2. The compass: a bump that holds and rotates

**What the graph fixes.** PEN has one non-ring input of size, GLNO (4 cells, sign 0, 19.4 % of PEN's raw input,
fully contralateral), and GLNO's own inputs are PEN 37 %, PS196_b 19 %, EPG 8 % -- efference-copy territory (NOTES
compass round 2; the PS196_b -> GLNO count 1,801 reproduced to the synapse against Wang 2026). The ring's write
loop returns to its own tile: EPG writes +2.16 / -2.18 wedges through PEN per side (`docs/audits/compass_ring_mechanism.md`
section 3). The DC term on the relays is two ExR6 and four ER6 (plus ER4m on EPG): EPG -> ExR6 is +11.4 mV per pair
against EPG -> PEN +5.05 (`compass_ring_mechanism.md`, the unitary-weights block of section 1). The rate model's
figure that ExR6 at 110-183 Hz delivers -9 to -15 mV to every PEN is a fixed point, not a measurement: the same fixed
point puts 2 ExR6 + 4 ER6 + 11 ER4m at a summed 484 Hz at background and 1,075 Hz during the pulse, while the LIF's
entire 308-cell ER / ExR population sums to 244 Hz after release -- the DC term is overstated by roughly 2x, and the
conclusions below rest on the measured PEN and ExR6 / ER6 rates (`compass_ring_mechanism.md` skeptic pass,
correction 5). ExR6 is glutamatergic and ER6 GABAergic by EASI-FISH (Wolff et al. 2025, eLife 104764, Fig. 9 source
data: SS53617 vGlut strong, SS58833 Gad1 weak; `docs/audits/exr6_evidence.md`); the receptors at the EB / GA contacts
are open, and a 2026 preprint (Eddy et al., *Divergent excitatory and inhibitory signaling in a head direction circuit*,
doi 10.64898/2026.01.18.700161, Fig. 2) reports that local PB glutamate puffs suppress E-PG calcium and increase PEN_b
calcium, with PEN_a increasing less consistently (not isolated to ExR6; `exr6_evidence.md` section 5.1, *not verified
here beyond the audit's quotation*).

**What the model does.** At the shipped gains the ring is silent (EPG / PEN at 0 Hz at rest: 42 of 42 PEN never spike in the CPU settle window, and the per-cell maximum is
0.13-0.58 Hz in the GPU post-pulse window; `docs/audits/compass_local_recurrence.md` sections 3.4 and 6) and no data-implied type-level change
gives a bump (`compass_ring_mechanism.md`, round 5A). The binding constraint is the DC brake: ExR6 fires 41-43 Hz and
ER6 23-24 Hz at rest in the model (`compass_local_recurrence.md` section 3.3), holding PEN down; lifting their edges
raises PEN from 0.3 to 45 Hz but yields a saturated ring, not a bump (`docs/audits/compass_dc_balance.md`). With the
hold plus GLNO relabelled glutamate a ~160 Hz hump forms (159-164 Hz, width 3.85-4.00 wedges, at the driven tile in
1 of 5 seeds; the H3G-vs-H3 contrast itself scores `undetermined`, `compass_dc_balance.md`); with a per-type EPG -> EPG
recurrence the hump is 7 wedges wide against a 2.5-5 target, sits at centres 3.08-3.23 against the driven 1.5 (a
+1.58-1.73 wedge offset, outside the 1.5-wedge bound, and at 11.39 in the fifth seed) and is confined in no seed;
with the global recurrence the ring flattens (`compass_local_recurrence.md` section 4, 6B). Given a signed afferent the hump
**does not rotate**: the confined-frame centre slope is +0.14 / -0.09 / -0.07 wedges/s against an ideal 4.0
(`compass_velocity_route.md` section 5, round 7), although the afferent reaches GLNO with a side (+2.22 Hz L-R,
`result`; the sign-flipped arm gives -2.07 +- 0.65, `docs/audits/compass_sign_control.md`), and a direct 90 Hz GLNO
challenge puts a correctly signed ~3 Hz into PEN while suppressing it from 24.0 to 3.6-5.0 Hz (`compass_velocity_route.md`
section 6). The LIF's effective input noise, measured for the first time, is 4.6 mV on the sub-threshold relays (the
11.8 mV upper edge is a quasi-static assumption), which puts the ring's critical recurrent gain (33.3 Hz/mV) 5.5x
above the maximum slope of the smoothed f-I (6.06 Hz/mV) -- the shipped ring cannot be an attractor at any
operating point the cell model provides (`compass_local_recurrence.md` section 2).

**The unknowns.** (a) GLNO's transmitter: three EM classifiers agree only that GLNO is inhibitory and disagree on
the transmitter (BANC Glu 0.505 vs FlyWire GABA at confidence 0.30-0.33, with MaleCNS `unclear` at 0.48), and both
inhibitory labels map to -1 in `NT_SIGN` (`docs/audits/glno_relabel.md` 1.2). (b) The receptor class and kinetics at
ExR6 / ER6 -> PEN, EPG. (c) EPG / PEN intrinsic gain and resting drive -- the ring's operating point. (d) The sign at
PS196_b (section 1).

**Predictions and experiments.**
1. *The DC-brake prediction.* If ExR6 fires tonically at tens of Hz in a resting fly, as it does in the model, and
   its synapse onto PEN is a plain inhibitory glutamate receptor, then P-EN should be nearly silent at rest. Real
   P-EN cells carry a bump at rest (*the standard reading of the head-direction literature -- literature pointer,
   unverified here*). So either ExR6's resting rate in vivo is far below 41 Hz, or the receptor at the
   ExR6 -> PEN contact is not the sign the table assumes (the preprint's excitatory P-EN glutamate response would be
   that case). One imaging session of ExR6 at rest, or one receptor-expression call at that contact, decides which.
2. *The GLNO prediction.* GLNO glutamate is safe on the suite and fixes nothing alone (`glno_relabel.md`, 5B); with
   the DC hold it is what turns a saturated ring into a hump (`compass_dc_balance.md`, H3 vs H3G). So the model
   predicts GLNO's sign matters only once the brake is off -- a transcriptome or EASI-FISH call on GLNO is worth
   making after, not before, the ExR6 receptor question.
3. *The gain prediction.* No attractor exists within a factor of five of the shipped LIF gain. A recorded EPG f-I
   slope, or evidence of a per-type intrinsic conductance that steepens it, is the measurement; without it the
   compass is not a wiring question.

## 3. Flight: a state the graph does not generate

**What the model does.** `raw` takes off: the giant fibre fires at loom (GF peak 44.1 Hz on MaleCNS against the 33 Hz threshold row, and
53.3 Hz on the synthetic BANC candidate lattice -- a single-run within-graph observation with two input asymmetries
favouring the candidate, not evidence of faster female visual processing; `docs/audits/banc_candidate_experiment.md`)
and the body's escape hop works. It does not fly: nothing in the graph supplies a sustained wing-motor drive from sensory input, and the
haltere loop closes only through a labelled stop-gap (`docs/audits/vnc_drive.md`). Under the labelled `flight`
instrument the wired wing-MN power path holds a 100 Hz servo target and the fly stays airborne for 59.7 s of a 60 s
room without feeding; under the corrected flight-priority policy artificial flight is unreachable in any room that
contains fruit, because the lateral-horn odour gate latches (`docs/audits/navigation_instruments.md`,
`docs/audits/flight_foraging_priority.md` and their skeptic sections). No episode shows powered flight and feeding
together.

**The unknowns.** The octopaminergic flight state -- OA release at take-off and its receptors on the wing motor
neurons and the haltere afferents -- is entirely within the 141 octopamine cells the model silences (section 5).

**Prediction and experiment.** The model predicts that the wing power motor pool is sufficient to hold the body airborne
once something drives it at ~100 Hz (a body-equation equilibrium of the shipped body, not a physiological rate; the
instrument writes Poisson directly onto the power MNs and no sensory path is involved, `navigation_instruments.md`
section 2), and that nothing in the wired graph supplies that drive: a fly whose
octopaminergic cells are silenced should take off (giant fibre) and fail to sustain flight. That is a known
phenotype in outline (*OA is required for flight maintenance -- literature pointer, unverified here*); the model's
contribution is which cells and which motor pool, and it is falsified if a non-octopaminergic tonic drive is found.

## 4. Fruit finding: a bilateral cue the model's own sensors cannot resolve

**What the model does.** Odour reaches the lateral horn and, through PFL3 -> DNa02, the legs; the anemotaxis
program and the plume instrument both steer through that wired stage (`flyverse/cx.py`,
`docs/audits/plume_steering.md`). Under the `instrumented` preset the shipped `plume` instrument -- which reads the
*physical* odour-concentration difference between antennae 1 mm apart, amplified by a 200x geometric gain -- fed in
6 of 6 rooms from six seed-drawn starts, including two starts facing ~157 deg away from a fruit 4-6 cm behind them.
Substituting the model's own ORN population rates for the physical contrast, at the same gain, fed in 1 of 6, and
its cue was uncorrelated with the true lateral contrast: per-run Pearson r -0.07 to +0.24, correct sign in 0.52 +-
0.10 of samples against 0.87-0.98 for the physical cue (`docs/audits/plume_transduced.md` section 7 and its skeptic
pass). The CPU characterisation says why: at the rooms' 0.26-0.57 % bilateral contrast the shipped ORN law
`1 + 150c/(c+0.5)` and the 100 ms / 250 ms filters deliver 0.022-0.056 Hz of L-R against 0.22-0.27 Hz of counting
noise (cascade SNR 0.16-0.34, and conservative -- shared synaptic drive lowers it), correct sign in ~0.64 of samples (0.667 +- 0.069 in one
eight-stream draw, 0.646 +- 0.067 in an independent draw, analytic 0.635; `plume_transduced.md` section 3). Removing the DNa02 feedback but keeping the physical goal fed in 3 of 6,
and those three are the starts already pointing at fruit (`plume_transduced.md` skeptic claim 4; six runs per arm,
one start each, B = 1: descriptive, not a significance result, not an SNR measurement, not a claim about flies). In
round 8's separate goal-only batch, on its own drawn starts at B = 6, the goal-only arm fed 4.33 +- 0.82 rows of six
and the predeclared full - goal-only test was `null` (`plume_goal_only.md`) -- a different unit and a different
batch, not the same count.

**The unknowns.** (a) ORN sensitivity and adaptation at low concentration -- the model's law is a fixed saturating
curve. (b) Temporal structure: the model's plume is a smooth field; real plumes are intermittent, and the model has
no onset detector. (c) Whether the antennal lobe expands bilateral contrast (the characterisation reads ORNs; PN
pooling is untested). (d) The ORN side assignment itself is a reconstruction proxy: 708 left / 1,396 right, with
per-glomerulus ratios from 0.08 to 13 (`plume_transduced.md` section 1).

**Prediction and experiment.** The model predicts that a 0.57 % concentration contrast at the rooms' median total concentration produces a
weighted bilateral ORN rate difference of 0.02-0.06 Hz (four odour mixtures), and that no downstream circuit can
steer on it at these filter times. A measurement of
bilateral ORN or PN rates in a fly at a 0.5 % bilateral contrast (Gaudry et al. 2013, Nature 493:424, verified here for the turn sign only; *that its stimuli are large and
unilateral is unverified here*) decides it: if real ORNs
resolve it, the transduction gain or adaptation is the data-driven change to `senses.Olfaction`; if they do not,
plume tracking in flies is temporal, and the model needs an onset channel, not more gain.

## 5. Neuromodulation: 3,312 cells at sign zero

**What the graph fixes.** MaleCNS labels 3,312 bodies with a transmitter `NT_SIGN` maps to 0 -- 2,361 unknown, 415
serotonin, 395 dopamine, 141 octopamine, in 530 types -- and flyverse silences their outputs. Read by type into the
female releases, dopamine is solid (FAFB DA 372 of 395; BANC verified 365), serotonin is the least corroborated
(BANC verified: serotonin 56, tyramine 46, glycine 8), and about 400 of the `unknown` cells carry a classical
prediction in BANC (ACh 150 / GABA 148 / Glu 111) (`flywire_banc_survey.md` section 4). Three disagreements are on
record: PFL3 (ACh in MaleCNS and FAFB, tyramine predicted in BANC), Delta7 (glutamate; BANC verified
glutamate + serotonin co-transmission), LAL074 (`docs/NT_INTEGRATION.md`).

**What the model does.** One slow scalar term cannot stand in for the monoamine class: round 3 tried it and every
per-transmitter unitary variant broke take-off or moved suite rows outside its gap (`docs/audits/monoamine_slow_term.md`,
`unitary_strength.md`, `guard_suites_r3.md`). Nothing was adopted.

**Prediction and experiment.** This is the one remaining default change with a data path: a slow term *per receptor
class* built from the receptor atlases already in the repository (`docs/audits/receptor_sources_*.md`) and BANC's
verified transmitters, rather than one scalar. The model predicts that the ~400 BANC-classical `unknown` cells,
switched on with their predicted sign, change specific suite rows and not others; that is a suite run, and the
rule for adopting it is written (`docs/PRESETS_SPEC.md` section 2 item 5; the round-2 rule).

## 6. Intrinsic properties and baseline activity

**What the model does.** Every cell is the same LIF (Shiu et al. 2024 parameters -- `w_syn` 0.275 mV -- plus the project's connection cap 60 and fan-in
normalisation, `flyverse/brain.py`); the only noise is sensory Poisson; there is no spontaneous activity. Measured on the
model's own relays the effective input noise is 4.6 mV (`compass_local_recurrence.md` section 2). Descending
neurons in a walking fly and ascending neurons have recorded baselines (*Aymanns 2022; Chen 2018 -- literature
pointers, to be entered as ledger rows, unverified here*); the model's DNa02 at rest is 0.015 Hz on the left and 0.080 Hz on the
right in the no-sense arm (`level_fixed_point.md` section 9, arm A).

**Prediction and experiment.** Per-type baseline rates as expectation-ledger rows, and a Poisson background per type
only where the data support it. The model predicts which behaviours a baseline unlocks: the ring (section 2) and
the steering choke (section 1) are both operating-point problems.

## 7. Small objects: LC11 and LC10a

**What the model does.** The small-object pathway is lost by opposite-signed carrier convergence at T3's inputs (Tm1 | Tm4, Mi1 | Tm3),
the residual is scrambled by spiking feedback and pooled away at LC11 / LC10a; on a fresh-seed B200 replication the
"suppress" carrier variant reproduces and "rectify" is partial, and 40 of 40 rectangle assays are null with no LC
localisation
(`docs/audits/deficit_object.md`, `object_rectangles_r3.md`, `object_localizer_r3.md`, `object_samedevice_r3.md`).
LC11 has 143 cells in MaleCNS, 127 in FAFB, 141 in BANC; LC10a 275 / 237 / 224 (`flywire_banc_survey.md` section 2).

**The unknown.** The signs of the Tm1 / Tm4 and Mi1 / Tm3 inputs onto T3 (and of T2 / T3 onto LC11) -- whether the
convergence is opposite-signed in the fly or only in the table.

**Prediction and experiment.** The model predicts that LC11 does not respond to a small dark object at the sizes tested on
the transmitter-signed graph (the audits do not license "no stimulus could localise LC11",
`object_localizer_r3.md`); a receptor call on T3's Tm / Mi inputs, or an LC11 recording at object sizes of a
few degrees, decides whether the sign table or the pooling is wrong.

## 8. Across connectomes: what is robust

The DNa02 balance, the PS196a -> PS059 contralateral loop, the IN12B014 pair and the absence of GLNO -> PS196b and
DNa02 -> AN04B003 edges hold in MaleCNS, FAFB v783 and BANC v888 with the same top rows (section 1;
`flywire_banc_survey.md` section 3), at synapse yields that scale roughly 1 : 0.6 : 0.3 and must not be compared
across releases without that scale. The female CNS, run through the same model with no female-tuned default, walks
straighter (yaw SD 0.28 against 2.64; `docs/audits/connectome_backends.md`, `docs/NOTES.md` Session 11) -- a
finding with the reconstruction and synapse-yield caveats attached, and a prediction: the difference should
localise to the DNa02 input budget, which the toolkit can test the same way it localised the male deficit.

## 9. The table

| unknown | gates | what the graph fixes | the model's quantitative statement | the measurement | audits |
|---|---|---|---|---|---|
| PS196_b's turn sign | turning, compass velocity input | 1,801 syn onto GLNO; contralateral chain to PEN | L-R same sign both ways under the haltere stop-gap (3 v 3, `underpowered`); a sided afferent reaches GLNO at +2.2 / -2.1 Hz | PS196_b / AN07B037 during self-turns | vnc_drive, compass_velocity_route, compass_sign_control |
| ExR6 / ER6 receptor at PEN, EPG | compass | transmitters settled (vGlut / Gad1); EPG -> ExR6 +11.4 mV/pair | ExR6 41-43 Hz at rest holds PEN at 0.3-0.7 Hz; brake off -> saturation, not a bump | ExR6 resting rate in vivo; receptor at the EB / GA contact | compass_dc_balance, compass_local_recurrence, exr6_evidence |
| GLNO transmitter | compass | sign 0; 19.4 % of PEN input | glutamate safe on the suite; matters only once the brake is off | EASI-FISH / transcriptome on GLNO | glno_relabel, compass_dc_balance |
| ring gain and operating point | compass | critical gain 33.3 Hz/mV | max f-I slope 6.06 Hz/mV at sigma 4.6: no attractor within 5.5x | EPG f-I; intrinsic conductances | compass_local_recurrence |
| OA flight state | flight | 141 OA cells silenced | wing-MN path sufficient at 100 Hz drive; nothing supplies it | OA-silenced take-off vs maintenance | flywire_banc_survey (the 141 cells), navigation_instruments, flight_foraging_priority |
| ORN gain / adaptation at low contrast | fruit finding | ORN law 1 + 150c/(c+0.5); 708 / 1,396 sided ORNs | 0.3-0.6 % contrast -> 0.02-0.06 Hz L-R vs 0.22-0.27 Hz noise; rooms 1/6 on the ORN cue | bilateral ORN / PN rates at 0.5 % contrast | plume_transduced, plume_goal_only |
| monoamine receptors per class | state, flight, turning | 3,312 sign-0 cells; BANC verified labels | one scalar fails; per-class term untested | receptor atlas + BANC labels -> suite run | monoamine_slow_term, flywire_banc_survey |
| per-type baselines | all state-dependent | none | DNa02-left 0.015 / -right 0.080 Hz at rest; no ring cell spikes in the settle window | Aymanns 2022 / Chen 2018 rows | level_fixed_point, compass_local_recurrence |
| T2 / T3 -> LC11 sign | small objects | 143 LC11 cells | null at all sizes on the signed graph | LC11 input receptor call | deficit_object, object_rectangles_r3 |

If one measurement had to be chosen, it is the first row: PS196_b during a self-generated turn. It is two cells,
it is named from both the body side and the compass side, and its answer moves both the turning and the compass
questions at once.
