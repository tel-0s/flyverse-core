# Proprioception transducer: the body state the VNC motor neurons produce -> afferent rates (opt-in)

Task build:transducer. Code: `flyverse/senses.py` (`Proprioception`), `flyverse/fly.py` (`FlyBrain.proprioception`, the
`proprioception` entry of `available_senses` -- inert unless a sense is attached), `flyverse/batch_body.py`
(`BatchBody.proprio_state`), `flyverse/batch_sim.py` (`BatchSim(..., proprioception=None)` and the per-frame call),
`scripts/batch_sustain.py` / `scripts/probe_walk_straightness.py` (`--proprioception SPEC`), the smoke generator
`scripts/probe_proprioception.py`, `tests/test_proprioception.py` (9 CPU tests), six `lit.*` rows appended to
`flyverse/data/expected_responses.csv` (stimulus `proprioception_transducer`, every one `op report`). **Nothing in the
shipped path changes: the default is OFF, and section 6(a) shows the shipped output is bit-identical with the sense absent,
attached-but-unfed, and through `BatchSim` with `proprioception=None`.** Nothing under `docs/NOTES.md`, `body.py`,
`brain.py`, `optic.py`, `room_demo.py` or `room_ui.py` was touched.

## 0. What this is, and what it is not

`deficit_turning.md` 6.3 and `deficit_rotation.md` 3(3) localised the same missing afferent: every `vnc_sensory` and
`sensory_ascending` proprioceptor sits at 0 Hz in the room, so the walking VNC runs open-loop (IN12B014 / IN19A003 at
12-14 Hz inhibiting DNa02 with no proprioceptive input), the ascending cells that carry a turn report (AN04B003 from
SNpp39 / SNppxx, AN07B035 from SNpp45, AN07B037_a / AN06A026 from the haltere SApp) never fire, and PS196_b -- the one
efference-copy input of GLNO -- stays at 0.03-0.16 Hz. `interp_atlas.md` 3 showed the wiring works when driven
(`superclass.vnc_sensory` is the rank-1 mover of the haltere and power readouts, `class.mechanosensory_proprioceptive`
rank 1 for `steer_LR`). The project rule names the legitimate fix: a **sensory transducer** -- a body-state -> afferent-rate
model like `Wind` -> JO or sugar -> GRN -- for an afferent population that exists in MaleCNS and is wired but never driven.
This is that transducer. It is a mechanism the connectome implies, not a tuning: no number in it was chosen by looking at
whether the fly turns, every rate law is stated with its literature range and citation (the ranges are `op report` ledger
rows, never scored), and the one term that would close a loop through a hand-written module (the haltere Coriolis term,
which needs `body.Locomotion`'s yaw scalar) is a labelled STOP-GAP option for a control arm and never the default.

What it is not: a leg model. `body.FlyState` carries `speed`, `yaw_rate`, `airborne` (and the flight vector); there is no
stance / swing phase, no joint angle, no step frequency, no load distribution over legs (scout notes, `body.py:28-59`,
`Locomotion.readout` `:236-255`). The only leg state the VNC produces is the motor-neuron rate per side
(`MotorRates.leg_L` / `leg_R`, 192 / 189 cells of subclass fl / ml / hl). So the leg channels read that rate, and the
document says so at every point where a real FeCO would read a joint.

## 1. Channels, cells, laws, ranges

Cell sets are selected by annotation only (`class == mechanosensory_proprioceptive` and `subclass`, plus `entryNerve` for
the leg / wing / haltere split of the campaniform sensilla). Counts on the shipped cache (`Proprioception(c, 'all').counts()`;
`tests/test_proprioception.py::CacheCountsTests` pins them):

| channel | selection | n | L / R / both | vnc_sensory / sensory_ascending | side source | named types (n; L / R / both) |
|---|---|---|---|---|---|---|
| `chordotonal` | class proprio, subclass `chordotonal organ` or `leg` | **615** | 271 / 259 / 85 | 593 / 22 | instance 22, laterality 593 | SNpp39 39 (6 / 14 / 19), SNpp50 62 (33 / 29 / 0), SNpp60 41 (22 / 18 / 1), SNpp52 17 (6 / 11 / 0), SNppxx 78 (34 / 37 / 7), SApp23 22 (11 / 11 / 0); entry nerves MetaLN 267, MesoLN 246, ProLN 59, ProCN 33, ProAN 8, VProN 2 |
| `hair_plate` | class proprio, subclass `hair plate` | **113** | 57 / 54 / 2 | 112 / 1 | instance 1, laterality 112 | SNpp45 52 (27 / 24; MesoLN / MetaLN / ProLN / VProN = leg), SNpp19 35 (18 / 17; **PrN = prosternal, i.e. neck hair plates**), others 26 (DProN / VProN) |
| `campaniform` | class proprio, subclass `campaniform sensilla`, entryNerve in {ProLN, MesoLN, MetaLN, ProAN, VProN, DProN, ProCN} | **12** | 6 / 6 / 0 | 12 / 0 | instance 12 | SNpp53 12 (4 per leg nerve). The other 414 campaniform cells enter by ADMN (218, wing) or DMetaN (195, haltere) and are **not** leg load: excluded |
| `haltere` | class proprio, subclass `haltere` | **201** | 99 / 95 / 7 | 53 / 148 | instance 148, laterality 53 | **SApp 148 (75 / 73 / 0)** in sensory_ascending (entryNerve DMetaN); SNpp34 / 25 / 23 / 35 / 14 / 15 / 20 / 21 / 12 in vnc_sensory |

**Side rule.** Every proprioceptor in these sets has `somaSide` NaN (the only exceptions are 2 SNpp42 cells outside the
sets), so side is (i) the `_L` / `_R` suffix of the MaleCNS `instance` where it exists -- all 183 `sensory_ascending`
cells and the 12 SNpp53 -- else (ii) the sign of the cell's output laterality on the reference graph against left- vs
right-soma targets (`senses.laterality`, the same function `Smell` / `Wind` use, threshold 0.2 as `Smell`). On the 183
instance-labelled cells (22 + 1 + 12 + 148 = 183) the two rules agree on the sign 177 / 177 where the laterality is
decisive (|lat| > 0.2) and 0 disagree; the 12 SNpp53 have |laterality| <= 0.2 on 6 cells, where the instance decides (`scripts/probe_proprioception.py structure`,
`out/proprio/structure.{txt,json}`). Cells with |laterality| <= 0.2 (85 chordotonal, 2 hair plate, 7 haltere) read the
mean of the two sides, as the unsided JO cells do in `Wind`. Annotation notes from the same file: SNpp52 is split by the
annotation between subclass `chordotonal organ` (17 cells, in the chordotonal channel) and `hair plate` (26, DProN /
MesoLN / MetaLN, in the hair-plate channel); the 67 untyped chordotonal cells are 25 L / 17 R / 25 unsided.

Consistency check on the targets -- raw synapses onto each watch cell from afferents by the side the rule assigns
(`structure.txt`; "outside" = proprioceptive cells in none of the four channels, i.e. wing / haltere campaniform,
abdomen, notum, wing):

| cell (soma) | from L-assigned | from R-assigned | unsided | outside the channels |
|---|---|---|---|---|
| AN04B003 L (3410 / 4161 / 7051) | 146 / 93 / 19 | 0 / 0 / 0 | 54 / 2 / 58 | 0 |
| AN04B003 R (1347 / 4060 / 6317) | 0 / 0 / 0 | 198 / 50 / 162 | 41 / 0 / 6 | 0 |
| AN07B035 L (19306 / 27824) | 44 / 78 | 0 / 0 | 0 / 0 | 0 |
| AN07B035 R (20265 / 24507) | 0 / 0 | 15 / 77 | 0 / 17 | 0 |
| AN07B037_a L (7698 / 134536) | 69 / 41 | 0 / 1 | 0 | 13 / 15 |
| AN07B037_a R (8292 / 8783) | 0 / 0 | 39 / 43 | 0 | 13 / 5 |
| AN06A026 L (79595 / 92741) | 155 / 185 | 3 / 8 | 0 | 54 / 100 |
| AN06A026 R (79926 / 80571) | 5 / 3 | 152 / 131 | 0 | 126 / 46 |
| IN19A003 L (145098 / 145224 / 161436) | 178 / 62 / 45 | 2 / 0 / 1 | 0 / 0 / 7 | 0 / 3 / 0 |
| IN19A003 R (145208 / 145267 / 145516) | 0 / 0 / 0 | 61 / 255 / 37 | 1 / 0 / 0 | 0 |
| DNa02 L (132380) / R (332) | 33 / 0 | 0 / 15 | 0 | 0 |
| IN12B014 L (145936, 146993) / R (146461, 147083) | 0, 0 / 0, 1 | 0, 3 / 1, 0 | 0 | **284, 0 / 225, 0** |
| PS196_b L / R | 0 | 0 | 0 | 0 (no proprioceptive input; it is reached through AN07B037_a / PS239) |

So the afferents project ipsilaterally: "left afferents from the left MN rates" reaches the left cell of every ascending
type, of IN19A003 and of DNa02 (SNpp45 -> DNa02, `deficit_turning.md` 6.1). Two facts to carry forward: **IN12B014, the
strongest VNC inhibitor of DNa02 (-89 / -124 mV/s in the room), receives its 509 proprioceptive synapses from cells
outside all four channels** (wing / haltere campaniform and body-wall afferents), so this transducer does not close that
loop and should not be expected to move IN12B014; and AN06A026 / AN07B037_a take 20-45 % of their proprioceptive input
from the DMetaN campaniform cells that the haltere channel does not include (section 7).

**Laws** (all Hz; `p` = `Proprioception.params`; `mn_ref_hz` = 30, the leg-MN rate `body.Locomotion` quotes its own
`k_leg_turn` per (100 deg/s per 30 Hz), i.e. the body model's existing full-scale MN rate, the same role "+1 = the
existing model's full-speed deflection" plays in `Wind`; every rate is `clip(., 0, max_hz)`):

| channel | law on the ground | airborne | defaults | literature range (ledger row) |
|---|---|---|---|---|
| chordotonal | `tonic + (max - tonic) * clip(legMN_side / mn_ref, 0, 1)` | `tonic` | tonic 10, max 150 | `lit.FeCO.afferent_rate_range_hz` 10-150 Hz |
| hair_plate | the same law with its own numbers | `tonic` | tonic 5, max 100 | `lit.hair_plate.afferent_rate_range_hz` 5-100 Hz |
| campaniform | `load_hz` (ground contact x body-weight support) | `0` | load 50, max 100 | `lit.campaniform_leg.afferent_rate_range_hz` 0-100 Hz |
| haltere | `k_h * haltereMN` (wingbeat term only) | same | k_h 1, max 250 | `lit.haltere.afferent_rate_range_hz` 0-250 Hz |
| haltere + STOP-GAP | `k_h * haltereMN * (1 + g * |yaw_rate|)` | same | g 1 per rad/s, **only** under `haltere_coriolis` | `lit.haltere.coriolis_modulation` (unmeasured) |

Injection: `FlyBrain.proprioception(leg_L, leg_R, haltere, airborne, yaw_rate)` -> `senses.Proprioception.rates` ->
`FlyBrain._input('proprioception_<channel>', idx, hz)`, i.e. `poisson_p = hz * dt / 1000` per LIF step, exactly the
path `wind` / `smell` / `taste` use. In `BatchSim.step` the call sits beside the other senses, before `fb.step`, and reads
`BatchBody.proprio_state(self.motor)`: the **previous frame's** motor readout (zeros before the first frame -> tonic /
load only) and each row's `airborne` flag and realised `yaw_rate`. The haltere MN group (`WingGroups.haltere`, 16 cells,
subclass hm) is not side-split in `MotorRates`, so both haltere sides read the one bilateral mean; a per-side split would be
a `motor.py` change (another task's file) and is listed in section 7.

## 2. Literature, and what is and is not measured

The ranges are brackets, not Drosophila spike rates, and the ledger rows say so in their `notes`:

* **FeCO (chordotonal).** Mamiya, Gurung & Tuthill 2018, Neuron 100:636: the femoral chordotonal organ's claw neurons
  encode tibia position tonically, hook neurons directional movement phasically, club neurons vibration -- by calcium
  imaging, so no spike rate. Tuthill & Wilson 2016, Curr Biol 26:R1022 (review). The tonic tens-of-Hz / phasic >100 Hz
  bracket is the locust's (Matheson 1992, J Exp Biol 162:229 -- range fractionation; tonic activity greatest at the joint
  extremes; Field & Matheson 1998, Adv Insect Physiol 27:1). **Uncertain**, other-insect extrapolation. And the
  transducer's independent variable is the MN rate, not the joint (section 0): the law is *"the FeCO fires more when the
  leg motor output is larger"*, the only monotone relation the body model can support, stated as such.
* **Hair plates.** Wong & Pearson 1976, J Exp Biol 64:233 (cockroach trochanteral hair plate: type I phasic, type II
  phasic-tonic with imposed joint displacement); Pearson, Wong & Fourtner 1976, J Exp Biol 64:251 (their afferents excite
  the extensor motoneurone and follow high afferent frequencies); Tuthill & Wilson 2016 (hair plates report joint angle).
  No Drosophila hair-plate rate. **Uncertain** bracket. The task brief's "Pratt et al. 2017" could not be resolved to a
  paper and is not cited. 35 of the 113 cells (SNpp19, entryNerve PrN) are prosternal (neck) hair plates; they are driven
  with the same leg-MN law because the brief assigns SNpp19 to this channel -- a documented mismatch (the model has no
  head), listed in section 7.
* **Leg campaniform sensilla (load).** Ridgel, Frazier, DiCaprio & Zill 2000, J Comp Physiol A 186:359 (cockroach tibial
  campaniform sensilla: phasico-tonic, level and rate of force, proximal sensilla tonic to sustained load); Zill et al.
  2013, Arthropod Struct Dev 42:455 (stick insect); Dinges et al. 2021, J Comp Neurol 529:905 (Drosophila anatomy, no
  physiology). Load on the ground = body-weight support = `load_hz` 50; airborne = 0. **Uncertain** bracket. The brief's
  "Agrawal et al. 2020" (eLife 9:e60299) is the central-processing paper of the FeCO (9Aa / 10Ba / 13Ba interneurons) and
  gives no hair-plate or campaniform rate; it is not used for a number.
* **Haltere afferents.** Fox & Daniel 2008, J Comp Physiol A 194:887 (Holorusia haltere campaniform afferents fire one
  precisely phase-locked spike per stroke and follow driven oscillation to 150 Hz); Dickerson, de Souza, Huda &
  Dickinson 2019, Curr Biol 29:3517 (Drosophila: stroke-synchronous haltere feedback to the wing steering system; the
  haltere beats antiphase to the wing at wingbeat frequency); Yarger & Fox 2016, Integr Comp Biol 56:865 (review);
  Lehmann & Dickinson 1997, J Exp Biol 200:1133 (Drosophila wingbeat ~200 Hz). So the afferent rate *is* the stroke rate:
  `k_h = 1` is the phase-locking identity (one spike per stroke), the ceiling 250 Hz the wingbeat, and the model's only
  wingbeat variable is the haltere motor-neuron rate. The Coriolis modulation (Pringle 1948, Phil Trans R Soc B 233:347;
  Fox & Daniel 2008: lateral Coriolis-like forcing shifts spike phase and recruits sensilla) has no gain in Hz per rad/s in
  the literature; `g = 1 per rad/s` is a placeholder for the control arm and nothing else.
* **Ascending neurons.** Chen et al. 2023, Nat Neurosci 26:682 (ascending neurons convey behavioural state; a large
  fraction are walking-active); Tsubouchi et al. 2017, Science 358:615. The brief's "Chen et al. 2021, Cell Reports" could
  not be resolved to a paper; the Nat Neurosci paper is the one cited (`lit.AN.walking_state_encoding`, gap 1).

## 3. Classification under the project rule

| channel | independent variable | classification | why |
|---|---|---|---|
| chordotonal | leg-MN rate per side, ground contact | **mechanism the connectome implies, with a documented body-model limitation** | the afferent population exists, is wired (SNpp39 -> AN04B003 -> DNa02 +221 mV^2, `deficit_turning.md` 6.1) and never driven; the transducer maps a body-produced variable to Hz within a literature range; the limitation is that the variable is the MN rate, not a joint angle, because the body has no leg |
| hair_plate | the same | the same; **plus** a documented mismatch for the 35 PrN (neck) cells | |
| campaniform | ground contact (airborne flag) | **mechanism the connectome implies** | load is exactly what the body model knows (on a surface vs airborne); 12 cells only |
| haltere (wingbeat term) | haltere MN rate | **mechanism the connectome implies** | one afferent spike per stroke is the measured physiology; the MN rate is the body's stroke rate |
| haltere Coriolis term | `body.Locomotion`'s realised yaw_rate | **STOP-GAP (labelled control arm), never default** | the yaw scalar is computed by a hand-written readout from DNa02 / leg MNs; feeding it back is a loop through a hand-written module, exactly the case `deficit_rotation.md` 3(3) names as not connectome-implied |
| `mn_ref_hz` 30, the tonic / ceiling numbers | -- | transducer scale constants with a literature bracket, exposed as parameters | none was set by looking at behaviour; the ceilings are literature; `mn_ref` is the body model's existing full-scale MN rate |

What would be hand-crafting, and is not done: a gain on any afferent -> AN or AN -> DNa02 / PS196_b link; a bias on
DNa02; choosing `mn_ref`, a tonic rate or a ceiling to produce a turn; driving the afferents from `yaw_rate` or `speed`
(the readout scalars) as a default; a per-cell pattern (e.g. a synthetic step cycle) the body does not produce.

## 4. The loop caveat, stated once more

`body.Locomotion` computes `speed` and `yaw` *from* the motor readout (`fwd_dn`, leg MNs, DNa02 L - R). The transducer reads
the motor readout too -- the leg-MN rates and the haltere-MN rate -- and the airborne flag (a body event: launch / land).
That is a loop VNC -> MN rate -> afferent -> VNC, which is the proprioceptive loop the animal has and the model lacked, and
it passes through no hand-written scalar. The Coriolis option reads `yaw_rate`, which *is* a hand-written scalar; hence the
label. `speed` is not read anywhere.

## 5. Smoke on the cluster (`scripts/probe_proprioception.py`)

Protocol: `BatchSim(4, program 'none', fruit 'all', fence, start (-0.15, 0.15, table top), energy 0.9, initial heading
uniform per environment seed, wind 0.3 m/s from 180 deg, cuda graphs / kernels / event-driven, cuSPARSE torch)`, 5 s =
500 frames, every frame recording `rate_hz` of the 941 afferent cells and of AN04B003, AN07B035, AN07B037_a, AN06A026,
PS196_b, IN12B014, IN19A003, DNa02 (by side) plus the motor readout; the commanded afferent Hz is read back from
`FlyBrain._base_poisson` (the injected probability x 1000 / dt) so the table shows what was injected, not what was
intended. Window means drop the first 1 s. Arms: `all` (seeds 0-3), `off` (the shipped path, seeds 0-3), and one
`all+haltere_coriolis` job (seed 0) to show the labelled option runs. One submission, 9 jobs, `--fetch out/proprio/`,
console `out/proprio_cluster.log`; JSONs `out/proprio/{on,off}_r{0..3}.json`, `coriolis_r0.json` (each with
`common.provenance`: resolved LIFParams / OpticParams / `type_path_gain`, realised device, compiled-W and
effective-weights md5, source fingerprint, git state), recordings `*_rec.npz`; the table `report --runs ...`
(`out/proprio/report.csv`).

RESULTS_PLACEHOLDER

## 6. Validation (CPU, `tests/test_proprioception.py`, 9 passed)

(a) **Bit-identity of the shipped path.** On the 22-cell synthetic graph: `FlyBrain` with no sense vs one with
`Proprioception('all')` attached but never fed, 8 frames of smell input -- `v`, `g`, `rate`, `poisson_p`, `spike_counts`
equal at `rtol=0, atol=0`; `available_senses` is `('smell',)` without and `('smell', 'proprioception')` with;
`BatchSim(2, ...)` vs `BatchSim(2, ..., proprioception=None)` equal after 5 frames with `poisson_p` zero on every afferent.
On the shipped cache: a `Connectome.subset` of the 941 afferents plus the eight watch types, `FlyBrain` with and without
the sense attached, 5 frames -- bit-identical, `poisson_p` all zero; then one `proprioception(2.5, 2.5, 0, False)` call
touches exactly the chordotonal + hair plate + campaniform cells (haltere at 0 Hz with no wingbeat) and nothing else.
(b) **Cell sets.** The counts of section 1 are asserted (615 / 113 / 12 / 201; SApp 148; SNpp39 39, SNpp50 62, SNpp60 41,
SNpp52 17, SNppxx 78, SApp23 22; the 12 campaniform enter by ProLN / MesoLN / MetaLN only; every SApp is sided); on the
synthetic graph the instance suffix overrides a contrary wiring, a wing (ADMN) campaniform cell and a bristle are excluded.
(c) **Laws.** Standing still: 10 / 5 / 50 / 0 Hz; left legs at half `mn_ref` raise only the left-sided and (by half)
the unsided chordotonal / hair-plate cells; 1e6 Hz of MN drive clamps every channel to its ceiling; airborne -> tonic /
tonic / 0 / unchanged haltere; the default haltere channel ignores `yaw_rate` and the `haltere_coriolis` arm applies
`(1 + g|yaw|)` and still clamps; negative MN rates and bad specs raise. The `BatchSim` per-frame call feeds tonic / load on
the first frame (no motor snapshot yet) and 0 load on a row flagged airborne.

The existing `tests/test_control.py` and `tests/test_batch_sim.py` CPU suites pass unchanged (section 8).

## 7. Limitations and next steps (data-driven)

1. **No leg cycle.** The chordotonal / hair-plate laws follow the MN rate, so they cannot produce the stance-phase
   modulation a real FeCO has; what they can produce is a sided tonic report of the leg motor output, which is what the
   `deficit_turning.md` 6.3 open loop lacks. A leg model in `body.py` (not this task's file) would replace the independent
   variable without touching this class.
2. **Haltere sides.** `MotorRates.haltere` is bilateral; a per-side haltere MN readout in `motor.py` would let the two
   SApp sides differ. Until then the haltere channel is symmetric by construction.
3. **Neck hair plates (SNpp19, 35 cells)** are driven by the leg law; they should read a head-neck angle the body does not
   have. Selecting them out is a one-line `entryNerve != 'PrN'` change if a later round wants the leg-only set.
4. **The numbers are brackets.** Every range is `op report`; the first Drosophila electrophysiology of FeCO / hair-plate /
   campaniform afferents that reports Hz replaces the defaults through the constructor, not through behaviour.
5. **What to run next** (from `deficit_rotation.md` 3(3)): repeat that audit's efferent arm with `--proprioception all`
   (>= 4 runs per arm in one batch) and read whether PS196_b / LAL / WED fire and whether GLNO's input gains a body-locked
   *sided* term; repeat `interp_apply_turning.py record` with the sense on and read DNa02's operating point and the
   IN12B014 / IN19A003 rates. Neither is done here: this task builds the transducer and shows it drives the wiring.

## 8. Process notes

* The cluster batch ships every file that differs from `origin/main` in this checkout, which at submission included
  another task's uncommitted edits to `flyverse/optic.py` (+215 / -17) and `flyverse/interp/common.py` (+30 / -2) and its
  untracked `tests/test_optic_hooks.py`; the run JSONs' `provenance.source_fingerprint` records the files actually loaded.
  Whether those edits alter the shipped model output is that task's claim; the `off` arm here is the model as shipped in
  *that* tree, and the on-vs-off comparison is within one batch of one tree.
* The CPU end-to-end smoke of the generator (`out/proprio_smoke_cpu/on_r0.{json,txt}`, `CUDA_VISIBLE_DEVICES=-1`, 2 flies
  x 0.3 s, device cpu) ran on this desktop before submission; no number here comes from it.
* The generator first wrote the recording to `<out>.json`, which `Recording.save` also uses; the recording now goes to
  `<out>_rec.{npz,json}` and the summary to `<out>.json`.
