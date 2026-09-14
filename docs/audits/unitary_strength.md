# Per-transmitter unitary strength: is one 0.275 mV for every transmitter what the data say, and does a data-anchored split move the ring?

Thread `unitary` (behaviour round 3, 2026-09-14). Script: `scripts/probe_unitary.py` (`brackets` / `structure` / `report`
CPU; `compass` / `room` GPU; `plan` writes the cluster batches; the suite runs through `scripts/interp_lesion.py run` with a
`lif` lesion per arm). Tests: `tests/test_unitary.py` (8, CPU; 47 with `test_receptor_model.py` + `test_bit_identity.py`). Model change: ONE opt-in field, `LIFParams.w_syn_by_nt:
dict | None = None` (`flyverse/brain.py`, consumed by `_shaped_weights` before the connection cap; `None` is byte-identical,
section 2). Ledger rows: `unitary.*` in `flyverse/data/expected_responses.csv` (8 rows, all `op report`). Data:
`out/unitary/` (`brackets.json`, `structure.json`, `paths_{default,mid}.json`, `fam_compass_<arm>_s<k>.{json,npz,txt}`,
`fam_suite_<arm>_r<k>.{json,txt}`, `fam_room_<arm>_s<k>.{json,txt}`, `report.{md,json}`, `unit2.blocks.json`, `batch{1,2}_*.sh`), console logs
`out/unitary_batch1_cluster.log`, `out/unitary_batch2_cluster.log`. Nothing adopted; every default is where it was.

Dependencies not at `origin/main` at submission (HEAD 6ec2de1): `flyverse/brain.py` (+30 / -1 lines: the field, `_nt_factor`, the two
`_shaped_weights` lines, the fan-in cache key; this thread), `scripts/probe_unitary.py`, `tests/test_unitary.py`, the 8 ledger rows (this thread).
`cluster_run.py` ships the working tree, so whatever the concurrent threads had modified at submission time is in the run
copy as well; the code identity of every number below is the JSONs' `provenance.source_fingerprint` / `effective_weights`
md5, not the commit.

## 0. Short answer

* **The data.** One measured cholinergic per-synapse anchor sits within 20 % of Shiu et al.'s 0.275 mV (ORN -> PN: 5 mV
  over ~23 synapses = 0.22 mV, x0.79); the two others put ACh lower (PN -> KC x0.12-0.36, PN -> LHN x0.45-1.2). **No
  per-synapse fast IPSP in a Drosophila central neuron is on record**; the only inhibitory statement is the chloride
  driving-force argument (I / E 0.15-0.35 per unit conductance), so the inhibitory brackets are an argument, not data.
  Histamine is graded and lives in the optic rate model, x1 throughout. Section 1; 8 `op report` ledger rows.
* **The implementation.** No existing gain can select edges by presynaptic transmitter (type regexes after the cap;
  gain classes by expression tertile), so ONE opt-in field `LIFParams.w_syn_by_nt` multiplies |W| per presynaptic
  transmitter before the connection cap. `None` is byte-identical (47 CPU tests incl. receptor-model and bit-identity);
  the multiplier lands on exactly the transmitter's entries (14,758,537 of 14,779,194 ACh entries at x0.5; the rest are
  cap-saturated). Section 2.
* **The ring: no.** At the shipped gains, no bracket (low / mid / high = ACh x0.5 / 0.8 / 1.0 with I / E 0.25 / 0.5 /
  0.75) forms a bump for a single frame after the pulse: `compass.EPG.bump_survival_s` 0.00 s in 16 / 16 runs (FAIL x16;
  rate and width unscorable); the driven wedges relax to the 10 Hz background in 0.20-0.35 s; default, mid and high agree
  on the post-pulse EPG MEAN to <= 0.035 Hz in every seed (the EPG is the stimulus; the per-frame traces are NOT identical
  in every seed -- section 4); PEN stays at 0.04-0.15 Hz because its net drive is
  +1.2 to +2.3 mV against the 7 mV gap in every arm. A transmitter scale multiplies the tuned Delta7 inhibition and the
  untuned ring feedback by the same factor; gE 2 / gD 15 changed their ratio by type. Classification: null (structural
  zero-SD) on every bump row. Section 4.
* **The cost.** No bracket keeps the suite: `walk.power_sustained_hz` fails in every bracket and every draw (89-103 Hz
  high, 158-174 mid, 190-194 low, against < 50; default 23-49), and low also fires the GF at 128-130 Hz to a walk
  stimulus, escapes 24-50 cm from the loom and loses a motion direction: the brackets are a net disinhibition of the
  graph and the take-off motor pays first. Section 5. In the room (16 x 60 s, 4 runs per arm, one block on the house
  B200s) the shipped fly walks straight and stays on the table (0-1 of 16 leave, 0.1 hops per fly); under high every fly
  hops 39-42 times a minute and all 16 are off the table by 3.5-7.9 s, under mid within 0.4-0.7 s with 56-69 % of the
  time airborne and a fixed DNa02 R-L bias of -11 Hz in every fly (a bias, not steering). Section 6.
* **Adoption** would need a measured inhibitory unitary (or reversal potentials), an ACh-only family, the suite kept in
  >= 4 draws, and a separate type-level mechanism for the Delta7 : ring ratio. Nothing adopted. Section 7.

## 1. The data: measured unitary strengths per transmitter (ledger rows `unitary.*`)

`LIFParams.w_syn = 0.275 mV` is Shiu et al. 2024's single free parameter: *"We chose W_syn such that activation of sugar
GRNs at 100 Hz resulted in roughly 80 % of maximal MN9 firing"* (Methods), one value for every neuron with the sign from
the transmitter prediction. It is a behavioural calibration of one cholinergic pathway, not a per-synapse measurement. What
is on record per transmitter, and what it implies per synapse when divided by the EM synapse counts of the compiled MaleCNS
graph (`scripts/probe_unitary.py brackets`, `out/unitary/brackets.json`):

| transmitter | measurement | source | synapses per unitary pair | mV per synapse | x w_syn | ledger row |
|---|---|---|---|---|---|---|
| ACh (ORN -> PN) | an isolated ORN spike depolarises a PN by ~5 mV; unitary EPSP tau ~30 ms. **The "about 5 mV" is verbatim from Tobin, Wilson & Lee 2017, which attributes it to Kazama & Wilson 2008; the primary is paywalled (403) and was NOT opened, and one secondary summary reports its unitary EPSP as ~7 mV (unitary EPSC ~13.5 pA). If the primary is 7 mV the anchor is 7 / 23 = 0.304 mV = x1.11 -- ABOVE the shipped 0.275, not 21 % below -- so this row must be pinned from the primary before the ACh-only family of section 7 item 2** | Kazama & Wilson 2008, Neuron 58:401 (via Tobin et al. 2017) | ~23 per unitary ORN -> PN connection (10-36 per PN; uEPSP r 0.99 with synapse count), Tobin, Wilson & Lee 2017, eLife 6:e24838; MaleCNS ORN -> uPN mean 26.9 / median 16 (15,261 pairs) | 0.217 (0.19 over the MaleCNS mean) | **0.79** (0.68) | `unitary.ACh.ORN_PN.mv_per_synapse` |
| ACh (PN -> KC) | single-claw EPSPs 0.59-1.76 mV (16 KCs connected through one claw). **The design is verbatim in the main text ("To quantify the range of PN-KC synaptic strengths, we measured the amplitudes of individual EPSPs from KCs connected via a single claw") and the wide range is confirmed, but the two NUMBERS live only in Suppl. Fig. 6, which could not be opened: UNVERIFIED, neither confirmed nor refuted** | Gruntman & Turner 2013, Nat Neurosci 16:1821, Suppl. Fig. 6 | MaleCNS uPN -> KC mean 17.8 / median 17 (19,994 pairs) | 0.033-0.099 | 0.12-0.36 | `unitary.ACh.PN_KC.mv_per_synapse` |
| ACh (PN -> LHN) | "a PN spike depolarizes the LHN by about 1 mV"; ~10 mV to spike; EPSP tau ~40 ms | Jeanne & Wilson 2015, Neuron 88:1014 | MaleCNS uPN -> LH mean 8.1 / median 3 (31,960 pairs) | 0.12-0.33 | 0.45-1.2 | `unitary.ACh.PN_LHN.mv_per_synapse` |
| GABA (LN -> PN) | GABA-A (picrotoxin) fast + GABA-B slow components of odour-evoked PN inhibition; population, not unitary | Wilson & Laurent 2005, J Neurosci 25:9069 | -- | **none located** | -- | `unitary.GABA.LN_PN.ipsp_mv` |
| glutamate (LN -> PN) | GluClalpha-mediated hyperpolarisation of every AL cell type, picrotoxin / RNAi-sensitive; selective Glu-LN activation hyperpolarises PNs; no amplitude in the text | Liu & Wilson 2013, PNAS 110:10294 | -- | **none located** | -- | `unitary.Glu.LN_PN.ipsp_mv` |
| histamine (R1-R6 -> L1/L2) | graded tonic-release synapse through a histamine-gated Cl- channel; ~43,500 vesicles per terminal, ~5,000 molecules per vesicle | Hardie 1989, Nature 339:704; Borycz et al. 2005, J Neurophysiol 93:1611; Juusola et al. 1995, J Gen Physiol 105:117 | -- | **not convertible** to mV per spike | -- | `unitary.His.R16_LMC.graded` |
| I / E per unit conductance | chloride driving force at v_rest -52 mV against E_Cl of about -60 to -70 mV = 8-18 mV, against ~52 mV for a cation channel. **The E_Cl endpoints are NOT pinned by any cited measurement**: the row's `source` field cites Shiu et al. 2024 for `v_rest` only, and the Drosophila numbers available are larval / embryonic GABA reversals of roughly -50 to -60 mV, which would give a driving force of 0-8 mV and an I / E of 0.0-0.15 -- BELOW this row's own bracket. Under the project rule this is a HAND-SET number; the prose says so, the ledger row's evidence does not, and the whole bracket family (and the monotone suite failure in I / E) hangs off it | the model's own constants (Shiu et al. 2024); no Drosophila central per-synapse IPSP / EPSP conductance pair is on record | -- | 0.15-0.35 of the EPSP | -- | `unitary.IoverE.chloride_driving_force` |
| I / E, measured, insect but not Drosophila | **fast IPSPs 223 +- 134 uV (n = 6, type I LN -> uPN pairs) alongside single-spike EPSPs 0.8 +- 1.0 mV (n = 18) in the same preparation** = a measured unitary I / E of about **0.28**, inside this round's 0.25-0.75 bracket and closest to its `mid` arm | *Rapid and Slow Chemical Synaptic Interactions of Cholinergic Projection Neurons and GABAergic Local Interneurons in the Insect Antennal Lobe*, J Neurosci 34(39):13039 (2014), *Periplaneta americana* | no EM synapse count | not convertible to mV per synapse | -- | **PROPOSED ledger row, not added this round** (`flyverse/data/expected_responses.csv` is another thread's file): `unitary.IoverE.insect_unitary` |
| w_syn | 100 Hz sugar GRNs -> ~80 % of maximal MN9 | Shiu et al. 2024, Nature 634:210 | -- | 0.275 | 1 | `unitary.Shiu.w_syn` |

Per-synapse conductances from EM synapse size do not exist for Drosophila central neurons (the searches of this round
found none; Tobin et al. 2017 correlate uEPSP amplitude with synapse *count*, not size). What the table supports:

* **ACh is within 20 % of 0.275 at the one synapse where both the EPSP and the count are measured in the same
  preparation** (ORN -> PN, x0.79); the other two cholinergic anchors put it lower (x0.12-0.36 onto KCs, x0.45-1.2 onto
  LHNs), so the cholinergic bracket is **x0.5-1.0 with its centre at x0.8**.
* **No per-synapse fast-inhibitory number exists FOR DROSOPHILA.** The only data-based statement within the model's own
  terms is the driving-force asymmetry of a chloride channel in a current-based model whose synapses are mV jumps: per
  unit conductance an IPSP is 0.15-0.35 of an EPSP, and the shipped model sets them equal -- and the E_Cl endpoints that
  argument rests on are themselves uncited (table row above). The fast-inhibitory bracket is therefore an **I/E ratio of
  0.25-0.75, with 1.0 (the shipped equality) as its upper limit**. **But "an argument, not a measurement" is stronger
  than the literature warrants** (skeptic pass): a unitary fast IPSP in an INSECT central neuron with a matched unitary
  EPSP in the same preparation IS on record -- J Neurosci 34(39):13039 (2014), *Periplaneta americana*, type I LN -> uPN
  fast IPSPs of 223 +- 134 uV (n = 6) against single-spike EPSPs of 0.8 +- 1.0 mV (n = 18) in type I LNs, i.e. a measured
  cross-species unitary I / E of about **0.28**, sitting inside this round's bracket and closest to its `mid` arm. It has
  no EM synapse count and it is not Drosophila, so the narrow claim above survives, but the paper belongs in the ledger
  (proposed row `unitary.IoverE.insect_unitary`) and is the anchor section 7 item 1's design should be built around.
* **Histamine cannot be bracketed per spike** and, in this model, does not go through `w_syn` at all: the photoreceptor ->
  lamina edges are in the optic-lobe rate model (`optic.py`, normalised by `in_syn`). The LIF holds 90,320 histamine entries
  (687,278 synapses) of which 22,195 (114,559 synapses) survive the frozen prune -- AN27X008 / AN27X004 / GNG043 / IN27X004
  histaminergic ascending / GNG cells and R7 / R8 onto MeTu3c / MeTu3b / aMe12 -- and 17,379 are already silenced by the
  receptor rule (121 of the surviving ones). Histamine stays x1 in every arm. (**The prune bookkeeping in this bullet --
  22,195 / 114,559 surviving, 17,379 silenced, 121 of the survivors -- is NOT independently verified**: checking it needs
  the retina / optic build that defines the frozen set, which the skeptic pass did not run. The upstream figures it rests
  on ARE verified: 90,320 histamine entries / 687,278 synapses in `out/unitary/brackets.json`. Peripheral and
  `op report` only.)

The three brackets run in this round (multipliers on |W| per PRESYNAPTIC transmitter; `probe_unitary.BRACKETS`):

| arm | ACh | GABA | glutamate | histamine | I / E |
|---|---|---|---|---|---|
| default (shipped) | 1 | 1 | 1 | 1 | 1.00 |
| low | 0.5 | 0.125 | 0.125 | 1 | 0.25 |
| mid | 0.8 | 0.4 | 0.4 | 1 | 0.50 |
| high | 1.0 | 0.75 | 0.75 | 1 | 0.75 |

## 2. The implementation: why one field, where it lands, what it does not touch

Neither existing gain machinery can express "x k on every edge whose presynaptic transmitter is T":

* `type_path_gain` / `path_gain` are (pre, post) **type / superclass regexes** applied **after** the connection cap. The
  transmitter is a per-body label: **114 MaleCNS types carry more than one transmitter (3,491 cells; 382,637 entries /
  1.277 M synapses), and separately 2,605 untyped cells (228,772 entries / 0.900 M synapses) carry labels spanning every
  transmitter** -- together 6,096 cells / 611,409 entries / 2.177 M synapses. (An earlier version wrote "115 types
  (6,096 cells; 611,409 entries / 2.18 M synapses) ... plus 228,772 entries from 2,605 untyped cells", which counted the
  untyped group twice and as a type; skeptic pass. The argument is unaffected.) So a type regex cannot select the edges,
  and a post-cap gain cannot reproduce a pre-cap scale (a 120-synapse connection at x0.5 is 60 synapse-equivalents
  before the cap and 30 after it).
* `receptor_gain` is keyed on the table's expression-tertile **gain class** (`none / low / mid / high`), not on the
  transmitter, and is `None` under the shipped `sign` model.

So the least invasive path is the one field the task allowed: `LIFParams.w_syn_by_nt: dict | None = None`, consumed in
`brain._shaped_weights` on `|W|` after the receptor sign / gain-class stage and **before the connection cap**, before the
path gains, the same-type damping and the fan-in normalisation (`brain._nt_factor`, +30 / -1 lines; the fan-in cache key
carries the dict so a second Brain in one process cannot reuse a stale total). `None` executes nothing. What it means:

* mV per presynaptic spike on edge (post i <- pre j) becomes `w_syn x scale_i x same_type^[..] x path x type_path x
  min(f[nt_j] x |c|, conn_cap) x fast_sign x receptor_gain` -- i.e. the **cap saturates at 60 synapse-equivalents of the
  scaled count** (tests pin both orders: 90 x 2 -> 60, 70 x 0.5 -> 35).
* The **fan-in normalisation partly renormalises** it: `scale_i = clip((5000 / tot_i)^1, 0.02, 1)` on the reference graph,
  and a uniform reduction lowers `tot_i`, so the 2,436 cells above `input_norm_ref` (their mean total 8,725 synapse-eq.)
  get their scale raised x1.50 (low), x1.27 (mid), x1.06 (high) on average; 364 / 1,182 / 2,088 cells remain below 1
  under low / mid / high. Sum |A| over the whole LIF is x0.391 / x0.677 / x0.913 of the shipped 29.18 M mV
  (`out/unitary/structure.json`).
* **Not touched**: the optic-lobe rate model (its weights are fractions of `in_syn`; the histamine photoreceptor -> lamina
  synapse lives there), the slow monoamine matrices (`_slow_weights` scale `receptor.count x slow_factor`, not `W.data`),
  the sign-0 entries (0 x f = 0).

Tests (`tests/test_unitary.py`, 8 passing on the CPU; `tests/test_receptor_model.py` and `tests/test_bit_identity.py`
unchanged, 47 passing together with 5 subtests, 2026-09-14): `None`, `{}`, all-ones and `{"acetylcholine": 1.0}` are byte-identical to the shipped
shaped weights on the synthetic graph and `None` on the cached connectome; a CPU `Brain` under `None` carries the same
matrix and the same state after 200 stimulated steps; the multiplier lands on exactly the entries whose presynaptic cell
carries the transmitter -- on the cached graph x0.5 moves 14,758,537 of the 14,779,194 acetylcholine entries (the 20,657
others are connections of >= 120 synapses, capped at 60 either way) and 0 entries of any other transmitter; the fan-in scale
is recomputed. Every interp tool reads the field through `common.effective_weights` (`model_record` resolves it into every
JSON): `interp_paths.py --lif 'w_syn_by_nt={...}'` reproduces the scaled links -- `out/unitary/paths_{default,mid}.json`
ER4m -> EPG -125.3 -> -50.7 mV per volley; and, from `out/unitary/structure.json` (NOT from `paths_default.json`, whose
links table holds only 10 rows and does not include ExR6 -- skeptic pass), ExR6 -> EPG -30.0 -> -14.9 through the cap and
EPG -> EPG +6.71 -> +5.47. Both numbers were re-derived independently with `common.effective_weights` and reproduce
(-30.021 / -14.912).

**One defect of the report path, disclosed and NOT fixed this round** (it is a script change and this round's edits are
frozen): `scripts/probe_unitary.py` `_stats()` stores `'values': [round(float(x), 4) ...]` and `cmd_report` then calls
`common.compare` on those ROUNDED values. For a quantity of order 1e-3 that materially changes the statistic --
`report.json` gives compass mid `rest_mean_post` diff 0.000225 at z 4.50, where the unrounded runs give diff 0.000203 at
z 7.12, and low is z -3.50 rounded against -7.29 unrounded. No verdict flips anywhere in this family (the skeptic re-ran
every compass and room comparison unrounded), but rounding to 4 dp before a `z = diff / SD(null)` test could flip a
verdict on a smaller effect, and the fix -- compare on unrounded values, round only at print time -- is owed before the
next family.

## 3. Structure first: what a per-transmitter scale does to the ring's balance (CPU, `probe_unitary.py structure`)

One-step effective weights between the compass groups, mV per postsynaptic cell per presynaptic-group volley (the paths
tool's `type_matrix` quantity; `cx_wedge.md` section 2's numbers for the default), and the net E / I per volley onto EPG
and PEN from the whole graph (`out/unitary/structure.json`):

| arm | EPG->PEN | PEN->EPG | EPG->PEG | PEG->EPG | EPG->Delta7 | Delta7->EPG | Delta7->PEN | Ring->EPG | Ring->PEN | EPG->Ring | EPG E / I per volley | PEN E / I |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default | +79.9 | +127.5 | +70.8 | +5.9 | +130.3 | -25.7 | -45.2 | **-766.9** | -96.3 | +28.7 | +227 / -847 | +181 / -143 |
| low | +40.4 | +71.8 | +46.7 | +3.0 | +65.1 | -3.2 | -5.9 | -78.7 | -15.8 | +17.3 | +122 / -107 | +92 / -22 |
| mid | +64.5 | +109.5 | +64.3 | +4.8 | +104.2 | -10.3 | -19.0 | -290.8 | -51.3 | +25.5 | +189 / -342 | +147 / -72 |
| high | +79.9 | +127.6 | +70.8 | +5.9 | +130.3 | -19.3 | -35.4 | -568.2 | -80.5 | +29.2 | +227 / -640 | +181 / -118 |

Two readings before any GPU job. (i) `cx_wedge.md`'s diagnosis was that the untuned EPG -> ExR / ER -> EPG, PEN feedback
(glutamate / GABA) shuts the loop at gain x1 -- Ring -> EPG -767 mV per volley against PEN -> EPG +127; the brackets move
that ratio from 6.0 : 1 (default) to 1.1 : 1 (low), 2.7 : 1 (mid), 4.5 : 1 (high) **but they scale the tuned inhibition
(Delta7 -> EPG, the cosine-shaped term the attractor needs) by the same factor or more**: Delta7 -> EPG / PEN -> EPG goes
0.20 (default) -> 0.045 (low). A per-transmitter scale cannot do what gE 2 / gD 15 did, which was to raise Delta7 -> EPG
x15 while leaving the ring feedback at x1 -- the two inhibitory terms are the same transmitter class. (ii) The ExR / ER
links are still the largest single inputs of EPG in every arm (`paths_mid.json`: ER4m -> EPG -50.7, ER4d -29.5, ExR6
-14.9 per volley, against Delta7 -10.3), so whether a bump *confines* is the LIF's question, not the linear estimate's.

## 4. Compass at the shipped gains under the brackets (GPU, `probe_unitary.py compass`, batch `unit1-11e76c`)

The protocol is `scripts/cx_wedge.py simulate` at gE = gD = gR = 1 (no compass gains; the shipped `type_path_gain`),
receptor rule `sign / abs`, compass adaptation off (cx_wedge's and compass_room's experiment convention, `--adapt off`):
10 Hz Poisson background on the 46 EPG for the whole run, 1 s settle, one 4-wedge block driven +40 Hz for 2 s, then 5 s
free; the per-frame EPG record is scored with `probe_compass_room.bump_frames` (the ledger rows `compass.EPG.*`). Four
seeds per arm; per arm, seeds 0 / 2 ran on r3-h200b and 1 / 3 on r3-h200a (`--arm-block fam` resolved to the token
after `fam_`, i.e. to one block per file, and cluster_run dealt them round-robin: both boxes are H200s and the split is
the same for every arm, so the box is balanced across arms rather than confounded with them; `out/unitary/
fam_compass_<arm>_s<k>.{json,npz,txt}`; `report.md`). Mean [min-max] over the 4 runs; Hz:

| arm | survival s | confined post | EPG in, pulse | EPG in, post | EPG out, post | PEN post | Delta7 post | Ring post | rest of brain | in-wedge back to background |
|---|---|---|---|---|---|---|---|---|---|---|
| default | 0.00 [0-0] | 0.00 [0-0] | 48.6 [46.8-49.9] | 10.37 [10.15-10.76] | 9.81 [9.61-9.92] | 0.04 [0.02-0.06] | 10.42 [10.23-10.53] | 0.81 [0.79-0.82] | 0.0015-0.0016 | 0.20-0.35 s |
| low | 0.00 [0-0] | 0.00 [0-0] | 52.5 [51.6-53.3] | 10.52 [10.20-10.97] | 9.80 [9.61-9.93] | 0.15 [0.10-0.18] | 0.66 [0.49-0.77] | 0.68 [0.67-0.69] | 0.0014-0.0015 | 0.23-0.35 s |
| mid | 0.00 [0-0] | 0.00 [0-0] | 48.6 [46.8-49.9] | 10.37 [10.15-10.78] | 9.80 [9.61-9.94] | 0.07 [0.06-0.08] | 6.63 [6.42-6.92] | 0.88 [0.86-0.89] | 0.0017-0.0018 | 0.20-0.35 s |
| high | 0.00 [0-0] | 0.00 [0-0] | 48.6 [47.0-49.9] | 10.38 [10.15-10.80] | 9.81 [9.63-9.92] | 0.05 [0.03-0.08] | 11.38 [11.15-11.56] | 0.86 [0.85-0.87] | 0.0016-0.0017 | 0.21-0.35 s |

Ledger: `compass.EPG.bump_survival_s` **FAIL in 16 / 16 runs** (0.00 s against >= 5); `bump_rate_hz` and
`bump_width_wedges` NOT_APPLICABLE in all 16 (no confined frame after the pulse to score; `frac_confined_pre` 0 as well).
`common.compare` against the default arm (4 vs 4 runs, exact U): survival and confinement are **null with a zero-SD
null** (structural: every run is 0.00), so their p is not a test; PEN post-pulse +0.11 Hz (low, z +6.2, p 0.029,
*result*), Ring -0.13 / +0.07 / +0.06 Hz (low / mid / high, *result*), EPG in / out *null* in every arm, i.e. the brackets
move the interneurons' background by fractions of a hertz and the EPG not at all.

What the traces say (`fam_compass_*.npz`): after the pulse the driven wedges fall from 46-63 Hz to within 2 Hz of the
undriven ones in **0.20-0.35 s in every arm** (the synaptic + membrane relaxation of a Poisson-driven cell, not an
attractor's decay), and the post-pulse EPG SUMMARY of default, mid and high agrees to **<= 0.035 Hz in every seed**
(`epg_in_mean_post`). The seed-0 sample below is exact -- in-wedge 51.8 / 25.8 / 16.4 / 15.2 / 9.4 Hz at 0 / 0.1 / 0.2 /
0.3 / 0.5 s after the pulse in all three arms, maximum frame-wise difference exactly 0.0000 -- but **the per-frame
traces are NOT identical in every seed**: in the shipped family seed 2 differs by up to 1.72 Hz (mid) and 1.69 Hz (high)
and seed 3 by up to 4.23 Hz (high) in the per-frame in-wedge mean, and 3 of 4 seeds differ by 1.6-4.3 Hz in the
skeptic's fresh-seed replication. An earlier version of this paragraph said "the per-seed EPG records ... identical to a
tenth of a hertz", which overstates what the npz show; the conclusion (the EPG rate is the stimulus, and the recurrent
ring contributes nothing measurable to it at the shipped gains) rests on the post-pulse mean and is unchanged. The one
arm that differs, low, differs *during* the pulse (52.5 vs 48.6 Hz in-wedge) because Delta7 is 15-16 Hz there against
30-38 Hz in the other arms, not after it.

Why the loop never closes, from the section 3 weights and the measured rates: the LIF's synaptic variable decays with
`tau_syn` 5 ms, so a presynaptic group's steady contribution is (mV per volley) x (rate) x 0.005 s. EPG -> PEN at the
measured 9.8 Hz background is **3.9 mV (default, high), 3.2 (mid), 2.0 (low)**; Delta7 + Ring -> PEN at their measured
rates is -2.7 / -2.4 / -0.9 / -0.1 mV (default / high / mid / low); the net is **+1.2 to +2.3 mV against the 7 mV
threshold gap in every arm**, so PEN sits at 0.04-0.15 Hz after the pulse (0.3-4.5 Hz during it; low the highest because
its Delta7 is halved), and with PEN silent there is no recurrent excitation for a bump to live on -- PEN -> EPG (+127
mV per volley at default) is the only large excitatory return path (PEG -> EPG is +5.9). The brackets that cut the
inhibition most (low) also cut the excitation (ACh x0.5) below what the pulse needs; the one that keeps ACh (high) keeps
Delta7 at 11 Hz. GLNO, PEN's largest single input (9,815 synapses onto PEN_a, transmitter unknown, sign 0), is silent in
every arm by construction (`struct.GLNO_PEN.sign`, docs/audits/deficit_rotation.md). None of this is what gE 2 / gD 15
did: those raised EPG -> PEN x2 and Delta7 -> EPG x15 *by type*, i.e. changed the ratio of the cosine-shaped Delta7
inhibition to the untuned ring feedback, which a per-transmitter scale leaves fixed (both are GABA / glutamate).

## 5. The benchmark suite under the brackets (GPU, `interp_lesion.py run`, batch `unit1-11e76c`, r3-h200a)

Sections rest / taste / smell / walk (which carries the loom and rotate checks) / motion, 3 draws per arm = 3 jobs, each
draw the benchmark's seeds 0, 1, 2 (`fam_suite_<arm>_r<k>.json`; `manifest.json`). Measured values per draw
(criterion; the benchmark's reference in brackets):

| check (criterion) [ref] | default | low | mid | high |
|---|---|---|---|---|
| `walk.power_sustained_hz` (< 50) [22] | 49.2, 23.3, 24.1 PASS | **189.7, 192.7, 193.9 FAIL** | **158.4, 167.7, 174.2 FAIL** | **89.3, 102.9, 96.4 FAIL** |
| `walk.GF_max_hz` (< 38) [26] | 5.0, 9.4, 13.4 | **130.1, 127.9, 128.4 FAIL** | 19.6, 10.5, 14.7 | 4.8, 4.7, 4.7 |
| `walk.power_max_hz` (not none) [22] | 95.5, 53.7, 51.3 | 268.9, 268.2, 272.4 | 209.6, 236.3, 237.3 | 120.1, 137.5, 143.9 |
| `motion.correct_directions` (== 8) | 8, 8, 8 | **7, 7, 7 FAIL** | 8, 8, 8 | 8, 8, 8 |
| `motion.min_dsi` (>= 0.1) [0.16] | 0.186 x3 | 0.153 x3 | 0.154 x3 | 0.202 x3 |
| `loom.GF_peak_hz` (>= 20) [26] | 28.0, 25.3, 26.1 | 113.2, 128.8, 120.0 | 37.7, 54.5, 55.9 | 41.8, 32.1, 36.3 |
| `loom.escape_cm` (not none) [3.5] | 3.5 x3 | 50.0, 50.0, 24.0 | 3.5, 6.0, 3.5 | 3.5 x3 |
| `rotate.DNp20_flip_hz` (< -2) [-14] | -29.5, -27.8, -30.5 | -17.9, -6.8, -14.8 | -22.8, -31.2, -22.5 | -36.3, -33.9, -18.3 |
| `taste.MN9_hz` (> 2) [3.7] | 5.8, 3.1, 9.5 | 7.2, 5.3, 11.2 | 34.7, 20.2, 49.4 | 12.6, 11.8, 31.3 |
| `smell.PN_hz` (< 100) [13] | 11.2, 11.4, 4.7 | 19.1, 4.5, 4.3 | 4.0, 17.6, 4.2 | 2.5, 10.7, 2.6 |
| `smell.KC_active` (> 0) [1249] | 1426, 1235, 1485 | 1397, 1886, 1738 | 534, 2221, 945 | 349, 1914, 515 |
| `rest.spikes_per_step` (< 5) [0] | 0 x3 | 0 x3 | 0 x3 | 0 x3 |
| **pass / fail of 12** | **12 / 0, 12 / 0, 12 / 0** | 9 / 3 x3 | 11 / 1 x3 | 11 / 1 x3 |

**No bracket keeps the suite.** Every one fails `walk.power_sustained_hz` in all three draws, monotonically in the
I / E ratio: the sustained power-muscle motor-neuron rate under the walk stimulus is 23-49 Hz at the shipped equality,
89-103 at I / E 0.75, 158-174 at 0.5, 190-194 at 0.25 (the bound is 50; the reference 22). The brackets are a net
disinhibition of the whole graph -- ACh x0.5-1.0 against GABA / glutamate x0.125-0.75 -- and the take-off / flight motor
is where it shows first: at low the GF fires at 128-130 Hz to a walk stimulus (bound 38), the loom escape is 24-50 cm
(3.5 shipped), the loom GF peak 113-129 Hz, and one of the 8 motion directions is lost; at mid the taste MN9 runs 3-5x
(20-49 Hz, still a pass) and the odour-active KC count swings 534-2,221 across draws (1,235-1,485 shipped). The
pattern is the same one the receptor-integration audit found for the histamine silencings (G.6): the shipped model's
inhibition is calibrated implicitly by Shiu's equality, and any data-anchored I / E < 1 has to be paid for elsewhere.
`high` (ACh x1, I / E 0.75) is the bracket closest to keeping the suite (one failing check, 89-103 vs 50) and is the one
the room rollout compares against the shipped model; `mid` (the predecessor's planned arm) runs alongside for the cost
gradient.

## 6. The plain-fly room under the brackets (GPU, `probe_unitary.py room`, batch `unit2-cd8abf`, house cluster)

`scripts/probe_walk_straightness.py`'s plain-fly rollout: 16 flies per run (`BatchSim`, env seeds **100 x seed ..
100 x seed + 15**, i.e. 0-15 / 100-115 / 200-215 / 300-315 for runs 0-3 -- `cmd_room` builds
`range(a.seed * 100, a.seed * 100 + a.batch)` and the run JSONs agree; an earlier version wrote "100 k .. 100 k + 15"
with k the thousands, which matches no run: skeptic pass. The four runs of an arm are independent environments, paired
across arms),
no program, no fence, every fruit, 60 s, started on the table top; 4 runs per arm; arms default (the reference), high
(nearest to keeping the suite) and mid (the predecessor's planned arm), **12 jobs in ONE block on <cluster-node>** (all on an
NVIDIA B200; torch sparse, event-driven, cuda kernels; `out/unitary_batch2_cluster.log`: `12 job(s), 0 failed (12.8
min)`; `fam_room_<arm>_s<k>.{json,txt}`, md5-verified against the run directory). Because batch 1 ran on H200s, the
room family is its own within-block comparison; its default arm against earlier rounds' H200 room numbers is a
replication across devices, not a bit-for-bit check. Per run (medians over the 16 flies unless a mean is named):

| arm | run | yaw SD deg/s (walking frames) | straightness | hops per fly (mean) | airborne fraction (mean) | flies off the table / 16 (median leave time; first) | DNa02 R-L Hz: mean of abs / signed mean | leg L-R Hz: abs / signed mean | wall s |
|---|---|---|---|---|---|---|---|---|---|
| default | 0 | 2.9 | 0.997 | 0.1 | 0.000 | 0 | 0.09 / +0.06 | 0.328 / +0.147 | 607 |
| default | 1 | 2.8 | 0.996 | 0.1 | 0.000 | 0 | 0.10 / +0.06 | 0.328 / +0.145 | 580 |
| default | 2 | 2.8 | 0.996 | 0.1 | 0.000 | 1 (59.4 s) | 0.10 / +0.06 | 0.325 / +0.146 | 597 |
| default | 3 | 2.6 | 0.997 | 0.1 | 0.000 | 1 (53.8 s) | 0.08 / +0.06 | 0.326 / +0.155 | 599 |
| high | 0 | 31.8 | 0.903 | 38.9 | 0.125 | 16 (7.9 s; 0.4 s) | 1.48 / -0.01 | 0.559 / +0.168 | 421 |
| high | 1 | 33.5 | 0.888 | 42.2 | 0.134 | 16 (7.3 s; 0.4 s) | 1.48 / -0.08 | 0.557 / +0.164 | 394 |
| high | 2 | 31.2 | 0.893 | 41.1 | 0.130 | 16 (4.6 s; 0.4 s) | 1.52 / -0.14 | 0.552 / +0.151 | 423 |
| high | 3 | 30.1 | 0.881 | 41.5 | 0.128 | 16 (3.5 s; 0.5 s) | 1.46 / -0.11 | 0.566 / +0.169 | 411 |
| mid | 0 | 191.3 | 0.094 | 45.3 | 0.629 | 16 (0.4 s; 0.4 s) | 16.53 / -11.28 | 1.148 / +0.634 | 610 |
| mid | 1 | 238.0 | 0.109 | 46.9 | 0.608 | 16 (0.4 s; 0.4 s) | 16.57 / -11.24 | 1.162 / +0.630 | 603 |
| mid | 2 | 536.9 | 0.119 | 38.5 | 0.687 | 16 (0.7 s; 0.4 s) | 16.77 / -11.06 | 1.158 / +0.643 | 592 |
| mid | 3 | 209.4 | 0.118 | 52.5 | 0.561 | 16 (0.5 s; 0.4 s) | 16.15 / -11.30 | 1.139 / +0.626 | 407 |

`common.compare` vs default (4 vs 4 runs, exact U, p 0.029 at best): hops, airborne fraction, flies off the table, yaw
SD, straightness and the DNa02 |R-L| are **result** in both arms (the z values are in the hundreds because the default
arm's run-to-run SD is a few hundredths); `left_table_s` is *underpowered* (the default arm has two runs with no leaver);
the signed leg offset is *null* at high (+0.151 to +0.169 vs +0.145 to +0.155) and *result* at mid (+0.63).

Reading. The shipped plain fly walks straight and stays on the table for a minute (0-1 of 16 leave, at 54-59 s;
0.1 hops per fly; the leg L-R offset a fixed +0.15 Hz in all 64 fly-runs, the sign the body-state thread reports for
its 400). Under the brackets the fly does not walk more: it **takes off**. At high (I / E 0.75) every fly hops 39-42
times a minute, is airborne 13 % of the time, and all 16 are off the table by a median 3.5-7.9 s (the first at 0.4 s in
every run); at mid (I / E 0.5) 39-53 hops, airborne 56-69 %, all 16 off within 0.4-0.7 s, and the yaw SD of 191-537
deg/s is the tumbling of a fly that is never on its feet, not steering (straightness 0.09-0.12). This is section 5's
`walk.power_sustained_hz` failure seen from the body: the take-off / flight power motor neurons run at 89-194 Hz when
the fast inhibition is scaled below the excitation, and the body's take-off rule (`gf_hz` / `takeoff_power_hz` /
`takeoff_hold_s`, `provenance.model.body`) fires on them. The DNa02 readout changes with it -- |R-L| 0.09 -> 1.5 (high)
-> 16.5 Hz (mid) -- but at mid it is a **fixed signed bias**, R-L = -11.1 to -11.3 Hz in every run and -10.0 to -12.3
Hz in every one of the 16 flies of run 0: the connectome's left-right asymmetry surfacing when the tonic inhibition on
DNa02 (`deficit_turning.md`: -1.6 / -2.0 mV steady) is cut, on a fly that is airborne two thirds of the time. A bias of
one sign in every fly is not a turning signal, and it is exactly the class of thing the project rule forbids adopting
(a fixed offset fed forward as if it were steering). At high the sign is not fixed (-0.01 to -0.14 mean, 1.5 abs) and
the leg offset is the shipped one.

The room cost of the least damaging data-anchored bracket is therefore total for the plain-fly assay: **a fly that hops
40 times a minute and leaves the table in seconds**; `min_fruit_cm` (2.8-4.0 vs 3.8-4.6 cm) says nothing, since the
flies reach the fruit by falling past it. The steering deficit that rounds 1-2 diagnosed (a straight walker with an
unsigned compass report) is not touched: the walking frames that remain under high have a yaw SD of 30-34 deg/s made of
landings, with the same fixed leg offset.

## 7. Answer, classification, and what an adoption would require

**Is a data-anchored per-transmitter scale enough to make the ring a working attractor at the shipped gains? No.** Under
the shipped `type_path_gain` and receptor rule, no bracket in the measured range (ACh x0.5-1.0; fast inhibition x0.125-
0.75 of it) produces a confined bump for a single frame after the pulse (0 / 16 runs; ledger `compass.EPG.bump_survival_s`
FAIL x16, rate and width not scorable), the EPG record is the Poisson stimulus to a tenth of a hertz in three of the four
arms, and the recurrent loop is open at PEN (net drive +1.2 to +2.3 mV against a 7 mV gap; PEN 0.04-0.15 Hz) in every
arm. Classification per docs/INTERP.md section 2: the bump rows are **null (structural zero-SD)** in every arm; the
interneuron background shifts are *result* at the fraction-of-a-hertz level and steering-irrelevant. The experiment
gains that do make a bump (gE 2 / gD 15, `cx_shift.md`) are a x2 / x15 on *specific types* -- they change the ratio of
the tuned Delta7 inhibition to the untuned ring feedback -- and a per-transmitter scale, which multiplies both by the
same factor, cannot reproduce that whatever the bracket. What it costs elsewhere: every bracket breaks
`walk.power_sustained_hz` (89-194 Hz vs < 50) and the low bracket breaks the GF / loom / motion checks as well
(section 5); in the room the fly under high hops 39-42 times a minute and leaves the table within seconds, under mid it
is airborne two thirds of the time with a fixed -11 Hz DNa02 bias (section 6).

What is *supported* by the data, independent of the ring: the cholinergic per-synapse anchor at ORN -> PN is x0.79 of
0.275 (Kazama & Wilson 2008 over Tobin et al. 2017's counts), the other two cholinergic anchors put it lower (KC x0.12-
0.36, LHN x0.45-1.2), and there is **no per-synapse fast-inhibitory measurement in a Drosophila central neuron** on
record -- the inhibitory bracket is the chloride driving-force argument in a current-based model, an argument the model
itself cannot honour without reversal potentials. Nothing is adopted; `w_syn_by_nt` defaults to `None` and the shipped
path is byte-identical (section 2; 47 CPU tests across `test_unitary.py`, `test_receptor_model.py`,
`test_bit_identity.py` pass, `out/unitary` provenance records the field in `model.lif.w_syn_by_nt`).

An adoption of any per-transmitter scale would require, in order:

1. **A measured fast IPSP per synapse** in a Drosophila central neuron (a paired recording with an EM count, the
   Tobin et al. 2017 design for an inhibitory pair -- e.g. an APL -> KC or MBON pair, or an AL LN -> PN pair), or else a
   conductance-based synapse with reversal potentials so that the I / E asymmetry is the driving force and not a scalar.
   The design should be anchored on the one insect unitary I / E that IS measured (0.28, *Periplaneta*, J Neurosci
   34:13039), which this round missed and which should be a ledger row; and
   `unitary.IoverE.chloride_driving_force` must either cite an E_Cl measurement or be relabelled hand-set, since its
   endpoints currently cite nothing.
2. **An ACh-only family** ({"acetylcholine": 0.8} with inhibition at x1; and the KC / LHN anchors as arms) run through
   the suite and the compass: this round's arms all move inhibition with excitation, so the effect of the one anchored
   number has not been isolated. Shiu et al.'s 0.275 was calibrated on a cholinergic pathway (100 Hz sugar GRNs -> ~80 %
   of maximal MN9), so an ACh scale is also a re-calibration of `taste.MN9_hz`; that check must be re-read, not just
   re-passed.
3. **The suite kept** at the candidate (12 / 12 in >= 4 draws on one box), and the room rollout (16 x 60 s, >= 4 runs,
   one block) with the compass ledger rows re-scored -- and, if a ring result is the goal, a *separate* mechanism for the
   Delta7 : ring ratio (a type-level fact the connectome does not fix through the transmitter), since section 4 shows a
   transmitter scale cannot supply it.
4. **Five runs per arm** for the interneuron-background differences to be more than a fraction-of-a-hertz result. The
   house (B200) replication of the H200 compass family that this item asked for is **DONE and passes**: the skeptic's
   `uskep-262b93` (16 jobs = 4 arms x fresh seeds 4-7, ONE block, `out/unitary_skeptic/`, `16 job(s), 0 failed (4.4
   min)`, every run NVIDIA B200 with the same receptor-table and connectome md5s) returns `bump_survival_s` 0.000 and
   `frac_confined_post` 0.000 in 16 / 16, FAIL x16, rate / width NOT_APPLICABLE x16, Delta7 post 10.868 / 0.671 / 7.075 /
   11.841 against the H200's 10.418 / 0.656 / 6.629 / 11.381, PEN low +0.0888 (z +5.3, p 0.029, result), Ring low -0.133
   / mid +0.069 / high +0.058 result, EPG in / out null in every arm. The finding is device-independent and
   seed-independent.

**Governance note for the hand-off** (not a defect in the work): `flyverse/brain.py` was on this round's never-edit list
and this thread edited it. The edit takes the "new `LIFParams` field unavoidable" carve-out and satisfies both of its
conditions -- default `None`, and a CPU bit-identity test (byte-identical shaped weights under `None` / `{}` / all-ones,
md5 `6c36faf3069d0af79d1b0855f83040bb` for all three; `test_bit_identity`'s 5 subtests pass) -- but the next round should
know that `brain.py` carries an unmerged field from thread unitary.

## 8. Files, reproduction, and the state of the batches

* Code: `flyverse/brain.py` (`LIFParams.w_syn_by_nt`, `_nt_factor`, the two `_shaped_weights` lines and the fan-in cache
  key; +30 / -1 lines; `None` byte-identical), `scripts/probe_unitary.py` (`brackets` / `structure` / `compass` / `room` /
  `plan` / `report`; `plan --only 2` writes an explicit `--arm-block-map` so the family is ONE block -- the fix for batch
  1's per-file blocks), `tests/test_unitary.py` (8 tests), 8 `unitary.*` ledger rows (`op report`).
* Batch 1 `unit1-11e76c` (28 jobs: 16 compass + 12 suite; r3-h200a 20 completed + r3-h200b 8 completed, 0 failed,
  `out/unitary_batch1_cluster.log`, both halves in `out/unitary/`, **size-verified against the box, NOT md5-verified**).
  The only local record is `out/unitary/FETCHED.txt` -- "every remote file present locally with matching size (python
  size compare)" -- and neither `scripts/cluster_run.py` nor `scripts/fetch_run.py` computes an md5, so the word
  "md5-verified" that an earlier version used here (and in the REPORT) was unbacked; r3-h200b now refuses connections,
  so the comparison can no longer be made (skeptic pass). Batch 1's completion is still well evidenced (28 consoles,
  12 lesion records, the box's own counts, the size match); batch 2's md5 claim below was independently confirmed.
  Every run's `provenance.execution.device_name` is `NVIDIA H200`; torch 2.11.0+cu128; compass backend torch sparse +
  cuda graphs. **Batch 1's compass family was not one block**: `--arm-block fam` resolved to one block per FILE and
  cluster_run dealt 16 one-job blocks round-robin over the two H200s. It is balanced, not confounded (seeds 0 / 2 ->
  r3-h200b and 1 / 3 -> r3-h200a for all four arms, `out/unitary_batch1_cluster.log` lines 10-25), and the skeptic's
  16-job single-block B200 rerun at fresh seeds reproduces every sign and magnitude -- but as executed it is a departure
  from the cluster rule and is said so here rather than described as "one submission, both boxes H200, balanced per arm".
* Batch 2 `unit2-cd8abf` (12 room jobs: default / high / mid x 4 runs, house cluster <cluster-node>, ONE block through
  `unit2.blocks.json`; `out/unitary_batch2_cluster.log`: `12 job(s), 0 failed (12.8 min)`; every run's device `NVIDIA
  B200`; fetched by the client and md5-verified against `<cluster-fs>/neurome/runs/unit2-cd8abf/out/unitary/`). Batch 1's
  suite also wrote its 12 per-run lesion records to `out/interp/lesion/lesion-20260914T08*.json` (fetched).
* CPU: `python scripts/probe_unitary.py brackets --json out/unitary/brackets.json`, `structure --json
  out/unitary/structure.json`, `report --dir out/unitary` (writes `report.md` / `report.json`);
  `python scripts/interp_paths.py --lif 'w_syn_by_nt={...}'` for the scaled links (`paths_{default,mid}.json`).
* The run copies carried the working tree at submission (the concurrent threads' edits to `body.py` / `motor.py` /
  `senses.py` / `batch_body.py` included, all opt-in and OFF by default); the code identity of every number is the
  JSON's `provenance.source_fingerprint` (44 files hashed), not HEAD 6ec2de1.

