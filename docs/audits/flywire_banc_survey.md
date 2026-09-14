# FlyWire FAFB v783 and BANC v888: what the two female connectomes offer flyverse

Survey of the two public Princeton / FlyWire releases in `D:\Datasets\flywire\` (2026-09-14), read against the
MaleCNS v1.0 graph flyverse ships on. Everything below is computed from the release tables with pandas and the
flyverse cache (CPU only); the script fragments are in the session scratchpad and are reproducible from the tables
alone. No model was changed.

## 1. What is on disk

| release | scope | cells | edges (table) | synapses | per-cell NT | typing |
|---|---|---:|---:|---:|---|---|
| **FAFB v783** (Female Adult Fly Brain) | brain incl. both optic lobes | 139,255 | 5.34 M rows (`connections_princeton`, pairs >= 5 syn; a 22.3 M-row `_no_threshold` table too) | 50.7 M | 6-class probabilities (`da/ser/gaba/glut/ach/oct_avg`) + `nt_type`; 14 % unlabelled | 8,772 primary types; `visual_neuron_types` (741 types, subsystem labels); `column_assignment` (45,528 cells, 31 types, hex `x,y,p,q`, ~790 columns/side) |
| **BANC v888** (Brain And Nerve Cord) | brain + VNC | 158,262 | 3.99 M rows (pairs >= 3 syn) | 23.6 M | predicted 9-class (`ACH/GLUT/GABA/DA/HIST/OCT/SER/TYR`) + **verified** transmitter for 65,369 cells (literature; incl. co-transmitters such as `glutamate,serotonin`) | 11,546 primary types; `Body Part` / `Function` / `Nerve` for sensory cells; 259 hemilineages |

Also present: `sk_lod1_783_healed.zip` (13.9 GB, FAFB skeletons), `labels.csv.gz` (community labels), `cell_stats`,
`coordinates`, `connectivity_tags`. BANC's `connections` carries no per-edge NT (all NaN): transmitter comes from the
neuron table.

Two caveats that shape every comparison below:

- **Synapse yield differs.** On the same post-synaptic cells the raw input counts scale MaleCNS : FAFB : BANC ~
  1 : 0.6 : 0.3 (DNa02, 2 cells: 47.7 k / 28.5 k / 13.3 k synapses; PS059: 28.1 k / 18.0 k / 9.4 k). MaleCNS is read at
  minconf 0.5; the FlyWire tables are thresholded per pair (>= 5 / >= 3). Counts are not comparable across releases
  without a per-release scale; **ratios within a release are**.
- **BANC's optic lobes are under-proofread** (T2 853 vs FAFB 1,466 vs MaleCNS 1,630; Tm3 1,098 / 1,755 / 2,054;
  T4a 1,094 / 1,457 / 1,684). FAFB's optic lobe is the complete one (~1,450 T4a over ~1,580 columns). BANC is the
  release with the VNC.

## 2. Vocabulary overlap with MaleCNS (the bridge already exists)

Exact type-name matches: MaleCNS 11,751 types vs FAFB 8,772 -> **4,715 shared (59 % of MaleCNS cells)**; vs BANC 11,546
-> **7,464 shared (72 % of MaleCNS cells)**, because BANC carries the VNC types (`AN04B003` 6/6, `IN12B014` 4/4,
`IN19A003` 6/6, `MN9` 2/2, `PS059` 4/4, `GLNO` 4/4 -- identical cell counts). The remaining differences are
notational (`PS196_b` <-> `PS196b`, `PEN_a` <-> `PEN_a/PEN1`, `DNg02` split a-e in FAFB; MaleCNS `Tm5Y` / `TmY21` /
`LPi34` are Nern-2025 names FAFB does not use) and `flyverse/data/type_aliases.csv` already carries 22,200 alias rows
built from MaleCNS `flywireType` / Schlegel 2024 -- so a loader adapter is a name-normalisation, not a re-typing.

Key populations (cells MaleCNS / FAFB / BANC): DNa02 2/2/2, DNa01 2/2/2, PS059 4/4/4, LT51 22/22/23, LC11 143/127/141,
LC10a 275/237/224, LC4 126/104/114, LPLC2 185/210/181, EPG 46/47/45, Delta7 42/42/40, PEG 18/20/19, hDeltaB 19/18/20,
MBON01 2/2/2.

## 3. The round-1..3 turning anatomy, re-read on the female CNS

The claim flyverse rests on (docs/audits/deficit_turning.md, vnc_drive.md, body_sided_state.md) is anatomical
before it is dynamical: DNa02 sits under a large sign-correct inhibitory budget, its lateralised excitation arrives
through AN04B003 / LT51, and the haltere-side loop closes through PS196 -> PS059 -> DNa02. All of it is in BANC, with
the same top rows:

| edge | MaleCNS (signed counts, by side pre->post) | BANC (counts) | agrees |
|---|---|---|---|
| PS049 -> DNa02 (GABA) | -492 L->L, -528 R->R | 132, 139 | yes, ipsilateral, top inhibitory row in all three |
| PS059 -> DNa02 (GABA) | -476 L->L, -522 R->R | 104, 115 | yes |
| VES051 / LAL126 / AOTU019 -> DNa02 | -607 / -634 / -586 | 597 / (in top 12) / 257 | yes (VES051 is BANC's single largest input) |
| AN04B003 -> DNa02 (ACh) | +425 L->L, +339 R->R | 56 L->L, 133 R->R | present, ipsilateral; BANC asymmetric |
| IN12B014 -> DNa02 | -112 L->R, -112 R->L (one edge each) | 58 L->R, 59 R->L | yes: the symmetric contralateral pair the body-state skeptic flagged |
| LT51 -> DNa02 | -155 / -165 | 70 / 76 | yes |
| PS196a -> PS059 (ACh) | +331 L->R, +306 R->L | 245 L->R, 183 R->L | yes, contralateral in both |
| PS196b -> PS059 | 11-16 per pair, all four combinations | 3-6, three combinations | weak in both |
| GLNO -> PS196b; DNa02 -> AN04B003 | absent | absent | yes |

Excitation : inhibition on DNa02 by presynaptic transmitter -- MaleCNS ACh 31.7 k vs GABA + Glu 16.1 k (2.0 : 1),
FAFB 18.2 k vs 9.3 k (1.96 : 1), BANC 7.7 k vs 4.6 k (1.7 : 1). The female releases put DNa02 in the same
count-balanced position; the net-inhibited resting state flyverse finds is therefore a rate statement, not a
reconstruction artefact of the male graph -- which is what a reviewer will ask.

Haltere afferents in BANC: 466 cells labelled `Body Part = haltere` (295 sensory_ascending, 144 sensory, 27 motor),
with `Function` labels (proprioception, mechanical strain); their top targets are `AN08B010`, `IN08B008`, `w-cHIN`,
`IN08B093`, `IN06B017` -- VNC interneurons, not PS196 -- consistent with the k = 3 route vnc_drive.md section 6 describes.

## 4. Transmitters: two independent labellings of the sign-0 set

MaleCNS silences 3,312 sign-0 bodies (2,361 `unknown`, 415 serotonin, 395 dopamine, 141 octopamine; 530 types). Read
by type name into the female releases:

| MaleCNS label | FAFB `nt_type` (cells) | BANC verified (cells) |
|---|---|---|
| dopamine (395) | DA 372, GLUT 4, absent 17 | dopamine 365 -- **solid** |
| octopamine (141) | OCT 37, **GABA 18**, absent 78 | octopamine 82, `?` 24 |
| serotonin (415) | SER 72, **DA 68**, ACH 20, GLUT 12, absent 239 | serotonin 56, **tyramine 46**, glycine 8, `?` 140 -- **least corroborated** |
| unknown (2,361) | absent 2,135; ACH 10 / GABA 55 / GLUT 93 / DA 28 | absent 1,727; BANC *predicted* ACH 150 / GABA 148 / GLUT 111 |

So about 400 of the `unknown` cells MaleCNS silences carry a classical-transmitter prediction in BANC, and the
serotonin class is split between serotonin, dopamine and tyramine across sources. Both are direct inputs to
docs/NT_INTEGRATION.md (a fourth and fifth NT source) and to the monoamine thread's target-load table
(docs/audits/monoamine_slow_term.md).

Disagreements worth a row in the NT table: **PFL3** -- MaleCNS ACh 24/24, FAFB ACH 24/24, BANC *predicts* TYR 24/25
(and BANC has PFL2 *verified* tyramine 12/12); **Delta7** -- BANC verified `glutamate,serotonin` (co-transmission;
MaleCNS glutamate); **LAL074** (a PS059 input) -- MaleCNS / FAFB glutamate, BANC predicts SER on 2 of 4.
BANC's predictor over-calls dopamine relative to FAFB (8,072 vs 584 DA cells brain-wide; 4.4 % vs 0.5 % of synapses),
so BANC *predicted* monoamine labels should be used only where its *verified* column agrees.

Brain-wide synapse share by presynaptic transmitter: FAFB ACh .56 / GABA .22 / Glu .13 / DA .005 / 5-HT .007 / OA
.003 / unlabelled .07; MaleCNS (|W|) ACh .61 / GABA .21 / Glu .17 / His .006 with the monoamine classes explicit zeros.

## 5. What flyverse can do with it, ranked

1. **Cross-connectome anatomy table (cheap, CPU, a day).** A `scripts/cross_connectome.py` that takes the named
   edges and input budgets of every anatomical claim in rounds 1-3 (section 3 is the seed) and prints MaleCNS /
   FAFB / BANC side by side, with per-release scaling. Turns "in MaleCNS, DNa02 is ..." into a cross-animal,
   cross-sex, cross-lab statement for the README and the announcement. No model change.
2. **A second `Connectome` backend (`load(dataset="fafb" | "banc")`, ~2 days).** `root_id` fits int64
   (7.2e17 < 9.2e18, so it slots into `bodyId` unchanged); type names via `type_aliases.csv`; superclass / NT /
   side / neuropil vocabularies are small maps (`ACH` -> `acetylcholine`, `optic_lobe_intrinsic` -> `ol_intrinsic`,
   `left` -> `L`, ...); the CSR cache format is already dataset-agnostic (`connectome_fingerprint` records the source).
   `receptors_by_type.csv` and `expected_responses.csv` are keyed by type name and transfer through the aliases. The
   optic-lobe rate model and the motor readout are keyed by type name too. **BANC then gives the female CNS as a
   whole-animal replicate of the walking result** -- either the female walks straight for the same reason, or she does
   not, and both are findings. This is the biggest scientific payoff and the best release story ("runs on both public
   fly connectomes"), and it is exactly what the extensibility layer was built to allow.
3. **Optic-lobe ground truth.** FAFB `column_assignment` (45,528 cells, 31 types, hex coordinates) is an independent
   column map to test `trace.column_of_cells` and the anatomical LC windows the size-tuning ladders used; LC11 /
   LC10a themselves are not column-assigned (0 rows), which is the same gap the localizer round hit.
4. **NT sources 4 and 5** for docs/NT_INTEGRATION.md: FAFB's per-cell probabilities and BANC's verified column,
   with the section-4 disagreements as new conflict rows.
5. **Morphology** (the 13.9 GB skeleton archive) for the room UI / figures; irrelevant to dynamics.

Not to do: mix counts across releases without the per-release scale; take BANC optic-lobe counts as complete; take
BANC predicted monoamine labels over its verified ones; assume sex-shared circuits for `fru` / `dsx` populations
(the male graph has male-specific cells the female releases lack, and vice versa).
