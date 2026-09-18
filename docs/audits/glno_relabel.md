# GLNO -> glutamate: the evidence case, the adoption suite and the rotation input (thread 5B)

Generator: `scripts/glno_relabel.py` (`compile` / `compare` / `plan` / `analyse` / `smoke` on the CPU; `suite` / `compass` are
the cluster job wrappers). Data: `out/cache_glno_glu/` (the candidate cache, scratch), `out/cx5b/` (`predeclared.json`,
`tree_state.json`, `jobs.json`, `batch.sh`, `entry_compare.{json,md}`, `fam_suite/`, `fam_c<seed>/`, `analysis/`, `smoke/`),
console `out/cx5b_cluster.log`. Nothing is adopted here: `cache/`, `connectome.TYPE_NT_OVERRIDE`, the receptor table and every
default are untouched; the candidate lives in a scratch cache and this audit prepares the adoption case under the round-2 rule
(a relabel enters `TYPE_NT_OVERRIDE` only if the full benchmark suite changes no check's status in >= 3 draws, and only cells with
NO usable fast label are relabelled).

Read first: `docs/audits/cx_glno.md` (rounds 2-4 on the GLNO sign in the ring; section 1 for the structure, section 3 for the
round-2 finding that a glutamatergic / GABAergic GLNO abolishes the bump at gE 1.75 / gD 15 in 3/3 seeds (the only six-seed row
at gE 1.75 is gD 8, where `gaba` is confined in 4/6 and persists in 3/6) and costs ~10 % of the bump at gE 2 / gD 15, section 5 for the three operating points robust to the sign), `docs/NT_INTEGRATION.md` (round 2 item 4, the
round-5 compass items), `docs/audits/level_matched_control.md` section 5 (round 4's efferent compass: GLNO L-R fixed at +26.9..+28.1
Hz in every phase of every arm, PEN / EPG flips <= 0.57 Hz, drift inside -0.0087..+0.0094 wedges/s).

## 1. The evidence case

### 1.1 What GLNO is in MaleCNS

4 cells, `GLNO(LAL-NO1)_L` x2 (bodyIds 12104, 23325) and `GLNO(LAL-NO1)_R` x2 (14881, 25939); class CX, superclass
cb_intrinsic; `nt` `unknown` in every MaleCNS column, so `sign` 0 under `NT_SIGN` and its 17,698 raw output synapses are stored
as explicit zeros in W (213 entries). 92.5 % of that output is PEN: 84 GLNO -> PEN entries, 16,371 raw synapses = 19.4 % of PEN's
raw input, every edge 88-345 synapses (mean 195, median 182) -- i.e. every one of the 84 above the connection cap 60 -- and exactly
2 GLNO inputs per PEN (42/42 PEN; contralateral: GLNO_L -> the R-glomerulus PENs, GLNO_R -> the L-glomerulus PENs,
`cx_glno.md` 1). GLNO's own input is 37 % PEN (84 entries, 3,496 raw synapses, ACh +1, capped: +4.1..+16.5 mV per PEN spike).
No expression source profiles it (section 1.3), and the receptor table has no row for it (tier `fallback` on its 646 input
entries, `cx_glno.md` section 0).

### 1.2 The sources, in the `TYPE_NT_OVERRIDE` entry format

The entry the adoption would add to `flyverse/connectome.py` (NOT added; the text is the evidence case in the format of the three
existing entries -- each names its cells, the sign-0 share it relabels and every source with its number):

```python
    # 4 cells, all 'unknown' (17,698 raw output synapses, 16,371 of them the 84 GLNO -> PEN edges = 19.4 % of PEN's raw input,
    # every edge above the connection cap; PEN's largest single unitary input once signed). EM classifiers only, no
    # transcriptome profile in any of the six sources, no EASI-FISH: MaleCNS v1.0 T-bar classifier glutamate 51 % /
    # acetylcholine 37 % / serotonin 7 % over 783 T-bars (consensus, cell-type and per-body columns all 'unclear' at
    # confidence 0.48; docs/audits/nt_audit.md row GLNO); BANC v888 'Predicted NT type' GLUT on 4/4 cells at confidence
    # 0.47 / 0.48 / 0.50 / 0.50 ('Verified NT type' empty; the banc backend takes the prediction without a threshold);
    # FlyWire v783 per-cell top_nt gaba x3 (conf 0.30-0.33) / glutamate x1 (0.30), below the fafb backend's 0.5 threshold
    # so 'unknown' 4/4 in the FAFB cache. Both female datasets and MaleCNS agree on INHIBITORY and disagree on which
    # inhibitory transmitter; under NT_SIGN glutamate and gaba are the same fast -1, so the W this entry produces is
    # bit-identical to {GLNO: gaba} (docs/audits/glno_relabel.md 1.4). Status: prepared, not adopted (section 4).
    "GLNO": "glutamate",
```

The sources one by one, with what each is and is not:

| # | source | what it says about GLNO | weight |
|---|---|---|---|
| 1 | MaleCNS v1.0 T-bar classifier (`docs/audits/nt_audit.md` rows GLNO, sections 1 and 1d) | 783 T-bars: glutamate 51 % / acetylcholine 37 % / serotonin 7 %; every consensus column `unclear` at confidence 0.48 -- below the release's own acceptance, hence `unknown` in the cache | one EM classifier; majority glutamate but 37 % ACh, i.e. the sign itself is the uncertain part |
| 2 | the name (Hulse et al. 2021 eLife 10:e66039; Scheffer et al. 2020 eLife 9:e57443; Wolff & Rubin 2018 J Comp Neurol 526:2585) | The task's premise was that Hulse et al. 2021 named the LAL-NO1 pair for its predicted transmitter. I could not verify that. Scheffer et al. 2020 (the hemibrain nomenclature) says only: "the nodulus neurons are now 'LNO' and 'GLNO' instead of 'LN' and 'GLN'" (to resolve the 'LN' name clash with clock and antennal-lobe neurons) and gives no transmitter; the accessible text of Hulse et al. 2021 lists GLNO (4 cells) in its Table 4 of NO input types with no transmitter statement; Wolff & Rubin 2018 is the source of the light-level 'LAL-NO1' name and its abbreviation table. No text I could read defines the 'G' or attaches a predicted transmitter to it | **none** -- an unverified reading of a name is not evidence; recorded here so it is not counted again |
| 3 | BANC v888 (`connectome.load(dataset='banc')`, `cache/banc/neurons.parquet`; raw `neurons.csv.gz` columns 'Predicted NT type' / 'Predicted NT confidence' / 'Verified NT type') | GLUT on 4/4 GLNO cells (root ids 720575941508852318, 720575941539276181, 720575941576317110, 720575941672018087) at confidence 0.48 / 0.50 / 0.50 / 0.47; 'Verified NT type' empty. The banc backend's rule is "verified-first; else the prediction" with NO confidence threshold (`flyverse/backends/banc.py` `verified_nt`; the `nt_threshold` argument is recorded in the manifest and not applied), so the cache's `glutamate` is four argmax calls at ~0.5, not four confident ones. For contrast the sister type LNO1 is `gaba` verified by EASI-FISH in the same file (conf 0.75-0.80) | a second, independent EM classifier (a different volume, a different classifier), same majority call, same low confidence |
| 4 | FlyWire / FAFB v783 (`data/external/typing/schlegel2024_Supplemental_file1_neuron_annotations.tsv`; `cache/fafb`) | per-cell `top_nt` gaba x3 (conf 0.304 / 0.330 / 0.333) / glutamate x1 (0.299); `known_nt` empty; the fafb backend thresholds at 0.5 so the cache says `unknown` 4/4. LNO1 there: gaba verified (Wolff et al. 2024 EASI-FISH), `top_nt` gaba x3 / acetylcholine x1 | a third EM classifier: inhibitory on 4/4, split GABA / glutamate, weakest confidences of the three |
| 5 | transcriptome / expression (`flyverse/data/expression_{central,davis2020,kurmangaliyev2020,ozel2021,ozel2021_mm}.csv`, `nt_by_type_transcriptome.csv`, `type_map_{central,davis2020,kurmangaliyev2020,nern2025,ozel2021}.csv`, `receptors_by_type.csv`) | no row matches GLNO, LNO1, LAL-NO1 or LAL_NO in any of them (grep, 0 hits in all 11 files). The only tables that carry the name are the typing maps: `type_map_typing.csv` / `type_aliases.csv` / `type_aliases_typing.csv` / `type_aliases_nern2025.csv` say GLNO = FlyWire `GLNO` (4 cells) = hemibrain `GLNO` (4 bodies), class CX -- an identity, not a profile. `LNO1` is a different hemibrain / FlyWire type (also 4 cells) and is not GLNO's alias; the MaleCNS instance `GLNO(LAL-NO1)` carries Wolff & Rubin 2018's light-level name | nothing: no transcriptome source, no Nern 2025 prediction, no EASI-FISH |

So the case rests on three EM classifiers that agree GLNO is inhibitory and disagree on the transmitter (MaleCNS glutamate 51 %,
BANC glutamate 4/4 at ~0.5, FlyWire GABA 3/4 at ~0.3), against nothing on the excitatory side except MaleCNS's 37 % ACh minority.
Under `NT_SIGN` the two inhibitory labels are the same fast -1; they differ only in the receptor model's column (section 1.4) and
the slow term (none for either). The existing `TYPE_NT_OVERRIDE` entries each have at least one transcriptome source or an
EASI-FISH validation; this one has none, which is why rounds 2-4 left GLNO out. What the rule can still decide is the suite half
(section 3).

### 1.3 The structural consequence (CPU, `scripts/glno_relabel.py compare`; `out/cx5b/entry_compare.{json,md}`)

The candidate cache `out/cache_glno_glu/` was compiled from the raw MaleCNS files with `connectome.TYPE_NT_OVERRIDE` + {GLNO:
glutamate} (`connectome.load(cache_dir=..., rebuild=True, type_nt_override=table)`, cx_wedge's temp-dir-then-rename pattern, plus
`build_sign0_counts` so the cache carries the same files as the shipped one) and compared with `cache/` entry by entry:

* neuron table: same 167,106 bodyIds in the same order, same types; `nt` differs on exactly 4 cells and `sign` on exactly 4
  cells, all GLNO (`unknown` / 0 -> `glutamate` / -1). nt counts: glutamate 29,707 -> 29,711, unknown 2,361 -> 2,357.
* W: same shape, same nnz 25,578,600, same sparsity pattern (`indptr` and `indices` equal); **213 entries differ, all 213 with a
  GLNO presynaptic cell, all 0 in the shipped cache, all negative in the candidate, and |candidate| equals the shipped
  `sign0_counts.npz` value on every one of them** (the explicit-zero entries' raw counts). GLNO has 213 presynaptic entries in
  total, 0 of them unchanged. sum|W| 121,460,584 -> 121,478,280 (+17,698 = GLNO's raw output); sign-0 counts 2,701,289 ->
  2,683,591 (-17,698). W md5 `ef23cc27` -> `7a10d93b`.
* **The candidate's W is bit-identical to the round-3/4 `gaba` scratch cache** (`out/cache_72164311`, md5 `7a10d93b`, sum|W|
  121,478,280): under `NT_SIGN` glutamate and gaba are the same -1, so the six-seed gain grid of `cx_glno.md` section 5 (both
  GLNO conditions, `receptor_model=None`) is already this candidate's ring dynamics -- the bump survives at gE 2 / gD 15,
  2.25 / 25 and 2.5 / 25 with -9 %, -9 %, -6.5 % of its rate, PEN -14..-16 %, and is lost at gE 1.75 / gD >= 15.

Where the 17,698 synapses go (entries / raw synapses / share of GLNO's output):

| post type | entries | raw syn | share |
|---|---|---|---|
| PEN_a(PEN1) | 40 | 9,806 | 55.4 % |
| PEN_b(PEN2) | 44 | 6,565 | 37.1 % |
| ExR8 | 8 | 435 | 2.5 % |
| GLNO (self / pair) | 4 | 400 | 2.3 % |
| FB4Y | 16 | 218 | 1.2 % |
| FB1C | 14 | 114 | 0.6 % |
| FB4M | 11 | 21 | 0.1 % |
| FB1H 2 / 21, LNO2 3 / 11, AN27X013 2 / 8, PFNa 6 / 8, PVLP060 1 / 6, PFNd 4 / 6, LCNOpm 3 / 6, DNpe023 2 / 5, and 43 further types at 1-4 synapses (EPG 2 / 2, PEG 2 / 2, PS196_b 2 / 3, ER6 1 / 2, ExR1/2/4/6 1-2 each ...) | 76 | 139 | 0.8 % |

(58 postsynaptic types in all; the columns now sum to 213 entries and 17,698 synapses. The 8 named types are 23 entries / 71
synapses; the 43 others are 53 entries / 68 synapses.)

The PEN edges after Brain's connection rules (`LIFParams` defaults, w_syn 0.275 mV, conn_cap 60; every PEN's capped raw fan-in
rises by exactly +120 = 2 x cap, 867-1597.5 -> 987-1717.5, which leaves all 42 far below `input_norm_ref` 5000, so the fan-in
scale stays 1.00 in both caches): **-16.50 mV per GLNO spike on every one of the 84 edges** (shipped 0),
i.e. -33.0 mV per PEN per volley of its two contralateral GLNO, against +5.05 per EPG -> PEN pair (`cx_glno.md` 1). The return leg
PEN -> GLNO is unchanged (+3,496 raw, +4.1..+16.5 mV per PEN spike, GLNO fan-in scale 1.00 in both), so the loop
**PEN -(ACh, +)-> GLNO -(glutamate, -)-> PEN** closes as negative feedback within one PB side (GLNO_L reads only R-glomerulus PENs
and writes only R-glomerulus PENs), with GLNO -> GLNO (400 syn) self- / pair-inhibition.

Beyond the 213 entries the effective-weight matrix A (mV) moves in one more place: 20 postsynaptic cells whose raw fan-in total
crosses `input_norm_ref` once GLNO's synapses count (FB4Y x4, LNO2 x2, LCNOpm x2, and one each of OA-VUMa1, ExR1, ExR4, ExR6,
DNbe003, DNpe023, FB4C, LAL073, LAL084, AVLP449, DNa13, FB1H) get a slightly smaller fan-in scale on ALL their inputs -- 14,828
entries in A differ, 213 of them GLNO's; scale ratios candidate / shipped 0.988-1.000 (the largest on the four FB4Y cells, which take 48-60 of GLNO's synapses each on a raw fan-in of ~5,000-5,500), and the largest change on any non-GLNO entry of A is 0.19 mV per spike. The 213 GLNO entries are the relabel; the 14,615 others are the fan-in normalisation's arithmetic on those 20 cells, all below 1.2 %.

### 1.4 The receptor model on the candidate's entries

The suite and the compass run under the shipped `LIFParams` (`receptor_model 'sign'`, net rule `abs`), which looks every entry
up by (postsynaptic type, presynaptic transmitter) -- so a `glutamate` label could in principle differ from a `gaba` one where a
postsynaptic type's table row calls glutamate excitatory. Checked on the candidate (`connectome.receptor_signs(c, net_rule)` on the
213 entries): **all 213 are -1 under `abs` and under `class`** (PEN_a / PEN_b tier `fuzzy`, FB4Y / FB1C tier `class`, ExR8 / GLNO
tier `fallback`; 0 entries +1, 0 entries 0). The runs below therefore install exactly `NT_SIGN`'s -1 on every GLNO synapse: under the
shipped `receptor_model 'sign'` the two labels resolve to the same `fast_sign` on all 25,578,600 entries, not only the 213, so
the suite and compass arms are the gaba cache's model exactly. The table is not blind to the difference, though: under
`sign+gain` or `full` the resolved gain and slow columns differ on the same entries -- `fast_gain` on 35 of the 213 (FB1C 14,
FB4M 11, FB2A 4, ExR2 2, FB1H 2, ER4m 1, OA-VUMa1 1: glutamate `high` x1.5 against gaba `mid` x1.0), `slow_sign` on 3 (EPG x2,
ER4m: glutamate 0 against gaba -1) and `slow_gain` on 57. Onto PEN the two labels are identical under every mode (the
`PEN_a(PEN1)` / `PEN_b(PEN2)` glutamate and gaba rows carry the same fast sign, the same `high` gain class and the same slow
columns; only the lead gene differs, GluClalpha against Rdl), which is why the ring result is label-free. The table already
holds 160 types with an excitatory glutamate `fast_sign` under the class rule (14 under `abs`) and four FB targets whose
glutamate net is `mixed`; none of them is a GLNO target, which is the actual reason the 213 come out -1. Neither label has a slow term (`SLOW_CLASS_OF_TRANSMITTER` is 0 for both), so the
candidate is fully described by section 1.3.

## 2. Design (predeclared; `out/cx5b/predeclared.json`, stamped 2026-09-15T05:35:08Z -- the same stamp `submit_tree.txt`, `tree_state.json` and `jobs.json` carry; the earliest scheduler evidence is the first job's `started_utc` 05:35:33Z, so the predeclaration precedes every run. This batch has no `scheduler_receipt.json`; `out/cx5b_cluster.log` records elapsed seconds, not wall clock.)

`python scripts/glno_relabel.py plan --dir out/cx5b --name cx5b --minutes 45` wrote `predeclared.json` (the questions, the
adoption rule in one direction, the primary families with their tests and Holm sizes, the expectations written before the runs,
the provenance every run must carry), `tree_state.json` (HEAD `0b3668f`, the porcelain status, sha256 of every `flyverse/*.py`,
`flyverse/interp/*.py`, `flyverse/backends/*.py` and the seven scripts the jobs load, and the shipped `scripts/probe_vnc_drive.py`
against HEAD's), `probe_diff_vs_HEAD.txt` (419 lines), `jobs.json` and `batch.sh`. ONE `cluster_run.py` submission, `--arm-block
fam`, house target (B200), 22 jobs:

| block | jobs | what |
|---|---|---|
| `fam_suite` | 6 | `glno_relabel.py suite --arm shipped\|glu --draw 1..3`: `scripts/benchmark.py --sections all --seeds 0,1,2` (the 29-check suite, native backend) through `benchmark.main()` with the loaded connectome captured; the `glu` arm passes `--cache-dir out/cache_glno_glu` (compiled on the box by the first glu job under a lock file, the others wait); `provenance` + a `glno_relabel` block appended, exit 3 if the cache the run used is not the arm's or the device is not a GPU |
| `fam_c0..3` | 4 x 4 | `glno_relabel.py compass --cache shipped\|glu --gains exp\|shipped --seed s`: `scripts/probe_vnc_drive.py compass --family level --arm C` (proprioception `all+leg_cycle`, the family that carries the leg cycle), `--gains 2:15` (the labelled instrument: 10 Hz EPG background, a 50 Hz pulse on wedges 0-3 for 2 s, compass adaptation 0; the only configuration with a bump to drift) or `--gains none` (the shipped path: no background, no pulse, adaptation as shipped), `--cache-dir` for the glu cache, DNa02_L / _R at 20 Hz for 10 s with 3 s skipped in each of the phases rest / ccw / rest2 / cw, `--sparse warp`; the same block appended to `<stem>_run.json` |

The rule and the families, as declared: **adoptable under the round-2 rule iff no check has a status in any glutamate draw that is
not the status of every shipped draw** (a check unstable among the shipped draws is reported, not counted); the draw-matched
reading and the 3 v 3 `common.compare` on the measured values are reported beside it (underpowered by construction). Compass
primaries, `common.compare` over runs (4 v 4, exact-U floor 0.029), Holm within: F1 glu vs shipped at the experiment gains {GLNO
flip, PEN_a flip, PEN_b flip, bump drift ccw - cw} (m 4); F2 the same at the shipped gains without the drift (m 3); F3 the signed
report on the glu / experiment-gains arm {GLNO, PEN_a, PEN_b flip vs its own null rest2 - rest} (m 3) -- "the signed self-turn
report PASSES GLNO iff the GLNO member is called"; F4 the same on the glu / shipped-gains arm (m 3); F5 drift on the glu /
experiment-gains arm {ccw vs the 8 rests, cw vs the 8 rests} (m 2; ideal +-4.0 wedges/s). Secondary, reported not called: the same
rows on the shipped arms (round 4 re-run), PS196_b / AN04B003 / EPG / DNa02 flips, chain rates, GLNO L-R per phase. Expectations
written before the runs: suite unknown (the compass gap row the one to watch); compass prior = GLNO flip still null, PEN flips
still null, drift still ~0 (rounds 2-4: nothing moves the bump; PS196_b's input to GLNO is symmetric).

The compass jobs ran the **working-tree** `scripts/probe_vnc_drive.py` (sha256 `dbd8b115...`; HEAD's is `1912dcfa...`), called as
it is with its own argv through `runpy` -- nothing in it was edited by this thread. `cluster_run.py` ships every file that differs
from origin/main (`git ls-files -m -o --exclude-standard`) and git-ignored paths cannot be shipped, so the batch necessarily
carries the other threads' uncommitted edits at submission (`out/cx5b/submit_tree.txt`: `flyverse/body.py`, `senses.py`,
`scripts/cx_wedge.py`, `probe_vnc_drive.py`, `tests/`; the diff of the probe against HEAD is `probe_diff_vs_HEAD.txt`). Those
edits are opt-in tokens (`unsided`, `leg_cycle_flat`, `flat_amplitude`, a `level2` family) and provenance additions (`sense`
record, `started_utc`, `mn_ref_hz` at the top level of the run JSON -- the round-4 defect this thread was told to check); arm C of
family `level` is `all+leg_cycle` in both versions (`FAMILIES['level']` unchanged; import check on the shipped tree before
submission). The independent check is that the only behavioural lines the diffs add on arm C's path are guarded off:
`sense_kwargs_of("C", "level", args)` returns `{}` (the `ARM_MN_REF` table has only the `L` key and `ARM_SENSE_KW` only
`level2/K`), `attach_cycle` builds the default `body.LegCycle()` because `leg_cycle_flat` is off, `senses.py`'s one new branch
is `if self.unsided:` (refused together with `leg_cycle`) and `body.py`'s is `if self.flat_amplitude:` on a default-False
field; every run JSON's `sense.tokens` records `unsided false, leg_cycle_flat false, haltere_sided false` and
`leg_cycle_params.flat_amplitude false`. `docs/INTERP.md` 10.1(7)(b) requires a cross-task dependency to be COMMITTED before
submission; it was not at 05:35:08Z. The owning thread has since committed exactly these files as `286dca3` ("Round 4b: the
three level controls ... and the AN04B003 single-cell check"), and `scripts/probe_vnc_drive.py` there still hashes
`dbd8b115...` -- the file the batch ran. Cite `286dca3` as the code identity of the compass numbers and say the rule was met
after the fact. CPU smokes before submission (`glno_relabel.py smoke`, `out/cx5b/smoke/`): the compass wrapper on the shipped cache
at the experiment gains and on the candidate at the shipped gains (`--quick --device cpu`), both exit 0, `problems []`, the right
md5 / GLNO label / gains / spec in each run JSON.

## 3. The runs (batch `cx5b-d08be3`, run dir `<cluster-fs>/neurome/runs/cx5b-d08be3`)

Submitted 2026-09-15 05:35 UTC, **`22 job(s), 0 failed (18.5 min)`**, fetched to `out/cx5b/` (6 suite JSONs + logs, 16 compass
`_run.json` + 64 phase JSONs + 128 npz + 16 logs; the client stayed attached and the wait was `box_status.py --target house
--prefix cx5b --wait` in the foreground: 22 running at 22:35, 13 done at 22:44, 22 done at 22:47 local). Every one of the 22 JSONs
records **NVIDIA B200**, `problems []`, and the cache it was asked for: the three shipped suite draws and the eight shipped-cache
compass runs `compiled_connectome.md5 ef23cc27` with GLNO `unknown` / sign 0 read back from the cache's `neurons.parquet`; the
three glu suite draws and the eight glu compass runs `7a10d93b` with GLNO `glutamate` / -1 (`cache_state` in the `glno_relabel`
block; `provenance.compiled_connectome.cache_dir` = `out/cache_glno_glu`, compiled once on the box by the first glu job; 12 local
files shipped). Suite: native backend, receptor `sign` / `abs`, 29 checks in all six, 481-532 s each. Compass: gains `[2, 15]`
with background 10 Hz / pulse 50 Hz / adaptation 0 in the eight `exp` runs, `None` / 0 / 0 / shipped in the eight `shipped`-gains
runs; `proprioception all+leg_cycle`, `leg_cycle true`, `mn_ref_hz 30.0` (the sense's default, recorded this time -- the round-4
JSONs had `null`), block `fam_c<seed>`; 92-591 s per run (the spread is the 22 jobs sharing one box, not the model).

## 4. Analysis (CPU, `scripts/glno_relabel.py analyse`; `out/cx5b/analysis/analysis.md`, `summary.json`, `suite_table.csv`, `decision_table.csv`, `compass_flip_<arm>.csv`, `compass_chain_<arm>.csv`)

### 4.1 The suite: 27 PASS / 0 FAIL / 2 KNOWN GAP in every draw of both arms; no check changes status

| check (criterion) | shipped d1 / d2 / d3 | glutamate d1 / d2 / d3 | status | value moved? |
|---|---|---|---|---|
| rest.spikes_per_step (< 5) | 0 x3 | 0 x3 | PASS x6 | no |
| taste.MN9_hz (> 2) | 10.93 x3 | 10.93 x3 | PASS x6 | no |
| smell.PN_hz (< 100) / smell.KC_active (> 0) | 7.861 / 816 x3 | 7.861 / 816 x3 | PASS x6 | no |
| dn.DNa02_L_leg_asym_hz (> 0.3) | 2.581 x3 | 2.581 x3 | PASS x6 | no |
| dn.MDN_top_hz / dn.DNp09_top_hz (< 250) | 153 / 152 x3 | 153 / 152 x3 | PASS x6 | no |
| walk.GF_max_hz (< 38) | 4.629 x3 | 4.629 x3 | PASS x6 | no |
| walk.power_max_hz (reported, notnone) | 48.48 x3 | 48.48 x3 | PASS x6 | no |
| walk.power_sustained_hz (< 50) | 20.11 x3 | 20.11 x3 | PASS x6 | no |
| loom.GF_peak_hz (>= 20) | 47.22 / 47.22 / 47.20 | 43.59 / 47.22 / 47.20 | PASS x6 | d1 only (43.6 is inside round 6's 43.6-47.2 range for the shipped default) |
| loom.escape_cm (reported) | 3.5 x3 | 3.5 x3 | PASS x6 | no |
| rotate.DNp20_flip_hz (< -2) | -32.4 / -45.6 / -36.1 | -28.0 / -36.5 / -30.5 | PASS x6 | inside scatter (diff +6.4, p 0.4) |
| motion.min_dsi (>= 0.1) | 0.24110 x3 | 0.24064 x3 | PASS x6 | yes: -0.00046 (-0.19 %); scatter ~2e-7 within each arm |
| motion.correct_directions (== 8) | 8 x3 | 8 x3 | PASS x6 | no |
| loom_escape.GF_peak_hz (>= 33) | 48.2 / 45.6 / 46.0 | 57.9 / 52.3 / 48.1 | PASS x6 | higher in 3/3 (diff +6.2, z 4.4, p 0.2: 3 v 3) |
| loom_escape.escapes (>= 1) | 3 x3 | 3 x3 | PASS x6 | no |
| walk_gf.p99_hz (< 38) | 22.6 / 19.2 / 14.8 | 23.0 / 16.0 / 20.5 | PASS x6 | inside scatter |
| rotation.group_flip_hz (<= -3) | -8.2 / -10.0 / -9.4 | -9.9 / -10.3 / -9.0 | PASS x6 | inside scatter |
| object.LC10a_flip_hz (abs >= 1) | 0.002 / 0.013 / -0.006 | 0.004 / -0.004 / -0.001 | KNOWN GAP x6 | inside scatter |
| bitter.calibrated_sugar_MN9_hz (> 2) / calibrated_sugar_bitter (< 1) | 5.518 / 0 x3 | 5.518 / 0 x3 | PASS x6 | no |
| bitter.shiu_sugar_MN9_hz (> 50) | 139.90 x3 | 135.15 x3 | PASS x6 | yes, deterministic: -4.75 Hz (-3.4 %) |
| bitter.shiu_sugar_bitter_MN9_hz (< 10) | 0.818 x3 | 0.988 x3 | PASS x6 | yes, deterministic: +0.17 Hz |
| wind.DNp18_flip_hz (>= 15) | 45.0 / 45.4 / 45.4 | 46.2 / 45.6 / 46.7 | PASS x6 | +0.9 (z 3.7, p 0.1: 3 v 3) |
| wind.DNp33_flip_hz (<= -15) | -49.7 / -50.4 / -50.1 | -50.4 / -49.6 / -49.9 | PASS x6 | inside scatter |
| odour.apple_channel_8cm_hz (>= 10) / clean (<= 6) | 17.5 / 4.4-4.5 | 17.4-17.6 / 4.2-4.6 | PASS x6 | inside scatter |
| compass.wedge_cells_persisting (>= 6) | 0 x3 | 0 x3 | KNOWN GAP x6 | no |

`summary.json`: `changed_pooled []`, `changed_matched []`, `unstable_shipped []`, `problems []` -> **ADOPTABLE by the suite half
of the round-2 rule** (3 + 3 draws, 174 check rows, one submission, one device). What the values say beyond the statuses: 16 of
the 29 checks read the same value in all six draws (the deterministic protocol sections: rest, taste, smell, dn, the walk
triple, loom.escape, motion.correct_directions, loom_escape.escapes, calibrated bitter, compass). "Identical to the printed
digits on this GPU batch" is what that means: under the project rule (`guard_suites_r3.md` 5) bit-identity is a CPU claim and
nothing here is one. Three checks move by a fixed amount in 3/3 draws -- `bitter.shiu_sugar_MN9` 139.8985 -> 135.1501 Hz and
`bitter.shiu_sugar_bitter_MN9` 0.81784 -> 0.98795 Hz, both constant within each arm, and `motion.min_dsi` 0.2410966 ->
0.2406374 (-0.19 %), whose within-arm scatter is ~2e-7, i.e. 2,500x smaller than the shift but not zero -- which is the
relabel reaching the optic-lobe / gustatory readouts through
GLNO's FB / LAL targets and the 20 renormalised cells, small and far from any bound; the stochastic sections are inside their
own scatter except `loom_escape.GF_peak` (higher in 3/3, +6.2 Hz, still far above 33) and `wind.DNp18_flip` (+0.9 Hz, 3/3). At 3
v 3 every `compare` verdict is `underpowered` by construction (floor p 0.10); these are directions on record, not results. The
compass gap row (`wedge_cells_persisting` 0) is the suite's shipped-gains protocol, where no bump forms with either cache.

**Not run here: the room take-off protocol** (the rate-half of the rule as round 7 sharpened it, `docs/audits/guard_suites_r3.md`
4; >= 6 runs per arm for a callable verdict). The verdict above is the suite half only, and says so.

### 4.2 The efferent compass: nothing moves, and the signed report still stops at AN04B003

Per-run scatter for every row is in `analysis.md` 2.1-2.4 (flip rows for 12 types x 4 arms, chain rates per phase, drift per
phase, the families); the essentials:

**Rates at rest at the experiment gains (mean of L and R over 4 runs, Hz).** shipped -> glutamate:
GLNO 133.2 -> 116.0 (-13 %), PEN_a 44.9 -> 38.8, PEN_b 50.7 -> 42.0 (PEN -15 %), EPG 55.4 -> 50.8 (-8 %; the wedge peak 252 -> 253,
vs 0.76-0.77 in both), Delta7 100.0 -> 95.0 (-5 %), PEG 33.3 -> 29.1, **FB4Y 84.2 -> 57.7 (-31 %)**, **FB1C 19.1 -> 6.2 (-68 %)**,
ExR8 0.4 -> 0.0 -- ExR8 is the one type the relabel silences outright: 0.00 Hz on both cells in all four runs (max single cell
0.00), against 0.86 / 0.00 Hz and a 2.14 Hz max cell on the shipped cache. It takes 435 of GLNO's synapses on 8 entries, GLNO's
third-largest target. The ring numbers reproduce cx_glno.md section 5's six-seed 2/15 figures (PEN -14..-16 %, Delta7 -4..-6 %,
GLNO -12..-14 %) on a different protocol (10 s, 50 Hz pulse, senses attached, receptor model on), and the two FB targets are the
relabel's largest effect outside the ring: 218 / 114 raw synapses on 16 / 14 capped entries from a GLNO at 116 Hz. At the
shipped gains the whole ring sits at 0.3-3.5 Hz with either cache (no bump: vs 0.10-0.25, peak 1-3 Hz), and GLNO at 3.4-3.5 Hz.

**GLNO's fixed L-R.** Round 4's +26.9..+28.1 Hz in every phase of every arm is reproduced on the shipped cache here (+26.89 /
+27.14 / +26.89 / +26.96 at rest / ccw / rest2 / cw; GLNO_L 146.6, GLNO_R 119.7 Hz). Under the relabel it is **+3.12 / +4.14 /
+4.48 / +3.30 Hz** (117.6 / 114.5): the asymmetry is 88 % smaller. It is not a turn signal in either cache -- it does not change
sign with the turn direction -- and its origin is the bump's position (the pulse sits on wedges 0-3, so the R-glomerulus PENs
that GLNO_L reads fire more: PEN_b L 39.6 / R 61.8 shipped); with GLNO signed, the PEN -> GLNO -> PEN loop and GLNO -> GLNO
(400 syn) compress it. At the shipped gains GLNO L-R is -0.14..+0.07 (glu) and -0.21..+0.18 (shipped).

**The flips (L-R at ccw minus cw against the run's own null rest2 - rest; 4 v 4).** Every predeclared member is `null` and none
is called (`decision_table.csv`):

| family | member | glutamate (4 runs) | reference (4 runs) | diff | z | p | p_holm |
|---|---|---|---|---|---|---|---|
| F1 glu vs shipped, gE 2 / gD 15 | GLNO flip | +0.64, +2.00, +0.29, +0.43 | +0.00, +0.07, +0.64, +0.00 | +0.66 | 2.1 | 0.20 | 0.60 |
| | PEN_a flip | -0.16, -0.94, -0.51, +0.01 | +0.17, +0.04, -0.30, +0.06 | -0.39 | -1.9 | 0.11 | 0.46 |
| | PEN_b flip | -0.49, -1.09, -0.34, -0.06 | -0.27, -0.13, -0.16, +0.33 | -0.44 | -1.7 | 0.20 | 0.60 |
| | drift ccw - cw (w/s) | -0.005, -0.006, +0.007, -0.004 | -0.003, -0.007, +0.000, -0.014 | +0.004 | 0.6 | 0.69 | 0.69 |
| F2 glu vs shipped, shipped gains | GLNO / PEN_a / PEN_b flip | diffs -0.27 / -0.05 / +0.02 | | | -0.5 / -0.5 / 0.3 | 0.89 / 0.69 / 0.69 | 1 / 1 / 1 |
| F3 signed report, glu at 2 / 15 | GLNO flip vs null | +0.64, +2.00, +0.29, +0.43 | +1.00, -0.29, +1.50, +3.21 | -0.52 | -0.4 | 0.69 | 1 |
| | PEN_a flip vs null | -0.16, -0.94, -0.51, +0.01 | -0.44, +0.21, -0.37, -1.04 | +0.01 | 0.0 | 1 | 1 |
| | PEN_b flip vs null | -0.49, -1.09, -0.34, -0.06 | -0.29, +0.26, -0.57, -1.14 | -0.06 | -0.1 | 1 | 1 |
| F4 signed report, glu at shipped gains | GLNO / PEN_a / PEN_b flip vs null | diffs -0.25 / -0.01 / -0.03 | | | -1.1 / -0.1 / -0.2 | 0.20 / 1 / 1 | 0.60 / 1 / 1 |
| F5 drift, glu at 2 / 15 | ccw vs rests / cw vs rests (w/s) | +0.002 / +0.004 vs ideal +-4.0 | | | 0.2 / 0.5 | 0.57 / 0.37 | 0.74 / 0.74 |

Secondary rows: **AN04B003 flips -11.2 +- 0.8 (glu, 2/15) / -11.8 +- 1.1 (glu, shipped gains) / -11.5 +- 0.9 / -11.5 +- 0.5
(shipped caches), `result` (z -10 to -17, p 0.0286 = the floor) in all four arms** -- round 4's depth-1 report (-10.5 +- 1.1
under C) reproduced and unchanged by the relabel; **PS196_b `null` in all four arms** (flips -0.6 +- 1.9 / -0.8 +- 0.9 / -0.9 +- 0.5 /
-1.5 +- 1.3 Hz, the same sign in both turn directions as in `vnc_drive.md` 6); EPG `null` (<= 0.45 Hz); DNa02 +37..+38 Hz `result`
(the stimulus itself). Bump drift at the experiment gains: glutamate rest / ccw / rest2 / cw +0.0015 / +0.0022 / -0.0004 /
+0.0044 wedges/s (every run inside -0.0088..+0.0137), shipped -0.0009 / -0.0011 / +0.0013 / +0.0050 (inside -0.0082..+0.0071),
against the ideal +-4.0; the heading itself turns +92..+106 / -79..-99 deg/s, so the body is turning and the ring does not know.

**Answers.** (i) *Does the signed self-turn report now pass GLNO?* **No.** With GLNO glutamatergic the report still ends at
AN04B003 (depth 1, -11 Hz, result in 4/4 arms) and is `null` at PS196_b, GLNO, PEN_a, PEN_b and EPG at both gain settings; the
GLNO flip on the glu arm is +0.84 +- 0.79 Hz against a null of +1.36 +- 1.45 (z -0.4). Signing GLNO closes the PEN <-> GLNO loop
but does not sign PS196_b's input to it, which is symmetric in both turn directions -- the round-5 diagnosis stands: the missing
piece is upstream of GLNO (a sided PS196_b), not GLNO's transmitter. (ii) *Does it move PEN or the bump?* PEN's rate, yes (-15 %,
as in the ring grid); PEN's flip and the bump, no (F1 all `null`; drift +0.002 / +0.004 w/s vs 4.0). (iii) The relabel's visible
dynamics are rate offsets: the ring -5..-15 %, GLNO's own L-R asymmetry -88 %, FB4Y -31 %, FB1C -68 %.

### 4.3 Is GLNO=glutamate adoptable under the round-2 rule?

**By the suite half, yes: 3 shipped and 3 glutamate draws in one submission, 27 / 0 / 2 in all six, no check changes status,
no shipped check unstable.** What the record cannot supply and this thread does not claim: (a) a transcriptome or EASI-FISH
source -- the case is three EM classifiers that agree on inhibitory and split on the transmitter, so the entry would be the first
in `TYPE_NT_OVERRIDE` with no expression evidence, and the honest label of the sign is "inhibitory (glutamate 2 of 3 classifiers,
GABA 1 of 3)"; under `NT_SIGN` and the present receptor table the two labels are bit-identical (W md5 `7a10d93b` for both), so
the adoption decision is really "sign GLNO -1" with the transmitter name a tie-break the data do not settle; (b) the room
take-off protocol (round 7's rate-half); (c) the compass operating point: gE 2 / gD 15 is robust to the sign (cx_glno.md 5), but
the round-2 finding stands that the bump is lost at gE 1.75 / gD >= 15 with GLNO signed, so a future gain default below gE 2 would
have to be re-run against this cache. The recommendation is in the Report block; nothing is adopted here.

### 4.3b Cross-check with thread 5A (`docs/audits/compass_ring_mechanism.md`, batch `cx5`)

Thread 5A's GLNO rows are the same candidate (its scratch cache `out/cache_c51b23e2` has the same compiled-W md5 `7a10d93b`) on
the pulse protocol without a body. Where the two threads meet they agree: (i) at the shipped gains (gE 1 / gD 1) the relabel is
inert because PEN never fires (5A: ExR6 / ER6 hold PEN 11-16 mV below threshold, GLNO 0.03-0.23 Hz; here, with the leg cycle
attached and DNa02 driven: PEN 0.6-1.3 Hz, GLNO 3.4-3.5 Hz, every flip and drift `null` with either cache -- the shipped-gains
arms are the record of that inertness under the efferent protocol); (ii) at the experiment instrument the sign costs ~10 % of the
bump and ~15 % of PEN without opening the loop (5A: RG 181-184 vs R 201-203 Hz; here EPG -8 %, PEN -15 %, wedge peak unchanged
at 253); (iii) 5A's collapse under the relabel (FG, 4/4 seeds within 0.15 s) is its damping-off configuration at gE 1 / gD 1, not
the gE 2 / gD 15 instrument used here, so the two results do not conflict: the instrument that can show a signed GLNO effect is
the one run here (GLNO signed vs silent at 2 / 15), and it shows rates moving and no report passing. (iv) 5A's C-family arms run
`receptor_model 'sign+gain'`, where glutamate and gaba are NOT the same model on GLNO's FB / EPG targets (section 1.4): 35 of
the 213 entries differ in resolved gain there. 5A's ring conclusions are unaffected -- the 84 PEN edges are identical under
every receptor mode -- but the "bit-identical to the gaba cache" statement is a `sign`-mode statement and should be qualified
when it is carried into a `sign+gain` discussion. One correction to carry over: 5A's table row (a) cites "the hemibrain name
(GLutamatergic LAL-NOduli)" as a source; section 1.2 above could not verify that expansion or any transmitter prediction in
Hulse 2021 / Scheffer 2020 / Wolff & Rubin 2018, so that item should not be counted as one of the three sources -- the two EM
classifiers (MaleCNS, BANC) plus FlyWire's inhibitory-but-GABA call are the evidence, and the entry text in section 1.2 is
written on those.

### 4.4 What remains

* The owner decision on the entry (suite half passed; the source standard of the three existing entries is not met).
* The room protocol under the candidate (>= 6 runs per arm, one submission) if the owner wants the rate-half.
* The compass: a sided PS196_b (the body-model item of `vnc_drive.md` 6 / `body_sided_state.md` 6) is what a signed GLNO needs
  to carry anything; until then the GLNO sign changes rates, not reports. The FB4Y / FB1C rate drops (-31 / -68 % at the
  experiment gains) are GLNO's second and third targets and nobody has looked at what those FB types do downstream.
* The `GLNO` name: no accessible text defines the 'G' or attaches a predicted transmitter; the claim should not be repeated
  without a page reference.
* The expectation ledger: `struct.GLNO_PEN.sign` (`flyverse/data/expected_responses.csv`, `expected 0`, `op == 0`, `gap 1`,
  "the largest single input of PEN carries no sign and is silent in the model") is the row this relabel is about, and the
  29-check suite does not carry it (no `check_key`). An adoption must re-score it; as written its polarity also looks inverted
  (meeting `== 0` prints "PASS (gap closed)" for what is the gap).

### 4.5 Caveats

1. One device (B200), one submission; the suite's stochastic sections scatter run to run on the GPU path (`guard_suites_r3.md`
   5), so "bit-identical" above is said only of the 17 checks whose three draws agree to the printed digits in both arms.
2. Three draws per suite arm is the rule's minimum and cannot call a value difference (floor p 0.10); the directions on record
   (loom_escape.GF_peak +6 Hz, wind.DNp18 +0.9 Hz) are for the next guard round. Both directions are quoted as
   z = diff / SD(shipped arm); the two-sample statistic is smaller because the glutamate arm scatters wider -- Welch 2.09 for
   `loom_escape.GF_peak` (SDs 4.91 against 1.40) and 2.72 for `wind.DNp18`. Quote the Welch beside the z so the direction is
   not read as firmer than the scatter.
3. The compass runs the working-tree `probe_vnc_drive.py` (section 2), with the other threads' opt-in edits inert on arm C;
   the run JSONs and `tree_state.json` carry the sha256s. Compass wall times 92-591 s reflect box sharing.
4. The receptor-table check (section 1.4) is on the current `receptors_by_type.csv`; a future PEN or FB row with an excitatory
   glutamate call would make the glutamate / gaba choice matter and must re-run `glno_relabel.py compare`.
5. `analysis.md` prints `z` for deterministic-null rows as large numbers (SD ~1e-16); `compare`'s verdict handles them
   (`underpowered` at 3 v 3); read the diff column there.

### 4.6 Further findings from the skeptic pass

Five points the independent pass adds that are not refutations (its section 3):

1. **The project's own expectation ledger carries a GLNO row this relabel changes.** `flyverse/data/expected_responses.csv`
   has `struct.GLNO_PEN.sign` (population `GLNO->PEN_a`, `expected 0`, `op ==`, `bound 0`, `gap 1`, notes "known gap: the
   largest single input of PEN carries no sign and is silent in the model") and `struct.GLNO_PEN.synaptic_pair_count`
   (`expected 16371`, range 15000-18000). The relabel takes that sign from 0 to -1: it is the documented deficit the ledger
   names. The row has no `check_key`, so it is not one of the 29 and the suite rule cannot see it, and the companion
   `struct.GLNO_PEN.synaptic_pair_count` row's `model_reference` says MISSING until `scripts/interp_paths.py` emits a
   per-type row for the GLNO -> PEN block. An adoption must re-score it -- and as written, with `op == 0` and
   `gap 1`, the row's polarity looks inverted (meeting `== 0` prints "PASS (gap closed)" for what is the gap). A ledger
   defect, not this thread's (section 4.4).
2. **`docs/INTERP.md` 10.1(7)(b) was met after the fact, not before.** `flyverse/body.py`, `flyverse/senses.py` and
   `scripts/probe_vnc_drive.py` went to the box uncommitted; the mitigation at submission was the sha256s, the 419-line diff
   and the diffstat. The owning thread has since committed exactly those files as **`286dca3`** ("Round 4b: the three level
   controls ... and the AN04B003 single-cell check"), where `scripts/probe_vnc_drive.py` still hashes `dbd8b115...` -- the
   file this batch ran. `286dca3` is the code identity of the compass numbers (section 2).
3. **"PEN raw fan-in totals unchanged in range" was loose.** Every PEN's capped raw fan-in rises by exactly **+120**
   (2 GLNO x cap 60): 867-1597.5 -> 987-1717.5. What is unchanged is that all 42 stay far below `input_norm_ref` 5000, so
   the scale stays 1.00 and the +120 buys no renormalisation (section 1.3).
4. **The md5 equality with `out/cache_72164311` has no artefact in `out/cx5b/`.** `validation` asserts it and the pass
   verified it at array and byte level (the two `W_post_pre.npz` files hash the same, `580e7dce...`, as thread 5A's
   `out/cache_c51b23e2`), but nothing in the run directory records it; a `compare --cache-dir out/cache_72164311` run would
   put it in a named file.
5. **The three draws of a suite arm use the same seeds 0, 1, 2**; the replicate unit is the job, which `docs/INTERP.md`
   10.1(3) explicitly sanctions ("two same-code batches with the same seeds are not the same draws"). The three draws are not
   three seeds (section 4.5).

## Report

```yaml
summary: >
  Thread 5B prepared (did not adopt) the transmitter relabel {GLNO: glutamate}. What the record supports: the SUITE HALF of
  the round-2 rule is met, the RATE HALF was not run, and the SOURCE STANDARD of the three existing `TYPE_NT_OVERRIDE` entries
  is not met. Two low-confidence EM classifiers say glutamate (MaleCNS v1.0 T-bar classifier glutamate 51 % / ACh 37 % over
  783 T-bars, every consensus column 'unclear' at conf 0.48; BANC v888 'Predicted NT type' GLUT 4/4 at conf 0.47-0.50,
  unverified, taken without a threshold by the banc backend), one says GABA (FlyWire v783 per-cell top_nt gaba 3/4 at conf
  0.30-0.33, glutamate 1/4 at 0.30), and all three say INHIBITORY; no transcriptome or EASI-FISH source exists in any of the
  11 local tables, and the 'named for its predicted transmitter' premise could not be verified in Hulse 2021 or Scheffer 2020
  (which only renamed GLN -> GLNO), so the hemibrain name is NOT one of the sources and is not counted. Because W is
  bit-identical to the round-3/4 gaba cache, the decision actually on the table is "sign GLNO -1", not "GLNO is
  glutamatergic": the transmitter name is a tie-break the data do not settle. Structure (CPU, entry by entry): the candidate
  cache differs from the shipped one in exactly the 4 GLNO cells and their 213 W entries (0 -> -raw count; 17,698 synapses;
  84 PEN edges at -16.50 mV per GLNO spike, all above the cap; the PEN -> GLNO -> PEN loop closes as negative feedback), plus
  a <= 1.2 % fan-in-scale shift on 20 minor targets; all 213 entries are -1 under the shipped receptor model, whose
  `fast_sign` is identical between the glutamate and gaba caches on all 25,578,600 entries -- under `sign+gain` / `full` the
  two labels are NOT the same model (resolved `fast_gain` differs on 35 of the 213, `slow_sign` on 3), though never onto PEN.
  One 22-job submission (cx5b-d08be3, B200, 0 failed, 18.5 min): the 29-check suite 3 draws x {shipped, glutamate} is 27 PASS
  / 0 FAIL / 2 KNOWN GAP in all six with no status change on any check (the suite half of the round-2 rule, met; the room
  rate-half not run); the efferent compass (family level, arm C = all+leg_cycle, 4 seeds x {shipped, glutamate} x {gE 2 /
  gD 15, shipped gains}) shows every predeclared member null: GLNO / PEN_a / PEN_b flips and the bump drift are unchanged by
  the relabel, and the signed self-turn report STILL STOPS AT AN04B003 (result, -11 Hz, 4/4 arms), null at PS196_b, GLNO, PEN
  and EPG. The relabel changes rates, not reports: ring -5..-15 %, FB4Y -31 %, FB1C -68 %, ExR8 silenced outright (0.00 Hz on
  both cells in all four runs), and GLNO's fixed L-R +27 -> +3..+4 Hz -- a STANDING ASYMMETRY of the pinned protocol's bump
  position (present in both rest phases, no sign change with the turn direction, and gone at the shipped gains), not a turn
  signal. Corrected throughout by an independent Opus skeptic pass (verdict MOSTLY SOUND; corrections C1-C14 applied).
skeptic:
  source: "independent Opus pass, 2026-09-15"
  verdict: "mostly sound"
key_claims:
  - "Entry comparison: 213 differing W entries, all with a GLNO presynaptic cell, all 0 -> negative, |value| = the shipped sign-0 counts; nt/sign differ on exactly 4 cells; same sparsity pattern; sum|W| 121,460,584 -> 121,478,280; md5 ef23cc27 -> 7a10d93b = the round-3/4 gaba cache (out/cx5b/entry_compare.json)."
  - "GLNO -> PEN: 84 entries, 16,371 raw syn, edges 88-345 (84 above cap 60), 2 GLNO per PEN (42/42), -16.50 mV per GLNO spike on every edge (PEN fan-in scale 1.00 both caches); PEN -> GLNO unchanged (+3,496; +4.1..+16.5 mV per PEN spike)."
  - "Elsewhere: ExR8 8 / 435 syn, GLNO -> GLNO 4 / 400, FB4Y 16 / 218, FB1C 14 / 114, FB4M 11 / 21, 51 further types at <= 21 syn (58 postsynaptic types in all); effective-weight matrix differs on 14,828 entries = the 213 + the fan-in renormalisation of 20 cells (scale ratio 0.988-1.000, max 0.19 mV per spike)."
  - "Receptor model (sign/abs and sign/class): 213 / 213 entries -1, 0 entries +1; `fast_sign` is identical between the glutamate and gaba caches on all 25,578,600 entries, so under the SHIPPED receptor model the two labels are the same model. Under `sign+gain` / `full` they are not: the resolved `fast_gain` differs on 35 of the 213 and `slow_sign` on 3 (never onto PEN, where the two labels are identical under every mode)."
  - "Suite (3 v 3, one submission, B200): 27/0/2 in every draw of both arms; changed_pooled [], changed_matched [], unstable_shipped []; 16 checks identical to the printed digits in all six draws, three checks move by a fixed amount (two of them constant within each arm) (motion.min_dsi 0.24110 -> 0.24064; bitter.shiu_sugar_MN9 139.90 -> 135.15 Hz; shiu_sugar_bitter 0.818 -> 0.988), loom_escape.GF_peak +6.2 Hz and wind.DNp18 +0.9 Hz higher in 3/3 (underpowered)."
  - "Compass at gE 2 / gD 15 (4 v 4): F1 GLNO flip +0.66 (z 2.1, p 0.20, p_holm 0.60), PEN_a -0.39 (p 0.11), PEN_b -0.44 (p 0.20), drift +0.004 w/s (p 0.69): all null; F3 GLNO flip vs null z -0.4 p 0.69, PEN_a / PEN_b p 1: the signed report does NOT pass GLNO; F5 drift ccw +0.002 / cw +0.004 w/s vs rests (p 0.57 / 0.37) against ideal +-4.0."
  - "AN04B003 flip -11.2 / -11.8 / -11.5 / -11.5 Hz result (p 0.0286) in all four arms; PS196_b null in all four; EPG null; DNa02 +37..+38 Hz (the stimulus)."
  - "Rates at rest, gE 2 / gD 15, shipped -> glutamate (Hz): GLNO 133.2 -> 116.0, PEN_a 44.9 -> 38.8, PEN_b 50.7 -> 42.0, EPG 55.4 -> 50.8, Delta7 100.0 -> 95.0, PEG 33.3 -> 29.1, FB4Y 84.2 -> 57.7, FB1C 19.1 -> 6.2, ExR8 0.43 -> 0.00 (the one type the relabel silences: 0.00 Hz on both cells and max single cell 0.00 in all four runs); GLNO L-R +26.9..+27.1 -> +3.1..+4.5 Hz -- a standing asymmetry of the pinned protocol's bump position in both caches (same size in both rest phases, no sign change with the turn direction, -0.21..+0.18 Hz at the shipped gains), not a turn signal."
  - "At the shipped gains no bump forms with either cache (vs 0.10-0.25, peak 1-3 Hz; ring 0.3-3.5 Hz); F2 / F4 all null."
  - "Adoption reading: the SUITE half of the round-2 rule is met (no check changes status in 3 + 3 draws, 174 rows) and nothing else -- the room rate-half (>= 6 runs per arm) was NOT run and the source standard of the three existing TYPE_NT_OVERRIDE entries (transcriptome or EASI-FISH) is NOT met; two low-confidence EM classifiers call glutamate, one calls GABA, all three call inhibitory; the W is bit-identical to the gaba cache, so the decision is 'sign GLNO -1'; the 'hemibrain name' is not a source and is withdrawn."
  - "Sources: expression_*.csv, nt_by_type_transcriptome.csv, type_map_*.csv, receptors_by_type.csv have 0 rows for GLNO / LNO1 / LAL-NO1; the typing maps only equate GLNO with FlyWire / hemibrain GLNO (4 cells); LNO1 is a different type (EASI-FISH GABA)."
files_written:
  - scripts/glno_relabel.py
  - docs/audits/glno_relabel.md
  - out/cache_glno_glu/ (scratch: W_post_pre.npz, neurons.parquet, sign0_counts.npz, TYPE_NT_OVERRIDE.json, manifest.json)
  - out/cx5b/entry_compare.json, out/cx5b/entry_compare.md, out/cx5b/compile_local.txt, out/cx5b/compare_local.txt
  - out/cx5b/predeclared.json, tree_state.json, probe_diff_vs_HEAD.txt, jobs.json, batch.sh, submit_tree.txt, smoke/, smoke_local.txt
  - out/cx5b/fam_suite/ (6 suite JSONs + .txt), out/cx5b/fam_c0..3/ (16 compass _run.json, 64 phase JSONs, 128 npz, 16 .txt)
  - out/cx5b/analysis/ (analysis.md, summary.json, suite_table.csv, decision_table.csv, compass_flip_<arm>.csv, compass_chain_<arm>.csv), out/cx5b/analysis_console.txt
  - out/cx5b_cluster.log
api:
  - "scripts/glno_relabel.py compile [--cache-dir out/cache_glno_glu]: compile TYPE_NT_OVERRIDE + {GLNO: glutamate} from the raw files (lock file, temp dir + rename, sign0 counts built)"
  - "scripts/glno_relabel.py compare [--cache-dir] [--out out/cx5b/entry_compare.json]: shipped vs candidate, neuron table, W entry by entry, by post type, PEN edges, effective mV, fan-in scale, receptor-model signs"
  - "scripts/glno_relabel.py plan [--dir out/cx5b] [--name cx5b] [--minutes 45] [--seeds 0,1,2,3] [--draws 3]: predeclared.json (stamped), tree_state.json, probe_diff_vs_HEAD.txt, jobs.json, batch.sh"
  - "scripts/glno_relabel.py suite --arm shipped|glu --draw N --out <json> (GPU job): benchmark.py --sections all --seeds 0,1,2 [--cache-dir] + provenance + glno_relabel block, exit 3 on cache / device mismatch"
  - "scripts/glno_relabel.py compass --cache shipped|glu --gains exp|shipped --seed S [--block] --out <stem> (GPU job): probe_vnc_drive.py compass --family level --arm C [--gains 2:15|none] [--cache-dir] via runpy + glno_relabel block, exit 3 on mismatch"
  - "scripts/glno_relabel.py analyse [--dir out/cx5b] [--out out/cx5b/analysis]: suite status table + verdict, flip tables, chain rates, drift, predeclared families with Holm"
  - "scripts/glno_relabel.py smoke [--suite]: CPU smoke of the wrappers (--quick --device cpu)"
validation:
  - "compile: nt counts move by exactly 4 cells (glutamate 29,707 -> 29,711, unknown 2,361 -> 2,357); the candidate's W md5 equals out/cache_72164311 (the round-3/4 gaba cache), sum|W| 121,478,280, nnz 25,578,600"
  - "compare: same bodyIds / order / types; nt and sign differ on the 4 GLNO cells only; same sparsity pattern; 213 differing entries all GLNO-pre, all 0 -> negative, |value| == shipped sign0 counts; sign0 totals differ by exactly 17,698"
  - "smoke: compass wrapper on both caches exit 0, problems [], right md5 / label / gains / spec in each run JSON; import check of cx_wedge / interp_apply_rotation / probe_vnc_drive / benchmark / glno_relabel on the shipped tree"
  - "batch cx5b-d08be3: 22 job(s), 0 failed (18.5 min); artefacts 6 + 16 + 64 + 128 + 22 as planned; every JSON NVIDIA B200, problems [], the arm's md5 (ef23cc27 / 7a10d93b) and GLNO label read back from the cache the run used; compass gains / background / pulse / adaptation / spec / leg_cycle / mn_ref_hz 30 recorded"
  - "round-4 reproduction on the shipped cache: GLNO L-R +26.9..+27.1 Hz every phase, AN04B003 flip -11.5 result, PS196_b null, drift |mean| <= 0.005 w/s"
  - "ring grid reproduction: PEN -15 %, Delta7 -5 %, GLNO -13 % at gE 2 / gD 15 match cx_glno.md section 5 (-14..-16 / -4..-6 / -12..-14 %)"
recommendations:
  - "Adoption is the owner's call: the suite half of the round-2 rule is met (no status change in 3/3 draws); the source standard of the existing entries (transcriptome or EASI-FISH) is not, and the transmitter name is a tie-break the data do not settle (glutamate 2 of 3 classifiers, GABA 1 of 3; bit-identical W). If adopted, write the entry as in section 1.2 and rebuild cache/ (backup as in round 2); the compass default gE 2 / gD 15 survives (cx_glno.md 5) but any gain default below gE 2 must be re-run."
  - "If the owner wants the rate-half: the room take-off protocol on out/cache_glno_glu at >= 6 runs per arm in one submission (guard_suites.sh's room jobs with FLYVERSE_CACHE / --cache-dir)."
  - "Do not spend more compass runs on GLNO's sign: the report is gone at PS196_b in every arm; the next compass item is the sided PS196_b input (the body-model thread), after which the four arms here are the ready-made comparison."
  - "Look at FB4Y and FB1C (-31 / -68 % at the experiment gains): GLNO's second and third targets, unexamined downstream."
open_questions:
  - "What the 'G' in GLNO denotes and whether any paper predicted GLNO's transmitter from light-level data -- unverified; needs a page reference before it is cited again."
  - "Is the deterministic bitter.shiu_sugar_MN9 -4.75 Hz / motion.min_dsi -0.19 % path through GLNO's FB / LAL outputs or through the 20 renormalised cells? (small, no status change; a trace from GLNO to MN9 would say)"
  - "Does GLNO's own L-R (+27 -> +3 Hz) matter for anything downstream of ExR8 / FB4Y / FB1C?"
  - "The room rate-half under the candidate (not run)."
```

## Skeptic pass (independent, 2026-09-15)


Everything below was recomputed from the artefacts with my own code (scratchpad `sk_cache.py`, `sk_suite.py`,
`sk_compass.py`, `sk_recep.py`), CPU only, `CUDA_VISIBLE_DEVICES=-1`. No cluster job was needed: every claim under
test is recomputable from `cache/`, `out/cache_glno_glu/`, `out/cache_72164311/`, `out/cx5b/` and the source tables.
Nothing in the repository was edited.

**Verdict: mostly sound.** The five headline claims all survive. The structural, receptor, suite-status and compass
results reproduce to the printed digit under my own implementations, the provenance chain holds, and the three
substantive judgements ("adoptable by the suite half only", "the source standard is not met", "the report still stops
at AN04B003") are correct and correctly hedged. What fails is a band of secondary arithmetic and three pieces of
over-claimed wording: a bit-identical count off by one, a summary table that does not sum to its own totals, a
receptor-model gloss that is false outside the shipped mode, a "never silent" hedge that hides the one type the
relabel does silence, an unsourced submission timestamp, and a reading-list gloss that overstates `cx_glno.md`.

---

### 1. Refuted

**R1. "17 of the 29 checks are bit-identical between the arms in every draw" (4.1; Report `key_claims`) -- it is 16.**
Recomputed from the six suite JSONs: exactly 16 checks have identical `measured` in all six draws -- `rest.spikes_per_step`,
`taste.MN9_hz`, `smell.PN_hz`, `smell.KC_active`, `dn.DNa02_L_leg_asym_hz`, `dn.MDN_top_hz`, `dn.DNp09_top_hz`,
`walk.GF_max_hz`, `walk.power_max_hz`, `walk.power_sustained_hz`, `loom.escape_cm`, `motion.correct_directions`,
`loom_escape.escapes`, `bitter.calibrated_sugar_MN9_hz`, `bitter.calibrated_sugar_bitter_MN9_hz`,
`compass.wedge_cells_persisting`. The audit's own parenthetical ("rest, taste, smell, dn, the walk triple, loom.escape,
motion.correct_directions, loom_escape.escapes, calibrated bitter, compass") enumerates exactly those 16, so the count
contradicts its own list. The partition is 16 bit-identical + 2 deterministic-and-moved + 11 with run scatter = 29.
*Second point on the same sentence:* `guard_suites_r3.md` caveat 1 says in terms that **"bit-identical" is a CPU claim
under the project rule** and that nothing from a GPU batch is a bit-identity claim. These six draws are B200.
(files: `out/cx5b/fam_suite/suite_{shipped,glu}_{1,2,3}.json`)

**R2. "three deterministic sections move" (4.1, twice; Report `key_claims`) -- only two of the three are deterministic.**
`motion.min_dsi` is *not* constant within either arm: shipped 0.241096598 / 0.241096418 / 0.241096508, glutamate
0.240637408 / 0.240637263 / 0.240637408. The other two are exactly constant in both arms (139.89849777 x3 -> 135.15008698
x3; 0.81783977 x3 -> 0.98794834 x3). The min_dsi *shift* is real and large against its own scatter (-4.59e-4 against a
within-arm SD of ~1.8e-7, a factor ~2,500), so the finding stands; the word "deterministic" does not.
(file: same six JSONs)

**R3. The tail row of the section 1.3 by-post-type table does not sum, under any reading.**
The row reads `| FB1H 2 / 21, LNO2 3 / 11, AN27X013 2 / 8, PFNa 6 / 8, PVLP060 1 / 6, PFNd 4 / 6, LCNOpm 3 / 6,
DNpe023 2 / 5, and 44 further types at 1-4 synapses (...) | 62 | 96 | 0.5 % |`. Recomputed: GLNO's output reaches **58**
postsynaptic types. The seven rows named above it are 137 entries / 17,559 synapses. The whole tail is therefore
**51 types, 76 entries, 139 raw synapses, 0.79 %** of GLNO's output -- not 62 / 96 / 0.5 %. Splitting it the other way:
the 8 explicitly listed types are 23 entries / 71 synapses, leaving **43** further types (not 44) at 53 entries /
68 synapses, max 4 synapses each. As printed the table's entries column sums to 199 (not 213) and its synapse column to
17,655 (not 17,698). The generated file `out/cx5b/entry_compare.{json,md}` has all 58 rows correctly; the defect is in
the hand-rolled summary row only.

**R4. `key_claims` "53 further types at <= 21 syn" -- it is 51.** 58 post types minus the 7 named ahead of it
(`PEN_a`, `PEN_b`, `ExR8`, `GLNO`, `FB4Y`, `FB1C`, `FB4M`). The "<= 21 syn" bound is right (FB1H 21 is the largest).

**R5. Section 1.4's gloss "the glutamate / gaba choice is invisible to the present receptor table (it would become
visible only if a PEN or FB row with an excitatory glutamate call were added)" is false as stated.**
What is true -- and *stronger* than the audit claims -- is that `receptor_signs(...).fast_sign` is identical between the
glutamate and the gaba cache on **all 25,578,600 entries**, under both `abs` and `class`, so under the shipped
`receptor_model='sign'` the two labels are the same model everywhere, not merely on the 213. But the table is not blind
to the difference on those same 213 entries:

| resolved field | entries of the 213 that differ glutamate vs gaba | which |
|---|---|---|
| `fast_sign` | 0 | -- |
| `fast_gain` | **35** | FB1C 14, FB4M 11, FB2A 4, ExR2 2, FB1H 2, ER4m 1, OA-VUMa1 1 (glutamate `high` = x1.5 vs gaba `mid` = x1.0 under `brain.DEFAULT_RECEPTOR_GAIN`) |
| `slow_sign` | 3 | EPG x2 and ER4m x1 (glutamate 0 vs gaba -1) |
| `slow_gain` | 57 | FB4Y 16, FB1C 14, FB4M 11, PFNd 4, FB2A 4, EPG 2, ExR2 2, FB1H 2, ER4m 1, OA-VUMa1 1 |

So under `receptor_model='sign+gain'` or `'full'` the candidate is **not** the gaba cache's model. Two further errors in
the same parenthetical: FB rows with a non-inhibitory glutamate net already exist (`FB4Y`, `FB1C`, `FB4M`, `FB1H` all
carry `fast_net = mixed` for glutamate), and the table already holds **160 of 613** types whose glutamate `fast_sign` is
+1 under the class rule (14 under `abs`) -- nothing would need to be "added". The reason the 213 come out -1 is not that
the table cannot express the distinction; it is that no GLNO target happens to be one of those 160. *Onto PEN
specifically* the claim does hold under every mode: the `PEN_a(PEN1)` and `PEN_b(PEN2)` glutamate and gaba rows carry the
same `fast_sign` -1, the same `fast_gain_class` `high`, the same `fast_sign_abs` and the same slow columns (they differ
only in the lead gene, `GluClalpha` vs `Rdl`), so all 84 PEN edges are identical under `sign`, `sign+gain` and `full`.
This matters for the 5A cross-check: thread 5A ran `sign+gain` arms (C / CF / CFG) with the same relabel, and the
"indistinguishable" statement does not carry into them.
(files: `flyverse/data/receptors_by_type.csv`, `flyverse/brain.py` `DEFAULT_RECEPTOR_GAIN`, both caches)

**R6. "never silent for a firing cell" (4.2 rates paragraph) conceals the one type the relabel silences.**
`ExR8` at the experiment gains: shipped L 0.86 / R 0.00 Hz, max single cell 2.14 Hz; glutamate **L 0.00 / R 0.00 Hz,
max single cell 0.00 Hz in all four runs**. ExR8 is GLNO's third-largest target (8 entries, 435 synapses, 2.5 % of its
output). The audit prints "ExR8 0.4 -> 0.0" in the same sentence as the hedge, so the number is on record, but the
parenthetical reads as a denial of exactly the event that occurred. (`analysis.md` 2.2's "never 'silent' for a cell with
nonzero firing" is a tautology and should not be carried into the prose.)
(files: `out/cx5b/fam_c*/compass_{shipped,glu}_exp_r*_rest.npz`)

**R7. "stamped 2026-09-15T05:35:08Z, before the 05:35:20Z submission" (section 2) -- 05:35:20Z is in no file.**
`predeclared.json.stamped_utc`, `tree_state.json.when`, `jobs.json.written_utc` and `submit_tree.txt` all read
**05:35:08Z**, and `submit_tree.txt`'s own first line is `submitted 2026-09-15T05:35:08Z` -- the same second as the
stamp, not 12 s after it. `out/cx5b_cluster.log` carries elapsed seconds only, no wall clock, and there is no
`scheduler_receipt.json` in `out/cx5b/` (thread 5A's batch has one; this one does not). The ordering that *is* in the
record is: stamp 05:35:08Z < earliest `glno_relabel.started_utc` **05:35:33Z** (six suite jobs and four compass jobs at
05:35:33-36Z). That is a real precedence and it should be the one quoted. (Note this also means the predeclaration and
the submission share a timestamp to the second: they were written by the same `plan` invocation seconds apart, which is
fine, but it is not what "before the 05:35:20Z submission" says.)

**R8. The "Read first" gloss "cx_glno.md ... section 3 for the round-2 finding that a glutamatergic / GABAergic GLNO
abolishes the bump at gE 1.75 in 6/6 seeds" overstates its source.** `cx_glno.md`'s gE 1.75 / gD 15 rows are **three**
seeds (0, 1, 2), `glu` 0/3 persisting; the only six-seed row at gE 1.75 in that audit is gD **8**, where `gaba` is
confined in 4/6 seeds and persists in 3/6 (bump 148-155 Hz in three of them). The audit's own section 1.3 phrasing
("lost at gE 1.75 / gD >= 15") is accurate; the header line is not.

---

### 2. Confirmed

#### 2a. Claim (1) -- the cache comparison (all recomputed with my own loader, differ code and `_shaped_weights` call)

* neuron table: 167,106 bodyIds, same order, same types; `nt` differs on exactly **4** cells, `sign` on exactly **4**,
  and the differing index set equals the GLNO index set exactly. bodyIds 12104 / 14881 / 23325 / 25939, instances
  `GLNO(LAL-NO1)_L` x2 / `_R` x2, `unknown`/0 -> `glutamate`/-1. nt counts glutamate 29,707 -> 29,711, unknown
  2,361 -> 2,357. OK
* W: same shape, nnz 25,578,600, `indptr` and `indices` equal; **213** differing entries, **213/213** with a GLNO
  presynaptic cell, all 0 in the shipped cache, all negative in the candidate, and `|candidate| == shipped
  sign0_counts` on every one (exact array equality, not `allclose`). GLNO has 213 presynaptic entries in total, 0
  unchanged. sum|W| 121,460,584 -> 121,478,280 (+17,698). sign-0 totals 2,701,289 -> 2,683,591, difference exactly
  17,698. md5 `ef23cc27...` -> `7a10d93b...`. OK
* **bit-identity with the round-3/4 gaba cache -- confirmed, and stronger than claimed.** `out/cache_72164311`
  (`TYPE_NT_OVERRIDE.json` = {TmY14, Mi19, aMe8, GLNO: **gaba**}) gives md5 `7a10d93ba2086f2c76bcdabdca79b4ec`, and
  `data`, `indices` and `indptr` are `np.array_equal` to the candidate's, 0 of 25,578,600 entries differing. The two
  `W_post_pre.npz` **files** also hash the same at the byte level (`580e7dce...`) as thread 5A's `out/cache_c51b23e2`,
  whose override table is {..., GLNO: **glutamate**} -- so 5A's glutamate cache and round 3/4's gaba cache are the same
  file. (The 5B candidate's own npz is `1870fd24...`: different container bytes, identical arrays.) This is the one
  claim in `validation` with no artefact in `out/cx5b/` recording it; it is true.
* PEN: 84 edges, 16,371 raw synapses, min 88 / max 345 / mean 194.89 / median 181.5, **84/84 above the cap 60**,
  exactly 2 GLNO per PEN on 42/42 PEN. Effective weight **exactly -16.500 mV** on all 84 (one distinct value;
  0.275 x 60 x scale 1.00), shipped 0.0, and **exactly -33.000 mV per PEN per volley** of its two GLNO (min = max).
  Return leg PEN -> GLNO unchanged: 84 entries, W sum +3,496 in both caches, +4.125 ... +16.500 mV per PEN spike, GLNO
  fan-in scale 1.00 in both. OK
* **The loop is contralateral and closes within one side**, as claimed: GLNO_L -> 42 edges onto somaSide-R PEN (8,236
  syn) and GLNO_R -> 42 onto somaSide-L PEN (8,135); PEN_R -> GLNO_L (42, 1,938 syn), PEN_L -> GLNO_R (42, 1,558). So
  GLNO_L reads only R PENs and writes only R PENs. OK
* Composition percentages all reproduce on the raw counts (|W| + sign-0): GLNO output -> PEN **92.50 %**; GLNO -> PEN as a
  share of PEN's raw input **19.36 %**; PEN as a share of GLNO's own raw input **37.3 %** (84 of 646 entries, 3,496 of
  9,371 raw synapses). OK
* fan-in: effective matrix A differs on **14,828** entries, 213 with a GLNO presynaptic cell, 14,615 outside;
  **20** cells' fan-in scale moves, ratio 0.98823 ... 0.99990 (**max shift 1.177 %**, "below 1.2 %" OK), types
  FB4Y x4, LNO2 x2, LCNOpm x2 + 12 singles; largest change on any non-GLNO entry of A **0.1927 mV**. OK
* Receptor model on the 213, my own call: **213 -1, 0 +1, 0 zero** under both `abs` and `class`. OK (Tier tally over the
  213: fuzzy 84, fallback 66, class 51, **alias 12** -- the audit names tiers only for its six main targets and does not
  mention the alias tier; 147 of 213 are matched by the table at all. Not an error, just incomplete.) The per-arm
  receptor coverage blocks in the suite JSONs corroborate this independently: `pre_unknown` 254,444 -> 254,231 (-213)
  and fuzzy +84 / class +51 / alias +12 / fallback +66 = 213.
* `LIFParams` defaults as quoted: `w_syn` 0.275, `conn_cap` 60, `input_norm_ref` 5000, `input_norm_alpha` 1.0,
  `receptor_model` `sign`, net rule `abs`. OK

#### 2b. Claim (2) -- the evidence case

Every source number checks out against the primary file, and the "could not verify the name" finding is correct.

* MaleCNS: `docs/audits/nt_audit.md` rows 154 and 550 -- 4 cells, cb_intrinsic, 17,698 output synapses, 16,371 to PEN
  (19.4 %), consensus / cell-type / per-body all `unclear` at conf 0.48, 783 T-bars, **glutamate 51 % / acetylcholine
  37 % / serotonin 7 %**. OK
* BANC v888 raw (`D:\Datasets\flywire\BANC v888\neurons.csv.gz`): the four Root IDs quoted, `Predicted NT type` GLUT
  x4 at **0.48 / 0.50 / 0.50 / 0.47**, `Verified NT type` empty x4. OK `cache/banc/neurons.parquet` carries
  `nt = glutamate`, `sign -1`, `nt_verified` NaN on all four. OK `flyverse/backends/banc.py`:
  `"nt": [verified_nt(v, p) for v, p in zip(d["Verified NT type"], d["Predicted NT type"])]` -- **no threshold is
  applied**, and `read(..., nt_threshold=0.5, ...)` is accepted and unused. OK exactly as the audit says.
  LNO1 in the same file: GABA predicted 0.75 / 0.76 / 0.77 / 0.80, `Verified NT type` = gaba x4. OK
  (One over-attribution: the BANC file records only "Verified NT type = gaba", not the method; the EASI-FISH source is
  FlyWire's, see below.)
* FlyWire v783 (`schlegel2024_Supplemental_file1_neuron_annotations.tsv`): GLNO `top_nt` gaba 0.3297 / 0.3038 / 0.3327
  and glutamate 0.2994, `known_nt` empty x4. OK `cache/fafb/neurons.parquet` GLNO `unknown` / sign 0 x4, and
  `flyverse/backends/fafb.py` thresholds at 0.5 (`d.nt_type.where(d.nt_type_score >= nt_threshold)`). OK
  LNO1: `known_nt` = "gaba, acetylcholine-negative, glutamate-negative", `known_nt_source` =
  **"Wolff et al., 2024 (EASI-FISH)"**, `top_nt` gaba x3 / acetylcholine x1. OK
* Expression / typing: no GLNO, LNO1, LAL-NO1 or LAL_NO row in any `expression_*.csv`, `nt_by_type_transcriptome.csv`,
  `type_map_*.csv` or `receptors_by_type.csv`; `receptors_by_type.csv` has **no GLNO row and no ExR8, LNO2 or PEG row**
  either. The only tables carrying the name are the alias / typing maps, which state an identity (GLNO = FlyWire GLNO =
  hemibrain GLNO, 4 cells, class CX), not a profile. OK
* **The name.** I ran the web checks the task permits. Nothing I could reach defines the 'G' or attaches a transmitter
  prediction to GLNO. Search returns for Scheffer 2020 surface exactly the sentence the audit quotes -- *"the nodulus
  neurons are now 'LNO' and 'GLNO' instead of 'LN' and 'GLN'"* -- with no transmitter statement; Wolff & Rubin 2018's
  abbreviation scheme is described as composing anatomical region names (LAL-NO(a) etc.), with no glutamate element.
  A search-engine summary did assert "GLNO is the current standardized name for glutamatergic nodulus neurons", but no
  quoted source text supports it -- it is the summariser inferring the expansion, which is precisely the failure mode the
  audit warns about. **5B's section 1.2 row 2 and its 4.3b correction to 5A are right, and I could not verify the
  expansion either.** The unsourced assertion is in `out/cx5/structure/evidence_glno.json`:
  `"hemibrain_name": "GLNO = GLutamatergic LAL-NOduli neuron (the hemibrain / Hulse et al. 2021 naming)"`.
  Sources consulted: [eLife 9:e57443 (Scheffer 2020)](https://elifesciences.org/articles/57443),
  [J Comp Neurol 526:2585 (Wolff & Rubin 2018)](https://onlinelibrary.wiley.com/doi/10.1002/cne.24512).

#### 2c. Claim (3) -- the suite

I re-evaluated every check's status from the recorded `measured` with `benchmark.evaluate` and `benchmark.REFERENCES`
rather than reading the recorded `status`: **0 mismatches over all 174 check rows.**

* 29 checks in every draw, 27 PASS / 0 FAIL / 2 KNOWN GAP (`object.LC10a_flip_hz`, `compass.wedge_cells_persisting`) in
  all six. OK
* Under the predeclared pooled rule: `changed_pooled` **[]**, `changed_matched` **[]**, `unstable_shipped` **[]**,
  per-run `problems` **[]** -- recomputed independently, matching `summary.json`. **No check changes status.** OK
* Provenance per draw: device `NVIDIA B200` x6; `config.receptor` model `sign` / net rule `abs` x6; backend
  `native (cuda_kernels, cuda_graphs, event_driven, warp)` x6; seeds `[0, 1, 2]` in all six; cache md5 `ef23cc27` x3 /
  `7a10d93b` x3; GLNO read back `unknown`/0 x3 and `glutamate`/-1 x3; wall 480.9-532.0 s. OK
  (One cosmetic note: the shipped suite arm read `.../runs/cx5b-d08be3/cache` while the shipped compass arm read
  `<cluster-fs>/datasets/flyverse/cache`; both fingerprint `ef23cc27`, so the arms are the same cache.)
* The moved values: `motion.min_dsi` 0.2410966 -> 0.2406374 (-0.000459, -0.19 %); `bitter.shiu_sugar_MN9_hz`
  139.8985 -> 135.1501 (-4.748, -3.39 %); `bitter.shiu_sugar_bitter_MN9_hz` 0.81784 -> 0.98795 (+0.17011). OK all three,
  and all far from their bounds (>= 0.1, > 50, < 10).
* `loom_escape.GF_peak_hz` +6.156 Hz, **z 4.39**, p 0.2, verdict `underpowered`; `wind.DNp18_flip_hz` +0.900 Hz,
  **z 3.72**, p 0.1, `underpowered`; `rotate.DNp20_flip_hz` +6.354, p 0.4. OK all as printed.
* **Is 3 v 3 the declared rule?** Yes for the suite half. `guard_suites_r3.md` 4: *"the candidate's own full 29-check
  suite x >= 3 draws with no check worse in status than the baseline ... and the room take-off protocol"*; the rate-half
  needs **>= 6 runs per arm** (its own power calculation: 0.50 at n=4, 0.85 at n=6). `connectome.py`'s round-2 comment
  says "adopt only if no suite check changes status". The audit's predeclared rule ("no status in any glutamate draw
  that is not the status of every shipped draw") is the **round-2 form and strictly stronger** than round 7's "no check
  worse" -- passing it implies passing the looser one. Correctly stated. OK
* **Is "underpowered" the right word?** Yes, and it is the tool's own verdict, not a euphemism. `common.compare` returns
  `underpowered` whenever `min(n_a, n_b) < CALL_REPLICATES = 4` **or** `p_floor > alpha`; `p_floor(3,3) = 0.10 > 0.05`,
  so at 3 v 3 no z can produce a `result` (`INTERP.md` 10.1(3) states the same rule). The audit's "directions on
  record, not results" is exactly right. **One refinement worth adding**: the z of 4.4 is `diff / SD(shipped arm)`, and
  the glutamate arm's own SD is 3.5x larger (4.91 vs 1.40 Hz); the two-sample statistic is **Welch 2.09** (and 2.72 for
  wind.DNp18, where z reads 3.72). Quoting z alone makes the direction look firmer than the scatter supports.

#### 2d. Claim (4) -- the efferent compass

I rebuilt the L-R / flip / null statistic from the 128 npz myself (per-type mean over somaSide-L cells minus somaSide-R
cells of each phase's time-mean `rate_hz`, graded cells on `optic_dr`; flip = LR(ccw) - LR(cw); null = LR(rest2) -
LR(rest)), rather than calling `interp_apply_rotation.flip_table`. Every predeclared row reproduces:

| family | member | audit | mine |
|---|---|---|---|
| F1 | GLNO flip | +0.66, z 2.1, p 0.20, p_holm 0.60 | +0.6607, z 2.122, p 0.2000, p_holm 0.600 |
| F1 | PEN_a flip | -0.39, z -1.9, p 0.11, p_holm 0.46 | -0.3929, z -1.930, p 0.1143, p_holm 0.457 |
| F1 | PEN_b flip | -0.44, z -1.7, p 0.20, p_holm 0.60 | -0.4383, z -1.667, p 0.2000, p_holm 0.600 |
| F1 | drift ccw-cw | +0.004, z 0.6, p 0.69 | +0.00390, z 0.633, p 0.6857, p_holm 0.686 |
| F2 | GLNO / PEN_a / PEN_b | -0.27 / -0.05 / +0.02; z -0.5 / -0.5 / 0.3 | -0.2679 / -0.0500 / +0.0195; z -0.475 / -0.542 / +0.318 |
| F3 | GLNO vs null | -0.52, z -0.4, p 0.69 | -0.5179, z -0.357, p 0.6857 |
| F3 | PEN_a / PEN_b vs null | +0.01 / -0.06, p 1 / 1 | +0.0107 / -0.0617, p 1 / 1 |
| F4 | GLNO / PEN_a / PEN_b vs null | -0.25 / -0.01 / -0.03; z -1.1 / -0.1 / -0.2 | -0.2500 / -0.0143 / -0.0325; z -1.055 / -0.111 / -0.174 |
| F5 | ccw / cw drift vs rests | +0.002 / +0.004; z 0.2 / 0.5; p 0.57 / 0.37 | +0.00167 / +0.00390; z 0.220 / 0.514; p 0.5697 / 0.3677 |

**All 15 predeclared members `null`, none called** (`result` requires `|z| >= 3` and `p_holm <= 0.05`). OK
The exact-U floor is `p_floor(4,4) = 2/70 = 0.028571` OK (F5 is 4 v 8, floor 0.00404, so F5 was genuinely callable and
came back null). Holm sizes as declared: m = 4 / 3 / 3 / 3 / 2. OK

Secondary rows, all recomputed:
* **AN04B003** flip -11.17 +- 0.79 (glu 2/15) / -11.80 +- 1.12 (glu shipped gains) / -11.52 +- 0.90 (shipped 2/15) /
  -11.45 +- 0.51 (shipped, shipped gains); z -9.86 / -14.67 / -10.44 / -17.35; p **0.02857 = the floor** in all four;
  verdict `result` in **4/4 arms**. OK exactly as printed ("-11.2 / -11.8 / -11.5 / -11.5", "z -10 to -17").
* **PS196_b** -0.61 +- 1.94 / -0.79 +- 0.94 / -0.89 +- 0.49 / -1.54 +- 1.26 Hz, `null` in 4/4. OK And its LR has the **same
  sign in both turn phases** (glu 2/15: rest +3.93, ccw +2.86, rest2 +3.89, cw +3.46), as claimed. OK
* **EPG** flips +0.011 / -0.036 / +0.450 / -0.012 -> `null`, max 0.45 Hz OK ("<= 0.45 Hz").
* **DNa02** +37.79 / +37.82 / +37.04 / +37.79 Hz, `result` in 4/4 (the stimulus). OK
* **So "the signed self-turn report stops at AN04B003" is confirmed**: AN04B003 is the only non-stimulus type called in
  any arm; PS196_b, GLNO, PEN_a, PEN_b and EPG are `null` at both gain settings, in both caches.
* Drift at the experiment gains: glu +0.0015 / +0.0022 / -0.0004 / +0.0044 w/s, every run in [-0.0088, +0.0137];
  shipped -0.0009 / -0.0011 / +0.0013 / +0.0050, every run in [-0.0082, +0.0071]. OK exactly. Heading ccw
  +91.9 ... +106.2, cw -98.6 ... -78.9 deg/s OK ("+92..+106 / -79..-99"). Bump `vs` 0.755-0.766 OK, peak 252 -> 253 OK.
* Rates at rest, 2/15, shipped -> glutamate: GLNO 133.15 -> 116.03 (-12.9 %), PEN_a 44.91 -> 38.77 (-13.7 %),
  PEN_b 50.70 -> 41.99 (-17.2 %), EPG 55.35 -> 50.81 (-8.2 %), Delta7 100.00 -> 94.98 (-5.0 %), PEG 33.29 -> 29.11
  (-12.6 %), FB4Y 84.17 -> 57.66 (-31.5 %), FB1C 19.12 -> 6.25 (-67.3 %), ExR8 0.43 -> 0.00. OK every figure.
  ("PEN -15 %" is the pooled figure, -15.5 %; the per-type figures -13.7 and -17.2 straddle the "-14..-16 %" band the
  audit says it reproduces from `cx_glno.md` 5, whose own per-point PEN values are -14.4 / -15.7 / -16.2 %. Fair as a
  pooled statement, loose as a per-type one.)
* **GLNO's L-R.** shipped 2/15: +26.91 / +27.14 / +26.89 / +26.96 at rest / ccw / rest2 / cw, GLNO_L 146.6 / GLNO_R
  119.7 OK; glutamate 2/15: **+3.12 / +4.14 / +4.48 / +3.30**, 117.6 / 114.5 OK; at the shipped gains -0.21...+0.18
  (shipped) and -0.14...+0.07 (glu) OK. PEN_b L 39.6 / R 61.8 shipped OK.
* **"is that right?" -- yes, the L-R is a standing asymmetry of the pinned protocol, not a turn signal**, and I can add
  the structural reason. (i) It is present in **both rest phases** at the same size and does not change sign with turn
  direction; (ii) the turn-contrast statistic on it is null in every arm (shipped +0.18 +- 0.31, glu +0.84 +- 0.79 against
  a null of +1.36 +- 1.45); (iii) it exists **only under the pulsed instrument** -- at the shipped gains, with no
  background and no pulse and therefore no bump, GLNO L-R collapses to +-0.2 Hz in both caches. The mechanism the audit
  gives is correct and I verified its wiring premise: the pulse sits on wedges 0-3, PEN_b fires R 61.8 > L 39.6, GLNO_L
  reads *only* the somaSide-R PENs (42 edges, 1,938 raw syn) and so runs hotter than GLNO_R. The relabel shrinks it
  because the now-negative PEN -> GLNO -> PEN loop and the 400-synapse GLNO <-> GLNO term compress a rate difference, not
  because any signal changed. (Scale: "88 % smaller" is the rest-phase figure; over the four phases it is **86 %**.)

#### 2e. Claim (5), and the provenance chain (e)

* **Predeclaration before the runs**: stamped 05:35:08Z, earliest job `started_utc` 05:35:33Z. OK (but see R7 on the
  "05:35:20Z" figure).
* **Reducer identity**: `tree_state.json` records sha256(LF) for 53 files; I recomputed all 53 against the working tree
  today -- **0 changed**, `scripts/glno_relabel.py` included. The analysis ran at 05:54:44Z with the same reducer as at
  the stamp. OK
* **Per-run provenance**: all 22 JSONs carry `NVIDIA B200`, `problems []`, the arm's md5 (`ef23cc27` / `7a10d93b`) and
  GLNO nt / sign read back from the cache the run used; the 16 compass runs carry `gains [2.0, 15.0]` x8 / `null` x8,
  `proprioception "all+leg_cycle"`, `leg_cycle true`, `mn_ref_hz 30.0`, background 10 / pulse 50 / adaptation 0,
  `dna02_hz 20`, `rate_dps 90`, `seconds 10`, `skip_s 3`, `sparse warp`, `probe_sha256_lf dbd8b115...`. OK `22 job(s),
  0 failed (18.5 min)` OK; artefact counts 6 + 16 + 64 + 128 as planned OK.
* **The working-tree probe -- arm C is identical under both versions. Confirmed by my own diff, not by the audit's
  assertion.** `ARMS_LEVEL` is byte-identical in HEAD and the working tree (`"C": "all+leg_cycle"`), and
  `FAMILIES["level"]` is `(ARMS_LEVEL, ARM_LABEL_LEVEL, "ALCD")` in both. The only behavioural changes on the compass
  path are: (i) `mn_ref_of(...)` -> `sense_kwargs_of(...)`, which for family `level` arm C returns `{}` because
  `ARM_MN_REF["level"]` has only the `"L"` key and `ARM_SENSE_KW` has only a `"level2"/"K"` key -- identical to HEAD's
  behaviour of passing nothing; (ii) `attach_cycle` gains a `body.LegCycle(flat_amplitude=True) if
  sense.leg_cycle_flat else body.LegCycle()` branch, and `leg_cycle_flat` is off. Everything else in the diff is
  `sense` / `started_utc` / `finished_utc` / top-level `mn_ref_hz` recording. In `flyverse/senses.py` the only new
  behavioural line is `if self.unsided: lL = lR = 0.5 * (lL + lR)` (and `unsided` + `leg_cycle` is a hard error); in
  `flyverse/body.py` it is `if self.flat_amplitude:` guarding a default-False field. **Arm C's numerical path is
  therefore byte-equivalent to HEAD's.** OK
* **No other thread's opt-in was active.** Every compass run JSON's `sense.tokens` reads
  `{haltere_coriolis: false, leg_cycle: true, haltere_sided: false, unsided: false, leg_cycle_flat: false}`,
  `overrides {}`, `defaults_used` all true (mn_ref 30.0, hair_plate_max 100.0, campaniform_load 50.0), and
  `leg_cycle_params.flat_amplitude: false`. OK
* **The wording "ADOPTABLE by the suite half of the round-2 rule ... the room rate-half not run ... the source standard of
  existing entries not met" is correct** (see section 4 below).

#### 2f. Claim (f) -- the 5A cross-check in section 4.3b

Consistent on every point I could check.
* 5A's scratch cache `out/cache_c51b23e2` does carry the same compiled W (md5 `7a10d93b`; in fact byte-identical npz)
  and its `TYPE_NT_OVERRIDE.json` is {..., GLNO: glutamate}. OK
* 5A section 0: "at the shipped gains (gE 1 / gD 1) the shipped path, GLNO = glutamate ... all score `bump_survival_s`
  0.00 in 4/4 seeds"; "GLNO fires 0.1 Hz while PEN is silent"; row (a) "GLNO fires 0.03-0.23 Hz". OK quoted correctly.
* 5A: "R (gE 2 / gD 15) 201-203 Hz in 4/4, RG 181-184 in 4/4" OK quoted correctly; and consistent with `cx_glno.md` 5's
  199.2 +- 4.5 -> 180.4 +- 3.0 (-9.4 %).
* 5A's FG collapse ("with GLNO = glutamate the same configuration (FG) loses the bump within 0.15 s in 4/4 seeds") is
  indeed the **damping-off** arm (`same_type_gain` 1) at gE 1 / gD 1, and 5A itself says CFG (sign+gain + damping off +
  relabel) restores it. So 5B's "the two results do not conflict" is right. OK
* 5A's section 0 and its section 3 row (a) do cite "the hemibrain name (GLutamatergic LAL-NOduli)" as one of three
  sources, and `evidence_glno.json` records it as a fact with no citation. **5B's carried-over correction is accurate
  and should stand.** OK
* 5B's own shipped-gains numbers (PEN 0.6-1.3 Hz, GLNO 3.4-3.5 Hz) reproduce exactly (PEN_b 0.58-0.61, PEN_a 1.18-1.26;
  GLNO 3.43-3.47). OK **And "inert" is defensible**: I tested every ring / target rate at the shipped gains 4 v 4 and
  every one is `null` -- EPG +0.245 (z 1.20), Delta7 +0.352 (z 1.04), PEG +0.264 (z 1.38), ExR8 -0.259 (z -1.45, Welch
  -2.07), GLNO -0.045, PEN_a -0.080, PEN_b -0.029, FB4Y -0.152, FB1C -0.187, PS196_b +0.196. Worth saying that "inert"
  means "no rate difference reaches `result` at 4 v 4", not "unchanged", since the mean shifts run to +-20-50 % on
  0.2-3.5 Hz rates.
* **One thing 4.3b should add**: 5A's arms run `receptor_model 'sign+gain'` in the C-family, and under `sign+gain` the
  glutamate and gaba labels are **not** the same model on GLNO's FB / EPG targets (R5). 5A's ring conclusions are
  unaffected (the PEN edges are identical under every mode), but the "bit-identical to the gaba cache" statement is a
  `receptor_model='sign'` statement and should not be carried into a `sign+gain` discussion unqualified.

---

### 3. Further findings not in the audit (not refutations)

1. **The project's own expectation ledger carries a GLNO row this relabel changes, and the audit never mentions it.**
   `flyverse/data/expected_responses.csv` has `struct.GLNO_PEN.sign` (population `GLNO->PEN_a`, `expected 0`, `op ==`,
   `bound 0`, `gap 1`, notes "known gap: the largest single input of PEN carries no sign and is silent in the model")
   and `struct.GLNO_PEN.synaptic_pair_count` (`expected 16371`, `range 15000,18000`). The relabel takes that sign from
   0 to -1, i.e. it is the documented deficit the ledger names. The row has no `check_key`, so it is not one of the 29
   and the suite rule cannot see it; its `model_reference` says MISSING until `interp_paths.py` emits the row. An
   adoption should re-score it (and, as written with `op == 0` and `gap 1`, the row's polarity looks inverted -- meeting
   `== 0` prints "PASS (gap closed)" for what is the gap; that is a ledger defect, not this thread's).
2. **`INTERP.md` 10.1(7)(b) was not satisfied at submission, and has since been satisfied retroactively.** The rule is
   that *"a cross-task dependency is committed before the batch is submitted -- `cluster_run.py` ships the working tree,
   so an uncommitted file from another task is in the run and in nobody's history"*. `flyverse/body.py`,
   `flyverse/senses.py` and `scripts/probe_vnc_drive.py` went to the box uncommitted. The audit records the sha256s, a
   419-line diff and a diffstat, which is the best available mitigation and more than 5A does, and I verified arm C is
   unaffected. **Update, mid-review:** the owning thread committed those files during this review as
   **`286dca3` "Round 4b: the three level controls (unsided, channel-matched, modulation-only) and the AN04B003
   single-cell check"**, on top of `0b3668f`; `scripts/probe_vnc_drive.py` still hashes `dbd8b115...`, i.e. the exact file
   the cx5b batch ran is now in history. The audit should cite `286dca3` as the code identity of the compass numbers
   (it was written when only the sha256 existed) and say plainly that the rule was met after the fact, not before --
   and name the owning thread, as 10.1(7)(b) requires.
3. **"PEN raw fan-in totals unchanged in range" (1.3) is loose.** Every PEN's capped raw fan-in rises by **exactly
   +120** (2 GLNO x cap 60): the range moves 867-1597.5 -> 987-1717.5. What is unchanged is that all 42 stay far below
   `input_norm_ref` 5000, so the scale stays 1.00 and the +120 buys no renormalisation.
4. **The md5 equality with `out/cache_72164311` has no artefact in `out/cx5b/`.** `validation` asserts it; nothing in
   the run directory records it. It is true (I verified array-level and byte-level equality), but a `compare --cache-dir
   out/cache_72164311` run would put it in a named file.
5. The three suite draws of an arm use the **same seeds 0,1,2**; the replicate unit is the job, which `INTERP.md`
   10.1(3) explicitly sanctions ("two same-code batches with the same seeds are not the same draws"). Worth one clause
   in 4.5 so a reader does not mistake the three draws for three seeds.

---

### 4. On the "adoptable" wording (g)

**The recommendation is phrased correctly**, and unusually carefully. Checking it against the two halves as the project
states them:

* **Suite half -- met, and the audit's rule is the stricter one.** Round 2 (`connectome.py` line 59): "adopt only if no
  suite check changes status". `guard_suites_r3.md` 4: ">= 3 draws with no check worse in status". The predeclared rule
  here forbids *any* status change in *any* glutamate draw relative to *every* shipped draw, which is round 2's form
  and implies round 7's. 3 + 3 draws, 174 rows, 0 changes, 0 unstable shipped rows, 0 provenance problems. OK The audit
  says exactly this and says "suite half only".
* **Rate half -- not run, and the audit says so twice (4.1 bold note, 4.3(b)).** `guard_suites_r3.md` 4 requires the
  room take-off protocol at **>= 6 runs per arm** for a callable verdict, two-sided, with the two-sample exact Poisson
  quoted beside the run-level `common.compare`. OK correctly cited.
* **Source standard -- not met, and the audit is the one that says so.** All three existing `TYPE_NT_OVERRIDE` entries
  name at least one transcriptome or EASI-FISH source (TmY14: Davie 2018 + FCA 2022 + Ozel 2021 + Nern 2025; Mi19:
  Nern 2025 EASI-FISH `validated_nt`; aMe8: Nern 2025 prediction). GLNO has none of those and no receptor-table row.
  The audit states this in 1.2, 1.3, 4.3(a), the Report summary and the recommendation, and names the honest label
  ("inhibitory (glutamate 2 of 3 classifiers, GABA 1 of 3)"). OK

**Exactly what is missing, stated compactly:** (1) an expression or EASI-FISH source for GLNO -- none exists in any of
the 11 local tables, and the two EM classifiers that call glutamate do so at 0.47-0.51 confidence while the third calls
GABA at 0.30-0.33; (2) the room take-off protocol at >= 6 runs per arm on `out/cache_glno_glu`; (3) a re-score of the
`struct.GLNO_PEN.sign` ledger row (above); (4) if the adoption is ever run under `sign+gain` or `full`, a re-run of
`compare`, because the two labels are *not* the same model there (R5) -- the audit's caveat 4 anticipates the right
thing for the wrong reason (it expects a future table row; the difference is already in the table's gain and slow
columns).

One framing point the audit gets right and is worth keeping in the entry text: because W is bit-identical to the gaba
cache under the shipped model, **the decision actually on the table is "sign GLNO -1", not "GLNO is glutamatergic"**.
All three EM classifiers agree on inhibitory; none of them settles which inhibitory transmitter; and the name that
would go in `TYPE_NT_OVERRIDE` is a tie-break the data do not make.

---

### 5. Corrections (exact replacement text)

**C1 -- section 4.1, the paragraph after `summary.json`. Replace**
> 17 of the 29 checks are bit-identical between the arms in every draw (the deterministic protocol sections: rest, taste, smell, dn, the walk triple, loom.escape, motion.correct_directions, loom_escape.escapes, calibrated bitter, compass); three deterministic sections move by a fixed amount in 3/3 draws -- `motion.min_dsi` 0.24110 -> 0.24064, `bitter.shiu_sugar_MN9` 139.90 -> 135.15 Hz, `bitter.shiu_sugar_bitter_MN9` 0.818 -> 0.988 Hz

**with**
> 16 of the 29 checks read the same value in all six draws (the deterministic protocol sections: rest, taste, smell, dn, the walk triple, loom.escape, motion.correct_directions, loom_escape.escapes, calibrated bitter, compass). "Identical to the printed digits on this GPU batch" is what that means: under the project rule (`guard_suites_r3.md` 5) bit-identity is a CPU claim and nothing here is one. Three checks move by a fixed amount in 3/3 draws -- `bitter.shiu_sugar_MN9` 139.8985 -> 135.1501 Hz and `bitter.shiu_sugar_bitter_MN9` 0.81784 -> 0.98795 Hz, both constant within each arm, and `motion.min_dsi` 0.2410966 -> 0.2406374 (-0.19 %), whose within-arm scatter is ~2e-7, i.e. 2,500x smaller than the shift but not zero

**C2 -- Report `key_claims`, the suite line. Replace** `17 checks bit-identical, three deterministic sections move`
**with** `16 checks identical to the printed digits in all six draws, three checks move by a fixed amount (two of them constant within each arm)`.

**C3 -- section 1.3 table, the last row. Replace**
> | FB1H 2 / 21, LNO2 3 / 11, AN27X013 2 / 8, PFNa 6 / 8, PVLP060 1 / 6, PFNd 4 / 6, LCNOpm 3 / 6, DNpe023 2 / 5, and 44 further types at 1-4 synapses (EPG 2 / 2, PEG 2 / 2, PS196_b 2 / 3, ER6 1 / 2, ExR1/2/4/6 1-2 each ...) | 62 | 96 | 0.5 % |

**with**
> | FB1H 2 / 21, LNO2 3 / 11, AN27X013 2 / 8, PFNa 6 / 8, PVLP060 1 / 6, PFNd 4 / 6, LCNOpm 3 / 6, DNpe023 2 / 5, and 43 further types at 1-4 synapses (EPG 2 / 2, PEG 2 / 2, PS196_b 2 / 3, ER6 1 / 2, ExR1/2/4/6 1-2 each ...) | 76 | 139 | 0.8 % |

(58 postsynaptic types in all; the columns now sum to 213 entries and 17,698 synapses. The 8 named types are 23 entries
/ 71 synapses; the 43 others are 53 entries / 68 synapses.)

**C4 -- Report `key_claims`, the "Elsewhere" line. Replace** `53 further types at <= 21 syn` **with**
`51 further types at <= 21 syn (58 postsynaptic types in all)`.

**C5 -- section 1.4, last two sentences. Replace**
> The runs below therefore install exactly `NT_SIGN`'s -1 on every GLNO synapse, and the glutamate / gaba choice is invisible to the present receptor table (it would become visible only if a PEN or FB row with an excitatory glutamate call were added).

**with**
> The runs below therefore install exactly `NT_SIGN`'s -1 on every GLNO synapse: under the shipped `receptor_model 'sign'` the two labels resolve to the same `fast_sign` on all 25,578,600 entries, not only the 213, so the suite and compass arms are the gaba cache's model exactly. The table is not blind to the difference, though: under `sign+gain` or `full` the resolved gain and slow columns differ on the same entries -- `fast_gain` on 35 of the 213 (FB1C 14, FB4M 11, FB2A 4, ExR2 2, FB1H 2, ER4m 1, OA-VUMa1 1: glutamate `high` x1.5 against gaba `mid` x1.0), `slow_sign` on 3 (EPG x2, ER4m: glutamate 0 against gaba -1) and `slow_gain` on 57. Onto PEN the two labels are identical under every mode (the `PEN_a(PEN1)` / `PEN_b(PEN2)` glutamate and gaba rows carry the same fast sign, the same `high` gain class and the same slow columns; only the lead gene differs, GluClalpha against Rdl), which is why the ring result is label-free. The table already holds 160 types with an excitatory glutamate `fast_sign` under the class rule (14 under `abs`) and four FB targets whose glutamate net is `mixed`; none of them is a GLNO target, which is the actual reason the 213 come out -1.

**C6 -- Report `key_claims`, the receptor line. Replace**
> "Receptor model (sign/abs and sign/class): 213 / 213 entries -1, 0 entries +1 -- glutamate and gaba are indistinguishable on the present table."

**with**
> "Receptor model (sign/abs and sign/class): 213 / 213 entries -1, 0 entries +1; `fast_sign` is identical between the glutamate and gaba caches on all 25,578,600 entries, so under the SHIPPED receptor model the two labels are the same model. Under `sign+gain` / `full` they are not: the resolved `fast_gain` differs on 35 of the 213 and `slow_sign` on 3 (never onto PEN, where the two labels are identical under every mode)."

**C7 -- section 4.2, the rates paragraph heading. Replace**
> **Rates at rest at the experiment gains (mean of L and R over 4 runs, Hz; never silent for a firing cell).**

**with**
> **Rates at rest at the experiment gains (mean of L and R over 4 runs, Hz).**

and after `ExR8 0.4 -> 0.0` **insert**
> -- ExR8 is the one type the relabel silences outright: 0.00 Hz on both cells in all four runs (max single cell 0.00), against 0.86 / 0.00 Hz and a 2.14 Hz max cell on the shipped cache. It takes 435 of GLNO's synapses on 8 entries, GLNO's third-largest target.

**C8 -- section 2 heading. Replace**
> ## 2. Design (predeclared; `out/cx5b/predeclared.json`, stamped 2026-09-15T05:35:08Z, before the 05:35:20Z submission)

**with**
> ## 2. Design (predeclared; `out/cx5b/predeclared.json`, stamped 2026-09-15T05:35:08Z -- the same stamp `submit_tree.txt`, `tree_state.json` and `jobs.json` carry; the earliest scheduler evidence is the first job's `started_utc` 05:35:33Z, so the predeclaration precedes every run. This batch has no `scheduler_receipt.json`; `out/cx5b_cluster.log` records elapsed seconds, not wall clock.)

**C9 -- the "Read first" paragraph, section 0. Replace**
> section 3 for the round-2 finding that a glutamatergic / GABAergic GLNO abolishes the bump at gE 1.75 in 6/6 seeds

**with**
> section 3 for the round-2 finding that a glutamatergic / GABAergic GLNO abolishes the bump at gE 1.75 / gD 15 in 3/3 seeds (the only six-seed row at gE 1.75 is gD 8, where `gaba` is confined in 4/6 and persists in 3/6)

**C10 -- section 1.3, the PEN-edge parenthetical. Replace**
> (`LIFParams` defaults, w_syn 0.275 mV, conn_cap 60, fan-in scale 1.00 on every PEN in both caches, PEN raw fan-in totals unchanged in range)

**with**
> (`LIFParams` defaults, w_syn 0.275 mV, conn_cap 60; every PEN's capped raw fan-in rises by exactly +120 = 2 x cap, 867-1597.5 -> 987-1717.5, which leaves all 42 far below `input_norm_ref` 5000, so the fan-in scale stays 1.00 in both caches)

**C11 -- section 4.1 and 4.5, on the two directional rows. Append to caveat 2**
> Both directions are quoted as z = diff / SD(shipped arm); the two-sample statistic is smaller because the glutamate arm scatters wider -- Welch 2.09 for `loom_escape.GF_peak` (SDs 4.91 against 1.40) and 2.72 for `wind.DNp18`. Quote the Welch beside the z so the direction is not read as firmer than the scatter.

**C12 -- section 4.4 "What remains", add a bullet**
> * The expectation ledger: `struct.GLNO_PEN.sign` (`flyverse/data/expected_responses.csv`, `expected 0`, `op == 0`, `gap 1`, "the largest single input of PEN carries no sign and is silent in the model") is the row this relabel is about, and the 29-check suite does not carry it (no `check_key`). An adoption must re-score it; as written its polarity also looks inverted (meeting `== 0` prints "PASS (gap closed)" for what is the gap).

**C13 -- section 2, the working-tree paragraph, after "...arm C of family `level` is `all+leg_cycle` in both versions". Insert**
> The independent check is that the only behavioural lines the diffs add on arm C's path are guarded off: `sense_kwargs_of("C", "level", args)` returns `{}` (the `ARM_MN_REF` table has only the `L` key and `ARM_SENSE_KW` only `level2/K`), `attach_cycle` builds the default `body.LegCycle()` because `leg_cycle_flat` is off, `senses.py`'s one new branch is `if self.unsided:` (refused together with `leg_cycle`) and `body.py`'s is `if self.flat_amplitude:` on a default-False field; every run JSON's `sense.tokens` records `unsided false, leg_cycle_flat false, haltere_sided false` and `leg_cycle_params.flat_amplitude false`. `docs/INTERP.md` 10.1(7)(b) requires a cross-task dependency to be COMMITTED before submission; it was not at 05:35:08Z. The owning thread has since committed exactly these files as `286dca3` ("Round 4b: the three level controls ... and the AN04B003 single-cell check"), and `scripts/probe_vnc_drive.py` there still hashes `dbd8b115...` -- the file the batch ran. Cite `286dca3` as the code identity of the compass numbers and say the rule was met after the fact.

**C14 -- 4.3b, add to the paragraph on 5A.** Append
> (iv) 5A's C-family arms run `receptor_model 'sign+gain'`, where glutamate and gaba are NOT the same model on GLNO's FB / EPG targets (section 1.4): 35 of the 213 entries differ in resolved gain there. 5A's ring conclusions are unaffected -- the 84 PEN edges are identical under every receptor mode -- but the "bit-identical to the gaba cache" statement is a `sign`-mode statement and should be qualified when it is carried into a `sign+gain` discussion.

---

### 6. Verdict

**Mostly sound.** Claims (1)-(5) all hold. The entry-by-entry cache comparison reproduces exactly under an independent
implementation, including the bit-identity with the round-3/4 gaba cache (which is in fact byte-identical at the npz
level and is the same file thread 5A used); the -16.50 mV per PEN edge and -33.00 mV per volley are exact, not
rounded; every source number in the evidence case matches its primary file, and the "the name is not a source"
finding survives an independent web check. The suite's 174 status rows re-evaluate from the raw measurements with zero
mismatches, `changed_pooled` / `changed_matched` / `unstable_shipped` are all genuinely empty, and "underpowered" is
the tool's own verdict at a p-floor of 0.10, used correctly. All 15 predeclared compass members reproduce to three
decimals under my own flip statistic and all 15 are null; AN04B003 is `result` at the exact-U floor in 4/4 arms and
nothing downstream of it is, so "the report still stops at AN04B003" is exactly right; the GLNO L-R reading -- a
standing asymmetry of the pinned protocol's bump position, not a turn signal -- is correct and I confirmed its wiring
premise (GLNO_L reads and writes only the somaSide-R PENs, which fire 61.8 against 39.6 Hz under the wedge-0-3 pulse).
The provenance chain holds: the predeclaration precedes every run, the reducer is unchanged from stamp to analysis,
arm C of family `level` is byte-equivalent under the shipped working tree and HEAD, and no other thread's opt-in token
was active in any of the 16 runs. What comes off is secondary: one off-by-one bit-identical count, a summary table
that does not sum to its own totals, one mislabelled "deterministic" row, a receptor-model gloss that is false outside
the shipped mode, a "never silent" hedge that hides ExR8's silencing, an unsourced submission timestamp, and one
reading-list gloss that overstates `cx_glno.md`. None of these touches the adoption question, and the adoption wording
itself -- suite half met under the round-2 rule, rate half not run, source standard not met, "sign GLNO -1" being the
real decision -- is the most careful part of the document.

**What rounds 5A and 5B jointly say about the compass and GLNO.** Taken together the two threads turn the GLNO sign
from an open question into a `null` with a known mechanism, and leave the compass's real failure where 5A found it.
5A establishes that at the shipped gains no type-level change reachable from the data -- the GLNO relabel included --
makes the ring hold a bump, because PEN sits 11-16 mV below threshold under ExR6 / ER6 and the relabel is therefore
inert (`bump_survival_s` 0.00 in 4/4 seeds, a structural zero-SD null); the only configuration that carries a bump at
those gains is the removal of a global hand rule, which is a labelled instrument, not a mechanism, and which the
correct GLNO sign then destroys. 5B runs the same candidate on the one instrument where a signed GLNO can act (gE 2 /
gD 15, body attached, leg cycle on, DNa02 driven) and finds the effect is entirely on rates -- ring -5 to -17 %, FB4Y
-31 %, FB1C -68 %, ExR8 silenced, GLNO's standing L-R compressed 86-88 % -- with every predeclared flip and the bump
drift `null` at 4 v 4, and with the signed self-turn report still terminating at AN04B003 (`result`, -11 Hz, p at the
0.0286 floor, 4/4 arms) and `null` at PS196_b, GLNO, PEN and EPG. The joint reading is therefore: GLNO's transmitter is
a **rate** parameter of the ring, not a **signal** parameter, and it is a parameter the fast model cannot even name,
since glutamate and gaba produce a bit-identical W and, under the shipped receptor model, an identical model
everywhere. The compass gap (`compass.wedge_cells_persisting` 0, KNOWN GAP in all six draws of both arms) is untouched
by it in both threads, and the self-turn report's break is upstream, at a PS196_b whose input to GLNO is symmetric in
both turn directions -- which is a body-model question, not a transmitter one. On the decision itself the two threads
now agree with one correction: the adoption is **adoptable by the suite half of the round-2 rule and by nothing else**,
the room rate-half is unrun, the source standard of the three existing `TYPE_NT_OVERRIDE` entries is unmet, and the
"hemibrain name" 5A counted as one of three sources is not a source at all -- leaving two low-confidence EM classifiers
calling glutamate and a third calling GABA, all three agreeing only that GLNO is inhibitory.
