# Spec: a second (and third) connectome backend -- FlyWire FAFB v783 and BANC v888

Status: specification for implementation (2026-09-14). Companion to `docs/audits/flywire_banc_survey.md` (what the
two female releases contain and how they read against MaleCNS) and `docs/EXTENSIBILITY_SPEC.md` (the same shape of
document that preceded the extensibility layer). The project rule applies throughout: **the shipped MaleCNS path
stays bit-identical**; nothing here tunes a default.

## 0. Goal

`flyverse.connectome.load(dataset="malecns" | "fafb" | "banc")` returns a `Connectome` that every downstream
component -- `Brain`, `OpticLobe` + `Retina` (where columns exist), `senses`, `motor`, `regions`, the receptor model,
`flyverse.interp`, the benchmark ledger -- consumes unchanged, because the *contract* of the `neurons` table and `W`
is met by every backend. Then:

- **BANC** gives the female brain + VNC: the walking / turning / compass results of rounds 1-3 rerun on an
  independent reconstruction of a different animal, sex and lab. Either outcome is a finding.
- **FAFB** gives the complete female optic lobe with an independent column map: the object / size-tuning work reruns
  there, and the column map tests our own.
- A cross-connectome anatomy table (`scripts/cross_connectome.py`) makes every anatomical claim of rounds 1-3 a
  three-release statement.

## 1. The contract every backend must meet

### 1.1 `neurons` (pandas, one row per cell, row index = matrix index)

| column | dtype | MaleCNS source | required by | FAFB source | BANC source |
|---|---|---|---|---|---|
| `bodyId` | int64, unique | `bodyId` | everything (`body_to_index`, provenance, `extend` reserves negatives) | `root_id` (7.2e17 < 2^63: store as is) | `Root ID` |
| `type` | str / NaN | `type` (Nern 2025 / MaleCNS names) | receptor table, ledger, `optic.DEFAULT_TAU_BY_TYPE`, `T4T5`, `motor` selects, every audit | `consolidated_cell_types.primary_type` **normalised to the MaleCNS name** through `flyverse/data/type_aliases.csv` (keep the source name in `flywireType`) | `Primary Cell Type`, same normalisation |
| `instance` | str | `instance` (`type_L` / `type_R`) | `senses` (haltere side suffix), `motor.read_haltere_sides` | synthesise `f"{type}_{side}"` | same |
| `superclass` | str | `superclass` | `regions`, `motor`, `senses`, `optic` (`ol_intrinsic`), `interp` | map `super_class` (section 2.1) | map `Super Class` |
| `class` | str | `class` | `senses` (`mechanosensory_proprioceptive`) | map `class` / `sub_class` | map `Class` / `Sub Class` |
| `subclass` | str | `subclass` | `senses` (`chordotonal organ`, `leg`, `hair plate`, `campaniform sensilla`, `haltere`), `motor` (`wm`, `hm`) | derive (section 2.3) | derive from `Class` / `Body Part` |
| `somaSide` | `L` / `R` / `M` / NaN | `somaSide` | everything sided | `side` (`left`->`L`, `right`->`R`, `center`->`M`) | `Soma side` |
| `somaNeuromere` | str / NaN | `somaNeuromere` | provenance only | NaN | derive from `Nerve` / neuropil if cheap, else NaN |
| `status` | str | `status` (`Traced` filter) | compile filter only | `Traced` for every row of the release | same |
| `entryNerve`, `exitNerve` | str / NaN | annotations | `senses` (`LEG_NERVES`) | `nerve` (brain nerves only) | `Nerve` mapped (section 2.4) |
| `flywireType`, `hemibrainType`, `mancType` | str / NaN | annotations | provenance, aliasing | `primary_type` verbatim; others NaN | `Primary Cell Type` verbatim; `Alternative Cell Type(s)` |
| `assignedOlHex1`, `assignedOlHex2` -> `hex1`, `hex2`, `hex_side`, `hex_source` | float / str | optic-lobe column annotation + `_assign_photoreceptor_columns` | `Retina`, `OpticLobe`, `interp.trace.column_of_cells`, the atlas | `column_assignment` (section 2.5) | **none by default** -- the biological release has no column map; `hex*` all NaN, `Retina` / `OpticLobe` unavailable. Explicit `vision="candidate"` adds experimental right-eye coordinates and synthetic R1-R6; see section 1.4. |
| `nt` | str in `TRANSMITTERS` + `unknown` (+ `tyramine`, new) | consensus > type prediction > body prediction | `NT_SIGN`, receptor model, slow term, fingerprint | section 2.2 | section 2.2 |
| `sign` | float32 | `NT_SIGN[nt]` | `W` | same rule | same rule |
| `in_syn`, `in_syn_l2` | float | computed | fan-in normalisation | computed | computed |

Anything downstream that reads a column not in this table is a bug to fix in that consumer, not a column to add.

### 1.2 `W`

CSR, shape (N, N), **rows = postsynaptic, columns = presynaptic**, float32, value = `sign(pre) * synapse count`,
`min_weight` 1 after summing the release's per-neuropil rows per (pre, post) pair. **Sign-0 presynaptic cells keep
explicit zero entries** (`W.data == 0`): `sign0_counts` and the monoamine slow term read them; do not `eliminate_zeros`.
FlyWire's shipped tables are pair-thresholded (FAFB >= 5 synapses per pair, BANC >= 3); record the threshold in the
cache manifest, and offer `--edges no_threshold` for FAFB (22.3 M rows) as an option, off by default.

### 1.3 Cache and provenance

- `cache/` stays the MaleCNS cache (bit-identical: same files, same md5). New datasets compile into
  `cache/<dataset>/` (or `$FLYVERSE_CACHE/<dataset>/`) with the same file set (`neurons.parquet`, `W_post_pre.npz`,
  `sign0_counts.npz`, `extension*`) plus `manifest.json` (dataset, release version, source file sha256s, pair
  threshold, alias table sha256, NT rule, compile date).
- `Connectome.dataset` (str) and `Connectome.release` (str) attributes; `interp.common.connectome_fingerprint` and
  `provenance()["model"]` carry both. Every Result JSON produced on a non-MaleCNS graph says so.
- Data location: `FLYVERSE_DATA_FAFB` / `FLYVERSE_DATA_BANC` environment variables, defaulting to
  `D:\Datasets\flywire\Female Adult Fly Brain v783` and `...\BANC v888`; `scripts/fetch_data.py --fafb / --banc` with
  the public URLs and sha256s added to `flyverse/data/manifest.json` (the tables are public; the 13.9 GB skeleton
  archive is not fetched).

### 1.4 API

```python
c = connectome.load(dataset="banc")                       # compiles on first call, then reads cache/banc/
c = connectome.load(dataset="fafb", edges="no_threshold")  # option recorded in the manifest and fingerprint
c.dataset, c.release                                        # "banc", "v888"
connectome.load()                                           # unchanged: MaleCNS, cache/, bit-identical
```

Subsequent owner-authorised experiment: `connectome.load(dataset="banc", vision="candidate",
vision_cache_dir=<unused scratch directory>)` appends synthetic R1-R6 through `Connectome.extend`
and enables a pinned right-eye candidate map. `vision=None` retains the source release; the optional
`vision_cache_dir` requires the candidate opt-in and otherwise defaults to a caller-owned temporary
directory. `cache_dir` continues to name the biological input cache. `has_vnc` describes the source
release; `has_optic_columns` may also be enabled by this explicit candidate extension, and both retain
their meaning on subsets. Pruning all synthetic nodes restores the original graph and its unavailable
optic capability. No biological synapse changes, and the left eye remains unavailable.

This is a **synthetic input layer on a candidate lattice**, not a validated BANC annotation. Its estimated
80-85% exact-column accuracy and failed anatomical gates travel in Result provenance. Functional motion
and loom acceptance is qualified by an inherited orientation convention, excess synthetic input budgets
and a more complete/wider eye than the MaleCNS comparator. See
[CONTROL_SURFACE.md](CONTROL_SURFACE.md#connectome-datasets) and the
[experiment audit](audits/banc_candidate_experiment.md) for controls, caveats and the pinned packaged assets.

`compile_connectome` becomes a thin dispatcher over `backends/malecns.py` (the current code, moved verbatim),
`backends/fafb.py`, `backends/banc.py`, each returning `(neurons, W_raw_pairs)` in the contract; the shared tail
(NT sign, overrides, photoreceptor columns, `in_syn`, save) stays in `connectome.py`. The MaleCNS-specific
constants (`ANNOT_FILE`, `NT_FILE`, `WEIGHTS_FILE`, `UNKNOWN_NT_OVERRIDE_REGEX`, `TYPE_NT_OVERRIDE`) apply to
MaleCNS only; a backend declares its own override tables (initially empty) so nothing MaleCNS-tuned leaks into a
female graph.

## 2. Vocabulary maps (small, explicit, tested)

### 2.1 superclass

MaleCNS values: `ol_intrinsic`, `cb_intrinsic`, `vnc_intrinsic`, `visual_projection`, `vnc_sensory`, `ol_sensory`,
`cb_sensory`, `ascending_neuron`, `descending_neuron`, `vnc_motor`, `visual_centrifugal`, `sensory_ascending`,
`cb_motor`, `vnc_efferent`, `cb_endocrine`, `ENS`, `vnc_endocrine`, `sensory_descending`, ...

| FAFB `super_class` | BANC `Super Class` | -> MaleCNS |
|---|---|---|
| `optic` | `optic_lobe_intrinsic` | `ol_intrinsic` (photoreceptors: `ol_sensory`) |
| `central` | `central_brain_intrinsic` | `cb_intrinsic` |
| -- | `ventral_nerve_cord_intrinsic` | `vnc_intrinsic` |
| `visual_projection` | `visual_projection` | `visual_projection` |
| `visual_centrifugal` | `visual_centrifugal` | `visual_centrifugal` |
| `sensory` (brain) | `sensory` + `Body Part` / `Nerve` | `cb_sensory` if the nerve enters the brain, `vnc_sensory` if a VNC nerve, `ol_sensory` for retina / interommatidial |
| `ascending` | `ascending` | `ascending_neuron` |
| `descending` | `descending` | `descending_neuron` |
| `sensory_ascending` | `sensory_ascending` | `sensory_ascending` |
| `motor` (brain) | `motor` + nerve | `cb_motor` if brain nerve, `vnc_motor` if VNC nerve |
| `endocrine` | `visceral_circulatory` | `cb_endocrine` / `vnc_endocrine` by location |
| -- | `glia`, `not_a_neuron`, `trachea` | **drop the row** |

`regions.py` reads `superclass` prefixes (`vnc_`, `ascending`, `ol_intrinsic`) -- with the map above it needs no change.

### 2.2 transmitter

Vocabulary: `ACH`->`acetylcholine`, `GABA`->`gaba`, `GLUT`->`glutamate`, `HIST`->`histamine`, `DA`->`dopamine`,
`OCT`->`octopamine`, `SER`->`serotonin`, `TYR`->`tyramine` (**new** `NT_SIGN` entry, 0.0 -- a monoamine; no MaleCNS
cell carries it, so the MaleCNS graph is unchanged), NaN->`unknown`.

- FAFB: `neurons.nt_type` when `nt_type_score >= 0.5`, else `unknown` (the per-class probabilities are kept in a
  side table `nt_scores.parquet` for the NT audit; the threshold is a backend parameter recorded in the manifest).
- BANC: **verified first**: `Verified NT type` when present -- for a co-transmitter string take the first *classical*
  entry (`acetylcholine` / `gaba` / `glutamate` / `histamine`) for `nt`, else the first monoamine, and keep the whole
  string in a new column `nt_verified`; otherwise `Predicted NT type`. Nitric oxide, neuropeptides are dropped from
  `nt` (kept in `nt_verified`).
- Photoreceptors are `histamine` in every backend (the MaleCNS rule).
- **No MaleCNS override table is applied to a female graph.** The survey's disagreements (PFL3 tyramine-predicted in
  BANC, Delta7 `glutamate,serotonin`, LAL074) go into `docs/NT_INTEGRATION.md` as conflict rows, not into code.

### 2.3 class / subclass for `senses` and `motor`

`senses.Proprioception` selects `class == mechanosensory_proprioceptive` and `subclass` in {`chordotonal organ`,
`leg`, `hair plate`, `campaniform sensilla`, `haltere`} with `entryNerve` in `LEG_NERVES`; `motor` selects
`superclass == vnc_motor` with `subclass` `wm` (wing) / `hm` (haltere) and a type regex for the steering MNs, plus
`MN9`. BANC gives `Class` (`chordotonal_organ_neuron`, `bristle_neuron`, `hair_plate_neuron`?, `campaniform_sensillum_neuron`?
-- enumerate the actual values), `Body Part` (`haltere`, `front_leg`, ...), `Function` (`proprioception`, ...) and
`Nerve`. Build `subclass` from those with an explicit table in `backends/banc.py`, and make the mapping's cell
counts part of the acceptance test (MaleCNS: chordotonal 425 + leg 868, hair plate 113, campaniform 426, haltere
205, wm 67, hm 16). FAFB has no VNC: those selections return empty sets and `senses` / `motor` must raise a clear
`NotAvailable("dataset fafb has no VNC")` rather than a silent empty group.

### 2.4 nerves

MaleCNS `entryNerve` codes (`ProLN`, `MesoLN`, `MetaLN`, `ProAN`, `VProN`, `DProN`, `ProCN`, `ADMN`, `DMetaN`, `AN`,
`MxLbN`, `AbN1-4`, ...) <-> BANC `Nerve` strings (`left_prothoracic_leg_nerve`, `right_anterior_dorsal_mesothoracic_nerve`,
`abdominal_nerve_trunk`, ...): a table, side stripped into `somaSide` where MaleCNS has none.

### 2.5 optic-lobe columns (FAFB only)

MaleCNS: `hex1` 1..36, `hex2` 1..39, ~880 columns per side, with `retina.py`'s conventions (`hex1 + hex2` increases
dorsally, `hex1 - hex2` anteriorly; the medulla is chiasm-inverted). FAFB `column_assignment`: `hemisphere`,
`column_id`, `x`, `y`, `p`, `q` for 45,528 cells of 31 types (`L1-L5`, `C2/C3`, `Mi1/4/9`, `T1-T5*`, `Tm1/2/3/4/9/20/21`,
`R7`, `R8`), ~790 columns per side. Required: an affine map `(p, q) -> (hex1, hex2)` per hemisphere, chosen so that
the two `retina.py` axis rules hold, **validated** by (i) the DRA photoreceptors (`R7d`-equivalents) landing on the
dorsal rim, (ii) the L/R mirror, (iii) T4a-d preferred directions (the four subtypes' column offsets must reproduce
the MaleCNS layout in `optic.py`'s T4/T5 wiring), (iv) `Retina.n_columns` ~ 1,580. Photoreceptors R1-R6 are not in
`column_assignment`: assign them exactly as MaleCNS does (`_assign_photoreceptor_columns`, strongest hexed
postsynaptic partner), which is the existing shared code path. Cells of the 31 types get `hex_source = "annotation"`.

BANC: no columns. `load(dataset="banc")` yields a graph on which `FlyBrain(optic=None)` runs; `build_retina` raises
`NotAvailable`. This is acceptable for the walking / turning / compass replicate (those experiments run in darkness
or with the retina blank anyway -- check each ledger row's `retina.mode`).

## 3. Type-name normalisation

`type_aliases.csv` (22,200 rows: `malecns_type, alias, system, tier, flag, evidence`) maps FlyWire / hemibrain /
MANC names to MaleCNS names. The backend applies `alias -> malecns_type` for `system in (flywire, ...)` on the
`exact` and `alias` tiers, records the source name in `flywireType`, and leaves unmapped names as they are. The
survey's notational cases (`PS196a` -> `PS196_a`, `PEN_a/PEN1` -> `PEN_a`, `DNg02_a..e` -> `DNg02`?) must be decided
one by one and added to the table with evidence, **never** by a regex in code. Coverage numbers to report in the
acceptance test: MaleCNS cells whose type name exists in FAFB / BANC (survey: 59 % / 72 % before aliasing).

## 4. Acceptance

1. **Bit-identity of the MaleCNS path**: `cache/` md5s unchanged after the refactor; `tests/test_bit_identity.py`
   and the full CPU suite green; `connectome_fingerprint` of `load()` unchanged.
2. **Compile**: `load(dataset="fafb")` -> N ~ 139 k, `W.nnz` and `nt` counts printed and recorded in the manifest;
   `load(dataset="banc")` -> N ~ 147 k (after dropping glia / not_a_neuron / trachea). Each under ~5 min on the desktop.
3. **Smoke on CPU**: a 200-cell subset of each female graph steps `Brain` for 100 ms; `receptor_signs()` builds (the
   table transfers by type name -- report the tier histogram); `regions.labels` counts per region.
4. **FAFB optic**: `build_retina` succeeds; `OpticLobe` steps one frame of the room on the house GPU; the four
   column-map validations of section 2.5 pass and are printed.
5. **BANC walking replicate** (house cluster, one submission, `--arm-block`): `scripts/probe_walk_straightness.py`
   and `scripts/probe_vnc_drive.py plan --family body` on `dataset="banc"` with the shipped defaults, 5 runs, beside
   the MaleCNS numbers already in `out/vncd3/` -- reported as a cross-dataset comparison (never row by row against
   the MaleCNS batch; each is a one-batch claim). The answer is whichever way it comes out.
6. **`scripts/cross_connectome.py`**: the section-3 table of `docs/audits/flywire_banc_survey.md` regenerated from
   the three graphs through the same `Connectome` API (no pandas over the raw releases), with a per-release synapse
   scale, for every named edge of rounds 1-3 (the list is in TODO.md section F).
7. **Tests**: everything data-dependent behind the existing `data` marker (`tests/conftest.py`); vocabulary maps and
   the `(p, q) -> hex` transform get CPU tests on synthetic tables.
8. **Docs**: `docs/CONTROL_SURFACE.md` gains the `dataset=` argument; `docs/INSTALL.md` the fetch lines; an audit
   `docs/audits/connectome_backends.md` with every number above and the sex-difference caveat.

## 5. Out of scope / do not

- Do not tune any default for either female graph (no override tables, no gains). If a female graph runs away
  or is silent under the shipped defaults, that is the finding to report.
- Do not mix counts across releases without the recorded scale; do not treat BANC's optic lobes as complete;
  do not take BANC *predicted* monoamine labels over its *verified* column.
- Do not assume `fru` / `dsx` circuits are sex-shared; list the male-specific MaleCNS types with no female
  counterpart (and vice versa) in the audit.
- The 13.9 GB FAFB skeleton archive is for the room UI later, not for this work.
