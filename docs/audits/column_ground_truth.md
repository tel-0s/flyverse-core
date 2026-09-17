# Column inference against the releases' column annotations (TODO.md F.4)

`flyverse.interp.trace.column_of_cells` is the inference that placed the anatomical LC windows of the object rounds
(`object_baseline_r2.md`: 405 of 418 LC windows are boxes around its column; `object_localizer_r3.md` 1.2: the prior).
It has never been checked against a truth. The two releases now carry one: FAFB v783 `column_assignment` (45,528
cells of 31 types, `hex_source == "annotation"` through the affine map `hex1 = q + 18, hex2 = p + 20` of
`connectome_backends.md`) and MaleCNS `assignedOlHex1/2` (23,720 cells of 15 types). This audit withholds an annotated
type, runs the same propagation, and compares; then it measures what the rule does to the LC types, which have no
annotation in either release. CPU only, 31 s; no simulation, no default changed.

Generator: `scripts/probe_column_ground_truth.py --out out/colgt` (run `trace-20260915T022803Z-9dc64269`, commit
`4ebf680`). Every number below is in `out/colgt/column_ground_truth.json` (a `Result`; tables named in each heading)
or the per-cell CSVs beside it. Test: `tests/test_column_ground_truth.py` (a 7-column synthetic graph, CPU).
**Regenerated after the independent skeptic pass** (`## Skeptic pass` below): the `<= 4.6 deg` and `within r50`
statistics now carry a 1e-6 tolerance and `frac_same_column` is tested on the column index, not on an `arccos`
distance (skeptic corrections 2 and 5 / R7). The original run was `trace-20260915T020015Z-5f4edd51`, commit
`28e862f`; nothing else in the generator changed.

## 1. Method

* **Inference under test.** `column_of_cells(c, retina, rate_idx)` exactly as the rounds called it (`rate_idx` =
  `ol_intrinsic` minus photoreceptors, the `OpticLobe.rate_idx` rule): an annotated rate cell keeps its column; an
  unannotated one takes the column of its strongest |W| rate input partner, three passes; every other cell (the
  spiking cells, LC included) takes the column of its strongest rate input partner.
* **Truth.** `hex_source == "annotation"`. The retina is built once from the FULL annotation (`build_retina`) and is
  never withheld: the question is whether the propagation recovers a column the retina knows, not whether the grid
  survives. MaleCNS's retina keeps only photoreceptor columns (1,466), so **3,903 of its 23,720 annotated cells sit in
  columns the retina does not express** and are already propagated in the baseline; FAFB's retina keeps every
  annotated column (1,581; `retina.py` special case), so its truth is complete. `frac_exact_in_retina` scores only
  cells whose truth column exists.
* **Leave-one-type-out (`loo`).** Per dataset, per annotated rate-unit type with >= 100 cells (photoreceptors
  excluded: the retina assigns them by a different rule, `_assign_photoreceptor_columns`): withhold that type's
  annotation, infer, score each withheld cell. FAFB 29 types / 42,999 cells; MaleCNS 15 / 23,720.
* **The MaleCNS-like annotation set (`loo_malecns_set`, FAFB only).** Withhold at once the 14 FAFB rate-unit types
  MaleCNS does not annotate (L4, T2, T2a, T3, T4a-d, T5a-d, Tm3, Tm6) and score each -- the propagation depth the
  MaleCNS LC windows relied on, measured where a truth exists.
* **Errors.** `col_err`: Euclidean distance in `retina.py`'s hex embedding (X = h1 - h2/2, Y = h2 sqrt(3)/2; one
  lattice step = 1), same side only. `deg_err`: great circle between `retina.col_az_el` of the inferred and the
  truth column (0 for the identical column). `frac_within_io_deg`: cells within the 4.6-deg interommatidial angle,
  over ALL truth cells (an unassigned cell counts as outside).
* **Chance (`chance_*`).** The inferred columns of the withheld type permuted among its cells of the same type and
  truth side (the side is trivially known), 5 draws, statistics averaged. For the LC case the single column is
  permuted within type and soma side and its distance to the unchanged input centroid remeasured.
* **The LC case.** Full annotation; per cell the inferred column, the |W|-weighted mean direction of the columns of
  its rate inputs (the round-3 prior, `object_round3_localizer.anatomical_prior`), the 50 % / 80 % weight radii,
  the distinct input columns, and the great-circle distance from centroid to inferred column. Cross-dataset only as
  distributions (quantiles, two-sample KS) -- two animals, never cell by cell. On FAFB additionally the per-cell
  distance between the columns inferred under the full and the MaleCNS-like annotation sets (one animal, so per
  cell is legitimate).

## 2. Leave-one-type-out (`tables.loo_per_type`, `side == both`)

`exact` = fraction of truth cells given exactly their column; `<= 4.6 deg` = within one interommatidial angle;
`p90` = 90th percentile of the error over assigned cells with a truth column in the retina; `chance` = the
permutation control (exact / median deg).

**The `<= 4.6 deg` and `within r50` columns are knife-edge statistics and carry a 1e-6 tolerance** (`DEG_TOL` in the
generator; skeptic correction 2). The modal one-lattice-step error is *exactly* 4.6 deg -- columns one step apart in
elevation -- so a bare `deg_err <= 4.6` on a floating-point `arccos` splits that mode arbitrarily: 194 of the 1,450
FAFB T2 cells sit within 1e-6 of 4.6 deg and only about 90 of them passed. With the tolerance FAFB T2 moves
**0.847 -> 0.919**, T3 **0.936 -> 0.970**, T4a **0.891 -> 0.951**, Tm3 0.887 -> 0.965, Tm4 0.816 -> 0.935, Tm6
0.852 -> 0.932, L4 0.719 -> 0.828 and MaleCNS Tm4 0.687 -> 0.864 (35 of the 44 rows of the two tables move; the
exact fractions and the p90s are untouched). An independently implemented great-circle distance gives 0.839 / 0.933 /
0.892 for the same FAFB T2 / T3 / T4a cells, so the mode itself, not the threshold, is what these figures are
sensitive to. The same applies to section 3's `frac_col_within_r50`, where the modal case is `d == r50` exactly
(500 of 1,466 FAFB T2 cells, 87 of 237 FAFB LC10a): the tolerance moves FAFB LC10a 0.671 -> 0.675, Tm5Y
0.865 -> 0.868 and TmY21 0.713 -> 0.716 here, and the same statistic computed down a different float path reaches
0.900 / 0.491 / 0.633 for MaleCNS T2 / MaleCNS LC10a / FAFB LC10a against the 0.925 / 0.527 / 0.675 reported below.
LC11 and LC4 have only 2-3 such ties, so the headline **7 % / 13 % / 18 %** are robust; the mid-range r50 fractions
are not quotable beyond the first decimal.

### FAFB v783 (29 types, 42,999 cells; the truth is complete)

| type | n | assigned | exact | <= 4.6 deg | median col / deg | p90 deg | chance exact / median deg |
|---|---:|---:|---:|---:|---|---:|---|
| C2 | 1,426 | 0.999 | 0.816 | 0.926 | 0 / 0 | 4.6 | 0.002 / 55.3 |
| C3 | 1,509 | 0.999 | 0.990 | 0.996 | 0 / 0 | 0.0 | 0.001 / 55.8 |
| L1 | 1,572 | 0.997 | 0.992 | 0.996 | 0 / 0 | 0.0 | 0.002 / 56.1 |
| L2 | 1,557 | 0.999 | 0.972 | 0.985 | 0 / 0 | 0.0 | 0.001 / 56.1 |
| **L3** | 1,412 | 0.984 | 0.318 | 0.507 | 1 / 4.6 | **23.8** | 0.001 / 55.9 |
| L4 | 1,332 | 0.917 | 0.348 | 0.828 | 1 / 4.0 | 4.6 | 0.001 / 54.5 |
| L5 | 1,556 | 1.000 | 0.994 | 0.997 | 0 / 0 | 0.0 | 0.002 / 56.9 |
| Mi1 | 1,581 | 1.000 | 0.969 | 0.980 | 0 / 0 | 0.0 | 0.002 / 56.8 |
| **Mi4** | 1,529 | 1.000 | 0.736 | 0.812 | 0 / 0 | **10.3** | 0.001 / 56.6 |
| Mi9 | 1,541 | 0.999 | 0.990 | 0.995 | 0 / 0 | 0.0 | 0.001 / 57.2 |
| T1 | 1,389 | 0.999 | 0.988 | 0.995 | 0 / 0 | 0.0 | 0.001 / 55.3 |
| **T2** | 1,450 | 0.998 | 0.494 | **0.919** | 1 / 2.9 | 4.6 | 0.002 / 55.5 |
| T2a | 1,530 | 0.999 | 0.711 | 0.947 | 0 / 0 | 4.6 | 0.001 / 56.0 |
| **T3** | 1,474 | 1.000 | 0.812 | **0.970** | 0 / 0 | 4.6 | 0.001 / 55.9 |
| T4a | 1,441 | 1.000 | 0.784 | 0.951 | 0 / 0 | 4.6 | 0.002 / 55.4 |
| T4b | 1,493 | 1.000 | 0.829 | 0.942 | 0 / 0 | 4.6 | 0.001 / 55.4 |
| T4c | 1,553 | 0.999 | 0.748 | 0.938 | 0 / 0 | 4.6 | 0.001 / 56.5 |
| T4d | 1,498 | 1.000 | 0.789 | 0.946 | 0 / 0 | 4.6 | 0.002 / 55.2 |
| **T5a** | 1,465 | 0.999 | 0.561 | 0.698 | 0 / 0 | **62.0** | 0.002 / 56.3 |
| **T5b** | 1,484 | 1.000 | 0.549 | 0.686 | 0 / 0 | **67.3** | 0.001 / 56.1 |
| **T5c** | 1,472 | 0.999 | 0.570 | 0.718 | 0 / 0 | **58.3** | 0.001 / 56.6 |
| **T5d** | 1,416 | 0.999 | 0.583 | 0.747 | 0 / 0 | **56.0** | 0.002 / 54.9 |
| Tm1 | 1,549 | 1.000 | 0.987 | 0.990 | 0 / 0 | 0.0 | 0.001 / 57.5 |
| Tm2 | 1,542 | 0.999 | 0.982 | 0.990 | 0 / 0 | 0.0 | 0.002 / 56.0 |
| Tm20 | 1,484 | 0.997 | 0.972 | 0.987 | 0 / 0 | 0.0 | 0.002 / 57.0 |
| Tm3 | 1,499 | 0.999 | 0.630 | 0.965 | 0 / 0 | 4.6 | 0.001 / 55.7 |
| Tm4 | 1,454 | 0.991 | 0.590 | 0.935 | 0 / 0 | 4.6 | 0.001 / 56.8 |
| Tm6 | 1,276 | 0.999 | 0.534 | 0.932 | 0 / 0 | 4.6 | 0.002 / 55.3 |
| Tm9 | 1,515 | 0.996 | 0.934 | 0.963 | 0 / 0 | 0.0 | 0.001 / 56.7 |

Under the MaleCNS-like annotation set (`condition == loo_malecns_set`, 14 types withheld together) the numbers agree
to within 0.017 in the exact fraction and 0.005 in the 4.6-deg fraction for every type, with p90 shifts up to 2.0 deg
for T5a-d: T2 exact 0.477 / <= 4.6 deg 0.918 / p90 4.6; T3 0.811 / 0.970 / 4.6; T4a-d 0.744-0.821 / 0.936-0.953 /
4.6; T5a-d 0.549-0.583 / 0.687-0.747 / p90 54.6-65.3; Tm3 0.630 / 0.963; Tm6 0.529 / 0.931. The largest single
deltas against the one-type-at-a-time condition are T2's exact fraction (0.494 -> 0.477), T4b's (0.829 -> 0.821),
T4a's (0.784 -> 0.777) and Tm6's (0.534 -> 0.529), and T5a / T5b p90 (61.95 -> 59.91 deg, 67.34 -> 65.32 deg).
Annotation depth is not what limits the propagation for these types.

### MaleCNS v1.0 (15 types, 23,720 cells; 3,903 truth columns absent from the retina)

| type | n | in retina | exact | exact (in retina) | <= 4.6 deg | p90 deg | chance exact / median deg |
|---|---:|---:|---:|---:|---:|---:|---|
| C2 (R only) | 874 | 773 | 0.868 | 0.982 | 0.874 | 0.0 | 0.001 / 58.5 |
| C3 | 1,770 | 1,463 | 0.824 | 0.997 | 0.826 | 0.0 | 0.002 / 57.6 |
| L1 | 1,767 | 1,464 | 0.797 | 0.962 | 0.805 | 0.0 | 0.001 / 57.5 |
| L2 | 1,767 | 1,464 | 0.827 | 0.999 | 0.828 | 0.0 | 0.002 / 57.5 |
| **L3 (R only)** | 892 | 784 | **0.057** | 0.065 | 0.096 | **65.0** (median 23.5) | 0.001 / 57.5 |
| L5 | 1,773 | 1,467 | 0.827 | 0.999 | 0.827 | 0.0 | 0.001 / 56.6 |
| Mi1 | 1,762 | 1,459 | 0.828 | 1.000 | 0.828 | 0.0 | 0.001 / 56.6 |
| **Mi4** | 1,758 | 1,458 | 0.444 | 0.535 | 0.511 | **67.2** | 0.001 / 61.4 |
| Mi9 | 1,760 | 1,459 | 0.722 | 0.870 | 0.731 | 12.7 | 0.001 / 58.2 |
| T1 | 1,764 | 1,464 | 0.830 | 1.000 | 0.830 | 0.0 | 0.001 / 55.9 |
| Tm1 | 1,767 | 1,465 | 0.827 | 0.998 | 0.827 | 0.0 | 0.001 / 57.0 |
| Tm2 | 1,758 | 1,461 | 0.829 | 0.997 | 0.830 | 0.0 | 0.001 / 56.6 |
| Tm20 | 1,732 | 1,439 | 0.671 | 0.808 | 0.691 | 22.0 | 0.001 / 58.2 |
| Tm4 (R only) | 833 | 743 | 0.335 | 0.376 | 0.864 | 4.6 | 0.001 / 56.3 |
| Tm9 | 1,743 | 1,454 | 0.635 | 0.761 | 0.649 | 37.7 | 0.001 / 57.3 |

Overall (`tables.loo_overall`): FAFB exact min / median 0.318 / 0.789, largest per-type median error 4.6 deg,
largest p90 67.3 deg; MaleCNS 0.057 / 0.824, 23.5 deg, 67.2 deg; chance exact <= 0.002 and chance median >= 54.5
deg on both. The inference is nowhere near chance, and it is exact or one column off for the lamina / medulla
columnar types whose strongest input is another columnar cell.

### Per side (`side in (L, R)`)

The sides agree on FAFB except where one eye's partner set differs: T2 L / R exact 0.475 / 0.515 (<= 4.6 deg 0.914 /
0.923), T3 0.821 / 0.802 (0.969 / 0.971), T5a 0.616 / 0.506 (p90 46.4 / 81.8 deg), Mi4 0.665 / 0.808, T4a 0.711 /
0.856. MaleCNS's annotation itself is asymmetric: C2, L3 and Tm4 are annotated on the right only, and on the left the
retina expresses fewer truth columns (Tm9 exact L 0.387 vs R 0.875; Tm20 0.460 vs 0.878; Mi9 0.564 vs 0.876; Mi4
p90 L 117.6 vs R 12.9 deg). The male left eye is the weaker ground truth and the weaker inference.

### Why the failures fail (`tables.loo_failure_partner`: the strongest partner of every withheld cell, and of the cells more than 10 deg off)

The rule has no notion of columnar versus wide-field. Every large error is a cell whose strongest rate input is a
wide-field cell that was itself propagated: withheld FAFB T5a cells more than 10 deg off (415 of 1,465) take **CT1**
in 411 cases (T5b 427 / 435, T5c 366 / 373, T5d 314 / 319); L3 takes **Dm12** (FAFB 331 of 413 bad cells, the rest
Lawf1 64 / Lai 16; MaleCNS 680 of 730); Mi4 takes **Dm4** (FAFB 153 of 160; MaleCNS 487 of 723); MaleCNS Tm9 / Tm20
fail through the already-failed **L3** (543 of 584, 445 of 510). The bad partners are annotated in 0-1 % of cases on
FAFB (T5 0.5-1.3 %, L3 0.2 %, Mi4 0 %)
against 15-96 % for the partners overall. T2's 31 bad cells (2 %) are split (LC14a-1 9, Tm2 7, Pm3 5). Where the
strongest partner is columnar and annotated (T3: Mi1 978 + Tm1 663 of 1,675 partners, 99.6 % annotated; T2: Tm2 990
+ L5 303, 96 %) the column is right to one lattice step.

**Caveat on the whole MaleCNS column of this table (skeptic correction 7).** The MaleCNS `n_bad` counts are inflated
by construction, not only for L3: `failure_partners` treats `deg_err = NaN` as `> bad_deg`, so every cell whose truth
column the male retina lacks is counted as a failure. MaleCNS Mi1 shows 303 "bad" cells -- exactly the 1,762 - 1,459
cells with no expressible truth -- whose partner is L1 in 303 / 303 cases and which are *exactly correct* in hex
terms (`frac_exact_in_retina` 1.000); the same floor of ~300 sits under C3, L2, L5, T1, Tm1 and Tm2. Only the FAFB
column, and the excess above that floor on MaleCNS (L3 730, Mi4 723, Tm9 584, Tm20 510, Mi9 454), is a real failure
count.

## 3. The LC case (`tables.lc_summary`, `side == both`; `tables.lc_strongest_partner`)

| dataset | type | n | no column | distinct columns (largest share) | input columns (median) | r50 / r80 deg | centroid -> column deg (median / p90) | chance median | column within r50 |
|---|---|---:|---:|---|---:|---|---|---:|---:|
| MaleCNS | **LC11** | 143 | 0 | **13** (0.350) | 94 | 21.7 / 67.3 | **66.9** / 115.5 | 71.0 | 0.070 |
| MaleCNS | **LC4** | 126 | 0 | 32 (0.286) | 78 | 19.7 / 55.1 | **61.4** / 111.1 | 64.9 | 0.183 |
| MaleCNS | LC10a | 275 | 0 | 88 (0.189) | 32 | 36.6 / 70.5 | 45.9 / 83.9 | 59.6 | 0.527 |
| MaleCNS | LPLC2 | 185 | 0 | 73 (0.238) | 86 | 27.7 / 47.6 | 30.8 / 131.4 | 57.7 | 0.497 |
| MaleCNS | T2 | 1,630 | 0 | 1,043 (0.023) | 40 | 14.9 / 53.2 | 13.0 / 24.0 | 54.6 | 0.925 |
| MaleCNS | T3 | 1,940 | 0 | 1,245 (0.023) | 19 | 11.6 / 41.4 | 10.4 / 19.6 | 54.3 | 0.954 |
| MaleCNS | Tm5Y | 898 | 0 | 478 (0.151) | 36 | 28.3 / 60.8 | 22.4 / 71.3 | 55.2 | 0.766 |
| MaleCNS | TmY21 | 372 | 0 | 110 (0.218) | 50 | 27.9 / 67.2 | 37.4 / 93.9 | 56.9 | 0.505 |
| FAFB | **LC11** | 127 | 0 | 42 (0.142) | 79 | 14.1 / 30.0 | **53.7** / 92.4 | 62.5 | 0.134 |
| FAFB | **LC4** | 104 | 0 | 14 (0.231) | 56 | 14.5 / 32.3 | **62.8** / 123.0 | 62.5 | 0.125 |
| FAFB | LC10a | 237 | 0.021 | 155 (0.073) | 6 | 11.8 / 24.3 | 12.2 / 46.3 | 53.2 | 0.675 |
| FAFB | LPLC2 | 210 | 0 | 149 (0.081) | 38 | 10.6 / 18.3 | 9.5 / 45.7 | 52.3 | 0.581 |
| FAFB | T2 | 1,466 | 0 | 1,450 (0.001) | 13 | 5.0 / 11.5 | 4.0 / 8.8 | 55.2 | 0.874 |
| FAFB | T3 | 1,676 | 0.001 | 1,474 (0.002) | 9 | 2.8 / 4.8 | 2.1 / 4.4 | 53.7 | 0.928 |
| FAFB | Tm5Y | 802 | 0.001 | 620 (0.021) | 11 | 12.2 / 34.8 | 10.0 / 25.3 | 51.5 | 0.868 |
| FAFB | TmY21 | 282 | 0.014 | 207 (0.029) | 7 | 10.4 / 18.8 | 9.2 / 25.5 | 45.2 | 0.716 |

The MaleCNS LC11 row reproduces `object_localizer_r3.md` 1.2 (13 columns, 50 / 44 / 14 / 13 cells, median 64.5 deg
there against 66.9 here: the same rule, that file's centroid used every input, this one the rate inputs only).

**The partner whose column an LC cell inherits.** MaleCNS LC11: TmY19b 59, Li15 36, MeLo10 11, Li26 11 -- 78
distinct partner cells for 143 LC11, **0 % annotated**, the partner carrying a median 1.7 % of the cell's rate input.
LC4: Li28 57, TmY3 25, Li39 16 (64 partners, 0 % annotated, 1.9 %). FAFB LC11: Li15 55, Li33 26, Li30 12 (65
partners, 3 % annotated, 2.3 %); LC4: Li28 68, Li33 29 (24 partners for 104 cells, 0 %, 4.5 %). LC10a and LPLC2
differ in kind: their strongest partner is Tm5Y for 36 / 66 (FAFB) and 70 / 59 (MaleCNS) cells, a columnar type,
and their column lands inside the cell's r50 for 50-68 % of cells (MaleCNS LC10a 53 %, LPLC2 50 %; FAFB LC10a 68 %,
LPLC2 58 %). Tm5Y is *not* the modal partner for all four of those rows: MaleCNS LPLC2's modal partner is the
wide-field **LPi43** (69 cells) with Tm5Y second (59), and MaleCNS LPLC2 is nonetheless well above chance
(30.8 deg against the 57.7-deg permutation of this table; the skeptic's paired Wilcoxon over cells gives p 2.3e-4).
A wide-field strongest partner does not by itself imply a chance column; LC11 and LC4 differ because their partners'
own columns are themselves collapsed onto a handful of positions. LC11 and LC4 inherit the column of a wide-field
lobula-intrinsic cell (Li / TmY19b), which is itself a propagated column, and the result is statistically
indistinguishable from a permuted column of the same type and soma side (paired Wilcoxon over cells: MaleCNS LC11
p 0.96, LC4 p 0.54; FAFB LC4 p 0.94). FAFB LC11 is the exception -- weakly but significantly better than the
permutation (53.7 vs 61.9 deg, 57 % of cells closer, p 3.6e-4). "Of the same eye" would be wrong in either
direction: 64 % of MaleCNS LC11 cells and 42 % of LC4 cells are given a column in the eye contralateral to their
soma side (FAFB 19 % each), and the permutation pool is contralateral in the same proportion. (The Wilcoxon tests
and the contralateral fractions are the independent skeptic pass's, not this audit's -- `## Skeptic pass` R5 below;
no `out/colgt` table reports a test.) The distances on file: MaleCNS LC11 66.9 deg from the cell's input centroid
against 71.0 by permutation, LC4 61.4 vs 64.9; FAFB LC11 53.7 vs 62.5, LC4 62.8 vs 62.5. Per side (`lc_summary`,
L / R): MaleCNS LC11 57.3 / 79.4 vs chance 65.8 / 74.1, 7 / 8 distinct columns; LC4 54.8 / 80.2 vs 55.0 / 74.4.

**The single column is not even stable under the sweep order.** `column_of_cells` updates in place while iterating in
index order, so its three passes are Gauss-Seidel; running them synchronously changes the column of 79 % of FAFB
LC11 cells and 79 % of LC4 cells (distinct columns 42 -> 64 and 14 -> 24), while leaving T2 100 % and T3 99.9 %
unchanged. The LC assignment is a function of the row order of the neuron table. (Measured by the independent
skeptic pass, C9.)

**Cross-dataset (`tables.lc_cross`; distributions of two animals).** The female graph's columns are tighter for every
type: centroid -> column median MaleCNS / FAFB LC11 66.9 / 53.7 (KS 0.32), LC10a 45.9 / 12.2 (0.48), LPLC2 30.8 / 9.5
(0.62), T2 13.0 / 4.0 (0.71), T3 10.4 / 2.1 (0.82), Tm5Y 22.4 / 10.0 (0.47), TmY21 37.4 / 9.2 (0.58); LC4 is equally
bad in both (61.4 / 62.8, KS 0.05, p 1.0). The input sets are also narrower on FAFB (T2 r50 14.9 vs 5.0 deg; T3 11.6
vs 2.8; LC11 21.7 vs 14.1; median distinct input columns T2 40 vs 13, T3 19 vs 9) -- FAFB annotates T2 / T3 / T4 /
T5 directly and its pair threshold (>= 5 synapses) prunes weak cross-column contacts, so no part of this is a sex
difference claim. The centroid positions themselves overlap across animals (az KS 0.13-0.26, el 0.02-0.18).

**Annotation depth is not the cause (`tables.fafb_annotation_depth`; one animal, per cell).** Re-inferring FAFB
with only the 15 MaleCNS-annotated types leaves the LC columns where they were: LC11 41 vs 42 distinct columns,
**70 %** of cells the identical column, median shift 0 deg, p90 16.2 deg, centroid distance 54.0 vs 53.7; LC4 **94 %**
identical, p90 0; LC10a **78 %** / p90 15.9; LPLC2 **82 %** / 7.7; T2 **48 %** identical but **92 %** within 4.6 deg
(median 3.0, p90 4.6), T3 **83 % / 97 %** / p90 4.6; Tm5Y **95 % / 97 %**. (`frac_same_column` is now tested on the
column index. It was previously `full_vs_malecns_set_deg < 1e-9`, and that distance is an `arccos` of a dot product
which returns up to 1.2e-6 deg for two *identical* columns -- the identity shortcut `score_cells` applies was missing
here -- so every identical-column figure in this table was understated: LC11 63 -> 70, LC4 92 -> 94, LC10a 66 -> 78,
LPLC2 68 -> 82, T2 38 -> 48, T3 67 -> 83, Tm5Y 75 -> 95. The same defect printed Tm5Y's
`p90_full_vs_malecns_set_deg` as 8.5e-7 deg, which was this noise and is now 0. The error direction *strengthens*
the conclusion below; skeptic correction 5 / R7.) The 13-column LC11 collapse on MaleCNS is the male graph's
own strongest-partner structure (TmY19b and Li15 cells whose propagated columns coincide), not a missing T2 / T3
annotation, and reducing FAFB to the MaleCNS annotation set does not reproduce it (42 -> 41 columns).

## 4. Verdict

1. **For the columnar stage the propagation is supported at the interommatidial level.** With a true annotation
   withheld, T2 is exactly right for 49 % of cells and within 4.6 deg for 92 % (p90 = 4.6 deg, one lattice step);
   T3 81 % / 97 % / 4.6 deg; T4a-d 75-83 % / 94-95 % / 4.6 deg; the medulla Tm types with a truth (Tm3 / Tm4 / Tm6,
   the nearest annotated relatives of Tm5Y) 53-63 % / 93-96 % / 4.6 deg; lamina and Mi1 / Mi9 / Tm1 / Tm2 / Tm9 /
   Tm20 / C3 / T1 93-99 % exact. Chance is 0.1-0.2 % exact and 55 deg. This holds unchanged when the annotation is
   reduced to the 15 MaleCNS types. Tm5Y itself has no truth in either release; its column sits 10.0 deg (FAFB) /
   22.4 deg (MaleCNS) from its input centroid against 52-55 deg by chance, i.e. right to a few columns, not to one.
2. **It is not supported for the types whose strongest input is wide-field**: T5a-d (CT1) p90 56-67 deg with 25-31 %
   of cells more than 4.6 deg off; L3 (Dm12) 49 % off on FAFB and 90 % on MaleCNS; Mi4 (Dm4) p90 10 / 67 deg. The
   rule needs a columnar-partner restriction for those types; none is changed here.
3. **The anatomical LC11 and LC4 windows of object rounds 2-3 are not supported.** Their column is inherited from
   a wide-field Li / TmY19b partner carrying ~2 % of the cell's input, and it lies at or near the permutation
   distance from the cell's own input field (MaleCNS LC11 66.9 deg vs 71.0 permuted, paired Wilcoxon p 0.96; LC4
   61.4 vs 64.9, p 0.54; FAFB LC4 62.8 vs 62.5, p 0.94). FAFB LC11 is the one exception -- 53.7 vs 62.5 permuted,
   weakly but significantly better than chance (p 3.6e-4, 57 % of cells closer than their permuted column), so the
   FAFB replicate does *not* say the same for LC11 as MaleCNS does. Only 7 % of MaleCNS LC11 cells and 18 % of LC4
   cells have their window column inside the half-weight radius of their inputs, and the assignment is not stable
   under the propagation's sweep order (79 % of FAFB LC11 cells change column under a synchronous sweep).
   **LC10a and LPLC2 are intermediate** (MaleCNS 45.9 / 30.8 deg vs chance 59.6 / 57.7, the column inside r50 for
   53 / 50 % of cells; on FAFB 12.2 / 9.5 deg): better than chance, not a receptive-field centre.
4. **What this implies for `object_baseline_r2.md`'s window sizes.** The LC11 boxes (143 / 143 `anat`, the type's
   median fitted width, else 15 deg) and the 262 anatomical LC10a boxes were centred a median 67 deg / 46 deg from
   the cells' input fields, whose half-weight radius is 22 deg / 37 deg (80 %: 67 / 70 deg). The box size is not what
   makes the window wrong: a 15-deg box on the inferred column still holds a median 15 % of an LC11 cell's rate-input
   weight (30-deg box 16 %; no cell holds zero), while half of that weight lies more than 73 deg away -- the box sits
   on a local weight spike, the strongest partner's own column, not on the centre of the field. Whether that
   mis-centring changed the round-2 reading was tested directly: re-windowing the same 84 fetched runs on the
   input-weighted centroid, and on a box of twice the cell's own r50 about that centroid, reproduces the LC11 null at
   every rung on the shipped lobe (smallest exact Mann-Whitney p 0.394 and 0.240 against 0.180 as shipped and 0.041
   whole-window; Holm needs 0.00417), and reproduces the LC10a result (0.026 and 0.132 against 0.0087 as shipped).
   The sphere track is at elevation exactly 0 while the LC11 input centroids sit at a median |el| of 24 deg, so the
   corrected window is entered by **fewer** bodies than the anatomical one (17 of 143 against 55 at the 4.5-deg rung;
   an r50-sized centroid box reaches 61). **The round-2 LC11 sphere-ladder result is therefore negative under the
   corrected window as well as the shipped one, not uninformative**; what the mis-centring costs is the per-cell
   interpretability of the window, not the family call. (The box / input-weight overlap and the re-windowing of the
   84 runs are the independent skeptic pass's measurements, not this audit's -- no `out/colgt` table measures either;
   `## Skeptic pass` R1-R4 below.) The round-3 prior (the input centroid with its r50 / r80 on file,
   `object_localizer_r3.md` 1.2) is the right centre; a window for an LC cell with no fitted RF should be that
   centroid with a box no smaller than the cell's r50 (MaleCNS LC11 median 21.7 deg, i.e. a ~45-deg box -- derived
   from `lc_summary.median_r50_deg`, not tabulated), and the single `column_of_cells` column should not be used for
   any `visual_projection` cell. The T2 / T3 / Tm-stage windows and the `at_null` reference of the trace validation
   are unaffected: those columns are right to one ommatidium.

## Report

```yaml
summary: >
  column_of_cells checked against the FAFB v783 column_assignment (29 rate-unit types, 42,999 cells) and MaleCNS
  assignedOlHex (15 types, 23,720 cells) by leave-one-type-out with the same propagation and a within-type-and-side
  permutation control. Columnar types are recovered to one ommatidium (T2 92 %, T3 97 %, T4 94-95 %, Tm3/4/6 93-96 %
  within 4.6 deg at a 1e-6 tolerance, p90 4.6 deg; lamina / Mi1 / Tm1 / Tm2 / Tm9 93-99 % exact; chance 0.1 % exact,
  55 deg), unchanged when FAFB's annotation is cut to the 15 MaleCNS types (exact fractions agree to 0.017, 4.6-deg
  fractions to 0.005, p90 to 2.0 deg). Types whose strongest input is wide-field fail (T5a-d via
  CT1 p90 56-67 deg, L3 via Dm12, Mi4 via Dm4). The LC11 / LC4 single column is inherited from a wide-field Li /
  TmY19b partner (~2 % of input, 0-3 % annotated), sits at or near the permutation distance from the cell's input
  centroid (MaleCNS LC11 66.9 vs 71.0 deg permuted, paired Wilcoxon p 0.96; LC4 61.4 vs 64.9, p 0.54; FAFB LC4
  62.8 vs 62.5, p 0.94 -- FAFB LC11 53.7 vs 62.5 is the one case weakly better than chance, p 3.6e-4), is shared by
  up to 50 cells at a time, and is not stable under the propagation's sweep order; LC10a / LPLC2 are
  intermediate (MaleCNS 45.9 / 30.8 vs 59.6 / 57.7). The anatomical LC11 / LC10a windows of the round-2 sphere
  ladder and the round-3 rectangle ladders were therefore centred a median 67 / 46 deg from the cells' input fields
  (r50 22 / 37 deg). That is a real defect of the windows, but it does NOT make those rounds' LC11 results
  uninformative: an independent pass re-windowed the same 84 round-2 runs on the input-weighted centroid and on a
  2 x r50 box about it and reproduced the null (smallest exact-U p 0.394 and 0.240 against 0.180 as shipped; Holm
  needs 0.00417), and because the sphere sweeps elevation 0 while the LC11 centroids sit at a median |el| of 24 deg,
  the corrected window is entered by FEWER bodies (17 of 143 against 55 at the 4.5-deg rung). The null stands under
  the corrected window; what the mis-centring costs is per-cell interpretability and coverage, not the family call.
skeptic:
  source: "independent Opus pass, 2026-09-15"
  verdict: "mostly sound"
key_claims:
  - "FAFB leave-one-type-out, T2: exact 0.494, within 4.6 deg 0.919, p90 4.6 deg; T3: 0.812 / 0.970 / 4.6; chance exact 0.002 / 0.001, chance median 55.5 / 55.9 deg (tables.loo_per_type). The 4.6-deg fractions carry a 1e-6 tolerance: the modal error IS exactly one lattice step, so a bare <= splits it (T2 0.847 -> 0.919 without / with)"
  - "Agrees to within 0.017 in the exact fraction, 0.005 in the 4.6-deg fraction and 2.0 deg in p90 with only the 15 MaleCNS-annotated types kept (loo_malecns_set: T2 0.477 / 0.918 / 4.6; T3 0.811 / 0.970 / 4.6)"
  - "Failures are wide-field partners (tables.loo_failure_partner): T5a-d p90 56-67 deg (CT1: 411 of 415 bad T5a cells), L3 p90 23.8 deg FAFB / 65.0 MaleCNS (Dm12: 331 / 413 and 680 / 730), Mi4 p90 10.3 / 67.2 (Dm4: 153 / 160 and 487 / 723); bad partners annotated in 0-1 % of cases"
  - "MaleCNS: 3,903 of 23,720 annotated cells sit in columns its photoreceptor-only retina lacks (exact therefore capped at ~0.83 -- derived from 1 - 3,903 / 23,720, not tabulated; exact-in-retina 0.96-1.00 for lamina / Mi1 / Tm1 / Tm2); C2 / L3 / Tm4 annotated on the right only; left-eye truth sparser (Tm9 exact L 0.387 vs R 0.875). The same absence inflates every MaleCNS n_bad in loo_failure_partner by a floor of ~300 (NaN deg_err counts as > bad_deg)"
  - "LC11 single column: MaleCNS 13 distinct columns for 143 cells (50 / 44 / 14 / 13), centroid-to-column 66.9 deg vs 71.0 permuted (paired Wilcoxon p 0.96), 7 % of cells inside their r50; FAFB 42 columns, 53.7 vs 62.5 deg (p 3.6e-4, the one type weakly better than its permutation), 13 %; strongest partner TmY19b 59 / Li15 36 (MaleCNS), Li15 55 / Li33 26 (FAFB), 0-3 % annotated, median 1.7-2.3 % of input (tables.lc_summary, lc_strongest_partner)"
  - "LC4: MaleCNS 61.4 vs 64.9 deg, FAFB 62.8 vs 62.5 (Li28 / Li33 / TmY3 partners); LC10a 45.9 vs 59.6 (FAFB 12.2 vs 53.2); LPLC2 30.8 vs 57.7 (FAFB 9.5 vs 52.3) -- Tm5Y is the partner for 36-70 LC10a / LPLC2 cells, but MaleCNS LPLC2's modal partner is the wide-field LPi43 (69, Tm5Y 59) and it is still well above chance, so a wide-field partner alone does not imply a chance column"
  - "FAFB re-inferred with the MaleCNS annotation set: LC11 41 vs 42 columns, 70 % identical column, median shift 0, p90 16.2 deg; LC4 94 % identical -- the MaleCNS 13-column collapse is the male graph's partner structure, not annotation depth (tables.fafb_annotation_depth; frac_same_column is tested on the column index, not on an arccos distance, which previously understated every row: LC11 63 -> 70, LC4 92 -> 94, T3 67 -> 83)"
  - "The LC single column is not order-stable: column_of_cells sweeps in index order and updates in place, so a synchronous sweep changes 79 % of FAFB LC11 and LC4 cells (distinct columns 42 -> 64, 14 -> 24) while leaving T2 100 % and T3 99.9 % unchanged (independent skeptic pass C9; not measured by this audit's tables)"
  - "Cross-dataset (distributions): FAFB columns tighter for every type (T3 2.1 vs 10.4 deg, KS 0.82; LC10a 12.2 vs 45.9, 0.48) except LC4 (KS 0.05, p 1.0); FAFB annotates T2 / T3 / T4 / T5 and thresholds pairs at 5 synapses, so not a sex claim (tables.lc_cross)"
files_written:
  - scripts/probe_column_ground_truth.py
  - tests/test_column_ground_truth.py
  - docs/audits/column_ground_truth.md
  - out/colgt/column_ground_truth.json
  - out/colgt/loo_per_type.csv
  - out/colgt/loo_failure_partner.csv
  - out/colgt/loo_per_cell_fafb.csv
  - out/colgt/loo_per_cell_malecns.csv
  - out/colgt/lc_per_cell.csv
  - out/colgt/lc_summary.csv
  - out/colgt/lc_cross.csv
  - out/colgt/lc_strongest_partner.csv
  - out/colgt/fafb_annotation_depth.csv
api:
  - "scripts/probe_column_ground_truth.py: rate_index, withhold, infer_columns, truth_table, score_cells, summarize, shuffle_within, failure_partners, leave_one_out, input_centroids, lc_case, strongest_partner, cross_dataset; CLI --out --datasets --min-cells --shuffles --seed"
  - "no library API changed; flyverse.interp.trace.column_of_cells is called as the rounds called it; no default changed"
validation:
  - "tests/test_column_ground_truth.py: 7 passed (CPU, synthetic 7-column graph: exact recovery, chance control, wide-field fallback = 1 column = 4.6 deg, permutation invariants, LC partner, lattice / retina distances)"
  - "Result.check() empty; provenance per dataset (commit 4ebf680, cache fingerprints, retina records) in summary.provenance_by_dataset"
  - "Regenerated after the independent skeptic pass (run trace-20260915T022803Z-9dc64269; the original was trace-20260915T020015Z-5f4edd51, commit 28e862f): frac_same_column and full_vs_malecns_set_deg now use the column-index identity shortcut, and the <= 4.6 deg / within-r50 statistics a 1e-6 tolerance. No other generator change; 7 tests still pass"
  - "reproduces object_localizer_r3.md 1.2 for MaleCNS LC11 (13 columns, 50 / 44 / 14 / 13 cells) and LC10a (88 columns)"
  - "FAFB cache compiled from D:/Datasets/flywire (139,255 cells, 3,732,460 pairs); MaleCNS cache untouched"
recommendations:
  - "Object rounds: never window a visual_projection cell on the single column_of_cells column; use the input-weighted centroid (round-3 prior) with a box >= the cell's r50 (LC11 ~22 deg radius), or a fitted RF"
  - "object_baseline_r2.md and object_rectangles_r3.md section 6: carry the 'Window caveat' applied there -- the 405 anatomical LC boxes (LC11 143/143, LC10a 262/275) were centred on column_of_cells's single column, a median 67 deg (MaleCNS) / 54 deg (FAFB) from the cell's input-weighted centroid and not cell-specific (13 distinct columns for 143 LC11 bodies), BUT re-windowing the same 84 runs on the centroid and on a 2 x r50 box reproduces the LC11 null (smallest exact-U p 0.394 and 0.240 against 0.180 as shipped; Holm needs 0.00417) and LC10a's (0.026 and 0.132 against 0.0087), and the corrected window is entered by fewer bodies (17 of 143 against 55 at the 4.5-deg rung) because the sphere sweeps elevation 0. The null stands; the caveat is about per-cell interpretability and coverage, not the family call. It does NOT belong on object_localizer_r3.md, which fitted inside a 20-deg box on the input-weighted centroid and kept the single column only as a reported distance"
  - "column_of_cells: restrict the strongest-partner vote to columnar (annotated or lamina / medulla columnar) types, or fall back to the input centroid, for T5 / L3 / Mi4 and every spiking cell -- a proposed change, not made here"
  - "MaleCNS retina: consider keeping annotated columns without a photoreceptor as FAFB does (3,903 annotated cells currently re-propagated); a default change, owner decision"
open_questions:
  - "Tm5Y / TmY21 have no annotation in either release; their columns are 10-37 deg from their input centroids -- the Tm5Y-like stage is right to a few columns, not one"
  - "Whether the LC11 / LC4 strongest partners (Li15, Li33, Li28, TmY19b) are reconstruction-complete in MaleCNS; the FAFB replicate shows the same partner types, so the rule, not the reconstruction, is the cause"
  - "MaleCNS left-eye annotation is sparser and one-sided for C2 / L3 / Tm4; the male left-eye windows were inferred from a weaker truth than the right"
```

## Skeptic pass (independent, 2026-09-15)

Branch `main`, HEAD `28e862f`. Nothing committed, nothing under `docs/`, `scripts/`, `flyverse/`, `tests/` edited.
CPU only. Every number below comes from my own code
(`scratchpad/recheck.py`, `recheck2.py`, `recheck3.py`, `recheck4.py`), which shares only
`flyverse.connectome` / `flyverse.retina` with the audit: my own withhold-and-propagate, my own great-circle
distance (through `retina.col_dir` dot products, not the audit's `az/el` `arccos`), my own permutation
(seed 12345, 20 draws, against the audit's seed 0, 5 draws), my own centroid and window code. The audit's
script was never run.

Headline: **the measurement layer (sections 1-3) reproduces almost exactly and is sound. The section-4
inference -- the part that "changes the reading of two published rounds and the Neurome delivery" -- does
not survive.** Re-windowing the actual round-2 runs on the audit's own recommended centre gives the same
null, and the load-bearing sentence about box/input overlap is false on its own terms.

---

### Refuted

#### R1. "no box of 15-30 deg placed at that centre overlaps the cell's inputs for most LC11 cells" (section 4 item 4)

FALSE, and it is in no file of `out/colgt` -- nothing in the audit measures box-vs-input-field overlap.
Share of a cell's |W| rate-input weight inside a box centred on the shipped round-2 anatomical column,
against the same box on the input-weighted centroid (MaleCNS, the shipped lobe):

| type | n | 15-deg box on the anat column | 15-deg box on the centroid | 30-deg anat | 30-deg centroid | cells with ZERO weight in the 15-deg anat box | ... in the 15-deg centroid box |
|---|---:|---:|---:|---:|---:|---:|---:|
| LC11 | 143 | **0.152** | 0.078 | 0.158 | 0.292 | **0 %** | 8.4 % |
| LC10a | 275 | **0.182** | 0.008 | 0.238 | 0.120 | **0 %** | 39.3 % |

Every one of the 143 LC11 bodies has non-zero input weight inside its shipped 15-deg box, with a median
15 % of its whole rate-input weight there. At matched box size the "wrong" centre holds *more* of the
cell's input weight than the "right" one for both LC types. (The centroid is a resultant-vector mean of a
broad, partly bimodal field, so it falls in a low-density place between the modes; the inherited column is
the strongest partner's column and sits on a local weight spike.)
File: `scratchpad/box_input_overlap.csv`.

#### R2. "the round-2 LC11 windowed statistics measured the drive of LC11 cells while the object was over a patch of eye those cells do not read" (section 4 item 4)

FALSE for this stimulus. Per frame I computed the input-weight-weighted coverage of the sphere (a column
counts as covered when its direction is within diam/2 of the disc centre; weights = the cell's |W| rate-input
weight on that column), then averaged it over the frames each window selects (ship lobe, seed 0, MaleCNS):

| type | rung | anat window | centroid window | whole run | per-cell peak |
|---|---:|---:|---:|---:|---:|
| LC11 | 15 deg | **0.0459** | 0.0273 | 0.0226 | 0.175 |
| LC11 | 30 deg | **0.1215** | 0.0877 | 0.0625 | 0.204 |
| LC10a | 15 deg | **0.1209** | 0.0112 | 0.0197 | 0.146 |
| LC10a | 30 deg | **0.1779** | 0.0621 | 0.0601 | 0.191 |

The mis-centred window selects frames in which *more* of the cell's input field is lit than the run average,
and more than a centroid window would select. File: `scratchpad/input_weight_coverage.csv`.

#### R3. "the round-2 LC11 numbers are uninformative rather than negative" (section 4 item 4; Report `recommendations`)

NOT SUPPORTED. I rebuilt the predeclared window rule from `out/objr2/rfmap_{ship,fb0}.csv` (it reproduces
`object_baseline_r2.md`'s window sources exactly: LC11 143/143 `anat`, LC10a 262 `anat` + 7 `rf_ship` +
6 `rf_fb0` = 405 of 418; and its per-rung windowed-body counts 55/99/99/99/103/103 and 73/76/76/76/79/90),
then re-windowed the **same 84 fetched runs** four ways and recomputed the primary statistic (population
median over windowed bodies of the windowed time-mean A-B drive; six object runs vs six nulls; exact
two-sided Mann-Whitney):

Shipped lobe, smallest p over the six rungs (Holm threshold for a 12-member family = 0.00417):

| type | anat (as shipped) | centroid, same box | centroid, box = 2 x r50 | whole window |
|---|---:|---:|---:|---:|
| LC11 | 0.180 | 0.394 | 0.240 | 0.041 |
| LC10a | 0.0087 | 0.026 | 0.132 | 0.065 |

Not one member of any windowing reaches Holm. The `gain_fb = 0` lobe is deterministic (all six seeds give
byte-identical medians; null SD ~1e-9), so no test is meaningful there, and its magnitudes show no ordering
by window quality and no size trend either (LC11 |effect| anat 0.015-0.125 mV, centroid 0.020-0.136,
r50 0.017-0.082, whole 0.010-0.042 mV). **Correcting the window to the audit's own recommended centre, and
to its own recommended box size, reproduces the LC11 null.** The window defect is real; the inference drawn
from it is not. Files: `scratchpad/rewindow_runs.csv`, `scratchpad/rewindow_compare.csv`.

#### R4. The geometric premise ("a window that misses the RF cannot show an effect") is backwards for this ladder

The sphere track is at elevation **exactly 0** across az +-50 deg (`geometry.elevation_deg` 0,
`track_check.realised_el_deg_maxdev` 0.0, `track_el_deg` a constant 0 in every npz). The round-2 anatomical
columns sit at a median |el| of **10.7 deg**; the input centroids the audit prefers sit at a median |el| of
**24.3 deg**. So the correct window is crossed by the stimulus *less* often:

| type | rung | bodies with >=1 frame: anat | centroid (same box) | centroid (2 x r50) | mean fraction of the 1200-frame track: anat / centroid / r50 |
|---|---:|---:|---:|---:|---|
| LC11 | 4.5 deg | **55** / 143 | **17** / 143 | 61 / 143 | 7.0 % / 1.9 % / 15.7 % |
| LC11 | 30 deg | 103 | 50 | 99 | 25.8 % / 11.4 % / 32.0 % |
| LC10a | 4.5 deg | 73 / 275 | 78 / 275 | 217 / 275 | 5.8 % / 6.1 % / 48.2 % |
| LC10a | 30 deg | 90 | 164 | 256 | 13.9 % / 23.7 % / 65.6 % |

The anat-to-centroid offset for LC11 decomposes as |d az| median 67.2 deg, |d el| median 21.6 deg -- and it
is the elevation half, not the azimuth half, that decides whether an elevation-0 sweep ever enters the box.
File: `scratchpad/window_geometry.csv`.

#### R5. "indistinguishable from a random column of the same eye" (section 3)

Two defects. **(a) No test is reported anywhere in the audit** -- the claim rests on comparing two medians.
On a paired, cell-by-cell Wilcoxon against the same permutation (my seed, 20 draws):

| dataset | type | median real | median permuted | cells closer than their permuted column | Wilcoxon p |
|---|---|---:|---:|---:|---:|
| MaleCNS | LC11 | 66.9 | 71.9 | 49.7 % | 0.958 |
| MaleCNS | LC4 | 61.4 | 65.4 | 50.0 % | 0.535 |
| FAFB | LC4 | 62.8 | 63.4 | 48.1 % | 0.935 |
| FAFB | **LC11** | 53.7 | 61.9 | **56.7 %** | **3.6e-4** |
| MaleCNS | LPLC2 | 30.8 | 62.1 | 67.0 % | 2.3e-4 |
| FAFB | LC10a | 12.2 | 52.4 | 91.3 % | 7.1e-36 |

Three of the four "chance" cases are confirmed indistinguishable; **FAFB LC11 is not** -- it is weakly but
significantly better than chance. The audit's phrase "the FAFB replicate says the same" is wrong for LC11.

**(b) "of the same eye" is false.** 64.3 % of MaleCNS LC11 cells and 42.1 % of MaleCNS LC4 cells are given a
column in the eye **contralateral** to their soma side (MaleCNS LC10a 30.2 %, LPLC2 36.8 %; FAFB LC11 18.9 %,
LC4 19.2 %; FAFB T2/T3 0.0 %). `hex_side` is `somaSide` for MaleCNS and equals `somaSide` for 100 % of FAFB's
45,528 annotated cells, so the comparison is like-for-like. The permutation control permutes within soma side
over a pool that is itself 64 % contralateral, so its draw is not "a column of the same eye" either.
Files: `scratchpad/lc_recheck.csv`, `scratchpad/eye_mixing.csv`.

#### R6. "the numbers are the same to the third decimal for every type" (section 2, `loo_malecns_set`)

False, and the sentence is contradicted by the numbers it itself then quotes. From `out/colgt/loo_per_type.csv`:
T2 exact 0.494 -> 0.477 (delta 0.017, second decimal), T4b 0.829 -> 0.821, T4a 0.784 -> 0.777, Tm6 0.534 -> 0.529;
T5a p90 61.95 -> 59.91 deg, T5b 67.34 -> 65.32 deg. Maximum deltas: 0.017 in the exact fraction, 0.005 in the
4.6-deg fraction, 2.0 deg in p90.

#### R7. `frac_same_column` is wrong in every row of `tables.fafb_annotation_depth`

A floating-point defect, not a science error. `full_vs_malecns_set_deg` is an `arccos` of a dot product and
returns up to **1.207e-6 deg for two IDENTICAL columns** -- the identity shortcut that `score_cells` applies
(`np.where(ic == tc, 0.0, deg)`, script line 117) was not applied at script line 395. `frac_same_column` then
tests `d < 1e-9` and drops those cells. Recomputed by integer column identity on the audit's own
`out/colgt/lc_per_cell.csv`:

| type | true identical | audit reports | true <= 4.6 deg | audit reports |
|---|---:|---:|---:|---:|
| LC11 | **0.701** | 0.630 | 0.717 | 0.717 |
| LC10a | **0.797** | 0.662 | 0.835 | 0.814 |
| LC4 | **0.942** | 0.923 | 0.971 | 0.942 |
| LPLC2 | **0.824** | 0.681 | 0.867 | 0.857 |
| T2 | **0.482** | 0.379 | 0.919 | 0.846 |
| T3 | **0.832** | 0.673 | 0.973 | 0.942 |
| Tm5Y | **0.954** | 0.753 | 0.966 | 0.961 |
| TmY21 | **0.879** | 0.677 | 0.901 | 0.883 |

The error direction *strengthens* the audit's own conclusion ("annotation depth is not the cause"). The same
table also prints `p90_full_vs_malecns_set_deg = 8.5e-7` for Tm5Y, which is this noise, not a distance.

---

### Confirmed

**C1. Leave-one-type-out, recomputed with my own propagation and my own distance.** My numbers against the
audit's, FAFB / MaleCNS:

| dataset | type | n | exact (mine / audit) | <= 4.6 deg (mine / audit) | p90 deg (mine / audit) |
|---|---|---:|---|---|---|
| FAFB | T2 | 1,450 | 0.494 / 0.494 | 0.839 / 0.847 | 4.6 / 4.6 |
| FAFB | T3 | 1,474 | 0.812 / 0.812 | 0.933 / 0.936 | 4.6 / 4.6 |
| FAFB | T4a | 1,441 | 0.784 / 0.784 | 0.892 / 0.891 | 4.6 / 4.6 |
| FAFB | Mi1 | 1,581 | 0.969 / 0.969 | 0.979 / 0.978 | 0.0 / 0.0 |
| FAFB | L1 | 1,572 | 0.992 / 0.992 | 0.996 / 0.996 | 0.0 / 0.0 |
| FAFB | T5a | 1,465 | 0.561 / 0.561 | 0.673 / 0.673 | 61.95 / 62.0 |
| MaleCNS | Mi1 | 1,762 | 0.828 / 0.828 (in-retina 1.000 / 1.000) | 0.828 / 0.828 | 0.0 / 0.0 |
| MaleCNS | L1 | 1,767 | 0.797 / 0.797 (in-retina 0.962 / 0.962) | 0.804 / 0.805 | 0.0 / 0.0 |

The `<= 4.6` residuals are the knife-edge described in Corrections 2. Failure partners reproduce exactly
(T5a: CT1 for 411 of 415 cells more than 10 deg off, 0.5 % annotated; T3: Mi1 871 + Tm1 578; T2's 31 bad cells
split LC14a-1 9 / Tm2 7 / Pm3 5). File: `scratchpad/loo_recheck.csv`.

**C2. Chance with a different RNG.** Seed 12345, 20 draws per type (audit: seed 0, 5 draws): chance exact
0.0008-0.0015 and chance median 55.0-57.9 deg on FAFB, 0.0012-0.0013 and 57.6-57.9 on MaleCNS. The audit's
"chance exact <= 0.002 and chance median >= 54.5 deg on both" holds. The inference is three orders of
magnitude above chance for the columnar stage.

**C3. The LC case reproduces to the third decimal.** MaleCNS LC11 143 cells / 13 distinct columns /
50-44-14-13 / largest share 0.350 / 94 median input columns / r50 21.68 / r80 67.25 / centroid-to-column
66.92 / 7.0 % inside r50. LC4 32 columns / 61.36 / 18.3 %. FAFB LC11 42 columns / 53.68 / 13.4 %; LC4 14
columns / 62.79 / 12.5 %. Per-cell values agree with `out/colgt/lc_per_cell.csv` to 1.2e-6 deg over all 418
LC bodies of both datasets. Strongest partners reproduce exactly, including the annotated fractions and the
median input share (MaleCNS LC11 TmY19b 59 / Li15 36 / MeLo10 11 / Li26 11, 78 distinct partner cells,
0 % annotated, 1.65 % of input; FAFB LC11 Li15 55 / Li33 26 / Li30 12, 65 partners, 3.1 %, 2.28 %;
LC4 FAFB Li28 68 / Li33 29, 24 partners, 0 %, 4.54 %).

**C4. The annotation-depth replicate (item 4 of the brief).** FAFB re-inferred with only the 15
MaleCNS-annotated types: LC11 **42 -> 41** distinct columns, median shift 0 deg, p90 16.21 deg, centroid
distance 53.68 -> 53.95 deg; LC4 14 -> 14 columns, p90 0. The claim "the MaleCNS 13-column collapse is the
male graph's partner structure, not annotation depth" stands (the "63 % identical" figure is R7's bug; the
true value is 70 %, which supports the claim more strongly). File: `scratchpad/depth_recheck.csv`.

**C5. The `object_localizer_r3.md` 1.2 cross-check, and the audit's explanation of the 64.5-vs-66.9 gap.**
Exactly right. Recomputing the centroid over **all** inputs rather than rate inputs gives MaleCNS LC11
median 64.5 deg / 100 input columns / r50 26.1 deg and LC10a 47.1 / 48 / 44.8 -- the four numbers of that
file's table, to the decimal.

**C6. The input-weighted centroid IS a sensible truth proxy for columnar cells (item 6 of the brief) -- a check
the audit never performs.** Distance from the centroid to the cell's **true annotated** column:

| dataset | L1 | Mi1 | T3 | T2 | Mi4 | L3 | T4a | Tm9 | T5a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FAFB (median deg) | 0.19 | 2.46 | 2.07 | 4.05 | 3.09 | 4.60 | 7.27 | 7.38 | 13.10 |
| MaleCNS (median deg) | 0.38 | 6.12 | -- | -- | 9.53 | 6.62 | -- | 9.39 | -- |

Against a chance level of ~55 deg, a 67-deg centroid-to-column distance is far outside anything a
retinotopically sensible assignment produces. Note especially **T5a: centroid-to-truth 13.1 deg while the
single inferred column's p90 is 62 deg** -- direct support for the audit's recommendation to fall back to
the centroid for T5 / L3 / Mi4. File: `scratchpad/proxy_recheck.csv`.

**C7. The symmetric form of the LC claim holds decisively.** Weighted 50 % input radius measured **about** the
inferred column, about the centroid, and about a uniformly random column:

| dataset | type | about the inferred column | about the centroid | about a random column |
|---|---|---:|---:|---:|
| MaleCNS | LC11 | **73.3** | 21.7 | 75.2 |
| MaleCNS | LC4 | **66.4** | 19.7 | 68.1 |
| MaleCNS | LC10a | 57.3 | 36.6 | 72.3 |
| MaleCNS | LPLC2 | 43.2 | 27.7 | 76.5 |
| FAFB | LC11 | 56.5 | 14.1 | 74.5 |
| FAFB | LC4 | **65.8** | 14.5 | 77.5 |
| MaleCNS | T3 | 4.6 | 11.6 | 77.1 |

Half of an LC11 cell's rate-input weight lies more than 73 deg from its inferred column, against 21.7 deg
from its centroid and 75.2 deg from a random column. **Section 3 and verdict item 3 are correct as
statements about the column; R1/R2 concern only what follows for a 15-deg box and for the round-2 statistic.**

**C8. Everything else checked.** 405 of 418 = 143 LC11 `anat` + 262 LC10a `anat`, reproduced from the maps.
Retinas 1,581 (FAFB, complete truth) and 1,466 (MaleCNS, 3,903 of 23,720 annotated cells in columns it lacks).
29 / 42,999 and 15 / 23,720 LOO cells. The audit uses **neither "silent" nor "inverted"** anywhere, in
keeping with `object_localizer_r3.md` 0. Every `files_written` path exists.

**C9. New, and in the audit's favour: the LC assignment is not even order-stable.** `column_of_cells` sweeps
`np.flatnonzero(col_r < 0)` in index order and writes in place, so a pass is Gauss-Seidel. Running the same
three passes synchronously (Jacobi) changes 18 % of all FAFB rate-unit columns, and for the LC types it
changes almost everything: LC11 **20.5 %** of cells keep their column and the distinct-column count goes
42 -> 64; LC4 21.2 %, 14 -> 24. T2 is 100.0 % stable and T3 99.9 %. The LC single column is a function of
the row order of `neurons.parquet`, which is an independent reason not to window on it -- worth adding.

---

### Corrections (exact replacement text)

**1. Section 4 item 4** -- replace from "The box size is not the issue:" to "uninformative rather than
negative." with:

> The box size is not what makes the window wrong: a 15-deg box on the inferred column still holds a median
> 15 % of an LC11 cell's rate-input weight (30-deg box 16 %; no cell holds zero), while half of that weight
> lies more than 73 deg away -- the box sits on a local weight spike, the strongest partner's own column,
> not on the centre of the field. Whether that mis-centring changed the round-2 reading was tested directly:
> re-windowing the same 84 fetched runs on the input-weighted centroid, and on a box of twice the cell's own
> r50 about that centroid, reproduces the LC11 null at every rung on the shipped lobe (smallest exact
> Mann-Whitney p 0.394 and 0.240 against 0.180 as shipped and 0.041 whole-window; Holm needs 0.00417), and
> reproduces the LC10a result (0.026 and 0.132 against 0.0087 as shipped). The sphere track is at elevation
> exactly 0 while the LC11 input centroids sit at a median |el| of 24 deg, so the corrected window is entered
> by **fewer** bodies than the anatomical one (17 of 143 against 55 at the 4.5-deg rung; an r50-sized centroid
> box reaches 61). **The round-2 LC11 sphere-ladder result is therefore negative under the corrected window
> as well as the shipped one, not uninformative**; what the mis-centring costs is the per-cell
> interpretability of the window, not the family call.

**2. Section 2** -- the `<= 4.6 deg` and `within r50` columns are knife-edge statistics and should carry a
tolerance. The modal one-lattice-step error is *exactly* 4.6 deg (columns one step apart in elevation), so
`de <= 4.6` on a floating-point `arccos` splits that mode arbitrarily: 194 of 1,450 FAFB T2 cells sit within
1e-6 of 4.6 deg and only ~90 pass. With a 1e-6 tolerance, T2 is 0.847 -> **0.919**, T3 0.936 -> **0.970**,
T4a 0.891 -> **0.951**. My independent distance gives 0.839 / 0.933 / 0.892 for the same cells. Add
`+ 1e-6` (or the `ic == tc`-style identity/adjacency shortcut) and restate; the same applies to
`frac_col_within_r50`, where the modal case is `d == r50` exactly (500 of 1,466 FAFB T2 cells, 87 of 237
FAFB LC10a) and the reported 0.925 / 0.527 / 0.671 move to 0.900 / 0.491 / 0.633 under a different float
path. LC11 and LC4 have 2-3 such ties, so the headline 7 % / 13 % / 18 % are robust.

**3. Section 3** -- replace "the result is **indistinguishable from a random column of the same eye**" with:

> the result is statistically indistinguishable from a permuted column of the same type and soma side
> (paired Wilcoxon over cells: MaleCNS LC11 p 0.96, LC4 p 0.54; FAFB LC4 p 0.94). FAFB LC11 is the exception --
> weakly but significantly better than the permutation (53.7 vs 61.9 deg, 57 % of cells closer, p 3.6e-4).
> "Of the same eye" would be wrong in either direction: 64 % of MaleCNS LC11 cells and 42 % of LC4 cells are
> given a column in the eye contralateral to their soma side (FAFB 19 % each), and the permutation pool is
> contralateral in the same proportion.

**4. Section 2, `loo_malecns_set`** -- replace "the numbers are the same to the third decimal for every type"
with "the numbers agree to within 0.017 in the exact fraction and 0.005 in the 4.6-deg fraction for every
type, with p90 shifts up to 2.0 deg for T5a-d".

**5. Section 3, `fafb_annotation_depth`** -- regenerate `frac_same_column` by column identity
(`inferred_col == inferred_col_malecns_set`) rather than `deg < 1e-9`, and restate: LC11 **70 %** identical
(not 63), LC4 **94** (not 92), LC10a **80** (not 66), LPLC2 **82** (not 68), T2 **48 % / 92 % within 4.6**
(not 38 / 85), T3 **83 / 97** (not 67 / 94), Tm5Y **95 / 97** (not 75 / 96). Same fix in the Report's
`key_claims` line ("63 % identical column" -> "70 %").

**6. Section 3** -- "their strongest partner is Tm5Y for 36 / 66 (FAFB) and 70 / 59 (MaleCNS) cells" reads as
though Tm5Y were the modal partner for all four. It is not for MaleCNS LPLC2, whose modal partner is the
wide-field **LPi43** (69 cells) with Tm5Y second (59) -- and MaleCNS LPLC2 is nonetheless well above chance
(30.8 vs 62.1 deg, p 2.3e-4). Add: a wide-field strongest partner does not by itself imply a chance column;
LC11 and LC4 differ because their partners' own columns are themselves collapsed onto a handful of positions.

**7. Section 2, "Why the failures fail"** -- the MaleCNS `n_bad` counts are inflated by construction, not only
for L3. `failure_partners` treats `deg_err = NaN` as `> bad_deg`, so every cell whose truth column the male
retina lacks is counted as a failure: MaleCNS Mi1 shows 303 "bad" cells (exactly the 1,762 - 1,459 cells with
no expressible truth) whose partner is L1 in 303/303 cases and which are *exactly correct* in hex terms
(`frac_exact_in_retina` 1.000). The caveat currently attached to L3 should be attached to the table.

**8. Add (supports the audit's own recommendation), section 3 or verdict item 3:**

> The single column is not even stable under the sweep order. `column_of_cells` updates in place while
> iterating in index order, so its three passes are Gauss-Seidel; running them synchronously changes the
> column of 79 % of FAFB LC11 cells and 79 % of LC4 cells (distinct columns 42 -> 64 and 14 -> 24), while
> leaving T2 100 % and T3 99.9 % unchanged. The LC assignment is a function of the row order of the neuron
> table.

**9. The precise wording the `object_baseline_r2.md` caveat should carry (replacing the Report's
`recommendations` line).** The current recommendation -- "add the caveat that its 405 anatomical LC boxes
were centred at chance relative to the cells' input fields, so the LC11 sphere-ladder null is uninformative
rather than negative" -- must not be written as it stands. Replace with:

> **Window caveat (from `docs/audits/column_ground_truth.md`).** The 405 anatomical LC boxes of this ladder
> (LC11 143/143, LC10a 262/275) were centred on `trace.column_of_cells`'s single column, which for LC11 and
> LC4 is inherited from a wide-field Li / TmY19b partner carrying ~2 % of the cell's input and lies a median
> 67 deg (MaleCNS) / 54 deg (FAFB) from the cell's input-weighted centroid, against 71 / 62 deg for a
> permuted column -- i.e. the box centre is not the cell's receptive-field centre, and it is not
> cell-specific (13 distinct columns for 143 LC11 bodies). The primary LC11 family was re-windowed on the
> input-weighted centroid and on a box of twice the cell's r50 about it, on these same 84 runs: the result is
> unchanged (smallest exact-U p 0.394 and 0.240 against 0.180 as shipped; Holm needs 0.00417), and so is
> LC10a's (0.026 and 0.132 against 0.0087). The sphere sweeps elevation 0 while the LC11 centroids sit at a
> median |el| of 24 deg, so the corrected window is entered by fewer bodies, not more (17 of 143 at the
> 4.5-deg rung against 55). **The null stands under the corrected window; the caveat is about per-cell
> interpretability and coverage, not about the family call.** Future rounds should place LC windows on the
> input-weighted centroid with a box no smaller than r50, or on a fitted RF, and should never use the single
> `column_of_cells` column for a `visual_projection` cell.

The same caveat belongs on `docs/audits/object_rectangles_r3.md` section 6, which inherits the identical
windows (all 143 LC11 anatomical, `RF_MAPS` explicitly not retrofitted) and is likewise null. It does **not**
belong on `object_localizer_r3.md`: that round fitted inside a 20-deg box on the input-weighted centroid and
kept the single column only as a reported distance, so "object rounds 2-3" over-reaches -- name the sphere
ladder and the round-3 rectangle ladders.

**10. Numbers in the Report with no file behind them.** Only one matters and it is R1's ("no box of 15-30 deg
placed at that centre overlaps the cell's inputs for most LC11 cells") -- no `out/colgt` table measures box /
input overlap, and the claim is false. Minor: "the 55 / 99 / 103 windowed bodies per rung" compresses
`object_baseline_r2.md`'s six-rung 55 / 99 / 99 / 99 / 103 / 103; "a ~45-deg box" and "exact capped at ~0.83"
are derived, not tabulated, which is fine but should be flagged as derived. Everything else in `key_claims`
traces to `loo_per_type.csv`, `loo_failure_partner.csv`, `lc_summary.csv`, `lc_strongest_partner.csv`,
`lc_cross.csv`, `fafb_annotation_depth.csv` or `summary.baseline`.

---

### Verdict

**Mostly sound.**

Sections 1-3 are a genuine, well-designed and independently reproducible ground-truth check. I rebuilt the
propagation, the distance, the permutation and the LC statistics from scratch and matched the audit to the
third decimal on every headline number, with a different RNG. The two central *measurement* claims survive
intact and are strengthened by checks the audit did not run: the propagation recovers columnar types to one
ommatidium three orders of magnitude above chance; the LC11 / LC4 single column really is at or near the
permutation distance from the cell's input field (half the input weight lies >73 deg away, against 75 deg
for a uniformly random column), it is inherited from an unannotated wide-field partner carrying ~2 % of the
input, it is shared by up to 50 cells at a time, and it is not even stable under the propagation's sweep
order. The input-weighted centroid is validated as a truth proxy for the first time here (0.2-7.4 deg from
the true column for FAFB columnar types).

What fails is section 4 -- the consequential part. The load-bearing sentence about box/input overlap is
false on its own terms (R1), its operational restatement is false for this stimulus (R2), and the conclusion
that follows from them -- that the round-2 LC11 null is uninformative rather than negative, which is the
claim that would change the reading of two published rounds and the Neurome delivery -- does not survive a
direct test on the runs themselves (R3): the null is reproduced under the audit's own recommended window,
and the audit's recommended window is in fact entered by *fewer* LC11 bodies than the one that was used (R4).
Add one inferential overstatement (R5, an untested "indistinguishable" that is false for FAFB LC11 and whose
"same eye" is wrong for 64 % of MaleCNS LC11 cells), one self-contradicting sentence (R6) and one
floating-point defect that corrupts every row of a shipped table (R7).

The recommendations themselves are right and should stand: do not window a `visual_projection` cell on the
single column, use the centroid with a box >= r50 or a fitted RF, and restrict the strongest-partner vote to
columnar partners for T5 / L3 / Mi4. It is the retro-active re-reading of round 2 that must be withdrawn and
replaced with Correction 9.
