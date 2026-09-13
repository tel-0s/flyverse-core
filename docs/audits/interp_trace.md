# interp:trace -- where along the depth from a sensory population is a stimulus lost (validation record)

**Tool.** `flyverse/interp/trace.py::trace` (CPU analysis), `scripts/interp_trace.py` (`record` on the cluster, `analyse` /
`audit` locally), `tests/test_interp.py::TraceTests` (7 CPU tests on the 8-neuron synthetic graph). Contract:
`docs/INTERP.md` 4.2. Date 2026-09-12/13. The toolkit reads the model; nothing in `flyverse/brain.py`, `optic.py`,
`body.py`, `programs.py`, `room_demo.py` or `room_ui.py` was touched.

**What the tool does.** Three arms of independent runs -- stimulus, matched control, control-again (the null) -- are
recorded as per-cell time-means over the scored window (every cell of the model: `rate_hz` from `spike_counts`,
`drive_mv` / `drive_mv_abs` from `brain.drive`, `optic_dr` / `optic_dr_abs` = `optic.rates() - r0` on the rate units)
plus a per-type pooled series. Per type and per run one statistic (`best_cell` = the object sweep's max-over-cells of
the per-cell (A - B) time-mean; `mean`; `figure_z` = probe_figure_stages' retinotopic figure; `dprime` = screen.rank on
the pooled series), then `common.compare` of the (stimulus - control) draws against the (null - control) draws: z on
the null SD, Welch, exact Mann-Whitney U / p, verdict. Types are ordered by their depth from the source over the
type-level input-share graph of the shaped weights (`TypeGraph`: edge P -> Q when the mean Q cell takes >= 2 % of its
|input| from P; untyped cells are never a node). Two first-lost rules are reported: the contract's depth rule and the
input rule (a non-carrier taking >= 20 % of its input from carriers; the first lost stage is the smallest such depth),
and the lost types are decomposed: the type-level input table with each input's own verdict and retinotopic signed
figure (the `optic_measures.md` 5.3 reading as a table), then the decompose tool's static and dynamic tables prefixed
`lost_static_` / `lost_dynamic_`.

---

## 1. The batch

One `cluster_run.py` call, 30 jobs (2 protocols x 3 arms x 5 runs), 0 failed, 8.0 min submit-to-fetch, run dir
`<cluster-fs>/neurome/runs/trv-dee1e7`, console log `out/trv_cluster.log`, recordings `out/trv/` (150 files, 375 MB;
`np.savez_compressed`, the pooled series every 5th frame). Every job realised `device cuda (NVIDIA B200)` (30/30
`out/trv/*.txt`); the JSON sidecars carry the resolved LIFParams (receptor `sign` / `abs`) and OpticParams, the cache
fingerprint `ef23cc27bea13be7f6a96f3c04fd3737`, the backend flags (event_driven, cuda_kernels, cuda_sparse warp) and
the retina record (1,466 columns, column -> photoreceptor bodies). `flyverse_commit` is `unknown` inside the per-run
cluster copy (no `.git` there); the analysis block of every Result records the local commit (`0d32fd6`, dirty with the
untracked toolkit files).

```
cmds=(); for proto in object odour; do pre=$([ "$proto" = object ] && echo obj || echo od); for a in stim ctrl null; do for s in 0 1 2 3 4; do
  cmds+=("python -c 'import torch; assert torch.cuda.is_available()' && mkdir -p out/trv && python scripts/interp_trace.py record --protocol $proto --arm $a --seed $s --series-every 5 --out out/trv/${pre}_${a}_r$s > out/trv/${pre}_${a}_r$s.txt 2>&1; tail -4 out/trv/${pre}_${a}_r$s.txt"); done; done; done
python scripts/cluster_run.py --name trv --minutes 40 "${cmds[@]}" --fetch out/trv/ > out/trv_cluster.log 2>&1
```

Protocols (each an existing probe's, reused): `object` = `probe_object_sweep.py` (pinned on the empty fenced apple
table, 1 cm black ball 5 cm ahead sweeping +-6 cm in 3 s per pass, 12 s scored after 3 s settle; `stim` = ball,
`ctrl` = no ball, `null` = no ball again) with `probe_figure_stages.run_ball`'s column sets (8 deg object radius, +20 deg
background margin); `odour` = `screen_odour.py --fruit apple` sites, pinned 30 s with 3 s skipped (`stim` =
`apple8_into_wind`, `ctrl` = `clean_into_wind`, `null` = `clean_into_wind` again). Per job: object 1,200 frames scored
in 95 s wall, odour 2,700 frames in 114 s (`out/trv/obj_stim_r0.txt`, `od_stim_r0.txt`).

**Why five runs per arm.** `common.compare` calls 'result' only at p <= 0.05, and the exact two-sided Mann-Whitney p of
n v n runs floors at 2 / C(2n, n): 0.10 at 3 v 3, 0.029 at 4 v 4, 0.0079 at 5 v 5. So the contract's minimum of three
runs can never yield a 'result', whatever the z (the toy test shows Mi4 at z 27 and p 0.10 -> 'null'). The tool writes
`p_floor` per row and a `p_floor_note` in the summary whenever the floor exceeds 0.05; five runs match
`object_sweep.md` 8.4.

---

## 2. The object stage (validation target 1)

`out/interp/trace/object_stage.json` (`--stat best_cell --stage-table family --decompose-at T3,T2,Tm5Y,TmY21`; console
`object_stage_console.txt`; analysis 1 m 47 s on the CPU). Source: the photoreceptors (5,930 cells); 5,908 types
scored (>= 3 recorded cells, reached within depth 6); stimulus 5 runs vs null 5 draws; p floor 0.0079. Statistic:
rate units `diff_abs_best_cell_mean` (max over cells of the (A - B) time-mean |deviation|, rate units), spiking VPNs
`diff_max_over_cells_mean_mv` (the same on the optic drive, mV) -- the statistic of `object_sweep.md` 8.4 / 8.7.

Reference columns: `object_sweep.md` 8.7 / 8.4 (5 seeds each, damped gains of round 3; z in the `off` and the
`sign-abs` mode) and `optic_measures.md` 6 (the shipped gains, 3 seeds). Numbers here are under the shipped defaults.

| type | depth | stage | stim mean | null mean +- SD | z | Welch | U | p | verdict | 8.7 z off / sign-abs | optic_measures 6 z |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L1 | 1 | 1 | 0.1256 | 0.0191 +- 0.0080 | +13.3 | +28.8 | 25 | 0.0079 | result | -- | -- |
| L2 | 1 | 1 | 0.1009 | 0.0205 +- 0.0058 | +13.9 | +30.1 | 25 | 0.0079 | result | -- | -- |
| **Mi4** | 2 | 2a | 0.0478 | 0.0112 +- 0.0055 | **+6.6** | +13.5 | 25 | 0.0079 | result | +28.6 / +22.3 | +35 |
| **Mi1** | 2 | 2a | 0.0724 | 0.0149 +- 0.0034 | **+17.2** | +36.9 | 25 | 0.0079 | result | +7.8 / +27.9 | +7.6 |
| **Tm3** | 2 | 2b | 0.1081 | 0.0239 +- 0.0080 | **+10.5** | +21.6 | 25 | 0.0079 | result | +7.8 / +15.6 | +12.5 |
| Mi9 | 2 | 2a | 0.0743 | 0.0103 +- 0.0045 | +14.2 | +30.9 | 25 | 0.0079 | result | -- | -- |
| Tm2 | 2 | 2b | 0.0840 | 0.0247 +- 0.0054 | +11.0 | +20.3 | 25 | 0.0079 | result | -- | -- |
| Tm1 | 2 | 2b | 0.0716 | 0.0249 +- 0.0106 | +4.4 | +9.4 | 25 | 0.0079 | result | -- | -- |
| Tm4 | 2 | 2b | 0.1035 | 0.0354 +- 0.0157 | +4.3 | +9.7 | 25 | 0.0079 | result | -- | -- |
| Tm20 | 2 | 2b | 0.1007 | 0.0343 +- 0.0155 | +4.3 | +9.0 | 25 | 0.0079 | result | -- | -- |
| Tm9 | 2 | 2b | 0.0667 | 0.0324 +- 0.0118 | +2.9 | +6.3 | 25 | 0.0079 | null (z < 3) | -- | -- |
| **Tm5Y** | 2 | 2b | 0.0512 | 0.0390 +- 0.0142 | **+0.9** | +1.8 | 20 | 0.15 | null | +0.4 / +2.6 | +1.1 |
| **TmY21** | 2 | 2b | 0.0399 | 0.0448 +- 0.0185 | **-0.3** | -0.5 | 9 | 0.55 | null | +1.1 / +0.6 | -0.1 |
| **T2** | 3 | 3 | 0.0409 | 0.0439 +- 0.0196 | **-0.2** | -0.3 | 13 | 1.00 | null | +0.5 / +1.2 | +0.2 |
| **T3** | 3 | 3 | 0.0278 | 0.0258 +- 0.0101 | **+0.2** | +0.3 | 14 | 0.84 | null | -0.0 / +0.7 | +0.4 |
| TmY13 | 3 | 2b | 0.0224 | 0.0263 +- 0.0086 | -0.5 | -0.9 | 9 | 0.55 | null | -0.6 / +0.5 | -0.3 |
| TmY5a | 3 | 2b | 0.0414 | 0.0430 +- 0.0138 | -0.1 | -0.2 | 12 | 1.00 | null | -1.0 / +1.2 | +0.1 |
| T4c | 3 | 3 | 0.0517 | 0.0103 +- 0.0029 | +14.5 | +30.1 | 25 | 0.0079 | result | -- | -- |
| T4d | 3 | 3 | 0.0502 | 0.0090 +- 0.0026 | +15.6 | +22.7 | 25 | 0.0079 | result | -- | -- |
| T5a | 3 | 3 | 0.0856 | 0.0345 +- 0.0074 | +6.9 | +12.3 | 25 | 0.0079 | result | -- | -- |
| LPi34 | 4 | 4 | 0.0709 | 0.0215 +- 0.0084 | +5.9 | +13.1 | 25 | 0.0079 | result | -- | -- |
| **LC11** (mV) | 3 | 5 | 0.0592 | 0.0567 +- 0.0271 | **+0.1** | +0.2 | 13 | 1.00 | null | -0.1 / +0.4 | -0.2 |
| **LC10a** (mV) | 2 | 5 | 0.0844 | 0.0828 +- 0.0365 | **+0.0** | +0.1 | 12 | 1.00 | null | +0.3 / -0.1 | -0.1 |
| LPLC2 (mV) | 3 | 5 | 0.3010 | 0.1653 +- 0.0661 | +2.1 | +4.0 | 25 | 0.0079 | null (z < 3) | +2.0 / +5.4 | +1.2 |
| LC10b (mV) | 3 | 5 | 0.1633 | 0.1319 +- 0.0859 | +0.4 | +0.5 | 15 | 0.69 | null | +0.7 / +1.0 | -1.2 |
| LC16 (mV) | 3 | 5 | 0.0901 | 0.0743 +- 0.0127 | +1.2 | +0.9 | 14 | 0.84 | null | -0.1 / -0.8 | +0.5 |
| LC4 (mV) | 3 | 5 | 0.0900 | 0.0653 +- 0.0248 | +1.0 | +1.3 | 19 | 0.22 | null | -0.1 / -1.0 | -0.6 |

Per-run values of the headline rows (stimulus draws / null draws): Mi4 .0485 .0485 .0498 .0484 .0436 / .0201 .0125
.0104 .0063 .0069; Mi1 .0725 .0734 .0711 .0731 .0720 / .0109 .0123 .0190 .0147 .0173; Tm3 .1081 .1042 .1073 .1136
.1074 / .0119 .0231 .0340 .0268 .0236; T3 .0206 .0387 .0331 .0190 .0278 / .0191 .0352 .0370 .0139 .0239; LC11 (mV)
.0471 .0911 .0527 .0491 .0561 / .0793 .0721 .0770 .0343 .0209; LPLC2 (mV) .2588 .2832 .3238 .2870 .3523 / .1451 .1565
.2502 .0733 .2016.

**Verdict: reproduced** (`validation.status`). The tool places Mi4 / Mi1 / Tm3 above the null (z +6.6 / +17.2 / +10.5,
all five stimulus draws above all five null draws) and T2 / T3 / Tm5Y / TmY21 / LC11 / LC10a at it (|z| <= 0.9, U 9-20).
LPLC2 repeats 8.4's `off` row exactly in kind (z +2.0 there, +2.1 here; U 25 / p 0.0079 in both) and is 'null' by the
|z| >= 3 rule in both. The digits differ from 8.7 -- Mi4's null SD is 0.0055 here against 0.0017-0.0023 there, so its
z is +6.6 against +22 to +35 -- which is the +-1.5 (and, on a 5-draw null SD, larger) scatter the design warned about:
the target is the ordering and the verdicts, and those agree with 8.7 and with `optic_measures.md` 6 row for row.

Carriers per depth / stage (result / scored): depth 1 7/27, 2 16/196, 3 15/670, 4 5/1967, 5 0/2223, 6 0/825; stage 1
lamina 7/11, 2a medulla intrinsic 9/83, 2b Tm/TmY 7/57, 3 T cells 8/23, 4 Li/LPi 3/41, 5 VPN 8/178, 6 VCN 1/38. The
carriers at depth 3-4 are the motion pathway (T4a-d, T5a/c/d, LPi34/43, LPi3412, LLPC1-3, LPC1/2, LC15, LoVP74, TmY14,
TmY4, TmY20, LPT111); the small-object pathway (T2, T3, Tm5Y, TmY21, TmY13, TmY5a -> LC11 / LC10a) carries nothing.

### 2.1 The same trace under probe_figure_stages' retinotopic statistic

`--stat figure_z` scores the figure itself -- (A - B) in the object columns minus (A - B) in the background columns,
per run -- and compares the figure values across runs (`object_figure_z.json`, signed deviation; `object_figure_z_abs.json`,
`--quantity graded=optic_dr_abs,spiking=drive_mv_abs`, the probe's `abs` measure). 188 types have >= 5 object and >= 20
background cells. Signed: only L1 / L2 (z -5.1 / -4.5), T4c / T4d (+7.2 / +5.9) and T2 (+3.3, U 25) carry -- for the
sweeping ball the ON / OFF transients cancel in the signed time-mean (`probe_figure_stages` says so) -- and the
validation reads **not reproduced** on that measure, which is the correct reading of a statistic the object does not
survive. Rectified: L2 +7.1, L1 +4.7, Tm1 +17.5, Mi9 +12.5, Tm4 +7.6, Tm3 +5.7, Mi1 +5.5, Tm2 +4.7, Mi4 +3.6, T4c +7.2,
T4d +5.9 result (U 25 each); Tm5Y +2.2 (U 22, p 0.056), TmY21 +0.2, T2 +0.5, T3 +0.5, TmY13 -0.2, TmY5a -0.1, LC11 -0.6,
LC10a +0.1, LPLC2 +2.4 (U 22, p 0.056), LC4 +0.3, LC16 +0.7, LC10b -0.8 null -- **reproduced**. The per-run figure z of a
single run is +1.2 to +2.7 on the medulla types here (the probe's single-run |z| >= 3 threshold would call most of them
non-carriers); it is the five-run comparison of the figure values against their own null that resolves them.

### 2.2 The lost stage decomposed: T3's ON / OFF summation (validation target 1b)

`lost_inputs` for `--decompose-at T3,T2,Tm5Y,TmY21` (type-level shaped weights `A`, mV per post cell per presynaptic
volley, share of the target's |input|; the input's own verdict, its z from the table above, its retinotopic signed
figure = mean over runs of the object-minus-background signed deviation, and the term sign x share x figure). Side by
side with `optic_measures.md` 5.3 (the static apple, the optic lobe's own normalised rate input):

| target | input | share here | 5.3 share | sign | mV / volley | input verdict (z) | signed figure | term |
|---|---|---|---|---|---|---|---|---|
| T3 (z +0.2) | Mi1 | 22.4 % | 22.5 % | + | +14.7 | result (+17.2) | +2.15e-4 | +4.80e-5 |
| | Tm1 | 16.0 % | 16.0 % | + | +10.5 | result (+4.4) | -2.18e-4 | -3.48e-5 |
| | Tm3 | 8.1 % | 8.2 % | + | +5.3 | result (+10.5) | +1.85e-5 | +1.50e-6 |
| | Tm4 | 6.6 % | 6.7 % | + | +4.3 | result (+4.3) | -1.27e-4 | -8.32e-6 |
| | Pm5 | 5.4 % | 5.6 % | - | -3.6 | null (+0.6) | -7.13e-4 | +3.88e-5 |
| | Pm1 | 5.1 % | 5.3 % | - | -3.4 | null (+1.5) | -8.58e-5 | +4.41e-6 |
| | Mi2 | 3.3 % | 3.3 % | - | -2.1 | null (-0.0) | +5.28e-4 | -1.72e-5 |
| | Li26 | 2.9 % | 3.2 % | - | -1.9 | null (-0.3) | -- | -- |
| T2 (z -0.2) | Tm2 11.9 % + (result +11.0, fig +5.3e-5); L5 8.7 % + (result +3.8, +6.3e-5); Tm3 7.4 % + (result); Pm2a 4.4 % - (null); Mi1 3.6 % + (result); Mi2 3.4 % -; TmY3 3.0 % + (null +1.7); C3 2.5 % - (result +7.4, fig +8.2e-4) | 5.3: Tm2 11.3, L5 8.3, Tm3 7.0, T2 4.7, Pm2a 4.2, Mi1 3.4, Mi2 3.2, TmY3 2.8 | | | | | |
| Tm5Y (z +0.9) | Tm20 13.6 % + (result +4.3, fig +1.6e-4); Li19 8.9 % - (null); Y3 7.4 % + (null); Li25 4.4 % -; Tm32 3.8 % -; Tm5a 3.4 % + (result +4.7, fig -6.8e-5); T2a 3.1 % +; Dm8a 2.7 % - (result); TmY20 1.7 % + (result, fig -1.0e-3) | 5.3: Tm20 14.0, Li19 9.3, Y3 7.5, Li25 4.6, Tm32 3.9, Tm5a 3.4, T2a 3.2 | | | | | |
| TmY21 (z -0.3) | TmY5a 8.4 % - (null); TmY13 6.4 % + (null); Tm20 4.9 % + (result); Tm5a 4.5 % + (result); Tm5Y 3.2 % + (null); TmY17 3.1 % + (null); Dm3a 2.9 % - (result) | 5.3: TmY5a 9.7, TmY13 7.6, Tm20 5.8, Tm5a 5.1, Tm5Y 3.8, TmY17 3.5, Dm3a 3.4 | | | | | |

`lost_cancellation` (summary): **T3** -- carriers raising {Mi1, Tm3} +4.95e-5 vs lowering {Tm1, Tm4} -4.31e-5, linear
estimate +6.4e-6 against T3's own signed figure -2.2e-5, cancellation fraction 0.93, carrier share 0.530, all of it
excitatory (`receptor:exact` on every entry). **T2** -- {Tm2, L5, Tm3, Mi1} +2.09e-5 vs {C3} -2.04e-5, cancellation 0.99,
carrier share 0.34. **Tm5Y** -- {Tm20, Dm8a} +2.55e-5 vs {Tm5a, TmY20} -2.03e-5, 0.89, carrier share 0.21. **TmY21** --
{Tm20} +8.0e-6 vs {Tm5a, Dm3a} -9.5e-6, 0.91, carrier share 0.12. This is `optic_measures.md` 5.3's hand reading
produced by the tool: at T3 the ON carriers (Mi1, Tm3) and the OFF carriers (Tm1, Tm4) arrive with opposite figures
through excitatory synapses of comparable weight (30.5 % vs 22.6 % of T3's input) and cancel; at Tm5Y / TmY21 the
carrier share is a fifth or an eighth of the input (dilution). Two differences from 5.3, both expected: the shares are
the LIF's shaped weights (`A`; `same_type_gain` removes T2 -> T2, which 5.3's optic matrix keeps at 4.7 %), and the
figure signs are those of the sweeping ball, not the static apple (Mi1 +, Tm1 - here; -8.8 / +4.5 z there).

Composition with the decompose tool (`summary.decompose`: static ok, dynamic ok): `lost_static_per_type` (1,654 rows)
gives T3 <- Mi1 +14.68 mV / volley over 6,659 entries from 1,773 cells, 103,700 raw synapses, share 0.221; Tm1 +10.50,
6,300 entries, 74,100 synapses, 0.158; Tm3 +5.31 / 37,490 / 0.080; Tm4 +4.31 / 30,380 / 0.065; Pm5 -3.58 (182 cells,
25,230 synapses, 0.054); Pm1 -3.38 (246 cells, 23,810, 0.051). `lost_dynamic_per_type` (kind `optic_input`, the
OpticParams rebuilt from the recording's provenance, 5 stimulus vs 5 control runs) is the whole-population
rate-weighted recurrent input: T3 <- Mi1 +1.74e-3 vs +1.66e-3 (ball vs none; z +0.5), Tm3 +4.0e-4 vs +3.4e-4, Pm1
-1.03e-3 vs -0.98e-3, Tm1 -5.9e-4 vs -5.4e-4 -- no term differs beyond scatter (|z| <= 0.85, every verdict null),
because the population mean over 1,940 T3 cells hides a figure that lives in ~40 columns. The retinotopic
`lost_inputs` table is the reading; the dynamic table is the whole-field context.

### 2.3 What the default first-lost rules name (`object_default.json`)

The contract's depth rule gives `first_lost_depth = 5` (depth 4 still holds 5 carriers of 1,967 scored types, depth 5
none of 2,223): for a stimulus that the motion pathway carries to the lobula plate, the depth at which *no* type carries
is past the optic lobe. The input rule lists 160 non-carriers taking >= 20 % of their input from carriers, by depth:
at depth 1 aMe12 (0.29 from Dm9), Tm5c (0.27, L3 + Dm9; z +2.8) and MeVP11 (0.24, L3 + Mi4) -- these are what
`--decompose-at first_lost` decomposes (Tm5c: L3 raising vs Dm9 lowering, cancellation 0.55; MeVP11: L3 vs Mi4, 0.96);
ranked by carrier share instead: Pm1 0.88 (Mi1, Tm3, TmY14), VS 0.88 (T5d, T4d, T5a, T4a, LPi34, T4b), Dm19 0.88 (L2,
Dm15), Li28 0.87, VST1 0.86, Pm7 0.85, Tlp14 0.84, Dm6 0.84; T3 sits at 0.53, TmY13 0.49, T2 0.34. So the tool's honest
default answer is "the figure is discarded first by the wide-field pooling cells (Dm, Pm, Li, VS / VST) at every
depth, and the small-object pathway is one of many non-carriers fed by carriers"; the targeted question -- T3's ON /
OFF summation -- is asked with `--decompose-at T3,T2,Tm5Y,TmY21`, and both are in the JSON (`lost_candidates` has all
160 with their carriers and shares).

---

## 3. The LH odour gate (validation target 2)

`out/interp/trace/odour_mean.json` (`--source class=olfactory --stat mean --min-cells 2`; the ORNs, 2,639 cells; 11,147
types scored; 1 m 54 s) and `odour_dprime.json` (`--stat dprime`, screen.rank's d' on the pooled series, stimulus vs
control per run). Levels in Hz over the 27 s window; references: NOTES session 8 (`screen_odour --fruit apple`:
LHPD4d1 20.6 Hz at 8 cm, 12.5 at 40 cm, 3.4 plume-free, d' 4.51) and `benchmark_suite.md` (channel 17.48 / 4.43;
LHPD4d1 19.4 / 3.4, LHAV4a1_a 20.4 / 4.5, LHAV4a1_b 18.0 / 4.7, LHCENT12_a 17.4 / 4.7, LHPD2a1 15.9 / 4.3, LHPD5c1
16.5 / 6.6).

| type | depth | cells | apple 8 cm | plume-free | diff | null mean +- SD | z | U | p | verdict | d' (stim vs ctrl) | reference (8 cm / clean) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ORN_DM1 | 0 | 74 | 88.30 | 4.93 | +83.37 | 0 +- 0 | -- (deterministic null) | 25 | 0.0079 | result* | 30.3 | -- |
| ORN_DM4 | 0 | 32 | 75.30 | 3.76 | +71.53 | 0 +- 0 | -- | 25 | 0.0079 | result* | 24.4 | -- |
| ORN_VA2 | 0 | 83 | 69.39 | 3.41 | +65.98 | 0 +- 0 | -- | 25 | 0.0079 | result* | 24.3 | -- |
| ORN_DC1 | 0 | 32 | 45.84 | 2.35 | +43.49 | -0.000 +- 0.006 | +7013 | 25 | 0.0079 | result | 16.5 | -- |
| DM1_lPN | 1 | 2 | 89.36 | 26.36 | +62.98 | +0.02 +- 0.13 | +473 | 25 | 0.0079 | result | 21.9 | -- |
| VA2_adPN | 1 | 2 | 77.67 | 15.95 | +61.67 | +0.06 +- 0.13 | +477 | 25 | 0.0079 | result | 27.0 | -- |
| DC1_adPN | 1 | 2 | 65.88 | 21.83 | +44.19 | -0.13 +- 0.38 | +116 | 25 | 0.0079 | result | 9.7 | -- |
| DM4_adPN | 1 | 2 | 56.80 | 15.16 | +41.67 | -0.03 +- 0.13 | +321 | 25 | 0.0079 | result | 18.0 | -- |
| **LHPD4d1** | 2 | 2 | **19.41** | **3.31** | +16.20 | -0.10 +- 0.24 | +66.9 | 25 | 0.0079 | result | 9.6 | 20.6 / 3.4 (screen); 19.4 / 3.4 (bench) |
| LHAV4a1_a | 2 | 2 | 20.36 | 4.59 | +15.83 | -0.05 +- 0.10 | +162 | 25 | 0.0079 | result | 8.5 | 21.9 / 4.5; 20.4 / 4.5 |
| LHAV4a1_b | 2 | 8 | 18.13 | 4.70 | +13.42 | +0.01 +- 0.07 | +188 | 25 | 0.0079 | result | 8.3 | 19.0 / 4.7; 18.0 / 4.7 |
| LHCENT12_a | 2 | 2 | 17.58 | 4.65 | +12.93 | -0.00 +- 0.08 | +168 | 25 | 0.0079 | result | 6.2 | 17.4 / 4.7 |
| LHPD2a1 | 2 | 9 | 15.95 | 4.38 | +11.63 | -0.05 +- 0.09 | +137 | 25 | 0.0079 | result | 6.7 | 15.9 / 4.3 |
| LHPD5c1 | 2 | 2 | 16.35 | 7.06 | +9.24 | +0.05 +- 0.18 | +52 | 25 | 0.0079 | result | 4.4 | 16.5 / 6.6 |
| KCab-m | 2 | 536 | 2.37 | 1.05 | +1.32 | -0.00 +- 0.02 | +88 | 25 | 0.0079 | result | 1.1 | -- |
| MBON07 | 3 | 4 | 11.61 | 5.14 | +6.48 | -0.02 +- 0.07 | +99 | 25 | 0.0079 | result | 3.4 | -- |
| MBON13 | 3 | 2 | 8.47 | 4.19 | +4.29 | -0.01 +- 0.04 | +120 | 25 | 0.0079 | result | 2.4 | -- |
| DNg99 | 4 | 2 | 10.63 | 8.15 | +2.58 | -0.10 +- 0.34 | +7.6 | 25 | 0.0079 | result | 1.2 | -- |
| DNp18 | 4 | 2 | 31.37 | 29.88 | +1.64 | -0.15 +- 0.26 | +6.3 | 25 | 0.0079 | result | 0.7 | wind-driven at both sites (heading into the wind) |
| DNp12 | 5 | 2 | 18.31 | 16.90 | +1.26 | +0.15 +- 0.24 | +5.2 | 25 | 0.0079 | result | 0.4 | -- |
| DNge016 | 5 | 2 | 25.19 | 24.23 | +0.95 | +0.01 +- 0.19 | +5.1 | 25 | 0.0079 | result | 0.3 | -- |
| DNp33 | 4 | 2 | 6.60 | 6.37 | +0.16 | +0.06 +- 0.18 | +0.9 | 18 | 0.31 | null | 0.05 | -- |
| DNa02 | 5 | 2 | 0.03 | 0.01 | +0.02 | -0.00 +- 0.02 | +1.1 | 19 | 0.22 | null | 0.02 | -- |
| DNa01 | 5 | 2 | 1.66 | 1.66 | -0.03 | +0.03 +- 0.10 | -0.3 | 13 | 1.00 | null | 0.01 | -- |
| DNp09 | 5 | 2 | 3.10 | 3.34 | -0.29 | +0.05 +- 0.35 | -0.8 | 6 | 0.22 | null | -0.2 | -- |
| MDN | 6 | 4 | 8.04 | 8.35 | -0.50 | +0.19 +- 0.61 | -0.8 | 6 | 0.22 | null | -0.3 | -- |
| WED080 | 3 | 2 | 21.87 | 22.07 | -0.19 | -0.01 +- 0.11 | -1.6 | 3 | 0.056 | null | -0.1 | -- |

\* `note = null_sd_zero`: the ORN input under a fixed plume is deterministic, every null draw is identical, so
`compare`'s z (diff / SD null) is undefined; the tool then decides by the exact rank test (p 0.0079, difference
non-zero) and records the note (section 5).

**Verdict: reproduced.** ORN -> PN -> LH is depth 0 -> 1 -> 2 from the ORNs (LHPD4d1 through VM7d_adPN 8.5 % /
DM1_lPN; LHAV4a1_a/b through DP1m_adPN; LHCENT12_a through DM4_adPN), and the apple channel's five types read 19.4 /
20.4 / 18.1 / 17.6 / 16.0 Hz at 8 cm against 3.3 / 4.6 / 4.7 / 4.7 / 4.4 plume-free -- the benchmark's numbers to 0.1-0.2
Hz, session 8's screen to 1-1.5 Hz (a 30 s window at one site there, 27 s here). LHPD4d1 is the sharpest of the five
against its own null (null SD 0.24 Hz on a +16.2 Hz difference). The d' column is screen.rank's statistic on this
run's stimulus-vs-control pair (skip 0, 1 s smoothing, the series at 20 Hz), not the screen's worst-fruit-site vs
best-clean-site d' of 4.51 -- same statistic, a different pair of conditions, so 9.6 vs 4.51 is not a discrepancy to
resolve.

**The loss beyond the LH.** 461 descending types were scored; 47 are 'result' and the largest difference among them
is +3.4 Hz (DNg56 16.0 vs 12.5; DNp32 +3.2, DNge054 +2.9, DNpe002 +2.8, DNg99 +2.6), i.e. the +12 to +16 Hz channel of the
LH arrives at the descending stage at a fifth of its size at best; DNp18 +1.6 Hz on 30 Hz (both sites face into the
wind; this is the wind response, not the odour), DNa02 0.03 vs 0.01 Hz, DNa01 / DNp09 / MDN / DNp33 null. The mushroom
body carries it (KCab-m +1.3 Hz over 536 cells, MBON07 +6.5, MBON14 +7.7, MBON13 +4.3). Carriers per depth (result /
scored): 0 6/53, 1 151/204, 2 514/901, 3 299/1901, 4 263/3523, 5 149/3682, 6 30/883 -- a brain-wide stimulus has a
carrier at every depth, so the depth rule returns `first_lost_depth = None` and the input rule's first candidates are
the multiglomerular PNs and the hygro-/thermo-receptor neurons fed by the antennal-lobe LNs (M_lvPNm30 0.51,
M_vPNml83 0.58, M_vPNml87 0.54, ...): the honest reading is that odour is not *lost* on the way to the DNs, it is
*small* there (a few Hz on tens of Hz of wind-driven activity), which is what NOTES session 6 found for the odour gate
("not at the DN level in this model") and why the program's gate reads the LH channel.

---

## 4. What the tool writes (schema, checks)

Every Result: `flyverse.interp.result/1`, `Result.check()` empty on all six JSONs; provenance from the recording
(realised `execution.device = cuda`, `device_name NVIDIA B200`, LIFParams / OpticParams resolved, cache md5
`ef23cc27...`, effective-weights md5 `ed1df661716d240b0f9289607f95320c`, retina record) plus an `analysis` block (local
git state, the arms' seeds, files and devices). Tables: `per_type` (per-run values of both arms, means, levels, z,
Welch, U, p, `p_floor`, verdict, note; `figure_z_runs` / `carry_runs` under `figure_z`), `lost_candidates`,
`lost_inputs`, `lost_cancellation`, `depth_edges` (the edge that placed each type), `readout_per_body` (Neurome
fields; LC11 143 bodies x {`upstream_drive_mV`, `output_Hz`} and LC10a 275 x 2 -- two rows per body, never pooled --
plus the decomposed and the top carrier types), and the decompose tool's `lost_static_*` / `lost_dynamic_*`.
`replicates` names every run (`{run_index, seed, arm, file, device, window_s, frames}`) and the null's source
(`null_runs`, or `control_pairs` when no null arm is given: every ordered pair of distinct control runs). `summary`
carries the carriers by depth, the counts per depth / stage, both first-lost rules, `lost_top_by_share`, the
cancellation records, the decompose composition status and the p floor.

---

## 5. Contract notes and deviations (for the design owner)

1. **`stat` default is `best_cell`, not the stub's `figure_z`.** The validation target's statistic is the object
   sweep's max-over-cells (8.4 / 8.7), and `figure_z` needs the column map. Under `figure_z` the tool now uses the
   recorded columns for every stat's signed-figure reading of the lost stage (section 2.2). Every stub parameter is kept
   (`StubTests` passes); the extra keyword parameters have defaults.
2. **Three runs per arm cannot give a 'result'** through `common.compare` (exact p floor 0.10 at 3 v 3, section 1).
   Either `MIN_REPLICATES` becomes 4 (floor 0.029) or `compare` should test p against its own floor; until then
   `trace` reports `p_floor` / `p_floor_note`, and the validation ran 5 v 5.
3. **A deterministic null (SD 0) makes `compare`'s z NaN and its verdict 'null' whatever the difference** (the ORNs:
   +83 Hz called 'null'). `trace` overrides to 'result' by the exact test when p <= 0.05 and the difference is non-zero,
   with `note = null_sd_zero` and z left NaN (JSON has no inf). `compare` itself should handle this.
4. **`flyverse.interp.trace` is the submodule, not the function, once `flyverse.interp.trace` has been imported**
   (Python binds the submodule on the package; the package `__getattr__` never runs). Same for every tool whose
   module and function share a name. The stable import is `from flyverse.interp.trace import trace`; the test asserts
   the package attribute is the module. `trace`'s composition with decompose takes the module's `.decompose` when the
   attribute is a module.
5. **Untyped cells** (type `''`, thousands of them across the brain) form one pseudo-type in any type-level graph and
   short-circuit every depth (LHPD4d1 read depth 1 from the ORNs through them; DNp18 and DNb05 too). `TypeGraph` never
   makes them a node; the per-type table never scores them.
6. **`Recording.save` writes an uncompressed npz** (27 MB for 167k cells x 5 quantities, mostly NaN and repeated type
   strings); `trace.save_recording` writes the same schema with `np.savez_compressed` (4-6 MB) and `Recording.load` reads
   both. The pooled series is kept every `--series-every` frames (5 here; the d' smoothing is 1 s).
7. **`load_runs` on a glob** must skip the console `.txt` next to a run (it once loaded every run twice, n = 10).
8. The depth rule (`first_lost_depth`) is only informative for a stimulus with a single carrying pathway; for the
   object (motion pathway carries to depth 4) it returns 5 and for odour (brain-wide) None. The input rule with the
   carrier-share threshold, the per-depth / per-stage counts and `lost_top_by_share` are what to read; the targeted
   `--decompose-at <types>` is the question the validation asks.
9. `VALIDATION['trace']['reference']` lists each 8.7 z pair in ascending order (Mi4 `[22.3, 28.6]` = sign-abs, off;
   Mi1 `[7.8, 27.9]` = off, sign-abs), and `docs/INTERP.md` 4.2 labels every pair `off / sign-abs`. No number is wrong;
   the audit subcommand labels the column "8.7, both modes" and the table in section 2 reads the modes from 8.4 / 8.7.
10. The CPU smoke test of `record` (`--quick --allow-cpu --device cpu`) initially ran on this desktop's GPU because
    `room_demo.Sim` picks CUDA when it is available; `--device cpu` now hides CUDA (`CUDA_VISIBLE_DEVICES=""`) before torch
    is imported. No number in this record comes from that smoke run.

## 6. Files

* `flyverse/interp/trace.py`, `scripts/interp_trace.py`, `tests/test_interp.py::TraceTests` (7 tests; the whole file
  65 tests pass on the CPU in 14 s; `tests/test_control.py` 18 pass).
* Recordings `out/trv/{obj,od}_{stim,ctrl,null}_r{0..4}.{npz,json}` + `_series`, console `.txt` per job,
  `out/trv_cluster.log`.
* Results `out/interp/trace/object_stage.json` (+ `_console.txt`), `object_default.json`, `object_figure_z.json`,
  `object_figure_z_abs.json`, `odour_mean.json`, `odour_dprime.json`; `python scripts/interp_trace.py audit --json <file>`
  prints the side-by-side of sections 2 and 3.
