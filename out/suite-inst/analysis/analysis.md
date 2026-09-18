# suite-inst analysis of `out/suite-inst`

problems 0

**Verdict: admissible by the suite half (no status change outside the declared gap rows); the room rate-half is owed**

| run | device | cache md5 | pass / fail / gap / missing | controllers | checks |
|---|---|---|---|---|---|
| raw s0 | NVIDIA B200 | ef23cc27 | 27 / 0 / 2 / 0 | 13 | ok |
| raw s1 | NVIDIA B200 | ef23cc27 | 26 / 1 / 2 / 0 | 13 | ok |
| raw s2 | NVIDIA B200 | ef23cc27 | 27 / 0 / 2 / 0 | 13 | ok |
| instrumented s0 | NVIDIA B200 | 7a10d93b | 27 / 0 / 2 / 0 | 19 | ok |
| instrumented s1 | NVIDIA B200 | 7a10d93b | 26 / 1 / 2 / 0 | 19 | ok |
| instrumented s2 | NVIDIA B200 | 7a10d93b | 27 / 0 / 2 / 0 | 19 | ok |

P PASS, P* PASS (gap closed), G KNOWN GAP, F FAIL, M MISSING; values by draw seed [0,1,2] (`suite_rows.csv`, columns value,status)

| check (criterion) | raw [0,1,2] | instrumented [0,1,2] | raw / instrumented statuses | status change | rejecting |
|---|---|---|---|---|---|
| rest.spikes_per_step (< 5) | 0 P, 0 P, 0 P | 0 P, 0 P, 0 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| taste.MN9_hz (> 2) | 10.9342 P, 1.69046 F, 3.12959 P | 10.9342 P, 1.69046 F, 3.12959 P | PASS/FAIL/PASS / PASS/FAIL/PASS | no (raw unstable) |  |
| smell.PN_hz (< 100) | 7.86116 P, 11.1984 P, 11.6049 P | 7.86116 P, 11.1984 P, 11.6049 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| smell.KC_active (> 0) | 816 P, 1345 P, 1518 P | 816 P, 1345 P, 1518 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| dn.DNa02_L_leg_asym_hz (> 0.3) | 2.58063 P, 2.40017 P, 2.38502 P | 2.58063 P, 2.40017 P, 2.38502 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| dn.MDN_top_hz (< 250) | 153 P, 172 P, 182 P | 153 P, 172 P, 182 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| dn.DNp09_top_hz (< 250) | 152 P, 157 P, 122 P | 152 P, 157 P, 122 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| walk.GF_max_hz (< 38) | 4.62916 P, 4.96265 P, 4.60406 P | 4.62916 P, 4.96265 P, 4.60406 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| walk.power_max_hz (notnone 0) | 48.4805 P, 53.5587 P, 56.1464 P | 48.4805 P, 53.5587 P, 55.9379 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| walk.power_sustained_hz (< 50) | 20.1091 P, 27.893 P, 25.5384 P | 20.1091 P, 27.893 P, 25.3919 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| loom.GF_peak_hz (>= 20) | 47.2279 P, 51.3004 P, 52.8799 P | 47.2162 P, 51.4427 P, 45.4985 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| loom.escape_cm (notnone 0) | 3.5 P, 3.5 P, 3.5 P | 3.5 P, 3.5 P, 3.5 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| rotate.DNp20_flip_hz (< -2) | -39.5479 P, -36.2069 P, -25.1727 P | -39.3318 P, -36.1501 P, -38.9102 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| motion.min_dsi (>= 0.1) | 0.241096 P, 0.24595 P, 0.241096 P | 0.240637 P, 0.248435 P, 0.240637 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| motion.correct_directions (== 8) | 8 P, 8 P, 8 P | 8 P, 8 P, 8 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| loom_escape.GF_peak_hz (>= 33) | 48.0924 P, 55.2507 P, 48.8023 P | 49.8767 P, 42.0818 P, 51.3857 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| loom_escape.escapes (>= 1) | 1 P, 1 P, 1 P | 1 P, 1 P, 1 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| walk_gf.p99_hz (< 38) | 20.099 P, 22.5368 P, 13.546 P | 20.2437 P, 16.3014 P, 25.8704 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| rotation.group_flip_hz (<= -3) | -9.48312 P, -10.4283 P, -10.9066 P | -9.25218 P, -9.80203 P, -9.68861 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| object.LC10a_flip_hz (abs>= 1.0) | 0.000725196 G, -0.00979827 G, 0.0091777 G | -0.00188607 G, 0.0209366 G, 0.00974457 G | KNOWN GAP/KNOWN GAP/KNOWN GAP / KNOWN GAP/KNOWN GAP/KNOWN GAP | no |  |
| bitter.calibrated_sugar_MN9_hz (> 2) | 5.51844 P, 4.28767 P, 3.92663 P | 5.51844 P, 4.28767 P, 3.92663 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| bitter.calibrated_sugar_bitter_MN9_hz (< 1) | 0 P, 0 P, 0 P | 0 P, 0 P, 0 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| bitter.shiu_sugar_MN9_hz (> 50) | 139.898 P, 138.934 P, 131.523 P | 140.26 P, 149.314 P, 132.249 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| bitter.shiu_sugar_bitter_MN9_hz (< 10) | 0.81784 P, 0 P, 0 P | 1.37552 P, 0 P, 0 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| wind.DNp18_flip_hz (>= 15) | 44.7584 P, 46.9604 P, 46.862 P | 45.839 P, 45.8612 P, 46.8457 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| wind.DNp33_flip_hz (<= -15) | -50.2723 P, -50.6726 P, -50.3201 P | -49.1691 P, -49.2368 P, -50.0192 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| odour.apple_channel_8cm_hz (>= 10) | 17.4191 P, 17.3895 P, 17.4635 P | 17.4431 P, 17.443 P, 17.4208 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| odour.apple_channel_clean_hz (<= 6) | 4.40823 P, 4.39386 P, 4.68342 P | 4.39271 P, 4.25439 P, 4.48907 P | PASS/PASS/PASS / PASS/PASS/PASS | no |  |
| compass.wedge_cells_persisting (>= 6) | 0 G, 0 G, 0 G | 0 G, 0 G, 0 G | KNOWN GAP/KNOWN GAP/KNOWN GAP / KNOWN GAP/KNOWN GAP/KNOWN GAP | no (declared gap row) |  |
