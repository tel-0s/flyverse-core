### Suite values by draw

Source: `out/compass_standin/analysis/suite_rows.csv`, columns `raw_value` and `instrumented_value`.
Each list is ordered by draw seed [0,1,2], unsorted by outcome. P=PASS, F=FAIL, G=KNOWN GAP.

| check | raw [0,1,2] | instrumented [0,1,2] | raw / instrumented status |
|---|---|---|---|
| rest.spikes_per_step | [0, 0, 0] | [3, 0, 0] | PPP / PPP |
| taste.MN9_hz | [10.9342, 1.69046, 3.12959] | [2.47865, 4.29596, 9.27711] | PFP / PPP |
| smell.PN_hz | [7.86116, 11.1984, 11.6049] | [9.35426, 6.60227, 2.98635] | PPP / PPP |
| smell.KC_active | [816, 1345, 1518] | [1190, 1202, 532] | PPP / PPP |
| dn.DNa02_L_leg_asym_hz | [2.58063, 2.40017, 2.38502] | [2.58063, 2.40017, 2.2464] | PPP / PPP |
| dn.MDN_top_hz | [153, 172, 182] | [149, 168, 188] | PPP / PPP |
| dn.DNp09_top_hz | [152, 157, 122] | [172, 157, 122] | PPP / PPP |
| walk.GF_max_hz | [4.62916, 4.96265, 4.60406] | [9.85212, 5.35751, 9.06606] | PPP / PPP |
| walk.power_max_hz | [48.4805, 53.5587, 56.1464] | [63.2179, 69.1525, 54.2596] | PPP / PPP |
| walk.power_sustained_hz | [20.1091, 27.893, 25.5384] | [29.9417, 36.543, 26.2494] | PPP / PPP |
| loom.GF_peak_hz | [47.1993, 51.4268, 52.8799] | [44.4985, 45.4109, 50.9155] | PPP / PPP |
| loom.escape_cm | [3.5, 3.5, 3.5] | [3.5, 3.5, 3.5] | PPP / PPP |
| rotate.DNp20_flip_hz | [-38.1083, -29.9594, -29.4401] | [-38.6436, -34.1511, -36.9592] | PPP / PPP |
| motion.min_dsi | [0.241097, 0.241097, 0.241097] | [0.251814, 0.240121, 0.244777] | PPP / PPP |
| motion.correct_directions | [8, 8, 8] | [8, 8, 8] | PPP / PPP |
| loom_escape.GF_peak_hz | [43.0381, 43.4018, 44.4386] | [44.3698, 45.9057, 45.8361] | PPP / PPP |
| loom_escape.escapes | [1, 1, 1] | [1, 1, 1] | PPP / PPP |
| walk_gf.p99_hz | [19.0357, 21.3438, 25.4141] | [24.4943, 23.7999, 19.1436] | PPP / PPP |
| rotation.group_flip_hz | [-9.27843, -9.16448, -9.50137] | [-9.26104, -9.5864, -9.59362] | PPP / PPP |
| object.LC10a_flip_hz | [-0.0017969, 0.000439985, 0.00587222] | [0.0130087, -0.00131853, -0.00937451] | GGG / GGG |
| bitter.calibrated_sugar_MN9_hz | [5.51844, 4.28767, 3.92663] | [6.08675, 4.83743, 4.48912] | PPP / PPP |
| bitter.calibrated_sugar_bitter_MN9_hz | [0, 0, 0] | [0, 0, 0] | PPP / PPP |
| bitter.shiu_sugar_MN9_hz | [139.898, 138.934, 131.523] | [135.257, 139.184, 127.327] | PPP / PPP |
| bitter.shiu_sugar_bitter_MN9_hz | [0.81784, 0, 0] | [1.25739, 0.0232895, 0] | PPP / PPP |
| wind.DNp18_flip_hz | [45.5796, 45.4505, 46.3454] | [45.5657, 45.2692, 45.7698] | PPP / PPP |
| wind.DNp33_flip_hz | [-50.6448, -50.8366, -50.6155] | [-49.9223, -50.2185, -50.2606] | PPP / PPP |
| odour.apple_channel_8cm_hz | [17.4499, 17.4347, 17.5586] | [17.4893, 17.394, 17.4277] | PPP / PPP |
| odour.apple_channel_clean_hz | [4.50877, 4.35076, 4.32929] | [4.19104, 4.43738, 4.53897] | PPP / PPP |
| compass.wedge_cells_persisting | [0, 0, 0] | [0, 0, 1] | GGG / GGG |

### Room values by row

Sources: initial `room_rows.csv`; r5 `summary.json:rooms`. Lists follow environment seeds [10,11,12,13,14,15],
all rows retained. Distance is XY path length between 0.1 s samples, not displacement toward fruit.

```text
initial raw:
  hops: [1, 0, 0, 0, 0, 0]
  distance_m: [0.679918, 0.521048, 0.535011, 0.530506, 0.522328, 0.527158]
  mean_abs_yaw_deg_s: [1.77279, 1.75847, 1.42505, 1.14191, 1.3436, 1.3173]
  mean_epg_hz: [0.0224894, 0.00703989, 0.0174705, 0.0175467, 0.0419367, 0.0271949]
initial instrumented:
  hops: [0, 0, 0, 0, 0, 0]
  distance_m: [0.530675, 0.534381, 0.522808, 0.536889, 0.532033, 0.532982]
  mean_abs_yaw_deg_s: [1.0914, 1.07517, 1.51923, 1.11505, 1.43504, 1.2524]
  mean_epg_hz: [11.4674, 11.6063, 11.5112, 11.51, 11.426, 11.6646]
room_eager_a:
  hops: [1, 0, 0, 0, 0, 0]
  distance_m: [0.554056, 0.457027, 0.529278, 0.524363, 0.531783, 0.537351]
  mean_abs_yaw_deg_s: [1.46975, 1.65636, 1.26687, 1.34526, 1.39128, 1.26183]
  mean_epg_hz: [11.4107, 11.5472, 11.3795, 11.5687, 11.4592, 11.58]
room_eager_b:
  hops: [0, 1, 0, 1, 0, 0]
  distance_m: [0.530097, 0.585012, 0.527049, 0.551996, 0.524185, 0.522268]
  mean_abs_yaw_deg_s: [1.37433, 1.82545, 1.16626, 1.20564, 1.15313, 1.34273]
  mean_epg_hz: [11.626, 11.5336, 11.4912, 11.4898, 11.5622, 11.6616]
room_instrumented:
  hops: [0, 0, 0, 0, 0, 0]
  distance_m: [0.527477, 0.53128, 0.532664, 0.530442, 0.444899, 0.53115]
  mean_abs_yaw_deg_s: [1.52903, 1.04244, 1.26682, 1.12158, 1.52392, 1.34602]
  mean_epg_hz: [11.5736, 11.6406, 11.4095, 11.4998, 11.4558, 11.6402]
room_native_raw:
  hops: [0, 0, 0, 0, 0, 0]
  distance_m: [0.528244, 0.524144, 0.528126, 0.532849, 0.529294, 0.533154]
  mean_abs_yaw_deg_s: [1.12232, 1.06213, 1.50035, 1.69534, 1.17801, 1.45325]
  mean_epg_hz: [0.0445646, 0.00864669, 0.064731, 0.0153902, 0.0332054, 0.030766]
room_native_instrumented:
  hops: [0, 0, 0, 0, 0, 0]
  distance_m: [0.531006, 0.529294, 0.529418, 0.464064, 0.523615, 0.454244]
  mean_abs_yaw_deg_s: [1.46202, 1.23686, 1.62061, 1.18091, 1.06345, 1.42151]
  mean_epg_hz: [11.5128, 11.6629, 11.4878, 11.4529, 11.4904, 11.5369]
```

### Performance repeats

Sources: r5 and r6 `profile_native.json:records`, synchronized wall ms per 10 ms frame.
Four alternating repeats in recorded order; repetitions on one device are not independent animals.

```text
compass_standin_r5 B=1: raw [0.49264909932389855, 0.49255914986133575, 0.4922550800256431, 0.4924878804013133]; instrumented [0.5465178401209414, 0.5465592816472054, 0.5464200605638325, 0.5465938802808523]
compass_standin_r5 B=8: raw [3.0008039600215852, 3.0007882486097515, 3.000880549661815, 3.000832989346236]; instrumented [3.095938719343394, 3.096106559969485, 3.096720341127366, 3.095995250623673]
compass_standin_r5 B=32: raw [12.692815940827131, 12.692558739800006, 12.693733689375222, 12.693404559977353]; instrumented [12.844424531795084, 12.843739108648151, 12.844947811681777, 12.84449314000085]
compass_standin_r6 B=1: raw [0.4924329509958625, 0.4921712982468307, 0.4922709590755403, 0.4924075095914304]; instrumented [0.546730412170291, 0.5465619801543653, 0.5464203492738307, 0.5465501686558127]
compass_standin_r6 B=8: raw [3.001607689075172, 3.0009281309321523, 3.0004799203015864, 3.0008422606624663]; instrumented [3.096698799636215, 3.096724129281938, 3.0971143790520728, 3.096653709653765]
compass_standin_r6 B=32: raw [12.69296603044495, 12.69172047963366, 12.69327879184857, 12.692816890776157]; instrumented [12.84789980854839, 12.84713874105364, 12.847986440174282, 12.847362190950662]
```
