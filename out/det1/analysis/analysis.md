# det1 analysis of `out/det1`

problems 0

**Decision (frozen rule): neither path repeats: items 3-4 are >= 6 draws, run = replicate unit, nothing quoted beyond its across-draw SD**

| pair | protocol | path | verdict | arrays equal / n | max abs diff (arrays) | metrics equal / n | max abs diff (metrics) | first diff frame (spikes) | device |
|---|---|---|---|---|---|---|---|---|---|
| cxS | cx_wedge S arm seed 0 | torch (cx_wedge default) | **repeats exactly** | 31 / 31 | 0 | 546 / 546 | 0 | None | NVIDIA B200 |
| plume_native | plume room 10 s B=6 seed 0 | native | **one draw** | 1 / 15 | 37083 | 992 / 1026 | 32271 | 30 | NVIDIA B200 |
| plume_torch | plume room 10 s B=6 seed 0 | torch | **one draw** | 1 / 15 | 41225 | 992 / 1026 | 26408 | 60 | NVIDIA B200 |
| raw_native | raw room 10 s B=6 seed 0 | native | **one draw** | 1 / 14 | 49326 | 142 / 178 | 26811 | 4 | NVIDIA B200 |
| raw_torch | raw room 10 s B=6 seed 0 | torch | **one draw** | 1 / 14 | 37865 | 142 / 178 | 29177 | 86 | NVIDIA B200 |

## Per array (`arrays.csv`)

| pair | array | shape | equal | max abs diff | first differing index |
|---|---|---|---|---|---|
| cxS | cells__Delta7 | (800, 42) | True | 0 | None |
| cxS | cells__EPGt | (800, 4) | True | 0 | None |
| cxS | cells__ER4m | (800, 11) | True | 0 | None |
| cxS | cells__ER6 | (800, 4) | True | 0 | None |
| cxS | cells__ExR6 | (800, 2) | True | 0 | None |
| cxS | cells__GLNO | (800, 4) | True | 0 | None |
| cxS | cells__PEG | (800, 18) | True | 0 | None |
| cxS | cells__PEN | (800, 42) | True | 0 | None |
| cxS | epg | (800, 46) | True | 0 | None |
| cxS | g__DNa02_L | (800,) | True | 0 | None |
| cxS | g__DNa02_R | (800,) | True | 0 | None |
| cxS | g__Delta7 | (800,) | True | 0 | None |
| cxS | g__EPGt | (800,) | True | 0 | None |
| cxS | g__ER4m | (800,) | True | 0 | None |
| cxS | g__ER6 | (800,) | True | 0 | None |
| cxS | g__ExR6 | (800,) | True | 0 | None |
| cxS | g__GLNO | (800,) | True | 0 | None |
| cxS | g__GLNO_L | (800,) | True | 0 | None |
| cxS | g__GLNO_R | (800,) | True | 0 | None |
| cxS | g__PEG | (800,) | True | 0 | None |
| cxS | g__PEN | (800,) | True | 0 | None |
| cxS | g__PEN_L | (800,) | True | 0 | None |
| cxS | g__PEN_R | (800,) | True | 0 | None |
| cxS | g__PS196b_L | (800,) | True | 0 | None |
| cxS | g__PS196b_R | (800,) | True | 0 | None |
| cxS | g__Ring | (800,) | True | 0 | None |
| cxS | g__rest | (800,) | True | 0 | None |
| cxS | inside | (46,) | True | 0 | None |
| cxS | t | (800,) | True | 0 | None |
| cxS | wedge_of | (46,) | True | 0 | None |
| cxS | yaw_deg_s | (800,) | True | 0 | None |
| plume_native | body | (1000, 6, 14) | False | 91.5984 | [31, 0, 13] |
| plume_native | final__adapt | (6, 167106) | False | 24.4138 | [0, 0] |
| plume_native | final__drive | (6, 167106) | False | 70 | [0, 2] |
| plume_native | final__g | (6, 167106) | False | 676.014 | [0, 0] |
| plume_native | final__g_slow | (6, 167106) | True | 0 | None |
| plume_native | final__poisson_p | (6, 167106) | False | 0.00356828 | [0, 501] |
| plume_native | final__rate | (6, 167106) | False | 146.194 | [0, 0] |
| plume_native | final__refrac | (6, 167106) | False | 2.2 | [0, 3] |
| plume_native | final__res | (6, 167106) | False | 0.420141 | [0, 82] |
| plume_native | final__spike_buf | (4, 6, 167106) | False | 1 | [0, 0, 3] |
| plume_native | final__spike_counts | (6, 167106) | False | 78 | [0, 0] |
| plume_native | final__spikes | (6, 167106) | False | 1 | [0, 82] |
| plume_native | final__v | (6, 167106) | False | 66.1006 | [0, 0] |
| plume_native | plume | (1000, 6, 7) | False | 17.8762 | [41, 0, 0] |
| plume_native | spikes_cum | (1000, 6) | False | 37083 | [30, 0] |
| plume_torch | body | (1000, 6, 14) | False | 93.9804 | [92, 4, 0] |
| plume_torch | final__adapt | (6, 167106) | False | 25.2432 | [0, 0] |
| plume_torch | final__drive | (6, 167106) | False | 70 | [0, 1] |
| plume_torch | final__g | (6, 167106) | False | 579.385 | [0, 0] |
| plume_torch | final__g_slow | (6, 167106) | True | 0 | None |
| plume_torch | final__poisson_p | (6, 167106) | False | 0.0031972 | [0, 501] |
| plume_torch | final__rate | (6, 167106) | False | 130.963 | [0, 0] |
| plume_torch | final__refrac | (6, 167106) | False | 2.2 | [0, 106] |
| plume_torch | final__res | (6, 167106) | False | 0.380767 | [0, 82] |
| plume_torch | final__spike_buf | (4, 6, 167106) | False | 1 | [0, 0, 710] |
| plume_torch | final__spike_counts | (6, 167106) | False | 84 | [0, 0] |
| plume_torch | final__spikes | (6, 167106) | False | 1 | [0, 218] |
| plume_torch | final__v | (6, 167106) | False | 110.672 | [0, 0] |
| plume_torch | plume | (1000, 6, 7) | False | 19.7951 | [97, 4, 0] |
| plume_torch | spikes_cum | (1000, 6) | False | 41225 | [60, 4] |
| raw_native | body | (1000, 6, 14) | False | 86.9035 | [5, 5, 0] |
| raw_native | final__adapt | (6, 167106) | False | 24.8557 | [0, 0] |
| raw_native | final__drive | (6, 167106) | False | 70 | [0, 1] |
| raw_native | final__g | (6, 167106) | False | 746.87 | [0, 0] |
| raw_native | final__g_slow | (6, 167106) | True | 0 | None |
| raw_native | final__poisson_p | (6, 167106) | False | 0.00269627 | [0, 7438] |
| raw_native | final__rate | (6, 167106) | False | 141.543 | [0, 0] |
| raw_native | final__refrac | (6, 167106) | False | 2.2 | [0, 0] |
| raw_native | final__res | (6, 167106) | False | 0.501304 | [0, 82] |
| raw_native | final__spike_buf | (4, 6, 167106) | False | 1 | [0, 0, 0] |
| raw_native | final__spike_counts | (6, 167106) | False | 59 | [0, 0] |
| raw_native | final__spikes | (6, 167106) | False | 1 | [0, 48] |
| raw_native | final__v | (6, 167106) | False | 56.1461 | [0, 0] |
| raw_native | spikes_cum | (1000, 6) | False | 49326 | [4, 5] |
| raw_torch | body | (1000, 6, 14) | False | 78.6018 | [137, 0, 0] |
| raw_torch | final__adapt | (6, 167106) | False | 23.6197 | [0, 0] |
| raw_torch | final__drive | (6, 167106) | False | 70 | [0, 1] |
| raw_torch | final__g | (6, 167106) | False | 450.206 | [0, 0] |
| raw_torch | final__g_slow | (6, 167106) | True | 0 | None |
| raw_torch | final__poisson_p | (6, 167106) | False | 0.000787832 | [0, 7438] |
| raw_torch | final__rate | (6, 167106) | False | 127.27 | [0, 0] |
| raw_torch | final__refrac | (6, 167106) | False | 2.2 | [0, 257] |
| raw_torch | final__res | (6, 167106) | False | 0.396919 | [0, 82] |
| raw_torch | final__spike_buf | (4, 6, 167106) | False | 1 | [0, 0, 639] |
| raw_torch | final__spike_counts | (6, 167106) | False | 86 | [0, 0] |
| raw_torch | final__spikes | (6, 167106) | False | 1 | [0, 473] |
| raw_torch | final__v | (6, 167106) | False | 100.105 | [0, 0] |
| raw_torch | spikes_cum | (1000, 6) | False | 37865 | [86, 0] |
