# suite-inst room rate-half analysis of `out/suite-inst-room`

problems 0

**Verdict: PASS: the instrumented take-off rate is not higher (one-sided exact p 0.291)**

| measure | role | K instrumented | K raw | rate inst / raw per 1,000 fly-s | one-sided exact p (inst higher) | two-sided | run-level compare (6 v 6) | instrumented runs | raw runs |
|---|---|---|---|---|---|---|---|---|---|
| hops | primary | 110 | 101 | 3.819 / 3.507 | 0.291 | 0.5819 | null (diff +1.500, z +0.44, p 0.9372) | [18, 16, 12, 17, 17, 30] | [13, 18, 17, 22, 18, 13] |
| escape | descriptive | 42 | 39 | 1.458 / 1.354 | 0.4122 | 0.8243 | null (diff +0.500, z +0.21, p 0.8182) | [5, 7, 5, 7, 7, 11] | [4, 8, 6, 10, 7, 4] |
| voluntary | descriptive | 68 | 62 | 2.361 / 2.153 | 0.3306 | 0.6612 | null (diff +1.000, z +0.83, p 1) | [13, 9, 7, 10, 10, 19] | [9, 10, 11, 12, 11, 9] |

| arm | run | brain seed | device | GPU | cache | hops | escape | voluntary | meals | path m | walking-GF median Hz | rows GF >= 33 | wall s | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0 | 0 | NVIDIA B200 | 4 | ef23cc27 | 13 | 4 | 9 | 4 | 3.602 | 31.027466773986816 | 4 | 2334.2883292390034 | ok |
| instrumented | 0 | 0 | NVIDIA B200 | 5 | 7a10d93b | 18 | 5 | 13 | 1 | 3.705 | 30.14659869670868 | 3 | 2400.3643049011007 | ok |
| raw | 1 | 1 | NVIDIA B200 | 6 | ef23cc27 | 18 | 8 | 10 | 7 | 3.610 | 32.63596057891846 | 7 | 2279.4105094280094 | ok |
| instrumented | 1 | 1 | NVIDIA B200 | 7 | 7a10d93b | 16 | 7 | 9 | 1 | 3.695 | 30.79933762550354 | 6 | 2390.0458512883633 | ok |
| raw | 2 | 2 | NVIDIA B200 | 4 | ef23cc27 | 17 | 6 | 11 | 1 | 3.703 | 31.040008544921875 | 5 | 2343.436711148359 | ok |
| instrumented | 2 | 2 | NVIDIA B200 | 5 | 7a10d93b | 12 | 5 | 7 | 2 | 3.663 | 31.06994280219078 | 3 | 2389.907826389186 | ok |
| raw | 3 | 3 | NVIDIA B200 | 6 | ef23cc27 | 22 | 10 | 12 | 6 | 3.678 | 32.115694999694824 | 7 | 2250.9035005392507 | ok |
| instrumented | 3 | 3 | NVIDIA B200 | 7 | 7a10d93b | 17 | 7 | 10 | 2 | 3.708 | 31.509700775146484 | 5 | 2392.8621950317174 | ok |
| raw | 4 | 4 | NVIDIA B200 | 4 | ef23cc27 | 18 | 7 | 11 | 7 | 3.617 | 31.363893270492554 | 7 | 2336.303450014442 | ok |
| instrumented | 4 | 4 | NVIDIA B200 | 5 | 7a10d93b | 17 | 7 | 10 | 4 | 3.635 | 30.804136753082275 | 6 | 2348.608551151119 | ok |
| raw | 5 | 5 | NVIDIA B200 | 6 | ef23cc27 | 13 | 4 | 9 | 0 | 3.693 | 31.47540843486786 | 4 | 2279.9273705026135 | ok |
| instrumented | 5 | 5 | NVIDIA B200 | 7 | 7a10d93b | 30 | 11 | 19 | 4 | 3.696 | 33.367876291275024 | 9 | 2353.7738149017096 | ok |
