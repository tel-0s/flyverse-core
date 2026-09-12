#!/usr/bin/env bash
# Compass-shift batch (docs/audits/cx_shift.md): one cluster call, ten jobs, all concurrent.
# Usage: PYTHONIOENCODING=utf-8 bash scripts/cx_shift_batch.sh   (from the repo root; console log -> out/cx_shift_cluster.log)
set -u
G="python -c 'import torch; assert torch.cuda.is_available()'"
S="python scripts/cx_shift.py --shift --seeds 0,1,2"
R="python scripts/cx_shift.py --rotation --seeds 0,1,2 --seconds 10 --rate 90"
PYTHONIOENCODING=utf-8 python scripts/cluster_run.py --name cx-shift --minutes 30 \
  "$G && $S --gains 2:15   --sides L,R,none --pen-hz 20 --no-scratch --out out/cx_shift_base_2_15.json > out/cx_shift_base_2_15.txt; cat out/cx_shift_base_2_15.txt" \
  "$G && $S --gains 2.5:25 --sides L,R,none --pen-hz 20 --no-scratch --out out/cx_shift_base_2.5_25.json > out/cx_shift_base_2.5_25.txt; cat out/cx_shift_base_2.5_25.txt" \
  "$G && $S --gains 2:15   --sides L,R,none --pen-hz 20 --nt-override GLNO=gaba --out out/cx_shift_gaba_2_15.json > out/cx_shift_gaba_2_15.txt; cat out/cx_shift_gaba_2_15.txt" \
  "$G && $S --gains 2.5:25 --sides L,R,none --pen-hz 20 --nt-override GLNO=gaba --out out/cx_shift_gaba_2.5_25.json > out/cx_shift_gaba_2.5_25.txt; cat out/cx_shift_gaba_2.5_25.txt" \
  "$G && $S --gains 2:15   --sides L,R --pen-hz 10,40 --no-scratch --out out/cx_shift_base_2_15_hz.json > out/cx_shift_base_2_15_hz.txt; cat out/cx_shift_base_2_15_hz.txt" \
  "$G && $S --gains 2:15   --sides L,R --pen-hz 10,40 --nt-override GLNO=gaba --out out/cx_shift_gaba_2_15_hz.json > out/cx_shift_gaba_2_15_hz.txt; cat out/cx_shift_gaba_2_15_hz.txt" \
  "$G && $R --condition default --out out/cx_shift_rot_default.json > out/cx_shift_rot_default.txt; cat out/cx_shift_rot_default.txt" \
  "$G && $R --condition gaba    --out out/cx_shift_rot_gaba.json > out/cx_shift_rot_gaba.txt; cat out/cx_shift_rot_gaba.txt" \
  "$G && $R --condition default --gains 2:15 --out out/cx_shift_rot_default_g2_15.json > out/cx_shift_rot_default_g2_15.txt; cat out/cx_shift_rot_default_g2_15.txt" \
  "$G && $R --condition gaba    --gains 2:15 --out out/cx_shift_rot_gaba_g2_15.json > out/cx_shift_rot_gaba_g2_15.txt; cat out/cx_shift_rot_gaba_g2_15.txt" \
  --fetch out/
