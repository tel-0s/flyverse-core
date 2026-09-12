#!/bin/sh
# Round 5, Brain-side vs optic-side attribution: ONE cluster batch, six jobs in one run directory.
#
#   sh scripts/r5_attr_batch.sh
#
# Each job rebuilds out/receptors_hold{Brain,Optic}.csv in its own run copy (out/ is not shipped by
# cluster_run.py; scripts/build_hold_tables.py writes atomically so the concurrent jobs of one run
# directory cannot tear the file) and then runs the Brain-only + walk/motion sections of the suite:
#   holdBrain x2  -- only the 44,463 optic-side entries applied  (the Brain side held at NT_SIGN)
#   holdOptic x2  -- only the 3,832 Brain-side entries applied   (the optic side held at NT_SIGN)
#   default  x1   -- all 48,295 entries, the shipped table, explicit sign/abs
#   off      x1   -- receptor_model=None
# The two anchors are in the SAME batch on purpose: the shipped working tree retires the GF x0.3
# damping (flyverse/brain.py DEFAULT_TYPE_PATH_GAIN, round-5 adoption task), so the round-4 walk
# numbers (default 79.47 / off 73.18) were measured under a different default and cannot anchor the
# walk.* rows of this attribution.
set -e
SEC=rest,taste,smell,walk,bitter,motion
BUILD="python scripts/build_hold_tables.py --groups Brain,Optic"
GPU="python -c 'import torch; assert torch.cuda.is_available()'"
job() {  # $1 = name, $2... = receptor flags
  name=$1; shift
  echo "$GPU && $BUILD && python scripts/benchmark.py --sections $SEC --seeds 0,1,2 $* --json out/r5_attr_$name.json > out/r5_attr_$name.txt; cat out/r5_attr_$name.txt"
}
PYTHONIOENCODING=utf-8 python scripts/cluster_run.py --name r5-attr --minutes 45 \
  "$(job holdBrain_1 --receptor-model sign --receptor-net-rule abs --receptor-table out/receptors_holdBrain.csv)" \
  "$(job holdBrain_2 --receptor-model sign --receptor-net-rule abs --receptor-table out/receptors_holdBrain.csv)" \
  "$(job holdOptic_1 --receptor-model sign --receptor-net-rule abs --receptor-table out/receptors_holdOptic.csv)" \
  "$(job holdOptic_2 --receptor-model sign --receptor-net-rule abs --receptor-table out/receptors_holdOptic.csv)" \
  "$(job default_1 --receptor-model sign --receptor-net-rule abs)" \
  "$(job off_1 --receptor-model off)" \
  --fetch out/
