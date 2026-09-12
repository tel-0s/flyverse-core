#!/bin/sh
# Skeptic recheck of the round-5 Brain-side vs optic-side attribution (task key: attribute).
#
#   sh scripts/skeptic_attr_recheck.sh
#
# A THIRD independent run directory for the same four arms the attribution rests on, with the same
# sections/seeds as scripts/r5_attr_batch.sh, so the reported per-arm values can be compared against
# a batch the reporting agent did not produce.  Each job rebuilds the hold tables in its own run copy
# (and prints their md5, which must equal the local d902daf5... / c3baf429...).  Outputs go to out/sk/
# so nothing in out/ written by the reported run is overwritten by the fetch.
set -e
SEC=rest,taste,smell,walk,bitter,motion
BUILD="python scripts/build_hold_tables.py --groups Brain,Optic"
GPU="python -c 'import torch; assert torch.cuda.is_available()'"
job() {
  name=$1; shift
  echo "$GPU && mkdir -p out/sk && $BUILD && python scripts/benchmark.py --sections $SEC --seeds 0,1,2 $* --json out/sk/sk_attr_$name.json > out/sk/sk_attr_$name.txt; cat out/sk/sk_attr_$name.txt"
}
PYTHONIOENCODING=utf-8 python scripts/cluster_run.py --name sk-attr --minutes 30 \
  "$(job holdBrain --receptor-model sign --receptor-net-rule abs --receptor-table out/receptors_holdBrain.csv)" \
  "$(job holdOptic --receptor-model sign --receptor-net-rule abs --receptor-table out/receptors_holdOptic.csv)" \
  "$(job default --receptor-model sign --receptor-net-rule abs)" \
  "$(job off --receptor-model off)" \
  --fetch out/sk
