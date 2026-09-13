#!/usr/bin/env bash
# Skeptic rerun of the round-5 take-off instrument (ONE cluster batch, 3 concurrent jobs).
#
#   bash scripts/sk5_hops_verify_batch.sh 2>&1 | tee out/sk5_hops_cluster.log
#
# Every job runs from the PRESERVED run directory of the batch under audit
# ($CLUSTER_RUNS/r5-hops-218d81), i.e. the exact code snapshot that produced
# out/r5_sustain_*.json and out/r5_hops_*.json -- brain.py there is HEAD 611f554's (the GF x0.3
# damping still in place), which the working tree no longer is.  Results are written into this
# run's own out/ so --fetch out/ brings them back.
#
#   1  benchmark.py --sections hops (shipped default)  -> does the 1.25 voluntary / KNOWN GAP verdict reproduce?
#   2  batch_sustain, identical command to r5-hops job 0 (default, live escape route, seed 0, env 0-15)
#   3  batch_sustain, identical command to r5-hops job 6 (off, live escape route, seed 0, env 0-15)
set -u
cd "$(dirname "$0")/.."
SNAP=$CLUSTER_RUNS/r5-hops-218d81
G="python -c 'import torch; assert torch.cuda.is_available()' && O=\$PWD/out && cd $SNAP"
S="python scripts/batch_sustain.py --batch 16 --minutes 5 --energy 0.9 --program cx --fruit apple --fence --cuda-graphs --cuda-kernels --event-driven --cuda-sparse torch --seed 0 --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"

PYTHONIOENCODING=utf-8 python scripts/cluster_run.py --name sk5-hops --minutes 70 \
  "$G && python scripts/benchmark.py --sections hops --json \$O/sk5_hops_default_rerun.json > \$O/sk5_hops_default_rerun.txt; cat \$O/sk5_hops_default_rerun.txt" \
  "$G && $S --json \$O/sk5_sustain_default_live_1r.json > \$O/sk5_sustain_default_live_1r.txt; cat \$O/sk5_sustain_default_live_1r.txt" \
  "$G && $S --receptor-model off --json \$O/sk5_sustain_off_live_1r.json > \$O/sk5_sustain_off_live_1r.txt; cat \$O/sk5_sustain_off_live_1r.txt" \
  --fetch out/
