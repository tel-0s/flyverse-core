#!/usr/bin/env bash
# Receptor round 5, GF-damping adoption: ONE cluster batch (7 concurrent jobs) on the EDITED default
# (flyverse/brain.py DEFAULT_TYPE_PATH_GAIN without the (SAD073|GNG300|DNp70|CL367|PVLP010 -> DNp01, 0.3) entry;
# the previous list is brain.GF_DAMPED_TYPE_PATH_GAIN).  Run from the repo root with Git Bash:
#     bash scripts/r5_adopt_batch.sh 2>&1 | tee out/r5_adopt_cluster.log
#
#    3 x scripts/benchmark.py --seeds 0,1,2, no flags (the no-flag suite on the edited default)
#    3 x scripts/batch_sustain.py --batch 16 --minutes 5 --energy 0.9 (cx / apple / fence; brain seeds 0,1,2 x env seeds
#        0-15 / 16-31 / 32-47), escape route live -- the instrument task's protocol (scripts/r5_cluster_batch.sh jobs 0-2)
#        under the edited default; the pre-retirement default and off arms are that batch's r5_sustain_{default,off}_live_*
#    1 x scripts/benchmark.py --sections hops (the opt-in section) on the edited default
# cluster_run.py ships the working-tree diff, so the edited brain.py and the round-5 instrument files go with the jobs.
set -u
cd "$(dirname "$0")/.."
G="python -c 'import torch; assert torch.cuda.is_available()' &&"
S="python scripts/batch_sustain.py --batch 16 --minutes 5 --energy 0.9 --program cx --fruit apple --fence --cuda-graphs --cuda-kernels --event-driven --cuda-sparse torch"
seeds=("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" "16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" "32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47")
cmds=()
for n in 1 2 3; do
  name="r5_adopt_default_$n"
  cmds+=("$G python scripts/benchmark.py --seeds 0,1,2 --json out/$name.json > out/$name.txt; cat out/$name.txt")
done
for k in 0 1 2; do
  n=$((k+1)); name="r5_adopt_sustain_live_$n"
  cmds+=("$G $S --seed $k --seeds ${seeds[$k]} --json out/$name.json > out/$name.txt; cat out/$name.txt")
done
cmds+=("$G python scripts/benchmark.py --sections hops --json out/r5_adopt_hops.json > out/r5_adopt_hops.txt; cat out/r5_adopt_hops.txt")
printf 'job: %s\n' "${cmds[@]}"
PYTHONIOENCODING=utf-8 python scripts/cluster_run.py --name r5-adopt --minutes 150 "${cmds[@]}" --fetch out/
