#!/usr/bin/env bash
# Receptor round 5: ONE cluster batch (14 concurrent jobs) -- run from the repo root with Git Bash:
#     bash scripts/r5_cluster_batch.sh 2>&1 | tee out/r5_cluster.log
#
#   12 x scripts/batch_sustain.py --batch 16 --minutes 5 --energy 0.9 (cx / apple / fence; brain seeds 0,1,2 x env seeds
#        0-15 / 16-31 / 32-47) for the shipped default and --receptor-model off, with the GF escape route live and with
#        --gf-hz 1e9 (voluntary route only)
#    2 x scripts/benchmark.py --sections hops (the new opt-in section) for default and off
set -u
cd "$(dirname "$0")/.."
G="python -c 'import torch; assert torch.cuda.is_available()' &&"
S="python scripts/batch_sustain.py --batch 16 --minutes 5 --energy 0.9 --program cx --fruit apple --fence --cuda-graphs --cuda-kernels --event-driven --cuda-sparse torch"
seeds=("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" "16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" "32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47")
cmds=()
for cond in default off; do
  flag=""; [ "$cond" = off ] && flag="--receptor-model off"
  for route in live nogf; do
    rflag=""; [ "$route" = nogf ] && rflag="--gf-hz 1e9"
    for k in 0 1 2; do
      n=$((k+1)); name="r5_sustain_${cond}_${route}_${n}"
      cmds+=("$G $S --seed $k --seeds ${seeds[$k]} $flag $rflag --json out/$name.json > out/$name.txt; cat out/$name.txt")
    done
  done
done
cmds+=("$G python scripts/benchmark.py --sections hops --json out/r5_hops_default.json > out/r5_hops_default.txt; cat out/r5_hops_default.txt")
cmds+=("$G python scripts/benchmark.py --sections hops --receptor-model off --json out/r5_hops_off.json > out/r5_hops_off.txt; cat out/r5_hops_off.txt")
printf 'job: %s\n' "${cmds[@]}"
PYTHONIOENCODING=utf-8 python scripts/cluster_run.py --name r5-hops --minutes 150 "${cmds[@]}" --fetch out/
