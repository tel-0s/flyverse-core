#!/usr/bin/env bash
# The compass under sensory input (docs/audits/compass_room.md): one cluster batch, 12 concurrent jobs.
#   2 GLNO-sign-robust operating points (gE 2 / gD 15, gE 2.5 / gD 25) x 2 programs (none, cx) x brain seeds 0, 1 = 8 jobs
#   + the shipped default (no compass gains) x 2 programs x seeds 0, 1 = 4 control jobs.
# Each job: BatchSim, 16 flies (environment seeds 16 s .. 16 s + 15), fenced single-apple table, 60 s, EPG pulse at 20 s
# (4 wedges per fly, +40 Hz, 2 s), Torch path + CUDA graphs. Console log -> out/cxroom_cluster.log.
set -e
cd "$(dirname "$0")/.."
mkdir -p out/cxroom
cmds=()
for prog in none cx; do
  for seed in 0 1; do
    for g in "2:15" "2.5:25" "ctrl"; do
      if [ "$g" = ctrl ]; then
        tag="ctrl_${prog}_s${seed}"; gains=""
      else
        gE=${g%%:*}; gD=${g##*:}
        tag="g${gE}-${gD}_${prog}_s${seed}"; gains="--gE $gE --gD $gD"
      fi
      cmds+=("python -c 'import torch; assert torch.cuda.is_available()' && mkdir -p out/cxroom && python scripts/probe_compass_room.py $gains --program $prog --seed $seed --seconds 60 --out out/cxroom/cxroom_${tag}.json > out/cxroom/cxroom_${tag}.txt; cat out/cxroom/cxroom_${tag}.txt")
    done
  done
done
printf '%s\n' "${cmds[@]}"
PYTHONIOENCODING=utf-8 python scripts/cluster_run.py --name cxroom --minutes 45 "${cmds[@]}" --fetch out/ 2>&1 | tee out/cxroom_cluster.log
