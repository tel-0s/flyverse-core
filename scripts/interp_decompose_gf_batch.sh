#!/usr/bin/env bash
# decompose validation (a): the walk.GF_max cancellation -- DNp01's presynaptic rates over benchmark.py's pinned walk
# section under the four receptor arms (off / default / holdBrain = optic side alone / holdOptic = Brain side alone),
# three independent runs (brain seeds 0, 1, 2) per arm, ONE cluster batch of 12 jobs (docs/INTERP.md section 5).
# Seed 0 is the suite's own seed: its GF maxima must reproduce receptor_integration.md G.4 (4.964 / 4.629 / 12.517 /
# 13.311) to the digit; seeds 1-2 give the scatter. Hold tables are rebuilt inside each job by
# flyverse.interp.decompose.arm_params (out/ is not shipped).
#
#     bash scripts/interp_decompose_gf_batch.sh 2>&1 | tee out/dec_gf_cluster.log
# then, on the CPU:
#     PYTHONIOENCODING=utf-8 python scripts/interp_decompose.py validate --case gf --dir out/dec --json out/interp/decompose/validate_gf.json
set -u
cd "$(dirname "$0")/.."
G="python -c 'import torch; assert torch.cuda.is_available()' && mkdir -p out/dec &&"
cmds=()
for arm in off default holdBrain holdOptic; do
  for s in 0 1 2; do
    cmds+=("$G python scripts/interp_decompose.py record --target DNp01 --protocol walk --arm $arm --seed $s --out out/dec/${arm}_r$s > out/dec/${arm}_r$s.txt 2>&1; tail -3 out/dec/${arm}_r$s.txt")
  done
done
printf 'job: %s\n' "${cmds[@]}"
PYTHONIOENCODING=utf-8 python scripts/cluster_run.py --name dec-gf --minutes 25 "${cmds[@]}" --fetch out/dec/
