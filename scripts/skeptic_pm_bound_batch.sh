#!/usr/bin/env bash
# SKEPTIC rerun of the walk.power_max bound study (run "pm-bound", docs/audits/anti_runaway.md round 6) at FRESH
# seeds, ONE cluster submission, results under out/skpm/.
#
# What it tests that the original batch cannot:
#   1. the original used seeds 0,1,2 only. The report asserts "walk.* draws no RNG, so the draws are GPU
#      nondeterminism of one deterministic section". If that holds, baseline / lpi_x1 / lpi_x2 at seeds 3,4,5
#      must return the SAME walk.power_max as at seeds 0,1,2. A different value refutes the determinism claim
#      and makes the 1.52 Hz margin seed-conditional.
#   2. two baseline jobs run --sections walk alone (instead of walk,a,b) to check that walk.power_max does not
#      depend on which other sections share the process.
#   3. six fresh pair_gain_lpi_x2 draws to re-estimate the excursion rate (1 of 6 in the original batch).
set -e
cd "$(dirname "$0")/.."
cmds=()
add() {  # add <tag> <cfg> <seed> <sections>
  cmds+=("mkdir -p out/skpm && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && PYTHONIOENCODING=utf-8 python scripts/retire_measures.py --configs $2 --seeds $3 --sections $4 --timeout 40 --out out/skpm/$1 > out/skpm/$1.txt 2>&1; cat out/skpm/$1.txt; tail -n 25 out/skpm/$1/$2.log")
}
for s in 3 4 5; do for k in 1 2; do
  add "baseline_s${s}_d${k}"         baseline         "$s" walk,a,b
  add "pair_gain_lpi_x2_s${s}_d${k}" pair_gain_lpi_x2 "$s" walk,a,b
done; done
for s in 3 4 5; do add "pair_gain_lpi_x1_s${s}_d1" pair_gain_lpi_x1 "$s" walk,a,b; done
add "baseline_walkonly_s3_d1" baseline 3 walk
add "baseline_walkonly_s4_d1" baseline 4 walk
if [ "$1" = "--print" ]; then printf '%s\n' "${cmds[@]}"; exit 0; fi
mkdir -p out/skpm
python scripts/cluster_run.py --name skpm --minutes 30 "${cmds[@]}" --fetch out/skpm/ 2>&1 | tee out/skpm_cluster.log
