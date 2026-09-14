#!/usr/bin/env bash
# The walk.power_max bound study (docs/audits/anti_runaway.md, round 6 / handover item 4): six configurations x seeds
# 0,1,2 x two independent draws, ONE cluster submission, every job its own directory under out/pm_bound/.
#
#   bash scripts/pm_bound_batch.sh            # submit + wait + fetch; console -> out/pm_bound_cluster.log
#   bash scripts/pm_bound_batch.sh --print    # only print the job commands
#
# Each job: scripts/retire_measures.py --configs <cfg> --seeds <s> --sections walk,a,b (the legacy walk section with
# its loom / rotate sub-sections, a = motion, b = loom_escape) under the shipped receptor default, native backend.
# The walk.* values draw no RNG, so the six jobs per configuration are six GPU draws of the same deterministic
# section (the round-4 corrections saw a 0.63 Hz excursion on the pre-retirement default); loom / rotate / motion /
# loom_escape do scatter (6-10 Hz within one configuration, optic_measures.md section 4) and the seeds are for them.
# Afterwards: python scripts/retire_measures.py --report-replicates out/pm_bound  and  scripts/pm_bound_report.py.
set -e
cd "$(dirname "$0")/.."
CONFIGS="baseline no_dn_vnc_gain no_path_gain no_drive_clip pair_gain_lpi_x1 pair_gain_lpi_x2"
SEEDS="0 1 2"
DRAWS="1 2"
cmds=()
for cfg in $CONFIGS; do
  for s in $SEEDS; do
    for k in $DRAWS; do
      tag="${cfg}_s${s}_d${k}"
      cmds+=("mkdir -p out/pm_bound && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' && PYTHONIOENCODING=utf-8 python scripts/retire_measures.py --configs $cfg --seeds $s --sections walk,a,b --timeout 40 --out out/pm_bound/$tag > out/pm_bound/$tag.txt 2>&1; cat out/pm_bound/$tag.txt; tail -n 45 out/pm_bound/$tag/$cfg.log")
    done
  done
done
if [ "$1" = "--print" ]; then
  printf '%s\n' "${cmds[@]}"; exit 0
fi
mkdir -p out/pm_bound
python scripts/cluster_run.py --name pmb --minutes 30 "${cmds[@]}" --fetch out/pm_bound/ 2>&1 | tee out/pm_bound_cluster.log
