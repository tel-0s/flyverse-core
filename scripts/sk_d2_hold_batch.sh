#!/usr/bin/env bash
# SKEPTIC replication of the take-off class split at FRESH brain seeds 4 / 5 / 6 (environment seeds 64-79 / 80-95 /
# 96-111), the same five arms and the identical G.5 protocol as scripts/d2_hold_batch.sh. ONE submission, 16 jobs.
# Fetch directory out/sk_d2_hold/ (never shared with the reported run's out/d2_hold/).
set -u
cd /d/Projects/flyverse
D=out/sk_d2_hold
P="mkdir -p $D && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' &&"
T="python scripts/build_hold_tables.py --groups OpticHis,OpticGlu,OpticRandom --out-dir $D"
S="python scripts/batch_sustain.py --batch 16 --minutes 5 --energy 0.9 --program cx --fruit apple --fence --cuda-graphs --cuda-kernels --event-driven --cuda-sparse torch"
declare -A SEEDSET
SEEDSET[4]="64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79"
SEEDSET[5]="80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95"
SEEDSET[6]="96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111"

PROV="$P md5sum flyverse/brain.py flyverse/body.py flyverse/batch_body.py flyverse/batch_sim.py flyverse/fly.py flyverse/optic.py flyverse/senses.py flyverse/programs.py flyverse/cx.py flyverse/connectome.py flyverse/data/receptors_by_type.csv scripts/batch_sustain.py scripts/build_hold_tables.py > $D/provenance.txt; git rev-parse HEAD >> $D/provenance.txt; python -c \"
import json,torch
from flyverse import brain
print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
p=brain.LIFParams()
print('DEFAULT_TYPE_PATH_GAIN',brain.DEFAULT_TYPE_PATH_GAIN)
print('LIFParams()',json.dumps({k:(str(v) if not isinstance(v,(int,float,str,bool,type(None),list)) else v) for k,v in vars(p).items()},default=str,sort_keys=True))
\" >> $D/provenance.txt 2>&1; $T --verify >> $D/provenance.txt 2>&1; md5sum $D/receptors_holdOpticHis.csv $D/receptors_holdOpticGlu.csv $D/receptors_holdOpticRandom.csv >> $D/provenance.txt; cat $D/provenance.txt"

cmds=()
cmds+=("$PROV")
for k in 4 5 6; do
  for arm in shipped off holdOpticHis holdOpticGlu holdOpticRandom; do
    name="${arm}_$k"
    case $arm in
      shipped) R="--receptor-model default" ; B="" ;;
      off)     R="--receptor-model off" ; B="" ;;
      *)       R="--receptor-model sign --receptor-net-rule abs --receptor-table $D/receptors_${arm}.csv" ; B="$T && " ;;
    esac
    cmds+=("$P $B$S $R --seed $k --seeds ${SEEDSET[$k]} --json $D/$name.json > $D/$name.txt 2>&1; tail -3 $D/$name.txt")
  done
done
printf 'job: %s\n' "${cmds[@]}"
PYTHONIOENCODING=utf-8 python scripts/cluster_run.py --name sk-d2hold --minutes 150 "${cmds[@]}" --fetch $D/
