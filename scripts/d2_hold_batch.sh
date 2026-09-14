#!/usr/bin/env bash
# DYNAMICS round 2 -- the take-off CLASS split with a DOSE control (docs/audits/receptor_integration.md G.6;
# the follow-up G.5 items 5-6 asked for). ONE cluster submission, 21 jobs, the G.5 room protocol exactly:
#
#   arm              table (scripts/build_hold_tables.py --groups OpticHis,OpticGlu,OpticRandom)       entries changed
#   shipped          --receptor-model default (both sides)                                              48,295
#   off              --receptor-model off (the presynaptic-sign rule)                                        0
#   holdOpticHis     out/d2_hold/receptors_holdOpticHis.csv    = the optic HISTAMINE silencings ALONE    17,256
#   holdOpticGlu     out/d2_hold/receptors_holdOpticGlu.csv    = the optic GLUTAMATE (iGluR) flips ALONE 27,207
#   holdOpticRandom  out/d2_hold/receptors_holdOpticRandom.csv = a seeded random optic row subset ALONE   3,832
#                    (= the Brain side's entry dose; the dose control)
#
# 'hold<G>' reads 'hold everything but <G>' for these three (build_hold_tables.py docstring): every other row that
# differs from NT_SIGN -- the rest of the optic side and the whole Brain side -- is held at the prior.
# x brain seeds 0 / 1 / 2 / 3 x environment seeds 0-15 / 16-31 / 32-47 / 48-63 (the four matched G.5 blocks),
# BatchSim 16 flies x 300 s = 4,800 fly-s per job; 4 runs per arm in ONE submission (docs/INTERP.md 10.4 rule 1).
# Job 0 is provenance: code md5s, torch / GPU, DEFAULT_TYPE_PATH_GAIN, LIFParams(), the tables built + --verify
# against the run box's own cache, and their md5s (which must equal the local out/d2_hold/d2_hold_tables_local.log).
# out/ is git-ignored, so every hold job rebuilds its tables in the run directory (atomic writes).
#
# Run from the repo root with Git Bash:
#     bash scripts/d2_hold_batch.sh 2>&1 | tee out/d2_hold_cluster.log
set -u
cd "$(dirname "$0")/.."
D=out/d2_hold
P="mkdir -p $D && source .venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available()' &&"
T="python scripts/build_hold_tables.py --groups OpticHis,OpticGlu,OpticRandom --out-dir $D"
S="python scripts/batch_sustain.py --batch 16 --minutes 5 --energy 0.9 --program cx --fruit apple --fence --cuda-graphs --cuda-kernels --event-driven --cuda-sparse torch"
seeds=("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" "16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" "32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47" "48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63")

PROV="$P md5sum flyverse/brain.py flyverse/body.py flyverse/batch_body.py flyverse/batch_sim.py flyverse/fly.py flyverse/optic.py flyverse/senses.py flyverse/programs.py flyverse/cx.py flyverse/connectome.py flyverse/data/receptors_by_type.csv scripts/batch_sustain.py scripts/build_hold_tables.py > $D/provenance.txt; python -c \"
import json,torch
from flyverse import brain
print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
p=brain.LIFParams()
print('DEFAULT_TYPE_PATH_GAIN',brain.DEFAULT_TYPE_PATH_GAIN)
print('DEFAULT_PATH_GAIN',brain.DEFAULT_PATH_GAIN)
print('LIFParams()',json.dumps({k:(str(v) if not isinstance(v,(int,float,str,bool,type(None),list)) else v) for k,v in vars(p).items()},default=str,sort_keys=True))
\" >> $D/provenance.txt 2>&1; $T --verify >> $D/provenance.txt 2>&1; md5sum $D/receptors_holdOpticHis.csv $D/receptors_holdOpticGlu.csv $D/receptors_holdOpticRandom.csv >> $D/provenance.txt; cat $D/provenance.txt"

cmds=()
cmds+=("$PROV")
for k in 0 1 2 3; do
  for arm in shipped off holdOpticHis holdOpticGlu holdOpticRandom; do
    name="${arm}_$k"
    case $arm in
      shipped) R="--receptor-model default" ; B="" ;;
      off)     R="--receptor-model off" ; B="" ;;
      *)       R="--receptor-model sign --receptor-net-rule abs --receptor-table $D/receptors_${arm}.csv" ; B="$T && " ;;
    esac
    cmds+=("$P $B$S $R --seed $k --seeds ${seeds[$k]} --json $D/$name.json > $D/$name.txt 2>&1; tail -3 $D/$name.txt")
  done
done
printf 'job: %s\n' "${cmds[@]}"
PYTHONIOENCODING=utf-8 python scripts/cluster_run.py --name d2-hold --minutes 150 "${cmds[@]}" --fetch $D/
