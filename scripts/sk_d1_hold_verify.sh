#!/usr/bin/env bash
# SKEPTIC replication of the DYNAMICS round-1 hold pair (thread takeoff-hold).
#
# An INDEPENDENT 4th replicate of all four room arms at a brain seed and 16 environment seeds that
# appear in none of the audited batches (brain seed 3, env 48-63), so nothing here re-uses the
# audited draws.  Matched env seeds across the four arms.
#
#   job 0  provenance: md5s of the run dir's code, the resolved LIFParams / type_path_gain that
#          batch_sustain actually builds (the room JSONs do not record it), and the hold-table md5s
#   job 1  holdBrain  = OPTIC side alone   (44,463 entries)
#   job 2  holdOptic  = BRAIN side alone   (3,832 entries)
#   job 3  off, shipped gains              (0)
#   job 4  shipped default, both sides     (48,295)
#
# Run from the repo root with Git Bash:
#     bash scripts/sk_d1_hold_verify.sh 2>&1 | tee out/sk_d1_hold_cluster.log
set -u
cd "$(dirname "$0")/.."
G="python -c 'import torch; assert torch.cuda.is_available()' &&"
T="python scripts/build_hold_tables.py --groups Brain,Optic --verify"
S="python scripts/batch_sustain.py --batch 16 --minutes 5 --energy 0.9 --program cx --fruit apple --fence --cuda-graphs --cuda-kernels --event-driven --cuda-sparse torch --seed 3 --seeds 48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63"

PROV="$G md5sum flyverse/brain.py flyverse/body.py flyverse/batch_body.py flyverse/batch_sim.py flyverse/optic.py flyverse/programs.py flyverse/cx.py flyverse/data/receptors_by_type.csv scripts/batch_sustain.py scripts/build_hold_tables.py && git -C . rev-parse HEAD 2>/dev/null; python -c \"
import json,torch
from flyverse import brain
print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
p=brain.LIFParams()
print('DEFAULT_TYPE_PATH_GAIN',brain.DEFAULT_TYPE_PATH_GAIN)
print('GF_DAMPED_TYPE_PATH_GAIN',brain.GF_DAMPED_TYPE_PATH_GAIN)
print('LIFParams()',json.dumps({k:(str(v) if not isinstance(v,(int,float,str,bool,type(None),list)) else v) for k,v in vars(p).items()},default=str,sort_keys=True))
\" && $T && md5sum out/receptors_holdBrain.csv out/receptors_holdOptic.csv && python scripts/batch_sustain.py --batch 2 --minutes 0.02 --energy 0.9 --program cx --fruit apple --fence --cuda-sparse torch --receptor-model sign --receptor-net-rule abs --receptor-table out/receptors_holdBrain.csv --seed 3 --seeds 48,49 --json out/sk_d1_prov_smoke.json > out/sk_d1_prov.txt 2>&1; tail -40 out/sk_d1_prov.txt; python -c \"
import json;d=json.load(open('out/sk_d1_prov_smoke.json'));print('SMOKE receptor',json.dumps(d['receptor']));print('SMOKE flight',json.dumps(d['flight']))\""

cmds=()
cmds+=("$PROV")
cmds+=("$G $T && $S --receptor-model sign --receptor-net-rule abs --receptor-table out/receptors_holdBrain.csv --json out/sk_d1_holdBrain_4.json > out/sk_d1_holdBrain_4.txt; cat out/sk_d1_holdBrain_4.txt")
cmds+=("$G $T && $S --receptor-model sign --receptor-net-rule abs --receptor-table out/receptors_holdOptic.csv --json out/sk_d1_holdOptic_4.json > out/sk_d1_holdOptic_4.txt; cat out/sk_d1_holdOptic_4.txt")
cmds+=("$G $S --receptor-model off --json out/sk_d1_off_4.json > out/sk_d1_off_4.txt; cat out/sk_d1_off_4.txt")
cmds+=("$G $S --receptor-model default --json out/sk_d1_shipped_4.json > out/sk_d1_shipped_4.txt; cat out/sk_d1_shipped_4.txt")

printf 'job: %s\n' "${cmds[@]}"
PYTHONIOENCODING=utf-8 python scripts/cluster_run.py --name sk-d1-hold --minutes 120 "${cmds[@]}" --fetch out/
