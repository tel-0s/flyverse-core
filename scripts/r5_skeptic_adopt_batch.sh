#!/usr/bin/env bash
# Skeptic verification of the round-5 GF-damping adoption (task key adopt-gf): ONE cluster batch, 4 concurrent jobs,
# on the working tree as shipped (flyverse/brain.py DEFAULT_TYPE_PATH_GAIN = [(LC4|LPLC2 -> DNp01, 3.0)] only).
#     bash scripts/r5_skeptic_adopt_batch.sh 2>&1 | tee out/r5_skeptic_adopt_cluster.log
#
#   job 0  provenance: print the run directory's flyverse/brain.py type-gain block and re-derive the four
#          _shaped_weights md5s on the CLUSTER's shared connectome cache (scripts/r5_adopt_structure.py, own JSON path)
#   job 1  a fourth no-flag suite replicate on the edited default (the adopt task ran three): 27/0/2? walk.power_max 48.48?
#   job 2  a second independent draw of the opt-in `hops` section on the edited default (the adopt task ran one draw;
#          its own caveat is that 2,400 fly-s under-samples the room rate)
#   job 3  a fourth room batch, brain seed 0 / env seeds 0-15, identical protocol to out/r5_adopt_sustain_live_1.json
#          (19 hops = 7 escape + 12 voluntary there): does the edited default's take-off rate replicate?
# Nothing of the adopt task's own out/ files is written: every output path here is r5_skeptic_*.
set -u
cd "$(dirname "$0")/.."
G="python -c 'import torch; assert torch.cuda.is_available()' &&"
S="python scripts/batch_sustain.py --batch 16 --minutes 5 --energy 0.9 --program cx --fruit apple --fence --cuda-graphs --cuda-kernels --event-driven --cuda-sparse torch"
cmds=(
  "$G sed -n '215,235p' flyverse/brain.py > out/r5_skeptic_brainpy.txt; python scripts/r5_adopt_structure.py --json out/r5_skeptic_structure.json > out/r5_skeptic_structure.txt; cat out/r5_skeptic_brainpy.txt out/r5_skeptic_structure.txt"
  "$G python scripts/benchmark.py --seeds 0,1,2 --json out/r5_skeptic_default_4.json > out/r5_skeptic_default_4.txt; cat out/r5_skeptic_default_4.txt"
  "$G python scripts/benchmark.py --sections hops --json out/r5_skeptic_hops.json > out/r5_skeptic_hops.txt; cat out/r5_skeptic_hops.txt"
  "$G $S --seed 0 --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --json out/r5_skeptic_sustain_live_4.json > out/r5_skeptic_sustain_live_4.txt; cat out/r5_skeptic_sustain_live_4.txt"
)
printf 'job: %s\n' "${cmds[@]}"
PYTHONIOENCODING=utf-8 python scripts/cluster_run.py --name r5-skeptic-adopt --minutes 90 "${cmds[@]}" --fetch out/
