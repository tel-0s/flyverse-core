#!/usr/bin/env bash
# Behaviour round 3 / anti-runaway round 7: the REGRESSION GUARDS (no model change), ONE cluster submission.
#
#   bash scripts/guard_suites.sh                 # submit + wait + fetch out/guard_r3/; console -> out/guard_r3_cluster.log
#   bash scripts/guard_suites.sh --print         # only print the job commands (and the block key of each)
#   bash scripts/guard_suites.sh --smoke         # CPU smoke of the embedded wrapper on this desktop (CUDA_VISIBLE_DEVICES=-1)
#   bash scripts/guard_suites.sh --wrapper       # print the embedded wrapper (what every room / hops job runs)
#   bash scripts/guard_suites.sh --report        # aggregate out/guard_r3/ -> out/guard_r3/guard_report.md + guard_summary.json (CPU)
#
# If the submitting client dies (it did: session limit + a 5 h box outage), attach by hand -- the batch keeps running on
# the boxes: `python scripts/box_status.py --target <box> --prefix guard7 --wait --poll 120` per box, then
# `python scripts/fetch_run.py --target <box> --run guard7-<id> --out guard_r3` per box (pulls only what is not local),
# then `--report`. Never resubmit a family that is already on a box.
#
# Two arm-blocks (--arm-block fam: the block is the directory token `fam_<block>` in every job's paths, so every
# arm of a comparison family runs on ONE box; the reference arm sits in the same block as its treatments):
#
#   fam_pinned (15 jobs)
#     suite_default_{1,2,3}   scripts/retire_measures.py --configs baseline        --sections all --seeds 0,1,2   (29 checks)
#     suite_noclip_{1,2,3}    scripts/retire_measures.py --configs no_drive_clip   (OpticParams.drive_clip_mv 35 -> 1e9)
#     suite_lpi1_{1,2,3}      scripts/retire_measures.py --configs pair_gain_lpi_x1 (LPi34/43 -> LPLC2 x4 -> x1)
#     hops_default_{1,2,3}    scripts/benchmark.py --sections hops (2,400 fly-s; the take-off reference for this round's boxes)
#     hops_proprio_{1,2,3}    the same section with senses.Proprioception('all') attached through BatchSim(proprioception='all')
#                             -- the ONLY section of the suite that runs through BatchSim; the pinned 29-check sections
#                             (Brain / OpticLobe protocols and scripts/room_demo.py's single-fly Sim) have no hook for the
#                             transducer (flyverse/fly.py attaches it only when a caller sets `proprioception_sense`;
#                             room_demo.Sim never does), so a 29-check "transducer on" suite would be the default suite
#                             bit for bit and is NOT run (docs/audits/guard_suites_r3.md says so).
#   fam_room (12 jobs)        the room take-off protocol: scripts/batch_sustain.py --batch 16 --minutes 5 --energy 0.9
#                             --program cx --fruit apple --fence, live escape route, brain seeds 0 / 1 / 2 x env seeds
#                             0-15 / 16-31 / 32-47 (the round-5 adopt-alone protocol, 3 x 4,800 fly-s per arm)
#     room_default_{1,2,3}    the shipped default
#     room_noclip_{1,2,3}     drive_clip_mv 1e9
#     room_lpi1_{1,2,3}       LPi34/43 -> LPLC2 x1
#     room_proprio_{1,2,3}    --proprioception all (the transducer arm; opt-in flag of batch_sustain.py)
#
# batch_sustain.py and benchmark.py have no optic-override flag, so the two candidates (and, for a single harness, every
# room / hops arm) run through the wrapper below, which each job materialises on the box with
# `bash scripts/guard_suites.sh --wrapper > out/.../wrap_<tag>.py` (this file is shipped with the working-tree diff;
# out/ is git-ignored; an inline base64 copy per job overflowed Windows' argument-list limit on the first submission). It applies the override through the module-level factory the simulators
# call (optic.OpticParams -> a subclass with the field set; BatchSim -> a subclass that records the instance), verifies
# on the built simulator that the override took (exit 3 otherwise), and appends a `guard` block and the mandatory
# provenance (flyverse.interp.common.provenance: resolved LIFParams / OpticParams, realised device, cache fingerprint)
# to the JSON the tool wrote. retire_measures.py writes its own provenance block.
#
# Exit codes are preserved (`st=$?; test -s <json> || st=1; tail -4 <txt>; exit $st`), never `; tail` alone.
set -u
cd "$(dirname "$0")/.."
OUT=out/guard_r3
MODE="${1:-submit}"

# ---------------------------------------------------------------------------------------------------- the wrapper
read -r -d '' WRAP <<'PYEOF'
"""guard_wrap.py -- materialised on the box by scripts/guard_suites.sh (anti-runaway round 7 regression guards).

    python guard_wrap.py sustain [--drive-clip MV] [--lpi F] -- <scripts/batch_sustain.py arguments>
    python guard_wrap.py hops    [--drive-clip MV] [--lpi F] [--proprioception SPEC] -- <scripts/benchmark.py arguments>

Runs the tool unchanged. The optic override reaches every OpticParams the simulator builds (optic.OpticParams is
rebound to a subclass that sets the field after the dataclass __init__); the transducer reaches sec_hops through
flyverse.BatchSim (rebound to a subclass that defaults proprioception=SPEC and records the instance). After the tool
returns, the built simulator is checked against the request (exit 3 on a mismatch) and the JSON gets `guard` and
`provenance` blocks."""
import dataclasses
import json
import os
import sys
import time

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
LPI = (r"^LPi(34|43)$", r"^LPLC2$")
CAPTURED = []


def parse(argv):
    if not argv or argv[0] not in ("sustain", "hops"):
        raise SystemExit("guard_wrap: mode must be 'sustain' or 'hops'")
    mode, opts, i = argv[0], {"drive_clip": None, "lpi": None, "proprioception": None}, 1
    while i < len(argv) and argv[i] != "--":
        k = argv[i]
        if k == "--drive-clip":
            opts["drive_clip"] = float(argv[i + 1]); i += 2
        elif k == "--lpi":
            opts["lpi"] = float(argv[i + 1]); i += 2
        elif k == "--proprioception":
            opts["proprioception"] = argv[i + 1]; i += 2
        else:
            raise SystemExit(f"guard_wrap: unknown option {k}")
    return mode, opts, (argv[i + 1:] if i < len(argv) else [])


def patch_optic(opts):
    from flyverse import optic
    if opts["drive_clip"] is None and opts["lpi"] is None:
        return
    Base = optic.OpticParams

    class GuardOpticParams(Base):
        def __init__(self, **kw):
            super().__init__(**kw)
            if opts["drive_clip"] is not None:
                self.drive_clip_mv = float(opts["drive_clip"])
            if opts["lpi"] is not None:
                out, found = [], 0
                for pre, post, g in optic.DEFAULT_PAIR_GAIN:
                    if (pre, post) == LPI:
                        found += 1; out.append((pre, post, float(opts["lpi"])))
                    else:
                        out.append((pre, post, g))
                if found != 1:
                    raise SystemExit(f"guard_wrap: optic.DEFAULT_PAIR_GAIN has {found} LPi -> LPLC2 entries, expected 1")
                self.pair_gain = out

    GuardOpticParams.__name__ = GuardOpticParams.__qualname__ = "OpticParams"
    optic.OpticParams = GuardOpticParams


def guard_batchsim(proprio):
    from flyverse.batch_sim import BatchSim as Base

    class GuardBatchSim(Base):
        def __init__(self, *a, **kw):
            if proprio is not None:
                kw.setdefault("proprioception", proprio)
            super().__init__(*a, **kw)
            CAPTURED.append(self)

    GuardBatchSim.__name__ = GuardBatchSim.__qualname__ = "BatchSim"
    return GuardBatchSim


def finish(mode, opts, rest, t0):
    import torch
    from flyverse import brain, optic
    from flyverse.interp import common
    path = rest[rest.index("--json") + 1]
    d = json.load(open(path, encoding="utf-8"))
    if not CAPTURED:
        raise SystemExit("guard_wrap: no BatchSim was built (exit 3)")
    sim = CAPTURED[-1]
    lif, op = sim.fb.brain.p, sim.optic.p
    pair_gain = optic.DEFAULT_PAIR_GAIN if op.pair_gain is None else op.pair_gain
    lpi_now = [float(g) for pre, post, g in pair_gain if (pre, post) == LPI]
    sense = getattr(sim.fb, "proprioception_sense", None)
    resolved = {"drive_clip_mv": float(op.drive_clip_mv), "lpi_lplc2_pair_gain": lpi_now, "pair_gain_is_default": op.pair_gain is None,
                "gain_out_mv": float(op.gain_out_mv), "optic_params_class": type(op).__module__ + "." + type(op).__name__,
                "proprioception": sim.proprioception, "proprioception_sense_attached": sense is not None,
                "proprioception_channels": sorted(getattr(sense, "channels", []) or []) if sense is not None else [],
                "receptor_model": lif.receptor_model, "receptor_net_rule": lif.receptor_net_rule, "w_syn": lif.w_syn, "conn_cap": lif.conn_cap,
                "path_gain": brain.DEFAULT_PATH_GAIN if lif.path_gain is None else lif.path_gain,
                "type_path_gain": brain.DEFAULT_TYPE_PATH_GAIN if lif.type_path_gain is None else lif.type_path_gain,
                "flight": {"gf_hz": float(sim.flights[0].gf_hz), "takeoff_power_hz": float(sim.flights[0].takeoff_power_hz),
                           "takeoff_hold_s": float(sim.flights[0].takeoff_hold_s)}}
    problems = []
    if opts["drive_clip"] is not None and resolved["drive_clip_mv"] != float(opts["drive_clip"]):
        problems.append(f"drive_clip_mv {resolved['drive_clip_mv']} != {opts['drive_clip']}")
    if opts["lpi"] is not None and lpi_now != [float(opts["lpi"])]:
        problems.append(f"LPi -> LPLC2 pair gain {lpi_now} != [{opts['lpi']}]")
    if opts["drive_clip"] is None and resolved["drive_clip_mv"] != 35.0:
        problems.append(f"drive_clip_mv {resolved['drive_clip_mv']} != the shipped 35")
    if opts["lpi"] is None and lpi_now != [4.0]:
        problems.append(f"LPi -> LPLC2 pair gain {lpi_now} != the shipped [4.0]")
    want_proprio = opts["proprioception"] if mode == "hops" else (rest[rest.index("--proprioception") + 1] if "--proprioception" in rest else None)
    if (want_proprio is not None) != (sense is not None):
        problems.append(f"proprioception requested {want_proprio!r}, sense attached {sense is not None}")
    dev = str(sim.fb.device)
    if "cuda" not in dev and os.environ.get("GUARD_ALLOW_CPU") != "1":
        problems.append(f"realised device {dev} is not a GPU")
    guard = {"mode": mode, "overrides_requested": opts, "resolved": resolved, "problems": problems, "device": dev,
             "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
             "wall_s": round(time.time() - t0, 1), "wrapper": "scripts/guard_suites.sh (embedded guard_wrap.py)", "tool_argv": rest}
    seeds = {"sustain": [int(rest[rest.index("--seed") + 1])] if "--seed" in rest else [0], "hops": [0]}[mode]
    stim = {"protocol": "batch_sustain.py --program cx --fruit apple --fence --energy 0.9, live escape route" if mode == "sustain"
            else "benchmark.py --sections hops (batch_sustain.py's protocol, 16 rooms, brain seed 0, env seeds 0-15)",
            "params": {"batch": sim.B, "env_seeds": list(sim.seeds), "program": sim.program_names[0], "fence": sim.fence, "fruit_set": sim.fruit_set,
                       "proprioception": sim.proprioception}, "control": "room_default / hops_default of the same block"}
    prov = common.provenance(sim.c, fb=sim.fb, device=dev, seeds=seeds, env_seeds=list(sim.seeds), batch=sim.B, stimulus=stim)
    d["guard"], d["provenance"] = guard, prov
    with open(path, "w", encoding="utf-8") as f:
        json.dump(common.to_jsonable(d), f, indent=1)
    print(f"guard: mode {mode} requested {opts} resolved drive_clip {resolved['drive_clip_mv']} LPi {lpi_now} proprio {sim.proprioception!r} "
          f"(sense attached {sense is not None}) device {dev} {guard['device_name']} md5 {prov['compiled_connectome'].get('md5')}", flush=True)
    if problems:
        print("guard: INVALID -- " + "; ".join(problems), flush=True)
        sys.exit(3)


def main():
    t0 = time.time()
    mode, opts, rest = parse(sys.argv[1:])
    patch_optic(opts)
    if mode == "sustain":
        import batch_sustain
        batch_sustain.BatchSim = guard_batchsim(None)
        sys.argv = ["batch_sustain.py"] + rest
        batch_sustain.main()
    else:
        import flyverse
        flyverse.BatchSim = guard_batchsim(opts["proprioception"])
        import benchmark
        sys.argv = ["benchmark.py"] + rest
        benchmark.main()
    finish(mode, opts, rest, t0)


if __name__ == "__main__":
    main()
PYEOF
if [ "$MODE" = --wrapper ]; then printf '%s\n' "$WRAP"; exit 0; fi   # the jobs materialise the wrapper from this (shipped) file
W='bash scripts/guard_suites.sh --wrapper >'

# ---------------------------------------------------------------------------------------------------- the jobs
G="python -c 'import torch; assert torch.cuda.is_available()'"
S="--batch 16 --minutes 5 --energy 0.9 --program cx --fruit apple --fence --cuda-graphs --cuda-kernels --event-driven --cuda-sparse torch"
seeds=("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" "16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" "32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47")
declare -A SUITE_CFG=([default]=baseline [noclip]=no_drive_clip [lpi1]=pair_gain_lpi_x1)
declare -A OPT=([default]="" [noclip]="--drive-clip 1e9" [lpi1]="--lpi 1.0" [proprio]="")
cmds=()
# fam_pinned: the 29-check suite x 3 arms x 3 draws, then hops x 2 arms x 3 draws
for arm in default noclip lpi1; do
  for n in 1 2 3; do
    D=$OUT/fam_pinned; tag=suite_${arm}_$n; cfg=${SUITE_CFG[$arm]}
    cmds+=("mkdir -p $D && source .venv/bin/activate && $G && PYTHONIOENCODING=utf-8 python scripts/retire_measures.py --configs $cfg --sections all --seeds 0,1,2 --timeout 60 --out $D/$tag > $D/$tag.txt 2>&1; st=\$?; test -s $D/$tag/$cfg.json || st=1; tail -4 $D/$tag.txt; exit \$st")
  done
done
for arm in default proprio; do
  for n in 1 2 3; do
    D=$OUT/fam_pinned; tag=hops_${arm}_$n; P=""; [ "$arm" = proprio ] && P="--proprioception all"
    cmds+=("mkdir -p $D && source .venv/bin/activate && $G && $W $D/wrap_$tag.py && PYTHONIOENCODING=utf-8 python $D/wrap_$tag.py hops $P -- --sections hops --json $D/$tag.json > $D/$tag.txt 2>&1; st=\$?; test -s $D/$tag.json || st=1; tail -4 $D/$tag.txt; exit \$st")
  done
done
# fam_room: the room take-off protocol x 4 arms x 3 seed-matched batches
for arm in default noclip lpi1 proprio; do
  for k in 0 1 2; do
    n=$((k+1)); D=$OUT/fam_room; tag=room_${arm}_$n; P=""; [ "$arm" = proprio ] && P="--proprioception all"
    cmds+=("mkdir -p $D && source .venv/bin/activate && $G && $W $D/wrap_$tag.py && PYTHONIOENCODING=utf-8 python $D/wrap_$tag.py sustain ${OPT[$arm]} -- $S --seed $k --seeds ${seeds[$k]} $P --json $D/$tag.json > $D/$tag.txt 2>&1; st=\$?; test -s $D/$tag.json || st=1; tail -4 $D/$tag.txt; exit \$st")
  done
done

case "$MODE" in
  --print)
    for c in "${cmds[@]}"; do printf '%s\n\n' "$c"; done
    echo "${#cmds[@]} jobs"; exit 0 ;;
  --smoke)
    # The wrapper on this desktop, CPU only (no GPU work here): a 2-row, 12-frame room batch with the clip removed and
    # with the transducer on, and a 2-row, 6-frame hops section with the transducer on. Checks the override plumbing,
    # the proprioception attachment and the provenance block; the numbers mean nothing.
    export CUDA_VISIBLE_DEVICES=-1 SDL_VIDEODRIVER=dummy PYTHONIOENCODING=utf-8 GUARD_ALLOW_CPU=1
    mkdir -p $OUT/smoke; printf '%s\n' "$WRAP" > $OUT/smoke/guard_wrap.py
    set -e
    python $OUT/smoke/guard_wrap.py sustain --drive-clip 1e9 -- --batch 2 --minutes 0.002 --energy 0.9 --program cx --fruit apple --fence --cuda-sparse torch --seed 0 --seeds 0,1 --json $OUT/smoke/sustain_noclip.json
    python $OUT/smoke/guard_wrap.py sustain --lpi 1.0 -- --batch 2 --minutes 0.002 --energy 0.9 --program cx --fruit apple --fence --cuda-sparse torch --seed 0 --seeds 0,1 --proprioception all --json $OUT/smoke/sustain_lpi1_proprio.json
    python $OUT/smoke/guard_wrap.py hops --proprioception all -- --sections hops --hops-batch 2 --hops-minutes 0.001 --eager --json $OUT/smoke/hops_proprio.json
    python - <<'EOF'
import json
for f in ["out/guard_r3/smoke/sustain_noclip.json", "out/guard_r3/smoke/sustain_lpi1_proprio.json", "out/guard_r3/smoke/hops_proprio.json"]:
    d = json.load(open(f)); g = d["guard"]; p = d["provenance"]
    print(f, "problems", g["problems"], "resolved", {k: g["resolved"][k] for k in ("drive_clip_mv", "lpi_lplc2_pair_gain", "proprioception", "proprioception_sense_attached")},
          "device", p["execution"]["device"], "md5", p["compiled_connectome"]["md5"], "optic.drive_clip", p["model"]["optic"].get("drive_clip_mv"))
EOF
    exit 0 ;;
  --report)
    PYTHONIOENCODING=utf-8 python - "$OUT" <<'PYEOF'
import glob, json, os, sys
import numpy as np
from scipy import stats
OUT = sys.argv[1]
RANK = {"PASS": 2, "PASS (gap closed)": 2, "KNOWN GAP": 1, "FAIL": 0, "MISSING": -1}
REPORTED = {"walk.power_max_hz", "loom.escape_cm"}          # notnone checks: reported, not scored (benchmark.py REFERENCES)
lines, summary = [], {"suite": {}, "hops": {}, "room": {}, "verdicts": {}, "devices": {}, "files": {}}

def load(pattern):
    return {f: json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(pattern))}

def poisson_ci(k, T, level=0.95):
    a = 1 - level
    lo = 0.0 if k == 0 else stats.chi2.ppf(a / 2, 2 * k) / 2
    hi = stats.chi2.ppf(1 - a / 2, 2 * k + 2) / 2
    return lo / T * 1000, hi / T * 1000

# ---- the suite
CFG = {"default": "baseline", "noclip": "no_drive_clip", "lpi1": "pair_gain_lpi_x1"}   # the configuration JSON (comparison.json sits beside it)
suite = {arm: load(f"{OUT}/fam_pinned/suite_{arm}_*/{CFG[arm]}.json") for arm in ("default", "noclip", "lpi1")}
lines += ["# Guard suites, round 7 (generated by scripts/guard_suites.sh --report)", "",
          "## 1. The 29-check suite, arm x draw (scripts/retire_measures.py, --sections all --seeds 0,1,2)", ""]
keys = []
for arm, runs in suite.items():
    for f, d in runs.items():
        summary["devices"][f] = d["config"].get("device"); summary["files"].setdefault("suite", []).append(f)
        for c in d["checks"]:
            if c["key"] not in keys: keys.append(c["key"])
def tally(d):
    n = {"pass": 0, "fail": 0, "gap": 0, "missing": 0}
    for c in d["checks"]:
        s = c["status"]; n["pass" if s.startswith("PASS") else "fail" if s == "FAIL" else "gap" if s == "KNOWN GAP" else "missing"] += 1
    return n
hdr = "| check (criterion) | " + " | ".join(f"{arm} d{i+1}" for arm in suite for i in range(len(suite[arm]))) + " |"
lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
worse = {arm: {} for arm in suite}
for k in keys:
    row, crit = [], ""
    base_ranks = []
    for arm, runs in suite.items():
        for f, d in runs.items():
            c = next((c for c in d["checks"] if c["key"] == k), None)
            if c is None:
                row.append("--"); continue
            crit = c["criterion"]; v = c["measured"]
            s = c["status"]; txt = f"{v:.4g}" if isinstance(v, (int, float)) and v is not None else str(v)
            tag = "R" if k in REPORTED else {"PASS": "P", "PASS (gap closed)": "P*", "KNOWN GAP": "G", "FAIL": "F", "MISSING": "M"}[s]
            row.append(f"{txt} {tag}")
            if arm == "default": base_ranks.append(RANK[s])
    for arm, runs in suite.items():
        if arm == "default" or k in REPORTED: continue
        for i, (f, d) in enumerate(runs.items()):
            c = next((c for c in d["checks"] if c["key"] == k), None)
            if c is not None and base_ranks and RANK[c["status"]] < min(base_ranks):
                worse[arm].setdefault(k, []).append(i + 1)
    lines.append(f"| {k} ({crit}){' [reported, not scored]' if k in REPORTED else ''} | " + " | ".join(row) + " |")
lines += ["", "P PASS, P* PASS (gap closed), G KNOWN GAP, F FAIL, M MISSING, R reported (notnone: walk.power_max_hz and loom.escape_cm count as PASS whenever measured).", ""]
lines += ["| arm | draw | file | device | pass / fail / known gap / missing | walk.power_max (reported) | walk.power_sustained | walk.GF_max |", "|---|---|---|---|---|---|---|---|"]
for arm, runs in suite.items():
    summary["suite"][arm] = []
    for i, (f, d) in enumerate(runs.items()):
        t = tally(d); w = d["sections"].get("walk", {}).get("walk", {})
        summary["suite"][arm].append({"file": f, "device": d["config"].get("device"), "tally": t, "checks": {c["key"]: [c["measured"], c["status"]] for c in d["checks"]},
                                      "provenance_md5": d.get("provenance", {}).get("compiled_connectome", {}).get("md5"),
                                      "execution_device": d.get("provenance", {}).get("execution", {}).get("device")})
        lines.append(f"| {arm} | {i+1} | {f} | {d['config'].get('device')} | {t['pass']} / {t['fail']} / {t['gap']} / {t['missing']} | "
                     f"{w.get('power_max_hz', float('nan')):.4f} | {w.get('power_sustained_hz', float('nan')):.4f} | {w.get('GF_max_hz', float('nan')):.4f} |")
lines += ["", "Checks worse in status than EVERY baseline draw (candidate draw numbers): " +
          "; ".join(f"{arm}: " + (", ".join(f"{k} (d{','.join(map(str, v))})" for k, v in w.items()) or "none") for arm, w in worse.items() if arm != "default"), ""]
summary["suite_worse"] = {a: w for a, w in worse.items() if a != "default"}

# ---- hops
lines += ["## 2. The hops section (benchmark.py --sections hops: 16 rooms x 150 s = 2,400 fly-s, brain seed 0, env seeds 0-15; 3 GPU draws per arm)", "",
          "| arm | draw | file | device | proprioception (sense attached) | hops = escape + voluntary | escape / voluntary per 1,000 fly-s | walking-GF median | checks |", "|---|---|---|---|---|---|---|---|---|"]
hops = {arm: load(f"{OUT}/fam_pinned/hops_{arm}_*.json") for arm in ("default", "proprio")}
for arm, runs in hops.items():
    summary["hops"][arm] = []
    for i, (f, d) in enumerate(runs.items()):
        h = d["sections"].get("hops", {}); g = d.get("guard", {}); r = g.get("resolved", {})
        summary["devices"][f] = d["config"].get("device"); summary["files"].setdefault("hops", []).append(f)
        summary["hops"][arm].append({"file": f, "device": d["config"].get("device"), "fly_s": h.get("fly_s"), "hops": h.get("hops_total"), "escape": h.get("escape_total"),
                                     "voluntary": h.get("voluntary_total"), "gf_median": h.get("walk_gf_max_median_hz"), "checks": {c["key"]: [c["measured"], c["status"]] for c in d["checks"]},
                                     "proprioception": r.get("proprioception"), "sense_attached": r.get("proprioception_sense_attached"), "problems": g.get("problems"),
                                     "provenance_md5": d.get("provenance", {}).get("compiled_connectome", {}).get("md5")})
        lines.append(f"| {arm} | {i+1} | {f} | {d['config'].get('device')} | {r.get('proprioception')!r} ({r.get('proprioception_sense_attached')}) | "
                     f"{h.get('hops_total')} = {h.get('escape_total')} + {h.get('voluntary_total')} | {h.get('escape_per_1000_fly_s', float('nan')):.2f} / {h.get('voluntary_per_1000_fly_s', float('nan')):.2f} | "
                     f"{h.get('walk_gf_max_median_hz', float('nan')):.2f} | " + ", ".join(f"{c['key'].split('.')[1]} {c['status']}" for c in d["checks"]) + " |")
for arm, runs in hops.items():
    if runs:
        T = sum(d["sections"]["hops"]["fly_s"] for d in runs.values()); K = sum(d["sections"]["hops"]["hops_total"] for d in runs.values())
        lo, hi = poisson_ci(K, T)
        lines.append(f"\n{arm}: pooled {K} take-offs in {T:,.0f} fly-s = {K / T * 1000:.2f} per 1,000 fly-s (exact Poisson 95 % CI {lo:.2f}-{hi:.2f})")
        summary["hops"][arm + "_pooled"] = {"hops": K, "fly_s": T, "rate": K / T * 1000, "ci95": [lo, hi]}
lines.append("")

# ---- the room
lines += ["## 3. The room take-off protocol (batch_sustain.py, 16 flies x 300 s per batch, brain seeds 0 / 1 / 2, env seeds 0-15 / 16-31 / 32-47, live escape route)", "",
          "| arm | batch | file | device | resolved (clip / LPi / proprio) | hops = escape + voluntary | all / escape / voluntary per 1,000 fly-s | walking-GF median (rows >= 33) |", "|---|---|---|---|---|---|---|---|"]
room = {arm: load(f"{OUT}/fam_room/room_{arm}_*.json") for arm in ("default", "noclip", "lpi1", "proprio")}
pooled = {}
for arm, runs in room.items():
    summary["room"][arm] = []; flies = []; K = {"all": 0, "escape": 0, "voluntary": 0}; T = 0.0
    for i, (f, d) in enumerate(runs.items()):
        g = d.get("guard", {}); r = g.get("resolved", {})
        d["device"] = g.get("device_name") or d.get("provenance", {}).get("execution", {}).get("device_name")   # batch_sustain.py writes none
        summary["devices"][f] = d.get("device"); summary["files"].setdefault("room", []).append(f)
        rows = d["rows"]; flies += rows; T += d["fly_s"]
        K["all"] += d["hops_total"]; K["escape"] += d["hops_escape_total"]; K["voluntary"] += d["hops_voluntary_total"]
        summary["room"][arm].append({"file": f, "device": d.get("device"), "brain_seed": d["brain_seed"], "fly_s": d["fly_s"], "hops": d["hops_total"], "escape": d["hops_escape_total"],
                                     "voluntary": d["hops_voluntary_total"], "gf_median": d["gf_max_walk_median_hz"], "rows_gf_at_threshold": d["rows_gf_at_threshold"],
                                     "resolved": r, "problems": g.get("problems"), "provenance_md5": d.get("provenance", {}).get("compiled_connectome", {}).get("md5"),
                                     "execution_device": d.get("provenance", {}).get("execution", {}).get("device")})
        lines.append(f"| {arm} | {i+1} (seed {d['brain_seed']}) | {f} | {d.get('device')} | {r.get('drive_clip_mv')} / {r.get('lpi_lplc2_pair_gain')} / {r.get('proprioception')!r} | "
                     f"{d['hops_total']} = {d['hops_escape_total']} + {d['hops_voluntary_total']} | {d['hops_total'] / d['fly_s'] * 1000:.2f} / {d['hops_escape_per_1000_fly_s']:.2f} / {d['hops_voluntary_per_1000_fly_s']:.2f} | "
                     f"{d['gf_max_walk_median_hz']:.2f} ({d['rows_gf_at_threshold']}/{d['batch']}) |")
    if runs:
        pooled[arm] = {"T": T, "K": K, "flies": flies, "rates": {k: v / T * 1000 for k, v in K.items()}, "ci": {k: poisson_ci(v, T) for k, v in K.items()},
                       "per_batch": [d["hops_total"] / d["fly_s"] * 1000 for d in runs.values()]}
lines += ["", "| arm | batches | fly-s | all per 1,000 fly-s (exact Poisson 95 % CI) [per-batch min-max] | escape (CI) | voluntary (CI) | walking-GF median over flies (rows >= 33 Hz) |", "|---|---|---|---|---|---|---|"]
for arm, p in pooled.items():
    gf = np.array([r["gf_max_walk_hz"] for r in p["flies"]])
    lines.append(f"| {arm} | {len(p['per_batch'])} | {p['T']:,.0f} | {p['rates']['all']:.3f} ({p['ci']['all'][0]:.3f}-{p['ci']['all'][1]:.3f}) [{min(p['per_batch']):.2f}-{max(p['per_batch']):.2f}] | "
                 f"{p['rates']['escape']:.3f} ({p['ci']['escape'][0]:.3f}-{p['ci']['escape'][1]:.3f}) | {p['rates']['voluntary']:.3f} ({p['ci']['voluntary'][0]:.3f}-{p['ci']['voluntary'][1]:.3f}) | "
                 f"{np.median(gf):.2f} ({int((gf >= 33).sum())}/{len(gf)}) |")
    summary["room"][arm + "_pooled"] = {"fly_s": p["T"], "counts": p["K"], "rates": p["rates"], "ci95": p["ci"], "per_batch_rates": p["per_batch"],
                                        "gf_median": float(np.median(gf)), "rows_gf_at_threshold": int((gf >= 33).sum()), "n_flies": len(gf)}
if "default" in pooled:
    lines += ["", "Mann-Whitney over flies vs the default (asymptotic two-sided; hop counts are tied so no exact p), and per-batch seed-matched counts:", "",
              "| arm | metric | U | p | candidate per batch | default per batch | candidate lower in n of 3 |", "|---|---|---|---|---|---|---|"]
    for arm, p in pooled.items():
        if arm == "default": continue
        summary["room"][arm + "_vs_default"] = {}
        for m in ("hops", "hops_escape", "hops_voluntary", "gf_max_walk_hz"):
            a = [r[m] for r in p["flies"]]; b = [r[m] for r in pooled["default"]["flies"]]
            u = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
            ca = [sum(r[m] for r in d["rows"]) if m != "gf_max_walk_hz" else float(np.median([r[m] for r in d["rows"]])) for d in room[arm].values()]
            cb = [sum(r[m] for r in d["rows"]) if m != "gf_max_walk_hz" else float(np.median([r[m] for r in d["rows"]])) for d in room["default"].values()]
            lower = sum(x < y for x, y in zip(ca, cb))
            summary["room"][arm + "_vs_default"][m] = {"U": float(u.statistic), "p": float(u.pvalue), "candidate_per_batch": ca, "default_per_batch": cb, "lower_in": lower}
            lines.append(f"| {arm} | {m} | {u.statistic:.1f} | {u.pvalue:.3g} | {' / '.join(f'{x:.4g}' for x in ca)} | {' / '.join(f'{x:.4g}' for x in cb)} | {lower} |")

# ---- interp.common.compare over runs (the replicate unit; 3 v 3 floors at p 0.10, so every verdict here is 'underpowered'
# by construction -- the scatter is what these rows carry, not a call)
sys.path.insert(0, os.getcwd())
from flyverse.interp import common
lines += ["", "## 3b. `flyverse.interp.common.compare` over runs (replicate unit = run; 3 v 3 is below the 4-per-arm calling floor, so 'underpowered' is structural here)", "",
          "| family | metric | candidate runs | default runs | diff | z | U | p (exact) | p floor | verdict |", "|---|---|---|---|---|---|---|---|---|---|"]
summary["compare"] = {}
def cmp_row(fam, metric, a, b):
    r = common.compare(a, b)
    summary["compare"][f"{fam}.{metric}"] = {k: r[k] for k in ("verdict", "diff", "z", "U", "p", "p_floor", "n_min")} | {"candidate": a, "default": b}
    lines.append(f"| {fam} | {metric} | {' / '.join(f'{x:.4g}' for x in a)} | {' / '.join(f'{x:.4g}' for x in b)} | {r['diff']:.4g} | {r['z']:.3g} | {r['U']:.1f} | {r['p']:.3g} | {r['p_floor']:.3g} | {r['verdict']} |")
if all(hops.get(a) for a in ("default", "proprio")):
    H = {a: list(runs.values()) for a, runs in hops.items()}
    for m, key in (("hops_total", "hops_total"), ("escape_total", "escape_total"), ("voluntary_total", "voluntary_total"), ("walk_gf_max_median_hz", "walk_gf_max_median_hz")):
        cmp_row("hops proprio vs default", m, [d["sections"]["hops"][key] for d in H["proprio"]], [d["sections"]["hops"][key] for d in H["default"]])
for arm in ("noclip", "lpi1", "proprio"):
    if room.get(arm) and room.get("default"):
        for m in ("hops_total", "hops_escape_total", "hops_voluntary_total", "gf_max_walk_median_hz", "rows_gf_at_threshold"):
            cmp_row(f"room {arm} vs default", m, [d[m] for d in room[arm].values()], [d[m] for d in room["default"].values()])
lines.append("")

# ---- the adopt-alone verdicts
lines += ["", "## 4. Adopt-alone verdicts (rule: no check worse in status than the baseline in 3/3 draws AND the pooled room take-off rate inside the baseline's exact Poisson 95 % CI)", ""]
for arm in ("noclip", "lpi1"):
    n_suite = len(suite.get(arm, {})); w = summary["suite_worse"].get(arm, {})
    suite_ok = n_suite >= 3 and not w
    room_ok = None; detail = ""
    if arm in pooled and "default" in pooled:
        lo, hi = pooled["default"]["ci"]["all"]; rate = pooled[arm]["rates"]["all"]
        room_ok = lo <= rate <= hi
        detail = (f"room all-route rate {rate:.3f} per 1,000 fly-s vs the default's {pooled['default']['rates']['all']:.3f} (CI {lo:.3f}-{hi:.3f}); "
                  f"escape {pooled[arm]['rates']['escape']:.3f} vs {pooled['default']['rates']['escape']:.3f} (default CI {pooled['default']['ci']['escape'][0]:.3f}-{pooled['default']['ci']['escape'][1]:.3f}); "
                  f"voluntary {pooled[arm]['rates']['voluntary']:.3f} vs {pooled['default']['rates']['voluntary']:.3f} (default CI {pooled['default']['ci']['voluntary'][0]:.3f}-{pooled['default']['ci']['voluntary'][1]:.3f})")
    verdict = "ADOPTABLE by the rule" if (suite_ok and room_ok) else "NOT adoptable"
    if n_suite < 3 or arm not in pooled: verdict = "INCOMPLETE (missing runs)"
    summary["verdicts"][arm] = {"suite_draws": n_suite, "suite_worse": w, "suite_ok": suite_ok, "room_in_default_ci": room_ok, "verdict": verdict, "detail": detail}
    lines.append(f"* **{arm}**: {verdict}. Suite: {n_suite} draws, checks worse than baseline: {w or 'none'}. Room: {detail or 'no runs'}.")
lines += ["", "Nothing is adopted here; the owner decides (docs/audits/anti_runaway.md round 7).", ""]
lines += ["## 5. Devices", "", "| file | device |", "|---|---|"] + [f"| {f} | {v} |" for f, v in summary["devices"].items()]
os.makedirs(OUT, exist_ok=True)
open(f"{OUT}/guard_report.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
json.dump(summary, open(f"{OUT}/guard_summary.json", "w", encoding="utf-8"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("\n".join(lines)); print(f"\nwrote {OUT}/guard_report.md and {OUT}/guard_summary.json")
PYEOF
    exit $? ;;
  submit) ;;
  *) echo "unknown mode $MODE (submit | --print | --smoke | --report | --wrapper)"; exit 2 ;;
esac

# ---------------------------------------------------------------------------------------------------- submit
mkdir -p $OUT
{ echo "submitted $(date -u +%FT%TZ) from $(git rev-parse HEAD)"; git status --short; echo; git diff --stat; } > $OUT/submit_tree.txt
printf '%s\n' "$WRAP" > $OUT/guard_wrap.py        # the wrapper as shipped (identical to the base64 in every job)
echo "${#cmds[@]} jobs; working tree at submission in $OUT/submit_tree.txt"
PYTHONIOENCODING=utf-8 python scripts/cluster_run.py --name guard7 --minutes 180 --arm-block fam "${cmds[@]}" --fetch $OUT/ 2>&1 | tee out/guard_r3_cluster.log
