#!/usr/bin/env bash
# Receptor round 5, completion step: fetch the finished r5-hops batch from the cluster run directory (the 14 jobs of
# scripts/r5_cluster_batch.sh keep running on the cluster after the submitting session ends) and print every number the
# 'Round 5: the take-off check' section needs.  Run from the repo root with Git Bash:
#     bash scripts/r5_fetch_and_report.sh [run-dir-name]      (default r5-hops-218d81)
# Logs: out/r5_hops_report.log (this script's console), out/r5_cluster.log (the submitting console, if it finished).
set -u
cd "$(dirname "$0")/.."
RUN="${1:-r5-hops-218d81}"
SSH=$(python -c "import json;print(json.load(open('.cluster.json'))['ssh'])")
RUNS=$(python -c "import json;print(json.load(open('.cluster.json'))['runs'].rstrip('/'))")
echo "== fetching $SSH:$RUNS/$RUN/out/ -> out/"
scp -q -r "$SSH:$RUNS/$RUN/out/." out/ || { echo "FETCH FAILED"; exit 1; }
ls -la out/r5_sustain_*.json out/r5_hops_default.json out/r5_hops_off.json 2>&1
echo "== job logs (device / receptor headers / final lines)"
for f in out/r5_sustain_*.txt out/r5_hops_default.txt out/r5_hops_off.txt; do
  echo "-- $f"; grep -E "receptor model|device=|take-off routes|^hops |rollouts saved|pass, .* fail|Traceback|WARNING" "$f" | cut -c1-200
done
C="python scripts/compare_sustain_runs.py"
echo "== A. live escape route: default vs off (3 seed-matched batches)"
$C --a out/r5_sustain_default_live_{1,2,3}.json --b out/r5_sustain_off_live_{1,2,3}.json --label-a default --label-b off --ref-proposal
echo "== B. escape route disabled (--gf-hz 1e9): default vs off"
$C --a out/r5_sustain_default_nogf_{1,2,3}.json --b out/r5_sustain_off_nogf_{1,2,3}.json --label-a default --label-b off --ref-proposal
echo "== C. default: live vs disabled route (is the voluntary rate independent of the escape route?)"
$C --a out/r5_sustain_default_live_{1,2,3}.json --b out/r5_sustain_default_nogf_{1,2,3}.json --label-a live --label-b nogf
echo "== D. off: live vs disabled route"
$C --a out/r5_sustain_off_live_{1,2,3}.json --b out/r5_sustain_off_nogf_{1,2,3}.json --label-a live --label-b nogf
echo "== E. the benchmark section, re-scored against the REFERENCES now in scripts/benchmark.py"
$C --benchmark-json out/r5_hops_default.json out/r5_hops_off.json
echo "== F. identical-seed check against round 4 (brain seed 0, env 0-15; round-4 default 21 / 17 / 24 hops, off 1 / 0 / 2)"
python - <<'EOF'
import json
for f in ("out/r5_sustain_default_live_1.json", "out/r5_sustain_off_live_1.json", "out/r4_sustain_default_1.json", "out/r4_sustain_default_1b.json",
          "out/sk4_route_default_1.json", "out/r4_sustain_off_1.json", "out/r4_sustain_off_1b.json", "out/sk4_route_off_1.json"):
    try:
        d = json.load(open(f))
    except FileNotFoundError:
        print(f, "missing"); continue
    rows = d["rows"]; rec = d.get("receptor", {})
    print(f"{f}: receptor {rec.get('model')} brain_seed {d.get('brain_seed')} hops {sum(r['hops'] for r in rows)}"
          + (f" = escape {sum(r['hops_escape'] for r in rows)} + voluntary {sum(r['hops_voluntary'] for r in rows)}" if "hops_escape" in rows[0] else "")
          + (f" (probe: escape {d.get('escape_total')} + voluntary {d.get('voluntary_total')})" if "escape_total" in d else ""))
EOF
