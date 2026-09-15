"""Per-family <scheduler> job counts on a rented box, over plain ssh (no tunnel).
usage: python box_status.py --target r3-h200a [--prefix mono] [--wait [--poll 60]]
--wait blocks until no job with the prefix is queued/running; exit 0 if 0 failed else 1.
Run from the repo root (reads the git-ignored .cluster.json)."""
import argparse, json, subprocess, sys, time
REMOTE = r'''
import json, sys, urllib.request
pre = sys.argv[1]
out = {}
for st in ("queued", "running", "failed", "completed", "cancelled"):
    jobs = json.load(urllib.request.urlopen(f"http://127.0.0.1:7000/api/v1/jobs?status={st}&limit=1000", timeout=30))["jobs"]
    names = [((j.get("spec") or {}).get("name") or j.get("name") or "") for j in jobs]
    out[st] = sorted(n for n in names if n.startswith(pre))
print(json.dumps(out))
'''
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True); ap.add_argument("--prefix", default="")
    ap.add_argument("--wait", action="store_true"); ap.add_argument("--poll", type=float, default=60)
    ap.add_argument("--names", action="store_true", help="print job names, not just counts")
    a = ap.parse_args()
    t = json.load(open(".cluster.json"))["targets"][a.target]
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "-o", "StrictHostKeyChecking=no", "-p", str(t.get("port", 22)), t["ssh"], "python3 - '" + a.prefix + "'"]
    while True:
        try:
            r = subprocess.run(ssh, input=REMOTE, capture_output=True, text=True, timeout=120)
            line = [l for l in r.stdout.splitlines() if l.startswith("{")]
            if not line: raise RuntimeError(r.stderr.strip()[-200:])
            d = json.loads(line[-1])
        except Exception as e:
            print(f"{time.strftime('%H:%M:%S')} {a.target}: poll failed: {e}", flush=True)
            if not a.wait: sys.exit(2)
            time.sleep(a.poll); continue
        counts = {k: len(v) for k, v in d.items()}
        print(f"{time.strftime('%H:%M:%S')} {a.target} prefix={a.prefix!r}: " + " ".join(f"{k}={v}" for k, v in counts.items()), flush=True)
        if a.names:
            for k in ("failed", "queued", "running"):
                if d[k]: print(f"  {k}: {' '.join(d[k])}")
        if not a.wait or (counts["queued"] == 0 and counts["running"] == 0):
            if d["failed"]: print("  failed: " + " ".join(d["failed"]))
            sys.exit(0 if not d["failed"] else 1)
        time.sleep(a.poll)
main()
