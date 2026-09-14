"""Pull a finished run directory's out/<name> from a rented box into local out/<name>/ (only files not already local).
usage: python fetch_run.py --target r3-h200a --run mono-fd76d2 --out monoamines [--src-sub monoamines] [--skip-npz] [--only-npz]
  --src-sub   remote subdir under /root/runs/<run>/out/ (default: same as --out; use e.g. fetch/slim for a slimmed copy)
  --skip-npz  pull everything except *.npz (do this first for big object batches; slim on the box, then fetch the slim dir)
Run from the repo root. Existing local files are never overwritten (collision = skipped, listed)."""
import argparse, json, os, subprocess, sys
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--src-sub", default=None); ap.add_argument("--skip-npz", action="store_true"); ap.add_argument("--only-npz", action="store_true")
    a = ap.parse_args()
    t = json.load(open(".cluster.json"))["targets"][a.target]
    src = f"{t['runs']}/{a.run}/out/{a.src_sub or a.out}"
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "-o", "StrictHostKeyChecking=no", "-p", str(t["port"]), t["ssh"]]
    r = subprocess.run(ssh + [f"cd {src} && find . -type f | sed 's#^[.]/##'"], capture_output=True, text=True, timeout=300)
    if r.returncode != 0: print("remote listing failed:", r.stderr.strip()[-300:]); sys.exit(2)
    there = sorted(l.strip().replace("\r", "") for l in r.stdout.splitlines() if l.strip())
    local = f"out/{a.out}"; os.makedirs(local, exist_ok=True)
    have = {os.path.relpath(os.path.join(dp, f), local).replace("\\", "/") for dp, _, fs in os.walk(local) for f in fs}
    want = [f for f in there if f not in have]
    if a.skip_npz: want = [f for f in want if not f.endswith(".npz")]
    if a.only_npz: want = [f for f in want if f.endswith(".npz")]
    skipped = [f for f in there if f in have]
    print(f"{a.target}:{src} -> {local}: {len(there)} remote, {len(skipped)} already local (skipped), {len(want)} to pull", flush=True)
    if not want: sys.exit(0)
    lst = "\n".join(want) + "\n"
    with open(os.path.join(local, ".fetch_list.txt"), "w", encoding="utf-8", newline="\n") as f: f.write(lst)
    cmd = ssh + [f"cd {src} && tar cf - -T -"]
    with open(os.path.join(local, ".fetch_list.txt"), "rb") as fin:
        p1 = subprocess.Popen(cmd, stdin=fin, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p2 = subprocess.Popen(["tar", "xf", "-", "-C", local], stdin=p1.stdout, stderr=subprocess.PIPE)
        p1.stdout.close(); e2 = p2.communicate()[1]; p1.wait(); e1 = p1.stderr.read()
    os.remove(os.path.join(local, ".fetch_list.txt"))
    now = {os.path.relpath(os.path.join(dp, f), local).replace("\\", "/") for dp, _, fs in os.walk(local) for f in fs}
    missing = [f for f in want if f not in now]
    print(f"pulled {len(want) - len(missing)} / {len(want)}; tar rc {p1.returncode}/{p2.returncode}" + (f"; MISSING {len(missing)}: {missing[:5]}" if missing else ""), (e1 or e2).decode(errors="replace")[-300:])
    sys.exit(0 if not missing and p1.returncode == 0 and p2.returncode == 0 else 1)
main()
