"""Run a flyverse command on the GPU cluster through the job manager, from this checkout.

    python scripts/cluster_run.py --name bench-fast "python scripts/benchmark.py --fast --json out/bench.json" --fetch out/bench.json
    python scripts/cluster_run.py --name sweep "python scripts/x.py --seed 0" "python scripts/x.py --seed 1" ... --fetch out/

Several commands in one call are a BATCH: one run directory, one job per command, all submitted at once and
run concurrently (each takes one GPU-share; the manager packs several onto a GPU), waited on together. Give
each command its own output file names. Prefer one batch call to a sequence of single calls.

For SWEEPS OF THE ROOM SIMULATION (seeds x programs), prefer the batched simulator to many single-fly jobs: one job
running `python scripts/batch_sustain.py --batch 16 --seeds ... --program ... --json out/x.json` steps 16 independent
rooms through one FlyBrain(batch=16) (see docs/BATCH_SIM.md; ~4x the aggregate throughput of 16 scalar processes, and
one process's worth of GPU memory). Use --cuda-sparse torch with batches (warp CSR is batch-1 only). Several batched
jobs in one call (one per program, say) are still a batch in the sense above.

What it does for each call: (1) collects the files that differ from origin/main here (unpushed commits, modified and
untracked files that are not git-ignored), (2) copies them over a fresh per-run copy of the cluster's
checkout (venv and connectome cache are shared by symlink), (3) submits the command as a job with one
GPU-share, (4) waits, prints the job log, exits with the job's exit code, and (5) copies back any
`--fetch` paths (relative to the repo root) into the same paths here.

Cluster addresses, paths and the submitting user come from `.cluster.json` at the repo root (git-ignored)
or the FLYVERSE_CLUSTER environment variable (same JSON). Nothing infrastructural is hard-coded here.

    {"ssh": "user@host", "api": "http://host:port/api/v1", "root": "/path/to/flyverse/checkout",
     "runs": "/path/for/per-run/copies", "user": "submitter", "env": {"PATH": "...", ...},
     "gpus": 1, "vram_gb": 24}

Options: --minutes (estimate, default 30), --no-wait, --priority, --node, --tags, --poll seconds, --sync-only.
Run directories are kept on the cluster (results and logs stay there under the printed path).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
import uuid

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_config() -> dict:
    raw = os.environ.get("FLYVERSE_CLUSTER")
    if not raw:
        p = os.path.join(ROOT, ".cluster.json")
        if not os.path.exists(p):
            sys.exit("no .cluster.json at the repo root and FLYVERSE_CLUSTER unset; see docs/CLUSTER.md")
        raw = open(p, encoding="utf-8").read()
    cfg = json.loads(raw)
    for k in ("ssh", "api", "root", "runs", "user"):
        if k not in cfg:
            sys.exit(f"cluster config lacks '{k}'")
    cfg.setdefault("env", {}); cfg.setdefault("gpus", 1); cfg.setdefault("vram_gb", 24)
    return cfg


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def ship_list() -> list[str]:
    files = set()
    try:
        files |= set(git("diff", "--name-only", "origin/main", "HEAD").split())
    except subprocess.CalledProcessError:
        pass                                                           # no origin/main: ship the working tree diff only
    files |= set(git("ls-files", "-m", "-o", "--exclude-standard").split())
    return sorted(f for f in files if os.path.isfile(os.path.join(ROOT, f)))


def ssh(cfg: dict, cmd: str, check: bool = True, input_bytes: bytes | None = None) -> str:
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", cfg["ssh"], cmd],
                       capture_output=True, input=input_bytes, check=False)
    if check and r.returncode != 0:
        sys.exit(f"ssh failed ({r.returncode}): {cmd}\n{r.stderr.decode(errors='replace')}")
    return r.stdout.decode(errors="replace")


def api(cfg: dict, path: str, body: dict | None = None, method: str | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(cfg["api"].rstrip("/") + path, data=data, method=method or ("POST" if data else "GET"),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("commands", nargs="+", help="one or more shell commands; several = a concurrent batch")
    ap.add_argument("--name", required=True, help="job name; a short unique suffix is appended")
    ap.add_argument("--minutes", type=int, default=30)
    ap.add_argument("--priority", type=int, default=40)
    ap.add_argument("--node", default=None)
    ap.add_argument("--tags", default="flyverse")
    ap.add_argument("--fetch", nargs="*", action="extend", default=[], help="paths (relative to the repo root) to copy back when the job ends; may be repeated; a directory (out/) is the robust form")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--poll", type=float, default=20.0)
    ap.add_argument("--sync-only", action="store_true", help="prepare the run directory and print it; submit nothing")
    args = ap.parse_args()
    cfg = load_config()
    run = f"{args.name}-{uuid.uuid4().hex[:6]}"
    rdir = f"{cfg['runs'].rstrip('/')}/{run}"
    root = cfg["root"].rstrip("/")

    # 1. fresh copy of the cluster checkout (pulled to origin/main), venv + cache shared
    ssh(cfg, f"set -e; cd {root} && git pull -q --ff-only || true; mkdir -p {rdir}; "
             f"rsync -a --exclude .venv --exclude cache --exclude out --exclude logs --exclude .torch_ext --exclude .git {root}/ {rdir}/; "
             f"ln -sfn {root}/.venv {rdir}/.venv; ln -sfn {root}/cache {rdir}/cache; mkdir -p {rdir}/out {rdir}/logs")
    # 2. overlay local changes
    files = ship_list()
    if files:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for f in files:
                tf.add(os.path.join(ROOT, f), arcname=f)
        ssh(cfg, f"cd {rdir} && tar xzf -", input_bytes=buf.getvalue())
    print(f"run dir {rdir}: {len(files)} local file(s) shipped" + (f" ({', '.join(files[:8])}{', ...' if len(files) > 8 else ''})" if files else ""))
    if args.sync_only:
        return 0
    # 3. submit one job per command
    env = {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
           "TORCH_EXTENSIONS_DIR": f"{root}/.torch_ext", **cfg["env"]}
    jobs = {}
    for i, command in enumerate(args.commands):
        jname = run if len(args.commands) == 1 else f"{run}-{i}"
        spec = {"job_type": "benchmark", "name": jname, "gpus": cfg["gpus"], "vram_gb": cfg["vram_gb"], "estimated_minutes": args.minutes,
                "priority": args.priority, "tags": args.tags.split(","), "working_dir": rdir, "log_path": f"{rdir}/logs/{jname}.log",
                "env": env, "command": f"source .venv/bin/activate && {command}"}
        if args.node:
            spec["node"] = args.node
        r = api(cfg, "/jobs", {"spec": spec, "submitted_by": cfg["user"]})
        job = r["job"]; jobs[job["id"]] = {"name": jname, "command": command, "job": job, "last": None}
        print(f"job {job['id']} {job['status']}  {jname}: {command}" + (f"  warnings: {r['warnings']}" if r.get("warnings") else ""))
    if args.no_wait:
        print(f"poll: {cfg['api']}/jobs/<id>   log: {cfg['api']}/jobs/<id>/logs?all=true")
        return 0
    # 4. wait for all of them
    terminal = ("completed", "failed", "cancelled", "killed", "timeout"); t0 = time.time()
    while True:
        for jid, j in jobs.items():
            if j["job"]["status"] in terminal:
                continue
            try:
                j["job"] = api(cfg, f"/jobs/{jid}")
            except Exception as e:                                      # VPN blips: keep waiting
                print(f"  (poll failed: {e})"); continue
            if j["job"]["status"] != j["last"]:
                print(f"  [{time.time() - t0:6.0f}s] {j['name']} {j['job']['status']}"); j["last"] = j["job"]["status"]
        if all(j["job"]["status"] in terminal for j in jobs.values()):
            break
        time.sleep(args.poll)
    failed = 0
    for jid, j in jobs.items():
        try:
            log = api(cfg, f"/jobs/{jid}/logs?all=true").get("log") or ""
        except Exception as e:
            log = f"(log fetch failed: {e})"
        code = j["job"].get("exit_code")
        print(f"---- {j['name']} ({jid}) {j['job']['status']} exit {code}: {j['command']} ----")
        print("\n".join(l for l in log.splitlines() if not l.startswith("ALSA lib")))
        if not (j["job"]["status"] == "completed" and code in (0, None)):
            failed += 1
    print("---- end logs ----")
    # 5. fetch results
    for p in args.fetch:
        dst = os.path.join(ROOT, p); os.makedirs(os.path.dirname(dst.rstrip("/\\")) or ".", exist_ok=True)
        rc = subprocess.run(["scp", "-q", "-r", f"{cfg['ssh']}:{rdir}/{p}", dst], capture_output=True, text=True).returncode
        print(f"fetched {p}" if rc == 0 else f"FETCH FAILED {p}")
    print(f"{len(jobs)} job(s), {failed} failed  ({(time.time() - t0) / 60:.1f} min)  run dir {rdir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
