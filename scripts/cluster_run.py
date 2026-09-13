"""Run flyverse commands on the GPU cluster(s) through the job manager, from this checkout.

    python scripts/cluster_run.py --name bench-fast "python scripts/benchmark.py --fast --json out/bench.json" --fetch out/bench.json
    python scripts/cluster_run.py --name sweep "python scripts/x.py --seed 0" "python scripts/x.py --seed 1" ... --fetch out/sweep/

Several commands in one call are a BATCH: one run directory per target, one job per command, all submitted at once and
run concurrently (each takes one GPU-share; the manager packs several onto a GPU), waited on together. Give
each command its own output file names. Prefer one batch call to a sequence of single calls.

For SWEEPS OF THE ROOM SIMULATION (seeds x programs), prefer the batched simulator to many single-fly jobs: one job
running `python scripts/batch_sustain.py --batch 16 --seeds ... --program ... --json out/x.json` steps 16 independent
rooms through one FlyBrain(batch=16) (see docs/BATCH_SIM.md; ~4x the aggregate throughput of 16 scalar processes, and
one process's worth of GPU memory). Use --cuda-sparse torch with batches (warp CSR is batch-1 only). Several batched
jobs in one call (one per program, say) are still a batch in the sense above.

What it does for each call: (1) collects the files that differ from origin/main here (unpushed commits, modified and
untracked files that are not git-ignored), (2) copies them over a fresh per-run copy of each selected target's
checkout (venv and connectome cache are shared by symlink), (3) submits each command as a job with one
GPU-share on the least-loaded target, (4) waits, prints the job log, exits with the jobs' exit codes, and (5) copies
back any `--fetch` paths (relative to the repo root) into the same paths here, merging the targets' results.

Cluster addresses, paths and the submitting user come from `.cluster.json` at the repo root (git-ignored)
or the FLYVERSE_CLUSTER environment variable (same JSON). Nothing infrastructural is hard-coded here.
One target (as before):

    {"ssh": "user@host", "api": "http://host:port/api/v1", "root": "/path/to/flyverse/checkout",
     "runs": "/path/for/per-run/copies", "user": "submitter", "env": {"PATH": "...", ...},
     "gpus": 1, "vram_gb": 24}

or several (rented single-box instances alongside the house cluster; keys outside "targets"/"default" are
inherited by every target). Extra per-target fields: "port" (ssh port, default 22), "tunnel" (reach the API
through an ssh local port-forward this script opens and closes itself; "api" is then the REMOTE url),
"slots" (concurrent jobs the box takes at this vram_gb; estimated from the API's node list if absent),
"disabled" (skip it):

    {"user": "submitter", "env": {...},
     "targets": {
       "house": {"ssh": "user@host", "api": "http://host:7000/api/v1", "root": "...", "runs": "...", "gpus": 1, "vram_gb": 24},
       "box-a": {"ssh": "root@1.2.3.4", "port": 41234, "tunnel": true, "api": "http://127.0.0.1:7000/api/v1",
                 "root": "/root/flyverse", "runs": "/root/runs", "gpus": 1, "vram_gb": 20, "slots": 4}},
     "default": ["house"]}

Options: --targets a,b,c / --target x, --minutes (estimate, default 30), --no-wait, --priority, --node, --tags,
--poll seconds, --sync-only, --allow-bare-fetch. Run directories are kept on each target (results and logs stay
there under the printed paths). See docs/CLUSTER.md section 14.
"""
from __future__ import annotations

import argparse
import atexit
import io
import json
import os
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.parse
import urllib.request
import uuid

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SINGLE = "cluster"            # target name given to a one-target (legacy) config
API_TIMEOUT = 30.0            # per HTTP call
PROBE_TIMEOUT = 20.0          # a target that does not answer within this (after its tunnel is up) sits this run out
RETRY_DELAY = 60.0            # one retry of a failed ssh / HTTP call, this long afterwards
HEADROOM_GB = 4.0             # VRAM left free per GPU when estimating slots from the node list

_FETCH_LOCK = threading.Lock()   # never two scp fetches into one local directory at once
_TUNNELS: list = []              # live ssh -L subprocesses, killed on exit


class ClusterError(RuntimeError):
    """An ssh / HTTP failure against one target; other targets may still be usable."""


# ── configuration ────────────────────────────────────────────────────────────

class Target:
    """One box: where to ssh, where its <scheduler> is, its checkout, and how many jobs it takes."""

    def __init__(self, name: str, cfg: dict):
        for k in ("ssh", "api", "root", "runs", "user"):
            if not cfg.get(k):
                sys.exit(f"cluster target '{name}' lacks '{k}'")
        self.name = name
        self.cfg = cfg
        self.host = cfg["ssh"]
        self.port = int(cfg.get("port", 22) or 22)
        self.root = str(cfg["root"]).rstrip("/")
        self.runs = str(cfg["runs"]).rstrip("/")
        self.user = cfg["user"]
        self.env = dict(cfg.get("env") or {})
        self.gpus = cfg.get("gpus", 1)
        self.vram_gb = cfg.get("vram_gb", 24)
        self.tunnel = bool(cfg.get("tunnel", False))
        self.remote_api = str(cfg["api"]).rstrip("/")
        self.api = self.remote_api                       # rewritten to 127.0.0.1:<local> when a tunnel is opened
        self.slots = cfg.get("slots")                    # None: estimated from the API's node list
        self.disabled = bool(cfg.get("disabled", False))
        self.headroom_gb = float(cfg.get("headroom_gb", HEADROOM_GB))
        self.proc = None                                 # the ssh -L subprocess, if any
        self.ok = True                                   # cleared when the box drops out of this run
        self.load = 0                                    # our pending/queued/running jobs, incl. what we submit now
        self.rdir = None                                 # per-run directory on this box
        self.njobs = 0
        self.nfailed = 0

    @property
    def free(self) -> int:
        return int(self.slots or 1) - self.load

    def __repr__(self) -> str:
        return f"<Target {self.name} {self.host}:{self.port}>"


def parse_config(cfg: dict) -> tuple[dict, list[str]]:
    """{targets: {...}, default: [...]} or one legacy target object -> ({name: Target}, default names)."""
    if not isinstance(cfg, dict):
        sys.exit("cluster config must be a JSON object")
    if "targets" in cfg:
        raw = cfg["targets"]
        if not isinstance(raw, dict) or not raw:
            sys.exit("cluster config: 'targets' must be a non-empty object of name -> target")
        shared = {k: v for k, v in cfg.items() if k not in ("targets", "default")}
        targets = {}
        for name, t in raw.items():
            if not isinstance(t, dict):
                sys.exit(f"cluster target '{name}' must be an object")
            merged = {**shared, **t}
            if shared.get("env") and t.get("env"):       # env merges key-wise, the target wins
                merged["env"] = {**shared["env"], **t["env"]}
            targets[name] = Target(name, merged)
        default = cfg.get("default") or [n for n, t in targets.items() if not t.disabled]
        if isinstance(default, str):
            default = [default]
        return targets, list(default)
    return {SINGLE: Target(SINGLE, cfg)}, [SINGLE]


def load_config() -> tuple[dict, list[str]]:
    raw = os.environ.get("FLYVERSE_CLUSTER")
    if not raw:
        p = os.path.join(ROOT, ".cluster.json")
        if not os.path.exists(p):
            sys.exit("no .cluster.json at the repo root and FLYVERSE_CLUSTER unset; see docs/CLUSTER.md")
        raw = open(p, encoding="utf-8").read()
    return parse_config(json.loads(raw))


def select_targets(targets: dict, default: list[str], want: list[str] | None) -> list:
    """--targets/--target names, else the config's "default" list, else every enabled target."""
    if want:
        out = []
        for n in want:
            if n not in targets:
                sys.exit(f"unknown target '{n}'; configured: {', '.join(targets) or '(none)'}")
            if targets[n].disabled:
                sys.exit(f"target '{n}' is disabled in the cluster config")
            if targets[n] not in out:
                out.append(targets[n])
        return out
    out = [targets[n] for n in default if n in targets and not targets[n].disabled]
    if not out:
        sys.exit("no enabled targets in the cluster config")
    return out


# ── ssh, tunnels, HTTP ───────────────────────────────────────────────────────

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


def ssh(t: Target, cmd: str, check: bool = True, input_bytes: bytes | None = None, retry: bool = True) -> str:
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-p", str(t.port), t.host, cmd]
    r = subprocess.run(argv, capture_output=True, input=input_bytes, check=False)
    if r.returncode != 0 and retry:
        print(f"  ({t.name}: ssh failed ({r.returncode}), one retry in {RETRY_DELAY:.0f}s)")
        time.sleep(RETRY_DELAY)
        r = subprocess.run(argv, capture_output=True, input=input_bytes, check=False)
    if check and r.returncode != 0:
        raise ClusterError(f"ssh to {t.name} failed ({r.returncode}): {cmd}\n{r.stderr.decode(errors='replace')}")
    return r.stdout.decode(errors="replace")


def scp(t: Target, src: str, dst: str, retry: bool = True) -> int:
    argv = ["scp", "-q", "-r", "-P", str(t.port), src, dst]
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode != 0 and retry:
        print(f"  ({t.name}: scp failed ({r.returncode}), one retry in {RETRY_DELAY:.0f}s)")
        time.sleep(RETRY_DELAY)
        r = subprocess.run(argv, capture_output=True, text=True)
    return r.returncode


def free_local_port() -> int:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def url_port(url: str) -> int:
    u = urllib.parse.urlsplit(url)
    return u.port or (443 if u.scheme == "https" else 80)


def local_api_url(remote: str, local_port: int) -> str:
    u = urllib.parse.urlsplit(remote)
    return urllib.parse.urlunsplit((u.scheme, f"127.0.0.1:{local_port}", u.path, "", "")).rstrip("/")


def tunnel_cmd(t: Target, local_port: int) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-N",
            "-L", f"{local_port}:127.0.0.1:{url_port(t.remote_api)}", "-p", str(t.port), t.host]


def open_tunnel(t: Target) -> str:
    """ssh -L <free local port>:127.0.0.1:<remote api port>, for the duration of this run."""
    port = free_local_port()
    t.proc = subprocess.Popen(tunnel_cmd(t, port), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    t.api = local_api_url(t.remote_api, port)
    _TUNNELS.append(t)
    print(f"[{t.name}] tunnel 127.0.0.1:{port} -> {t.host}:{t.port} -> {t.remote_api}")
    return t.api


def close_tunnels() -> None:
    while _TUNNELS:
        t = _TUNNELS.pop()
        p, t.proc = t.proc, None
        if p is None or p.poll() is not None:
            continue
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


atexit.register(close_tunnels)


def api(t: Target, path: str, body: dict | None = None, method: str | None = None,
        retry: bool = True, timeout: float = API_TIMEOUT) -> dict:
    def once() -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(t.api.rstrip("/") + path, data=data,
                                     method=method or ("POST" if data else "GET"),
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    try:
        return once()
    except Exception as e:
        if not retry:
            raise
        print(f"  ({t.name}: API {path} failed: {e}; one retry in {RETRY_DELAY:.0f}s)")
        time.sleep(RETRY_DELAY)
        return once()


def wait_for_api(t: Target, timeout: float | None = None) -> str | None:
    """None when the target's API answers within `timeout`, else the reason it did not."""
    timeout = PROBE_TIMEOUT if timeout is None else timeout
    t0 = time.time()
    last = "no answer"
    while True:
        if t.proc is not None and t.proc.poll() is not None:
            err = (t.proc.stderr.read().decode(errors="replace").strip() if t.proc.stderr else "")
            return f"ssh tunnel exited ({t.proc.returncode}) {err}".strip()
        try:
            api(t, "/status", retry=False, timeout=min(5.0, timeout))
            return None
        except Exception as e:
            last = str(e)
        if time.time() - t0 >= timeout:
            return last
        time.sleep(1.0)


# ── capacity and scheduling ──────────────────────────────────────────────────

def estimate_slots(t: Target) -> int:
    """floor((gpu memory - headroom) / vram_gb) * gpus, from the API's node list."""
    nodes = api(t, "/nodes", retry=False)
    states = list(nodes.values()) if isinstance(nodes, dict) else list(nodes or [])
    gpus = []
    for st in states:
        if not isinstance(st, dict) or st.get("reachable") is False or st.get("fence_reason"):
            continue
        gpus.extend(g for g in (st.get("gpus") or []) if isinstance(g, dict))
    if not gpus:
        raise ClusterError("node list has no GPUs")
    mem_gb = min(float(g.get("memory_total_mb") or 0) for g in gpus) / 1024.0
    if not t.vram_gb:
        return max(1, len(gpus))
    return max(1, int((mem_gb - t.headroom_gb) // float(t.vram_gb)) * len(gpus))


def current_load(t: Target) -> int:
    """How many of our jobs the box already has pending / queued / running."""
    n = 0
    for status in ("pending", "queued", "running"):
        r = api(t, f"/jobs?status={status}&limit=200", retry=False)
        for j in (r.get("jobs") or []):
            if j.get("submitted_by") == t.user:
                n += 1
    return n


def pick_target(targets: list) -> Target:
    """The available target with the most free slots; ties go to the earlier target."""
    live = [t for t in targets if t.ok]
    if not live:
        raise ClusterError("no target available")
    return max(live, key=lambda t: t.free)


# ── run steps ────────────────────────────────────────────────────────────────

def bare_fetch_paths(paths: list[str]) -> list[str]:
    return [p for p in paths if p.strip().strip("/\\").lower() in ("", ".", "out")]


BARE_FETCH_WHY = (
    "refusing --fetch {paths}: the bare run-output directory is not safe to copy back. A scratch cache_* "
    "directory under a run's out/ (152 MB, interp round) made scp -r hang and left two threads reporting "
    "finished batches as pending. Fetch a named subdirectory (--fetch out/<name>/) or a file, or pass "
    "--allow-bare-fetch if you really mean the whole directory."
)


def make_tarball(files: list[str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for f in files:
            tf.add(os.path.join(ROOT, f), arcname=f)
    return buf.getvalue()


def ship(t: Target, run: str, tarball: bytes | None) -> None:
    """Fresh copy of the target's checkout (venv + cache shared by symlink), local changes overlaid."""
    t.rdir = f"{t.runs}/{run}"
    ssh(t, f"set -e; cd {t.root} && git pull -q --ff-only || true; mkdir -p {t.rdir}; "
           f"rsync -a --exclude .venv --exclude cache --exclude out --exclude logs --exclude .torch_ext --exclude .git {t.root}/ {t.rdir}/; "
           f"ln -sfn {t.root}/.venv {t.rdir}/.venv; ln -sfn {t.root}/cache {t.rdir}/cache; mkdir -p {t.rdir}/out {t.rdir}/logs")
    if tarball:
        ssh(t, f"cd {t.rdir} && tar xzf -", input_bytes=tarball)


def ship_all(targets: list, run: str, tarball: bytes | None, files: list[str]) -> None:
    """Ship to every selected target in parallel; a target that fails to take the diff sits this run out."""
    errs: dict = {}

    def one(t: Target) -> None:
        try:
            ship(t, run, tarball)
        except Exception as e:                                         # noqa: BLE001 - any failure drops the box
            errs[t.name] = str(e)
            t.ok = False

    threads = [threading.Thread(target=one, args=(t,), daemon=True) for t in targets]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    shown = ", ".join(files[:8]) + (", ..." if len(files) > 8 else "")
    for t in targets:
        if t.name in errs:
            print(f"[{t.name}] SHIP FAILED, skipping this target: {errs[t.name].splitlines()[0]}")
        else:
            print(f"[{t.name}] run dir {t.rdir}: {len(files)} local file(s) shipped" + (f" ({shown})" if files else ""))


def build_spec(t: Target, jname: str, command: str, args) -> dict:
    env = {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
           "TORCH_EXTENSIONS_DIR": f"{t.root}/.torch_ext", **t.env}
    spec = {"job_type": "benchmark", "name": jname, "gpus": t.gpus, "vram_gb": t.vram_gb,
            "estimated_minutes": args.minutes, "priority": args.priority, "tags": args.tags.split(","),
            "working_dir": t.rdir, "log_path": f"{t.rdir}/logs/{jname}.log", "env": env,
            "command": f"source .venv/bin/activate && {command}"}
    if args.node:
        spec["node"] = args.node
    return spec


def submit_all(targets: list, commands: list[str], run: str, args) -> tuple[list, int]:
    """One job per command, least-loaded target first; returns the job records and the submit failures."""
    jobs, lost = [], 0
    for i, command in enumerate(commands):
        jname = run if len(commands) == 1 else f"{run}-{i}"
        tried: list = []
        while True:
            live = [t for t in targets if t.ok and t not in tried]
            if not live:
                print(f"SUBMIT FAILED {jname}: no target took it: {command}")
                lost += 1
                break
            t = pick_target(live)
            try:
                r = api(t, "/jobs", {"spec": build_spec(t, jname, command, args), "submitted_by": t.user})
                job = r["job"]
            except Exception as e:                                     # noqa: BLE001 - try the next box
                print(f"[{t.name}] submit failed ({e}); marking unavailable for this run")
                t.ok = False
                tried.append(t)
                continue
            t.load += 1
            t.njobs += 1
            jobs.append({"t": t, "id": job["id"], "name": jname, "command": command, "job": job, "last": None})
            print(f"job {job['id']} {job['status']}  @{t.name}  {jname}: {command}"
                  + (f"  warnings: {r['warnings']}" if r.get("warnings") else ""))
            break
    return jobs, lost


def fetch_all(targets: list, paths: list[str]) -> None:
    """Copy each --fetch path back from every target that ran something, merged into one local path.

    Serialised on purpose: two scp -r into the same local directory at once produced internally
    inconsistent files before (docs/CLUSTER.md section 13).
    """
    for p in paths:
        dst = os.path.join(ROOT, p)
        os.makedirs(os.path.dirname(dst.rstrip("/\\")) or ".", exist_ok=True)
        for t in targets:
            if not t.njobs or not t.rdir:
                continue
            src = f"{t.host}:{t.rdir}/{p.rstrip('/')}/." if p.endswith("/") else f"{t.host}:{t.rdir}/{p}"
            with _FETCH_LOCK:
                rc = scp(t, src, dst)
            print(f"fetched {p} @{t.name}" if rc == 0 else f"FETCH FAILED {p} @{t.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("commands", nargs="+", help="one or more shell commands; several = a concurrent batch")
    ap.add_argument("--name", required=True, help="job name; a short unique suffix is appended")
    ap.add_argument("--minutes", type=int, default=30)
    ap.add_argument("--priority", type=int, default=40)
    ap.add_argument("--node", default=None)
    ap.add_argument("--tags", default="flyverse")
    ap.add_argument("--fetch", nargs="*", action="extend", default=[], help="paths (relative to the repo root) to copy back when the jobs end; may be repeated; a NAMED directory (out/<name>/) is the robust form")
    ap.add_argument("--allow-bare-fetch", action="store_true", help="permit --fetch out/ (see the refusal message)")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--poll", type=float, default=20.0)
    ap.add_argument("--sync-only", action="store_true", help="prepare the run directories and print them; submit nothing")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--targets", default=None, help="comma-separated target names (default: the config's \"default\" list, else every enabled target)")
    g.add_argument("--target", default=None, help="a single target name")
    args = ap.parse_args()

    bare = bare_fetch_paths(args.fetch)
    if bare and not args.allow_bare_fetch:
        sys.exit(BARE_FETCH_WHY.format(paths=" ".join(bare)))

    targets, default = load_config()
    want = [n.strip() for n in (args.targets or args.target or "").split(",") if n.strip()] or None
    sel = select_targets(targets, default, want)
    run = f"{args.name}-{uuid.uuid4().hex[:6]}"
    t0 = time.time()

    # 1. tunnels + availability: a box whose API stays silent sits this run out
    for t in sel:
        if t.tunnel:
            open_tunnel(t)
    if not args.sync_only:
        for t in sel:
            why = wait_for_api(t)
            if why:
                print(f"[{t.name}] UNAVAILABLE (API silent for {PROBE_TIMEOUT:.0f}s): {why}")
                t.ok = False
        sel_ok = [t for t in sel if t.ok]
        if not sel_ok:
            print("no target answered; nothing submitted")
            return 2
        # slots: configured, or estimated from the node list; plus what of ours the box already holds
        for t in sel_ok:
            if t.slots:
                print(f"[{t.name}] slots {t.slots} (configured)")
            else:
                try:
                    t.slots = estimate_slots(t)
                    print(f"[{t.name}] slots {t.slots} (estimated at vram_gb {t.vram_gb}, headroom {t.headroom_gb:.0f} GB)")
                except Exception as e:                                 # noqa: BLE001
                    t.slots = max(1, int(t.gpus or 1))
                    print(f"[{t.name}] slots {t.slots} (fallback; node list unavailable: {e})")
            try:
                t.load = current_load(t)
                if t.load:
                    print(f"[{t.name}] {t.load} job(s) of {t.user} already pending/queued/running")
            except Exception as e:                                     # noqa: BLE001
                print(f"[{t.name}] job list unavailable ({e}); assuming idle")
                t.load = 0

    # 2. ship the local diff to every selected target, in parallel
    files = ship_list()
    tarball = make_tarball(files) if files else None
    ship_all([t for t in sel if t.ok], run, tarball, files)
    sel_ok = [t for t in sel if t.ok]
    if not sel_ok:
        print("no target took the diff; nothing submitted")
        return 2
    if args.sync_only:
        return 0

    # 3. submit least-loaded-first
    jobs, lost = submit_all(sel_ok, args.commands, run, args)
    used = [t for t in sel if t.njobs]
    if args.no_wait:
        for t in used:
            u = t.remote_api if t.tunnel else t.api                     # the local port dies with this process
            print(f"[{t.name}] poll: {u}/jobs/<id>   log: {u}/jobs/<id>/logs?all=true"
                  + (f"   (through ssh -p {t.port} {t.host})" if t.tunnel else ""))
        return 1 if lost else 0

    # 4. wait for all of them, on every target
    terminal = ("completed", "failed", "cancelled", "killed", "timeout")
    while jobs:
        for j in jobs:
            if j["job"]["status"] in terminal:
                continue
            try:
                j["job"] = api(j["t"], f"/jobs/{j['id']}", retry=False)
            except Exception as e:                                     # VPN blips: keep waiting
                print(f"  (poll {j['t'].name} failed: {e})")
                continue
            if j["job"]["status"] != j["last"]:
                print(f"  [{time.time() - t0:6.0f}s] {j['name']} @{j['t'].name} {j['job']['status']}")
                j["last"] = j["job"]["status"]
        if all(j["job"]["status"] in terminal for j in jobs):
            break
        time.sleep(args.poll)

    failed = lost
    for j in jobs:
        t = j["t"]
        try:
            log = api(t, f"/jobs/{j['id']}/logs?all=true", retry=False).get("log") or ""
        except Exception as e:                                         # noqa: BLE001
            log = f"(log fetch failed: {e})"
        code = j["job"].get("exit_code")
        print(f"---- {j['name']} @{t.name} ({j['id']}) {j['job']['status']} exit {code}: {j['command']} ----")
        print("\n".join(l for l in log.splitlines() if not l.startswith("ALSA lib")))
        if not (j["job"]["status"] == "completed" and code in (0, None)):
            failed += 1
            t.nfailed += 1
    print("---- end logs ----")

    # 5. fetch results from every target that ran one of our jobs, merged into the same local paths
    fetch_all(used, args.fetch)
    for t in used:
        print(f"  @{t.name}: {t.njobs} job(s), {t.nfailed} failed  run dir {t.rdir}")
    where = used[0].rdir if len(used) == 1 else f"{run} on {len(used)} target(s)"
    print(f"{len(jobs)} job(s), {failed} failed  ({(time.time() - t0) / 60:.1f} min)  run dir {where}")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ClusterError as e:
        sys.exit(str(e))
    except KeyboardInterrupt:
        close_tunnels()
        sys.exit(130)
    finally:
        close_tunnels()
