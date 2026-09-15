"""Reproduce the demo media under docs/media/ (TODO.md section A, 'Demo media').

Three stages, each a subcommand; docs/media/README.md quotes the exact lines that produced every committed file.

    record   (GPU; the cluster or a local CUDA box)  one clip of the room console -> out/media/<clip>_raw.gif (the
             full 1360x820 canvas at 12.5 fps, exactly what `room_demo.py --headless --gif` records), <clip>_last.png,
             and <clip>_log.json (the fly's state at every recorded frame: pose, heading relative to the wind, giant-
             fibre / jump-muscle rates, distance to the nearest fruit, tasting / feeding; plus TAKEOFF / LANDED events)
    encode   (CPU, needs ffmpeg)  raw GIF -> docs/media/<clip>.mp4 (h264, full canvas) and docs/media/<clip>.gif
             (palette GIF, downscaled and at a lower frame rate so it stays under ~15 MB)
    figure   (CPU)  the toolkit figure docs/media/toolkit_loom_gf.png: the `paths` stage map of LC4 / LPLC2 -> DNp01
             (structural, runs here if out/media/paths_loom_gf.json is missing) next to the `atlas` readout of the
             giant fibre when those populations are stimulated (from the GPU runs of `cluster`)
    cluster  submit the GPU half (two clips + the atlas runs, three jobs) through scripts/cluster_run.py and fetch out/media/

The record loop is room_demo.py's headless loop (Sim, draw, the same frame cadence: a draw every 4th 10 ms frame and a
recorded frame every 8th = 80 ms) with a state log added; `CLIPS` names the room_demo.py command line each clip is
equivalent to. Nothing is staged: the loom is the demo's `--loom-at`, the wind and the fruit are the demo's defaults
or its documented flags, and the caption in docs/media/README.md reports what the log says happened.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)

RAW_DIR = "out/media"
DST_DIR = "docs/media"
FRAME_DT_S = 0.08                         # room_demo records every 8th 10 ms frame

# Every clip = the room demo at documented flags. `demo` is the equivalent room_demo.py line (the record loop
# reproduces it, adding the log); `note` is what the newcomer is meant to look at.
CLIPS = {
    "loom": dict(seconds=25.0, loom_at=8.0, seed=0, fruit="all", fence=False, wind_speed=0.3, wind_dir=180.0,
                 demo="python scripts/room_demo.py --headless --seconds 25 --loom-at 8 --seed 0 --gif out/media/loom_raw.gif",
                 note="the fly walking on the table; at 8 s a black ball looms from its left (the demo's L key / --loom-at)"),
    "wind_apple": dict(seconds=25.0, loom_at=-1.0, seed=0, fruit="apple", fence=False, wind_speed=0.3, wind_dir=180.0,
                       demo="python scripts/room_demo.py --headless --seconds 25 --fruit apple --seed 0 --gif out/media/wind_apple_raw.gif",
                       note="one apple on the table, the default 0.3 m/s wind from the door (+x); wind sense and the apple's plume are on"),
}

ATLAS_POPS = "LC4|LPLC2|LC6|LPLC1|DNp01"
ATLAS_PATTERN = "^(DNp01|DNp11|PVLP151|PVLP122|PVLP024)$"
PATHS_CMD = ["python", "scripts/interp_paths.py", "--a", "LC4|LPLC2", "--b", "DNp01", "--k", "2", "--top", "12",
             "--json", f"{RAW_DIR}/paths_loom_gf.json"]


def git_commit() -> str:
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        return h + (" (dirty)" if dirty else "")
    except Exception:
        return "unknown (no .git in this copy)"


def wrap_deg(a):
    return float((a + 180.0) % 360.0 - 180.0)


# ----------------------------------------------------------------------------------------------------- record
def fly_state(sim, rd) -> dict:
    fly = sim.fly
    wind_to = float(np.rad2deg(sim.air.direction))                  # the direction the wind blows towards
    heading = float(np.rad2deg(fly.heading))
    name, dist = sim.nearest_fruit()
    cmd, wcmd = getattr(sim, "cmd", {}), getattr(sim, "wcmd", {})
    return dict(t_s=round(sim.brain.t / 1000, 3), x=round(float(fly.x), 4), y=round(float(fly.y), 4), z=round(float(fly.z), 4),
                heading_deg=round(heading, 1), wind_from_deg=round((wind_to + 180.0) % 360.0, 1),
                upwind_rel_deg=round(wrap_deg(heading - (wind_to + 180.0)), 1),   # 0 = facing into the wind
                speed=round(float(getattr(fly, "speed", 0.0)), 4), airborne=bool(fly.airborne),
                gf_hz=round(float(wcmd.get("gf", 0.0)), 1), ttm_hz=round(float(wcmd.get("ttm", 0.0)), 1),
                power_hz=round(float(wcmd.get("power", 0.0)), 1), fruit=name, fruit_dist_m=round(float(dist), 4),
                tasting=bool(sim.tasting), feeding=bool(getattr(sim, "feeding", False)),
                walk_speed_cmd=round(float(cmd.get("speed", 0.0)), 4), yaw_cmd=round(float(cmd.get("yaw", 0.0)), 3),
                proboscis=round(float(cmd.get("proboscis", 0.0)), 3), loom_t=round(float(sim.loom_t), 2))


def cmd_record(args) -> int:
    spec = CLIPS[args.clip]
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import imageio
    import pygame
    import torch
    import room_demo as rd

    out = args.out
    os.makedirs(out, exist_ok=True)
    pygame.init()
    DW, DH = rd.canvas_size(rd.DEFAULT_SIZE)
    pygame.display.set_mode((DW, DH))
    canvas = pygame.Surface((DW, DH))
    ui = rd.RoomUI()
    ui.loading(canvas)
    sim = rd.Sim(spec["seed"], fruit_set=spec["fruit"], fence=spec["fence"], wind_speed=spec["wind_speed"], wind_dir=spec["wind_dir"])
    on_cuda = torch.device(sim.brain.device).type == "cuda"
    if not on_cuda and not args.allow_cpu:
        raise SystemExit("record: the brain is not on CUDA (pass --allow-cpu for a slow CPU render)")
    device = str(sim.brain.device)
    device_name = torch.cuda.get_device_name(0) if on_cuda else "cpu"
    orbit = rd.OrbitCam((0.0, 0.0, sim.info["table_top_z"]))
    sim._ui = ui
    ui.layout = rd.Layout(canvas.get_size())
    ui.mouse = (-1.0, -1.0)
    frames, log, events = [], [], []
    loom_at = spec["loom_at"]
    was_air = False
    seconds = args.seconds or spec["seconds"]
    t0 = time.time()
    while True:
        if loom_at >= 0 and sim.brain.t >= loom_at * 1000:
            sim.start_loom(); ui.notify("Loom stimulus presented"); loom_at = -1
            events.append(dict(t_s=round(sim.brain.t / 1000, 2), event="LOOM", text="black ball, 1 m/s from 0.5 m on the fly's left"))
        sim.step()
        if sim.fly.airborne and not was_air:
            e = dict(t_s=round(sim.brain.t / 1000, 2), event="TAKEOFF", x=round(float(sim.fly.x), 3), y=round(float(sim.fly.y), 3),
                     gf_hz=round(float(sim.wcmd["gf"]), 1), ttm_hz=round(float(sim.wcmd["ttm"]), 1), power_hz=round(float(sim.wcmd["power"]), 1))
            events.append(e); print("t=%(t_s).2fs TAKEOFF at (%(x)+.2f,%(y)+.2f) GF %(gf_hz).0f Hz TTMn %(ttm_hz).0f power %(power_hz).0f Hz" % e)
        if not sim.fly.airborne and was_air:
            e = dict(t_s=round(sim.brain.t / 1000, 2), event="LANDED", x=round(float(sim.fly.x), 3), y=round(float(sim.fly.y), 3),
                     z=round(float(sim.fly.z), 3), air_s=round(float(sim.fly.air_time), 2))
            events.append(e); print("t=%(t_s).2fs LANDED at (%(x)+.2f,%(y)+.2f,%(z).2f) after %(air_s).2fs" % e)
        was_air = sim.fly.airborne
        frame_no = sim.brain.step_count // int(rd.FRAME_MS / sim.brain.p.dt)
        if frame_no % 4 == 0:
            rd.draw(sim, canvas, None, orbit, False, None)
            if frame_no % 8 == 0:
                frames.append(np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2)))
                log.append(fly_state(sim, rd))
        if sim.brain.t >= seconds * 1000:
            break
    wall = time.time() - t0
    stem = os.path.join(out, args.clip)
    pygame.image.save(canvas, stem + "_last.png")
    imageio.mimsave(stem + "_raw.gif", frames, duration=FRAME_DT_S, loop=0)
    meta = dict(clip=args.clip, spec=spec, job=args.job, commit=git_commit(), device=device, device_name=device_name,
                torch=torch.__version__, host=os.uname().nodename if hasattr(os, "uname") else "", canvas=[DW, DH],
                frames=len(frames), frame_dt_s=FRAME_DT_S, brain_seconds=seconds, wall_seconds=round(wall, 1),
                n_neurons=int(sim.c.n), events=events, log=log)
    with open(stem + "_log.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {stem}_raw.gif ({len(frames)} frames), {stem}_last.png, {stem}_log.json; device {device} ({device_name}); "
          f"{seconds:.0f} s of brain time in {wall:.0f} s wall")
    pygame.quit()
    return 0


# ----------------------------------------------------------------------------------------------------- encode
def ffprobe_duration(path) -> float:
    o = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], text=True)
    return float(o.strip())


def cmd_encode(args) -> int:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("encode needs ffmpeg on PATH")
    os.makedirs(args.dst, exist_ok=True)
    clips = args.clip or list(CLIPS)
    for clip in clips:
        raw = os.path.join(args.out, f"{clip}_raw.gif")
        if not os.path.exists(raw):
            print(f"skip {clip}: {raw} missing"); continue
        mp4 = os.path.join(args.dst, f"{clip}.mp4")
        gif = os.path.join(args.dst, f"{clip}.gif")
        # imageio's Pillow GIF writer stores a 100 ms delay for the 0.08 s duration room_demo asks for (its --gif has the
        # same quirk: the raw GIF plays 25 s of brain time in 31 s), so every frame is re-stamped at the true 80 ms cadence
        fps = 1 / FRAME_DT_S
        retime = f"setpts=N/{fps:g}/TB"
        subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-i", raw,
                               "-vf", f"{retime},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p", "-r", f"{fps:g}",
                               "-c:v", "libx264", "-crf", str(args.crf), "-preset", "slow", "-movflags", "+faststart", mp4])
        vf = (f"{retime},fps={args.gif_fps:g},scale={args.gif_width}:-1:flags=lanczos,split[s0][s1];"
              f"[s0]palettegen=max_colors={args.gif_colors}:stats_mode=diff[p];"
              f"[s1][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle")
        subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-vf", vf, gif])
        for p in (mp4, gif):
            print(f"{p}: {os.path.getsize(p) / 1e6:.2f} MB, {ffprobe_duration(p):.1f} s")
    return 0


# ----------------------------------------------------------------------------------------------------- figure
def _rows(result, table):
    return result["tables"][table]


def cmd_figure(args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    if not os.path.exists(args.paths_json):
        print("running", " ".join(PATHS_CMD))
        subprocess.check_call(PATHS_CMD, cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    paths = json.load(open(args.paths_json))
    atlas_json = os.path.join(args.out, "atlas_loom.json")
    runs = sorted(set(os.path.splitext(p)[0] for p in glob.glob(args.atlas_runs + ".npz")))
    atlas = None
    if runs:
        if not os.path.exists(atlas_json) or args.refresh:
            subprocess.check_call(["python", "scripts/interp_atlas.py", "analyse", "--runs", args.atlas_runs, "--no-connectome",
                                   "--json", atlas_json], cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        atlas = json.load(open(atlas_json))
    else:
        print("no atlas runs matched", args.atlas_runs, "-- the figure gets the paths panel only")

    # colours: one accent per sign, neutral ink; readable on white (the README renders on both themes, so no dark bg)
    INK, MUTED, GRID = "#1f2430", "#6b7280", "#e5e7eb"
    POS, NEG, SRC, TGT = "#0f766e", "#b45309", "#1d4ed8", "#111827"
    fig, axes = plt.subplots(1, 2 if atlas else 1, figsize=(15 if atlas else 8.5, 6.4), dpi=130,
                             gridspec_kw=dict(width_ratios=[1.25, 1]) if atlas else None)
    axes = np.atleast_1d(axes)

    # --- left: the stage map of the k <= 2 signed walks LC4 / LPLC2 -> DNp01 (paths tool, type level)
    ax = axes[0]
    walks = [r for r in _rows(paths, "paths") if r["kind"] == "signed"]
    links = {(r["pre"], r["post"]): r["mv_per_post_volley"] for r in _rows(paths, "links")}
    relays = []
    for r in sorted((w for w in walks if w["k"] == 2), key=lambda w: -abs(w["gain"])):
        mid = r["path"].split(" -> ")[1]
        if mid not in relays:
            relays.append(mid)
    relays = relays[: args.max_intermediates]
    src = list(paths["summary"]["a_types"])
    n = max(len(relays), 2)
    X = {"src": 0.0, "relay": 1.0, "tgt": 2.0}
    ys = {"a:" + s: (n - 1) * (0.3 + 0.4 * i / max(1, len(src) - 1)) for i, s in enumerate(src)}
    ys.update({m: float(i) for i, m in enumerate(relays)})
    ys["b"] = (n - 1) / 2
    ax.set_xlim(-0.3, 2.45); ax.set_ylim(n - 0.3, -1.1); ax.axis("off")
    vmax = max(abs(v) for (pre, post), v in links.items() if pre.startswith("a:")) or 1.0

    def arrow(p0, p1, v, rad=0.0):
        col = POS if v >= 0 else NEG
        lw = 0.7 + 7.0 * min(1.0, abs(v) / vmax)
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=lw, color=col, alpha=0.85,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=1))
        return col

    for m in relays:                                       # source -> relay (fat), relay -> GF (thin: a few mV)
        ins = []
        for s_ in src:
            v = links.get(("a:" + s_, m))
            if v is not None and abs(v) >= 0.005 * vmax:
                arrow((X["src"] + 0.16, ys["a:" + s_]), (X["relay"] - 0.22, ys[m]), v)
                ins.append((s_, v))
        ins.sort(key=lambda t: -abs(t[1]))
        ax.text(X["relay"] - 0.25, ys[m] - 0.02, "   ".join(f"{s_} {v:+.0f}" for s_, v in ins), fontsize=6.4, color=INK,
                ha="right", va="bottom", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85), zorder=3)
        v_out = links.get((m, "b"), 0.0)
        col = arrow((X["relay"] + 0.22, ys[m]), (X["tgt"] - 0.24, ys["b"]), v_out)
        ax.text(X["relay"] + 0.25, ys[m] - 0.02, f"{v_out:+.1f}", fontsize=6.4, color=col, ha="left", va="bottom", zorder=3)
    direct = {}
    for i, s_ in enumerate(src):                           # the direct monosynaptic input, bowed around the relay column
        v = links.get(("a:" + s_, "b"), 0.0); direct[s_] = v
        rad = -0.42 if i == 0 else 0.42
        arrow((X["src"] + 0.16, ys["a:" + s_]), (X["tgt"] - 0.24, ys["b"]), v, rad=rad)
        ax.text(1.0, ys["a:" + s_] + (-0.95 if i == 0 else 0.95) * (n - 1) * 0.22, f"direct {s_} -> DNp01 {v:+.0f} mV",
                fontsize=7, color=POS if v >= 0 else NEG, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9), zorder=3)
    for node, x in [("a:" + s_, X["src"]) for s_ in src] + [(m, X["relay"]) for m in relays] + [("b", X["tgt"])]:
        label = node[2:] if node.startswith("a:") else ("DNp01" + chr(10) + "(giant fibre)" if node == "b" else node)
        fc = SRC if node.startswith("a:") else (TGT if node == "b" else "white")
        tc = "white" if fc != "white" else INK
        ax.text(x, ys[node], label, fontsize=8, ha="center", va="center", color=tc, weight="bold" if tc == "white" else None,
                bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec=INK if fc == "white" else fc, lw=0.8), zorder=4)
    ax.set_title("paths: signed gain LC4 / LPLC2 -> giant fibre, k <= 2 (type level, structural)", fontsize=10, color=INK, loc="left")
    ax.text(0.0, -0.01, f"numbers = mean input to one post cell per presynaptic-type volley (mV); teal +, amber -; width ~ |mV|." + chr(10) +
                      f"The top {len(relays)} two-step relays by |gain|; every relay -> DNp01 link is a few mV against "
                      f"the direct {direct.get('LC4', 0):+.0f} / {direct.get('LPLC2', 0):+.0f}.",
            fontsize=7, color=MUTED, ha="left", va="top", transform=ax.transAxes)

    # --- right: the atlas readout -- the giant fibre's rate when each population is stimulated, replicate scatter
    if atlas:
        ax = axes[1]
        rows = [r for r in _rows(atlas, "atlas") if r["readout"] == args.readout]
        rows.sort(key=lambda r: -r["stim_mean"])
        names = [r["population"] for r in rows]
        y = np.arange(len(rows))
        for i, r in enumerate(rows):
            vals = r.get("stim_values") or []
            ax.scatter(vals, [i] * len(vals), s=14, color=MUTED, zorder=3, alpha=0.9)
            col = GRID if r["verdict"] == "null" else (POS if r["diff"] > 0 else NEG)     # sign of the effect; 'null' = no effect
            ax.barh(i, r["stim_mean"], color=col, height=0.62, zorder=2, alpha=0.9 if col != GRID else 1.0)
            ax.text(max([r["stim_mean"], 0.0, *vals]) + 0.015 * max(1.0, rows[0]["stim_mean"]), i,
                    f"{r['stim_mean']:.1f} Hz  {r['verdict']}", va="center", fontsize=7, color=INK)
        null = rows[0]["null_mean"] if rows else 0.0
        ax.axvline(null, color=INK, lw=0.8, ls="--", zorder=1)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis(); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.set_xlabel(f"MotorRates '{args.readout}' = DNp01 mean rate over the pulse window (Hz), per stimulated population", fontsize=8.5, color=INK)
        prov = atlas["provenance"]["execution"]
        n_runs = atlas["summary"]["n_runs"]
        hz, ms = rows[0]["hz"], rows[0]["ms"]
        ax.set_title(f"atlas: population pulsed at {hz:.0f} Hz for {ms:.0f} ms, {n_runs} independent runs (dots), null dashed",
                     fontsize=10, color=INK, loc="left")
        ax.text(0, -0.16, f"device {prov.get('device')} ({prov.get('device_name', '')}); bar = mean over runs, dots = the runs; verdict = "
                          f"common.compare() against the unstimulated rows of the same batches (docs/INTERP.md 2.4): 'undetermined' = the null "
                          f"rows are deterministic (the giant fibre never fires unstimulated, SD 0), so z is undefined and the row is read by its "
                          f"diff and exact Mann-Whitney p. Model output; nothing tuned for the figure.",
                fontsize=7, color=MUTED, ha="left", va="top", transform=ax.transAxes, wrap=True)
    commit = paths["provenance"].get("flyverse_commit", {}).get("commit", "?")
    fig.suptitle("flyverse.interp on the shipped MaleCNS model: the loom -> giant-fibre path", fontsize=11, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, f"paths JSON commit {str(commit)[:7]}, compiled W md5 {paths['provenance']['compiled_connectome']['md5']}; "
                          f"generator scripts/make_demo_media.py figure", fontsize=6.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    fig.savefig(args.dst, facecolor="white")
    print("wrote", args.dst, f"{os.path.getsize(args.dst) / 1e6:.2f} MB")
    return 0


# ----------------------------------------------------------------------------------------------------- cluster
def cluster_commands() -> list[str]:
    gpu = "python -c 'import torch; assert torch.cuda.is_available()'"
    cmds = []
    for clip in CLIPS:
        cmds.append(f"mkdir -p {RAW_DIR} && export FAM=fam_media && {gpu} && python scripts/make_demo_media.py record --clip {clip} --job $FAM "
                    f"> {RAW_DIR}/{clip}_record.txt 2>&1; st=$?; tail -8 {RAW_DIR}/{clip}_record.txt; exit $st")
    cmds.append(f"mkdir -p {RAW_DIR} && export FAM=fam_media && {gpu} && for s in 0 1 2; do python scripts/interp_atlas.py run "
                f"--populations '{ATLAS_POPS}' --pattern '{ATLAS_PATTERN}' --seed $s --out {RAW_DIR}/atlas_loom_r$s "
                f"> {RAW_DIR}/atlas_loom_r$s.txt 2>&1 || exit 1; done && tail -3 {RAW_DIR}/atlas_loom_r2.txt")
    return cmds


def cmd_cluster(args) -> int:
    cmds = cluster_commands()
    call = ["python", "scripts/cluster_run.py", "--name", args.name, "--minutes", str(args.minutes), "--arm-block", "fam", *cmds,
            "--fetch", RAW_DIR + "/"]
    print(" \\\n  ".join(call))
    if args.dry_run:
        return 0
    return subprocess.call(call, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="GPU: one clip -> out/media/<clip>_raw.gif + _last.png + _log.json")
    r.add_argument("--clip", required=True, choices=list(CLIPS))
    r.add_argument("--seconds", type=float, default=0, help="override the clip's brain time (s)")
    r.add_argument("--out", default=RAW_DIR)
    r.add_argument("--job", default="", help="a label recorded in the log (the cluster batch passes fam_media)")
    r.add_argument("--allow-cpu", action="store_true")
    e = sub.add_parser("encode", help="CPU: raw GIF -> docs/media/<clip>.mp4 + .gif (ffmpeg)")
    e.add_argument("--clip", action="append", default=[], choices=list(CLIPS))
    e.add_argument("--out", default=RAW_DIR)
    e.add_argument("--dst", default=DST_DIR)
    e.add_argument("--crf", type=int, default=23)
    e.add_argument("--gif-width", type=int, default=800)
    e.add_argument("--gif-fps", type=float, default=10.0)
    e.add_argument("--gif-colors", type=int, default=128)
    f = sub.add_parser("figure", help="CPU: the toolkit figure (paths stage map + atlas readout)")
    f.add_argument("--paths-json", default=f"{RAW_DIR}/paths_loom_gf.json")
    f.add_argument("--atlas-runs", default=f"{RAW_DIR}/atlas_loom_r*")
    f.add_argument("--readout", default="gf")
    f.add_argument("--max-intermediates", type=int, default=8)
    f.add_argument("--out", default=RAW_DIR)
    f.add_argument("--dst", default=f"{DST_DIR}/toolkit_loom_gf.png")
    f.add_argument("--refresh", action="store_true", help="re-run the atlas analyse even if its JSON exists")
    c = sub.add_parser("cluster", help="submit the GPU half through scripts/cluster_run.py")
    c.add_argument("--name", default="media")
    c.add_argument("--minutes", type=int, default=30)
    c.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return {"record": cmd_record, "encode": cmd_encode, "figure": cmd_figure, "cluster": cmd_cluster}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
