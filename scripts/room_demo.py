"""The fly brain in a room: a table, fruit, colour vision, and a live view of what the connectome does.

    python scripts/room_demo.py                 # interactive pygame window
    python scripts/room_demo.py --gif out/room.gif --seconds 20 --headless
    python scripts/room_demo.py --brain-map     # + every soma projected, activity as highlights

Panels: fly's-eye camera (human colours) | scene view from an orbiting camera with the fly marked
        fly's-eye hex mosaic (fly false colour: UV=magenta, G=green, B=blue) + R1-R6 contrast
        brain activity: superclass rates, motor readout, spike count trace   [+ brain map column]
Brain: graded optic lobe (optic.py, 89k rate units) -> spiking LIF central brain + VNC (brain.py, 72k).
Keys: SPACE pause, R reset fly, T teleport next to fruit, L loom a black ball at the fly, F stimulate the
giant fibre (escape jump), W stimulate the flight DNs DNg02_a/DNa08 for 1 s (wingbeat), ESC quit.
Scene camera: arrow keys orbit, +/- (or mouse wheel over the view) zoom, mouse drag in the view orbits,
C follows the fly, HOME resets. The window is resizable; the UI keeps its layout and scales to fit.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flyverse import body, brain, brainmap, connectome, olfaction, optic, retina, world  # noqa: E402

FRAME_MS = 10.0          # brain time per frame (20 LIF steps at 0.5 ms)
W, H = 1280, 760         # design size of the UI; the window is resizable and the UI is scaled to fit
MAP_W = 330              # extra width of the brain-map column (--brain-map)
SCENE = (500, 10, 480, 300)   # scene-view rectangle on the canvas (x, y, w, h)


class Camera:
    def __init__(self, pos, look_at, width, height, fov_deg):
        self.pos = np.array(pos, float); f = np.array(look_at, float) - self.pos
        self.f = f / np.linalg.norm(f)
        r = np.cross(self.f, [0, 0, 1.0])
        if np.linalg.norm(r) < 1e-6:
            r = np.array([0.0, 1.0, 0.0])
        self.r = r / np.linalg.norm(r); self.u = np.cross(self.r, self.f)
        self.w, self.h, self.tan = width, height, np.tan(np.deg2rad(fov_deg) / 2)

    def render(self, wd: world.World):
        return wd.render_camera(self.pos, self.f, self.u, self.w, self.h, np.rad2deg(2 * np.arctan(self.tan)))

    def project(self, p):
        v = np.array(p, float) - self.pos
        z = v @ self.f
        if z <= 1e-6:
            return None
        x = (v @ self.r) / z / (self.tan * self.w / self.h); y = (v @ self.u) / z / self.tan
        return int((-x * 0.5 + 0.5) * self.w), int((-y * 0.5 + 0.5) * self.h)


class OrbitCam:
    """Scene camera orbiting a target: azimuth / elevation / distance, optionally following the fly."""

    def __init__(self, target):
        self.home = (np.array(target, float), -139.0, 28.0, 2.25)
        self.reset()

    def reset(self):
        self.target, self.az, self.el, self.dist = self.home[0].copy(), self.home[1], self.home[2], self.home[3]
        self.follow = False

    def orbit(self, daz, del_):
        self.az = (self.az + daz) % 360.0
        self.el = float(np.clip(self.el + del_, -5.0, 89.0))

    def zoom(self, factor):
        self.dist = float(np.clip(self.dist * factor, 0.05, 6.0))

    def key(self):
        return (tuple(np.round(self.target, 4)), round(self.az, 2), round(self.el, 2), round(self.dist, 4))

    def camera(self, fly=None) -> Camera:
        if self.follow and fly is not None:
            self.target = fly.eye_pos.copy()
        az, el = np.deg2rad(self.az), np.deg2rad(self.el)
        pos = self.target + self.dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        return Camera(pos, self.target, SCENE[2], SCENE[3], 60)


class Sim:
    def __init__(self, seed=0, brain_dt=0.5, optic_dt=1.0, cam_scale=1, start=None, trail_seconds=20.0):
        t0 = time.time()
        self.start = start                       # (x, y, z) or None = default spot on the table
        self.trail_seconds = trail_seconds
        self.trail = []                          # (brain time s, position) samples, for the scene view
        self.cam_scale = int(cam_scale)          # fly's-eye camera rendered at 1/cam_scale resolution, upscaled
        self.c = connectome.load(verbose=False)
        self.r = retina.build_retina(self.c)
        self.optic = optic.OpticLobe(self.c, self.r, optic.OpticParams(dt_ms=optic_dt))
        self.optic.relax()
        self.brain = brain.Brain(self.c, brain.LIFParams(dt=brain_dt), seed=seed)
        self.brain.freeze(self.optic.rate_idx)     # optic-lobe neurons are rate units, not LIF
        self.groups = body.motor_groups(self.c)
        self.loco = body.Locomotion()
        self.wings = body.wing_groups(self.c)
        self.flight = body.Flight()
        self.world, self.info = world.make_room(seed)
        self.world.spheres.append(world.Sphere((9, 9, 9), (0.03, 0.03, 0.03), "black"))   # looming ball (L key)
        self.loom_idx = len(self.world.spheres) - 1
        self.loom_t = -1.0
        self.dirs_b, self.wts = self.r.ray_directions()
        self.wts_t = torch.from_numpy(self.wts).float().to(self.world.device)
        # sweet taste: labellar sugar GRNs identified by connectivity to the known sweet second-order
        # neurons (scripts/find_sweet_grns.py -> flyverse/data/taste_grns.csv)
        taste = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "flyverse", "data", "taste_grns.csv"))
        self.sweet = self.c.index_of(taste.bodyId[taste.taste == "sweet"].to_numpy())
        # smell: every fruit is an odour source for the ORNs of its glomeruli (flyverse/olfaction.py)
        self.olf = olfaction.Olfaction(self.c, [(name, cen, 1.0) for name, cen, rad in self.info["fruit"]])
        self.reset_fly()
        self.superclasses = ["ol_intrinsic", "visual_projection", "cb_intrinsic", "descending_neuron",
                             "vnc_intrinsic", "vnc_motor", "cb_motor", "vnc_sensory"]
        self.sc_idx = {s: self.c.select(superclass=s) for s in self.superclasses}
        self.spike_hist = []
        self.t_wall = time.time()
        print(f"ready in {time.time() - t0:.1f}s: {self.c.n} neurons, {self.r.n_columns} columns, device {self.brain.device}")

    def reset_fly(self):
        x, y, z = self.start if self.start is not None else (-0.5, 0.05, self.info["table_top_z"])
        self.fly = body.FlyState(x=x, y=y, z=z, heading=np.deg2rad(5))
        self.optic.reset()
        self.tasting = 0.0
        self.trail = []

    def teleport_to_fruit(self):
        name, (x, y, z), rad = self.info["fruit"][0]
        self.fly.x, self.fly.y = x - rad - 0.005, y     # front legs touching the apple
        self.fly.heading = 0.0

    def column_radiance(self):
        d = self.fly.body_to_world(self.dirs_b.reshape(-1, 3))
        o = np.broadcast_to(self.fly.eye_pos, d.shape)
        rad = self.world.trace(torch.from_numpy(np.ascontiguousarray(o)).float(), torch.from_numpy(d).float())
        rad = rad.reshape(self.dirs_b.shape[0], self.dirs_b.shape[1], 4)
        return (rad * self.wts_t[None, :, None]).sum(1)

    def load_decoder(self, path):
        """Walking commands from a trained linear decoder over descending-neuron rates (train_decoder.py)."""
        d = np.load(path)
        self.dec_W, self.dec_b = d["W"], d["b"]
        self.dec_idx = torch.as_tensor(self.c.select(superclass="descending_neuron"), device=self.brain.device)
        self.decoder = True

    def decoder_cmd(self):
        obs = (self.brain.rate[0, self.dec_idx] / 10.0).clamp(0, 4).cpu().numpy()
        a = np.tanh(obs @ self.dec_W + self.dec_b)
        return {"speed": float(a[0] * 0.02), "yaw": float(a[1] * np.deg2rad(200)), "proboscis": self.cmd["proboscis"], "rates": self.cmd["rates"]}

    def surface_z(self, x, y):
        x0, x1, y0, y1 = self.info["table_extent"]
        return self.info["table_top_z"] if (x0 <= x <= x1 and y0 <= y <= y1) else 0.0

    def start_loom(self):
        self.loom_t = 0.0

    def update_loom(self):
        if self.loom_t < 0:
            return
        self.loom_t += FRAME_MS / 1000
        d = max(0.5 - 1.0 * self.loom_t, 0.035)
        eye = self.fly.eye_pos
        if self.loom_t > 0.8:
            self.loom_t = -1.0; self.world.move_sphere(self.loom_idx, (9, 9, 9))
        else:
            self.world.move_sphere(self.loom_idx, eye + np.array([0.0, d, 0.01]))

    def stimulate_wing_dns(self, ms=1000.0):
        """Drive the flight DNs the screen found (DNg02_a, DNa08: wingbeat) at 120 Hz for `ms`."""
        idx = self.c.select(type=["DNg02_a", "DNa08"])
        self.brain.set_poisson(idx, 120.0)
        self.wing_pulse = int(ms / FRAME_MS); self.wing_idx = idx

    def stimulate_gf(self):
        """Like a giant-fibre optogenetic pulse: 200 Hz for 30 ms."""
        self.brain.set_poisson(self.wings.gf, 200.0)
        self.gf_pulse = 3

    def nearest_fruit(self):
        best = (None, 1e9)
        for name, cen, rad in self.info["fruit"]:
            dxy = np.hypot(self.fly.x - cen[0], self.fly.y - cen[1]) - rad
            if dxy < best[1]:
                best = (name, dxy)
        return best

    def step(self):
        self.col_rad = self.column_radiance()
        self.brain.drive = self.optic.step_frame(self.col_rad, self.brain.rate, FRAME_MS)
        # taste: front legs touching fruit -> sweet GRNs fire (Poisson 120 Hz, Shiu-style)
        name, dist = self.nearest_fruit()
        self.tasting = 1.0 if (dist < 0.01 and not self.fly.airborne) else 0.0
        if getattr(self, "wing_pulse", 0) > 0:
            self.wing_pulse -= 1
            if self.wing_pulse == 0:
                self.brain.set_poisson(self.wing_idx, 0.0)
        if getattr(self, "gf_pulse", 0) > 0:
            self.gf_pulse -= 1
            if self.gf_pulse == 0:
                self.brain.set_poisson(self.wings.gf, 0.0)
        self.olf.apply(self.brain, self.fly.eye_pos)
        self.brain.set_poisson(self.sweet, 120.0 * self.tasting)
        self.brain.step(int(FRAME_MS / self.brain.p.dt))
        self.cmd = self.loco.readout(self.brain, self.groups)
        self.wcmd = self.flight.readout(self.brain, self.wings)
        x0, x1, y0, y1 = self.info["table_extent"]
        if self.fly.airborne:
            self.flight.step(self.fly, self.wcmd, FRAME_MS / 1000, self.surface_z, (-2, 2, -2, 2, 2.6))
        elif not self.flight.maybe_takeoff(self.fly, self.wcmd):
            on_table = abs(self.fly.z - self.info["table_top_z"]) < 1e-3
            bounds = (x0 + 0.02, x1 - 0.02, y0 + 0.02, y1 - 0.02) if on_table else (-1.95, 1.95, -1.95, 1.95)
            self.loco.step(self.fly, self.decoder_cmd() if getattr(self, "decoder", False) else self.cmd, FRAME_MS / 1000, bounds)
        self.update_loom()
        if self.trail_seconds > 0 and int(self.brain.t / FRAME_MS) % 5 == 0:      # 20 samples per second
            t_now = self.brain.t / 1000
            self.trail.append((t_now, self.fly.eye_pos.copy()))
            while self.trail and t_now - self.trail[0][0] > self.trail_seconds:
                self.trail.pop(0)
        self.spike_hist.append(self.brain.total_spikes())
        self.spike_hist = self.spike_hist[-400:]


# ---------------------------------------------------------------------------- drawing
def draw(sim: Sim, screen, font, orbit: OrbitCam, paused: bool, bmap=None):
    """Draw the whole UI onto `screen` (the fixed-size design canvas)."""
    import pygame
    screen.fill((18, 18, 22))
    fly = sim.fly
    # 1. fly's-eye pinhole camera (human colours), following the full body orientation (pitch/roll too)
    s = sim.cam_scale
    img = sim.world.render_camera(fly.eye_pos, fly.forward, fly.up, 480 // s, 300 // s, 110)
    surf = pygame.surfarray.make_surface(np.transpose(world.to_rgb8(img, exposure=2.5), (1, 0, 2)))
    if s > 1:
        surf = pygame.transform.scale(surf, (480, 300))
    screen.blit(surf, (10, 10))
    blit_text(screen, font, f"fly's-eye camera (110 deg)  pitch {np.rad2deg(fly.pitch):+.0f}  roll {np.rad2deg(fly.roll):+.0f}", 12, 312)
    # 2. scene view from the orbit camera (re-rendered when the camera moves or follows the fly)
    sx, sy, sw, sh = SCENE
    cam_over = orbit.camera(fly)
    key = orbit.key()
    if getattr(sim, "_over_key", None) != key:
        sim._over = pygame.surfarray.make_surface(np.transpose(world.to_rgb8(cam_over.render(sim.world), exposure=2.0), (1, 0, 2)))
        sim._over_key = key
    screen.blit(sim._over, (sx, sy))
    # time-decaying trail: bright yellow now -> dark with age
    if sim.trail:
        t_now = sim.brain.t / 1000
        prev = None
        for t_s, pos in sim.trail:
            q = cam_over.project(pos)
            if q is None or not (0 <= q[0] < sw and 0 <= q[1] < sh):
                prev = None; continue
            a = max(0.0, 1.0 - (t_now - t_s) / max(sim.trail_seconds, 1e-3))
            col = (int(40 + 215 * a), int(40 + 190 * a), int(20 + 40 * a))
            pt = (sx + q[0], sy + q[1])
            if prev is not None:
                pygame.draw.line(screen, col, prev, pt, 2)
            prev = pt
    p = cam_over.project(fly.eye_pos)
    if p and 0 <= p[0] < sw and 0 <= p[1] < sh:
        pygame.draw.circle(screen, (255, 255, 0), (sx + p[0], sy + p[1]), 5)
        q = cam_over.project(fly.eye_pos + 0.04 * fly.forward)
        if q:
            pygame.draw.line(screen, (255, 255, 0), (sx + p[0], sy + p[1]), (sx + q[0], sy + q[1]), 2)
    nf = sim.nearest_fruit()
    blit_text(screen, font, f"scene [{'follow' if orbit.follow else 'orbit'} az {orbit.az:.0f} el {orbit.el:.0f} d {orbit.dist:.2f}]  hdg {np.rad2deg(fly.heading) % 360:.0f}  "
              f"{fly.speed * 100:.1f} cm/s  {nf[0]} {nf[1] * 100:.0f} cm" + ("  TASTING" if sim.tasting else ""), sx + 2, 312)
    mode = f"AIRBORNE z={fly.z:.2f} v=({fly.vx:+.2f},{fly.vy:+.2f},{fly.vz:+.2f})" if fly.airborne else ("on table" if abs(fly.z - sim.info["table_top_z"]) < 1e-3 else "on floor")
    blit_text(screen, font, f"({fly.x:+.2f},{fly.y:+.2f}) {mode}  smell: " + sim.olf.summary(), sx + 2, 328)
    # 3. hex mosaics: fly false colour and drive
    fc = world.to_fly_false_color(sim.col_rad, exposure=2.5).astype(int)
    cf = sim.optic.last["contrast"][0][:, 0] if "contrast" in sim.optic.last else np.zeros(sim.r.n_columns)
    v = np.clip(128 + 127 * cf / 0.5, 0, 255).astype(int)
    if not hasattr(sim, "_hex_xy"):
        az, el = sim.r.col_az_el[:, 0], sim.r.col_az_el[:, 1]
        sim._hex_xy = list(zip((240 - az * 1.85).astype(int).tolist(), (355 + 130 - el * 1.6).astype(int).tolist()))
    for k, (title, ox) in enumerate([("what the fly's photoreceptors see (UV=magenta, G=green, B=blue)", 10),
                                     ("R1-R6 contrast: ON (white) / OFF (black)", 500)]):
        blit_text(screen, font, title, ox, 335)
        cols = fc.tolist() if k == 0 else [(c, c, c) for c in v.tolist()]
        for (x, y), col in zip(sim._hex_xy, cols):
            pygame.draw.circle(screen, col, (ox + x, y), 3)
    # 4. brain panel
    ox, oy = 990, 10
    blit_text(screen, font, f"brain t = {sim.brain.t / 1000:.2f} s   spikes/step {sim.spike_hist[-1] if sim.spike_hist else 0:.0f}", ox, oy)
    y = oy + 24
    for sc in sim.superclasses:
        rate = sim.brain.mean_rate(sim.sc_idx[sc])
        pygame.draw.rect(screen, (70, 120, 200), (ox + 120, y, int(min(rate, 50) * 3), 12))
        blit_text(screen, font, f"{sc[:18]:18s} {rate:5.1f} Hz", ox, y - 2); y += 16
    y += 8
    blit_text(screen, font, "motor readout (Hz)", ox, y); y += 18
    for k, val in sim.cmd["rates"].items():
        pygame.draw.rect(screen, (220, 120, 60), (ox + 90, y, int(min(val, 80) * 2), 12))
        blit_text(screen, font, f"{k:9s} {val:5.1f}", ox, y - 2); y += 16
    for k, val in sim.wcmd.items():
        pygame.draw.rect(screen, (120, 160, 240), (ox + 90, y, int(min(val, 80) * 2), 12))
        blit_text(screen, font, f"{k:9s} {val:5.1f}", ox, y - 2); y += 16
    y += 8
    blit_text(screen, font, f"speed cmd {sim.cmd['speed'] * 100:+.2f} cm/s  yaw {np.rad2deg(sim.cmd['yaw']):+.0f} deg/s  proboscis {sim.cmd['proboscis']:.2f}", ox, y); y += 22
    if len(sim.spike_hist) > 2:
        hh = np.array(sim.spike_hist); mx = max(hh.max(), 1)
        pts = [(ox + i * 280 / 400, y + 80 - 80 * val / mx) for i, val in enumerate(hh)]
        pygame.draw.lines(screen, (120, 220, 120), False, pts, 1)
        blit_text(screen, font, f"spikes/step (max {mx:.0f})", ox, y + 82)
    y += 105
    dt_wall = time.time() - sim.t_wall; sim.t_wall = time.time()
    blit_text(screen, font, f"{1 / max(dt_wall, 1e-3):.0f} fps  ({FRAME_MS / max(dt_wall, 1e-3) / 1000:.2f}x real time)" + ("   PAUSED" if paused else ""), ox, y)
    blit_text(screen, font, "SPACE pause  R reset  T apple  L loom  F giant fibre  W wing DNs  ESC", ox, y + 18)
    blit_text(screen, font, "arrows/drag orbit  +/- wheel zoom  C follow  HOME reset", ox, y + 34)
    # 5. brain map (optional column)
    if bmap is not None:
        mx0 = W + 12
        act = bmap.activity(sim.brain, sim.optic)
        dors, lat = bmap.render(act)
        blit_text(screen, font, "brain map: dorsal (brain left, VNC right)", mx0, 10)
        screen.blit(pygame.surfarray.make_surface(np.transpose(dors, (1, 0, 2))), (mx0, 28))
        blit_text(screen, font, "lateral (dorsal up)", mx0, 28 + bmap.Hd + 6)
        screen.blit(pygame.surfarray.make_surface(np.transpose(lat, (1, 0, 2))), (mx0, 28 + bmap.Hd + 24))
        yy = 28 + bmap.Hd + 24 + bmap.Hl + 10
        blit_text(screen, font, f"active neurons (>0.1): {int((act > 0.1).sum())}", mx0, yy); yy += 16
        blit_text(screen, font, "most active types:", mx0, yy); yy += 18
        for t, val in bmap.top_types(act).items():
            blit_text(screen, font, f"  {t:14s} {val:.2f}", mx0, yy); yy += 16


def blit_text(screen, font, text, x, y, color=(220, 220, 220)):
    screen.blit(font.render(text, True, color), (x, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0, help="stop after this much brain time (0 = run until quit)")
    ap.add_argument("--gif", type=str, default="", help="record a GIF to this path")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teleport", action="store_true", help="start next to the apple")
    ap.add_argument("--loom-at", type=float, default=-1, help="launch a loom at this brain time (s)")
    ap.add_argument("--decoder", type=str, default="", help="drive walking from a trained DN decoder (out/decoder.npz)")
    ap.add_argument("--wing-at", type=float, default=-1, help="stimulate the flight DNs (DNg02_a, DNa08) at this brain time (s)")
    ap.add_argument("--fast", action="store_true", help="speed preset for slower GPUs (Apple MPS): brain dt 1 ms, optic dt 2 ms, half-res camera")
    ap.add_argument("--brain-dt", type=float, default=None, help="LIF step (ms), default 0.5")
    ap.add_argument("--optic-dt", type=float, default=None, help="optic-lobe substep (ms), default 1 (the RL env uses 2)")
    ap.add_argument("--cam-scale", type=int, default=None, help="fly's-eye camera downscale factor, default 1")
    ap.add_argument("--brain-map", action="store_true", help="show every soma with activity highlights (extra column)")
    ap.add_argument("--window", type=str, default="", help="initial window size WxH (default: the design size)")
    ap.add_argument("--start", type=str, default="", help="start pose: 'x,y' (on the table), 'x,y,z', or 'floor'")
    ap.add_argument("--trail-seconds", type=float, default=20.0, help="how long the fly's trail persists in the scene view (0 = off)")
    args = ap.parse_args()
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame
    pygame.init()
    DW, DH = W + (MAP_W if args.brain_map else 0), H
    win = tuple(int(v) for v in args.window.lower().split("x")) if args.window else (DW, DH)
    screen = pygame.display.set_mode(win, pygame.RESIZABLE)
    canvas = pygame.Surface((DW, DH))
    pygame.display.set_caption("flyverse: MaleCNS fly brain in a room")
    font = pygame.font.SysFont("consolas", 14)
    fast = {"brain_dt": 1.0, "optic_dt": 2.0, "cam_scale": 2} if args.fast else {"brain_dt": 0.5, "optic_dt": 1.0, "cam_scale": 1}
    for k, v in (("brain_dt", args.brain_dt), ("optic_dt", args.optic_dt), ("cam_scale", args.cam_scale)):
        if v is not None:
            fast[k] = v
    start = None
    if args.start == "floor":
        start = (-1.2, 0.9, 0.0)
    elif args.start:
        v = [float(t) for t in args.start.split(",")]
        start = (v[0], v[1], v[2] if len(v) > 2 else 0.75)
    sim = Sim(args.seed, start=start, trail_seconds=args.trail_seconds, **fast)
    if args.decoder:
        sim.load_decoder(args.decoder)
    if args.teleport:
        sim.teleport_to_fruit()
    bmap = brainmap.BrainMap(sim.c, sim.optic) if args.brain_map else None
    orbit = OrbitCam((0.0, 0.0, sim.info["table_top_z"]))
    frames = []
    paused = False
    dragging = False
    clock = pygame.time.Clock()
    running = True

    def view_transform():
        """canvas -> window: uniform scale, centred (letterboxed)."""
        ww, wh = screen.get_size()
        sc = min(ww / DW, wh / DH)
        return sc, (ww - DW * sc) / 2, (wh - DH * sc) / 2

    def to_canvas(pos):
        sc, ox, oy = view_transform()
        return (pos[0] - ox) / sc, (pos[1] - oy) / sc

    def in_scene(pos):
        cx, cy = to_canvas(pos)
        return SCENE[0] <= cx < SCENE[0] + SCENE[2] and SCENE[1] <= cy < SCENE[1] + SCENE[3]

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((ev.w, ev.h), pygame.RESIZABLE)
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_r:
                    sim.reset_fly()
                elif ev.key == pygame.K_t:
                    sim.teleport_to_fruit()
                elif ev.key == pygame.K_l:
                    sim.start_loom()
                elif ev.key == pygame.K_f:
                    sim.stimulate_gf()
                elif ev.key == pygame.K_w:
                    sim.stimulate_wing_dns()
                elif ev.key == pygame.K_LEFT:
                    orbit.orbit(-10, 0)
                elif ev.key == pygame.K_RIGHT:
                    orbit.orbit(10, 0)
                elif ev.key == pygame.K_UP:
                    orbit.orbit(0, 5)
                elif ev.key == pygame.K_DOWN:
                    orbit.orbit(0, -5)
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    orbit.zoom(0.8)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    orbit.zoom(1.25)
                elif ev.key == pygame.K_c:
                    orbit.follow = not orbit.follow
                    if orbit.follow:
                        orbit.dist = min(orbit.dist, 0.4)
                elif ev.key == pygame.K_HOME:
                    orbit.reset()
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and in_scene(ev.pos):
                dragging = True
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging = False
            elif ev.type == pygame.MOUSEMOTION and dragging:
                sc, _, _ = view_transform()
                orbit.orbit(-ev.rel[0] / sc * 0.5, ev.rel[1] / sc * 0.4)
            elif ev.type == pygame.MOUSEWHEEL and in_scene(pygame.mouse.get_pos()):
                orbit.zoom(0.85 if ev.y > 0 else 1.18)
        if not paused:
            if args.loom_at >= 0 and sim.brain.t >= args.loom_at * 1000:
                sim.start_loom(); args.loom_at = -1
            if args.wing_at >= 0 and sim.brain.t >= args.wing_at * 1000:
                sim.stimulate_wing_dns(); args.wing_at = -1
            sim.step()
            if sim.fly.airborne and not getattr(sim, "_was_air", False):
                print(f"t={sim.brain.t / 1000:.2f}s TAKEOFF at ({sim.fly.x:+.2f},{sim.fly.y:+.2f}) GF {sim.wcmd['gf']:.0f} Hz TTMn {sim.wcmd['ttm']:.0f} power {sim.wcmd['power']:.0f} Hz")
            if not sim.fly.airborne and getattr(sim, "_was_air", False):
                print(f"t={sim.brain.t / 1000:.2f}s LANDED at ({sim.fly.x:+.2f},{sim.fly.y:+.2f},{sim.fly.z:.2f}) after {sim.fly.air_time:.2f}s")
            sim._was_air = sim.fly.airborne
        frame_no = sim.brain.step_count // int(FRAME_MS / sim.brain.p.dt)
        if frame_no % 4 == 0 or paused:
            draw(sim, canvas, font, orbit, paused, bmap)
            sc, ox, oy = view_transform()
            screen.fill((0, 0, 0))
            if abs(sc - 1.0) < 1e-6:
                screen.blit(canvas, (int(ox), int(oy)))
            else:
                screen.blit(pygame.transform.smoothscale(canvas, (int(DW * sc), int(DH * sc))), (int(ox), int(oy)))
            pygame.display.flip()
            if args.gif and frame_no % 8 == 0:
                frames.append(np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2)))
        if args.seconds and sim.brain.t >= args.seconds * 1000:
            running = False
        clock.tick(240)
    if args.gif and frames:
        import imageio
        imageio.mimsave(args.gif, frames, duration=0.08, loop=0)
        print("wrote", args.gif, len(frames), "frames")
    pygame.quit()


if __name__ == "__main__":
    main()
