"""The fly brain in a room: a table, fruit, colour vision, and a live view of what the connectome does.

    python scripts/room_demo.py                 # interactive pygame window
    python scripts/room_demo.py --gif out/room.gif --seconds 20 --headless

Panels: fly's-eye camera (human colours) | overview of the table with the fly marked
        fly's-eye hex mosaic (fly false colour: UV=magenta, G=green, B=blue) + R1-R6 contrast
        brain activity: superclass rates, motor readout, spike count trace
Brain: graded optic lobe (optic.py, 89k rate units) -> spiking LIF central brain + VNC (brain.py, 72k).
Keys: SPACE pause, R reset fly, T teleport next to fruit, L loom a black ball at the fly, F stimulate the
giant fibre (escape jump), W stimulate the flight DNs DNg02_a/DNa08 for 1 s (wingbeat), ESC quit.
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
from flyverse import body, brain, connectome, olfaction, optic, retina, world  # noqa: E402

FRAME_MS = 10.0          # brain time per frame (20 LIF steps at 0.5 ms)
W, H = 1280, 760


class Camera:
    def __init__(self, pos, look_at, width, height, fov_deg):
        self.pos = np.array(pos, float); f = np.array(look_at, float) - self.pos
        self.f = f / np.linalg.norm(f)
        r = np.cross(self.f, [0, 0, 1.0]); self.r = r / np.linalg.norm(r); self.u = np.cross(self.r, self.f)
        self.w, self.h, self.tan = width, height, np.tan(np.deg2rad(fov_deg) / 2)

    def render(self, wd: world.World):
        return wd.render_camera(self.pos, self.f, [0, 0, 1], self.w, self.h, np.rad2deg(2 * np.arctan(self.tan)))

    def project(self, p):
        v = np.array(p, float) - self.pos
        z = v @ self.f
        if z <= 1e-6:
            return None
        x = (v @ self.r) / z / (self.tan * self.w / self.h); y = (v @ self.u) / z / self.tan
        return int((-x * 0.5 + 0.5) * self.w), int((-y * 0.5 + 0.5) * self.h)


class Sim:
    def __init__(self, seed=0, brain_dt=0.5, optic_dt=1.0, cam_scale=1):
        t0 = time.time()
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
        self.fly = body.FlyState(x=-0.5, y=0.05, z=self.info["table_top_z"], heading=np.deg2rad(5))
        self.optic.reset()
        self.tasting = 0.0

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
        self.loom_from = self.fly.eye_pos + np.array([0.0, 0.5, 0.01])

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
        self.spike_hist.append(self.brain.total_spikes())
        self.spike_hist = self.spike_hist[-400:]


# ---------------------------------------------------------------------------- drawing
def draw(sim: Sim, screen, font, cam_over: Camera, paused: bool):
    import pygame
    screen.fill((18, 18, 22))
    fly = sim.fly
    # 1. fly's-eye pinhole camera (human colours)
    s = sim.cam_scale
    img = sim.world.render_camera(fly.eye_pos, fly.forward, [0, 0, 1], 480 // s, 300 // s, 110)
    surf = pygame.surfarray.make_surface(np.transpose(world.to_rgb8(img, exposure=2.5), (1, 0, 2)))
    if s > 1:
        surf = pygame.transform.scale(surf, (480, 300))
    screen.blit(surf, (10, 10))
    blit_text(screen, font, "fly's-eye camera (human colours, 110 deg)", 12, 312)
    # 2. overview
    if not hasattr(sim, "_over"):
        sim._over = pygame.surfarray.make_surface(np.transpose(world.to_rgb8(cam_over.render(sim.world), exposure=2.0), (1, 0, 2)))
    screen.blit(sim._over, (500, 10))
    p = cam_over.project(fly.eye_pos)
    if p:
        pygame.draw.circle(screen, (255, 255, 0), (500 + p[0], 10 + p[1]), 5)
        q = cam_over.project(fly.eye_pos + 0.04 * fly.forward)
        if q:
            pygame.draw.line(screen, (255, 255, 0), (500 + p[0], 10 + p[1]), (500 + q[0], 10 + q[1]), 2)
    nf = sim.nearest_fruit()
    blit_text(screen, font, f"overview  fly ({fly.x:+.2f},{fly.y:+.2f}) m  hdg {np.rad2deg(fly.heading) % 360:.0f} deg  "
              f"{fly.speed * 100:.1f} cm/s  {nf[0]} {nf[1] * 100:.0f} cm" + ("  TASTING" if sim.tasting else ""), 502, 312)
    mode = f"AIRBORNE z={fly.z:.2f} m v=({fly.vx:+.2f},{fly.vy:+.2f},{fly.vz:+.2f})" if fly.airborne else ("walking on table" if abs(fly.z - sim.info["table_top_z"]) < 1e-3 else "walking on floor")
    blit_text(screen, font, mode + "   smell: " + sim.olf.summary(), 502, 328)
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
    for s in sim.superclasses:
        rate = sim.brain.mean_rate(sim.sc_idx[s])
        pygame.draw.rect(screen, (70, 120, 200), (ox + 120, y, int(min(rate, 50) * 3), 12))
        blit_text(screen, font, f"{s[:18]:18s} {rate:5.1f} Hz", ox, y - 2); y += 16
    y += 8
    blit_text(screen, font, "motor readout (Hz)", ox, y); y += 18
    for k, v in sim.cmd["rates"].items():
        pygame.draw.rect(screen, (220, 120, 60), (ox + 90, y, int(min(v, 80) * 2), 12))
        blit_text(screen, font, f"{k:9s} {v:5.1f}", ox, y - 2); y += 16
    for k, v in sim.wcmd.items():
        pygame.draw.rect(screen, (120, 160, 240), (ox + 90, y, int(min(v, 80) * 2), 12))
        blit_text(screen, font, f"{k:9s} {v:5.1f}", ox, y - 2); y += 16
    y += 8
    blit_text(screen, font, f"speed cmd {sim.cmd['speed'] * 100:+.2f} cm/s  yaw {np.rad2deg(sim.cmd['yaw']):+.0f} deg/s  proboscis {sim.cmd['proboscis']:.2f}", ox, y); y += 22
    if len(sim.spike_hist) > 2:
        hh = np.array(sim.spike_hist); mx = max(hh.max(), 1)
        pts = [(ox + i * 280 / 400, y + 80 - 80 * v / mx) for i, v in enumerate(hh)]
        pygame.draw.lines(screen, (120, 220, 120), False, pts, 1)
        blit_text(screen, font, f"spikes/step (max {mx:.0f})", ox, y + 82)
    y += 105
    dt_wall = time.time() - sim.t_wall; sim.t_wall = time.time()
    blit_text(screen, font, f"{1 / max(dt_wall, 1e-3):.0f} fps  ({FRAME_MS / max(dt_wall, 1e-3) / 1000:.2f}x real time)" + ("   PAUSED" if paused else ""), ox, y)
    blit_text(screen, font, "SPACE pause  R reset  T apple  L loom  F giant fibre  W wing DNs  ESC", ox, y + 18)


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
    args = ap.parse_args()
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("flyverse: MaleCNS fly brain in a room")
    font = pygame.font.SysFont("consolas", 14)
    fast = {"brain_dt": 1.0, "optic_dt": 2.0, "cam_scale": 2} if args.fast else {"brain_dt": 0.5, "optic_dt": 1.0, "cam_scale": 1}
    for k, v in (("brain_dt", args.brain_dt), ("optic_dt", args.optic_dt), ("cam_scale", args.cam_scale)):
        if v is not None:
            fast[k] = v
    sim = Sim(args.seed, **fast)
    if args.decoder:
        sim.load_decoder(args.decoder)
    if args.teleport:
        sim.teleport_to_fruit()
    cam_over = Camera((-1.5, -1.3, 1.8), (0.0, 0.0, sim.info["table_top_z"]), 480, 300, 60)
    frames = []
    paused = False
    clock = pygame.time.Clock()
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
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
            draw(sim, screen, font, cam_over, paused)
            pygame.display.flip()
            if args.gif and frame_no % 8 == 0:
                frames.append(np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2)))
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
