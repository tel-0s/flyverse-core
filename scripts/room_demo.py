"""The fly brain in a room: a table, fruit, colour vision, and a live view of what the connectome does.

    python scripts/room_demo.py                 # interactive pygame window
    python scripts/room_demo.py --gif out/room.gif --seconds 20 --headless
    python scripts/room_demo.py --brain-map     # open the soma activity atlas

Console: body camera, small orbit view, both retinal mosaics, spike history and antennal samples.
         Live telemetry shows populations, motors and body state together; long readouts scroll.
Brain: graded optic lobe (optic.py, 89k rate units) -> spiking LIF central brain + VNC (brain.py, 72k).
Keys: SPACE pause, R reset fly, T teleport next to fruit, L loom a black ball at the fly, F stimulate the
giant fibre (escape jump), W stimulate the flight DNs DNg02_a/DNa08 for 1 s (wingbeat), ESC quit.
Scene camera: arrow keys orbit, +/- (or mouse wheel over the view) zoom, mouse drag in the view orbits,
C follows the fly, HOME resets. The console reflows when resized; ? opens the key bindings.
State: F5 / F9 quick-save / quick-load (out/quicksave.pt), S timestamped save, --load file to resume.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from flyverse import air, body, brain, brainmap, optic, programs, surfaces, world  # noqa: E402
from flyverse.fly import FlyBrain  # noqa: E402
from flyverse.room_ui import RoomUI, Layout, DEFAULT_SIZE, canvas_size, fit_transform  # noqa: E402

FRAME_MS = 10.0          # brain time per frame (20 LIF steps at 0.5 ms)
W, H = DEFAULT_SIZE


class Camera:
    def __init__(self, pos, look_at, width, height, fov_deg):
        self.pos = np.array(pos, float); f = np.array(look_at, float) - self.pos
        self.f = f / np.linalg.norm(f)
        r = np.cross(self.f, [0, 0, 1.0])
        if np.linalg.norm(r) < 1e-6:
            r = np.array([0.0, 1.0, 0.0])
        self.r = r / np.linalg.norm(r); self.u = np.cross(self.r, self.f)
        self.w, self.h, self.tan = width, height, np.tan(np.deg2rad(fov_deg) / 2)

    def render(self, wd: world.World, scale=1):
        return wd.render_camera(self.pos, self.f, self.u, max(1,self.w//scale), max(1,self.h//scale),
                                np.rad2deg(2 * np.arctan(self.tan)))

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
        self.home = (np.array(target, float), -126.0, 43.0, 1.25)
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

    def camera(self, fly=None, width=480, height=300) -> Camera:
        if self.follow and fly is not None:
            self.target = fly.eye_pos.copy()
        az, el = np.deg2rad(self.az), np.deg2rad(self.el)
        pos = self.target + self.dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        return Camera(pos, self.target, width, height, 60)


class Sim:
    def __init__(self, seed=0, brain_dt=0.5, optic_dt=1.0, cam_scale=1, start=None, trail_seconds=20.0,
                 wind_speed=0.3, wind_dir=180.0, cuda_graphs=False, weight_dtype="float32",
                 sensory_cuda_graphs=None, program="none", escape_gating=False, dt_by_module=None, prune_frozen=True,
                 fruit_set="all", fence=False,
                 cuda_kernels=None, event_driven=None, cuda_sparse="torch", cuda_compact=True, receptor_model="default",
                 preset="raw", instruments=None, gf_threshold=None):
        if gf_threshold is not None and (not np.isfinite(gf_threshold) or gf_threshold < 0):
            raise ValueError("gf_threshold must be finite and nonnegative (Hz)")
        t0 = time.time()
        self.start = start                       # (x, y, z) or None = default spot on the table
        self.trail_seconds = trail_seconds
        self.trail = []                          # (brain time s, position) samples, for the scene view
        self.cam_scale = int(cam_scale)          # fly's-eye camera rendered at 1/cam_scale resolution, upscaled
        self.sensory_cuda_graphs = cuda_graphs if sensory_cuda_graphs is None else sensory_cuda_graphs
        lif_kw = dict(dt=brain_dt, weight_dtype=weight_dtype, dt_by_module=dt_by_module, prune_frozen=prune_frozen, event_driven=event_driven)
        if receptor_model != "default":          # 'off' = the presynaptic-sign rule; otherwise a receptor model name (docs/NT_INTEGRATION.md)
            lif_kw["receptor_model"] = None if receptor_model == "off" else receptor_model
        self.fb = FlyBrain(seed=seed, lif_params=brain.LIFParams(**lif_kw),
                           optic_params=optic.OpticParams(dt_ms=optic_dt), cuda_graphs=cuda_graphs, cuda_kernels=cuda_kernels,
                           cuda_sparse=cuda_sparse, cuda_compact=cuda_compact, preset=preset, instruments=instruments)
        self.c, self.r, self.optic, self.brain = self.fb.c, self.fb.retina, self.fb.optic, self.fb.brain
        if self.brain.cuda and self.optic is not None:
            self.optic.diagnostics = False  # the UI samples contrast only when drawing
        self.groups, self.wings = self.fb.groups, self.fb.wings
        self.loco = body.Locomotion()
        self.flight = body.Flight()
        if gf_threshold is not None:
            self.flight.gf_hz = float(gf_threshold)
        self.metabolism = body.Metabolism()
        # behaviour programs (hand-designed stand-ins, off by default): flyverse/programs.py
        self.program = programs.make_program(program)
        self.gating = programs.EscapeGating() if escape_gating else None
        self.world, self.info = world.make_room(seed, fruit_set)
        self.surfaces = surfaces.Surfaces(self.info["room"], self.info["solids"], self.info["solid_labels"])
        # --fence: an invisible glass box around the table top (walls the fly can walk on, no visual change):
        # holds the fly on the table so foraging can be scored independently of the escape-hop problem
        self.fence = fence                     # a test fixture: the fly cannot leave the table top by walking or hopping
        self.world.spheres.append(world.Sphere((9, 9, 9), (0.03, 0.03, 0.03), "black"))   # looming ball (L key)
        self.loom_idx = len(self.world.spheres) - 1
        self.loom_t = -1.0
        self.dirs_b, self.wts = self.r.ray_directions()
        self.wts_t = torch.from_numpy(self.wts).float().to(self.world.device)
        # air: wind + a plume from every fruit; smelled bilaterally by the antennae, wind felt by the
        # Johnston's organ (flyverse/air.py)
        # odour emission scales with the fruit's size (surface): strength 1 for a 2 cm radius, so the apple
        # (4 cm) emits 2x, a blueberry (6 mm) 0.3x, the banana (9 cm) 4.5x. One value per source, not per fruit type.
        self.air = air.Air([(name, cen, rad / 0.02, rad) for name, cen, rad in self.info["fruit"]],
                           air.WindParams(speed=wind_speed, direction_deg=wind_dir), seed=seed)
        self.smell_values = ({}, {})
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
        self.fly.ground_time = -3.0                      # a jump cut is a loom to the model: no escape for 3 s (UI convenience)

    def column_radiance(self):
        d = self.fly.body_to_world(self.dirs_b.reshape(-1, 3))
        o = np.broadcast_to(self.fly.eye_pos, d.shape)
        rad = self.world.trace(torch.from_numpy(np.ascontiguousarray(o)).float(), torch.from_numpy(d).float(),
                               cuda_graphs=self.sensory_cuda_graphs)
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
        return self.surfaces.support(x, y)

    def start_loom(self, speed=1.0, radius=0.03, final=0.035, start=0.5):
        """A black ball approaches from the fly's left at `speed` m/s from `start` m to `final` m (l/v = radius / speed:
        30 ms at the defaults, the fast loom; slower approaches probe the long-mode escape and the landing response)."""
        self.loom_t = 0.0; self.loom_speed = speed; self.loom_final = final; self.loom_start = start
        self.world.move_sphere(self.loom_idx, (9, 9, 9), (radius, radius, radius))

    def update_loom(self):
        if self.loom_t < 0:
            return
        self.loom_t += FRAME_MS / 1000
        speed = getattr(self, "loom_speed", 1.0); final = getattr(self, "loom_final", 0.035); start = getattr(self, "loom_start", 0.5)
        d = max(start - speed * self.loom_t, final)
        eye = self.fly.eye_pos
        if self.loom_t > (start - final) / speed + 0.3:
            self.loom_t = -1.0; self.world.move_sphere(self.loom_idx, (9, 9, 9))
        else:
            self.world.move_sphere(self.loom_idx, eye + d * np.asarray(self.fly.left) + np.array([0.0, 0.0, 0.01]))   # from the fly's left, whatever its heading

    def stimulate_wing_dns(self, ms=1000.0):
        """Drive the flight DNs the screen found (DNg02_a, DNa08: wingbeat) at 120 Hz for `ms`."""
        self.fb.stimulate({"type": ["DNg02_a", "DNa08"]}, 120.0, ms)

    def stimulate_gf(self):
        """Like a giant-fibre optogenetic pulse: 200 Hz for 30 ms."""
        self.fb.stimulate({"type": "DNp01"}, 200.0, 30.0)

    # ------------------------------------------------------------------ save / load
    BRAIN_TENSORS = ("v", "g", "refrac", "drive", "poisson_p", "rate", "spikes", "adapt", "res", "spike_buf")
    OPTIC_TENSORS = ("v", "adapt", "I_lp", "I_mean", "_fresh")

    def save_state(self, path):
        """Everything needed to resume exactly: LIF + optic-lobe state, RNG, fly pose, loom, trail."""
        import dataclasses
        b, o = self.brain, self.optic
        state = {
            "controller": self.fb.state_dict(),
            "locomotion": dict(vars(self.loco)),
            "program": self.program.state() if self.program else None, "gating": self.gating.state() if self.gating else None,
            "flight": vars(self.flight).copy(),
            "cmd": self.cmd, "wcmd": self.wcmd, "feeding": getattr(self, "feeding", False),
            "air_phase": self.air.phase.copy(),
            "fly": dataclasses.asdict(self.fly),
            "loom_t": self.loom_t, "loom_center": self.world.spheres[self.loom_idx].center,
            "trail": self.trail, "spike_hist": self.spike_hist, "tasting": self.tasting,
            "pulses": {"wing_pulse": getattr(self, "wing_pulse", 0), "gf_pulse": getattr(self, "gf_pulse", 0)},
            "flight_hold": getattr(self.flight, "_power_hold", 0.0),
            "air_t": self.air.t,
            "metabolism": dataclasses.asdict(self.metabolism),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(state, path)
        print(f"saved state at t={b.t / 1000:.2f}s -> {path}")

    def load_state(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        b, o = self.brain, self.optic
        if "controller" in state:
            self.fb.load_state_dict(state["controller"])
            self.loco.__dict__.update(state["locomotion"])
            self.flight.__dict__.update(state["flight"])
            self.air.phase = state["air_phase"].copy()
        else:
            for k in self.BRAIN_TENSORS:
                getattr(b, k).copy_(state["brain"][k].to(b.device))
            for k, v in state["brain_scalars"].items():
                setattr(b, k, v)
            b._rate_np_key = None
            b.gen.set_state(state["rng"])
            for k in self.OPTIC_TENSORS:
                getattr(o, k).copy_(state["optic"][k].to(o.device))
            self.fb._base_poisson.copy_(b.poisson_p)
            self.fb._inputs_on = {"legacy": b._poisson_on}
        self.fly = body.FlyState(**state["fly"])
        self.loom_t = state["loom_t"]; self.world.move_sphere(self.loom_idx, state["loom_center"])
        self.trail = state["trail"]; self.spike_hist = state["spike_hist"]; self.tasting = state["tasting"]
        self.wing_pulse = state["pulses"]["wing_pulse"]; self.gf_pulse = state["pulses"]["gf_pulse"]
        if self.wing_pulse:
            self.wing_idx = self.c.select(type=["DNg02_a", "DNa08"])
        if "controller" not in state:
            # Old saves counted frames before applying the pulse, hence N-1 remaining frames.
            for count, selection, hz in ((self.wing_pulse, {"type": ["DNg02_a", "DNa08"]}, 120.0),
                                         (self.gf_pulse, {"type": "DNp01"}, 200.0)):
                if count:
                    idx = self.c.select(**selection)
                    self.fb._base_poisson[:, b._idx(idx)] = 0.0
                    b.set_poisson(idx, 0.0)
                    if count > 1:
                        self.fb.stimulate(selection, hz, (count - 1) * FRAME_MS)
        self.flight._power_hold = state["flight_hold"]
        self.air.t = state.get("air_t", 0.0)
        if "metabolism" in state:
            self.metabolism = body.Metabolism(**state["metabolism"])
        if self.program and state.get("program"):
            self.program.load_state(state["program"])
        if self.gating and state.get("gating"):
            self.gating.load_state(state["gating"])
        self.col_rad = self.column_radiance()
        if "cmd" in state:
            self.cmd, self.wcmd = state["cmd"], state["wcmd"]
            self.feeding = state["feeding"]
        else:
            self.cmd = self.loco.readout(self.fb.motor()); self.wcmd = self.flight.readout(self.fb.motor())
        print(f"loaded state at t={b.t / 1000:.2f}s <- {path}")

    def nearest_fruit(self):
        best = (None, 1e9)
        for name, cen, rad in self.info["fruit"]:
            dxy = float(np.linalg.norm(self.fly.pos - np.asarray(cen))) - rad     # 3-D: a fly on the floor under a berry is 75 cm from it
            if dxy < best[1]:
                best = (name, dxy)
        return best

    def step(self):
        self.col_rad = self.column_radiance()
        self.fb.vision(self.col_rad)
        # taste: front legs touching fruit -> sweet GRNs fire (Poisson 120 Hz, Shiu-style)
        name, dist = self.nearest_fruit()
        self.tasting = 1.0 if (dist < 0.015 and not self.fly.airborne) else 0.0
        self.air.step(FRAME_MS / 1000)
        self.smell_values = self.air.antennae(self.fly.eye_pos, self.fly.left, self.fly.forward)
        self.fb.smell(*self.smell_values)
        self.fb.wind(*self.air.deflections(self.fly.forward, self.fly.left))
        self.fb.taste(self.tasting)
        if 'proprioception' in self.fb.available_senses:
            self.fb.proprioception(0., 0., 0., self.fly.airborne, yaw_rate=self.fly.yaw_rate)
        if 'interoception' in self.fb.available_senses:
            self.fb.interoception(self.metabolism.energy, sated=self.metabolism.sated,
                                  airborne=self.fly.airborne,
                                  feeding=bool(self.tasting) and not self.metabolism.sated)
        self.fb.step(FRAME_MS)
        motor = self.fb.motor()
        self.cmd = self.loco.readout(motor, dt_s=FRAME_MS / 1000)
        self.wcmd = self.flight.readout(motor)
        if self.program is not None:
            parts = self.program.parts if isinstance(self.program, programs.Composite) else [self.program]
            for part in parts:                            # each part by its kind: brain-side, sensor-side, or body-side
                if hasattr(part, "pfl_hz"):               # cx.CompassSteering: drives PFL3 / DNp09 in the brain, the body is untouched
                    info = part.apply(self.fb, motor, self.fly, self.metabolism, FRAME_MS / 1000)
                    self.cmd = dict(self.cmd, mode=info["mode"], rates=dict(self.cmd["rates"], **{"odour Hz": info["odour_hz"], "gate x10": info["gate"] * 10, "steer err x10": info["error"] * 10}))
                elif isinstance(part, programs.KlinotaxisProgram):
                    self.cmd = part.apply(motor, self.cmd, self.fly, self.metabolism, FRAME_MS / 1000, antennae=self.smell_values)
                else:
                    self.cmd = part.apply(motor, self.cmd, self.fly, self.metabolism, FRAME_MS / 1000)
        if self.gating is not None:
            self.wcmd = self.gating.apply(motor, self.wcmd, self.fly, FRAME_MS / 1000)
        if self.fly.airborne:
            self.feeding = False
            self.metabolism.update(False, 0.0, FRAME_MS / 1000)
            if self.fence:
                x0, x1, y0, y1 = self.info["table_extent"]
                self.flight.step(self.fly, self.wcmd, FRAME_MS / 1000, lambda x, y: self.info["table_top_z"], (x0, x1, y0, y1, 2.6))
            else:
                self.flight.step(self.fly, self.wcmd, FRAME_MS / 1000, self.surfaces, (-2, 2, -2, 2, 2.6))
        elif not self.flight.maybe_takeoff(self.fly, self.wcmd):
            cmd = self.decoder_cmd() if getattr(self, "decoder", False) else self.cmd
            self.feeding = self.metabolism.update(bool(self.tasting), self.fly.speed, FRAME_MS / 1000)
            if self.feeding:                                   # a fly that is feeding stops walking
                cmd = dict(cmd, speed=0.0, yaw=0.0)
            if self.fence:
                x0, x1, y0, y1 = self.info["table_extent"]
                self.loco.step(self.fly, cmd, FRAME_MS / 1000, (x0 + 0.02, x1 - 0.02, y0 + 0.02, y1 - 0.02))
            else:
                self.loco.step(self.fly, cmd, FRAME_MS / 1000, self.surfaces)
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
    """Render the observatory; retain this entry point for profiling and captures."""
    if not hasattr(sim, "_ui"):
        sim._ui = RoomUI()
    sim._ui.draw(sim, screen, orbit, paused, bmap)


def parse_dt_by_module(text):
    if not text:
        return None
    return {k.strip(): float(v) for k, v in (item.split("=") for item in text.split(","))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0, help="stop after this much brain time (0 = run until quit)")
    ap.add_argument("--gif", type=str, default="", help="record a GIF to this path")
    ap.add_argument("--screenshot", type=str, default="", help="save the final observatory frame as a PNG")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teleport", action="store_true", help="start next to the apple")
    ap.add_argument("--loom-at", type=float, default=-1, help="launch a loom at this brain time (s)")
    ap.add_argument("--decoder", type=str, default="", help="drive walking from a trained DN decoder (out/decoder.npz)")
    ap.add_argument("--wing-at", type=float, default=-1, help="stimulate the flight DNs (DNg02_a, DNa08) at this brain time (s)")
    ap.add_argument("--fast", action="store_true", help="speed preset for slower GPUs (Apple MPS): brain dt 1 ms, optic dt 2 ms, half-res camera")
    ap.add_argument("--brain-dt", type=float, default=None, help="LIF step (ms), default 0.5")
    ap.add_argument("--dt-by-module", default=None, help="per-module LIF clocks, e.g. vnc=1.0 or vnc=1.0,descending=1.0 (ms; multiples of --brain-dt)")
    ap.add_argument("--no-prune", action="store_true", help="keep the optic-lobe synapses in the LIF matrix (they are zeros; for timing comparisons)")
    ap.add_argument("--receptor-model", default="default", choices=["default", "off", "sign", "sign+gain", "full"],
                    help="synapse-sign model: 'default' = LIFParams' own (receptor-corrected 'sign'/'abs' since session 10); 'off' = the presynaptic-sign rule "
                         "(the pre-session-10 weights); see docs/NT_INTEGRATION.md")
    ap.add_argument("--fruit", default="all", choices=["all", "apple"], help="fruit on the table: all 19 items, or the apple alone (a single source)")
    ap.add_argument("--fence", action="store_true", help="test fixture: the fly cannot leave the table top by walking or hopping (scores foraging without the escape problem)")
    ap.add_argument("--program", default="none",
                    help="hand-designed behaviour program between the brain and the body (flyverse/programs.py): none | anemotaxis | klinotaxis | cx, or a+b to compose; default none, the plain model")
    ap.add_argument('--preset', choices=['raw', 'instrumented'], default='raw', help='raw brain or explicitly named experimental instruments')
    from flyverse.instruments import add_cli_arguments
    add_cli_arguments(ap)
    ap.add_argument("--escape-gating", action="store_true", help="habituation + efference-copy gating of the giant-fibre escape (programs.EscapeGating)")
    ap.add_argument("--gf-threshold", type=float, default=None, metavar="HZ",
                    help=f"base giant-fibre escape threshold before --escape-gating (Hz, default {body.Flight.gf_hz:g}); --load restores the saved value")
    ap.add_argument("--cuda-graphs", action="store_true", help="capture and replay controller frames and sensory ray tracing on CUDA")
    ap.add_argument("--cuda-kernels", action=argparse.BooleanOptionalAction, default=None,
                    help="fused CUDA neuron updates (requires nvcc and a host C++ compiler)")
    ap.add_argument("--event-driven", action=argparse.BooleanOptionalAction, default=None,
                    help="traverse active synapses; CUDA graph capture requires --cuda-kernels")
    ap.add_argument("--cuda-sparse", choices=["torch", "warp"], default="torch",
                    help="sparse products: cuSPARSE or experimental CUDA warp CSR (batch 1; requires --cuda-kernels)")
    ap.add_argument("--cuda-compact", action=argparse.BooleanOptionalAction, default=True,
                    help="compact native CUDA event traversal and skip frozen cells at rest; preserves full state")
    ap.add_argument("--sensory-cuda-graphs", action=argparse.BooleanOptionalAction, default=None,
                    help="override sensory ray capture independently (default: follows --cuda-graphs)")
    ap.add_argument("--weight-dtype", choices=["float32", "float16"], default="float32", help="LIF sparse weights; float16 requires CUDA")
    ap.add_argument("--optic-dt", type=float, default=None, help="optic-lobe substep (ms), default 1 (the RL env uses 2)")
    ap.add_argument("--cam-scale", type=int, default=None, help="fly's-eye camera downscale factor, default 1")
    ap.add_argument("--brain-map", action="store_true", help="open the brain atlas with soma activity highlights")
    ap.add_argument("--brain-map-mode", choices=("activity","nt"), default="activity",
                    help="map layer (nt opens live levels from an optional NTSource; does not enable NT dynamics)")
    ap.add_argument("--window", type=str, default="", help="initial window size WxH (default: the design size)")
    ap.add_argument("--start", type=str, default="", help="start pose: 'x,y' (on the table), 'x,y,z', or 'floor'")
    ap.add_argument("--trail-seconds", type=float, default=20.0, help="how long the fly's trail persists in the scene view (0 = off)")
    ap.add_argument("--map-every", type=int, default=4, help="brain map: sample the brain every N drawn frames (activity fades in between)")
    ap.add_argument("--map-no-blur", action="store_true", help="brain map: no fading between samples")
    ap.add_argument("--load", type=str, default="", help="resume from a saved state (.pt); F5/F9 quick-save/load, S timestamped save")
    ap.add_argument("--wind-speed", type=float, default=0.3, help="m/s (0 = still air: no plume, no wind cue)")
    ap.add_argument("--wind-dir", type=float, default=180.0, help="direction the wind blows towards, deg (180 = from the door at +x)")
    args = ap.parse_args()
    if args.gf_threshold is not None and (not np.isfinite(args.gf_threshold) or args.gf_threshold < 0):
        ap.error("--gf-threshold must be finite and nonnegative (Hz)")
    if args.instrument and args.preset != 'instrumented':
        ap.error('--instruments requires --preset instrumented')
    from flyverse.instruments import validate_cli
    validate_cli(ap, args.instrument)
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame
    pygame.init()
    try:
        win = tuple(int(v) for v in args.window.lower().split("x")) if args.window else (W, H)
    except ValueError:
        ap.error("--window must be a positive size, e.g. 1360x820")
    if len(win) != 2 or min(win) <= 0:
        ap.error("--window must be a positive size, e.g. 1360x820")
    DW, DH = canvas_size(win)
    screen = pygame.display.set_mode(win, pygame.RESIZABLE)
    canvas = pygame.Surface((DW, DH))
    recording_size = (DW, DH)
    pygame.display.set_caption("flyverse // room console")
    font = None  # draw's compatibility argument; RoomUI owns its typography
    ui = RoomUI()
    ui.loading(canvas)
    screen.blit(pygame.transform.smoothscale(canvas,screen.get_size()),(0,0))
    pygame.display.flip()
    pygame.event.pump()
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
    sim = Sim(args.seed, start=start, trail_seconds=args.trail_seconds, wind_speed=args.wind_speed, wind_dir=args.wind_dir,
              cuda_graphs=args.cuda_graphs, weight_dtype=args.weight_dtype, program=args.program, escape_gating=args.escape_gating,
              dt_by_module=parse_dt_by_module(args.dt_by_module), prune_frozen=not args.no_prune, fruit_set=args.fruit, fence=args.fence,
              cuda_kernels=args.cuda_kernels, event_driven=args.event_driven, cuda_sparse=args.cuda_sparse,
              cuda_compact=args.cuda_compact, receptor_model=args.receptor_model,
              sensory_cuda_graphs=args.sensory_cuda_graphs, preset=args.preset, instruments=args.instrument,
              gf_threshold=args.gf_threshold, **fast)
    if args.decoder:
        sim.load_decoder(args.decoder)
    if args.teleport:
        sim.teleport_to_fruit()
    bmap = brainmap.BrainMap(sim.c, sim.optic) if args.brain_map or args.brain_map_mode == "nt" else None
    sim.map_every, sim.map_no_blur = args.map_every, args.map_no_blur
    if args.load:
        sim.load_state(args.load)
    orbit = OrbitCam((0.0, 0.0, sim.info["table_top_z"]))
    sim._ui = ui
    ui.layout = Layout(canvas.get_size())
    if bmap is not None:
        ui.bmap, ui.tab = bmap, "atlas"
        ui.map_mode = args.brain_map_mode
    frames = []
    paused = False
    dragging = False
    clock = pygame.time.Clock()
    running = True
    dirty = True

    def view_transform():
        """canvas -> window: uniform scale, centred (letterboxed)."""
        return fit_transform(canvas.get_size(), screen.get_size())

    def to_canvas(pos):
        sc, ox, oy = view_transform()
        return (pos[0] - ox) / sc, (pos[1] - oy) / sc

    def in_scene(pos):
        return not ui.help_open and ui.layout.scene_view.collidepoint(to_canvas(pos))

    def perform(action):
        nonlocal paused, dirty
        dirty = True
        if action == "help":
            if not ui.help_open:
                ui.resume_after_help = not paused
                paused = True
            else:
                paused = not ui.resume_after_help
        if ui.handle_action(action, sim):
            return
        if action == "pause":
            paused = not paused
        elif action == "reset":
            sim.reset_fly(); ui.notify("Fly returned to its starting pose")
            ui._map_time = None; ui._map_shown = None; ui._map_count = 0
        elif action == "apple":
            sim.teleport_to_fruit(); ui.notify("Fly moved next to the apple")
        elif action == "loom":
            sim.start_loom(); ui.notify("Loom stimulus presented")
        elif action == "escape":
            sim.stimulate_gf(); ui.notify("Giant fibre stimulated")
        elif action == "flight":
            sim.stimulate_wing_dns(); ui.notify("Flight descending neurons stimulated")
        elif action == "follow":
            orbit.follow = not orbit.follow
            if orbit.follow:
                orbit.dist = min(orbit.dist, .4)
        elif action == "home":
            orbit.reset()
        elif action in ("save", "load", "save_timestamp"):
            path = time.strftime("out/state_%Y%m%d_%H%M%S.pt") if action == "save_timestamp" else "out/quicksave.pt"
            try:
                if action == "load":
                    if not os.path.exists(path):
                        ui.notify("No quick-save yet. Save a state with F5 first.")
                        return
                    sim.load_state(path)
                    ui._eye_key = ui._scene_key = ui._map_time = None
                    ui._map_shown = None; ui._map_count = 0
                else:
                    if not hasattr(sim,"cmd"):
                        ui.notify("Start the simulation before saving a state.")
                        return
                    sim.save_state(path)
                ui.notify(("Loaded " if action == "load" else "Saved ") + path)
            except (OSError, ValueError, RuntimeError) as exc:
                ui.notify(f"Could not {action}: {exc}")

    key_actions = {pygame.K_SPACE:"pause",pygame.K_r:"reset",pygame.K_t:"apple",pygame.K_l:"loom",
                   pygame.K_f:"escape",pygame.K_w:"flight",pygame.K_c:"follow",pygame.K_HOME:"home",
                   pygame.K_F5:"save",pygame.K_F9:"load",pygame.K_s:"save_timestamp",
                   pygame.K_1:"tab:regions",pygame.K_2:"tab:motor",pygame.K_3:"tab:senses",pygame.K_4:"tab:atlas",
                   pygame.K_n:"map:toggle"}

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((max(1,ev.w), max(1,ev.h)), pygame.RESIZABLE)
                canvas = pygame.Surface(canvas_size(screen.get_size()))
                ui.layout = Layout(canvas.get_size())
                ui.hits = []
                dirty = True
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    if ui.help_open:
                        perform("help")
                    else:
                        running = False
                elif ev.key in (pygame.K_SLASH, pygame.K_QUESTION, pygame.K_h):
                    perform("help")
                elif ui.help_open:
                    continue
                elif ev.key in key_actions:
                    perform(key_actions[ev.key])
                elif ev.key == pygame.K_v:
                    modes = ("both","colour","contrast")
                    perform("retina:" + modes[(modes.index(ui.retina_mode)+1)%len(modes)])
                elif ev.key == pygame.K_LEFT:
                    orbit.orbit(-10, 0); dirty = True
                elif ev.key == pygame.K_RIGHT:
                    orbit.orbit(10, 0); dirty = True
                elif ev.key == pygame.K_UP:
                    orbit.orbit(0, 5); dirty = True
                elif ev.key == pygame.K_DOWN:
                    orbit.orbit(0, -5); dirty = True
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    orbit.zoom(0.8); dirty = True
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    orbit.zoom(1.25); dirty = True
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                action = ui.action_at(to_canvas(ev.pos))
                if action:
                    perform(action)
                elif in_scene(ev.pos):
                    dragging = True
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging = False
            elif ev.type == pygame.MOUSEMOTION and dragging and not ui.help_open:
                sc, _, _ = view_transform()
                orbit.orbit(-ev.rel[0] / sc * 0.5, ev.rel[1] / sc * 0.4)
                dirty = True
            elif ev.type == pygame.MOUSEWHEEL:
                pos = pygame.mouse.get_pos()
                if ui.wheel(to_canvas(pos),ev.y):
                    dirty = True
                elif in_scene(pos):
                    orbit.zoom(0.85 if ev.y > 0 else 1.18); dirty = True
        if not running:
            break
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
        if frame_no % 4 == 0 or paused or dirty:
            ui.mouse = to_canvas(pygame.mouse.get_pos())
            draw(sim, canvas, font, orbit, paused, bmap)
            sc, ox, oy = view_transform()
            screen.fill((0, 0, 0))
            if abs(sc - 1.0) < 1e-6:
                screen.blit(canvas, (int(ox), int(oy)))
            else:
                screen.blit(pygame.transform.smoothscale(canvas, (max(1,int(canvas.get_width() * sc)), max(1,int(canvas.get_height() * sc)))), (int(ox), int(oy)))
            pygame.display.flip()
            dirty = False
            if args.gif and frame_no % 8 == 0:
                frame = canvas if canvas.get_size() == recording_size else pygame.transform.smoothscale(canvas,recording_size)
                frames.append(np.transpose(pygame.surfarray.array3d(frame), (1, 0, 2)))
        if args.seconds and sim.brain.t >= args.seconds * 1000:
            running = False
        clock.tick(60 if paused else 240)
    if args.screenshot:
        os.makedirs(os.path.dirname(args.screenshot) or ".", exist_ok=True)
        if ui._last_sim_time != sim.brain.t or dirty:
            draw(sim,canvas,font,orbit,paused,bmap)
        pygame.image.save(canvas,args.screenshot)
        print("wrote",args.screenshot)
    if args.gif and frames:
        import imageio
        imageio.mimsave(args.gif, frames, duration=0.08, loop=0)
        print("wrote", args.gif, len(frames), "frames")
    pygame.quit()


if __name__ == "__main__":
    main()
