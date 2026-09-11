"""FlyRoomEnv: B flies with the connectome brain, walking on the picnic table, as a vectorised RL env.

The question this environment poses: *does the fly's brain state carry enough information to find the
fruit?* The agent (a decoder) reads a slice of brain activity -- by default the firing rates of all
1,314 descending neurons, the only channel by which the real brain talks to the legs -- and outputs
walking commands. Vision, smell and taste are produced by the world and delivered to the brain by the
same machinery as the demo. The connectome itself is not trained.

    obs     (B, n_obs)   descending-neuron rates (Hz) / 10, clipped to [0, 4]
    action  (B, 2)       [forward in -1..1 -> cm/s * max_speed, yaw in -1..1 -> deg/s * max_yaw]
    reward  (B,)         progress towards the nearest fruit (cm) + 1.0 per frame of tasting - 0.01
    done    (B,)         episode length reached (or fell off the table -> teleported back, no done)

Vectorised API in the PufferLib / gymnasium-vector style (reset() -> obs; step(actions) -> obs, reward,
done, info). PufferLib itself does not build on this Windows box (see docs/NOTES.md); on Linux wrap
this with pufferlib.emulation / vector and PuffeRL. Frame = 10 ms of brain time; ~40 ms wall for B=32.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from . import air, body, brain, connectome, optic, retina, world


@dataclass
class EnvParams:
    frame_ms: float = 10.0
    episode_s: float = 8.0
    max_speed: float = 0.02      # m/s
    max_yaw: float = np.deg2rad(200)
    taste_reward: float = 1.0
    step_penalty: float = 0.01
    obs: str = "descending"      # "descending" | "descending+motor" | "pn" (per-glomerulus PN rates + their change)
                                 # | "pn3" (pn + rectified negative change: the klinotaxis feature)
    pn_lag_frames: int = 50      # for obs="pn": the derivative is taken over this many frames (0.5 s)
    seed: int = 0
    spawn_radius: float = 0.0    # > 0: curriculum -- spawn within this distance (m) of a random fruit
    wind_speed: float = 0.3      # m/s; the plume blows towards -x
    rays_per_ommatidium: int = 1 # 1 for training speed (the demo uses 7); the ray tracer dominates at large B
    optic_dt_ms: float = 2.0     # optic-lobe substep (the demo uses 1 ms)


class FlyRoomEnv:
    def __init__(self, batch: int = 16, params: EnvParams | None = None, device=None):
        self.B = int(batch)
        self.p = params or EnvParams()
        self.rng = np.random.default_rng(self.p.seed)
        self.c = connectome.load(verbose=False)
        self.r = retina.build_retina(self.c, retina.EyeGeometry(rays_per_ommatidium=self.p.rays_per_ommatidium))
        self.optic = optic.OpticLobe(self.c, self.r, optic.OpticParams(dt_ms=self.p.optic_dt_ms), batch=self.B, device=device)
        self.optic.relax()
        self.brain = brain.Brain(self.c, seed=self.p.seed, batch=self.B, device=device)
        self.brain.freeze(self.optic.rate_idx)
        self.world, self.info = world.make_room(self.p.seed)
        self.dirs_b, self.wts = self.r.ray_directions()
        self.wts_t = torch.from_numpy(self.wts).float().to(self.world.device)
        taste = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "taste_grns.csv"))
        self.sweet = self.c.index_of(taste.bodyId[taste.taste == "sweet"].to_numpy())
        self.air = air.Air([(name, cen, 1.0) for name, cen, rad in self.info["fruit"]], air.WindParams(speed=self.p.wind_speed), seed=self.p.seed)
        self.olf = air.BilateralOlfaction(self.c, self.air)
        self.windsense = air.WindSense(self.c, self.air)
        n = self.c.neurons
        if self.p.obs == "descending":
            self.obs_idx = self.c.select(superclass="descending_neuron")
        elif self.p.obs == "descending+motor":
            self.obs_idx = np.concatenate([self.c.select(superclass="descending_neuron"), self.c.select(superclass="vnc_motor"), self.c.select(superclass="cb_motor")])
        elif self.p.obs in ("pn", "pn3"):
            # projection neurons grouped by glomerulus (type prefix before '_'): the odour code the mushroom
            # body and lateral horn read. obs = [glomerulus means now, means now - means `lag` frames ago]
            pn = self.c.select(type="~_l2PN|_adPN|_lPN|_lvPN|_ilPN|_ivPN|_vPN")
            glom = np.array([t.split("_")[0] for t in n.type.to_numpy()[pn]])
            self.pn_gloms = sorted(set(glom))
            gid = np.array([self.pn_gloms.index(g) for g in glom])
            A = np.zeros((len(self.pn_gloms), len(pn)), np.float32)
            A[gid, np.arange(len(pn))] = 1.0
            A /= A.sum(1, keepdims=True)
            self.pn_A = torch.from_numpy(A).to(self.brain.device)
            self.obs_idx = pn
            self.pn_hist = []
        else:
            raise ValueError(self.p.obs)
        self.obs_idx_t = torch.as_tensor(self.obs_idx, device=self.brain.device)
        self.n_obs = {"pn": 2, "pn3": 3}.get(self.p.obs, 0) * len(self.pn_gloms) if self.p.obs in ("pn", "pn3") else len(self.obs_idx)
        self.fruit_xy = np.array([[cen[0], cen[1]] for _, cen, _ in self.info["fruit"]])
        self.fruit_r = np.array([rad for _, _, rad in self.info["fruit"]])
        self.x0, self.x1, self.y0, self.y1 = self.info["table_extent"]
        self.z = self.info["table_top_z"]
        self.eye_h = 0.0012
        self.x = np.zeros(self.B); self.y = np.zeros(self.B); self.heading = np.zeros(self.B)
        self.t_frames = 0
        self.max_frames = int(self.p.episode_s * 1000 / self.p.frame_ms)

    # ------------------------------------------------------------------ helpers
    def fruit_dist(self) -> np.ndarray:
        d = np.hypot(self.x[:, None] - self.fruit_xy[None, :, 0], self.y[:, None] - self.fruit_xy[None, :, 1]) - self.fruit_r[None]
        return d.min(axis=1)

    def _place(self, rows):
        k = len(rows)
        if self.p.spawn_radius > 0:
            j = self.rng.integers(0, len(self.fruit_xy), k)
            ang = self.rng.uniform(-np.pi, np.pi, k)
            d = self.fruit_r[j] + self.rng.uniform(0.02, self.p.spawn_radius, k)
            self.x[rows] = np.clip(self.fruit_xy[j, 0] + d * np.cos(ang), self.x0 + 0.03, self.x1 - 0.03)
            self.y[rows] = np.clip(self.fruit_xy[j, 1] + d * np.sin(ang), self.y0 + 0.03, self.y1 - 0.03)
        else:
            self.x[rows] = self.rng.uniform(self.x0 + 0.05, self.x1 - 0.05, k)
            self.y[rows] = self.rng.uniform(self.y0 + 0.05, self.y1 - 0.05, k)
        self.heading[rows] = self.rng.uniform(-np.pi, np.pi, k)

    def _radiance(self) -> torch.Tensor:
        """(B, n_col, 4) per-column radiance for all flies in one trace."""
        n_col, k, _ = self.dirs_b.shape
        cos, sin = np.cos(self.heading), np.sin(self.heading)
        d = self.dirs_b.reshape(-1, 3)                                                  # (n_col*k, 3) body frame
        dx = cos[:, None] * d[None, :, 0] - sin[:, None] * d[None, :, 1]
        dy = sin[:, None] * d[None, :, 0] + cos[:, None] * d[None, :, 1]
        dz = np.broadcast_to(d[None, :, 2], dx.shape)
        dirs = np.stack([dx, dy, dz], axis=-1).reshape(-1, 3)                            # (B*n_col*k, 3)
        eye = np.stack([self.x, self.y, np.full(self.B, self.z + self.eye_h)], axis=1)   # (B, 3)
        org = np.repeat(eye, n_col * k, axis=0)
        rad = self.world.trace(torch.from_numpy(np.ascontiguousarray(org, dtype=np.float32)), torch.from_numpy(dirs.astype(np.float32)))
        rad = rad.view(self.B, n_col, k, 4)
        return (rad * self.wts_t[None, None, :, None]).sum(2)

    def _obs(self) -> np.ndarray:
        if self.p.obs in ("pn", "pn3"):
            g = (self.brain.rate[:, self.obs_idx_t] @ self.pn_A.T / 50.0).clamp(0, 4)      # (B, n_glom), PN ~0-200 Hz
            self.pn_hist.append(g)
            self.pn_hist = self.pn_hist[-(self.p.pn_lag_frames + 1):]
            d = g - self.pn_hist[0]
            parts = [g, d] + ([(-d).clamp_min(0)] if self.p.obs == "pn3" else [])
            return torch.cat(parts, dim=1).cpu().numpy().astype(np.float32)
        # descending neurons mostly fire 0-20 Hz here: /10 puts the typical active unit at O(1)
        return (self.brain.rate[:, self.obs_idx_t] / 10.0).clamp(0, 4).cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------ API
    def reset(self, same_start: bool = False) -> np.ndarray:
        """same_start: all B flies spawn at one random pose (common random numbers for ES)."""
        self._place(np.arange(self.B))
        if same_start:
            self.x[:] = self.x[0]; self.y[:] = self.y[0]; self.heading[:] = self.heading[0]
        self.brain.reset()
        self.optic.reset()
        self.t_frames = 0
        if self.p.obs in ("pn", "pn3"):
            self.pn_hist = []
        self.prev_dist = self.fruit_dist()
        self._sense_and_think()
        return self._obs()

    def _sense_and_think(self):
        rad = self._radiance()
        self.brain.drive = self.optic.step_frame(rad, self.brain.rate, self.p.frame_ms)
        self.air.step(self.p.frame_ms / 1000)
        eye = np.stack([self.x, self.y, np.full(self.B, self.z + self.eye_h)], axis=1)
        fwd = np.stack([np.cos(self.heading), np.sin(self.heading), np.zeros(self.B)], axis=1)
        left = np.stack([-np.sin(self.heading), np.cos(self.heading), np.zeros(self.B)], axis=1)
        self.brain.set_poisson(self.olf.orn_idx, self.olf.rates_batch(eye, left, fwd))
        w = self.air.vector
        wb_x = fwd @ w; wb_y = left @ w                                                  # wind in each body frame
        c45 = np.cos(np.pi / 4)
        dL = -(wb_x * c45 + wb_y * c45) / self.windsense.full_speed; dR = -(wb_x * c45 - wb_y * c45) / self.windsense.full_speed
        rE, rC = self.windsense.rates_batch(dL, dR)
        self.brain.set_poisson(self.windsense.joE, rE); self.brain.set_poisson(self.windsense.joC, rC)
        tasting = (self.fruit_dist() < 0.01).astype(np.float32)
        self.tasting = tasting
        self.brain.set_poisson(self.sweet, np.repeat(tasting[:, None] * 120.0, len(self.sweet), axis=1))
        self.brain.step(int(self.p.frame_ms / self.brain.p.dt))

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1).reshape(self.B, 2)
        dt = self.p.frame_ms / 1000
        speed = action[:, 0] * self.p.max_speed
        self.heading += action[:, 1] * self.p.max_yaw * dt
        nx = self.x + speed * dt * np.cos(self.heading)
        ny = self.y + speed * dt * np.sin(self.heading)
        inside = (nx > self.x0 + 0.02) & (nx < self.x1 - 0.02) & (ny > self.y0 + 0.02) & (ny < self.y1 - 0.02)
        self.x = np.where(inside, nx, self.x); self.y = np.where(inside, ny, self.y)
        self._sense_and_think()
        dist = self.fruit_dist()
        reward = (self.prev_dist - dist) * 100.0 + self.p.taste_reward * self.tasting - self.p.step_penalty
        self.prev_dist = dist
        self.t_frames += 1
        done = np.full(self.B, self.t_frames >= self.max_frames)
        info = {"dist_cm": dist * 100, "tasting": self.tasting}
        return self._obs(), reward.astype(np.float32), done, info

    # convenience for evaluating hand-written policies
    def klinotaxis_action(self, obs: np.ndarray, gain: float = 4.0) -> np.ndarray:
        """Hand-written chemotaxis on the PN observation: walk forward; turn (left) at a rate proportional
        to the total *drop* in odour over the last 0.5 s. Needs obs='pn' or 'pn3'."""
        n = len(self.pn_gloms)
        drop = np.clip(-obs[:, n:2 * n].sum(1), 0, None)
        return np.stack([np.ones(self.B), np.clip(gain * drop, 0, 1)], axis=1)

    def oracle_action(self) -> np.ndarray:
        """Head straight for the nearest fruit (ground truth, for reward calibration)."""
        d = np.hypot(self.x[:, None] - self.fruit_xy[None, :, 0], self.y[:, None] - self.fruit_xy[None, :, 1]) - self.fruit_r[None]
        j = d.argmin(axis=1)
        ang = np.arctan2(self.fruit_xy[j, 1] - self.y, self.fruit_xy[j, 0] - self.x)
        err = (ang - self.heading + np.pi) % (2 * np.pi) - np.pi
        return np.stack([np.ones(self.B), np.clip(err / 0.5, -1, 1)], axis=1)
