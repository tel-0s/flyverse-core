"""mars -- a fly connectome rides a Mars rover (games/PLAN.md, section 8).

A six-wheeled rover crosses the floor of a Jezero-like crater (a GAME scene: butterscotch sky, a small pale sun, red
regolith, dark dunes, the crater rim on the horizon, dust devils) toward a waypoint past a fixed boulder field. The
shipped fly (MaleCNS v1.0, preset 'raw') rides on a post above the mast's camera head: its 1,466 ommatidial columns are
ray-traced every 10 ms through the camera's scene, and a dust devil's wind reaches its antennae through the shipped
antenna model and `fb.wind`. A GAME autopilot drives the route line at cruise speed. Two declared, read-only decoders
are the only things the brain touches:

    wind_steer   yaw rate from the wind descending neurons' L-R index (DNp18 / DNp33), towards the side it reports;
                 added to the autopilot's (whose limit is above the decoder's cap), held while recovering from a collision
    hazard_stop  giant fibre DNp01 >= 33 Hz (the shipped body model's takeoff line) -> brake, then back up

Everything else (the planet, the rover, its speeds, the autopilot, the devils and their wind, collisions and the
detour after one, the camera, dust, tracks) is GAME. Controls, same seed: `--control off` (both decoders read,
neither applied), `--control nowind` (fb.wind gets still air), `--control hidden` (the hazard boulders removed from
the fly's scene).

    python games/mars.py                                             # interactive window
    python games/mars.py --seed 100 --record out/games/mars/dev.mp4  # headless clip + run log
    python games/mars.py --seed 100 --control off --record out/games/mars/dev_off.mp4
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import games.common as gc  # noqa: E402

import torch  # noqa: E402

DEG = math.pi / 180.0

# ======================================================================================================== the scene (GAME)
# World frame: x east (the route runs along +x), y north (+y = the rover's left at the start), z up; metres.
SUN_AZ, SUN_EL = 58.0, 24.0                    # deg: ahead-left of the route and low, for long shadows
SUN_RADIUS_DEG = 0.55                          # the camera's sun disc (Mars' real sun is 0.35 deg across)
SKY_HORIZON = np.array([0.66, 0.44, 0.27])     # butterscotch (linear RGB)
SKY_ZENITH = np.array([0.30, 0.20, 0.14])
SUN_HALO = np.array([0.62, 0.68, 0.80])        # the pale blue-white glow round the Martian sun
SUN_RGB = np.array([1.0, 0.94, 0.86])
REGOLITH = np.array([0.34, 0.165, 0.085])      # linear albedo
DUNE_SAND = np.array([0.15, 0.085, 0.055])     # dark basaltic sand
ROCK_DARK = np.array([0.085, 0.07, 0.06])
ROCK_LIGHT = np.array([0.24, 0.16, 0.11])
HAZE_RGB = np.array([0.62, 0.44, 0.30])
DUST_RGB = np.array([0.33, 0.20, 0.12])        # the dust devils' dust (linear), before sun / shade
HAZE_M = 2600.0                                # aerial perspective: 1 - exp(-t / HAZE_M) of haze
FAR_L = 2048.0                                 # the heightfield spans [-FAR_L, FAR_L]^2
FAR_N = 2048
DETAIL_T = 64.0                                # the detail heightfield tiles every DETAIL_T m
DETAIL_N = 1024
CRATER_C = (160.0, 60.0)                       # Jezero-like crater: its rim rings the plain on the horizon
CRATER_R, CRATER_W, CRATER_H = 1450.0, 300.0, 120.0
ZMAX = 420.0                                   # nothing in the scene is above this (rays above it that climb are sky)
TMAX = 5200.0


def unit_from_az_el(az_deg, el_deg):
    a, e = az_deg * DEG, el_deg * DEG
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


SUN_DIR = unit_from_az_el(SUN_AZ, SUN_EL)


def spectral_fbm(n, beta, seed, lowcut=1.0, highcut=None):
    """Periodic fractal noise on an n x n grid (amplitude spectrum f^-beta), zero mean, unit SD (CPU float32)."""
    g = torch.Generator().manual_seed(int(seed))
    noise = torch.randn(n, n, generator=g)
    F = torch.fft.rfft2(noise)
    fy = torch.fft.fftfreq(n)[:, None] * n
    fx = torch.fft.rfftfreq(n)[None, :] * n
    f = torch.sqrt(fx ** 2 + fy ** 2)
    amp = torch.where(f >= lowcut, f.clamp_min(1e-6) ** (-beta), torch.zeros_like(f))
    if highcut is not None:
        amp = amp * torch.exp(-(f / highcut) ** 2)
    h = torch.fft.irfft2(F * amp, s=(n, n))
    return ((h - h.mean()) / h.std()).float()


def smoothstep(a, b, x):
    t = ((x - a) / (b - a)).clamp(0, 1) if isinstance(x, torch.Tensor) else np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


class Tex2D:
    """A scalar or vector texture on a square domain, sampled bilinearly (clamped or wrapped) on any device."""

    def __init__(self, data, lo, size, wrap=False):
        data = data if data.ndim == 3 else data[..., None]               # (n, n, C): rows are y, columns x
        self.C = data.shape[2]
        self.img = data.permute(2, 0, 1)[None].contiguous()             # (1, C, n, n) for grid_sample
        self.lo, self.size, self.wrap = float(lo), float(size), wrap

    def __call__(self, x, y):
        u = (x - self.lo) / self.size
        v = (y - self.lo) / self.size
        if self.wrap:                           # periodic textures: fold into the tile (half-texel seam, negligible)
            u = u - torch.floor(u)
            v = v - torch.floor(v)
        g = torch.stack([u * 2 - 1, v * 2 - 1], -1).reshape(1, 1, -1, 2)
        out = torch.nn.functional.grid_sample(self.img, g, mode="bilinear", padding_mode="border", align_corners=False)
        out = out[0, :, 0, :].T                                         # (M, C)
        return out[:, 0] if self.C == 1 else out


@dataclass
class Rock:
    """A boulder: the union of a core ellipsoid and two lobes, sunk into the ground (GAME art and geometry)."""
    pos: np.ndarray                # centre of the core (world), m
    radii: np.ndarray              # core semi-axes, m
    yaw: float                     # rad
    albedo: np.ndarray
    hazard: bool = False           # on the route: counted in the tally
    name: str = ""
    lobes: list = field(default_factory=list)   # [(offset_local (3,), radii (3,))]

    @property
    def r_ground(self) -> float:
        """Horizontal radius of the footprint used for contact (the largest horizontal semi-axis of any part)."""
        rs = [max(self.radii[0], self.radii[1])]
        for off, r in self.lobes:
            rs.append(math.hypot(off[0], off[1]) + max(r[0], r[1]))
        return float(max(rs))

    @property
    def top(self) -> float:
        return float(self.pos[2] + max([self.radii[2]] + [o[2] + r[2] for o, r in self.lobes]))


def make_rock(rng, pos, size, *, dark=True, hazard=False, name=""):
    """A lumpy boulder of about `size` m radius with its base sunk into the ground."""
    rx = size * rng.uniform(0.9, 1.15)
    ry = size * rng.uniform(0.75, 1.0)
    rz = size * rng.uniform(0.65, 0.85)
    lobes = []
    for _ in range(2):
        ang = rng.uniform(0, 2 * math.pi)
        off = np.array([math.cos(ang) * rx * 0.45, math.sin(ang) * ry * 0.45, rng.uniform(-0.1, 0.25) * rz])
        lobes.append((off, np.array([rx, ry, rz]) * rng.uniform(0.5, 0.7, 3)))
    alb = (ROCK_DARK if dark else ROCK_LIGHT) * rng.uniform(0.85, 1.15, 3) ** 0.3
    p = np.array([pos[0], pos[1], pos[2] + rz * 0.35], float)
    return Rock(p, np.array([rx, ry, rz]), rng.uniform(0, 2 * math.pi), alb, hazard, name, lobes)


class MarsScene:
    """Terrain (a baked heightfield plus a tiled detail field), boulders (unions of ellipsoids), a Martian sky with a
    small pale sun, aerial haze and dust devils; one torch ray tracer for both the fly's 1,466 x 7 eye rays and the
    human camera. Everything here is GAME."""

    def __init__(self, device, *, terrain_seed=7, flat=False):
        self.device = torch.device(device)
        self.flat = flat
        dev = self.device
        n = FAR_N
        xs = (torch.arange(n) + 0.5) / n * 2 * FAR_L - FAR_L
        Y, X = torch.meshgrid(xs, xs, indexing="ij")
        if flat:
            H = torch.zeros(n, n)
            dune = torch.zeros(n, n)
        else:
            base = spectral_fbm(n, 1.9, terrain_seed, lowcut=2.0) * 3.2
            hills = spectral_fbm(n, 2.4, terrain_seed + 1, lowcut=1.0, highcut=400.0) * 14.0
            r = torch.sqrt((X - CRATER_C[0]) ** 2 + (Y - CRATER_C[1]) ** 2)
            ang = torch.atan2(Y - CRATER_C[1], X - CRATER_C[0])
            wob = (torch.sin(3 * ang + 1.3) * 0.06 + torch.sin(7 * ang + 0.4) * 0.03 + torch.sin(13 * ang) * 0.015)
            rr = r / (CRATER_R * (1 + wob))
            rough = spectral_fbm(n, 1.8, terrain_seed + 2, lowcut=6.0, highcut=260.0)
            rim = CRATER_H * torch.exp(-((rr - 1.0) * CRATER_R / CRATER_W) ** 2) * (1 + 0.3 * rough)
            outer = CRATER_H * 0.55 * smoothstep(1.0, 1.25, rr)
            # transverse dunes north of the route, dark sand
            dir_ = torch.tensor([math.cos(0.5), math.sin(0.5)])
            phase = (X * dir_[0] + Y * dir_[1]) / 17.0 * 2 * math.pi + 2.2 * spectral_fbm(n, 2.2, terrain_seed + 3)
            crest = (0.5 + 0.5 * torch.sin(phase)) ** 2.5
            dune = smoothstep(26.0, 60.0, Y) * (1 - smoothstep(380.0, 520.0, torch.sqrt((X - 150) ** 2 + (Y - 150) ** 2)))
            route = smoothstep(5.0, 18.0, torch.abs(Y)) * 0.7 + 0.3          # a smoother track along the route
            H = base * route + hills * (0.25 + 0.75 * smoothstep(40.0, 200.0, torch.abs(Y))) + rim + outer \
                + crest * 1.6 * dune
            H = H - self._center_height(H, xs)
        self.far = Tex2D(H.to(dev), -FAR_L, 2 * FAR_L)
        self.dune = Tex2D(dune.to(dev), -FAR_L, 2 * FAR_L)
        det = spectral_fbm(DETAIL_N, 1.5, terrain_seed + 10, lowcut=2.0) * 0.045
        self.detail = Tex2D(det.to(dev), 0.0, DETAIL_T, wrap=True)
        alb = torch.stack([spectral_fbm(512, 1.2, terrain_seed + 20, lowcut=2.0),
                           spectral_fbm(512, 1.0, terrain_seed + 21, lowcut=8.0)], -1)
        self.albedo_noise = Tex2D(alb.to(dev), 0.0, 96.0, wrap=True)
        self.rocks: list[Rock] = []
        self._pack_rocks()
        self.sun = torch.tensor(SUN_DIR, dtype=torch.float32, device=dev)
        # colour constants as device tensors made once (no host-to-device copies inside a trace: capture-safe)
        self.K = {k: torch.tensor(np.asarray(v, np.float32), device=dev) for k, v in (
            ("sky_horizon", SKY_HORIZON), ("sky_zenith", SKY_ZENITH), ("sun_halo", SUN_HALO), ("sun_rgb", SUN_RGB),
            ("regolith", REGOLITH), ("dune_sand", DUNE_SAND), ("haze", HAZE_RGB), ("dust", DUST_RGB))}
        self.devils: list[dict] = []
        self.tracks = None                             # wheel tracks (display: the camera's scene only)

    @staticmethod
    def _center_height(H, xs):
        i = int(torch.argmin(torch.abs(xs)))
        return H[i, i]

    # ------------------------------------------------------------------ geometry
    def height(self, x, y):
        return self.far(x, y) + self.detail(x, y)

    def height_np(self, x, y) -> float:
        t = self.height(torch.tensor([float(x)], device=self.device), torch.tensor([float(y)], device=self.device))
        return float(t.item())

    def set_rocks(self, rocks):
        self.rocks = list(rocks)
        self._pack_rocks()

    def _pack_rocks(self):
        self.ell = self._pack(self.rocks)
        self.ell_scenery = self._pack([r for r in self.rocks if not r.hazard])   # the hidden control's scene

    def _pack(self, rocks):
        dev = self.device
        parts = []
        for k, r in enumerate(rocks):
            c, s = math.cos(r.yaw), math.sin(r.yaw)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            parts.append((k, r.pos, r.radii, Rz, r.albedo))
            for off, rad in r.lobes:
                parts.append((k, r.pos + Rz @ off, rad, Rz, r.albedo))
        if not parts:
            return None
        return {
            "rock": torch.tensor([p[0] for p in parts], device=dev),
            "c": torch.tensor(np.array([p[1] for p in parts]), dtype=torch.float32, device=dev),
            "r": torch.tensor(np.array([p[2] for p in parts]), dtype=torch.float32, device=dev),
            "R": torch.tensor(np.array([p[3] for p in parts]), dtype=torch.float32, device=dev),
            "alb": torch.tensor(np.array([p[4] for p in parts]), dtype=torch.float32, device=dev),
        }

    def _ray_tex(self, o, d):
        """Per-ray affine maps from the distance t along o + t d to both height textures' grid_sample coordinates,
        so that the marches' inner loop costs a handful of kernels (it is launch-bound on a GPU). o (3,) or (M,3);
        d (M,3) or (1,3)."""
        far, det = self.far, self.detail
        o = o if o.ndim == 2 else o[None]
        kf, kd = 2.0 / far.size, 2.0 / det.size
        return ((o[:, :2] - far.lo) * kf - 1.0, (d[:, :2] * kf).contiguous(),     # far: clamped, [-1, 1]
                (o[:, :2] - det.lo) * kd, (d[:, :2] * kd).contiguous(),           # detail: wrapped below
                o[:, 2].contiguous(), d[:, 2].contiguous())

    def _above(self, maps, t):
        """(height of o + t d above the terrain, z of that point) for each ray: the same bilinear samples as
        height(), in fewer kernels."""
        Af, Bf, Ad, Bd, oz, dzr = maps
        M = t.shape[0]
        tt = t[:, None]
        gs = torch.nn.functional.grid_sample
        gf = torch.addcmul(Af, Bf, tt)
        hf = gs(self.far.img, gf.view(1, 1, M, 2), mode="bilinear", padding_mode="border", align_corners=False)
        gd = torch.remainder(torch.addcmul(Ad, Bd, tt), 2.0).sub_(1.0)             # = 2 frac(u) - 1: the tile wraps
        hd = gs(self.detail.img, gd.view(1, 1, M, 2), mode="bilinear", padding_mode="border", align_corners=False)
        pz = torch.addcmul(oz, dzr, t)
        return pz - hf.view(M) - hd.view(M), pz

    def march(self, o, d, n_steps=300, growth=0.012):
        """Terrain hit distance along each ray (inf for none). o (3,), d (M,3). Steps grow with distance and with the
        height above ground; the first crossing is refined by bisection. Fixed step count, masked: no host syncs."""
        M = d.shape[0]
        dev = d.device
        maps = self._ray_tex(o, d)
        t = torch.full((M,), 0.02, device=dev)
        lo = torch.zeros(M, device=dev)
        found = torch.zeros(M, dtype=torch.bool, device=dev)
        zero = torch.zeros(M, device=dev)
        c = torch.full((), 0.02, device=dev)
        for _ in range(n_steps):
            dz, _ = self._above(maps, t)
            found |= dz < 0
            step = torch.where(found, zero, torch.maximum(torch.add(c, t, alpha=growth), dz.mul_(0.42)))
            lo = torch.where(found, lo, t)
            t = t + step
        a, b = lo, t
        for _ in range(8):
            m = 0.5 * (a + b)
            under = self._above(maps, m)[0] < 0
            b = torch.where(under, m, b)
            a = torch.where(under, a, m)
        return torch.where(found, 0.5 * (a + b), torch.full_like(t, float("inf")))

    def _ell_t(self, o, d, sl, E):
        """Entry distance (inf if none) of rays o + t d into the ellipsoids of packing E in slice sl: (P, M)."""
        c, r, R = E["c"][sl], E["r"][sl], E["R"][sl]                        # (P,3), (P,3), (P,3,3)
        if o.ndim == 1:
            ol = torch.einsum("pj,pjk->pk", o[None] - c, R)[:, None, :]      # (P,1,3)
        else:
            ol = torch.einsum("mj,pjk->pmk", o, R) - torch.einsum("pj,pjk->pk", c, R)[:, None, :]
        dl = torch.einsum("mj,pjk->pmk", d, R)                              # (P,M,3)
        oq, dq = ol / r[:, None, :], dl / r[:, None, :]
        a = (dq * dq).sum(-1)
        b = (dq * oq).sum(-1)
        cc = (oq * oq).sum(-1) - 1
        disc = b * b - a * cc
        sq = torch.sqrt(disc.clamp_min(0))
        t0 = (-b - sq) / a
        t1 = (-b + sq) / a
        tt = torch.where(t0 > 1e-3, t0, t1)
        return torch.where((disc > 0) & (tt > 1e-3), tt, torch.full_like(tt, float("inf"))), t1, disc

    def rocks_hit(self, o, d, t_best, chunk=None, E="all"):
        """Nearest boulder hit closer than t_best: returns (t, normal (M,3), albedo (M,3), rock id (M,), hit mask).
        E: a rock packing ('all' = self.ell, every rock; the hidden control passes self.ell_scenery)."""
        M = d.shape[0]
        dev = d.device
        E = self.ell if isinstance(E, str) else E
        t_best = t_best.clone()
        best_k = torch.full((M,), -1, dtype=torch.long, device=dev)
        if E is None:
            return t_best, torch.zeros(M, 3, device=dev), torch.zeros(M, 3, device=dev), best_k, best_k >= 0
        P = E["c"].shape[0]
        chunk = chunk or max(1, min(P, int(2e7 // max(M, 1))))
        for s0 in range(0, P, chunk):
            sl = slice(s0, min(P, s0 + chunk))
            tt, _, _ = self._ell_t(o, d, sl, E)
            tmin, arg = tt.min(0)
            better = tmin < t_best
            t_best = torch.where(better, tmin, t_best)
            best_k = torch.where(better, arg + s0, best_k)
        hit = best_k >= 0
        k = best_k.clamp_min(0)
        c, r, R = E["c"][k], E["r"][k], E["R"][k]
        p = (o[None] if o.ndim == 1 else o) + d * torch.where(hit, t_best, torch.zeros_like(t_best))[:, None]
        pl = torch.einsum("mj,mjk->mk", p - c, R)
        nw = torch.einsum("mk,mjk->mj", pl / (r * r), R)
        normal = nw / nw.norm(dim=1, keepdim=True).clamp_min(1e-9)
        alb = E["alb"][k]
        rid = torch.where(hit, E["rock"][k], best_k)
        return t_best, normal, alb, rid, hit

    # ------------------------------------------------------------------ light
    def sky(self, d, *, camera=False):
        """Linear RGB of the sky along each direction (the sun disc only for the camera; the eye gets its glow)."""
        z = d[:, 2].clamp(-1, 1)
        up = smoothstep(0.0, 0.55, z.clamp_min(0)) ** 0.8
        hz, ze = self.K["sky_horizon"], self.K["sky_zenith"]
        col = hz[None] * (1 - up[:, None]) + ze[None] * up[:, None]
        cosg = (d @ self.sun).clamp(-1, 1)
        gam = torch.acos(cosg)
        halo = torch.exp(-gam / (9 * DEG)) * 0.55 + torch.exp(-gam / (2.5 * DEG)) * 0.8
        col = col + halo[:, None] * self.K["sun_halo"][None]
        below = z < 0
        col = torch.where(below[:, None], hz[None] * 0.85, col)
        if camera:
            disc = smoothstep(SUN_RADIUS_DEG * DEG, SUN_RADIUS_DEG * DEG * 0.8, gam) * 40.0
            col = col + disc[:, None] * self.K["sun_rgb"][None]
        return col

    def ground_albedo(self, p):
        nz = self.albedo_noise(p[:, 0], p[:, 1])
        dn = self.dune(p[:, 0], p[:, 1])
        reg, sand = self.K["regolith"], self.K["dune_sand"]
        a = reg[None] * (1 + 0.16 * nz[:, :1] + 0.08 * nz[:, 1:2])
        speck = (nz[:, 1:2] > 1.6).float() * 0.45
        a = a * (1 - speck)
        return a * (1 - dn[:, None] * 0.8) + sand[None] * dn[:, None] * 0.8

    def ground_normal(self, p, t):
        eps = (0.04 + 0.0015 * t)
        x, y = p[:, 0], p[:, 1]
        hx = self.height(x + eps, y) - self.height(x - eps, y)
        hy = self.height(x, y + eps) - self.height(x, y - eps)
        n = torch.stack([-hx, -hy, 2 * eps], -1)
        return n / n.norm(dim=1, keepdim=True)

    def sun_shadow(self, p, n_steps=40, E="all"):
        """1 where the sun reaches p, 0 where terrain or a boulder (of packing E) blocks it (hard shadows)."""
        M = p.shape[0]
        dev = p.device
        s = self.sun[None]
        o = p + s * 0.03
        t = torch.full((M,), 0.05, device=dev)
        blocked = torch.zeros(M, dtype=torch.bool, device=dev)
        maps = self._ray_tex(o, s)
        c = torch.full((), 0.05, device=dev)
        for _ in range(n_steps):
            dz, qz = self._above(maps, t)
            blocked |= (dz < 0) & (qz < ZMAX)
            t = t + torch.maximum(torch.add(c, t, alpha=0.05), dz.mul_(0.5))
        E = self.ell if isinstance(E, str) else E
        if E is not None:
            P = E["c"].shape[0]
            chunk = max(1, min(P, int(2e7 // max(M, 1))))
            dsun = s.expand(M, 3)
            for s0 in range(0, P, chunk):
                _, t1, disc = self._ell_t(o, dsun, slice(s0, min(P, s0 + chunk)), E)
                blocked |= ((disc > 0) & (t1 > 1e-3)).any(0)
        return (~blocked).float()

    def devil_layer(self, o, d, t_hit, t_s, devil_xy=None):
        """Dust along each ray from the devils: (opacity (M,), dust colour (M,3) linear RGB). Each devil is a column
        whose width flares with height and whose axis wanders a little, with twisting streaks, a denser wall, a skirt
        of dust at its foot, and a sunlit and a shaded side. One sample per ray, at its closest approach to the axis.
        devil_xy: (n_devils, 2) centres as a device tensor (the eye's captured trace), else from each devil's `pos`;
        t_s may then be a 0-d device tensor."""
        M = d.shape[0]
        dev = d.device
        alpha = torch.zeros(M, device=dev)
        col = torch.zeros(M, 3, device=dev)
        if not self.devils:
            return alpha, col
        oxy = o[..., :2] if o.ndim == 2 else o[None, :2]
        dxy = d[:, :2]
        dd = (dxy * dxy).sum(-1).clamp_min(1e-9)
        sun_h = self.sun[:2] / self.sun[:2].norm().clamp_min(1e-9)
        base = self.K["dust"]
        for i, dv in enumerate(self.devils):
            if devil_xy is None:
                c = torch.tensor(dv["pos"](t_s), dtype=torch.float32, device=dev)
            else:
                c = devil_xy[i]
            ts = ((c[None] - oxy) * dxy).sum(-1) / dd
            ts = ts.clamp_min(0)
            p = oxy + dxy * ts[:, None]
            oz = o[..., 2] if o.ndim == 2 else o[2]
            z = oz + d[:, 2] * ts
            zg = self.height(p[:, 0], p[:, 1])
            h = (z - zg).clamp_min(0)
            wob = torch.stack([torch.sin(h * 0.011 + t_s * 0.35 + dv["w0"]), torch.cos(h * 0.008 - t_s * 0.3)], -1)
            rel = p - c[None] - wob * (h * 0.045)[:, None]
            dist = rel.norm(dim=1)
            w = dv["w0"] + dv["flare"] * h
            q = dist / w
            ang = torch.atan2(rel[:, 1], rel[:, 0])
            swirl = 0.68 + 0.32 * torch.sin(h * 0.09 - t_s * 3.2 * dv["spin"] + ang * 2 + 0.8 * torch.sin(h * 0.031))
            wall = 0.75 + 0.25 * smoothstep(0.1, 0.7, q)
            column = (1 - smoothstep(0.45, 1.05, q)) * wall * swirl * torch.exp(-h / dv["height"])
            skirt = torch.exp(-(dist / (2.2 * dv["w0"])) ** 2) * torch.exp(-h / 6.0)
            dens = (column + skirt).clamp(0, 1.5)
            dens = dens * ((z - zg) > -0.5).float() * (ts < t_hit).float()
            a = dv["opacity"] * (1 - torch.exp(-3.6 * dens))
            side = torch.tanh(0.8 * (rel @ sun_h) / w)          # -1 shaded side .. +1 sunlit side, smooth across
            lit = 0.62 + 0.6 * (0.5 + 0.5 * side) - 0.15 * torch.exp(-h / 3.0)
            c_dv = base[None] * lit[:, None]
            # front-to-back: this devil's dust over what lies behind it (devils rarely overlap; order is by list)
            col = col * (1 - a[:, None]) + c_dv * a[:, None]
            alpha = 1 - (1 - alpha) * (1 - a)
        col = col / alpha.clamp_min(1e-6)[:, None]
        return alpha, col

    def trace(self, o, d, t_s=0.0, *, camera=False, rover=None, rover_shadow=True, hide_hazards=False, devil_xy=None):
        """Linear RGB along each ray. o (3,), d (M,3) unit, on self.device. Returns (rgb (M,3), depth (M,)).
        The camera marches finer (300 steps growing 1.2 % of the distance, 40 shadow steps); the eye, whose ommatidia
        average 4.5 deg, marches 170 steps growing 5.5 % (24 shadow steps). Dense and masked: few host syncs.
        hide_hazards: trace a scene without the hazard boulders (and their shadows): the hidden control's eye.
        devil_xy: see devil_layer. With camera=False the trace makes no host syncs and no host-to-device copies, so the
        eye's trace can be captured in a CUDA graph (MarsRover._eye_radiance)."""
        dev = self.device
        M = d.shape[0]
        E = self.ell_scenery if hide_hazards else self.ell
        n_march, growth, n_shadow = (300, 0.012, 40) if camera else (170, 0.055, 24)
        t_ter = self.march(o, d, n_march, growth)
        t, n_r, alb_r, rid, hit_r = self.rocks_hit(o, d, t_ter, E=E)
        hit_t = torch.isfinite(t_ter) & ~hit_r
        depth = torch.where(hit_r | hit_t, t, torch.full_like(t, float("inf")))
        is_rv = torch.zeros(M, dtype=torch.bool, device=dev)
        if rover is not None:
            t_rv, n_rv, alb_rv, is_rv = rover.hit(o, d, depth, dense=not camera)
            depth = torch.where(is_rv, t_rv, depth)
        g = hit_t & ~is_rv
        r_ = hit_r & ~is_rv
        surf = g | r_ | is_rv
        p = o[None] + d * torch.where(surf, depth, torch.zeros_like(depth))[:, None]
        n = self.ground_normal(p, torch.where(surf, depth, torch.zeros_like(depth)))
        alb = self.ground_albedo(p)
        if camera and self.tracks is not None:
            alb = alb * (1 - 0.3 * self.tracks.sample(p[:, 0], p[:, 1]))[:, None]
        bump = self.albedo_noise(p[:, 0] * 7.0 + p[:, 2] * 5.0, p[:, 1] * 7.0 - p[:, 2] * 3.0)
        nn = n_r + 0.18 * torch.stack([bump[:, 0], bump[:, 1], bump[:, 0] * 0.5], -1)
        nn = nn / nn.norm(dim=1, keepdim=True).clamp_min(1e-9)
        n = torch.where(r_[:, None], nn, n)
        alb = torch.where(r_[:, None], alb_r * (1 + 0.25 * bump[:, 1:2]).clamp_min(0.4), alb)
        if rover is not None:
            n = torch.where(is_rv[:, None], n_rv, n)
            alb = torch.where(is_rv[:, None], alb_rv, alb)
        lam = (n @ self.sun).clamp_min(0)
        lit = self.sun_shadow(p + n * 0.02, n_shadow, E=E)
        lit = torch.where(depth < 900.0, lit, torch.ones_like(lit))
        if rover is not None and rover_shadow:
            lit = lit * rover.shadow(p + n * 0.02, self.sun, dense=not camera)
        skyamb = 0.28 + 0.14 * n[:, 2:3].clamp(-1, 1)
        amb_col = self.K["sky_horizon"][None] * skyamb
        sunc = self.K["sun_rgb"][None] * 1.55
        col = alb * (sunc * (lam * lit)[:, None] + amb_col)
        if rover is not None:
            hvec = -d + self.sun[None]
            hvec = hvec / hvec.norm(dim=1, keepdim=True)
            spec = (n * hvec).sum(1).clamp_min(0) ** 40 * lit * 0.6
            col = col + torch.where(is_rv, spec, torch.zeros_like(spec))[:, None]
        haze = 1 - torch.exp(-torch.where(surf, depth, torch.zeros_like(depth)) / HAZE_M)
        hz = self.K["haze"][None]
        col = col * (1 - haze[:, None]) + hz * haze[:, None]
        rgb = torch.where(surf[:, None], col, self.sky(d, camera=camera))
        a, dust = self.devil_layer(o, d, depth, t_s, devil_xy)
        rgb = rgb * (1 - a[:, None]) + dust * a[:, None]
        return rgb, depth


class Tracks:
    """Wheel tracks pressed into the regolith (display: only the camera's scene samples them; the fly's eye does not).
    A (H, W) map over the route's area; each wheel's contact patch is stamped every tick."""

    def __init__(self, device, lo=(-30.0, -40.0), size=(220.0, 80.0), res=0.05, wheel_r=0.13):
        self.dev = torch.device(device)
        self.lo, self.size, self.res = lo, size, res
        self.W, self.H = int(round(size[0] / res)), int(round(size[1] / res))
        self.img = torch.zeros(1, 1, self.H, self.W, device=self.dev)
        k = int(math.ceil(wheel_r / res))
        oy, ox = np.mgrid[-k:k + 1, -k:k + 1]
        keep = (ox ** 2 + oy ** 2) * res * res <= wheel_r * wheel_r
        self.off = torch.tensor(np.stack([oy[keep], ox[keep]], 1), device=self.dev)       # (P, 2) row, col

    def stamp(self, xy):
        """xy: (n, 2) world points (wheel contacts)."""
        c = torch.as_tensor(np.asarray(xy, np.float32), device=self.dev)
        ix = torch.floor((c[:, 0] - self.lo[0]) / self.res).long()
        iy = torch.floor((c[:, 1] - self.lo[1]) / self.res).long()
        r = (iy[:, None] + self.off[None, :, 0]).reshape(-1).clamp(0, self.H - 1)
        q = (ix[:, None] + self.off[None, :, 1]).reshape(-1).clamp(0, self.W - 1)
        self.img[0, 0, r, q] = 1.0

    def sample(self, x, y):
        u = (x - self.lo[0]) / self.size[0] * 2 - 1
        v = (y - self.lo[1]) / self.size[1] * 2 - 1
        g = torch.stack([u, v], -1).reshape(1, 1, -1, 2)
        out = torch.nn.functional.grid_sample(self.img, g, mode="bilinear", padding_mode="zeros", align_corners=False)
        return out.reshape(-1)


def look_basis(forward, up=(0.0, 0.0, 1.0)):
    f = np.asarray(forward, float)
    f = f / np.linalg.norm(f)
    left = np.cross(np.asarray(up, float), f)
    left = left / max(np.linalg.norm(left), 1e-9)
    u = np.cross(f, left)
    return f, left, u


def camera_rays(forward, up, W, H, fov_deg, device):
    """(H*W, 3) unit ray directions of a pinhole camera; row 0 is the top, column 0 the left (not mirrored)."""
    f, left, u = look_basis(forward, up)
    tan = math.tan(fov_deg * DEG / 2)
    xs = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W * 2 - 1
    ys = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H * 2 - 1
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    ft, lt, ut = (torch.tensor(v, dtype=torch.float32, device=device) for v in (f, left, u))
    d = ft[None] + (-xx.reshape(-1, 1) * tan) * lt[None] + (-yy.reshape(-1, 1) * tan * H / W) * ut[None]
    return d / d.norm(dim=1, keepdim=True)


def tonemap(rgb, exposure=1.0):
    """Linear RGB -> display [0, 1]: ACES (Narkowicz fit), then sRGB gamma."""
    x = rgb * exposure
    y = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    return y.clamp(0, 1) ** (1 / 2.2)


# ======================================================================================================== the rover (GAME)
def rot_zyx(yaw, pitch, roll):
    """World-from-rover rotation: yaw about z, then pitch about the rover's y (nose down = +), then roll about x."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


WHITE_PAINT = (0.62, 0.60, 0.55)
GOLD = (0.50, 0.36, 0.14)
DARK = (0.07, 0.07, 0.08)
ALU = (0.36, 0.36, 0.38)
GREY = (0.22, 0.22, 0.23)
ROVER_L, ROVER_W = 3.1, 2.7                    # footprint used for contact (m), about Perseverance's
WHEEL_R = 0.27
MAST_X, HEAD_Z = 1.08, 2.08                    # the mast (rover frame) and the centre of its camera head, m
# Where the fly's eyes can sit (rover frame, m). Measured on dev seeds >= 100 (games/mars_assets/probe_game.py and
# the caption's measurements); EYE_MOUNT is the one the game uses.
POST_H = 0.35                                  # the post that carries the fly's eyes above the camera head, m
EYE_MOUNTS = {
    "mast_top": (1.08, 0.0, 2.26),             # 9 cm above the camera head (the first build's; 421 columns see rover)
    "mast_post": (1.08, 0.0, 2.17 + POST_H),   # on a 35 cm post above the head (71 columns see the rover)
}
EYE_MOUNT = "mast_post"
EYE_LOCAL = np.array(EYE_MOUNTS[EYE_MOUNT])
WHEELS_LOCAL = [(1.1, 1.12), (0.0, 1.2), (-1.1, 1.12), (1.1, -1.12), (0.0, -1.2), (-1.1, -1.12)]


def rover_parts():
    """Boxes (centre, half extents, colour), capsules (a, b, radius, colour) and wheels (a, b, radius) in the rover
    frame: x forward, y left, z up, origin on the ground under the rover's centre."""
    mx, hz = MAST_X, HEAD_Z
    boxes = [((0.05, 0.0, 1.0), (1.25, 0.8, 0.22), WHITE_PAINT),
             ((-0.3, 0.0, 1.27), (0.62, 0.56, 0.05), GOLD),
             ((mx, 0.0, hz), (0.1, 0.27, 0.09), DARK),
             ((mx + 0.02, 0.0, hz), (0.085, 0.29, 0.03), WHITE_PAINT),
             ((-0.55, -0.45, 1.36), (0.3, 0.3, 0.025), ALU)]
    caps = [((mx, 0.0, 1.2), (mx, 0.0, hz - 0.07), 0.055, WHITE_PAINT),
            ((mx, 0.0, hz + 0.09), (mx, 0.0, hz + 0.09 + POST_H - 0.03), 0.012, GREY),     # the eyes' post
            ((-1.2, 0.0, 1.05), (-1.7, 0.0, 1.5), 0.2, GREY),
            ((1.33, 0.5, 0.70), (1.33, -0.48, 0.70), 0.05, WHITE_PAINT)]
    for s in (1, -1):
        caps += [((0.55, s * 0.95, 0.95), (1.1, s * 1.02, WHEEL_R + 0.02), 0.045, GREY),
                 ((0.55, s * 0.95, 0.95), (-0.45, s * 1.02, 0.62), 0.045, GREY),
                 ((-0.45, s * 1.02, 0.62), (0.0, s * 1.08, WHEEL_R + 0.02), 0.04, GREY),
                 ((-0.45, s * 1.02, 0.62), (-1.1, s * 1.02, WHEEL_R + 0.02), 0.04, GREY)]
    wheels = []
    for wx, wy in WHEELS_LOCAL:
        s = 1 if wy > 0 else -1
        wheels.append(((wx, wy - s * 0.13, WHEEL_R), (wx, wy + s * 0.13, WHEEL_R), WHEEL_R))
    return boxes, caps, wheels


def _t(v, dev):
    return torch.tensor(np.asarray(v, np.float32), device=dev)


class Rover:
    """The rover's shape for ray tracing: analytic boxes, capsules and finite cylinders (wheels) in its own frame."""

    def __init__(self, device):
        self.dev = torch.device(device)
        boxes, caps, wheels = rover_parts()
        dev = self.dev
        self.bc = _t([b[0] for b in boxes], dev)
        self.bh = _t([b[1] for b in boxes], dev)
        self.bcol = _t([b[2] for b in boxes], dev)
        self.ca = _t([c[0] for c in caps], dev)
        self.cb = _t([c[1] for c in caps], dev)
        self.cr = _t([c[2] for c in caps], dev)
        self.ccol = _t([c[3] for c in caps], dev)
        self.wa = _t([w[0] for w in wheels], dev)
        self.wb = _t([w[1] for w in wheels], dev)
        self.wr = _t([w[2] for w in wheels], dev)
        self._alu = _t(ALU, dev)
        self.center_local = np.array([0.0, 0.0, 1.05])
        self.radius = 2.35
        # the pose lives in fixed device buffers, updated in place (a captured trace reads them)
        self._pos_t = torch.zeros(3, device=dev)
        self._R_t = torch.eye(3, device=dev)
        self._phase_t = torch.zeros((), device=dev)
        self.set_pose(np.zeros(3), np.eye(3), 0.0)

    def set_pose(self, pos, R, wheel_phase):
        self.pos = np.asarray(pos, float)
        self.R = np.asarray(R, float)
        self.wheel_phase = float(wheel_phase)
        self._pos_t.copy_(torch.as_tensor(np.asarray(self.pos, np.float32)))
        self._R_t.copy_(torch.as_tensor(np.asarray(self.R, np.float32)))
        self._phase_t.fill_(self.wheel_phase)
        self.center = self.pos + self.R @ self.center_local

    def _local(self, o, d):
        R = self._R_t
        return (o - self._pos_t) @ R, d @ R

    def _intersect(self, ol, dl, any_hit=False):
        """(t (M,), normal_local (M,3), colour (M,3)) of the nearest part; t = inf for none. ol, dl (M,3) local."""
        M = dl.shape[0]
        dev = self.dev
        # boxes (axis-aligned in the rover frame)
        inv = 1.0 / torch.where(dl.abs() < 1e-9, torch.full_like(dl, 1e-9), dl)
        lo = (self.bc - self.bh)[:, None, :]
        hi = (self.bc + self.bh)[:, None, :]
        t1 = (lo - ol[None]) * inv[None]
        t2 = (hi - ol[None]) * inv[None]
        tn, ax = torch.minimum(t1, t2).max(-1)
        tf = torch.maximum(t1, t2).min(-1).values
        okb = (tn < tf) & (tf > 1e-4)
        tb = torch.where(okb, torch.where(tn > 1e-4, tn, tf), torch.full_like(tn, float("inf")))
        tbm, kb = tb.min(0)
        # capsules
        ba = (self.cb - self.ca)[:, None, :]
        oa = ol[None] - self.ca[:, None, :]
        baba = (ba * ba).sum(-1)
        bard = (ba * dl[None]).sum(-1)
        baoa = (ba * oa).sum(-1)
        rdoa = (dl[None] * oa).sum(-1)
        oaoa = (oa * oa).sum(-1)
        r = self.cr[:, None]
        a = baba - bard * bard
        b = baba * rdoa - baoa * bard
        c = baba * oaoa - baoa * baoa - r * r * baba
        h = b * b - a * c
        tc = (-b - torch.sqrt(h.clamp_min(0))) / a.clamp_min(1e-12)
        y = baoa + tc * bard
        body_ok = (h >= 0) & (y > 0) & (y < baba) & (tc > 1e-4)
        ocap = torch.where((y <= 0)[..., None], oa, ol[None] - self.cb[:, None, :])
        b2 = (dl[None] * ocap).sum(-1)
        c2 = (ocap * ocap).sum(-1) - r * r
        h2 = b2 * b2 - c2
        tcap = -b2 - torch.sqrt(h2.clamp_min(0))
        cap_ok = (h >= 0) & ~body_ok & (h2 > 0) & (tcap > 1e-4)
        tcs = torch.where(body_ok, tc, torch.where(cap_ok, tcap, torch.full_like(tc, float("inf"))))
        tcm, kc = tcs.min(0)
        # wheels (finite cylinders)
        ba = (self.wb - self.wa)[:, None, :]
        oc = ol[None] - self.wa[:, None, :]
        baba = (ba * ba).sum(-1)
        bard = (ba * dl[None]).sum(-1)
        baoc = (ba * oc).sum(-1)
        r = self.wr[:, None]
        k2 = baba - bard * bard
        k1 = baba * (oc * dl[None]).sum(-1) - baoc * bard
        k0 = baba * (oc * oc).sum(-1) - baoc * baoc - r * r * baba
        hh = k1 * k1 - k2 * k0
        sq = torch.sqrt(hh.clamp_min(0))
        tw = (-k1 - sq) / k2.clamp_min(1e-12)
        yw = baoc + tw * bard
        side_ok = (hh >= 0) & (yw > 0) & (yw < baba) & (tw > 1e-4)
        den = torch.where(bard.abs() < 1e-9, torch.full_like(bard, 1e-9), bard)
        tcapw = (torch.where(yw < 0, torch.zeros_like(yw), baba) - baoc) / den
        capw_ok = (hh >= 0) & ~side_ok & ((k1 + k2 * tcapw).abs() < sq) & (tcapw > 1e-4)
        tws = torch.where(side_ok, tw, torch.where(capw_ok, tcapw, torch.full_like(tw, float("inf"))))
        twm, kw = tws.min(0)
        t = torch.minimum(torch.minimum(tbm, tcm), twm)
        if any_hit:
            return t, None, None
        fin = torch.isfinite(t)
        n = torch.zeros(M, 3, device=dev)
        col = torch.zeros(M, 3, device=dev)
        ar = torch.arange(M, device=dev)
        isb = fin & (tbm <= t)
        isc = fin & ~isb & (tcm <= t)
        isw = fin & ~isb & ~isc
        p = ol + dl * torch.where(fin, t, torch.zeros_like(t))[:, None]
        axb = ax[kb, ar]
        nb = torch.zeros(M, 3, device=dev)
        nb[ar, axb] = -torch.sign(dl[ar, axb])
        n = torch.where(isb[:, None], nb, n)
        col = torch.where(isb[:, None], self.bcol[kb], col)
        A, B = self.ca[kc], self.cb[kc]
        BA = B - A
        hproj = (((p - A) * BA).sum(-1) / (BA * BA).sum(-1).clamp_min(1e-12)).clamp(0, 1)
        nc = p - A - BA * hproj[:, None]
        n = torch.where(isc[:, None], nc / nc.norm(dim=1, keepdim=True).clamp_min(1e-9), n)
        col = torch.where(isc[:, None], self.ccol[kc], col)
        A, B = self.wa[kw], self.wb[kw]
        BA = B - A
        yy = ((p - A) * BA).sum(-1) / (BA * BA).sum(-1)
        radial = p - A - BA * yy[:, None]
        on_side = (yy > 1e-3) & (yy < 1 - 1e-3)
        axis = BA / BA.norm(dim=1, keepdim=True)
        nw = torch.where(on_side[:, None], radial / radial.norm(dim=1, keepdim=True).clamp_min(1e-9),
                         axis * torch.where(yy < 0.5, -1.0, 1.0)[:, None])
        ang = torch.atan2(radial[:, 2], radial[:, 0]) + self._phase_t
        groove = 0.55 + 0.45 * (torch.cos(ang * 16) > -0.2).float()
        hub = 0.75 + 0.25 * (radial.norm(dim=1) < WHEEL_R * 0.55).float()
        wc = self._alu[None] * torch.where(on_side, groove, hub)[:, None]
        n = torch.where(isw[:, None], nw, n)
        col = torch.where(isw[:, None], wc, col)
        return t, n, col

    def _bound(self, o, d):
        """Rays (world) that pass within the bounding sphere."""
        c = _t(self.center, self.dev)
        oc = (c - o) if o.ndim == 1 else (c[None] - o)
        tca = (d @ oc) if o.ndim == 1 else (oc * d).sum(-1)
        oo = (oc * oc).sum() if o.ndim == 1 else (oc * oc).sum(-1)
        return (oo - tca * tca < self.radius ** 2) & (tca > -self.radius)

    def hit(self, o, d, depth, dense=False):
        """Nearest rover hit in front of `depth`: (t (M,), world normal (M,3), albedo (M,3), mask (M,)). `dense`
        intersects every ray (no host sync; for the eye's 10k rays); otherwise only rays near the rover."""
        M = d.shape[0]
        dev = self.dev
        if dense:
            ol, dl = self._local(o[None].expand(M, 3) if o.ndim == 1 else o, d)
            t, n, col = self._intersect(ol, dl)
            mask = torch.isfinite(t) & (t < depth)
            return t, n @ self._R_t.T, col, mask
        t_all = torch.full((M,), float("inf"), device=dev)
        n_all = torch.zeros(M, 3, device=dev)
        c_all = torch.zeros(M, 3, device=dev)
        cand = torch.nonzero(self._bound(o, d)).flatten()
        if cand.numel() == 0:
            return t_all, n_all, c_all, torch.zeros(M, dtype=torch.bool, device=dev)
        oo = o[None].expand(len(cand), 3) if o.ndim == 1 else o[cand]
        ol, dl = self._local(oo, d[cand])
        t, n, col = self._intersect(ol, dl)
        t_all[cand] = t
        n_all[cand] = n @ self._R_t.T
        c_all[cand] = col
        mask = torch.isfinite(t_all) & (t_all < depth)
        return t_all, n_all, c_all, mask

    def shadow(self, p, sun, dense=False):
        """0 where the rover blocks the sun from p (world points), else 1."""
        M = p.shape[0]
        lit = torch.ones(M, device=self.dev)
        d = sun[None].expand(M, 3)
        if dense:
            ol, dl = self._local(p, d)
            t, _, _ = self._intersect(ol, dl, any_hit=True)
            return torch.where(torch.isfinite(t), torch.zeros_like(t), lit)
        cand = torch.nonzero(self._bound(p, d)).flatten()
        if cand.numel() == 0:
            return lit
        ol, dl = self._local(p[cand], d[cand])
        t, _, _ = self._intersect(ol, dl, any_hit=True)
        lit[cand] = torch.where(torch.isfinite(t), torch.zeros_like(t), torch.ones_like(t))
        return lit


# ======================================================================================================== rules (pure; tested)
GF_HZ = 33.0                  # flyverse/body.py Flight.gf_hz: the shipped body model's takeoff threshold
LOOM_CELLS = ("LPLC2", "LC4")
WARM_S = 1.0                  # GAME: the rover waits this long before it starts (the brain settles on the scene)
V_CRUISE = 5.0                # GAME: autopilot cruise speed, m/s
A_DRIVE = 1.5                 # GAME: acceleration limit, m/s^2
A_BRAKE = 6.0                 # GAME: braking deceleration of a hazard stop, m/s^2
V_REVERSE = 1.2               # GAME: reverse speed after a stop or a collision, m/s
REVERSE_S = 1.0               # GAME: reverse duration after a hazard stop, s
AP_LOOKAHEAD = 12.0           # GAME: autopilot pure-pursuit look-ahead along the route, m
AP_GAIN = 1.0                 # GAME: autopilot yaw rate per rad of heading error while cruising, 1/s
AP_MAX = 24.0 * DEG           # GAME: autopilot yaw-rate limit, rad/s; above the wind steer's cap (WIND_MAX), so the
                              # autopilot can always bring the rover back to the route (the rule that ends a tug of war)
# GAME: going round a rock the rover hit (class Detour): back up straight until the footprint is BACK_CLEAR from it (at
# most BACK_MAX_S), pivot in place to PIVOT_DEG off the route towards the side the rover's centre is on, drive out at
# up to V_SIDE until the footprint will clear the rock by DETOUR_MARGIN, pivot back parallel to the route, and drive
# past (pursuing that lane: look-ahead DETOUR_LOOKAHEAD, gain DETOUR_GAIN, limit DETOUR_MAX) until the rear is past it
BACK_CLEAR = 1.2
BACK_MAX_S = 3.0
PIVOT_DEG = 55.0
PIVOT_RATE = 45.0 * DEG
V_SIDE = 2.0
DETOUR_LOOKAHEAD = 6.0
DETOUR_GAIN = 1.5
DETOUR_MAX = 14.0 * DEG
DETOUR_MARGIN = 0.8
# the dust devils' wind at the fly's antennae (GAME physics; see devil_wind and antenna_deflections)
MARS_RHO, EARTH_RHO = 0.020, 1.225   # kg/m^3: air density at Jezero's surface and at Earth's sea level
WIND_EQUIV = math.sqrt(MARS_RHO / EARTH_RHO)   # 0.128: the Earth wind with the same dynamic pressure, per m/s on Mars
JO_FULL_SPEED = 0.5           # flyverse.air.Air.deflections' default: full antennal deflection at 0.5 m/s (Earth air)
WIND_TAU = 0.2                # DECODER: EMA of the wind-DN index u, s
WIND_DEAD = 8.0               # DECODER: dead band on |u_s|, Hz (dev seeds >= 100: still air kept |u_s| <= 3.8 Hz)
WIND_GAIN = 1.5 * DEG         # DECODER: yaw rate per Hz of u_s beyond the dead band, rad/s per Hz
WIND_MAX = 18.0 * DEG         # DECODER: yaw-rate limit, rad/s; below AP_MAX (see there)
ONSET_GAP_S = 0.5             # log: a wind-steer onset follows >= 0.5 s without the decoder active (no chatter)
VEER_FRAC = 0.6               # log + banner: a VEER is the decoder's command passing 0.6 of its cap ...
VEER_REARM_S = 2.0            # ... after >= 2 s below it
ROUTE_END = 170.0             # GAME: the waypoint (x, m) on the route line y = 0; the run ends there
LANE_HALF = 8.0               # GAME: scenery rocks are kept this far (m) from the route line; only hazards sit on it
ENCOUNTER_M = 45.0            # log: the dust-devil encounter is while D3's centre is within this distance of the rover


def steer_applied(dec_yaw, recovering):
    """GAME arbitration of the wind steer's command (rad/s): applied in full while the rover drives forward, held (0)
    while it recovers (backing up, or going round a rock it hit), so the decoder can never hold it against a rock. Returns (applied yaw rate, held: the decoder commanded a turn that this rule withheld)."""
    if recovering:
        return 0.0, dec_yaw != 0.0
    return dec_yaw, False


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def autopilot_yaw_rate(x, y, yaw, target_y=0.0, lookahead=AP_LOOKAHEAD, gain=AP_GAIN, limit=AP_MAX):
    """GAME: pure pursuit of the line y = target_y (the route runs along +x); a yaw rate in rad/s. The limit is above
    the wind steer's, so their sum always leaves the autopilot the last word."""
    desired = math.atan2(target_y - y, lookahead)
    return float(np.clip(gain * wrap(desired - yaw), -limit, limit))


class Ema:
    def __init__(self, tau_s):
        self.tau_s, self.y = tau_s, None

    def __call__(self, x, dt_s):
        if self.y is None:
            self.y = float(x)
        else:
            a = 1.0 - math.exp(-dt_s / self.tau_s) if self.tau_s > 0 else 1.0
            self.y += a * (float(x) - self.y)
        return self.y


WIND_CELLS = {"DNp18_L": {"type": "DNp18", "somaSide": "L"}, "DNp18_R": {"type": "DNp18", "somaSide": "R"},
              "DNp33_L": {"type": "DNp33", "somaSide": "L"}, "DNp33_R": {"type": "DNp33", "somaSide": "R"}}


def wind_index(p18L, p18R, p33L, p33R):
    """The wind descending neurons' left-right index (games/hairdryer.py's): u = 1/2 [(DNp18 L-R) - (DNp33 L-R)],
    Hz; > 0 when the wind comes from the left (DNp18 fires on the wind side, DNp33 on the other)."""
    return 0.5 * ((p18L - p18R) - (p33L - p33R))


class WindSteerLaw:
    """DECODER, wind steer: u = the wind-DN index, u_s = EMA(u, tau); beyond a dead band it commands a yaw rate
    TOWARDS the side u reports (+ = a left turn), i.e. into the wind. With the dust devil's inward-spiralling surface
    wind that is away from the devil while the rover approaches it (measured and reported, not assumed)."""

    def __init__(self, gain=WIND_GAIN, dead_hz=WIND_DEAD, tau_s=WIND_TAU, limit=WIND_MAX):
        self.gain, self.dead, self.tau_s, self.limit = gain, dead_hz, tau_s, limit
        self.lp = Ema(tau_s)

    def parameters(self):
        return {"cells": "DNp18 L/R, DNp33 L/R (one cell each; rate_hz, the brain's 100 ms running rate estimate)",
                "index": "u = 1/2 [(DNp18_L - DNp18_R) - (DNp33_L - DNp33_R)]", "ema_tau_s": self.tau_s,
                "dead_band_Hz": self.dead, "gain_deg_s_per_Hz": round(self.gain / DEG, 3),
                "limit_deg_s": round(self.limit / DEG, 2),
                "sign": "u > 0 (wind reported on the left) -> positive yaw rate (turn left, into the wind)"}

    def from_rates(self, p18L, p18R, p33L, p33R, dt_s):
        u = wind_index(p18L, p18R, p33L, p33R)
        u_s = self.lp(u, dt_s)
        excess = math.copysign(max(abs(u_s) - self.dead, 0.0), u_s)
        raw = self.gain * excess
        yaw_rate = float(np.clip(raw, -self.limit, self.limit))
        return {"u": float(u), "u_s": float(u_s), "yaw_rate": yaw_rate, "active": excess != 0.0,
                "at_cap": abs(raw) > self.limit,
                "p18L": float(p18L), "p18R": float(p18R), "p33L": float(p33L), "p33R": float(p33R)}

    def __call__(self, dt_ms, x):
        r = [float(x[k].float().mean().item()) for k in WIND_CELLS]
        return self.from_rates(*r, dt_ms / 1000.0)


class StopLaw:
    """DECODER, hazard stop: the giant fibre DNp01 (mean rate of both cells, the value `fb.motor().gf` reports and the
    shipped body model thresholds) >= 33 Hz -> the rover's hazard stop."""

    def __init__(self, threshold_hz=GF_HZ):
        self.threshold = threshold_hz

    def parameters(self):
        return {"cells": "DNp01 (both)", "threshold_Hz": self.threshold, "source": "flyverse.body.Flight.gf_hz"}

    def __call__(self, dt_ms, x):
        gf = float(x["gf"].float().mean().item())
        return {"gf": gf, "fire": gf >= self.threshold}


class DriveState:
    """GAME: the rover's speed schedule. 'drive' accelerates to cruise; a hazard stop brakes to rest ('stop'), then
    backs up ('reverse') and drives on; a collision stops dead and hands the speed to a Detour ('detour')."""

    def __init__(self, v_cruise=V_CRUISE):
        self.state, self.v, self.timer, self.v_cruise = "wait", 0.0, 0.0, v_cruise

    def hazard_stop(self) -> bool:
        if self.state == "drive" and self.v > 0.3:
            self.state = "stop"
            return True
        return False

    def collide(self):
        self.state, self.v, self.timer = "detour", 0.0, 0.0

    def update(self, dt, t_s):
        if self.state == "wait":
            if t_s >= WARM_S:
                self.state = "drive"
        elif self.state == "drive":
            self.v = min(self.v_cruise, self.v + A_DRIVE * dt)
        elif self.state == "stop":
            self.v = max(0.0, self.v - A_BRAKE * dt)
            if self.v == 0.0:
                self.state, self.timer = "reverse", REVERSE_S
        elif self.state == "reverse":
            self.v = -V_REVERSE
            self.timer -= dt
            if self.timer <= 0:
                self.state, self.v = "drive", 0.0
        return self.v                                   # 'detour': the Detour sets v


class Detour:
    """GAME: going round a rock the rover hit, as a rover that can turn in place does it (constants above). `step`
    returns the commanded speed and yaw rate for this tick and whether the detour is over; the rover then rejoins
    the route under the autopilot."""

    def __init__(self, rock, x, y, yaw):
        self.rock = rock
        self.side = 1.0 if y >= rock["y"] else -1.0
        self.lane = rock["y"] + self.side * (rock["r"] + ROVER_W / 2 + DETOUR_MARGIN)
        self.phase, self.t = "back", 0.0

    def _pivot(self, yaw, target, dt, next_phase):
        err = wrap(target - yaw)
        if abs(err) <= PIVOT_RATE * dt:
            self.phase = next_phase
            return 0.0, err / dt
        return 0.0, math.copysign(PIVOT_RATE, err)

    def step(self, dt, x, y, yaw, v, gap):
        """(speed m/s, yaw rate rad/s, done). gap: the footprint's clearance to the rock now, m."""
        self.t += dt
        out = self.side * (y - self.lane) >= 0              # the centre is on the lane (or beyond it)
        if self.phase == "back":
            if gap < BACK_CLEAR and self.t < BACK_MAX_S:
                return -V_REVERSE, 0.0, False
            self.phase = "pivot_back" if out else "pivot_out"
            return 0.0, 0.0, False
        if self.phase == "pivot_out":
            return (*self._pivot(yaw, self.side * PIVOT_DEG * DEG, dt, "side"), False)
        if self.phase == "side":
            if not out:
                return min(V_SIDE, v + A_DRIVE * dt), 0.0, False
            self.phase = "pivot_back"
            return 0.0, 0.0, False
        if self.phase == "pivot_back":
            return (*self._pivot(yaw, 0.0, dt, "pass"), False)
        if x - ROVER_L / 2 > self.rock["x"] + self.rock["r"]:
            return v, 0.0, True
        return (min(V_CRUISE, v + A_DRIVE * dt),
                autopilot_yaw_rate(x, y, yaw, self.lane, DETOUR_LOOKAHEAD, DETOUR_GAIN, DETOUR_MAX), False)


def rect_circle_clearance(cx, cy, yaw, half_l, half_w, px, py, r):
    """Signed gap (m) between an oriented rectangle (centre, yaw, half sizes) and a circle; < 0 means overlap."""
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = px - cx, py - cy
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    qx, qy = abs(lx) - half_l, abs(ly) - half_w
    outside = math.hypot(max(qx, 0.0), max(qy, 0.0))
    inside = min(max(qx, qy), 0.0)
    return outside + inside - r


def rover_step(x, y, yaw, v, yaw_rate, dt):
    """GAME kinematics: a skid-steered rover on the ground plane."""
    yaw = wrap(yaw + yaw_rate * dt)
    return x + v * math.cos(yaw) * dt, y + v * math.sin(yaw) * dt, yaw


# The boulder field (GAME): the same for every seed. (x along the route, y offset from the line, size m). Two
# boulders, an open stretch of ~110 m where dust devil D3 passes the route, then two more.
HAZARDS = [(20.0, 0.5, 1.15), (33.0, -0.6, 1.25), (150.0, 0.2, 1.3), (163.0, -0.3, 1.2)]
LAYOUT_SEED = 0x3A25                     # the rocks' shapes and the scenery come from this, never from --seed


def build_rocks(scene: MarsScene, n_scenery=70):
    rng = np.random.default_rng(LAYOUT_SEED)
    rocks = []
    for k, (x, y, size) in enumerate(HAZARDS):
        rocks.append(make_rock(rng, (x, y, scene.height_np(x, y)), size, dark=True, hazard=True, name=f"H{k + 1}"))
    placed = 0
    while placed < n_scenery:
        x = rng.uniform(-40, 260)
        y = rng.uniform(-70, 70)
        if abs(y) < LANE_HALF and -5 < x < ROUTE_END + 10:    # keep the lane clear except for the hazards
            continue
        size = float(np.clip(rng.lognormal(math.log(0.45), 0.55), 0.15, 2.4))
        rocks.append(make_rock(rng, (x, y, scene.height_np(x, y)), size, dark=rng.random() < 0.7))
        placed += 1
    return rocks


# Dust devils (GAME): look (w0 core width m, flare m/m, height m, opacity) and wind (peak tangential speed vt m/s at
# the core radius rc m, decay length m outside it, spin +1 = counter-clockwise seen from above, inflow angle deg).
# D1 and D2 are far scenery; D3 drifts along the route 16 m to its right, in the open stretch between H2 and H3, and
# turns counter-clockwise seen from above, so the rover meets its wind from the left on the approach.
DEVILS = [{"name": "D1", "xy0": (420.0, 330.0), "vel": (-2.2, -0.6), "w0": 5.0, "flare": 0.22, "height": 260.0,
           "opacity": 0.55, "vt": 14.0, "rc": 6.0, "decay_m": 20.0, "spin": 1, "inflow_deg": 35.0},
          {"name": "D2", "xy0": (150.0, -380.0), "vel": (1.2, 1.4), "w0": 3.5, "flare": 0.18, "height": 180.0,
           "opacity": 0.45, "vt": 12.0, "rc": 5.0, "decay_m": 20.0, "spin": -1, "inflow_deg": 35.0},
          {"name": "D3", "xy0": (84.0, -16.0), "vel": (0.55, 0.0), "w0": 3.8, "flare": 0.06, "height": 260.0,
           "opacity": 0.9, "vt": 20.0, "rc": 7.0, "decay_m": 20.0, "spin": 1, "inflow_deg": 35.0}]


def devil_specs():
    out = []
    for d in DEVILS:
        x0, y0 = d["xy0"]
        vx, vy = d["vel"]
        spec = dict(d)
        spec["pos"] = (lambda t, x0=x0, y0=y0, vx=vx, vy=vy: (x0 + vx * t, y0 + vy * t))
        out.append(spec)
    return out


def devil_wind(xy, devils, t_s):
    """GAME: the dust devils' surface wind (m/s, world x/y) at a point: each is a vortex whose tangential speed rises
    linearly inside its core (radius rc) to vt and falls outside as vt (rc / r) exp(-(r - rc) / decay), with an inward
    spiral (the surface air converges on the devil: wind = v_t (e_theta - tan(inflow) e_r))."""
    w = np.zeros(2)
    for dv in devils:
        cx, cy = dv["pos"](t_s)
        rx, ry = xy[0] - cx, xy[1] - cy
        r = math.hypot(rx, ry)
        if r < 1e-6:
            continue
        er = np.array([rx / r, ry / r])
        et = dv["spin"] * np.array([-er[1], er[0]])
        rc = dv["rc"]
        vt = dv["vt"] * (r / rc if r < rc else rc / r * math.exp(-(r - rc) / dv["decay_m"]))
        w += vt * (et - math.tan(dv["inflow_deg"] * DEG) * er)
    return w


def devil_wind_many(xy, devils, t_s):
    """devil_wind for many points at once: xy (N, 2) -> (N, 2) m/s (the display's wind-borne dust)."""
    xy = np.asarray(xy, float)
    w = np.zeros_like(xy)
    for dv in devils:
        cx, cy = dv["pos"](t_s)
        rel = xy - np.array([cx, cy])
        r = np.maximum(np.hypot(rel[:, 0], rel[:, 1]), 1e-6)
        er = rel / r[:, None]
        et = dv["spin"] * np.stack([-er[:, 1], er[:, 0]], 1)
        rc = dv["rc"]
        vt = dv["vt"] * np.where(r < rc, r / rc, rc / r * np.exp(-(r - rc) / dv["decay_m"]))
        w += vt[:, None] * (et - math.tan(dv["inflow_deg"] * DEG) * er)
    return w


def wind_from_deg(w_xy, yaw):
    """Where the wind comes FROM relative to the heading (deg, + = from the left, 0 = dead ahead)."""
    return math.degrees(wrap(math.atan2(-w_xy[1], -w_xy[0]) - yaw))


def antenna_deflections(w_xy, yaw, air=None):
    """GAME -> shipped sense: the Mars wind w_xy (m/s) as the Earth wind of the same dynamic pressure (x WIND_EQUIV),
    through the shipped antenna model (flyverse.air.Air.deflections, antennae at +-45 deg, full deflection at 0.5 m/s)
    for a fly facing `yaw`. Returns (dL, dR) for fb.wind."""
    from flyverse.air import Air, WindParams
    if air is None:
        air = Air([], WindParams(speed=0.0, direction_deg=0.0, meander_deg=0.0))
    we = np.asarray(w_xy, float) * WIND_EQUIV
    air.wind.speed = float(math.hypot(we[0], we[1]))
    air.wind.direction_deg = float(math.degrees(math.atan2(we[1], we[0])))
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    dL, dR = air.deflections(fwd, left, JO_FULL_SPEED)
    return float(dL[0]), float(dR[0])


class HazardTally:
    """GAME scoring of the route's boulders. A hazard is PASSED when the rover's rear clears the rock's far side
    along the route; it is a COLLISION if the footprint ever touched it, else CLEAN. Hazard stops are credited to the
    nearest hazard ahead within `stop_window_m` of the rover's front."""

    def __init__(self, rocks, stop_window_m=12.0):
        # min_clear_m is over the poses the rover actually took: a pose that would overlap a rock is rejected (the
        # collision rule), so it is >= 0 even for a rock it hit; collision_gap_m keeps the rejected pose's overlap
        self.rows = [{"name": r.name, "x": float(r.pos[0]), "y": float(r.pos[1]), "r": r.r_ground, "contacts": 0,
                      "stops": 0, "min_clear_m": float("inf"), "collision_gap_m": None, "outcome": None,
                      "t_passed": None} for r in rocks if r.hazard]
        self.stop_window = stop_window_m

    def ahead(self, x_front):
        live = [h for h in self.rows if h["outcome"] is None and h["x"] + h["r"] > x_front - ROVER_L]
        return min(live, key=lambda h: h["x"]) if live else None

    def observe(self, t_s, x, y, yaw):
        """Update clearances; return the hazards passed this tick."""
        done = []
        for h in self.rows:
            if h["outcome"] is not None or abs(h["x"] - x) > 25:
                continue
            gap = rect_circle_clearance(x, y, yaw, ROVER_L / 2, ROVER_W / 2, h["x"], h["y"], h["r"])
            h["min_clear_m"] = min(h["min_clear_m"], gap)
            if x - ROVER_L / 2 > h["x"] + h["r"]:
                h["outcome"] = "collision" if h["contacts"] else "clean"
                h["t_passed"] = round(t_s, 2)
                done.append(h)
        return done

    def credit_stop(self, x_front):
        h = self.ahead(x_front)
        if h is not None and h["x"] - h["r"] - x_front <= self.stop_window:
            h["stops"] += 1
            return h
        return None

    def summary(self):
        passed = [h for h in self.rows if h["outcome"] is not None]
        return {"hazards": len(self.rows), "passed": len(passed),
                "passed_clean": sum(h["outcome"] == "clean" for h in passed),
                "passed_after_collision": sum(h["outcome"] == "collision" for h in passed),
                "contacts": sum(h["contacts"] for h in self.rows),
                "hazard_stops_credited": sum(h["stops"] for h in self.rows),
                "not_reached": [h["name"] for h in self.rows if h["outcome"] is None],
                "per_hazard": [{k: (round(v, 3) if isinstance(v, float) and math.isfinite(v) else v)
                                for k, v in h.items()} for h in self.rows]}


# ======================================================================================================== the game
DESIGN = (1920, 1080)
VIEW = (0, 64, 1536, 864)                      # main view rect on the design canvas
RCOL = 1536                                    # right column x
BANNER_S = 1.3
TRACE_S = 6.0


def _mean(x):
    return float(np.mean(x)) if len(x) else 0.0


def banner_text(font, text, color):
    """Render a banner line; a '→' is drawn as an arrow shape (display fonts often lack the glyph; as games/doom.py)."""
    import pygame
    parts = text.split("→")
    imgs = [font.render(p.strip(), True, color) for p in parts]
    if len(imgs) == 1:
        return imgs[0]
    hgt = max(i.get_height() for i in imgs)
    aw = int(hgt * 0.9)
    out = pygame.Surface((sum(i.get_width() for i in imgs) + aw * (len(imgs) - 1), hgt), pygame.SRCALPHA)
    x = 0
    for k, img in enumerate(imgs):
        out.blit(img, (x, (hgt - img.get_height()) // 2))
        x += img.get_width()
        if k < len(imgs) - 1:
            cy, th = hgt // 2, max(3, hgt // 11)
            pygame.draw.rect(out, color, (x + aw * 0.18, cy - th // 2, aw * 0.45, th))
            pygame.draw.polygon(out, color, [(x + aw * 0.55, cy - hgt * 0.22), (x + aw * 0.86, cy),
                                             (x + aw * 0.55, cy + hgt * 0.22)])
            x += aw
    return out


class MarsRover(gc.Game):
    title = "mars"
    subtitle = "a fly connectome rides a rover across a Jezero-like crater"

    def __init__(self, args):
        super().__init__(args)
        self.apply = set() if args.control == "off" else set(a for a in args.apply.split(",") if a)
        self.control = args.control
        self.hide_hazards = args.control == "hidden"        # the hazard boulders are removed from the fly's scene only
        self.deliver_wind = args.control != "nowind"        # nowind: fb.wind(0, 0), still air, every tick
        self.eye_mount = getattr(args, "eye", None) or EYE_MOUNT
        self.eye_local = np.array(EYE_MOUNTS[self.eye_mount], float)
        self.fb = gc.build_brain(args)
        fb = self.fb
        dev = fb.device
        self.dev = dev
        self.eyes = gc.Eyes(fb)
        self.scene = MarsScene(dev)
        self.rocks = build_rocks(self.scene)
        self.scene.set_rocks(self.rocks)
        self.scene.devils = self.devils = devil_specs()
        self.scene.tracks = Tracks(dev)                         # display: sampled by the camera only
        self.rover = Rover(dev)
        self.hazards = [r for r in self.rocks if r.hazard]
        self.tally = HazardTally(self.rocks)
        from flyverse.air import Air, WindParams
        self.air = Air([], WindParams(speed=0.0, direction_deg=0.0, meander_deg=0.0))   # the shipped antenna model
        # the two decoders: attached, read-only (kind='decoder'), recorded in module_records()
        self.wind_law = WindSteerLaw()
        self.wind_dec = gc.ReadDecoder(
            "wind_steer", WIND_CELLS, self.wind_law,
            law=("yaw rate = %.1f deg/s per Hz x (u_s beyond a %.0f Hz dead band), u_s = EMA %.1f s of u = 1/2 "
                 "[(DNp18 L-R) - (DNp33 L-R)]; clipped at %.0f deg/s; + = a left turn: towards the side the wind "
                 "DNs report (into the wind)" % (WIND_GAIN / DEG, WIND_DEAD, WIND_TAU, WIND_MAX / DEG)),
            parameters={**self.wind_law.parameters(), "applied": "steer" in self.apply})
        self.stop_law = StopLaw()
        self.stop_dec = gc.ReadDecoder(
            "hazard_stop", {"gf": {"type": "DNp01"}}, self.stop_law,
            law="hazard stop when the giant fibres' (DNp01) mean rate >= 33 Hz: brake at %.0f m/s^2, back up %.1f s" % (
                A_BRAKE, REVERSE_S),
            parameters={**self.stop_law.parameters(), "applied": "stop" in self.apply})
        fb.attach(self.wind_dec)
        fb.attach(self.stop_dec)
        # CONNECTOME readouts for the HUD and the log (display only)
        c = fb.c
        side = c.neurons.somaSide.fillna("").to_numpy()
        groups = {}
        for t in LOOM_CELLS + ("DNp01", "MDN"):
            idx = c.select(type=t)
            for s in "LR":
                groups[f"{t}_{s}"] = idx[side[idx] == s]
        self.read_names = list(groups)
        self.read_seg = np.cumsum([0] + [len(groups[k]) for k in self.read_names])
        self.read_idx = torch.as_tensor(np.concatenate([groups[k] for k in self.read_names]), device=dev)
        self.rates = {k: 0.0 for k in self.read_names}
        # rover state (GAME)
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.drive = DriveState()
        self.dist = 0.0
        self.detour = None                         # (rock, target_y) while going round a rock it hit
        self.gf_armed = True
        self.steer_was_active = False
        self.steer_last_active_t = -1e9            # an onset needs >= ONSET_GAP_S of inactivity before it
        self.yaw_rate_dec = 0.0
        self.yaw_rate_ap = 0.0
        self.wind = np.zeros(2)                    # the devils' wind at the rover (Mars m/s, world)
        self.defl = (0.0, 0.0)                     # what fb.wind got this tick
        self.trail = []
        self.pose_R = np.eye(3)
        # the eye's trace inputs, in fixed device buffers (see _eye_radiance)
        self._eye_dirs = torch.as_tensor(self.eyes.dirs_body.reshape(-1, 3), device=dev)
        self._eye_o = torch.zeros(3, device=dev)
        self._eye_R = torch.eye(3, device=dev)
        self._eye_t = torch.zeros((), device=dev)
        self._eye_dxy = torch.zeros(len(self.devils), 2, device=dev)
        self._eye_graph, self._eye_out, self._eye_graph_state = None, None, "untried"
        self._update_pose()
        self.rad = None
        n_hist = int(TRACE_S * 100)
        self.hist = {k: [0.0] * n_hist for k in ("loomL", "loomR", "gf", "u", "u_s", "yaw")}
        # gf_crossings counts every DNp01 >= 33 Hz crossing, applied or not; hazard_stops_applied only those that
        # braked the rover. steer_onsets and veers likewise count the decoder's own events, applied or not.
        # Overrides of the wind steer: steer_held_* (the GAME collision-recovery rule withheld a commanded turn);
        # steer_cap_ticks (the decoder's own declared cap clipped its command; part of its law, counted anyway).
        self.stats = {"gf_crossings": 0, "hazard_stops_applied": 0, "collisions": 0, "steer_onsets": 0, "veers": 0,
                      "veers_applied": 0,
                      "steer_active_ticks": 0, "gf_max": 0.0, "mdn_max": 0.0, "mdn_max_t_lt_1s": 0.0,
                      "loomL_max": 0.0, "loomR_max": 0.0, "u_s_min": 0.0, "u_s_max": 0.0,
                      "wind_equiv_max_m_s": 0.0, "yaw_dec_abs_max_deg_s": 0.0, "steer_held_ticks": 0,
                      "steer_held_episodes": 0, "steer_held_abs_deg": 0.0, "steer_cap_ticks": 0,
                      "steer_applied_abs_deg": 0.0}
        self.held_prev = False
        self.veer_high_t = -1e9                    # last tick the command was >= VEER_FRAC of the cap
        self.held = False
        # the encounter with D3 (and every devil's closest approach): facts for the caption, GAME geometry
        self.devil_min = {dv["name"]: {"distance_m": float("inf"), "t_s": None, "rover_y_m": None}
                          for dv in self.devils}
        self.enc = {"t_in": None, "t_out": None, "y_min": 0.0, "y_max": 0.0, "yaw_min_deg": 0.0, "yaw_max_deg": 0.0,
                    "dec_applied_deg": 0.0, "dec_left_deg": 0.0, "dec_right_deg": 0.0, "held_ticks": 0}
        self.series = []                            # 10 Hz: SERIES_COLS
        self.lanes = {}
        self.dust = np.zeros((0, 7), np.float32)   # x y z vx vy vz age
        self.rng = np.random.default_rng([LAYOUT_SEED, args.seed])   # dust only (display)
        self.windp = np.zeros((0, 4), np.float32)  # wind-borne dust streaks (display): x y z age
        self.cam_state = None
        self.cam_yaw = None                        # the chase camera's smoothed heading (display)
        self._frame = None
        self._layers = {}
        self._declare()
        self.log.event(0.0, "start", control=self.control, applied=sorted(self.apply), hazards=len(self.hazards),
                       v_cruise=V_CRUISE, eye_mount=self.eye_mount, wind_delivered=self.deliver_wind,
                       hazards_hidden_from_the_fly=self.hide_hazards)

    # ------------------------------------------------------------------ provenance
    def _declare(self):
        m = self.log.meta
        m["mars"] = {"hazards": [{"name": r.name, "x": round(float(r.pos[0]), 2), "y": round(float(r.pos[1]), 2),
                                  "r_ground": round(r.r_ground, 3), "top_m": round(r.top - self.scene.height_np(
                                      r.pos[0], r.pos[1]), 3)} for r in self.hazards],
                     "devils": [{k: v for k, v in d.items() if k != "pos"} for d in self.devils],
                     "layout_seed": LAYOUT_SEED, "control": self.control, "applied": sorted(self.apply),
                     "hazards_hidden_from_the_fly": self.hide_hazards, "wind_delivered": self.deliver_wind,
                     "eye_mount": self.eye_mount, "eye_local_m": self.eye_local.tolist(),
                     "rover_footprint_m": [ROVER_L, ROVER_W], "wind_equiv": round(WIND_EQUIV, 4)}
        L = self.log
        gc.declare(L, "eyes: on a post above the mast's camera head", "game",
                   "the fly's 1,466 x 7 ommatidial rays start %.2f m above the ground (rover at rest), on a %.0f cm post "
                   "above the mast's camera head, facing the rover's heading, and are ray-traced every 10 ms through the "
                   "camera's scene: the same terrain, rocks, rover, dust devils and sky (the sun's halo but not its "
                   "%.2f deg disc), marched coarser (170 steps vs 300), without the wheel dust, the wheel tracks and the "
                   "wind-borne dust streaks (display only); linear RGB -> radiance with UV = 0.5 B"
                   % (self.eye_local[2], POST_H * 100, SUN_RADIUS_DEG), eye_local_m=self.eye_local.tolist(),
                   mount=self.eye_mount)
        d3 = [d for d in self.devils if d["name"] == "D3"][0]
        gc.declare(L, "dust devils' wind", "game",
                   "each devil is a vortex: tangential speed vt x r / rc inside its core, vt (rc / r) exp(-(r - rc) / "
                   "decay) outside, spiralling inward at the inflow angle; D3 (vt %.0f m/s, rc %.0f m, decay %.0f m, "
                   "%s seen from above) starts at (%.0f, %.0f) m and drifts along the route at %.2f m/s, %.0f m to its "
                   "right. No ambient wind; the rover's own motion is not delivered as wind" % (
                       d3["vt"], d3["rc"], d3["decay_m"], "counter-clockwise" if d3["spin"] > 0 else "clockwise",
                       d3["xy0"][0], d3["xy0"][1], d3["vel"][0], -d3["xy0"][1]),
                   devils=[{k: v for k, v in d.items() if k != "pos"} for d in self.devils])
        gc.declare(L, "antennae", "game",
                   "the devils' wind at the rover, as the Earth wind of the same dynamic pressure (x %.3f = sqrt(%.3f / "
                   "%.3f kg/m^3)), through the shipped antenna model flyverse.air.Air.deflections (full deflection at "
                   "0.5 m/s) for the rover's heading -> fb.wind(dL, dR) every 10 ms%s" % (
                       WIND_EQUIV, MARS_RHO, EARTH_RHO, "" if self.deliver_wind else "; CONTROL: fb.wind(0, 0) instead"),
                   wind_equiv=WIND_EQUIV)
        gc.declare(L, "autopilot", "game",
                   "pure pursuit of the route line (look-ahead %.0f m, %.1f /s per rad, limit %.0f deg/s) at %.1f m/s "
                   "(accel %.1f m/s^2); yaw rate = autopilot + the wind steer's applied command. The autopilot's limit "
                   "is above the wind steer's cap (%.0f deg/s), so it can always bring the rover back to the route" % (
                       AP_LOOKAHEAD, AP_GAIN, AP_MAX / DEG, V_CRUISE, A_DRIVE, WIND_MAX / DEG),
                   v_cruise=V_CRUISE, lookahead_m=AP_LOOKAHEAD, gain=AP_GAIN, limit_deg_s=AP_MAX / DEG)
        gc.declare(L, "wind steer arbitration", "game",
                   "the wind steer's command is added to the autopilot's in full while the rover drives forward; while "
                   "it backs up (after a hazard stop or a collision) or goes round a rock it hit, the command is held "
                   "(not applied). Every held tick is counted (summary steer_held_ticks / steer_held_episodes)")
        gc.declare(L, "hazard stop mechanics", "game",
                   "on the stop decoder's trigger (re-armed once DNp01 falls below 33 Hz), brake at %.0f m/s^2 to rest, "
                   "reverse %.1f s at %.1f m/s, drive on" % (A_BRAKE, REVERSE_S, V_REVERSE))
        gc.declare(L, "collision", "game",
                   "contact = the %.1f x %.1f m footprint overlaps a boulder's ground circle; the rover stops dead and "
                   "goes round that rock as a rover that turns in place: it backs up straight at %.1f m/s until the "
                   "footprint is %.1f m clear (at most %.0f s), pivots in place (%.0f deg/s) to %.0f deg off the route "
                   "towards the side its centre is on, drives out at up to %.0f m/s until the footprint will clear "
                   "the rock by %.1f m, pivots back parallel to the route and drives past; then the autopilot rejoins "
                   "the route" % (ROVER_L, ROVER_W, V_REVERSE, BACK_CLEAR, BACK_MAX_S, PIVOT_RATE / DEG, PIVOT_DEG,
                                  V_SIDE, DETOUR_MARGIN))
        gc.declare(L, "boulder field", "game",
                   "%d hazard boulders on the route line (fixed layout, LAYOUT_SEED, the same for every --seed) and %d "
                   "scenery rocks at least %.0f m off it; terrain, crater rim, dunes, sky, sun, haze and dust devils are "
                   "fixed" % (len(self.hazards), len(self.rocks) - len(self.hazards), LANE_HALF))
        gc.declare(L, "display only", "game", "wheel dust, wheel tracks and wind-borne dust streaks are drawn for the "
                   "camera only; the fly's scene has none of them")
        if self.control == "off":
            gc.declare(L, "control: decoders disconnected", "game",
                       "both decoders still read their cells, but neither command is applied")
        elif self.control == "nowind":
            gc.declare(L, "control: no wind", "game",
                       "the devils' wind is not delivered: fb.wind(0, 0) every tick (still air); decoders applied")
        elif self.control == "hidden":
            gc.declare(L, "control: boulders hidden from the fly", "game",
                       "the %d hazard boulders and their shadows are removed from the scene the fly's eyes trace (the "
                       "camera, the collision rule and the tally keep them); decoders applied" % len(self.hazards))
        if self.control != "off" and self.apply != {"steer", "stop"}:
            gc.declare(L, "ablation", "game", "only these decoder outputs are applied: %s" % sorted(self.apply))

    def brains(self):
        return {"brain": self.fb}

    # ------------------------------------------------------------------ pose
    def _update_pose(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        pts = []
        for wx, wy in WHEELS_LOCAL:
            pts.append((self.x + c * wx - s * wy, self.y + s * wx + c * wy))
        pts.append((self.x, self.y))
        P = torch.tensor(pts, dtype=torch.float32, device=self.dev)
        h = self.scene.height(P[:, 0], P[:, 1]).cpu().numpy()
        front = (h[0] + h[3]) / 2
        rear = (h[2] + h[5]) / 2
        left = (h[0] + h[1] + h[2]) / 3
        right = (h[3] + h[4] + h[5]) / 3
        pitch = math.atan2(rear - front, 2.2)                  # nose down = +
        roll = math.atan2(left - right, 2.3) * -1.0             # left side higher -> roll right (negative about +x)
        z = float(np.mean(h[:6]))
        self.pose_R = rot_zyx(self.yaw, pitch, -roll)
        self.pose_pos = np.array([self.x, self.y, z])
        self.rover.set_pose(self.pose_pos, self.pose_R, self.dist / WHEEL_R)

    def eye_pos(self):
        return self.pose_pos + self.pose_R @ self.eye_local

    def _eye_trace(self):
        """The eye's trace from the fixed input buffers (pose, time, devil centres): the fly's 1,466 x 7 rays ->
        (1466, 4) radiance. Makes no host syncs, so it can be captured in a CUDA graph."""
        d = self._eye_dirs @ self._eye_R.T                     # body -> world: columns of pose_R are fwd, left, up
        rgb, _ = self.scene.trace(self._eye_o, d, self._eye_t, rover=self.rover, hide_hazards=self.hide_hazards,
                                  devil_xy=self._eye_dxy)
        return self.eyes.pool(gc.rgb_to_radiance(rgb))

    def _eye_radiance(self):
        """What the fly sees this tick. On CUDA the eye's trace is replayed from a CUDA graph (the same kernels on the
        same inputs, without ~3,000 kernel launches a tick); it is checked against the eager trace once at the start,
        and the run falls back to the eager trace (logged) if capture fails or the two differ."""
        self._eye_o.copy_(torch.as_tensor(self.eye_pos().astype(np.float32)))
        self._eye_R.copy_(torch.as_tensor(self.pose_R.astype(np.float32)))
        self._eye_t.fill_(float(self.t_s))
        self._eye_dxy.copy_(torch.as_tensor(np.array([dv["pos"](self.t_s) for dv in self.devils], np.float32)))
        if self._eye_graph_state == "untried":
            self._eye_graph_state = "eager"
            if self.dev.type == "cuda" and not getattr(self.args, "no_eye_graph", False):
                self._capture_eye_graph()
        if self._eye_graph_state == "graph":
            self._eye_graph.replay()
            return self._eye_out.clone()
        return self._eye_trace()

    def _capture_eye_graph(self):
        info = {"used": False}
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._eye_trace()
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self._eye_trace()
            g.replay()
            ref = self._eye_trace()
            diff = float((out - ref).abs().max().item())
            info.update(max_abs_diff_vs_eager=diff, checked_at_t_s=round(self.t_s, 3))
            if diff <= 1e-5:
                self._eye_graph, self._eye_out, self._eye_graph_state = g, out, "graph"
                info["used"] = True
            else:
                info["why_not"] = "graph and eager traces differ"
        except Exception as e:                                  # any capture problem: the eager trace, logged
            info["why_not"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            torch.cuda.synchronize()
        self.log.meta["eye_cuda_graph"] = info

    def _read_rates(self):
        r = self.fb.brain.rate[0, self.read_idx].float().cpu().numpy()
        for i, k in enumerate(self.read_names):
            self.rates[k] = _mean(r[self.read_seg[i]:self.read_seg[i + 1]])

    def _banner(self, lane, text, color, size, chips=(), sub=None):
        """Show a banner in a fixed lane; `chips` are the provenance classes of what it reports (drawn as tabs)."""
        self.lanes[lane] = (self.t_s, text, color, size, tuple(chips), sub)

    def _veer(self, t, ws, wfrom, d3, steer_on, held):
        """Log (and banner) a VEER of the wind steer's command."""
        S = self.stats
        S["veers"] += 1
        left = ws["yaw_rate"] > 0
        near = math.isfinite(d3[0]) and d3[0] < ENCOUNTER_M
        away = (left != (d3[1] > 0)) if (near and 3.0 < abs(d3[1]) < 177.0) else None
        towards = (left == (wfrom > 0)) if 3.0 < abs(wfrom) < 177.0 else None
        applied = steer_on and not held
        S["veers_applied"] += int(applied)
        self.log.event(t, "veer", direction="left" if left else "right", yaw_rate_deg_s=round(ws["yaw_rate"] / DEG, 1),
                       u_s_hz=round(ws["u_s"], 1), applied=applied, held=held,
                       DNp18_L=round(ws["p18L"], 1), DNp18_R=round(ws["p18R"], 1), DNp33_L=round(ws["p33L"], 1),
                       DNp33_R=round(ws["p33R"], 1), wind_from_deg=round(wfrom, 1),
                       wind_mars_m_s=round(float(np.hypot(*self.wind)), 2), towards_wind=towards,
                       D3_distance_m=round(d3[0], 1), D3_bearing_deg=round(d3[1], 1), away_from_D3=away)
        side = "LEFT" if left else "RIGHT"
        sub = "DNp18 L %.0f · R %.0f Hz   DNp33 L %.0f · R %.0f Hz   index %+.0f Hz → %.0f deg/s %s" % (
            ws["p18L"], ws["p18R"], ws["p33L"], ws["p33R"], ws["u_s"], abs(ws["yaw_rate"]) / DEG, side.lower())
        if applied:
            self._banner("steer", "WIND DNs → VEER %s" % side, gc.TEAL, 50, ("connectome", "decoder"),
                         sub + (", away from the dust devil" if away else ""))
        else:
            why = "HELD: COLLISION RECOVERY" if held else "NOT APPLIED"
            self._banner("steer", "WIND DNs → VEER %s (%s)" % (side, why), gc.MUTED, 40, ("connectome", "decoder"),
                         sub + ("" if held else "   (control: decoder disconnected)"))

    # ------------------------------------------------------------------ loop
    def tick(self):
        fb = self.fb
        dt = gc.TICK_MS / 1000.0
        # senses (GAME -> shipped surface): the eyes' radiance, and the devils' wind on the antennae
        self.rad = self._eye_radiance()
        fb.vision(self.rad)
        self.wind = devil_wind((self.x, self.y), self.devils, self.t_s)
        self.defl = antenna_deflections(self.wind, self.yaw, self.air) if self.deliver_wind else (0.0, 0.0)
        fb.wind(*self.defl)
        fb.step(gc.TICK_MS)
        self.t_s += dt
        t = self.t_s
        self._read_rates()
        ws = self.wind_dec.value or {"u": 0.0, "u_s": 0.0, "yaw_rate": 0.0, "active": False,
                                     "p18L": 0.0, "p18R": 0.0, "p33L": 0.0, "p33R": 0.0}
        sp = self.stop_dec.value or {"gf": 0.0, "fire": False}
        gf = sp["gf"]
        loomL = 0.5 * (self.rates["LPLC2_L"] + self.rates["LC4_L"])
        loomR = 0.5 * (self.rates["LPLC2_R"] + self.rates["LC4_R"])
        mdn = 0.5 * (self.rates["MDN_L"] + self.rates["MDN_R"])
        for key, val in (("loomL", loomL), ("loomR", loomR), ("gf", gf), ("u", ws["u"]), ("u_s", ws["u_s"]),
                         ("yaw", ws["yaw_rate"] / DEG)):
            h = self.hist[key]
            h.append(val)
            del h[0]
        S = self.stats
        S["gf_max"] = max(S["gf_max"], gf)
        S["mdn_max"] = max(S["mdn_max"], mdn)
        if t < WARM_S:
            S["mdn_max_t_lt_1s"] = max(S["mdn_max_t_lt_1s"], mdn)
        S["loomL_max"] = max(S["loomL_max"], loomL)
        S["loomR_max"] = max(S["loomR_max"], loomR)
        S["u_s_min"] = min(S["u_s_min"], ws["u_s"])
        S["u_s_max"] = max(S["u_s_max"], ws["u_s"])
        S["wind_equiv_max_m_s"] = max(S["wind_equiv_max_m_s"], float(np.hypot(*self.wind)) * WIND_EQUIV)
        S["yaw_dec_abs_max_deg_s"] = max(S["yaw_dec_abs_max_deg_s"], abs(ws["yaw_rate"]) / DEG)
        x_front = self.x + ROVER_L / 2 * math.cos(self.yaw)
        # hazard stop (DECODER -> GAME brake)
        if sp["fire"] and self.gf_armed:
            self.gf_armed = False
            S["gf_crossings"] += 1
            nearest = self.tally.ahead(x_front)
            ahead_m = None if nearest is None else round(nearest["x"] - nearest["r"] - x_front, 2)
            applied = "stop" in self.apply and self.drive.hazard_stop()
            if applied:
                S["hazard_stops_applied"] += 1
                self.tally.credit_stop(x_front)
            self.log.event(t, "gf_cross", gf_hz=round(gf, 1), applied=applied, state=self.drive.state,
                           speed=round(self.drive.v, 2), nearest_hazard=None if nearest is None else nearest["name"],
                           hazard_gap_m=ahead_m, footprint_gap_m=self._nearest_gap(), loom_L=round(loomL, 2),
                           loom_R=round(loomR, 2))
            self._banner("gf", "GIANT FIBRE %.0f Hz → HAZARD STOP%s" % (gf, "" if applied else " (NOT APPLIED)"),
                         gc.RED if applied else gc.MUTED, 52 if applied else 40, ("connectome", "decoder"),
                         "DNp01 >= 33 Hz, the shipped body model's takeoff line")
        elif not sp["fire"]:
            self.gf_armed = True
        # wind steer (DECODER)
        active = bool(ws["active"])
        if active:
            S["steer_active_ticks"] += 1
        if ws.get("at_cap"):
            S["steer_cap_ticks"] += 1
        wfrom = wind_from_deg(self.wind, self.yaw)
        d3 = self._devil_rel("D3")
        steer_on = "steer" in self.apply
        if active and not self.steer_was_active and t - self.steer_last_active_t > ONSET_GAP_S:
            S["steer_onsets"] += 1
            turn_left = ws["yaw_rate"] > 0
            self.log.event(t, "wind_steer_on", u_hz=round(ws["u"], 1), u_s_hz=round(ws["u_s"], 1),
                           yaw_rate_deg_s=round(ws["yaw_rate"] / DEG, 1), applied=steer_on,
                           wind_from_deg=round(wfrom, 1), wind_equiv_m_s=round(float(np.hypot(*self.wind)) * WIND_EQUIV, 3),
                           towards_wind=(turn_left == (wfrom > 0)) if 3.0 < abs(wfrom) < 177.0 else None,
                           D3_distance_m=round(d3[0], 1), D3_bearing_deg=round(d3[1], 1),
                           away_from_D3=(turn_left != (d3[1] > 0)) if 3.0 < abs(d3[1]) < 177.0 else None)
        self.steer_was_active = active
        if active:
            self.steer_last_active_t = t
        # autopilot (GAME), or the detour round a rock the rover hit
        v_detour = None
        if self.detour is not None:
            dr = self.detour.rock
            gap = rect_circle_clearance(self.x, self.y, self.yaw, ROVER_L / 2, ROVER_W / 2, dr["x"], dr["y"], dr["r"])
            v_detour, self.yaw_rate_ap, done = self.detour.step(dt, self.x, self.y, self.yaw, self.drive.v, gap)
            if done:
                self.log.event(t, "detour_done", rock=dr["name"], x=round(self.x, 2), y=round(self.y, 3))
                self.detour, v_detour = None, None
                self.drive.state = "drive"
        if self.detour is None:
            self.yaw_rate_ap = autopilot_yaw_rate(self.x, self.y, self.yaw)
        # GAME arbitration: the decoder's command is held while the rover recovers from a collision or backs up
        recovering = self.detour is not None or self.drive.state == "reverse"
        self.yaw_rate_dec, held = steer_applied(ws["yaw_rate"] if steer_on else 0.0, recovering)
        self.held = held
        if held:
            S["steer_held_ticks"] += 1
            S["steer_held_abs_deg"] += abs(ws["yaw_rate"]) * dt / DEG
            if not self.held_prev:                 # one episode per collision recovery in which a turn was held
                S["steer_held_episodes"] += 1
                self.log.event(t, "steer_held", yaw_rate_deg_s=round(ws["yaw_rate"] / DEG, 1),
                               u_s_hz=round(ws["u_s"], 1), drive_state=self.drive.state,
                               rock=None if self.detour is None else self.detour.rock["name"])
        self.held_prev = (self.held_prev or held) if recovering else False
        S["steer_applied_abs_deg"] += abs(self.yaw_rate_dec) * dt / DEG
        # a VEER: the decoder's command passes VEER_FRAC of its cap after >= VEER_REARM_S below it (log + banner)
        if abs(ws["yaw_rate"]) >= VEER_FRAC * WIND_MAX:
            if t - self.veer_high_t > VEER_REARM_S:
                self._veer(t, ws, wfrom, d3, steer_on, held)
            self.veer_high_t = t
        if v_detour is not None:
            self.drive.v = v_detour
        v = self.drive.update(dt, t)
        yaw_rate = self.yaw_rate_ap + self.yaw_rate_dec
        nx, ny, nyaw = rover_step(self.x, self.y, self.yaw, v, yaw_rate, dt)
        # collisions (GAME)
        hit = self._contact(nx, ny, nyaw)
        if hit is not None and v < 0:
            # backing up (after a stop or a collision) into a rock behind: the reverse ends there (GAME); not a
            # collision, the rover does not move this tick and carries on from where it is
            if self.detour is not None:
                self.detour.t = BACK_MAX_S
            else:
                self.drive.state, self.drive.v, self.drive.timer = "drive", 0.0, 0.0
            self.log.event(t, "reverse_blocked", rock=hit["name"])
        elif hit is not None:
            hit["contacts"] += 1
            S["collisions"] += 1
            gap = rect_circle_clearance(nx, ny, nyaw, ROVER_L / 2, ROVER_W / 2, hit["x"], hit["y"], hit["r"])
            if "collision_gap_m" in hit:
                hit["collision_gap_m"] = round(min(hit["collision_gap_m"] or 0.0, gap), 3)
            self.log.event(t, "collision", hazard=hit["name"], speed=round(self.drive.v, 2),
                           yaw_deg=round(self.yaw / DEG, 1), y=round(self.y, 2), rejected_pose_gap_m=round(gap, 3),
                           gf_hz=round(gf, 1), loom_L=round(loomL, 2), loom_R=round(loomR, 2))
            self.drive.collide()
            self.detour = Detour(hit, self.x, self.y, self.yaw)
            self._banner("hit", "COLLISION · %s" % hit["name"], gc.AMBER, 48, ("game",),
                         "giant fibre %.0f Hz: below the 33 Hz stop line" % gf)
        else:
            self.dist += abs(v) * dt
            self.x, self.y, self.yaw = nx, ny, nyaw
        self._update_pose()
        if hit is None and v != 0.0:
            self._stamp_tracks()
        self._encounter(t, dt)
        for h in self.tally.observe(t, self.x, self.y, self.yaw):
            self.log.event(t, "hazard_passed", hazard=h["name"], outcome=h["outcome"],
                           min_clearance_m=round(h["min_clear_m"], 3), stops=h["stops"], contacts=h["contacts"])
            if h["outcome"] == "clean":
                self._banner("pass", "%s PASSED · CLEARANCE %.2f m" % (h["name"], h["min_clear_m"]), gc.SAGE, 40,
                             ("game",))
        if int(round(t * 100)) % 10 == 0:
            self.trail.append((self.x, self.y))
            self.series.append([round(t, 2), round(self.x, 3), round(self.y, 3), round(self.yaw / DEG, 2),
                                round(self.drive.v, 2), round(loomL, 3), round(loomR, 3), round(gf, 2), round(mdn, 2),
                                round(ws["p18L"], 2), round(ws["p18R"], 2), round(ws["p33L"], 2), round(ws["p33R"], 2),
                                round(ws["u"], 2), round(ws["u_s"], 2), round(ws["yaw_rate"] / DEG, 2),
                                round(self.yaw_rate_dec / DEG, 2), int(self.held),
                                round(self.yaw_rate_ap / DEG, 2), round(float(np.hypot(*self.wind)), 3),
                                round(wfrom, 1), round(self.defl[0], 3), round(self.defl[1], 3), round(d3[0], 2),
                                round(d3[1], 1), self._nearest_gap()])
        self._dust_step(dt, v)

    def _contact(self, nx, ny, nyaw):
        """The rock (a hazard's tally row, or a dict for a scenery rock) the footprint would overlap at a pose."""
        for h in self.tally.rows:
            if abs(h["x"] - nx) < 6 and rect_circle_clearance(nx, ny, nyaw, ROVER_L / 2, ROVER_W / 2,
                                                              h["x"], h["y"], h["r"]) < 0:
                return h
        for r in self.rocks:
            if not r.hazard and abs(r.pos[0] - nx) < 6 and abs(r.pos[1] - ny) < 6 and \
                    rect_circle_clearance(nx, ny, nyaw, ROVER_L / 2, ROVER_W / 2, r.pos[0], r.pos[1], r.r_ground) < 0:
                return {"name": "scenery", "x": float(r.pos[0]), "y": float(r.pos[1]), "r": r.r_ground, "contacts": 0}
        return None

    def _stamp_tracks(self):
        """Display: press the six wheels' contact patches into the camera's track map."""
        c, sn = math.cos(self.yaw), math.sin(self.yaw)
        self.scene.tracks.stamp([(self.x + c * wx - sn * wy, self.y + sn * wx + c * wy) for wx, wy in WHEELS_LOCAL])

    def _encounter(self, t, dt):
        """GAME geometry for the log: every devil's closest approach to the rover's centre, and, while D3 is within
        ENCOUNTER_M, the rover's lateral excursion and heading range outside collision detours and the wind steer's
        applied turn."""
        for dv in self.devils:
            cx, cy = dv["pos"](t)
            dist = math.hypot(cx - self.x, cy - self.y)
            m = self.devil_min[dv["name"]]
            if dist < m["distance_m"]:
                m.update(distance_m=dist, t_s=round(t, 2), rover_x_m=round(self.x, 2), rover_y_m=round(self.y, 3))
        dist, _ = self._devil_rel("D3")
        if dist < ENCOUNTER_M:
            e = self.enc
            if e["t_in"] is None:
                e["t_in"] = round(t, 2)
                self.log.event(t, "encounter_start", D3_distance_m=round(dist, 1), x=round(self.x, 2),
                               y=round(self.y, 3))
            e["t_out"] = round(t, 2)
            if self.detour is None:                # the excursion while driving (a detour's pivots are GAME)
                e["y_min"], e["y_max"] = min(e["y_min"], self.y), max(e["y_max"], self.y)
                yd = self.yaw / DEG
                e["yaw_min_deg"], e["yaw_max_deg"] = min(e["yaw_min_deg"], yd), max(e["yaw_max_deg"], yd)
            a = self.yaw_rate_dec * dt / DEG
            e["dec_applied_deg"] += a
            e["dec_left_deg"] += max(a, 0.0)
            e["dec_right_deg"] += max(-a, 0.0)
            e["held_ticks"] += int(self.held)

    def _devil_rel(self, name):
        """(distance m, bearing deg relative to the heading, + = left) of a dust devil's centre from the rover."""
        for dv in self.devils:
            if dv["name"] == name:
                cx, cy = dv["pos"](self.t_s)
                return (math.hypot(cx - self.x, cy - self.y),
                        math.degrees(wrap(math.atan2(cy - self.y, cx - self.x) - self.yaw)))
        return (float("nan"), float("nan"))

    def _nearest_gap(self):
        """Footprint clearance (m) to the nearest hazard not yet passed (None past the last one)."""
        live = [h for h in self.tally.rows if h["outcome"] is None]
        if not live:
            return None
        return round(min(rect_circle_clearance(self.x, self.y, self.yaw, ROVER_L / 2, ROVER_W / 2, h["x"], h["y"],
                                               h["r"]) for h in live), 3)

    def finished(self):
        return self.x > ROUTE_END

    # ------------------------------------------------------------------ dust (display only)
    def _dust_step(self, dt, v):
        d = self.dust
        if len(d):
            d[:, 6] += dt
            d[:, 5] -= 0.6 * dt
            d[:, 0:3] += d[:, 3:6] * dt
            d[:, 3] += 0.25 * dt                     # a light breeze toward +x... carried along
            d = d[d[:, 6] < 2.2]
        if abs(v) > 0.3 and int(round(self.t_s * 100)) % 2 == 0:
            R = self.pose_R
            new = []
            for wx, wy in WHEELS_LOCAL:
                if wx > 0.5:
                    continue
                p = self.pose_pos + R @ np.array([wx - 0.25, wy, 0.08])
                vel = R @ np.array([-0.25 * v, self.rng.normal(0, 0.25), 0.35 + self.rng.uniform(0, 0.35)])
                new.append([*p, *vel, 0.0])
            d = np.concatenate([d, np.asarray(new, np.float32)]) if len(d) else np.asarray(new, np.float32)
        self.dust = d

    # ------------------------------------------------------------------ camera
    # the opening crane shot: behind-right of the rover, low, dollying in and rising, the sun in the upper left
    VISTA_P0, VISTA_H0 = np.array([-19.0, -13.5]), 1.3
    VISTA_P1, VISTA_H1 = np.array([-13.5, -10.5]), 3.0
    VISTA_LOOK0, VISTA_LOOK1 = (48.0, 9.5), (42.0, 6.5)   # azimuth, elevation (deg) at the start and the end
    VISTA_FOV = 68.0
    VISTA_END, CHASE_FULL = 3.2, 6.0               # the crane shot holds to 3.2 s, the chase camera is in by 6.0 s

    def camera_pose(self):
        """GAME camera, called once per video frame: a wide crane shot of the crater floor that dollies in and rises
        from t = 0 (the rover and the sun in frame), then a chase camera behind-left of the rover, which follows a
        smoothed heading (1 s) and holds the route's direction while the rover goes round a rock it hit (so its
        pivots show as the rover turning, not as the camera swinging). While dust devil D3 is within ~50 m and not
        yet well behind the rover, the camera sits behind-left of the rover in the route's frame and aims between the
        rover and the devil, never more than 20 deg off the rover, so the rover's turns show as turns in frame."""
        t = self.t_s
        dt = 1.0 / float(getattr(self.args, "fps", 50) or 50)
        target_yaw = 0.0 if self.detour is not None else self.yaw
        if self.cam_yaw is None:
            self.cam_yaw = target_yaw
        self.cam_yaw = wrap(self.cam_yaw + wrap(target_yaw - self.cam_yaw) * (1 - math.exp(-dt / 1.0)))
        fwd = np.array([math.cos(self.cam_yaw), math.sin(self.cam_yaw), 0.0])
        left = np.array([-math.sin(self.cam_yaw), math.cos(self.cam_yaw), 0.0])
        base = self.pose_pos
        chase_pos = base - fwd * 10.5 + left * 4.6 + np.array([0, 0, 3.7])
        chase_tgt = base + fwd * 7.0 + np.array([0, 0, 0.9])
        dist, _ = self._devil_rel("D3")
        dx, dy = [dv for dv in self.devils if dv["name"] == "D3"][0]["pos"](t)
        rel_x = dx - base[0]                          # + = the devil is ahead of the rover along the route
        wd = float(smoothstep(55.0, 38.0, dist) * smoothstep(-24.0, -6.0, rel_x)) if math.isfinite(dist) else 0.0
        if wd > 0:
            enc_pos = base + np.array([-11.0, 6.5, 4.4])
            aim = base + np.array([0.35 * (dx - base[0]), 0.35 * (dy - base[1]), 3.5]) - enc_pos
            to_rover = base + np.array([0.0, 0.0, 1.2]) - enc_pos
            a_aim, a_rov = math.atan2(aim[1], aim[0]), math.atan2(to_rover[1], to_rover[0])
            off = wrap(a_aim - a_rov)
            if abs(off) > 20.0 * DEG:                 # keep the rover within 20 deg of the frame's centre
                a_new = a_rov + math.copysign(20.0 * DEG, off)
                r = math.hypot(aim[0], aim[1])
                aim = np.array([r * math.cos(a_new), r * math.sin(a_new), aim[2]])
            chase_pos = chase_pos * (1 - wd) + enc_pos * wd
            chase_tgt = chase_tgt * (1 - wd) + (enc_pos + aim) * wd
        u = float(smoothstep(0.0, self.VISTA_END + 0.8, t))
        xy = self.VISTA_P0 * (1 - u) + self.VISTA_P1 * u
        hgt = self.VISTA_H0 * (1 - u) + self.VISTA_H1 * u
        wide_pos = np.array([xy[0], xy[1], self.scene.height_np(xy[0], xy[1]) + hgt])
        az = self.VISTA_LOOK0[0] * (1 - u) + self.VISTA_LOOK1[0] * u
        el = self.VISTA_LOOK0[1] * (1 - u) + self.VISTA_LOOK1[1] * u
        wide_tgt = wide_pos + 30.0 * unit_from_az_el(az, el)
        k = float(smoothstep(self.VISTA_END, self.CHASE_FULL, t))
        pos = wide_pos * (1 - k) + chase_pos * k
        tgt = wide_tgt * (1 - k) + chase_tgt * k
        fov = self.VISTA_FOV * (1 - k) + (50.0 + 16.0 * wd) * k
        if self.cam_state is None or k < 1.0:
            self.cam_state = (pos, tgt)
        else:
            a = 1 - math.exp(-dt / 0.35)
            p0, t0 = self.cam_state
            self.cam_state = (p0 + (pos - p0) * a, t0 + (tgt - t0) * a)
        pos, tgt = self.cam_state
        return pos, tgt - pos, fov

    def render_view(self, W, H):
        pos, fwd, fov = self.camera_pose()
        s = float(getattr(self.args, "view_scale", 1.0) or 1.0)          # dev only: trace fewer camera rays
        w, h = (W, H) if s >= 1.0 else (max(16, int(W * s)), max(9, int(H * s)))
        d = camera_rays(fwd, (0, 0, 1), w, h, fov, self.dev)
        o = torch.tensor(pos, dtype=torch.float32, device=self.dev)
        rgb, depth = self.scene.trace(o, d, self.t_s, camera=True, rover=self.rover)
        img = rgb.reshape(h, w, 3).permute(2, 0, 1)[None]
        if (w, h) != (W, H):
            img = torch.nn.functional.interpolate(img, size=(H, W), mode="bilinear", align_corners=False)
            depth = torch.nn.functional.interpolate(depth.reshape(1, 1, h, w), size=(H, W), mode="nearest")
        bright = (img - 1.0).clamp_min(0)
        small = torch.nn.functional.avg_pool2d(bright, 8)
        for _ in range(2):
            small = torch.nn.functional.avg_pool2d(small, 5, stride=1, padding=2, count_include_pad=False)
        bloom = torch.nn.functional.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
        img = img + 0.35 * bloom
        yy = torch.linspace(-1, 1, H, device=self.dev)[:, None]
        xx = torch.linspace(-1, 1, W, device=self.dev)[None, :]
        vig = (1 - 0.22 * (xx ** 2 + yy ** 2)).clamp(0.6, 1)
        out = tonemap(img[0] * vig[None], 1.3).permute(1, 2, 0)
        arr = (out * 255).to(torch.uint8).cpu().numpy()
        return arr, depth.reshape(H, W).cpu().numpy(), (pos, fwd, fov)

    @staticmethod
    def project(points, cam, W, H):
        """World points (N,3) -> pixel (N,2) and depth along the view axis (N,); for overlays on the main view."""
        pos, fwd, fov = cam
        f, left, u = look_basis(fwd)
        rel = np.asarray(points, float) - pos[None]
        zc = rel @ f
        tan = math.tan(fov * DEG / 2)
        xx = -(rel @ left) / np.maximum(zc, 1e-6) / tan
        yy = -(rel @ u) / np.maximum(zc, 1e-6) / (tan * H / W)
        return np.stack([(xx + 1) / 2 * W, (yy + 1) / 2 * H], 1), zc

    # ------------------------------------------------------------------ drawing
    def _layer(self, key, size):
        """A reusable transparent surface (cleared on each use)."""
        import pygame
        s = self._layers.get(key)
        if s is None or s.get_size() != tuple(size):
            s = self._layers[key] = pygame.Surface(size, pygame.SRCALPHA)
        s.fill((0, 0, 0, 0))
        return s

    def draw(self, surface):
        import pygame
        size = surface.get_size()
        if size != DESIGN:
            if self._frame is None:
                self._frame = pygame.Surface(DESIGN)
            canvas = self._frame
        else:
            canvas = surface
        hud = self.hud
        canvas.fill(gc.BG)
        vx, vy, vw, vh = VIEW
        arr, depth, cam = self.render_view(vw, vh)
        view = pygame.image.frombuffer(arr.tobytes(), (vw, vh), "RGB").copy()
        self._windp_step(1.0 / self.args.fps, cam)
        self._draw_windp(view, depth, cam, vw, vh)
        self._draw_dust(view, depth, cam, vw, vh)
        self._draw_devil_tag(view, depth, cam, vw, vh)
        self._draw_callout(view, cam, vw, vh)
        canvas.blit(view, (vx, vy))
        self._draw_view_tags(canvas)
        if self.t_s >= self.VISTA_END + 0.6:                  # a clean opening shot: no overlays on the view yet
            self._draw_wind_widget(canvas)
            self._draw_minimap(canvas, pygame.Rect(VIEW[2] - 330, VIEW[1] + 12, 318, 150))
        self._draw_banners(canvas)
        hud.header(canvas, "mars", self.subtitle, gc.model_line(self.fb))
        self._draw_right(canvas)
        self._draw_bottom(canvas)
        laws = ["wind steer: %.1f deg/s per Hz of wind-DN L-R beyond %.0f Hz, cap %.0f, into the wind; held in "
                "collision recovery" % (WIND_GAIN / DEG, WIND_DEAD, WIND_MAX / DEG),
                "stop: DNp01 >= 33 Hz -> brake", "rover, autopilot, wind: GAME"]
        hud.footer(canvas, laws)
        if canvas is not surface:
            surface.blit(pygame.transform.smoothscale(canvas, size), (0, 0))

    def _puff(self, r):
        """A soft round dust puff of radius r px (a radial falloff baked into the alpha), cached per radius."""
        import pygame
        cache = self._layers.setdefault("_puffs", {})
        r = int(max(2, min(r, 90)))
        if r not in cache:
            n = 2 * r + 1
            yy, xx = np.mgrid[0:n, 0:n] - r
            a = np.clip(1 - np.sqrt(xx ** 2 + yy ** 2) / r, 0, 1) ** 1.6
            s = pygame.Surface((n, n), pygame.SRCALPHA)
            s.fill((200, 146, 100, 0))
            pygame.surfarray.pixels_alpha(s)[:] = (a.T * 255).astype(np.uint8)
            cache[r] = s
        return cache[r]

    WINDP_N = 520                                  # wind-borne dust streaks (display): how many are kept alive

    def _windp_step(self, dt, cam):
        """Display only: dust grains carried by the devils' GAME wind field near the rover, drawn as streaks."""
        p = self.windp
        if len(p):
            w = devil_wind_many(p[:, :2], self.devils, self.t_s)
            p[:, 0:2] += w * dt
            p[:, 2] += 0.35 * dt
            p[:, 3] += dt
            keep = (p[:, 3] < 1.8) & (np.abs(p[:, 0] - self.x) < 30) & (np.abs(p[:, 1] - self.y) < 30)
            p = p[keep]
        n = self.WINDP_N - len(p)
        if n > 0:
            r = self.rng
            ahead = np.array([math.cos(self.yaw), math.sin(self.yaw)])
            c = np.array([self.x, self.y]) + ahead * 6.0
            xy = c[None] + r.uniform(-24, 24, (n, 2))
            xy_t = torch.as_tensor(xy, dtype=torch.float32, device=self.dev)
            z = self.scene.height(xy_t[:, 0], xy_t[:, 1]).cpu().numpy()
            new = np.c_[xy, z + r.uniform(0.1, 3.2, n), r.uniform(0, 1.2, n)].astype(np.float32)
            p = np.concatenate([p, new]) if len(p) else new
        self.windp = p

    def _draw_windp(self, view, depth, cam, W, H):
        import pygame
        p = self.windp
        if not len(p) or self.t_s < self.VISTA_END:
            return
        w = devil_wind_many(p[:, :2], self.devils, self.t_s)
        sp = np.hypot(w[:, 0], w[:, 1])
        vis = sp > 0.9
        if not vis.any():
            return
        p, w, sp = p[vis], w[vis], sp[vis]
        tail = p[:, :3].copy()
        tail[:, :2] -= w * 0.16
        a_px, za = self.project(p[:, :3], cam, W, H)
        b_px, zb = self.project(tail, cam, W, H)
        layer = self._layer("windp", (W, H))
        age = p[:, 3]
        fade = np.clip(np.minimum(age / 0.3, (1.8 - age) / 0.5), 0, 1)
        alpha = (np.clip((sp - 0.9) / 3.0, 0, 1) * fade * 150).astype(int)
        for (x0, y0), (x1, y1), z0, z1, al in zip(a_px, b_px, za, zb, alpha):
            if al < 6 or z0 <= 0.5 or z1 <= 0.5 or not (0 <= x0 < W and 0 <= y0 < H):
                continue
            if depth[int(y0), int(x0)] < z0 * 0.98:
                continue
            width = 2 if z0 > 12 else 3
            pygame.draw.line(layer, (226, 178, 128, int(al)), (int(x1), int(y1)), (int(x0), int(y0)), width)
        view.blit(layer, (0, 0))

    def _draw_dust(self, view, depth, cam, W, H):
        if not len(self.dust):
            return
        px, zc = self.project(self.dust[:, :3], cam, W, H)
        tan = math.tan(cam[2] * DEG / 2)
        for (x, y), z, age in zip(px, zc, self.dust[:, 6]):
            if z <= 0.3 or not (0 <= x < W and 0 <= y < H):
                continue
            if depth[int(y), int(x)] < z * 0.98:
                continue
            a = 75 * (1 - age / 2.2) ** 1.5
            if a <= 3:
                continue
            r = max(2.0, (0.06 + 0.24 * age) / (z * tan) * W / 2)
            s = self._puff(r)
            s.set_alpha(int(a))
            view.blit(s, s.get_rect(center=(int(x), int(y))))

    def _draw_devil_tag(self, view, depth, cam, W, H):
        """A small GAME label beside dust devil D3's column, while it is near (display)."""
        dist, _ = self._devil_rel("D3")
        if not (math.isfinite(dist) and 8.0 < dist < 60.0) or self.t_s < self.CHASE_FULL:
            return
        dv = [d for d in self.devils if d["name"] == "D3"][0]
        cx, cy = dv["pos"](self.t_s)
        h = 14.0
        _, left, _ = look_basis(cam[1])
        side = np.array([cx, cy, self.scene.height_np(cx, cy) + h]) - left * 1.4 * (dv["w0"] + dv["flare"] * h)
        p, z = self.project(side[None], cam, W, H)
        x, y = float(p[0, 0]), float(p[0, 1])
        if z[0] <= 1 or not (0 <= x < W - 240 and 40 <= y < H - 40):
            return
        if x > W - 580:
            y = max(y, 200.0)                     # below the route minimap (top right of the view)
        hud = self.hud
        img = hud.font(20, True).render("DUST DEVIL · %.0f m" % dist, True, gc.AMBER)
        plate = self._layer("devil_tag", (img.get_width() + 16, img.get_height() + 8))
        plate.fill((9, 12, 14, 170))
        plate.blit(img, (8, 4))
        view.blit(plate, plate.get_rect(midleft=(int(x) + 6, int(y))))

    def _draw_callout(self, view, cam, W, H):
        """A tag on the post above the mast head: where the fly's eyes are (display). Gone before the first hazard."""
        import pygame
        k = float(smoothstep(4.0, 4.6, self.t_s)) * (1 - float(smoothstep(5.4, 5.9, self.t_s)))
        if k <= 0.01:
            return
        px, z = self.project(self.eye_pos()[None], cam, W, H)
        x, y = float(px[0, 0]), float(px[0, 1])
        if z[0] <= 0 or not (0 <= x < W and 0 <= y < H):
            return
        layer = self._layer("callout", (W, H))
        a = int(255 * k)
        pygame.draw.circle(layer, (*gc.TEAL, a), (int(x), int(y)), 10, 3)
        tx, ty = x - 170, y - 110
        pygame.draw.line(layer, (*gc.TEAL, a), (int(x) - 8, int(y) - 8), (int(tx) + 170, int(ty) + 34), 3)
        img = self.hud.font(30, True, True).render("the fly's eyes", True, gc.TEAL)
        img2 = self.hud.font(22).render("1,466 ommatidia on a post above the mast", True, gc.WHITE)
        plate = pygame.Surface((max(img.get_width(), img2.get_width()) + 20, img.get_height() + img2.get_height() + 14),
                               pygame.SRCALPHA)
        plate.fill((9, 12, 14, 170))
        plate.blit(img, (10, 4))
        plate.blit(img2, (10, 8 + img.get_height()))
        plate.set_alpha(a)
        layer.blit(plate, plate.get_rect(bottomright=(int(tx) + 170, int(ty) + 30)))
        view.blit(layer, (0, 0))

    def _banner_surface(self, text, color, size, chips=(), sub=None, pad=18, max_w=None):
        """A banner on a translucent box: its provenance chips as tabs above the box's left edge (as games/doom.py),
        the headline, and an optional sub-line with the numbers it reports."""
        import pygame
        h = self.hud
        img = banner_text(h.font(size, True, True), text, color)
        sub_img = h.font(21).render(sub, True, gc.TEXT) if sub else None
        inner = max(img.get_width(), sub_img.get_width() if sub_img else 0)
        if max_w and inner + 2 * pad > max_w:
            f = (max_w - 2 * pad) / inner
            img = pygame.transform.smoothscale(img, (int(img.get_width() * f), int(img.get_height() * f)))
            if sub_img:
                sub_img = pygame.transform.smoothscale(sub_img, (int(sub_img.get_width() * f),
                                                                 int(sub_img.get_height() * f)))
            inner = max(img.get_width(), sub_img.get_width() if sub_img else 0)
        tab = 26 if chips else 0
        bw = inner + 2 * pad
        bh = img.get_height() + pad + (sub_img.get_height() + 6 if sub_img else 0)
        s = pygame.Surface((bw, bh + tab), pygame.SRCALPHA)
        pygame.draw.rect(s, (9, 12, 14, 196), (0, tab, bw, bh))
        pygame.draw.rect(s, (*color, 255), (0, tab, 6, bh))
        s.blit(img, img.get_rect(midtop=(bw // 2, tab + pad // 2)))
        if sub_img:
            s.blit(sub_img, sub_img.get_rect(midtop=(bw // 2, tab + pad // 2 + img.get_height() + 4)))
        x = 0
        for kind in chips:
            x = h.chip(s, kind, (x, 0), 12).right + 6
        return s

    BANNER_SLOTS = {"gf": 70, "steer": 150, "hit": 262, "pass": 262}   # top of each lane, view y

    def _draw_banners(self, canvas):
        vx, vy, vw, vh = VIEW
        for lane in list(self.lanes):
            t0, txt, col, size, chips, sub = self.lanes[lane]
            age = self.t_s - t0
            if not 0 <= age < BANNER_S:
                del self.lanes[lane]
                continue
            s = self._banner_surface(txt, col, size, chips, sub, max_w=vw - 380)
            s.set_alpha(int(255 * (1 - (age / BANNER_S) ** 2)))
            canvas.blit(s, s.get_rect(midtop=(vx + (vw - 340) // 2, vy + self.BANNER_SLOTS[lane])))

    def _plate_text(self, canvas, text, pos, size, color, chip=None):
        import pygame
        hud = self.hud
        img = hud.font(size, True).render(text, True, color)
        cw = 0
        if chip:
            label, _ = gc.PROVENANCE[chip]
            cw = hud.font(12, True).size(label)[0] + 22
        plate = pygame.Surface((img.get_width() + 20 + cw, img.get_height() + 12), pygame.SRCALPHA)
        plate.fill((9, 12, 14, 190))
        canvas.blit(plate, pos)
        canvas.blit(img, (pos[0] + 10, pos[1] + 6))
        if chip:
            hud.chip(canvas, chip, (pos[0] + 16 + img.get_width(), pos[1] + (img.get_height() + 12) // 2 - 10), 12)

    def _draw_view_tags(self, canvas):
        import pygame
        vx, vy, vw, vh = VIEW
        if self.control != "on" or self.apply != {"steer", "stop"}:
            pygame.draw.rect(canvas, gc.AMBER, (vx + 2, vy + 2, vw - 4, vh - 4), 4)
            label = {"off": "CONTROL · DECODERS DISCONNECTED", "nowind": "CONTROL · NO WIND TO THE ANTENNAE",
                     "hidden": "CONTROL · BOULDERS HIDDEN FROM THE FLY"}.get(self.control)
            if label is None:
                label = "ABLATION · APPLIED: %s" % (", ".join(sorted(self.apply)).upper() or "NONE")
            self._plate_text(canvas, label, (vx + 16, vy + 14), 28, gc.AMBER)
        # the opening title, bottom-left of the crane shot (the rover is right of centre, the sun upper left)
        k = 1.0 - float(smoothstep(3.0, 3.8, self.t_s))
        if k > 0.02:
            hud = self.hud
            t1 = hud.font(40, True, True).render("A JEZERO-LIKE CRATER, MARS", True, gc.WHITE)
            t2 = hud.font(20).render("a GAME scene, not survey data · a fly connectome rides the rover", True,
                                     gc.TEXT)
            w = max(t1.get_width(), t2.get_width() + 110) + 36
            hgt = t1.get_height() + t2.get_height() + 30
            s = pygame.Surface((w, hgt), pygame.SRCALPHA)
            s.fill((9, 12, 14, 168))
            pygame.draw.rect(s, (*gc.AMBER, 255), (0, 0, 6, hgt))
            s.blit(t1, (20, 10))
            s.blit(t2, (20, 18 + t1.get_height()))
            hud.chip(s, "game", (28 + t2.get_width(), 16 + t1.get_height()), 12)
            s.set_alpha(int(255 * k))
            canvas.blit(s, (vx + 22, vy + vh - hgt - 22))

    def _draw_wind_widget(self, canvas):
        """GAME -> the shipped sense: the devils' wind at the rover and the antennal deflections handed to fb.wind
        (bottom-right of the view; not during the opening crane shot)."""
        import pygame
        hud = self.hud
        vx, vy, vw, vh = VIEW
        speed = float(np.hypot(*self.wind))
        hot = self.deliver_wind and speed > 0.8
        r = pygame.Rect(vx + vw - 386, vy + vh - 164, 372, 150)
        s = pygame.Surface(r.size, pygame.SRCALPHA)
        s.fill((9, 12, 14, 196))
        if hot:
            pygame.draw.rect(s, (*gc.AMBER, 255), (0, 0, r.w, r.h), 2)
        canvas.blit(s, r.topleft)
        t = hud.text(canvas, "WIND AT THE ANTENNAE", (r.x + 12, r.y + 9), 15, gc.MUTED, bold=True)
        hud.chip(canvas, "game", (t.right + 8, r.y + 7), 11)
        c = (r.x + 66, r.y + 90)
        pygame.draw.circle(canvas, gc.LINE, c, 44, 2)
        # the rover (heading up) in the dial's centre
        pygame.draw.rect(canvas, gc.MUTED, (c[0] - 6, c[1] - 9, 12, 18), border_radius=2)
        pygame.draw.polygon(canvas, gc.DIM, [(c[0], c[1] - 54), (c[0] - 6, c[1] - 44), (c[0] + 6, c[1] - 44)])
        if not self.deliver_wind:
            hud.text(canvas, "not delivered", (r.x + 130, r.y + 50), 24, gc.AMBER, bold=True)
            hud.text(canvas, "control: still air,", (r.x + 130, r.y + 82), 17, gc.MUTED)
            hud.text(canvas, "fb.wind(0, 0)", (r.x + 130, r.y + 102), 17, gc.MUTED)
            return
        if speed > 0.05:
            wf = wind_from_deg(self.wind, self.yaw)
            a = math.radians(wf)                                          # where it comes from, + = left; up = ahead
            fx, fy = -math.sin(a), -math.cos(a)                           # unit vector towards the source, screen
            ln = 30 + 14 * min(1.0, speed / 4.0)
            tail = (c[0] + fx * (ln + 8), c[1] + fy * (ln + 8))
            head = (c[0] + fx * 14, c[1] + fy * 14)
            col = gc.AMBER if hot else gc.DIM
            pygame.draw.line(canvas, col, tail, head, 5)
            nx, ny = -fy, fx
            pygame.draw.polygon(canvas, col, [(head[0] + fx * 12 + nx * 10, head[1] + fy * 12 + ny * 10),
                                              (head[0] + fx * 12 - nx * 10, head[1] + fy * 12 - ny * 10),
                                              (head[0] - fx * 4, head[1] - fy * 4)])
            where = "ahead" if abs(wf) < 5 else ("behind" if abs(wf) > 175 else
                                                "%.0f deg %s" % (abs(wf), "left" if wf > 0 else "right"))
            hud.text(canvas, "from " + where, (r.x + 130, r.y + 70), 18, gc.TEXT)
        hud.text(canvas, "%.1f m/s" % speed, (r.x + 130, r.y + 34), 30, gc.AMBER if hot else gc.TEXT, bold=True,
                 display=True)
        hud.text(canvas, "Earth equiv. %.2f m/s" % (speed * WIND_EQUIV), (r.x + 130, r.y + 94), 15, gc.MUTED)
        # the antennal deflections handed to fb.wind (the shipped antenna model's output)
        dl, dr = self.defl
        y = r.y + 118
        hud.text(canvas, "antenna L", (r.x + 130, y), 15, gc.TEXT)
        hud.text(canvas, "R", (r.x + 262, y), 15, gc.TEXT)
        for x0, v in ((r.x + 212, dl), (r.x + 280, dr)):
            pygame.draw.rect(canvas, gc.INSET, (x0, y + 3, 46, 12))
            wv = int(23 * max(-1.0, min(1.0, v)))
            pygame.draw.rect(canvas, gc.AMBER, (x0 + 23 + min(0, wv), y + 3, abs(wv), 12))
            pygame.draw.line(canvas, gc.LINE, (x0 + 23, y + 1), (x0 + 23, y + 16))

    def _draw_right(self, canvas):
        hud = self.hud
        x0, w = RCOL + 8, DESIGN[0] - RCOL - 16
        # the fly's eyes
        r = hud.panel(canvas, (x0, 72, w, 250), "what the fly sees")
        if self.rad is not None:
            hud.mosaic(canvas, (r.x, r.y, r.w, r.h - 16), self.eyes, gc.eye_colors(self.rad, "human", 2.0))
        hud.text(canvas, "the radiance handed to fb.vision · UV not shown", (r.centerx, r.bottom - 14), 13,
                 gc.MUTED, anchor="midtop")
        # wind DNs (connectome)
        r = hud.panel(canvas, (x0, 330, w, 206), "wind DNs  DNp18 · DNp33", "connectome")
        ws = self.wind_dec.value or {"p18L": 0.0, "p18R": 0.0, "p33L": 0.0, "p33R": 0.0}
        hud.text(canvas, "DNp18 L %.0f · R %.0f" % (ws["p18L"], ws["p18R"]), (r.x, r.y - 2), 22, gc.SAGE, bold=True)
        hud.text(canvas, "DNp33 L %.0f · R %.0f Hz" % (ws["p33L"], ws["p33R"]), (r.x, r.y + 24), 22, gc.LILAC,
                 bold=True)
        tr = (r.x, r.y + 56, r.w, r.h - 56)
        hud.trace(canvas, tr, self.hist["u"], -60, 60, gc.DIM, fill=False)
        self._band(canvas, tr, -WIND_DEAD, WIND_DEAD, -60, 60)
        self._hline(canvas, tr, 0.0, -60, 60, gc.LINE)
        self._overlay_line(canvas, tr, self.hist["u_s"], -60, 60, gc.WHITE)
        hud.text(canvas, "wind reported L", (tr[0] + 4, tr[1] + 2), 13, gc.MUTED)
        hud.text(canvas, "R", (tr[0] + 4, tr[1] + tr[3] - 16), 13, gc.MUTED)
        lab = hud.font(15).render("L-R index %+.0f Hz" % self.hist["u_s"][-1], True, gc.WHITE)
        box = lab.get_rect(topright=(tr[0] + tr[2] - 4, tr[1] + 2)).inflate(8, 2)
        canvas.fill(gc.INSET, box)
        canvas.blit(lab, lab.get_rect(center=box.center))
        # wind steer (decoder)
        r = hud.panel(canvas, (x0, 544, w, 162), "wind steer", "decoder")
        applied = "steer" in self.apply
        yaw = (self.wind_dec.value["yaw_rate"] / DEG if self.wind_dec.value else 0.0) + 0.0
        state = "" if applied and not self.held else (" held" if applied else " off")
        hud.text(canvas, "%s%.0f deg/s%s" % ("L " if yaw > 0.5 else ("R " if yaw < -0.5 else ""), abs(yaw), state),
                 (r.x, r.y - 2), 26, gc.TEAL if applied and not self.held else gc.MUTED, bold=True)
        note = "held (GAME)" if self.held else ("not applied" if not applied else "+ autopilot")
        hud.text(canvas, note, (r.right, r.y + 4), 14, gc.AMBER if self.held else gc.MUTED, anchor="topright")
        tr = (r.x, r.y + 30, r.w, r.h - 30)
        hud.trace(canvas, tr, self.hist["yaw"], -WIND_MAX / DEG, WIND_MAX / DEG, gc.TEAL if applied else gc.MUTED,
                  fill=False)
        self._hline(canvas, tr, 0.0, -WIND_MAX / DEG, WIND_MAX / DEG, gc.LINE)
        hud.text(canvas, "left", (tr[0] + 4, tr[1] + 2), 13, gc.MUTED)
        hud.text(canvas, "right", (tr[0] + 4, tr[1] + tr[3] - 16), 13, gc.MUTED)
        hud.text(canvas, "+-%.0f deg/s" % (WIND_MAX / DEG), (tr[0] + tr[2] - 4, tr[1] + 3), 13, gc.MUTED,
                 anchor="topright")
        # giant fibre
        r = hud.panel(canvas, (x0, 714, w, 150), "giant fibre DNp01", "connectome")
        gf = self.hist["gf"]
        hud.trace(canvas, (r.x, r.y + 30, r.w, r.h - 30), gf, 0, 60, gc.RED, threshold=GF_HZ)
        hud.text(canvas, "%.0f Hz" % gf[-1], (r.x, r.y - 2), 26, gc.RED if gf[-1] >= GF_HZ else gc.TEXT, bold=True)
        hud.text(canvas, "stop at 33 Hz", (r.right, r.y + 2), 16, gc.MUTED, anchor="topright")
        # loom detectors
        r = hud.panel(canvas, (x0, 872, w, 152), "LPLC2 + LC4  L · R", "connectome")
        L, R = self.hist["loomL"], self.hist["loomR"]
        vmax = max(1.0, max(max(L), max(R)) * 1.15)
        tr = (r.x, r.y + 30, r.w, r.h - 30)
        hud.trace(canvas, tr, L, 0, vmax, gc.SAGE, fill=False)
        self._overlay_line(canvas, tr, R, 0, vmax, gc.LILAC)
        hud.text(canvas, "L %.1f" % L[-1], (r.x, r.y - 2), 24, gc.SAGE, bold=True)
        hud.text(canvas, "R %.1f Hz" % R[-1], (r.x + 104, r.y - 2), 24, gc.LILAC, bold=True)
        hud.text(canvas, "scale 0-%.1f" % vmax, (r.right, r.y + 2), 14, gc.MUTED, anchor="topright")

    def _band(self, canvas, rect, lo, hi, vmin, vmax):
        """Shade the band lo..hi of a trace (the decoder's dead band)."""
        import pygame
        x, y, w, h = rect
        y0 = y + h - 1 - (hi - vmin) / (vmax - vmin) * (h - 2)
        y1 = y + h - 1 - (lo - vmin) / (vmax - vmin) * (h - 2)
        band = pygame.Surface((w, max(1, int(y1 - y0))), pygame.SRCALPHA)
        band.fill((*gc.LINE, 120))
        canvas.blit(band, (x, int(y0)))

    def _hline(self, canvas, rect, v, vmin, vmax, color):
        import pygame
        x, y, w, h = rect
        yy = y + h - 1 - (v - vmin) / (vmax - vmin) * (h - 2)
        pygame.draw.line(canvas, color, (x, yy), (x + w, yy), 1)

    def _overlay_line(self, canvas, rect, values, vmin, vmax, color):
        import pygame
        rect = pygame.Rect(rect)
        v = np.asarray(values, float)
        x = rect.x + np.linspace(0, rect.w - 1, len(v))
        y = rect.bottom - 1 - np.clip((v - vmin) / (vmax - vmin), 0, 1) * (rect.h - 2)
        pygame.draw.lines(canvas, color, False, np.c_[x, y].round().astype(int).tolist(), 2)

    def _draw_bottom(self, canvas):
        import pygame
        hud = self.hud
        y0, h = VIEW[1] + VIEW[3] + 6, DESIGN[1] - 48 - (VIEW[1] + VIEW[3] + 6) - 6
        s = self.tally.summary()
        stops = ("%d" % self.stats["hazard_stops_applied"] if "stop" in self.apply
                 else "%d not applied" % self.stats["gf_crossings"])
        steers = ("%d" % self.stats["veers_applied"] if "steer" in self.apply
                  else "%d not applied" % self.stats["veers"])
        cells = [("HAZARDS PASSED", "%d / %d" % (s["passed"], s["hazards"]), gc.SAGE, "game"),
                 ("COLLISIONS", "%d" % self.stats["collisions"], gc.AMBER, "game"),
                 ("HAZARD STOPS", stops, gc.RED, "decoder"),
                 ("WIND VEERS", steers, gc.TEAL, "decoder"),
                 ("SPEED", "%+.1f m/s" % (self.drive.v + 0.0), gc.TEXT, "game"),
                 ("TO WAYPOINT", "%.0f m" % max(0.0, ROUTE_END - self.x), gc.TEXT, "game")]
        cw = (VIEW[2] - 16) / len(cells)
        for i, (label, val, col, chip) in enumerate(cells):
            r = pygame.Rect(8 + i * cw, y0, cw - 8, h)
            pygame.draw.rect(canvas, gc.PANEL, r, border_radius=6)
            pygame.draw.rect(canvas, gc.LINE, r, 1, border_radius=6)
            t = hud.text(canvas, label, (r.x + 12, r.y + 8), 15, gc.MUTED, bold=True)
            hud.chip(canvas, chip, (t.right + 8, r.y + 6), 11)
            hud.text(canvas, val, (r.x + 12, r.y + 32), 34 if len(val) < 10 else 26, col, bold=True, display=True)

    def _draw_minimap(self, canvas, rect):
        import pygame
        s = pygame.Surface(rect.size, pygame.SRCALPHA)
        s.fill((9, 12, 14, 170))
        x_lo = self.x - 25
        scale = rect.w / 110.0

        def P(x, y):
            return (int((x - x_lo) * scale), int(rect.h / 2 - y * scale))
        pygame.draw.line(s, (*gc.DIM, 255), P(x_lo, 0), P(x_lo + 110, 0), 1)
        for r in self.rocks:
            if x_lo - 5 < r.pos[0] < x_lo + 115 and abs(r.pos[1]) < rect.h / 2 / scale + 3:
                col = gc.RED if r.hazard else (120, 96, 80)
                pygame.draw.circle(s, (*col, 255), P(r.pos[0], r.pos[1]), max(2, int(r.r_ground * scale)))
        for dv in self.devils:
            cx, cy = dv["pos"](self.t_s)
            if x_lo - 10 < cx < x_lo + 120 and abs(cy) < rect.h / 2 / scale + 5:
                pygame.draw.circle(s, (*gc.AMBER, 255), P(cx, cy), max(3, int(dv["rc"] * scale)), 2)
        if len(self.trail) > 1:
            pts = [P(x, y) for x, y in self.trail if x > x_lo - 2]
            if len(pts) > 1:
                pygame.draw.lines(s, (*gc.TEAL, 255), False, pts, 2)
        c, sn = math.cos(self.yaw), math.sin(self.yaw)
        corners = [(self.x + c * a - sn * b, self.y + sn * a + c * b) for a, b in
                   ((ROVER_L / 2, ROVER_W / 2), (ROVER_L / 2, -ROVER_W / 2), (-ROVER_L / 2, -ROVER_W / 2),
                    (-ROVER_L / 2, ROVER_W / 2))]
        pygame.draw.polygon(s, (*gc.WHITE, 255), [P(*q) for q in corners])
        canvas.blit(s, rect.topleft)
        self.hud.text(canvas, "ROUTE", (rect.x + 8, rect.y + 6), 13, gc.MUTED, bold=True)
        self.hud.chip(canvas, "game", (rect.x + 62, rect.y + 4), 10)


SERIES_COLS = ["t_s", "x_m", "y_m", "yaw_deg", "v_m_s", "loomL_hz", "loomR_hz", "gf_hz", "mdn_hz", "DNp18_L",
               "DNp18_R", "DNp33_L", "DNp33_R", "u_hz", "u_s_hz", "yaw_decoder_deg_s", "yaw_decoder_applied_deg_s",
               "steer_held", "yaw_autopilot_deg_s",
               "wind_mars_m_s", "wind_from_deg", "defl_L", "defl_R", "D3_distance_m", "D3_bearing_deg",
               "footprint_gap_m"]


def add_args(ap):
    ap.add_argument("--control", choices=("on", "off", "nowind", "hidden"), default="on",
                    help="off: both decoders read but neither is applied; nowind: the devils' wind is not delivered "
                         "(fb.wind(0, 0)); hidden: the hazard boulders are removed from the fly's scene only (the "
                         "camera still shows them); decoders applied in nowind and hidden")
    ap.add_argument("--apply", default="steer,stop", help="which decoder outputs are applied (ablations): steer,stop")
    ap.add_argument("--eye", choices=sorted(EYE_MOUNTS), default=EYE_MOUNT,
                    help="where the fly's eyes sit on the rover (dev; the clips use the default)")
    ap.add_argument("--no-eye-graph", action="store_true",
                    help="trace the eye eagerly every tick instead of replaying a CUDA graph (same kernels)")
    return ap


def run_summary(game):
    s = game.tally.summary()
    s.update({k: (round(v, 2) if isinstance(v, float) else v) for k, v in game.stats.items()})
    s["x_end_m"] = round(game.x, 2)
    s["finished"] = game.finished()
    s["devils_closest"] = {k: {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                           for k, v in game.devil_min.items()}
    s["D3_encounter"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in game.enc.items()}
    s["D3_encounter"]["within_m"] = ENCOUNTER_M
    s["series_cols"] = SERIES_COLS
    s["series_10hz"] = game.series
    return s


def main(argv=None):
    # SDL otherwise traps SIGTERM (it posts a quit event the recorder never reads), so a preempted cluster job's
    # recording kept running beside its retry, both writing the same mp4 (games.common.run sets this too)
    os.environ.setdefault("SDL_NO_SIGNAL_HANDLERS", "1")
    ap = add_args(gc.standard_args(__doc__.splitlines()[0], seconds=70.0))
    ap.add_argument("--log-only", metavar="JSON", default=None,
                    help="dev: tick headless for --seconds without the camera or the HUD; write only the run log")
    ap.add_argument("--view-scale", type=float, default=1.0,
                    help="dev: trace the main view at this fraction of its resolution and upscale (the clips use 1)")
    args = ap.parse_args(argv)
    game = MarsRover(args)
    if args.log_only:
        import time
        t0 = time.time()
        for k in range(int(round(args.seconds * 1000 / gc.TICK_MS))):
            game.tick()
            if k % 100 == 0:
                print(f"  mars: {game.t_s:5.1f} s brain, {time.time() - t0:6.1f} s wall, x {game.x:6.1f} "
                      f"y {game.y:+5.2f} u_s {game.hist['u_s'][-1]:+5.1f} gf {game.hist['gf'][-1]:4.1f}", flush=True)
            if game.finished():
                break
        game.log.meta["wall_s"] = round(time.time() - t0, 1)
        game.log.meta["brain_s"] = round(game.t_s, 3)
        game.log.meta["log_only"] = True
        path = args.log_only
    else:
        path = gc.run(game, args)
    if path:
        import json
        s = run_summary(game)
        game.log.summary = s
        game.log.save(path, game.brains())
        print(json.dumps({k: v for k, v in s.items() if k not in ("per_hazard", "series_10hz", "series_cols")}))


if __name__ == "__main__":
    main()
