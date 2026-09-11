"""A tiny spectral ray tracer (torch) for fly-scale scenes: a room, a table, fruit.

Light is carried in 4 channels [UV, B, G, R]. Materials have a 4-channel reflectance so that fruit can
look different to a fly (UV!) than to us. Geometry is spheres/ellipsoids, axis-aligned boxes and planes,
shaded with one point light + ambient, Lambertian, with hard shadows. Units: metres. World frame:
x forward, y left, z up (right-handed).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .device import default_device

INF = 1e9


def _hash3(i: torch.Tensor) -> torch.Tensor:
    """Deterministic pseudo-random value in [-1, 1] per integer lattice point (M, 3)."""
    h = (i[:, 0] * 374761393 + i[:, 1] * 668265263 + i[:, 2] * 2147483647) % 2147483647
    h = (h ^ (h >> 13)) * 1274126177 % 2147483647
    return (h % 65536).float() / 32768.0 - 1.0


def value_noise(p: torch.Tensor, scale: float) -> torch.Tensor:
    """Trilinearly interpolated lattice noise in [-1, 1] at spatial period `scale` (metres), (M,)."""
    q = p / scale
    i0 = torch.floor(q).long(); f = q - i0.float()
    f = f * f * (3 - 2 * f)                                          # smoothstep
    out = torch.zeros(p.shape[0], device=p.device)
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1 - f[:, 0]
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1 - f[:, 1]
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1 - f[:, 2]
                out += wx * wy * wz * _hash3(i0 + torch.tensor([dx, dy, dz], device=p.device))
    return out


@dataclass
class Material:
    name: str
    refl: tuple            # (UV, B, G, R) reflectance in [0, 1]
    emit: tuple = (0, 0, 0, 0)
    pattern: int = 0       # 0 plain, 1 checked cloth (refl2 = second colour), 2 vertical wall stripes, 3 floor planks
    refl2: tuple = (0, 0, 0, 0)
    scale: float = 0.1     # pattern period (m)


# Reflectances are rough guesses; fruit skins often reflect UV weakly, leaves strongly in green.
MATERIALS = {
    "wall":     Material("wall",   (0.45, 0.92, 0.95, 0.95), pattern=2, refl2=(0.20, 0.45, 0.50, 0.55), scale=0.5),
    "floor":    Material("floor",  (0.04, 0.10, 0.16, 0.24), pattern=3, refl2=(0.02, 0.05, 0.09, 0.14), scale=0.15),
    "ceiling":  Material("ceiling", (0.40, 0.90, 0.90, 0.90)),
    "table":    Material("table",  (0.04, 0.10, 0.18, 0.30)),      # dark wood (legs)
    "cloth":    Material("cloth",  (0.50, 0.95, 0.95, 0.95), pattern=1, refl2=(0.05, 0.08, 0.15, 0.85), scale=0.03),  # red-white picnic cloth (3 cm checks)
    "apple":    Material("apple",  (0.05, 0.08, 0.20, 0.85)),      # red apple
    "banana":   Material("banana", (0.10, 0.10, 0.85, 0.90)),      # yellow
    "orange":   Material("orange", (0.05, 0.05, 0.50, 0.95)),
    "grape":    Material("grape",  (0.30, 0.40, 0.15, 0.30)),      # purple-ish, some UV
    "lime":     Material("lime",   (0.15, 0.20, 0.80, 0.30)),
    "blueberry": Material("blueberry", (0.55, 0.70, 0.20, 0.20)),  # bluish with UV bloom
    "plate":    Material("plate",  (0.60, 0.95, 0.95, 0.95)),
    "lamp":     Material("lamp",   (0, 0, 0, 0), emit=(0.6, 1.0, 1.0, 1.0)),
    "black":    Material("black",  (0.01, 0.02, 0.02, 0.02)),      # looming object / predator
}


@dataclass
class Sphere:
    center: tuple
    radii: tuple          # (rx, ry, rz) -> ellipsoid
    material: str


@dataclass
class Box:
    lo: tuple
    hi: tuple
    material: str


@dataclass
class Plane:
    point: tuple
    normal: tuple
    material: str


@dataclass
class World:
    spheres: list = field(default_factory=list)
    boxes: list = field(default_factory=list)
    planes: list = field(default_factory=list)
    light_pos: tuple = (0.0, 0.0, 2.3)
    light_color: tuple = (0.9, 1.8, 1.8, 1.8)    # UV, B, G, R power (a bright lamp with some UV)
    ambient: tuple = (0.12, 0.15, 0.15, 0.15)
    detail: float = 1.4                          # amplitude of the multi-scale surface noise (0 = flat colours)
    device: torch.device = field(default_factory=default_device)

    # ---------------------------------------------------------------- dynamic objects
    def move_sphere(self, idx: int, center, radii=None) -> None:
        """Move (and optionally resize) a sphere; the packed GPU tensors are refreshed on the next trace."""
        s = self.spheres[idx]
        s.center = tuple(float(v) for v in center)
        if radii is not None:
            s.radii = tuple(float(v) for v in radii)
        self._packed = False

    # ---------------------------------------------------------------- packing
    def _pack(self):
        d = self.device
        mats = [MATERIALS[s.material] for s in self.spheres] + [MATERIALS[b.material] for b in self.boxes] + \
               [MATERIALS[p.material] for p in self.planes]
        self._refl = torch.tensor([m.refl for m in mats] + [(0, 0, 0, 0)], dtype=torch.float32, device=d)
        self._refl2 = torch.tensor([m.refl2 for m in mats] + [(0, 0, 0, 0)], dtype=torch.float32, device=d)
        self._pattern = torch.tensor([m.pattern for m in mats] + [0], dtype=torch.long, device=d)
        self._pscale = torch.tensor([m.scale for m in mats] + [1.0], dtype=torch.float32, device=d)
        self._emit = torch.tensor([m.emit for m in mats] + [(0, 0, 0, 0)], dtype=torch.float32, device=d)
        self._sc = torch.tensor([s.center for s in self.spheres] or np.zeros((0, 3)), dtype=torch.float32, device=d)
        self._sr = torch.tensor([s.radii for s in self.spheres] or np.zeros((0, 3)), dtype=torch.float32, device=d)
        self._blo = torch.tensor([b.lo for b in self.boxes] or np.zeros((0, 3)), dtype=torch.float32, device=d)
        self._bhi = torch.tensor([b.hi for b in self.boxes] or np.zeros((0, 3)), dtype=torch.float32, device=d)
        self._pp = torch.tensor([p.point for p in self.planes] or np.zeros((0, 3)), dtype=torch.float32, device=d)
        pn = torch.tensor([p.normal for p in self.planes] or np.zeros((0, 3)), dtype=torch.float32, device=d)
        self._pn = pn / pn.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        self._lp = torch.tensor(self.light_pos, dtype=torch.float32, device=d)
        self._lc = torch.tensor(self.light_color, dtype=torch.float32, device=d)
        self._amb = torch.tensor(self.ambient, dtype=torch.float32, device=d)
        self._packed = True

    # ---------------------------------------------------------------- intersection
    def _intersect(self, o: torch.Tensor, d: torch.Tensor):
        """o, d: (M, 3). Returns t (M,), normal (M, 3), material id (M,) (-1 = miss -> last slot)."""
        M = o.shape[0]
        best_t = torch.full((M,), INF, device=o.device)
        best_n = torch.zeros(M, 3, device=o.device)
        best_id = torch.full((M,), self._refl.shape[0] - 1, dtype=torch.long, device=o.device)
        eps = 1e-4
        # ellipsoids: scale space so they become unit spheres
        if len(self.spheres):
            oc = (o[:, None, :] - self._sc[None]) / self._sr[None]           # (M, S, 3)
            dd = d[:, None, :] / self._sr[None]
            a = (dd * dd).sum(-1); b = 2 * (oc * dd).sum(-1); c = (oc * oc).sum(-1) - 1
            disc = b * b - 4 * a * c
            ok = disc > 0
            sq = torch.sqrt(disc.clamp_min(0))
            t0 = (-b - sq) / (2 * a); t1 = (-b + sq) / (2 * a)
            t = torch.where(t0 > eps, t0, t1)
            t = torch.where(ok & (t > eps), t, torch.full_like(t, INF))
            tmin, si = t.min(dim=1)
            hit = tmin < best_t
            pt = o + d * tmin[:, None]
            n = (pt - self._sc[si]) / (self._sr[si] ** 2)
            n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            best_n = torch.where(hit[:, None], n, best_n); best_t = torch.where(hit, tmin, best_t)
            best_id = torch.where(hit, si, best_id)
        if len(self.boxes):
            inv = 1.0 / torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
            t_lo = (self._blo[None] - o[:, None, :]) * inv[:, None, :]       # (M, B, 3)
            t_hi = (self._bhi[None] - o[:, None, :]) * inv[:, None, :]
            tn = torch.minimum(t_lo, t_hi); tf = torch.maximum(t_lo, t_hi)
            t_enter, ax = tn.max(dim=-1); t_exit = tf.min(dim=-1).values
            t = torch.where((t_exit > t_enter) & (t_enter > eps), t_enter, torch.full_like(t_enter, INF))
            tmin, bi = t.min(dim=1)
            hit = tmin < best_t
            axis = ax.gather(1, bi[:, None])[:, 0]
            sign = -torch.sign(d.gather(1, axis[:, None])[:, 0])
            n = torch.zeros(M, 3, device=o.device); n.scatter_(1, axis[:, None], sign[:, None])
            best_n = torch.where(hit[:, None], n, best_n); best_t = torch.where(hit, tmin, best_t)
            best_id = torch.where(hit, bi + len(self.spheres), best_id)
        if len(self.planes):
            denom = (d[:, None, :] * self._pn[None]).sum(-1)                  # (M, P)
            t = ((self._pp[None] - o[:, None, :]) * self._pn[None]).sum(-1) / torch.where(denom.abs() < 1e-9, torch.full_like(denom, 1e-9), denom)
            t = torch.where((t > eps) & (denom.abs() > 1e-9), t, torch.full_like(t, INF))
            tmin, pi = t.min(dim=1)
            hit = tmin < best_t
            n = self._pn[pi]
            n = torch.where((n * d).sum(-1, keepdim=True) > 0, -n, n)
            best_n = torch.where(hit[:, None], n, best_n); best_t = torch.where(hit, tmin, best_t)
            best_id = torch.where(hit, pi + len(self.spheres) + len(self.boxes), best_id)
        return best_t, best_n, best_id

    @torch.no_grad()
    def trace(self, origins: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
        """Radiance (M, 4) for rays from origins (M,3) along unit dirs (M,3)."""
        if not getattr(self, "_packed", False):
            self._pack()
        o = origins.to(self.device, torch.float32); d = dirs.to(self.device, torch.float32)
        t, n, mid = self._intersect(o, d)
        hit = t < INF
        p = o + d * t[:, None]
        to_l = self._lp[None] - p
        dist = to_l.norm(dim=-1, keepdim=True)
        l = to_l / dist.clamp_min(1e-9)
        lam = (n * l).sum(-1).clamp_min(0)
        # shadow ray
        ts, _, _ = self._intersect(p + n * 1e-3, l)
        lit = (ts >= dist[:, 0]).float()
        falloff = 4.0 / (1.0 + dist[:, 0] ** 2)
        refl = self._texture(p, n, mid)
        rad = refl * (self._amb[None] + self._lc[None] * (lam * lit * falloff)[:, None]) + self._emit[mid]
        return torch.where(hit[:, None], rad, torch.zeros_like(rad))

    def _texture(self, p: torch.Tensor, n: torch.Tensor, mid: torch.Tensor) -> torch.Tensor:
        """Procedural surface patterns: pick refl or refl2 per hit point."""
        pat = self._pattern[mid]; sc = self._pscale[mid]
        fx = torch.floor(p[:, 0] / sc); fy = torch.floor(p[:, 1] / sc)
        checks = ((fx + fy) % 2) == 1
        # walls: stripes along the wall's horizontal coordinate (whichever of x, y is not the normal)
        horiz = torch.where(n[:, 0].abs() > 0.5, fy, fx)
        stripes = (horiz % 2) == 1
        planks = ((fy % 2) == 1) | (((fx + 3 * fy) % 7) == 0)
        second = torch.where(pat == 1, checks, torch.where(pat == 2, stripes, torch.where(pat == 3, planks, torch.zeros_like(checks))))
        refl = torch.where(second[:, None], self._refl2[mid], self._refl[mid])
        # fine, non-periodic detail at fly scale: value noise at 2 cm, 8 mm and 3 mm (weave, grain, fibres,
        # fruit-skin speckle). A fly 1 mm above a surface sees millimetre structure as degrees of texture.
        if self.detail > 0:
            m = 1.0 + self.detail * (0.5 * value_noise(p, 0.02) + 0.3 * value_noise(p, 0.008) + 0.2 * value_noise(p, 0.003))
            refl = refl * m.clamp_min(0.1)[:, None]
        return refl

    @torch.no_grad()
    def render_camera(self, pos, forward, up, width=320, height=200, fov_deg=90.0) -> torch.Tensor:
        """Pinhole render -> (height, width, 4) radiance."""
        pos = torch.tensor(pos, dtype=torch.float32, device=self.device)
        f = torch.tensor(forward, dtype=torch.float32, device=self.device); f = f / f.norm()
        u = torch.tensor(up, dtype=torch.float32, device=self.device)
        r = torch.cross(f, u, dim=0); r = r / r.norm(); u = torch.cross(r, f, dim=0)
        asp = width / height
        tan = np.tan(np.deg2rad(fov_deg) / 2)
        xs = torch.linspace(-tan * asp, tan * asp, width, device=self.device)
        ys = torch.linspace(tan, -tan, height, device=self.device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        d = f[None, None] + gx[..., None] * (-r)[None, None] + gy[..., None] * u[None, None]
        d = d / d.norm(dim=-1, keepdim=True)
        d = d.reshape(-1, 3)
        o = pos[None].expand_as(d)
        return self.trace(o, d).reshape(height, width, 4)


def make_room(seed: int = 0) -> tuple[World, dict]:
    """A 4 x 4 x 2.6 m room with a table in the middle, fruit on the table. Returns (world, info)."""
    w = World()
    # room: floor z=0, ceiling z=2.6, walls at x=+-2, y=+-2
    w.planes += [Plane((0, 0, 0), (0, 0, 1), "floor"), Plane((0, 0, 2.6), (0, 0, -1), "ceiling"),
                 Plane((2, 0, 0), (-1, 0, 0), "wall"), Plane((-2, 0, 0), (1, 0, 0), "wall"),
                 Plane((0, 2, 0), (0, -1, 0), "wall"), Plane((0, -2, 0), (0, 1, 0), "wall")]
    # landmarks: a dark picture on the +y wall, a bright door on the +x wall, a black skirting strip
    w.boxes.append(Box((-0.6, 1.99, 1.0), (0.2, 2.0, 1.6), "black"))
    w.boxes.append(Box((1.99, -1.3, 0.0), (2.0, -0.5, 2.0), "plate"))
    w.boxes.append(Box((-2.0, -2.0, 0.0), (2.0, -1.99, 0.08), "black"))
    # table: top 1.2 x 0.8 m, height 0.75
    top_z = 0.75
    w.boxes.append(Box((-0.6, -0.4, top_z - 0.03), (0.6, 0.4, top_z), "cloth"))
    for sx in (-0.55, 0.55):
        for sy in (-0.35, 0.35):
            w.boxes.append(Box((sx - 0.025, sy - 0.025, 0), (sx + 0.025, sy + 0.025, top_z - 0.03), "table"))
    # fruit on the table
    fruit = [
        Sphere((0.25, 0.15, top_z + 0.04), (0.04, 0.04, 0.04), "apple"),
        Sphere((-0.20, -0.10, top_z + 0.035), (0.035, 0.035, 0.035), "orange"),
        Sphere((0.05, -0.22, top_z + 0.02), (0.09, 0.02, 0.02), "banana"),
        Sphere((-0.30, 0.20, top_z + 0.025), (0.025, 0.025, 0.025), "lime"),
    ]
    rng = np.random.default_rng(seed)
    for k in range(9):   # a bunch of grapes
        fruit.append(Sphere((0.35 + 0.02 * rng.normal(), -0.05 + 0.015 * (k % 3) + 0.01 * rng.normal(), top_z + 0.008 + 0.006 * (k // 3)),
                            (0.008, 0.008, 0.008), "grape"))
    for k in range(6):
        fruit.append(Sphere((-0.05 + 0.03 * rng.normal(), 0.28 + 0.03 * rng.normal(), top_z + 0.006), (0.006, 0.006, 0.006), "blueberry"))
    w.spheres += fruit
    w.spheres.append(Sphere(w.light_pos, (0.08, 0.08, 0.08), "lamp"))
    info = {"table_top_z": top_z, "table_extent": (-0.6, 0.6, -0.4, 0.4),
            "fruit": [(s.material, s.center, max(s.radii)) for s in fruit]}
    return w, info


def to_rgb8(rad: torch.Tensor, exposure: float = 1.0, gamma: float = 2.2) -> np.ndarray:
    """(…, 4) radiance -> uint8 RGB for humans (drops UV)."""
    x = rad[..., [3, 2, 1]].clamp_min(0) * exposure
    x = (x / (1 + x)) ** (1 / gamma)
    return (x.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()


def to_fly_false_color(rad: torch.Tensor, exposure: float = 1.0, gamma: float = 2.2) -> np.ndarray:
    """(…, 4) radiance -> uint8 false-colour: UV->magenta, B->blue, G->green (what the fly channels carry)."""
    uv, b, g = rad[..., 0], rad[..., 1], rad[..., 2]
    x = torch.stack([uv, g, 0.5 * b + 0.5 * uv], dim=-1).clamp_min(0) * exposure
    x = (x / (1 + x)) ** (1 / gamma)
    return (x.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
