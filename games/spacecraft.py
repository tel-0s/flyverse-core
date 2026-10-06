"""spacecraft -- 6DOF: a fly flies a spaceship (games/PLAN.md section 3).

A rigid ship (position, velocity, orientation quaternion, angular velocity) starts tumbling in low orbit near a
rotating ring station while five rocks are sent at it. The fly sits in the cockpit; its 1,466 ommatidial columns look
out along the ship's axes (the hull is not drawn in its view). What it sees is ray traced from the same scene the chase
camera shows: a nebula-filled sky, the planet, sun and moon at infinity (so a pure rotation gives a purely rotational
image flow), the station and the rocks in 3D.

Brain -> thrusters, every link a declared decoder (`games.common.ReadDecoder`, read-only) reading a population that
carried that signal on dev seeds >= 100 (numbers in games/captions/spacecraft.md). Each hold signal is side-equalised
(the beta term) and low-passed 0.5 s:

    yaw hold        s = (H2 - 0.47 HSN - 1.15 HSE)(L - R) + 0.20 sum|w|(L + R)  ->  yaw accel = -4 s  (deg/s^2 per Hz)
    roll hold       s = VST2 (L - R) + 0.31 (L + R)                             ->  roll accel = -3 s
    (pitch)         not built: no lobula-plate population carried it
    dodge           LPLC2 + LC4 left minus right, |.| >= 5 Hz -> a sideways push away from the looming side
    escape burn     giant fibre DNp01 >= 33 Hz (the body model's takeoff line) -> a forward / up burn

Everything else is GAME: the physics, the rocks, the cruise autopilot, the station, the camera. The seed draws the
scenario (the initial tumble and each rock's direction, aim and shape) on its own RNG stream; the raw brain has no
Poisson input, so without that the seed would change nothing.

    python games/spacecraft.py                        # interactive; keys: K kick the ship, R send a rock, C thrusters on/off
    python games/spacecraft.py --seed 100 --seconds 30 --record out/games/spacecraft/dev.mp4
    python games/spacecraft.py --seed 100 --seconds 30 --record out/games/spacecraft/dev_control.mp4 --control
    python games/spacecraft.py --seed 100 --seconds 15 --record out/games/spacecraft/dev_still.mp4 --still
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (puts the repo on sys.path)
import torch  # noqa: E402

DEG = math.pi / 180.0


# ======================================================================================================== math
def quat_mul(a, b):
    """Hamilton product of quaternions (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def quat_to_matrix(q):
    """Unit quaternion (w, x, y, z) -> 3x3 rotation matrix whose columns are the body axes in the world frame."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    s = math.sin(angle / 2)
    return np.array([math.cos(angle / 2), *(axis * s)])


def rot_axis(axis, angle):
    return quat_to_matrix(quat_from_axis_angle(axis, angle))


def look_basis(forward, up=(0, 0, 1)):
    """(forward, left, up) unit vectors of a frame looking along `forward` (world z up unless told)."""
    f = np.asarray(forward, float)
    f = f / np.linalg.norm(f)
    u = np.asarray(up, float)
    left = np.cross(u, f)
    if np.linalg.norm(left) < 1e-6:
        left = np.cross((1.0, 0.0, 0.0), f)
    left /= np.linalg.norm(left)
    return f, left, np.cross(f, left)


def unit_from_az_el(az_deg, el_deg):
    a, e = az_deg * DEG, el_deg * DEG
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


# ======================================================================================================== scene
# World frame: x, y, z (right-handed, z points away from the planet). Units: metres, seconds. Colours: linear RGB;
# the fly's radiance is rgb_to_radiance(rgb) (UV = 0.5 B, logged by RunLog as `uv_from_rgb`).
PLANET_DIR = np.array([0.0, 0.0, -1.0])        # direction to the planet's centre (a sphere at infinity)
PLANET_ANG = 66.0                              # its angular radius, deg: low orbit -- the limb sits 24 deg below level
SUN_AZ = 0.0                                   # the sun stands over the limb straight ahead (+x) ...
SUN_EL0, SUN_RISE = 6.0, 0.0                   # ... 30 deg above it, so the planet below is day-lit (the fly's
                                               # photoreceptors code temporal contrast over I + 0.02: a dark planet
                                               # would be nearly invisible to it)
SUN_RADIUS = 1.3                               # deg (enlarged for the camera)
MOON_DIR = unit_from_az_el(172.0, 12.0)        # a nearly full moon, opposite the sun
MOON_RADIUS = 2.6
GALAXY_POLE = unit_from_az_el(140.0, 35.0)     # the Milky Way's great circle is normal to this
GALAXY_CENTRE = unit_from_az_el(35.0, 22.0)

SUN_RGB = np.array([1.0, 0.93, 0.82])
NEBULA = 0.22                                  # nebula brightness (the planet's day side is ~0.3)
PLANETSHINE_RGB = np.array([0.20, 0.30, 0.45])
N_LOBES = 4                                    # a rock is the union of this many ellipsoids (GAME art and shape)
ROCK_ALBEDO = (0.21, 0.19, 0.17)               # dark grey-brown: reads as rock, not as a small moon
EYE_SUN = 2.2                                  # direct sunlight on local objects in the fly's eye scene
CAM_SUN = 0.62                                 # ... and in the chase camera (art: at 2.2 the sunlit hull burns out)


@dataclass
class Ellipsoid:
    center: np.ndarray                   # world (or body) position
    radii: np.ndarray
    rot: np.ndarray = field(default_factory=lambda: np.eye(3))    # local -> parent
    albedo: tuple = (0.8, 0.8, 0.8)
    kind: str = "hull"                   # hull | glass | rock | eye | dark | emit | panel
    name: str = ""


def ship_parts():
    """The ship in its own body frame (x forward, y left, z up), about 9 m long. Pure art (GAME); only the chase
    camera draws it -- the fly's view has no hull in it. The same parts are the ship's collision shape. Albedos are
    a mid grey: the sun is bright and often straight behind the chase camera, and a whiter hull burns out."""
    parts = []
    E = Ellipsoid
    parts.append(E(np.array([0.0, 0.0, 0.0]), np.array([4.3, 1.05, 0.78]), albedo=(0.56, 0.57, 0.58), kind="hull",
                   name="fuselage"))
    parts.append(E(np.array([-0.2, 0.0, -0.35]), np.array([3.4, 1.2, 0.5]), albedo=(0.30, 0.32, 0.35), kind="panel",
                   name="belly"))
    sweep = 32 * DEG
    for s in (1, -1):
        side = "left" if s > 0 else "right"
        R = rot_axis((0, 0, 1), -s * sweep)
        parts.append(E(np.array([-1.3, s * 2.2, -0.22]), np.array([1.35, 2.0, 0.08]), R, (0.52, 0.53, 0.55), "hull",
                       f"{side} wing"))
        parts.append(E(np.array([-2.55, s * 3.75, -0.2]), np.array([0.42, 0.14, 0.1]), R, (0.25, 0.55, 0.58), "panel",
                       f"{side} wingtip"))
        parts.append(E(np.array([-3.0, s * 1.25, -0.05]), np.array([1.5, 0.38, 0.38]), albedo=(0.45, 0.46, 0.48),
                       kind="hull", name=f"{side} engine pod"))
        parts.append(E(np.array([-4.47, s * 1.25, -0.05]), np.array([0.1, 0.3, 0.3]), albedo=(0.1, 0.1, 0.1),
                       kind="emit", name=f"{side} engine pod"))
    parts.append(E(np.array([-3.2, 0.0, 0.85]), np.array([1.15, 0.09, 0.95]), rot_axis((0, 1, 0), -20 * DEG),
                   (0.55, 0.56, 0.57), "hull", "tail fin"))
    parts.append(E(np.array([-4.2, 0.0, 0.0]), np.array([0.12, 0.42, 0.42]), albedo=(0.1, 0.1, 0.1), kind="emit",
                   name="main engine"))
    # the pilot, under the glass: a stylised fly (head with red eyes, thorax, folded wings)
    parts.append(E(np.array([2.08, 0.0, 0.78]), np.array([0.2, 0.26, 0.2]), albedo=(0.12, 0.10, 0.08), kind="dark"))
    for s in (1, -1):
        parts.append(E(np.array([2.18, s * 0.2, 0.82]), np.array([0.13, 0.12, 0.17]), albedo=(0.75, 0.08, 0.05), kind="eye"))
        parts.append(E(np.array([1.45, s * 0.2, 0.92]), np.array([0.55, 0.16, 0.025]), rot_axis((0, 0, 1), s * 12 * DEG),
                       (0.55, 0.6, 0.65), "dark"))
    parts.append(E(np.array([1.72, 0.0, 0.72]), np.array([0.34, 0.26, 0.24]), albedo=(0.22, 0.2, 0.14), kind="dark"))
    parts.append(E(np.array([1.2, 0.0, 0.68]), np.array([0.45, 0.24, 0.22]), albedo=(0.30, 0.25, 0.12), kind="dark"))
    return parts


CANOPY = (np.array([1.75, 0.0, 0.62]), 0.9)    # glass dome (body frame centre, radius)
EYE_BODY = np.array([2.18, 0.0, 0.82])         # where the fly's eyes sit (body frame)


def _fib_sphere(n):
    """n roughly evenly spread unit vectors (a Fibonacci sphere)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def ship_surface_points(spacing=0.28):
    """GAME collision shape: points on the surface of every hull part and the canopy (body frame), about `spacing`
    apart, and the name of the part each lies on. A rock touches the ship when one of these enters it."""
    pts, names = [], []
    parts = [p for p in ship_parts() if p.kind not in ("eye", "dark")]
    parts.append(Ellipsoid(CANOPY[0], np.array([CANOPY[1], CANOPY[1] * 0.8, CANOPY[1] * 0.75]), name="canopy"))
    for p in parts:
        a, b, c = p.radii
        area = 4 * math.pi * (((a * b) ** 1.6 + (a * c) ** 1.6 + (b * c) ** 1.6) / 3) ** (1 / 1.6)
        n = max(24, int(area / spacing ** 2))
        pts.append((_fib_sphere(n) * p.radii) @ np.asarray(p.rot).T + p.center)
        names += [p.name] * n
    return np.concatenate(pts), names


SHIP_POINTS, SHIP_POINT_PART = ship_surface_points()
SHIP_BOUND = float(np.linalg.norm(SHIP_POINTS, axis=1).max())   # bounding sphere of the collision shape, m

# thrusters, body frame: (position, exhaust direction) -- drawn when they fire
RCS = {
    "yaw+": [((3.4, -0.7, 0.0), (0, -1, 0)), ((-3.6, 0.9, 0.3), (0, 1, 0))],
    "yaw-": [((3.4, 0.7, 0.0), (0, 1, 0)), ((-3.6, -0.9, 0.3), (0, -1, 0))],
    "pitch+": [((3.3, 0.0, -0.55), (0, 0, -1)), ((-3.8, 0.0, 0.9), (0, 0, 1))],
    "pitch-": [((3.3, 0.0, 0.55), (0, 0, 1)), ((-3.8, 0.0, -0.4), (0, 0, -1))],
    "roll+": [((-2.5, 3.75, -0.2), (0, 0, -1)), ((-2.5, -3.75, -0.2), (0, 0, 1))],
    "roll-": [((-2.5, 3.75, -0.2), (0, 0, 1)), ((-2.5, -3.75, -0.2), (0, 0, -1))],
}
MAIN_ENGINES = [((-4.55, 1.25, -0.05), (-1, 0, 0)), ((-4.55, -1.25, -0.05), (-1, 0, 0)), ((-4.3, 0.0, 0.0), (-1, 0, 0))]


# -------------------------------------------------------------------------------------------------------- torch
def _hash3(i):
    """Integer lattice (M, 3) int64 -> pseudo-random values in [-1, 1]. (flyverse.world's hash drops z -- its z
    coefficient is 0 mod its modulus -- which is harmless on its flat surfaces but streaks a sky; this one mixes all
    three.)"""
    h = (i[:, 0] * 73856093) ^ (i[:, 1] * 19349663) ^ (i[:, 2] * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFF).float() / 32767.5 - 1.0


def _noise(p, scale, corners):
    """Trilinear value noise in [-1, 1] at spatial period `scale`."""
    q = p / scale
    i0 = torch.floor(q).long()
    f = q - i0.float()
    f = f * f * (3 - 2 * f)
    out = torch.zeros(p.shape[0], device=p.device)
    for k in range(8):
        dx, dy, dz = (k >> 2) & 1, (k >> 1) & 1, k & 1
        w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
        out = out + w * _hash3(i0 + corners[k])
    return out


def _fbm(p, scale, corners, octaves=4, gain=0.5):
    out = torch.zeros(p.shape[0], device=p.device)
    amp, tot = 1.0, 0.0
    for k in range(octaves):
        out = out + amp * _noise(p, scale / (2.0 ** k), corners)
        tot += amp
        amp *= gain
    return out / tot


def _smoothstep(a, b, x):
    t = ((x - a) / (b - a)).clamp(0, 1)
    return t * t * (3 - 2 * t)


class SpaceRenderer:
    """A small torch ray tracer for the space scene. Everything at infinity (stars, the Milky Way, the planet with its
    atmosphere, the moon, the sun) is a function of ray direction only, so a pure rotation of the viewer gives a purely
    rotational image flow. Local objects (the station, the asteroids, the ship) are intersected in 3D."""

    def __init__(self, device, sky_seed=7, sky_size=(2048, 1024)):
        self.device = torch.device(device)
        self.corners = torch.tensor([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)], device=self.device)
        self.sun_dir = unit_from_az_el(SUN_AZ, SUN_EL0)
        self._build_sky(sky_seed, *sky_size)
        self.parts = ship_parts()

    # ------------------------------------------------------------------------------------ the sky at infinity
    def _build_sky(self, seed, W, H):
        dev = self.device
        rng = np.random.default_rng(seed)
        lon = (0.5 - (torch.arange(W, device=dev) + 0.5) / W) * 2 * math.pi
        lat = (0.5 - (torch.arange(H, device=dev) + 0.5) / H) * math.pi
        la, lo = torch.meshgrid(lat, lon, indexing="ij")
        d = torch.stack([torch.cos(la) * torch.cos(lo), torch.cos(la) * torch.sin(lo), torch.sin(la)], -1).reshape(-1, 3)
        pole = torch.tensor(GALAXY_POLE, dtype=torch.float32, device=dev)
        gc = torch.tensor(GALAXY_CENTRE, dtype=torch.float32, device=dev)
        glat = torch.asin((d @ pole).clamp(-1, 1))
        n1 = _fbm(d * 1.0 + 3.1, 0.35, self.corners, 5)
        n2 = _fbm(d * 1.0 - 7.7, 0.12, self.corners, 4)
        band = torch.exp(-(glat / (13 * DEG)) ** 2) * (0.55 + 0.9 * n1).clamp_min(0.05)
        dust = torch.exp(-((glat - 1.5 * DEG) / (3.5 * DEG)) ** 2) * (0.5 + 0.8 * n2).clamp(0, 1)
        bulge = torch.exp(-(torch.acos((d @ gc).clamp(-1, 1)) / (16 * DEG)) ** 2)
        mw = (band * (1 - 0.75 * dust) + 1.6 * bulge * (1 - 0.6 * dust)).clamp_min(0)
        warm = torch.tensor([1.0, 0.86, 0.70], device=dev)
        cool = torch.tensor([0.62, 0.72, 1.0], device=dev)
        rgb = 0.070 * mw[:, None] * (warm * (0.6 + 0.4 * bulge[:, None]) + cool * 0.4 * (1 - bulge[:, None]))
        # nebulae: coloured gas with dark lanes over the whole sky, structure from ~3 to ~40 deg. Besides colour it
        # is what gives the fly's eye texture to see its own rotation against (a bare starfield is far below the
        # photoreceptors' contrast floor).
        gas = _fbm(d * 1.0 + 0.7, 0.55, self.corners, 6)
        fil = _fbm(d * 1.0 - 3.3, 0.22, self.corners, 5)
        lanes = _smoothstep(0.05, 0.35, _fbm(d * 1.0 + 9.1, 0.3, self.corners, 5))
        dens = (_smoothstep(-0.25, 0.45, gas) * (0.35 + 0.65 * _smoothstep(-0.2, 0.4, fil)) * (1 - 0.8 * lanes)).clamp_min(0)
        hue = _smoothstep(-0.3, 0.3, _fbm(d * 1.0 + 20.0, 0.8, self.corners, 3))
        c1 = torch.tensor([1.0, 0.32, 0.42], device=dev)                 # hydrogen red / magenta
        c2 = torch.tensor([0.25, 0.55, 1.0], device=dev)                 # reflection blue
        c3 = torch.tensor([1.0, 0.7, 0.35], device=dev)                  # dusty amber
        tint = c1 * hue[:, None] + c2 * (1 - hue)[:, None]
        tint = tint * (0.75 + 0.25 * fil[:, None]) + c3 * (0.3 * _smoothstep(0.2, 0.6, fil))[:, None]
        rgb = rgb + NEBULA * dens[:, None] * tint
        rgb = rgb + 0.0012                                              # zodiacal / airglow floor
        # stars: a catalogue (the camera draws them crisp) and the same stars blurred into the texture for the eye
        n_star = 9000
        v = rng.normal(size=(n_star, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        # pull a third of them toward the galactic plane
        k = n_star // 3
        pl = np.asarray(GALAXY_POLE)
        v[:k] -= np.outer(v[:k] @ pl, pl) * 0.85
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        mag = rng.pareto(1.6, n_star) + 1.0
        bright = np.clip(0.05 * mag, 0.05, 3.0)
        temp = rng.uniform(0, 1, n_star)
        col = np.stack([0.75 + 0.35 * temp, 0.85 + 0.1 * np.sin(temp * 3), 1.1 - 0.45 * temp], 1)
        self.star_dir = torch.tensor(v, dtype=torch.float32, device=dev)
        self.star_rgb = torch.tensor(bright[:, None] * col, dtype=torch.float32, device=dev)
        # blurred into the texture (sigma ~1 deg), flux scaled for a ~4.5 deg acceptance
        img = rgb.reshape(H, W, 3)
        slon = np.arctan2(v[:, 1], v[:, 0]); slat = np.arcsin(np.clip(v[:, 2], -1, 1))
        su = ((0.5 - slon / (2 * np.pi)) * W).astype(np.int64) % W
        sv = np.clip(((0.5 - slat / np.pi) * H).astype(np.int64), 0, H - 1)
        acc = torch.zeros(H * W, 3, device=dev)
        acc.index_add_(0, torch.tensor(sv * W + su, device=dev), self.star_rgb * 0.02)
        acc = acc.reshape(H, W, 3).permute(2, 0, 1)[None]
        ks = 9
        g = torch.exp(-0.5 * (torch.arange(ks, device=dev) - ks // 2) ** 2 / 2.5 ** 2)
        g = g / g.sum()
        acc = torch.nn.functional.conv2d(torch.nn.functional.pad(acc, (ks // 2, ks // 2, 0, 0), mode="circular"),
                                         g.view(1, 1, 1, ks).expand(3, 1, 1, ks), groups=3)
        acc = torch.nn.functional.conv2d(torch.nn.functional.pad(acc, (0, 0, ks // 2, ks // 2), mode="replicate"),
                                         g.view(1, 1, ks, 1).expand(3, 1, ks, 1), groups=3)
        self.sky_soft = (img.permute(2, 0, 1)[None] + acc).contiguous()   # (1, 3, H, W)
        self.sky_plain = img.permute(2, 0, 1)[None].contiguous()

    def _sample_sky(self, d, tex):
        lon = torch.atan2(d[:, 1], d[:, 0]); lat = torch.asin(d[:, 2].clamp(-1, 1))
        gx = -lon / math.pi                                   # grid_sample x in [-1, 1] left -> right = lon + -> -
        gy = -lat / (math.pi / 2)
        grid = torch.stack([gx, gy], -1).view(1, 1, -1, 2)
        out = torch.nn.functional.grid_sample(tex, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return out.view(3, -1).t()

    def sun_visibility(self, sun_dir=None):
        s = self.sun_dir if sun_dir is None else sun_dir
        ang = math.degrees(math.acos(float(np.clip(np.dot(s, PLANET_DIR), -1, 1))))
        return float(np.clip((ang - PLANET_ANG) / (2 * SUN_RADIUS) + 0.5, 0.0, 1.0))

    def _build_planet(self, n_tex=2048):
        """Planet albedo and night lights on the visible cap of normals (a sphere at infinity shows only normals within
        90 - PLANET_ANG deg of the viewer), stored as a texture over the normal's (x, y); shaded per frame."""
        dev = self.device
        L = math.cos(PLANET_ANG * DEG) * 1.02
        g = torch.linspace(-L, L, n_tex, device=dev)
        gy, gx = torch.meshgrid(g, g, indexing="ij")
        nz = torch.sqrt((1 - gx ** 2 - gy ** 2).clamp_min(0))
        q = torch.stack([gx, gy, nz], -1).reshape(-1, 3) * 1.0
        cont = _fbm(q + 11.0, 0.22, self.corners, 6)
        land = _smoothstep(0.0, 0.08, cont)
        cloud = _smoothstep(0.05, 0.5, _fbm(q + 5.3, 0.08, self.corners, 6)) * 0.95
        streak = _smoothstep(0.2, 0.6, _fbm(q * torch.tensor([1.0, 4.0, 1.0], device=dev) + 2.1, 0.12, self.corners, 3))
        cloud = (cloud + 0.35 * streak).clamp(0, 1)
        tone = _fbm(q, 0.03, self.corners, 3)
        ocean = torch.tensor([0.015, 0.06, 0.19], device=dev) * (1 + 0.3 * tone[:, None])
        landc = torch.tensor([0.16, 0.16, 0.07], device=dev) + torch.tensor([0.14, 0.07, 0.0], device=dev) * tone[:, None]
        alb = ocean * (1 - land)[:, None] + landc * land[:, None]
        alb = alb * (1 - cloud)[:, None] + torch.tensor([0.88, 0.9, 0.93], device=dev) * cloud[:, None]
        city = land * (1 - cloud) * _smoothstep(0.45, 0.75, _fbm(q + 2.0, 0.005, self.corners, 2))
        tex = torch.cat([alb, city[:, None]], -1).reshape(n_tex, n_tex, 4)
        self.planet_tex = tex.permute(2, 0, 1)[None].contiguous()
        self.planet_L = L

    def infinity(self, d, *, eye=False):
        """Radiance (linear RGB) of everything at infinity along unit directions d (M, 3). Returns (rgb, planet_mask)."""
        if not hasattr(self, "planet_tex"):
            self._build_planet()
        k = self._consts()
        s, pc = k["sun"], k["pc"]
        rgb = self._sample_sky(d, self.sky_soft if eye else self.sky_plain)
        sun_vis = self.sun_visibility()
        sunc = k["sun_rgb"]
        # --- sun: disc + corona + wide glow (hidden behind the planet until it rises)
        if sun_vis > 0:
            ang = torch.acos((d @ s).clamp(-1, 1)) / DEG
            disc = _smoothstep(SUN_RADIUS + 0.25, SUN_RADIUS - 0.25, ang)
            corona = torch.exp(-ang / 2.2) * 0.9 + torch.exp(-ang / 9.0) * 0.12
            rgb = rgb + ((disc * 40.0 + corona) * sun_vis)[:, None] * sunc
        # --- moon (a lit sphere at infinity)
        mc = k["moon"]
        km = math.sin(MOON_RADIUS * DEG)
        b = d @ mc
        disc_m = b * b - (1 - km * km)
        im = torch.nonzero((disc_m > 0) & (b > 0)).squeeze(1)
        if im.numel():
            dm = d[im]
            tm = b[im] - torch.sqrt(disc_m[im])
            nm = (tm[:, None] * dm - mc) / km
            alb = 0.55 + 0.5 * self._vol_sample(nm * 1.2)[:, 3]
            rgb[im] = ((nm @ s).clamp_min(0) * 1.6 * alb)[:, None] * self._c([0.9, 0.88, 0.85]) + 0.002
        # --- planet: a sphere at unit distance whose angular radius is PLANET_ANG
        kp = math.sin(PLANET_ANG * DEG)
        bp = d @ pc
        limb = torch.acos(bp.clamp(-1, 1)) / DEG                  # angle from the planet centre
        disc_p = bp * bp - (1 - kp * kp)
        hit_p = (disc_p > 0) & (bp > 0)
        # atmosphere: a blue rim on the day side, orange where the terminator crosses the limb (the sunrise)
        near = limb < PLANET_ANG + 12
        ia = torch.nonzero(near).squeeze(1)
        if ia.numel():
            da, ba, la = d[ia], bp[ia], limb[ia]
            dl = da - pc * ba[:, None]
            dl = dl / dl.norm(dim=-1, keepdim=True).clamp_min(1e-6)   # the limb point's normal below this direction
            lam_l = dl @ s
            glow_day = _smoothstep(-0.25, 0.3, lam_l)
            to_sun = torch.acos((dl @ s).clamp(-1, 1))
            sunset = torch.exp(-(lam_l / 0.2) ** 2) + 1.5 * torch.exp(-(to_sun / (25 * DEG)) ** 2) * sun_vis
            atm = (self._c([0.25, 0.55, 1.2]) * glow_day[:, None] * 0.9
                   + self._c([1.4, 0.55, 0.18]) * sunset[:, None] * 1.1)
            rim_out = torch.exp(-(la - PLANET_ANG).clamp_min(0) / 1.3) * (la >= PLANET_ANG)
            rgb[ia] = rgb[ia] + atm * (0.5 * rim_out)[:, None]
        ip = torch.nonzero(hit_p).squeeze(1)
        if ip.numel():
            dp = d[ip]
            tp = bp[ip] - torch.sqrt(disc_p[ip])
            n = (tp[:, None] * dp - pc) / kp
            n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            grid = (n[:, :2] / self.planet_L).view(1, 1, -1, 2)
            tex = torch.nn.functional.grid_sample(self.planet_tex, grid, mode="bilinear", padding_mode="border",
                                                  align_corners=True).view(4, -1).t()
            alb, city = tex[:, :3], tex[:, 3]
            lam = n @ s
            day = _smoothstep(-0.06, 0.2, lam)
            lit = alb * (lam.clamp_min(0) * 2.0 + 0.02)[:, None] * self._c([1.0, 0.97, 0.92])
            planet = lit * day[:, None] + alb * 0.005 + (city * (1 - day) * 0.10)[:, None] * self._c([1.0, 0.62, 0.28])
            la = limb[ip]
            rim_in = torch.exp(-(PLANET_ANG - la).clamp_min(0) / 2.2)
            dl = dp - pc * bp[ip][:, None]
            dl = dl / dl.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            lam_l = dl @ s
            to_sun = torch.acos(lam_l.clamp(-1, 1))
            atm = (self._c([0.25, 0.55, 1.2]) * _smoothstep(-0.25, 0.3, lam_l)[:, None] * 0.9
                   + self._c([1.4, 0.55, 0.18])
                   * (torch.exp(-(lam_l / 0.2) ** 2) + 1.5 * torch.exp(-(to_sun / (25 * DEG)) ** 2) * sun_vis)[:, None] * 1.1)
            rgb[ip] = planet * (1 - 0.5 * rim_in)[:, None] + atm * (0.5 * rim_in)[:, None]
        return rgb, hit_p

    # ------------------------------------------------------------------------------------ local objects
    # Each object function takes a batch of rays (o (3,) or (M, 3), d (M, 3)) and returns (t, rgb) with t = inf where
    # it is missed. No host syncs: the camera restricts each object to its screen-space bounding box instead.
    @staticmethod
    def _ellipsoid_hit(o, d, center, radii, rot):
        """Ray (o, d) against an ellipsoid (rot: its local -> parent axes); t = inf where missed."""
        oc = (o - center) @ rot
        dl = d @ rot
        oq, dq = oc / radii, dl / radii
        a = (dq * dq).sum(-1)
        b = (oq * dq).sum(-1)
        c = (oq * oq).sum(-1) - 1
        disc = b * b - a * c
        sq = torch.sqrt(disc.clamp_min(0))
        t0 = (-b - sq) / a
        t1 = (-b + sq) / a
        t = torch.where(t0 > 1e-3, t0, t1)
        return torch.where((disc > 0) & (t > 1e-3), t, torch.full_like(t, float("inf")))

    def _noise_volume(self, n=64):
        """A tileable 4-channel fBm volume (bump xyz, albedo), so rock and moon surfaces cost one texture fetch."""
        if getattr(self, "_vol", None) is None:
            dev = self.device
            g = (torch.arange(n, device=dev) + 0.5) / n
            z, y, x = torch.meshgrid(g, g, g, indexing="ij")
            p = torch.stack([x, y, z], -1).reshape(-1, 3)
            chans = []
            for k in range(4):
                acc = torch.zeros(p.shape[0], device=dev)
                amp, tot = 1.0, 0.0
                for o_ in range(4):
                    per = 4 * 2 ** o_                              # lattice cells per tile: tileable noise
                    acc = acc + amp * self._tile_noise(p * per + 31.0 * k, per)
                    tot += amp
                    amp *= 0.5
                chans.append(acc / tot)
            self._vol = torch.stack(chans, 0).reshape(1, 4, n, n, n).contiguous()
        return self._vol

    def _tile_noise(self, q, per):
        i0 = torch.floor(q).long()
        f = q - i0.float()
        f = f * f * (3 - 2 * f)
        out = torch.zeros(q.shape[0], device=q.device)
        for k in range(8):
            dx, dy, dz = (k >> 2) & 1, (k >> 1) & 1, k & 1
            w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
            out = out + w * _hash3(torch.remainder(i0 + self.corners[k], per))
        return out

    def _vol_sample(self, q):
        """q (M, 3) in tile units (1 = one tile) -> (M, 4) fBm."""
        v = self._noise_volume()
        g = (torch.remainder(q, 1.0) * 2 - 1).view(1, 1, 1, -1, 3)
        out = torch.nn.functional.grid_sample(v, g, mode="bilinear", padding_mode="border", align_corners=False)
        return out.view(4, -1).t()

    def _c(self, values):
        """A cached device tensor for a constant list."""
        key = tuple(values)
        cache = self.__dict__.setdefault("_cc", {})
        if key not in cache:
            cache[key] = torch.tensor(values, dtype=torch.float32, device=self.device)
        return cache[key]

    def _consts(self):
        """Device constants, made once (a host -> device copy per object per frame is what made this slow on a
        shared GPU)."""
        if getattr(self, "_k", None) is None:
            dev = self.device
            vis = self.sun_visibility()
            parts = self.parts
            kinds = {"hull": 0, "panel": 1, "glass": 2, "rock": 3, "eye": 4, "dark": 5, "emit": 6}
            self._k = {
                "sun": torch.tensor(self.sun_dir, dtype=torch.float32, device=dev),
                "sunc": torch.tensor(SUN_RGB * CAM_SUN * vis, dtype=torch.float32, device=dev),
                "ps": torch.tensor(PLANETSHINE_RGB, dtype=torch.float32, device=dev),
                "pc": torch.tensor(PLANET_DIR, dtype=torch.float32, device=dev),
                "moon": torch.tensor(MOON_DIR, dtype=torch.float32, device=dev),
                "sun_rgb": torch.tensor(SUN_RGB, dtype=torch.float32, device=dev),
                "eye3": torch.eye(3, device=dev),
                "spokes": torch.tensor([[math.cos(k * math.pi / 2), math.sin(k * math.pi / 2), 0.0] for k in range(4)],
                                       dtype=torch.float32, device=dev),
                "p_c": torch.tensor(np.stack([p.center for p in parts]), dtype=torch.float32, device=dev),
                "p_r": torch.tensor(np.stack([p.radii for p in parts]), dtype=torch.float32, device=dev),
                "p_rot": torch.tensor(np.stack([p.rot for p in parts]), dtype=torch.float32, device=dev),
                "p_alb": torch.tensor(np.stack([p.albedo for p in parts]), dtype=torch.float32, device=dev),
                "p_kind": torch.tensor([kinds[p.kind] for p in parts], device=dev),
                "p_spec": torch.tensor([{"hull": 0.35, "panel": 0.2, "eye": 0.6}.get(p.kind, 0.05) for p in parts],
                                       dtype=torch.float32, device=dev),
                "p_main": torch.tensor([p.kind == "emit" and p.center[1] == 0 for p in parts], device=dev),
                "p_pods": torch.tensor([p.kind == "emit" and p.center[1] != 0 for p in parts], device=dev),
                "glow_c": torch.tensor([0.35, 0.6, 1.0], dtype=torch.float32, device=dev),
                "can_c": torch.tensor(CANOPY[0], dtype=torch.float32, device=dev),
                "can_r": torch.tensor([CANOPY[1], CANOPY[1] * 0.8, CANOPY[1] * 0.75], dtype=torch.float32, device=dev),
                "ring_c": torch.tensor([0.82, 0.83, 0.85], dtype=torch.float32, device=dev),
                "hub_c": torch.tensor([0.7, 0.72, 0.74], dtype=torch.float32, device=dev),
                "win_c": torch.tensor([1.1, 0.8, 0.45], dtype=torch.float32, device=dev) * 0.9,
                "tint": torch.tensor([0.01, 0.02, 0.03], dtype=torch.float32, device=dev),
                "planet_avg": torch.tensor([0.10, 0.13, 0.18], dtype=torch.float32, device=dev),
            }
        return self._k

    def _shade(self, n, alb, view=None, spec=0.0):
        k = self._consts()
        lam = (n @ k["sun"]).clamp_min(0)
        pl = (n @ k["pc"]).clamp_min(0) * 0.8 + 0.2
        col = alb * (lam[:, None] * k["sunc"] + pl[:, None] * k["ps"] * 0.55 + 0.012)
        if view is not None and (not isinstance(spec, float) or spec > 0):
            h = k["sun"] - view
            h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            sp = spec if isinstance(spec, float) else spec[:, None]
            col = col + sp * ((n * h).sum(-1).clamp_min(0) ** 60)[:, None] * k["sunc"]
        return col

    ROW = 16 + 6 * N_LOBES                     # floats per packed object row (a rock's; station and ship rows pad)

    @staticmethod
    def pack_rock(a):
        """[center 3, rot 9, albedo 3, seed 1, lobe centres N_LOBES x 3, lobe radii N_LOBES x 3] (lobes in the rock's
        own frame) -> ROW floats."""
        return np.concatenate([np.asarray(a["pos"], float), np.asarray(a["rot"], float).ravel(),
                               np.asarray(a["albedo"], float), [float(a["seed"])],
                               np.asarray(a["lobe_c"], float).ravel(), np.asarray(a["lobe_r"], float).ravel()])

    def _rock(self, o, d, g):
        """g: a (ROW,) device row from pack_rock. The rock is the union of its lobes (ellipsoids sharing its rotation),
        which gives a lumpy silhouette; a noise volume adds bump and a little albedo variation."""
        c, rot, albedo, seed = g[0:3], g[3:12].view(3, 3), g[12:15], g[15]
        lc = g[16:16 + 3 * N_LOBES].view(N_LOBES, 3)
        lr = g[16 + 3 * N_LOBES:16 + 6 * N_LOBES].view(N_LOBES, 3)
        ol = (o - c) @ rot                                          # ray origin(s) in the rock frame
        dl = d @ rot                                                # (M, 3)
        oq = (ol.unsqueeze(-2) - lc) / lr                           # (L, 3) or (M, L, 3)
        dq = dl[:, None, :] / lr[None]                              # (M, L, 3)
        a = (dq * dq).sum(-1)
        b = (dq * oq).sum(-1)
        cc = (oq * oq).sum(-1) - 1
        disc = b * b - a * cc
        sq = torch.sqrt(disc.clamp_min(0))
        t0 = (-b - sq) / a
        t1 = (-b + sq) / a
        t = torch.where(t0 > 1e-3, t0, t1)
        t = torch.where((disc > 0) & (t > 1e-3), t, torch.full_like(t, float("inf")))
        t, li = t.min(1)                                            # nearest lobe per ray
        p = ol + dl * t.clamp_max(1e6)[:, None]                     # rock-frame hit point
        nl = (p - lc[li]) / lr[li] ** 2
        nl = nl / nl.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        q = p / lr[0].max() * 0.45 + seed * 0.137
        nz = self._vol_sample(q)
        nz2 = self._vol_sample(q * 2.3 + 0.5)
        nl = nl + 1.1 * nz[:, :3] + 0.55 * nz2[:, :3]
        n = nl @ rot.t()
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        mott = (0.85 + 0.45 * nz[:, 3] + 0.25 * nz2[:, 3]).clamp(0.5, 1.2)
        return t, self._shade(n, albedo * mott[:, None])

    STATION_STEPS = 36

    def _station_sdf(self, p, R, r, spokes):
        rho = torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
        torus = torch.sqrt((rho - R) ** 2 + p[:, 2] ** 2) - r
        hub = torch.sqrt(rho ** 2 + (p[:, 2] / 1.6) ** 2) - 7.0
        axle = torch.maximum(rho - 2.2, p[:, 2].abs() - 16.0)
        h = (p @ spokes.t()).clamp(6.0, R - r * 0.5)                           # (M, 4)
        sp = (p[:, None, :] - h[..., None] * spokes[None]).norm(dim=-1).amin(1) - 1.1
        return torch.minimum(torch.minimum(torus, hub), torch.minimum(axle, sp))

    def _station(self, o, d, g, R, r):
        """g: (12,) device row [pos 3, rot 9 (local -> world, spin included)]."""
        k = self._consts()
        c, rot = g[0:3], g[3:12].view(3, 3)
        bound = R + r + 2.0
        oc = c - o
        tc = (oc * d).sum(-1)
        miss2 = (oc * oc).sum(-1) - tc * tc
        half = torch.sqrt((bound * bound - miss2).clamp_min(0))
        t = (tc - half).clamp_min(0.01)
        t_end = tc + half
        ol = (-oc) @ rot
        dl = d @ rot
        spokes = k["spokes"]
        hit = torch.zeros_like(t, dtype=torch.bool)
        for _ in range(self.STATION_STEPS):
            dist = self._station_sdf(ol + dl * t[:, None], R, r, spokes)
            hit = hit | (dist < 0.03 * (1 + t * 0.003))
            t = torch.where(hit, t, t + dist.clamp_min(0.05))
        hit = hit & (t < t_end) & (miss2 < bound * bound)
        p = ol + dl * t[:, None]
        E = k["eye3"] * 0.05
        nl = torch.stack([self._station_sdf(p + E[i], R, r, spokes) - self._station_sdf(p - E[i], R, r, spokes)
                          for i in range(3)], -1)
        nl = nl / nl.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        n = nl @ rot.t()
        theta = torch.atan2(p[:, 1], p[:, 0])
        rho = torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
        on_ring = (rho - R).abs() < r + 0.3
        seg = torch.floor(theta / (2 * math.pi) * 48) % 2
        alb = torch.where(on_ring[:, None], k["ring_c"] * (0.88 + 0.12 * seg)[:, None], k["hub_c"])
        col = self._shade(n, alb, d, 0.25)
        psi = torch.atan2(p[:, 2], rho - R)
        win = on_ring & (psi.abs() < 0.35) & (torch.remainder(theta * 480 / (2 * math.pi), 1.0) < 0.45) & (seg > 0.5)
        col = col + win[:, None].float() * k["win_c"]
        return torch.where(hit, t, torch.full_like(t, float("inf"))), col

    def _ship(self, o, d, g, glow):
        """g: (12,) device row [pos 3, rot 9 (body -> world)]; all 18 hull / pilot parts at once, then the glass."""
        k = self._consts()
        c, Rb = g[0:3], g[3:12].view(3, 3)
        ob = (o - c) @ Rb                                           # (3,)
        db = d @ Rb                                                 # (M, 3)
        pc, pr, prot = k["p_c"], k["p_r"], k["p_rot"]               # (P, 3), (P, 3), (P, 3, 3)
        oq = torch.einsum("pi,pij->pj", ob[None] - pc, prot) / pr    # (P, 3)
        dq = torch.einsum("mi,pij->mpj", db, prot) / pr[None]        # (M, P, 3)
        a = (dq * dq).sum(-1)
        b = (dq * oq[None]).sum(-1)
        cc = (oq * oq).sum(-1) - 1
        disc = b * b - a * cc[None]
        sq = torch.sqrt(disc.clamp_min(0))
        t0 = (-b - sq) / a
        t1 = (-b + sq) / a
        t = torch.where(t0 > 1e-3, t0, t1)
        t = torch.where((disc > 0) & (t > 1e-3), t, torch.full_like(t, float("inf")))
        tb, pi = t.min(1)                                           # nearest part per ray
        hit = torch.isfinite(tb)
        p = ob + db * tb.clamp_max(1e6)[:, None]                    # body-frame hit point
        rot_i, c_i, r_i = prot[pi], pc[pi], pr[pi]
        nl = torch.einsum("mi,mij->mj", p - c_i, rot_i) / (r_i * r_i)
        n = torch.einsum("mj,mij->mi", nl, rot_i)
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        kind = k["p_kind"][pi]
        alb = k["p_alb"][pi]
        lines = (torch.remainder(p[:, 0] * 1.4, 1.0) < 0.04) | (torch.remainder(p[:, 1] * 1.4 + 0.5, 1.0) < 0.03)
        plated = (kind <= 1).float()
        cell = torch.stack([torch.floor(p[:, 0] * 1.4), torch.floor(p[:, 1] * 1.4 + 0.5), torch.floor(p[:, 2] * 1.4)],
                           -1).long()
        panel = 1 + 0.25 * _hash3(cell + pi[:, None] * 131)         # each hull plate a slightly different grey
        alb = alb * ((1 - 0.5 * (lines.float() * plated)) * (1 + (panel - 1) * plated))[:, None]
        emit = ((k["p_main"][pi].float() * (0.02 + 6.0 * glow.get("main", 0.0))
                 + k["p_pods"][pi].float() * (0.02 + 6.0 * glow.get("pods", 0.0)))[:, None] * k["glow_c"])
        n_w = n @ Rb.t()
        col = self._shade(n_w, alb, d, k["p_spec"][pi]) + emit
        # canopy glass: tint what is behind it and reflect the sky (texture + the planet's mean colour)
        tg = self._ellipsoid_hit(ob, db, k["can_c"], k["can_r"], k["eye3"])
        glass = torch.isfinite(tg) & (tg < tb)
        pg_ = ob + db * tg.clamp_max(1e6)[:, None]
        ng = (pg_ - k["can_c"]) / k["can_r"] ** 2
        ng = ng / ng.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        ngw = ng @ Rb.t()
        refl = d - 2 * (d * ngw).sum(-1, keepdim=True) * ngw
        refl = refl / refl.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        env = self._sample_sky(refl, self.sky_plain)
        below = (refl @ k["pc"]) > math.cos(PLANET_ANG * DEG)
        env = torch.where(below[:, None], k["planet_avg"], env)
        cosi = (-(d * ngw).sum(-1)).clamp(0, 1)
        fres = 0.08 + 0.92 * (1 - cosi) ** 5
        behind = torch.where(hit[:, None], col, torch.zeros_like(col))
        col_g = behind * 0.8 * (1 - fres)[:, None] + env * (0.25 + fres)[:, None] + k["tint"]
        col = torch.where(glass[:, None], col_g, col)
        tb = torch.where(glass, tg, tb)
        return tb, col

    def _pack_frame(self, asteroids, station, ship):
        """One host -> device copy for everything that moves this frame."""
        rows = [self.pack_rock(a) for a in asteroids]
        n_r = len(rows)
        if station is not None:
            rows.append(np.concatenate([np.asarray(station["pos"], float), np.asarray(station["rot"], float).ravel(),
                                        np.zeros(self.ROW - 12)]))
        if ship is not None:
            rows.append(np.concatenate([np.asarray(ship["pos"], float), np.asarray(ship["rot"], float).ravel(),
                                        np.zeros(self.ROW - 12)]))
        if not rows:
            return None, 0
        return torch.tensor(np.stack(rows), dtype=torch.float32, device=self.device), n_r

    # ------------------------------------------------------------------------------------ views
    @torch.no_grad()
    def trace(self, o, d, t_s, *, asteroids=(), station=None, ship=None, eye=False):
        """Linear RGB along a flat batch of rays (no culling): local objects over everything at infinity."""
        rgb, occ = self.infinity(d, eye=eye)
        best = torch.full((d.shape[0],), float("inf"), device=self.device)
        G, n_r = self._pack_frame(asteroids, station, ship)
        objs = [(lambda o_, d_, i=i: self._rock(o_, d_, G[i])) for i in range(n_r)]
        if station is not None:
            objs.append(lambda o_, d_: self._station(o_, d_, G[n_r], station["R"], station["r"]))
        if ship is not None:
            objs.append(lambda o_, d_: self._ship(o_, d_, G[-1], ship.get("glow", {})))
        for fn in objs:
            t, col = fn(o, d)
            closer = t < best
            best = torch.where(closer, t, best)
            rgb = torch.where(closer[:, None], col, rgb)
        return rgb, occ | torch.isfinite(best)

    @torch.no_grad()
    def camera(self, pos, forward, up, width, height, fov_deg, t_s, *, asteroids=(), station=None, ship=None,
               stars=True, exposure=1.6):
        """A pinhole render -> (height, width, 3) uint8 sRGB (filmic tone map, crisp stars, sun flare). Each local
        object is traced only inside its screen-space bounding box."""
        dev = self.device
        f, left, u = look_basis(forward, up)
        right = -left
        tan = math.tan(fov_deg * DEG / 2)
        asp = width / height
        basis = torch.tensor(np.stack([f, right, u, np.asarray(pos, float)]), dtype=torch.float32, device=dev)
        F, Rt, U, o = basis[0], basis[1], basis[2], basis[3]
        xs = torch.linspace(-tan, tan, width, device=dev)
        ys = torch.linspace(tan / asp, -tan / asp, height, device=dev)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        d = F[None, None] + gx[..., None] * Rt + gy[..., None] * U
        d = d / d.norm(dim=-1, keepdim=True)
        rgb, occluded = self.infinity(d.reshape(-1, 3))
        rgb = rgb.reshape(height, width, 3)
        occluded = occluded.reshape(height, width)
        best = torch.full((height, width), float("inf"), device=dev)
        pos = np.asarray(pos, float)

        def bbox(center, radius):
            v = np.asarray(center, float) - pos
            z = float(np.dot(v, f))
            dist = float(np.linalg.norm(v))
            if dist <= radius * 1.05 or (-radius < z <= radius):
                return 0, width, 0, height
            if z <= -radius:
                return None
            ang = math.asin(min(1.0, radius / dist))
            cx = float(np.dot(v, right)) / z / tan
            cy = float(np.dot(v, u)) / z / (tan / asp)
            rx = math.tan(ang) / tan * (1 + abs(cx)) * 1.15 + 0.01
            ry = math.tan(ang) / (tan / asp) * (1 + abs(cy)) * 1.15 + 0.01
            x0 = int(math.floor(((cx - rx) * 0.5 + 0.5) * (width - 1)))
            x1 = int(math.ceil(((cx + rx) * 0.5 + 0.5) * (width - 1))) + 1
            y0 = int(math.floor((0.5 - (cy + ry) * 0.5) * (height - 1)))
            y1 = int(math.ceil((0.5 - (cy - ry) * 0.5) * (height - 1))) + 1
            x0, x1, y0, y1 = max(0, x0), min(width, x1), max(0, y0), min(height, y1)
            if x1 <= x0 or y1 <= y0:
                return None
            return x0, x1, y0, y1

        G, n_r = self._pack_frame(asteroids, station, ship)
        objs = [((lambda o_, d_, i=i: self._rock(o_, d_, G[i])), a["pos"], float(a["bound"]))
                for i, a in enumerate(asteroids)]
        if station is not None:
            objs.append(((lambda o_, d_: self._station(o_, d_, G[n_r], station["R"], station["r"])), station["pos"],
                         station["R"] + station["r"] + 2.0))
        if ship is not None:
            objs.append(((lambda o_, d_: self._ship(o_, d_, G[-1], ship.get("glow", {}))), ship["pos"], 5.2))
        for fn, center, radius in objs:
            box = bbox(center, radius)
            if box is None:
                continue
            x0, x1, y0, y1 = box
            t, col = fn(o, d[y0:y1, x0:x1].reshape(-1, 3))
            t = t.view(y1 - y0, x1 - x0)
            closer = t < best[y0:y1, x0:x1]
            best[y0:y1, x0:x1] = torch.where(closer, t, best[y0:y1, x0:x1])
            rgb[y0:y1, x0:x1] = torch.where(closer[..., None], col.view(y1 - y0, x1 - x0, 3), rgb[y0:y1, x0:x1])
        occluded = occluded | torch.isfinite(best)
        img = rgb
        if stars:
            sd = self.star_dir
            z = sd @ F
            px = ((sd @ Rt) / z.clamp_min(1e-3) / tan * 0.5 + 0.5) * (width - 1)
            py = (0.5 - (sd @ U) / z.clamp_min(1e-3) / (tan / asp) * 0.5) * (height - 1)
            ok = (z > 0.05) & (px >= 0) & (px < width - 1) & (py >= 0) & (py < height - 1)
            c = self.star_rgb * ok[:, None].float()
            px = torch.where(ok, px, torch.zeros_like(px))
            py = torch.where(ok, py, torch.zeros_like(py))
            ix, iy = px.floor().long(), py.floor().long()
            fx, fy = (px - ix)[:, None], (py - iy)[:, None]
            acc = torch.zeros(height * width, 3, device=dev)
            for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
                acc.index_add_(0, (iy + dy) * width + (ix + dx), c * w)
            img = img + acc.reshape(height, width, 3) * (~occluded)[..., None].float()
        # sun flare: an anamorphic streak and a soft bloom when the sun is in view and up
        s = self.sun_dir
        zs = float(np.dot(s, f))
        vis = self.sun_visibility()
        if zs > 0.05 and vis > 0:
            sx = (float(np.dot(s, right)) / zs / tan * 0.5 + 0.5) * (width - 1)
            sy = (0.5 - float(np.dot(s, u)) / zs / (tan / asp) * 0.5) * (height - 1)
            if -0.3 * width < sx < 1.3 * width and -0.3 * height < sy < 1.3 * height:
                xx = torch.arange(width, device=dev)[None, :].float()
                yy = torch.arange(height, device=dev)[:, None].float()
                dx, dy = (xx - sx) / width, (yy - sy) / width
                streak = torch.exp(-(dy.abs() / 0.004)) * torch.exp(-(dx.abs() / 0.35)) * 0.35
                bloom = torch.exp(-torch.sqrt(dx * dx + dy * dy) / 0.05) * 0.5
                img = img + ((streak + bloom) * vis)[..., None] * torch.tensor([1.0, 0.85, 0.7], device=dev)
        x = img * exposure
        x = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)          # ACES-like filmic curve
        x = x.clamp(0, 1) ** (1 / 2.2)
        return (x * 255).to(torch.uint8).cpu().numpy()


# -------------------------------------------------------------------------------------------------------- the eye (CPU)
class EyeScene:
    """The same scene for the fly's 1,466 x 7 rays, traced on the CPU in numpy: the sky at infinity (planet, sun, moon,
    nebulae, blurred stars) is static, so it is rendered once into an equirectangular texture by SpaceRenderer and
    sampled bilinearly; rocks (unions of ellipsoids) and the station (signed distance field) are intersected per tick with plain
    Lambert shading (the eye's 4.5 deg acceptance averages away the camera's surface detail). The GPU stays free for
    the brain."""

    def __init__(self, renderer: SpaceRenderer, size=(2048, 1024)):
        W, H = size
        dev = renderer.device
        lon = (0.5 - (torch.arange(W, device=dev) + 0.5) / W) * 2 * math.pi
        lat = (0.5 - (torch.arange(H, device=dev) + 0.5) / H) * math.pi
        la, lo = torch.meshgrid(lat, lon, indexing="ij")
        d = torch.stack([torch.cos(la) * torch.cos(lo), torch.cos(la) * torch.sin(lo), torch.sin(la)], -1).reshape(-1, 3)
        chunks = [renderer.infinity(d[i:i + 262144], eye=True)[0] for i in range(0, d.shape[0], 262144)]
        self.sky = torch.cat(chunks).reshape(H, W, 3).cpu().numpy().astype(np.float32)
        self.W, self.H = W, H
        self.sun = np.asarray(renderer.sun_dir, float)
        self.sunc = SUN_RGB * EYE_SUN * renderer.sun_visibility()
        self.spokes = np.array([[math.cos(k * math.pi / 2), math.sin(k * math.pi / 2), 0.0] for k in range(4)])

    def sample_sky(self, d):
        lon = np.arctan2(d[:, 1], d[:, 0]); lat = np.arcsin(np.clip(d[:, 2], -1, 1))
        u = (0.5 - lon / (2 * np.pi)) * self.W - 0.5
        v = np.clip((0.5 - lat / np.pi) * self.H - 0.5, 0, self.H - 1.001)
        u0 = np.floor(u).astype(np.int64); v0 = np.floor(v).astype(np.int64)
        fu = (u - u0)[:, None]; fv = (v - v0)[:, None]
        u0 %= self.W; u1 = (u0 + 1) % self.W; v1 = np.minimum(v0 + 1, self.H - 1)
        s = self.sky
        return (s[v0, u0] * (1 - fu) * (1 - fv) + s[v0, u1] * fu * (1 - fv) + s[v1, u0] * (1 - fu) * fv + s[v1, u1] * fu * fv)

    def shade(self, n, alb):
        lam = np.clip(n @ self.sun, 0, None)
        pl = np.clip(n @ PLANET_DIR, 0, None) * 0.8 + 0.2
        return alb * (lam[:, None] * self.sunc + pl[:, None] * PLANETSHINE_RGB * 0.55 + 0.012)

    @staticmethod
    def ellipsoid(o, d, center, radii, rot):
        oc = (o - center) @ rot
        dl = d @ rot
        oq, dq = oc / radii, dl / radii
        a = (dq * dq).sum(-1); b = (dq * oq).sum(-1); c = (oq * oq).sum(-1) - 1
        disc = b * b - a * c
        sq = np.sqrt(np.clip(disc, 0, None))
        t0 = (-b - sq) / a; t1 = (-b + sq) / a
        t = np.where(t0 > 1e-3, t0, t1)
        return np.where((disc > 0) & (t > 1e-3), t, np.inf)

    def station_sdf(self, p, R, r):
        rho = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
        torus = np.sqrt((rho - R) ** 2 + p[:, 2] ** 2) - r
        hub = np.sqrt(rho ** 2 + (p[:, 2] / 1.6) ** 2) - 7.0
        axle = np.maximum(rho - 2.2, np.abs(p[:, 2]) - 16.0)
        h = np.clip(p @ self.spokes.T, 6.0, R - r * 0.5)
        sp = np.linalg.norm(p[:, None, :] - h[..., None] * self.spokes[None], axis=-1).min(1) - 1.1
        return np.minimum(np.minimum(torus, hub), np.minimum(axle, sp))

    def trace(self, o, d, rocks=(), station=None):
        """o (3,), d (M, 3) world -> linear RGB (M, 3)."""
        rgb = self.sample_sky(d)
        best = np.full(d.shape[0], np.inf)
        eye3 = np.eye(3)
        for a in rocks:
            c = np.asarray(a["pos"], float); rot = np.asarray(a["rot"], float)
            oc = c - o
            tc = d @ oc
            miss2 = oc @ oc - tc * tc
            cand = np.flatnonzero((miss2 < a["bound"] ** 2) & (tc > -a["bound"]))
            if not len(cand):
                continue
            ol, dl = (o - c) @ rot, d[cand] @ rot                    # rock frame
            t = np.full(len(cand), np.inf)
            nl = np.zeros((len(cand), 3))
            for lc, lr in zip(np.asarray(a["lobe_c"], float), np.asarray(a["lobe_r"], float)):
                tl = self.ellipsoid(ol, dl, lc, lr, eye3)
                better = tl < t
                if better.any():
                    t[better] = tl[better]
                    nl[better] = ((ol + dl[better] * tl[better, None]) - lc) / lr ** 2
            closer = t < best[cand]
            if not closer.any():
                continue
            idx = cand[closer]
            n = nl[closer] @ rot.T
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            best[idx] = t[closer]
            rgb[idx] = self.shade(n, np.asarray(a["albedo"], float)[None] * 0.85)   # the camera's mean mottling
        if station is not None:
            c = np.asarray(station["pos"], float); rot = np.asarray(station["rot"], float)
            R, r = station["R"], station["r"]
            bound = R + r + 2.0
            oc = c - o
            tc = d @ oc
            miss2 = oc @ oc - tc * tc
            cand = np.flatnonzero(miss2 < bound * bound)
            if len(cand):
                half = np.sqrt(bound * bound - miss2[cand])
                t = np.clip(tc[cand] - half, 0.01, None); t_end = tc[cand] + half
                ol = (o - c) @ rot; dl = d[cand] @ rot
                hit = np.zeros(len(cand), bool)
                for _ in range(40):
                    dist = self.station_sdf(ol + dl * t[:, None], R, r)
                    hit |= dist < 0.03 * (1 + t * 0.003)
                    t = np.where(hit, t, t + np.maximum(dist, 0.05))
                hit &= (t < t_end) & (t < best[cand])
                if hit.any():
                    idx, t = cand[hit], t[hit]
                    p = ol + dl[hit] * t[:, None]
                    E = np.eye(3) * 0.05
                    nl = np.stack([self.station_sdf(p + E[i], R, r) - self.station_sdf(p - E[i], R, r) for i in range(3)], -1)
                    nl /= np.maximum(np.linalg.norm(nl, axis=1, keepdims=True), 1e-9)
                    best[idx] = t
                    rgb[idx] = self.shade(nl @ rot.T, np.array([0.8, 0.81, 0.83]))
        return rgb


# ======================================================================================================== physics (GAME)
INERTIA = np.array([1.0, 1.7, 2.0])            # principal moments about body x (roll), y (pitch), z (yaw); normalised
ALPHA_MAX = np.array([90.0, 90.0, 90.0]) * DEG  # thruster authority: max angular acceleration per axis, rad/s^2
CRUISE_V = np.array([2.0, 0.0, 0.0])           # autopilot's world-frame cruise velocity, m/s
CRUISE_TAU, CRUISE_AMAX = 2.5, 1.5             # it relaxes the velocity toward CRUISE_V with this time constant (s), capped


@dataclass
class ShipState:
    pos: np.ndarray                            # world, m
    vel: np.ndarray                            # world, m/s
    q: np.ndarray                              # body -> world unit quaternion (w, x, y, z)
    w: np.ndarray                              # body-frame angular velocity, rad/s (x roll, y pitch, z yaw)

    @property
    def R(self):
        return quat_to_matrix(self.q)

    def copy(self):
        return ShipState(self.pos.copy(), self.vel.copy(), self.q.copy(), self.w.copy())


def rigid_step(s: ShipState, alpha_cmd, accel_world, dt, inertia=INERTIA):
    """Advance a torque-free-plus-thrusters rigid body by dt: Euler's equations in the body frame (torque = I alpha_cmd,
    RK4), quaternion integration, and a world-frame linear acceleration. Returns a new state."""
    I = np.asarray(inertia, float)
    tau = I * np.asarray(alpha_cmd, float)

    def wdot(w):
        return (tau - np.cross(w, I * w)) / I
    w0 = s.w
    k1 = wdot(w0)
    k2 = wdot(w0 + 0.5 * dt * k1)
    k3 = wdot(w0 + 0.5 * dt * k2)
    k4 = wdot(w0 + dt * k3)
    w = w0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)                  # RK4 on Euler's equations (torque held over dt)
    wm = 0.5 * (w0 + w)                                           # rotate by the mid-step rate
    ang = float(np.linalg.norm(wm)) * dt
    q = quat_mul(s.q, quat_from_axis_angle(wm, ang)) if ang > 0 else s.q.copy()
    q = q / np.linalg.norm(q)
    vel = s.vel + np.asarray(accel_world, float) * dt
    pos = s.pos + vel * dt
    return ShipState(pos, vel, q, w)


def angular_momentum_world(s: ShipState, inertia=INERTIA):
    return s.R @ (np.asarray(inertia) * s.w)


def cruise_autopilot(vel, target=CRUISE_V, tau=CRUISE_TAU, amax=CRUISE_AMAX):
    """GAME: a translation-only autopilot that relaxes the ship's world velocity toward `target`; it never touches
    attitude."""
    a = (np.asarray(target) - np.asarray(vel)) / tau
    n = np.linalg.norm(a)
    return a if n <= amax else a * (amax / n)


# ---------------------------------------------------------------------------------------------------- asteroids
@dataclass
class Rock:
    name: str
    pos: np.ndarray                            # world, m
    vel: np.ndarray                            # world, m/s
    lobe_c: np.ndarray                         # (N_LOBES, 3) lobe centres in the rock's own frame, m (lobe 0: the core)
    lobe_r: np.ndarray                         # (N_LOBES, 3) lobe radii, m
    spin_axis: np.ndarray
    spin: float                                # rad/s
    albedo: tuple
    seed: float                                # texture offset
    threat: bool = False
    t_arrive: float = 0.0
    gap: float = float("inf")                  # current surface clearance to the ship's collision shape, m
    closest: float = float("inf")              # the smallest gap so far, m (negative: it touched)
    t_closest: float = 0.0
    resolved: str = ""                         # '' | 'miss' | 'impact'
    part: str = ""                             # the ship part it hit
    angle: float = 0.0
    approach: dict = field(default_factory=dict)   # peak loom / GF rates while it was within APPROACH_M (Hz)

    @property
    def bound(self):
        """Radius of a sphere about `pos` that holds every lobe, m."""
        return float(np.max(np.linalg.norm(self.lobe_c, axis=1) + self.lobe_r.max(1)))

    def rot(self):
        return rot_axis(self.spin_axis, self.angle)

    def render_spec(self):
        return {"pos": self.pos, "rot": self.rot(), "albedo": self.albedo, "seed": self.seed,
                "lobe_c": self.lobe_c, "lobe_r": self.lobe_r, "bound": self.bound}


def rock_shape(rng, radius):
    """GAME: a lumpy rock of size `radius` m: a core ellipsoid radius x (1, 0.75-0.9, 0.65-0.8) and N_LOBES - 1
    smaller lobes pushed out from it (0.45-0.6 of the core's radii; each 0.5-0.7 of the core's size). Returns
    (lobe_c, lobe_r) in the rock's frame."""
    core = radius * np.array([1.0, rng.uniform(0.75, 0.9), rng.uniform(0.65, 0.8)])
    cs, rs = [np.zeros(3)], [core * 0.9]
    for _ in range(N_LOBES - 1):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        cs.append(u * core * rng.uniform(0.45, 0.6))
        rs.append(core * rng.uniform(0.5, 0.7, 3))
    return np.array(cs), np.array(rs)


# The chase camera's path (GAME): it looks along azimuth CAM_AZ0 + CAM_AZ_RATE * t, CAM_EL deg below level, from
# CAM_DIST m behind the ship. The scenario sends each rock in from beyond the ship close to that line of sight, so the
# approach and the contact happen in the open middle of the frame, not under the HUD columns.
CAM_AZ0, CAM_AZ_RATE, CAM_EL, CAM_DIST, CAM_FOV = 200.0, 5.0, -10.0, 25.0, 60.0


def cam_look(t_s):
    """The chase camera's look direction at brain time t_s (world, unit)."""
    return unit_from_az_el(CAM_AZ0 + CAM_AZ_RATE * t_s, CAM_EL)


# The threat schedule (GAME, the same for every seed): arrival time (brain s), speed (m/s), size (m), and whether the
# rock is aimed to hit the ship ('hit': its centre 0.8-2.2 m off the ship's) or to pass close by ('close': 5.5-7.5 m
# off; the wings can still catch it). The seed draws each rock's direction, exact offset and shape (make_scenario).
THREAT_SLOTS = [(9.0, 24.0, 2.6, "hit"), (14.0, 26.0, 2.8, "close"), (19.0, 24.0, 3.0, "hit"),
                (24.0, 22.0, 2.6, "close"), (28.5, 28.0, 2.8, "hit")]
LAUNCH_LEAD = 5.0                              # seconds before arrival that a threat is spawned
W0_BASE = np.array([40.0, 18.0, -45.0])        # the initial tumble's central axis, body (roll, pitch, yaw) deg/s
SCENARIO_KEY = 0x5CAF                          # the scenario RNG is default_rng([SCENARIO_KEY, seed]): not the brain's


@dataclass
class Threat:
    t_arrive: float
    az: float                                  # direction from the ship toward the rock's start, world, deg
    el: float
    speed: float                               # m/s
    offset: float                              # the rock's centre passes this far from the ship's predicted centre, m
    lobe_c: np.ndarray
    lobe_r: np.ndarray
    spin_axis: np.ndarray
    spin: float
    tex_seed: float

    def describe(self):
        return {"t_arrive": self.t_arrive, "az_deg": round(self.az, 1), "el_deg": round(self.el, 1),
                "speed_m_s": self.speed, "aim_offset_m": round(self.offset, 2),
                "core_radius_m": round(float(np.max(self.lobe_r[0])), 2)}


def draw_threat(rng, t_arrive, speed, radius, kind, side):
    """One threat: it comes in from beyond the ship, near the chase camera's line of sight at arrival -- from its
    right (`side` -1: 12-26 deg off it) or almost straight behind the ship (`side` +1: 4 deg right to 6 deg left) --
    and 8 deg below to 4 deg above it. The camera pans left at 5 deg/s, so either way the rock stays in the open
    middle of the frame for most of its approach."""
    az = CAM_AZ0 + CAM_AZ_RATE * t_arrive + (rng.uniform(-4.0, 6.0) if side > 0 else -rng.uniform(12.0, 26.0))
    el = CAM_EL + rng.uniform(-8.0, 4.0)
    offset = rng.uniform(0.8, 2.2) if kind == "hit" else rng.uniform(5.5, 7.5)
    lobe_c, lobe_r = rock_shape(rng, radius)
    axis = rng.normal(size=3)
    return Threat(t_arrive, float(az), float(el), float(speed), float(offset), lobe_c, lobe_r,
                  axis / np.linalg.norm(axis), float(rng.uniform(0.2, 0.6)), float(rng.uniform(0, 50)))


@dataclass
class Scenario:
    seed: int
    still: bool
    w0_deg: np.ndarray                         # initial body angular velocity (roll, pitch, yaw), deg/s
    threats: list

    def describe(self):
        return {"seed": self.seed, "still": self.still, "w0_deg_s": [round(float(v), 2) for v in self.w0_deg],
                "w0_speed_deg_s": round(float(np.linalg.norm(self.w0_deg)), 2),
                "threats": [t.describe() for t in self.threats]}


def make_scenario(seed, still=False):
    """GAME: the run's scenario, drawn from the seed on its own RNG stream (the brain's RNG is separate and, with no
    Poisson input, unused): the initial tumble -- 55-70 deg/s about an axis scattered around W0_BASE's (a normal of
    sd 0.35 added to its unit vector) -- and each threat's direction, aim offset and shape. `still`: the ship starts
    at rest and no rocks come (a check of what the decoders do to a still ship)."""
    rng = np.random.default_rng([SCENARIO_KEY, int(seed)])
    axis = W0_BASE / np.linalg.norm(W0_BASE) + rng.normal(0.0, 0.35, 3)
    w0 = axis / np.linalg.norm(axis) * rng.uniform(55.0, 70.0)
    side0 = 1 if rng.uniform() < 0.5 else -1
    threats = [draw_threat(rng, t, v, r, kind, side0 * (-1) ** k) for k, (t, v, r, kind) in enumerate(THREAT_SLOTS)]
    if still:
        return Scenario(int(seed), True, np.zeros(3), [])
    return Scenario(int(seed), False, w0, threats)


def predict_position(pos, vel, lead, dt=0.01):
    """Where the cruise autopilot alone would take the ship in `lead` s (the same Euler steps as rigid_step)."""
    p, v = np.array(pos, float), np.array(vel, float)
    for _ in range(int(round(lead / dt))):
        v = v + cruise_autopilot(v) * dt
        p = p + v * dt
    return p


def make_threat(name, th: Threat, ship_pos, ship_vel, t_now):
    """Launch a threat so that, if nothing but the autopilot moved the ship, the rock's centre would pass `th.offset` m
    from the ship's centre at `th.t_arrive`; the offset is sideways, on the side away from the chase camera."""
    lead = th.t_arrive - t_now
    u = unit_from_az_el(th.az, th.el)
    target = predict_position(ship_pos, ship_vel, lead)
    side = np.cross(u, (0.0, 0.0, 1.0))
    side /= max(np.linalg.norm(side), 1e-9)
    if np.dot(side, -cam_look(th.t_arrive)) > 0:
        side = -side
    start = target + side * th.offset + u * th.speed * lead
    rock = Rock(name, start, -u * th.speed, th.lobe_c, th.lobe_r, th.spin_axis, th.spin, ROCK_ALBEDO, th.tex_seed,
                threat=True, t_arrive=th.t_arrive)
    rock.gap = float(np.linalg.norm(start - ship_pos)) - SHIP_BOUND - rock.bound
    return rock


def scenery_rocks(rng):
    """Big slow rocks far away (never threats): depth for the camera, a little texture for the eye."""
    out = []
    for k in range(7):
        u = unit_from_az_el(rng.uniform(-180, 180), rng.uniform(-10, 40))
        pos = u * rng.uniform(140, 320)
        vel = rng.normal(size=3) * 0.6
        lobe_c, lobe_r = rock_shape(rng, rng.uniform(5, 14))
        axis = rng.normal(size=3)
        out.append(Rock(f"scenery {k}", pos, vel, lobe_c, lobe_r, axis / np.linalg.norm(axis),
                        float(rng.uniform(0.02, 0.1)), (0.26, 0.24, 0.22), float(rng.uniform(0, 50))))
    return out


def rock_proximity(ship_pos, ship_R, rock: Rock):
    """GAME: the rock's surface clearance to the ship's collision points, m (negative: a point is inside the rock),
    measured along each lobe's centre ray, and for the nearest point (contact point in the world, unit normal from
    the rock toward the ship, name of the ship part). A rock farther than the two bounding spheres gets only the gap
    between them, and None."""
    dist = float(np.linalg.norm(rock.pos - ship_pos))
    bound = rock.bound
    if dist > SHIP_BOUND + bound + 2.0:
        return dist - SHIP_BOUND - bound, None
    Rr = rock.rot()
    pts = ship_pos + SHIP_POINTS @ np.asarray(ship_R).T               # (N, 3) world
    v = ((pts - rock.pos) @ Rr)[:, None, :] - rock.lobe_c[None]       # (N, L, 3) rock frame, from each lobe centre
    q = np.linalg.norm(v / rock.lobe_r[None], axis=-1)                # < 1 inside that lobe
    gap = (q - 1.0) * np.linalg.norm(v, axis=-1) / np.maximum(q, 1e-9)
    i, lobe = np.unravel_index(int(np.argmin(gap)), gap.shape)
    n = Rr @ (v[i, lobe] / rock.lobe_r[lobe] ** 2)
    n /= max(float(np.linalg.norm(n)), 1e-12)
    return float(gap[i, lobe]), (pts[i], n, SHIP_POINT_PART[i])


def collide(ship: ShipState, rock: Rock, contact=None, restitution=0.4, ship_mass=1.0, rock_mass=1.5):
    """GAME: if the rock touches the ship (a collision point of the hull is inside one of its lobes), bounce them apart
    with a momentum-conserving impulse along the contact normal, and give the ship an angular kick of
    min(0.45 rad/s, 0.016 x impulse) x min(1, lever / 1.5 m) about the blow's lever axis (contact offset x normal),
    scaled per body axis by mean inertia / axis inertia. Returns {dv, kick, point, normal, part} or None."""
    if contact is None:
        gap, contact = rock_proximity(ship.pos, ship.R, rock)
        if gap >= 0 or contact is None:
            return None
    p, n, part = contact
    R = ship.R
    r = p - ship.pos
    v_point = ship.vel + R @ np.cross(ship.w, R.T @ r)                # the hull point's velocity (spin included)
    vrel = float(np.dot(v_point - rock.vel, n))
    if vrel >= 0:
        return None
    j = -(1 + restitution) * vrel / (1 / ship_mass + 1 / rock_mass)
    dv = n * j / ship_mass
    ship.vel = ship.vel + dv
    rock.vel = rock.vel - n * j / rock_mass
    lever = R.T @ np.cross(r, n)                                      # body frame; zero for a blow through the centre
    size = float(np.linalg.norm(lever))
    kick = lever / size * min(0.45, 0.016 * j) * min(1.0, size / 1.5) if size > 1e-9 else np.zeros(3)
    ship.w = ship.w + kick / INERTIA * INERTIA.mean()
    return {"dv": dv, "kick": kick, "point": p, "normal": n, "part": part}


# ======================================================================================================== decoders
# Populations (Connectome.select criteria) and the laws that turn their rates into thruster commands. Weights and
# thresholds were measured / tuned on dev seeds >= 100 only (games/captions/spacecraft.md, "Design").
YAW_CELLS = {"HSN": -0.47, "HSE": -1.15, "H2": 1.0}   # yaw signal = sum w * (L - R), Hz; least-squares fit (no
                                                     # intercept) of body yaw rate on dev seeds 100 + 101, scaled to H2
ROLL_CELLS = {"VST2": 1.0}                            # roll signal = VST2 L - R, Hz (the one VS-family population whose
                                                     # weight kept its sign across both dev seeds)
# Side equalisation: during any rotation the right-side cells fire more than the left (the L - R of both signals falls
# with total activity), which a plain -G (L - R) turns into a steady torque that spins the ship up. Each signal is
# therefore (L - R) - BETA * (L + R), with BETA = mean(L - R) / mean(L + R) over the dev-seed tumbles (seeds 100, 101):
# zero activity is still zero torque.
BETA = {"roll": -0.31, "yaw": -0.20}
LOOM_CELLS = ("LPLC2", "LC4")


def side_sel(t, s):
    return {"type": t, "somaSide": s}


class Ema:
    """First-order low-pass with time constant tau (s); state is a float or an array."""

    def __init__(self, tau_s):
        self.tau, self.y = float(tau_s), None

    def __call__(self, x, dt_s):
        x = np.asarray(x, float)
        if self.y is None:
            self.y = np.zeros_like(x)
        a = math.exp(-dt_s / self.tau) if self.tau > 0 else 0.0
        self.y = a * self.y + (1 - a) * x
        return self.y


def population_means(x) -> dict:
    """{label: (B, n) rate tensor} -> {label: mean Hz of row 0}, with one device -> host copy."""
    keys = list(x)
    v = torch.stack([x[k][0].float().mean() for k in keys]).cpu().numpy()
    return dict(zip(keys, (float(a) for a in v)))


def weighted_lr(means: dict, weights: dict, beta: float = 0.0) -> float:
    """sum_t w_t * (L_t - R_t) - beta * sum_t |w_t| (L_t + R_t), from mean rates keyed 't_L' / 't_R' (Hz)."""
    d = sum(w * (means[f"{t}_L"] - means[f"{t}_R"]) for t, w in weights.items())
    s = sum(abs(w) * (means[f"{t}_L"] + means[f"{t}_R"]) for t, w in weights.items())
    return float(d - beta * s)


class AttitudeLaw:
    """Attitude hold: alpha_roll = -G_roll * roll_signal, alpha_yaw = -G_yaw * yaw_signal (deg/s^2 per Hz), where each
    signal is a weighted left-minus-right of lobula-plate tangential cells, low-passed with `tau_s`. Pitch is not
    built (no population carried it; see the caption). The value is a dict: raw means (CONNECTOME), signals and the
    commanded angular acceleration (DECODER)."""

    def __init__(self, g_roll=3.0, g_yaw=4.0, tau_s=0.5, hp_s=0.0, clip_hz=0.0):
        self.g_roll, self.g_yaw, self.tau_s, self.hp_s, self.clip_hz = g_roll, g_yaw, tau_s, hp_s, clip_hz
        self.lp = Ema(tau_s)
        self.slow = Ema(hp_s) if hp_s > 0 else None

    def parameters(self):
        return {"yaw_weights_Hz": YAW_CELLS, "roll_weights_Hz": ROLL_CELLS, "side_equalisation_beta": BETA,
                "G_roll_deg_s2_per_Hz": self.g_roll,
                "G_yaw_deg_s2_per_Hz": self.g_yaw, "lowpass_tau_s": self.tau_s, "highpass_tau_s": self.hp_s,
                "signal_clip_Hz": self.clip_hz or None, "pitch": "not built"}

    def from_means(self, means, dt_s):
        raw = np.array([weighted_lr(means, ROLL_CELLS, BETA["roll"]), weighted_lr(means, YAW_CELLS, BETA["yaw"])])
        sig = self.lp(raw, dt_s)
        if self.slow is not None:                           # remove slow drift: signal minus its hp_s running mean
            sig = sig - self.slow(sig, dt_s)
        if self.clip_hz > 0:                                # saturate: a noise excursion cannot command much torque
            sig = np.clip(sig, -self.clip_hz, self.clip_hz)
        roll_s, yaw_s = sig
        alpha = np.array([-self.g_roll * roll_s, 0.0, -self.g_yaw * yaw_s]) * DEG      # rad/s^2, body (x, y, z)
        return {"means": means, "roll_sig": float(roll_s), "yaw_sig": float(yaw_s), "alpha": alpha}

    def __call__(self, dt_ms, x):
        return self.from_means(population_means(x), dt_ms / 1000)


class DodgeLaw:
    """Dodge: side = mean(LPLC2_L, LC4_L) - mean(LPLC2_R, LC4_R), low-passed; |side| >= threshold fires a sideways push
    away from the looming side (the push itself is GAME: size and refractory in the game)."""

    def __init__(self, threshold_hz=5.0, tau_s=0.05):
        self.threshold, self.tau_s = threshold_hz, tau_s
        self.lp = Ema(tau_s)

    def parameters(self):
        return {"cells": list(LOOM_CELLS), "threshold_Hz": self.threshold, "lowpass_tau_s": self.tau_s}

    def from_means(self, means, dt_s):
        L = np.mean([means[f"{t}_L"] for t in LOOM_CELLS])
        R = np.mean([means[f"{t}_R"] for t in LOOM_CELLS])
        side = float(self.lp(L - R, dt_s))
        fire = 0 if abs(side) < self.threshold else (1 if side > 0 else -1)   # +1: loom on the left -> push right
        return {"means": means, "L": float(L), "R": float(R), "side": side, "fire": fire}

    def __call__(self, dt_ms, x):
        return self.from_means(population_means(x), dt_ms / 1000)


class EscapeLaw:
    """Escape burn: the giant fibre DNp01 (mean of both cells, the shipped body's `motor.gf`) >= 33 Hz, the shipped
    body model's takeoff threshold (flyverse/body.py Flight.gf_hz)."""

    def __init__(self, threshold_hz=33.0):
        self.threshold = threshold_hz

    def parameters(self):
        return {"cells": "DNp01 (both)", "threshold_Hz": self.threshold}

    def from_means(self, gf_l, gf_r):
        gf = 0.5 * (gf_l + gf_r)
        return {"gf": gf, "L": gf_l, "R": gf_r, "fire": gf >= self.threshold}

    def __call__(self, dt_ms, x):
        m = population_means(x)
        return self.from_means(m["GF_L"], m["GF_R"])


# ======================================================================================================== the game
DODGE_DV, DODGE_REFRACTORY = 4.0, 1.0          # m/s sideways push, s (GAME)
BURN_DV, BURN_S, BURN_REFRACTORY = 6.0, 0.4, 1.5   # escape burn: m/s along body (forward + up)/sqrt 2, over s; refractory s
BURN_DIR = np.array([1.0, 0.0, 1.0]) / math.sqrt(2)
STATION = {"pos": unit_from_az_el(213.0, -4.0) * 420.0, "axis": np.array([0.35, 0.8, 0.5]), "R": 45.0, "r": 3.6,
           "spin": 0.10}                        # rad/s (GAME scenery; its rim moves ~4.5 m/s, far from the ship)
TRACE_S = 6.0                                  # seconds of history in the HUD traces
HOLD_BANNER_DEG_S = 15.0                       # the one-shot 'attitude hold' banner: |w| first below this
NEAR_M = 50.0                                  # event banners say 'no rock within 50 m' beyond this
APPROACH_M = 30.0                              # each rock's log carries the peak loom / GF rates within this clearance

# HUD layout, on the 1920 x 1080 design canvas (another --size gets this canvas scaled to it)
DESIGN = (1920, 1080)
TOP, FOOT_H = 64, 48                           # header and footer strips (common.Hud)
PAD, LW, RW = 20, 420, 430                     # side gutter; left and right column widths
FREE = (PAD + LW + 16, DESIGN[0] - PAD - RW - 16)   # x range of the open main view between the columns


class Banner:
    def __init__(self, t, text, sub="", color=common.WHITE, hold=1.2, fade=0.8):
        self.t, self.text, self.sub, self.color, self.hold, self.fade = t, text, sub, color, hold, fade

    def alpha(self, now):
        age = now - self.t
        if age < 0.12:
            return age / 0.12
        if age < self.hold:
            return 1.0
        return max(0.0, 1.0 - (age - self.hold) / self.fade)


class Series:
    """A growable per-tick history buffer: amortised O(1) appends, and `all()` / `tail(n)` are views, not copies."""

    def __init__(self, width=None, cap=4096):
        self.a = np.zeros((cap,) if width is None else (cap, width))
        self.n = 0

    def append(self, v):
        if self.n == len(self.a):
            self.a = np.concatenate([self.a, np.zeros_like(self.a)])
        self.a[self.n] = v
        self.n += 1

    def all(self):
        return self.a[:self.n]

    def tail(self, k):
        return self.a[max(0, self.n - k):self.n]

    def __len__(self):
        return self.n


class Spacecraft(common.Game):
    title = "spacecraft"
    subtitle = "6DOF: a fly flies a spaceship"

    def __init__(self, args):
        super().__init__(args)
        self.control = bool(args.control)
        self.still = bool(getattr(args, "still", False))
        self.scenario = make_scenario(args.seed, self.still)
        self.fb = common.build_brain(args)                       # preset 'raw'; only read-only decoders attached
        self.dev = self.fb.device
        self.eyes = common.Eyes(self.fb)
        self.render = SpaceRenderer(self.dev)
        self.eye_scene = EyeScene(self.render)
        self.dirs_body_np = self.eyes.dirs_body.reshape(-1, 3).astype(np.float64)
        cam_scale = getattr(args, "cam_scale", None)
        self.cam_scale = float(cam_scale) if cam_scale is not None else (1.0 if args.record else 0.5)
        self.play_rng = np.random.default_rng([SCENARIO_KEY, int(args.seed), 1])   # interactive keys only
        self.fx_rng = np.random.default_rng([SCENARIO_KEY, int(args.seed), 2])     # impact sparks (drawing only)
        # ---- decoders: read-only modules, recorded in fb.module_records()
        self.att_law = AttitudeLaw(args.g_roll, args.g_yaw, args.att_tau, args.att_hp, getattr(args, "att_clip", 0.0))
        att_reads = {f"{t}_{s}": side_sel(t, s) for t in list(YAW_CELLS) + list(ROLL_CELLS) for s in "LR"}
        self.att = self.fb.attach(common.ReadDecoder(
            "attitude_hold", att_reads, self.att_law,
            law=("roll accel = -G_roll * s_roll, yaw accel = -G_yaw * s_yaw; s = sum w (L-R) - beta sum |w| (L+R) with "
                 "roll: VST2 (w 1, beta -0.31), yaw: H2 1, HSN -0.47, HSE -1.15 (beta -0.20); low-passed 0.5 s; "
                 "pitch not built"),
            parameters=self.att_law.parameters()))
        self.dodge_law = DodgeLaw(args.dodge_hz)
        self.dodge = self.fb.attach(common.ReadDecoder(
            "dodge", {f"{t}_{s}": side_sel(t, s) for t in LOOM_CELLS for s in "LR"}, self.dodge_law,
            law="side = mean(LPLC2, LC4) left - right, low-passed 50 ms; |side| >= threshold -> push away from that side",
            parameters=self.dodge_law.parameters()))
        self.esc_law = EscapeLaw(33.0)
        self.esc = self.fb.attach(common.ReadDecoder(
            "escape_burn", {"GF_L": side_sel("DNp01", "L"), "GF_R": side_sel("DNp01", "R")}, self.esc_law,
            law="giant fibre DNp01 (mean of both) >= 33 Hz -> escape burn", parameters=self.esc_law.parameters()))
        L = self.log
        sc = self.scenario
        common.declare(L, "rigid_body", "game", "Euler's equations for a rigid ship (principal moments below) + "
                       "quaternion attitude; torque = I * commanded angular acceleration, clipped per axis",
                       inertia=INERTIA.tolist(), alpha_max_deg_s2=(ALPHA_MAX / DEG).tolist())
        common.declare(L, "scenario", "game", "drawn from the seed on its own RNG stream (default_rng([0x5CAF, seed])): "
                       "the initial tumble (55-70 deg/s about an axis scattered around (40, 18, -45)) and each rock's "
                       "direction (from beyond the ship, near the chase camera's line of sight), aim offset and shape; "
                       "arrival times, speeds and sizes are the same for every seed", **sc.describe())
        common.declare(L, "torque_free_reference", "game", "a copy of the ship's rotation integrated with no torque "
                       "and no rock impacts, from the same start: what the tumble would be with the thrusters off and "
                       "nothing hitting it (the dashed line in the TUMBLE panel)")
        common.declare(L, "cruise_autopilot", "game", "translation only: relaxes world velocity toward CRUISE_V; "
                       "never touches attitude", cruise_v=CRUISE_V.tolist(), tau_s=CRUISE_TAU, amax=CRUISE_AMAX)
        common.declare(L, "dodge_thruster", "game", "a fired dodge adds a sideways velocity change along body y",
                       dv_m_s=DODGE_DV, refractory_s=DODGE_REFRACTORY)
        common.declare(L, "escape_thruster", "game", "a fired burn accelerates along body (forward + up)",
                       dv_m_s=BURN_DV, duration_s=BURN_S, refractory_s=BURN_REFRACTORY)
        common.declare(L, "rocks", "game", "each threat is launched LAUNCH_LEAD s before its arrival so that it would "
                       "pass its aim offset from the ship's centre if only the autopilot moved the ship; a rock is the "
                       "union of N_LOBES ellipsoids; contact when a point of the ship's hull (ship_surface_points) is "
                       "inside it; momentum-conserving bounce along the contact normal (rock 1.5x the ship's mass, "
                       "restitution 0.4) plus an angular kick of min(0.45 rad/s, 0.016 x impulse) x min(1, lever/1.5 m). "
                       "Each rock's impact / miss event carries the peak LPLC2+LC4 per side and GF rates (population "
                       "means, per 10 ms tick) while its clearance was under APPROACH_M",
                       lead_s=LAUNCH_LEAD, slots=THREAT_SLOTS, lobes=N_LOBES, hull_points=int(len(SHIP_POINTS)),
                       approach_m=APPROACH_M)
        common.declare(L, "eye_mount", "game", "the fly's eyes look out along the ship's axes from the cockpit "
                       "(x forward, y left, z up); the hull is not drawn in its view. The eye's rays are traced on the "
                       "CPU (EyeScene): the sky at infinity from a 2048 x 1024 texture of the camera's own sky function, "
                       "rocks (their lobes) and station with plain Lambert shading (no surface bump detail)",
                       eye_body_m=EYE_BODY.tolist())
        common.declare(L, "scene", "game", "planet, sun, moon, nebulae and stars at infinity; station, rocks local; "
                       "UV = 0.5 B", planet_angular_radius_deg=PLANET_ANG, sun_el_deg=SUN_EL0, nebula=NEBULA)
        common.declare(L, "hud_display", "game", "the HUD's CONNECTOME per-type rates are 0.5 s running means of the "
                       "decoders' population means (LPLC2+LC4: 0.1 s); the decoded signals and the GF trace are shown "
                       "as the decoders compute them")
        if self.control:
            common.declare(L, "control_arm", "game", "decoders attached and read, their outputs NOT applied: "
                           "thrusters off")
        if self.still:
            common.declare(L, "still_start", "game", "the ship starts at rest and no rocks are sent")
        # ---- world state
        w0 = sc.w0_deg * DEG
        self.ship = ShipState(np.zeros(3), CRUISE_V.copy(), np.array([1.0, 0, 0, 0]), w0.copy())
        self.ghost = self.ship.copy()                            # torque-free reference (GAME)
        self.rocks = scenery_rocks(np.random.default_rng(1234))  # scenery: the same sky for every seed
        self.pending = list(enumerate(sc.threats))
        self.n_player_rocks = 0
        self.station_angle = 0.0
        self.burn_left = 0.0
        self.last_burn = -1e9
        self.last_dodge = -1e9
        self.dodge_side = 1
        self.alpha_cmd = np.zeros(3)
        self.banners = []
        self.cam_anchor = self.ship.pos.copy()
        self.shake = 0.0
        self.last_rad = None
        self.flash = None                                        # (t, world point) of the last contact
        self.sparks = None                                       # impact debris: dict of arrays (drawing only)
        self.hold_announced = self.still or float(np.linalg.norm(sc.w0_deg)) < 2 * HOLD_BANNER_DEG_S
        self.disp_att = np.zeros(0)                              # display means (set on the first tick)
        self.disp_loom = np.zeros(2)
        self.disp_att_lp = Ema(0.5)
        self.disp_loom_lp = Ema(0.1)
        self._canvas = None
        self._final_summary = False
        self.h = {k: Series(3 if k == "w" else None) for k in
                  ("t", "w", "speed", "ghost", "yaw_sig", "roll_sig", "gf", "loomL", "loomR", "alpha_r", "alpha_y")}
        self.counts = {"burns": 0, "dodges": 0, "impacts": 0, "misses": 0, "burns_disconnected": 0,
                       "dodges_disconnected": 0}
        self.rate_samples = []
        self.t_below = None
        L.event(0.0, "tumble_start", w_deg_s=[round(float(v), 2) for v in sc.w0_deg],
                speed_deg_s=round(float(np.linalg.norm(sc.w0_deg)), 2), control=self.control, still=self.still)

    def brains(self):
        return {"brain": self.fb}

    # ------------------------------------------------------------------------------------------------ world
    def station_spec(self):
        ax = STATION["axis"] / np.linalg.norm(STATION["axis"])
        _, e1, e2 = look_basis(ax)
        base = np.stack([e1, e2, ax], 1)                         # columns: two in-plane axes, then the spin axis
        spin = rot_axis((0, 0, 1), self.station_angle)
        return {"pos": STATION["pos"], "rot": base @ spin, "R": STATION["R"], "r": STATION["r"]}

    def eye_radiance(self):
        R = self.ship.R
        d = self.dirs_body_np @ R.T
        o = self.ship.pos + R @ EYE_BODY
        rgb = self.eye_scene.trace(o, d, [r.render_spec() for r in self.rocks], self.station_spec())
        return self.eyes.pool(common.rgb_to_radiance(rgb.astype(np.float32)))

    def banner(self, text, sub="", color=common.WHITE):
        self.banners.append(Banner(self.t_s, text, sub, color))

    def nearest_threat(self):
        """(rock, surface clearance m) of the nearest unresolved threat, or None."""
        live = [r for r in self.rocks if r.threat and not r.resolved]
        return min(((r, r.gap) for r in live), key=lambda x: x[1]) if live else None

    @staticmethod
    def near_text(near):
        if near is None or near[1] > NEAR_M:
            return f"no rock within {NEAR_M:.0f} m"
        return f"{near[0].name} {max(near[1], 0.0):.0f} m away"

    # ------------------------------------------------------------------------------------------------ tick
    def tick(self):
        dt = common.TICK_MS / 1000
        t = self.t_s
        # scripted threats (GAME)
        while self.pending and self.pending[0][1].t_arrive - LAUNCH_LEAD <= t + 1e-9:
            k, th = self.pending.pop(0)
            rock = make_threat(f"rock {k + 1}", th, self.ship.pos, self.ship.vel, t)
            self.rocks.append(rock)
            self.log.event(t, "rock_launch", rock=rock.name, **th.describe())
        # the fly sees, the brain steps (the decoders read the previous frame's rates)
        rad = self.eye_radiance()
        self.last_rad = rad
        self.fb.vision(rad)
        self.fb.step(common.TICK_MS)
        att, dod, esc = self.att.value, self.dodge.value, self.esc.value
        self.alpha_cmd = np.clip(att["alpha"], -ALPHA_MAX, ALPHA_MAX)
        accel = cruise_autopilot(self.ship.vel)
        live = not self.control
        w_now = float(np.linalg.norm(self.ship.w) / DEG)
        # dodge (DECODER) -> sideways push (GAME)
        if dod["fire"] and t - self.last_dodge >= DODGE_REFRACTORY:
            self.last_dodge = t
            self.dodge_side = dod["fire"]
            near = self.nearest_threat()
            info = dict(side_hz=round(dod["side"], 2), L_hz=round(dod["L"], 2), R_hz=round(dod["R"], 2),
                        w_deg_s=round(w_now, 1), nearest_rock=near[0].name if near else None,
                        nearest_clearance_m=round(near[1], 1) if near else None)
            side = "LEFT" if dod["fire"] > 0 else "RIGHT"
            if live:
                self.ship.vel = self.ship.vel - dod["fire"] * self.ship.R[:, 1] * DODGE_DV   # loom left -> push to body -y
                self.counts["dodges"] += 1
                self.log.event(t, "dodge", away_from=side.lower(), **info)
                self.banner(f"LPLC2/LC4 {side} {abs(dod['side']):.1f} Hz  ->  DODGE",
                            f"decoder: loom side -> sideways thrusters  ·  {self.near_text(near)}", common.TEAL)
            else:
                self.counts["dodges_disconnected"] += 1
                self.log.event(t, "dodge_disconnected", **info)
                self.banner(f"LPLC2/LC4 {side} {abs(dod['side']):.1f} Hz  ->  DODGE NOT APPLIED",
                            f"control run: decoders disconnected  ·  {self.near_text(near)}", common.MUTED)
        # escape burn (DECODER) -> burn (GAME)
        if esc["fire"] and t - self.last_burn >= BURN_REFRACTORY:
            self.last_burn = t
            near = self.nearest_threat()
            info = dict(gf_hz=round(esc["gf"], 1), gf_L_hz=round(esc["L"], 1), gf_R_hz=round(esc["R"], 1),
                        w_deg_s=round(w_now, 1), nearest_rock=near[0].name if near else None,
                        nearest_clearance_m=round(near[1], 1) if near else None)
            if live:
                self.burn_left = BURN_S
                self.counts["burns"] += 1
                self.log.event(t, "escape_burn", **info)
                self.banner(f"GIANT FIBRE {esc['gf']:.0f} Hz  ->  ESCAPE BURN",
                            f"DNp01 over the body's 33 Hz line  ·  tumbling {w_now:.0f} deg/s  ·  {self.near_text(near)}",
                            common.AMBER)
            else:
                self.counts["burns_disconnected"] += 1
                self.log.event(t, "escape_burn_disconnected", **info)
                self.banner(f"GIANT FIBRE {esc['gf']:.0f} Hz  ->  BURN NOT APPLIED",
                            f"control run: decoders disconnected  ·  {self.near_text(near)}", common.MUTED)
        if live and self.burn_left > 0:
            accel = accel + self.ship.R @ BURN_DIR * (BURN_DV / BURN_S)
            self.burn_left -= dt
        alpha = self.alpha_cmd if live else np.zeros(3)
        self.ship = rigid_step(self.ship, alpha, accel, dt)
        self.ghost = rigid_step(self.ghost, np.zeros(3), np.zeros(3), dt)
        # rocks move, spin, collide, and are scored (GAME)
        R = self.ship.R
        for r in self.rocks:
            r.pos = r.pos + r.vel * dt
            r.angle += r.spin * dt
            if not r.threat or r.resolved:
                continue
            r.gap, contact = rock_proximity(self.ship.pos, R, r)
            if r.gap < r.closest:
                r.closest, r.t_closest = r.gap, t
            if r.gap < APPROACH_M:                               # what the loom pathway did on the way in (CONNECTOME)
                ap = r.approach
                for k, v in (("loom_L_hz", dod["L"]), ("loom_R_hz", dod["R"]), ("gf_hz", esc["gf"]),
                             ("gf_L_hz", esc["L"]), ("gf_R_hz", esc["R"])):
                    ap[k] = round(max(ap.get(k, 0.0), float(v)), 2)
            hit = collide(self.ship, r, contact) if (r.gap < 0 and contact is not None) else None
            if hit is not None:
                r.resolved, r.part = "impact", hit["part"]
                self.counts["impacts"] += 1
                w_after = float(np.linalg.norm(self.ship.w) / DEG)
                self.log.event(t, "impact", rock=r.name, part=hit["part"],
                               dv_m_s=round(float(np.linalg.norm(hit["dv"])), 2),
                               kick_deg_s=round(float(np.linalg.norm(hit["kick"]) / DEG), 1),
                               w_before_deg_s=round(w_now, 1), w_after_deg_s=round(w_after, 1),
                               approach_peak=dict(r.approach))
                self.banner("IMPACT", f"{r.name} hit the {hit['part']}  ·  tumble now {w_after:.0f} deg/s", common.RED)
                self.shake = 1.0
                self.flash = (self.t_s, hit["point"].copy())
                self._spawn_sparks(hit["point"], hit["normal"], self.ship.vel, r.vel)
            elif r.gap > r.closest + 3.0 and np.dot(r.vel - self.ship.vel, r.pos - self.ship.pos) > 0:
                r.resolved = "miss"
                self.counts["misses"] += 1
                self.log.event(t, "miss", rock=r.name, clearance_m=round(r.closest, 2), t_closest=round(r.t_closest, 3),
                               approach_peak=dict(r.approach))
                self.banner(f"MISS  {r.closest:.1f} m", f"{r.name}  ·  closest surface clearance", common.SAGE)
        self.rocks = [r for r in self.rocks if r.threat or np.linalg.norm(r.pos - self.ship.pos) < 900]
        self._step_sparks(dt)
        self.station_angle += STATION["spin"] * dt
        self.shake = max(0.0, self.shake - dt * 2.5)
        self.t_s = round(t + dt, 6)
        speed = float(np.linalg.norm(self.ship.w) / DEG)
        ghost = float(np.linalg.norm(self.ghost.w) / DEG)
        # the one-shot hold banner (a fact about the physics, whichever arm)
        if self.t_below is None and speed < HOLD_BANNER_DEG_S:
            self.t_below = self.t_s
            self.log.event(self.t_s, "tumble_below_15", w_deg_s=round(speed, 1), torque_free_deg_s=round(ghost, 1))
            if not self.hold_announced:
                self.hold_announced = True
                w0 = float(np.linalg.norm(self.scenario.w0_deg))
                if live:
                    self.banner(f"ATTITUDE HOLD  {w0:.0f} -> {HOLD_BANNER_DEG_S:.0f} deg/s IN {self.t_s:.1f} s",
                                f"decoders -> roll / yaw thrusters  ·  torque-free it would spin at {ghost:.0f} deg/s",
                                common.TEAL)
                else:
                    self.banner(f"TUMBLE BELOW {HOLD_BANNER_DEG_S:.0f} deg/s", "control run: thrusters off",
                                common.MUTED)
        # history for the HUD and the summary
        m = att["means"]
        vec = np.array([m[f"{t_}_{s}"] for t_ in ("H2", "HSN", "HSE", "VST2") for s in "LR"])
        self.disp_att = self.disp_att_lp(vec, dt)
        self.disp_loom = self.disp_loom_lp(np.array([dod["L"], dod["R"]]), dt)
        h = self.h
        for k, v in (("t", self.t_s), ("w", self.ship.w / DEG), ("speed", speed), ("ghost", ghost),
                     ("yaw_sig", att["yaw_sig"]), ("roll_sig", att["roll_sig"]), ("gf", esc["gf"]),
                     ("loomL", dod["L"]), ("loomR", dod["R"]), ("alpha_r", self.alpha_cmd[0] / DEG),
                     ("alpha_y", self.alpha_cmd[2] / DEG)):
            h[k].append(v)
        if round(self.t_s * 100) % 50 == 0:                           # every 0.5 s
            self.rate_samples.append([round(self.t_s, 2), *[round(float(v), 2) for v in self.ship.w / DEG],
                                      round(speed, 2), round(ghost, 2)])
        if round(self.t_s * 100) % 25 == 0:
            self._summary()

    def _spawn_sparks(self, point, normal, ship_vel, rock_vel, n=56):
        """GAME, drawing only: debris thrown off the contact point, mostly along the surface."""
        rng = self.fx_rng
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        d = d - 0.7 * (d @ normal)[:, None] * normal[None]
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        hot = rng.uniform(size=n) < 0.6
        speed = np.where(hot, rng.uniform(5, 16, n), rng.uniform(1.5, 6, n))
        new = {"pos": np.repeat(np.asarray(point, float)[None], n, 0), "vel": 0.5 * (ship_vel + rock_vel)[None] + d * speed[:, None],
               "age": np.zeros(n), "life": np.where(hot, rng.uniform(0.5, 1.2, n), rng.uniform(1.2, 2.2, n)), "hot": hot}
        if self.sparks is None:
            self.sparks = new
        else:
            self.sparks = {k: np.concatenate([self.sparks[k], new[k]]) for k in new}

    def _step_sparks(self, dt):
        s = self.sparks
        if s is None:
            return
        s["pos"] = s["pos"] + s["vel"] * dt
        s["age"] = s["age"] + dt
        keep = s["age"] < s["life"]
        self.sparks = {k: v[keep] for k, v in s.items()} if keep.any() else None

    def _summary(self):
        t = self.h["t"].all()
        if not len(t):
            return
        sp, gh, w = self.h["speed"].all(), self.h["ghost"].all(), self.h["w"].all()

        def win(a, b, x=sp):
            m = (t > a) & (t <= b)
            return round(float(x[m].mean()), 2) if m.any() else None

        wins = {"0-1s": (0, 1), "4-5s": (4, 5), "9-10s": (9, 10), "14-15s": (14, 15), "19-20s": (19, 20),
                "24-25s": (24, 25), "last_2s": (t[-1] - 2, t[-1])}
        rocks = {}
        for r in self.rocks:
            if r.threat:
                rocks[r.name] = {"resolved": r.resolved or "pending", "t_arrive": r.t_arrive,
                                 "closest_clearance_m": round(r.closest, 2) if r.closest < 1e8 else None,
                                 "t_closest": round(r.t_closest, 2) if r.closest < 1e8 else None}
                if r.resolved == "impact":
                    rocks[r.name]["part"] = r.part
                rocks[r.name]["approach_peak_within_30m"] = dict(r.approach)
        gf = self.h["gf"].all()
        self.log.summary = {
            "control": self.control, "still": self.still, "seed": self.scenario.seed, "brain_s": round(float(t[-1]), 3),
            "w0_deg_s": [round(float(v), 2) for v in self.scenario.w0_deg],
            "w0_speed_deg_s": round(float(np.linalg.norm(self.scenario.w0_deg)), 2),
            "speed_mean_deg_s": {k: win(a, b) for k, (a, b) in wins.items()},
            "torque_free_speed_mean_deg_s": {k: win(a, b, gh) for k, (a, b) in wins.items()},
            "axis_rms_last_5s_deg_s": [round(float(np.sqrt((w[t > t[-1] - 5, i] ** 2).mean())), 2) for i in range(3)],
            "first_below_15_deg_s": self.t_below,
            "frac_time_gf_over_33": round(float((gf >= 33).mean()), 4),
            "counts": dict(self.counts),
            "rocks": rocks,
            "rate_samples_t_roll_pitch_yaw_speed_torquefree": self.rate_samples,
        }

    def finished(self):
        if not self._final_summary and self.t_s >= self.args.seconds - 0.011:
            self._final_summary = True                  # the end-of-run summary (tick refreshes it every 0.25 s)
            self._summary()
        return False

    # ------------------------------------------------------------------------------------------------ camera
    def camera_pose(self):
        """A world-stable chase camera (it does not tumble with the ship) that slowly orbits: it starts looking past
        the ship toward the station and swings round toward the sun over ~30 s (cam_look). Its position trails the
        ship (so a burn or a knock shows), but it always looks at the ship."""
        look = cam_look(self.t_s)
        pos = self.cam_anchor - look * CAM_DIST + np.array([0.0, 0.0, 1.2])
        if self.shake > 0:
            k = self.shake ** 2 * 0.6
            pos = pos + k * np.array([math.sin(self.t_s * 91.0), math.cos(self.t_s * 77.0), math.sin(self.t_s * 63.0)])
        fwd = self.ship.pos + look * 4.0 + np.array([0.0, 0.0, -1.2]) - pos
        return pos, fwd / np.linalg.norm(fwd)

    def project(self, p, cam, W, H):
        pos, f, right, up, tan = cam
        v = np.asarray(p, float) - pos
        z = float(np.dot(v, f))
        if z < 0.3:
            return None
        asp = W / H
        x = (float(np.dot(v, right)) / z / tan * 0.5 + 0.5) * (W - 1)
        y = (0.5 - float(np.dot(v, up)) / z / (tan / asp) * 0.5) * (H - 1)
        return x, y, z

    # ------------------------------------------------------------------------------------------------ draw
    def draw(self, surface):
        """The HUD is laid out on the 1920 x 1080 design canvas; any other --size gets it scaled."""
        if surface.get_size() == DESIGN:
            self._draw(surface)
            return
        if self._canvas is None:
            self._canvas = self.hud.pg.Surface(DESIGN)
        self._draw(self._canvas)
        self.hud.pg.transform.smoothscale(self._canvas, surface.get_size(), surface)

    def _draw(self, surface):
        pg = self.hud.pg
        Wc, Hc = DESIGN
        top, bottom = TOP, Hc - FOOT_H
        vw, vh = Wc, bottom - top
        surface.fill(common.BG)
        # ---- main view: the chase camera
        self.cam_anchor = self.cam_anchor + (self.ship.pos - self.cam_anchor) * 0.06
        pos, f = self.camera_pose()
        rw, rh = max(64, int(vw * self.cam_scale)), max(36, int(vh * self.cam_scale))
        glow = {"main": 1.0 if self.burn_left > 0 else 0.0, "pods": 1.0 if self.burn_left > 0 else 0.0}
        img = self.render.camera(pos, f, (0, 0, 1), rw, rh, CAM_FOV, self.t_s,
                                 asteroids=[r.render_spec() for r in self.rocks], station=self.station_spec(),
                                 ship={"pos": self.ship.pos, "rot": self.ship.R, "glow": glow})
        view = pg.image.frombuffer(img.tobytes(), (rw, rh), "RGB")
        if (rw, rh) != (vw, vh):
            view = pg.transform.smoothscale(view, (vw, vh))
        surface.blit(view, (0, top))
        _, left, up = look_basis(f)
        cam = (pos, f, -left, up, math.tan(CAM_FOV * DEG / 2))
        view_rect = pg.Rect(0, top, vw, vh)
        self._draw_thrusters_3d(surface, cam, view_rect)
        self._draw_impact_fx(surface, cam, view_rect)
        self._draw_reticles(surface, cam, view_rect)
        if self.control:                                         # the control arm is unmistakable in any cut
            pg.draw.rect(surface, common.AMBER, view_rect, 4)
        # ---- header, footer
        self.hud.header(surface, "spacecraft", "6DOF  ·  decoders read the fly's lobula plate and giant fibre to fire "
                        "the thrusters", common.model_line(self.fb))
        g = self.att_law
        self.hud.footer(surface, [f"yaw = -{g.g_yaw:g}[(H2 -.47HSN -1.15HSE)(L-R) +.2 sum|w|(L+R)]",
                                  f"roll = -{g.g_roll:g}[VST2 (L-R) +.31(L+R)]",
                                  "deg/s^2 per Hz, LP 0.5 s, no pitch",
                                  "dodge |LPLC2+LC4 L-R| >= %.0f Hz" % self.dodge_law.threshold, "burn GF >= 33 Hz"])
        # ---- overlays
        self._draw_left(surface)
        self._draw_right(surface)
        self._draw_banners(surface)

    def _glass(self, surface, rect, title=None, kind=None, alpha=205, border=common.LINE):
        pg = self.hud.pg
        rect = pg.Rect(rect)
        box = pg.Surface(rect.size, pg.SRCALPHA)
        box.fill((*common.PANEL, alpha))
        surface.blit(box, rect.topleft)
        pg.draw.rect(surface, border, rect, 1 if border == common.LINE else 2, border_radius=6)
        if not title:
            return rect.inflate(-16, -16)
        t = self.hud.text(surface, title.upper(), (rect.x + 12, rect.y + 9), 15, common.MUTED, bold=True)
        if kind:
            self.hud.chip(surface, kind, (t.right + 10, rect.y + 6), 12)
        return pg.Rect(rect.x + 10, rect.y + 34, rect.w - 20, rect.h - 42)

    def _tail(self, key, seconds=TRACE_S):
        return self.h[key].tail(int(seconds * 100))

    def _polyline(self, surface, rect, values, vmin, vmax, color, width=2, dashed=False, max_pts=240):
        """values (oldest first) across the full width of rect."""
        pg = self.hud.pg
        v = np.asarray(values, float)
        if len(v) < 2:
            return
        step = max(1, len(v) // max_pts)
        v = v[::step]
        x = rect.x + np.linspace(0, rect.w - 1, len(v))
        y = rect.bottom - 1 - np.clip((v - vmin) / (vmax - vmin), 0, 1) * (rect.h - 2)
        pts = np.c_[x, y].round().astype(int).tolist()
        if not dashed:
            pg.draw.lines(surface, color, False, pts, width)
            return
        for i in range(len(pts) - 1):
            if (i // 3) % 2 == 0:
                pg.draw.line(surface, color, pts[i], pts[i + 1], width)

    def _draw_left(self, surface):
        hud, pg = self.hud, self.hud.pg
        w = np.array(self.ship.w) / DEG
        sp = float(np.linalg.norm(w))
        x0 = PAD
        # ---- tumble
        r = self._glass(surface, (x0, TOP + 16, LW, 292))
        t = hud.text(surface, "TUMBLE", (r.x + 4, r.y), 17, common.MUTED, bold=True)
        hud.chip(surface, "game", (t.right + 10, r.y - 2), 12)
        hud.text(surface, "ship's angular speed (physics)", (r.right, r.y + 1), 14, common.DIM, anchor="topright")
        col = common.RED if sp > 30 else (common.AMBER if sp > HOLD_BANNER_DEG_S else common.SAGE)
        big = hud.text(surface, f"{sp:3.0f}", (r.x, r.y + 14), 76, col, bold=True, display=True)
        hud.text(surface, "deg/s", (big.right + 8, big.bottom - 36), 24, common.MUTED)
        ax = hud.text(surface, f"roll {w[0]:+4.0f}  pitch {w[1]:+4.0f}  yaw {w[2]:+4.0f}", (r.x + 2, big.bottom - 2),
                      22, common.TEXT)
        tr = pg.Rect(r.x, ax.bottom + 10, r.w, r.bottom - ax.bottom - 12)
        pg.draw.rect(surface, common.INSET, tr, border_radius=3)
        span = max(float(self.args.seconds), self.t_s, 1.0)
        vmax = 100.0
        n = len(self.h["speed"])
        if n > 2:
            sub = pg.Rect(tr.x, tr.y, max(2, int(tr.w * min(1.0, self.t_s / span))), tr.h)
            self._polyline(surface, sub, self.h["ghost"].all(), 0, vmax, common.DIM, 2, dashed=True)
            self._polyline(surface, sub, self.h["speed"].all(), 0, vmax, col, 2)
            gy = tr.bottom - 1 - min(1.0, self.h["ghost"].all()[-1] / vmax) * (tr.h - 2)
            hud.text(surface, "torque-free", (min(sub.right + 6, tr.right - 80), min(gy - 16, tr.bottom - 36)), 13,
                     common.DIM)
        hud.text(surface, "0", (tr.x + 4, tr.bottom - 17), 13, common.DIM)
        hud.text(surface, f"{span:.0f} s", (tr.right - 4, tr.bottom - 17), 13, common.DIM, anchor="topright")
        # ---- rocks
        y = TOP + 16 + 292 + 12
        r2 = self._glass(surface, (x0, y, LW, 88))
        t = hud.text(surface, "ROCKS", (r2.x + 4, r2.y), 17, common.MUTED, bold=True)
        hud.chip(surface, "game", (t.right + 10, r2.y - 2), 12)
        incoming = sum(1 for rk in self.rocks if rk.threat and not rk.resolved) + len(self.pending)
        xx = r2.x + 4
        for label, k, c in (("missed", self.counts["misses"], common.SAGE), ("impacts", self.counts["impacts"], common.RED),
                            ("to come", incoming, common.MUTED)):
            rr = hud.text(surface, f"{k}", (xx, r2.y + 20), 34, c, bold=True, display=True)
            rr2 = hud.text(surface, label, (rr.right + 6, r2.y + 34), 17, common.TEXT)
            xx = rr2.right + 24
        # ---- the arm
        y = y + 88 + 12
        if self.control:
            msg, c = "CONTROL  ·  THRUSTERS OFF", common.AMBER
        else:
            msg, c = "DECODERS DRIVE THRUSTERS", common.TEAL
        rr = self._glass(surface, (x0, y, LW, 50), border=c)
        hud.text(surface, msg, (rr.centerx, rr.centery), 22, c, bold=True, anchor="center")
        tb = self._glass(surface, (x0, DESIGN[1] - FOOT_H - 8 - 40, 236, 40))
        hud.text(surface, f"t = {self.t_s:5.2f} s brain", (tb.x + 2, tb.y), 18, common.TEXT)

    def _draw_right(self, surface):
        hud, pg = self.hud, self.hud.pg
        x0 = DESIGN[0] - PAD - RW
        live = not self.control
        # ---- the fly's eye
        r = self._glass(surface, (x0, TOP + 16, RW, 250), "What the fly sees")
        hud.text(surface, "radiance handed to fb.vision", (r.right, r.y - 27), 13, common.DIM, anchor="topright")
        if self.last_rad is not None:
            hud.mosaic(surface, (r.x + 6, r.y + 4, r.w - 12, r.h - 8), self.eyes,
                       common.eye_colors(self.last_rad, "human", 2.2))
        # ---- attitude hold
        y = TOP + 16 + 250 + 12
        r = self._glass(surface, (x0, y, RW, 340), "Attitude hold", "decoder")
        da = self.disp_att if len(self.disp_att) else np.zeros(8)
        h2, hsn, hse, vst = da[0] - da[1], da[2] - da[3], da[4] - da[5], (da[6], da[7])
        rows = [("YAW", "s_yaw", f"L-R  H2 {h2:+.1f}  HSN {hsn:+.1f}  HSE {hse:+.1f}", "yaw_sig", 2, "alpha_y", 6.0),
                ("ROLL", "s_roll", f"VST2  L {vst[0]:.1f}  R {vst[1]:.1f} Hz  (n 3/4)", "roll_sig", 0, "alpha_r", 10.0)]
        yy = r.y
        for name, sname, raw, key, axis, akey, vmax in rows:
            t = hud.text(surface, name, (r.x, yy), 20, common.WHITE, bold=True)
            c = hud.chip(surface, "decoder", (t.right + 8, yy + 2), 11)
            hud.text(surface, sname, (c.right + 8, yy + 2), 16, common.MUTED)
            sig = self._tail(key)
            now = float(sig[-1]) if len(sig) else 0.0
            hud.text(surface, f"{now:+5.1f} Hz", (r.right, yy - 3), 22, common.TEAL, bold=True, anchor="topright")
            c2 = hud.chip(surface, "connectome", (r.x, yy + 28), 11)
            hud.text(surface, raw, (c2.right + 8, yy + 27), 16, common.SAGE)
            tr = pg.Rect(r.x, yy + 52, r.w - 78, 56)
            hud.trace(surface, tr, sig, -vmax, vmax, common.TEAL)
            pg.draw.line(surface, common.LINE, (tr.x, tr.centery), (tr.right, tr.centery), 1)
            wt = self._tail("w")
            if len(wt) > 2:                                      # the ship's true rate, ±80 deg/s full scale (GAME)
                true = wt[:, axis]
                ys = tr.centery - np.clip(true / 80.0, -1, 1) * (tr.h / 2 - 2)
                xs = tr.x + np.linspace(0, tr.w - 1, len(true))
                pts = np.c_[xs, ys].round().astype(int)[:: max(1, len(true) // 200)].tolist()
                if len(pts) > 1:
                    pg.draw.lines(surface, common.AMBER, False, pts, 1)
            a = self._tail(akey)
            av = float(a[-1]) if len(a) else 0.0
            bx = pg.Rect(tr.right + 10, tr.y, r.right - tr.right - 10, tr.h)
            pg.draw.rect(surface, common.INSET, bx, border_radius=3)
            frac = float(np.clip(av / (ALPHA_MAX[axis] / DEG), -1, 1))
            hbar = int(abs(frac) * (bx.h / 2 - 3))
            if hbar > 0:
                pg.draw.rect(surface, common.TEAL if live else common.DIM,
                             (bx.x + 8, bx.centery - hbar if frac > 0 else bx.centery, bx.w - 16, hbar), border_radius=2)
            pg.draw.line(surface, common.LINE, (bx.x + 4, bx.centery), (bx.right - 4, bx.centery), 1)
            hud.text(surface, "thrust" if live else "not applied", (bx.centerx, bx.bottom + 1), 12, common.DIM,
                     anchor="midtop")
            yy = tr.bottom + 18
        hud.text(surface, "teal: decoded s = sum w(L-R) - beta sum|w|(L+R), LP 0.5 s", (r.x, yy - 4), 13, common.DIM)
        hud.text(surface, "amber: ship's true rate, ±80 deg/s full scale (GAME)", (r.x, yy + 11), 13, common.DIM)
        hud.text(surface, "PITCH: not built (no population carried it)", (r.x, yy + 26), 13, common.DIM)
        # ---- loom -> escape
        y = y + 340 + 12
        r = self._glass(surface, (x0, y, RW, 186), "Loom  ->  escape", "connectome")
        lv, rv = self.disp_loom
        half = r.w // 2 - 8
        for i, (lab, val) in enumerate((("LPLC2+LC4 L", lv), ("LPLC2+LC4 R", rv))):
            bx = pg.Rect(r.x + i * (half + 16), r.y - 2, half, 34)
            hud.text(surface, lab, (bx.x, bx.y + 4), 15, common.MUTED)
            hud.text(surface, f"{val:.1f} Hz", (bx.right, bx.y - 2), 22, common.SAGE, bold=True, anchor="topright")
            track = pg.Rect(bx.x, bx.bottom - 6, bx.w, 8)
            pg.draw.rect(surface, common.INSET, track, border_radius=3)
            fr = float(np.clip(val / 12.0, 0, 1))
            if fr > 0:
                pg.draw.rect(surface, common.SAGE, (track.x, track.y, max(2, int(track.w * fr)), track.h), border_radius=3)
        gf = self._tail("gf")
        gnow = float(gf[-1]) if len(gf) else 0.0
        yy = r.y + 44
        t = hud.text(surface, "GIANT FIBRE DNp01", (r.x, yy + 3), 16, common.TEXT, bold=True)
        hud.text(surface, "33 Hz line", (t.right + 10, yy + 4), 14, common.RED)
        hud.text(surface, f"{gnow:4.0f} Hz", (r.right, yy - 4), 26, common.RED if gnow >= 33 else common.SAGE,
                 bold=True, anchor="topright")
        tr = pg.Rect(r.x, yy + 28, r.w, r.bottom - yy - 28)
        hud.trace(surface, tr, gf, 0, 80, common.SAGE, threshold=33.0)
        # ---- thrusters
        y = y + 186 + 12
        r = self._glass(surface, (x0, y, RW, 76))
        t = hud.text(surface, "THRUSTERS", (r.x + 2, r.y - 4), 15, common.MUTED, bold=True)
        hud.chip(surface, "decoder", (t.right + 10, r.y - 7), 12)
        if not live:
            hud.text(surface, "control: would fire, not applied", (r.right, r.y - 3), 13, common.DIM, anchor="topright")
        now = self.t_s
        states = [("ROLL", abs(self.alpha_cmd[0]) / ALPHA_MAX[0]), ("YAW", abs(self.alpha_cmd[2]) / ALPHA_MAX[2]),
                  ("DODGE", 1.0 if now - self.last_dodge < 0.4 else 0.0),
                  ("BURN", 1.0 if (self.burn_left > 0 or now - self.last_burn < 0.4) else 0.0)]
        lw = (r.w - 3 * 8) // 4
        for i, (name, lvl) in enumerate(states):
            lvl = float(np.clip(lvl, 0, 1))
            if live:
                on = common.AMBER if name == "BURN" else common.TEAL
                c = on if (name in ("DODGE", "BURN") and lvl > 0) else \
                    tuple(int(common.INSET[k] + (on[k] - common.INSET[k]) * lvl) for k in range(3))
                edge, txt = common.LINE, (common.WHITE if lvl > 0.3 else common.MUTED)
            else:                                                # the control: what they would do, greyed
                grey = (78, 88, 92)
                c = tuple(int(common.INSET[k] + (grey[k] - common.INSET[k]) * lvl) for k in range(3))
                edge, txt = common.DIM, (common.TEXT if lvl > 0.3 else common.DIM)
            box = pg.Rect(r.x + i * (lw + 8), r.y + 16, lw, 32)
            pg.draw.rect(surface, c, box, border_radius=5)
            pg.draw.rect(surface, edge, box, 1, border_radius=5)
            hud.text(surface, name, box.center, 18, txt, bold=True, anchor="center")

    def _draw_banners(self, surface):
        pg, hud = self.hud.pg, self.hud
        self.banners = [b for b in self.banners if b.alpha(self.t_s) > 0 or self.t_s - b.t < 0.2]
        gap_l, gap_r = FREE
        cx, max_w = (gap_l + gap_r) // 2, gap_r - gap_l
        y = TOP + 22
        for b in self.banners[-2:][::-1]:
            a = b.alpha(self.t_s)
            if a <= 0:
                continue
            big = hud.font(52, True, True).render(b.text, True, b.color)
            if big.get_width() > max_w - 40:
                k = (max_w - 40) / big.get_width()
                big = pg.transform.smoothscale(big, (int(big.get_width() * k), int(big.get_height() * k)))
            sub = hud.font(20, False, False).render(b.sub, True, common.TEXT) if b.sub else None
            if sub is not None and sub.get_width() > max_w - 24:
                k = (max_w - 24) / sub.get_width()
                sub = pg.transform.smoothscale(sub, (int(sub.get_width() * k), int(sub.get_height() * k)))
            w = min(max_w, max(big.get_width(), sub.get_width() if sub else 0) + 48)
            h = big.get_height() + (sub.get_height() + 4 if sub else 0) + 20
            box = pg.Surface((w, h), pg.SRCALPHA)
            box.fill((*common.BG, int(185 * a)))
            pg.draw.rect(box, (*b.color, int(255 * a)), box.get_rect(), 2, border_radius=8)
            big.set_alpha(int(255 * a))
            box.blit(big, big.get_rect(midtop=(w // 2, 8)))
            if sub:
                sub.set_alpha(int(255 * a))
                box.blit(sub, sub.get_rect(midtop=(w // 2, 10 + big.get_height())))
            surface.blit(box, box.get_rect(midtop=(cx, y)))
            y += h + 10

    def _free_rect(self):
        return self.hud.pg.Rect(FREE[0], TOP + 8, FREE[1] - FREE[0], DESIGN[1] - FOOT_H - TOP - 16)

    def _draw_reticles(self, surface, cam, view_rect):
        """Corner brackets on each incoming rock, kept inside the open view; a rock under a HUD column or out of frame
        gets an edge marker pointing at it instead."""
        pg, hud = self.hud.pg, self.hud
        free = self._free_rect()
        inner = free.inflate(-40, -40)
        for r in self.rocks:
            if not r.threat or r.resolved:
                continue
            p = self.project(r.pos, cam, view_rect.w, view_rect.h)
            if p is None:
                continue
            x, y, z = p
            x, y = x + view_rect.x, y + view_rect.y
            dist = float(np.linalg.norm(r.pos - self.ship.pos))
            c = common.RED if dist < 60 else common.AMBER
            speed = float(np.linalg.norm(r.vel - self.ship.vel))
            label = f"{r.name.upper()}  {dist:3.0f} m  {speed:2.0f} m/s"
            if inner.collidepoint(x, y):
                rad = int(np.clip(r.bound / z / cam[4] * view_rect.w / 2 * 1.15, 16, 170))
                surface.set_clip(free)
                for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                    cxp, cyp = x + sx * rad, y + sy * rad
                    pg.draw.line(surface, c, (cxp, cyp), (cxp - sx * 14, cyp), 3)
                    pg.draw.line(surface, c, (cxp, cyp), (cxp, cyp - sy * 14), 3)
                surface.set_clip(None)
                img = hud.font(18, True).render(label, True, c)
                lx = x + rad + 10 if x + rad + 10 + img.get_width() < free.right else x - rad - 10 - img.get_width()
                ly = int(np.clip(y - rad, free.y + 4, free.bottom - 26))
                lx = int(np.clip(lx, free.x + 4, free.right - img.get_width() - 4))
                surface.blit(img, (lx, ly))
            else:                                                # an edge marker toward it
                cx0, cy0 = free.center
                dx, dy = x - cx0, y - cy0
                k = min((inner.w / 2) / max(abs(dx), 1e-6), (inner.h / 2) / max(abs(dy), 1e-6))
                ex, ey = cx0 + dx * k, cy0 + dy * k
                ang = math.atan2(dy, dx)
                tip = (ex + 12 * math.cos(ang), ey + 12 * math.sin(ang))
                l1 = (ex + 10 * math.cos(ang + 2.5), ey + 10 * math.sin(ang + 2.5))
                l2 = (ex + 10 * math.cos(ang - 2.5), ey + 10 * math.sin(ang - 2.5))
                pg.draw.polygon(surface, c, [tip, l1, l2])
                img = hud.font(16, True).render(label, True, c)
                lx = ex - img.get_width() - 16 if dx > 0 else ex + 16
                lx = int(np.clip(lx, free.x + 4, free.right - img.get_width() - 4))
                ly = int(np.clip(ey - 9, free.y + 4, free.bottom - 24))
                surface.blit(img, (lx, ly))

    def _draw_impact_fx(self, surface, cam, view_rect):
        """GAME, drawing only: a flash at the contact point and debris thrown off it (additive, not depth-tested)."""
        ppm = 0.5 * view_rect.w / cam[4]                        # pixels per metre at 1 m
        if self.flash is not None:
            age = self.t_s - self.flash[0]
            if age < 0.5:
                p = self.project(self.flash[1], cam, view_rect.w, view_rect.h)
                if p is not None:
                    x, y = view_rect.x + p[0], view_rect.y + p[1]
                    k = (1 - age / 0.5) ** 1.5
                    self._glow(surface, x, y, np.clip(7.0 * ppm / p[2], 60, 220), (255, 200, 150), 1.0 * k)
                    self._glow(surface, x, y, np.clip(2.5 * ppm / p[2], 24, 120), (255, 255, 240), 1.0 * k)
        s = self.sparks
        if s is None:
            return
        for pos, age, life, hot in zip(s["pos"], s["age"], s["life"], s["hot"]):
            p = self.project(pos, cam, view_rect.w, view_rect.h)
            if p is None:
                continue
            u = age / life
            x, y = view_rect.x + p[0], view_rect.y + p[1]
            if hot:
                col = (255, int(220 - 110 * u), int(150 - 110 * u))
                self._glow(surface, x, y, max(3.0, 0.16 * ppm / p[2]), col, 0.95 * (1 - u) ** 1.2)
            else:
                self._glow(surface, x, y, max(3.0, 0.35 * ppm / p[2]), (120, 108, 96), 0.7 * (1 - u))

    def _draw_thrusters_3d(self, surface, cam, view_rect):
        """Exhaust plumes drawn over the render where the decoders fire (additive glow)."""
        if self.control:
            return
        R, p0 = self.ship.R, self.ship.pos
        jets = []
        for axis, key in ((0, "roll"), (2, "yaw")):
            a = self.alpha_cmd[axis] / ALPHA_MAX[axis]
            if abs(a) > 0.08:
                for pos_b, dir_b in RCS[key + ("+" if a > 0 else "-")]:
                    jets.append((pos_b, dir_b, 0.8 + 1.4 * min(1.0, abs(a)), (140, 190, 255)))
        if self.burn_left > 0:
            for pos_b, dir_b in MAIN_ENGINES:
                jets.append((pos_b, dir_b, 14.0, (255, 165, 80)))
        if self.t_s - self.last_dodge < 0.35:
            side = self.dodge_side
            for sx in (2.0, -1.5):
                jets.append(((sx, side * 1.1, 0.0), (0, side, 0), 3.5, (180, 230, 255)))
        for pos_b, dir_b, length, col in jets:
            a = p0 + R @ np.asarray(pos_b, float)
            b = a + R @ np.asarray(dir_b, float) * length
            pa = self.project(a, cam, view_rect.w, view_rect.h)
            pb = self.project(b, cam, view_rect.w, view_rect.h)
            if pa is None or pb is None:
                continue
            base = 0.5 * view_rect.w / cam[4] / max(pa[2], 1.0)          # pixels per metre at the nozzle
            n = 16
            flick = 0.85 + 0.15 * math.sin(self.t_s * 157.0 + length)
            for k in range(n):
                u = k / (n - 1)
                x = view_rect.x + pa[0] + (pb[0] - pa[0]) * u
                y = view_rect.y + pa[1] + (pb[1] - pa[1]) * u
                rad = base * (0.25 + 0.55 * u) * (0.55 + 0.05 * length)
                self._glow(surface, x, y, rad, col, (0.9 if length > 10 else 0.55) * flick * (1 - u) ** 1.3)
            self._glow(surface, view_rect.x + pa[0], view_rect.y + pa[1], base * 0.5, (255, 255, 255), 0.9)

    def _glow(self, surface, x, y, rad, col, strength):
        """Additive radial glow (premultiplied sprite, cached by quantised size / colour / strength)."""
        pg = self.hud.pg
        r = int(max(2, min(rad, 220)))
        q = (r, tuple(int(v) for v in col), int(strength * 20))
        if q[2] <= 0:
            return
        cache = self.__dict__.setdefault("_glow_cache", {})
        spr = cache.get(q)
        if spr is None:
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            fall = np.exp(-(xx ** 2 + yy ** 2) / (0.35 * r * r)) * (q[2] / 20.0)
            rgb = np.clip(np.asarray(q[1], float)[None, None] * fall[..., None], 0, 255).astype(np.uint8)
            spr = pg.surfarray.make_surface(np.ascontiguousarray(rgb.transpose(1, 0, 2)))
            cache[q] = spr
            if len(cache) > 3000:
                cache.clear()
        surface.blit(spr, (int(x) - r, int(y) - r), special_flags=pg.BLEND_ADD)

    # ------------------------------------------------------------------------------------------------ input
    def handle(self, event, canvas_pos=None):
        pg = self.hud.pg if self.hud else None
        if pg is None or event.type != pg.KEYDOWN:
            return
        rng = self.play_rng
        if event.key == pg.K_k:                                  # GAME: a random kick, to watch the hold work
            v = rng.normal(size=3)
            self.ship.w = self.ship.w + v / np.linalg.norm(v) * 50 * DEG
            self.log.event(self.t_s, "kick", by="player")
            self.banner("KICK", "a random 50 deg/s tumble (GAME)", common.AMBER)
        elif event.key == pg.K_r:                                # GAME: a rock at the ship, arriving in LAUNCH_LEAD s
            self.n_player_rocks += 1
            th = draw_threat(rng, self.t_s + LAUNCH_LEAD, 25.0, 2.6, "hit", 1 if rng.uniform() < 0.5 else -1)
            rock = make_threat(f"player rock {self.n_player_rocks}", th, self.ship.pos, self.ship.vel, self.t_s)
            self.rocks.append(rock)
            self.log.event(self.t_s, "rock_launch", rock=rock.name, by="player", **th.describe())
        elif event.key == pg.K_c:
            self.control = not self.control
            self.log.event(self.t_s, "control_toggled", control=self.control)
            self.banner("THRUSTERS OFF" if self.control else "THRUSTERS LIVE", "", common.AMBER)


def main(argv=None):
    ap = common.standard_args("spacecraft -- 6DOF: a fly flies a spaceship", seconds=30.0)
    ap.add_argument("--control", action="store_true", help="the control arm: decoders read but NOT applied (thrusters off)")
    ap.add_argument("--still", action="store_true", help="start at rest with no rocks (what the decoders do to a still ship)")
    ap.add_argument("--g-roll", type=float, default=3.0, help="roll gain, deg/s^2 per Hz of the roll signal")
    ap.add_argument("--g-yaw", type=float, default=4.0, help="yaw gain, deg/s^2 per Hz of the yaw signal")
    ap.add_argument("--att-tau", type=float, default=0.5, help="attitude low-pass time constant, s")
    ap.add_argument("--att-hp", type=float, default=0.0, help="attitude high-pass time constant, s (0 = off)")
    ap.add_argument("--att-clip", type=float, default=0.0, help="saturate each decoded attitude signal at +-this, Hz (0 = off)")
    ap.add_argument("--dodge-hz", type=float, default=5.0, help="dodge threshold on LPLC2+LC4 L-R, Hz")
    ap.add_argument("--cam-scale", type=float, default=None,
                    help="chase-camera render scale (default 1.0 when recording, 0.5 interactive)")
    args = ap.parse_args(argv)
    game = Spacecraft(args)
    return common.run(game, args)


if __name__ == "__main__":
    main()
