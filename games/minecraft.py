"""minecraft -- the fly in real Minecraft.

MaleCNS v1.0 (167,106 neurons, preset 'raw') drives a mineflayer bot on a local offline-mode vanilla 1.21.4 server.

  EYES   The 1,466 x 7 ommatidial rays are cast from the player's eye (1.62 m above its feet) through the server's
         real block grid (fetched around the bot through the bridge) on the GPU: block textures from the game's own
         1.21.4 texture atlas, Minecraft's face shading, a sky with the sun at the world's time of day, living mobs
         as boxes of their hitbox size (a box that contains the eye is culled, as the vanilla client does). RGB ->
         radiance with UV = 0.5 B. A declared scale: the fly's eye becomes a player's eye.
  BODY   The shipped body readout on `fb.motor()`: `Locomotion.readout` + its 80 ms smoothing gives a walking speed
         and a yaw rate; the speed is scaled into the player's walk (DECODER, 2 cm/s brisk fly walk = Minecraft's
         4.317 m/s walk; about 90% of the fly's ~0.9 cm/s is the readout's constant 0.8 cm/s intrinsic drive), the
         yaw rate passes 1:1 (angles are scale-free). `Flight.maybe_takeoff` (giant fibre DNp01 >= 33 Hz, 1 s
         refractory after landing) -> a Minecraft jump (the jump itself is Minecraft's physics).
  LOCKSTEP  The world is frozen (/tick freeze). Every 50 ms of brain: one client physics tick of the bot
         (prismarine-physics, Minecraft's player movement), `/tick step 1` over RCON, and a chat marker that proves
         the bot has received that tick. The fly sees the world the way the vanilla client draws it: the player
         interpolated between its last two ticks, mobs between the last two server ticks.
  GAME   A fixed script (the same for every seed): three zombies walk at the player under Minecraft's own AI (each
         removed 0.6 s after its first hit), then two phantom dives, stepped every 10 ms brain frame against the
         eye's position in that frame: straight at the eye until the hitbox is 0.10 m from it, then up and over the
         head, never closer than 0.10 m. Minecraft's Auto-Jump option (jump into a 1-block step while walking) is on.
         Natural regeneration is off.
  FOOTAGE  Real Minecraft rendering: prismarine-viewer (three.js, the mineflayer project's Minecraft renderer) in
         headless Chrome, driven by puppeteer, from an orbiting chase camera (third person) and the player's eye.
         games/minecraft_bridge/web/fv.js corrects three things the viewer draws wrongly: limb pivots, the zombie /
         husk / sheep texture mapping, and the phantom's model offset (drawn on its hitbox, as vanilla does).

Run: python games/minecraft.py [--record out/games/minecraft/dev.mp4 --seconds 30 --seed 100]
Needs: Java 21+, Node 18+, `npm install` in games/minecraft_bridge, ffmpeg for --record.
"""
from __future__ import annotations

import base64
import io
import json
import math
import socket
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:                     # `python games/minecraft.py`, or `import games.minecraft` in tests
    sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

VIEWER_PUBLIC = HERE / "minecraft_bridge" / "node_modules" / "prismarine-viewer" / "public"
MC_VERSION = "1.21.4"

# Biome tints (the plains values of the vanilla colour maps; a declared simplification: one biome's tint everywhere)
GRASS_TINT = (0x91, 0xBD, 0x59)
FOLIAGE_TINT = (0x77, 0xAB, 0x2F)
WATER_TINT = (0x3F, 0x76, 0xE4)
FIXED_FOLIAGE = {"birch_leaves": (0x80, 0xA7, 0x55), "spruce_leaves": (0x61, 0x99, 0x61)}
# Minecraft's directional face shading (the vanilla renderer's constants): top, bottom, north/south, east/west
FACE_SHADE = {"top": 1.0, "bottom": 0.5, "z": 0.8, "x": 0.6}
LIQUIDS = {"water": ("water_still", WATER_TINT), "lava": ("lava_still", None), "bubble_column": ("water_still", WATER_TINT)}

TOP, BOTTOM, SIDE = 0, 1, 2


def srgb8_to_linear(x):
    x = np.asarray(x, np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _first_model(state: dict):
    var = state.get("variants")
    if var:
        first = next(iter(var.values()))
    else:
        mp = state.get("multipart") or []
        first = mp[0].get("apply") if mp else None
    if isinstance(first, list):
        first = first[0] if first else None
    if not first:
        return None
    return first.get("model", first)


class BlockTextures:
    """Per block name: three 16 x 16 linear-RGB textures (top, bottom, side) cut from the viewer's 1.21.4 atlas, with
    the biome tint applied where the model asks for it and any tinted overlay (the grass block's side fringe)
    composited. Blocks whose model is not at least a half cube (plants, torches, rails, snow layers ...) are
    see-through for the fly's rays: a declared simplification. Leaves are opaque; their cut-out texels take 0.75 x
    the mean of the opaque ones (a stand-in for the leaves seen through the gaps)."""

    def __init__(self, public: Path = VIEWER_PUBLIC, version: str = MC_VERSION):
        self.public = Path(public)
        self.version = version
        self.states = json.loads((self.public / "blocksStates" / f"{version}.json").read_text(encoding="utf-8"))
        from PIL import Image
        self.atlas = np.asarray(Image.open(self.public / "textures" / f"{version}.png").convert("RGBA"))
        self.A = self.atlas.shape[0]
        self.cache: dict[str, np.ndarray | None] = {}

    def _cut(self, tex: dict) -> np.ndarray:
        u0 = int(round(tex["u"] * self.A)); v0 = int(round(tex["v"] * self.A))
        su = max(1, int(round(tex["su"] * self.A))); sv = max(1, int(round(tex["sv"] * self.A)))
        px = self.atlas[v0:v0 + sv, u0:u0 + su]
        if px.shape[:2] != (16, 16):                    # animated or odd-sized: resample to 16 x 16
            iy = (np.arange(16) * px.shape[0] / 16).astype(int); ix = (np.arange(16) * px.shape[1] / 16).astype(int)
            px = px[iy][:, ix]
        return px.astype(np.float32)

    def _face(self, faces: dict, names, tint, overlay_faces=None) -> np.ndarray | None:
        f = next((faces[n] for n in names if n in faces), None)
        if f is None:
            return None
        px = self._cut(f["texture"])
        rgb, a = px[..., :3], px[..., 3:] / 255.0
        if f.get("tintindex") is not None and tint is not None:
            rgb = rgb * np.asarray(tint, np.float32) / 255.0
        if (a < 0.5).any() and (a >= 0.5).any():         # cut-out texels (leaves): the mean of the opaque ones, a
            fill = 0.75 * rgb[a[..., 0] >= 0.5].mean(0)  # stand-in for the leaves seen through the gaps
            rgb = rgb * a + fill * (1 - a)
        if overlay_faces is not None:
            o = next((overlay_faces[n] for n in names if n in overlay_faces), None)
            if o is not None:
                opx = self._cut(o["texture"])
                orgb, oa = opx[..., :3], opx[..., 3:] / 255.0
                if o.get("tintindex") is not None and tint is not None:
                    orgb = orgb * np.asarray(tint, np.float32) / 255.0
                rgb = rgb * (1 - oa) + orgb * oa
        return srgb8_to_linear(np.clip(rgb, 0, 255))

    def _liquid(self, name: str) -> np.ndarray:
        tex_name, tint = LIQUIDS[name]
        from PIL import Image
        p = self.public / "textures" / self.version / "blocks" / f"{tex_name}.png"
        px = np.asarray(Image.open(p).convert("RGBA"))[:16, :16].astype(np.float32)   # first animation frame
        rgb = px[..., :3] * (np.asarray(tint, np.float32) / 255.0 if tint else 1.0)
        lin = srgb8_to_linear(rgb)
        return np.stack([lin, lin, lin], 0)

    def get(self, name: str) -> np.ndarray | None:
        """(3, 16, 16, 3) linear RGB [top, bottom, side], or None when the fly's rays pass through the block."""
        if name in self.cache:
            return self.cache[name]
        out = None
        if name in LIQUIDS:
            out = self._liquid(name)
        elif name in self.states and name not in ("air", "cave_air", "void_air"):
            m = _first_model(self.states[name])
            els = (m or {}).get("elements") or []
            big = [e for e in els if np.prod(np.subtract(e["to"], e["from"])) >= 0.5 * 16 ** 3]
            if big:
                tint = FIXED_FOLIAGE.get(name) or (FOLIAGE_TINT if name.endswith("leaves") else GRASS_TINT)
                faces = big[0]["faces"]
                overlay = big[1]["faces"] if len(big) > 1 else None
                top = self._face(faces, ("up",), tint, overlay)
                bottom = self._face(faces, ("down",), tint, overlay)
                side = self._face(faces, ("north", "south", "east", "west"), tint, overlay)
                sides = [t for t in (top, bottom, side) if t is not None]
                if sides:
                    top = top if top is not None else sides[0]
                    bottom = bottom if bottom is not None else sides[0]
                    side = side if side is not None else sides[0]
                    out = np.stack([top, bottom, side], 0)
        self.cache[name] = out
        return out


class TextureTable:
    """Palette indices -> a stacked texture tensor on the device. Row 0 is 'see-through' (air, plants)."""

    def __init__(self, textures: BlockTextures, device):
        self.textures, self.device = textures, torch.device(device)
        self.rows: dict[str, int] = {}
        self._stack = [np.zeros((3, 16, 16, 3), np.float32)]
        self.tensor = torch.zeros((1, 3, 16, 16, 3), device=self.device)

    def row(self, name: str) -> int:
        if name not in self.rows:
            tex = self.textures.get(name)
            if tex is None:
                self.rows[name] = 0
            else:
                self.rows[name] = len(self._stack)
                self._stack.append(tex)
                self.tensor = torch.as_tensor(np.stack(self._stack), device=self.device)
        return self.rows[name]

    def map_palette(self, names) -> np.ndarray:
        return np.asarray([self.row(n) for n in names], np.int64)


# ---------------------------------------------------------------------------------------------------- frames
def heading_from_mineflayer_yaw(yaw: float) -> float:
    """mineflayer yaw (rad, 0 = north, + counter-clockwise) -> heading psi (rad, CCW from east)."""
    return float(yaw) + np.pi / 2


def mineflayer_yaw_from_heading(psi: float) -> float:
    return float(psi) - np.pi / 2


def body_axes(psi: float):
    """(forward, left, up) of a level fly with heading psi, in the z-up frame (X east, Y north, Z up)."""
    c, s = np.cos(psi), np.sin(psi)
    return (c, s, 0.0), (-s, c, 0.0), (0.0, 0.0, 1.0)


def zup_to_mc(v):
    """(..., 3) z-up (X east, Y north, Z up) -> Minecraft (x east, y up, z south)."""
    v = np.asarray(v, np.float32)
    return np.stack([v[..., 0], v[..., 2], -v[..., 1]], -1)


def mc_to_zup(v):
    v = np.asarray(v, np.float32)
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], -1)


# ---------------------------------------------------------------------------------------------------- the sky
@dataclass
class Sky:
    """A vanilla-like daytime sky: zenith blue fading to a pale horizon, a sun disc (Minecraft's sun is a square;
    this one is a soft disc of the same 10 deg size) on the east-west arc at the world's time of day, and fog that
    blends distant blocks into the horizon colour. Linear RGB, then scaled by `light` (the declared exposure)."""
    zenith: tuple = (0.18, 0.36, 0.95)
    horizon: tuple = (0.62, 0.78, 1.00)
    sun_rgb: tuple = (6.0, 5.4, 4.2)
    sun_radius_deg: float = 5.0
    sun_dir_mc: tuple = (0.866, 0.5, 0.0)     # MC axes; default: 30 deg above the east horizon
    fog_start: float = 28.0                    # blocks
    fog_end: float = 60.0
    light: float = 0.42                        # radiance of a white texel in full sun

    @staticmethod
    def sun_direction(time_of_day: int) -> tuple:
        """Minecraft's celestial angle (sun rises in the east at time 0, noon at 6000), MC axes."""
        f = ((time_of_day / 24000.0) - 0.25) % 1.0
        f = f + ((1.0 - (np.cos(f * np.pi) + 1.0) / 2.0) - f) / 3.0      # vanilla's easing
        ang = 2 * np.pi * f
        # angle 0 = noon (zenith); at time 0 (angle 270 deg) the sun is on the east horizon (+x), at 12000 west
        return (float(-np.sin(ang)), float(np.cos(ang)), 0.0)


# ---------------------------------------------------------------------------------------------------- entities
# Mob colours as vertical bands (bottom fraction, top fraction, sRGB): zombie and phantom are the mean opaque texels of
# the entity textures prismarine-viewer draws (zombie front faces: legs, shirt, head; phantom: whole skin); the rest
# are rough palette values. Mobs are drawn as axis-aligned boxes of their hitbox size -- the PLAN's "mobs as boxes".
MOB_BANDS = {
    "zombie": [(0.0, 0.38, (75, 65, 151)), (0.38, 0.74, (12, 148, 158)), (0.74, 1.0, (77, 111, 60))],
    "phantom": [(0.0, 1.0, (83, 82, 98))],
    "creeper": [(0.0, 0.22, (0x3F, 0x8F, 0x33)), (0.22, 0.72, (0x5C, 0xB0, 0x4C)), (0.72, 1.0, (0x4B, 0xA0, 0x3E))],
    "skeleton": [(0.0, 1.0, (0xB8, 0xB8, 0xB8))],
    "spider": [(0.0, 1.0, (0x35, 0x2E, 0x28))],
    "cow": [(0.0, 0.45, (0x3A, 0x2C, 0x22)), (0.45, 1.0, (0x5B, 0x44, 0x33))],
    "sheep": [(0.0, 0.40, (0xC8, 0xAA, 0x90)), (0.40, 1.0, (0xE8, 0xE8, 0xE8))],
    "pig": [(0.0, 1.0, (0xF0, 0xA0, 0xA0))],
    "chicken": [(0.0, 1.0, (0xF2, 0xF2, 0xF2))],
    "player": [(0.0, 0.38, (0x46, 0x3A, 0xA5)), (0.38, 0.74, (0x00, 0xAF, 0xAF)), (0.74, 1.0, (0xB4, 0x84, 0x6D))],
}
DEFAULT_BAND = [(0.0, 1.0, (0x70, 0x60, 0x50))]


MAX_ENTITIES = 8          # entity boxes per cast (nearest first); empty slots sit far away and are never hit


def entity_arrays(entities, origin=(0, 0, 0), k_max: int = MAX_ENTITIES):
    """[{name, x, y, z, width, height}] (MC coords, feet at y) -> numpy (lo (K, 3), hi (K, 3), bands (K, 3, 3) linear
    RGB, cuts (K, 2)) relative to `origin`, padded to K = k_max; three colour bands per box (legs, body, head)."""
    lo = np.full((k_max, 3), 1e6, np.float32)
    hi = np.full((k_max, 3), 1e6 + 1, np.float32)
    cols = np.zeros((k_max, 3, 3), np.float32)
    cuts = np.ones((k_max, 2), np.float32)
    o = np.asarray(origin, np.float64)
    for i, e in enumerate(list(entities)[:k_max]):
        w, h = float(e.get("width") or 0.6), float(e.get("height") or 1.8)
        lo[i] = np.array((e["x"] - w / 2, e["y"], e["z"] - w / 2)) - o
        hi[i] = np.array((e["x"] + w / 2, e["y"] + h, e["z"] + w / 2)) - o
        bands = MOB_BANDS.get(e.get("name", ""), DEFAULT_BAND)
        b3 = (bands + [bands[-1]] * 3)[:3]
        cols[i] = srgb8_to_linear(np.asarray([bb[2] for bb in b3], np.float32))
        cuts[i] = (b3[0][1], b3[1][1] if len(bands) > 1 else 1.0)
    return lo, hi, cols, cuts


# ---------------------------------------------------------------------------------------------------- the caster
class VoxelCaster:
    """Cast rays from one eye point through a block region (and entity boxes) and shade what they hit.

    `_core` is pure tensor code with fixed shapes and no host synchronisation, so on CUDA the whole cast (about
    150 small kernels) is captured once as a CUDA graph and replayed with one launch per call (`graphs=True`,
    the default on CUDA). The region's grid is updated in place, which keeps the graph valid."""

    def __init__(self, table: TextureTable, sky: Sky | None = None, device="cpu", graphs: bool | None = None):
        self.table, self.sky, self.device = table, sky or Sky(), torch.device(device)
        self.grid = None            # (dx, dy, dz) int64 texture rows on the device
        self.origin = np.zeros(3, np.int64)
        self.graphs = (self.device.type == "cuda") if graphs is None else bool(graphs)
        self._graph = None
        self._static = None
        self.tex = table.tensor
        dev, sky = self.device, self.sky
        sun = np.asarray(sky.sun_dir_mc, np.float32)
        f32 = lambda v: torch.tensor(v, dtype=torch.float32, device=dev)  # noqa: E731
        self.k = {"zenith": f32(sky.zenith), "horizon": f32(sky.horizon), "sun": f32(sun / np.linalg.norm(sun)),
                  "sun_rgb": f32(sky.sun_rgb), "glow": f32((1.0, 0.9, 0.7)),
                  "far": f32(srgb8_to_linear((0x6E, 0x9A, 0x48))),
                  "shade": f32((FACE_SHADE["x"], FACE_SHADE["top"], FACE_SHADE["z"]))}   # indexed by hit axis x / y / z

    def set_region(self, origin, rows_grid: np.ndarray):
        """origin: MC block coords of grid[0, 0, 0]; rows_grid: (dx, dy, dz) texture rows (0 = see-through)."""
        self.origin = np.asarray(origin, np.int64)
        g = torch.as_tensor(np.ascontiguousarray(rows_grid), dtype=torch.int64, device=self.device)
        if self.grid is not None and self.grid.shape == g.shape:
            self.grid.copy_(g)
        else:
            self.grid = g
            self._graph = None

    def _core(self, o, d, lo, hi, cols, cuts, max_t: float = 80.0):
        dev = d.device
        M = d.shape[0]
        grid = self.grid
        dims = grid.shape
        flat_grid = grid.reshape(-1)
        strides = (dims[1] * dims[2], dims[2], 1)
        inf = torch.full((M,), float("inf"), device=dev)
        best_t = inf.clone()
        best_axis = torch.zeros(M, dtype=torch.int64, device=dev)
        best_row = torch.zeros(M, dtype=torch.int64, device=dev)
        for a in range(3):
            b, c = [i for i in range(3) if i != a]
            n = int(dims[a]) + 1
            fo = torch.floor(o[a])
            da = d[:, a]
            pos = (da > 0)[:, None]
            j = torch.arange(1, n + 1, device=dev, dtype=torch.float32)[None, :]
            plane = torch.where(pos, fo + j, fo - (j - 1))
            vox_a = torch.where(pos, plane, plane - 1).long()
            nz = da.abs() > 1e-9
            safe = torch.where(nz, da, torch.full_like(da, 1e-9))
            t = (plane - o[a]) / safe[:, None]
            ib = torch.floor(o[b] + d[:, b:b + 1] * t).long()
            ic = torch.floor(o[c] + d[:, c:c + 1] * t).long()
            inside = (nz[:, None] & (t > 0) & (t < max_t) & (vox_a >= 0) & (vox_a < dims[a])
                      & (ib >= 0) & (ib < dims[b]) & (ic >= 0) & (ic < dims[c]))
            flat = vox_a * strides[a] + ib * strides[b] + ic * strides[c]
            rows = flat_grid[torch.where(inside, flat, torch.zeros_like(flat))]
            solid = inside & (rows > 0)
            tt = torch.where(solid, t, torch.full_like(t, float("inf")))
            tmin, jmin = tt.min(1)
            better = tmin < best_t
            best_t = torch.where(better, tmin, best_t)
            best_axis = torch.where(better, torch.full_like(best_axis, a), best_axis)
            best_row = torch.where(better, rows.gather(1, jmin[:, None])[:, 0], best_row)
        hit = torch.isfinite(best_t)
        tb = torch.where(hit, best_t, torch.zeros_like(best_t))
        p = o[None, :] + d * tb[:, None]
        frac = p - torch.floor(p)
        ax = best_axis
        u = torch.where(ax == 0, frac[:, 2], frac[:, 0])
        v = torch.where(ax == 1, frac[:, 2], 1 - frac[:, 1])
        top = d[:, 1] < 0
        face = torch.where(ax == 1, torch.where(top, TOP, BOTTOM), SIDE)
        tu = (u * 16).long().clamp(0, 15)
        tv = (v * 16).long().clamp(0, 15)
        rgb = self.tex[best_row, face, tv, tu]
        shade = torch.where(ax == 1, torch.where(top, FACE_SHADE["top"], FACE_SHADE["bottom"]), self.k["shade"][ax])
        rgb = rgb * shade[:, None]
        # entity boxes (slab test), nearer than the block hit wins
        inv = 1.0 / torch.where(d.abs() > 1e-9, d, torch.full_like(d, 1e-9))
        t0 = (lo[None] - o[None, None]) * inv[:, None]
        t1 = (hi[None] - o[None, None]) * inv[:, None]
        tn = torch.minimum(t0, t1).amax(-1)
        tf = torch.maximum(t0, t1).amin(-1)
        ok = (tf >= tn) & (tf > 0)
        tn = torch.where(ok, tn.clamp_min(0), torch.full_like(tn, float("inf")))
        tk, kk = tn.min(1)
        ent_hit = torch.isfinite(tk)
        pe = o[None] + d * torch.where(ent_hit, tk, torch.zeros_like(tk))[:, None]
        lk, hk = lo[kk], hi[kk]
        h = (pe[:, 1] - lk[:, 1]) / (hk[:, 1] - lk[:, 1]).clamp_min(1e-6)
        band = (h > cuts[kk, 0]).long() + (h > cuts[kk, 1]).long()
        ec = cols[kk, band]
        dist_faces = torch.minimum((pe - lk).abs(), (pe - hk).abs())
        ent_axis = dist_faces.argmin(-1)
        es = torch.where(ent_axis == 1, 1.0, self.k["shade"][ent_axis])
        ent_rgb = ec * es[:, None]
        use_ent = ent_hit & (tk < best_t)
        t_final = torch.where(use_ent, tk, torch.where(hit, best_t, inf))
        surf = torch.where(use_ent[:, None], ent_rgb, rgb)
        any_hit = use_ent | hit
        # sky: zenith-to-horizon gradient, a glow and the sun disc
        up = d[:, 1].clamp(-1, 1)
        kz = up.clamp(0, 1).pow(0.5)[:, None]
        sky = self.k["horizon"] * (1 - kz) + self.k["zenith"] * kz
        ang = torch.rad2deg(torch.arccos((d * self.k["sun"]).sum(-1).clamp(-1, 1)))
        disc = torch.clamp(self.sky.sun_radius_deg - ang, 0, 1)[:, None]
        sky = sky + torch.exp(-ang / 12.0)[:, None] * 0.35 * self.k["glow"] + disc * self.k["sun_rgb"]
        below = (d[:, 1] < -0.02)[:, None]
        far = torch.where(below, self.k["far"].expand(M, 3), sky)         # below the horizon: distant ground
        fog = ((t_final - self.sky.fog_start) / (self.sky.fog_end - self.sky.fog_start)).clamp(0, 1)
        fog = torch.where(any_hit, fog, torch.ones_like(fog))[:, None]
        fogcol = torch.where(below, far, self.k["horizon"].expand(M, 3))
        col = torch.where(any_hit[:, None], surf * (1 - fog) + fogcol * fog, far) * self.sky.light
        rad = torch.stack([0.5 * col[:, 2], col[:, 2], col[:, 1], col[:, 0]], -1)
        return rad, t_final, use_ent, any_hit

    def cast(self, eye_mc, dirs_mc, entities=(), max_t: float = 80.0, graph: bool | None = None):
        """eye_mc: (3,) MC coords of the eye; dirs_mc: (M, 3) unit directions in MC axes; entities: up to
        MAX_ENTITIES boxes (the nearest first). Returns (M, 4) radiance [UV, B, G, R] (UV = 0.5 B, the declared RGB
        convention) and a dict of per-ray hit info. Outputs of a graph replay are reused buffers: copy to keep."""
        dev = self.device
        o = torch.as_tensor(np.asarray(eye_mc, np.float64) - self.origin, dtype=torch.float32, device=dev)
        d = torch.as_tensor(np.asarray(dirs_mc, np.float32), device=dev)
        arr = [torch.as_tensor(x, device=dev) for x in entity_arrays(entities, self.origin)]
        use_graph = self.graphs if graph is None else graph
        if not use_graph:
            self.tex = self.table.tensor
            rad, t, ent, hit = self._core(o, d, *arr, max_t=max_t)
            return rad, {"t": t, "entity": ent, "hit": hit}
        st = self._static
        if self._graph is None or st["d"].shape != d.shape or st["tex"] is not self.table.tensor:
            self.tex = self.table.tensor          # a new block type grows the table: recapture on the new tensor
            st = self._static = {"o": o.clone(), "d": d.clone(), "arr": [x.clone() for x in arr], "tex": self.tex}
            s = torch.cuda.Stream(device=dev)
            s.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._core(st["o"], st["d"], *st["arr"], max_t=max_t)
            torch.cuda.current_stream(dev).wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                st["out"] = self._core(st["o"], st["d"], *st["arr"], max_t=max_t)
            self._graph = g
        st["o"].copy_(o)
        st["d"].copy_(d)
        for dst, src in zip(st["arr"], arr):
            dst.copy_(src)
        self._graph.replay()
        rad, t, ent, hit = st["out"]
        return rad, {"t": t, "entity": ent, "hit": hit}



# ---------------------------------------------------------------------------------------------------- the server
REPO = HERE.parent
SERVER_DIR = REPO / "out" / "minecraft" / "server"
SERVER_JAR = f"server-{MC_VERSION}.jar"
SERVER_PORT, RCON_PORT, RCON_PASSWORD, VIEWER_PORT = 25631, 25632, "flyverse", 3031
BRIDGE_JS = HERE / "minecraft_bridge" / "bridge.js"
LEVEL_SEED = "flyverse"
SERVER_PROPERTIES = {
    "online-mode": "false", "enable-rcon": "true", "rcon.password": RCON_PASSWORD, "rcon.port": str(RCON_PORT),
    "server-port": str(SERVER_PORT), "server-ip": "127.0.0.1", "level-name": "world", "level-seed": LEVEL_SEED,
    "difficulty": "easy", "gamemode": "survival", "spawn-protection": "0", "view-distance": "8",
    "simulation-distance": "6", "max-players": "4", "motd": "flyverse games/minecraft", "allow-flight": "true",
    "generate-structures": "true", "sync-chunk-writes": "false", "spawn-monsters": "true",
    "broadcast-rcon-to-ops": "false", "broadcast-console-to-ops": "false", "log-ips": "false",
}


class Rcon:
    """Minimal Source-RCON client (the protocol the vanilla server speaks)."""

    def __init__(self, host="127.0.0.1", port=RCON_PORT, password=RCON_PASSWORD, timeout=10.0):
        self.s = socket.create_connection((host, port), timeout=timeout)
        self.i = 0
        if self._req(3, password)[0] == -1:
            raise RuntimeError("RCON authentication failed")

    def _recv(self, n):
        b = b""
        while len(b) < n:
            c = self.s.recv(n - len(b))
            if not c:
                raise ConnectionError("RCON connection closed")
            b += c
        return b

    def _req(self, typ, body):
        self.i += 1
        data = struct.pack("<ii", self.i, typ) + body.encode("utf-8") + b"\x00\x00"
        self.s.sendall(struct.pack("<i", len(data)) + data)
        n = struct.unpack("<i", self._recv(4))[0]
        p = self._recv(n)
        rid, _ = struct.unpack("<ii", p[:8])
        return rid, p[8:-2].decode("utf-8", "replace")

    def __call__(self, cmd: str) -> str:
        return self._req(2, cmd)[1]

    def gametime(self) -> int:
        return int(self("time query gametime").split()[-1])

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


def rcon_up(port=RCON_PORT) -> bool:
    try:
        Rcon(port=port, timeout=2.0).close()
        return True
    except (OSError, RuntimeError):
        return False


class Server:
    """The local offline-mode vanilla server under out/minecraft/server/ (git-ignored). Downloads the jar from
    Mojang's version manifest if it is missing (sha1-checked), writes eula=true (owner-approved) and
    server.properties, and starts java unless a server already answers on the RCON port. `fresh=True` deletes the
    world first, so every run starts from the same generated terrain (the level seed is fixed)."""

    def __init__(self, directory=SERVER_DIR, fresh=True, java="java", xmx="2G"):
        self.dir, self.fresh, self.java, self.xmx = Path(directory), fresh, java, xmx
        self.proc = None
        self._log = None

    def ensure_jar(self):
        jar = self.dir / SERVER_JAR
        if jar.exists():
            return jar
        import hashlib
        import urllib.request
        self.dir.mkdir(parents=True, exist_ok=True)
        man = json.loads(urllib.request.urlopen("https://piston-meta.mojang.com/mc/game/version_manifest_v2.json").read())
        url = next(v["url"] for v in man["versions"] if v["id"] == MC_VERSION)
        s = json.loads(urllib.request.urlopen(url).read())["downloads"]["server"]
        data = urllib.request.urlopen(s["url"]).read()
        if hashlib.sha1(data).hexdigest() != s["sha1"]:
            raise RuntimeError("server.jar sha1 mismatch")
        jar.write_bytes(data)
        return jar

    def start(self, timeout=180.0):
        if rcon_up():
            if self.fresh:
                raise RuntimeError(f"a server already answers on RCON port {RCON_PORT}; stop it or pass --reuse-server")
            return self
        jar = self.ensure_jar()
        (self.dir / "eula.txt").write_text("eula=true\n", encoding="utf-8")
        (self.dir / "server.properties").write_text("".join(f"{k}={v}\n" for k, v in SERVER_PROPERTIES.items()),
                                                    encoding="utf-8")
        if self.fresh:
            import shutil
            shutil.rmtree(self.dir / "world", ignore_errors=True)
        self._log = open(self.dir / "server_console.log", "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen([self.java, f"-Xmx{self.xmx}", "-Xms1G", "-jar", jar.name, "nogui"], cwd=self.dir,
                                     stdin=subprocess.PIPE, stdout=self._log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited early; see {self.dir / 'server_console.log'}")
            if rcon_up():
                return self
            time.sleep(0.5)
        raise RuntimeError("server did not come up")

    def stop(self):
        if self.proc is None:
            return
        try:
            r = Rcon(timeout=5.0)
            r("stop")
            r.close()
        except (OSError, RuntimeError):
            pass
        try:
            self.proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        if self._log:
            self._log.close()
        self.proc = None


class Bridge:
    """games/minecraft_bridge/bridge.js over stdio, one JSON object per line each way."""

    def __init__(self, node="node"):
        if not (HERE / "minecraft_bridge" / "node_modules").exists():
            raise SystemExit("run `npm install` in games/minecraft_bridge first")
        (REPO / "out" / "minecraft").mkdir(parents=True, exist_ok=True)
        self.errlog = open(REPO / "out" / "minecraft" / "bridge_stderr.log", "w", encoding="utf-8", errors="replace")
        self.p = subprocess.Popen([node, str(BRIDGE_JS)], cwd=HERE / "minecraft_bridge", stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.errlog, bufsize=1, text=True, encoding="utf-8")
        self.i = 0
        first = json.loads(self.p.stdout.readline())
        if first.get("result") != "ready":
            raise RuntimeError(f"bridge did not start: {first}")

    def __call__(self, cmd, **args):
        self.i += 1
        self.p.stdin.write(json.dumps({"id": self.i, "cmd": cmd, "args": args}) + "\n")
        self.p.stdin.flush()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"bridge exited during {cmd} (see out/minecraft/bridge_stderr.log)")
            msg = json.loads(line)
            if msg.get("id") == self.i:
                break
        if not msg.get("ok"):
            raise RuntimeError(f"bridge {cmd}: {msg.get('error')}")
        return msg["result"]

    def close(self):
        try:
            self("quit")
        except (RuntimeError, OSError, ValueError):
            pass
        try:
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.errlog.close()


def decode_region(reg: dict):
    """bridge `region` result -> (origin (3,), palette indices grid[x, y, z], palette names)."""
    import base64
    dx, dy, dz = reg["dims"]
    idx = np.frombuffer(base64.b64decode(reg["data"]), dtype=np.uint16).reshape(dy, dz, dx)
    return np.asarray(reg["origin"], np.int64), np.ascontiguousarray(idx.transpose(2, 0, 1)), list(reg["palette"])


# ---------------------------------------------------------------------------------------------------- the mapping
TICK_S = C.TICK_MS / 1000.0
BRAIN_TICKS_PER_MC = 5                 # 50 ms of brain per Minecraft tick (20 TPS)
EYE_HEIGHT = 1.62                      # Minecraft's standing eye height (m above the feet)
WALK_SCALE = 4.317 / 0.02              # DECODER: player m/s per fly m/s (2 cm/s brisk fly walk -> 4.317 m/s walk)
SPEED_PER_ATTR = 43.17                 # vanilla: steady ground speed (m/s) = 43.17 x movement_speed attribute
YAW_GAIN = 1.0                         # DECODER: player yaw rate = fly yaw rate (rad/s)
MOVE_DEADBAND = 0.05                   # m/s: below this the player's movement keys are released
REGION = (96, 64, 96)                  # blocks fetched around the player for the eye (x, y, z)
TIME_OF_DAY = 1000                     # Minecraft time (1000 = morning, the sun 28 deg up in the east)
START = (-296, 63, -294, 157.5)        # block x, y of the grass top, z, heading (deg, from +x towards +z) -- a flat lakeshore runway
DESPAWN_AFTER_HIT_S = 0.6
ENCOUNTER_TIMEOUT_S = 9.0
NEAR_M = 3.0                           # an encounter's "GF peak" is taken only while its mob is this close (see dist_to)
# entity types (minecraft-data 1.21.4) the eye draws as boxes: living mobs and players. 'other' (falling blocks,
# display entities, boats ...) and 'projectile' are left out; the viewer does not draw most of them either.
EYE_ENTITY_TYPES = {"hostile", "mob", "animal", "passive", "water_creature", "ambient", "player"}
CAM_RATE = np.deg2rad(90.0)            # GAME camera: the chase camera orbits the player at most this fast
CAM_MIN_DIST = 3.0                     # ... and never comes closer than this to the player's centre


def mc_heading_to_yaw(a_deg: float) -> float:
    """A direction (cos a, 0, sin a) in Minecraft's x-z plane -> mineflayer yaw (rad)."""
    a = np.deg2rad(a_deg)
    return float(np.arctan2(-np.cos(a), -np.sin(a)))


def yaw_forward(yaw: float) -> np.ndarray:
    """mineflayer yaw -> unit forward vector in MC axes."""
    return np.array([-np.sin(yaw), 0.0, -np.cos(yaw)])


def notch_yaw_deg(yaw: float) -> float:
    return float((180.0 - np.rad2deg(yaw)) % 360.0)


def player_command(v_fly: float) -> dict:
    """DECODER: the shipped body's walking speed (m/s, negative = backing up) -> Minecraft movement keys and the
    movement_speed attribute that makes the player's steady ground speed WALK_SCALE x the fly's."""
    v = WALK_SCALE * float(v_fly)
    return {"forward": v > MOVE_DEADBAND, "back": v < -MOVE_DEADBAND, "speed": abs(v) / SPEED_PER_ATTR, "v_player": v}


@dataclass
class Encounter:
    """GAME: one mob, scripted (the fixed SCRIPT) or summoned by a key in interactive mode. Bearing is relative to
    the player's heading at summon time (+ = left)."""
    t_s: float
    bearing_deg: float
    dist: float
    mob: str = "zombie"
    height: float = 0.0
    pass_gap: float = 0.10             # phantoms: eye-to-hitbox gap where the dive turns into a pass
    n: int = 0
    scripted: bool = True              # False: summoned by a key (interactive), not part of the fixed script
    c: np.ndarray | None = None        # phantoms: current hitbox centre (MC), stepped every 10 ms brain frame
    mode: str = ""                     # phantoms: dive / pass
    direction: np.ndarray | None = None
    pass_dir: np.ndarray | None = None
    pass_t: float | None = None
    entity_id: int | None = None
    summoned_at: float | None = None
    pos: tuple | None = None
    first_hit_t: float | None = None
    removed_at: float | None = None
    min_dist: float = float("inf")
    min_gap: float = float("inf")
    gf_max_window: float = 0.0         # GF max while this was the nearest live mob, at any distance
    gf_peak_near: float = 0.0          # GF max while it was the nearest live mob and within NEAR_M
    gf_peak_dive: float = 0.0          # phantoms: the same, split before / after the dive turned into the pass
    gf_peak_pass: float = 0.0
    gf_cross: list = field(default_factory=list)
    cross_at: dict | None = None       # the first crossing: GF, distance, gap, dive mode
    jumps: int = 0
    eye_inside_ticks: int = 0
    hits: int = 0
    outcome: str = ""


# GAME: (time s, mob, bearing deg (+ = left of the player's heading at that time), distance m, height m). Zombies walk
# under Minecraft's AI. The phantom is a scripted dive (dive_step): straight at the player's eye at PHANTOM_SPEED
# until its hitbox is PHANTOM_PASS_GAP from the eye, then a pass up and over the head.
SCRIPT = [(1.5, "zombie", 0.0, 12.0, 0.0), (7.5, "zombie", 35.0, 10.0, 0.0), (13.0, "zombie", -35.0, 10.0, 0.0),
          (19.0, "phantom", 0.0, 22.0, 11.0), (24.5, "phantom", 60.0, 20.0, 9.0)]
PHANTOM_SPEED = 10.0                   # m/s
PHANTOM_PASS_GAP = 0.10                # m between the eye and the phantom's hitbox where the dive turns into a pass
                                       # (the swoop reaches the head, as a phantom's attack does); the pass keeps at
                                       # least this gap, so the eye never enters the box
PHANTOM_BOX = (0.9, 0.5)               # hitbox width, height (vanilla, size 0)
PHANTOM_HALF = (PHANTOM_BOX[0] / 2, PHANTOM_BOX[1] / 2, PHANTOM_BOX[0] / 2)


def ground_y(grid_rows: np.ndarray, origin: np.ndarray, x: float, z: float, y_hint: float) -> float | None:
    """Feet height of the first walkable surface at (x, z): the top of the highest solid block under y_hint + 4."""
    ix, iz = int(math.floor(x)) - origin[0], int(math.floor(z)) - origin[2]
    if not (0 <= ix < grid_rows.shape[0] and 0 <= iz < grid_rows.shape[2]):
        return None
    top = min(grid_rows.shape[1] - 1, int(math.floor(y_hint)) - origin[1] + 4)
    for iy in range(top, -1, -1):
        if grid_rows[ix, iy, iz] > 0:
            return float(origin[1] + iy + 1)
    return None


def autojump_wanted(grid_rows, origin, feet, fwd, moving: bool) -> bool:
    """GAME (Minecraft's Auto-Jump option): walking into a 1-block step with room above it -> jump."""
    if not moving:
        return False
    p = np.asarray(feet, float) + 0.55 * np.asarray(fwd, float)
    ix, iy, iz = int(math.floor(p[0])) - origin[0], int(math.floor(feet[1] + 0.01)) - origin[1], int(math.floor(p[2])) - origin[2]
    if not (0 <= ix < grid_rows.shape[0] and 0 <= iz < grid_rows.shape[2] and 0 <= iy < grid_rows.shape[1] - 3):
        return False
    return bool(grid_rows[ix, iy, iz] > 0 and grid_rows[ix, iy + 1, iz] == 0 and grid_rows[ix, iy + 2, iz] == 0)


def fly_view_of(center_mc, eye_mc, psi, radius_m):
    """A point in Minecraft coordinates -> (azimuth deg (+ left), elevation deg (+ up), angular radius deg, distance)
    as seen from the eye of a level fly with heading psi."""
    v = mc_to_zup(np.asarray(center_mc, float) - np.asarray(eye_mc, float))
    f, left, up = (np.asarray(a, float) for a in body_axes(psi))
    x, y, z = float(v @ f), float(v @ left), float(v @ up)
    dist = max(math.sqrt(x * x + y * y + z * z), 1e-6)
    return (math.degrees(math.atan2(y, x)), math.degrees(math.asin(max(-1.0, min(1.0, z / dist)))),
            math.degrees(math.asin(min(1.0, radius_m / dist))), dist)


def mosaic_points(az_el, side, rect, az, el):
    """Where (az, el) falls on games.common.Hud.mosaic's layout of the two eyes (the same shift-and-scale it uses):
    a list of (x, y, scale px/deg), one per eye whose field could contain the direction."""
    a = np.asarray(az_el, np.float64)[:, 0].copy()
    e = np.asarray(az_el, np.float64)[:, 1]
    side = np.asarray(side)
    L, R = side == "L", side == "R"
    half = max(0.0, (a[R].max() - a[L].min()) / 2) + 3.0 if (L.any() and R.any()) else 0.0
    a2 = a.copy()
    a2[L] += half
    a2[R] -= half
    x0, y0, w, h = rect
    scale = min((w - 8) / max(np.ptp(a2), 1), (h - 8) / max(np.ptp(e), 1))
    cx0, cy0 = x0 + w / 2, y0 + h / 2
    out = []
    for eye_mask, shift in ((L, half), (R, -half)):
        if not eye_mask.any():
            continue
        lo, hi = a[eye_mask].min() - 5, a[eye_mask].max() + 5
        if not (lo <= az <= hi):
            continue
        x = -((az + shift) - (a2.max() + a2.min()) / 2) * scale + cx0
        y = -(el - (e.max() + e.min()) / 2) * scale + cy0
        out.append((x, y, scale))
    return out


def box_gap(c, half, p) -> float:
    """Distance from point p to the axis-aligned box with centre c and half-extents half (0 inside)."""
    d = np.abs(np.asarray(p, float) - np.asarray(c, float)) - np.asarray(half, float)
    return float(np.linalg.norm(np.maximum(d, 0.0)))


def inside_box(c, half, p) -> bool:
    return bool((np.abs(np.asarray(p, float) - np.asarray(c, float)) < np.asarray(half, float)).all())


def keep_clear(c, half, eye, gap):
    """GAME: move the box vertically (up if it is above the eye, which it is on a pass) just enough that the eye is at
    least `gap` from it: the phantom pulls up over a rising (jumping) head instead of swallowing it."""
    c = np.asarray(c, float)
    if box_gap(c, half, eye) >= gap:
        return c
    eye = np.asarray(eye, float)
    gh = float(np.linalg.norm(np.maximum(np.abs((eye - c)[[0, 2]]) - np.asarray(half, float)[[0, 2]], 0.0)))
    gv = math.sqrt(max(gap * gap - gh * gh, 0.0))
    c = c.copy()
    if c[1] >= eye[1]:
        c[1] = max(c[1], eye[1] + half[1] + gv)
    else:
        c[1] = min(c[1], eye[1] - half[1] - gv)
    return c


def dive_step(c, eye, mode, pass_dir, step, pass_gap, half=PHANTOM_HALF):
    """GAME: one step of the scripted phantom dive. `c` is the hitbox centre. While diving it moves `step` straight at
    the eye, but stops where its hitbox is `pass_gap` from the eye; in that step it turns into a pass: the horizontal
    direction to the eye + 0.8 up, normalised, kept from then on, lifted where needed so the eye stays `pass_gap`
    from the hitbox (`keep_clear`). Returns (c, mode, pass_dir, direction, turned)."""
    c = np.asarray(c, float)
    eye = np.asarray(eye, float)
    turned = False
    if mode == "dive":
        to = eye - c
        dist = float(np.linalg.norm(to))
        direction = to / max(dist, 1e-9)
        if box_gap(c + direction * step, half, eye) > pass_gap:
            return c + direction * step, mode, pass_dir, direction, turned
        # the pass point is within this step. Along the line to the eye the gap only shrinks (the eye moves radially
        # towards the box centre), so bisect for the advance that leaves exactly `pass_gap`.
        lo, hi = 0.0, step
        if box_gap(c, half, eye) <= pass_gap:
            hi = 0.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if box_gap(c + direction * mid, half, eye) > pass_gap:
                lo = mid
            else:
                hi = mid
        # (if the eye itself moved into the gap within this frame -- a jump rises 8 cm per frame -- the box is lifted
        # back out to the gap, as on the pass)
        c = keep_clear(c + direction * lo, half, eye, pass_gap)
        h = direction.copy()
        h[1] = 0.0
        h = h / max(float(np.linalg.norm(h)), 1e-6)
        v = h + np.array([0.0, 0.8, 0.0])
        pass_dir = v / np.linalg.norm(v)
        return c, "pass", pass_dir, direction, True
    c = keep_clear(c + np.asarray(pass_dir, float) * step, half, eye, pass_gap)
    return c, mode, pass_dir, np.asarray(pass_dir, float), turned


def orbit_step(phi, target, max_step, frac):
    """GAME camera: move an azimuth angle `phi` towards `target` (rad) the short way round, by `frac` of the gap but
    never more than `max_step`: an orbit around the player, never a straight line through it."""
    d = (target - phi + np.pi) % (2 * np.pi) - np.pi
    return phi + float(np.clip(d * frac, -max_step, max_step))


def lerp(a, b, f):
    return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * f


def _r(x, nd=3):
    """Round a float for the run log; None for None / inf / nan."""
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# ---------------------------------------------------------------------------------------------------- the game
class MinecraftGame(C.Game):
    title = "minecraft"
    subtitle = "the fly in real Minecraft 1.21.4"

    def __init__(self, args):
        super().__init__(args)
        self.server = self.bridge = self.rcon = None
        import atexit
        atexit.register(self._kill_children)
        try:
            self._setup(args)
        except BaseException:
            self.close(summary=False)
            raise

    def _kill_children(self):
        """Last resort at interpreter exit: no orphaned java / node / chrome processes."""
        for proc in (getattr(self.bridge, "p", None), getattr(self.server, "proc", None)):
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                except OSError:
                    pass

    def _build_brain(self, args, attempts=6, wait_s=60.0):
        """GPU etiquette on a shared card: on CUDA OOM wait and retry."""
        for k in range(attempts):
            try:
                return C.build_brain(args)
            except torch.OutOfMemoryError:
                if k == attempts - 1:
                    raise
                torch.cuda.empty_cache()
                print(f"  minecraft: CUDA out of memory building the brain; retry in {wait_s:.0f} s", flush=True)
                time.sleep(wait_s)

    def _setup(self, args):
        from flyverse.body import Flight, FlyState, Locomotion
        self.dev = C.pick_device(args.device)
        self.log.meta["sources"].update(C.source_hashes([BRIDGE_JS, HERE / "minecraft_bridge" / "web" / "fv.js",
                                                          HERE / "minecraft_bridge" / "package.json",
                                                          HERE / "minecraft_bridge" / "package-lock.json"]))
        self.log.meta["minecraft"] = {"version": MC_VERSION, "level_seed": LEVEL_SEED, "start": START,
                                      "time_of_day": TIME_OF_DAY, "script": SCRIPT, "control": args.control}
        # --- the server and the bot
        self.server = Server(fresh=not args.reuse_server)
        t0 = time.time()
        self.server.start()
        self.rcon = Rcon()
        for c in ["tick unfreeze", "gamerule doDaylightCycle false", f"time set {TIME_OF_DAY}", "weather clear 1000000",
                  "gamerule doWeatherCycle false", "gamerule doMobSpawning false", "gamerule naturalRegeneration false",
                  "gamerule doInsomnia false", "gamerule mobGriefing false", "gamerule doMobLoot false",
                  "gamerule spawnRadius 0", "gamerule doImmediateRespawn true", "gamerule sendCommandFeedback false",
                  "gamerule announceAdvancements false", "gamerule doTraderSpawning false", "gamerule doPatrolSpawning false",
                  "difficulty easy", "kill @e[type=!player,tag=fv]"]:
            self.rcon(c)
        self.bridge = Bridge()
        self.bridge("connect", username="FlyBot")
        x, y, z, head = START
        self.yaw0 = mc_heading_to_yaw(head)
        self.rcon("gamemode survival FlyBot")
        self.rcon(f"tp FlyBot {x + 0.5} {y + 1} {z + 0.5} {notch_yaw_deg(self.yaw0):.2f} 0")
        self.rcon(f"spawnpoint FlyBot {x} {y + 1} {z}")
        self.rcon("effect clear FlyBot")
        self.rcon("clear FlyBot")
        time.sleep(1.0)
        self.bridge("wait_chunks")
        time.sleep(1.5)
        # clear leftovers (a reused server keeps mobs in chunks that were unloaded until the bot arrived)
        for mob in ("zombie", "husk", "skeleton", "creeper", "spider", "witch", "enderman", "slime", "phantom", "item"):
            self.rcon(f"kill @e[type=minecraft:{mob}]")
        self.rcon("kill @e[tag=fv]")
        self.rcon("effect give FlyBot minecraft:instant_health 1 4 true")
        time.sleep(1.2)
        self.rcon("tick freeze")
        self.rcon(f"tp FlyBot {x + 0.5} {y + 1} {z + 0.5} {notch_yaw_deg(self.yaw0):.2f} 0")
        time.sleep(0.3)
        self.gt0 = self.rcon.gametime()
        st = self.bridge("state")
        self.log.meta["minecraft"]["setup_wall_s"] = round(time.time() - t0, 1)
        # --- the eyes' world
        self.textures = BlockTextures()
        self.table = TextureTable(self.textures, self.dev)
        self.sky = Sky(sun_dir_mc=Sky.sun_direction(TIME_OF_DAY))
        self.caster = VoxelCaster(self.table, self.sky, self.dev)
        self.rows_grid = None
        self.region_center = None
        self.P_prev = self.P_cur = np.asarray(st["pos"], float)
        self.vel = np.zeros(3)
        self.on_ground = True
        self.fetch_region()
        # --- the viewer (real Minecraft rendering) or our own render
        self.viewer_ok = False
        self.viewer_note = ""
        if not args.no_viewer:
            try:
                info = self.bridge("viewer_start", port=VIEWER_PORT, width=1536, height=864, viewDistance=8)
                self.bridge("viewer_setup", sunDir=list(self.sky.sun_dir_mc), zenith="#5b8cf0", horizon="#bcd6ff",
                            fogNear=64, fogFar=120)
                w = self.bridge("viewer_wait", timeoutMs=90000, settleMs=1500)
                self.viewer_ok = True
                self.log.meta["minecraft"]["viewer"] = {"gl": info.get("gl"), "wait": w}
            except RuntimeError as e:
                self.viewer_note = f"viewer failed: {e}"[:300]
                if args.record is not None:            # one recording per seed: never spend it on the fallback
                    raise RuntimeError(f"{self.viewer_note} -- refusing to record the fallback render; pass "
                                       "--no-viewer to record it on purpose") from e
                print(self.viewer_note, file=sys.stderr)
        self.log.meta["minecraft"]["footage"] = ("prismarine-viewer (three.js) in headless Chrome: real Minecraft blocks, "
                                                 "textures and entity models" if self.viewer_ok else
                                                 "fallback: our own render of the server's block data")
        # --- the brain
        self.fb = self._build_brain(args)
        self.eyes = C.Eyes(self.fb)
        sel = lambda **k: self.fb.c.select(**k)  # noqa: E731
        self.sel = {"LC4_L": sel(type="LC4", somaSide="L"), "LC4_R": sel(type="LC4", somaSide="R"),
                    "LPLC2_L": sel(type="LPLC2", somaSide="L"), "LPLC2_R": sel(type="LPLC2", somaSide="R")}
        self.loco, self.flight = Locomotion(), Flight()
        self.fly = FlyState()
        self.fly.ground_time = 0.0
        self.airborne = False
        self.jump_pending = None               # "gf" / "auto"
        # --- state
        self.psi = heading_from_mineflayer_yaw(self.yaw0)
        self.k10 = 0
        self.mc_k = 0
        self.v_acc = []
        self.cmd = player_command(0.0)
        self.E_prev, self.E_cur = {}, {}
        self.health = float(st.get("health", 20))
        script = json.loads(args.script_json) if args.script_json else SCRIPT
        self.log.meta["minecraft"]["script"] = script
        self.encounters = ([Encounter(row[0], row[2], row[3], mob=row[1], height=row[4],
                                      pass_gap=row[5] if len(row) > 5 else PHANTOM_PASS_GAP, n=i + 1)
                            for i, row in enumerate(script)] if not args.no_script else [])
        self.trace = {k: [] for k in ("t", "gf", "lpL", "lpR", "lc4L", "lc4R", "mdn", "v_fly", "v_player", "mob_dist",
                                      "mob_gap")}
        self.pending_ids = set()
        self.hidden_ids = set()
        self.rad = None
        self.hist = {k: [] for k in ("gf", "lc4L", "lc4R", "lpL", "lpR", "mdn", "v", "yaw")}
        self.banners = []                      # (t_s, text, color, kinds)
        self.walk_phase = 0.0
        self.ent_walk = {}
        self.geom = {}                         # encounter n -> (dist, gap, centre, (w, h)) at the current brain frame
        self.eye_names = set()                 # entity names that entered the fly's ray cast
        self.eye_inside = False
        self.cam_phi = None
        self.cam_lock = None                   # (encounter n, phi) while a phantom passes overhead
        self.cam_par = None                    # smoothed (back, side, up)
        self.cam_y = None
        self.cam_look = None
        self.frames = 0
        self.mdn_on = False
        self.gf_above = False
        self.deaths = 0
        self.autojumps = 0
        self.gf_jumps = 0
        self.voluntary_jumps = 0
        self.bumps = 0
        self.prev_collided = False
        self.path_len = 0.0
        self.max_gf = 0.0
        self.prof = {}
        self.declare_all()
        print(f"  minecraft: ready ({'viewer' if self.viewer_ok else 'own render'}), setup {time.time() - t0:.0f} s", flush=True)

    # ------------------------------------------------------------------------------------------------ declarations
    def declare_all(self):
        L = self.log
        C.declare(L, "eyes", "game", "the 1,466 x 7 ommatidial rays cast from the player's eye (feet + 1.62 m) through the "
                  "server's block grid; 1.21.4 atlas textures, vanilla face shading, sky + sun, living mobs and players "
                  "as hitbox boxes (the 8 nearest; a box that contains the eye is not drawn, as the vanilla client "
                  "culls it); RGB -> [0.5 B, B, G, R]", reads="bridge region + entities",
                  eye_height_m=EYE_HEIGHT, region_blocks=REGION, fog_blocks=(self.sky.fog_start, self.sky.fog_end),
                  light=self.sky.light, time_of_day=TIME_OF_DAY, plants="see-through (not drawn for the eye)",
                  entity_types=sorted(EYE_ENTITY_TYPES),
                  control=("the scripted mobs (zombies, phantoms) are left out of the cast; other entities stay"
                           if self.args.control == "blind-to-mobs" else "none"))
        C.declare(L, "walk", "decoder", "player ground speed = 215.85 x the shipped body's walking speed, which is "
                  "Locomotion.readout (+ 80 ms smoothing) = a constant 0.8 cm/s intrinsic drive (baseline_speed, about "
                  "90% of the walk: 1.73 m/s of the player's) + forward-DN, leg-MN and MDN-backing terms; sign -> "
                  "forward / back keys",
                  reads="fb.motor(): fwd_dn, leg_L/R, back_dn (MDN above 15 Hz)", walk_scale=WALK_SCALE,
                  baseline_speed_mps=self.loco.baseline_speed, k_fwd=self.loco.k_fwd, k_leg=self.loco.k_leg,
                  k_back=self.loco.k_back, speed_per_attribute=SPEED_PER_ATTR, deadband_mps=MOVE_DEADBAND)
        C.declare(L, "turn", "decoder", "player yaw rate = the shipped body's yaw rate, 1:1 rad/s (DNa02 L-R, leg MN L-R)",
                  reads="fb.motor(): turn_L/R, leg_L/R", gain=YAW_GAIN)
        C.declare(L, "escape_jump", "decoder", "body model (flyverse/body.py Flight.maybe_takeoff): giant fibre DNp01 >= 33 Hz, "
                  "1 s refractory after landing -> one Minecraft jump (the jump is Minecraft's physics); the same call's "
                  "voluntary branch (power MNs >= 50 Hz for 0.3 s) would also jump and is logged as 'voluntary'",
                  reads="fb.motor().gf (DNp01 mean rate), fb.motor().power", gf_hz=self.flight.gf_hz,
                  landing_refractory_s=self.flight.landing_refractory_s)
        C.declare(L, "zombies", "game", "scripted: zombies summoned at t, bearing, distance (same for every seed), walking "
                  "under Minecraft's own AI; removed 0.6 s after its first hit or at min(9 s, next summon - 0.4 s); a leather helmet "
                  "(not drawn by the viewer) keeps it from burning in daylight", script=SCRIPT)
        C.declare(L, "phantom", "game", "scripted dive (dive_step), stepped every 10 ms brain frame against the eye's "
                  "position in that frame: straight at the player's eye at 10 m/s until its hitbox is 0.10 m from the "
                  "eye, then a pass up and over the head (horizontal + 0.8 up), lifted where needed so the eye stays "
                  "0.10 m from the hitbox; the fly's eye and the footage take its position from the script, and a NoAI "
                  "phantom (fire resistance, never attacks) is /tp'd there every Minecraft tick",
                  speed_mps=PHANTOM_SPEED, pass_gap_m=PHANTOM_PASS_GAP, hitbox=PHANTOM_BOX)
        C.declare(L, "auto_jump", "game", "Minecraft's Auto-Jump option: walking into a 1-block step with 2 blocks of air "
                  "above it -> jump")
        C.declare(L, "lockstep", "game", "server frozen; per 50 ms of brain: one prismarine-physics tick of the bot, "
                  "/tick step 1, chat-marker sync; the eye interpolates the player between its last two ticks and mobs "
                  "between the last two server ticks", brain_ticks_per_mc_tick=BRAIN_TICKS_PER_MC)
        C.declare(L, "rules", "game", "difficulty easy; natural regeneration, mob spawning, weather and daylight cycle "
                  "off; the player is healed to full whenever a scripted mob is summoned")
        C.declare(L, "camera", "game", "chase camera for the footage only (the fly never sees it): orbits the player at "
                  "most 90 deg/s towards the nearest scripted mob, holds its bearing while a phantom passes overhead, "
                  "stays >= 3 m from the player", max_rate_deg_s=90.0, min_dist_m=CAM_MIN_DIST)

    # ------------------------------------------------------------------------------------------------ world
    def fetch_region(self):
        dx, dy, dz = REGION
        c = np.floor(self.P_cur).astype(int)
        reg = self.bridge("region", x0=int(c[0] - dx // 2), y0=int(c[1] - 24), z0=int(c[2] - dz // 2), dx=dx, dy=dy, dz=dz)
        origin, idx, pal = decode_region(reg)
        rows = self.table.map_palette(pal)
        self.rows_grid = rows[idx]
        self.region_origin = origin
        self.caster.set_region(origin, self.rows_grid)
        self.region_center = c.astype(float)

    def maybe_refetch(self):
        d = self.P_cur - self.region_center
        if abs(d[0]) > 20 or abs(d[2]) > 20 or abs(d[1]) > 10:
            self.fetch_region()

    def live_phantoms(self):
        return [e for e in self.encounters if e.mob == "phantom" and e.c is not None and e.removed_at is None]

    def ents_now(self, f):
        """Entities as the vanilla client would draw them this brain frame: mobs interpolated between the last two
        server ticks; scripted phantoms at their scripted position (the server's copy is /tp'd there each tick)."""
        out = []
        ph_ids = {e.entity_id for e in self.live_phantoms()}
        for eid, e in self.E_cur.items():
            if eid in self.hidden_ids or eid in ph_ids:
                continue
            p = lerp(self.E_prev[eid]["pos"], e["pos"], f) if eid in self.E_prev else np.asarray(e["pos"], float)
            out.append({"id": eid, "name": e["name"], "x": p[0], "y": p[1], "z": p[2], "width": e.get("width") or 0.6,
                        "height": e.get("height") or 1.8, "yaw": e.get("yaw"), "pitch": e.get("pitch"),
                        "type": e.get("type")})
        for enc in self.live_phantoms():
            d = enc.direction if enc.direction is not None else np.array([1.0, 0.0, 0.0])
            yaw = mc_heading_to_yaw(np.rad2deg(np.arctan2(d[2], d[0])))
            out.append({"id": enc.entity_id if enc.entity_id is not None else -enc.n, "name": "phantom",
                        "x": enc.c[0], "y": enc.c[1] - PHANTOM_HALF[1], "z": enc.c[2], "width": PHANTOM_BOX[0],
                        "height": PHANTOM_BOX[1], "yaw": yaw, "pitch": float(np.arcsin(np.clip(d[1], -1, 1))),
                        "type": "mob", "enc": enc.n})
        return out

    def scripted_ids(self):
        return {e.entity_id for e in self.encounters if e.entity_id is not None} | {-e.n for e in self.encounters}

    def eye_entities(self, ents, eye):
        """The boxes the fly's rays can hit: living mobs and players (EYE_ENTITY_TYPES), nearest first. A box that
        contains the eye is dropped, as the vanilla client culls a model the camera is inside (logged if it
        happens). Control arm: the scripted mobs are left out."""
        out = [e for e in ents if e.get("type") in EYE_ENTITY_TYPES]
        if self.args.control == "blind-to-mobs":
            ids = self.scripted_ids()
            out = [e for e in out if e["id"] not in ids]
        inside = []
        for e in out:
            w, h = float(e.get("width") or 0.6), float(e.get("height") or 1.8)
            if inside_box((e["x"], e["y"] + h / 2, e["z"]), (w / 2, h / 2, w / 2), eye):
                inside.append(e)
        out = [e for e in out if e not in inside]
        out.sort(key=lambda e: (e["x"] - eye[0]) ** 2 + (e["y"] - eye[1]) ** 2 + (e["z"] - eye[2]) ** 2)
        return out[:MAX_ENTITIES], inside

    def eye_radiance(self, eye, psi, eye_ents):
        f, left, up = body_axes(psi)
        dirs = zup_to_mc(self.eyes.world_dirs(f, left, up))
        self.eye_names.update(e["name"] for e in eye_ents)
        rad, info = self.caster.cast(np.asarray(eye), dirs, eye_ents)
        return self.eyes.pool(rad)

    # ------------------------------------------------------------------------------------------------ minecraft tick
    def mc_begin(self):
        v_fly = float(np.mean(self.v_acc)) if self.v_acc else float(self.fly.speed)
        self.v_acc = []
        self.cmd = player_command(v_fly)
        jump = self.jump_pending is not None
        st = self.bridge("step", yaw=mineflayer_yaw_from_heading(self.psi), pitch=0.0, forward=self.cmd["forward"],
                         back=self.cmd["back"], jump=jump, speed=self.cmd["speed"])
        self.P_prev, self.P_cur = self.P_cur, np.asarray(st["pos"], float)
        self.vel = np.asarray(st["vel"], float)
        was_ground = self.on_ground
        self.on_ground = bool(st["onGround"])
        step = np.linalg.norm((self.P_cur - self.P_prev)[[0, 2]])
        self.path_len += step
        self.walk_phase += step * 2.2
        if jump and not self.on_ground:
            src = self.jump_pending
            self.jump_pending = None
            self.airborne = True
            self.log.event(self.t_s, "jump", source=src, pos=self.P_cur.round(2).tolist())
        if self.airborne and self.on_ground and not was_ground:
            self.airborne = False
            self.fly.airborne = False
            self.fly.ground_time = 0.0
            self.log.event(self.t_s, "land", pos=self.P_cur.round(2).tolist())
        if st.get("collidedH") and not self.prev_collided and (self.cmd["forward"] or self.cmd["back"]):
            self.bumps += 1
            self.log.event(self.t_s, "bump", pos=self.P_cur.round(2).tolist())
        self.prev_collided = bool(st.get("collidedH"))
        self.rcon("tick step 1")
        self.mc_k += 1

    def mc_end(self):
        k = self.mc_k
        t0 = time.time()
        while self.rcon.gametime() < self.gt0 + k:
            if time.time() - t0 > 5:
                raise RuntimeError("server did not step")
            time.sleep(0.002)
        self.rcon(f'tellraw FlyBot "#fv {k}"')
        snap = self.bridge("sync", k=str(k))
        self.E_prev = self.E_cur
        self.E_cur = {e["id"]: e for e in snap["entities"]}
        for eid, e in self.E_cur.items():
            if eid in self.E_prev:
                d = np.linalg.norm(np.subtract(e["pos"], self.E_prev[eid]["pos"])[[0, 2]])
                self.ent_walk[eid] = self.ent_walk.get(eid, 0.0) + d * 2.2
        b = snap["bot"]
        h = float(b.get("health", self.health))
        for ev in snap.get("hurt", []):
            if ev.get("death"):
                self.deaths += 1
                self.log.event(self.t_s, "death")
                self.banner("THE FLY DIED  (zombie)", C.RED, "game")
        if h < self.health - 1e-6:
            enc = self.nearest_encounter()
            self.log.event(self.t_s, "hit", health=h, damage=round(self.health - h, 2),
                           encounter=enc.n if enc else None)
            self.banner(f"HIT  -{self.health - h:.1f} HP", C.RED, "game")
            if enc is not None:
                enc.hits += 1
                if enc.first_hit_t is None:
                    enc.first_hit_t = self.t_s
        self.health = h
        if not b.get("alive", True):
            self.health = 0.0
        self.script_update()
        self.maybe_refetch()
        if self.mc_k % 20 == 0:
            self.summary_quiet()
        fwd = yaw_forward(mineflayer_yaw_from_heading(self.psi))
        if (self.jump_pending is None and not self.airborne and self.on_ground
                and autojump_wanted(self.rows_grid, self.region_origin, self.P_cur, fwd, self.cmd["forward"])):
            self.jump_pending = "auto"
            self.autojumps += 1
            self.banner("AUTO-JUMP  (Minecraft option)", C.AMBER, "game")

    # ------------------------------------------------------------------------------------------------ the script
    def live_encounters(self):
        return [e for e in self.encounters if e.summoned_at is not None and e.removed_at is None
                and (e.entity_id is not None or e.c is not None)]

    def nearest_encounter(self):
        """The live mob whose hitbox is nearest the eye (this brain frame's geometry, `self.geom`)."""
        live = self.live_encounters()
        if not live:
            return None
        return min(live, key=lambda e: self.geom.get(e.n, (np.inf, np.inf))[1])

    def dist_to(self, enc):
        """Zombies: horizontal centre-to-centre distance to the player. Phantom: its hitbox centre to the eye."""
        g = self.geom.get(enc.n)
        return g[0] if g is not None else float("inf")

    def gap_to(self, enc):
        """The eye-to-hitbox distance (0 = the eye is inside the box)."""
        g = self.geom.get(enc.n)
        return g[1] if g is not None else float("inf")

    def update_geometry(self, ents, feet, eye):
        """This frame's distance and eye-to-hitbox gap of every live encounter, from the same entity positions the
        eye is cast against."""
        by_id = {e["id"]: e for e in ents}
        self.geom = {}
        for enc in self.live_encounters():
            e = by_id.get(enc.entity_id)
            if enc.mob == "phantom":
                c, w, h = enc.c, PHANTOM_BOX[0], PHANTOM_BOX[1]
                dist = float(np.linalg.norm(c - eye))
            elif e is not None:
                w, h = float(e.get("width") or 0.6), float(e.get("height") or 1.95)
                c = np.array([e["x"], e["y"] + h / 2, e["z"]])
                dist = float(np.linalg.norm((c - feet)[[0, 2]]))
            else:
                continue
            self.geom[enc.n] = (dist, box_gap(c, (w / 2, h / 2, w / 2), eye), c, (w, h))

    def summon(self, enc):
        a = self.psi + np.deg2rad(enc.bearing_deg)
        # z-up heading -> MC x/z
        dx, dz = np.cos(a), -np.sin(a)
        x, z = self.P_cur[0] + enc.dist * dx, self.P_cur[2] + enc.dist * dz
        face = notch_yaw_deg(mc_heading_to_yaw(np.rad2deg(np.arctan2(-dz, -dx))))
        if enc.mob == "phantom":
            y = self.P_cur[1] + EYE_HEIGHT + enc.height
            nbt = ('{Tags:["fv","fv%d"],NoAI:1b,NoGravity:1b,Silent:1b,PersistenceRequired:1b,Size:0,'
                   'Rotation:[%.1ff,30f]}' % (enc.n, face))
            enc.c, enc.mode = np.array([x, y + PHANTOM_HALF[1], z]), "dive"
            eye = self.P_cur + [0, EYE_HEIGHT, 0]
            enc.direction = (eye - enc.c) / np.linalg.norm(eye - enc.c)
        else:
            y = ground_y(self.rows_grid, self.region_origin, x, z, self.P_cur[1] + 3)
            y = self.P_cur[1] if y is None else y
            nbt = ('{Tags:["fv","fv%d"],PersistenceRequired:1b,CanPickUpLoot:0b,IsBaby:0b,'
                   'ArmorItems:[{},{},{},{id:"minecraft:leather_helmet",count:1}],ArmorDropChances:[0f,0f,0f,0f],'
                   'Rotation:[%.1ff,0f]}' % (enc.n, face))
        out = self.rcon(f"summon {enc.mob} {x:.2f} {y:.2f} {z:.2f} {nbt}")
        if enc.mob == "phantom":
            self.rcon(f"effect give @e[tag=fv{enc.n}] minecraft:fire_resistance infinite 0 true")
        if self.health < 20:
            self.rcon("effect give FlyBot minecraft:instant_health 1 4 true")    # GAME: full health for each encounter
            self.log.event(self.t_s, "heal", health_before=self.health)
        enc.summoned_at, enc.pos = self.t_s, (round(x, 2), round(y, 2), round(z, 2))
        self.pending_ids.add(enc.n)
        self.log.event(self.t_s, "summon", encounter=enc.n, mob=enc.mob, bearing_deg=enc.bearing_deg, dist=enc.dist,
                       pos=enc.pos, scripted=enc.scripted, rcon=out[:80])
        side = "AHEAD" if abs(enc.bearing_deg) < 20 else ("FROM THE LEFT" if enc.bearing_deg > 0 else "FROM THE RIGHT")
        if abs(enc.bearing_deg) > 150:
            side = "FROM BEHIND"
        if enc.mob == "phantom":
            self.banner(f"PHANTOM DIVING  ({enc.dist:.0f} m out, {enc.height:.0f} m up)", C.AMBER, "game")
        else:
            self.banner(f"ZOMBIE #{enc.n} {side}  ({enc.dist:.0f} m)", C.AMBER, "game")

    def script_update(self):
        for enc in self.encounters:
            if enc.scripted and enc.summoned_at is None and self.t_s + 1e-9 >= enc.t_s:
                self.summon(enc)
            if enc.summoned_at is not None and enc.entity_id is None and enc.removed_at is None:
                taken = {x.entity_id for x in self.encounters} | self.hidden_ids
                cands = [(float(np.linalg.norm(np.subtract(e["pos"], enc.pos))), eid) for eid, e in self.E_cur.items()
                         if e["name"] == enc.mob and eid not in taken]
                cands = [c for c in cands if c[0] < 4.0]      # the entity at the summon point, not a stranger
                if cands:
                    enc.entity_id = min(cands)[1]
            if enc.mob == "phantom" and enc.c is not None and enc.removed_at is None and enc.entity_id is not None:
                # the server's copy follows the script (the eye and the footage use the scripted position itself)
                d = enc.direction
                yaw = notch_yaw_deg(mc_heading_to_yaw(np.rad2deg(np.arctan2(d[2], d[0]))))
                pitch = float(np.rad2deg(-np.arcsin(np.clip(d[1], -1, 1))))
                p = enc.c - [0, PHANTOM_HALF[1], 0]
                self.rcon(f"tp @e[tag=fv{enc.n}] {p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {yaw:.1f} {pitch:.1f}")
            if enc.summoned_at is not None and enc.removed_at is None and (enc.entity_id is not None or enc.c is not None):
                done = (enc.first_hit_t is not None and self.t_s - enc.first_hit_t >= DESPAWN_AFTER_HIT_S) or \
                       (self.t_s - enc.summoned_at >= self.timeout_for(enc) - 1e-9) or \
                       (enc.pass_t is not None and self.t_s - enc.pass_t >= 1.5 - 1e-9)
                if done:
                    self.remove(enc)

    def remove(self, enc):
        self.rcon(f"kill @e[tag=fv{enc.n}]")
        if enc.entity_id is not None:
            self.hidden_ids.add(enc.entity_id)
        enc.removed_at = self.t_s
        enc.outcome = ("jumped" if enc.jumps else ("GF crossed, no jump" if enc.gf_cross else "no jump")) + \
            (f", hit x{enc.hits}" if enc.hits else ", not hit")
        name = f"ZOMBIE #{enc.n}" if enc.mob == "zombie" else "PHANTOM"
        if enc.jumps and enc.cross_at:
            self.banner(f"{name}: GF {enc.cross_at['hz']:.0f} Hz AT {enc.cross_at['gap']:.1f} m · JUMPED", C.SAGE,
                        ("connectome", "decoder"))
        else:
            self.banner(f"{name}: GF PEAK {enc.gf_peak_near:.0f} Hz WITHIN {NEAR_M:.0f} m · NO JUMP", C.LILAC,
                        ("connectome", "decoder"))
        self.log.event(self.t_s, "despawn", encounter=enc.n, min_dist=_r(enc.min_dist), min_gap=_r(enc.min_gap),
                       gf_peak_near=round(enc.gf_peak_near, 1), gf_max_window=round(enc.gf_max_window, 1),
                       gf_crossings=enc.gf_cross, jumps=enc.jumps, hits=enc.hits, eye_inside_frames=enc.eye_inside_ticks)

    def timeout_for(self, enc):
        """A scripted mob leaves 0.4 s before the next scripted summon (or after ENCOUNTER_TIMEOUT_S); a mob summoned
        by a key gets the full ENCOUNTER_TIMEOUT_S."""
        if not enc.scripted:
            return ENCOUNTER_TIMEOUT_S
        later = [e.t_s for e in self.encounters if e.scripted and e.t_s > enc.t_s]
        return min(ENCOUNTER_TIMEOUT_S, (min(later) - enc.t_s - 0.4) if later else ENCOUNTER_TIMEOUT_S)

    def phantom_frame(self, eye):
        """GAME: one 10 ms brain frame of every live scripted dive (`dive_step`) against the eye's position in this
        frame, so the geometry the fly sees is exactly the script's."""
        step = PHANTOM_SPEED * TICK_S
        for enc in self.live_phantoms():
            enc.c, enc.mode, enc.pass_dir, enc.direction, turned = dive_step(enc.c, eye, enc.mode, enc.pass_dir, step,
                                                                               enc.pass_gap)
            if turned:
                enc.pass_t = self.t_s
                self.log.event(self.t_s, "phantom_pass", encounter=enc.n,
                               eye_dist=round(float(np.linalg.norm(eye - enc.c)), 3),
                               eye_gap=round(box_gap(enc.c, PHANTOM_HALF, eye), 3))

    # ------------------------------------------------------------------------------------------------ the brain tick
    def tick(self):
        j = self.k10 % BRAIN_TICKS_PER_MC
        t0 = time.perf_counter()
        if j == 0:
            self.mc_begin()
            t0 = self._t("mc_begin", t0)
        f = (j + 1) / BRAIN_TICKS_PER_MC
        feet = lerp(self.P_prev, self.P_cur, f)
        eye = feet + [0, EYE_HEIGHT, 0]
        self.phantom_frame(eye)
        ents = self.ents_now(f)
        self.update_geometry(ents, feet, eye)
        eye_ents, inside = self.eye_entities(ents, eye)
        self.note_eye_inside(inside)
        self.rad = self.eye_radiance(eye, self.psi, eye_ents)
        t0 = self._t("eye", t0)
        self.fb.vision(self.rad)
        self.fb.step(C.TICK_MS)
        t0 = self._t("brain", t0)
        self.k10 += 1
        self.t_s = round(self.k10 * TICK_S, 9)         # from the integer tick count: no drift against the script
        m = self.fb.motor()
        cmd = self.loco.readout(m, dt_s=TICK_S)
        self.loco.step(self.fly, cmd, TICK_S, bounds=(-1e9, 1e9, -1e9, 1e9))
        self.psi += YAW_GAIN * self.fly.yaw_rate * TICK_S
        self.v_acc.append(self.fly.speed)
        br = self.fb.brain
        gf = float(m.gf)
        vals = {"gf": gf, "lc4L": br.mean_rate(self.sel["LC4_L"]), "lc4R": br.mean_rate(self.sel["LC4_R"]),
                "lpL": br.mean_rate(self.sel["LPLC2_L"]), "lpR": br.mean_rate(self.sel["LPLC2_R"]),
                "mdn": float(m.back_dn), "v": self.fly.speed, "yaw": self.fly.yaw_rate}
        for k, v in vals.items():
            self.hist[k].append(float(v))
        self.max_gf = max(self.max_gf, gf)
        enc = self.nearest_encounter()
        d_enc = self.dist_to(enc) if enc else None
        g_enc = self.gap_to(enc) if enc else None
        where = {"encounter": enc.n if enc else None, "mob": enc.mob if enc else None, "mob_dist": _r(d_enc),
                 "mob_gap": _r(g_enc)}
        for e in self.live_encounters():
            if e.n in self.geom:
                e.min_dist, e.min_gap = min(e.min_dist, self.geom[e.n][0]), min(e.min_gap, self.geom[e.n][1])
        if enc is not None:
            enc.gf_max_window = max(enc.gf_max_window, gf)
            if d_enc <= NEAR_M:
                enc.gf_peak_near = max(enc.gf_peak_near, gf)
                if enc.mob == "phantom":
                    if enc.mode == "dive":
                        enc.gf_peak_dive = max(enc.gf_peak_dive, gf)
                    else:
                        enc.gf_peak_pass = max(enc.gf_peak_pass, gf)
        # giant fibre: the shipped body threshold decides the escape jump
        crossed = gf >= self.flight.gf_hz
        if crossed and not self.gf_above:
            self.log.event(self.t_s, "gf_cross", hz=round(gf, 1), phase=enc.mode if enc else None, **where)
            if enc is not None:
                enc.gf_cross.append(round(self.t_s, 3))
                if enc.cross_at is None:
                    enc.cross_at = {"t_s": round(self.t_s, 3), "hz": round(gf, 1), "dist": _r(d_enc), "gap": _r(g_enc),
                                    "phase": enc.mode or None}
        self.gf_above = crossed
        if not self.airborne and self.jump_pending is None:
            w = self.flight.readout(m)
            if self.flight.maybe_takeoff(self.fly, w, TICK_S):
                if w["gf"] >= w.get("gf_threshold", self.flight.gf_hz):
                    self.jump_pending = "gf"
                    self.gf_jumps += 1
                    if enc is not None:
                        enc.jumps += 1
                    self.log.event(self.t_s, "escape", gf_hz=round(gf, 1), **where)
                    self.banner(f"GIANT FIBRE {round(gf, 1):.0f} Hz  ->  JUMP", C.SAGE, ("connectome", "decoder"))
                else:                                  # the body's voluntary-takeoff branch (power MNs), not the GF
                    self.jump_pending = "voluntary"
                    self.voluntary_jumps += 1
                    self.log.event(self.t_s, "voluntary_takeoff", power_hz=round(float(w["power"]), 1), gf_hz=round(gf, 1),
                                   **where)
                    self.banner(f"WING POWER MNs {float(w['power']):.0f} Hz  ->  TAKEOFF JUMP", C.LILAC,
                                ("connectome", "decoder"))
        # MDN (moonwalker) above the body's threshold -> the shipped readout backs the fly up
        mdn_on = float(m.back_dn) > self.loco.mdn_threshold + 10 and self.fly.speed < 0
        if mdn_on and not self.mdn_on:
            self.log.event(self.t_s, "backing_up", mdn_hz=round(float(m.back_dn), 1), **where)
            self.banner(f"MOONWALKER DN {float(m.back_dn):.0f} Hz  ->  BACKING UP", C.LILAC, ("connectome", "decoder"))
        self.mdn_on = mdn_on if mdn_on else (self.mdn_on and self.fly.speed < 0)
        if self.k10 % 2 == 0:
            tr = self.trace
            tr["t"].append(round(self.t_s, 3))
            for key in ("gf", "lpL", "lpR", "lc4L", "lc4R", "mdn"):
                tr[key].append(round(vals[key], 2))
            tr["v_fly"].append(round(self.fly.speed * 100, 4))
            tr["v_player"].append(round(self.cmd["v_player"], 3))
            tr["mob_dist"].append(_r(d_enc, 2))
            tr["mob_gap"].append(_r(g_enc, 2))
        t0 = self._t("readout", t0)
        if j == BRAIN_TICKS_PER_MC - 1:
            self.mc_end()
            self._t("mc_end", t0)

    def note_eye_inside(self, inside):
        """Log each episode of the eye inside a mob's hitbox (the box is culled for the eye while it lasts)."""
        now = bool(inside)
        if now:
            ids = {e["id"] for e in inside}
            for enc in self.encounters:
                if enc.entity_id in ids or -enc.n in ids:
                    enc.eye_inside_ticks += 1
        if now and not self.eye_inside:
            self.log.event(self.t_s, "eye_in_mob_box", mobs=[e["name"] for e in inside],
                           encounters=[enc.n for enc in self.encounters
                                       if enc.entity_id in {e["id"] for e in inside} or -enc.n in {e["id"] for e in inside}])
        self.eye_inside = now

    def _t(self, key, t0):
        if self.args.profile:
            if str(self.dev).startswith("cuda"):
                torch.cuda.synchronize()
            self.prof[key] = self.prof.get(key, 0.0) + time.perf_counter() - t0
        return time.perf_counter()

    def banner(self, text, color, kind):
        self.banners.append((self.t_s, text, color, kind))
        self.banners = self.banners[-6:]

    def finished(self):
        return bool(self.args.record is None and self.args.quit_after and self.t_s >= self.args.quit_after)

    def brains(self):
        return {"brain": self.fb}

    def close(self, summary=True):
        if summary:
            self.summary()
        steps = []
        if self.rcon is not None:
            steps.append(lambda: self.rcon("kill @e[tag=fv]"))
            steps.append(lambda: self.rcon("tick unfreeze"))
        if self.bridge is not None:
            steps.append(self.bridge.close)
        if self.server is not None:
            steps.append(self.server.stop)
        for fn in steps:
            try:
                fn()
            except (OSError, RuntimeError, ValueError):
                pass
        self._kill_children()

    def summary_quiet(self):
        prof, self.prof = self.prof, {}
        self.summary()
        self.prof = prof

    def summary(self):
        try:
            self.server_ticks = self.rcon.gametime() - self.gt0
        except (OSError, RuntimeError, ValueError):
            pass
        if self.prof:
            print("  profile (wall s): " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(self.prof.items(), key=lambda kv: -kv[1])))
        gf = np.asarray(self.hist["gf"])
        v = np.asarray(self.hist["v"])
        base = self.loco.baseline_speed
        md = np.asarray([np.inf if x is None else x for x in self.trace["mob_dist"]], float)
        far = np.asarray(self.trace["gf"], float)[md > 8.0] if len(md) else np.zeros(0)
        tt = np.asarray(self.trace["t"], float)
        quiet = np.ones(len(tt), bool)
        for e in self.encounters:                       # outside every encounter: summon .. removal + 1 s
            if e.summoned_at is not None:
                quiet &= ~((tt >= e.summoned_at) & (tt <= (e.removed_at if e.removed_at is not None else np.inf) + 1.0))
        gq = np.asarray(self.trace["gf"], float)[quiet] if len(tt) else np.zeros(0)
        self.log.summary = {
            "brain_s": round(self.t_s, 3), "path_m": round(self.path_len, 2), "gf_max_hz": round(float(gf.max()) if len(gf) else 0, 1),
            "gf_jumps": self.gf_jumps, "voluntary_jumps": self.voluntary_jumps, "auto_jumps": self.autojumps,
            "bumps": self.bumps, "deaths": self.deaths,
            "health_end": self.health, "footage": self.log.meta["minecraft"]["footage"],
            "minecraft_ticks_stepped": self.mc_k, "server_ticks_elapsed": getattr(self, "server_ticks", None),
            "control": self.args.control,
            "eye_entity_names": sorted(self.eye_names),
            "gf_no_mob_within_8m": ({"p99_hz": round(float(np.percentile(far, 99)), 1), "max_hz": round(float(far.max()), 1),
                                     "samples_50hz": int(len(far))} if len(far) else None),
            "gf_outside_encounters": ({"p99_hz": round(float(np.percentile(gq, 99)), 1), "max_hz": round(float(gq.max()), 1),
                                       "samples_50hz": int(len(gq)), "window": "summon .. removal + 1 s excluded"}
                                      if len(gq) else None),
            "encounters": [{"n": e.n, "mob": e.mob, "scripted": e.scripted, "bearing_deg": e.bearing_deg, "dist": e.dist,
                            "summoned_at": _r(e.summoned_at), "min_dist": _r(e.min_dist), "min_gap": _r(e.min_gap),
                            "gf_peak_near_hz": round(e.gf_peak_near, 1), "gf_max_window_hz": round(e.gf_max_window, 1),
                            **({"gf_peak_dive_hz": round(e.gf_peak_dive, 1), "gf_peak_pass_hz": round(e.gf_peak_pass, 1),
                                "pass_t": _r(e.pass_t)} if e.mob == "phantom" else {}),
                            "gf_crossings_t": e.gf_cross, "first_crossing": e.cross_at, "jumps": e.jumps, "hits": e.hits,
                            "eye_inside_frames": e.eye_inside_ticks, "removed_at": _r(e.removed_at), "outcome": e.outcome}
                           for e in self.encounters],
            "gf_peak_definitions": f"gf_peak_near: GF max while the mob was the nearest live one and within {NEAR_M} m "
                                   "(mob_dist); gf_max_window: at any distance while it was the nearest live one; "
                                   "phantoms: split at the pass (dive / pass)",
            "fly_speed_median_cm_s": round(float(np.median(v)) * 100, 3) if len(v) else None,
            "fly_speed_split": ({"baseline_cm_s": round(base * 100, 3),
                                 "neural_median_cm_s": round(float(np.median(v - base)) * 100, 3),
                                 "baseline_fraction_of_median": round(base / float(np.median(v)), 3)}
                                if len(v) and float(np.median(v)) > 0 else None),
            "mdn_max_hz": round(float(np.max(self.hist["mdn"])), 1) if self.hist["mdn"] else None,
            "traces_50hz": dict(self.trace, units="gf, lpL, lpR, lc4L, lc4R, mdn: Hz; v_fly: cm/s; v_player: m/s; "
                                "mob_dist: m (zombie: horizontal centre distance; phantom: hitbox centre to eye); "
                                "mob_gap: m (eye to the mob's hitbox); both for the nearest live mob"),
        }

    # ------------------------------------------------------------------------------------------------ camera (GAME)
    def camera(self, feet, fwd, ents):
        """GAME: the chase camera for the footage (never seen by the fly). It orbits the player: its bearing turns at
        most CAM_RATE towards the nearest live mob of an encounter (behind-and-beside the player on the line to it,
        so both are in frame), or towards the player's heading when there is none. While a phantom passes over the
        head (or any mob is within 2 m horizontally) the bearing is held, so the camera never swings through the
        player; afterwards it eases back. Distance, height and look point are smoothed (0.6 s)."""
        P = np.asarray(feet, float)
        dt = 1.0 / self.args.fps
        fwd_h = np.array([fwd[0], 0.0, fwd[2]])
        fwd_h /= max(np.linalg.norm(fwd_h), 1e-6)
        tgt_phi = math.atan2(fwd_h[2], fwd_h[0])
        par = np.array([6.0, 2.4, 2.5])                   # back, side, up
        look = P + fwd_h * 4.0 + [0, 1.0, 0]
        hold = False
        live = {(e.entity_id if e.entity_id is not None else -e.n): e for e in self.live_encounters()}
        zs = [e for e in ents if e["id"] in live and np.linalg.norm(np.subtract((e["x"], e["y"], e["z"]), P)) < 30]
        if zs:
            z = min(zs, key=lambda e: np.linalg.norm(np.subtract((e["x"], e["y"], e["z"]), P)))
            enc = live[z["id"]]
            Z = np.array([z["x"], z["y"], z["z"]])
            if z["name"] == "phantom":
                Z = P + (Z - P) * [1.0, 0.8, 1.0]         # keep the sky in frame but not the zenith
            u = Z - P
            u[1] = 0
            dist = max(float(np.linalg.norm(u)), 1e-3)
            hold = enc.mode == "pass" or dist < 2.0
            if hold:
                if self.cam_lock is None or self.cam_lock[0] != enc.n:
                    self.cam_lock = (enc.n, self.cam_phi if self.cam_phi is not None else tgt_phi)
                tgt_phi = self.cam_lock[1]
                look = P + np.array([math.cos(tgt_phi), 0.0, math.sin(tgt_phi)]) * 1.5 + [0, 1.4, 0]
            else:
                tgt_phi = math.atan2(u[2], u[0])
                look = P + (Z - P) * 0.42 + [0, 1.1, 0]
            par = np.array([5.0 + 0.25 * min(dist, 14), 2.6 + 0.12 * min(dist, 14), 2.3])
        if not hold:
            self.cam_lock = None
        if self.cam_phi is None:
            self.cam_phi, self.cam_par, self.cam_y, self.cam_look = tgt_phi, par, P[1], look
        a = 1 - math.exp(-dt / 0.6)
        self.cam_phi = orbit_step(self.cam_phi, tgt_phi, CAM_RATE * dt, 1 - math.exp(-dt / 0.35))
        self.cam_y = self.cam_y + (P[1] - self.cam_y) * (1 - math.exp(-dt / 0.5))   # no bob with the jump
        u = np.array([math.cos(self.cam_phi), 0.0, math.sin(self.cam_phi)])
        side = np.cross(u, [0, 1, 0])

        def place(par):
            return np.array([P[0], self.cam_y, P[2]]) - u * par[0] + side * par[1] + [0, par[2], 0]
        # stay out of terrain: raise the target height a block at a time, then smooth towards it
        for _ in range(8):
            ix, iy, iz = (np.floor(place(par)).astype(int) - self.region_origin)
            if (0 <= ix < self.rows_grid.shape[0] and 0 <= iy < self.rows_grid.shape[1] and 0 <= iz < self.rows_grid.shape[2]
                    and self.rows_grid[ix, iy, iz] > 0):
                par = par + [0, 0, 1.0]
            else:
                break
        self.cam_par = self.cam_par + (par - self.cam_par) * a
        self.cam_look = self.cam_look + (look - self.cam_look) * a
        pos = place(self.cam_par)
        c = P + [0, 0.9, 0]
        r = pos - c
        if np.linalg.norm(r) < CAM_MIN_DIST:
            pos = c + r / max(np.linalg.norm(r), 1e-6) * CAM_MIN_DIST
        return pos, self.cam_look

    def render_views(self, main_size, fp_size):
        """Real Minecraft (prismarine-viewer) frames: main chase view + the player's first-person view."""
        import pygame
        j = (self.k10 - 1) % BRAIN_TICKS_PER_MC
        f = (j + 1) / BRAIN_TICKS_PER_MC
        feet = lerp(self.P_prev, self.P_cur, f)
        yaw = mineflayer_yaw_from_heading(self.psi)
        fwd = yaw_forward(yaw)
        ents = self.ents_now(f)
        cam, look = self.camera(feet, fwd, ents)
        eye = feet + [0, EYE_HEIGHT, 0]
        fp_look = eye + fwd + [0, -0.12, 0]
        if not self.viewer_ok:
            return self.render_own(main_size, eye, fwd, ents), None
        vents = []
        for e in ents:
            if e["id"] is None or e["id"] < 0:              # a scripted phantom not yet matched to its server entity
                continue
            wp = self.ent_walk.get(e["id"], 0.0)
            spd = 0.0
            if e["id"] in self.E_prev and e["id"] in self.E_cur:
                spd = float(np.linalg.norm(np.subtract(self.E_cur[e["id"]]["pos"], self.E_prev[e["id"]]["pos"])[[0, 2]])) * 20
            vents.append({"id": e["id"], "x": e["x"], "y": e["y"], "z": e["z"], "yaw": e["yaw"], "pitch": e.get("pitch"),
                          "name": e["name"], "walk": wp, "speed": spd})
        for hid in self.hidden_ids:
            vents.append({"id": hid, "x": 0, "y": -500, "z": 0, "name": "", "walk": 0, "speed": 0})
        spd = float(np.linalg.norm((self.P_cur - self.P_prev)[[0, 2]])) * 20
        cams = [{"pos": cam.tolist(), "look": look.tolist(), "fov": 62, "w": main_size[0], "h": main_size[1]},
                {"pos": eye.tolist(), "look": fp_look.tolist(), "fov": 80, "firstPerson": True, "w": fp_size[0], "h": fp_size[1]}]
        imgs = self.bridge("frame", q=0.9, cams=cams, wait=1500,
                           bot={"x": feet[0], "y": feet[1], "z": feet[2], "yaw": yaw, "walk": self.walk_phase, "speed": spd},
                           ents=vents)
        out = []
        for im in imgs:
            raw = base64.b64decode(im.split(",", 1)[1])
            out.append(pygame.image.load(io.BytesIO(raw), "frame.jpg"))
        return out[0], out[1]

    def render_own(self, size, eye, fwd, ents):
        """Fallback when no real Minecraft renderer is available: our own ray cast of the server's block data."""
        import pygame
        W, H = 384, 216
        right = np.cross(fwd, [0, 1, 0])
        xs = (np.arange(W) + 0.5) / W * 2 - 1
        ys = (np.arange(H) + 0.5) / H * 2 - 1
        X, Y = np.meshgrid(xs, ys)
        t = np.tan(np.deg2rad(40))
        d = fwd[None, None] + X[..., None] * t * right + (-Y[..., None]) * t * H / W * np.array([0, 1.0, 0])
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        boxes = [e for e in ents if e.get("type") in EYE_ENTITY_TYPES]
        boxes.sort(key=lambda e: (e["x"] - eye[0]) ** 2 + (e["y"] - eye[1]) ** 2 + (e["z"] - eye[2]) ** 2)
        rad, _ = self.caster.cast(eye, d.reshape(-1, 3), boxes[:MAX_ENTITIES], graph=False)
        rgb = C.eye_colors(rad, exposure=2.5).reshape(H, W, 3)
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(rgb.transpose(1, 0, 2)))
        return pygame.transform.scale(surf, size)

    # ------------------------------------------------------------------------------------------------ HUD
    def draw(self, surface):
        import pygame
        hud = self.hud
        W, H = surface.get_size()
        sx = W / 1920
        S = lambda v: int(round(v * sx))  # noqa: E731
        surface.fill(C.BG)
        MX, MY, MW, MH = 0, S(64), S(1536), S(864)
        fp_w = W - S(1548) - S(12) - S(20)        # the right column panel's content width
        fp_h = int(fp_w * 9 / 16)
        t0 = time.perf_counter()
        main, fp = self.render_views((MW, MH), (fp_w, fp_h))
        t0 = self._t("viewer", t0)
        surface.blit(main, (MX, MY))
        self.frames += 1
        # footage label
        lab = "REAL MINECRAFT 1.21.4  ·  prismarine-viewer" if self.viewer_ok else "OUR RENDER OF THE SERVER'S BLOCK DATA"
        tag = hud.font(S(15), True).render(lab, True, C.WHITE)
        bg = pygame.Surface((tag.get_width() + S(20), tag.get_height() + S(10)), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 120))
        surface.blit(bg, (MX + S(16), MY + S(14)))
        surface.blit(tag, (MX + S(26), MY + S(19)))
        if self.args.record is None:              # interactive: say what the keys do
            hint = hud.font(S(15), True).render("keys 1-4: summon a zombie ahead / 35 deg left / 35 deg right / "
                                                "behind (GAME)", True, C.AMBER)
            bg = pygame.Surface((hint.get_width() + S(20), hint.get_height() + S(10)), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 120))
            surface.blit(bg, (MX + S(16), MY + S(48)))
            surface.blit(hint, (MX + S(26), MY + S(53)))
        # fly's-eye mosaic over the footage, bottom-left
        mw, mh = S(560), S(372)
        mrect = pygame.Rect(MX + S(16), MY + MH - mh - S(16), mw, mh)
        pan = pygame.Surface(mrect.size, pygame.SRCALPHA)
        pan.fill((9, 12, 14, 215))
        surface.blit(pan, mrect.topleft)
        pygame.draw.rect(surface, C.LINE, mrect, 1, border_radius=6)
        t = hud.text(surface, "WHAT THE FLY SEES", (mrect.x + S(14), mrect.y + S(10)), S(22), C.WHITE, bold=True,
                     display=True)
        hud.chip(surface, "game", (t.right + S(10), mrect.y + S(12)), S(11))
        hud.text(surface, "1,466 ommatidia · ray-cast through the server's blocks", (mrect.x + S(14), mrect.y + S(40)), S(14), C.MUTED)
        if self.args.control == "blind-to-mobs":
            hud.text(surface, "CONTROL: NO SCRIPTED MOBS", (mrect.right - S(14), mrect.y + S(12)), S(16), C.RED,
                     bold=True, anchor="topright")
        inner = pygame.Rect(mrect.x + S(10), mrect.y + S(62), mw - S(20), mh - S(92))
        if self.rad is not None:
            cols = C.eye_colors(self.rad, "human", exposure=2.5)
            hud.mosaic(surface, inner, self.eyes, cols)
        hud.text(surface, "black: no column in the eye map", (mrect.x + S(14), mrect.bottom - S(24)), S(13), C.DIM)
        self.draw_mosaic_marker(surface, inner, S, (mrect.right - S(14), mrect.bottom - S(24)))
        self.draw_banners(surface, pygame.Rect(MX, MY, MW, MH), S)
        # header / footer
        sub = ("CONTROL: the scripted mobs left out of the fly's ray cast" if self.args.control == "blind-to-mobs"
               else "the fly in real Minecraft 1.21.4")
        hud.header(surface, "MINECRAFT", sub, C.model_line(self.fb))
        laws = ["walk = 216 x body speed (0.8 of ~0.9 cm/s a constant drive)", "turn = body yaw 1:1",
                "GF >= 33 Hz -> jump", "mobs scripted · Auto-Jump · lockstep 50 ms"]
        hud.footer(surface, laws)
        # right column
        RX = S(1548)
        RW = W - RX - S(12)
        y = S(72)
        r = hud.panel(surface, (RX, y, RW, fp_h + S(46)), "player's eye · minecraft", None)
        if fp is not None:
            surface.blit(fp, (r.x, r.y))
        else:
            hud.text(surface, "(first-person view: viewer unavailable)", (r.x, r.y + S(80)), S(14), C.DIM)
        y += fp_h + S(54)
        y = self.draw_loom_panel(surface, RX, y, RW, S)
        y = self.draw_walk_panel(surface, RX, y, RW, S)
        y = self.draw_game_panel(surface, RX, y, RW, H - S(56) - y, S)
        # bottom strip: the giant fibre trace
        self.draw_gf_strip(surface, pygame.Rect(S(12), MY + MH + S(8), MW - S(24), H - S(56) - (MY + MH + S(8))), S)
        self._t("hud", t0)

    def draw_mosaic_marker(self, surface, rect, S, label_at):
        """GAME overlay: a ring where the nearest mob of an encounter is in the fly's field, its radius the mob's
        angular size (the box's bounding sphere), from this frame's geometry (the positions the eye was cast
        against). The fly's eye itself is the mosaic under it."""
        import pygame
        enc = self.nearest_encounter()
        if enc is None or enc.n not in self.geom or self.args.control == "blind-to-mobs":
            return
        _, gap, centre, (w, h) = self.geom[enc.n]
        f = ((self.k10 - 1) % BRAIN_TICKS_PER_MC + 1) / BRAIN_TICKS_PER_MC
        eye = lerp(self.P_prev, self.P_cur, f) + [0, EYE_HEIGHT, 0]
        az, el, rad_deg, _ = fly_view_of(centre, eye, self.psi, 0.5 * math.sqrt(2 * w * w + h * h))
        clip = surface.get_clip()
        surface.set_clip(rect)
        drawn = 0
        for x, y, scale in mosaic_points(self.eyes.az_el, self.eyes.side, tuple(rect), az, el):
            r = int(max(S(8), min(rad_deg * scale, rect.h * 0.5)))
            pygame.draw.circle(surface, C.AMBER, (int(x), int(y)), r, max(2, S(3)))
            drawn += 1
        surface.set_clip(clip)
        if drawn:
            across = rad_deg * 2
            size = "> 120" if across > 120 else f"{across:.0f}"
            label = f"ring (GAME): {enc.mob} {size} deg, {gap:.1f} m"
            self.hud.text(surface, label, label_at, S(13), C.AMBER, anchor="topright")

    def draw_gf_strip(self, surface, rect, S):
        import pygame
        hud = self.hud
        hud.panel(surface, rect, None)
        gf = self.hist["gf"][-600:]
        now = gf[-1] if gf else 0.0
        hud.text(surface, "GIANT FIBRE DNp01", (rect.x + S(14), rect.y + S(10)), S(18), C.TEXT, bold=True)
        hud.chip(surface, "connectome", (rect.x + S(14), rect.y + S(38)), S(12))
        hud.text(surface, f"{now:5.1f} Hz", (rect.x + S(14), rect.y + S(58)), S(24), C.SAGE if now < 33 else C.RED, bold=True)
        tr = pygame.Rect(rect.x + S(190), rect.y + S(8), rect.w - S(190) - S(215), rect.h - S(16))
        hud.trace(surface, tr, gf, 0, 120, C.SAGE, threshold=self.flight.gf_hz)
        hud.text(surface, "33 Hz body threshold -> jump", (tr.x + S(6), tr.y + S(2)), S(12), C.RED)
        hud.text(surface, "last 6 s · 0-120 Hz", (tr.right - S(6), tr.y + S(2)), S(12), C.DIM, anchor="topright")
        hud.text(surface, f"max {self.max_gf:5.1f} Hz", (rect.right - S(14), rect.y + S(10)), S(22), C.MUTED,
                 anchor="topright")
        j = hud.text(surface, f"jumps {self.gf_jumps}", (rect.right - S(14), rect.y + S(44)), S(24), C.TEAL,
                     anchor="topright", bold=True)
        cw = hud.font(S(11), True).size("DECODER")[0] + S(12)
        hud.chip(surface, "decoder", (j.x - cw - S(10), j.y + S(6)), S(11))

    def draw_loom_panel(self, surface, x, y, w, S):
        import pygame
        hud = self.hud
        h = S(212)
        r = hud.panel(surface, (x, y, w, h), "loom detectors (mean rate)", "connectome")
        last = lambda k: self.hist[k][-1] if self.hist[k] else 0.0  # noqa: E731
        rows = [("LPLC2  left", last("lpL"), C.TEAL), ("LPLC2  right", last("lpR"), C.TEAL),
                ("LC4  left", last("lc4L"), C.LILAC), ("LC4  right", last("lc4R"), C.LILAC)]
        for i, (lab, v, col) in enumerate(rows):
            yy = r.y + i * S(41)
            hud.text(surface, lab, (r.x, yy + S(4)), S(15), C.TEXT)
            hud.text(surface, f"{v:5.1f} Hz", (r.right, yy - S(2)), S(22), col, bold=True, anchor="topright")
            track = pygame.Rect(r.x, yy + S(26), r.w, S(8))
            pygame.draw.rect(surface, C.INSET, track, border_radius=3)
            frac = float(np.clip(v / 12.0, 0, 1))
            if frac > 0:
                pygame.draw.rect(surface, col, (track.x, track.y, max(2, round(track.w * frac)), track.h), border_radius=3)
        return y + h + S(8)

    def draw_walk_panel(self, surface, x, y, w, S):
        import pygame
        hud = self.hud
        h = S(196)
        r = hud.panel(surface, (x, y, w, h), "body", None)
        mdn = self.hist["mdn"][-1] if self.hist["mdn"] else 0.0
        t = hud.text(surface, "MDN (moonwalker)", (r.x, r.y + S(4)), S(15), C.TEXT)
        hud.chip(surface, "connectome", (t.right + S(8), r.y + S(2)), S(11))
        hud.text(surface, f"{mdn:5.1f} Hz", (r.right, r.y - S(2)), S(22), C.LILAC, bold=True, anchor="topright")
        track = pygame.Rect(r.x, r.y + S(26), r.w, S(8))
        pygame.draw.rect(surface, C.INSET, track, border_radius=3)
        frac = float(np.clip(mdn / 80.0, 0, 1))
        if frac > 0:
            pygame.draw.rect(surface, C.LILAC, (track.x, track.y, max(2, round(track.w * frac)), track.h), border_radius=3)
        xt = track.x + round(track.w * self.loco.mdn_threshold / 80.0)
        pygame.draw.line(surface, C.RED, (xt, track.y - S(4)), (xt, track.bottom + S(3)), 2)
        v = self.hist["v"][-1] if self.hist["v"] else 0.0
        base = self.loco.baseline_speed
        hud.text(surface, "fly walk (body model)", (r.x, r.y + S(46)), S(15), C.TEXT)
        hud.text(surface, f"{v * 100:+.2f} cm/s", (r.x, r.y + S(66)), S(24), C.TEAL, bold=True)
        hud.chip(surface, "decoder", (r.x + S(150), r.y + S(70)), S(11))
        hud.text(surface, "player", (r.x + S(222), r.y + S(46)), S(15), C.TEXT)
        hud.text(surface, f"{self.cmd['v_player']:+.2f} m/s", (r.x + S(222), r.y + S(66)), S(24), C.TEAL, bold=True)
        hud.text(surface, f"= {base * 100:.2f} constant {(v - base) * 100:+.2f} from DN/MN rates",
                 (r.x, r.y + S(98)), S(14), C.MUTED)
        yaw = self.hist["yaw"][-1] if self.hist["yaw"] else 0.0
        t = hud.text(surface, f"yaw {np.rad2deg(yaw):+5.1f} deg/s", (r.x, r.y + S(124)), S(16), C.MUTED)
        hud.chip(surface, "decoder", (t.right + S(10), r.y + S(124)), S(11))
        return y + h + S(8)

    def draw_game_panel(self, surface, x, y, w, h, S):
        import pygame
        hud = self.hud
        r = hud.panel(surface, (x, y, w, h), "game", "game")
        # hearts
        for i in range(10):
            hx, hy = r.x + i * S(33), r.y + S(4)
            fill = np.clip(self.health / 2 - i, 0, 1)
            col = C.RED if fill >= 1 else ((150, 70, 66) if fill > 0 else C.INSET)
            pygame.draw.circle(surface, col, (hx + S(8), hy + S(8)), S(8))
            pygame.draw.circle(surface, col, (hx + S(20), hy + S(8)), S(8))
            pygame.draw.polygon(surface, col, [(hx, hy + S(11)), (hx + S(28), hy + S(11)), (hx + S(14), hy + S(28))])
        hud.text(surface, f"{self.health:.1f} / 20 HP", (r.x, r.y + S(36)), S(16), C.MUTED)
        enc = self.nearest_encounter()
        g = self.gap_to(enc) if enc else None
        hud.text(surface, (f"{enc.mob}: eye to hitbox" if enc else "mob"), (r.x, r.y + S(64)), S(15), C.TEXT)
        hud.text(surface, "--" if g is None or not np.isfinite(g) else f"{g:4.1f} m", (r.x, r.y + S(84)), S(28), C.AMBER, bold=True)
        done = [e for e in self.encounters if e.summoned_at is not None]
        hud.text(surface, "encounters", (r.x + S(222), r.y + S(64)), S(15), C.TEXT)
        hud.text(surface, f"{len(done)}", (r.x + S(222), r.y + S(84)), S(28), C.AMBER, bold=True)
        yy = r.y + S(128)
        kinds = {"summon": C.AMBER, "hit": C.AMBER, "despawn": C.AMBER, "death": C.AMBER, "gf_cross": C.SAGE,
                 "backing_up": C.SAGE, "escape": C.TEAL, "voluntary_takeoff": C.TEAL, "eye_in_mob_box": C.AMBER}
        for ev in [e for e in self.log.events if e["kind"] in kinds][-6:][::-1]:
            if yy > r.bottom - S(18):
                break
            pygame.draw.rect(surface, kinds[ev["kind"]], (r.x, yy + S(5), S(8), S(8)), border_radius=2)
            hud.text(surface, self.event_text(ev), (r.x + S(14), yy), S(14), C.MUTED)
            yy += S(21)
        return y + h

    @staticmethod
    def event_text(ev):
        k = ev["kind"]
        t = f"{ev['t_s']:5.2f}s "
        if k == "summon":
            return t + f"{ev['mob']} #{ev['encounter']} summoned {ev['dist']:.0f} m"
        if k == "gf_cross":
            g = ev.get("mob_gap")
            return t + f"GF {ev['hz']:.0f} Hz" + (f", hitbox {g:.1f} m" if g is not None else "")
        if k == "escape":
            return t + "GF >= 33 Hz -> jump"
        if k == "voluntary_takeoff":
            return t + "power MNs -> takeoff jump"
        if k == "jump":
            return t + f"jump ({ev['source']})"
        if k == "hit":
            return t + f"hit, {ev['health']:.1f} HP left"
        if k == "backing_up":
            return t + f"MDN {ev['mdn_hz']:.0f} Hz: backing up"
        if k == "despawn":
            return t + f"#{ev['encounter']} removed, GF peak {ev['gf_peak_near']:.0f} Hz (< {NEAR_M:.0f} m)"
        if k == "death":
            return t + "died"
        if k == "eye_in_mob_box":
            return t + "eye inside a mob's hitbox"
        return t + k

    def draw_banners(self, surface, rect, S):
        import pygame
        hud = self.hud
        shown = [b for b in self.banners if self.t_s - b[0] < 2.0][-2:]
        for i, (t0, text, col, kinds) in enumerate(reversed(shown)):
            kinds = (kinds,) if isinstance(kinds, str) else tuple(kinds)
            age = self.t_s - t0
            alpha = 1.0 if age < 1.2 else max(0.0, 1 - (age - 1.2) / 0.8)
            size = S(50)
            img = hud.font(size, True, True).render(text, True, col)
            while img.get_width() + S(56) > rect.w - S(640) and size > S(26):   # fit beside the mosaic
                size -= S(4)
                img = hud.font(size, True, True).render(text, True, col)
            box = pygame.Surface((img.get_width() + S(56), img.get_height() + S(26)), pygame.SRCALPHA)
            box.fill((8, 10, 12, int(200 * alpha)))
            pygame.draw.rect(box, (*col, int(255 * alpha)), (0, 0, S(8), box.get_height()))
            img.set_alpha(int(255 * alpha))
            bx = rect.x + S(600) + (rect.w - S(600) - box.get_width()) // 2
            bx = min(bx, rect.right - S(12) - box.get_width())
            by = rect.bottom - S(40) - (i + 1) * (box.get_height() + S(22))
            surface.blit(box, (bx, by))
            surface.blit(img, (bx + S(32), by + S(13)))
            cx = bx + box.get_width()                       # chips along the box's top edge, right-aligned inside it
            for kind in reversed(kinds):                    # (drawn on their own layer so they fade with the banner)
                w = hud.font(S(12), True).size(C.PROVENANCE[kind][0])[0] + S(12)
                cx -= w + S(6)
                layer = pygame.Surface((w + S(4), S(26)), pygame.SRCALPHA)
                hud.chip(layer, kind, (0, 0), S(12))
                layer.set_alpha(int(255 * alpha))
                surface.blit(layer, (cx, by - S(20)))

    # ------------------------------------------------------------------------------------------------ interactive
    def handle(self, event, canvas_pos=None):
        """Interactive: keys 1-4 summon a zombie 12 m ahead / 35 deg left / 35 deg right / behind (GAME). These
        are not part of the script and get the full ENCOUNTER_TIMEOUT_S."""
        import pygame
        if event.type != pygame.KEYDOWN:
            return
        bearings = {pygame.K_1: 0.0, pygame.K_2: 35.0, pygame.K_3: -35.0, pygame.K_4: 180.0}
        if event.key in bearings:
            enc = Encounter(self.t_s, bearings[event.key], 12.0, n=len(self.encounters) + 1, scripted=False)
            self.encounters.append(enc)
            self.summon(enc)


# ---------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = C.standard_args("the fly in real Minecraft: MaleCNS v1.0 drives a mineflayer bot on a local 1.21.4 server",
                         seconds=30.0)
    ap.add_argument("--reuse-server", action="store_true", help="use a server already running on the RCON port "
                    "(no fresh world)")
    ap.add_argument("--no-viewer", action="store_true", help="skip the real Minecraft renderer; draw our own render "
                    "of the server's block data (labelled)")
    ap.add_argument("--no-script", action="store_true", help="no scripted mobs (interactive: keys 1-4 summon a zombie ahead / left / right / behind)")
    ap.add_argument("--profile", action="store_true", help="print a wall-time breakdown at the end")
    ap.add_argument("--quit-after", type=float, default=0.0, help="interactive: quit after this many brain seconds")
    ap.add_argument("--script-json", default=None, help="dev: replace the GAME script, e.g. '[[2, \"phantom\", 0, 22, 11]]' "
                    "(time s, mob, bearing deg, distance m, height m)")
    ap.add_argument("--control", choices=("none", "blind-to-mobs"), default="none",
                    help="control arm: 'blind-to-mobs' leaves the mobs out of the fly's ray cast (same script)")
    args = ap.parse_args(argv)
    game = MinecraftGame(args)
    try:
        path = C.run(game, args)
    finally:
        game.close()
    if path:
        game.log.save(path, game.brains())
    return path


if __name__ == "__main__":
    main()
