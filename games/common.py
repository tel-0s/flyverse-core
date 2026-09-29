"""Shared plumbing for the games in this directory: build the brain, give it eyes on any scene, read it out through
declared decoders, draw one consistent HUD, keep an honest run log, and record a clip.

Nothing in this file changes the model. The brain is `flyverse.FlyBrain` exactly as shipped (preset 'raw' unless a
game says otherwise and logs it). A game touches the brain only through the public control surface
(docs/CONTROL_SURFACE.md): `fb.vision / smell / wind / taste / stimulate` in, `fb.motor()` and attached read-only
modules out. Every mapping from neural activity to a game control is a `ReadDecoder` -- a module with
`kind="decoder"` that writes nothing, so attaching it leaves the simulation numerically unchanged
(docs/EXTENSIBILITY.md, "a module that writes nothing ... is numerically neutral") and it is listed in
`fb.module_records()`, which the run log saves.

The HUD sorts every number on screen into one of three provenance classes, and says which:

    CONNECTOME  a rate read straight from named cells: the model's own output
    DECODER     a mapping from neural activity to a game control, written for the game (declared, `kind="decoder"`)
    GAME        game logic, physics, autopilot or any hand-designed drive that is not the fly

Run any game interactively (`python games/<name>.py`) or headless to a clip
(`python games/<name>.py --record out/games/<name>.mp4 --seconds 20`). A clip's run log (`--log`, default: next to
the clip) carries the command, commit, device, seed, attached modules and the game's own event list, which is what
its caption in games/README.md is written from.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                      # `python games/x.py` from a checkout, without an install
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

TICK_MS = 10.0                 # brain time per game tick: one FlyBrain.step(10) = 20 LIF steps at 0.5 ms
DEFAULT_FPS = 50               # clip frame rate; must divide 1000 / TICK_MS so every video frame is whole ticks
DEFAULT_SIZE = (1920, 1080)    # clip canvas (the trailer's resolution)
DEFAULT_WINDOW = (1280, 720)   # interactive window; the canvas is scaled into it

# The room console's palette (flyverse/room_ui.py), so the games and the console read as one project.
BG = (13, 16, 18)
PANEL = (19, 24, 27)
INSET = (9, 12, 14)
LINE = (43, 53, 58)
TEXT = (213, 223, 219)
MUTED = (151, 171, 172)
DIM = (109, 135, 139)
SAGE = (185, 219, 126)
TEAL = (107, 201, 208)
AMBER = (232, 179, 104)
LILAC = (176, 157, 216)
RED = (240, 104, 96)
WHITE = (240, 244, 242)

PROVENANCE = {                 # class -> (chip label, colour); see the module docstring
    "connectome": ("CONNECTOME", SAGE),
    "decoder": ("DECODER", TEAL),
    "game": ("GAME", AMBER),
}

MONO_FONTS = "Cascadia Mono,Consolas,DejaVu Sans Mono,Liberation Mono,Menlo"
DISPLAY_FONTS = "Bahnschrift,Segoe UI,DejaVu Sans,Liberation Sans,Helvetica"


# ---------------------------------------------------------------------------------------------------- arguments
def standard_args(description: str, *, seconds: float = 20.0) -> argparse.ArgumentParser:
    """The flags every game shares. Games add their own on the returned parser."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--seed", type=int, default=0, help="brain RNG seed (and the game's, unless it says otherwise)")
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available, else cpu)")
    ap.add_argument("--record", metavar="MP4", default=None, help="run headless and write the clip here")
    ap.add_argument("--seconds", type=float, default=seconds, help="brain seconds to run when recording")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS, choices=(20, 25, 50), help="clip frame rate")
    ap.add_argument("--size", default="%dx%d" % DEFAULT_SIZE, help="canvas size WxH")
    ap.add_argument("--window", default="%dx%d" % DEFAULT_WINDOW, help="interactive window size WxH")
    ap.add_argument("--log", metavar="JSON", default=None, help="run log (default: <record>.json, or none)")
    ap.add_argument("--screenshot", metavar="PNG", default=None, help="save the last frame")
    return ap


def parse_size(text: str) -> tuple[int, int]:
    w, h = (int(v) for v in text.lower().split("x"))
    if w <= 0 or h <= 0:
        raise ValueError(f"size must be positive: {text!r}")
    return w, h


def pick_device(name: str | None) -> str:
    if name:
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_brain(args, **kw):
    """`FlyBrain(device=..., seed=args.seed, **kw)`. Nothing is attached and the preset is 'raw' unless `kw` says so."""
    from flyverse import FlyBrain
    return FlyBrain(device=pick_device(args.device), seed=args.seed, **kw)


# ---------------------------------------------------------------------------------------------------- decoders
class ReadDecoder:
    """A read-only module (`kind="decoder"`): each `fb.step()` call it receives the previous frame's quantity for its
    `reads` selections, computes `value = fn(dt_ms, inputs)`, and keeps it. It writes nothing, so it cannot perturb
    the brain, and `fb.attach(decoder)` records it in provenance. `law` is one sentence saying what the mapping is,
    in plain words; it is saved in the run log and printed in the HUD legend.

        dec = ReadDecoder("paddle", {"L": {"type": "DNp18", "somaSide": "L"}, "R": {"type": "DNp18", "somaSide": "R"}},
                          lambda dt, x: float((x["L"].mean() - x["R"].mean()).item()),
                          law="paddle velocity = DNp18 left minus right, Hz")
        fb.attach(dec); fb.step(10); dec.value

    Selections use the `interp.common.resolve` grammar (a dict of Connectome.select criteria, a type name, a
    regex with '~', body ids ...). `quantity_in` is 'rate_hz' (LIF rates), 'drive_mv', 'spike_count' or
    'optic_rate' (graded optic-lobe units; slow to gather for thousands of cells -- reading `fb.optic` directly in
    the game is acceptable when it is declared the same way, see `declare`)."""

    kind = "decoder"

    def __init__(self, name, reads, fn, *, law, quantity_in="rate_hz", parameters=None):
        self.name, self.reads, self.writes = name, dict(reads), {}
        self.fn, self.law = fn, law
        self.quantity_in, self.channel_out = quantity_in, "poisson_hz"
        self.parameters = dict(parameters or {})
        self.value = None

    def reset(self, B, device):
        self.B, self.device, self.value = B, torch.device(device), None

    def reset_rows(self, rows):
        pass

    def step(self, dt_ms, inputs):
        self.value = self.fn(dt_ms, inputs)
        return {}

    def state_dict(self):
        return {}

    def load_state_dict(self, d):
        pass

    def describe(self):
        fn = getattr(self.fn, "__qualname__", type(self.fn).__name__)
        return {"name": self.name, "class": "games.common.ReadDecoder", "kind": self.kind, "callable": fn,
                "law": self.law, "parameters": self.parameters, "trainable": False, "checkpoint_hash": None}


def declare(log, name: str, kind: str, law: str, reads: str = "", **parameters):
    """Record a mapping that is not an attached module (a decoder reading `fb.optic` directly, a game rule, an
    autopilot, a stand-in drive) in the run log with the same fields a module record has. `kind` is one of
    'decoder', 'stop-gap', 'game'."""
    if kind not in ("decoder", "stop-gap", "game"):
        raise ValueError(f"kind must be decoder / stop-gap / game, got {kind!r}")
    log.meta.setdefault("declared", []).append({"name": name, "kind": kind, "law": law, "reads": reads,
                                                "parameters": parameters})


# ---------------------------------------------------------------------------------------------------- eyes
def rgb_to_radiance(rgb):
    """(..., 3) linear RGB in [0, 1] -> (..., 4) radiance [UV, B, G, R] for sources that only have RGB (a game
    frame, a Minecraft block colour). UV is not in an RGB image; it is **approximated as half the blue channel**
    (the convention scripts/probe_motion.py uses for its gratings). A declared assumption, logged by `Eyes`."""
    if isinstance(rgb, torch.Tensor):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        return torch.stack([0.5 * b, b, g, r], dim=-1)
    rgb = np.asarray(rgb, np.float32)
    return np.stack([0.5 * rgb[..., 2], rgb[..., 2], rgb[..., 1], rgb[..., 0]], axis=-1)


def srgb_to_linear(rgb8):
    x = np.asarray(rgb8, np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


class Eyes:
    """The fly's 1,466 ommatidial columns as rays. Each column samples 7 rays (centre + hexagon, the retina's
    acceptance function); `radiance` averages them with the retina's weights into the (n_columns, 4) array that
    `fb.vision` takes.

    Directions are in the fly's body frame: x forward, y left, z up. `world_dirs(forward, left, up)` rotates them
    into a world frame for a scene that lives in world coordinates."""

    def __init__(self, fb):
        if fb.retina is None:
            raise ValueError("this brain has no retina (FAFB and MaleCNS have one; default BANC does not)")
        self.fb, self.retina = fb, fb.retina
        dirs, w = fb.retina.ray_directions()
        self.dirs_body = np.asarray(dirs, np.float32)                 # (n_col, k, 3)
        self.weights = np.asarray(w, np.float32)                      # (k,), sums to 1
        self.n_col, self.k = self.dirs_body.shape[:2]
        self.az_el = np.asarray(fb.retina.col_az_el, np.float32)      # (n_col, 2) deg, azimuth + = left, elevation + = up
        self.side = np.asarray(fb.retina.col_side)
        self._w_t = torch.from_numpy(self.weights).to(fb.device)

    def world_dirs(self, forward=(1, 0, 0), left=(0, 1, 0), up=(0, 0, 1)) -> np.ndarray:
        """(n_col * k, 3) unit ray directions in a world frame whose axes are the fly's forward / left / up."""
        R = np.stack([np.asarray(forward, np.float32), np.asarray(left, np.float32), np.asarray(up, np.float32)], 0)
        return self.dirs_body.reshape(-1, 3) @ R

    def pool(self, samples) -> torch.Tensor:
        """(n_col * k, 4) per-ray radiance -> (n_col, 4) per-column radiance on the brain's device."""
        s = torch.as_tensor(samples, dtype=torch.float32, device=self.fb.device).reshape(self.n_col, self.k, 4)
        return (s * self._w_t[None, :, None]).sum(1)

    def radiance(self, shade, forward=(1, 0, 0), left=(0, 1, 0), up=(0, 0, 1)) -> torch.Tensor:
        """`shade(dirs)` maps (M, 3) world directions to (M, 4) radiance [UV, B, G, R] (numpy or torch)."""
        return self.pool(shade(self.world_dirs(forward, left, up)))

    def from_equirect(self, image, forward=(1, 0, 0), left=(0, 1, 0), up=(0, 0, 1)) -> torch.Tensor:
        """Sample an equirectangular panorama (H, W, 3) linear RGB or (H, W, 4) radiance, in the world frame whose
        +x is longitude 0 and +z is the top row."""
        img = np.asarray(image, np.float32)
        d = self.world_dirs(forward, left, up)
        lon = np.arctan2(d[:, 1], d[:, 0]); lat = np.arcsin(np.clip(d[:, 2], -1, 1))
        H, W = img.shape[:2]
        u = ((0.5 - lon / (2 * np.pi)) * W).astype(np.int64) % W
        v = np.clip(((0.5 - lat / np.pi) * H).astype(np.int64), 0, H - 1)
        px = img[v, u]
        return self.pool(px if px.shape[-1] == 4 else rgb_to_radiance(px))

    def from_pinhole(self, image, hfov_deg, forward=(1, 0, 0), left=(0, 1, 0), up=(0, 0, 1), outside=(0, 0, 0, 0)):
        """Sample a flat camera frame (H, W, 3) linear RGB (or 4 radiance) rendered looking along `forward` with
        horizontal field of view `hfov_deg`; rays outside the frame get `outside` radiance. For sources that only
        render a screen (a game window): the eye sees the screen as if it filled that field."""
        img = np.asarray(image, np.float32)
        H, W = img.shape[:2]
        d = self.dirs_body.reshape(-1, 3)                             # camera = body frame: x fwd, y left, z up
        fwd = d[:, 0]
        ok = fwd > 1e-3
        tan = np.tan(np.deg2rad(hfov_deg) / 2)
        x = np.where(ok, -d[:, 1] / np.maximum(fwd, 1e-3), 9.0) / tan          # -1 .. 1 left to right
        y = np.where(ok, -d[:, 2] / np.maximum(fwd, 1e-3), 9.0) / (tan * H / W)  # -1 .. 1 top to bottom
        inside = ok & (np.abs(x) <= 1) & (np.abs(y) <= 1)
        u = np.clip(((x + 1) / 2 * (W - 1)).round().astype(np.int64), 0, W - 1)
        v = np.clip(((y + 1) / 2 * (H - 1)).round().astype(np.int64), 0, H - 1)
        px = img[v, u]
        rad = px if px.shape[-1] == 4 else rgb_to_radiance(px)
        rad = np.where(inside[:, None], rad, np.asarray(outside, np.float32)[None])
        return self.pool(rad)


# ---------------------------------------------------------------------------------------------------- HUD
class Hud:
    """Fonts and a few drawing primitives in the console's style. Sizes are in canvas pixels; the canvas is
    DEFAULT_SIZE for clips, and games lay out in fractions of it."""

    def __init__(self):
        import pygame
        self.pg = pygame
        if not pygame.font.get_init():
            pygame.font.init()
        self._fonts = {}
        self._mosaic = {}

    def font(self, size=18, bold=False, display=False):
        key = (size, bold, display)
        if key not in self._fonts:
            self._fonts[key] = self.pg.font.SysFont(DISPLAY_FONTS if display else MONO_FONTS, size, bold=bold)
        return self._fonts[key]

    def text(self, surface, s, pos, size=18, color=TEXT, *, bold=False, display=False, anchor="topleft"):
        img = self.font(size, bold, display).render(str(s), True, color)
        rect = img.get_rect(**{anchor: (int(pos[0]), int(pos[1]))})
        surface.blit(img, rect)
        return rect

    def chip(self, surface, kind, pos, size=13):
        """A provenance chip (CONNECTOME / DECODER / GAME) with its top-left at `pos`; returns its rect."""
        label, color = PROVENANCE[kind]
        img = self.font(size, True).render(label, True, BG)
        r = img.get_rect(topleft=(pos[0] + 6, pos[1] + 3))
        box = r.inflate(12, 6)
        box.topleft = pos
        self.pg.draw.rect(surface, color, box, border_radius=4)
        surface.blit(img, img.get_rect(center=box.center))
        return box

    def panel(self, surface, rect, title=None, kind=None):
        """A dark panel with an optional title and provenance chip; returns the content rect below the title."""
        rect = self.pg.Rect(rect)
        self.pg.draw.rect(surface, PANEL, rect, border_radius=6)
        self.pg.draw.rect(surface, LINE, rect, 1, border_radius=6)
        if not title:
            return rect.inflate(-16, -16)
        t = self.text(surface, title.upper(), (rect.x + 12, rect.y + 9), 15, MUTED, bold=True)
        if kind:
            self.chip(surface, kind, (t.right + 10, rect.y + 6), 12)
        return self.pg.Rect(rect.x + 10, rect.y + 36, rect.w - 20, rect.h - 46)

    def bar(self, surface, rect, value, vmax, label, color=SAGE, *, threshold=None, unit="Hz", fmt="{:.1f}"):
        """A horizontal bar: label left, value right, optional threshold tick."""
        rect = self.pg.Rect(rect)
        self.text(surface, label, (rect.x, rect.y), 15, TEXT)
        self.text(surface, (fmt.format(value) + (" " + unit if unit else "")), (rect.right, rect.y), 15, color,
                  anchor="topright")
        track = self.pg.Rect(rect.x, rect.y + 22, rect.w, max(6, rect.h - 24))
        self.pg.draw.rect(surface, INSET, track, border_radius=3)
        frac = float(np.clip(value / vmax, 0, 1)) if vmax > 0 else 0.0
        if frac > 0:
            self.pg.draw.rect(surface, color, (track.x, track.y, max(2, round(track.w * frac)), track.h),
                              border_radius=3)
        if threshold is not None and vmax > 0:
            x = track.x + round(track.w * min(threshold / vmax, 1.0))
            self.pg.draw.line(surface, RED, (x, track.y - 4), (x, track.bottom + 3), 2)

    def trace(self, surface, rect, values, vmin, vmax, color=SAGE, *, threshold=None, fill=True):
        """A scrolling line plot of `values` (oldest first) filling `rect`."""
        rect = self.pg.Rect(rect)
        self.pg.draw.rect(surface, INSET, rect, border_radius=3)
        v = np.asarray(values, np.float64)
        if threshold is not None:
            y = rect.bottom - (threshold - vmin) / (vmax - vmin) * rect.h
            if rect.top <= y <= rect.bottom:
                self.pg.draw.line(surface, RED, (rect.x, y), (rect.right, y), 1)
        if len(v) < 2:
            return
        x = rect.x + np.linspace(0, rect.w - 1, len(v))
        y = rect.bottom - 1 - np.clip((v - vmin) / (vmax - vmin), 0, 1) * (rect.h - 2)
        pts = np.c_[x, y].round().astype(int).tolist()
        if fill:
            poly = [(pts[0][0], rect.bottom - 1)] + pts + [(pts[-1][0], rect.bottom - 1)]
            shade = tuple(int(c * 0.28 + b * 0.72) for c, b in zip(color, INSET))
            self.pg.draw.polygon(surface, shade, poly)
        self.pg.draw.lines(surface, color, False, pts, 2)

    def mosaic(self, surface, rect, eyes: Eyes, colors):
        """Draw the fly's-eye view: one hexagon per ommatidial column at its azimuth / elevation, left eye on the
        left. `colors` is (n_col, 3) uint8 (see `eye_colors`)."""
        rect = self.pg.Rect(rect)
        key = (id(eyes), tuple(rect))
        if key not in self._mosaic:
            az, el = eyes.az_el[:, 0].copy(), eyes.az_el[:, 1]
            # the eyes overlap in a ~26 deg frontal binocular strip; draw them apart so neither hides the other
            L, R = eyes.side == "L", eyes.side == "R"
            if L.any() and R.any():
                half = max(0.0, (az[R].max() - az[L].min()) / 2) + 3.0
                az[L] += half; az[R] -= half
            scale = min((rect.w - 8) / max(np.ptp(az), 1), (rect.h - 8) / max(np.ptp(el), 1))
            cx = -(az - (az.max() + az.min()) / 2) * scale + rect.centerx
            cy = -(el - (el.max() + el.min()) / 2) * scale + rect.centery
            radius = max(1.5, scale * 4.6 / np.sqrt(3) * 1.02)
            ang = np.arange(6) * np.pi / 3 + np.pi / 6
            polys = np.round(np.stack([cx, cy], 1)[:, None, :] + radius * np.c_[np.cos(ang), np.sin(ang)][None]).astype(int)
            self._mosaic[key] = polys.tolist()
        self.pg.draw.rect(surface, INSET, rect, border_radius=4)
        for poly, c in zip(self._mosaic[key], np.asarray(colors).tolist()):
            self.pg.draw.polygon(surface, c, poly)

    def header(self, surface, title, subtitle="", right=""):
        """The strip across the top of every game: FLYVERSE / TITLE, a subtitle, and the model on the right."""
        w = surface.get_width()
        self.pg.draw.rect(surface, PANEL, (0, 0, w, 64))
        self.pg.draw.line(surface, LINE, (0, 64), (w, 64))
        x = self.text(surface, "FLYVERSE", (28, 14), 30, SAGE, bold=True, display=True).right
        x = self.text(surface, "/", (x + 12, 14), 30, DIM, display=True).right
        x = self.text(surface, title.upper(), (x + 12, 14), 30, TEXT, bold=True, display=True).right
        if subtitle:
            self.text(surface, subtitle, (x + 22, 25), 17, MUTED)
        if right:
            self.text(surface, right, (w - 28, 25), 17, MUTED, anchor="topright")

    def footer(self, surface, laws=()):
        """The provenance legend along the bottom: the three chips, then each declared law in one line."""
        w, h = surface.get_size()
        y = h - 40
        self.pg.draw.rect(surface, PANEL, (0, y - 8, w, 48))
        self.pg.draw.line(surface, LINE, (0, y - 8), (w, y - 8))
        x = 28
        for kind in ("connectome", "decoder", "game"):
            x = self.chip(surface, kind, (x, y + 4), 12).right + 18
        text = "   ".join(laws)
        self.text(surface, text, (x + 8, y + 7), 15, DIM)


def eye_colors(radiance, mode="human", exposure=2.5) -> np.ndarray:
    """(n_col, 4) radiance -> (n_col, 3) uint8 for `Hud.mosaic`: 'human' drops UV (what we would see through those
    1,466 pixels), 'fly' is the console's UV / B / G false colour."""
    from flyverse import world
    rad = radiance if isinstance(radiance, torch.Tensor) else torch.as_tensor(np.asarray(radiance, np.float32))
    rad = rad.detach().float().cpu()
    return (world.to_rgb8 if mode == "human" else world.to_fly_false_color)(rad, exposure=exposure).astype(np.uint8)


def model_line(fb) -> str:
    """'MaleCNS v1.0 · 167,106 neurons · raw' -- the right side of the header."""
    ds = {"malecns": "MaleCNS", "fafb": "FlyWire FAFB", "banc": "BANC"}.get(fb.c.dataset, str(fb.c.dataset))
    return f"{ds} {fb.c.release} · {fb.c.n:,} neurons · {fb.preset}"


# ---------------------------------------------------------------------------------------------------- run log
def git_commit() -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def source_hashes(paths) -> dict:
    """sha256 (first 16 hex) of each file, keyed by its path relative to the repo: a clip recorded from an
    uncommitted tree is still traceable to the exact game code that made it."""
    import hashlib
    out = {}
    for p in paths:
        p = Path(p).resolve()
        if p.is_file():
            key = p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.name
            out[key] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


class RunLog:
    """What a clip's caption is written from: command, commit, device, seed, attached modules, declared mappings
    and the game's own events (t in brain seconds). Events are facts about the run, not interpretations."""

    def __init__(self, game: str, args=None):
        dev = pick_device(getattr(args, "device", None))
        self.meta = {"game": game, "argv": [Path(sys.argv[0]).as_posix()] + sys.argv[1:],
                     "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                     **git_commit(), "python": platform.python_version(), "torch": torch.__version__,
                     "device": dev, "device_name": torch.cuda.get_device_name(0) if dev.startswith("cuda") else platform.processor(),
                     "seed": getattr(args, "seed", None), "uv_from_rgb": "UV = 0.5 * B wherever a scene only has RGB",
                     "sources": source_hashes([sys.argv[0], __file__])}
        self.events = []
        self.summary = {}

    def event(self, t_s: float, kind: str, **detail):
        self.events.append({"t_s": round(float(t_s), 4), "kind": kind, **detail})

    def attach_model(self, fb, name="brain"):
        """Record a brain: dataset, release, size, preset, instruments and every attached module."""
        self.meta.setdefault("brains", {})[name] = {
            "dataset": fb.c.dataset, "release": fb.c.release, "neurons": int(fb.c.n), "preset": fb.preset,
            "instruments": sorted(fb.instruments), "modules": fb.module_records(), "senses": list(fb.available_senses)}

    def save(self, path, fbs=None):
        for name, fb in (fbs or {}).items():
            self.attach_model(fb, name)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = {"meta": self.meta, "summary": self.summary, "events": self.events}
        path.write_text(json.dumps(out, indent=1, default=_json_default), encoding="utf-8")
        return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return str(o)


# ---------------------------------------------------------------------------------------------------- recording
def find_ffmpeg():
    """ffmpeg on PATH, else the static binary the optional `imageio-ffmpeg` package ships (it includes libx264)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


class Recorder:
    """Pipe RGB frames into ffmpeg: H.264, yuv420p, CRF 16, +faststart (an .mp4 any player and X accept)."""

    def __init__(self, path, size, fps, crf=16):
        ffmpeg = find_ffmpeg()
        if ffmpeg is None:
            raise SystemExit("--record needs ffmpeg on PATH (or `pip install imageio-ffmpeg`)")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        w, h = size
        self.proc = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
             "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(self.path)], stdin=subprocess.PIPE)
        self.frames = 0

    def write(self, surface):
        import pygame
        self.proc.stdin.write(pygame.image.tobytes(surface, "RGB"))
        self.frames += 1

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")


# ---------------------------------------------------------------------------------------------------- the loop
class Game:
    """Subclass and fill in. `tick()` advances the world and the brain by exactly TICK_MS of brain time (call
    `fb.step(TICK_MS)` once inside it); `draw(surface)` paints the whole canvas; `handle(event)` takes pygame
    events in interactive mode (canvas coordinates are provided by `run`). `t_s` is brain time in seconds.
    Put facts in `self.log.event(...)` as they happen and a dict in `self.log.summary` by the end."""

    title = "game"
    subtitle = ""

    def __init__(self, args):
        self.args = args
        self.log = RunLog(self.title, args)
        self.t_s = 0.0
        self.hud = None                           # set by run() once pygame is up

    def brains(self) -> dict:
        """{name: FlyBrain} for the run log."""
        return {}

    def tick(self):
        raise NotImplementedError

    def draw(self, surface):
        raise NotImplementedError

    def handle(self, event, canvas_pos=None):
        pass

    def finished(self) -> bool:
        return False


def run(game: Game, args):
    """Interactive window, or headless recording when `--record` is given. Returns the run log path (or None)."""
    headless = args.record is not None
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    # SDL otherwise turns SIGTERM into a quit event that a headless loop never reads, so a job scheduler's
    # preemption leaves the process running beside its retry (two writers on one clip). Let SIGTERM terminate.
    os.environ.setdefault("SDL_NO_SIGNAL_HANDLERS", "1")
    import pygame
    pygame.init()
    size = parse_size(args.size)
    canvas = pygame.Surface(size)
    game.hud = Hud()
    ticks_per_frame = 1000 / args.fps / TICK_MS
    if abs(ticks_per_frame - round(ticks_per_frame)) > 1e-9:
        raise SystemExit("fps must make each video frame a whole number of ticks")
    ticks_per_frame = int(round(ticks_per_frame))
    log_path = args.log or (str(Path(args.record).with_suffix(".json")) if headless else None)
    t_wall = time.time()
    if headless:
        rec = Recorder(args.record, size, args.fps)
        n_frames = int(round(args.seconds * args.fps))
        try:
            for f in range(n_frames):
                for _ in range(ticks_per_frame):
                    game.tick()
                game.draw(canvas)
                rec.write(canvas)
                if f % args.fps == 0:
                    print(f"  {game.title}: {f / args.fps:5.1f} / {args.seconds:.1f} s brain, "
                          f"{time.time() - t_wall:6.1f} s wall", flush=True)
                if game.finished():
                    break
        finally:
            rec.close()
        print(f"wrote {rec.path} ({rec.frames} frames, {rec.frames / args.fps:.2f} s) in {time.time() - t_wall:.0f} s wall")
    else:
        window = pygame.display.set_mode(parse_size(args.window), pygame.RESIZABLE)
        pygame.display.set_caption(f"flyverse / {game.title}")
        clock = pygame.time.Clock()
        running = True
        while running and not game.finished():
            wsize = window.get_size()
            scale = min(wsize[0] / size[0], wsize[1] / size[1])
            ox, oy = (wsize[0] - size[0] * scale) / 2, (wsize[1] - size[1] * scale) / 2
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    running = False
                pos = getattr(ev, "pos", None)
                cpos = None if pos is None else ((pos[0] - ox) / scale, (pos[1] - oy) / scale)
                game.handle(ev, cpos)
            for _ in range(ticks_per_frame):
                game.tick()
            game.draw(canvas)
            window.fill(BG)
            window.blit(pygame.transform.smoothscale(canvas, (int(size[0] * scale), int(size[1] * scale))), (ox, oy))
            pygame.display.flip()
            clock.tick(args.fps)                  # never faster than real time; slower if the brain is
    if args.screenshot:
        Path(args.screenshot).parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(canvas, args.screenshot)
    if log_path:
        game.log.meta["wall_s"] = round(time.time() - t_wall, 1)
        game.log.meta["brain_s"] = round(game.t_s, 3)
        p = game.log.save(log_path, game.brains())
        print(f"run log {p}")
        return p
    return None
