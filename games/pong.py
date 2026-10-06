"""pong -- the first game of Pong between two connectomes.

MaleCNS v1.0 (a male CNS, `FlyBrain()`) against FlyWire FAFB v783 (a female brain with the complete optic lobe,
`FlyBrain(dataset="fafb")`), both `preset="raw"`. Only the eyes are used; FAFB has no VNC and nothing here reads a
motor neuron. Not a sex comparison: the two reconstructions differ in how many photoreceptors face the court (the HUD
shows the counts), in proofreading and in naming.

What is what (each is declared in the run log and named in the footer):

  GAME        The court. Each fly watches its own LED arena: the court fills az +-50 deg x el +-35 deg of its field, its
              own end at the bottom (el -35), the far end at the top; a bright disc 12 deg across (about three
              ommatidia) is the ball. The two arenas are the same court seen from opposite ends. The flies do NOT see
              the paddles (measured on dev seeds: a visible, moving paddle becomes the decoder's peak -- see the
              caption). Ball physics (a return is never flatter than MIN_ANGLE), serves (a fixed schedule, the same for
              every seed), the paddle speed limit, the paddle that missed stopping until the point, the score.
  DECODER     Per fly, read-only, from the optic lobe's rate units: L2 (lamina) + Mi1 (medulla) cells whose retinotopic
              column (`interp.trace.column_of_cells`, the fly's own map) looks at the court. Per cell the high-pass
              |dr - EMA(dr, 0.15 s)|; the 12 strongest cells; the densest cluster among them (cells within 12 deg of one
              another, weighted by activity); its weighted mean azimuth is this tick's reading. The paddle target
              follows the reading (EMA 50 ms) while the strongest cell exceeds a gate, and holds otherwise.
  CONNECTOME  What the reconstruction and the model give under the ball: how many reconstructed photoreceptors look at
              it, and the mean rate deviation dr of the L2 cells whose column looks at it.

Run:  python games/pong.py                         (fly vs fly, interactive)
      python games/pong.py --left human             (you, W/S or the mouse, against FAFB)
      python games/pong.py --record out/games/pong/seed100.mp4 --seed 100 --seconds 40
"""
from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402
from common import (AMBER, BG, DIM, INSET, LILAC, LINE, MUTED, PANEL, RED, SAGE, TEAL, TEXT, WHITE)  # noqa: E402

import torch  # noqa: E402

# ---------------------------------------------------------------------------------------------------- the court (GAME)
# Court coordinates are degrees of fly A's arena: X along the court (0 = A's end line, 70 = B's), Y across it
# (+50 = A's left = the top of the screen). Fly A sees (az, el) = (Y, X - 35); fly B, facing it, (-Y, 35 - X).
COURT_LEN = 70.0
HALF_W = 50.0
PAD_X = (3.0, 67.0)          # paddle lines
PAD_HALF = 12.0              # paddle half-length (deg of Y)
PAD_THICK = 0.9              # paddle half-thickness (deg of X)
PAD_SPEED = 80.0             # deg / s: the paddle's GAME speed limit
BALL_R_EYE = 6.0             # the ball the flies see: a disc of this angular radius (deg)
BALL_RY = 6.0                # the ball's collision radius across the court (deg of Y) = the main view's ellipse
BALL_RX = 1.8                # ... and along it (deg of X): the main view's X scale is about 3.3x its Y scale
V0 = 52.0                    # serve speed (deg / s)
SPEEDUP = 1.15               # x speed per return
VMAX = 200.0                 # speed cap (deg / s)
ANGLE_KEEP = 0.5             # fraction of the incoming angle a return keeps
ENGLISH = 35.0               # deg added to the ball's angle for a hit on the paddle's tip (scaled by the hit offset)
MIN_ANGLE = 20.0             # |angle| floor for a return (deg): a flatter return is steepened to this, same direction
MAX_ANGLE = 55.0             # |angle| cap (deg from the court axis)
SERVE_PAUSE = 1.0            # s between a point and the next serve (the ball is hidden)
FIRST_SERVE = 1.5            # s of clip before the first serve (the title card; the flies watch the empty court)
GAME_OVER_HOLD = 2.0         # s the final banner stays up when --points ends the game
# The serve schedule (GAME): the same for every seed, cycled. (receiver, angle deg, start Y deg); the ball is put in
# play at the server's end line.
SERVES = [("A", 22.0, 8.0), ("B", -28.0, -6.0), ("A", -14.0, 12.0), ("B", 32.0, 4.0), ("A", 30.0, -10.0),
          ("B", -18.0, 10.0), ("A", -34.0, 0.0), ("B", 12.0, -12.0), ("A", 16.0, 6.0), ("B", -30.0, 8.0),
          ("A", 26.0, -4.0), ("B", 20.0, -8.0), ("A", -24.0, 10.0), ("B", -12.0, 2.0), ("A", 8.0, -12.0),
          ("B", 34.0, 12.0)]


def view_of(side: str, X: float, Y: float) -> tuple[float, float]:
    """Court position -> (azimuth, elevation) in that fly's arena (deg; azimuth + = the fly's left)."""
    return (Y, X - COURT_LEN / 2) if side == "A" else (-Y, COURT_LEN / 2 - X)


def court_y_of(side: str, az: float) -> float:
    """A decoded azimuth -> the court's Y (the paddle axis)."""
    return az if side == "A" else -az


def face_x(side: str) -> float:
    """The ball centre's X when it touches that side's paddle."""
    return PAD_X[0] + PAD_THICK + BALL_RX if side == "A" else PAD_X[1] - PAD_THICK - BALL_RX


def other(side: str) -> str:
    return "B" if side == "A" else "A"


class Court:
    """Pong physics in court degrees (GAME). Pure Python: no brain, no drawing. Parameters are read from the
    module's constants when the court is made."""

    def __init__(self, serves=None):
        self.serves = list(SERVES if serves is None else serves)
        self.pad_half, self.pad_speed = PAD_HALF, PAD_SPEED
        self.v0, self.speedup, self.vmax = V0, SPEEDUP, VMAX
        self.min_angle, self.max_angle = MIN_ANGLE, MAX_ANGLE
        self.angle_keep, self.english = ANGLE_KEEP, ENGLISH
        self.paddle = {"A": 0.0, "B": 0.0}
        self.score = {"A": 0, "B": 0}
        self.X, self.Y, self.vX, self.vY, self.speed = COURT_LEN / 2, 0.0, 0.0, 0.0, 0.0
        self.live = False
        self.serving = True              # False once the game is decided (--points): no more serves
        self.missed = None               # the side that missed, until the point: its paddle stays where it was
        self.next_serve_t = FIRST_SERVE
        self.n_serves = 0
        self.rally = 0                   # returns in the current point
        self.rallies = []                # returns per finished point

    @property
    def paddle_limit(self) -> float:
        """The paddle centre's |Y| limit: the paddle stays on the court."""
        return HALF_W - self.pad_half

    # -------------------------------------------------------------- paddles
    def move_paddle(self, side: str, target: float, dt: float):
        if self.missed == side:                                   # the point is decided; the paddle stops
            return
        lim = self.paddle_limit
        target = float(np.clip(target, -lim, lim))
        p = self.paddle[side]
        step = self.pad_speed * dt
        self.paddle[side] = float(np.clip(p + np.clip(target - p, -step, step), -lim, lim))

    # -------------------------------------------------------------- ball
    def serve(self):
        receiver, angle, y0 = self.serves[self.n_serves % len(self.serves)]
        self.n_serves += 1
        server = other(receiver)
        self.X, self.Y, self.speed = face_x(server), float(y0), self.v0
        sgn = -1.0 if receiver == "A" else 1.0
        a = math.radians(angle)
        self.vX, self.vY = sgn * self.speed * math.cos(a), self.speed * math.sin(a)
        self.live, self.rally, self.missed = True, 0, None
        return {"kind": "serve", "toward": receiver, "angle_deg": angle, "y0": y0, "n": self.n_serves}

    def return_angle(self, a_in: float, offset: float) -> tuple[float, bool]:
        """The outgoing angle (deg) for an incoming angle `a_in` and a hit offset in -1 .. 1: part of the incoming
        angle plus english from the offset, capped at max_angle; a return flatter than min_angle is steepened to
        min_angle in the direction it was going (the offset's, then away from the nearer side wall, if exactly flat).
        Returns (angle, steepened)."""
        a = float(np.clip(self.angle_keep * a_in + self.english * float(np.clip(offset, -1, 1)),
                          -self.max_angle, self.max_angle))
        if abs(a) >= self.min_angle:
            return a, False
        if a != 0.0:
            sgn = math.copysign(1.0, a)
        elif offset != 0.0:
            sgn = math.copysign(1.0, offset)
        else:
            sgn = -math.copysign(1.0, self.Y) if self.Y != 0.0 else 1.0
        return sgn * self.min_angle, True

    def _return(self, side: str, offset: float) -> tuple[float, bool]:
        self.speed = min(self.speed * self.speedup, self.vmax)
        a_in = math.degrees(math.atan2(self.vY, abs(self.vX)))
        a_deg, steep = self.return_angle(a_in, offset)
        a = math.radians(a_deg)
        sgn = 1.0 if side == "A" else -1.0
        self.vX, self.vY = sgn * self.speed * math.cos(a), self.speed * math.sin(a)
        self.rally += 1
        return a_deg, steep

    def step(self, t: float, dt: float) -> list:
        """Advance the ball by dt (s) at brain time t; returns this step's events."""
        ev = []
        if not self.live:
            if self.serving and t + 1e-9 >= self.next_serve_t:
                ev.append(self.serve())
            return ev
        X0 = self.X
        self.X += self.vX * dt
        self.Y += self.vY * dt
        lim = HALF_W - BALL_RY
        if abs(self.Y) > lim:                                     # side walls
            self.Y = math.copysign(2 * lim - abs(self.Y), self.Y)
            self.vY = -self.vY
            ev.append({"kind": "wall"})
        for side, sgn in (("A", -1.0), ("B", 1.0)):
            if self.vX * sgn <= 0 or self.missed is not None:
                continue
            face = face_x(side)
            if not ((X0 - face) * sgn <= 0 < (self.X - face) * sgn):
                continue
            d = self.Y - self.paddle[side]
            reach = self.pad_half + BALL_RY
            if abs(d) <= reach:
                self.X = face - (self.X - face)                   # reflect the overshoot
                a_deg, steep = self._return(side, d / reach)
                ev.append({"kind": "return", "side": side, "offset": round(d / reach, 3),
                           "speed_deg_s": round(self.speed, 1), "rally": self.rally, "angle_deg": round(a_deg, 1),
                           "steepened": steep})
            else:
                self.missed = side
                ev.append({"kind": "miss", "side": side, "by_deg": round(abs(d) - reach, 2),
                           "ball_vy_deg_s": round(self.vY, 1)})
        if self.X < -BALL_RX or self.X > COURT_LEN + BALL_RX:     # out: a point
            scorer = "B" if self.X < 0 else "A"
            self.score[scorer] += 1
            self.live = False
            self.next_serve_t = t + SERVE_PAUSE
            self.rallies.append(self.rally)
            ev.append({"kind": "point", "scorer": scorer, "score": dict(self.score), "rally": self.rally})
        return ev


def game_floor(seconds: float, dt: float = 0.01) -> dict:
    """GAME only, no brain: both paddles frozen at the centre, the same serve schedule and rules, for `seconds` of
    clip. Deterministic. The floor that a run's return and miss counts sit on (paddles with no ball information)."""
    c = Court()
    out = {"returns": {"A": 0, "B": 0}, "misses": {"A": 0, "B": 0}}
    tally = {"return": "returns", "miss": "misses"}
    for i in range(int(round(seconds / dt))):
        for e in c.step(i * dt, dt):
            if e["kind"] in tally:
                out[tally[e["kind"]]][e["side"]] += 1
    out.update(seconds=seconds, score=dict(c.score), returns_per_point=list(c.rallies))
    return out


# ---------------------------------------------------------------------------------------------------- the arena (GAME)
# Radiance [UV, B, G, R]. UV is given explicitly (no RGB source here).
RAD_OUTSIDE = np.array((0.015, 0.025, 0.03, 0.03), np.float32)
RAD_COURT = np.array((0.05, 0.12, 0.16, 0.12), np.float32)
RAD_LINE = np.array((0.16, 0.30, 0.34, 0.30), np.float32)
RAD_BALL = np.array((0.80, 1.00, 1.00, 1.00), np.float32)


class Arena:
    """What one fly sees: the court as an LED arena in its own field (GAME). `radiance(ball)` returns the per-ray
    radiance (n_col * k, 4); `Eyes.pool` turns it into what `fb.vision` gets."""

    def __init__(self, dirs_body: np.ndarray):
        d = np.asarray(dirs_body, np.float32).reshape(-1, 3)
        self.az = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
        self.el = np.degrees(np.arcsin(np.clip(d[:, 2], -1, 1)))
        self.front = d[:, 0] > 0.05
        self.cos_el = np.cos(np.radians(self.el))
        base = np.tile(RAD_OUTSIDE, (len(d), 1))
        court = self.front & (np.abs(self.az) <= HALF_W) & (np.abs(self.el) <= COURT_LEN / 2)
        base[court] = RAD_COURT
        edge = court & ((np.abs(self.az) >= HALF_W - 1.2) | (np.abs(self.el) >= COURT_LEN / 2 - 1.0))
        net = court & (np.abs(self.el) <= 0.6) & ((np.floor((self.az + HALF_W) / 5.0) % 2) == 0)
        base[edge | net] = RAD_LINE
        self.base = base

    def radiance(self, ball=None) -> np.ndarray:
        """`ball` = (az, el) of its centre in this fly's field, or None when no ball is in play."""
        rad = self.base.copy()
        if ball is not None:
            dist = np.hypot((self.az - ball[0]) * self.cos_el, self.el - ball[1])
            alpha = np.clip(BALL_R_EYE + 0.5 - dist, 0.0, 1.0)[:, None] * self.front[:, None]   # 1 deg soft edge
            m = alpha[:, 0] > 0
            rad[m] = rad[m] * (1 - alpha[m]) + RAD_BALL * alpha[m]
        return rad


# ---------------------------------------------------------------------------------------------------- the decoder
DECODER_TYPES = ("L2", "Mi1")
DECODER_PARAMS = {"types": list(DECODER_TYPES), "highpass_tau_s": 0.15, "top_k": 12, "cluster_deg": 12.0,
                  "gate": 0.40, "target_tau_s": 0.05, "window_deg": [HALF_W + 5, COURT_LEN / 2 + 5]}
DECODER_LAW = ("paddle target = court position of the peak of high-passed L2 + Mi1: per cell |dr - EMA(dr, 0.15 s)|, "
               "top 12 cells, densest 12-deg cluster, activity-weighted mean azimuth of its columns (the reading), "
               "EMA 50 ms (the target); held while the top cell < 0.40")


class BallDecoder:
    """DECODER (read-only): where is the ball, from one fly's L2 / Mi1 columns. Reads rate-unit deviations
    `dr` (the optic lobe's `last["dr"]` row), writes nothing.

    `pos` indexes the decoder's cells in `dr`; `cell_az` / `cell_el` are their columns' directions (deg).
    `az_hat` is this tick's reading; `target_az` is what the paddle follows (the reading through a 50 ms EMA, held
    while the strongest cell is below the gate)."""

    def __init__(self, pos, cell_az, cell_el, *, highpass_tau_s=0.15, top_k=12, cluster_deg=12.0, gate=0.40,
                 target_tau_s=0.05, device="cpu", **_):
        dev = torch.device(device)
        self.pos = torch.as_tensor(np.asarray(pos), dtype=torch.long, device=dev)
        self.az = torch.as_tensor(np.asarray(cell_az, np.float32), device=dev)
        self.el = torch.as_tensor(np.asarray(cell_el, np.float32), device=dev)
        self.tau, self.k, self.cluster, self.gate, self.target_tau = highpass_tau_s, int(top_k), cluster_deg, gate, target_tau_s
        self.ema = None
        self.hp = None
        self.az_hat = self.el_hat = float("nan")     # this tick's reading
        self.conf = 0.0                               # the strongest cell's |dr - EMA|
        self.valid = False
        self.target_az = 0.0                          # the smoothed, gated reading the paddle follows
        self.target_el = 0.0

    @property
    def n(self) -> int:
        return int(self.pos.numel())

    def reset_target(self):
        self.target_az = self.target_el = 0.0

    def update(self, dr: torch.Tensor, dt_s: float):
        x = dr[self.pos]
        if self.ema is None:
            self.ema = x.clone()
        hp = (x - self.ema).abs()
        a = math.exp(-dt_s / self.tau)
        self.ema.mul_(a).add_((1 - a) * x)
        self.hp = hp
        k = min(self.k, hp.numel())
        v, i = hp.topk(k)
        pa, pe = self.az[i], self.el[i]
        near = (torch.hypot(pa[:, None] - pa[None, :], pe[:, None] - pe[None, :]) <= self.cluster).to(v.dtype)
        score = near @ v
        j = int(score.argmax())
        w = near[j] * v
        s = float(w.sum())
        self.conf = float(v[0])
        if s <= 0.0:
            self.valid = False
            return self
        self.az_hat, self.el_hat = float((w * pa).sum()) / s, float((w * pe).sum()) / s
        self.valid = self.conf >= self.gate
        if self.valid:
            b = 1.0 - math.exp(-dt_s / self.target_tau)
            self.target_az += (self.az_hat - self.target_az) * b
            self.target_el += (self.el_hat - self.target_el) * b
        return self

    @classmethod
    def for_brain(cls, fb, params=None):
        """The decoder's cells on this brain: L2 / Mi1 rate units whose column (the brain's own map) is in the window."""
        from flyverse.interp.trace import column_of_cells
        params = DECODER_PARAMS if params is None else params
        ol, r = fb.optic, fb.retina
        col, _ = column_of_cells(fb.c, r, ol.rate_idx)
        types = fb.c.neurons.type.fillna("").to_numpy()
        cells = np.flatnonzero(np.isin(types, params["types"]))
        cols = col[cells]
        ok = cols >= 0
        cells, cols = cells[ok], cols[ok]
        az, el = r.col_az_el[cols, 0], r.col_az_el[cols, 1]
        win = (np.abs(az) <= params["window_deg"][0]) & (np.abs(el) <= params["window_deg"][1])
        cells, cols, az, el = cells[win], cols[win], az[win], el[win]
        pos = np.searchsorted(ol.rate_idx, cells)
        assert (ol.rate_idx[pos] == cells).all()
        dec = cls(pos, az, el, device=fb.device, **{k: v for k, v in params.items() if k not in ("types", "window_deg")})
        dec.cells, dec.columns, dec.types = cells, cols, types[cells]
        return dec


# ---------------------------------------------------------------------------------------------------- measures
UNDER_DEG = BALL_R_EYE                    # "under the ball": columns whose direction lies within the ball's disc
PR_BINS = ((0, 0), (1, 3), (4, 8), (9, 20), (21, 10 ** 9))   # photoreceptors under the ball, for the error breakdown


def pr_bin_label(lo: int, hi: int) -> str:
    return str(lo) if lo == hi else (f"{lo}+" if hi >= 10 ** 9 else f"{lo}-{hi}")


def angular_dist(az, el, ball) -> np.ndarray:
    """Degrees between column directions (az, el) and the ball's centre (small-angle, cos(el)-scaled azimuth)."""
    return np.hypot((np.asarray(az) - ball[0]) * math.cos(math.radians(ball[1])), np.asarray(el) - ball[1])


def err_stats(err, mask=None) -> dict:
    """{median, p90, n} of |error| samples (nan ignored), optionally where `mask` is True."""
    e = np.asarray(err, np.float64)
    if mask is not None:
        e = e[np.asarray(mask, bool)]
    e = e[np.isfinite(e)]
    if not len(e):
        return {"median": None, "p90": None, "n": 0}
    return {"median": round(float(np.median(e)), 2), "p90": round(float(np.percentile(e, 90)), 2), "n": int(len(e))}


# ---------------------------------------------------------------------------------------------------- players
FLIES = {"malecns": {"dataset": None, "name": "MaleCNS v1.0", "short": "MaleCNS"},
         "fafb": {"dataset": "fafb", "name": "FlyWire FAFB v783", "short": "FAFB"}}
RIBBON_BINS = 26                          # azimuth bins of 4 deg across the court (+-52 deg)
RIBBON_SCALE = 0.8                        # |dr - EMA| shown at full brightness
HISTORY = 160                             # frames kept for ribbons and traces (3.2 s at 50 fps)


class Player:
    def __init__(self, side: str, kind: str, color, args):
        self.side, self.kind, self.color = side, kind, color
        self.fb = self.eyes = self.arena = self.decoder = None
        self.radiance = None
        self.ribbon = deque(maxlen=HISTORY)       # per frame: (RIBBON_BINS,) max |dr - EMA| of L2 + Mi1 by azimuth
        self.ribbon_dec = deque(maxlen=HISTORY)   # per frame: the decoder's target, court Y
        self.ribbon_ball = deque(maxlen=HISTORY)  # per frame: the ball's court Y (nan if none)
        self.conf_hist = deque(maxlen=HISTORY)
        self.target_err_hist = deque(maxlen=HISTORY)   # per frame: |target - ball| (deg) while the ball is live
        self.under = (0, 0, 0.0)                  # CONNECTOME: photoreceptors, L2 cells under the ball, their mean dr
        # per live tick (the summary is computed from these)
        self.err_target = []                      # |target az - ball az| (deg): what the paddle follows
        self.err_reading = []                     # |reading az - ball az| (deg): this tick's reading, before EMA + gate
        self.own_half = []                        # the ball is in this fly's half
        self.pr_under = []                        # reconstructed photoreceptors under the ball
        self.gated = 0
        self.ticks_live = 0
        self.ticks_no_l2 = 0                      # live ticks with no L2 column under the ball
        self.ticks_at_clamp = 0                   # live ticks with the paddle at the court edge
        self.returns = self.misses = 0
        self.human_target = 0.0
        if kind == "human":
            self.name = self.short = "HUMAN"
            self.key = f"{side}_human"
            return
        info = FLIES[kind]
        self.name, self.short = info["name"], info["short"]
        self.key = f"{side}_{self.short}"
        kw = {"dataset": info["dataset"]} if info["dataset"] else {}
        self.fb = common.build_brain(args, **kw)
        self.eyes = common.Eyes(self.fb)
        self.arena = Arena(self.eyes.dirs_body)
        self.decoder = BallDecoder.for_brain(self.fb)
        d = self.decoder
        bins = np.floor((d.az.cpu().numpy() + 52.0) / 4.0).astype(np.int64)
        self.ribbon_bin = torch.as_tensor(np.clip(bins, 0, RIBBON_BINS - 1), device=self.fb.device)
        l2 = np.flatnonzero(d.types == "L2")
        self.l2_pos = d.pos[torch.as_tensor(l2, device=d.pos.device)]
        self.l2_az, self.l2_el = d.az.cpu().numpy()[l2], d.el.cpu().numpy()[l2]
        r = self.fb.retina
        self.no_pr = np.asarray(r.coverage()["without_photoreceptors_indices"], np.int64)
        npr = np.bincount(np.asarray(r.pr_column, np.int64), minlength=r.n_columns)
        caz, cel = np.asarray(r.col_az_el[:, 0]), np.asarray(r.col_az_el[:, 1])
        court = (np.abs(caz) <= HALF_W) & (np.abs(cel) <= COURT_LEN / 2)
        self.coverage = {"court_window_deg": [HALF_W, COURT_LEN / 2], "columns": int(court.sum()),
                         "columns_with_photoreceptors": int((court & (npr > 0)).sum()),
                         "photoreceptors": int(npr[court].sum()),
                         "photoreceptors_per_column_median": float(np.median(npr[court])) if court.any() else 0.0}
        self.pr_court = self.coverage["photoreceptors"]
        near = (np.abs(caz) <= HALF_W + UNDER_DEG + 3) & (np.abs(cel) <= COURT_LEN / 2 + UNDER_DEG + 3) & (npr > 0)
        self.pr_az, self.pr_el, self.pr_n = caz[near], cel[near], npr[near]

    @property
    def is_fly(self) -> bool:
        return self.fb is not None

    def see(self, ball):
        """Hand the arena to the eyes: `ball` = (az, el) in this fly's field or None."""
        rays = self.arena.radiance(ball)
        self.radiance = self.eyes.pool(rays)
        self.fb.vision(self.radiance)

    def measure_under(self, ball):
        """CONNECTOME, for the HUD and the error breakdown (drives nothing): under the ball's disc (columns within
        UNDER_DEG of its centre), the number of reconstructed photoreceptors, the number of L2 cells, and their mean
        rate deviation dr (L2 hyperpolarises under a bright ball, so this is negative)."""
        if ball is None:
            self.under = (0, 0, 0.0)
            return self.under
        n_pr = int(self.pr_n[angular_dist(self.pr_az, self.pr_el, ball) <= UNDER_DEG].sum())
        m = np.flatnonzero(angular_dist(self.l2_az, self.l2_el, ball) <= UNDER_DEG)
        if not len(m):
            self.under = (n_pr, 0, 0.0)
            return self.under
        dr = self.fb.optic.last["dr"][0][self.l2_pos[torch.as_tensor(m, device=self.l2_pos.device)]]
        self.under = (n_pr, len(m), float(dr.mean()))
        return self.under

    def ribbon_sample(self) -> np.ndarray:
        """The decoder's input by azimuth: max |dr - EMA| of its L2 + Mi1 cells per 4-deg azimuth bin."""
        hp = self.decoder.hp
        out = torch.zeros(RIBBON_BINS, device=hp.device).scatter_reduce(0, self.ribbon_bin, hp, "amax", include_self=True)
        return out.cpu().numpy()

    def summary(self) -> dict:
        d = {"side": self.side, "returns": self.returns, "misses": self.misses}
        if not self.is_fly:
            return d
        own = np.asarray(self.own_half, bool)
        pr = np.asarray(self.pr_under, np.int64)
        t_all, t_own = err_stats(self.err_target), err_stats(self.err_target, own)
        r_all, r_own = err_stats(self.err_reading), err_stats(self.err_reading, own)
        by_pr = {}
        for lo, hi in PR_BINS:
            m = (pr >= lo) & (pr <= hi)
            by_pr[pr_bin_label(lo, hi)] = {"ticks": int(m.sum()), "target_median": err_stats(self.err_target, m)["median"],
                                           "reading_median": err_stats(self.err_reading, m)["median"]}
        live = max(self.ticks_live, 1)
        d.update({
            "decoder_cells": self.decoder.n, "court_coverage": self.coverage,
            "decoder_target_abs_err_deg_median": t_all["median"], "decoder_target_abs_err_deg_p90": t_all["p90"],
            "decoder_target_abs_err_own_half_median": t_own["median"], "decoder_target_abs_err_own_half_p90": t_own["p90"],
            "decoder_reading_abs_err_deg_median": r_all["median"], "decoder_reading_abs_err_deg_p90": r_all["p90"],
            "decoder_reading_abs_err_own_half_median": r_own["median"], "decoder_reading_abs_err_own_half_p90": r_own["p90"],
            "decoder_err_by_photoreceptors_under_ball": by_pr,
            "decoder_held_fraction": round(self.gated / live, 3),
            "ball_over_no_l2_column_fraction": round(self.ticks_no_l2 / live, 3),
            "paddle_at_court_edge_fraction": round(self.ticks_at_clamp / live, 3),
            "live_ticks": self.ticks_live})
        return d


# ---------------------------------------------------------------------------------------------------- drawing helpers
COURT_BG = (6, 8, 9)
NOT_SEX = "not a sex comparison: one male and one female reconstruction, with different photoreceptor coverage"


def blend(c1, c2, a):
    return tuple(int(round(x * (1 - a) + y * a)) for x, y in zip(c1, c2))


def colormap(v, color):
    """(..., ) values in [0, 1] -> (..., 3) uint8 from near-black through the fly's colour to white."""
    v = np.clip(np.asarray(v, np.float32), 0, 1)[..., None]
    lo = np.array(INSET, np.float32)
    mid = np.array(color, np.float32)
    hi = np.array(WHITE, np.float32)
    out = np.where(v < 0.6, lo + (mid - lo) * (v / 0.6), mid + (hi - mid) * ((v - 0.6) / 0.4))
    return out.astype(np.uint8)


def fmt_deg(v: float) -> str:
    """Miss distances: one decimal under 2 deg (so a near miss never reads '0 deg'), whole degrees above."""
    return f"{v:.1f}°" if abs(v) < 2 else f"{v:.0f}°"


def truncate_polyline(pts, max_len):
    """The last part of a polyline (oldest first), at most `max_len` long, measured back from its newest point."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 2:
        return pts
    out = [pts[-1]]
    left = float(max_len)
    for i in range(len(pts) - 1, 0, -1):
        seg = pts[i - 1] - pts[i]
        L = float(np.hypot(*seg))
        if L >= left:
            out.append(pts[i] + seg * (left / L if L > 0 else 0.0))
            break
        out.append(pts[i - 1])
        left -= L
    return np.asarray(out[::-1])


class FrontalMosaic:
    """The frontal part of the fly's-eye view (WHAT THE FLY SEES): one hexagon per ommatidial column at its true
    azimuth / elevation (both eyes, so the binocular strip overlaps), cropped to +-az_half x +-el_half deg. Columns
    without a reconstructed photoreceptor (`retina.coverage()`) are drawn dark: no light reaches the model there."""

    def __init__(self, eyes, no_pr=(), az_half=55.0, el_half=40.0):
        self.eyes, self.az_half, self.el_half = eyes, az_half, el_half
        az, el = eyes.az_el[:, 0], eyes.az_el[:, 1]
        self.cols = np.flatnonzero((np.abs(az) <= az_half + 3) & (np.abs(el) <= el_half + 3))
        self.dark = np.isin(self.cols, np.asarray(no_pr, np.int64))
        self._cache = {}

    def fit(self, rect):
        """The mosaic's own rect inside `rect`: centred, the window's aspect ratio."""
        scale = min(rect[2] / (2 * self.az_half), rect[3] / (2 * self.el_half))
        w, h = 2 * self.az_half * scale, 2 * self.el_half * scale
        return (int(rect[0] + (rect[2] - w) / 2), int(rect[1] + (rect[3] - h) / 2), int(w), int(h))

    def layout(self, rect):
        key = tuple(int(v) for v in rect)
        if key not in self._cache:
            scale = min(rect[2] / (2 * self.az_half), rect[3] / (2 * self.el_half))
            cx, cy = rect[0] + rect[2] / 2, rect[1] + rect[3] / 2
            az, el = self.eyes.az_el[self.cols, 0], self.eyes.az_el[self.cols, 1]
            x = cx - az * scale
            y = cy - el * scale
            radius = max(1.5, scale * 4.6 / math.sqrt(3) * 1.13)
            ang = np.arange(6) * np.pi / 3 + np.pi / 6
            polys = np.round(np.stack([x, y], 1)[:, None, :] + radius * np.c_[np.cos(ang), np.sin(ang)][None]).astype(int)
            self._cache[key] = (polys.tolist(), scale, cx, cy)
        return self._cache[key]

    def to_px(self, rect, az, el):
        _, scale, cx, cy = self.layout(rect)
        return cx - az * scale, cy - el * scale

    def draw(self, pg, surface, rect, colors):
        polys, *_ = self.layout(rect)
        clip = surface.get_clip()
        surface.set_clip(pg.Rect(rect))
        pg.draw.rect(surface, (3, 4, 5), rect)
        cols = np.asarray(colors)[self.cols].tolist()
        for poly, c, dark in zip(polys, cols, self.dark):
            pg.draw.polygon(surface, (26, 30, 32) if dark else c, poly)
        surface.set_clip(clip)


# ---------------------------------------------------------------------------------------------------- the game
class PongGame(common.Game):
    title = "pong"
    subtitle = "the first game of Pong between two connectomes"

    def __init__(self, args):
        super().__init__(args)
        self.court = Court()
        self.control = args.control
        self.players = {"A": Player("A", args.left, SAGE, args), "B": Player("B", args.right, LILAC, args)}
        self.mixed = args.left != args.right and "human" not in (args.left, args.right)   # MaleCNS vs FAFB
        self.banners = []                     # dicts: kind, text, color, t0, size, subs, hold
        self.trail = deque(maxlen=12)
        self.flash = {"A": -9.0, "B": -9.0}   # last return time per side (paddle glow)
        self.last_miss = None                 # dict: side, Y, t, by, vy
        self.tick_n = 0
        self.best_rally = 0
        self.keys = set()                     # W/S/arrow keys held (interactive)
        self.game_over_t = None
        self.steepened = 0
        self.return_angles = []
        self.floor = game_floor(args.seconds)
        self._declare()
        self._layout()
        self._warmup(args.warmup)
        a, b = self.players["A"], self.players["B"]
        subs = ["each paddle follows a decoder reading its own fly's L2 + Mi1 columns"]
        if self.mixed:
            subs.append(f"not a sex comparison: {a.pr_court:,} vs {b.pr_court:,} reconstructed photoreceptors face the court")
        self.banner("title", f"{a.name}  vs  {b.name}", WHITE, 60, subs=subs, hold=0.7)

    # -------------------------------------------------------------- provenance
    def _declare(self):
        log, a = self.log, self.args
        log.meta["mode"] = {"left": a.left, "right": a.right, "control": a.control, "points": a.points,
                            "warmup_s": a.warmup}
        c = self.court
        common.declare(log, "court", "game", "Pong in court degrees (1 deg = 1 deg of each fly's arena): constant-speed "
                       "ball, reflecting side walls; a paddle returns the ball if |ball Y - paddle Y| <= paddle half + "
                       "ball radius when the ball reaches it: the return keeps part of the incoming angle plus english "
                       "proportional to the hit offset, never flatter than the minimum angle (steepened in the direction "
                       f"it was going) nor steeper than the cap, at x{c.speedup} speed (capped); otherwise a miss: the "
                       "paddle that missed stops until the ball leaves the court and the other side scores",
                       court_len_deg=COURT_LEN, half_width_deg=HALF_W, paddle_lines_deg=list(PAD_X),
                       paddle_half_deg=c.pad_half, ball_collision_radius_deg=[BALL_RX, BALL_RY], serve_speed_deg_s=c.v0,
                       speedup=c.speedup, max_speed_deg_s=c.vmax, angle_keep=c.angle_keep, english_deg=c.english,
                       min_angle_deg=c.min_angle, max_angle_deg=c.max_angle, serve_pause_s=SERVE_PAUSE,
                       first_serve_s=FIRST_SERVE)
        common.declare(log, "serves", "game", "a fixed serve schedule (receiver, angle, start Y), the same for every "
                       "seed, cycled; the ball is put in play at the server's end line", schedule=[list(s) for s in c.serves])
        common.declare(log, "arena", "game", "each fly sees the court as an LED arena in its own field: "
                       "(az, el) = (Y, X - 35) for the left fly, (-Y, 35 - X) for the right one, i.e. its own end at "
                       "el -35 and the far end at el +35, az +-50; the ball is a bright disc of 6 deg radius; the "
                       "paddles are NOT shown to the flies; radiance [UV, B, G, R] given directly (no RGB conversion)",
                       ball_radius_deg=BALL_R_EYE, court=RAD_COURT.tolist(), lines=RAD_LINE.tolist(),
                       outside=RAD_OUTSIDE.tolist(), ball=RAD_BALL.tolist(), paddles_visible_to_flies=False)
        common.declare(log, "paddle", "game", "the paddle moves toward its target (its decoder's, or a human's) at up "
                       "to the speed limit, its centre clamped to the court; after a miss it stops until the point",
                       speed_limit_deg_s=c.pad_speed, centre_limit_deg=c.paddle_limit)
        for side, p in self.players.items():
            if p.is_fly:
                common.declare(log, f"ball_decoder_{side}", "decoder", DECODER_LAW,
                               reads=f"{p.name}: fb.optic.last['dr'] of {p.decoder.n} L2 + Mi1 rate units "
                                     "(columns from flyverse.interp.trace.column_of_cells on this brain)",
                               n_cells=p.decoder.n, **DECODER_PARAMS)
        log.meta["court_coverage"] = {p.key: p.coverage for p in self.players.values() if p.is_fly}
        log.meta["display"] = {
            "ribbons": "the decoder's input, max |dr - EMA| of its L2 + Mi1 cells per 4-deg azimuth bin, last 3.2 s; "
                       "rows aligned with the court's Y; dots: the ball's Y; line: the decoder's target",
            "court_names": "under each name: neurons in the brain and reconstructed photoreceptors whose column lies in "
                           "the court window (az +-50 x el +-35), counted from the retina (retina.pr_column)",
            "under_the_ball": f"CONNECTOME, for the HUD and the error breakdown only (drives nothing): columns within "
                              f"{UNDER_DEG} deg of the ball's true direction; the number of reconstructed "
                              "photoreceptors there (a count from the reconstruction) and the mean dr of the L2 cells "
                              "there (bar full at -0.5)",
            "decoder_error": "HUD: |decoder target - ball| (what the paddle follows), median of the last 3 s of live "
                             "frames; the summary also gives the per-tick reading's error, before the 50 ms EMA and gate",
            "mosaic": "frontal +-55 x +-40 deg of the exact radiance handed to fb.vision, eye_colors('human', exposure "
                      "1.2); columns without a reconstructed photoreceptor drawn dark (MaleCNS's retina has none of "
                      "those columns, so its gaps are black)",
            "decoded_ring": "the decoder's target (az, el); teal while above the gate, grey while held",
            "trail": "the ball's last 0.24 s, at most 220 px"}
        if self.control == "blind":
            common.declare(log, "control_blind", "game", "CONTROL ARM: the ball is never drawn in either arena "
                           "(the flies see the empty court); decoders and paddles run unchanged")
        elif self.control == "frozen":
            common.declare(log, "control_frozen", "game", "CONTROL ARM: decoders disconnected; both paddles stay "
                           "at the centre")

    def brains(self):
        return {p.key: p.fb for p in self.players.values() if p.is_fly}

    def _warmup(self, seconds):
        """Brain time before the clip: both flies look at the empty court so the photoreceptors adapt."""
        n = int(round(seconds * 1000 / common.TICK_MS))
        for _ in range(n):
            for p in self.players.values():
                if p.is_fly:
                    p.see(None)
                    p.fb.step(common.TICK_MS)
                    p.decoder.update(p.fb.optic.last["dr"][0], common.TICK_MS / 1000)
        for p in self.players.values():
            if p.is_fly:
                p.decoder.reset_target()

    # -------------------------------------------------------------- simulation
    def ball_view(self, side):
        c = self.court
        if not c.live or self.control == "blind":
            return None
        return view_of(side, c.X, c.Y)

    def target_y(self, side) -> float:
        """The court Y the paddle is steered to: its decoder's target, a human's, or the centre (frozen control)."""
        p = self.players[side]
        if not p.is_fly:
            return p.human_target
        if self.control == "frozen":
            return 0.0
        return court_y_of(side, p.decoder.target_az)

    def key_dir(self) -> float:
        pg = self.hud.pg if self.hud else None
        if pg is None:
            return 0.0
        up = any(k in self.keys for k in (pg.K_w, pg.K_UP))
        down = any(k in self.keys for k in (pg.K_s, pg.K_DOWN))
        return float(up) - float(down)

    def tick(self):
        dt = common.TICK_MS / 1000.0
        c = self.court
        for ev in c.step(self.t_s, dt):
            self._on_event(ev)
        kd = self.key_dir() if any(not p.is_fly for p in self.players.values()) else 0.0
        for side, p in self.players.items():
            if p.is_fly:
                p.see(self.ball_view(side))
                p.fb.step(common.TICK_MS)
                d = p.decoder.update(p.fb.optic.last["dr"][0], dt)
                true_ball = view_of(side, c.X, c.Y) if c.live else None
                n_pr, n_l2, _ = p.measure_under(true_ball)
                if c.live:
                    p.ticks_live += 1
                    p.gated += 0 if d.valid else 1
                    p.ticks_no_l2 += 1 if n_l2 == 0 else 0
                    p.ticks_at_clamp += 1 if abs(c.paddle[side]) >= c.paddle_limit - 0.25 else 0
                    p.err_target.append(abs(d.target_az - true_ball[0]))
                    p.err_reading.append(abs(d.az_hat - true_ball[0]) if np.isfinite(d.az_hat) else float("nan"))
                    p.own_half.append((c.X < COURT_LEN / 2) == (side == "A"))
                    p.pr_under.append(n_pr)
            elif kd:
                lim = c.paddle_limit
                p.human_target = float(np.clip(p.human_target + kd * c.pad_speed * dt, -lim, lim))
            c.move_paddle(side, self.target_y(side), dt)
        self.tick_n += 1
        self.t_s = self.tick_n * dt

    def _on_event(self, ev):
        t = self.t_s
        kind = ev["kind"]
        c = self.court
        if kind == "wall":
            return
        if kind == "serve":
            self.trail.clear()
            self.log.event(t, "serve", toward=ev["toward"], angle_deg=ev["angle_deg"], y0_deg=ev["y0"], n=ev["n"])
            return
        side = ev.get("side")
        if kind == "return":
            p = self.players[side]
            p.returns += 1
            self.flash[side] = t
            self.best_rally = max(self.best_rally, ev["rally"])
            self.steepened += int(ev["steepened"])
            self.return_angles.append(ev["angle_deg"])
            self.log.event(t, "return", side=side, player=p.short, offset=ev["offset"], speed_deg_s=ev["speed_deg_s"],
                           angle_out_deg=ev["angle_deg"], steepened=ev["steepened"], rally=ev["rally"],
                           paddle_minus_ball_deg=round(c.paddle[side] - c.Y, 2),
                           target_minus_ball_deg=round(self.target_y(side) - c.Y, 2))
            if ev["rally"] % 5 == 0:
                self.banner("rally", f"RALLY {ev['rally']}", WHITE, 72, hold=0.5)
        elif kind == "miss":
            p = self.players[side]
            p.misses += 1
            self.last_miss = {"side": side, "Y": c.Y, "t": t, "by": ev["by_deg"], "vy": ev["ball_vy_deg_s"],
                              "paddle": c.paddle[side]}
            self.log.event(t, "miss", side=side, player=p.short, by_deg=ev["by_deg"],
                           paddle_minus_ball_deg=round(c.paddle[side] - c.Y, 2),
                           target_minus_ball_deg=round(self.target_y(side) - c.Y, 2),
                           speed_deg_s=round(c.speed, 1), ball_vy_deg_s=ev["ball_vy_deg_s"],
                           photoreceptors_under_ball=p.under[0] if p.is_fly else None,
                           l2_cells_under_ball=p.under[1] if p.is_fly else None)
        elif kind == "point":
            s = ev["scorer"]
            p = self.players[s]
            loser = self.players[other(s)]
            self.log.event(t, "point", scorer=s, player=p.short, score=[ev["score"]["A"], ev["score"]["B"]],
                           rally=ev["rally"])
            miss = ""
            if self.last_miss and self.last_miss["side"] == loser.side:
                miss = f"  ·  {loser.short}'s paddle missed by {fmt_deg(self.last_miss['by'])}"
            self.banner("point", f"POINT {p.short}", p.color, 120, subs=[f"{ev['score']['A']} – {ev['score']['B']}{miss}"])
            n = self.args.points
            if n and ev["score"][s] >= n:
                c.serving = False
                self.game_over_t = t
                self.log.event(t, "game_over", winner=s, player=p.short, score=[ev["score"]["A"], ev["score"]["B"]])
                self.banner("point", f"GAME {p.short}", p.color, 120,
                            subs=[f"{ev['score']['A']} – {ev['score']['B']}{miss}"], hold=GAME_OVER_HOLD - 0.8)

    def banner(self, kind, text, color, size, subs=(), hold=0.4):
        """A big centred banner (full for `hold` s, then fading over 0.8 s); a new one replaces the last. A 'rally'
        banner shows the live rally count and ball speed while the rally lasts."""
        self.banners = [{"kind": kind, "text": text, "color": color, "t0": self.t_s, "size": size,
                         "subs": list(subs), "hold": hold}]

    def finished(self):
        self._summarise()                       # cheap; keeps the run log's summary current for whenever run() stops
        return self.game_over_t is not None and self.t_s - self.game_over_t >= GAME_OVER_HOLD

    def _summarise(self):
        c = self.court
        ang = np.abs(np.asarray(self.return_angles, np.float64))
        s = {"brain_s": round(self.t_s, 2), "score": {p.key: c.score[k] for k, p in self.players.items()},
             "points_played": len(c.rallies), "returns_per_point": c.rallies, "longest_rally": max(c.rallies or [0]),
             "rally_in_progress": c.rally if c.live else None, "serves": c.n_serves,
             "return_angle_abs_median_deg": round(float(np.median(ang)), 1) if len(ang) else None,
             "returns_steepened_to_min_angle": self.steepened,
             "game_floor_frozen_paddles": self.floor}
        for p in self.players.values():
            s[p.key] = p.summary()
        self.log.summary = s

    # -------------------------------------------------------------- input
    def handle(self, event, canvas_pos=None):
        pg = self.hud.pg if self.hud else None
        humans = [p for p in self.players.values() if not p.is_fly]
        if pg is None or not humans:
            return
        if event.type == pg.MOUSEMOTION and canvas_pos is not None:
            y = self._court_y_from_px(canvas_pos[1])
            lim = self.court.paddle_limit
            for p in humans:
                p.human_target = float(np.clip(y, -lim, lim))
        if event.type in (pg.KEYDOWN, pg.KEYUP) and event.key in (pg.K_w, pg.K_UP, pg.K_s, pg.K_DOWN):
            if event.type == pg.KEYDOWN:
                self.keys.add(event.key)
            else:
                self.keys.discard(event.key)

    # -------------------------------------------------------------- layout and drawing
    def _layout(self):
        W, H = common.parse_size(self.args.size)
        s = W / 1920.0
        self.s = s
        g = int(20 * s)
        top, mid_h = int(74 * s), int(700 * s)
        rib_w = int(150 * s)
        gap = int(8 * s)
        self.r_ribbon = {"A": (g, top, rib_w, mid_h), "B": (W - g - rib_w, top, rib_w, mid_h)}
        cx0 = g + rib_w + gap
        self.r_court = (cx0, top, W - 2 * cx0, mid_h)
        y2 = top + mid_h + int(10 * s)
        h2 = H - int(54 * s) - y2
        mw = int(560 * s)
        self.r_mosaic = {"A": (g, y2, mw, h2), "B": (W - g - mw, y2, mw, h2)}
        self.r_center = (g + mw + int(10 * s), y2, W - 2 * (g + mw + int(10 * s)), h2)
        self.mosaics = {k: FrontalMosaic(p.eyes, p.no_pr) for k, p in self.players.items() if p.is_fly}
        self.log.meta.setdefault("display", {})["layout_px"] = {
            "canvas": [W, H], "court": list(self.r_court), "ribbons": [list(v) for v in self.r_ribbon.values()],
            "court_fraction": round(self.r_court[2] * self.r_court[3] / (W * H), 3),
            "court_and_ribbons_fraction": round((W - 2 * g) * mid_h / (W * H), 3)}

    def _court_scale(self):
        """Pixels per court degree along X and across Y."""
        _, _, w, h = self.r_court
        return w / COURT_LEN, (h - 24 * self.s) / (2 * HALF_W)

    def _court_px(self, X, Y):
        x, y, w, h = self.r_court
        m = 12 * self.s
        return x + X / COURT_LEN * w, y + m + (HALF_W - Y) / (2 * HALF_W) * (h - 2 * m)

    def _court_y_from_px(self, py):
        _, y, _, h = self.r_court
        m = 12 * self.s
        return float(np.clip(HALF_W - (py - y - m) / (h - 2 * m) * 2 * HALF_W, -HALF_W, HALF_W))

    def _tag(self, surf, text, pos, size, color, *, anchor="topleft", bold=False, display=False, alpha=1.0,
             back=(6, 8, 9, 215), pad=5, back_fade=True):
        """Text on a dark rounded backdrop (for labels drawn over the court or a trace). The backdrop fades with the
        text unless `back_fade` is False (then its own alpha, back[3], is used as given)."""
        pg = self.hud.pg
        img = self.hud.font(size, bold, display).render(str(text), True, color)
        r = img.get_rect(**{anchor: (int(pos[0]), int(pos[1]))})
        if back:
            b = pg.Surface((r.w + 2 * pad, r.h + 2 * pad), pg.SRCALPHA)
            pg.draw.rect(b, (*back[:3], int(back[3] * (alpha if back_fade else 1.0))), b.get_rect(), border_radius=pad)
            surf.blit(b, (r.x - pad, r.y - pad))
        if alpha < 1.0:
            img.set_alpha(int(255 * alpha))
        surf.blit(img, r)
        return r

    def draw(self, surface):
        pg, hud = self.hud.pg, self.hud
        surface.fill(BG)
        c = self.court
        for p in self.players.values():
            if p.is_fly:
                p.ribbon.append(p.ribbon_sample())
                p.ribbon_ball.append(c.Y if c.live else np.nan)
                p.ribbon_dec.append(court_y_of(p.side, p.decoder.target_az))
                p.conf_hist.append(p.decoder.conf)
                if c.live:
                    p.target_err_hist.append(p.err_target[-1] if p.err_target else np.nan)
        if c.live:
            self.trail.append((c.X, c.Y))
        else:
            self.trail.clear()
        names = " vs ".join(p.name for p in self.players.values())
        hud.header(surface, "PONG", self.subtitle, right=f"{names} · preset raw")
        self._draw_court(pg, hud, surface)
        for side in ("A", "B"):
            self._draw_ribbon(pg, hud, surface, side)
            self._draw_mosaic(pg, hud, surface, side)
        self._draw_center(pg, hud, surface)
        hud.footer(surface, ["court, serves, paddle speed: GAME",
                             "paddle target: DECODER (peak of high-passed L2 + Mi1)",
                             "under the ball: CONNECTOME (photoreceptors, L2 Δr)"])

    def _draw_court(self, pg, hud, surf):
        s = self.s
        x, y, w, h = self.r_court
        pg.draw.rect(surf, COURT_BG, (x, y, w, h), border_radius=int(8 * s))
        pg.draw.rect(surf, LINE, (x, y, w, h), 2, border_radius=int(8 * s))
        c = self.court
        nx = int(x + w / 2)
        for yy in range(int(y + 10 * s), int(y + h - 10 * s), int(28 * s)):
            pg.draw.rect(surf, (40, 48, 52), (nx - int(3 * s), yy, int(6 * s), int(15 * s)))
        for side, fx in (("A", 0.3), ("B", 0.7)):
            p = self.players[side]
            cx = x + w * fx
            hud.text(surf, str(c.score[side]), (cx, y + 6 * s), int(150 * s), blend(COURT_BG, p.color, 0.72),
                     bold=True, display=True, anchor="midtop")
            hud.text(surf, p.name, (cx, y + 170 * s), int(32 * s), p.color, bold=True, display=True, anchor="midtop")
            sub = (f"{p.fb.c.n:,} neurons · {p.pr_court:,} photoreceptors face the court" if p.is_fly
                   else "a human: W / S or the mouse")
            hud.text(surf, sub, (cx, y + 210 * s), int(21 * s), TEXT, display=True, anchor="midtop")
        if self.mixed:
            self._tag(surf, NOT_SEX, (nx, y + 250 * s), int(17 * s), MUTED, anchor="midtop", display=True,
                      back=(*COURT_BG, 255), pad=int(4 * s))
        # rally and speed, on a backdrop over the net, with the GAME chip beside it
        lab = pg.Rect(0, 0, int(360 * s), int(58 * s))
        lab.midbottom = (nx, int(y + h - 8 * s))
        pg.draw.rect(surf, COURT_BG, lab)
        hud.text(surf, (f"RALLY {c.rally}" if c.live else f"LONGEST RALLY {self.best_rally}"),
                 (nx, lab.y + 2 * s), int(24 * s), TEXT, bold=True, anchor="midtop")
        hud.text(surf, (f"ball {c.speed:.0f}°/s in the flies' arenas" if c.live else
                        ("game over" if not c.serving else "next serve")),
                 (nx, lab.bottom - 4 * s), int(16 * s), DIM, anchor="midbottom")
        img = hud.font(int(12 * s), True).render("GAME", True, BG)             # the chip's size, to right-align it
        cw, ch = img.get_width() + 12, img.get_height() + 6
        hud.chip(surf, "game", (lab.x - cw - 10 * s, lab.bottom - ch - 8 * s), int(12 * s))
        self._draw_banners(pg, hud, surf)
        clip = surf.get_clip()
        surf.set_clip(pg.Rect(x + 2, y + 2, w - 4, h - 4))         # paddles, flashes and the ball stay on the court
        sx, sy = self._court_scale()
        miss_side = c.missed if c.live else None
        # the ball: a comet trail (at most 220 px), a soft glow, the ball; drawn under a paddle that missed
        if c.live and miss_side:
            self._draw_ball(pg, surf, sx, sy)
        for side in ("A", "B"):
            p = self.players[side]
            px = PAD_X[0] if side == "A" else PAD_X[1]
            X, Y = self._court_px(px, c.paddle[side])
            half = PAD_HALF * sy
            thick = max(6, int(2 * PAD_THICK * sx))
            age = self.t_s - self.flash[side]
            if 0 <= age < 0.45:
                grow = (8 + 70 * age / 0.45) * s
                ring = pg.Rect(0, 0, thick + 2 * grow, 2 * half + 2 * grow)
                ring.center = (X, Y)
                pg.draw.rect(surf, blend(COURT_BG, p.color, 1.0 - age / 0.45), ring, max(2, int(3 * s)),
                             border_radius=int(10 * s + grow / 2))
            body_col = blend(WHITE, p.color, min(1.0, max(0.0, (age - 0.06) / 0.25))) if age >= 0 else p.color
            if miss_side == side:
                body_col = blend(COURT_BG, p.color, 0.55)
            pg.draw.rect(surf, body_col, (X - thick / 2, Y - half, thick, 2 * half), border_radius=int(4 * s))
            if p.is_fly and self.control != "frozen":
                ty = self._court_px(px, float(np.clip(court_y_of(side, p.decoder.target_az), -HALF_W, HALF_W)))[1]
                sgn = 1 if side == "A" else -1
                bx = X + sgn * (thick / 2 + 12 * s)
                col = TEAL if p.decoder.valid else DIM
                pg.draw.polygon(surf, col, [(bx, ty), (bx + sgn * 18 * s, ty - 11 * s), (bx + sgn * 18 * s, ty + 11 * s)])
        if c.live and not miss_side:
            self._draw_ball(pg, surf, sx, sy)
        if c.live and self.control == "blind":
            X, Y = self._court_px(c.X, c.Y)
            hud.text(surf, "hidden from both flies", (X, Y - BALL_RY * sy - 6 * s), int(16 * s), AMBER,
                     anchor="midbottom")
        self._draw_miss(pg, hud, surf, sy)
        surf.set_clip(clip)

    def _draw_ball(self, pg, surf, sx, sy):
        c = self.court
        pts = truncate_polyline([self._court_px(tx, ty) for tx, ty in self.trail], 220 * self.s)
        rx, ry = BALL_RX * sx, BALL_RY * sy
        n = len(pts)
        if n > 1:
            seg = np.hypot(*np.diff(pts, axis=0).T)
            total = float(seg.sum())
            if total > 1:
                k = max(2, int(total / (6 * self.s)))
                cum = np.r_[0.0, np.cumsum(seg)]
                for f in np.linspace(0.0, 1.0, k, endpoint=False):
                    q = np.array([np.interp(f * total, cum, pts[:, 0]), np.interp(f * total, cum, pts[:, 1])])
                    r = max(1, int(ry * (0.25 + 0.6 * f)))
                    pg.draw.circle(surf, blend(COURT_BG, (190, 205, 202), 0.5 * f ** 1.5), q, r)
        X, Y = self._court_px(c.X, c.Y)
        glow = pg.Rect(0, 0, int(2 * rx * 1.22), int(2 * ry * 1.22))
        glow.center = (int(X), int(Y))
        pg.draw.ellipse(surf, blend(COURT_BG, WHITE, 0.18), glow)
        ball = pg.Rect(0, 0, int(2 * rx), int(2 * ry))
        ball.center = (int(X), int(Y))
        pg.draw.ellipse(surf, WHITE, ball)

    def _draw_miss(self, pg, hud, surf, sy):
        """A red ring where the ball crossed the paddle line, and 'missed by N deg' on the side of the ring the ball
        is moving away from (so the escaping ball never runs under the label), toward the court."""
        lm = self.last_miss
        if not lm or self.t_s - lm["t"] >= 1.2:
            return
        s = self.s
        side = lm["side"]
        px = PAD_X[0] if side == "A" else PAD_X[1]
        X, Y = self._court_px(px, lm["Y"])
        a = 1.0 - (self.t_s - lm["t"]) / 1.2
        rr = int(34 * s)
        pg.draw.circle(surf, blend(COURT_BG, RED, a), (X, Y), rr, max(2, int(4 * s)))
        _, y, _, h = self.r_court
        # screen-up is +Y; the label goes where the ball is not heading (or away from the paddle if it goes straight)
        away_up = lm["vy"] < 0 if abs(lm["vy"]) > 1e-6 else lm["Y"] > lm["paddle"]
        off = (BALL_RY * sy + 18 * s)
        room_up, room_down = Y - off - 30 * s > y, Y + off + 30 * s < y + h
        if (away_up and not room_up) or (not away_up and not room_down):
            away_up = not away_up
        ly = Y - off if away_up else Y + off
        lx = X + (1 if side == "A" else -1) * 62 * s
        anchor = ("bottom" if away_up else "top") + ("left" if side == "A" else "right")
        self._tag(surf, f"missed by {fmt_deg(lm['by'])}", (lx, ly), int(24 * s), blend(COURT_BG, RED, a), bold=True,
                  anchor=anchor, alpha=a, back=(6, 8, 9, int(240 * min(1.0, 1.6 * a))), back_fade=False)

    def _draw_banners(self, pg, hud, surf):
        keep = []
        x, y, w, h = self.r_court
        s = self.s
        c = self.court
        for b in self.banners:
            age = self.t_s - b["t0"]
            if age > b["hold"] + 0.8:
                continue
            keep.append(b)
            a = 1.0 if age < b["hold"] else max(0.0, 1.0 - (age - b["hold"]) / 0.8)
            text, subs = b["text"], b["subs"]
            if b["kind"] == "rally" and c.live:
                text, subs = f"RALLY {c.rally}", [f"ball {c.speed:.0f}°/s"]
            img = hud.font(int(b["size"] * s), True, True).render(text, True, b["color"])
            img.set_alpha(int(255 * a))
            r = img.get_rect(center=(x + w / 2, y + h * 0.56))
            simgs = [hud.font(int(24 * s)).render(t, True, TEXT) for t in subs]
            bw = max([r.w] + [im.get_width() for im in simgs]) + int(70 * s)
            bh = r.h + int(26 * s) + sum(im.get_height() + int(8 * s) for im in simgs)
            back = pg.Surface((bw, bh), pg.SRCALPHA)
            back.fill((6, 8, 9, int(210 * a)))
            surf.blit(back, back.get_rect(midtop=(r.centerx, r.y - int(13 * s))))
            surf.blit(img, r)
            yy = r.bottom + 6 * s
            for im in simgs:
                im.set_alpha(int(255 * a))
                surf.blit(im, im.get_rect(midtop=(r.centerx, yy)))
                yy += im.get_height() + 8 * s
        self.banners = keep

    def _draw_ribbon(self, pg, hud, surf, side):
        """The decoder's input by azimuth, rows aligned with the court's Y: max |dr - EMA| of the L2 + Mi1 cells per
        4-deg azimuth bin over the last 3.2 s (the newest column at the court's edge), with the ball's Y (dots) and the
        decoder's target (line) on top."""
        s = self.s
        p = self.players[side]
        rect = pg.Rect(self.r_ribbon[side])
        pg.draw.rect(surf, PANEL, rect, border_radius=int(6 * s))
        pg.draw.rect(surf, LINE, rect, 1, border_radius=int(6 * s))
        inner = pg.Rect(int(rect.x + 6 * s), int(self.r_court[1] + 12 * s), int(rect.w - 12 * s),
                        int(self.r_court[3] - 24 * s))
        if not p.is_fly:
            hud.text(surf, "no brain", (inner.x + 4 * s, inner.y + 40 * s), int(15 * s), DIM)
            return
        T = len(p.ribbon)
        age = np.arange(inner.w) / max(inner.w - 1, 1) * (HISTORY - 1)
        if side == "A":
            age = age[::-1]                                        # the court edge (right) is now
        idx = T - 1 - np.round(age).astype(int)
        ok = idx >= 0
        Yrow = HALF_W - (np.arange(inner.h) + 0.5) / inner.h * 2 * HALF_W
        az = Yrow if side == "A" else -Yrow
        bins = np.clip(np.floor((az + 52.0) / 4.0).astype(int), 0, RIBBON_BINS - 1)
        img = np.zeros((inner.w, inner.h), np.float32)
        if T:
            arr = np.stack(list(p.ribbon), 0) / RIBBON_SCALE
            img[ok] = arr[idx[ok]][:, bins]
        surf.blit(pg.surfarray.make_surface(np.ascontiguousarray(colormap(img, p.color))), inner.topleft)
        clip = surf.get_clip()
        surf.set_clip(inner)

        def xs_of(n):
            a = np.arange(n)[::-1] / (HISTORY - 1) * inner.w
            return inner.right - 1 - a if side == "A" else inner.x + a

        def ys_of(v):
            return inner.y + (HALF_W - np.asarray(v, np.float64)) / (2 * HALF_W) * inner.h

        xb, yb = xs_of(len(p.ribbon_ball)), ys_of(p.ribbon_ball)
        for k in range(len(xb) % 4, len(xb), 4):
            if np.isfinite(yb[k]):
                pg.draw.circle(surf, WHITE, (xb[k], yb[k]), max(2, int(2.5 * s)))
        xd, yd = xs_of(len(p.ribbon_dec)), ys_of(p.ribbon_dec)
        if len(xd) > 1:
            pg.draw.lines(surf, TEAL, False, np.c_[xd, yd].tolist(), max(2, int(3 * s)))
        surf.set_clip(clip)
        lab = pg.Surface((inner.w, int(54 * s)), pg.SRCALPHA)
        lab.fill((19, 24, 27, 225))
        surf.blit(lab, inner.topleft)
        hud.text(surf, p.short, (inner.x + 5 * s, inner.y + 3 * s), int(19 * s), p.color, bold=True)
        chip = hud.chip(surf, "decoder", (inner.x + 5 * s, inner.y + 29 * s), int(11 * s))
        hud.text(surf, "input", (chip.right + 6 * s, inner.y + 30 * s), int(13 * s), MUTED)
        foot = pg.Surface((inner.w, int(40 * s)), pg.SRCALPHA)
        foot.fill((19, 24, 27, 225))
        surf.blit(foot, (inner.x, inner.bottom - int(40 * s)))
        near, far = (inner.right - 4 * s, "bottomright"), (inner.x + 4 * s, "bottomleft")
        if side == "B":
            near, far = (inner.x + 4 * s, "bottomleft"), (inner.right - 4 * s, "bottomright")
        hud.text(surf, "now", (near[0], inner.bottom - 22 * s), int(13 * s), MUTED, anchor=near[1])
        hud.text(surf, "−3 s", (far[0], inner.bottom - 22 * s), int(13 * s), MUTED, anchor=far[1])
        hud.text(surf, "● ball ― target", (inner.centerx, inner.bottom - 4 * s), int(13 * s), TEAL, anchor="midbottom")

    def _draw_mosaic(self, pg, hud, surf, side):
        s = self.s
        p = self.players[side]
        rect = pg.Rect(self.r_mosaic[side])
        pg.draw.rect(surf, PANEL, rect, border_radius=6)
        pg.draw.rect(surf, LINE, rect, 1, border_radius=6)
        pad = int(12 * s)
        text_w = int(222 * s)
        if side == "A":
            tx, marea = rect.x + pad, (rect.x + pad + text_w + pad, rect.y + pad, rect.w - text_w - 3 * pad, rect.h - 2 * pad)
        else:
            tx, marea = rect.right - pad - text_w, (rect.x + pad, rect.y + pad, rect.w - text_w - 3 * pad, rect.h - 2 * pad)
        hud.text(surf, f"WHAT {p.short.upper()} SEES", (tx, rect.y + 10 * s), int(16 * s), MUTED, bold=True)
        if not p.is_fly:
            hud.text(surf, "a human plays this side", (tx, rect.y + 40 * s), int(16 * s), DIM)
            return
        hud.text(surf, "the input to fb.vision", (tx, rect.y + 32 * s), int(13 * s), DIM)
        chip = hud.chip(surf, "connectome", (tx, rect.y + 58 * s), int(11 * s))
        hud.text(surf, "reconstructed", (chip.right + 6 * s, rect.y + 59 * s), int(13 * s), MUTED)
        num = hud.text(surf, f"{p.pr_court:,}", (tx, rect.y + 80 * s), int(44 * s), p.color, bold=True, display=True)
        hud.text(surf, "photoreceptors", (num.right + 8 * s, num.y + 6 * s), int(15 * s), TEXT)
        hud.text(surf, "face the court", (num.right + 8 * s, num.y + 26 * s), int(15 * s), TEXT)
        gap = "black" if not len(p.no_pr) else "dark"
        hud.text(surf, f"{gap}: no photoreceptor", (tx, num.bottom + 8 * s), int(13 * s), DIM)
        hud.text(surf, "reconstructed there", (tx, num.bottom + 26 * s), int(13 * s), DIM)
        ly = rect.bottom - pad - 38 * s
        chip = hud.chip(surf, "decoder", (tx, ly), int(11 * s))
        hud.text(surf, "ring: decoded ball", (chip.right + 6 * s, ly + 1 * s), int(13 * s), TEAL)
        hud.text(surf, "bar: paddle (unseen)", (tx, ly + 22 * s), int(13 * s), p.color)
        m = self.mosaics[side]
        mrect = m.fit(marea)
        colors = common.eye_colors(p.radiance, exposure=1.2) if p.radiance is not None else np.zeros((p.eyes.n_col, 3), np.uint8)
        m.draw(pg, surf, mrect, colors)
        d = p.decoder
        clip = surf.get_clip()
        surf.set_clip(pg.Rect(mrect))
        if self.control != "frozen":
            X, Y = m.to_px(mrect, d.target_az, d.target_el)
            col = TEAL if d.valid else DIM
            r = int(15 * s)
            lw = max(2, int(3 * s))
            pg.draw.circle(surf, col, (X, Y), r, lw)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                pg.draw.line(surf, col, (X + dx * (r - 5 * s), Y + dy * (r - 5 * s)),
                             (X + dx * (r + 8 * s), Y + dy * (r + 8 * s)), lw)
            tx2, ty2 = m.to_px(mrect, d.target_az, -COURT_LEN / 2 + 1.0)
            pg.draw.polygon(surf, col, [(tx2, ty2 + 7 * s), (tx2 - 7 * s, ty2 - 5 * s), (tx2 + 7 * s, ty2 - 5 * s)])
        paz = court_y_of(side, self.court.paddle[side])
        x0, y0 = m.to_px(mrect, paz + PAD_HALF, -COURT_LEN / 2 - 2.0)
        x1, _ = m.to_px(mrect, paz - PAD_HALF, -COURT_LEN / 2 - 2.0)
        pg.draw.line(surf, p.color, (x0, y0), (x1, y0), max(3, int(5 * s)))
        surf.set_clip(clip)

    def _draw_center(self, pg, hud, surf):
        s = self.s
        body = hud.panel(surf, self.r_center, "the paddles follow the decoders", "decoder")
        row_h = body.h / 2
        live = self.court.live
        for i, side in enumerate(("A", "B")):
            p = self.players[side]
            y = body.y + i * row_h
            hud.text(surf, p.short, (body.x, y), int(22 * s), p.color, bold=True)
            if not p.is_fly:
                chip = hud.chip(surf, "game", (body.x, y + 34 * s), int(11 * s))
                hud.text(surf, f"returns {p.returns}  misses {p.misses}", (chip.right + 8 * s, y + 30 * s), int(22 * s), TEXT)
                hud.text(surf, "human (W / S or mouse)", (body.x + 120 * s, y + 5 * s), int(14 * s), DIM)
                continue
            d = p.decoder
            hud.text(surf, f"{d.n:,} L2 + Mi1 cells", (body.x + 118 * s, y + 5 * s), int(13 * s), MUTED)
            chip = hud.chip(surf, "game", (body.x, y + 34 * s), int(11 * s))
            hud.text(surf, f"returns {p.returns}  misses {p.misses}", (chip.right + 8 * s, y + 29 * s), int(22 * s), TEXT)
            chip = hud.chip(surf, "decoder", (body.x, y + 66 * s), int(11 * s))
            e = np.asarray(p.target_err_hist, np.float64)
            e = e[np.isfinite(e)]
            err = f"{np.median(e):.0f}°" if len(e) else "–"
            hud.text(surf, f"|target − ball| {err}", (chip.right + 8 * s, y + 61 * s), int(22 * s), TEAL)
            hud.text(surf, "median of the last 3 s", (chip.right + 8 * s, y + 86 * s), int(12 * s), DIM)
            # CONNECTOME: what lies under the ball
            bx = body.x + 336 * s
            bw = body.right - bx
            n_pr, n_l2, mdr = p.under
            chip = hud.chip(surf, "connectome", (bx, y + 3 * s), int(11 * s))
            hud.text(surf, "photoreceptors under the ball", (chip.right + 8 * s, y + 3 * s), int(13 * s), MUTED)
            hud.text(surf, f"{n_pr}" if live else "–", (bx + bw, y - 3 * s), int(24 * s),
                     (SAGE if n_pr else RED) if live else DIM, bold=True, anchor="topright")
            ly = y + 30 * s
            t = hud.text(surf, "L2 mean Δr", (bx, ly), int(13 * s), MUTED)
            val = hud.text(surf, (f"{mdr:+.2f} ({n_l2} cells)" if n_l2 else ("no L2 column" if live else "–")),
                           (bx + bw, ly - 2 * s), int(16 * s), SAGE if n_l2 else DIM, anchor="topright")
            track = pg.Rect(t.right + 10 * s, ly + 3 * s, max(10, val.x - t.right - 20 * s), 10 * s)
            pg.draw.rect(surf, INSET, track, border_radius=3)
            if live and n_l2 and mdr < 0:
                pg.draw.rect(surf, SAGE, (track.x, track.y, max(2, track.w * min(-mdr / 0.5, 1)), track.h), border_radius=3)
            tr = pg.Rect(bx, y + 50 * s, bw, row_h - 56 * s)
            hud.trace(surf, tr, list(p.conf_hist), 0.0, 1.2, TEAL, threshold=d.gate)
            self._tag(surf, "decoder's top cell |Δr − mean|, red: gate", (tr.x + 5 * s, tr.y + 3 * s), int(12 * s),
                      MUTED, back=(9, 12, 14, 200), pad=2)
            self._tag(surf, f"{d.conf:.2f}", (tr.right - 5 * s, tr.y + 3 * s), int(16 * s), TEAL if d.valid else DIM,
                      bold=True, anchor="topright", back=(9, 12, 14, 200), pad=2)


# ---------------------------------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = common.standard_args(__doc__.split("\n")[0], seconds=40.0)
    ap.add_argument("--left", default="malecns", choices=("malecns", "fafb", "human"), help="left player")
    ap.add_argument("--right", default="fafb", choices=("malecns", "fafb", "human"), help="right player")
    ap.add_argument("--control", default="none", choices=("none", "blind", "frozen"),
                    help="control arm: 'blind' hides the ball from both flies; 'frozen' disconnects the decoders")
    ap.add_argument("--points", type=int, default=0, help="end the game when a side reaches this score (0: run --seconds)")
    ap.add_argument("--warmup", type=float, default=1.0, help="brain seconds of empty court before the clip")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return common.run(PongGame(args), args)


if __name__ == "__main__":
    main()
