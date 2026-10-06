"""Sunrise: the whole fly CNS as a point cloud of 141,781 somata, lit by its own activity, while the room's lamp
comes on.

The fly stands on the table in the room (scripts/room_demo.py's `Sim`, preset 'raw'). The room is dark: the
lamp's point light and the ambient light are scaled to `--dark` of their shipped values. At `--lamp-on-at` seconds
both return to their shipped values (GAME: we flip the light switch). In the shipped room only the ambient term
reaches the fly's eyes: the point light sits inside its own lamp globe, which shadows it from every surface
(games/captions/sunrise.md, "A flyverse finding"). Nothing else is done to the fly. Every
soma's brightness is the model's own activity: spiking cells rate / 30 Hz; graded optic-lobe units
(|r - baseline| - 0.12) / 0.3, where 0.12 sits just above the optic lobe's median |r - baseline| in total darkness
(0.06-0.10 on a dev seed; the dark optic lobe is never silent, its 90th percentile is ~0.3). Both clipped to [0, 1].
That offset is a display choice, declared in the run log; the console's brain map uses |r - baseline| x 2.

    python games/sunrise.py                                   # interactive
    python games/sunrise.py --record out/games/sunrise/seed0.mp4 --seconds 16 --lamp-on-at 6.2 --clean

Soma coordinates are `somaLocation` from the MaleCNS body annotations (8 nm voxels); neurons without one
(25,325 of 167,106, mostly VNC and sensory) are not drawn. The camera orbit is decoration.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as gc  # noqa: E402

sys.path.insert(0, str(gc.ROOT / "scripts"))


OPTIC_DARK, OPTIC_SPAN = 0.12, 0.3        # display mapping for graded units (see the module docstring)


class Sunrise(gc.Game):
    title = "sunrise"
    subtitle = "141,781 somata, lit by their own activity"

    def __init__(self, args):
        super().__init__(args)
        import room_demo
        from flyverse import connectome
        self.sim = room_demo.Sim(seed=args.seed, trail_seconds=0, fence=True)
        self.fb, self.w = self.sim.fb, self.sim.world
        self.eyes = gc.Eyes(self.fb)
        self.dev = self.fb.device
        c = self.fb.c
        # --- soma cloud
        ann = pd.read_feather(connectome.DATA_DIR / connectome.ANNOT_FILE, columns=["bodyId", "somaLocation"])
        loc = ann.set_index("bodyId").somaLocation.reindex(c.neurons.bodyId)
        xyz = np.full((c.n, 3), np.nan, np.float32)
        for i, v in enumerate(loc.to_numpy()):
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 3:
                xyz[i] = v
        has = np.isfinite(xyz).all(1)
        self.idx = torch.as_tensor(np.flatnonzero(has), device=self.dev)
        p = xyz[has] * 8e-3                                            # voxels -> micrometres
        # display frame: X = left-right, Y = anterior-posterior (brain at the front), Z = dorsal up
        P = np.stack([p[:, 0], -p[:, 2], -p[:, 1]], 1)
        sc = c.neurons.superclass.fillna("").to_numpy()[has]
        brain = ~np.char.startswith(sc.astype(str), "vnc")
        P -= np.median(P[brain], axis=0)                               # orbit about the brain; the VNC trails behind it
        self.P = torch.as_tensor(P / np.percentile(np.abs(P[brain]), 99.5), dtype=torch.float32, device=self.dev)
        tint = np.tile(np.array([[0.30, 0.42, 0.46]], np.float32), (len(sc), 1))    # anatomy: dim teal-grey
        tint[np.isin(sc, ["ol_intrinsic", "ol_sensory", "visual_projection"])] = (0.28, 0.40, 0.52)
        tint[np.char.startswith(sc.astype(str), "vnc")] = (0.34, 0.40, 0.40)
        self.tint = torch.as_tensor(tint, device=self.dev)
        self.log.meta["somata_drawn"] = int(has.sum())
        # optic-unit positions inside the full vector (graded units use |dr| x 2, as flyverse/brainmap.py)
        self.rate_idx = torch.as_tensor(self.fb.optic.rate_idx, device=self.dev)
        # --- the light switch (GAME)
        self.light0, self.amb0 = tuple(self.w.light_color), tuple(self.w.ambient)
        self.set_light(args.dark)
        self.lit = False
        gc.declare(self.log, "light switch", "game",
                   f"lamp power and ambient x{args.dark} of the shipped room until t = {args.lamp_on_at} s, then x1",
                   lamp_on_at_s=args.lamp_on_at, dark_factor=args.dark)
        self.act_hist, self.of_hist = [], []
        self.dr_ema = None
        gc.declare(self.log, "soma glow", "game", "display only: spiking rate/30 Hz; optic (|dr| - 0.12)/0.3; clipped to [0,1]",
                   optic_dark=OPTIC_DARK, optic_span=OPTIC_SPAN)
        self.rad = torch.zeros(self.eyes.n_col, 4)

    def brains(self):
        return {"brain": self.fb}

    def set_light(self, k):
        self.w.light_color = tuple(k * v for v in self.light0)
        self.w.ambient = tuple(k * v for v in self.amb0)
        self.w.invalidate()

    def update_ema(self):
        dr = self.fb.optic.last.get("dr")
        if dr is None:
            return
        k = gc.TICK_MS / 300.0
        self.dr_ema = dr[0].clone() if self.dr_ema is None else self.dr_ema + k * (dr[0] - self.dr_ema)

    def activity(self):
        a = (self.fb.brain.rate[0] / 30.0).clamp(0, 1).clone()
        dr = self.fb.optic.last.get("dr")
        if dr is not None and self.dr_ema is not None:
            a[self.rate_idx] = ((dr[0].abs() - OPTIC_DARK) / OPTIC_SPAN).clamp(0, 1)
        return a

    def tick(self):
        if not self.lit and self.t_s + 1e-9 >= self.args.lamp_on_at:
            self.set_light(1.0)
            self.lit = True
            self.log.event(self.t_s, "lamp_on")
        self.sim.step()
        self.t_s += gc.TICK_MS / 1000
        self.rad = self.sim.col_rad
        a = self.activity()
        self.update_ema()
        ol = a[self.rate_idx].mean().item()
        self.act_hist.append(a.mean().item()); self.of_hist.append(ol)
        self.act_hist, self.of_hist = self.act_hist[-300:], self.of_hist[-300:]
        if int(round(self.t_s * 100)) % 10 == 0:                      # every 100 ms: the model's own numbers
            dr = self.fb.optic.last["dr"][0].abs()
            q50, q90 = torch.quantile(dr[:: max(1, dr.numel() // 50000)], torch.tensor([0.5, 0.9], device=dr.device)).tolist()
            self.log.event(self.t_s, "activity", optic_absdr_median=round(q50, 4), optic_absdr_p90=round(q90, 4),
                           spiking_mean_hz=round(float(self.fb.brain.rate[0].mean()), 3),
                           spikes_per_step=round(float(self.fb.brain.total_spikes()), 1), gf_hz=round(self.sim.wcmd["gf"], 2),
                           display_mean_optic=round(ol, 4))

    # ------------------------------------------------------------------------------------------ drawing
    def render_cloud(self, size):
        W, H = size
        yaw = 0.25 + 0.03 * self.t_s                                   # slow orbit (decoration)
        pitch = 0.32
        cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
        R = torch.tensor([[cy, sy, 0], [-sy * cp, cy * cp, sp], [sy * sp, -cy * sp, cp]], dtype=torch.float32, device=self.dev)
        q = self.P @ R.T                                               # x right, y depth, z up
        depth = q[:, 1] + 3.0
        f = 0.62 * H * 3.0 / depth
        u = (W * 0.50 + q[:, 0] * f).long(); v = (H * 0.47 - q[:, 2] * f).long()
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        act = self.activity()[self.idx]
        base = self.tint * 0.10
        hot = torch.stack([act * 1.0, act * 0.72, act * 0.28], 1)      # amber-white glow
        col = (base + hot * 0.9)[ok]
        img = torch.zeros(H * W, 3, device=self.dev)
        img.index_add_(0, (v[ok] * W + u[ok]), col)
        img = img.reshape(H, W, 3).permute(2, 0, 1)[None]
        # bloom: two blurred copies of the hot part added back
        glow = torch.nn.functional.avg_pool2d(img, 4)
        for k in (9, 25):
            g = torch.arange(k, device=self.dev, dtype=torch.float32) - k // 2
            g = torch.exp(-(g / (k / 5)) ** 2); g = g / g.sum()
            b = torch.nn.functional.conv2d(glow, g.view(1, 1, 1, k).repeat(3, 1, 1, 1), padding=(0, k // 2), groups=3)
            b = torch.nn.functional.conv2d(b, g.view(1, 1, k, 1).repeat(3, 1, 1, 1), padding=(k // 2, 0), groups=3)
            img = img + 0.9 * torch.nn.functional.interpolate(b, size=(H, W), mode="bilinear", align_corners=False)
        img = 1 - torch.exp(-img * 1.4)                                # soft tone map
        bg = torch.tensor(gc.BG, device=self.dev, dtype=torch.float32)[:, None, None] / 255
        img = (bg + img[0]).clamp(0, 1)
        return (img.permute(1, 2, 0) * 255).byte().cpu().numpy()

    def draw(self, s):
        pg = self.hud.pg
        W, H = s.get_size()
        arr = self.render_cloud((W, H))
        s.blit(pg.surfarray.make_surface(arr.swapaxes(0, 1)), (0, 0))
        h = self.hud
        if not self.args.clean:
            h.header(s, "sunrise", self.subtitle, gc.model_line(self.fb))
            h.footer(s, ["lamp switched on at t = %.2f s (GAME); every soma's glow is the model's own activity" % self.args.lamp_on_at])
        # the fly's eye, small, bottom-right: black until the light comes on
        r = pg.Rect(W - 470, H - 330 - (0 if self.args.clean else 50), 440, 290)
        pg.draw.rect(s, gc.PANEL, r, border_radius=6)
        h.text(s, "WHAT THE FLY SEES", (r.x + 12, r.y + 8), 15, gc.MUTED, bold=True)
        h.mosaic(s, pg.Rect(r.x + 8, r.y + 32, r.w - 16, r.h - 40), self.eyes, gc.eye_colors(self.rad, exposure=2.5))
        tl = "LAMP ON" if self.lit else "DARK"
        h.text(s, f"t = {self.t_s:5.2f} s   {tl}", (32, H - 60 - (0 if self.args.clean else 50)), 22,
               gc.AMBER if self.lit else gc.DIM)


def main():
    ap = gc.standard_args(__doc__.splitlines()[0], seconds=16.0)
    ap.add_argument("--lamp-on-at", type=float, default=4.0, help="brain seconds at which the lamp comes on")
    ap.add_argument("--dark", type=float, default=0.0, help="lamp and ambient factor before the switch (0 = black)")
    ap.add_argument("--clean", action="store_true", help="no header / footer (for the trailer)")
    args = ap.parse_args()
    gc.run(Sunrise(args), args)


if __name__ == "__main__":
    main()
