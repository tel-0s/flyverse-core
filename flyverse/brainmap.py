"""Brain map: every neuron's soma projected onto two views, activity drawn as glowing highlights.

Soma positions come from `somaLocation` in the annotations (8 nm voxels; x = left-right, y = dorsal-
ventral, z = anterior-posterior; the brain sits at low z, the VNC at high z). Two views are drawn:
dorsal (z across, x down: brain left, VNC right, both optic lobes top/bottom) and lateral (z across,
y down). Neurons are binned into pixels once; per frame the activity vector is summed per pixel with
np.bincount, so 167k neurons cost well under a millisecond.

Activity: spiking neurons use their running rate (Hz / 40, clipped to 1); optic-lobe rate units use
|r - baseline| * 2 (clipped to 1), so both halves of the hybrid model light up on the same scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import connectome


class BrainMap:
    def __init__(self, c, optic, width=300, dorsal_h=220, lateral_h=155):
        ann = pd.read_feather(connectome.DATA_DIR / connectome.ANNOT_FILE, columns=["bodyId", "somaLocation"])
        loc = ann.set_index("bodyId").somaLocation.reindex(c.neurons.bodyId)
        self.has = loc.notna().to_numpy()
        xyz = np.array([list(p) for p in loc[self.has]], dtype=np.float32)
        self.idx = np.flatnonzero(self.has)
        # fixed extents (1-99 percentiles of the CNS) so the map does not jump between datasets
        zlo, zhi = np.percentile(xyz[:, 2], [0.5, 99.5]); xlo, xhi = np.percentile(xyz[:, 0], [0.5, 99.5]); ylo, yhi = np.percentile(xyz[:, 1], [0.5, 99.5])
        self.W, self.Hd, self.Hl = width, dorsal_h, lateral_h
        u = np.clip(((xyz[:, 2] - zlo) / (zhi - zlo) * (width - 1)).astype(int), 0, width - 1)
        vd = np.clip(((xyz[:, 0] - xlo) / (xhi - xlo) * (dorsal_h - 1)).astype(int), 0, dorsal_h - 1)
        vl = np.clip(((xyz[:, 1] - ylo) / (yhi - ylo) * (lateral_h - 1)).astype(int), 0, lateral_h - 1)
        self.bin_d = vd * width + u
        self.bin_l = vl * width + u
        cnt_d = np.bincount(self.bin_d, minlength=width * dorsal_h).reshape(dorsal_h, width)
        cnt_l = np.bincount(self.bin_l, minlength=width * lateral_h).reshape(lateral_h, width)
        self.bg_d = (np.log1p(cnt_d) / np.log1p(cnt_d.max()) * 70 + 18).astype(np.uint8)
        self.bg_l = (np.log1p(cnt_l) / np.log1p(cnt_l.max()) * 70 + 18).astype(np.uint8)
        self.norm_d = np.maximum(cnt_d, 1).reshape(-1); self.norm_l = np.maximum(cnt_l, 1).reshape(-1)
        self.types = c.neurons.type.fillna("").to_numpy()
        self.type_names, self.type_code = np.unique(self.types, return_inverse=True)
        self.type_count = np.bincount(self.type_code, minlength=len(self.type_names))
        self.rate_idx = optic.rate_idx
        self.is_rate = np.zeros(c.n, bool); self.is_rate[optic.rate_idx] = True
        self.baseline = optic.b_vec.cpu().numpy()
        self.n = c.n

    def activity(self, brain, optic) -> np.ndarray:
        """(N,) activity in [0, 1] for spiking and rate units alike."""
        act = np.clip(brain.rate_np() / 40.0, 0, 1)
        if "dr" in optic.last:
            dr = optic.last["dr"][0].detach().cpu().numpy()
            act[self.rate_idx] = np.clip(np.abs(dr) * 2.0, 0, 1)
        return act

    def render(self, act: np.ndarray):
        """-> (dorsal RGB uint8 (Hd, W, 3), lateral RGB uint8 (Hl, W, 3))."""
        a = act[self.idx]
        out = []
        for bins, norm, bg, h in ((self.bin_d, self.norm_d, self.bg_d, self.Hd), (self.bin_l, self.norm_l, self.bg_l, self.Hl)):
            heat = np.bincount(bins, weights=a, minlength=self.W * h) / np.sqrt(norm)    # sum / sqrt(count): dense regions do not wash out
            heat = np.clip(heat / 1.5, 0, 1).reshape(h, self.W)
            img = np.empty((h, self.W, 3), np.uint8)
            img[..., 0] = np.clip(bg + heat * (255 - bg), 0, 255)
            img[..., 1] = np.clip(bg + heat * (150 - bg), 0, 255)
            img[..., 2] = np.clip(bg + heat * (40 - bg) * 0.3, 0, 255)
            out.append(img)
        return out

    def top_types(self, act: np.ndarray, k: int = 6):
        mean = np.bincount(self.type_code, weights=act, minlength=len(self.type_names)) / np.maximum(self.type_count, 1)
        mean[(self.type_count < 2) | (self.type_names == "")] = -1
        order = np.argsort(-mean)[:k]
        return {str(self.type_names[i]): round(float(mean[i]), 2) for i in order if mean[i] > 0}
