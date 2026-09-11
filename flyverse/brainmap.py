"""Brain map: every neuron's soma projected onto two views, activity drawn as glowing highlights.

Soma positions come from `somaLocation` in the annotations (8 nm voxels; x = left-right, y = dorsal-
ventral, z = anterior-posterior; the brain sits at low z, the VNC at high z). Two views are drawn:
dorsal (z across, x down: brain left, VNC right, both optic lobes top/bottom) and lateral (z across,
y down). Neurons are binned into pixels once; per frame the activity vector is summed per pixel with
np.bincount, so 167k neurons cost well under a millisecond.

Activity: spiking neurons use their running rate (Hz / 40, clipped to 1); optic-lobe rate units use
|r - baseline| * 2 (clipped to 1), so both halves of the hybrid model light up on the same scale.
Optional scalar fields use render_field, with explicit ranges and missing samples. The projection
does not know neurotransmitter identities, dynamics, or the implementation supplying a field.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import connectome


class BrainMap:
    def __init__(self, c, optic=None, width=300, dorsal_h=220, lateral_h=155, *, locations=None):
        """Project a connectome, with optional graded activity and external soma coordinates.

        ``locations`` is a Series or mapping of bodyId -> xyz. Prefer coordinates in the
        neuron table when present; otherwise load the dataset annotations. Row alignment
        always follows the supplied connectome, including reordered subsets.
        """
        if min(width, dorsal_h, lateral_h) < 1:
            raise ValueError("brain map dimensions must be positive")
        if locations is None:
            if "somaLocation" in c.neurons:
                locations = c.neurons.set_index("bodyId").somaLocation
            else:
                ann = pd.read_feather(connectome.DATA_DIR / connectome.ANNOT_FILE, columns=["bodyId", "somaLocation"])
                locations = ann.set_index("bodyId").somaLocation
        loc = pd.Series(locations).reindex(c.neurons.bodyId)
        coords = np.full((c.n, 3), np.nan, dtype=np.float32)
        for i, value in enumerate(loc):
            if isinstance(value, (list, tuple, np.ndarray)) and np.shape(value) == (3,):
                coords[i] = value
        self.has = np.isfinite(coords).all(axis=1)
        xyz = coords[self.has]
        self.idx = np.flatnonzero(self.has)
        # Robust extents, including empty subsets and degenerate coordinate axes.
        lo, hi = np.percentile(xyz, [0.5, 99.5], axis=0) if len(xyz) else (np.zeros(3), np.ones(3))
        span = np.where(hi > lo, hi-lo, 1.)
        self.W, self.Hd, self.Hl = width, dorsal_h, lateral_h
        u = np.clip(((xyz[:, 2] - lo[2]) / span[2] * (width - 1)).astype(int), 0, width - 1)
        vd = np.clip(((xyz[:, 0] - lo[0]) / span[0] * (dorsal_h - 1)).astype(int), 0, dorsal_h - 1)
        vl = np.clip(((xyz[:, 1] - lo[1]) / span[1] * (lateral_h - 1)).astype(int), 0, lateral_h - 1)
        self.bin_d = vd * width + u
        self.bin_l = vl * width + u
        cnt_d = np.bincount(self.bin_d, minlength=width * dorsal_h).reshape(dorsal_h, width)
        cnt_l = np.bincount(self.bin_l, minlength=width * lateral_h).reshape(lateral_h, width)
        self.bg_d = (np.log1p(cnt_d) / np.log1p(max(1,cnt_d.max())) * 70 + 18).astype(np.uint8)
        self.bg_l = (np.log1p(cnt_l) / np.log1p(max(1,cnt_l.max())) * 70 + 18).astype(np.uint8)
        self.norm_d = np.maximum(cnt_d, 1).reshape(-1); self.norm_l = np.maximum(cnt_l, 1).reshape(-1)
        self.types = c.neurons.get("type", pd.Series("", index=c.neurons.index)).fillna("").to_numpy()
        self.type_names, self.type_code = np.unique(self.types, return_inverse=True)
        self.type_count = np.bincount(self.type_code, minlength=len(self.type_names))
        self.rate_idx = np.asarray(optic.rate_idx) if optic is not None else np.empty(0, dtype=np.int64)
        self.n = c.n
        self.body_ids = c.neurons.bodyId.to_numpy(copy=True)

    def activity(self, brain, optic=None) -> np.ndarray:
        """(N,) activity in [0, 1] for spiking and rate units alike."""
        act = np.clip(brain.rate_np() / 40.0, 0, 1)
        if optic is not None and "dr" in optic.last:
            dr = optic.last["dr"][0].detach().cpu().numpy()
            act[optic.rate_idx] = np.clip(np.abs(dr) * 2.0, 0, 1)
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

    def render_field(self, values, minimum, maximum, color=(107,201,208)):
        """Project scalar samples in declared physical units, without changing them.

        Samples sharing a pixel are averaged *before* display clipping. NaN means no
        sample; gray anatomy remains there. Valid samples use a fixed linear colour
        scale; no frame-wise normalization and no inference from neural activity.
        """
        values = np.asarray(values, dtype=np.float64)
        color = np.asarray(color, dtype=np.float64)
        if values.shape != (self.n,) or np.isinf(values).any():
            raise ValueError(f"field values must have shape ({self.n},), finite or NaN")
        if not np.isfinite([minimum,maximum]).all() or maximum <= minimum:
            raise ValueError("field display range must be finite and increasing")
        if color.shape != (3,) or not np.isfinite(color).all() or np.any((color<0)|(color>255)):
            raise ValueError("field color must be RGB in [0, 255]")
        samples = values[self.idx]
        valid = np.isfinite(samples)
        low = np.array([21,37,43.])
        out = []
        for bins,bg,h in ((self.bin_d,self.bg_d,self.Hd),(self.bin_l,self.bg_l,self.Hl)):
            count = np.bincount(bins[valid],minlength=self.W*h)
            sums = np.bincount(bins[valid],weights=samples[valid],minlength=self.W*h)
            mean = sums / np.maximum(count,1)
            fraction = np.clip((mean-minimum)/(maximum-minimum),0,1)
            image = np.repeat(bg.reshape(-1,1),3,axis=1).astype(np.float64)
            observed = count>0
            image[observed] = low+fraction[observed,None]*(color-low)
            out.append(image.reshape(h,self.W,3).astype(np.uint8))
        return out

    def top_types(self, act: np.ndarray, k: int = 6):
        mean = np.bincount(self.type_code, weights=act, minlength=len(self.type_names)) / np.maximum(self.type_count, 1)
        mean[(self.type_count < 2) | (self.type_names == "")] = -1
        order = np.argsort(-mean)[:k]
        return {str(self.type_names[i]): round(float(mean[i]), 2) for i in order if mean[i] > 0}
