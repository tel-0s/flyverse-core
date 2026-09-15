"""Compound-eye geometry and spectral sensitivity for the MaleCNS photoreceptors.

Every photoreceptor has been assigned to an optic-lobe hex column (side, hex1, hex2) by connectome.py.
Here each column gets a viewing direction in the fly's body frame, and each photoreceptor gets a
spectral sensitivity over the world's light channels.

Hex axes (from soma-position regressions of L1/Mi1 against assignedOlHex1/2):
    hex1 + hex2  increases towards DORSAL
    hex1 - hex2  increases towards ANTERIOR  (lamina topology; the medulla is chiasm-inverted)
The two hex axes are 120 deg apart, so the Cartesian embedding is X = h1 - h2/2, Y = h2*sqrt(3)/2, in
which the (h1+h2) and (h1-h2) directions are orthogonal. Column spacing ~ 4.6 deg (Drosophila
interommatidial angle); the eye comes out ~33 columns tall x ~26 wide (~154 x 120 deg).

Light channels everywhere in flyverse are [UV, B, G, R] (R is for humans only; flies barely see it).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .connectome import Connectome, PHOTORECEPTOR_TYPES

CHANNELS = ["UV", "B", "G", "R"]

# Relative spectral sensitivity of each photoreceptor class to the four light channels.
# Rh1 (R1-R6) peaks ~480nm with a UV-sensitizing pigment; Rh3/Rh4 (R7p/R7y) are UV; Rh5 (R8p) blue;
# Rh6 (R8y) green; DRA R7d/R8d both use Rh3 (UV, polarization-sensitive).
SENSITIVITY = {
    "R1-R6":        [0.50, 0.70, 1.00, 0.05],
    "R7p":          [1.00, 0.05, 0.00, 0.00],
    "R7y":          [1.00, 0.15, 0.00, 0.00],
    "R7d":          [1.00, 0.05, 0.00, 0.00],
    "R7_unclear":   [1.00, 0.10, 0.00, 0.00],
    "R8p":          [0.20, 1.00, 0.10, 0.00],
    "R8y":          [0.05, 0.30, 1.00, 0.10],
    "R8d":          [1.00, 0.05, 0.00, 0.00],
    "R8_unclear":   [0.12, 0.65, 0.55, 0.05],
    "R7R8_unclear": [0.60, 0.40, 0.30, 0.02],
}


@dataclass
class EyeGeometry:
    interommatidial_deg: float = 4.6
    eye_center_azimuth_deg: float = 55.0     # lateral offset of each eye's central column (field ~ -5..115 deg)
    anterior_sign: float = +1.0              # +1: anterior = increasing (h1 - h2)
    dorsal_sign: float = +1.0                # +1: dorsal   = increasing (h1 + h2)
    acceptance_deg: float = 4.5              # FWHM-ish of the ommatidial acceptance function
    rays_per_ommatidium: int = 7             # 1 = centre only; 7 = centre + hexagon


@dataclass
class Retina:
    pr_index: np.ndarray            # connectome row index of each photoreceptor (n_pr,)
    pr_column: np.ndarray           # column id (0..n_col-1) of each photoreceptor
    pr_sens: np.ndarray             # (n_pr, 4) spectral sensitivity
    col_side: np.ndarray            # 'L'/'R' per column
    col_hex: np.ndarray             # (n_col, 2) hex1, hex2
    col_dir: np.ndarray             # (n_col, 3) unit view direction in body frame (x fwd, y left, z up)
    col_az_el: np.ndarray           # (n_col, 2) azimuth (deg, +left), elevation (deg, +up)
    geometry: EyeGeometry = field(default_factory=EyeGeometry)

    @property
    def n_columns(self) -> int:
        return len(self.col_side)

    def ray_directions(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-column sample rays: (n_col, k, 3) unit directions and (k,) weights, body frame."""
        g = self.geometry
        k = g.rays_per_ommatidium
        if k <= 1:
            return self.col_dir[:, None, :], np.ones(1)
        r = np.deg2rad(g.acceptance_deg / 2)
        ang = np.linspace(0, 2 * np.pi, k - 1, endpoint=False)
        offs = np.concatenate([[[0.0, 0.0]], np.c_[r * np.cos(ang), r * np.sin(ang)]])   # (k, 2)
        w = np.exp(-0.5 * (np.hypot(offs[:, 0], offs[:, 1]) / r) ** 2 * 2.0)
        w /= w.sum()
        az = np.deg2rad(self.col_az_el[:, 0])[:, None] + offs[None, :, 0]
        el = np.deg2rad(self.col_az_el[:, 1])[:, None] + offs[None, :, 1]
        dirs = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)
        return dirs, w

    def photoreceptor_intensity(self, col_radiance: np.ndarray | torch.Tensor) -> np.ndarray:
        """(n_col, 4) radiance per column -> (n_pr,) scalar light intensity per photoreceptor."""
        if isinstance(col_radiance, torch.Tensor):
            col_radiance = col_radiance.detach().cpu().numpy()
        return np.einsum("pc,pc->p", col_radiance[self.pr_column], self.pr_sens)


def build_retina(c: Connectome, geometry: EyeGeometry | None = None) -> Retina:
    c.require("optic_columns")
    g = geometry or EyeGeometry()
    if c.reference is not c:
        # Pruning columns must not recenter the remaining eye's viewing directions.
        full = build_retina(c.reference, g)
        body_ids = c.reference.neurons.bodyId.to_numpy()[full.pr_index]
        keep = np.isin(body_ids, c.neurons.bodyId.to_numpy())
        columns = np.unique(full.pr_column[keep])
        col_map = np.full(full.n_columns, -1, dtype=np.int64)
        col_map[columns] = np.arange(len(columns))
        return Retina(pr_index=c.index_of(body_ids[keep]), pr_column=col_map[full.pr_column[keep]],
                      pr_sens=full.pr_sens[keep], col_side=full.col_side[columns],
                      col_hex=full.col_hex[columns], col_dir=full.col_dir[columns],
                      col_az_el=full.col_az_el[columns], geometry=g)
    nrn = c.neurons
    m = nrn.type.isin(PHOTORECEPTOR_TYPES) & nrn.hex1.notna()
    pr = nrn[m]
    pr_index = np.flatnonzero(m.to_numpy())

    keys = list(zip(pr.hex_side, pr.hex1.astype(int), pr.hex2.astype(int)))
    uniq = sorted(set(keys))
    if c.dataset == "fafb":
        # The release supplies the complete column grid independently of R-cell
        # reconstruction. Keep annotated columns even if no R axon reaches them.
        columns = nrn[nrn.hex1.notna() & nrn.hex2.notna() & nrn.hex_side.notna()]
        uniq = sorted(set(uniq) | set(zip(columns.hex_side, columns.hex1.astype(int), columns.hex2.astype(int))))
    col_of = {k: i for i, k in enumerate(uniq)}
    pr_column = np.array([col_of[k] for k in keys])
    col_side = np.array([k[0] for k in uniq])
    col_hex = np.array([[k[1], k[2]] for k in uniq], dtype=float).reshape(-1, 2)
    pr_sens = np.array([SENSITIVITY[t] for t in pr.type], dtype=np.float32)

    # hex -> Cartesian (column spacing = 1). The two hex axes are 120 deg apart ((1,1) is a nearest
    # neighbour): the occupied grid is a diagonal band elongated along h1+h2, and with 120-deg axes the
    # eye comes out ~33 x 26 columns (154 x 120 deg), the right aspect ratio for a Drosophila eye.
    X = col_hex[:, 0] - 0.5 * col_hex[:, 1]
    Y = col_hex[:, 1] * np.sqrt(3) / 2
    e_dv = 0.5 * X + (np.sqrt(3) / 2) * Y        # unit vector along (h1+h2): 0.5*(h1+h2)
    e_ap = (np.sqrt(3) / 2) * X - 0.5 * Y        # unit vector along (h1-h2): 0.866*(h1-h2)
    az = np.zeros(len(uniq)); el = np.zeros(len(uniq))
    for side in ("L", "R"):
        s = col_side == side
        if not s.any():
            continue
        ap = (e_ap[s] - e_ap[s].mean()) * g.anterior_sign * g.interommatidial_deg   # + = anterior
        dv = (e_dv[s] - e_dv[s].mean()) * g.dorsal_sign * g.interommatidial_deg     # + = dorsal
        lateral = g.eye_center_azimuth_deg - ap                                        # + = lateral
        az[s] = lateral if side == "L" else -lateral                                   # azimuth + = left
        el[s] = dv
    azr, elr = np.deg2rad(az), np.deg2rad(el)
    col_dir = np.stack([np.cos(elr) * np.cos(azr), np.cos(elr) * np.sin(azr), np.sin(elr)], axis=-1)

    return Retina(pr_index=pr_index, pr_column=pr_column, pr_sens=pr_sens, col_side=col_side,
                  col_hex=col_hex, col_dir=col_dir, col_az_el=np.c_[az, el], geometry=g)


def summarize(r: Retina, c: Connectome) -> str:
    t = c.neurons.type.to_numpy()[r.pr_index]
    lines = [f"retina: {len(r.pr_index)} photoreceptors in {r.n_columns} columns "
             f"(L {int((r.col_side == 'L').sum())}, R {int((r.col_side == 'R').sum())})"]
    for side in ("L", "R"):
        s = r.col_side == side
        lines.append(f"  {side}: azimuth {r.col_az_el[s, 0].min():.0f}..{r.col_az_el[s, 0].max():.0f} deg, "
                     f"elevation {r.col_az_el[s, 1].min():.0f}..{r.col_az_el[s, 1].max():.0f} deg")
    for fam, pat in [("R1-R6", "R1-R6"), ("R7", "R7"), ("R8", "R8")]:
        sel = np.char.startswith(t.astype(str), pat)
        lines.append(f"  {fam}: {int(sel.sum())} cells over {len(np.unique(r.pr_column[sel]))} columns")
    # dorsal-rim check: DRA photoreceptors should sit at the dorsal edge (high elevation)
    dra = np.isin(t, ["R7d", "R8d"])
    if dra.any():
        el_dra = r.col_az_el[r.pr_column[dra], 1]
        lines.append(f"  DRA (R7d/R8d) mean elevation {el_dra.mean():+.1f} deg (eye mean {r.col_az_el[:, 1].mean():+.1f}); "
                     f"should be strongly positive if dorsal_sign is right")
    return "\n".join(lines)
