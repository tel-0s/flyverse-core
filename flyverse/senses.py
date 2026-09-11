"""Physical sensor values -> neural drive, independent of bodies, air and worlds."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def laterality(c, pre_idx, post_idx_L, post_idx_R):
    Wabs = abs(c.W).tocsr()
    sL = np.asarray(Wabs[post_idx_L][:, pre_idx].sum(axis=0)).ravel()
    sR = np.asarray(Wabs[post_idx_R][:, pre_idx].sum(axis=0)).ravel()
    return (sL - sR) / np.maximum(sL + sR, 1.0)


def batch_values(value, batch, name):
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(batch, float(array))
    if array.shape != (batch,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite and scalar or shape ({batch},)")
    return array


class Smell:
    def __init__(self, c, base_hz=1.0, max_hz=150.0, half_conc=0.5, side_threshold=0.2):
        self.c = c
        self.base_hz, self.max_hz, self.half_conc = base_hz, max_hz, half_conc
        self.orn_idx = c.select(**{"class": "olfactory"})
        self.glom = np.array([t.replace("ORN_", "") for t in c.neurons.type.fillna("").to_numpy()[self.orn_idx]])
        ref = c.reference
        pn = ref.select(type="~_l2PN|_adPN|_lPN|_lvPN|_ilPN|_ivPN|_vPN")
        side = ref.neurons.somaSide.to_numpy()
        pre = ref.index_of(c.neurons.bodyId.to_numpy()[self.orn_idx])
        lat = laterality(ref, pre, pn[side[pn] == "L"], pn[side[pn] == "R"])
        self.side = np.where(lat > side_threshold, 1, np.where(lat < -side_threshold, -1, 0))

    def rates(self, cL: dict, cR: dict, batch=1):
        """Concentrations keyed by glomerulus; omitted keys mean zero concentration."""
        def expand(values):
            if not isinstance(values, dict):
                raise TypeError("concentrations must be dictionaries keyed by glomerulus")
            result = np.zeros((batch, len(self.orn_idx)))
            for name, value in values.items():
                v = batch_values(value, batch, name)
                if np.any(v < 0):
                    raise ValueError("concentrations must be nonnegative")
                result[:, self.glom == name] = v[:, None]
            return result
        left, right = expand(cL), expand(cR)
        concentration = np.where(self.side[None] > 0, left, np.where(self.side[None] < 0, right, .5 * (left + right)))
        return self.base_hz + self.max_hz * concentration / (concentration + self.half_conc)


class Wind:
    def __init__(self, c, max_hz=50.0, base_hz=2.0):
        self.max_hz, self.base_hz = max_hz, base_hz
        self.joC, self.joE = c.select(type="~^JO-C"), c.select(type="~^JO-E")
        ref = c.reference
        targets = ref.select(superclass="cb_intrinsic")
        side = ref.neurons.somaSide.to_numpy()
        tL, tR = targets[side[targets] == "L"], targets[side[targets] == "R"]
        self.sideC = np.sign(laterality(ref, ref.index_of(c.neurons.bodyId.to_numpy()[self.joC]), tL, tR))
        self.sideE = np.sign(laterality(ref, ref.index_of(c.neurons.bodyId.to_numpy()[self.joE]), tL, tR))

    def rates(self, dL, dR, batch=1):
        """Normalized backward deflection: +1 is the existing model's full-speed deflection."""
        left = batch_values(dL, batch, "left deflection")[:, None]
        right = batch_values(dR, batch, "right deflection")[:, None]
        def rate(d):
            return self.base_hz + self.max_hz * np.clip(d, 0, 1)
        rE = np.where(self.sideE[None] > 0, rate(left), np.where(self.sideE[None] < 0, rate(right), rate(.5 * (left + right))))
        rC = np.where(self.sideC[None] > 0, rate(-left), np.where(self.sideC[None] < 0, rate(-right), rate(-.5 * (left + right))))
        return rE, rC


class Taste:
    def __init__(self, c):
        table = pd.read_csv(Path(__file__).parent / "data" / "taste_grns.csv")
        ids = table.bodyId[table.taste == "sweet"].to_numpy()
        self.sweet = c.index_of(ids[np.isin(ids, c.neurons.bodyId)])

    def rates(self, sugar, batch=1):
        contact = batch_values(sugar, batch, "sugar contact")
        if np.any((contact < 0) | (contact > 1)):
            raise ValueError("sugar contact must be between 0 and 1")
        return np.repeat(contact[:, None] * 120.0, len(self.sweet), axis=1)
