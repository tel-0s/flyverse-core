"""Olfaction: fruit odours -> olfactory receptor neurons (ORNs), by glomerulus, as Poisson rates.

The annotations label 2,639 ORNs by target glomerulus (ORN_DA1, ORN_DM2, ...; 53 classes). Each fruit
in the world emits an odour "plume" (isotropic, concentration falling as 1/(1 + (d/d0)^2)); each odour
is a sparse vector over glomeruli taken from the DoOR / Hallem & Carlson tuning literature (rough):

    banana   isoamyl acetate, ethyl butyrate      -> DM2 (Or22a), VM2 (Or43b), DM3 (Or47a), VA2 (Or92a)
    apple    ethyl acetate, hexanol, acetic acid   -> DM1 (Or42b), DM4 (Or59b), VA2 (Or92a), DC1 (Or19a)
    orange   limonene, valencene                   -> DC1 (Or19a), DL5 (Or7a), DA4m (Or2a)
    lime     limonene, citral                      -> DC1 (Or19a), DA4m (Or2a), VC4 (Or67c)
    grape    vinegar-ish, 2-phenylethanol          -> DM1, DM4, VA2 (Or92a), DA1? no (pheromone) -> VM7d (Or42a)
    blueberry ethyl butyrate, ethyl hexanoate      -> DM2, DM4, DM3
    vinegar  (any rotting fruit) acetic acid        -> DM1, DP1m (Or33b), VM4 (Ir84a-ish)

ORNs of a glomerulus fire at rate = base + max_rate * saturating(total concentration * weight). Real
ORNs have spontaneous rates (~5-20 Hz) -- we give every ORN `base_hz`, which also puts a steady drive
into the antennal lobe. All ORNs are cholinergic (verified in the NT table).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .connectome import Connectome

ODOURS = {
    "banana":    {"DM2": 1.0, "VM2": 0.8, "DM3": 0.6, "VA2": 0.5},
    "apple":     {"DM1": 1.0, "DM4": 0.7, "VA2": 0.6, "DC1": 0.3},
    "orange":    {"DC1": 1.0, "DL5": 0.6, "DA4m": 0.5},
    "lime":      {"DC1": 1.0, "DA4m": 0.6, "VC4": 0.5},
    "grape":     {"DM1": 0.7, "DM4": 0.6, "VA2": 0.5, "VM7d": 0.6},
    "blueberry": {"DM2": 0.8, "DM4": 0.7, "DM3": 0.5},
    "vinegar":   {"DM1": 1.0, "DP1m": 0.8, "VM4": 0.5},
}


@dataclass
class OlfactionParams:
    d0: float = 0.10          # m, distance at which concentration halves
    max_hz: float = 150.0     # ORN rate at saturating concentration (Hallem & Carlson: up to ~250 Hz)
    base_hz: float = 3.0      # spontaneous ORN rate (real ~8-20 Hz, but Shiu-strength ORN->PN synapses saturate PNs)
    half_conc: float = 0.5    # concentration giving half-max response


class Olfaction:
    def __init__(self, c: Connectome, sources: list, params: OlfactionParams | None = None):
        """sources: list of (odour_name, (x, y, z), strength)."""
        self.c, self.p = c, params or OlfactionParams()
        self.sources = [(name, np.array(pos, float), s) for name, pos, s in sources if name in ODOURS]
        n = c.neurons
        orn = n["class"].fillna("") == "olfactory"
        self.orn_idx = np.flatnonzero(orn.to_numpy())
        glom = n.type.fillna("").to_numpy()[self.orn_idx]
        self.glom = np.array([g.replace("ORN_", "") for g in glom])
        self.gloms = sorted(set(self.glom))
        self.glom_id = {g: i for i, g in enumerate(self.gloms)}
        self.orn_glom_id = np.array([self.glom_id[g] for g in self.glom])
        self.last = {}

    def concentrations(self, pos) -> np.ndarray:
        """(n_glom,) odour concentration per glomerulus at position pos."""
        conc = np.zeros(len(self.gloms))
        pos = np.asarray(pos, float)
        for name, spos, strength in self.sources:
            d = np.linalg.norm(pos - spos)
            cd = strength / (1.0 + (d / self.p.d0) ** 2)
            for g, w in ODOURS[name].items():
                if g in self.glom_id:
                    conc[self.glom_id[g]] += cd * w
        return conc

    def rates(self, pos) -> np.ndarray:
        """(n_orn,) Poisson rate (Hz) for every ORN."""
        conc = self.concentrations(pos)
        resp = conc / (conc + self.p.half_conc)
        r = self.p.base_hz + self.p.max_hz * resp
        self.last = {"conc": conc}
        return r[self.orn_glom_id]

    def apply(self, brain, pos) -> None:
        brain.set_poisson(self.orn_idx, self.rates(pos))

    def summary(self) -> str:
        conc = self.last.get("conc")
        if conc is None:
            return ""
        top = np.argsort(conc)[::-1][:4]
        return "  ".join(f"{self.gloms[i]}={conc[i]:.2f}" for i in top if conc[i] > 0.01)
