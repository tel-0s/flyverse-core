"""Air: wind, odour plumes, and the two sensory channels that read them (antennae -> ORNs by side,
antennal deflection -> Johnston's organ wind neurons by side).

Flies find fruit by odour-gated anemotaxis: an odour hit gates upwind turning, losing the plume
triggers crosswind casting. That needs (1) a plume that is directional, (2) a wind cue. Here:

  Wind      a horizontal wind vector with slow random meander of its direction.
  Plume     every fruit is a source; downwind of it a Gaussian plume (sigma growing with distance)
            carries its odour; a near-source isotropic term makes touching fruit intense. A slow
            per-source gain fluctuation (0.5-1.5) gives the puffiness of real plumes.
  Antennae  the two antennae sample the plume 1 mm apart; ORNs are assigned to the left or right
            antenna by the laterality of their projection-neuron targets (each ORN innervates both
            antennal lobes, ipsilateral more strongly: 46% are lateralised beyond 0.5), so left and
            right antennae drive different ORN sets.
  JO wind   Johnston's organ C and E neurons respond to sustained antennal deflection (Yorozu et al.
            2009): wind pushes the arista back (E) or forward (C). The antennae point forward-lateral
            at +-45 deg, so a crosswind deflects the two sides differently -- that asymmetry is how the
            fly knows the wind direction. JO neurons have no soma side in the annotations; they are
            sided by the laterality of their AMMC/WED targets, like the ORNs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .olfaction import ODOURS


@dataclass
class WindParams:
    speed: float = 0.3                 # m/s
    direction_deg: float = 180.0       # where the wind blows TOWARDS (world frame, 0 = +x); 180: from +x (the door) to -x
    meander_deg: float = 20.0          # amplitude of the slow direction wander
    meander_period_s: float = 8.0


@dataclass
class PlumeParams:
    sigma0: float = 0.015              # m, plume width at the source
    spread: float = 0.12               # sigma grows by this fraction of the downwind distance
    near_d0: float = 0.03              # m, half-distance of the isotropic near-source term
    near_gain: float = 1.0
    plume_gain: float = 3.0            # concentration on the plume axis just downwind of the source
    puff_period_s: float = 1.5         # per-source gain fluctuation
    puff_depth: float = 0.5


class Air:
    def __init__(self, sources, wind: WindParams | None = None, plume: PlumeParams | None = None, seed: int = 0):
        """sources: list of (odour_name, (x, y, z), strength)."""
        self.wind = wind or WindParams()
        self.plume = plume or PlumeParams()
        self.sources = [(name, np.array(pos, float), s) for name, pos, s in sources if name in ODOURS]
        self.rng = np.random.default_rng(seed)
        self.phase = self.rng.uniform(0, 2 * np.pi, len(self.sources))
        self.t = 0.0

    def step(self, dt_s: float) -> None:
        self.t += dt_s

    @property
    def direction(self) -> float:
        """Current wind direction (rad, towards) including meander."""
        w = self.wind
        return np.deg2rad(w.direction_deg + w.meander_deg * np.sin(2 * np.pi * self.t / w.meander_period_s))

    @property
    def vector(self) -> np.ndarray:
        d = self.direction
        return self.wind.speed * np.array([np.cos(d), np.sin(d), 0.0])

    @property
    def gloms(self):
        if not hasattr(self, "_gloms"):
            self._gloms = sorted({g for name, _, _ in self.sources for g in ODOURS[name]})
            gid = {g: i for i, g in enumerate(self._gloms)}
            self._odour_mat = np.zeros((len(self.sources), len(self._gloms)))       # source -> glomerulus weights
            for k, (name, _, _) in enumerate(self.sources):
                for g, w in ODOURS[name].items():
                    self._odour_mat[k, gid[g]] = w
        return self._gloms

    def concentration_batch(self, pos: np.ndarray) -> np.ndarray:
        """(M, 3) positions -> (M, n_glom) concentrations (glomeruli in self.gloms order)."""
        pos = np.atleast_2d(np.asarray(pos, float))
        gl = self.gloms
        if not self.sources:
            return np.zeros((pos.shape[0], len(gl)))
        pp = self.plume
        d_hat = self.vector / max(self.wind.speed, 1e-6)
        src = np.stack([s for _, s, _ in self.sources]); strength = np.array([st for _, _, st in self.sources])
        d = pos[:, None, :] - src[None, :, :]                                        # (M, S, 3)
        x = d @ d_hat                                                                  # (M, S) downwind distance
        perp = d - x[..., None] * d_hat
        r2 = perp[..., 0] ** 2 + perp[..., 1] ** 2 + (2 * perp[..., 2]) ** 2
        conc = pp.near_gain / (1.0 + (np.linalg.norm(d, axis=-1) / pp.near_d0) ** 2)
        sig = pp.sigma0 + pp.spread * np.maximum(x, 0)
        puff = 1.0 + pp.puff_depth * np.sin(2 * np.pi * self.t / pp.puff_period_s + self.phase)[None, :]
        conc = conc + np.where(x > 0, pp.plume_gain * puff * (pp.sigma0 / sig) ** 2 * np.exp(-r2 / (2 * sig ** 2)), 0.0)
        return (conc * strength[None, :]) @ self._odour_mat                            # (M, n_glom)

    def concentration(self, pos) -> dict:
        """{glomerulus: concentration} at one world position."""
        c = self.concentration_batch(np.asarray(pos, float)[None])[0]
        return {g: float(v) for g, v in zip(self.gloms, c)}


# ------------------------------------------------------------------------------------------ senses
def _laterality(c, pre_idx, post_idx_L, post_idx_R):
    Wabs = abs(c.W).tocsr()
    sL = np.asarray(Wabs[post_idx_L][:, pre_idx].sum(axis=0)).ravel()
    sR = np.asarray(Wabs[post_idx_R][:, pre_idx].sum(axis=0)).ravel()
    return (sL - sR) / np.maximum(sL + sR, 1.0)


class BilateralOlfaction:
    """ORN Poisson rates from the plume sampled at each antenna."""

    def __init__(self, c, air: Air, base_hz: float = 1.0, max_hz: float = 150.0, half_conc: float = 0.5,
                 antenna_sep: float = 0.001, side_threshold: float = 0.2):
        self.c, self.air = c, air
        self.base_hz, self.max_hz, self.half_conc, self.antenna_sep = base_hz, max_hz, half_conc, antenna_sep
        n = c.neurons
        self.orn_idx = np.flatnonzero((n["class"] == "olfactory").to_numpy())
        self.glom = np.array([t.replace("ORN_", "") for t in n.type.fillna("").to_numpy()[self.orn_idx]])
        pn = c.select(type="~_l2PN|_adPN|_lPN|_lvPN|_ilPN|_ivPN|_vPN")
        side = n.somaSide.to_numpy(dtype=object)
        lat = _laterality(c, self.orn_idx, pn[side[pn] == "L"], pn[side[pn] == "R"])
        self.side = np.where(lat > side_threshold, 1, np.where(lat < -side_threshold, -1, 0))   # +1 left antenna, -1 right, 0 both
        self.last = {}

    def _orn_matrix(self):
        """(n_orn, n_glom) 0/1 map from the air's glomerulus list to ORNs (built lazily: sources may change)."""
        gl = self.air.gloms
        if getattr(self, "_mat_key", None) != tuple(gl):
            gid = {g: i for i, g in enumerate(gl)}
            M = np.zeros((len(self.orn_idx), len(gl)))
            for i, g in enumerate(self.glom):
                if g in gid:
                    M[i, gid[g]] = 1.0
            self._M = M; self._mat_key = tuple(gl)
        return self._M

    def rates_batch(self, eye: np.ndarray, left: np.ndarray, forward: np.ndarray) -> np.ndarray:
        """(B, 3) eye positions, body left/forward vectors -> (B, n_orn) Poisson rates."""
        pL = eye + left * self.antenna_sep / 2 + forward * 0.0005
        pR = eye - left * self.antenna_sep / 2 + forward * 0.0005
        M = self._orn_matrix()
        gL = self.air.concentration_batch(pL); gR = self.air.concentration_batch(pR)     # (B, n_glom)
        cL = gL @ M.T; cR = gR @ M.T                                                       # (B, n_orn)
        c = np.where(self.side[None] > 0, cL, np.where(self.side[None] < 0, cR, 0.5 * (cL + cR)))
        self.last = {"cL": dict(zip(self.air.gloms, gL[0])), "cR": dict(zip(self.air.gloms, gR[0]))}
        return self.base_hz + self.max_hz * c / (c + self.half_conc)

    def rates(self, fly) -> np.ndarray:
        return self.rates_batch(fly.eye_pos[None], fly.left[None], fly.forward[None])[0]

    def apply(self, brain, fly) -> None:
        brain.set_poisson(self.orn_idx, self.rates(fly))

    def summary(self) -> str:
        cL, cR = self.last.get("cL", {}), self.last.get("cR", {})
        gl = sorted(set(cL) | set(cR), key=lambda g: -(cL.get(g, 0) + cR.get(g, 0)))[:3]
        return "  ".join(f"{g}={cL.get(g, 0):.2f}/{cR.get(g, 0):.2f}" for g in gl if cL.get(g, 0) + cR.get(g, 0) > 0.02)


class WindSense:
    """Johnston's organ C/E neurons driven by wind-induced antennal deflection, per side."""

    def __init__(self, c, air: Air, max_hz: float = 50.0, base_hz: float = 2.0, full_speed: float = 0.5):
        self.c, self.air, self.max_hz, self.base_hz, self.full_speed = c, air, max_hz, base_hz, full_speed
        n = c.neurons
        t = n.type.fillna("")
        self.joC = np.flatnonzero(t.str.match(r"^JO-C").to_numpy())
        self.joE = np.flatnonzero(t.str.match(r"^JO-E").to_numpy())
        side = n.somaSide.to_numpy(dtype=object)
        targets = np.flatnonzero((n.superclass == "cb_intrinsic").to_numpy())
        tL, tR = targets[side[targets] == "L"], targets[side[targets] == "R"]
        self.sideC = np.sign(_laterality(c, self.joC, tL, tR))
        self.sideE = np.sign(_laterality(c, self.joE, tL, tR))
        self.last = {}

    def deflections(self, fly):
        """Backward deflection (+) of the left and right antenna from the wind, in body frame."""
        w = self.air.vector
        R = np.stack([fly.forward, fly.left, fly.up], axis=1)
        wb = R.T @ w                                                        # wind in body frame
        aL = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)]); aR = np.array([np.cos(np.pi / 4), -np.sin(np.pi / 4)])
        # wind blowing against an antenna's axis pushes it back: backward deflection = -(w . axis)
        dL = -float(wb[:2] @ aL) / self.full_speed
        dR = -float(wb[:2] @ aR) / self.full_speed
        return dL, dR

    def rates_batch(self, dL: np.ndarray, dR: np.ndarray):
        """(B,) backward deflections per side -> ((B, nE), (B, nC)) Poisson rates."""
        def rate(d):
            return self.base_hz + self.max_hz * np.clip(d, 0, 1)
        dL, dR = dL[:, None], dR[:, None]
        rE = np.where(self.sideE[None] > 0, rate(dL), np.where(self.sideE[None] < 0, rate(dR), rate(0.5 * (dL + dR))))
        rC = np.where(self.sideC[None] > 0, rate(-dL), np.where(self.sideC[None] < 0, rate(-dR), rate(-0.5 * (dL + dR))))
        return rE, rC

    def apply(self, brain, fly) -> None:
        dL, dR = self.deflections(fly)
        rE, rC = self.rates_batch(np.array([dL]), np.array([dR]))
        brain.set_poisson(self.joE, rE[0])
        brain.set_poisson(self.joC, rC[0])
        self.last = {"dL": dL, "dR": dR}
