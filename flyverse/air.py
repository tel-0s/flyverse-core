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
        """sources: list of (odour_name, (x, y, z), strength[, radius]). The radius makes the plume's vertical
        extent that of the fruit: odour from a fruit on a substrate fills the boundary layer down to the
        surface, so a walking fly under the fruit's rim is on the plume axis, not 4 cm below it."""
        self.wind = wind or WindParams()
        self.plume = plume or PlumeParams()
        self.sources = [(src[0], np.array(src[1], float), src[2]) for src in sources if src[0] in ODOURS]
        self.src_rad = np.array([(src[3] if len(src) > 3 else 0.0) for src in sources if src[0] in ODOURS], float)
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
        pz = np.maximum(np.abs(perp[..., 2]) - self.src_rad[None, :], 0.0)     # vertical offset outside the fruit's extent
        r2 = perp[..., 0] ** 2 + perp[..., 1] ** 2 + (2 * pz) ** 2
        conc = pp.near_gain / (1.0 + (np.linalg.norm(d, axis=-1) / pp.near_d0) ** 2)
        sig = np.maximum(pp.sigma0, self.src_rad)[None, :] + pp.spread * np.maximum(x, 0)   # a plume starts as wide as its source
        puff = 1.0 + pp.puff_depth * np.sin(2 * np.pi * self.t / pp.puff_period_s + self.phase)[None, :]
        conc = conc + np.where(x > 0, pp.plume_gain * puff * (pp.sigma0 / sig) ** 2 * np.exp(-r2 / (2 * sig ** 2)), 0.0)
        return (conc * strength[None, :]) @ self._odour_mat                            # (M, n_glom)

    def concentration(self, pos) -> dict:
        """{glomerulus: concentration} at one world position."""
        c = self.concentration_batch(np.asarray(pos, float)[None])[0]
        return {g: float(v) for g, v in zip(self.gloms, c)}

    def antennae(self, eye, left, forward, antenna_sep=0.001):
        """Physical concentrations at two antennae, keyed by glomerulus, for one or B poses."""
        eye, left, forward = (np.atleast_2d(x) for x in (eye, left, forward))
        pL = eye + left * antenna_sep / 2 + forward * 0.0005
        pR = eye - left * antenna_sep / 2 + forward * 0.0005
        cL, cR = self.concentration_batch(pL), self.concentration_batch(pR)
        return ({g: cL[:, i] for i, g in enumerate(self.gloms)},
                {g: cR[:, i] for i, g in enumerate(self.gloms)})

    def deflections(self, forward, left, full_speed=0.5):
        """Normalized backward antennal deflections; no neuron identities are involved."""
        fwd, left = np.atleast_2d(forward), np.atleast_2d(left)
        wx, wy = fwd @ self.vector, left @ self.vector
        c45 = np.cos(np.pi / 4)
        return -(wx * c45 + wy * c45) / full_speed, -(wx * c45 - wy * c45) / full_speed


# Compatibility adapters for the existing probes. New environments use Air.antennae /
# Air.deflections and FlyBrain; all neural encoding lives in senses.py.
from .senses import Smell, Wind, laterality as _laterality


class BilateralOlfaction(Smell):
    def __init__(self, c, air: Air, base_hz=1.0, max_hz=150.0, half_conc=0.5,
                 antenna_sep=0.001, side_threshold=0.2):
        super().__init__(c, base_hz, max_hz, half_conc, side_threshold)
        self.air, self.antenna_sep, self.last = air, antenna_sep, {}

    def rates_batch(self, eye, left, forward):
        cL, cR = self.air.antennae(eye, left, forward, self.antenna_sep)
        self.last = {"cL": {g: float(v[0]) for g, v in cL.items()},
                     "cR": {g: float(v[0]) for g, v in cR.items()}}
        return super().rates(cL, cR, len(eye))

    def rates(self, fly):
        return self.rates_batch(fly.eye_pos[None], fly.left[None], fly.forward[None])[0]

    def apply(self, brain, fly):
        brain.set_poisson(self.orn_idx, self.rates(fly))

    def summary(self):
        cL, cR = self.last.get("cL", {}), self.last.get("cR", {})
        gl = sorted(set(cL) | set(cR), key=lambda g: -(cL.get(g, 0) + cR.get(g, 0)))[:3]
        return "  ".join(f"{g}={cL.get(g, 0):.2f}/{cR.get(g, 0):.2f}" for g in gl if cL.get(g, 0) + cR.get(g, 0) > 0.02)


class WindSense(Wind):
    def __init__(self, c, air: Air, max_hz=50.0, base_hz=2.0, full_speed=0.5):
        super().__init__(c, max_hz, base_hz)
        self.air, self.full_speed, self.last = air, full_speed, {}

    def deflections(self, fly):
        dL, dR = self.air.deflections(fly.forward, fly.left, self.full_speed)
        return float(dL[0]), float(dR[0])

    def rates_batch(self, dL, dR):
        return super().rates(dL, dR, len(dL))

    def apply(self, brain, fly):
        dL, dR = self.deflections(fly)
        rE, rC = self.rates_batch(np.array([dL]), np.array([dR]))
        brain.set_poisson(self.joE, rE[0]); brain.set_poisson(self.joC, rC[0])
        self.last = {"dL": dL, "dR": dR}
