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


class Proprioception:
    """Opt-in transducer: the body state the VNC motor neurons produce -> proprioceptive afferent rates (Hz).

    Nothing builds this by default; ``FlyBrain`` grows a ``proprioception`` sense only when a caller attaches an
    instance (``BatchSim(..., proprioception='all')``), so the shipped path is untouched. The afferent populations exist
    in MaleCNS and are wired (docs/audits/deficit_turning.md 6.3, deficit_rotation.md 3(3)) but never driven; this
    class drives them the way ``Wind`` drives JO-C / JO-E: a physical variable of the body model -> Hz, injected through
    ``FlyBrain._input`` as Poisson spikes. Channels (docs/audits/proprioception_transducer.md for the laws, literature
    ranges, cell counts and the classification of each under the project rule):

      chordotonal  class mechanosensory_proprioceptive, subclass 'chordotonal organ' | 'leg' (SNpp39/50/60/52, the
                   unnamed leg proprioceptors SNppxx, SApp23): femoral chordotonal organ. rate = tonic + (max - tonic)
                   * clip(legMN_side / mn_ref, 0, 1) on the ground, tonic when airborne. The model has NO leg cycle
                   (no stance / swing / joint angle: body.FlyState carries speed, yaw_rate, airborne only), so the only
                   leg state the VNC produces is the side's leg motor-neuron rate (MotorRates.leg_L / leg_R); that is
                   the documented limitation of this channel, not a claim that FeCO rate follows MN rate in the animal.
      hair_plate   subclass 'hair plate' (SNpp45, SNpp19 ...): joint-angle proxy, the same MN-rate law with its own
                   literature range (tonic when airborne).
      campaniform  subclass 'campaniform sensilla' with a LEG entry nerve (SNpp53, 12 cells; the 414 wing (ADMN) and
                   haltere (DMetaN) campaniform cells are not leg load and are excluded): load = ground contact x body
                   weight support: load_hz on the ground, 0 airborne.
      haltere      subclass 'haltere' (SApp 148 in sensory_ascending + 53 in vnc_sensory): rate = k_h * haltere motor
                   rate (WingGroups.haltere; MotorRates.haltere is one bilateral mean -- the group is not side-split),
                   wingbeat term only. The Coriolis term (1 + g |yaw_rate|) is a STOP-GAP: the only body angular rate
                   is body.Locomotion's realised yaw_rate, computed from the DNa02 / leg-MN readout, so feeding it back
                   closes a loop through a hand-written module. It is off unless the spec names 'haltere_coriolis'
                   (a labelled control arm), never the default.

    Sides: an afferent's side is the '_L' / '_R' suffix of its MaleCNS ``instance`` when present (every
    sensory_ascending cell), else the sign of its output laterality on the reference graph against left- vs right-soma
    targets (``laterality``, threshold ``side_threshold`` as in ``Smell``; the vnc_sensory cells carry no somaSide and
    no instance suffix). Left afferents read the left MN rate; cells with |laterality| <= threshold read the mean of
    the two sides, as ``Wind`` does. Every rate is clamped to its channel's literature ceiling.

    Spec grammar: 'all' | 'all+haltere_coriolis' | comma-separated channel names, 'haltere_coriolis' meaning the
    haltere channel with the stop-gap term on. Parameters are Hz; ``mn_ref_hz`` is the leg-MN rate at which the
    MN-rate channels saturate (body.Locomotion's own reference: the 30 Hz its k_leg_turn is quoted per). No parameter
    here was chosen to make the fly turn.
    """
    CHANNELS = ("chordotonal", "hair_plate", "campaniform", "haltere")
    LEG_NERVES = ("ProLN", "MesoLN", "MetaLN", "ProAN", "VProN", "DProN", "ProCN")

    def __init__(self, c, channels="all", *, side_threshold=0.2, mn_ref_hz=30.0,
                 chordotonal_tonic_hz=10.0, chordotonal_max_hz=150.0,
                 hair_plate_tonic_hz=5.0, hair_plate_max_hz=100.0,
                 campaniform_load_hz=50.0, campaniform_max_hz=100.0,
                 haltere_k=1.0, haltere_max_hz=250.0, coriolis_gain_per_rad_s=1.0):
        self.c = c
        self.channels, self.haltere_coriolis = self.parse_spec(channels)
        for name, value in (("mn_ref_hz", mn_ref_hz), ("chordotonal_tonic_hz", chordotonal_tonic_hz),
                            ("chordotonal_max_hz", chordotonal_max_hz), ("hair_plate_tonic_hz", hair_plate_tonic_hz),
                            ("hair_plate_max_hz", hair_plate_max_hz), ("campaniform_load_hz", campaniform_load_hz),
                            ("campaniform_max_hz", campaniform_max_hz), ("haltere_k", haltere_k),
                            ("haltere_max_hz", haltere_max_hz), ("coriolis_gain_per_rad_s", coriolis_gain_per_rad_s)):
            if not np.isfinite(value) or value < 0 or (name == "mn_ref_hz" and value == 0):
                raise ValueError(f"{name} must be finite and nonnegative (mn_ref_hz positive)")
        self.mn_ref_hz = float(mn_ref_hz)
        self.params = {
            "chordotonal": dict(tonic_hz=float(chordotonal_tonic_hz), max_hz=float(chordotonal_max_hz)),
            "hair_plate": dict(tonic_hz=float(hair_plate_tonic_hz), max_hz=float(hair_plate_max_hz)),
            "campaniform": dict(load_hz=float(campaniform_load_hz), max_hz=float(campaniform_max_hz)),
            "haltere": dict(k=float(haltere_k), max_hz=float(haltere_max_hz),
                            coriolis_gain_per_rad_s=float(coriolis_gain_per_rad_s) if self.haltere_coriolis else 0.0),
        }
        for ch in ("chordotonal", "hair_plate", "campaniform"):
            p = self.params[ch]
            if p.get("tonic_hz", p.get("load_hz")) > p["max_hz"]:
                raise ValueError(f"{ch}: the tonic / load rate exceeds the channel ceiling")
        n = c.neurons
        pro = (n["class"] == "mechanosensory_proprioceptive").to_numpy()
        sub = n.subclass.fillna("").to_numpy().astype(str)
        nerve = n.entryNerve.fillna("").to_numpy().astype(str) if "entryNerve" in n else np.full(c.n, "")
        masks = {
            "chordotonal": pro & np.isin(sub, ["chordotonal organ", "leg"]),
            "hair_plate": pro & (sub == "hair plate"),
            "campaniform": pro & (sub == "campaniform sensilla") & np.isin(nerve, self.LEG_NERVES),
            "haltere": pro & (sub == "haltere"),
        }
        self.idx = {ch: np.flatnonzero(masks[ch]) for ch in self.channels}
        self.side = {ch: self._sides(idx, side_threshold) for ch, idx in self.idx.items()}
        self.side_source = {ch: self._side_source(idx) for ch, idx in self.idx.items()}

    @classmethod
    def parse_spec(cls, spec):
        """'all' | 'all+haltere_coriolis' | 'chordotonal,haltere' | 'haltere_coriolis' -> (channels, coriolis flag)."""
        if isinstance(spec, (list, tuple)):
            spec = ",".join(spec)
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError("proprioception spec must be a non-empty string such as 'all' or 'chordotonal,haltere'")
        parts = [p.strip() for p in spec.replace("+", ",").split(",") if p.strip()]
        coriolis = "haltere_coriolis" in parts
        names = []
        for p in parts:
            if p == "all":
                names += list(cls.CHANNELS)
            elif p == "haltere_coriolis":
                names.append("haltere")
            elif p in cls.CHANNELS:
                names.append(p)
            else:
                raise ValueError(f"unknown proprioception channel {p!r}; choose from {cls.CHANNELS} (+ 'haltere_coriolis')")
        channels = tuple(ch for ch in cls.CHANNELS if ch in names)
        if not channels:
            raise ValueError("proprioception spec selects no channel")
        return channels, coriolis

    @property
    def spec(self):
        s = "all" if self.channels == self.CHANNELS else ",".join(self.channels)
        return s + "+haltere_coriolis" if self.haltere_coriolis else s

    def _instance_side(self, idx):
        if "instance" not in self.c.neurons:
            return np.zeros(len(idx), dtype=int)
        inst = self.c.neurons.instance.fillna("").to_numpy().astype(str)[idx]
        return np.where(np.char.endswith(inst, "_L"), 1, np.where(np.char.endswith(inst, "_R"), -1, 0))

    def _sides(self, idx, threshold):
        if not len(idx):
            return np.zeros(0, dtype=int)
        side = self._instance_side(idx)
        missing = side == 0
        if missing.any():
            ref = self.c.reference
            soma = ref.neurons.somaSide.to_numpy()
            tL, tR = np.flatnonzero(soma == "L"), np.flatnonzero(soma == "R")
            pre = ref.index_of(self.c.neurons.bodyId.to_numpy()[idx[missing]])
            lat = laterality(ref, pre, tL, tR)
            side[missing] = np.where(lat > threshold, 1, np.where(lat < -threshold, -1, 0))
        return side

    def _side_source(self, idx):
        inst = self._instance_side(idx)
        return {"instance": int((inst != 0).sum()), "laterality": int((inst == 0).sum())}

    def counts(self):
        """Per channel: cells, left / right / unsided, and the superclass split -- the numbers the audit reports."""
        sc = self.c.neurons.superclass.fillna("").to_numpy().astype(str)
        out = {}
        for ch, idx in self.idx.items():
            s = self.side[ch]
            out[ch] = dict(n=int(len(idx)), L=int((s > 0).sum()), R=int((s < 0).sum()), both=int((s == 0).sum()),
                           vnc_sensory=int((sc[idx] == "vnc_sensory").sum()),
                           sensory_ascending=int((sc[idx] == "sensory_ascending").sum()), side_source=self.side_source[ch])
        return out

    def _lateral(self, ch, left, right):
        side = self.side[ch][None]
        return np.where(side > 0, left[:, None], np.where(side < 0, right[:, None], .5 * (left + right)[:, None]))

    def rates(self, leg_L, leg_R, haltere, airborne, yaw_rate=0.0, batch=1):
        """Leg MN rates (Hz, per side), haltere MN rate (Hz), airborne flag and the realised yaw rate (rad/s; read only
        by the labelled stop-gap term) -> [(channel, cell indices, (batch, n) Hz)], each clamped to its ceiling."""
        lL = batch_values(leg_L, batch, "leg_L rate"); lR = batch_values(leg_R, batch, "leg_R rate")
        h = batch_values(haltere, batch, "haltere rate")
        air = batch_values(np.asarray(airborne, dtype=float), batch, "airborne") > 0.5
        yaw = batch_values(yaw_rate, batch, "yaw_rate")
        if np.any(lL < 0) or np.any(lR < 0) or np.any(h < 0):
            raise ValueError("motor rates must be nonnegative")
        ground = (~air).astype(float)
        out = []
        for ch in self.channels:
            idx = self.idx[ch]
            p = self.params[ch]
            if ch in ("chordotonal", "hair_plate"):
                drive = self._lateral(ch, np.clip(lL / self.mn_ref_hz, 0, 1) * ground, np.clip(lR / self.mn_ref_hz, 0, 1) * ground)
                hz = p["tonic_hz"] + (p["max_hz"] - p["tonic_hz"]) * drive
            elif ch == "campaniform":
                hz = np.repeat((p["load_hz"] * ground)[:, None], len(idx), axis=1)
            else:
                mod = 1.0 + p["coriolis_gain_per_rad_s"] * np.abs(yaw)
                hz = np.repeat((p["k"] * h * mod)[:, None], len(idx), axis=1)
            out.append((ch, idx, np.clip(hz, 0.0, p["max_hz"])))
        return out
