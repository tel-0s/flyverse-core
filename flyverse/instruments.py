"""Instruments: labelled, opt-in stand-ins for physiology the connectome names but the model cannot supply.

The contract is docs/PRESETS_SPEC.md; what ships and how to turn it on is docs/INSTRUMENTS.md. Nothing here is built
by default: ``FlyBrain(preset="raw")`` -- the default -- constructs none of these objects, and
``FlyBrain(preset="instrumented", instruments=[...])`` records every instrument's ``describe()`` in ``provenance()``
beside ``compiled_connectome`` (keys ``preset`` and ``instruments``), so a JSON always says which of its numbers came
from the connectome and which from a hand-written stand-in.

The registry includes:

* ``compass.CompassDriver`` (kind ``stop-gap``): an imposed angular memory, enabled by name ``compass``.
  It receives held yaw velocity through ``FlyBrain.proprioception`` and writes only EPG Poisson Hz through
  the ordinary attached-module scheduler. This explicit motion receiver is specific to named instruments;
  it does not give modules access to the body or supply a world-heading/goal oracle.

* ``SidedTurnAfferent`` (kind ``stop-gap``): the body's signed yaw rate -> Poisson spikes on PS196_b's named ascending
  afferents, on the side the connectome's contralateral routing implies. It is a *body-derived* signal, so it lives
  where the ``haltere_coriolis`` stop-gap lives -- the ``senses.Proprioception`` transducer, token ``'turn_afferent'``
  -- and reaches the cells through the channel that already exists (``FlyBrain.proprioception(..., yaw_rate=)``,
  ``FlyBrain._input``). Its runtime remains in the sense, separate from CompassDriver's imposed memory.
* ``EdgeHold`` (kind ``edges``): the record of a factor-0 hold installed through ``LIFParams.type_path_gain``
  (``cx_wedge.py --hold-edges``, round 6A). PRESETS_SPEC section 2 item 4: a held edge is an instrument too. The
  object does not install the hold -- the caller's ``LIFParams`` does -- but ``install()`` refuses a FlyBrain whose
  gain list does not carry it, so the record cannot lie.
* ``TypeRelabel`` (kind ``relabel``): the record of a ``TYPE_NT_OVERRIDE`` row compiled into a scratch cache
  (``cx_wedge.py --nt-override``, round 2 / 5B). ``install()`` refuses a connectome whose cells do not carry it.
"""
from __future__ import annotations

import numpy as np

PRESETS = ("raw", "instrumented")
INSTRUMENT_KINDS = {"stop-gap", "mechanism", "edges", "relabel"}
# What the stand-in stands in for (PRESETS_SPEC section 5, owner extension of 2026-09-17): 'input' supplies a missing
# *input* the connectome names, 'computation' replaces a *computation* the model cannot do (a program-shaped stand-in,
# admitted only under the section-5 extension), 'configuration' is a record of a caller configuration (edges/relabel).
REPLACES = {"input", "computation", "configuration"}
NAMED_INSTRUMENTS = ('compass', 'compass_ring', 'plume', 'hunger', 'flight')


def add_cli_arguments(parser):
    """Plural list syntax plus the existing repeatable singular alias, in command-line order."""
    import argparse
    parser.add_argument('--instruments', dest='instrument', nargs='+', action='extend', default=[],
                        metavar='NAME', help='explicit instruments: '+', '.join(NAMED_INSTRUMENTS))
    parser.add_argument('--instrument', dest='instrument', action='append', default=argparse.SUPPRESS,
                        metavar='NAME', help='compatibility alias; repeatable')


def validate_cli(parser, specs):
    """Fail on named conflicts before loading a connectome or allocating a brain."""
    names=[s.split(':',1)[0] for s in specs]
    if len(names)!=len(set(names)):
        parser.error('instrument names must be unique')
    unknown=set(names)-set(NAMED_INSTRUMENTS)-set(REGISTRY)
    if unknown:
        parser.error(f'unknown instruments: {sorted(unknown)}')
    if {'compass','compass_ring'}<=set(names):
        parser.error('compass and compass_ring are incompatible heading providers')
    if 'plume' in names and not {'compass','compass_ring'}.intersection(names):
        parser.error('plume requires compass or compass_ring')
    if 'hunger' in names and not {'plume','flight'}.intersection(names):
        parser.error('hunger requires plume or flight')


def validate_composition(instruments):
    """Validate dependencies and neural write conflicts before installing any instrument."""
    names=[_check_instrument(m) for m in instruments]
    if len(names)!=len(set(names)):
        raise ValueError(f'instrument names must be unique: {names}')
    present=set(names)
    claimed={}
    for m in instruments:
        conflict=present.intersection(getattr(m,'incompatible',()))
        if conflict:
            raise ValueError(f'instrument {m.name!r} is incompatible with {sorted(conflict)}')
        required=set(getattr(m,'requires_any',()))
        if required and not required.intersection(present):
            raise ValueError(f'instrument {m.name!r} requires one of {sorted(required)}')
        for idx in getattr(m,'writes',{}).values():
            if not isinstance(idx,np.ndarray):continue  # general selector conflicts are resolved by attach()
            for i in idx:
                key=(getattr(m,'channel_out',None),int(i))
                if key in claimed:
                    raise ValueError(f'instrument {m.name!r} overlaps {claimed[key]!r} on {key[0]}')
                claimed[key]=m.name
    return names


def make_instrument(c, name):
    """Resolve an explicit named instrument after the connectome/subset has been selected."""
    if name == 'compass':
        from .compass import CompassDriver
        return CompassDriver(c)
    if name in NAMED_INSTRUMENTS:
        from .navigation import RecurrentCompass, PlumeNavigation, HungerGain, FlightDrive
        return {'compass_ring':RecurrentCompass,'plume':PlumeNavigation,'hunger':HungerGain,'flight':FlightDrive}[name](c)
    if isinstance(name,str) and name.split(':',1)[0] in REGISTRY:
        return parse_instrument(name,c)
    raise ValueError(f'unknown instrument {name!r}; named instruments: {", ".join(NAMED_INSTRUMENTS)}')


def identifier(obj):
    t = obj if isinstance(obj, type) else type(obj)
    return f"{t.__module__}:{t.__qualname__}"


def _check_instrument(inst):
    """The minimum an object must carry to be listed as an instrument (PRESETS_SPEC section 2 items 1, 2, 6)."""
    name = getattr(inst, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"an instrument needs a nonempty string `name`: {inst!r}")
    if getattr(inst, "kind", None) not in INSTRUMENT_KINDS:
        raise ValueError(f"instrument {name!r}: `kind` must be one of {sorted(INSTRUMENT_KINDS)}")
    if not callable(getattr(inst, "describe", None)):
        raise ValueError(f"instrument {name!r} must implement describe()")
    if not callable(getattr(inst, "install", None)):
        raise ValueError(f"instrument {name!r} must implement install()")
    d = inst.describe()
    for key in ("name", "kind", "law", "gap", "removal", "audits", "replaces"):
        if key not in d:
            raise ValueError(f"instrument {name!r}: describe() lacks {key!r} (PRESETS_SPEC section 2)")
    if d["name"] != name or d["kind"] != inst.kind:
        raise ValueError(f"instrument {name!r}: describe() disagrees with its name or kind")
    if any(not d[key] for key in ("law", "gap", "removal", "audits")):
        raise ValueError(f"instrument {name!r}: law, gap, removal and audits must be nonempty")
    if d["replaces"] not in REPLACES:
        raise ValueError(f"instrument {name!r}: describe()['replaces'] must be one of {sorted(REPLACES)} "
                         "(PRESETS_SPEC section 5: a program-shaped stand-in says so)")
    # PRESETS_SPEC section 2 item 2: either the law is marked `unverified`, or the record names where it came from.
    if d["law"] != "unverified" and not (d.get("source") or d.get("sources")):
        raise ValueError(f"instrument {name!r}: law {d['law']!r} is not 'unverified', so describe() must carry a "
                         "nonempty `source` or `sources` (PRESETS_SPEC section 2 item 2)")
    return name


def _sides(c, idx, what):
    """+1 left / -1 right from `somaSide`; every cell of a sided stand-in must carry one."""
    soma = c.neurons.somaSide.fillna("").to_numpy().astype(str)[idx] if "somaSide" in c.neurons else np.full(len(idx), "")
    side = np.where(soma == "L", 1, np.where(soma == "R", -1, 0))
    if (side == 0).any():
        bad = c.neurons.bodyId.to_numpy()[idx[side == 0]].tolist()
        raise ValueError(f"{what}: {len(bad)} cell(s) carry no somaSide and cannot be sided: {bad[:8]}")
    return side


def _type_indices(c, type_name):
    idx = np.flatnonzero(c.neurons.type.fillna("").to_numpy().astype(str) == type_name)
    return idx


def _counts_matrix(c):
    """Raw synapse counts (post x pre) where the cache carries them (sign-0 edges included), else |W|."""
    import warnings
    try:
        from .interp.common import raw_counts
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # a synthetic graph beside a compiled cache: the sign-0 file is ignored, |W| is the count
            C, _ = raw_counts(c, build=False)
        return C.tocsr()
    except Exception:  # noqa: BLE001 -- a synthetic graph with no cache: |W| is the count
        return abs(c.W).tocsr()


def side_block(c, pre_type, post_type, counts=None):
    """{'LL', 'LR', 'RL', 'RR': {'synapses', 'pairs'}} for pre_type -> post_type by soma side, plus the two type
    counts and whether the routing is contralateral (cross-side synapses > same-side). None when a type is absent."""
    pre, post = _type_indices(c, pre_type), _type_indices(c, post_type)
    if not len(pre) or not len(post):
        return None
    counts = _counts_matrix(c) if counts is None else counts
    soma = c.neurons.somaSide.fillna("").to_numpy().astype(str)
    out = {"pre": pre_type, "post": post_type, "n_pre": int(len(pre)), "n_post": int(len(post))}
    for ps in ("L", "R"):
        for qs in ("L", "R"):
            a, b = pre[soma[pre] == ps], post[soma[post] == qs]
            if len(a) and len(b):
                sub = counts[b][:, a]
                out[ps + qs] = {"synapses": float(sub.sum()), "pairs": int(sub.nnz)}
            else:
                out[ps + qs] = {"synapses": 0.0, "pairs": 0}
    cross = out["LR"]["synapses"] + out["RL"]["synapses"]
    same = out["LL"]["synapses"] + out["RR"]["synapses"]
    out["contralateral"] = bool(cross > same)
    out["cross_synapses"], out["same_synapses"] = float(cross), float(same)
    return out


class SidedTurnAfferent:
    """PS196_b's signed turn input, as a Poisson drive on its named ascending afferents. LABELLED STOP-GAP, off unless
    named; the law is UNVERIFIED.

    THE GAP (PRESETS_SPEC section 2 item 1). PS196_b (2 cells, L / R) is GLNO's largest non-ring input (1,801 syn,
    19-21 % of GLNO's input: Wang 2026 finding 3, reproduced in our cache to the synapse) and GLNO is PEN's one nodulus
    input of size (19.4 % of PEN's raw input). In the shipped body the report that reaches PS196_b is UNSIGNED: the only
    afferent class two steps from it is the haltere SApp, whose motor rate is one bilateral number, so under the
    labelled Coriolis stop-gap PS196_b's L-R moves the same way in both turn directions (+3.52 ccw, +3.95 cw;
    docs/audits/vnc_drive.md section 6, docs/NOTES.md compass round 2). Its named ascending inputs -- AN07B037_a / _b
    (419 / 52 syn), CB0675, GNG580, PS047_b -- exist in the connectome and are driven by nothing in the model. This
    object drives them.

    THE LAW (item 2): rate = k * max(0, sign * yaw_deg_s) on the LEFT afferents and k * max(0, -sign * yaw_deg_s) on
    the RIGHT ones (positive yaw is a left turn in body.Locomotion: body.py, the leg-cycle note), clamped to `max_hz`.
    `k` is a declared parameter in Hz per deg/s with the predeclared levels K_LEVELS = (0.25, 0.5, 1.0)
    (docs/audits/compass_velocity_route.md section 2; 0.5 is the primary). It is **unverified**: no recording of
    PS196_b, AN07B037 or their turn tuning exists -- searched 2026-09-15 in Wang's fly-circuit-exploration audit,
    Hulse et al. 2021 (eLife 66039), and the two Rockefeller theses named in docs/NOTES.md (Janke 2025, Avritzer
    2026); PRESETS_SPEC section 3 names the search. `sign` = -1 is the HGV- control arm (the sign flip).

    THE SIDE (from the graph, never assumed): the routing is read from W[post, pre] with somaSide at construction
    and recorded in describe()['routing']. On the shipped cache every hop is contralateral -- AN07B037_a L -> PS196_b R
    202 syn / R -> L 214 vs 0 / 3 ipsilateral; AN07B037_b 28 / 24 vs 0 / 0; PS196_b L -> GLNO R 832 / R -> L 966 vs
    0 / 3; GLNO -> PEN cross-side only -- so with sign +1 a LEFT turn drives the LEFT afferents and the chain lands on
    PS196_b_R -> GLNO_L -> PEN_R (describe()['chain_for_positive_yaw']). Which PEN side *should* carry a left turn is
    exactly what no recording settles, hence the sign arm. The `cells` variants CB0675 / GNG580 / PS047_b are
    IPSILATERAL onto PS196_b (51 / 51, 10 / 13, 185 / 217 syn same-side), so their chain lands one side over; the
    record says so per variant.

    THE BOUNDARY (item 3): reads the body's yaw rate through the channel that already exists
    (``FlyBrain.proprioception(..., yaw_rate=)``, the one the Coriolis stop-gap reads) and writes ``poisson_hz`` on
    the afferent cells through ``FlyBrain._input``. It never writes the body, a weight, a receptor or NT_SIGN.

    REMOVAL (item 6): retired by any of (a) a recording of PS196_b or AN07B037 during turning, which replaces `k` and
    `sign` with a measured tuning (then this becomes a `mechanism` with a source, or is dropped if the tuning is
    flat); (b) a sided haltere / leg readout in the body whose ascending report reaches PS196_b signed on its own
    (the round-3 sided tokens did not: docs/audits/body_sided_state.md); (c) a round-7 `null` on measure 3 at every
    declared level, which says the route is not these afferents.
    """
    name = "sided_turn_afferent"
    kind = "stop-gap"
    law = "unverified"
    K_LEVELS = (0.25, 0.5, 1.0)
    CELL_VARIANTS = {"AN07B037": ("AN07B037_a", "AN07B037_b"), "CB0675": ("CB0675",), "GNG580": ("GNG580",),
                     "PS047_b": ("PS047_b",), "all": ("AN07B037_a", "AN07B037_b", "CB0675", "GNG580", "PS047_b")}
    CHAIN = ("PS196_b", "GLNO", ("PEN_a(PEN1)", "PEN_b(PEN2)"))
    AUDITS = ("docs/audits/compass_velocity_route.md", "docs/audits/vnc_drive.md section 6",
              "docs/NOTES.md compass round 2 (PS196_b unsigned)", "docs/PRESETS_SPEC.md section 3")

    def __init__(self, c, *, k_hz_per_deg_s=0.5, sign=1, cells="AN07B037", max_hz=250.0):
        k = float(k_hz_per_deg_s)
        if not np.isfinite(k) or k <= 0:
            raise ValueError("k_hz_per_deg_s must be finite and positive")
        if sign not in (1, -1, 1.0, -1.0):
            raise ValueError("sign must be +1 or -1")
        if not np.isfinite(max_hz) or max_hz <= 0:
            raise ValueError("max_hz must be finite and positive")
        if cells not in self.CELL_VARIANTS:
            raise ValueError(f"cells must be one of {sorted(self.CELL_VARIANTS)}, got {cells!r}")
        self.c = c
        self.k, self.sign, self.cells, self.max_hz = k, int(sign), cells, float(max_hz)
        self.types = tuple(self.CELL_VARIANTS[cells])
        ty = c.neurons.type.fillna("").to_numpy().astype(str)
        missing = [t for t in self.types if not (ty == t).any()]
        if missing:
            raise ValueError(f"{self.name}: the connectome has no cells of type {missing}; the variant {cells!r} cannot be built")
        self.idx = np.flatnonzero(np.isin(ty, self.types))
        self.side = _sides(c, self.idx, self.name)
        self.type_of = ty[self.idx]
        self.routing = self._routing()
        self.chain_for_positive_yaw = self._chain()
        self.parameters = {"k_hz_per_deg_s": self.k, "sign": self.sign, "cells": self.cells, "types": list(self.types),
                           "max_hz": self.max_hz, "k_levels": list(self.K_LEVELS),
                           "k_is_declared_level": any(abs(self.k - lv) < 1e-12 for lv in self.K_LEVELS)}

    # ------------------------------------------------------------------------------------------------ the graph
    def _routing(self):
        counts = _counts_matrix(self.c)
        ps, glno, pens = self.CHAIN
        out = {"afferent_to_PS196_b": {t: side_block(self.c, t, ps, counts) for t in self.types},
               "PS196_b_to_GLNO": side_block(self.c, ps, glno, counts),
               "GLNO_to_PEN": {p: side_block(self.c, glno, p, counts) for p in pens}}
        return out

    def _chain(self):
        """Follow the majority side of each hop from a LEFT afferent (what sign +1 drives on a left turn)."""
        def step(block, side):
            if block is None:
                return None
            same, cross = block[side + side]["synapses"], block[side + ("R" if side == "L" else "L")]["synapses"]
            return side if same >= cross else ("R" if side == "L" else "L")
        chain = []
        for t in self.types:
            side = "L"
            path = [f"{t}_L"]
            side = step(self.routing["afferent_to_PS196_b"][t], side)
            path.append(f"PS196_b_{side}" if side else "PS196_b_?")
            side = step(self.routing["PS196_b_to_GLNO"], side) if side else None
            path.append(f"GLNO_{side}" if side else "GLNO_?")
            pen_sides = {p: step(b, side) if side else None for p, b in self.routing["GLNO_to_PEN"].items()}
            sides = {s for s in pen_sides.values() if s}
            path.append("PEN_" + ("/".join(sorted(sides)) if sides else "?"))
            chain.append(" -> ".join(path))
        return chain

    # ------------------------------------------------------------------------------------------------ the law
    def rates(self, yaw_rate, batch=1):
        """Realised yaw rate (rad/s; scalar or (batch,)) -> (batch, n_cells) Hz. Zero yaw is zero everywhere."""
        yaw = np.asarray(yaw_rate, dtype=float)
        if yaw.ndim == 0:
            yaw = np.full(batch, float(yaw))
        if yaw.shape != (batch,) or not np.isfinite(yaw).all():
            raise ValueError(f"yaw_rate must be finite and scalar or shape ({batch},)")
        toward = self.sign * np.rad2deg(yaw)
        left = self.k * np.maximum(0.0, toward) + 0.0          # + 0.0: no -0.0 at zero yaw
        right = self.k * np.maximum(0.0, -toward) + 0.0
        hz = np.where(self.side[None] > 0, left[:, None], right[:, None])
        return np.clip(hz, 0.0, self.max_hz)

    # ------------------------------------------------------------------------------------------------ the record
    def describe(self):
        ids = self.c.neurons.bodyId.to_numpy()
        return {
            "name": self.name, "class": identifier(self), "kind": self.kind, "law": self.law,
            "replaces": "input",                        # a missing body-derived input, not a computation
            "law_text": "poisson_hz = k * max(0, sign * yaw_deg_s) on the left afferents, k * max(0, -sign * yaw_deg_s) on the right; "
                        "clamped to max_hz; positive yaw = a left turn (body.Locomotion)",
            "parameters": dict(self.parameters), "trainable": False, "checkpoint_hash": None,
            "n_cells": int(len(self.idx)),
            "cells": {"L": ids[self.idx[self.side > 0]].tolist(), "R": ids[self.idx[self.side < 0]].tolist()},
            "cells_by_type": {t: int((self.type_of == t).sum()) for t in self.types},
            "routing": self.routing, "chain_for_positive_yaw": self.chain_for_positive_yaw,
            "reads": "body yaw_rate (rad/s) through FlyBrain.proprioception, the channel the haltere_coriolis stop-gap reads",
            "writes": "poisson_hz on the afferent cells (FlyBrain._input 'proprioception_turn_afferent')",
            "gap": "PS196_b receives no signed turn input in the shipped body: the ascending report arrives unsigned "
                   "(vnc_drive.md 6, NOTES compass round 2); Wang 2026 finding 3 names PS196_b as GLNO's largest non-ring input",
            "source": "no PS196_b / AN07B037 recording exists (searched 2026-09-15: Wang's fly-circuit-exploration audit, "
                      "Hulse 2021, the two Rockefeller theses); k and sign are declared levels, unverified",
            "removal": "a recording of PS196_b / AN07B037 during turning; a sided ascending report that reaches PS196_b on its own; "
                       "or a round-7 null on measure 3 at every declared k",
            "audits": list(self.AUDITS),
        }

    def install(self, fb):
        """Attach through the existing proprioception pattern: the sense grows the 'turn_afferent' channel."""
        from .senses import Proprioception
        sense = getattr(fb, "proprioception_sense", None)
        if sense is None:
            fb.proprioception_sense = Proprioception(fb.c, "turn_afferent", turn_afferent=self)
        elif getattr(sense, "turn_afferent", None) is not self:
            sense.add_turn_afferent(self)


class EdgeHold:
    """The record of an `edges`-kind hold: one presynaptic class onto one postsynaptic class at `factor` (0 = held),
    installed by the caller through ``LIFParams.type_path_gain`` (``cx_wedge.parse_hold_edges``; round 6A,
    docs/audits/compass_dc_balance.md). A LABELLED COUNTERFACTUAL, never a default; PRESETS_SPEC section 2 item 4.

    The default name `ring_dc_hold` is the 6A hold `^(ExR6|ER6|ER4m)$:^(PEN_|EPG$)` -- the ExR6 / ER6 / ER4m DC term
    on PEN / EPG that removes the ring's resting state. `resolved` is the caller's count record
    (``cx_wedge.hold_edge_counts``: cells, entries, synapses silenced). Removal: receptor rows at the EB / GA contacts
    of the three types. ExR6 glutamate and ER6 GABA now have type-linked evidence (exr6_evidence.md);
    receptor placement and kinetics at their EB / GA contacts remain open."""
    kind = "edges"
    law = "counterfactual"

    def __init__(self, pre_re, post_re, factor=0.0, *, name="ring_dc_hold", resolved=None, description=None):
        self.name = name
        self.pre_re, self.post_re, self.factor = str(pre_re), str(post_re), float(factor)
        self.resolved = resolved
        self.description = dict(description or {})
        if set(self.description) - {"gap", "source", "removal", "audits"}:
            raise ValueError("hold description may only specialize gap, source, removal and audits")

    def describe(self):
        return {"name": self.name, "class": identifier(self), "kind": self.kind, "law": self.law,
                "replaces": "configuration",            # a record of a caller configuration: neither an input nor a computation
                "law_text": f"W[post ~ {self.post_re}, pre ~ {self.pre_re}] x {self.factor:g} through LIFParams.type_path_gain",
                "parameters": {"pre": self.pre_re, "post": self.post_re, "factor": self.factor},
                "resolved": self.resolved, "trainable": False, "checkpoint_hash": None,
                "gap": "the ExR6 / ER6 / ER4m DC term on PEN / EPG removes the ring's resting state (compass_dc_balance.md); "
                       "receptor placement and kinetics at those contacts are open",
                "source": "a factor-0 hold claims no transfer; the DC term it removes is the 5A fixed point, recorded in "
                          "docs/audits/compass_dc_balance.md and docs/audits/compass_ring_mechanism.md",
                "removal": "sourced receptor placement and kinetics at the EB / GA contacts of ExR6 / ER6 / ER4m that reproduce the physiological operating state",
                "audits": ["docs/audits/compass_dc_balance.md", "docs/audits/compass_ring_mechanism.md", "docs/audits/exr6_evidence.md"],
                **self.description}

    def install(self, fb):
        gains = [(str(p), str(q), float(f)) for p, q, f in (getattr(fb.brain.p, "type_path_gain", None) or [])]
        if (self.pre_re, self.post_re, self.factor) not in gains:
            raise ValueError(f"instrument {self.name!r}: the FlyBrain's type_path_gain does not carry "
                             f"({self.pre_re!r}, {self.post_re!r}, {self.factor:g}); the hold is a record of a gain the caller installs")


class TypeRelabel:
    """The record of a transmitter relabel compiled into a scratch cache (``connectome.TYPE_NT_OVERRIDE`` extended by
    ``cx_wedge.py --nt-override TYPE=nt``). `glno_sign` (GLNO = glutamate) is round 7's component: GLNO's sign is 0 in
    the shipped cache because two EM predictions disagree (docs/audits/glno_relabel.md); glutamate is the stronger of
    the two and is safe on the suite but fixes nothing alone (5B). UNVERIFIED as a transmitter call. Removal: a
    transmitter source for GLNO that is not one EM classifier, at which point the row enters TYPE_NT_OVERRIDE itself."""
    kind = "relabel"
    law = "unverified"

    def __init__(self, type_name, nt, *, name=None):
        self.type_name, self.nt = str(type_name), str(nt)
        self.name = name or ("glno_sign" if self.type_name == "GLNO" else f"relabel_{self.type_name}")

    def describe(self):
        return {"name": self.name, "class": identifier(self), "kind": self.kind, "law": self.law,
                "replaces": "configuration",            # a record of a caller configuration: neither an input nor a computation
                "law_text": f"type {self.type_name} compiled as {self.nt} (NT_SIGN sign) in a scratch cache",
                "parameters": {"type": self.type_name, "nt": self.nt}, "trainable": False, "checkpoint_hash": None,
                "gap": "GLNO's transmitter is unknown (MaleCNS `unclear`; two EM predictions disagree and both fall below 0.5)",
                "source": "the stronger of two disagreeing EM predictions, recorded in docs/audits/glno_relabel.md and "
                          "docs/audits/cx_glno.md; no sourced transmitter call exists",
                "removal": "a transmitter source for the type that is not one EM classifier (then a TYPE_NT_OVERRIDE row)",
                "audits": ["docs/audits/glno_relabel.md", "docs/audits/cx_glno.md"]}

    def install(self, fb):
        n = fb.c.neurons
        idx = np.flatnonzero(n.type.fillna("").to_numpy().astype(str) == self.type_name)
        if not len(idx):
            raise ValueError(f"instrument {self.name!r}: no cells of type {self.type_name!r} in this connectome")
        nts = set(n.nt.to_numpy()[idx].tolist())
        if nts != {self.nt}:
            raise ValueError(f"instrument {self.name!r}: the connectome carries nt {sorted(nts)} on {self.type_name}, not {self.nt!r}; "
                             "compile the relabel (cx_wedge.py --nt-override) before recording it")


class EdgeGain(EdgeHold):
    """A per-type gain through the same stage (``cx_wedge.py --edge-gain``, round 6B): a LABELLED INSTRUMENT, never a
    default. Same record as EdgeHold with a non-zero factor and its own name."""
    law = "instrument (a per-type gain, no transfer claimed)"

    def __init__(self, pre_re, post_re, factor, *, name="edge_gain", resolved=None):
        super().__init__(pre_re, post_re, factor, name=name, resolved=resolved)

    def describe(self):
        d = super().describe()
        d.update(gap="a per-type undamping the 5A / 6A rounds asked for (compass_local_recurrence.md)",
                 source="no measurement: a gain factor is not one. The round that asked for it and the arms it was run "
                        "in are docs/audits/compass_local_recurrence.md",
                 removal="a measured recurrence gain, or the round that retires the question",
                 audits=["docs/audits/compass_local_recurrence.md"])
        return d


REGISTRY = {"sided_turn_afferent": SidedTurnAfferent}
SPEC_HELP = ("NAME[:key=value]... e.g. sided_turn_afferent:k=0.5:sign=-1:cells=CB0675  "
             "(keys for sided_turn_afferent: k = Hz per deg/s in {0.25, 0.5, 1.0}, sign = +1 | -1, "
             "cells = AN07B037 | CB0675 | GNG580 | PS047_b | all, max_hz)")


def parse_instrument(spec, c):
    """'NAME[:key=value]...' -> an instrument instance on connectome `c` (the cx_wedge --instrument grammar)."""
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError(f"instrument spec must be a nonempty string: {SPEC_HELP}")
    if spec in NAMED_INSTRUMENTS:
        return make_instrument(c,spec)
    parts = [p.strip() for p in spec.split(":")]
    name, kv = parts[0], parts[1:]
    if name not in REGISTRY:
        raise ValueError(f"unknown instrument {name!r}; choose from {sorted(REGISTRY)}")
    kwargs = {}
    seen = set()
    for item in kv:
        if "=" not in item:
            raise ValueError(f"instrument option {item!r} is not key=value: {SPEC_HELP}")
        key, value = item.split("=", 1)
        key = key.strip()
        if key in seen:
            raise ValueError(f"duplicate instrument option {key!r}")
        seen.add(key)
        if name == "sided_turn_afferent":
            if key == "k":
                kwargs["k_hz_per_deg_s"] = float(value)
            elif key == "sign":
                kwargs["sign"] = float(value)
            elif key == "cells":
                kwargs["cells"] = value.strip()
            elif key == "max_hz":
                kwargs["max_hz"] = float(value)
            else:
                raise ValueError(f"unknown option {key!r} for {name}: {SPEC_HELP}")
    return REGISTRY[name](c, **kwargs)


def records(instruments):
    """[describe() of each], JSON-ready, in attachment order -- what provenance()['instruments'] carries."""
    from .interp.common import to_jsonable
    return [to_jsonable(inst.describe()) for inst in instruments]
