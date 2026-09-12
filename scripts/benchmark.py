"""Behaviour benchmark suite: score one parameter set of the model (LIF rules in brain.py, the optic model in
optic.py, the readouts in motor.py) on every measured behaviour at once, each section against a reference
value from docs/NOTES.md, and write the numbers as JSON for regression comparisons.

    python scripts/benchmark.py                                  # everything (~4 min on the 4090, ~5 on a shared B200; native backend)
    python scripts/benchmark.py --sections a,b,f --json out/bench.json
    python scripts/benchmark.py --fast                           # shorter recordings, one seed (~half the time)
    python scripts/benchmark.py --eager                          # demo sections on the torch path instead of the CUDA kernels
    python scripts/benchmark.py --std-u 0 --same-type-gain 0.1   # LIF / optic overrides apply to EVERY section

Sections (letters or names in --sections; "legacy" = the five original ones, "all" = everything):
  legacy (Brain + OpticLobe, eager torch; the session 3-4 calibration protocol)
    rest       no input, 500 ms                              -> spikes/step
    taste      labellar sweet GRNs 100 Hz, 600 ms            -> MN9, GNG175
    smell      ORNs 1 Hz base + apple odour                  -> PN, KC, LN
    dn         DNa02_L / DNp09 / MDN at 150 Hz               -> leg MN L/R, wing power, top clique
    walk       1.5 s walking (smell on) -> loom from the left -> 0.8 s yaw each way
                                                            -> GF walking / loom peak / escape range / DN flips (onset)
  a motion       T4/T5 direction selectivity: DSI and preferred direction per subtype (scripts/probe_motion.py)
  b loom_escape  the demo Sim (scripts/room_demo.py): 5 s walking, then the L-key loom; GF burst peak and whether
                 body.Flight (gf_hz 33) escapes -- two seeds
  c walk_gf      15 s of walking in the demo Sim, escapes disabled: per-second maxima of the GF, 99th percentile
  d rotation     sustained +-90 deg/s imposed yaw, pinned, no wind (scripts/screen_rotation.py protocol):
                 L - R flip of the optomotor group DNp20 + HSN + HSE
  e object       apple 5 cm ahead-left vs ahead-right on the single-apple table, heading oscillating +-20 deg at
                 0.5 Hz (scripts/screen_object.py + NOTES 'LC10'): LC10a L - R flip -- a known gap
  f bitter       sugar -> MN9 and sugar + bitter -> MN9 under the calibrated rules and under Shiu's rules
                 (scripts/probe_bitter.py)
  g wind         wind on the fly's left vs right, pinned (scripts/screen_steering.py sites): DNp18 / DNp33 L - R flips
  h odour        the lateral-horn apple channel (motor.LH_ODOUR_CHANNELS['apple']) 8 cm downwind of the apple vs a
                 plume-free spot (scripts/screen_odour.py --fruit apple sites)
  i compass      a 12-cell EPG wedge driven at 60 Hz for 2 s: cells still firing 0.5 s later -- a known gap

Every section builds what it needs and frees it. The demo sections (b-e, g, h) run scripts/room_demo.py's Sim on
the native backend (cuda_kernels + cuda_graphs + event_driven + warp CSR) unless --eager. Runs are chaotic: repeat
before trusting a 20% change in a magnitude; the pass/fail bounds are set well inside the run-to-run scatter.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from flyverse import body, brain, connectome, motor, olfaction, optic, retina, screen, world  # noqa: E402


# ------------------------------------------------------------------------------------------------ references
@dataclass
class Ref:
    value: object          # the number docs/NOTES.md established (the reference the table prints)
    op: str                # pass criterion: measured <op> bound  (">", "<", ">=", "<=", "==", "abs>=", "notnone")
    bound: float
    session: str           # NOTES session(s) that established the reference
    gap: bool = False      # a documented gap: missing the bound is 'KNOWN GAP' (not FAIL); meeting it is 'PASS (gap closed)'
    note: str = ""


# Reference values. The comment / `session` field says which docs/NOTES.md session established each number.
REFERENCES = {
    # --- legacy sections: session 3 (conn_cap 60, adaptation 1.5 mV, same-type 0.1) and session 4 (path gains)
    "rest.spikes_per_step": Ref(0, "<", 5, "3", note="no input -> the CNS is silent"),
    "taste.MN9_hz": Ref(3.7, ">", 2, "3-4", note="labellar sweet GRNs 100 Hz -> proboscis MN9 (3.7-5.8 Hz across sessions)"),
    "smell.PN_hz": Ref(13, "<", 100, "3", note="apple odour: PN mean 13 Hz (max 108); the antennal lobe must not run hot"),
    "smell.KC_active": Ref(1249, ">", 0, "3", note="Kenyon cells above 1 Hz: sparse but not silent"),
    "dn.DNa02_L_leg_asym_hz": Ref(3.0, ">", 0.3, "4", note="DNa02_L at 150 Hz: left minus right leg-MN mean (3.1 vs 0.1 Hz)"),
    "dn.MDN_top_hz": Ref(152, "<", 250, "4", note="MDN drives its backward-walking set (IN06B020 ~150 Hz), no 300 Hz storm"),
    "dn.DNp09_top_hz": Ref(45, "<", 250, "4", note="DNp09 drives its premotor set (~45 Hz), no AVLP storm"),
    "walk.GF_max_hz": Ref(26, "<", 38, "4-8", note="walking optic flow (smell on) keeps the GF below Flight.gf_hz"),
    "walk.power_max_hz": Ref(22, "<", 50, "4", note="per-frame max of the wing-power MN mean (22 Hz with DN->VNC x3, 50 at x6)"),
    "walk.power_sustained_hz": Ref(22, "<", 50, "4", note="max 0.3 s running mean: a voluntary takeoff needs Flight.takeoff_power_hz (50) sustained for 0.3 s"),
    "loom.GF_peak_hz": Ref(26, ">=", 20, "4", note="black ball at 1 m/s from the left -> GF burst (20-76 Hz across sessions)"),
    "loom.escape_cm": Ref(3.5, "notnone", 0, "2-4", note="range at which the GF crosses 20 Hz (3.5 cm)"),
    "rotate.DNp20_flip_hz": Ref(-14, "<", -2, "3", note="0.8 s yaw onset: DNp20 (L-R)_left - (L-R)_right (rightward: 15/1 Hz)"),
    # --- a: session 3 (DS achieved: DSI 0.16-0.26, correct directions), session 8 (T5a 0.38, T4c/d 0.23/0.24)
    "motion.min_dsi": Ref(0.16, ">=", 0.1, "3, 8", note="minimum DSI over T4a-d / T5a-d, 60 deg/s 30 deg grating"),
    "motion.correct_directions": Ref(8, "==", 8, "3", note="a front-to-back, b back-to-front, c up, d down, both T4 and T5"),
    # --- b: session 9 (LPi x4, edge_len 20 mm, gf_hz 33: loom peaks 34-56 Hz, 5/6 looms escape; the check is bistable at the threshold, use --seeds 0..5)
    "loom_escape.GF_peak_hz": Ref(41, ">=", 33, "9", note="best seed's GF peak in the 1.5 s after the demo loom"),
    "loom_escape.escapes": Ref(1, ">=", 1, "8", note="seeds in which body.Flight escapes within 1.5 s of the loom"),
    # --- c: session 8 (per-second walking GF maxima: median 20, 90th pct 28, 99th 32; one outlier 41.5)
    "walk_gf.p99_hz": Ref(32, "<", 38, "8", note="99th percentile of per-second GF maxima over 15 s of walking"),
    # --- d: session 8 (screen_rotation.py: HSN -1.4 ccw / +5.7 cw, DNp20 -4.5 / +7.8, HSE d' 2.2)
    "rotation.group_flip_hz": Ref(-8, "<=", -3, "8", note="(L-R)_ccw - (L-R)_cw of DNp20 + HSN + HSE (L - R grows under clockwise yaw)"),
    # --- e: session 8 (screen_object.py: no object-position signal; LC10a 0.02 Hz with or without the apple)
    "object.LC10a_flip_hz": Ref(0.0, "abs>=", 1.0, "8", gap=True, note="LC10a (L-R)_apple-left - (L-R)_apple-right; known gap: the rate optic lobe carries no small-object signal"),
    # --- f: session 8 (Shiu's sugar / bitter result on MaleCNS: 123.5 -> 2.1 Hz under Shiu's rules, 4.6 -> 0.0 calibrated)
    "bitter.calibrated_sugar_MN9_hz": Ref(4.6, ">", 2, "8", note="sweet GRNs 100 Hz, this project's rules"),
    "bitter.calibrated_sugar_bitter_MN9_hz": Ref(0.0, "<", 1, "8", note="sweet + bitter GRNs 100 Hz: bitter shuts MN9 off"),
    "bitter.shiu_sugar_MN9_hz": Ref(123.5, ">", 50, "8", note="Shiu et al. rules (uniform 0.275 mV, no cap / adaptation / damping / fan-in)"),
    "bitter.shiu_sugar_bitter_MN9_hz": Ref(2.1, "<", 10, "8", note="Shiu reported 78 -> 3 Hz on FlyWire"),
    # --- g: session 8 (screen_steering.py clean site: DNp18 +45 Hz, DNp33 -49 Hz, identical with and without odour)
    "wind.DNp18_flip_hz": Ref(45, ">=", 15, "8", note="DNp18 (L-R)_wind-left - (L-R)_wind-right: fires on the wind side"),
    "wind.DNp33_flip_hz": Ref(-49, "<=", -15, "8", note="DNp33: fires on the side away from the wind"),
    # --- h: session 8 (screen_odour.py --fruit apple: LHPD4d1 20.6 Hz at 8 cm, 12.5 at 40 cm, 3.4 plume-free)
    "odour.apple_channel_8cm_hz": Ref(20.6, ">=", 10, "8", note="LH apple channel mean, fly pinned 8 cm downwind of the apple"),
    "odour.apple_channel_clean_hz": Ref(3.4, "<=", 6, "8", note="same channel at the plume-free spot"),
    # --- i: session 8 (compass grid: a driven wedge dies within 0.5 s under every gain / adaptation setting tried)
    "compass.wedge_cells_persisting": Ref(0, ">=", 6, "8", gap=True, note="wedge cells above 5 Hz 0.5 s after a 2 s 60 Hz pulse; known gap: no attractor"),
}


def evaluate(ref: Ref, value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "MISSING"
    op, b = ref.op, ref.bound
    ok = {">": lambda v: v > b, "<": lambda v: v < b, ">=": lambda v: v >= b, "<=": lambda v: v <= b,
          "==": lambda v: v == b, "abs>=": lambda v: abs(v) >= b, "notnone": lambda v: v is not None}[op](value)
    if ok:
        return "PASS (gap closed)" if ref.gap else "PASS"
    return "KNOWN GAP" if ref.gap else "FAIL"


# ------------------------------------------------------------------------------------------------ context
class Context:
    """Shared connectome, parameter overrides, backend flags, the check list and the timing of each section."""

    def __init__(self, args):
        self.args = args
        self.fast = args.fast
        self.native = not args.eager
        self.seeds = [int(s) for s in args.seeds.split(",")]
        if self.fast:
            self.seeds = self.seeds[:1]
        self.cache_dir = args.cache_dir or os.environ.get("FLYVERSE_CACHE") or None   # a scratch cache (e.g. TYPE_NT_OVERRIDE trials)
        self.c = connectome.load(verbose=False) if self.cache_dir is None else connectome.load(cache_dir=self.cache_dir, verbose=False)
        connectome.load = lambda *a, **k: self.c        # every FlyBrain / demo Sim built here shares this one graph
        self.receptor_table, self.dopamine_lead_info = None, None
        if args.dopamine_lead not in (None, "all") and self.receptor_model is not None:
            self.receptor_table, self.dopamine_lead_info = self._build_dopamine_lead_table(args.dopamine_lead)
        self.checks = []
        self.results = {}
        self.runtime = {}

    # ---- parameters (CLI overrides reach every section, the demo Sims through the patched factories)
    @property
    def has_overrides(self):
        a = self.args
        return any(v is not None for v in [a.std_u, a.std_tau, a.adapt_jump, a.same_type_gain, a.norm_alpha, a.norm_ref,
                                           a.w_syn, a.conn_cap, a.dn_vnc_gain, a.vp_dn_gain, a.gain_out, a.t4_gain]) or self.receptor_model is not None

    @property
    def receptor_model(self):
        """LIFParams.receptor_model from --receptor-model ('off' -> None)."""
        m = self.args.receptor_model
        return None if m in (None, "off") else m

    def _apply_lif(self, p):
        a = self.args
        for k, v in [("std_u", a.std_u), ("std_tau", a.std_tau), ("adapt_jump", a.adapt_jump), ("same_type_gain", a.same_type_gain),
                     ("input_norm_alpha", a.norm_alpha), ("input_norm_ref", a.norm_ref), ("w_syn", a.w_syn), ("conn_cap", a.conn_cap)]:
            if v is not None:
                setattr(p, k, v)
        self._apply_receptor(p)
        if a.dn_vnc_gain is not None or a.vp_dn_gain is not None:
            p.path_gain = [(r"^descending_neuron$", r"^vnc_", 3.0 if a.dn_vnc_gain is None else a.dn_vnc_gain),
                           (r"^visual_projection$", r"^descending_neuron$", 2.0 if a.vp_dn_gain is None else a.vp_dn_gain)]
        return p

    def _apply_receptor(self, p):
        """The receptor-model flags (model, net rule, class fallback, the slow term's mode / class scales / taus and the
        --dopamine-lead table) on a LIFParams; a no-op with --receptor-model off."""
        a = self.args
        if self.receptor_model is None:
            return p
        p.receptor_model = self.receptor_model                # the receptor model (docs/NT_INTEGRATION.md) reaches every section
        p.receptor_net_rule = a.receptor_net_rule
        p.receptor_nt_class_fallback = bool(a.receptor_nt_class_fallback)
        if a.receptor_gain:
            low, mid, high = (float(x) for x in a.receptor_gain.split(","))
            p.receptor_gain = {"none": 1.0, "low": low, "mid": mid, "high": high}
        p.slow_mode = a.slow_mode
        gains, taus = {}, {}
        if a.slow_gain_monoamine is not None:
            gains["monoamine"] = a.slow_gain_monoamine
        if a.slow_gain_classical is not None:
            gains["metabotropic_classical"] = a.slow_gain_classical
        if a.slow_tau_monoamine is not None:
            taus["monoamine"] = a.slow_tau_monoamine
        if a.slow_tau_classical is not None:
            taus["metabotropic_classical"] = a.slow_tau_classical
        p.slow_gain_by_class = gains or None
        p.slow_tau_by_class = taus or None
        if self.receptor_table is not None:
            p.receptor_table = self.receptor_table
        return p

    def _build_dopamine_lead_table(self, lead):
        """--dopamine-lead dop1r1: rebuild receptors_by_type.csv in memory with the dopamine slow + group restricted to
        Dop1R1 / Dop1R2 (DopEcR ignored) through scripts/build_receptor_table's own rule (its RECEPTOR_GROUPS patched;
        without the raw weights, which only feed the per-type synapse columns -- every sign / class column is identical
        to the shipped table under the unpatched groups), written to out/receptors_by_type_<lead>.csv for the run."""
        import build_receptor_table as brt
        if lead != "dop1r1":
            raise ValueError(f"unknown --dopamine-lead {lead!r}")
        brt.RECEPTOR_GROUPS["dopamine"]["slow"] = [(+1, "Dop1R", ["Dop1R1", "Dop1R2"]), (-1, "Dop2R", ["Dop2R"])]
        t0 = time.time()
        _, rec, *_ = brt.build_tables(self.c, None, log=lambda *a, **k: None)
        path = os.path.join(os.path.dirname(__file__), "..", "out", f"receptors_by_type_{lead}.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# receptors_by_type.csv rebuilt by scripts/benchmark.py --dopamine-lead {lead}: dopamine slow + group = Dop1R1 / Dop1R2 only\n")
            rec.to_csv(f, index=False)
        d = rec[(rec.transmitter == "dopamine") & ~rec.malecns_type.astype(str).str.startswith("<")]
        info = {"lead": lead, "path": path, "seconds": round(time.time() - t0, 1),
                "dopamine_rows": int(len(d)), "slow_net": {k: int(v) for k, v in d.slow_net.value_counts().items()},
                "slow_pos_lead": {str(k): int(v) for k, v in d.slow_pos_lead.fillna("").value_counts().items()}}
        print(f"dopamine lead {lead}: table rebuilt in {info['seconds']} s -> {path}; dopamine rows {info['dopamine_rows']}, "
              f"slow_net {info['slow_net']}, + lead {info['slow_pos_lead']}", flush=True)
        return path, info

    def _apply_optic(self, op):
        a = self.args
        if a.gain_out is not None:
            op.gain_out_mv = a.gain_out
        if a.t4_gain is not None:
            op.pair_gain = [g for g in optic.DEFAULT_PAIR_GAIN if not g[0].startswith("^T[45]")] + [(r"^T[45][abcd]$", r".*", a.t4_gain)]
        return op

    def lif(self, **kw):
        return self._apply_lif(brain.LIFParams(**kw))

    def optic_params(self, **kw):
        return self._apply_optic(optic.OpticParams(**kw))

    @contextlib.contextmanager
    def patched_params(self):
        """While a demo Sim is built, brain.LIFParams / optic.OpticParams produce instances with the CLI overrides."""
        if not self.has_overrides:
            yield
            return
        L, O = brain.LIFParams, optic.OpticParams
        brain.LIFParams = lambda **kw: self._apply_lif(L(**kw))
        optic.OpticParams = lambda **kw: self._apply_optic(O(**kw))
        try:
            yield
        finally:
            brain.LIFParams, optic.OpticParams = L, O

    # ---- the demo simulation
    def sim(self, seed=0, **kw):
        import room_demo as rd
        flags = dict(cuda_kernels=True, cuda_graphs=True, event_driven=True, cuda_sparse="warp") if self.native else {}
        with self.patched_params():
            return rd.Sim(seed, trail_seconds=0.0, **flags, **kw)

    def free(self, *objs):
        del objs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def report(self, key, value):
        ref = REFERENCES[key]
        status = evaluate(ref, value)
        self.checks.append({"key": key, "measured": value, "reference": ref.value, "criterion": f"{ref.op} {ref.bound}",
                            "status": status, "session": ref.session, "note": ref.note})
        return status


def F(seconds):
    """frames of 10 ms"""
    return int(round(seconds * 100))


def smooth_skip(x, skip_s=3.0):
    return screen._smooth(np.asarray(x, np.float32), 1.0)[F(skip_s):]


def top_types(c, rt, k=4):
    g = c.neurons.assign(rate=rt).groupby("type").rate.agg(["mean", "size"])
    return g[g["size"] >= 2].sort_values("mean", ascending=False).head(k)["mean"].round(0).to_dict()


def lr_flip(rec, runs, t, a, b, skip_s=3.0):
    """(L - R) mean of type t in condition a minus in condition b, from by-side TypeRecorder runs (T, keys)."""
    pos = {k: i for i, k in enumerate(rec.keys)}
    if f"{t}_L" not in pos or f"{t}_R" not in pos:      # a type with no cells on one side (or none at all) has no asymmetry
        return None, {k: None for k in runs}
    iL, iR = pos[f"{t}_L"], pos[f"{t}_R"]
    m = {k: float((smooth_skip(v, skip_s)[:, iL] - smooth_skip(v, skip_s)[:, iR]).mean()) for k, v in runs.items()}
    return m[a] - m[b], m


# ------------------------------------------------------------------------------------------------ legacy sections
def sec_rest(ctx):
    b = brain.Brain(ctx.c, ctx.lif()); b.run_ms(500)
    res = {"spikes_per_step": float(b.total_spikes())}
    ctx.report("rest.spikes_per_step", res["spikes_per_step"])
    print(f"rest      spikes/step {res['spikes_per_step']:.0f}")
    ctx.free(b)
    return res


def sec_taste(ctx):
    c = ctx.c; n = c.neurons
    taste = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "flyverse", "data", "taste_grns.csv"))
    sweet = c.index_of(taste.bodyId[taste.taste == "sweet"].to_numpy())
    sweet = sweet[np.isin(n.subclass.to_numpy()[sweet], ["labellar bristle", "taste peg"])]
    b = brain.Brain(c, ctx.lif()); b.set_poisson(sweet, 100.0); b.run_ms(600); rt = b.rate_np()
    res = {"MN9_hz": float(rt[c.select(type="MN9")].mean()), "GNG175_hz": float(rt[c.select(type="GNG175")].mean()),
           "frac_active": float((rt > 1).mean()), "top": top_types(c, rt)}
    ctx.report("taste.MN9_hz", res["MN9_hz"])
    print(f"taste     MN9 {res['MN9_hz']:.1f} Hz  GNG175 {res['GNG175_hz']:.0f}  frac {res['frac_active']:.3f}  top {res['top']}")
    ctx.free(b)
    return res


def sec_smell(ctx):
    c = ctx.c
    olf = olfaction.Olfaction(c, [("apple", (0.25, 0.15, 0.79), 1.0)])
    b = brain.Brain(c, ctx.lif()); olf.apply(b, (0.19, 0.15, 0.75)); b.run_ms(800); rt = b.rate_np()
    pn = c.select(type="~_l2PN|_adPN|_lPN|_lvPN|_ilPN"); kc = c.select(type="~^KC"); ln = c.select(type="~^(lLN|v2LN)")
    res = {"PN_hz": float(rt[pn].mean()), "PN_max_hz": float(rt[pn].max()), "KC_hz": float(rt[kc].mean()),
           "KC_active": int((rt[kc] > 1).sum()), "LN_hz": float(rt[ln].mean()), "frac_active": float((rt > 1).mean()), "top": top_types(c, rt)}
    ctx.report("smell.PN_hz", res["PN_hz"]); ctx.report("smell.KC_active", res["KC_active"])
    print(f"smell     PN {res['PN_hz']:.0f} (max {res['PN_max_hz']:.0f})  KC {res['KC_hz']:.2f} ({res['KC_active']} active)  LN {res['LN_hz']:.0f}  "
          f"frac {res['frac_active']:.3f}  top {res['top']}")
    ctx.free(b)
    return res


def sec_dn(ctx):
    c = ctx.c; side = c.neurons.somaSide.to_numpy()
    leg = c.select(superclass="vnc_motor", subclass=["fl", "ml", "hl"]); legL = leg[side[leg] == "L"]; legR = leg[side[leg] == "R"]
    wg = motor.wing_groups(c)
    res = {}
    for name, idx in [("DNa02_L", c.select(type="DNa02", somaSide="L")), ("DNp09", c.select(type="DNp09")), ("MDN", c.select(type="MDN"))]:
        b = brain.Brain(c, ctx.lif()); b.set_poisson(idx, 150.0); b.run_ms(400); rt = b.rate_np()
        res[name] = {"legL": float(rt[legL].mean()), "legR": float(rt[legR].mean()), "power": float(rt[wg.power].mean()),
                     "frac": float((rt > 1).mean()), "top": top_types(c, rt)}
        d = res[name]
        print(f"dn {name:8s} legL {d['legL']:.1f} legR {d['legR']:.1f} power {d['power']:.1f} frac {d['frac']:.3f} top {d['top']}")
        ctx.free(b)
    ctx.report("dn.DNa02_L_leg_asym_hz", res["DNa02_L"]["legL"] - res["DNa02_L"]["legR"])
    for name in ("MDN", "DNp09"):
        ctx.report(f"dn.{name}_top_hz", max(res[name]["top"].values()) if res[name]["top"] else 0.0)
    return res


def sec_walk(ctx):
    """The original hybrid protocol: 1.5 s walking with smell on, a loom from the left, 0.8 s of yaw each way."""
    c = ctx.c; n = c.neurons; side = n.somaSide.to_numpy()
    leg = c.select(superclass="vnc_motor", subclass=["fl", "ml", "hl"]); wg = motor.wing_groups(c)
    r = retina.build_retina(c)
    w, info = world.make_room()
    w.spheres.append(world.Sphere((9, 9, 9), (0.03,) * 3, "black")); loom_idx = len(w.spheres) - 1
    dirs_b, wts = r.ray_directions(); wts_t = torch.from_numpy(wts).float().to(w.device)
    ol = optic.OpticLobe(c, r, ctx.optic_params()); ol.relax()
    b = brain.Brain(c, ctx.lif()); b.freeze(ol.rate_idx)
    fly = body.FlyState(x=-0.3, y=0.0, z=info["table_top_z"], heading=0.0)

    def col_rad():
        d = fly.body_to_world(dirs_b.reshape(-1, 3)); o = np.broadcast_to(fly.eye_pos, d.shape)
        rad = w.trace(torch.from_numpy(np.ascontiguousarray(o)).float(), torch.from_numpy(d).float())
        return (rad.reshape(dirs_b.shape[0], dirs_b.shape[1], 4) * wts_t[None, :, None]).sum(1)

    gf = c.select(type="DNp01"); a02L = c.select(type="DNa02", somaSide="L"); a02R = c.select(type="DNa02", somaSide="R")
    res = {}
    gf_walk = []; pw_walk = []
    olf_walk = olfaction.Olfaction(c, [(name, cen, 1.0) for name, cen, rad in info["fruit"]])   # smell on, as in the demo
    for k in range(150):
        fly.x += 0.004 * 0.01; olf_walk.apply(b, fly.eye_pos); b.drive = ol.step_frame(col_rad(), b.rate, 10.0); b.step(20)
        if k >= 50:
            gf_walk.append(float(b.rate[0, b._idx(gf)].mean())); pw_walk.append(float(b.rate[0, b._idx(wg.power)].mean()))
    rt = b.rate_np()
    pw_sustained = float(np.max(np.convolve(pw_walk, np.ones(30) / 30, mode="valid")))   # 0.3 s running mean: what a voluntary takeoff needs
    res["walk"] = {"GF_mean_hz": float(np.mean(gf_walk)), "GF_max_hz": float(np.max(gf_walk)), "frac_active": float((rt > 1).mean()),
                   "power_mean_hz": float(np.mean(pw_walk)), "power_max_hz": float(np.max(pw_walk)), "power_sustained_hz": pw_sustained,
                   "leg_hz": float(rt[leg].mean()), "top": top_types(c, rt)}
    eye = fly.eye_pos + np.array([0, 0, 0.01]); esc = None; gf_peak = 0.0
    for k in range(80):
        d = max(0.5 - 1.0 * k * 0.01, 0.035)
        w.move_sphere(loom_idx, eye + np.array([0.0, d, 0.0]))
        b.drive = ol.step_frame(col_rad(), b.rate, 10.0); b.step(20)
        g = float(b.rate[0, b._idx(gf)].mean()); gf_peak = max(gf_peak, g)
        if esc is None and g >= 20:
            esc = d * 100
    res["loom"] = {"GF_peak_hz": gf_peak, "escape_cm": esc}
    w.move_sphere(loom_idx, (9, 9, 9))
    out = {}; dnp20 = {}
    dn_all = c.select(superclass="descending_neuron"); dn_side = side[dn_all]; dn_type = n.type.fillna("").to_numpy()[dn_all]
    res["rotate"] = {}
    for name, sgn in [("left", 1), ("right", -1)]:
        acc = np.zeros(len(dn_all))
        for k in range(80):
            fly.heading += sgn * np.deg2rad(90) * 0.01
            b.drive = ol.step_frame(col_rad(), b.rate, 10.0); b.step(20)
            if k >= 20:
                acc += b.rate[0, b._idx(dn_all)].cpu().numpy()
        acc /= 60
        out[name] = (float(b.rate[0, b._idx(a02L)].mean()), float(b.rate[0, b._idx(a02R)].mean()))
        df = pd.DataFrame({"type": dn_type, "side": dn_side, "rate": acc})
        piv = df.groupby(["type", "side"]).rate.mean().unstack().fillna(0)
        piv["asym"] = piv.get("L", 0) - piv.get("R", 0)
        dnp20[name] = float(piv.asym.get("DNp20", 0.0))
        piv = piv[(piv[["L", "R"]].max(axis=1) > 3)]
        res["rotate"][f"{name}_asym_dns"] = piv.reindex(piv.asym.abs().sort_values(ascending=False).index).head(6).round(1).to_dict("index")
        for k in range(50):
            b.drive = ol.step_frame(col_rad(), b.rate, 10.0); b.step(20)
    res["rotate"]["left_DNa02_L_R"] = out["left"]; res["rotate"]["right_DNa02_L_R"] = out["right"]
    res["rotate"]["DNp20_LR_left"] = dnp20["left"]; res["rotate"]["DNp20_LR_right"] = dnp20["right"]
    res["rotate"]["DNp20_flip_hz"] = dnp20["left"] - dnp20["right"]
    wk, lm = res["walk"], res["loom"]
    ctx.report("walk.GF_max_hz", wk["GF_max_hz"]); ctx.report("walk.power_max_hz", wk["power_max_hz"]); ctx.report("walk.power_sustained_hz", pw_sustained)
    ctx.report("loom.GF_peak_hz", lm["GF_peak_hz"]); ctx.report("loom.escape_cm", lm["escape_cm"])
    ctx.report("rotate.DNp20_flip_hz", res["rotate"]["DNp20_flip_hz"])
    print(f"walk      GF mean {wk['GF_mean_hz']:.1f} max {wk['GF_max_hz']:.0f}  wing power mean {wk['power_mean_hz']:.1f} max {wk['power_max_hz']:.0f} "
          f"(0.3 s sustained max {pw_sustained:.0f})  "
          f"leg MN {wk['leg_hz']:.1f}  frac {wk['frac_active']:.3f}  top {wk['top']}")
    print(f"loom      GF peak {lm['GF_peak_hz']:.0f} Hz  escape at {lm['escape_cm']} cm")
    print(f"rotate    left: DNa02 L/R {out['left'][0]:.1f}/{out['left'][1]:.1f}   right: {out['right'][0]:.1f}/{out['right'][1]:.1f}   "
          f"DNp20 L-R left {dnp20['left']:+.1f} right {dnp20['right']:+.1f}")
    for name in ["left", "right"]:
        print(f"  rotate {name:5s} most lateralised DNs (L, R, L-R Hz): " + "; ".join(f"{t}: {v['L']:.0f}/{v['R']:.0f} ({v['asym']:+.0f})"
                                                                                  for t, v in res["rotate"][f"{name}_asym_dns"].items()))
    ctx.free(b, ol, w)
    return res


# ------------------------------------------------------------------------------------------------ a: motion
def sec_motion(ctx):
    from probe_motion import DIRS, grating
    c = ctx.c; types = c.neurons.type.fillna("").to_numpy()
    r = retina.build_retina(c)
    ol = optic.OpticLobe(c, r, ctx.optic_params()); ol.relax(); rt = types[ol.rate_idx]
    b = brain.Brain(c, ctx.lif()); b.freeze(ol.rate_idx)
    if b.p.prune_frozen:
        b.prune(ol.rate_idx)
    subtypes = optic.T4T5
    expected = {"a": "front->back", "b": "back->front", "c": "up", "d": "down"}
    n = F(1.0 if ctx.fast else 1.5)
    resp = {t: {} for t in subtypes}
    for name in DIRS:
        ol.reset(); b.reset(); acc = {t: [] for t in subtypes}
        for k in range(n):
            b.drive = ol.step_frame(grating(r, k * 0.01, name, 60.0, 30.0, 0.5), b.rate, 10.0); b.step(20)
            if k >= n // 3:
                dr = ol.last["dr"][0].cpu().numpy()
                for t in subtypes:
                    acc[t].append(np.maximum(dr[rt == t], 0).mean())
        for t in subtypes:
            resp[t][name] = float(np.mean(acc[t]))
    res = {"speed_deg_s": 60.0, "period_deg": 30.0, "subtypes": {}}
    for t in subtypes:
        vals = [resp[t][d] for d in DIRS]; best = list(DIRS)[int(np.argmax(vals))]
        dsi = (max(vals) - min(vals)) / (abs(max(vals)) + abs(min(vals)) + 1e-6)
        res["subtypes"][t] = {"dsi": float(dsi), "best": best, "expected": expected[t[-1]], "correct": best == expected[t[-1]], "response": resp[t]}
        print(f"  {t}: DSI {dsi:.2f} best {best:12s} expected {expected[t[-1]]:12s} {'ok' if best == expected[t[-1]] else 'WRONG'}")
    res["min_dsi"] = float(min(v["dsi"] for v in res["subtypes"].values()))
    res["correct_directions"] = int(sum(v["correct"] for v in res["subtypes"].values()))
    ctx.report("motion.min_dsi", res["min_dsi"]); ctx.report("motion.correct_directions", res["correct_directions"])
    print(f"motion    min DSI {res['min_dsi']:.2f}  correct preferred directions {res['correct_directions']}/8")
    ctx.free(b, ol)
    return res


# ------------------------------------------------------------------------------------------------ b: loom escape in the demo
def sec_loom_escape(ctx):
    res = {"seeds": {}}
    walk_s = 3.0 if ctx.fast else 5.0
    n_walk, n_loom = F(walk_s), F(1.5)
    for seed in ctx.seeds:
        sim = ctx.sim(seed)
        gf_walk, gf_loom = [], []; hops_before = 0; was = False; esc = None
        for k in range(n_walk + n_loom):
            if k == n_walk:
                sim.start_loom()
            sim.step()
            g, air = float(sim.wcmd["gf"]), bool(sim.fly.airborne)
            if k < n_walk:
                gf_walk.append(g)
                if air and not was:
                    hops_before += 1
            else:
                gf_loom.append(g)
                if air and not was and esc is None:
                    esc = {"t_after_loom_s": (k - n_walk + 1) * 0.01, "range_cm": max(0.5 - (k - n_walk + 1) * 0.01, 0.035) * 100, "GF_hz": g}
            was = air
        res["seeds"][seed] = {"GF_walk_max_hz": float(np.max(gf_walk)), "hops_before_loom": hops_before,
                              "GF_loom_peak_hz": float(np.max(gf_loom)), "escape": esc is not None, "escape_info": esc,
                              "gf_threshold": float(sim.flight.gf_hz)}
        s = res["seeds"][seed]
        print(f"loom_escape seed {seed}: walking GF max {s['GF_walk_max_hz']:.0f} Hz ({hops_before} hops in {walk_s:.0f} s), "
              f"loom GF peak {s['GF_loom_peak_hz']:.0f} Hz, escape {s['escape']}" + (f" at {esc['range_cm']:.1f} cm, {esc['t_after_loom_s']:.2f} s" if esc else ""))
        ctx.free(sim)
    res["GF_peak_hz"] = float(max(s["GF_loom_peak_hz"] for s in res["seeds"].values()))
    res["escapes"] = int(sum(s["escape"] for s in res["seeds"].values()))
    ctx.report("loom_escape.GF_peak_hz", res["GF_peak_hz"]); ctx.report("loom_escape.escapes", res["escapes"])
    return res


# ------------------------------------------------------------------------------------------------ c: walking GF
def sec_walk_gf(ctx):
    T = 8 if ctx.fast else 15
    sim = ctx.sim(0)
    sim.flight.gf_hz = 1e9          # the GF is measured, not acted on (no escape hops)
    gf = np.zeros(F(T)); hops = 0; was = False
    for k in range(F(T)):
        sim.step(); gf[k] = float(sim.wcmd["gf"])
        air = bool(sim.fly.airborne)
        if air and not was:
            hops += 1
        was = air
    maxima = gf.reshape(T, 100).max(axis=1)
    res = {"seconds": T, "per_second_max_hz": maxima.round(1).tolist(), "median_hz": float(np.median(maxima)), "p90_hz": float(np.percentile(maxima, 90)),
           "p99_hz": float(np.percentile(maxima, 99)), "max_hz": float(maxima.max()), "mean_hz": float(gf.mean()), "voluntary_takeoffs": hops,
           "gf_threshold": float(body.Flight().gf_hz)}
    ctx.report("walk_gf.p99_hz", res["p99_hz"])
    print(f"walk_gf   {T} s: GF per-second maxima median {res['median_hz']:.0f} p90 {res['p90_hz']:.0f} p99 {res['p99_hz']:.0f} max {res['max_hz']:.0f} Hz "
          f"(mean {res['mean_hz']:.1f}); voluntary takeoffs {hops}")
    ctx.free(sim)
    return res


# ------------------------------------------------------------------------------------------------ d: sustained rotation
def sec_rotation(ctx):
    T = 6.0 if ctx.fast else 10.0
    sim = ctx.sim(0, start=(0.0, 0.0, 0.75), wind_speed=0.0)
    types = ["DNp20", "HSN", "HSE", "DNp04", "LPT27", "LPT30"]
    rec = screen.TypeRecorder.build(sim.c, types=types, by_side=True)
    h = 0.0; runs = {}; group = {}
    for name, rate in [("rest", 0.0), ("ccw", 90.0), ("rest2", 0.0), ("cw", -90.0)]:
        rows = np.zeros((F(T), len(rec.keys)), np.float32); opt = np.zeros((F(T), 2), np.float32)
        for k in range(F(T)):
            h += np.deg2rad(rate) * 0.01
            sim.fly.place(0.0, 0.0, 0.75, heading=h); sim.step()
            rows[k] = rec.snapshot(sim.fb.brain.rate_np()); opt[k] = (sim.cmd["rates"]["opto_L"], sim.cmd["rates"]["opto_R"])
        runs[name] = rows; group[name] = opt
    gm = {k: float((smooth_skip(v)[:, 0] - smooth_skip(v)[:, 1]).mean()) for k, v in group.items()}
    res = {"seconds_per_condition": T, "group": "DNp20 + HSN + HSE", "group_LR": gm, "group_flip_hz": gm["ccw"] - gm["cw"],
           "group_rest_LR": 0.5 * (gm["rest"] + gm["rest2"]), "types": {}}
    for t in types:
        flip, m = lr_flip(rec, runs, t, "ccw", "cw")
        res["types"][t] = {"flip_hz": flip, "LR": m}
    ctx.report("rotation.group_flip_hz", res["group_flip_hz"])
    print(f"rotation  group (DNp20+HSN+HSE) L-R: rest {gm['rest']:+.1f} ccw {gm['ccw']:+.1f} rest2 {gm['rest2']:+.1f} cw {gm['cw']:+.1f} -> flip (ccw - cw) {res['group_flip_hz']:+.1f} Hz")
    print("          per type flip: " + "  ".join(f"{t} {v['flip_hz']:+.1f}" for t, v in res["types"].items()))
    ctx.free(sim)
    return res


# ------------------------------------------------------------------------------------------------ e: object side
def sec_object(ctx):
    T = 6.0 if ctx.fast else 10.0
    apple = np.array([0.25, 0.15]); r = 0.04 + 0.05
    sites = {}
    for name, ang in [("apple_left", np.deg2rad(135)), ("apple_right", np.deg2rad(45))]:   # fly faces +y; apple 45 deg ahead-left / ahead-right
        pos = apple - r * np.array([np.cos(ang), np.sin(ang)])
        sites[name] = (float(pos[0]), float(pos[1]))
    x0, y0 = sites["apple_left"]
    sim = ctx.sim(0, start=(x0, y0, 0.75), fruit_set="apple", fence=True, wind_speed=0.0)
    types = ["LC10a", "LC10b", "LC10d", "LC16", "LC11", "DNa02"]      # (LC10c has no cells in MaleCNS)
    rec = screen.TypeRecorder.build(sim.c, types=types, by_side=True)
    runs = {}; dist = {}
    for name, (x, y) in sites.items():
        rows = np.zeros((F(T), len(rec.keys)), np.float32)
        for k in range(F(T)):
            hd = np.pi / 2 + np.deg2rad(20) * np.sin(2 * np.pi * 0.5 * k * 0.01)
            sim.fly.place(x, y, 0.75, heading=hd); sim.step()
            rows[k] = rec.snapshot(sim.fb.brain.rate_np())
        runs[name] = rows; dist[name] = float(sim.nearest_fruit()[1] * 100)
    pos = {k: i for i, k in enumerate(rec.keys)}
    res = {"seconds_per_site": T, "apple_surface_cm": dist, "types": {}}
    for t in types:
        flip, m = lr_flip(rec, runs, t, "apple_left", "apple_right")
        cols = [pos[k] for k in (f"{t}_L", f"{t}_R") if k in pos]
        rate = float(np.mean([smooth_skip(v)[:, cols].mean() for v in runs.values()])) if cols else None
        res["types"][t] = {"flip_hz": flip, "LR": m, "rate_hz": rate}
    res["LC10a_flip_hz"] = res["types"]["LC10a"]["flip_hz"]; res["LC10a_hz"] = res["types"]["LC10a"]["rate_hz"]
    ctx.report("object.LC10a_flip_hz", res["LC10a_flip_hz"])
    print(f"object    LC10a L-R apple-left {res['types']['LC10a']['LR']['apple_left']:+.2f} apple-right {res['types']['LC10a']['LR']['apple_right']:+.2f} "
          f"-> flip {res['LC10a_flip_hz']:+.2f} Hz (rate {res['LC10a_hz']:.2f} Hz); " + "  ".join(f"{t} {v['flip_hz']:+.2f}" for t, v in res["types"].items() if t != "LC10a"))
    ctx.free(sim)
    return res


# ------------------------------------------------------------------------------------------------ f: sugar / bitter
def sec_bitter(ctx):
    c = ctx.c
    table = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "flyverse", "data", "taste_grns.csv"))
    ids = {k: table.bodyId[table.taste == k].to_numpy() for k in ("sweet", "bitter")}
    sweet = c.index_of(ids["sweet"][np.isin(ids["sweet"], c.neurons.bodyId)])
    bitter = c.index_of(ids["bitter"][np.isin(ids["bitter"], c.neurons.bodyId)])
    mn9 = c.select(type="MN9")
    ms = 1000.0 if ctx.fast else 1500.0
    settings = {"calibrated": ctx.lif(),
                "shiu": ctx._apply_receptor(brain.LIFParams(adapt_jump=0.0, conn_cap=0.0, same_type_gain=1.0, input_norm_alpha=0.0, std_u_by_type={},
                                                            path_gain=[], type_path_gain=[]))}   # the receptor model (incl. the slow term) applies to both, as in probe_bitter.py
    res = {"ms": ms, "rate_hz": 100.0}
    steps = int(ms / 0.5)
    for label, p in settings.items():
        for cond, drive in [("sugar", {"sweet": 100.0}), ("sugar_bitter", {"sweet": 100.0, "bitter": 100.0})]:
            b = brain.Brain(c, p, seed=0)
            for k, hz in drive.items():
                b.set_poisson(sweet if k == "sweet" else bitter, hz)
            b.step(steps // 3)                                        # settle, then measure the last two thirds
            acc = 0.0
            for _ in range(20):
                b.step(steps // 30); acc += float(b.rate_np()[mn9].mean())
            res[f"{label}_{cond}_MN9_hz"] = acc / 20
            ctx.free(b)
        print(f"bitter    {label:10s} MN9: sugar {res[f'{label}_sugar_MN9_hz']:.1f} Hz, sugar + bitter {res[f'{label}_sugar_bitter_MN9_hz']:.1f} Hz")
    for key in ["calibrated_sugar", "calibrated_sugar_bitter", "shiu_sugar", "shiu_sugar_bitter"]:
        ctx.report(f"bitter.{key}_MN9_hz", res[f"{key}_MN9_hz"])
    return res


# ------------------------------------------------------------------------------------------------ g: wind direction
def sec_wind(ctx):
    T = 6.0 if ctx.fast else 10.0
    x, y = 0.55, 0.35                                    # the plume-free spot; wind 0.3 m/s from +x (default)
    sim = ctx.sim(0, start=(x, y, 0.75))
    types = ["DNp18", "DNp33", "DNge016", "DNg99", "DNg05_a", "WED080"]
    rec = screen.TypeRecorder.build(sim.c, types=types, by_side=True)
    runs = {}
    for name, hd in [("windL", -90.0), ("windR", 90.0)]:  # heading -90 (facing -y): +x, where the wind comes from, is on the fly's left
        rows = np.zeros((F(T), len(rec.keys)), np.float32)
        for k in range(F(T)):
            sim.fly.place(x, y, 0.75, heading=np.deg2rad(hd)); sim.step()
            rows[k] = rec.snapshot(sim.fb.brain.rate_np())
        runs[name] = rows
    res = {"seconds_per_condition": T, "types": {}}
    for t in types:
        flip, m = lr_flip(rec, runs, t, "windL", "windR")
        res["types"][t] = {"flip_hz": flip, "LR": m}
    res["DNp18_flip_hz"] = res["types"]["DNp18"]["flip_hz"]; res["DNp33_flip_hz"] = res["types"]["DNp33"]["flip_hz"]
    ctx.report("wind.DNp18_flip_hz", res["DNp18_flip_hz"]); ctx.report("wind.DNp33_flip_hz", res["DNp33_flip_hz"])
    print("wind      L-R flip (wind left - wind right): " + "  ".join(f"{t} {v['flip_hz']:+.1f}" for t, v in res["types"].items()))
    ctx.free(sim)
    return res


# ------------------------------------------------------------------------------------------------ h: odour gate
def sec_odour(ctx):
    T = 6.0 if ctx.fast else 10.0
    sites = {"apple8": (0.17, 0.15), "clean": (0.55, 0.35)}      # 8 cm downwind of the apple at (0.25, 0.15); plume-free
    sim = ctx.sim(0, start=(*sites["apple8"], 0.75), fruit_set="apple")
    channel = motor.LH_ODOUR_CHANNELS["apple"]
    rec = screen.TypeRecorder.build(sim.c, types=channel + ["LHPD5c1"])
    pos = {k: i for i, k in enumerate(rec.keys)}
    runs = {}; chan = {}
    for name, (x, y) in sites.items():
        rows = np.zeros((F(T), len(rec.keys)), np.float32); ch = np.zeros(F(T), np.float32)
        for k in range(F(T)):
            sim.fly.place(x, y, 0.75, heading=0.0); sim.step()           # facing into the wind
            rows[k] = rec.snapshot(sim.fb.brain.rate_np()); ch[k] = sim.fb.motor().lh_odour["apple"]
        runs[name] = rows; chan[name] = ch
    res = {"seconds_per_site": T, "channel": channel, "channel_hz": {k: float(smooth_skip(v).mean()) for k, v in chan.items()},
           "types": {t: {k: float(smooth_skip(v)[:, pos[t]].mean()) for k, v in runs.items()} for t in rec.keys}}
    res["apple_channel_8cm_hz"] = res["channel_hz"]["apple8"]; res["apple_channel_clean_hz"] = res["channel_hz"]["clean"]
    ctx.report("odour.apple_channel_8cm_hz", res["apple_channel_8cm_hz"]); ctx.report("odour.apple_channel_clean_hz", res["apple_channel_clean_hz"])
    print(f"odour     LH apple channel: 8 cm downwind {res['apple_channel_8cm_hz']:.1f} Hz, plume-free {res['apple_channel_clean_hz']:.1f} Hz; "
          + "  ".join(f"{t} {v['apple8']:.1f}/{v['clean']:.1f}" for t, v in res["types"].items()))
    ctx.free(sim)
    return res


# ------------------------------------------------------------------------------------------------ i: compass
def sec_compass(ctx):
    c = ctx.c
    epg = c.select(type="EPG")
    inst = c.neurons.instance.fillna("").to_numpy()
    wedge = epg[pd.Series(inst[epg]).str.contains(r"_(?:L3|L4|R5|R6)$", regex=True).to_numpy()]   # 12 cells: two PB glomeruli per side
    rest = np.setdiff1d(epg, wedge)
    pen = c.select(type="~^PEN"); d7 = c.select(type="Delta7"); pfl3 = c.select(type="PFL3")
    b = brain.Brain(c, ctx.lif())
    b.set_poisson(wedge, 60.0); b.run_ms(2000); rt = b.rate_np()
    during = {"wedge_hz": float(rt[wedge].mean()), "rest_EPG_hz": float(rt[rest].mean()), "PEN_hz": float(rt[pen].mean()),
              "Delta7_hz": float(rt[d7].mean()), "PFL3_hz": float(rt[pfl3].mean())}
    b.set_poisson(wedge, 0.0); b.run_ms(500); rt = b.rate_np()
    after = {"wedge_hz": float(rt[wedge].mean()), "rest_EPG_hz": float(rt[rest].mean()), "PEN_hz": float(rt[pen].mean()), "Delta7_hz": float(rt[d7].mean()),
             "wedge_cells_above_5hz": int((rt[wedge] > 5).sum()), "rest_cells_above_5hz": int((rt[rest] > 5).sum())}
    res = {"wedge_cells": int(len(wedge)), "wedge_instances": sorted(set(inst[wedge])), "drive_hz": 60.0, "drive_ms": 2000, "after_ms": 500,
           "during": during, "after": after, "wedge_cells_persisting": after["wedge_cells_above_5hz"]}
    ctx.report("compass.wedge_cells_persisting", res["wedge_cells_persisting"])
    print(f"compass   during drive: wedge {during['wedge_hz']:.1f} Hz, other EPG {during['rest_EPG_hz']:.1f}, PEN {during['PEN_hz']:.1f}, Delta7 {during['Delta7_hz']:.1f}; "
          f"0.5 s after: wedge {after['wedge_hz']:.1f} Hz ({after['wedge_cells_above_5hz']}/{len(wedge)} cells > 5 Hz), other EPG {after['rest_EPG_hz']:.1f} "
          f"({after['rest_cells_above_5hz']}/{len(rest)}), PEN {after['PEN_hz']:.1f}")
    ctx.free(b)
    return res


# ------------------------------------------------------------------------------------------------ registry / main
SECTIONS = [  # (letter or None, name, function)
    (None, "rest", sec_rest), (None, "taste", sec_taste), (None, "smell", sec_smell), (None, "dn", sec_dn), (None, "walk", sec_walk),
    ("a", "motion", sec_motion), ("b", "loom_escape", sec_loom_escape), ("c", "walk_gf", sec_walk_gf), ("d", "rotation", sec_rotation),
    ("e", "object", sec_object), ("f", "bitter", sec_bitter), ("g", "wind", sec_wind), ("h", "odour", sec_odour), ("i", "compass", sec_compass),
]
LEGACY = ["rest", "taste", "smell", "dn", "walk"]


def select_sections(text):
    if not text or text == "all":
        return [s for s in SECTIONS]
    want = set()
    for tok in text.split(","):
        tok = tok.strip()
        if tok == "legacy":
            want.update(LEGACY)
        elif tok in ("new", "abcdefghi"):
            want.update(name for letter, name, _ in SECTIONS if letter)
        else:
            want.add(tok)
    chosen = [s for s in SECTIONS if s[1] in want or (s[0] and s[0] in want)]
    unknown = want - {s[1] for s in chosen} - {s[0] for s in chosen if s[0]}
    if unknown:
        raise SystemExit(f"unknown sections {sorted(unknown)}; known: " + ", ".join(f"{l or '-'}={n}" for l, n, _ in SECTIONS))
    return chosen


def fmt(v):
    if v is None:
        return "--"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"
    return str(v)


def summary_table(ctx):
    rows = [("check", "measured", "reference", "criterion", "status", "NOTES")]
    for ch in ctx.checks:
        rows.append((ch["key"], fmt(ch["measured"]), fmt(ch["reference"]), ch["criterion"], ch["status"], ch["session"]))
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    lines = ["| " + " | ".join(str(v).ljust(w) for v, w in zip(r, widths)) + " |" for r in rows]
    lines.insert(1, "|" + "|".join("-" * (w + 2) for w in widths) + "|")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", default="all", help="comma-separated letters a-i and/or names (rest, taste, smell, dn, walk, motion, ...); 'legacy', 'new', 'all'")
    ap.add_argument("--json", type=str, default="", help="write every measured number and the check table to this file")
    ap.add_argument("--fast", action="store_true", help="shorter recordings and one seed (~half the runtime)")
    ap.add_argument("--eager", action="store_true", help="demo sections on the torch path (default: cuda_kernels + cuda_graphs + event_driven + warp CSR)")
    ap.add_argument("--seeds", default="0,1", help="seeds for the demo loom-escape section (default 0,1; --fast keeps the first)")
    ap.add_argument("--std-u", type=float, default=None)
    ap.add_argument("--std-tau", type=float, default=None)
    ap.add_argument("--adapt-jump", type=float, default=None)
    ap.add_argument("--same-type-gain", type=float, default=None)
    ap.add_argument("--norm-alpha", type=float, default=None)
    ap.add_argument("--norm-ref", type=float, default=None)
    ap.add_argument("--w-syn", type=float, default=None)
    ap.add_argument("--conn-cap", type=float, default=None)
    ap.add_argument("--dn-vnc-gain", type=float, default=None, help="gain on descending -> VNC synapses (default 3)")
    ap.add_argument("--vp-dn-gain", type=float, default=None, help="gain on visual projection -> descending synapses (default 2)")
    ap.add_argument("--gain-out", type=float, default=None, help="optic lobe -> spiking drive gain (mV)")
    ap.add_argument("--t4-gain", type=float, default=None, help="T4/T5 output gain (default 2)")
    ap.add_argument("--receptor-model", default="off", choices=["off", "sign", "sign+gain", "full"],
                    help="LIFParams.receptor_model for every section (default off = the presynaptic NT_SIGN rule); 'full' needs --eager")
    ap.add_argument("--receptor-net-rule", default="class", choices=["class", "abs", "nonmda"])
    ap.add_argument("--receptor-nt-class-fallback", action="store_true",
                    help="LIFParams.receptor_nt_class_fallback: unprofiled targets take the Davis 2020 ChAT / Gad1 / VGlut class baseline (tier nt_class)")
    ap.add_argument("--cache-dir", default=None,
                    help="connectome cache directory (default cache/, or $FLYVERSE_CACHE); e.g. a scratch cache built with a different TYPE_NT_OVERRIDE")
    # the slow term of --receptor-model full (LIFParams.slow_*; docs/audits/slow_term.md)
    ap.add_argument("--slow-mode", default="additive", choices=list(brain.SLOW_MODES),
                    help="how the slow tone acts on its target: added to the membrane input, a multiplicative gain on the fast input, or a threshold shift")
    ap.add_argument("--slow-gain-monoamine", type=float, default=None, help="scale of the monoamine (DA / OA / 5-HT) slow class, x w_syn per synapse per spike (default LIFParams.slow_gain = 0.02)")
    ap.add_argument("--slow-gain-classical", type=float, default=None, help="scale of the classical metabotropic class (mAChR / GABA-B / mGluR; default 0 = off)")
    ap.add_argument("--slow-tau-monoamine", type=float, default=None, help="time constant (ms) of the monoamine slow class (default LIFParams.slow_tau_ms = 200)")
    ap.add_argument("--slow-tau-classical", type=float, default=None, help="time constant (ms) of the classical metabotropic class (default 100)")
    ap.add_argument("--dopamine-lead", default="all", choices=["all", "dop1r1"],
                    help="'dop1r1': rebuild the receptor table with the dopamine slow + group = Dop1R1 / Dop1R2 only (DopEcR ignored) and use it")
    ap.add_argument("--receptor-gain", default=None,
                    help="gain-class factors of 'sign+gain' / 'full' as low,mid,high (default 0.5,1,1.5); '1,1,1' = the 'sign' fast weights under 'full'")
    args = ap.parse_args()
    t_all = time.time()
    ctx = Context(args)
    lif, op = ctx.lif(), ctx.optic_params()
    if ctx.receptor_model is not None:      # the coverage the model runs under (connectome.receptor_signs' tier summary)
        rs = brain._receptor(ctx.c, lif, with_counts=ctx.receptor_model == "full")   # counts for the slow-term summary below
        cov = rs.coverage(ctx.c.W)
        print(f"receptor model {ctx.receptor_model} ({args.receptor_net_rule}); fast sign changed on "
              f"{int((rs.fast_sign != np.sign(ctx.c.W.data)).sum()):,} of {ctx.c.W.nnz:,} entries; coverage by tier:")
        print(cov.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        receptor_cfg = {"model": ctx.receptor_model, "net_rule": args.receptor_net_rule, "nt_class_fallback": bool(args.receptor_nt_class_fallback),
                        "coverage": cov.to_dict("records"), "fast_sign_changed_entries": int((rs.fast_sign != np.sign(ctx.c.W.data)).sum()),
                        "table": rs.table_path, "dopamine_lead": ctx.dopamine_lead_info, "gain_classes": brain._receptor_gain(lif)}
        if ctx.receptor_model == "full":                                    # the slow term in force: spec and per-class entry counts
            spec = brain._slow_spec(lif)
            slow_cfg = {"mode": lif.slow_mode, "gain_by_class": brain._slow_gains(lif), "tau_by_class": brain._slow_taus(lif),
                        "active": spec is not None, "entries_by_class": {}}
            for i, name in enumerate(connectome.SLOW_CLASSES[1:], 1):
                m = rs.slow_class == i
                slow_cfg["entries_by_class"][name] = {"entries": int(m.sum()),
                                                      "syn_eq": float((rs.count[m] * np.abs(rs.slow_sign[m])).sum()) if rs.count is not None else None}
            receptor_cfg["slow"] = slow_cfg
            print(f"slow term: mode {lif.slow_mode}, gains {slow_cfg['gain_by_class']}, taus {slow_cfg['tau_by_class']} ms, "
                  f"active {spec is not None}; entries {slow_cfg['entries_by_class']}")
        del rs
    else:
        receptor_cfg = {"model": None}
    config = {"lif": {k: getattr(lif, k) for k in ["std_u", "std_tau", "adapt_jump", "same_type_gain", "input_norm_alpha", "input_norm_ref", "w_syn", "conn_cap"]},
              "receptor": receptor_cfg,
              "path_gain": brain.DEFAULT_PATH_GAIN if lif.path_gain is None else lif.path_gain,
              "type_path_gain": brain.DEFAULT_TYPE_PATH_GAIN if lif.type_path_gain is None else lif.type_path_gain,
              "optic": {"gain_out_mv": op.gain_out_mv, "pair_gain": optic.DEFAULT_PAIR_GAIN if op.pair_gain is None else op.pair_gain},
              "fast": ctx.fast, "backend": "native (cuda_kernels, cuda_graphs, event_driven, warp)" if ctx.native else "eager torch",
              "seeds": ctx.seeds, "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
              "gf_hz": float(body.Flight().gf_hz), "neurons": int(ctx.c.n),
              "cache_dir": str(ctx.cache_dir or connectome.CACHE_DIR),
              "nt_counts": {k: int(v) for k, v in ctx.c.neurons.nt.value_counts().items()}}
    print("LIF:", config["lif"], " gain_out", op.gain_out_mv, " backend:", config["backend"], " fast:", ctx.fast, flush=True)
    for letter, name, fn in select_sections(args.sections):
        print(f"\n=== [{letter or '-'}] {name}", flush=True)
        t0 = time.time()
        try:
            ctx.results[name] = fn(ctx)
        except Exception as e:      # one broken section must not hide the others; it shows as MISSING in the table
            import traceback
            traceback.print_exc()
            ctx.results[name] = {"error": repr(e)}
            for key in REFERENCES:
                if key.split(".")[0] == name or (name == "walk" and key.split(".")[0] in ("walk", "loom", "rotate")):
                    ctx.report(key, None)
        ctx.runtime[name] = time.time() - t0
        print(f"    [{name}: {ctx.runtime[name]:.0f} s]", flush=True)
    total = time.time() - t_all
    print("\n" + summary_table(ctx))
    n_pass = sum(c["status"].startswith("PASS") for c in ctx.checks); n_fail = sum(c["status"] == "FAIL" for c in ctx.checks)
    n_gap = sum(c["status"] == "KNOWN GAP" for c in ctx.checks); n_miss = sum(c["status"] == "MISSING" for c in ctx.checks)
    print(f"\n{n_pass} pass, {n_fail} fail, {n_gap} known gap, {n_miss} missing; runtime {total / 60:.1f} min "
          f"(" + ", ".join(f"{k} {v:.0f} s" for k, v in ctx.runtime.items()) + ")")
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"config": config, "sections": ctx.results, "checks": ctx.checks, "runtime_s": ctx.runtime, "total_runtime_s": total,
                       "date": time.strftime("%Y-%m-%d %H:%M")}, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        print("wrote", args.json)


if __name__ == "__main__":
    main()
