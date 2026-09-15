"""The first instrument (docs/PRESETS_SPEC.md, docs/INSTRUMENTS.md): `sided_turn_afferent`, the presets keyword, the
provenance fields and the cx_wedge threading.

    CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python -m pytest tests/test_instruments.py -q

Everything but the last class runs on a synthetic graph that carries the velocity route with the shipped cache's
routing (AN07B037_a / _b contralateral onto PS196_b, PS196_b contralateral onto GLNO, GLNO contralateral onto PEN;
CB0675 ipsilateral), so the side decision is tested as a property of the graph, not of a constant. The last class
is a CPU smoke of one V-arm `cx_wedge.py` command on the real cache and skips without it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from flyverse import instruments as fi                     # noqa: E402
from flyverse import senses                                # noqa: E402
from flyverse.brain import LIFParams                       # noqa: E402
from flyverse.connectome import CACHE_DIR, Connectome      # noqa: E402
from flyverse.fly import FlyBrain                          # noqa: E402
from flyverse.interp import common                         # noqa: E402

HAVE_CACHE = (CACHE_DIR / "W_post_pre.npz").exists()


def graph(glno_nt="unknown"):
    """The velocity route in 20 cells, wired with the shipped cache's sidedness (counts are small stand-ins)."""
    rows = [
        # bodyId, type, instance, superclass, somaSide, nt
        (1, "AN07B037_a", "AN07B037_a_L", "ascending_neuron", "L", "acetylcholine"),
        (2, "AN07B037_a", "AN07B037_a_L", "ascending_neuron", "L", "acetylcholine"),
        (3, "AN07B037_a", "AN07B037_a_R", "ascending_neuron", "R", "acetylcholine"),
        (4, "AN07B037_a", "AN07B037_a_R", "ascending_neuron", "R", "acetylcholine"),
        (5, "AN07B037_b", "AN07B037_b_L", "ascending_neuron", "L", "acetylcholine"),
        (6, "AN07B037_b", "AN07B037_b_R", "ascending_neuron", "R", "acetylcholine"),
        (7, "CB0675", "CB0675_L", "cb_intrinsic", "L", "acetylcholine"),
        (8, "CB0675", "CB0675_R", "cb_intrinsic", "R", "acetylcholine"),
        (9, "PS196_b", "PS196_b_L", "cb_intrinsic", "L", "acetylcholine"),
        (10, "PS196_b", "PS196_b_R", "cb_intrinsic", "R", "acetylcholine"),
        (11, "GLNO", "GLNO(LAL-NO1)_L", "cb_intrinsic", "L", glno_nt),
        (12, "GLNO", "GLNO(LAL-NO1)_R", "cb_intrinsic", "R", glno_nt),
        (13, "PEN_a(PEN1)", "PEN_a(PB06a)_L3", "cb_intrinsic", "L", "acetylcholine"),
        (14, "PEN_a(PEN1)", "PEN_a(PB06a)_R3", "cb_intrinsic", "R", "acetylcholine"),
        (15, "PEN_b(PEN2)", "PEN_b(PB06b)_L3", "cb_intrinsic", "L", "acetylcholine"),
        (16, "PEN_b(PEN2)", "PEN_b(PB06b)_R3", "cb_intrinsic", "R", "acetylcholine"),
        (17, "EPG", "EPG(PB08)_L1", "cb_intrinsic", "L", "acetylcholine"),
        (18, "DNa02", "DNa02_L", "descending_neuron", "L", "acetylcholine"),
        (19, "DNa02", "DNa02_R", "descending_neuron", "R", "acetylcholine"),
        (20, "ExR6", "ExR6_L", "cb_intrinsic", "L", "glutamate"),
    ]
    n = pd.DataFrame(rows, columns=["bodyId", "type", "instance", "superclass", "somaSide", "nt"])
    n["class"] = ""; n["subclass"] = ""; n["entryNerve"] = None
    N = len(n); w = np.zeros((N, N), np.float32)
    k = {b: i for i, b in enumerate(n.bodyId)}

    def syn(pre, post, count):
        w[k[post], k[pre]] = count
    syn(1, 10, 100); syn(2, 10, 102); syn(3, 9, 107); syn(4, 9, 107); syn(4, 10, 3)      # AN07B037_a: contralateral
    syn(5, 10, 28); syn(6, 9, 24)                                                       # AN07B037_b: contralateral
    syn(7, 9, 51); syn(7, 10, 1); syn(8, 10, 51); syn(8, 9, 4)                          # CB0675: ipsilateral
    syn(9, 12, 832); syn(10, 11, 966); syn(10, 12, 3)                                   # PS196_b -> GLNO: contralateral
    for pen_L, pen_R in ((13, 14), (15, 16)):                                           # GLNO -> PEN: contralateral
        syn(11, pen_R, 250); syn(12, pen_L, 240)
    syn(13, 17, 30); syn(17, 13, 30); syn(20, 17, 60); syn(20, 13, 40); syn(17, 18, 20)
    return Connectome(n, sp.csr_matrix(w), pd.Series(np.arange(N), index=n.bodyId))


# ---------------------------------------------------------------------------------------------- the instrument
class SidedTurnAfferentTests(unittest.TestCase):
    def test_the_side_is_read_from_the_graph(self):
        inst = fi.SidedTurnAfferent(graph())
        r = inst.routing
        for t in ("AN07B037_a", "AN07B037_b"):
            self.assertTrue(r["afferent_to_PS196_b"][t]["contralateral"], t)
            self.assertEqual(r["afferent_to_PS196_b"][t]["LL"]["synapses"], 0.0)
        self.assertEqual(r["afferent_to_PS196_b"]["AN07B037_a"]["LR"]["synapses"], 202.0)
        self.assertEqual(r["afferent_to_PS196_b"]["AN07B037_a"]["RL"]["synapses"], 214.0)
        self.assertTrue(r["PS196_b_to_GLNO"]["contralateral"])
        self.assertTrue(all(b["contralateral"] for b in r["GLNO_to_PEN"].values()))
        self.assertEqual(inst.chain_for_positive_yaw, ["AN07B037_a_L -> PS196_b_R -> GLNO_L -> PEN_R",
                                                       "AN07B037_b_L -> PS196_b_R -> GLNO_L -> PEN_R"])
        # the ipsilateral variant lands one side over, and the record says so
        cb = fi.SidedTurnAfferent(graph(), cells="CB0675")
        self.assertFalse(cb.routing["afferent_to_PS196_b"]["CB0675"]["contralateral"])
        self.assertEqual(cb.chain_for_positive_yaw, ["CB0675_L -> PS196_b_L -> GLNO_R -> PEN_L"])

    def test_rates_side_sign_levels_and_zero(self):
        c = graph()
        for k in fi.SidedTurnAfferent.K_LEVELS:
            inst = fi.SidedTurnAfferent(c, k_hz_per_deg_s=k)
            self.assertTrue(inst.parameters["k_is_declared_level"])
            hz = inst.rates(np.deg2rad(90.0), 1)[0]
            self.assertEqual(hz.shape, (6,))
            np.testing.assert_allclose(hz[inst.side > 0], 90.0 * k)          # a left turn drives the LEFT afferents
            np.testing.assert_allclose(hz[inst.side < 0], 0.0)
            hz = inst.rates(np.deg2rad(-90.0), 1)[0]                          # a right turn the RIGHT ones
            np.testing.assert_allclose(hz[inst.side < 0], 90.0 * k)
            np.testing.assert_allclose(hz[inst.side > 0], 0.0)
        inst = fi.SidedTurnAfferent(c, k_hz_per_deg_s=0.5)
        zero = inst.rates(0.0, 3)
        self.assertEqual(zero.shape, (3, 6)); self.assertTrue((zero == 0.0).all())
        self.assertFalse(np.signbit(zero).any())                              # no -0.0 either
        # the sign flip (the HGV- control) swaps the sides exactly
        flipped = fi.SidedTurnAfferent(c, k_hz_per_deg_s=0.5, sign=-1)
        np.testing.assert_array_equal(flipped.rates(np.deg2rad(60.0), 1), inst.rates(np.deg2rad(-60.0), 1))
        # per-row yaw in a batch, and the ceiling
        hz = inst.rates(np.deg2rad([45.0, -45.0]), 2)
        np.testing.assert_allclose(hz[0][inst.side > 0], 22.5); np.testing.assert_allclose(hz[0][inst.side < 0], 0.0)
        np.testing.assert_allclose(hz[1][inst.side < 0], 22.5); np.testing.assert_allclose(hz[1][inst.side > 0], 0.0)
        capped = fi.SidedTurnAfferent(c, k_hz_per_deg_s=1.0, max_hz=100.0)
        self.assertEqual(capped.rates(np.deg2rad(400.0), 1).max(), 100.0)
        self.assertFalse(fi.SidedTurnAfferent(c, k_hz_per_deg_s=0.3).parameters["k_is_declared_level"])

    def test_describe_is_the_presets_spec_record(self):
        d = fi.SidedTurnAfferent(graph(), k_hz_per_deg_s=0.5, sign=-1).describe()
        for key in ("name", "class", "kind", "law", "parameters", "gap", "source", "removal", "audits", "routing",
                    "chain_for_positive_yaw", "cells", "reads", "writes"):
            self.assertIn(key, d)
        self.assertEqual(d["name"], "sided_turn_afferent"); self.assertEqual(d["kind"], "stop-gap")
        self.assertEqual(d["law"], "unverified")                              # no PS196_b recording exists
        self.assertEqual(d["parameters"]["k_hz_per_deg_s"], 0.5); self.assertEqual(d["parameters"]["sign"], -1)
        self.assertEqual(d["parameters"]["k_levels"], [0.25, 0.5, 1.0])
        self.assertEqual(d["parameters"]["types"], ["AN07B037_a", "AN07B037_b"])
        self.assertEqual(d["cells"], {"L": [1, 2, 5], "R": [3, 4, 6]})
        self.assertTrue(any("compass_velocity_route" in a for a in d["audits"]))
        self.assertIn("PS196_b", d["gap"]); self.assertIn("recording", d["removal"])
        json.dumps(common.to_jsonable(d))                                     # lands in every JSON

    def test_variants_and_refusals(self):
        c = graph()
        self.assertEqual(fi.SidedTurnAfferent(c, cells="CB0675").types, ("CB0675",))
        for variant in ("GNG580", "PS047_b", "all"):                          # types absent from this graph: never assumed
            with self.assertRaises(ValueError):
                fi.SidedTurnAfferent(c, cells=variant)
        with self.assertRaises(ValueError):
            fi.SidedTurnAfferent(c, cells="nonsense")
        for bad in ({"k_hz_per_deg_s": 0.0}, {"k_hz_per_deg_s": -1.0}, {"sign": 2}, {"max_hz": 0.0}):
            with self.assertRaises(ValueError):
                fi.SidedTurnAfferent(c, **bad)
        # a cell without a somaSide cannot be sided
        c2 = graph(); c2.neurons.loc[c2.neurons.bodyId == 1, "somaSide"] = None
        with self.assertRaises(ValueError):
            fi.SidedTurnAfferent(c2)

    def test_parse_instrument_grammar(self):
        c = graph()
        inst = fi.parse_instrument("sided_turn_afferent:k=0.25:sign=-1:cells=CB0675", c)
        self.assertEqual((inst.k, inst.sign, inst.cells), (0.25, -1, "CB0675"))
        self.assertEqual(fi.parse_instrument("sided_turn_afferent", c).k, 0.5)
        for bad in ("", "nope", "sided_turn_afferent:k", "sided_turn_afferent:gain=3", "sided_turn_afferent:cells=XX",
                    "sided_turn_afferent:sign=1.5", "sided_turn_afferent:k=0.5:k=1"):
            with self.assertRaises(ValueError):
                fi.parse_instrument(bad, c)


# ---------------------------------------------------------------------------------------------- the sense token
class ProprioceptionTokenTests(unittest.TestCase):
    def test_turn_afferent_is_an_extra_channel_all_never_selects(self):
        c = graph()
        sense = senses.Proprioception(c, "turn_afferent")
        self.assertEqual(sense.channels, ("turn_afferent",)); self.assertEqual(sense.spec, "turn_afferent")
        self.assertIsInstance(sense.turn_afferent, fi.SidedTurnAfferent)
        out = sense.rates(0.0, 0.0, 0.0, False, yaw_rate=np.deg2rad(90.0), batch=1)
        self.assertEqual([o[0] for o in out], ["turn_afferent"])
        ch, idx, hz = out[0]
        np.testing.assert_array_equal(idx, sense.turn_afferent.idx)
        np.testing.assert_allclose(hz[0][sense.turn_afferent.side > 0], 45.0)
        self.assertEqual(sense.counts()["turn_afferent"]["instrument"], "sided_turn_afferent")
        self.assertNotIn("turn_afferent", senses.Proprioception.parse_flags("all")[0])
        spec = senses.Proprioception.parse_flags("all+turn_afferent+haltere_coriolis")
        self.assertEqual(spec[0], ("chordotonal", "hair_plate", "campaniform", "haltere", "turn_afferent"))
        s2 = senses.Proprioception(c, "all+turn_afferent", turn_afferent=fi.SidedTurnAfferent(c, sign=-1))
        self.assertEqual(s2.spec, "all+turn_afferent"); self.assertEqual(s2.turn_afferent.sign, -1)
        with self.assertRaises(TypeError):
            senses.Proprioception(c, "turn_afferent", turn_afferent=object())


# ---------------------------------------------------------------------------------------------- FlyBrain presets
class PresetTests(unittest.TestCase):
    def test_token_cannot_bypass_raw_or_omit_provenance(self):
        from flyverse.batch_sim import BatchSim
        c = graph()
        with self.assertRaisesRegex(ValueError, "instrumented"):
            BatchSim(c=c, device="cpu", proprioception="all+turn_afferent", preset="raw")
        raw = FlyBrain(c, device="cpu", lif_params=LIFParams(receptor_model=None))
        raw.proprioception_sense = senses.Proprioception(c, "turn_afferent")
        with self.assertRaisesRegex(ValueError, "instrumented"):
            raw.proprioception(0, 0, 0, False, yaw_rate=1)
        with self.assertRaisesRegex(ValueError, "instrumented"):
            raw.instrument_records()
        fb = FlyBrain(c, device="cpu", preset="instrumented", lif_params=LIFParams(receptor_model=None))
        fb.proprioception_sense = senses.Proprioception(c, "turn_afferent")
        fb.proprioception(0, 0, 0, False, yaw_rate=1)
        self.assertEqual([d["name"] for d in fb.instrument_records()], ["sided_turn_afferent"])

    def test_checkpoint_rejects_a_different_afferent_sign_before_loading(self):
        c = graph()
        make = lambda sign: FlyBrain(c, device="cpu", preset="instrumented", lif_params=LIFParams(receptor_model=None),
                                    instruments=[fi.SidedTurnAfferent(c, sign=sign)])
        fb, other = make(1), make(-1)
        state = fb.state_dict()
        with self.assertRaisesRegex(ValueError, "instruments"):
            other.load_state_dict(state)
        make(1).load_state_dict(state)

    def test_wrong_graph_and_describe_only_instruments_are_refused(self):
        c = graph(); reordered = graph()
        reordered.neurons = reordered.neurons.iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "ordering"):
            senses.Proprioception(reordered, "turn_afferent", turn_afferent=fi.SidedTurnAfferent(c))
        class DescriptionOnly:
            name, kind = "unused", "stop-gap"
            def describe(self):
                return dict(name=self.name, kind=self.kind, law="unverified", gap="gap", removal="recording", audits=["audit"])
        with self.assertRaisesRegex(ValueError, "install"):
            FlyBrain(c, device="cpu", preset="instrumented", instruments=[DescriptionOnly()])

    def test_raw_is_the_default_and_refuses_instruments(self):
        c = graph()
        fb = FlyBrain(c, device="cpu", lif_params=LIFParams(receptor_model=None))
        self.assertEqual(fb.preset, "raw"); self.assertEqual(fb.instruments, {}); self.assertEqual(fb.instrument_records(), [])
        self.assertNotIn("proprioception", fb.available_senses)
        with self.assertRaises(ValueError):
            FlyBrain(c, device="cpu", lif_params=LIFParams(receptor_model=None), preset="raw", instruments=[fi.SidedTurnAfferent(c)])
        with self.assertRaises(ValueError):
            FlyBrain(c, device="cpu", lif_params=LIFParams(receptor_model=None), preset="tuned")
        with self.assertRaises(ValueError):
            FlyBrain(c, device="cpu", lif_params=LIFParams(receptor_model=None), preset="instrumented", instruments=[object()])
        prov = common.provenance(c, fb=fb, device="cpu", seeds=[0])
        self.assertEqual(prov["preset"], "raw"); self.assertEqual(prov["instruments"], [])

    def test_instrumented_drives_the_sided_cells_and_lands_in_provenance(self):
        c = graph()
        for sign in (1, -1):
            inst = fi.SidedTurnAfferent(c, k_hz_per_deg_s=0.5, sign=sign)
            fb = FlyBrain(c, device="cpu", lif_params=LIFParams(receptor_model=None), preset="instrumented", instruments=[inst])
            self.assertEqual(fb.preset, "instrumented"); self.assertIn("proprioception", fb.available_senses)
            self.assertIs(fb.proprioception_sense.turn_afferent, inst)
            fb.proprioception(0.0, 0.0, 0.0, False, yaw_rate=np.deg2rad(90.0))       # a left turn
            hz = fb.brain.poisson_p[0, fb.brain._idx(inst.idx)].cpu().numpy() * 1000.0 / fb.brain.p.dt
            driven = inst.side > 0 if sign == 1 else inst.side < 0
            np.testing.assert_allclose(hz[driven], 45.0, rtol=1e-5); np.testing.assert_allclose(hz[~driven], 0.0)
            other = np.setdiff1d(np.arange(c.n), inst.idx)
            self.assertEqual(float(fb.brain.poisson_p[0, fb.brain._idx(other)].abs().sum()), 0.0)   # nothing else is touched
            prov = common.provenance(c, fb=fb, device="cpu", seeds=[0])
            self.assertEqual(prov["preset"], "instrumented")
            self.assertEqual([i["name"] for i in prov["instruments"]], ["sided_turn_afferent"])
            self.assertEqual(prov["instruments"][0]["parameters"]["sign"], sign)
            self.assertEqual(prov["instruments"][0]["law"], "unverified")
            keys = list(prov)
            self.assertLess(keys.index("compiled_connectome"), keys.index("preset"))
            json.dumps(prov)
            fb.step(20.0)                                                                 # and it simulates

    def test_hold_and_relabel_records_verify_what_they_record(self):
        c = graph()
        hold = fi.EdgeHold(r"^(ExR6|ER6|ER4m)$", r"^(PEN_|EPG$)", 0.0, resolved=[{"n_entries": 2}])
        with self.assertRaises(ValueError):                                              # the gain list lacks the hold
            FlyBrain(c, device="cpu", lif_params=LIFParams(receptor_model=None), preset="instrumented", instruments=[hold])
        p = LIFParams(receptor_model=None, type_path_gain=[(r"^(ExR6|ER6|ER4m)$", r"^(PEN_|EPG$)", 0.0)])
        fb = FlyBrain(c, device="cpu", lif_params=p, preset="instrumented", instruments=[hold])
        self.assertEqual(fb.instrument_records()[0]["kind"], "edges")
        self.assertEqual(fb.instrument_records()[0]["name"], "ring_dc_hold")
        relabel = fi.TypeRelabel("GLNO", "glutamate")
        with self.assertRaises(ValueError):                                              # this graph's GLNO is 'unknown'
            FlyBrain(c, device="cpu", lif_params=p, preset="instrumented", instruments=[relabel])
        fb = FlyBrain(graph(glno_nt="glutamate"), device="cpu", lif_params=p, preset="instrumented", instruments=[relabel, hold])
        self.assertEqual([r["name"] for r in fb.instrument_records()], ["glno_sign", "ring_dc_hold"])
        with self.assertRaises(ValueError):                                              # names unique
            FlyBrain(graph(glno_nt="glutamate"), device="cpu", lif_params=p, preset="instrumented", instruments=[hold, hold])


# ---------------------------------------------------------------------------------------------- cx_wedge threading
class CxWedgeTests(unittest.TestCase):
    def setUp(self):
        import importlib
        self.cw = importlib.import_module("cx_wedge")

    def test_preset_resolution_and_turn_window(self):
        cw = self.cw
        self.assertEqual(cw.resolve_preset(None, None), "raw")
        self.assertEqual(cw.resolve_preset(None, ["sided_turn_afferent"]), "instrumented")
        self.assertEqual(cw.resolve_preset("instrumented", None), "instrumented")
        with self.assertRaises(SystemExit):
            cw.resolve_preset("raw", ["sided_turn_afferent"])
        self.assertEqual(cw.parse_turn_window(None), (0.5, 3.5)); self.assertEqual(cw.parse_turn_window("0:1.5"), (0.0, 1.5))
        for bad in ("x", "2:1", "-1:2"):
            with self.assertRaises(SystemExit):
                cw.parse_turn_window(bad)

    def test_build_instruments_is_empty_under_raw_whatever_the_flags(self):
        cw = self.cw
        c = graph(glno_nt="glutamate")
        holds = cw.parse_hold_edges([r"^(ExR6|ER6|ER4m)$:^(PEN_|EPG$)"])
        self.assertEqual(cw.build_instruments(c, [], "raw", hold_edges=holds, nt_override={"GLNO": "glutamate"}), [])
        insts = cw.build_instruments(c, ["sided_turn_afferent:k=1.0"], "instrumented", hold_edges=holds, nt_override={"GLNO": "glutamate"},
                                     edge_gains=cw.parse_edge_gains(["^EPG$:^EPG$:10"]))
        self.assertEqual([i.name for i in insts], ["sided_turn_afferent", "ring_dc_hold", "glno_sign", "edge_gain_0"])
        self.assertEqual(insts[1].resolved["n_entries"], 2); self.assertEqual(insts[3].kind, "edges")
        with self.assertRaises(SystemExit):
            cw.build_instruments(c, ["sided_turn_afferent:cells=XX"], "instrumented")

    def test_bump_follow_and_side_groups(self):
        cw = self.cw
        tt = np.arange(300) * 0.01
        centre = (2.0 + 4.0 * tt) % 16                                     # a bump moving +4 wedges / s in ring order
        conf = np.ones(300, bool)
        m = cw.bump_follow(centre, conf, tt, 0.5, 2.5, +90.0)
        self.assertAlmostEqual(m["bump_follow_wedges_per_s"], 4.0, places=6); self.assertEqual(m["bump_follow_ideal_wedges_per_s"], 4.0)
        self.assertEqual(m["bump_follow_n_frames"], 200); self.assertEqual(m["bump_follow_confined_frac"], 1.0)
        self.assertAlmostEqual(cw.bump_follow(centre, conf, tt, 0.5, 2.5, -90.0)["bump_follow_wedges_per_s"], -4.0, places=6)
        self.assertTrue(np.isnan(cw.bump_follow(centre, conf, tt, 0.5, 2.5, None)["bump_follow_wedges_per_s"]))
        g = cw.side_groups(graph(), {"PEN": ("PEN_a(PEN1)", "PEN_b(PEN2)"), "GLNO": "GLNO", "DNa02": "DNa02", "PS196b": "PS196_b"})
        self.assertEqual({k: len(v) for k, v in g.items()}, {"PEN_L": 2, "PEN_R": 2, "GLNO_L": 1, "GLNO_R": 1, "DNa02_L": 1, "DNa02_R": 1, "PS196b_L": 1, "PS196b_R": 1})


@unittest.skipUnless(HAVE_CACHE, "needs the compiled MaleCNS cache")
class CxWedgeVArmSmoke(unittest.TestCase):
    def test_one_v_arm_command_runs_on_cpu(self):
        """One predeclared V-arm command line (docs/audits/compass_velocity_route.md 2) at --sim 1:1, shortened to
        0.8 s of simulated time, on CPU: the JSON carries the preset, the instrument's describe() and the round-7
        measures, and the afferents fire on the left for a left turn."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "V_s0.json"
            cmd = [sys.executable, str(ROOT / "scripts" / "cx_wedge.py"), "--no-structure", "--sim", "1:1", "--ledger", "--seed", "0",
                   "--arm", "V", "--block", "fam_s0", "--receptor-model", "shipped", "--instrument", "sided_turn_afferent:k=0.5",
                   "--turn", "90", "--turn-window", "0.0:0.3", "--settle-s", "0.2", "--pulse-s", "0.2", "--seconds", "0.4",
                   "--device", "cpu", "--no-graphs", "--out", tmp, "--sim-out", str(out)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8")
            r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=900)
            self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
            row = json.load(open(out, encoding="utf-8"))[0]
            self.assertEqual(row["preset"], "instrumented"); self.assertEqual(row["instruments"], ["sided_turn_afferent"])
            self.assertEqual(row["device"], "cpu"); self.assertEqual(row["arm"], "V")
            prov = row["provenance"]
            import cx_velocity_route as cvr
            # --receptor-model shipped resolves BOTH fields from LIFParams(), overriding the separate
            # CLI net-rule default. The frozen protocol must match the actual simulation path.
            self.assertEqual(row["receptor_net_rule"], cvr.PROTOCOL["receptor_net_rule"])
            self.assertEqual(prov["model"]["lif"], cvr.resolved_lif_by_arm()["V"])
            self.assertEqual(prov["preset"], "instrumented")
            self.assertEqual(prov["instruments"][0]["law"], "unverified")
            self.assertEqual(prov["instruments"][0]["chain_for_positive_yaw"][0], "AN07B037_a_L -> PS196_b_R -> GLNO_L -> PEN_R")
            m = row["metrics"]
            self.assertTrue(m["turn_fed"]); self.assertEqual(m["turn_deg_s"], 90.0)
            self.assertGreater(m["AFF_LR_hz"], 10.0)                       # the left afferents fire for a left turn
            for key in ("PS196b_LR_hz", "GLNO_LR_hz", "PEN_LR_hz", "DNa02_LR_hz", "bump_follow_wedges_per_s", "bump_follow_ideal_wedges_per_s"):
                self.assertIn(key, m)
            self.assertEqual(m["bump_follow_ideal_wedges_per_s"], 4.0)
            self.assertEqual(m["AFF_n_L"], 3); self.assertEqual(m["AFF_n_R"], 3)


if __name__ == "__main__":
    unittest.main()
