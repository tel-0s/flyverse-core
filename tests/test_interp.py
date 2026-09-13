"""Interpretability toolkit (flyverse/interp): the common module's contracts on a synthetic graph, and the package's
stub / implementation hand-off. CPU only; no dataset, no GPU. Each tool's implementer appends a class here
(`DecomposeTests`, `TraceTests`, ...) that exercises the tool on `graph()` or on a Connectome.subset.

    PYTHONIOENCODING=utf-8 python -m unittest discover -s tests -p test_interp.py
"""
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flyverse import connectome as cn
from flyverse.connectome import Connectome, save, load
from flyverse import interp
from flyverse.interp import common
from flyverse.screen import TypeRecorder


def graph() -> Connectome:
    """8 neurons: a photoreceptor (R1-R6, histamine), an optic rate unit (Mi4, GABA), a visual projection cell (LC4, ACh),
    the giant fibre (DNp01, ACh), DNa02 L / R (ACh), a sign-0 GLNO (unknown) and a PEN_a (ACh); W[post, pre] signed counts
    with GLNO's outputs stored as explicit zeros, as in the cache."""
    types = ["R1-R6", "Mi4", "LC4", "DNp01", "DNa02", "DNa02", "GLNO", "PEN_a"]
    nts = ["histamine", "gaba", "acetylcholine", "acetylcholine", "acetylcholine", "acetylcholine", "unknown", "acetylcholine"]
    sc = ["ol_sensory", "ol_intrinsic", "visual_projection", "descending_neuron", "descending_neuron", "descending_neuron",
          "cb_intrinsic", "cb_intrinsic"]
    n = pd.DataFrame({"bodyId": np.array([1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008], dtype=np.int64), "type": types,
                      "instance": [t + "_x" for t in types], "superclass": sc, "class": [""] * 8, "subclass": [""] * 8,
                      "somaSide": ["L", "L", "L", "R", "L", "R", "L", "R"], "nt": nts,
                      "sign": np.array([cn.NT_SIGN[x] for x in nts], dtype=np.float32)})
    cnt = np.zeros((8, 8), np.float32)
    cnt[1, 0] = 40    # R1-R6 -> Mi4
    cnt[2, 1] = 30    # Mi4 -> LC4
    cnt[3, 2] = 120   # LC4 -> DNp01 (above the cap)
    cnt[4, 3] = 10    # DNp01 -> DNa02_L
    cnt[5, 4] = 20    # DNa02_L -> DNa02_R (same type)
    cnt[7, 6] = 200   # GLNO -> PEN_a (sign 0: explicit zero)
    cnt[6, 7] = 50    # PEN_a -> GLNO
    cnt[7, 2] = 5     # LC4 -> PEN_a
    sign = n.sign.to_numpy()
    coo = sp.coo_matrix(cnt)
    W = sp.csr_matrix((coo.data * sign[coo.col], (coo.row, coo.col)), shape=(8, 8), dtype=np.float32)
    W.sort_indices()
    return Connectome(n, W, pd.Series(np.arange(8), index=n.bodyId))


def fake_fb(c, rates, B=1):
    """A FlyBrain stand-in with the attributes Recorder reads (numpy instead of torch tensors)."""
    r = np.asarray(rates, np.float32)
    brain = SimpleNamespace(B=B, t=0.0, rate_np=lambda: r, rate=r, drive=np.full_like(r, 2.0), v=np.full_like(r, -50.0),
                            adapt=np.zeros_like(r), refrac=np.zeros_like(r), spike_counts=np.arange(len(r), dtype=np.float32),
                            p=SimpleNamespace(dt=0.5), device="cpu")
    optic = SimpleNamespace(rate_idx=np.array([1]), delta_rate=np.array([0.25], np.float32), rates=lambda: np.array([0.75], np.float32))
    return SimpleNamespace(brain=brain, optic=optic, c=c, cuda_graphs=False)


class SelectionTests(unittest.TestCase):
    def test_grammar(self):
        c = graph()
        np.testing.assert_array_equal(common.resolve(c, "DNa02"), [4, 5])
        np.testing.assert_array_equal(common.resolve(c, "DNa02&somaSide=R"), [5])
        np.testing.assert_array_equal(common.resolve(c, "~^DN"), [3, 4, 5])
        np.testing.assert_array_equal(common.resolve(c, "type:LC4|PEN_a"), [2, 7])
        np.testing.assert_array_equal(common.resolve(c, "LC4|PEN_a"), [2, 7])
        np.testing.assert_array_equal(common.resolve(c, "module=optic"), [0, 1])
        np.testing.assert_array_equal(common.resolve(c, "module=descending"), [3, 4, 5])
        np.testing.assert_array_equal(common.resolve(c, "nt=unknown"), [6])
        np.testing.assert_array_equal(common.resolve(c, "body:1008|1001"), [0, 7])
        np.testing.assert_array_equal(common.resolve(c, ["GLNO", "PEN_a", "PEN_a"]), [6, 7])
        np.testing.assert_array_equal(common.resolve(c, {"type": "~^DNa"}), [4, 5])
        np.testing.assert_array_equal(common.resolve(c, np.array([5, 2])), [2, 5])
        np.testing.assert_array_equal(common.resolve(c, np.arange(8) >= 6), [6, 7])
        self.assertEqual(len(common.resolve(c, [])), 0)
        with self.assertRaises(ValueError):
            common.resolve(c, "module=nowhere")
        with self.assertRaises(ValueError):
            common.resolve(c, "colour=red")

    def test_population_and_grouping(self):
        c = graph()
        pops = common.populations(c, {"steer": "DNa02", "ring": ["GLNO", "PEN_a"]})
        self.assertEqual([p.label for p in pops], ["steer", "ring"])
        self.assertEqual(pops[0].record()["body_ids"], ["1005", "1006"])
        groups = common.by_type(c, common.resolve(c, "module=descending"), by_side=True)
        self.assertEqual(list(groups), ["DNa02_L", "DNa02_R", "DNp01_R"])
        self.assertEqual(common.body_str(np.array([7, 8])), ["7", "8"])


class EffectiveWeightTests(unittest.TestCase):
    def test_matches_brain_install(self):
        from flyverse.brain import Brain, LIFParams
        c = graph()
        for p in [LIFParams(receptor_model=None, event_driven=False),
                  LIFParams(receptor_model=None, conn_cap=0, same_type_gain=0.3, path_gain=[], type_path_gain=[],
                            input_norm_ref=40, input_norm_alpha=0.7, event_driven=False),
                  LIFParams(receptor_model=None, input_norm_alpha=0, event_driven=False)]:
            ew = common.effective_weights(c, p)
            installed = Brain(c, p, device="cpu")._W_cpu
            np.testing.assert_allclose(ew.A.toarray(), installed.toarray(), rtol=1e-6, atol=1e-7)
        # the type matrix is the mean over post cells of the summed input from the pre type
        M = ew.type_matrix(c, ["DNa02"], ["DNp01", "DNa02"])
        self.assertAlmostEqual(M.loc["DNa02", "DNp01"], float(ew.A[[4, 5]][:, [3]].sum(axis=1).mean()))
        self.assertGreater(abs(M.loc["DNa02", "DNa02"]), 0)

    def test_subset_keeps_reference_normalisation(self):
        from flyverse.brain import LIFParams
        c = graph(); p = LIFParams(receptor_model=None, input_norm_ref=20, event_driven=False)
        full = common.effective_weights(c, p)
        sub = common.effective_weights(c.subset([3, 2, 4]), p)
        np.testing.assert_allclose(sub.A.toarray(), full.A.toarray()[[3, 2, 4]][:, [3, 2, 4]], rtol=1e-6)

    def test_raw_counts_merge_the_sign0_counts(self):
        """`raw_counts` MERGES |W.data| with cache/sign0_counts.npz -- maximum(|W.data|, sign0_counts) -- instead of
        substituting the sign-0 array for the whole vector, which reported 0 synapses for every signed edge and wrote
        `synaptic_pair_count 0` into the Neurome column (docs/INTERP.md 11, defect 1)."""
        from unittest import mock
        c = graph(); W = c.W.tocsr()
        C, ok = common.raw_counts(c, with_sign0=False)
        self.assertFalse(ok)                                   # 'exclude the sign-0 entries': they stay at 0
        np.testing.assert_allclose(C.data, np.abs(W.data))
        self.assertEqual(C[7, 6], 0.0); self.assertEqual(C[3, 2], 120.0)
        s0 = np.zeros(W.nnz, np.float32); s0[W.data == 0] = 200.0        # what the npz holds: non-zero ONLY there
        with mock.patch.object(cn, "sign0_counts", lambda *a, **k: s0):
            C2, ok2 = common.raw_counts(c)
            self.assertEqual(common.raw_counts(c, dtype=np.float64)[0].dtype, np.float64)   # health's exact total
        self.assertTrue(ok2)
        self.assertEqual(C2[7, 6], 200.0)                      # the GLNO -> PEN_a sign-0 entry's raw count, rescued
        self.assertEqual(C2[3, 2], 120.0)                      # and every signed entry keeps its own (read 0 before)
        self.assertAlmostEqual(float(C2.sum()), float(np.abs(W.data).sum()) + 200.0)
        # the five tools that patched this privately now all call the shared accessor and agree entry for entry
        from flyverse.interp import paths as P, trace as tr, health as H, export as ex, decompose as dec
        with mock.patch.object(cn, "sign0_counts", lambda *a, **k: s0):
            for got in (P.raw_counts(c)[0], tr.full_raw_counts(c)[0], H.full_counts(c)[0], ex.raw_counts(c)[0],
                        dec.counts_matrix(c)[0]):
                np.testing.assert_allclose(got.toarray(), C2.toarray())

    def test_links_and_silence(self):
        from flyverse.brain import LIFParams
        c = graph(); p = LIFParams(receptor_model=None, event_driven=False)
        ew = common.effective_weights(c, p)
        counts, sign0_available = common.raw_counts(c, with_sign0=False)
        flags = common.silent_flags(c, np.arange(8), frozen_idx=[1], rates=np.array([[0, 0, 5, 0.1, 0, 0, 0, 3]]))
        df = common.links(c, ew, pre=["GLNO", "LC4", "Mi4"], post=["PEN_a", "DNp01", "LC4"], counts=counts, flags=flags)
        self.assertEqual(list(df.columns[:6]), ["pre_index", "post_index", "body_pre", "body_post", "pre_type", "post_type"])
        glno = df[df.pre_type == "GLNO"].iloc[0]
        self.assertEqual(glno.effective_mv, 0.0); self.assertEqual(glno.sign, 0); self.assertIn("sign0", glno.silent)
        self.assertIn("never_firing", glno.silent)
        self.assertEqual(glno.body_pre, "1007"); self.assertEqual(glno.synaptic_pair_count, 0.0)   # explicit zero without sign0 counts
        mi4 = df[df.pre_type == "Mi4"].iloc[0]
        self.assertIn("frozen", mi4.silent); self.assertIn("pruned", mi4.silent); self.assertLess(mi4.effective_mv, 0)
        lc4 = df[(df.pre_type == "LC4") & (df.post_type == "DNp01")].iloc[0]
        self.assertEqual(lc4.synaptic_pair_count, 120.0); self.assertEqual(lc4.silent, "")
        # the cap and the LC4 -> DNp01 x3 type gain, the VP -> DN x2 path gain, and the fan-in scale are in the entry
        self.assertAlmostEqual(lc4.effective_mv, 60 * 3 * 2 * p.w_syn * ew.scale[3], places=5)
        self.assertEqual(lc4.sign_rule, "nt_sign")
        self.assertEqual(list(common.unit_kinds(c)), ["photoreceptor", "graded", "spiking", "spiking", "spiking", "spiking", "spiking", "spiking"])
        self.assertEqual(list(common.unit_kinds(c, fake_fb(c, np.zeros(8))))[1], "graded")
        # with no rollout the never_firing question was not asked: the flag is False, and no structural row claims it
        # (it was NaN, which every bool(flag) reader turned into True -- docs/INTERP.md 11, defect 2)
        structural = common.silent_flags(c, np.arange(8), frozen_idx=[1])
        self.assertEqual(structural.never_firing.dtype, bool); self.assertFalse(structural.never_firing.any())
        ds = common.links(c, ew, pre=["GLNO", "LC4", "Mi4"], post=["PEN_a", "DNp01", "LC4"], counts=counts, flags=structural)
        self.assertEqual(set(ds.silent), {"", "sign0", "frozen|pruned"})
        self.assertNotIn("never_firing", "|".join(ds.silent))


class RecordingTests(unittest.TestCase):
    def test_capture_pool_window_roundtrip(self):
        c = graph()
        rec = common.Recorder(c, ["DNa02", "Mi4", "LC4"], quantities=("rate_hz", "drive_mv", "optic_dr", "refrac", "spike_count"))
        frames = [np.array([0, 0, 10, 0, 4, 8, 0, 0]), np.array([0, 0, 12, 0, 6, 8, 0, 0]), np.array([0, 0, 14, 0, 8, 8, 0, 0])]
        for k, r in enumerate(frames):
            fb = fake_fb(c, r); fb.brain.t = 10.0 * k
            rec.capture(fb, motor=SimpleNamespace(gf=1.0, power=np.float32(2.0), lh_odour={"apple": 3.0}, time_ms=0.0))
        R = rec.finish(meta={"protocol": "toy"})
        np.testing.assert_array_equal(R.idx, [1, 2, 4, 5])
        self.assertEqual(R.quantities["rate_hz"].shape, (3, 4))
        keys, pooled = R.per_type("rate_hz")
        self.assertEqual(list(keys), ["DNa02", "LC4", "Mi4"])
        np.testing.assert_allclose(pooled[:, 0], [6, 7, 8])                    # DNa02 L / R mean
        keys_s, pooled_s = R.per_type("rate_hz", by_side=True)
        self.assertIn("DNa02_L", keys_s)
        self.assertTrue(np.isnan(R.quantities["optic_dr"][0, 1]) and R.quantities["optic_dr"][0, 0] == 0.25)
        self.assertEqual(R.motor["gf"].shape, (3,)); self.assertIn("lh_odour.apple", R.motor)
        # TypeRecorder from screen.py is what pools
        tr = R.recorder(); self.assertIsInstance(tr, TypeRecorder)
        w = R.window(0.01, 0.03); self.assertEqual(w.n_frames, 2)
        with tempfile.TemporaryDirectory() as d:
            path = R.save(Path(d) / "rec")
            back = common.Recording.load(path)
        np.testing.assert_array_equal(back.quantities["rate_hz"], R.quantities["rate_hz"])
        self.assertEqual(back.meta["protocol"], "toy"); self.assertEqual(list(back.types), ["Mi4", "LC4", "DNa02", "DNa02"])
        with self.assertRaises(ValueError):
            common.Recorder(c, "DNa02", quantities=("voltage",))

    def test_batched_rows(self):
        c = graph()
        rec = common.Recorder(c, "DNa02")
        r = np.array([[0, 0, 0, 0, 1, 2, 0, 0], [0, 0, 0, 0, 3, 4, 0, 0]], np.float32)
        rec.capture(fake_fb(c, r, B=2))
        R = rec.finish()
        self.assertTrue(R.batched); self.assertEqual(R.quantities["rate_hz"].shape, (1, 2, 2))
        np.testing.assert_array_equal(R.row(1).quantities["rate_hz"], [[3, 4]])
        with self.assertRaises(ValueError):
            R.per_type()
        n = {"k": 0}
        def step():
            n["k"] += 1
            return fake_fb(c, r, B=2)
        common.record_frames(step, rec, frames=4, every=2)
        self.assertEqual(n["k"], 4); self.assertEqual(rec.finish().n_frames, 3)


class NullHelperTests(unittest.TestCase):
    def test_compare_reproduces_object_sweep_rule(self):
        # LPLC2 sign-abs, docs/audits/object_sweep.md 8.4: z +5.4, Welch +8.6, U 25, p 0.0079
        stim = [0.369, 0.370, 0.326, 0.392, 0.299]; null = [0.124, 0.081, 0.178, 0.137, 0.174]
        r = common.compare(stim, null)
        self.assertAlmostEqual(r["z"], 5.4, delta=0.15); self.assertAlmostEqual(r["welch"], 8.6, delta=0.2)
        self.assertEqual(r["U"], 25.0); self.assertAlmostEqual(r["p"], 0.0079, places=3); self.assertEqual(r["verdict"], "result")
        # LC11 off: z -0.1, U 12, p 1.0 -> null
        r2 = common.compare([0.081, 0.118, 0.040, 0.058, 0.032], [0.080, 0.067, 0.026, 0.097, 0.071])
        self.assertEqual(r2["verdict"], "null"); self.assertAlmostEqual(r2["z"], -0.1, delta=0.1); self.assertEqual(r2["U"], 12.0)
        # two runs are never a result, whatever the numbers
        self.assertEqual(common.compare([10, 10], [0, 0])["verdict"], "underpowered")
        self.assertEqual(common.replicate_seeds(3, 4), [4, 5, 6])
        a = common.ArmStats.of([1, np.nan, 3]); self.assertEqual(a.n, 2); self.assertEqual(a.mean, 2.0)
        self.assertAlmostEqual(r["p_floor"], 2 / 252); self.assertEqual(r["alpha"], 0.05)

    def test_the_p_floor_is_part_of_the_verdict(self):
        """MIN_REPLICATES stays 3 (the scatter rule) but a difference is CALLED only where the exact rank test can
        reach alpha: compare reports `p_floor` and says 'underpowered' while it exceeds alpha (docs/INTERP.md 11,
        defect 3). 3 v 3 floors at 0.10, 4 v 4 at 0.029, 5 v 5 at 0.0079."""
        self.assertEqual(common.MIN_REPLICATES, 3)
        self.assertAlmostEqual(common.p_floor(3, 3), 0.1); self.assertAlmostEqual(common.p_floor(4, 4), 2 / 70)
        self.assertAlmostEqual(common.p_floor(5, 5), 2 / 252); self.assertTrue(np.isnan(common.p_floor(0, 3)))
        r3 = common.compare([10.0, 10.1, 9.9], [0.0, 0.1, -0.1])          # a huge separation over three runs
        self.assertGreater(r3["z"], 3); self.assertAlmostEqual(r3["p"], 0.1)
        self.assertAlmostEqual(r3["p_floor"], 0.1); self.assertEqual(r3["verdict"], "underpowered")
        r4 = common.compare([10.0, 10.1, 9.9, 10.2], [0.0, 0.1, -0.1, 0.2])
        self.assertAlmostEqual(r4["p_floor"], 2 / 70); self.assertEqual(r4["verdict"], "result")

    def test_a_deterministic_null_is_undetermined(self):
        """SD(null) == 0 (bit-identical draws) leaves z undefined: the verdict is 'undetermined' and the magnitude is
        `diff` -- not a z of NaN ('null' on +83 Hz), not one of 1e41 on a near-zero group, and not a per-tool
        override (docs/INTERP.md 11, defect 4)."""
        r = common.compare([83.0, 83.0, 83.0, 83.0], [0.0, 0.0, 0.0, 0.0])
        self.assertTrue(np.isnan(r["z"])); self.assertTrue(r["null_sd_zero"])
        self.assertEqual(r["verdict"], "undetermined"); self.assertEqual(r["diff"], 83.0)
        # a null that is deterministic only to float noise is the same case, not a z of 1e41
        tiny = common.compare([1e-20, 1.1e-20, 0.9e-20, 1e-20], [0.0, 1e-40, 0.0, 0.0])
        self.assertEqual(tiny["verdict"], "undetermined")
        # identical arms: there is nothing to call, deterministic or not
        self.assertEqual(common.compare([5.0] * 4, [5.0] * 4)["verdict"], "null")
        # and the run count still comes first
        self.assertEqual(common.compare([83.0] * 3, [0.0] * 3)["verdict"], "underpowered")


class ResultSchemaTests(unittest.TestCase):
    def test_provenance_fingerprint_and_roundtrip(self):
        from flyverse.brain import LIFParams
        c = graph()
        with tempfile.TemporaryDirectory() as d:
            save(c, Path(d)); loaded = load(Path(d))
            fp = common.connectome_fingerprint(loaded, cache_dir=d)
            import hashlib
            with np.load(Path(d) / "W_post_pre.npz") as z:
                self.assertEqual(fp["md5_data"], hashlib.md5(z["data"].tobytes()).hexdigest())
            self.assertEqual(fp["nnz"], 8); self.assertEqual(fp["n_neurons"], 8); self.assertEqual(fp["sum_abs_W"], 275.0)
            self.assertEqual(fp["cache_dir"], d); self.assertIsNone(fp["subset"])
            self.assertIn("TmY14", fp["type_nt_override"])
        self.assertEqual(common.connectome_fingerprint(c.subset([1, 2]))["subset"], {"n": 2, "of": 8})
        prov = common.provenance(c, LIFParams(receptor_model=None, w_syn=0.3), fb=fake_fb(c, np.zeros(8)), seeds=[0, 1],
                                 stimulus={"protocol": "toy", "params": {"hz": 1}, "control": "none"})
        for k in common.REQUIRED_PROVENANCE:
            self.assertIn(k, prov)
        self.assertEqual(len(prov["dataset_release"]["files"]), 4)
        self.assertTrue(all(len(f["sha256"]) == 64 for f in prov["dataset_release"]["files"]))
        self.assertEqual(prov["model"]["lif"]["w_syn"], 0.3)
        self.assertEqual(prov["model"]["lif"]["type_path_gain"], [("^(LC4|LPLC2)$", "^DNp01$", 3.0)])
        self.assertEqual(prov["model"]["body"]["gf_hz"], 33.0)
        self.assertEqual(prov["model"]["optic"]["gain_out_mv"], 100.0); self.assertIn("pair_gain", prov["model"]["optic"])
        self.assertEqual(prov["execution"]["device"], "cpu"); self.assertEqual(prov["execution"]["dt"]["lif_ms"], 0.5)
        self.assertEqual(prov["execution"]["replicate_unit"], "runs")
        self.assertIn(prov["flyverse_commit"]["commit"] == "unknown" or len(prov["flyverse_commit"]["commit"]) == 40, [True])
        # the code identity: git when git can, the export's source fingerprint when it cannot (a cluster copy has no
        # .git), 'unknown' only as the last resort -- docs/INTERP.md 11, defect 8
        sf = prov["source_fingerprint"]
        if prov["flyverse_commit"]["commit"] == "unknown":
            self.assertTrue(sf["computed"])
        else:
            self.assertFalse(sf["computed"]); self.assertEqual(sf["commit"], prov["flyverse_commit"]["commit"])
        from unittest import mock
        with mock.patch.object(common, "git_state", lambda: {"commit": "unknown", "dirty": False, "modified_files": []}):
            cluster = common.provenance(c, fb=fake_fb(c, np.zeros(8)))["source_fingerprint"]
        self.assertTrue(cluster["computed"]); self.assertGreater(cluster["n_files"], 20)
        self.assertIn("flyverse/interp/common.py", cluster["files"])
        self.assertIn("flyverse/interp/common.py", cluster["files_loaded"])
        from flyverse.interp import export as ex
        self.assertEqual(ex.match_sources(cluster)["differ"], [])       # it matches this checkout, by content
        res = common.Result.new("paths", prov)
        res.add_population(common.population(c, "PEN_a", "target"), unit_kind="spiking")
        res.add_table("contributions", pd.DataFrame([{k: (0 if k in ("value", "synaptic_pair_count") else "x") for k in common.EXPORT_TABLES["contributions"]}]))
        res.summary = {"n": np.int64(3), "z": np.float32(1.5), "arr": np.arange(2)}
        self.assertEqual(res.check(), [])
        self.assertEqual(res.validation["name"], common.VALIDATION["paths"]["name"]); self.assertEqual(res.validation["status"], "not run")
        with tempfile.TemporaryDirectory() as d:
            path = res.save(Path(d) / "r.json")
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            back = common.Result.load(path)
        self.assertEqual(raw["schema"], common.SCHEMA); self.assertEqual(raw["summary"], {"n": 3, "z": 1.5, "arr": [0, 1]})
        self.assertTrue(back.run_id.startswith("paths-")); self.assertEqual(back.populations[0]["body_ids"], ["1008"])
        self.assertEqual(back.table("contributions").shape[0], 1)
        bad = common.Result.new("paths", {"model": {}})
        bad.add_table("sensitivity", [{"lesion_id": "x"}])
        problems = bad.check()
        self.assertTrue(any("execution" in p for p in problems) and any("sensitivity" in p for p in problems))
        with self.assertRaises(ValueError):
            common.Result.new("nonsense", prov)

    def test_cli_helpers(self):
        import argparse
        ap = argparse.ArgumentParser(); common.add_common_args(ap)
        args = ap.parse_args(["--receptor-model", "off", "--lif", "w_syn=0.3", "--lif", "path_gain=[]", "--optic", "gain_fb=0", "--replicates", "5"])
        lif, op = common.params_from_args(args)
        self.assertIsNone(lif.receptor_model); self.assertEqual(lif.w_syn, 0.3); self.assertEqual(lif.path_gain, []); self.assertEqual(op.gain_fb, 0)
        self.assertEqual(args.replicates, 5)
        self.assertEqual(common.parse_kv(["a=None", "b=x", "c={\"k\": 1}"]), {"a": None, "b": "x", "c": {"k": 1}})
        self.assertEqual(common.to_jsonable({"p": Path("x"), "f": float("nan"), "t": (1, 2)}), {"p": "x", "f": None, "t": [1, 2]})
        s = common.print_table(pd.DataFrame({"a": [1.23456]}))
        self.assertIn("+1.235", s)


class StubTests(unittest.TestCase):
    def test_the_two_namespaces_are_separate(self):
        """`flyverse.interp.<tool>` is the MODULE and `flyverse.interp.tools.<tool>` / `interp.tool(<tool>)` is the
        FUNCTION, whatever has been imported already (docs/INTERP.md 11, defect 6: the attribute used to be the
        function until the submodule was imported and the module afterwards, so this class passed or failed on
        test order)."""
        import importlib
        from types import ModuleType
        for name in common.TOOLS:
            importlib.import_module(f"flyverse.interp.{name}")          # the shadowing import, done first on purpose
        for name in common.TOOLS:
            self.assertIsInstance(getattr(interp, name), ModuleType, f"interp.{name} is not the module")
            fn = interp.tool(name)
            self.assertIs(fn, getattr(interp.tools, name))
            self.assertTrue(callable(fn) and not isinstance(fn, ModuleType))
            self.assertIs(fn, getattr(getattr(interp, name), name))     # the implementation, not the stub
        self.assertEqual(sorted(dir(interp.tools)), sorted(common.TOOLS))
        with self.assertRaises(ValueError):
            interp.tool("nonesuch")
        with self.assertRaises(AttributeError):
            interp.tools.nonesuch
        # the stub is what the function namespace returns while a module is missing; it names the file to write
        self.assertTrue(inspect.getdoc(interp.stubs.paths))
        self.assertIs(interp._implementation("no_such_tool", "no_such_tool", interp.stubs.paths), interp.stubs.paths)

    def test_stubs_import_and_signatures(self):
        for name in common.TOOLS:
            fn = interp.tool(name)
            self.assertTrue(callable(fn))
            self.assertTrue(inspect.getdoc(getattr(interp.stubs, name)))
            sig = inspect.signature(getattr(interp.stubs, name))
            real = inspect.signature(fn)
            # an implementation keeps every parameter of the contract (it may add keyword-only ones with defaults)
            self.assertTrue(set(sig.parameters) <= set(real.parameters), f"{name}: {set(sig.parameters) - set(real.parameters)} dropped")
            for extra in set(real.parameters) - set(sig.parameters):
                self.assertIsNot(real.parameters[extra].default, inspect.Parameter.empty, f"{name}: new parameter {extra} needs a default")
        try:
            interp.stubs.paths(None, "EPG", "PEN_a")
        except NotImplementedError as e:
            self.assertIn("flyverse/interp/paths.py", str(e))
        self.assertTrue(callable(interp.HealthReadout))
        with self.assertRaises(AttributeError):
            interp.nonexistent_tool
        self.assertEqual(set(common.VALIDATION), set(common.TOOLS))
        for v in common.VALIDATION.values():
            self.assertTrue(v["name"] and v["reference"] and v["source"])


class HealthTests(unittest.TestCase):
    """flyverse/interp/health.py on graph(): the per-type operating-point columns, the structural-only mode, the
    Neurome rows, replicate / arm helpers and the NTSource readout (checked through FlyBrain.neurotransmitters, the
    observatory's own hook)."""

    def _recording(self, c, frames=4):
        from flyverse.interp import health as H
        rec = common.Recorder(c, np.arange(8), quantities=("rate_hz", "v_mv", "adapt_mv", "refrac", "spike_count"))
        #           R1-R6 Mi4  LC4  DNp01 DNa02L DNa02R GLNO PEN_a
        rate = np.array([0, 10, 200, 3.0, 4.0, 0.0, 0.0, 30.0], np.float32)
        v = np.array([-52, -52, -47, -45.5, -50, -52, -52, -48], np.float32)
        adapt = np.array([0, 0, 3.5, 0, 0, 0, 0, 0.7], np.float32)
        for k in range(frames):
            fb = fake_fb(c, rate)
            fb.brain.t = 10.0 * (k + 1); fb.brain.v = v; fb.brain.adapt = adapt
            fb.brain.refrac = np.array([0, 0, 1 if k % 2 else 0, 0, 0, 0, 0, 0], np.float32)   # LC4 refractory every other frame
            fb.brain.spike_counts = np.array([0, 0, 2 * k, 0, 0, 0, 0, 0.3 * k], np.float32)
            rec.capture(fb)
        return rec.finish(meta={"protocol": "toy", "seed": 7}), H

    def _counts(self, c):
        counts = abs(c.W).tocsr().copy()
        counts[7, 6] = 200.0                       # the sign-0 GLNO -> PEN_a entry's raw count (cache/sign0_counts.npz in the real graph)
        return counts.tocsr()

    def test_per_type_operating_points(self):
        from flyverse.brain import LIFParams
        c = graph(); R, H = self._recording(c)
        p = LIFParams(receptor_model=None, event_driven=False)
        res = H.health(R, c=c, params=p, counts=self._counts(c), fb=fake_fb(c, np.zeros(8)))
        self.assertEqual(res.tool, "health"); self.assertEqual(res.check(), [])
        t = res.table("per_type").set_index("group")
        self.assertEqual(list(t.index), ["DNa02", "DNp01", "GLNO", "LC4", "Mi4", "PEN_a", "R1-R6"])
        lc4 = t.loc["LC4"]
        self.assertAlmostEqual(lc4.refractory_load, 200 * 2.2 / 1000, places=6)          # 0.44: refractory-limited
        self.assertIn("refractory_limited", lc4.state_flags); self.assertIn("high_rate", lc4.state_flags)
        self.assertAlmostEqual(lc4.refrac_measured, 0.5); self.assertAlmostEqual(lc4.adapt_load, 0.5)
        self.assertAlmostEqual(lc4.spike_rate_hz, 2 * 3 / 0.03, places=3)                 # 6 spikes in 30 ms
        self.assertAlmostEqual(lc4.v_margin_mv, 2.0)                                         # non-refractory frames only
        self.assertAlmostEqual(lc4.ei_balance, -1.0)                                         # Mi4 (GABA, 10 Hz) is its only recorded input
        self.assertLess(lc4.i_in_mv_s, 0); self.assertEqual(lc4.e_in_mv_s, 0)
        self.assertEqual(t.loc["DNa02"].silent_frac, 0.5); self.assertEqual(t.loc["DNa02"].n_cells, 2)
        self.assertIn("mostly_silent", t.loc["DNa02"].state_flags)
        self.assertEqual(t.loc["DNp01"].at_threshold_frac, 1.0); self.assertIn("at_threshold", t.loc["DNp01"].state_flags)
        self.assertAlmostEqual(t.loc["DNp01"].ei_balance, 1.0)                              # LC4 (ACh) at 200 Hz
        self.assertAlmostEqual(t.loc["DNp01"].e_in_mv_s, 60 * 3 * 2 * p.w_syn * float(t.loc["DNp01"].fanin_scale_med) * 200, places=3)
        self.assertEqual(t.loc["GLNO"].state_flags, "silent|sign0_output"); self.assertEqual(t.loc["GLNO"].sign0_out_share, 1.0)
        self.assertAlmostEqual(t.loc["PEN_a"].sign0_in_share, 200 / 205)                     # GLNO's 200 of PEN_a's 205 raw input synapses
        self.assertIn("sign0_input", t.loc["PEN_a"].state_flags)
        self.assertEqual(t.loc["Mi4"].unit_kind, "graded"); self.assertEqual(t.loc["LC4"].frozen_in_share, 1.0)
        self.assertEqual(t.loc["Mi4"].frozen_out_share, 1.0)
        self.assertEqual(res.summary["sign0"]["n_sign0_cells"], 1); self.assertEqual(res.summary["sign0"]["n_sign0_presynaptic"], 1)
        self.assertAlmostEqual(res.summary["sign0"]["sign0_synapse_share"], 200 / 475)
        self.assertEqual(res.summary["lif"]["t_ref_ms"], 2.2); self.assertEqual(res.summary["flagged"]["LC4"], lc4.state_flags)
        rb = res.table("readout_per_body")
        self.assertEqual(list(rb.columns), common.EXPORT_TABLES["readout_per_body"])
        lc4_rows = rb[rb.bodyId == "1003"].set_index("quantity")
        self.assertAlmostEqual(lc4_rows.loc["refractory_load"].stimulus_value, 0.44, places=6)
        self.assertEqual(lc4_rows.loc["rate_hz"].unit, "Hz"); self.assertEqual(res.populations[0]["body_ids"][0], "1001")
        # window and thresholds
        w = H.health(R, c=c, params=p, counts=self._counts(c), window=(0.02, 0.04), thresholds={"high_rate_hz": 500})
        self.assertEqual(w.summary["n_frames"], 2); self.assertNotIn("high_rate", w.table("per_type").set_index("group").loc["LC4"].state_flags)
        # custom groups, a presynaptic recording, and dynamic-only (no Connectome) use
        g = H.health(R, c=c, params=p, by={"steer": "DNa02", "ring": ["GLNO", "PEN_a"]}, presyn=R, counts=self._counts(c))
        self.assertEqual(list(g.table("per_type").group), ["ring", "steer"]); self.assertEqual(g.summary["presyn"]["source"], "presyn recording")
        d = H.health(R, params=p)
        self.assertEqual(d.table("per_type").set_index("group").loc["LC4"].refractory_load, lc4.refractory_load)
        self.assertNotIn("fanin_scale_med", d.table("per_type").columns); self.assertIn("execution.device", " ".join(d.check()))
        with self.assertRaises(ValueError):
            H.health(None)
        with self.assertRaises(ValueError):
            H.health(R, c=c.subset([2, 3]), params=p)

    def test_full_counts_is_abs_w_plus_sign0(self):
        from flyverse.interp import health as H
        c = graph()
        counts, avail = H.full_counts(c)                       # graph() has no sign0_counts cache: |W| with the sign-0 zero kept
        self.assertFalse(avail); self.assertEqual(counts.dtype, np.float64)
        np.testing.assert_array_equal(counts.toarray(), np.abs(c.W.toarray()))
        self.assertEqual(counts.nnz, c.W.nnz); self.assertEqual(counts[7, 6], 0.0)
        s0 = H.sign0_summary(c, self._counts(c), c.neurons["sign"].to_numpy() == 0, False)
        self.assertEqual((s0["n_sign0_cells"], s0["n_sign0_presynaptic"], s0["sign0_synapses"]), (1, 1, 200.0))

    def test_structure_only_reproduces_group_shares(self):
        from flyverse.brain import LIFParams
        c = graph(); from flyverse.interp import health as H
        p = LIFParams(receptor_model=None, event_driven=False)
        res = H.health(None, c=c, params=p, by="superclass", counts=self._counts(c))
        t = res.table("per_type").set_index("group")
        self.assertAlmostEqual(t.loc["cb_intrinsic"].sign0_out_share, 200 / 250)   # GLNO -> PEN 200 of the 250 output synapses of GLNO + PEN_a
        self.assertAlmostEqual(t.loc["cb_intrinsic"].sign0_in_share, 200 / 255)    # of 255 synapses onto GLNO + PEN_a
        self.assertEqual(t.loc["cb_intrinsic"].n_sign0_pre, 1); self.assertNotIn("rate_mean", t.columns)
        self.assertEqual(res.summary["n_frames"], 0); self.assertEqual(res.check(), [])
        m = H.health(None, c=c, params=p, by="module").table("per_type").set_index("group")
        self.assertEqual(m.loc["optic"].frozen_out_share, 30 / 70)                 # Mi4's 30 of the optic module's 70 output synapses
        with self.assertRaises(ValueError):
            H.health(None, c=c, params=p, by="colour")

    def test_replicates_and_arms(self):
        from flyverse.brain import LIFParams
        from flyverse.interp import health as H
        c = graph(); p = LIFParams(receptor_model=None, event_driven=False)
        runs = [H.health(self._recording(c)[0], c=c, params=p, counts=self._counts(c)) for _ in range(3)]
        rt = H.replicate_table(runs).set_index(["group", "stat"])
        self.assertEqual(rt.loc[("LC4", "refractory_load")].n, 3); self.assertAlmostEqual(rt.loc[("LC4", "refractory_load")].sd, 0.0, places=12)
        cmp = H.compare_arms(runs, runs).set_index(["group", "stat"])
        # an arm against itself: no difference at all, and at 3 v 3 runs the exact U cannot reach alpha either
        # (common.compare reports p_floor 0.10 and says so) -- neither reading is a result
        self.assertEqual(cmp.loc[("LC4", "rate_mean")].verdict, "underpowered"); self.assertEqual(cmp.loc[("LC4", "rate_mean")]["diff"], 0.0)
        self.assertEqual(H.compare_arms(runs[:2], runs).set_index(["group", "stat"]).loc[("LC4", "rate_mean")].verdict, "underpowered")
        rec = self._recording(c)[0]
        with tempfile.TemporaryDirectory() as d:
            path = rec.save(Path(d) / "r0")
            back = common.Recording.load(path)
        self.assertIsNone(H.params_from_meta(back.meta)); self.assertEqual(H.resolve_params(None, back).t_ref, 2.2)
        prov = common.provenance(c, LIFParams(receptor_model=None, t_ref=3.0, event_driven=False))
        self.assertEqual(H.params_from_meta({"provenance": prov}).t_ref, 3.0)
        self.assertEqual(H.params_from_meta({"provenance": prov}).type_path_gain, [("^(LC4|LPLC2)$", "^DNp01$", 3.0)])
        np.testing.assert_array_equal(H.presynaptic_indices(c, [7]), [2, 6]); np.testing.assert_array_equal(H.presynaptic_indices(c, [7], exclude=[6]), [2])

    def test_readout_source_through_the_observatory_hook(self):
        from flyverse import FlyBrain, NTSnapshot
        from flyverse.interp import HealthReadout
        from flyverse.interp import health as H
        self.assertIs(HealthReadout, H.HealthReadout)
        from test_control import graph as control_graph
        fb = FlyBrain(control_graph(), device="cpu", batch=2, seed=1)
        fb.nt_source = HealthReadout(fb)
        fb.stimulate([0], 200.0, 20.0); fb.step(10.0)
        snap = fb.neurotransmitters(batch_index=1)
        self.assertIsInstance(snap, NTSnapshot)
        self.assertEqual([ch.name for ch in snap.channels], ["rate_hz", "v_margin_mv", "adapt_mv", "refractory", "fanin_scale"])
        self.assertEqual([ch.unit for ch in snap.channels], ["Hz", "mV", "mV", "0/1", "factor"])
        self.assertEqual((snap.channels[1].display_min, snap.channels[1].display_max), (0.0, 7.0))
        a = snap.aligned(fb.c.neurons.bodyId.to_numpy())
        self.assertEqual(a.shape, (6, 5)); self.assertTrue(np.isfinite(a).all()); self.assertFalse(snap.levels.flags.writeable)
        self.assertEqual(snap.time_ms, 10.0)
        self.assertTrue(((a[:, 3] == 0) | (a[:, 3] == 1)).all()); self.assertTrue(((a[:, 4] > 0) & (a[:, 4] <= 1)).all())
        self.assertTrue((a[:, 1] >= -1e-6).all())            # v never above threshold between steps
        self.assertGreater(a[0, 0], 0)                       # the stimulated ORN fires
        with self.assertRaises(IndexError):
            fb.nt_source.readout(batch_index=2)
        with self.assertRaises(ValueError):
            HealthReadout(fb, channels=("rate_hz", "dopamine"))
        # graded units are NaN on the LIF channels and carry the optic rate
        c = graph(); ffb = fake_fb(c, np.arange(8, dtype=np.float32))
        src = HealthReadout(ffb, channels=("rate_hz", "optic_rate", "drive_mv"))
        s = src.readout()
        row = s.aligned(np.array([1002, 1003]))
        self.assertTrue(np.isnan(row[0, 0]) and np.isnan(row[0, 2])); self.assertEqual(row[0, 1], 0.75)
        self.assertEqual(row[1, 0], 2.0); self.assertTrue(np.isnan(row[1, 1])); self.assertEqual(row[1, 2], 2.0)

    def test_cli_module(self):
        import importlib
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        cli = importlib.import_module("interp_health")
        self.assertEqual(cli.parse_window("5.5,8"), (5.5, 8.0)); self.assertIsNone(cli.parse_window(None))
        self.assertEqual(len(cli.compass_type_path_gain(2.0, 15.0)), 7)
        self.assertEqual(cli.compass_type_path_gain(2.0, 15.0)[5], (r"^Delta7$", r"^EPG$", 15.0))
        c = graph(); R, H = self._recording(c)
        with tempfile.TemporaryDirectory() as d:
            R.save(Path(d) / "toy_r0"); R.save(Path(d) / "toy_r1"); R.save(Path(d) / "toy_r2")
            from flyverse.brain import LIFParams
            out = cli.analyse_runs(sorted(str(p)[:-4] for p in Path(d).glob("toy_r*.npz")), c=c, params=LIFParams(receptor_model=None, event_driven=False),
                                   window=None, by="type", groups=None, counts=self._counts(c))
            self.assertEqual(out.replicates["n"], 3); self.assertEqual(out.replicates["unit"], "runs")
            rt = out.table("replicates").set_index(["group", "stat"])
            self.assertEqual(rt.loc[("LC4", "refractory_load")].n, 3)
            self.assertEqual(out.table("readout_per_body").iloc[0].n_trials, 3)
            path = out.save(Path(d) / "health.json")
            self.assertEqual(common.Result.load(path).tool, "health")


class TraceTests(unittest.TestCase):
    """flyverse/interp/trace.py on graph(): the type-level depth graph, the four statistics, the arm comparison, the
    first-lost rules, the lost-stage input table, the accumulator that writes the recordings, and the CLI parser."""

    @staticmethod
    def _rec(c, values, arm, seed, protocol="object", columns=None):
        v = {k: np.asarray(x, np.float32)[None] for k, x in values.items()}
        meta = {"protocol": protocol, "arm": arm, "seed": seed, "window_s": [3.0, 15.0], "accumulated": True, "frames": 1200,
                "default_quantity": {"spiking": "drive_mv" if protocol == "object" else "rate_hz"},
                "provenance": common.provenance(c, fb=fake_fb(c, np.zeros(8)), seeds=[seed], stimulus={"protocol": protocol, "params": {}, "control": "ctrl"})}
        if columns is not None:
            meta.update(columns)
        r = common.Recording(np.array([15000.0]), np.arange(8), c.neurons.bodyId.to_numpy(), c.neurons.type.fillna("").to_numpy().astype(str), v, {}, meta)
        r.series = None; r.path = f"mem:{arm}_r{seed}"
        return r

    def _arms(self, c, n=4, signal=True):
        """n stimulus / control / null recordings (4: the exact U p of 3 v 3 runs floors at 0.10, so no 'result' is possible
        below four runs per arm -- trace.p_floor): Mi4 (graded) and LC4 (spiking, drive) carry a signal, DNp01 does not."""
        rng = np.random.default_rng(0)

        def one(arm, seed):
            dr = np.full(8, np.nan); dra = np.full(8, np.nan)
            dr[1] = rng.normal(0, 0.002); dra[1] = 0.010 + rng.normal(0, 0.002)
            drive = rng.normal(0, 0.01, 8); rate = rng.normal(5, 0.2, 8); rate[1] = np.nan
            if arm == "stim" and signal:
                dra[1] += 0.06; dr[1] -= 0.05; drive[2] += 0.5; rate[2] += 3.0
            return {"rate_hz": rate, "drive_mv": drive, "drive_mv_abs": np.abs(drive), "optic_dr": dr, "optic_dr_abs": dra}
        return ([self._rec(c, one("stim", s), "stim", s) for s in range(n)], [self._rec(c, one("ctrl", s), "ctrl", s) for s in range(n)],
                [self._rec(c, one("null", s), "null", s) for s in range(n)])

    def test_type_graph_and_depth(self):
        from flyverse.brain import LIFParams
        from flyverse.interp import trace as tr
        c = graph(); ew = common.effective_weights(c, LIFParams(receptor_model=None, event_driven=False))
        tg = tr.TypeGraph(c, ew, common.raw_counts(c, with_sign0=False)[0])
        self.assertEqual(list(tg.keys), ["DNa02", "DNp01", "GLNO", "LC4", "Mi4", "PEN_a", "R1-R6"])
        # M[post, pre] is the mean over post cells of the summed input: DNa02 (2 cells) <- DNp01 (1 cell onto DNa02_L only)
        M_dnp01 = float(ew.A[[4, 5]][:, [3]].sum(axis=1).mean())
        self.assertAlmostEqual(tg.M[tg.pos["DNa02"], tg.pos["DNp01"]], M_dnp01, places=6)
        self.assertAlmostEqual(tg.share[tg.pos["LC4"], tg.pos["Mi4"]], 1.0, places=6)          # LC4's only input
        self.assertEqual(tg.raw[tg.pos["DNp01"], tg.pos["LC4"]], 120.0)                         # raw, uncapped
        # GLNO -> PEN_a is sign 0: no edge in the depth graph; PEN_a is reached through LC4 (5 synapses, 100 % of its |input|)
        d = tg.depths(common.resolve(c, "R1-R6"), min_share=0.02, depth_max=6)
        self.assertEqual({k: int(v) for k, v in zip(tg.keys, d)}, {"R1-R6": 0, "Mi4": 1, "LC4": 2, "DNp01": 3, "PEN_a": 3, "DNa02": 4, "GLNO": 4})
        d0 = tg.depths(common.resolve(c, "R1-R6"), min_share=0.0)
        np.testing.assert_array_equal(d0, d)                                                     # no weak edges here: identical
        inp = tg.inputs("DNp01")
        self.assertEqual(list(inp.pre_type), ["LC4"]); self.assertEqual(int(inp.sign.iloc[0]), 1)
        self.assertEqual(tr.stage_of("Mi4", "ol_intrinsic", {}), 2); self.assertEqual(tr.stage_of("T3", "ol_intrinsic", {}), 4)
        self.assertEqual(tr.stage_of("LC4", "visual_projection", {}), 6); self.assertIsNone(tr.stage_of("DNp01", "descending_neuron", {}))
        self.assertEqual(tr.load_stage_table("family"), {}); self.assertIsNone(tr.load_stage_table(None))

    def test_statistics(self):
        from flyverse.interp import trace as tr
        # figure_stats is probe_figure_stages': (A - B) obj mean - bg mean over the bg population sd
        xA = np.array([1.0, 1.2, 0.0, 0.1, -0.1, 0.0]); xB = np.zeros(6)
        obj = np.array([1, 1, 0, 0, 0, 0], bool); bg = ~obj
        f = tr.figure_stats(xA, xB, obj, bg)
        self.assertAlmostEqual(f["figure"], 1.1); self.assertAlmostEqual(f["z"], 1.1 / (np.array([0, 0.1, -0.1, 0]).std() + 1e-9), places=4)
        inv = np.array([0, 0, 1, 1, 1, 1]); d = xA - xB
        v, _ = tr._group_stat(d, inv, 2, "best_cell"); np.testing.assert_allclose(v, [1.2, 0.1])
        v, _ = tr._group_stat(d, inv, 2, "mean"); np.testing.assert_allclose(v, [1.1, 0.0])
        v, z = tr._group_stat(np.r_[d, d], np.r_[inv, inv] * 0, 1, "figure_z", np.r_[obj, obj], np.r_[bg, bg], min_obj=2, min_bg=4)
        self.assertAlmostEqual(v[0], 1.1); self.assertAlmostEqual(z[0], f["z"], places=4)
        v, _ = tr._group_stat(np.array([np.nan, 2.0, np.nan]), np.array([0, 0, 1]), 2, "best_cell")
        self.assertEqual(v[0], 2.0); self.assertTrue(np.isnan(v[1]))

    def test_trace_finds_the_lost_stage(self):
        from flyverse.brain import LIFParams
        from flyverse import interp
        c = graph(); p = LIFParams(receptor_model=None, event_driven=False)
        stim, ctrl, nul = self._arms(c)
        from flyverse.interp import trace as tr
        self.assertIs(interp.trace, tr)                       # the package attribute is the submodule once imported (docs/INTERP.md 2.1 caveat)
        res = tr.trace(c, "R1-R6", stimulus=stim, control=ctrl, null=nul, params=p, stat="best_cell", min_cells=1, decompose_at="first_lost")
        self.assertIsInstance(res, common.Result); self.assertEqual(res.tool, "trace")
        pt = res.table("per_type").set_index("type")
        self.assertEqual(list(res.table("per_type").depth), sorted(res.table("per_type").depth))
        self.assertEqual(pt.loc["Mi4", "verdict"], "result"); self.assertEqual(pt.loc["Mi4", "quantity"], "optic_dr_abs")
        self.assertEqual(pt.loc["LC4", "verdict"], "result"); self.assertEqual(pt.loc["LC4", "quantity"], "drive_mv")
        self.assertEqual(pt.loc["DNp01", "verdict"], "null"); self.assertGreater(pt.loc["Mi4", "z"], 3)
        self.assertEqual(len(pt.loc["Mi4", "stim_values"]), 4); self.assertEqual(pt.loc["Mi4", "unit_kind"], "graded")
        self.assertAlmostEqual(pt.loc["Mi4", "p_floor"], 2 / 70); self.assertIsNone(res.summary["p_floor_note"])
        self.assertAlmostEqual(tr.p_floor(3, 3), 0.1); self.assertAlmostEqual(tr.p_floor(5, 5), 2 / 252)
        self.assertEqual(set(res.summary["carriers"]), {"Mi4", "LC4"})
        # the depth rule and the input rule agree: DNp01 (depth 3) is the first stage lost, fed by the carrier LC4
        self.assertEqual(res.summary["first_lost_depth"], 3); self.assertEqual(res.summary["lost_depth"], 3)
        self.assertEqual(set(res.summary["lost_types"]), {"DNp01", "PEN_a"}); self.assertEqual(set(res.summary["decomposed"]), {"DNp01", "PEN_a"})
        li = res.table("lost_inputs")
        r = li[(li.target_type == "DNp01") & (li.pre_type == "LC4")].iloc[0]
        self.assertAlmostEqual(r.share, 1.0); self.assertEqual(r.sign, 1); self.assertEqual(r.pre_verdict, "result"); self.assertEqual(r.raw_synapses_per_post, 120.0)
        self.assertGreater(r.pre_signed_figure, 0); self.assertGreater(r.term, 0)
        cz = {x["target_type"]: x for x in res.summary["cancellation"]}
        self.assertEqual(cz["DNp01"]["carriers_raising"], ["LC4"]); self.assertAlmostEqual(cz["DNp01"]["excitatory_share_of_carriers"], 1.0)
        # Neurome table: LC4's body with its scored quantity; the Result is exportable
        rb = res.table("readout_per_body")
        self.assertIn("1003", list(rb.bodyId)); self.assertEqual(list(rb.columns), common.EXPORT_TABLES["readout_per_body"])
        self.assertEqual(res.check(), []); self.assertEqual(res.replicates["n"], 4); self.assertEqual(res.replicates["null"]["source"], "null_runs")
        self.assertEqual(res.validation["status"], "not run")                        # no validation type in the toy graph
        with tempfile.TemporaryDirectory() as d:
            path = res.save(Path(d) / "t.json"); back = common.Result.load(path)
        self.assertEqual(back.summary["lost_depth"], 3)
        # decompose composition: the stub reports itself, a missing module never fails the trace
        self.assertIn(res.summary["decompose"]["status"].split(":")[0], ("stub", "ok", "unavailable"))
        # without a null arm the null draws are the ordered control pairs; two runs per arm are underpowered
        res2 = tr.trace(c, "R1-R6", stimulus=stim[:2], control=ctrl[:2], params=p, min_cells=1)
        self.assertEqual(res2.summary["null_source"], "control_pairs"); self.assertEqual(res2.table("per_type").set_index("type").loc["Mi4", "verdict"], "underpowered")
        # three runs per arm: z is huge but the exact p floors at 0.10 -> compare says 'underpowered' (its p_floor rule)
        # and the summary says why
        res3b = tr.trace(c, "R1-R6", stimulus=stim[:3], control=ctrl[:3], null=nul[:3], params=p, min_cells=1)
        pt3b = res3b.table("per_type").set_index("type")
        self.assertGreater(pt3b.loc["Mi4", "z"], 3); self.assertEqual(pt3b.loc["Mi4", "verdict"], "underpowered")
        self.assertAlmostEqual(pt3b.loc["Mi4", "p"], 0.1); self.assertAlmostEqual(pt3b.loc["Mi4", "p_floor"], 0.1)
        self.assertIn("floors at 0.100", res3b.summary["p_floor_note"])
        # stat 'mean' on the odour-style quantity: LC4's rate rises 3 Hz
        for r in stim + ctrl + nul:
            r.meta["protocol"] = "odour"; r.meta["default_quantity"] = {"spiking": "rate_hz"}
        res3 = tr.trace(c, "R1-R6", stimulus=stim, control=ctrl, null=nul, params=p, stat="mean", min_cells=1)
        pt3 = res3.table("per_type").set_index("type")
        self.assertEqual(pt3.loc["LC4", "quantity"], "rate_hz"); self.assertAlmostEqual(pt3.loc["LC4", "diff"], 3.0, delta=0.5)
        self.assertEqual(pt3.loc["LC4", "verdict"], "result")

    def test_figure_z_and_dprime(self):
        from flyverse.brain import LIFParams
        from flyverse.interp import trace as tr
        # a 13-cell graph: 6 Mi4 columns feeding 6 LC4 cells (one column each); the object sits in columns 0-1
        types = ["R1-R6"] + ["Mi4"] * 6 + ["LC4"] * 6
        nts = ["histamine"] + ["gaba"] * 6 + ["acetylcholine"] * 6
        sc = ["ol_sensory"] + ["ol_intrinsic"] * 6 + ["visual_projection"] * 6
        n = pd.DataFrame({"bodyId": np.arange(2001, 2014, dtype=np.int64), "type": types, "instance": types, "superclass": sc, "class": [""] * 13,
                          "subclass": [""] * 13, "somaSide": ["L"] * 13, "nt": nts, "sign": np.array([cn.NT_SIGN[x] for x in nts], np.float32)})
        cnt = np.zeros((13, 13), np.float32)
        for k in range(6):
            cnt[1 + k, 0] = 20; cnt[7 + k, 1 + k] = 30
        coo = sp.coo_matrix(cnt); W = sp.csr_matrix((coo.data * n.sign.to_numpy()[coo.col], (coo.row, coo.col)), shape=(13, 13), dtype=np.float32)
        c = Connectome(n, W, pd.Series(np.arange(13), index=n.bodyId))
        p = LIFParams(receptor_model=None, event_driven=False)
        cols = {"column": [-1] + list(range(6)) + list(range(6)), "columns_obj": [0, 1], "columns_bg": [2, 3, 4, 5]}
        rng = np.random.default_rng(1)

        def rec(arm, seed):
            dr = np.full(13, np.nan); dr[1:7] = rng.normal(0, 0.001, 6)
            drive = np.zeros(13); drive[7:] = rng.normal(0, 0.01, 6)
            if arm == "stim":
                dr[1:3] -= 0.05; drive[7:9] += 0.3
            v = {"rate_hz": np.zeros(13), "drive_mv": drive, "drive_mv_abs": np.abs(drive), "optic_dr": dr, "optic_dr_abs": np.abs(dr)}
            r = common.Recording(np.array([15000.0]), np.arange(13), n.bodyId.to_numpy(), np.array(types), {k: np.asarray(x, np.float32)[None] for k, x in v.items()}, {},
                                 dict({"protocol": "object", "arm": arm, "seed": seed, "window_s": [3.0, 15.0], "default_quantity": {"spiking": "drive_mv"}}, **cols))
            T = 50
            series = common.Recording(np.arange(T) * 10.0, np.arange(3), -np.ones(3, np.int64), np.array(["LC4", "Mi4", "R1-R6"]),
                                      {"pooled": (np.tile([5.0 + (arm == "stim") * 4, 0.0, 0.0], (T, 1)) + rng.normal(0, 0.05, (T, 3))).astype(np.float32)}, {},
                                      {"pooled": True, "n_cells": [6, 6, 1]})
            r.series = series; r.path = f"mem:{arm}{seed}"
            return r
        stim = [rec("stim", s) for s in range(4)]; ctrl = [rec("ctrl", s) for s in range(4)]; nul = [rec("null", s) for s in range(4)]
        res = tr.trace(c, "R1-R6", stimulus=stim, control=ctrl, null=nul, params=p, stat="figure_z", min_cells=1, per_body="all", min_obj=2, min_bg=4)
        pt = res.table("per_type").set_index("type")
        self.assertEqual(pt.loc["Mi4", "verdict"], "result"); self.assertLess(pt.loc["Mi4", "stim_mean"], -0.04)      # the dark object lowers Mi4
        self.assertEqual(pt.loc["Mi4", "carry_runs"], 4); self.assertEqual(pt.loc["Mi4", "null_carry_runs"], 0)
        self.assertEqual(pt.loc["LC4", "verdict"], "result"); self.assertEqual(len(pt.loc["Mi4", "figure_z_runs"]), 4)
        self.assertIsNone(res.summary["first_lost_depth"])                                       # nothing after LC4 to lose
        # d' from the pooled series through screen.rank
        res = tr.trace(c, "R1-R6", stimulus=stim, control=ctrl, null=nul, params=p, stat="dprime", min_cells=1)
        pt = res.table("per_type").set_index("type")
        self.assertGreater(pt.loc["LC4", "stim_mean"], 3.0); self.assertEqual(pt.loc["LC4", "verdict"], "result"); self.assertLess(abs(pt.loc["Mi4", "stim_mean"]), 0.5)
        for r in stim:
            r.meta.pop("column")
        with self.assertRaises(ValueError):
            tr.trace(c, "R1-R6", stimulus=stim, control=ctrl, null=nul, params=p, stat="figure_z", min_cells=1, min_obj=2, min_bg=4)

    def test_accumulator_and_run_files(self):
        """ArmAccumulator turns per-frame FlyBrain reads into the accumulated recording + the pooled series."""
        from flyverse.interp import trace as tr
        c = graph()
        fb = fake_fb(c, np.zeros(8)); fb.c = c
        fb.optic.r0 = np.array([0.5], np.float32)
        frames = [np.array([0, 0, 10, 0, 4, 8, 0, 0], np.float32), np.array([0, 0, 12, 0, 6, 8, 0, 0], np.float32), np.array([0, 0, 14, 0, 8, 8, 0, 0], np.float32)]
        acc = tr.ArmAccumulator(fb)
        for k, r in enumerate(frames):
            fb.brain.rate = r; fb.brain.rate_np = (lambda r=r: r); fb.brain.t = 10.0 * k
            fb.brain.spike_counts = np.arange(8, dtype=np.float32) * (k + 1)          # index i spikes i times per frame
            fb.optic.rates = (lambda k=k: np.array([0.5 + 0.1 * (k + 1)], np.float32))
            acc.add(fb, k, skip=1)
        cells, series = acc.finish({"protocol": "toy", "arm": "stim", "seed": 0})
        self.assertEqual(acc.n, 2); self.assertEqual(cells.quantities["rate_hz"].shape, (1, 8))
        np.testing.assert_allclose(cells.quantities["rate_hz"][0, [0, 2, 7]], [0.0, 2 * 2 / 0.02, 7 * 2 / 0.02])       # counts over 2 frames = 20 ms
        self.assertTrue(np.isnan(cells.quantities["rate_hz"][0, 1]))                                                 # the rate unit
        self.assertAlmostEqual(float(cells.quantities["optic_dr"][0, 1]), 0.25, places=5)                          # mean of +0.2, +0.3
        self.assertAlmostEqual(float(cells.quantities["drive_mv"][0, 3]), 2.0)
        self.assertEqual(series.quantities["pooled"].shape, (2, 7)); self.assertTrue(series.meta["pooled"])
        keys = list(series.types); self.assertAlmostEqual(float(series.quantities["pooled"][0, keys.index("DNa02")]), 7.0)     # frame 1: (6 + 8) / 2
        self.assertAlmostEqual(float(series.quantities["pooled"][1, keys.index("Mi4")]), 0.3, places=5)                 # graded: deviation
        with tempfile.TemporaryDirectory() as d:
            tr.save_recording(cells, Path(d) / "stim_r0"); tr.save_recording(series, Path(d) / "stim_r0_series")      # savez_compressed, same schema
            runs = tr.load_runs(str(Path(d) / "stim_r*"))
            self.assertEqual(len(runs), 1); self.assertIsNotNone(runs[0].series); self.assertEqual(runs[0].meta["arm"], "stim")
            self.assertEqual(runs[0].series.quantities["pooled"].shape, (2, 7))
            np.testing.assert_allclose(runs[0].quantities["drive_mv"], cells.quantities["drive_mv"]); self.assertEqual(list(runs[0].types), list(cells.types))
            self.assertEqual(runs[0].series.meta["hz"], 100.0)
        acc2 = tr.ArmAccumulator(fb, series_every=2)
        for k, r in enumerate(frames):
            fb.brain.rate = r; fb.brain.rate_np = (lambda r=r: r); fb.brain.spike_counts = np.arange(8, dtype=np.float32) * (k + 1)
            acc2.add(fb, k, skip=0)
        cells2, series2 = acc2.finish({"arm": "stim"})
        self.assertEqual(acc2.n, 3); self.assertEqual(series2.quantities["pooled"].shape, (2, 7)); self.assertEqual(series2.meta["hz"], 50.0)
        # the column sets of the figure statistic
        ang = np.array([0.0, 5.0, 10.0, 27.0, 28.0, 60.0])
        o, b = tr.figure_columns(None, ang, 8.0, 20.0)
        np.testing.assert_array_equal(o, [0, 1]); np.testing.assert_array_equal(b, [4, 5])

    def test_untyped_cells_and_deterministic_null(self):
        """Untyped cells ('' type) are not a node of the depth graph (they would short-circuit every depth); a null arm
        whose draws are identical (SD 0: a deterministic input stage) is 'undetermined' -- z is not defined there, so
        the magnitude and the exact p are what the reader gets (common.compare; docs/INTERP.md 11, defect 4)."""
        from flyverse.brain import LIFParams
        from flyverse.interp import trace as tr
        c = graph(); n = c.neurons.copy()
        n.loc[n.type == "GLNO", "type"] = ""                     # GLNO becomes an untyped cell that projects onto PEN_a
        n.loc[n.type == "", "nt"] = "acetylcholine"; n.loc[n.type == "", "sign"] = 1.0
        W = c.W.tolil(); W[7, 6] = 5.0; W = W.tocsr()            # PEN_a <- the untyped cell, so a typed edge would reach PEN_a at depth 1 through it
        c2 = Connectome(n, W, pd.Series(np.arange(8), index=n.bodyId))
        ew = common.effective_weights(c2, LIFParams(receptor_model=None, event_driven=False))
        tg = tr.TypeGraph(c2, ew, common.raw_counts(c2, with_sign0=False)[0])
        self.assertEqual(tg.untyped, tg.pos[""])
        d = dict(zip(tg.keys, tg.depths(common.resolve(c2, "R1-R6"), min_share=0.0)))
        self.assertEqual(d[""], -1); self.assertEqual(d["PEN_a"], 3)                       # not 1 via the untyped node
        self.assertNotIn("", list(tg.inputs("PEN_a", min_share=0.0).pre_type))
        # deterministic null: every null draw identical
        p = LIFParams(receptor_model=None, event_driven=False)
        stim, ctrl, nul = self._arms(c, n=5)
        for r in ctrl + nul:
            r.quantities["drive_mv"][:] = 0.0; r.quantities["rate_hz"][:] = 5.0
        res = tr.trace(c, "R1-R6", stimulus=stim, control=ctrl, null=nul, params=p, stat="best_cell", min_cells=1, decompose_at=None, per_body="none")
        pt = res.table("per_type").set_index("type")
        self.assertEqual(pt.loc["LC4", "note"], "null_sd_zero"); self.assertEqual(pt.loc["LC4", "verdict"], "undetermined")
        self.assertTrue(np.isnan(pt.loc["LC4", "z"])); self.assertAlmostEqual(pt.loc["LC4", "p"], 2 / 252); self.assertGreater(pt.loc["LC4", "diff"], 0)
        # a deterministic null the rank test DOES settle (p 0.15 > alpha, the stim draws straddle it) is a plain null
        self.assertEqual(pt.loc["DNp01", "verdict"], "null"); self.assertEqual(pt.loc["DNp01", "note"], "null_sd_zero")
        # OpticParams rebuilt from a provenance block (pair_gain rows back to tuples)
        prov = common.provenance(c, fb=fake_fb(c, np.zeros(8)))
        op = tr.optic_params_from_provenance(prov)
        self.assertIsNotNone(op); self.assertEqual(op.gain_fb, prov["model"]["optic"]["gain_fb"])
        self.assertTrue(all(isinstance(x, tuple) for x in op.pair_gain))
        self.assertEqual(tr.optic_params_from_provenance({"model": {"optic": {"no_such_field": 1}}}).gain_fb, op.gain_fb)   # unknown keys dropped

    def test_cli_parser_and_stage_copy(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("interp_trace", Path(__file__).resolve().parents[1] / "scripts" / "interp_trace.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        self.assertEqual(set(mod.PROTOCOLS), {"object", "apple", "odour"})
        self.assertEqual(mod.ODOUR_ARMS, {"stim": "apple8_into_wind", "ctrl": "clean_into_wind", "null": "clean_into_wind"})
        # the family-rule copy in trace.py equals the script's (read as text, so pygame / torch are not imported)
        from flyverse.interp import trace as tr
        src = (Path(__file__).resolve().parents[1] / "scripts" / "probe_figure_stages.py").read_text(encoding="utf-8")
        for pat, s in tr.FAMILY_STAGE:
            self.assertIn(pat, src)
        self.assertIn(repr(tr.NERN_GROUP_STAGE), src)
        self.assertIn(tr.STAGES[4], src)


class LesionTests(unittest.TestCase):
    """flyverse/interp/lesion.py: the manifest grammar, how each lesion is applied (never by editing the model), the
    planned batch, and the matrix / dissociation extraction on the round-4 / round-5 hold numbers."""

    def manifest(self):
        return {"name": "toy", "sections": "taste,smell",
                "baseline": {"id": "baseline", "kind": "none", "receptor_model": "sign", "receptor_net_rule": "abs"},
                "lesions": [{"id": "no_LC4", "kind": "population", "spec": "LC4"},
                            {"id": "no_ach_dn", "kind": "transmitter", "spec": "acetylcholine", "within": "module=descending"},
                            {"id": "off", "kind": "lif", "spec": {"receptor_model": None}, "receptor_model": "off"}]}

    def test_manifest_grammar_and_kinds(self):
        from flyverse.interp import lesion as L
        man = L.load_manifest(self.manifest())
        self.assertEqual([x.id for x in L.lesions_of(man)], ["baseline", "no_LC4", "no_ach_dn", "off"])
        self.assertEqual(L.find_lesion(man, "off").kind, "lif")
        self.assertEqual([x.id for x in L.lesions_of(L.load_manifest("holds"))],      # the builtin validation manifest
                         ["baseline", "holdKC", "holdDN1", "holdBrain", "holdOptic", "off"])
        self.assertEqual(L.load_manifest(json.dumps(self.manifest()))["name"], "toy")  # a JSON string (what plan embeds)
        with self.assertRaises(ValueError):
            L.Lesion.of({"id": "x", "kind": "nonsense"})
        with self.assertRaises(SystemExit):
            L.find_lesion(man, "absent")
        c = graph()
        np.testing.assert_array_equal(L.pre_indices(c, L.find_lesion(man, "no_LC4")), [2])
        np.testing.assert_array_equal(L.pre_indices(c, L.find_lesion(man, "no_ach_dn")), [3, 4, 5])
        self.assertEqual(len(L.pre_indices(c, L.find_lesion(man, "off"))), 0)
        lif, optic, rec = L.overrides(L.Lesion.of({"id": "lpi", "kind": "pair_gain", "spec": ["^LPi(34|43)$", "^LPLC2$", 1.0]}))
        self.assertIn(("^LPi(34|43)$", "^LPLC2$", 1.0), optic["pair_gain"])            # the x4 stop-gap set back to x1

    def test_weight_mask_silences_output_and_restores(self):
        from flyverse.brain import LIFParams
        from flyverse.interp import lesion as L
        c = graph(); p = LIFParams(receptor_model=None, event_driven=False)
        before = common.effective_weights(c, p).A.toarray()
        nnz, sum_abs = c.W.nnz, float(abs(c.W).sum())
        with L.weight_mask([2], c=c, scope="lif"):                       # LC4 makes no output
            A = common.effective_weights(c, p).A.toarray()
            self.assertTrue(np.all(A[:, 2] == 0))                        # its presynaptic column is gone
            self.assertEqual(c.W.nnz, nnz)                               # scope 'lif' leaves the graph alone
            self.assertAlmostEqual(float(abs(c.W).sum()), sum_abs)
            self.assertAlmostEqual(A[4, 3], before[4, 3], places=6)      # an untouched entry (DNp01 -> DNa02_L)
        np.testing.assert_allclose(common.effective_weights(c, p).A.toarray(), before, rtol=1e-6)
        with L.weight_mask([2], c=c, scope="graph"):                     # also in c.W: the rate optic lobe sees it too
            self.assertEqual(c.W.nnz, nnz)                               # the sparsity pattern is kept (explicit zeros)
            self.assertAlmostEqual(float(abs(c.W).sum()), sum_abs - 125.0)
        self.assertAlmostEqual(float(abs(c.W).sum()), sum_abs)
        np.testing.assert_allclose(common.effective_weights(c, p).A.toarray(), before, rtol=1e-6)

    def test_resolve_records_bodies_and_writes_a_tier_table(self):
        from flyverse.interp import lesion as L
        c = graph()
        rec = L.resolve_lesion(c, L.Lesion.of({"id": "no_LC4", "kind": "population", "spec": "LC4"}))
        self.assertEqual(rec["n_bodies"], 1); self.assertEqual(rec["bodies"], ["1003"])
        self.assertEqual(rec["n_entries_removed"], 2)                    # LC4 -> DNp01 and LC4 -> PEN_a
        self.assertEqual(rec["synapses_removed"], 125.0)                 # 120 + 5, raw and uncapped
        self.assertIn("NOT masked", rec["applies_to"])                   # the optic lobe reads c.W directly
        with tempfile.TemporaryDirectory() as d:                         # a receptor-tier hold, written here
            path, n = L.tier_hold_table("exact", d)
            t = pd.read_csv(path, comment="#")
            held = t[t.fast_net_abs == "held"]
            self.assertEqual(len(held), n); self.assertGreater(n, 0)
            self.assertTrue((held.tier == "exact").all())
            self.assertTrue((held.fast_sign_abs.astype(float) == held.transmitter.map(cn.NT_SIGN)).all())
            self.assertEqual(len(t), len(pd.read_csv(cn.RECEPTOR_TABLE, comment="#")))

    def test_plan_writes_one_batch_line(self):
        from flyverse.interp import lesion as L
        with tempfile.TemporaryDirectory() as d:
            res = L.lesion(self.manifest(), out_dir=d, mode="plan", replicates=2, minutes=15, c=graph(), quiet=True)
            batch = (Path(d) / "batch.sh").read_text(encoding="utf-8")
            resolved = json.loads((Path(d) / "manifest.resolved.json").read_text(encoding="utf-8"))
        self.assertEqual(batch.count("cluster_run.py"), 1)                # ONE batch call, not one call per job
        self.assertEqual(batch.count("--one "), 8)                        # 4 arms x 2 replicates
        self.assertIn("--fetch", batch); self.assertIn("assert torch.cuda.is_available()", batch)
        self.assertIn("--one off --replicate 1", batch)
        self.assertEqual(batch.count("mkdir -p"), 9)                  # the local one plus one inside every job: out/ is
        self.assertTrue(all(l.strip().lstrip("'").startswith("mkdir -p")   # git-ignored, so the cluster run copy has
                            for l in batch.splitlines() if "--one " in l))  # no out/ and the redirection would fail
        self.assertEqual(len(resolved["jobs"]), 8); self.assertEqual(len(resolved["lesions"]), 4)
        self.assertEqual(res.summary["n_jobs"], 8)
        self.assertTrue(res.table("jobs").json.iloc[0].endswith("baseline_r0.json"))
        self.assertIn('{"name": "toy"', batch)                        # a dict manifest travels inside the command

    def test_a_builtin_manifest_travels_by_name(self):
        from flyverse.interp import lesion as L
        # out/ is not shipped to the cluster, so a manifest has to reach the job as a name it can load or as JSON;
        # 'holds' is a builtin, and naming it keeps the 12 command lines readable (2 kB of embedded JSON each otherwise).
        man = L.load_manifest("holds")
        self.assertEqual(L.manifest_argument(man), "holds")
        man["lesions"][0]["spec"] = "out/elsewhere.csv"                # edited: it is no longer the builtin
        self.assertTrue(L.manifest_argument(man).startswith("{"))
        self.assertIn("--manifest holds --one holdKC --replicate 1",
                      L.job_command("holds", "holdKC", 1, out_dir="out/les_holds", sections="taste"))

    # ---- the matrix and the dissociations, on the audit's own numbers --------------------------------------------
    def _job(self, d, lesion_id, replicate, seed, checks, prov, kind="hold_table", spec="out/x.csv"):
        from flyverse.interp import lesion as L
        rows = [{"key": k, "measured": v, "reference": 0, "criterion": "> 0", "status": "PASS", "session": "E"}
                for k, v in checks.items()]
        job = {"schema": L.JOB_SCHEMA, "tool": "lesion", "manifest": "holds",
               "lesion": {"id": lesion_id, "kind": kind, "spec": spec, "bodies": None},
               "replicate": replicate, "brain_seed": seed, "sections": "taste,smell", "checks": rows,
               "results": {}, "probes": [], "runtime_s": {}, "provenance": prov}
        (Path(d) / f"{lesion_id}_r{replicate}.json").write_text(json.dumps(common.to_jsonable(job)), encoding="utf-8")

    def _prov(self):
        c = graph()
        return common.to_jsonable(common.provenance(c, fb=fake_fb(c, np.zeros(8)),
                                                    stimulus={"protocol": "benchmark sections"}))

    def test_matrix_and_dissociation_reproduce_E4(self):
        from flyverse.interp import lesion as L
        # docs/audits/receptor_integration.md E.4 (scripts/r5_attr_taste_cpu.py, device cpu, brain seeds 0, 1, 2):
        # holdBrainGlu = default on taste and off on smell; holdBrainHis the mirror image, every digit.
        taste = {"off": [1.554839, 4.341760, 2.309808], "baseline": [5.090923, 4.315772, 2.360074]}
        kc = {"off": [1079, 427, 1178], "baseline": [486, 412, 543]}
        arms = {"baseline": ("baseline", "baseline"), "off": ("off", "off"),
                "holdBrainGlu": ("baseline", "off"), "holdBrainHis": ("off", "baseline")}
        prov = self._prov()
        with tempfile.TemporaryDirectory() as d:
            for les, (t_src, k_src) in arms.items():
                for r, seed in enumerate([0, 1, 2]):
                    self._job(d, les, r, seed, {"taste.MN9_hz": taste[t_src][r], "smell.KC_active": kc[k_src][r]},
                              prov, kind="none" if les == "baseline" else "hold_table")
            res = L.lesion("holds_cpu", out_dir=d, mode="analyse", quiet=True)
            mat = res.table("matrix")
        row = mat[(mat.check == "taste.MN9_hz") & (mat.lesion_id == "holdBrainHis")].iloc[0]
        self.assertTrue(row.paired_by_seed); self.assertEqual(row.moved, "yes")
        self.assertEqual(row.replicate_values, taste["off"])                       # = off, every digit
        m = mat.set_index(["check", "lesion_id"])
        self.assertEqual(m.loc[("taste.MN9_hz", "holdBrainGlu")].moved, "no")      # = the default, every digit
        self.assertTrue(m.loc[("taste.MN9_hz", "holdBrainGlu")].identical_to_baseline)
        self.assertEqual(m.loc[("smell.KC_active", "holdBrainGlu")].moved, "yes")
        self.assertEqual(m.loc[("smell.KC_active", "holdBrainHis")].moved, "no")
        dis = res.table("dissociations")
        self.assertEqual(len(dis), 1)
        r0 = dis.iloc[0]
        self.assertEqual((r0.lesion_a, r0.lesion_b), ("holdBrainGlu", "holdBrainHis"))
        self.assertEqual((r0.check_a, r0.check_b), ("smell.KC_active", "taste.MN9_hz"))
        self.assertIn("paired by brain seed", r0.criterion)
        self.assertEqual(res.validation["arm"], "cpu"); self.assertEqual(res.validation["status"], "reproduced")
        self.assertEqual(res.check(), [])                                          # exportable
        self.assertEqual(res.replicates["unit"], "runs"); self.assertEqual(res.replicates["n"], 3)
        sens = res.table("sensitivity")
        self.assertEqual(list(sens.columns), common.EXPORT_TABLES["sensitivity"])
        self.assertNotIn("baseline", set(sens.lesion_id))

    def test_a_scattering_check_with_two_runs_is_underpowered(self):
        from flyverse.interp import lesion as L
        # out/les_holds (12 jobs, 2 draws per arm): rotate.DNp20_flip_hz scatters (baseline -30.5 / -40.8) and E.2
        # refused to attribute it. Two runs may not call it whatever the delta -- and an uncalled row cannot dissociate.
        rot = {"baseline": [-30.5, -40.8], "holdOptic": [-12.9, -24.2], "holdBrain": [-29.2, -32.1]}
        taste = {"baseline": [10.9342, 10.9342], "holdOptic": [10.9342, 10.9342], "holdBrain": [5.8455, 5.8455]}
        prov = self._prov()
        with tempfile.TemporaryDirectory() as d:
            for les in rot:
                for r in (0, 1):
                    self._job(d, les, r, None, {"rotate.DNp20_flip_hz": rot[les][r], "taste.MN9_hz": taste[les][r]},
                              prov, kind="none" if les == "baseline" else "hold_table")
            res = L.lesion("holds", out_dir=d, mode="analyse", quiet=True)
        m = res.table("matrix").set_index(["check", "lesion_id"])
        self.assertEqual(m.loc[("rotate.DNp20_flip_hz", "holdOptic")].moved, "underpowered")   # +17.1 Hz, 2 runs
        self.assertIn("quoted, not called", m.loc[("rotate.DNp20_flip_hz", "holdOptic")].moved_criterion)
        self.assertEqual(m.loc[("taste.MN9_hz", "holdBrain")].moved, "yes")                    # bit-identical: still called
        self.assertEqual(len(res.table("dissociations")), 0)
        self.assertEqual(res.summary["n_underpowered"], 2)             # the two lesion arms; the baseline row is neither
        self.assertEqual(res.summary["underpowered_checks"], ["rotate.DNp20_flip_hz"])

    def test_matrix_bit_identity_rule_on_the_gpu_arms(self):
        from flyverse.interp import lesion as L
        # E.2 / the round-4 re-score table: two draws per arm, no brain seed, every value bit-identical within its
        # arm -- then any difference at all is a move, and a size threshold has to be asked for.
        gpu = {"baseline": {"taste.MN9_hz": 10.9342, "smell.KC_active": 816, "walk.power_max_hz": 48.4805},
               "holdKC": {"taste.MN9_hz": 10.9342, "smell.KC_active": 1155, "walk.power_max_hz": 55.63},
               "holdBrain": {"taste.MN9_hz": 5.8455, "smell.KC_active": 1426, "walk.power_max_hz": 47.0012},
               "holdOptic": {"taste.MN9_hz": 10.9342, "smell.KC_active": 816, "walk.power_max_hz": 64.9147}}
        prov = self._prov()
        with tempfile.TemporaryDirectory() as d:
            for les, checks in gpu.items():
                for r in (0, 1):
                    self._job(d, les, r, None, checks, prov, kind="none" if les == "baseline" else "hold_table")
            res = L.lesion("holds", out_dir=d, mode="analyse", quiet=True)
            loose = L.lesion("holds", out_dir=d, mode="analyse", rel_tolerance=0.05, quiet=True)
        mat = res.table("matrix")
        self.assertTrue(mat.bit_identical_within_arms.all()); self.assertFalse(mat.paired_by_seed.any())
        m = mat.set_index(["check", "lesion_id"])
        self.assertEqual(m.loc[("taste.MN9_hz", "holdOptic")].moved, "no")         # the Brain side alone = the default
        self.assertEqual(m.loc[("taste.MN9_hz", "holdBrain")].moved, "yes")        # the optic side alone = off
        self.assertAlmostEqual(m.loc[("taste.MN9_hz", "holdBrain")].delta, -5.0887, places=4)
        self.assertEqual(m.loc[("smell.KC_active", "holdKC")].value, 1155)         # between 816 and 1426: the groups interact
        self.assertGreater(m.loc[("walk.power_max_hz", "holdOptic")].delta, 0)     # non-monotone: both holds above the
        self.assertLess(m.loc[("walk.power_max_hz", "holdBrain")].delta, 0)        # default, holdOptic by 16 Hz
        self.assertEqual(len(res.table("dissociations")), 0)                       # nothing dissociates at bit-identity
        pairs = {(r.lesion_a, r.lesion_b, r.check_a, r.check_b) for r in loose.table("dissociations").itertuples()}
        self.assertIn(("holdBrain", "holdOptic", "taste.MN9_hz", "walk.power_max_hz"), pairs)
        v = res.validation
        self.assertEqual(v["arm"], "gpu")
        got = {(r["check"], r["lesion"]): r for r in v["measured"]["checks"]}
        self.assertEqual(got[("taste.MN9_hz", "holdBrain")]["status"], "reproduced")
        self.assertEqual(got[("smell.KC_active", "holdKC")]["expected"], 1155)
        self.assertEqual(got[("loom.GF_peak_hz", "holdBrain")]["status"], "not run")


class AtlasTests(unittest.TestCase):
    """flyverse/interp/atlas.py on a subset of graph(): the stimulation list and its specs, the readout groups and
    their left-right derivations, one real FlyBrain run (CPU, 5 flies, 3 of them stimulated), the AtlasRun npz/json
    round trip, the null comparison, the Neurome rows and the validation extractor. The subset drops the
    photoreceptor and the optic rate unit -- graph() has no hex coordinates, so no retina can be built -- and
    everything else is the shipped code path (FlyBrain.stimulate, screen.TypeRecorder, motor.motor_groups)."""

    def brain_subset(self):
        return graph().subset([2, 3, 4, 5, 6, 7])          # LC4, DNp01, DNa02_L, DNa02_R, GLNO, PEN_a

    def params(self):
        from flyverse.brain import LIFParams
        return LIFParams(receptor_model=None, event_driven=False)

    def runs(self, c, n=3, **kw):
        from flyverse.interp import atlas as A
        pops = A.make_populations(c, ["DNa02", "LC4"], by_side=True, hz=200.0, ms=100.0)
        return pops, [A.run_once(c, pops, hz=200.0, ms=100.0, settle_ms=50.0, batch=5, n_null=2,
                                 pattern=r"^(DNa02|PEN_a|LC4)$", params=self.params(), device="cpu", seed=s, **kw)
                      for s in common.replicate_seeds(n, 0)]

    def test_population_list_and_specs(self):
        from flyverse.interp import atlas as A
        c = graph()
        pops = A.make_populations(c, "~^DN", by_side=True)
        # a bilateral type splits by side; a type present on one side only keeps its bare name
        self.assertEqual([p.label for p in pops], ["DNa02_L", "DNa02_R", "DNp01"])
        self.assertEqual([p.spec for p in pops], ["type=DNa02&somaSide=L", "type=DNa02&somaSide=R", "type=DNp01"])
        for p in pops:                                     # every recorded spec resolves back to exactly its cells
            np.testing.assert_array_equal(common.resolve(c, p.spec), p.idx)
        self.assertEqual([float(p.hz[0]) for p in pops], [150.0, 150.0, 150.0])
        self.assertEqual([p.ms for p in pops], [400.0, 400.0, 400.0])
        # split=None keeps one population per spec; a dict entry names it and may carry per-cell rates (the JO arm)
        one = A.make_populations(c, {"steer": "DNa02"}, split=None)
        self.assertEqual([(p.label, len(p.idx)) for p in one], [("steer", 2)])
        rates = A.make_populations(c, [{"label": "jo", "spec": "DNa02", "idx": np.array([4, 5]),
                                        "hz": np.array([7.0, 9.0]), "ms": 1000.0}], split=None)
        np.testing.assert_allclose(rates[0].hz, [7.0, 9.0]); self.assertEqual(rates[0].ms, 1000.0)
        self.assertEqual(rates[0].record(c, frozen=np.array([4]))["frozen_frac"], 0.5)
        self.assertEqual(rates[0].record(c)["body_ids"], ["1005", "1006"])
        self.assertEqual(A.make_populations(c, "DNa02", min_cells=99), [])
        with self.assertRaises(ValueError):
            A.make_populations(c, [{"label": "x", "idx": np.array([0, 1]), "hz": np.array([1.0])}], split=None)

    def test_readout_groups_and_derived(self):
        from flyverse.interp import atlas as A
        c = graph()
        groups, derived = A.readout_groups(c, ("motor",), pattern=r"^(DNa02|PEN_a)$", by_side=True)
        for name in A.MOTOR_FIELDS:                        # every pooled MotorRates field is a readout
            self.assertIn(name, groups)
        self.assertIn("power", groups); self.assertIn("gf", groups)
        self.assertEqual(sorted(k for k in groups if k.startswith("type.")), ["type.DNa02_L", "type.DNa02_R", "type.PEN_a_R"])
        np.testing.assert_array_equal(groups["type.DNa02_L"], [4])
        self.assertIn(("type.DNa02_LR", "type.DNa02_L", "type.DNa02_R"), derived)
        self.assertIn(("leg_LR", "leg_L", "leg_R"), derived)
        self.assertNotIn("type.PEN_a_LR", [d[0] for d in derived])       # one side only: no asymmetry readout

    def test_wind_deflections_and_contexts(self):
        from flyverse.interp import atlas as A
        # wind blows towards 180 deg (from +x). Heading -90 faces -y, so +x is the fly's LEFT (screen_steering.py)
        dL, dR = A.wind_deflections(-90.0)
        self.assertGreater(dL, 0); self.assertLess(dR, 0); self.assertAlmostEqual(dL, -dR, places=6)
        self.assertAlmostEqual(dL, 0.3 * np.cos(np.pi / 4) / 0.5, places=6)
        for got, want in zip(A.wind_deflections(+90.0), (dR, dL)):       # the mirror image
            self.assertAlmostEqual(got, want, places=9)
        head_on = A.wind_deflections(0.0)
        self.assertAlmostEqual(head_on[0], head_on[1], places=6); self.assertGreater(head_on[0], 0)
        self.assertEqual(A.CONTEXTS["wind_left"]["wind"], (dL, dR))
        self.assertEqual(A.CONTEXTS["wind_off"]["wind"], (0.0, 0.0))
        self.assertEqual(A.apply_context(None, None), {"context": None})
        seen = {}
        self.assertEqual(A.apply_context(SimpleNamespace(taste=lambda v: seen.setdefault("taste", v)), "sugar")["context"], "sugar")
        self.assertEqual(seen, {"taste": 1.0})
        with self.assertRaises(ValueError):
            A.apply_context(SimpleNamespace(), {"colour": 1.0})

    def test_run_roundtrip_and_null_comparison(self):
        from flyverse.interp import atlas as A
        c = self.brain_subset()
        pops, runs = self.runs(c, 3)
        self.assertEqual([p.label for p in pops], ["DNa02_L", "DNa02_R", "LC4"])
        r0 = runs[0]
        self.assertEqual(r0.values["mean"].shape, (3, len(r0.readouts)))
        self.assertEqual(len(r0.null_ids), 2)                             # batch 5 - 3 populations
        self.assertEqual(r0.null_values["mean"].shape, (2, len(r0.readouts)))
        self.assertEqual(r0.meta["provenance"]["execution"]["device"], "cpu")
        self.assertEqual(r0.meta["provenance"]["execution"]["replicate_unit"], "runs")
        for key in common.REQUIRED_PROVENANCE:
            self.assertIn(key, r0.meta["provenance"])
        self.assertEqual(r0.frame("mean").loc["DNa02_L", "type.DNa02_L"], r0.values["mean"][0, r0.readouts.index("type.DNa02_L")])
        with tempfile.TemporaryDirectory() as d:
            back = A.AtlasRun.load(r0.save(Path(d) / "run_r0"))
        np.testing.assert_allclose(back.values["mean"], r0.values["mean"])
        np.testing.assert_allclose(back.null_values["final"], r0.null_values["final"])
        self.assertEqual(back.readouts, r0.readouts); self.assertEqual(back.labels, r0.labels)

        res = A.analyse_runs(runs, c=c, top=2)
        df = res.table("atlas")
        self.assertEqual(res.summary["n_runs"], 3); self.assertEqual(res.summary["n_populations"], 3)
        self.assertEqual(res.check(), [])
        self.assertEqual(res.replicates["unit"], "runs"); self.assertEqual(res.replicates["n"], 3)
        self.assertEqual(res.replicates["null"]["n_draws"], 6)
        def row(pop, ro):
            return df[(df.population == pop) & (df.readout == ro)].iloc[0]
        self.assertGreater(row("DNa02_L", "type.DNa02_L")["diff"], 10.0)   # the pulse drives the cell it names
        # the atlas' null rows are bit-identical (SD 0), so compare's verdict is 'undetermined' and the declared
        # effect size z_floor (diff / max(SD, 0.05 Hz)) is what calls the row a mover (atlas.called)
        self.assertEqual(row("DNa02_L", "type.DNa02_L")["verdict"], "undetermined")
        self.assertEqual(row("DNa02_L", "type.DNa02_L")["verdict_z_floor"], "result")
        self.assertEqual(row("DNa02_L", "type.DNa02_L")["n_runs"], 3)
        self.assertEqual(row("LC4", "type.DNa02_L")["verdict"], "null")    # an unconnected pulse moves nothing
        self.assertEqual(list(A.called(df[df.readout == "type.DNa02_L"]).population), ["DNa02_L"])
        self.assertAlmostEqual(row("LC4", "type.DNa02_L")["diff"], 0.0, places=6)
        self.assertLess(row("DNa02_R", "type.DNa02_LR")["diff"], -10.0)    # the left-right readout has the sign
        # movers lists only populations that actually moved the readout, and marks the ones inside it
        mv = res.table("movers")
        m = mv[mv.readout == "type.DNa02_L"]
        self.assertEqual(list(m.population), ["DNa02_L"]); self.assertEqual(list(m.self_drive), [True])
        self.assertEqual(int(m.readout_shared_cells.iloc[0]), 1)
        self.assertFalse(bool(row("DNa02_R", "type.DNa02_L")["self_drive"]))
        self.assertEqual(res.summary["top_population_per_readout"]["type.DNa02_L"]["self_drive"], True)
        self.assertEqual(sorted(p["label"] for p in res.populations), ["DNa02_L", "DNa02_R", "LC4"])
        # two runs are never a result, whatever the numbers (common.compare's scatter rule)
        two = A.analyse_runs(runs[:2], c=None)
        self.assertEqual(set(two.table("atlas").verdict), {"underpowered"})

    def test_atlas_end_to_end(self):
        """The contract entry point itself: populations -> runs -> Result, in process (what --device cpu does)."""
        from flyverse.interp import atlas as A
        c = self.brain_subset()
        with tempfile.TemporaryDirectory() as d:
            res = A.atlas(c, ["DNa02", "LC4"], hz=200.0, ms=100.0, settle_ms=50.0, batch=5, n_null=2, replicates=3,
                          pattern=r"^(DNa02|PEN_a|LC4)$", params=self.params(), device="cpu", out_dir=d, top=3)
            self.assertEqual(sorted(p.name for p in Path(d).glob("run_r*.npz")),
                             ["run_r0.npz", "run_r1.npz", "run_r2.npz"])
        self.assertEqual(res.tool, "atlas"); self.assertEqual(res.check(), [])
        self.assertEqual(res.replicates["n"], 3); self.assertEqual(res.replicates["unit"], "runs")
        self.assertEqual([r["seed"] for r in res.replicates["runs"]], [0, 1, 2])
        self.assertEqual(res.provenance["execution"]["device"], "cpu")
        self.assertIn("flyverse_commit_analysis", res.provenance)
        df = res.table("atlas")
        self.assertEqual(set(df.population), {"DNa02_L", "DNa02_R", "LC4"})
        self.assertGreater(float(df[(df.population == "DNa02_L") & (df.readout == "type.DNa02_L")]["diff"].iloc[0]), 10.0)
        self.assertEqual(res.files["generator"], "flyverse/interp/atlas.py::atlas")

    def test_per_body_rows_and_validation(self):
        from flyverse.interp import atlas as A
        c = self.brain_subset()
        _, runs = self.runs(c, 3)
        r = runs[0]
        r.body_idx = np.array([2, 3])                                     # stand in for the motor groups this graph lacks
        r.body_values = np.tile(np.array([[7.0, 1.0]], np.float32), (len(r.pops), 1))
        r.body_null = np.full((2, 2), 1.0, np.float32)
        tbl = A._per_body_table(c, [r], A.analyse_runs([r], c=None).table("atlas"), ["DNa02_L"], "mean")
        self.assertEqual(len(tbl), 2)
        for col in common.EXPORT_TABLES["readout_per_body"]:
            self.assertIn(col, tbl.columns)
        self.assertEqual(list(tbl.bodyId), ["1005", "1006"])              # decimal strings
        self.assertEqual(list(tbl.type), ["DNa02", "DNa02"]); self.assertEqual(list(tbl.unit_kind), ["spiking"] * 2)
        self.assertAlmostEqual(float(tbl.stimulus_minus_control.iloc[0]), 6.0)
        self.assertEqual(tbl.window_start_s.iloc[0], 0.05); self.assertAlmostEqual(tbl.window_end_s.iloc[0], 0.15)

        # the validation extractor reads the reference arms out of an atlas table (VALIDATION['atlas'], docs/INTERP.md 6)
        ref = common.VALIDATION["atlas"]["reference"]
        rows = [{"population": "JO_wind_left", "readout": "type.DNp18_LR", "stim_mean": 30.0, "final_mean": 30.0, "pre_mean": 0.0},
                {"population": "JO_wind_right", "readout": "type.DNp18_LR", "stim_mean": -15.0, "final_mean": -15.0, "pre_mean": 0.0},
                {"population": "JO_wind_left", "readout": "type.DNp33_LR", "stim_mean": -30.0, "final_mean": -30.0, "pre_mean": 0.0},
                {"population": "JO_wind_right", "readout": "type.DNp33_LR", "stim_mean": 19.0, "final_mean": 19.0, "pre_mean": 0.0},
                {"population": "DNa02_L", "readout": "leg_L", "stim_mean": 3.1, "final_mean": 3.1, "pre_mean": 0.0},
                {"population": "DNa02_L", "readout": "leg_R", "stim_mean": 0.1, "final_mean": 0.1, "pre_mean": 0.0},
                {"population": "DNa02_L", "readout": "leg_LR", "stim_mean": 3.0, "final_mean": 3.0, "pre_mean": 0.0},
                {"population": "PFL3_L", "readout": "type.DNa02_R", "stim_mean": 22.6, "final_mean": 22.6, "pre_mean": 0.0},
                {"population": "PFL3_L", "readout": "type.DNa02_L", "stim_mean": 0.0, "final_mean": 0.0, "pre_mean": 0.0}]
        for r in rows:                                   # the per-run values the scatter is quoted from
            r["stim_values"] = [r["stim_mean"] - 1.0, r["stim_mean"], r["stim_mean"] + 1.0]
            r["stim_sd"] = 1.0
        v = A.validate(pd.DataFrame(rows), runs)
        self.assertEqual(v["status"], "reproduced")
        self.assertEqual(v["measured"]["wind.DNp18_flip_hz"]["flip_per_run_hz"], [45.0, 45.0, 45.0])
        self.assertEqual(v["measured"]["wind.DNp18_flip_hz"]["flip_sd_hz"], 0.0)
        self.assertEqual(v["measured"]["wind.DNp18_flip_hz"]["wind_head_on_LR_hz"], None)
        self.assertEqual(v["measured"]["DNa02_L_150Hz"]["leg_asym_sd_hz"], 1.0)
        self.assertAlmostEqual(v["measured"]["wind.DNp18_flip_hz"]["flip_hz"], 45.0)
        self.assertAlmostEqual(v["measured"]["wind.DNp33_flip_hz"]["flip_hz"], -49.0)
        self.assertEqual(v["measured"]["wind.DNp18_flip_hz"]["reference"], ref["wind.DNp18_flip_hz"])
        self.assertAlmostEqual(v["measured"]["DNa02_L_150Hz"]["leg_asym_hz"], 3.0)
        self.assertAlmostEqual(v["measured"]["PFL3_L_80Hz"]["DNa02_R_hz"], 22.6)
        self.assertEqual(v["measured"]["n_runs"], 3)
        # a flip of the wrong sign is a finding, not a pass
        flipped = pd.DataFrame(rows)
        flipped.loc[flipped.population == "JO_wind_left", "stim_mean"] *= -1
        self.assertIn(A.validate(flipped, runs)["status"], ("not reproduced", "partly reproduced"))
        self.assertEqual(A.validate(pd.DataFrame([{"population": "x", "readout": "y", "stim_mean": 0.0,
                                                   "final_mean": 0.0, "pre_mean": 0.0}]), runs)["status"], "not run")


class PathsTests(unittest.TestCase):
    """paths (flyverse/interp/paths.py): the type-level link statistic and walk gains against the effective weights on
    graph(), the silent-link rules (sign 0 with raw counts, frozen / pruned, never_firing from a Recording), the
    target-input table, the exact k-best walks against a brute-force enumeration on a random typed graph, the cell
    level with the two-step table and cx_wedge's wedge aggregation on a synthetic 16-wedge ring, the Result schema
    round trip and the CLI on a saved cache."""

    @staticmethod
    def _params():
        from flyverse.brain import LIFParams
        return LIFParams(receptor_model=None, event_driven=False)

    @staticmethod
    def _counts(c):
        """Raw counts of graph() including the sign-0 GLNO -> PEN_a entry (200), which |W| stores as an explicit zero."""
        C = abs(c.W).tocsr().copy(); C[7, 6] = 200.0
        return C.tocsr()

    def test_type_level_walk_matches_effective_weights(self):
        from flyverse.interp import paths as P
        c, p = graph(), self._params()
        ew = common.effective_weights(c, p)
        res = P.paths(c, "Mi4", "DNa02", params=p, k_max=3, top=5, frozen=[1], counts=self._counts(c))
        pt = res.table("paths")
        self.assertEqual(len(pt), 1)                                   # the only walk: Mi4 -> LC4 -> DNp01 -> DNa02_L (k = 3)
        w = pt.iloc[0]
        self.assertEqual((int(w.k), w.path), (3, "a:Mi4 -> LC4 -> DNp01 -> b"))
        A = ew.A.toarray()
        gain = A[2, 1] * A[3, 2] * (A[4, 3] + A[5, 3]) / 2            # M[b, DNp01] is the mean over the two DNa02 cells
        self.assertAlmostEqual(w.gain_if_signed, gain, places=6)
        self.assertEqual(w.signs, "-,+,+")
        self.assertEqual(w.kind, "silent")                             # Mi4 is a frozen (pruned) rate unit: the link carries nothing in the LIF
        self.assertEqual(w.gain, 0.0)
        self.assertIn("a:Mi4->LC4:frozen|pruned", w.silent_links)
        lt = res.table("links").set_index(["pre", "post"])
        self.assertEqual(lt.loc[("a:Mi4", "LC4")].share_of_post_input, 1.0)
        self.assertAlmostEqual(lt.loc[("DNp01", "b")].mv_per_pair, A[4, 3], places=6)
        self.assertAlmostEqual(lt.loc[("DNp01", "b")].mv_per_post_volley, A[4, 3] / 2, places=6)
        self.assertEqual(lt.loc[("LC4", "DNp01")].raw_count, 120.0)   # raw, uncapped
        self.assertAlmostEqual(lt.loc[("LC4", "DNp01")].mv_per_pair, 60 * 3 * 2 * p.w_syn * ew.scale[3], places=5)   # capped, gained
        bi = res.table("b_inputs").set_index("pre_type")
        self.assertEqual(list(bi.index), ["DNa02", "DNp01"])           # DNa02_L -> DNa02_R (20) outranks DNp01 -> DNa02_L (10)
        self.assertAlmostEqual(bi.loc["DNp01"].share_of_b_input, 10 / 30, places=6)
        self.assertEqual(res.summary["direct"]["raw_count"], 0.0)
        self.assertEqual(res.summary["b_raw_input_total"], 30.0)
        self.assertEqual(res.check(), [])
        self.assertEqual(res.provenance["execution"]["device"], "cpu")
        with tempfile.TemporaryDirectory() as d:
            path = res.save(Path(d) / "p.json"); back = common.Result.load(path)
        self.assertEqual(back.table("paths").iloc[0].path, w.path)
        self.assertEqual(back.summary["strongest_silent_link_per_k"]["3"]["pre"], "a:Mi4")

    def test_sign0_loop_and_never_firing(self):
        from flyverse.interp import paths as P
        c, p = graph(), self._params()
        ew = common.effective_weights(c, p)
        C = self._counts(c)
        res = P.paths(c, "PEN_a", "PEN_a", params=p, k_max=2, top=5, counts=C)
        pt = res.table("paths")
        self.assertEqual(pt.kind.tolist(), ["silent"]); self.assertEqual(pt.iloc[0].path, "a:PEN_a -> GLNO -> b")
        if_signed = p.w_syn * ew.scale[7] * min(200.0, p.conn_cap)        # what the GLNO -> PEN_a entry would carry if signed
        self.assertAlmostEqual(pt.iloc[0].gain_if_signed, ew.A[6, 7] * if_signed, places=6)
        self.assertEqual(pt.iloc[0].gain, 0.0); self.assertEqual(pt.iloc[0].signs, "+,0")
        self.assertEqual(pt.iloc[0].silent_links, "GLNO->b:sign0")
        bi = res.table("b_inputs").set_index("pre_type")
        self.assertEqual(list(bi.index), ["GLNO", "LC4"])
        g = bi.loc["GLNO"]
        self.assertEqual((int(g.entries), g.raw_count, int(g.sign), g.silent), (1, 200.0, 0, "sign0"))
        self.assertAlmostEqual(g.share_of_b_input, 200 / 205, places=6)
        self.assertAlmostEqual(g.mv_per_pair_if_signed, if_signed, places=6)
        self.assertEqual(g.mv_per_post_volley, 0.0)
        self.assertEqual(res.summary["dominant_silent_input_of_b"]["pre_type"], "GLNO")
        self.assertAlmostEqual(res.summary["b_sign0_input_share"], 200 / 205, places=6)
        s = res.summary["strongest_silent_link_per_k"]
        self.assertIsNone(s[1]); self.assertEqual((s[2]["pre"], s[2]["post"], s[2]["silent"]), ("GLNO", "b", "sign0"))
        self.assertAlmostEqual(s[2]["mv_per_post_volley_if_signed"], if_signed, places=6)
        # without raw counts for the explicit zero the link is invisible (count 0): the walk disappears
        none = P.paths(c, "PEN_a", "PEN_a", params=p, k_max=2, top=5, counts=abs(c.W).tocsr())
        self.assertEqual(len(none.table("paths")), 0)
        # never_firing from a Recording: LC4 silent over the rollout makes LC4 -> PEN_a a silent walk
        rec = common.Recording(t_ms=np.array([0.0, 10.0]), idx=np.arange(8), body_ids=c.neurons.bodyId.to_numpy(), types=np.array(c.neurons.type),
                               quantities={"rate_hz": np.array([[5, 5, 0.0, 5, 5, 5, 5, 5], [5, 5, 0.2, 5, 5, 5, 5, 5]], np.float32)}, motor={}, meta={})
        r2 = P.paths(c, "LC4", "PEN_a", params=p, k_max=1, top=5, counts=C, recording=rec)
        w = r2.table("paths").iloc[0]
        self.assertEqual((w.kind, w.silent_links), ("silent", "a:LC4->b:never_firing"))
        self.assertEqual(r2.table("b_inputs").set_index("pre_type").loc["LC4"].silent, "never_firing")
        r3 = P.paths(c, "LC4", "PEN_a", params=p, k_max=1, top=5, counts=C, recording=rec, rates_min_hz=0.1)
        self.assertEqual(r3.table("paths").iloc[0].kind, "signed")
        # contributions carry the Neurome fields, decimal-string bodies and the if-signed value of the sign-0 entry
        ct = res.table("contributions")
        self.assertTrue(set(common.EXPORT_TABLES["contributions"]) <= set(ct.columns))
        z = ct[(ct.body_pre == "1007") & (ct.body_post == "1008")].iloc[0]
        self.assertEqual((z.value, z.synaptic_pair_count, z.silent), (0.0, 200.0, "sign0"))
        self.assertAlmostEqual(z.value_if_signed, if_signed, places=6)

    @staticmethod
    def _random_graph(seed=3, n=40, n_types=8):
        rng = np.random.default_rng(seed)
        types = [f"T{i}" for i in range(n_types)]
        nts = ["acetylcholine", "gaba", "glutamate", "acetylcholine", "unknown", "acetylcholine", "gaba", "acetylcholine"]
        ty = rng.integers(0, n_types, n)
        nt = [nts[t] for t in ty]
        neurons = pd.DataFrame({"bodyId": np.arange(5000, 5000 + n, dtype=np.int64), "type": [types[t] for t in ty], "instance": [f"{types[t]}_x" for t in ty],
                                "superclass": ["cb_intrinsic"] * n, "class": [""] * n, "subclass": [""] * n, "somaSide": ["L"] * n, "nt": nt,
                                "sign": np.array([cn.NT_SIGN[x] for x in nt], dtype=np.float32)})
        cnt = (rng.random((n, n)) < 0.18) * rng.integers(1, 90, (n, n)); np.fill_diagonal(cnt, 0)
        coo = sp.coo_matrix(cnt.astype(np.float32))
        sign = neurons.sign.to_numpy()
        W = sp.csr_matrix((coo.data * sign[coo.col], (coo.row, coo.col)), shape=(n, n), dtype=np.float32); W.sort_indices()
        C = sp.csr_matrix((coo.data, (coo.row, coo.col)), shape=(n, n), dtype=np.float32); C.sort_indices()
        return Connectome(neurons, W, pd.Series(np.arange(n), index=neurons.bodyId)), C

    @staticmethod
    def _brute_force(g, k_max, K):
        from flyverse.interp import paths as P
        v, u, w, sgn, sil = P.edges(g)
        out_edges = {}
        for i in range(len(v)):
            out_edges.setdefault(int(u[i]), []).append((int(v[i]), float(w[i]), int(sil[i])))
        is_mid, is_dst = set(g.mid.tolist()), set(g.dst.tolist())
        found = {(k, s): [] for k in range(1, k_max + 1) for s in (0, 1)}

        def walk(node, depth, gain, state, nodes):
            for nv, nw, ns in out_edges.get(node, []):
                if nv in is_dst:
                    found[(depth + 1, state | ns)].append((gain * nw, nodes + [nv]))
                if nv in is_mid and depth + 1 < k_max:
                    walk(nv, depth + 1, gain * nw, state | ns, nodes + [nv])
        for s in g.src:
            walk(int(s), 0, 1.0, 0, [int(s)])
        return {key: (sorted(vals, key=lambda x: -x[0])[:K], [x[0] for x in vals]) for key, vals in found.items()}

    def test_kbest_walks_match_brute_force(self):
        from flyverse.interp import paths as P
        p = self._params()
        for level, seed in (("type", 3), ("type", 4), ("cell", 3)):
            c, C = self._random_graph(seed)
            ew = common.effective_weights(c, p)
            a, b = common.resolve(c, "T0"), common.resolve(c, "T7")
            g = P.build_graph(c, ew, a, b, counts=C, k_max=3, level=level)
            walks = P.kbest_walks(g, 3, 4)
            ref = self._brute_force(g, 3, 4)
            n_checked = 0
            for k in (1, 2, 3):
                for state, kind in ((0, "signed"), (1, "silent")):
                    got = [w for w in walks[k] if w["kind"] == kind]
                    exp, all_gains = ref[(k, state)]
                    self.assertEqual(len(got), len(exp), f"{level} seed {seed} k={k} {kind}")
                    np.testing.assert_allclose([w["abs_gain"] for w in got], [e[0] for e in exp], rtol=1e-6, err_msg=f"{level} seed {seed} k={k} {kind}")
                    for w, e in zip(got, exp):
                        if e[0] > 0 and sum(1 for x in all_gains if abs(x - e[0]) <= 1e-9 * e[0]) == 1:      # unique gain: the walk itself must match
                            self.assertEqual(w["nodes"], e[1])
                        gain = np.prod([w2["mv_per_post_volley_if_signed"] for w2 in w["links"]])
                        self.assertAlmostEqual(abs(gain), w["abs_gain"], places=6)
                        self.assertEqual(int(np.sign(gain)) if gain != 0 else 1, w["sign_prod"])
                    n_checked += len(got)
            self.assertGreater(n_checked, 10)
            # a silent walk has at least one silent link and the graph has sign-0 (T4, unknown) links to find
            silent = [w for k in (1, 2, 3) for w in walks[k] if w["kind"] == "silent"]
            self.assertTrue(all(any(L["silent"] for L in w["links"]) for w in silent))
            self.assertTrue(any(L["silent"] == "sign0" for w in silent for L in w["links"]))
        # min_abs_mv drops links, exclude drops types
        c, C = self._random_graph(3); ew = common.effective_weights(c, p)
        full = P.paths(c, "T0", "T7", params=p, k_max=2, top=3, counts=C)
        cut = P.paths(c, "T0", "T7", params=p, k_max=2, top=3, counts=C, exclude=("T1", "~^T[23]$"))
        self.assertFalse(any(t in " ".join(cut.table("paths").path) for t in (" T1 ", " T2 ", " T3 ")))
        self.assertGreaterEqual(len(full.table("paths")), len(cut.table("paths")))
        strong = P.paths(c, "T0", "T7", params=p, k_max=2, top=3, counts=C, min_abs_mv=5.0)
        lt = strong.table("links")
        if len(lt):
            self.assertTrue((lt.mv_per_post_volley_if_signed.abs() >= 5.0).all())

    @staticmethod
    def _ring(count_ep=30, count_pe=30):
        """A synthetic 16-wedge ring: EPG_i -> PEN_i and PEN_i -> EPG_{i +- 1}, PB glomeruli in the instance names in
        scripts/cx_wedge.py's RING16 order (wedge index = ring position), so the wedge aggregation is checkable by hand."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import cx_wedge
        ring = cx_wedge.RING16
        n = 32
        types = ["EPG"] * 16 + ["PEN_a(PEN1)"] * 16
        inst = [f"EPG_{g}_L" for g in ring] + [f"PEN_a(PEN1)_{g}_L" for g in ring]
        neurons = pd.DataFrame({"bodyId": np.arange(7000, 7000 + n, dtype=np.int64), "type": types, "instance": inst, "superclass": ["cb_intrinsic"] * n,
                                "class": [""] * n, "subclass": [""] * n, "somaSide": ["L"] * n, "nt": ["acetylcholine"] * n, "sign": np.ones(n, np.float32)})
        cnt = np.zeros((n, n), np.float32)
        for i in range(16):
            cnt[16 + i, i] = count_ep                                  # EPG_i -> PEN_i
            cnt[(i + 1) % 16, 16 + i] = count_pe                       # PEN_i -> EPG_{i+1}
            cnt[(i - 1) % 16, 16 + i] = count_pe                       # PEN_i -> EPG_{i-1}
        W = sp.csr_matrix(cnt); W.sort_indices()
        return Connectome(neurons, W, pd.Series(np.arange(n), index=neurons.bodyId)), W.copy()

    def test_cell_level_two_step_and_wedge_profile(self):
        from flyverse.interp import paths as P
        p = self._params()
        c, C = self._ring()
        ew = common.effective_weights(c, p); A = ew.A.toarray()
        res = P.paths(c, "EPG", "EPG", params=p, k_max=2, top=3, level="cell", wedge=True, counts=C, wedge_groups={"PEN": r"^PEN_"})
        pt = res.table("paths")
        self.assertTrue((pt.k == 2).all()); self.assertTrue((pt.kind == "signed").all())
        a_ep, a_pe = A[16, 0], A[1, 16]                                # EPG_0 -> PEN_0, PEN_0 -> EPG_1 (all equal by symmetry)
        np.testing.assert_allclose(pt.gain.to_numpy(), a_ep * a_pe, rtol=1e-6)
        self.assertTrue(pt.path.str.startswith("EPG#").all())
        ts = res.table("two_step_by_type").set_index("type")
        self.assertEqual(list(ts.index), ["PEN_a(PEN1)"])
        self.assertAlmostEqual(ts.iloc[0].two_step_total_per_post_mv2, 2 * a_ep * a_pe, places=6)   # each EPG hears two PEN neighbours
        self.assertAlmostEqual(ts.iloc[0].a_to_type_mv_per_pair, a_ep, places=6)
        self.assertAlmostEqual(ts.iloc[0].type_to_b_mv_per_pair, a_pe, places=6)
        wp = res.table("wedge_profile").set_index("group")
        self.assertEqual(set(wp.index), {"direct", "PEN"})
        prof = [wp.loc["PEN", f"d{j}"] for j in range(9)]
        self.assertAlmostEqual(prof[1], a_ep * a_pe, places=6)         # the two-step lands one wedge away, nowhere else
        for j in (0, 2, 3, 4, 5, 6, 7, 8):
            self.assertAlmostEqual(prof[j], 0.0, places=9)
        self.assertTrue(all(abs(wp.loc["direct", f"d{j}"]) < 1e-12 for j in range(9)))
        M16 = np.array(res.files["M16"]["PEN"])
        self.assertEqual(M16.shape, (16, 16)); self.assertAlmostEqual(M16[1, 0], a_ep * a_pe, places=2); self.assertEqual(M16[2, 0], 0.0)   # M16 is stored to 3 decimals
        self.assertEqual(res.summary["wedge"]["dropped"], {"a": 0, "b": 0})
        # the synonym convenience: 'PEN_a' selects 'PEN_a(PEN1)' as 'PEN_a|PEN_b' does in the CLI
        np.testing.assert_array_equal(P.resolve_loose(c, "PEN_a"), np.arange(16, 32))
        np.testing.assert_array_equal(P.resolve_loose(c, "PEN_a|PEN_b"), np.arange(16, 32))
        np.testing.assert_array_equal(P.resolve_loose(c, ["~^PEN_a", "~^PEN_b"]), np.arange(16, 32))
        one = P.paths(c, "PEN_a", "EPG", params=p, k_max=1, top=3, counts=C)
        d = one.summary["direct"]
        self.assertEqual((d["pairs"], d["n_pre_cells"]), (32, 16)); self.assertAlmostEqual(d["mv_per_pair"], a_pe, places=6)
        self.assertAlmostEqual(d["mv_per_post_volley"], 2 * a_pe, places=6)

    def test_cli_on_saved_cache(self):
        import importlib
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        cli = importlib.import_module("interp_paths")
        c = graph()
        with tempfile.TemporaryDirectory() as d:
            save(c, Path(d) / "cache")
            out = Path(d) / "r.json"
            rc = cli.main(["--a", "Mi4", "--b", "DNa02", "--k", "3", "--cache-dir", str(Path(d) / "cache"), "--json", str(out), "--quiet",
                           "--receptor-model", "off", "--frozen", "static", "--lif", "event_driven=False"])
            self.assertEqual(rc, 0)
            res = common.Result.load(out)
        self.assertEqual(res.tool, "paths"); self.assertEqual(res.check(), [])
        self.assertEqual(res.table("paths").iloc[0].path, "a:Mi4 -> LC4 -> DNp01 -> b")
        self.assertIn("frozen", res.table("paths").iloc[0].silent_links)   # 'static': Mi4 (ol_intrinsic) is a rate unit
        self.assertEqual(res.provenance["model"]["lif"]["receptor_model"], None)
        self.assertTrue(res.files["generator"].startswith("scripts/interp_paths.py"))
        self.assertEqual(cli.P.spec_from_cli("LNO1|LNO2|~^LAL"), ["LNO1", "LNO2", "~^LAL"])
        self.assertEqual(cli.P.spec_from_cli("LC10a|LC11"), "LC10a|LC11")


class ExportTests(unittest.TestCase):
    """flyverse/interp/export.py -- the Neurome read-only probe export (docs/NEUROME_INTERFACE.md section 1) as a
    serializer over common.Result: the manifest's mandatory provenance, decimal bodyIds, one SHA-256 per table, the
    retina record (columns / photoreceptor bodies / radiance), the two-rows-per-body rule for LC11 and LC10a, the raw
    uncapped synapse count Neurome joins on, the structural `contributions` path and the object-sweep adapter's arms.
    CPU, the synthetic graph of `graph()`, no dataset and no GPU (docs/audits/interp_export.md)."""

    # ----------------------------------------------------------------------------------------------- fixtures
    def _prov(self, c, stimulus=None):
        from flyverse.brain import LIFParams
        return common.to_jsonable(common.provenance(
            c, LIFParams(receptor_model=None), fb=fake_fb(c, np.zeros(8)), seeds=[0],
            stimulus=stimulus or {"protocol": "toy", "params": {"hz": 1.0}, "control": "none"}))

    def _readout_rows(self, spec=(("LC11", (1004, 1005)), ("LC10a", (1006,))),
                      quantities=("upstream_drive_mV", "output_Hz")):
        rows = []
        for ty, bodies in spec:
            for i, b in enumerate(bodies):
                for q in quantities:
                    rows.append({"bodyId": int(b), "model_index": 3 + i, "type": ty, "unit_kind": "spiking",
                                 "quantity": q, "window_start_s": 3.0, "window_end_s": 15.0,
                                 "stimulus_value": 1.0 + i, "control_value": 0.5, "stimulus_minus_control": 0.5 + i,
                                 "unit": "mV" if q == "upstream_drive_mV" else "Hz", "n_trials": 3, "trial_sd": 0.125,
                                 "control_ids": "", "verdict": "null"})
        return rows

    def _result(self, c, **kw):
        res = common.Result.new("export", self._prov(c))
        res.add_table("readout_per_body", self._readout_rows(**kw))
        res.add_table("per_type", [{"type": "LC11", "z": 1.5}, {"type": "LC10a", "z": -0.25}])
        return res

    def _neurons(self, d, c):
        p = Path(d) / "neurons.parquet"
        pd.DataFrame({"bodyId": c.neurons.bodyId.to_numpy()}).to_parquet(p)
        return p

    def _sweep_json(self, path, values, n_cells=2, stat="diff_max_over_cells_mean_mv"):
        payload = {"config": {"mode": "off", "seed": 0, "seconds": 12.0, "settle": 3.0, "angular_diameter_deg": 11.4,
                              "null": False, "condition_a": "ball", "condition_b": "none"},
                   "ball": {t: {"kind": "spiking", "n_cells": n_cells, stat: float(v)} for t, v in values.items()},
                   "none": {}}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def _cells_npz(self, path, drive, rate, dev=0.25):
        """Two LC11 bodies and one graded Mi4 in the layout `scripts/interp_export.py record` writes."""
        np.savez(path, types=np.array(["LC11", "Mi4"]),
                 body__LC11=np.array([1004, 1005], np.int64), index__LC11=np.array([3, 4], np.int64),
                 unitkind__LC11=np.array(["spiking", "spiking"]),
                 drive_mean__LC11__a=np.array([drive, drive + 1.0], np.float32),
                 drive_mean__LC11__b=np.array([0.0, 1.0], np.float32),
                 rate_hz__LC11__a=np.array([rate, rate], np.float32), rate_hz__LC11__b=np.zeros(2, np.float32),
                 body__Mi4=np.array([1002], np.int64), index__Mi4=np.array([1], np.int64),
                 unitkind__Mi4=np.array(["graded"]),
                 dev_mean__Mi4__a=np.array([dev], np.float32), dev_mean__Mi4__b=np.array([0.2], np.float32))
        return str(path)

    # ----------------------------------------------------------------------------------------------- the export
    def test_manifest_tables_and_round_trip(self):
        from flyverse.interp import export as ex
        c = graph()
        res = self._result(c)
        with tempfile.TemporaryDirectory() as d:
            run = ex.export(res, out_root=Path(d) / "export")
            man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            # the manifest alone names the commit, the four MaleCNS files, the cache fingerprint, the model,
            # the realised device and the stimulus (docs/NEUROME_INTERFACE.md section 1)
            for field in ("run_id", "flyverse_commit", "dataset_release", "compiled_connectome", "model", "execution",
                          "stimulus", "retina", "units", "tables"):
                self.assertIn(field, man)
            self.assertEqual(man["interchange_key"], ["dataset.name", "dataset.release", "bodyId"])
            self.assertEqual(man["execution"]["device"], "cpu")
            self.assertIsNone(man["model"]["lif"]["receptor_model"])
            self.assertEqual(man["compiled_connectome"]["nnz"], 8)
            self.assertEqual(man["compiled_connectome"]["sum_abs_W"], 275.0)
            self.assertEqual(len(man["dataset_release"]["files"]), 4)
            self.assertEqual(len(man["units"]), len(common.UNITS))
            names = {t["name"]: t for t in man["tables"]}
            self.assertEqual(names["readout_per_body"]["role"], "interchange")
            self.assertEqual(names["per_type"]["role"], "tool")          # the tool's own table travels and is hashed
            for t in man["tables"]:                                       # every SHA-256 matches the file
                self.assertEqual(ex.sha256_file(run / t["file"]), t["sha256"])
            self.assertEqual(ex.sha256_file(run / "result.json"), man["source_result"]["sha256"])
            df = ex.read_table(run, "readout_per_body")
            self.assertEqual(list(df.columns)[:5], ["dataset", "release", "bodyId", "model_index", "type"])
            self.assertEqual(set(df.dataset), {common.DATASET_NAME})
            self.assertTrue(all(isinstance(b, str) and b.isdigit() for b in df.bodyId))
            self.assertEqual(names["readout_per_body"]["units"]["window_start_s"], "s")
            # 'null' is a verdict, not a missing value: read_table says so and the manifest declares the convention
            self.assertEqual(list(df.verdict), ["null"] * 6)
            self.assertTrue(pd.read_csv(run / "readout_per_body.csv").verdict.isna().all())   # pandas' defaults eat it
            self.assertIn("keep_default_na", man["conventions"]["missing"])
            trip = ex.round_trip_check(res, run)
            self.assertEqual(trip["readout_per_body"]["rows"], 6)
            self.assertEqual(trip["readout_per_body"]["max_abs_diff"], 0.0)
            self.assertTrue(trip["readout_per_body"]["ids_match"])
            self.assertTrue(trip["readout_per_body"]["text_columns_match"])
            info = ex.verify(run, neurons=self._neurons(d, c), expect_counts={"LC11": 2, "LC10a": 1})
            self.assertEqual(info["problems"], [])
            self.assertEqual((info["LC11_bodies"], info["LC10a_bodies"]), (2, 1))
            # a table edited after the export is caught by its hash
            p = run / names["readout_per_body"]["file"]
            p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            self.assertTrue(any("SHA-256" in s for s in ex.verify(run)["problems"]))

    def test_refusals_and_the_paired_quantity_rule(self):
        from flyverse.interp import export as ex
        c = graph()
        with tempfile.TemporaryDirectory() as d:
            bad = common.Result.new("export", {"model": {}})          # no provenance, no realised device
            with self.assertRaises(ValueError) as e:
                ex.export(bad, out_root=d)
            self.assertIn("refuses", str(e.exception))
            # LC11 / LC10a must carry BOTH quantities for every body, never pooled
            half = common.Result.new("export", self._prov(c))
            half.add_table("readout_per_body", self._readout_rows(spec=(("LC11", (1004,)),), quantities=("output_Hz",)))
            run = ex.export(half, out_root=Path(d) / "export")
            problems = ex.verify(run, expect_paired=("LC11", "LC10a"))["problems"]
            self.assertTrue(any("1 LC11 bodies lack both" in s for s in problems))
            self.assertTrue(any("no rows for LC10a" in s for s in problems))
            self.assertEqual(ex.verify(run, expect_paired=("LC11",), expect_counts={"LC11": 2})["problems"][0][:40],
                             "readout_per_body: 1 LC11 bodies lack bot"[:40])
            # an id outside the connectome is a problem, not a silent row
            self.assertTrue(any("not in the connectome" in s
                                for s in ex.verify(run, neurons=np.array([1, 2, 3]))["problems"]))

    def test_retina_tables(self):
        from flyverse.interp import export as ex
        z = {"radiance": np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4), "t_s": np.array([0.0, 0.01]),
             "ball_offset_m": np.array([0.0, 0.01]), "col_dir": np.eye(3, dtype=np.float32),
             "col_az_el": np.zeros((3, 2)), "col_hex": np.zeros((3, 2), np.int64),
             "col_side": np.array(["L", "L", "R"]), "pr_index": np.array([0, 1, 2], np.int64),
             "pr_body": np.array([1001, 1002, 1003], np.int64), "pr_column": np.array([0, 0, 2], np.int64),
             "pr_sens": np.ones((3, 4), np.float32), "pr_type": np.array(["R1-R6", "R1-R6", "R7"])}
        with tempfile.TemporaryDirectory() as d:
            metas, block = ex.retina_tables(z, d)
            self.assertEqual([m["name"] for m in metas], ["retina_columns", "retina_bodies", "retina_radiance"])
            cols = pd.read_csv(Path(d) / "retina_columns.csv")
            self.assertEqual(len(cols), 3)
            self.assertEqual(list(cols.n_photoreceptors), [2, 0, 1])
            self.assertEqual(cols.photoreceptor_bodies[0], "1001|1002")   # the column -> body map Neurome asked for
            bodies = pd.read_csv(Path(d) / "retina_bodies.csv", dtype={"bodyId": str})
            self.assertEqual(list(bodies.bodyId), ["1001", "1002", "1003"])
            self.assertEqual(set(bodies.unit_kind), {"photoreceptor"})
            self.assertEqual(list(bodies.column_id), [0, 0, 2])
            rad = pd.read_csv(Path(d) / "retina_radiance.csv")
            self.assertEqual(len(rad), 2 * 3)                              # long: frame x column, four channels wide
            self.assertEqual(list(rad.radiance_uv[:3]), [0.0, 4.0, 8.0])
            self.assertEqual(list(rad.columns)[-1], "ball_offset_m")
            self.assertEqual((block["n_columns"], block["n_photoreceptors"], block["n_frames"]), (3, 3, 2))
            self.assertEqual(block["shape"], [2, 3, 4])
            self.assertEqual(block["channels"], ["UV", "B", "G", "R"])
            self.assertEqual(ex.retina_tables(None, d)[1]["n_columns"], None)
            # above parquet_rows a table is Parquet, and reads back identically
            meta = ex.write_table(pd.DataFrame({"a": [1.5, 2.5]}), Path(d), "big", parquet_rows=1)
            self.assertEqual(meta["format"], "parquet")
            self.assertEqual(list(pd.read_parquet(Path(d) / meta["file"]).a), [1.5, 2.5])

    # ------------------------------------------------------------------------------- the contributions path
    def test_raw_counts_keep_the_uncapped_synapse_count(self):
        from unittest import mock
        from flyverse.interp import export as ex
        c = graph()
        C, ok = ex.raw_counts(c)
        self.assertFalse(ok)                                   # no cache/sign0_counts.npz for a synthetic graph
        self.assertEqual(C[3, 2], 120.0)                       # LC4 -> DNp01: the raw count, above the cap of 60
        self.assertEqual(C[7, 6], 0.0)                         # GLNO -> PEN_a is an explicit zero in W
        W = c.W.tocsr()
        a, b = W.indptr[7], W.indptr[8]
        s0 = np.zeros(W.nnz, np.float32)
        s0[a + int(np.flatnonzero(W.indices[a:b] == 6)[0])] = 200.0
        with mock.patch.object(cn, "sign0_counts", lambda *args, **kw: s0):
            C2, ok2 = ex.raw_counts(c)
        self.assertTrue(ok2)
        self.assertEqual(C2[7, 6], 200.0)                       # the sign-0 entry carries its real count
        self.assertEqual(C2[3, 2], 120.0)                       # and every ordinary entry is unchanged
        self.assertEqual(float(C2.sum()), float(abs(c.W).sum()) + 200.0)

    def test_static_decompose_result_and_its_export(self):
        from unittest import mock
        from flyverse.interp import export as ex
        c = graph()
        W = c.W.tocsr()
        a, b = W.indptr[7], W.indptr[8]
        s0 = np.zeros(W.nnz, np.float32)
        s0[a + int(np.flatnonzero(W.indices[a:b] == 6)[0])] = 200.0
        with mock.patch.object(cn, "sign0_counts", lambda *args, **kw: s0):
            res = ex.static_decompose_result(c, "PEN_a")
        self.assertEqual(res.check(), [])
        con = res.table("contributions")
        for col in common.EXPORT_TABLES["contributions"]:
            self.assertIn(col, con.columns)
        glno = con[con.pre_type == "GLNO"].iloc[0]
        self.assertEqual((glno.body_pre, glno.body_post), ("1007", "1008"))
        self.assertEqual(glno.value, 0.0)                       # sign 0: the link exists and carries nothing
        self.assertEqual(glno.synaptic_pair_count, 200.0)
        self.assertIn("sign0", glno.silent)
        self.assertNotIn("never_firing", glno.silent)           # not evaluated without a rollout
        lc4 = con[con.pre_type == "LC4"].iloc[0]
        self.assertGreater(lc4.value, 0.0)
        self.assertEqual(lc4.synaptic_pair_count, 5.0)
        self.assertEqual(lc4.kind, "effective_weight_mV")
        self.assertEqual(lc4.sign_rule, "nt_sign")
        self.assertIn("conn_cap", lc4.gain_rule)
        per = res.table("per_type")
        self.assertAlmostEqual(float(per[per.pre_type == "LC4"].share_of_raw_input.iloc[0]), 5 / 205, places=6)
        self.assertEqual(res.summary["n_post_cells"], 1)
        self.assertEqual(res.summary["sign0_entries"], 1)
        self.assertEqual(res.summary["sign0_raw_synapses"], 200.0)
        with tempfile.TemporaryDirectory() as d:
            run = ex.export(res, out_root=Path(d) / "export")
            info = ex.verify(run, neurons=self._neurons(d, c), expect_paired=())
            self.assertEqual(info["problems"], [])
            self.assertEqual(info["tables"]["contributions"], len(con))
            self.assertEqual(ex.round_trip_check(res, run)["contributions"]["max_abs_diff"], 0.0)

    def test_source_fingerprint_names_the_commit_of_a_cluster_run(self):
        from flyverse.interp import export as ex
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "flyverse").mkdir()
            (d / "flyverse" / "brain.py").write_text("x = 1\n", encoding="utf-8")
            (d / "flyverse" / "optic.py").write_text("y = 2\n", encoding="utf-8")
            fp = ex.source_fingerprint(d, patterns=("flyverse/*.py",))
            self.assertEqual(fp["n_files"], 2)
            self.assertEqual(set(fp["files"]), {"flyverse/brain.py", "flyverse/optic.py"})
            self.assertEqual(fp["files"]["flyverse/brain.py"], ex.sha256_file(d / "flyverse" / "brain.py"))
            self.assertIn("commit", fp["git"])
            # the same tree: every file identical, so the run is pinned to this checkout's commit
            m = ex.match_sources(fp, d)
            self.assertTrue(m["verified"]); self.assertEqual((m["n_recorded"], m["n_identical"]), (2, 2))
            self.assertEqual((m["differ"], m["missing_here"]), ([], []))
            # one file changed and one gone: not verified, and both are named
            (d / "flyverse" / "brain.py").write_text("x = 2\n", encoding="utf-8")
            (d / "flyverse" / "optic.py").unlink()
            m2 = ex.match_sources(fp, d)
            self.assertFalse(m2["verified"])
            self.assertEqual((m2["differ"], m2["missing_here"]), (["flyverse/brain.py"], ["flyverse/optic.py"]))
            self.assertFalse(ex.match_sources(None, d)["verified"])       # a run that recorded no hashes proves nothing
            # the same source with the other line endings is the same source: the cluster tree is LF, a Windows
            # working tree CRLF, and only files that differ from origin/main are shipped (scripts/cluster_run.py)
            (d / "flyverse" / "brain.py").write_bytes(b"x = 1\nz = 3\n")
            lf = ex.source_fingerprint(d, patterns=("flyverse/brain.py",))
            (d / "flyverse" / "brain.py").write_bytes(b"x = 1\r\nz = 3\r\n")
            crlf = ex.source_fingerprint(d, patterns=("flyverse/brain.py",))
            self.assertNotEqual(lf["files"], crlf["files"])               # the raw hashes differ
            self.assertEqual(lf["files_lf"], crlf["files_lf"])            # the content hash does not
            self.assertTrue(ex.match_sources(lf, d)["verified"])          # and the match is on content
            (d / "flyverse" / "brain.py").write_bytes(b"x = 1\r\nz = 4\r\n")
            self.assertFalse(ex.match_sources(lf, d)["verified"])         # a real edit is still a difference
            # the verdict is taken on what the run actually imported, not on a glob of the tree: a sibling module
            # edited while the job ran cannot have changed its numbers
            fp2 = ex.source_fingerprint(d, patterns=("flyverse/*.py",))
            fp2["files_loaded"] = {"flyverse/brain.py": ex.sha256_file(d / "flyverse" / "brain.py")}
            (d / "flyverse" / "sibling.py").write_text("never imported\n", encoding="utf-8")
            fp2["files"]["flyverse/sibling.py"] = "0" * 64               # a glob entry that does not match
            m3 = ex.match_sources(fp2, d)
            self.assertEqual(m3["scope"], "loaded")
            self.assertTrue(m3["verified"])
            self.assertFalse(m3["glob_scope"]["verified"])
            self.assertEqual(m3["glob_scope"]["differ"], ["flyverse/sibling.py"])
            self.assertIn("flyverse/interp/export.py", ex.loaded_sources())   # this process imported the module

    # ------------------------------------------------------------------------------- the object-sweep adapter
    def test_object_sweep_result_arms_and_readouts(self):
        from flyverse.interp import export as ex
        c = graph()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            stim = [self._sweep_json(d / f"stim_s{i}.json", {"LC11": v, "LC10a": 0.05 + 0.01 * i})
                    for i, v in enumerate((10.0, 10.5, 11.0))]
            null = [self._sweep_json(d / f"null_s{i}.json", {"LC11": v, "LC10a": 0.06 + 0.01 * i})
                    for i, v in enumerate((0.0, 0.1, 0.2))]
            ref = [self._sweep_json(d / f"ref_s{i}.json", {"LC11": v, "LC10a": 0.07}) for i, v in enumerate((9.0, 9.5))]
            refn = [self._sweep_json(d / f"refn_s{i}.json", {"LC11": v, "LC10a": 0.07}) for i, v in enumerate((0.0, 0.3))]
            cells = [self._cells_npz(d / f"stim_s{i}_cells.npz", 2.0 + i, 3.0) for i in range(3)]
            ncells = [self._cells_npz(d / f"null_s{i}_cells.npz", 0.5, 0.25) for i in range(3)]
            res = ex.result_from_object_sweep(stim, null, cells=cells, null_cells=ncells, provenance=self._prov(c),
                                              reference=ref, reference_null=refn, control_ids=["null_s0"])
            self.assertEqual(res.check(), [])
            per = res.table("per_type")
            r = per[(per.type == "LC11") & (per.statistic == "diff_max_over_cells_mean_mv")].iloc[0]
            self.assertEqual((r.stim_n, r.null_n), (3, 3))
            self.assertAlmostEqual(r.stim_mean, 10.5); self.assertAlmostEqual(r.null_mean, 0.1)
            self.assertGreater(r.z, 3.0)
            # 3 runs per arm is the floor of the scatter rule but not of the exact test: the smallest two-sided
            # Mann-Whitney p at n = 3, 3 is 0.1, so no separation can be called -- compare says 'underpowered'
            self.assertAlmostEqual(r.p, 0.1); self.assertEqual(r.verdict, "underpowered")
            self.assertEqual(common.compare([10.0, 10.1, 9.9, 10.2, 9.8], [0.0, 0.1, -0.1, 0.2, -0.2])["verdict"], "result")
            # the reference arm is reduced exactly as the reproduction, so the two are comparable row by row
            rep = res.validation["measured"]["reproduction"]["LC11"]
            self.assertAlmostEqual(rep["reference_mean"], 9.25); self.assertAlmostEqual(rep["reproduced_mean"], 10.5)
            self.assertEqual(rep["reference_runs"], [9.0, 9.5])
            self.assertEqual(len(res.provenance["stimulus"]["reference_runs"]), 4)
            self.assertTrue(all(len(x["sha256"]) == 64 for x in res.provenance["stimulus"]["reference_runs"]))
            self.assertEqual(res.replicates["unit"], "runs"); self.assertEqual(res.replicates["n"], 3)
            # two rows per spiking body, one per graded optic rate unit, never pooled
            ro = res.table("readout_per_body")
            self.assertEqual(set(ro[ro.type == "LC11"].quantity), {"upstream_drive_mV", "output_Hz"})
            self.assertEqual(ro[ro.type == "LC11"].bodyId.nunique(), 2)
            drive = ro[(ro.type == "LC11") & (ro.quantity == "upstream_drive_mV") & (ro.bodyId == "1004")].iloc[0]
            self.assertAlmostEqual(drive.stimulus_value, 3.0)              # mean of 2, 3, 4
            self.assertAlmostEqual(drive.stimulus_minus_control, 3.0)      # control 0.0 in every run
            self.assertEqual(drive.n_trials, 3); self.assertAlmostEqual(drive.trial_sd, 1.0)
            self.assertEqual(drive.unit, "mV"); self.assertEqual(drive.unit_kind, "spiking")
            self.assertEqual((drive.window_start_s, drive.window_end_s), (3.0, 15.0))
            self.assertEqual(drive.control_ids, "null_s0")
            mi4 = ro[ro.type == "Mi4"].iloc[0]
            self.assertEqual((mi4.unit_kind, mi4.quantity, mi4.unit), ("graded", "rate_deviation", "rate units [0-1]"))
            run = ex.export(res, out_root=d / "export")
            info = ex.verify(run, neurons=self._neurons(d, c), expect_paired=("LC11",), expect_counts={"LC11": 2})
            self.assertEqual(info["problems"], [])
            self.assertEqual({t["name"] for t in json.loads((run / "manifest.json").read_text(encoding="utf-8"))["tables"]},
                             {"readout_per_body", "per_type", "reference_per_type"})

    def test_cli_analyse_and_verify(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import interp_export as cli
        c = graph()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            res = self._result(c)
            rp = res.save(d / "r.json")
            neurons = self._neurons(d, c)
            self.assertEqual(cli.main(["analyse", "--result", str(rp), "--out", str(d / "export"),
                                       "--neurons", str(neurons), "--control-ids", "a,b"]), 0)
            run = d / "export" / res.run_id
            checks = json.loads((run / "checks.json").read_text(encoding="utf-8"))
            self.assertEqual(checks["verify"]["problems"], [])
            self.assertEqual(checks["round_trip"]["readout_per_body"]["max_abs_diff"], 0.0)
            self.assertEqual(set(pd.read_csv(run / "readout_per_body.csv").control_ids), {"a|b"})
            self.assertEqual(cli.main(["verify", "--run-dir", str(run), "--neurons", str(neurons)]), 0)
            self.assertEqual(cli.main(["verify", "--run-dir", str(run), "--neurons", str(neurons), "--paired", ""]), 0)
            # the bare form of docs/INTERP.md section 7 defaults to `analyse`
            self.assertEqual(cli.main(["--result", str(rp), "--out", str(d / "export"), "--neurons", str(neurons)]), 0)
        self.assertEqual(cli._glob(["out/expobj/nothing_here_*.json"]), [])


class DecomposeTests(unittest.TestCase):
    """flyverse/interp/decompose.py on graph() plus two edges (DNa02_L -> DNp01 and DNa02_L -> LC4) so a target has
    two signed inputs: the static table is the effective-weight matrix, the dynamic one is I = A r frame by frame,
    contrast splits sign / gain changes from fan-in rescaling, the arm convention round-trips, the CLI helpers parse."""

    @staticmethod
    def graph2() -> Connectome:
        c = graph()
        W = c.W.tolil()
        W[3, 4] = 3.0          # DNa02_L -> DNp01 (ACh, +): DNp01 now has an LC4 (x3 type gain, x2 path gain) and a plain input
        W[2, 4] = 6.0          # DNa02_L -> LC4 (ACh, +) next to Mi4 -> LC4 (GABA, -): an E / I pair on LC4
        W = W.tocsr().astype(np.float32); W.sort_indices()
        return Connectome(c.neurons.copy(), W, pd.Series(np.arange(8), index=c.neurons.bodyId.to_numpy()))

    @staticmethod
    def params(**kw):
        from flyverse.brain import LIFParams
        return LIFParams(**{"receptor_model": None, "event_driven": False, **kw})

    def recording(self, c, rates, seed, p, drop=None):
        """A single-fly Recording of every cell (or all but `drop`) with the provenance block a CLI record writes."""
        idx = np.array([i for i in range(8) if i != drop])
        r = np.asarray(rates, np.float32)[:, idx]
        n = c.neurons
        prov = common.provenance(c, p, fb=fake_fb(c, np.zeros(8)), seeds=[seed], stimulus={"protocol": "toy", "params": {}, "control": "off"})
        return common.Recording(10.0 * np.arange(len(r)), idx, n.bodyId.to_numpy()[idx], n.type.to_numpy()[idx].astype(str),
                                {"rate_hz": r}, {}, {"seed": seed, "protocol": "toy", "provenance": prov, "sides": ["L"] * len(idx)})

    def test_static_is_the_effective_weight_table(self):
        from flyverse.interp import decompose as dec
        c, p = self.graph2(), self.params(input_norm_ref=100)
        ew = common.effective_weights(c, p)
        res = dec.decompose(c, "DNp01|LC4|PEN_a", params=p, by=("type",), tiers=True)
        self.assertEqual(res.tool, "decompose"); self.assertEqual(res.check(), []); self.assertEqual(res.validation["status"], "not run")
        t = res.table("per_type").set_index(["post_type", "pre_group"])
        # DNp01: LC4 (cap 60, x3 type gain, x2 path gain) and DNa02 (3 synapses), both through DNp01's fan-in scale and w_syn
        self.assertAlmostEqual(t.loc[("DNp01", "LC4")].value, float(ew.A[3, 2]), places=6)
        self.assertAlmostEqual(t.loc[("DNp01", "DNa02")].value, float(ew.A[3, 4]), places=6)
        self.assertEqual(t.loc[("DNp01", "LC4")].raw_count, 120.0); self.assertAlmostEqual(t.loc[("DNp01", "LC4")].share, 120 / 123)
        self.assertEqual(t.loc[("DNp01", "LC4")].sign, "1"); self.assertEqual(t.loc[("DNp01", "LC4")].sign_rule, "nt_sign")
        self.assertEqual(t.loc[("DNp01", "LC4")].unit, dec.STATIC_UNIT); self.assertEqual(t.loc[("DNp01", "LC4")].kind, dec.STATIC_KIND)
        # LC4: an E / I pair -> the cancelling pair of the summary; PEN_a: GLNO is a sign-0 (silent) entry of weight 0
        self.assertLess(t.loc[("LC4", "Mi4")].value, 0); self.assertGreater(t.loc[("LC4", "DNa02")].value, 0)
        cp = res.summary["static"]["LC4"]["cancelling_pair"]
        self.assertEqual((cp["E"], cp["I"]), ("DNa02", "Mi4")); self.assertAlmostEqual(cp["cancellation"], min(cp["E_value"], -cp["I_value"]))
        self.assertEqual(t.loc[("PEN_a", "GLNO")].value, 0.0); self.assertEqual(t.loc[("PEN_a", "GLNO")].silent_entries, 1)
        self.assertEqual(res.summary["silent_entries"]["sign0"], 1); self.assertEqual(res.summary["n_presynaptic"], 4)
        # the tier table appends the sign rule to the group; other groupings work on the same edge table
        self.assertIn(("DNp01", "LC4/nt_sign"), res.table("per_tier").set_index(["post_type", "pre_group"]).index)
        by_nt = dec.decompose(c, "LC4", params=p, by=("transmitter", "sign"), tiers=False).table("per_type")
        self.assertEqual(set(by_nt.pre_group), {"gaba/I", "acetylcholine/E"})
        with self.assertRaises(ValueError):
            dec.decompose(c, "LC4", params=p, by=("colour",))
        # Neurome tables: the export columns are present, bodyIds are decimal strings, counts raw
        ct = res.table("contributions")
        self.assertTrue(set(common.EXPORT_TABLES["contributions"]) <= set(ct.columns))
        row = ct[(ct.pre_type == "LC4") & (ct.post_type == "DNp01")].iloc[0]
        self.assertEqual((row.body_pre, row.body_post, row.synaptic_pair_count), ("1003", "1004", 120.0))
        rb = res.table("readout_per_body")
        self.assertEqual(set(rb.bodyId), {"1003", "1004", "1008"}); self.assertEqual(rb.quantity.iloc[0], "total_effective_input_mV_per_volley")
        self.assertEqual(res.populations[0]["label"], "target"); self.assertEqual(res.provenance["execution"]["device"], "cpu")
        self.assertTrue(dec.print_per_type(res, top=5))
        with tempfile.TemporaryDirectory() as d:
            path = res.save(Path(d) / "r.json"); back = common.Result.load(path)
        self.assertEqual(back.summary["n_presynaptic"], 4)

    def test_dynamic_is_A_times_rate_per_frame(self):
        from flyverse.interp import decompose as dec
        c, p = self.graph2(), self.params(input_norm_ref=100)
        A = common.effective_weights(c, p).A.toarray()
        rng = np.random.default_rng(0)
        base = np.zeros((6, 8)); base[:, 2] = [10, 20, 30, 40, 30, 20]; base[:, 4] = [5, 5, 8, 8, 5, 5]; base[:, 3] = [0, 1, 4, 9, 2, 1]
        stim = [self.recording(c, base * (1 + 0.1 * k) + rng.uniform(0, 0.5, base.shape), k, p) for k in range(3)]
        null = [self.recording(c, base * 0.5 + rng.uniform(0, 0.5, base.shape), 10 + k, p) for k in range(3)]
        res = dec.decompose(c, "DNp01", recording={"stim": stim}, null_recording={"null": null}, params=p, by=("type",), tiers=True, window=(0.01, 0.05))
        self.assertEqual(res.check(), []); self.assertEqual(res.summary["kind"], "current"); self.assertEqual(res.replicates["n"], 3)
        t = res.table("per_type").set_index("pre_group")
        r0 = stim[0].quantities["rate_hz"][1:5]                                    # the window: frames 1..4 (t_ms 10..40)
        expect_lc4 = [float((A[3, 2] * s.quantities["rate_hz"][1:5, 2]).mean()) for s in stim]
        np.testing.assert_allclose(t.loc["LC4"].stim_values, expect_lc4, rtol=1e-5)
        self.assertAlmostEqual(t.loc["DNa02"].stim_mean, float(np.mean([(A[3, 4] * s.quantities["rate_hz"][1:5, 4]).mean() for s in stim])), places=5)
        self.assertEqual(t.loc["LC4"].unit, dec.DYNAMIC_UNIT); self.assertAlmostEqual(t.loc["LC4"].weight_mv_per_volley, A[3, 2], places=6)
        self.assertAlmostEqual(t.loc["LC4"].stim_g_mv, t.loc["LC4"].stim_mean * p.tau_syn / 1000.0)
        # the arm comparison: z against the null scatter, and -- at exactly three runs -- never 'result': the exact U
        # floors p at 0.1, which compare reports as p_floor and calls 'underpowered'
        self.assertGreater(t.loc["LC4"].stim_z, 3); self.assertEqual(t.loc["LC4"].stim_verdict, "underpowered")
        self.assertAlmostEqual(t.loc["LC4"].stim_p, 0.1)
        two = dec.decompose(c, "DNp01", recording={"stim": stim[:2]}, null_recording={"null": null}, params=p, tiers=False)
        self.assertEqual(two.table("per_type").set_index("pre_group").loc["LC4"].stim_verdict, "underpowered")
        # the peak view: every group's input at the frame of the target's peak output
        k = int(np.argmax(r0[:, 3])); self.assertAlmostEqual(t.loc["LC4"].stim_peak_t_ms, 10.0 * (k + 1))
        np.testing.assert_allclose(t.loc["LC4"].stim_value_at_peak, np.mean([A[3, 2] * s.quantities["rate_hz"][1:5][int(np.argmax(s.quantities["rate_hz"][1:5, 3])), 2] for s in stim]), rtol=1e-5)
        # E / I totals, the time series of the first run, the body-level tables and the coverage record
        dyn = res.summary["dynamic"]["stim"]["DNp01"]
        self.assertAlmostEqual(dyn["E_total"], t.loc["LC4"].stim_mean + t.loc["DNa02"].stim_mean, places=6); self.assertEqual(dyn["I_total"], 0.0)
        ts = res.table("timeseries"); self.assertEqual(ts.t_ms.nunique(), 4); self.assertEqual(set(ts.pre_group), {"LC4", "DNa02"})
        rb = res.table("readout_per_body")
        self.assertEqual(set(rb.quantity), {"input_current_mV_per_s", "output_Hz"}); self.assertEqual(set(rb.arm), {"stim", "null"})
        row = rb[(rb.quantity == "output_Hz") & (rb.arm == "stim")].iloc[0]
        self.assertEqual(row.n_trials, 3); self.assertEqual(row.control_ids, ["null[0]", "null[1]", "null[2]"]); self.assertEqual(row.bodyId, "1004")
        self.assertEqual(res.summary["coverage"]["stim[0]"]["presynaptic_cells_missing"], 0)
        ct = res.table("contributions"); self.assertTrue(set(common.EXPORT_TABLES["contributions"]) <= set(ct.columns)); self.assertEqual(ct.kind.iloc[0], "current")
        # a recording without one presynaptic cell reports the gap; batched rows and graded targets are refused
        gap = dec.decompose(c, "DNp01", recording=[self.recording(c, base, 0, p, drop=2)], params=p, tiers=False)
        self.assertEqual(gap.summary["coverage"]["stim[0]"]["presynaptic_cells_missing"], 1)
        self.assertEqual(int(gap.table("per_type").set_index("pre_group").loc["LC4"].stim_n_entries_missing), 1)
        batched = common.Recording(np.arange(2.0), np.arange(8), c.neurons.bodyId.to_numpy(), c.neurons.type.to_numpy().astype(str),
                                   {"rate_hz": np.zeros((2, 2, 8), np.float32)}, {}, {})
        with self.assertRaises(ValueError):
            dec.decompose(c, "DNp01", recording=batched, params=p)
        with self.assertRaises(ValueError):
            dec.decompose(c, "Mi4", recording=stim, params=p)
        with self.assertRaises(ValueError):
            dec.decompose(c, "DNp01", recording=stim, params=p, kind="voltage")
        # an arm recorded under other parameters is weighted by its own effective weights (here named explicitly)
        pb = self.params(input_norm_ref=100, type_path_gain=[["LC4", "DNp01", -3.0]])
        own = dec.decompose(c, "DNp01", recording={"stim": stim}, null_recording={"null": null}, params=p, tiers=False, window=(0.01, 0.05),
                            arm_weights_override={"stim": pb})
        o = own.table("per_type").set_index("pre_group")
        np.testing.assert_allclose(o.loc["LC4"].stim_values, -np.asarray(expect_lc4), rtol=1e-5)
        np.testing.assert_allclose(o.loc["LC4"].null_values, t.loc["LC4"].null_values, rtol=1e-6)
        self.assertAlmostEqual(o.loc["LC4"].stim_weight_mv_per_volley, -A[3, 2], places=6); self.assertAlmostEqual(o.loc["LC4"].weight_mv_per_volley, A[3, 2], places=6)
        self.assertEqual(own.summary["arm_weights"]["stim"]["entries_differing_from_params"], 1); self.assertEqual(own.summary["arm_weights"]["null"]["source"], "params")

    def test_contrast_separates_sign_changes_from_rescaling(self):
        from flyverse.interp import decompose as dec
        c = self.graph2()
        pa = self.params(input_norm_ref=100)
        pb = self.params(input_norm_ref=100, type_path_gain=[["LC4", "DNp01", -3.0]])        # the LC4 -> DNp01 entry flips sign, same magnitude
        res = dec.contrast(c, "DNp01|LC4|PEN_a", pa, pb, labels=("a", "b"))
        s = res.summary
        self.assertEqual(res.check(), []); self.assertEqual(s["entries_total"], 6)          # DNp01: LC4, DNa02; LC4: Mi4, DNa02; PEN_a: GLNO, LC4
        # one entry changes sign; DNp01's row sum is unchanged (|W| preserved) so nothing is rescaled
        self.assertEqual((s["entries_sign_changed"], s["entries_rescaled_only"]), (1, 0)); self.assertEqual(s["synapses_sign_changed"], 120.0)
        self.assertEqual(list(s["by_transmitter"]), ["acetylcholine +1->-1"]); self.assertEqual(s["pre_types_changed"], ["LC4"])
        link = res.table("delta_links").iloc[0]
        self.assertEqual((link.body_pre, link.body_post, link.change, link.sign_a, link.sign_b), ("1003", "1004", "sign", 1, -1))
        self.assertAlmostEqual(link.delta_mv, -2 * link.effective_mv_a, places=6)
        # a cap change moves the magnitude only -> 'gain', and DNp01's fan-in scale with it -> the DNa02 entry is rescaled only
        pc = self.params(input_norm_ref=100, conn_cap=0)
        res2 = dec.contrast(c, "DNp01|LC4|PEN_a", pa, pc, labels=("a", "c"))
        s2 = res2.summary
        self.assertEqual((s2["entries_sign_changed"], s2["entries_rescaled_only"], s2["target_cells_with_moved_fanin_scale"]), (1, 1, 1))
        self.assertEqual(res2.table("delta_links").change.iloc[0], "gain")
        cells = res2.table("rescaled_cells").iloc[0]
        self.assertEqual((cells.bodyId, cells.type, int(cells.rescaled_entries), int(cells.sign_changed_entries)), ("1004", "DNp01", 1, 1))
        self.assertAlmostEqual(cells.tot_a, 60 * 3 * 2 + 3); self.assertAlmostEqual(cells.tot_b, 120 * 3 * 2 + 3)
        self.assertLess(cells.fanin_scale_b, cells.fanin_scale_a)
        ct = res2.table("contributions"); self.assertEqual(list(ct.change), ["gain", "rescaled"])
        self.assertTrue(set(common.EXPORT_TABLES["contributions"]) <= set(ct.columns))
        # nothing changes on a target the parameters do not touch
        self.assertEqual(dec.contrast(c, "PEN_a", pa, pb).summary["entries_changed"], 0)

    def test_arms_and_cli_helpers(self):
        import dataclasses
        from flyverse.interp import decompose as dec
        self.assertIsNone(dec.arm_params("off").receptor_model); self.assertEqual(dec.arm_of(dec.arm_params("off")), "off")
        d = dec.arm_params("default"); self.assertEqual((d.receptor_model, d.receptor_net_rule, d.receptor_table), ("sign", "abs", None))
        self.assertEqual(dec.arm_of(d), "default"); self.assertIs(dec.arm_params("as-given", d), d)
        with self.assertRaises(ValueError):
            dec.arm_params("holdEverything")
        with tempfile.TemporaryDirectory() as tmp:
            h = dec.arm_params("holdBrainHis", out_dir=tmp)                  # built by scripts/build_hold_tables.py's hold_table
            self.assertTrue(Path(h.receptor_table).is_file()); self.assertEqual(dec.arm_of(h), "holdBrainHis")
            header = [ln for ln in open(h.receptor_table, encoding="utf-8") if ln.startswith("#")]
            self.assertTrue(any("histamine rows only" in ln for ln in header))
            csv = dec.arm_params(h.receptor_table); self.assertEqual(dec.arm_of(csv), "holdBrainHis")
            # the hold table applies to a synthetic graph too (no held row matches it: the same weights as the shipped table)
            c = self.graph2()
            ew_h = common.effective_weights(c, dataclasses.replace(h, event_driven=False))
            ew_d = common.effective_weights(c, dataclasses.replace(d, event_driven=False))
            np.testing.assert_allclose(ew_h.A.toarray(), ew_d.A.toarray())
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import interp_decompose as cli
        self.assertEqual(cli._arms_from_items(["default=out/x_r*.npz", "out/y_r*.npz"]), {"default": "out/x_r*.npz", "arm1": "out/y_r*.npz"})
        self.assertEqual(cli._window("0.5,1.5"), (0.5, 1.5)); self.assertIsNone(cli._window(None))
        self.assertEqual(set(cli.PROTOCOL_FN), set(cli.PROTOCOLS)); self.assertEqual(cli.GF_ARMS, ("off", "default", "holdBrain", "holdOptic"))
        c = self.graph2()
        counts, ok = dec.counts_matrix(c)
        self.assertFalse(ok); self.assertEqual(counts[3, 2], 120.0); self.assertEqual(counts[7, 6], 0.0)     # the explicit zero stays 0 without sign0 counts
        self.assertEqual(list(dec.static_frozen(c)), [1])


class LedgerTests(unittest.TestCase):
    """flyverse/interp/ledger.py: the shipped expectation table, the source readers, the null / replicate rule and
    the agreement with scripts/benchmark.py's own statuses. CPU, synthetic sources, seconds."""

    @classmethod
    def tearDownClass(cls):
        # importing flyverse.interp.ledger binds the SUBMODULE as flyverse.interp.ledger, which shadows the
        # __getattr__ hand-off in flyverse/interp/__init__.py (_implementation imports the module and returns the
        # function but leaves the module bound, so the second `interp.ledger` is a module, not a callable).
        # Drop the binding again so StubTests sees the contract's state. See docs/audits/interp_ledger.md section 7.
        interp.__dict__.pop("ledger", None)

    # ---- fixtures ---------------------------------------------------------------------------------------------
    def bench_doc(self, gf_peak=37.116951, criterion=">= 38", status="FAIL"):
        """A scripts/benchmark.py JSON in miniature: the sections its readers walk plus the `checks` list."""
        return {"config": {"receptor": None, "device": "NVIDIA GeForce RTX 4090"},
                "sections": {"rest": {"spikes_per_step": 0.0},
                             "taste": {"MN9_hz": 5.845469},
                             "motion": {"subtypes": {"T4a": {"dsi": 0.1664, "best": "front->back", "correct": True},
                                                     "T5a": {"dsi": 0.39, "best": "front->back", "correct": True}}},
                             "walk": {"loom.GF_peak_hz": 30.797726},
                             "loom_escape": {"GF_peak_hz": gf_peak, "escapes": 0},
                             "rotation": {"types": {"HSN": {"flip_hz": -7.064, "rate_hz": 2.19}}},
                             "dn": {"MDN": {"legL": 3.171, "legR": 2.185, "power": 0.0, "top": {"IN06B020": 152.0}}}},
                "checks": [{"key": "taste.MN9_hz", "measured": 5.845469, "reference": 3.7, "criterion": "> 2", "status": "PASS"},
                           {"key": "motion.min_dsi", "measured": 0.1664, "reference": 0.16, "criterion": ">= 0.1", "status": "PASS"},
                           {"key": "loom_escape.GF_peak_hz", "measured": gf_peak, "reference": 46,
                            "criterion": criterion, "status": status}]}

    def sweep_doc(self, values, null=False, mode="sign-abs"):
        """A probe_object_sweep.py JSON: `config.null` marks the none-vs-none arm (object_sweep.md 8.4)."""
        return {"config": {"mode": mode, "null": null, "condition_a": "ball", "condition_b": "none", "device": "cuda"},
                "none": {}, "verdict": {"pass": False},
                "ball": {t: {"diff_max_over_cells_mean_mv": v} for t, v in values.items()}}

    def stage_doc(self, ab_z, cb_z, config="baseline"):
        """A probe_figure_stages.py stage JSON: the AB (object vs none) and CB (none vs none) blocks."""
        return {"config": {"receptor_model": "sign", "receptor_net_rule": "abs", "optic_config": config, "device": "cuda"},
                "apple": {"types": {t: {"stage": 3, "kind": "rate", "n_cells": 10,
                                        "signed": {"AB": {"z": z, "figure": z / 100.0},
                                                   "CB": {"z": cb_z[t], "figure": cb_z[t] / 100.0}},
                                        "abs": {"AB": {"z": 0.0, "figure": 0.0}}}
                                    for t, z in ab_z.items()}, "columns": {}, "geometry": {}}}

    def tiny_table(self, rows, d):
        """Write a table CSV with ledger.COLUMNS and return the path."""
        from flyverse.interp import ledger as LG
        df = pd.DataFrame([{c: r.get(c, "") for c in LG.COLUMNS} for r in rows])
        p = Path(d) / "table.csv"
        df.to_csv(p, index=False)
        return p

    # ---- the shipped table ------------------------------------------------------------------------------------
    def test_shipped_table_covers_the_mandated_literature(self):
        from flyverse.interp import ledger as LG
        t = LG.load_table()
        self.assertEqual([c for c in LG.COLUMNS if c not in t.columns], [])
        self.assertTrue(set(t.op) <= set(LG.OPS))
        self.assertEqual(len(set(t.row_id)), len(t))
        self.assertEqual(sorted({r for r in t.requires if r} - set(t.row_id)), [])
        self.assertTrue((t.source.str.len() > 0).all())               # every row carries a citation
        self.assertTrue((t.label.str.len() > 0).all())
        # the validation table of the task brief: one row per named result, with the paper it comes from
        need = {"Maisak": ["motion.T4a.dsi", "motion.T4c.direction", "motion.T4_T5.min_dsi"],
                "Ache": ["loom.LC4.rate_peak_hz"], "Klapoetke": ["loom.LPLC2.rate_peak_hz", "objsweep.LPLC2.figure_z"],
                "Schnell": ["rotation.HSN.dprime", "rotation.HSE.dprime", "rotation.VS.dprime"],
                "Hallem": ["orn.ORN_DM2.rate_hz", "orn.ORN_DM4.rate_hz", "orn.ORN_DC1.rate_hz"],
                "Shiu": ["taste.MN9.rate_hz", "taste.MN9.rate_hz_calibrated_bitter", "taste.MN9.rate_hz_shiu"],
                "Seelig": ["compass.EPG.bump_rate_hz", "compass.EPG.bump_width_wedges"],
                "Bidaye": ["dn.MDN.top_rate_hz", "dn.DNp09.top_rate_hz"],
                "von Reyn": ["loom.DNp01.rate_peak_hz", "loom.DNp01.escape_latency_ms"],
                "Ribeiro": ["object.LC10a.flip_hz", "objsweep.LC10a.figure_z"]}
        by_id = t.set_index("row_id")
        for author, ids in need.items():
            for rid in ids:
                self.assertIn(rid, by_id.index, f"{author}: {rid} missing from the expectation table")
                self.assertIn(author, by_id.loc[rid, "source"], f"{rid} does not cite {author}")
        # the walk.power_max bound of dynamics round 1 is recorded, never scored
        self.assertEqual(by_id.loc["walk.power_MN.rate_max_hz", "op"], "report")
        # every check_key names a real scripts/benchmark.py REFERENCES key (read from the source, not imported:
        # scripts/benchmark.py pulls in torch and the whole simulator at import time)
        import re as _re
        src = (Path(__file__).resolve().parents[1] / "scripts" / "benchmark.py").read_text(encoding="utf-8")
        keys = set(_re.findall(r'^\s*"([\w.]+)":\s*Ref\(', src, _re.M))
        self.assertTrue(keys, "could not read benchmark.REFERENCES")
        self.assertEqual(sorted({k for k in t.check_key if k} - keys), [])

    def test_table_validation_rejects_bad_rows(self):
        from flyverse.interp import ledger as LG
        base = {"row_id": "a.b.c", "population": "DNp01", "stimulus": "loom_left", "quantity": "rate_hz",
                "expected": "1", "op": ">=", "bound": "1", "source": "x"}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                LG.load_table(self.tiny_table([dict(base, op="~~")], d))
            with self.assertRaises(ValueError):
                LG.load_table(self.tiny_table([base, dict(base)], d))
            with self.assertRaises(ValueError):
                LG.load_table(self.tiny_table([dict(base, requires="nope")], d))
            t = LG.load_table(self.tiny_table([dict(base, label="")], d))
            self.assertEqual(t.label.iloc[0], "DNp01")                 # a blank label falls back to the spec
            self.assertFalse(bool(t.gap.iloc[0]))

    def test_evaluate_every_op(self):
        from flyverse.interp import ledger as LG
        ev = LG.evaluate
        self.assertEqual(ev(">", 3, 2), "PASS"); self.assertEqual(ev(">", 1, 2), "FAIL")
        self.assertEqual(ev("<", 1, 2), "PASS"); self.assertEqual(ev(">=", 2, 2), "PASS"); self.assertEqual(ev("<=", 2, 2), "PASS")
        self.assertEqual(ev("==", 8, 8), "PASS"); self.assertEqual(ev("abs>=", -5.4, 3), "PASS")
        self.assertEqual(ev("abs<=", -1, 3), "PASS"); self.assertEqual(ev("sign", -7.06, -1), "PASS")
        self.assertEqual(ev("sign", 7.06, -1), "FAIL"); self.assertEqual(ev("range", 3.6, "2.5,5.0"), "PASS")
        self.assertEqual(ev("range", 220, "5,60"), "FAIL"); self.assertEqual(ev("notnone", 3.5, 0), "PASS")
        self.assertEqual(ev("notnone", None, 0), "MISSING"); self.assertEqual(ev("is", "up", "up"), "PASS")
        self.assertEqual(ev("is", "down", "up"), "FAIL"); self.assertEqual(ev("report", 79.05, 50), "RECORDED")
        self.assertEqual(ev(">=", None, 1), "MISSING"); self.assertEqual(ev(">=", float("nan"), 1), "MISSING")
        # the battery's gap labels, and the tolerance
        self.assertEqual(ev(">=", 0.0, 6, gap=True), "KNOWN GAP"); self.assertEqual(ev(">=", 7, 6, gap=True), "PASS (gap closed)")
        self.assertEqual(ev(">", 1.9, 2), "FAIL"); self.assertEqual(ev(">", 1.9, 2, tolerance=0.1), "PASS")

    # ---- readers ----------------------------------------------------------------------------------------------
    def test_benchmark_reader_and_battery_agreement(self):
        from flyverse.interp import ledger as LG
        obs, checks, row = LG.read_source(self.bench_doc())
        self.assertEqual(row["kind"], "benchmark")
        o = LG._obs_frame(obs)
        self.assertEqual(float(o[(o.population == "MN9") & (o.quantity == "rate_hz")].value.iloc[0]), 5.845469)
        self.assertEqual(o[(o.population == "T4a") & (o.quantity == "preferred_direction")].value.iloc[0], "front->back")
        self.assertEqual(float(o[o.population == "T4/T5"].set_index("quantity").value["min_dsi"]), 0.1664)
        self.assertEqual(float(o[(o.population == "MDN") & (o.quantity == "leg_asym_hz")].value.iloc[0]), 3.171 - 2.185)
        self.assertEqual(set(checks), {"taste.MN9_hz", "motion.min_dsi", "loom_escape.GF_peak_hz"})
        with tempfile.TemporaryDirectory() as d:
            tab = self.tiny_table([
                {"row_id": "taste.MN9.rate_hz", "population": "MN9", "stimulus": "sugar", "quantity": "rate_hz",
                 "expected": "3.7", "op": ">", "bound": "2", "check_key": "taste.MN9_hz", "source": "Shiu 2024"},
                {"row_id": "loom.demo", "population": "DNp01", "stimulus": "loom_demo", "quantity": "rate_peak_hz",
                 "expected": "41", "op": ">=", "bound": "33", "check_key": "loom_escape.GF_peak_hz", "source": "von Reyn 2014"},
            ], d)
            res = LG.ledger([self.bench_doc()], table=tab, c=graph())
            led = res.table("ledger").set_index("row_id")
        self.assertEqual(led.loc["taste.MN9.rate_hz", "status"], "PASS")
        self.assertEqual(led.loc["taste.MN9.rate_hz", "battery_status"], "PASS")
        self.assertTrue(bool(led.loc["taste.MN9.rate_hz", "battery_agrees"]))
        self.assertTrue(bool(led.loc["taste.MN9.rate_hz", "battery_criterion_same"]))
        # the stored file kept the pre-session-9 criterion: a status difference the ledger explains, not a bug
        self.assertEqual(led.loc["loom.demo", "status"], "PASS")
        self.assertEqual(led.loc["loom.demo", "battery_status"], "FAIL")
        self.assertFalse(bool(led.loc["loom.demo", "battery_criterion_same"]))
        self.assertEqual(res.summary["battery_disagree"], [])
        self.assertEqual([m["row_id"] for m in res.summary["battery_criterion_mismatch"]], ["loom.demo"])
        self.assertEqual(res.summary["sources"]["devices"], ["NVIDIA GeForce RTX 4090"])
        # a file whose criterion is the table's and whose status differs IS a disagreement
        with tempfile.TemporaryDirectory() as d:
            tab = self.tiny_table([{"row_id": "loom.demo", "population": "DNp01", "stimulus": "loom_demo",
                                    "quantity": "rate_peak_hz", "expected": "41", "op": ">=", "bound": "33",
                                    "check_key": "loom_escape.GF_peak_hz", "source": "von Reyn 2014"}], d)
            bad = LG.ledger([self.bench_doc(criterion=">= 33", status="FAIL")], table=tab, c=graph())
        self.assertEqual([x["row_id"] for x in bad.summary["battery_disagree"]], ["loom.demo"])

    def test_object_sweep_null_reproduces_the_audit(self):
        """docs/audits/object_sweep.md 8.4, LPLC2 sign-abs: z +5.4, Welch +8.6, U 25, p 0.0079 -- and LC11 at the null."""
        from flyverse.interp import ledger as LG
        ball = [0.369, 0.370, 0.326, 0.392, 0.299]
        null = [0.124, 0.081, 0.178, 0.137, 0.174]
        lc11_b, lc11_n = [0.081, 0.118, 0.040, 0.058, 0.032], [0.080, 0.067, 0.026, 0.097, 0.071]
        stim = [self.sweep_doc({"LPLC2": ball[i], "LC11": lc11_b[i]}) for i in range(5)]
        nulls = [self.sweep_doc({"LPLC2": null[i], "LC11": lc11_n[i]}, null=True) for i in range(5)]
        with tempfile.TemporaryDirectory() as d:
            tab = self.tiny_table([
                {"row_id": "objsweep.LPLC2.figure_z", "population": "LPLC2", "stimulus": "object_sweep",
                 "quantity": "diff_max_over_cells_mean_mv@z", "expected": "5.4", "op": "abs>=", "bound": "3",
                 "source": "Klapoetke 2017"},
                {"row_id": "objsweep.LC11.figure_z", "population": "LC11", "stimulus": "object_sweep",
                 "quantity": "diff_max_over_cells_mean_mv@z", "expected": "0.4", "op": "abs>=", "bound": "3",
                 "gap": "1", "source": "Keles & Frye 2017"}], d)
            res = LG.ledger(stim, table=tab, null=nulls, c=graph())
        led = res.table("ledger").set_index("row_id")
        r = led.loc["objsweep.LPLC2.figure_z"]
        self.assertEqual(r["scored_on"], "z"); self.assertEqual(int(r["n"]), 5); self.assertEqual(int(r["null_n"]), 5)
        self.assertAlmostEqual(r["measured"], 0.3512, places=3); self.assertAlmostEqual(r["null_mean"], 0.1388, places=3)
        self.assertAlmostEqual(r["z"], 5.4, delta=0.15); self.assertAlmostEqual(r["welch"], 8.6, delta=0.2)
        self.assertEqual(r["U"], 25.0); self.assertAlmostEqual(r["p"], 0.0079, places=3)
        self.assertEqual(r["null_verdict"], "result"); self.assertEqual(r["status"], "PASS")
        g = led.loc["objsweep.LC11.figure_z"]
        self.assertAlmostEqual(g["z"], -0.1, delta=0.1); self.assertEqual(g["U"], 12.0)
        self.assertEqual(g["status"], "KNOWN GAP")                     # a documented gap, not a FAIL
        # three runs is the floor: two are 'underpowered' whatever the numbers
        with tempfile.TemporaryDirectory() as d:
            tab = self.tiny_table([{"row_id": "objsweep.LPLC2.figure_z", "population": "LPLC2", "stimulus": "object_sweep",
                                    "quantity": "diff_max_over_cells_mean_mv@z", "expected": "5.4", "op": "abs>=",
                                    "bound": "3", "source": "Klapoetke 2017"}], d)
            few = LG.ledger(stim[:2], table=tab, null=nulls[:2], c=graph())
        r2 = few.table("ledger").iloc[0]
        self.assertEqual(r2["null_verdict"], "underpowered"); self.assertFalse(bool(r2["replicates_ok"]))
        self.assertEqual(few.summary["underpowered_rows"], ["objsweep.LPLC2.figure_z"])

    def test_stage_json_carries_its_own_null(self):
        from flyverse.interp import ledger as LG
        doc = self.stage_doc({"Mi1": -8.8, "T2": 2.7}, {"Mi1": 0.18, "T2": -0.07})
        obs, _, row = LG.read_source(doc)
        self.assertEqual(row["kind"], "figure_stages")
        o = LG._obs_frame(obs)
        self.assertEqual(set(o.arm), {"sign-abs"})
        ab = o[(o.population == "Mi1") & (o.quantity == "figure_z") & (~o.is_null)]
        cb = o[(o.population == "Mi1") & (o.quantity == "figure_z") & (o.is_null)]
        self.assertEqual(float(ab.value.iloc[0]), -8.8); self.assertEqual(float(cb.value.iloc[0]), 0.18)
        self.assertAlmostEqual(float(o[(o.population == "Mi1") & (o.quantity == "figure")].value.iloc[0]), -0.088)
        # a non-baseline optic configuration is its own arm
        o2 = LG._obs_frame(LG.read_source(self.stage_doc({"Mi1": -1.0}, {"Mi1": 0.0}, config="no_spk_feedback"))[0])
        self.assertEqual(set(o2.arm), {"sign-abs+no_spk_feedback"})

    def test_text_and_csv_readers(self):
        from flyverse.interp import ledger as LG
        loom = ("receptor model sign (abs); fast sign changed on 63,380 entries\n"
                "device: cuda; cuda available: True\n"
                "[walking 1] spikes/step 3 | OL T4a=0.01 | LPLC2=1 LC4=0 DNp01=0 TTMn=0\n"
                "[t=0.5s d=3cm] spikes/step 33 | OL T4a=0.01 | LPLC2=2 LC4=0 DNp01=14 TTMn=2\n"
                "  *** escape triggered at t=0.60s, object 3.5 cm away (GF 35 Hz, TTMn 20 Hz)\n"
                "[t=0.6s d=3cm] spikes/step 21 | OL T4a=0.03 | LPLC2=4 LC4=0 DNp01=35 TTMn=20\n"
                "escape: True\n")
        o = LG._obs_frame(LG.read_loom_text(loom, "out/loom2_abs_s0.txt"))
        self.assertEqual(set(o.arm), {"sign-abs"})                      # the console header, not the file name
        peak = o[(o.stimulus == "loom_left") & (o.quantity == "rate_peak_hz")].set_index("population").value
        self.assertEqual(peak["DNp01"], 35.0); self.assertEqual(peak["LC4"], 0.0); self.assertEqual(peak["TTMn"], 20.0)
        self.assertEqual(float(o[o.quantity == "escape_range_cm"].value.iloc[0]), 3.5)
        self.assertEqual(int(o[o.quantity == "escape"].value.iloc[0]), 1)
        self.assertEqual(float(o[(o.stimulus == "walking") & (o.population == "DNp01")].value.iloc[0]), 0.0)
        bitter = ("receptor model full (class): matched 7,590,509 edges\n"
                  "this project (calibrated)              sugar           : MN9   5.8 Hz |  | spikes/step 25\n"
                  "this project (calibrated)              sugar + bitter  : MN9   0.0 Hz |  | spikes/step 25\n"
                  "Shiu et al. rules (uniform 0.275 mV)   sugar           : MN9  74.2 Hz |  | spikes/step 440\n"
                  "Shiu et al. rules (uniform 0.275 mV)   sugar + bitter  : MN9   1.8 Hz |  | spikes/step 256\n")
        b = LG._obs_frame(LG.read_bitter_text(bitter, "out/bitter_full.txt")).set_index(["stimulus", "quantity"]).value
        self.assertEqual(set(LG._obs_frame(LG.read_bitter_text(bitter, "x.txt")).arm), {"full-class"})
        self.assertEqual(b[("sugar", "rate_hz_calibrated")], 5.8); self.assertEqual(b[("sugar+bitter", "rate_hz_calibrated")], 0.0)
        self.assertEqual(b[("sugar", "rate_hz_shiu")], 74.2); self.assertEqual(b[("sugar+bitter", "rate_hz_shiu")], 1.8)
        rot = pd.DataFrame([{"type": "HSN", "cells": 2, "flip_d": -4.148, "flip_hz": -7.064, "rate_hz": 2.185}])
        r = LG._obs_frame(LG.read_rotation_csv(rot, "out/screen_rotation.csv")).set_index("quantity").value
        self.assertEqual(r["dprime"], -4.148); self.assertEqual(r["flip_hz"], -7.064)
        fig = pd.DataFrame([{"type": "L1", "figure": 0.0162, "figure_z": 19.25, "bg_level": 0.0027}])
        f = LG._obs_frame(LG.read_figure_csv(fig, "out/fg_off.csv"))
        self.assertEqual(set(f.stimulus), {"figure_ground_apple"}); self.assertEqual(set(f.arm), {"off"})
        self.assertEqual(float(f.set_index("quantity").value["figure_z"]), 19.25)
        # an unreadable / unrecognised file is a row in the sources table, never an exception
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "nonsense.txt"; p.write_text("nothing here", encoding="utf-8")
            _, _, row = LG.read_source(p)
            self.assertIn("not a loom or bitter console log", row["error"])
            _, _, row2 = LG.read_source(Path(d) / "missing.json")
            self.assertTrue(row2["error"])

    # ---- arms, preconditions, schema, CLI ---------------------------------------------------------------------
    def test_arms_preconditions_and_result_schema(self):
        from flyverse.interp import ledger as LG
        from flyverse.brain import LIFParams
        compass = [{"config": {"gE": g, "gD": 15.0, "program": "cx"}, "n_epg": 16,
                    "summary": {"survival_s": {"mean": s, "n": 16}, "bump_hz_post": {"mean": hz, "n": 16}}}
                   for g, s, hz in ((None, 0.033, 34.9), (2.0, 38.0, 219.5))]
        with tempfile.TemporaryDirectory() as d:
            tab = self.tiny_table([
                {"row_id": "compass.EPG.bump_survival_s", "population": "EPG", "stimulus": "compass_room",
                 "quantity": "bump_survival_s", "expected": "38", "op": ">=", "bound": "5", "source": "Seelig 2015"},
                {"row_id": "compass.EPG.bump_rate_hz", "population": "EPG", "stimulus": "compass_room",
                 "quantity": "bump_rate_hz", "expected": "220", "op": "range", "bound": "5,60", "gap": "1",
                 "requires": "compass.EPG.bump_survival_s", "source": "Seelig 2015"}], d)
            res = LG.ledger(compass, table=tab, c=graph(), lif=LIFParams(receptor_model=None))
        led = res.table("ledger").set_index(["row_id", "arm"])
        self.assertEqual(sorted(res.summary["arms"]), ["control+cx", "gE2/gD15+cx"])
        self.assertEqual(led.loc[("compass.EPG.bump_survival_s", "control+cx"), "status"], "FAIL")
        self.assertEqual(led.loc[("compass.EPG.bump_survival_s", "gE2/gD15+cx"), "status"], "PASS")
        # the shipped default's 34.9 Hz 'bump' dies in 0.033 s: its rate is not a verdict at all
        self.assertEqual(led.loc[("compass.EPG.bump_rate_hz", "control+cx"), "status"], "NOT_APPLICABLE")
        self.assertIn("bump_survival_s", led.loc[("compass.EPG.bump_rate_hz", "control+cx"), "precondition"])
        self.assertEqual(led.loc[("compass.EPG.bump_rate_hz", "gE2/gD15+cx"), "status"], "KNOWN GAP")   # 219.5 Hz
        self.assertEqual([r["row_id"] for r in res.summary["not_applicable"]], ["compass.EPG.bump_rate_hz"])
        # the Result is exportable and analysis-only: the realised device is the CPU that scored
        self.assertEqual(res.check(), [])
        for k in common.REQUIRED_PROVENANCE:
            self.assertIn(k, res.provenance)
        self.assertEqual(res.provenance["execution"]["device"], "cpu")
        self.assertIn("scored_devices", res.provenance["execution"])
        self.assertIsNone(res.provenance["model"]["lif"]["receptor_model"])
        self.assertIn("type_path_gain", res.provenance["model"]["lif"])
        self.assertEqual(res.provenance["compiled_connectome"]["n_neurons"], 8)
        self.assertEqual(set(res.tables), {"ledger", "sources", "observations"})
        self.assertTrue(any(p["label"] == "EPG" for p in res.populations) or not res.populations)
        with tempfile.TemporaryDirectory() as d:
            back = common.Result.load(res.save(Path(d) / "led.json"))
        self.assertEqual(back.tool, "ledger"); self.assertEqual(len(back.table("ledger")), len(led))
        # group_by=None pools every arm into one row and says how many runs went in
        with tempfile.TemporaryDirectory() as d:
            tab = self.tiny_table([{"row_id": "compass.EPG.bump_survival_s", "population": "EPG",
                                    "stimulus": "compass_room", "quantity": "bump_survival_s", "expected": "38",
                                    "op": ">=", "bound": "5", "source": "Seelig 2015"}], d)
            pooled = LG.ledger(compass, table=tab, group_by=None, c=graph()).table("ledger").iloc[0]
        self.assertEqual(pooled["arm"], "pooled"); self.assertEqual(int(pooled["n"]), 2)
        self.assertEqual(pooled["runs_passing"], "1/2"); self.assertTrue(bool(pooled["unstable"]))

    def test_validate_reproduces_the_battery(self):
        """VALIDATION['ledger'] on a synthetic benchmark JSON + rotation CSV: the statuses and the numbers."""
        from flyverse.interp import ledger as LG
        self.assertEqual({m[2] for m in LG.VALIDATION_MAP} - set(LG.load_table().row_id), set())
        self.assertTrue(LG._within(4.566, 4.6, False)); self.assertFalse(LG._within(4.566, 10.0, False))
        self.assertTrue(LG._within(-4.148, 4.2, True)); self.assertFalse(LG._within(-4.148, 4.2, False))
        self.assertTrue(LG._within(37.1, [34, 56], False)); self.assertFalse(LG._within(30.8, [34, 56], False))
        self.assertIsNone(LG._within(None, 1.0, False))
        self.assertTrue(LG._same_criterion(">=", "33", ">= 33")); self.assertFalse(LG._same_criterion(">=", "33", ">= 38"))
        self.assertTrue(LG._same_criterion(">=", "33", ""))            # a file that stored no criterion is not stale

    def test_cli(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import interp_ledger as cli
        self.assertEqual(cli.main(["record"]), 2)                       # no GPU part: says so and stops
        self.assertEqual(cli.main(["run"]), 2)
        self.assertEqual(cli.main(["table", "--only", "compass", "--quiet", "--max-rows", "3"]), 0)
        df = pd.DataFrame([{"row_id": "a.b.c", "status": "FAIL", "arm": "off"},
                           {"row_id": "x.y.z", "status": "PASS", "arm": "sign-abs"}])
        self.assertEqual(list(cli._filter(df, ["a."], None, None).row_id), ["a.b.c"])
        self.assertEqual(list(cli._filter(df, None, ["PASS"], None).row_id), ["x.y.z"])
        self.assertEqual(list(cli._filter(df, None, None, ["off"]).row_id), ["a.b.c"])
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "bench.json"
            with open(src, "w", encoding="utf-8") as f:
                json.dump(LedgerTests().bench_doc(), f)
            out = Path(d) / "led.json"
            rc = cli.main(["analyse", str(src), "--no-connectome", "--quiet", "--json", str(out),
                           "--csv", str(Path(d) / "led.csv"), "--max-rows", "5"])
            self.assertEqual(rc, 0)
            res = common.Result.load(out)
        self.assertEqual(res.tool, "ledger")
        self.assertEqual(res.provenance["execution"]["device"], "cpu")
        led = res.table("ledger").set_index("row_id")
        self.assertEqual(led.loc["taste.MN9.rate_hz", "status"], "PASS")      # the shipped table, scored end to end
        self.assertEqual(led.loc["motion.T4_T5.min_dsi", "status"], "PASS")


# --------------------------------------------------------------------------------------------------------------------
# The three application scripts (scripts/interp_apply_{turning,object,rotation}.py).  CPU only, no connectome cache and
# no GPU: each class runs the script's own `selftest` logic plus the pieces the skeptic round fixed.  Before this the
# ~2,500 lines of analysis behind docs/audits/deficit_*.md were exercised only by hand-run `selftest` subcommands.
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load_script(name):
    """Import scripts/<name>.py as a module (the apply scripts are CLIs, not a package)."""
    import importlib.util
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


class ApplyTurningTests(unittest.TestCase):
    """scripts/interp_apply_turning.py's CPU analysis on a synthetic room recording: the L-R tables (and the start-up
    skip that now reaches their totals), the cancellation summary (which used to copy DNa02's readout SD into the leg
    pair's block), and the per-fly readout correlations that deficit_turning.md 4.4 / 5 quote."""

    T, B, G = 400, 3, 6      # frames, flies, presynaptic groups
    SKIP_S = 1.0             # 100 frames at the 10 ms dt below

    @classmethod
    def setUpClass(cls):
        cls.m = _load_script("interp_apply_turning")

    def _run(self, seed=0):
        """A Run object with the arrays lr_tables / cancellation_tables / per_fly_readout_corr read.  The first
        SKIP_S seconds carry a large transient on the totals, so dropping them changes the totals and nothing else."""
        m = self.m
        rng = np.random.default_rng(seed)
        T, B, G = self.T, self.B, self.G
        keys = np.array(["IN12B014/L", "IN12B014/R", "LLPC1/L", "LLPC1/R", "DNge035/L", "DNge035/R"])
        targets = ["DNa02_L", "DNa02_R", "leg_L", "leg_R"]
        k0 = int(round(self.SKIP_S / 0.01))

        tot = rng.normal(0.0, 1.0, (T, 4, B)).astype(np.float32)
        tot[:, 0] -= 300.0; tot[:, 1] -= 400.0                       # DNa02 L / R steady inhibition
        tot[:k0] -= 1000.0                                            # the start-up transient the skip removes
        totE = np.abs(tot) + 700.0
        totI = tot - totE

        # per-fly, per-group mean input to each target (the GPU pass accumulates these over ALL frames)
        gm = rng.normal(0.0, 2.0, (4, B, G)).astype(np.float32)
        gm[0, :, 0] -= 90.0; gm[1, :, 1] -= 124.0                     # the inhibitor's fixed L-R offset

        # a planted within-fly correlation: LLPC1 L-R drives DNa02 L-R, DNge035 L-R drives leg L-R contralaterally
        watch = ["DNa02_L", "DNa02_R", "LLPC1_L", "LLPC1_R", "DNge035_L", "DNge035_R"]
        wr = rng.normal(2.0, 0.5, (T, B, len(watch))).astype(np.float32)
        llp = wr[:, :, 2] - wr[:, :, 3]
        dng = wr[:, :, 4] - wr[:, :, 5]
        turn_l = (0.5 * llp + rng.normal(0, 0.05, (T, B))).astype(np.float32)
        turn_r = np.zeros((T, B), np.float32)
        leg_l = (-0.5 * dng + rng.normal(0, 0.05, (T, B))).astype(np.float32)
        leg_r = np.zeros((T, B), np.float32)

        keep = np.arange(G, dtype=np.int32)
        series = rng.normal(0.0, 5.0, (T, B, G)).astype(np.float16)

        z = {"t_ms": (np.arange(T) * 10.0).astype(np.float64), "group_keys": keys, "target_names": np.array(targets),
             "pre_idx": np.arange(G), "pre_group": np.arange(G, dtype=np.int32), "sel_idx": np.arange(G),
             "rate_mean_flies": rng.uniform(0.0, 3.0, (B, G)).astype(np.float32),
             "rate_max_flies": np.array([[20.0, 20.0, 20.0, 20.0, 0.2, 20.0]] * B, np.float32),   # DNge035/L never fires
             "group_mean_flies": gm, "tot": tot, "totE": totE, "totI": totI,
             "watch_rates": wr, "yaw_cmd": rng.normal(0, 0.05, (T, B)).astype(np.float32),
             "airborne": np.zeros((T, B), bool), "wind_angle": rng.uniform(-1, 1, (T, B)).astype(np.float32),
             "w__DNa02_L": np.array([-5.0, -5.0, 33.0, 0.6, 0.0, 0.0], np.float32),
             "w__DNa02_R": np.array([-5.0, -5.0, 0.6, 55.0, 0.0, 0.0], np.float32),
             "w__leg_L": np.zeros(G, np.float32), "w__leg_R": np.zeros(G, np.float32),
             "keep__DNa02_L|DNa02_R": keep, "keep__leg_L|leg_R": keep,
             "series__DNa02_L": series, "series__DNa02_R": series * 0.5,
             "series__leg_L": series * 0.25, "series__leg_R": series * 0.1,
             "m__turn_L": turn_l, "m__turn_R": turn_r, "m__leg_L": leg_l, "m__leg_R": leg_r}

        r = object.__new__(m.Run)
        r.prefix = Path(f"mem:plain_r{seed}")
        r.z = z
        r.meta = {"seed": seed, "watch": watch, "device": "cpu"}
        r.keys = keys
        r.targets = targets
        r.B = B
        r.seed = seed
        r.flies = None
        return r

    def test_lr_tables_totals_take_the_skip_and_entries_do_not(self):
        """The skeptic's protocol item: `--skip` could not reach lr_tables at all, so the audit's totals row included
        the start-up transient.  It now reaches the per-frame totals; the per-(pre group) entries come from
        `group_mean_flies`, a whole-rollout mean, and cannot be re-windowed without re-recording."""
        m = self.m
        runs = [self._run(s) for s in range(3)]
        full = m.lr_tables(runs, None, 0.0)
        skipped = m.lr_tables(runs, None, self.SKIP_S)

        tf = full["DNa02_L|DNa02_R:totals"]; ts = skipped["DNa02_L|DNa02_R:totals"]
        self.assertAlmostEqual(float(tf.window_skip_s.iloc[0]), 0.0)
        self.assertAlmostEqual(float(ts.window_skip_s.iloc[0]), self.SKIP_S)
        self.assertAlmostEqual(float(ts.entries_window_skip_s.iloc[0]), 0.0)     # declared, not silently implied
        # the transient pulls the unskipped total down by ~1000 * k0 / T = 250 mV/s
        self.assertLess(tf.I_DNa02_L.mean(), ts.I_DNa02_L.mean() - 200.0)
        self.assertAlmostEqual(ts.I_DNa02_L.mean(), -300.0, delta=5.0)
        self.assertAlmostEqual(ts.I_DNa02_R.mean(), -400.0, delta=5.0)
        # E / I follow the same window, and E + I is the net
        self.assertTrue(np.allclose(ts.E_DNa02_L + ts.I_neg_DNa02_L, ts.I_DNa02_L, atol=1e-3))
        # the per-type entries are identical between the two calls (they cannot take a window)
        for level in ("type", "type_side"):
            a = full[f"DNa02_L|DNa02_R:{level}"].set_index("pre_group")
            b = skipped[f"DNa02_L|DNa02_R:{level}"].set_index("pre_group")
            self.assertTrue(np.allclose(a.LR_mean.sort_index(), b.LR_mean.sort_index()))

    def test_lr_tables_report_the_room_activity_of_each_group(self):
        m = self.m
        runs = [self._run(s) for s in range(3)]
        t = m.lr_tables(runs, None, self.SKIP_S)["DNa02_L|DNa02_R:type"].set_index("pre_group")
        self.assertEqual(int(t.loc["LLPC1", "n_pre_cells"]), 2)
        self.assertAlmostEqual(float(t.loc["LLPC1", "w_DNa02_R_mv_per_volley"]), 55.6, places=4)
        # DNge035/L never exceeds NEVER_FIRING_HZ, DNge035/R does: half the type's cells fire
        self.assertAlmostEqual(float(t.loc["DNge035", "frac_cells_firing"]), 0.5)
        self.assertAlmostEqual(float(t.loc["IN12B014", "frac_cells_firing"]), 1.0)

    def test_cancellation_summary_reports_each_pair_s_own_readout(self):
        """The skeptic's tool bug: cancellation_tables() wrote DNa02's readout SD into the leg pair's summary block,
        so `summary.cancellation['leg_L|leg_R'].sd_DNa02_LR_hz_mean` was DNa02's number, not the legs'."""
        m = self.m
        runs = [self._run(s) for s in range(3)]
        canc = m.cancellation_tables(runs, self.SKIP_S, n_shuffle=8, seed=0)
        dna = canc["DNa02_L|DNa02_R"]["summary"]
        leg = canc["leg_L|leg_R"]["summary"]
        self.assertEqual(dna["readout_LR"], "DNa02_L-DNa02_R")
        self.assertEqual(leg["readout_LR"], "leg_L-leg_R")
        # the two readouts have different temporal SDs, and each block now carries its own
        self.assertAlmostEqual(dna["sd_readout_LR_hz_mean"], dna["sd_DNa02_LR_hz_mean"], places=9)
        self.assertNotAlmostEqual(leg["sd_readout_LR_hz_mean"], leg["sd_DNa02_LR_hz_mean"], places=3)
        per_fly = canc["leg_L|leg_R"]["per_fly"]
        self.assertAlmostEqual(leg["sd_readout_LR_hz_mean"], float(per_fly.sd_leg_LR_hz.mean()), places=6)
        self.assertAlmostEqual(dna["sd_readout_LR_hz_mean"], float(canc["DNa02_L|DNa02_R"]["per_fly"].sd_DNa02_LR_hz.mean()), places=6)

    def test_per_fly_readout_corr_is_a_shipped_generator(self):
        """deficit_turning.md 4.4's corr(LLPC1 L-R, DNa02 L-R) and 5's median corr(DNge035 L-R, leg L-R) had no
        generator in scripts/; `wind-side --per-fly-corr` now emits both."""
        m = self.m
        runs = [self._run(s) for s in range(3)]
        df, st = m.per_fly_readout_corr(runs, self.SKIP_S)
        self.assertEqual(len(df), 3 * self.B * 2)                                 # 3 runs x B flies x 2 resolvable pairs
        st = st.set_index(["pre_LR", "readout_LR"])
        self.assertGreater(st.loc[("LLPC1_LR", "DNa02_LR"), "median_r_over_runs"], 0.9)
        self.assertLess(st.loc[("DNge035_LR", "leg_LR"), "median_r_over_runs"], -0.9)
        self.assertEqual(int(st.loc[("LLPC1_LR", "DNa02_LR"), "n_runs"]), 3)
        self.assertEqual(len(st.loc[("LLPC1_LR", "DNa02_LR"), "median_r_per_run"]), 3)
        # HSS / LPT22 are not in this recording's watch list: the generator skips them rather than failing
        self.assertNotIn(("HSS_LR", "DNa02_LR"), st.index)


class ApplyObjectTests(unittest.TestCase):
    """scripts/interp_apply_object.py on graph(): the `edges` lesion kind's arithmetic, the size geometry, the batch
    lines, and the trace-path derivation that used to be a fixed path."""

    @classmethod
    def setUpClass(cls):
        cls.m = _load_script("interp_apply_object")

    def test_edge_lesion_keeps_the_pattern_and_edits_only_the_block(self):
        m = self.m
        c = graph(); W0 = c.W.tocsr().copy()
        rec = m.apply_edge_lesion(c, "R1-R6", "Mi4", 0.0)
        self.assertEqual(rec["n_entries"], 1)
        self.assertEqual(rec["synapses"], 40.0)
        self.assertEqual(c.W.nnz, W0.nnz)                       # explicit zeros: the sparsity pattern survives
        self.assertEqual(c.W[1, 0], 0.0)
        self.assertEqual(c.W[2, 1], W0[2, 1])                   # every other entry untouched

        c = graph()
        rec = m.apply_edge_lesion(c, "LC4", "DNp01", -1.0)
        self.assertEqual(rec["n_entries"], 1)
        self.assertEqual(c.W[3, 2], -W0[3, 2])                  # the counterfactual sign flip

        c = graph()
        rec = m.apply_edge_lesion(c, "LC4", "DNp01|PEN_a", 2.0)
        self.assertEqual(rec["n_entries"], 2)
        self.assertEqual(c.W[3, 2], 2 * W0[3, 2])
        self.assertEqual(c.W[7, 2], 2 * W0[7, 2])

    def test_edge_selection_and_the_receptor_fast_sign_follow_the_factor(self):
        from flyverse import brain
        m = self.m
        c = graph()
        sel, _, _ = m.edge_selection(c, "LC4", "DNp01")
        self.assertEqual(int(sel.sum()), 1)
        r = brain._receptor(c, brain.LIFParams())
        self.assertIsNotNone(r)
        self.assertEqual(r.fast_sign[sel][0], 1.0)
        fs = np.array(r.fast_sign, copy=True); fs[sel] *= -1
        self.assertEqual(fs[sel][0], -1.0)
        self.assertTrue((fs[~sel] == r.fast_sign[~sel]).all())   # a matched entry's table sign must not leak

    def test_size_geometry_and_batch_lines(self):
        m = self.m
        g = m.angular_size_from_eye(m.ball_radius_for(11.4))
        self.assertAlmostEqual(g["angular_diameter_deg_probe"], 11.4, delta=0.05)
        self.assertAlmostEqual(g["ball_radius_m"], 0.00499, delta=2e-4)
        for deg in m.SIZES_DEG:
            gg = m.angular_size_from_eye(m.ball_radius_for(deg))
            self.assertAlmostEqual(gg["angular_diameter_deg_probe"], deg, delta=0.05)
        line = m.job_line("t3_off_held", "stim", 2, "out/x")
        self.assertTrue(line.startswith("mkdir -p out/x && python -c 'import torch; assert torch.cuda.is_available()' && "))
        self.assertIn("--lesion t3_off_held --arm stim --seed 2", line)
        lj = m.ladder_job(30.0, "null", 0, "out/y", True, "default")
        self.assertIn("--null", lj)
        self.assertIn("--retina", lj)
        self.assertIn(f"--ball-radius {m.ball_radius_for(30.0)}", lj)
        for k in ("base", "fb0", "inl1", "outl2", "rect", "t3_off_held", "t3_on_held", "t3_off_flip"):
            self.assertIn(k, m.LESIONS)

    def test_trace_path_is_derived_from_json_or_trace_dir(self):
        """The skeptic's tool defect: `analyse` wrote trace_<lesion>.json to a FIXED path regardless of --dir and
        --json, so a second recording set silently overwrote the first set's per-arm traces."""
        m = self.m
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.assertEqual(m.trace_dir_for(SimpleNamespace(trace_dir=None, json=None)), m.OUT_JSON)
            self.assertEqual(m.trace_dir_for(SimpleNamespace(trace_dir=None, json=str(d / "a" / "les.json"))), d / "a")
            self.assertEqual(m.trace_dir_for(SimpleNamespace(trace_dir=str(d / "t"), json=str(d / "a" / "les.json"))), d / "t")
            # two analyses with their own --json no longer collide
            one = m.trace_dir_for(SimpleNamespace(trace_dir=None, json=str(d / "one" / "les.json")))
            two = m.trace_dir_for(SimpleNamespace(trace_dir=None, json=str(d / "two" / "les.json")))
            self.assertNotEqual(one / "trace_base.json", two / "trace_base.json")


class ApplyRotationTests(unittest.TestCase):
    """scripts/interp_apply_rotation.py's statistics on synthetic data: the sided flip table, the bump / heading
    metrics across the 16-wedge wrap, and `verify-batch`, the check that catches two clients fetching one directory."""

    @classmethod
    def setUpClass(cls):
        cls.m = _load_script("interp_apply_rotation")

    def _fake_c(self):
        types = np.array(["HSN", "HSN", "GLNO", "GLNO", "PEN", "PEN", "T4", "T4"])
        n = pd.DataFrame({"type": types, "somaSide": np.array(["L", "R"] * 4), "bodyId": np.arange(8) + 100,
                          "superclass": ["visual_projection"] * 6 + ["ol_intrinsic"] * 2})
        return SimpleNamespace(n=8, neurons=n), types, n

    def test_flip_table_finds_a_planted_sided_flip_and_not_its_neighbour(self):
        m = self.m
        c, types, n = self._fake_c()
        rng = np.random.default_rng(0)
        orig = common.unit_kinds
        common.unit_kinds = lambda cc, fb=None: np.array(["spiking"] * 6 + ["graded"] * 2, dtype=object)
        try:
            def rec(phase, k):
                x = rng.normal(5.0, 0.3, 8); x[6:] = np.nan
                dr = np.full(8, np.nan); dr[6:] = rng.normal(0.0, 0.01, 2)
                if phase == "ccw":
                    x[0] += 3.0; dr[6] += 0.05
                if phase == "cw":
                    x[1] += 3.0; dr[7] += 0.05
                return common.Recording(np.array([0.0]), np.arange(8), n.bodyId.to_numpy(), types,
                                        {"rate_hz": x[None].astype(np.float32), "optic_dr": dr[None].astype(np.float32)},
                                        {}, {"seed": k, "arm": phase})
            runs = {ph: [rec(ph, k) for k in range(5)] for ph in ("rest", "ccw", "rest2", "cw")}
            df = m.flip_table(c, runs).set_index("type")
        finally:
            common.unit_kinds = orig
        self.assertEqual(df.loc["HSN", "verdict"], "result")
        self.assertGreater(df.loc["HSN", "flip_mean"], 5.0)
        self.assertEqual(df.loc["GLNO", "verdict"], "null")          # the unflipped neighbour stays null
        self.assertEqual(df.loc["T4", "verdict"], "result")
        self.assertEqual(df.loc["T4", "unit_kind"], "graded")        # a graded unit is read off optic_dr

    def test_bump_and_heading_metrics_across_the_wrap(self):
        m = self.m
        track = []
        for k in range(1000):
            cen = (13.0 + 4.0 * k * 0.01) % 16                        # +4 wedges/s through the 16-wedge wrap
            prof = np.exp(-0.5 * ((np.arange(16) - cen + 8) % 16 - 8) ** 2 / 1.5) * 200
            cc, vs = m.circ_centre(prof)
            track.append((cc, vs, prof.max()))
        b = m.bump_metrics(track, 300)
        self.assertAlmostEqual(b["drift_wedges_per_s"], 4.0, delta=0.01)
        self.assertAlmostEqual(b["net_wedges"], 4.0 * 6.99, delta=0.1)
        self.assertGreater(b["vs"], 0.8)
        h = m.heading_metrics(np.deg2rad(90.0) * np.arange(1000) * 0.01, 300)
        self.assertAlmostEqual(h["rate_dps"], 90.0, delta=0.01)

    def test_verify_batch_flags_an_interleaved_fetch(self):
        """One job writes the `_run.json`, the recording meta, the `_pen` meta and the console, so they must agree;
        a directory two clients fetched into holds a mix and this is the check that says so (2.1 of the audit)."""
        m = self.m

        def write(d, stem, drift, heading, drift_rec=None, run_dir="rot-cf0c43"):
            drift_rec = drift if drift_rec is None else drift_rec
            phases = {ph: {"bump": {"drift_wedges_per_s": drift}, "heading": {"rate_dps": heading}} for ph, _ in m.PHASES}
            (d / f"{stem}_run.json").write_text(json.dumps({"device": "cuda", "phases": phases}), encoding="utf-8")
            txt = [f"run dir /mnt/beegfs/neurome/runs/{run_dir}"]
            for ph, _ in m.PHASES:
                (d / f"{stem}_{ph}.json").write_text(json.dumps({"meta": {"bump": {"drift_wedges_per_s": drift_rec}}}), encoding="utf-8")
                (d / f"{stem}_{ph}_pen.json").write_text(json.dumps({"meta": {"bump": {"drift_wedges_per_s": drift_rec}}}), encoding="utf-8")
                txt.append(f"  {ph} (10.0 s): drift {drift:+.4f} w/s, heading {heading:+.2f} deg/s")
            (d / f"{stem}.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write(d, "default_visual_r0", +0.0040, +90.00)
            write(d, "default_visual_r1", -0.0020, -90.00)
            df = m.verify_batch(str(d))
            self.assertEqual(len(df), 2 * len(m.PHASES))
            self.assertTrue(df.consistent.all())
            self.assertEqual(sorted(set(df.run_dirs_in_console)), ["rot-cf0c43"])

            # the interleaved case: the recording meta comes from a different job than the run json / console
            write(d, "default_visual_r2", +0.0040, +90.00, drift_rec=-2.6570, run_dir="rot-7de91e")
            df = m.verify_batch(str(d))
            bad = df[~df.consistent]
            self.assertEqual(len(bad), len(m.PHASES))
            self.assertEqual(sorted(set(bad.run)), ["default_visual_r2"])
            self.assertEqual(int(df.consistent.sum()), 2 * len(m.PHASES))

if __name__ == "__main__":
    unittest.main()
