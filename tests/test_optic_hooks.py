"""The opt-in per-stream hooks of the rate optic lobe (OpticParams.stream_rectify / stream_adapt / spatial_suppress /
fb_hold; docs/audits/optic_stream_hooks.md). CPU only, explicit device 'cpu'.

    CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python -m unittest discover -s tests -p test_optic_hooks.py

(a) the shipped defaults are bit-identical with the hook code in place (against the committed optic.py, and against a
    hook that matches nothing); (b) each hook changes only its own stream -- the unmatched contributions are
    bit-identical; (c) the sign of every weight is preserved under every rectification mode; (d) spatial suppression
    sends a spatially uniform input to (1 - k) x itself (0 at k = 1) and scales a single-column input by 1 - k / n;
    (e) fb_hold ('.*', '.*') is gain_fb = 0 and a narrow hold touches only its block; (f) the CLI grammar."""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")        # a CPU test never touches this machine's GPU (the cluster rule)

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyverse import connectome as cn                      # noqa: E402
from flyverse import optic, retina as rt                   # noqa: E402
from flyverse.interp import common                         # noqa: E402
from flyverse.interp.trace import column_of_cells          # noqa: E402

CACHE = (cn.CACHE_DIR / "W_post_pre.npz").exists()


def to_scipy(M) -> sp.csr_matrix:
    if isinstance(M, sp.spmatrix):
        return M.tocsr()
    M = M.cpu()
    if M.layout == torch.sparse_csr:
        return sp.csr_matrix((M.values().numpy(), M.col_indices().numpy(), M.crow_indices().numpy()), shape=tuple(M.shape))
    M = M.coalesce()
    return sp.csr_matrix((M.values().numpy(), (M.indices()[0].numpy(), M.indices()[1].numpy())), shape=tuple(M.shape))


# ---------------------------------------------------------------------------------------------- a synthetic hex patch
HEX = [(0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, -1)]      # a centre column and its six nearest neighbours


def hex_graph():
    """7 photoreceptors (one per column), 7 Mi1 (ACh, one per column), one Pm1 (GABA, centre column), one T3 (ACh, no
    column: it takes the column of its strongest input) and one LC11 (spiking). Edges: R -> Mi1 -10 (histamine) per column,
    Mi1 -> T3 +10 each, Pm1 -> T3 -5, Mi1(centre) -> Pm1 +4, T3 -> LC11 +20."""
    rows = []
    for k, (h1, h2) in enumerate(HEX):
        rows.append(dict(bodyId=100 + k, type="R1-R6", superclass="ol_sensory", nt="histamine", hex1=h1, hex2=h2, hex_side="L"))
    for k, (h1, h2) in enumerate(HEX):
        rows.append(dict(bodyId=200 + k, type="Mi1", superclass="ol_intrinsic", nt="acetylcholine", hex1=h1, hex2=h2, hex_side="L"))
    rows.append(dict(bodyId=300, type="Pm1", superclass="ol_intrinsic", nt="gaba", hex1=0, hex2=0, hex_side="L"))
    rows.append(dict(bodyId=400, type="T3", superclass="ol_intrinsic", nt="acetylcholine", hex1=np.nan, hex2=np.nan, hex_side=None))
    rows.append(dict(bodyId=500, type="LC11", superclass="visual_projection", nt="acetylcholine", hex1=np.nan, hex2=np.nan, hex_side=None))
    n = pd.DataFrame(rows)
    n["class"] = ""; n["subclass"] = ""; n["somaSide"] = "L"; n["instance"] = n.type + "_x"
    n["sign"] = np.array([cn.NT_SIGN[x] for x in n.nt], dtype=np.float32)
    N = len(n); i_pr = list(range(0, 7)); i_mi = list(range(7, 14)); i_pm, i_t3, i_lc = 14, 15, 16
    post, pre, val = [], [], []
    for k in range(7):
        post.append(i_mi[k]); pre.append(i_pr[k]); val.append(-10.0)
        post.append(i_t3); pre.append(i_mi[k]); val.append(10.0)
    post.append(i_t3); pre.append(i_pm); val.append(-5.0)
    post.append(i_pm); pre.append(i_mi[0]); val.append(4.0)
    post.append(i_lc); pre.append(i_t3); val.append(20.0)
    W = sp.csr_matrix((np.array(val, np.float32), (post, pre)), shape=(N, N)); W.sort_indices()
    return cn.Connectome(n, W, pd.Series(np.arange(N), index=n.bodyId.to_numpy()))


def plain_params(**kw):
    """No pair gains / per-type constants: the equations reduce to the docstring's."""
    return optic.OpticParams(pair_gain=[], tau_by_type={}, baseline_by_type={}, adapt_gain=0.0, gain_fb=0.0, **kw)


class HexPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = hex_graph(); cls.r = rt.build_retina(cls.c)
        cls.ol0 = optic.OpticLobe(cls.c, cls.r, plain_params(), device="cpu")
        ridx = cls.ol0.rate_idx; cls.types = cls.c.neurons.type.to_numpy()[ridx]
        cls.mi = np.flatnonzero(cls.types == "Mi1"); cls.pm = int(np.flatnonzero(cls.types == "Pm1")[0]); cls.t3 = int(np.flatnonzero(cls.types == "T3")[0])

    def test_retina_and_columns(self):
        self.assertEqual(self.r.n_columns, 7)
        col, n_ann = column_of_cells(self.c, self.r, self.ol0.rate_idx)
        self.assertEqual(n_ann, 8)                                     # 7 Mi1 + Pm1 annotated; T3 propagated
        self.assertGreaterEqual(int(col[self.ol0.rate_idx][self.t3]), 0)

    def neighbours(self, radius_deg):
        """n_j per Mi1 cell from the retina geometry alone: Mi1 columns within radius of the cell's column, itself included."""
        cd = np.asarray(self.r.col_dir); col = column_of_cells(self.c, self.r, self.ol0.rate_idx)[0][self.ol0.rate_idx]
        cm = col[self.mi]
        ang = np.degrees(np.arccos(np.clip(cd[cm] @ cd[cm].T, -1, 1)))
        return (ang <= radius_deg).sum(1), ang

    def test_spatial_suppress_uniform_and_single_column(self):
        for k in (1.0, 0.5):
            ol = optic.OpticLobe(self.c, self.r, plain_params(spatial_suppress=[("^Mi1$", k, 5.0)]), device="cpu")
            self.assertTrue(ol._hooks); self.assertFalse(ol.cuda); self.assertFalse(ol.metal)
            self.assertEqual(ol.hook_info["spatial"]["entries"][0]["cells_with_column"], 7)
            n_j, ang = self.neighbours(5.0)
            self.assertEqual(int(n_j[0]), 7)                            # the centre sees all seven columns
            self.assertEqual(ol.hook_info["spatial"]["entries"][0]["neighbours_max"], 7)
            # (d1) uniform input over the Mi1 stream -> (1 - k) x itself, i.e. 0 at k = 1
            dr = torch.zeros(1, ol.n_rate); dr[0, self.mi] = 0.3
            u = ol._stream_signals(dr)[0]
            np.testing.assert_allclose(u[0, self.mi].numpy(), 0.3 * (1 - k), atol=1e-6)
            if k == 1.0:
                self.assertLess(float(u[0, self.mi].abs().max()), 1e-6)
            self.assertEqual(float(u[0, self.pm]), 0.0); self.assertEqual(float(u[0, self.t3]), 0.0)   # untouched cells
            # (d2) a single column: the cell itself scaled by 1 - k / n_j, each neighbour j' gets -k x / n_j'
            dr = torch.zeros(1, ol.n_rate); j = 0; dr[0, self.mi[j]] = 0.5
            u = ol._stream_signals(dr)[0][0, self.mi].numpy()
            self.assertAlmostEqual(float(u[j]), 0.5 * (1 - k / n_j[j]), places=6)
            for jj in range(1, 7):
                self.assertAlmostEqual(float(u[jj]), -k * 0.5 / n_j[jj], places=6, msg=f"neighbour {jj}")
            self.assertTrue((ang[0, 1:] <= 5.0).all())

    def test_rectify_sign_preserved_per_entry(self):
        """Every (post <- pre) entry of the stream contributes W_ij x_j with the sign of W_ij under pos / neg / abs; the
        block's weights are the shipped W_rr entries, untouched."""
        W0 = to_scipy(self.ol0.W_rr)
        gen = torch.Generator().manual_seed(1)
        for mode in ("pos", "neg", "abs"):
            ol = optic.OpticLobe(self.c, self.r, plain_params(stream_rectify=[("^(Mi1|Pm1)$", "^T3$", mode)]), device="cpu")
            self.assertEqual(len(ol.streams), 1); blk = to_scipy(ol.streams[0]["W_rr"]).tocoo()
            self.assertEqual(blk.nnz, 8)                                 # 7 Mi1 + Pm1 -> T3
            np.testing.assert_array_equal(blk.data, np.asarray(W0[blk.row, blk.col]).ravel())
            self.assertTrue((blk.data[blk.col == self.pm] < 0).all())
            dr = torch.rand(1, ol.n_rate, generator=gen) * 2 - 1        # both signs
            x = ol._stream_signals(dr)[0][0].numpy()
            self.assertTrue((x >= 0).all())
            contrib = blk.data * x[blk.col]
            self.assertTrue((contrib[blk.data < 0] <= 0).all(), mode)   # an inhibitory entry never contributes positively
            self.assertTrue((contrib[blk.data > 0] >= 0).all(), mode)
            y = dr[0].numpy()
            expect = {"pos": np.maximum(y, 0), "neg": np.maximum(-y, 0), "abs": np.abs(y)}[mode]
            np.testing.assert_allclose(x, expect, atol=1e-7)
            # the same entries under the plain sum flip with the presynaptic sign (the deviation model's disinhibition)
            lin = blk.data * y[blk.col]
            self.assertTrue((lin[blk.data < 0] > 0).any() or (lin[blk.data > 0] < 0).any())

    def test_rectify_adds_cancelling_channels(self):
        """Two excitatory inputs of opposite deviation cancel in the sum; 'pos' on one and 'neg' on the other add."""
        ol = optic.OpticLobe(self.c, self.r, plain_params(stream_rectify=[("^Mi1$", "^T3$", "pos"), ("^Pm1$", "^T3$", "neg")]), device="cpu")
        dr = torch.zeros(1, ol.n_rate); dr[0, self.mi] = 0.1; dr[0, self.pm] = -0.2
        lin = self.ol0._recurrent(dr)[0, self.t3].item(); hooked = ol._recurrent(dr)[0, self.t3].item()
        W0 = to_scipy(self.ol0.W_rr)
        w_mi = float(W0[self.t3, self.mi[0]]); w_pm = float(W0[self.t3, self.pm])
        self.assertAlmostEqual(lin, 7 * w_mi * 0.1 + w_pm * (-0.2), places=6)      # Pm1 (GABA) below rest: disinhibits
        self.assertAlmostEqual(hooked, 7 * w_mi * 0.1 + w_pm * 0.2, places=6)      # 'neg': OFF half-wave as a positive x; W < 0 keeps inhibiting

    def test_adapt_state_dynamics_and_reset(self):
        tau, gain = 20.0, 1.0
        ol = optic.OpticLobe(self.c, self.r, plain_params(stream_adapt=[("^Mi1$", "^T3$", tau, gain)]), device="cpu")
        self.assertEqual(ol.stream_adapt_state.shape, (1, 1, ol.n_rate))
        dr = torch.zeros(1, ol.n_rate); dr[0, self.mi] = 0.4
        a = np.exp(-ol.p.dt_ms / tau)
        for n in range(1, 6):
            x_before = ol._stream_signals(dr)[0][0, self.mi[0]].item()
            self.assertAlmostEqual(x_before, 0.4 - gain * 0.4 * (1 - a ** (n - 1)), places=6)
            ol._recurrent(dr, update=True)
            self.assertAlmostEqual(ol.stream_adapt_state[0, 0, self.mi[0]].item(), 0.4 * (1 - a ** n), places=6)
        ol.reset()
        self.assertEqual(float(ol.stream_adapt_state.abs().sum()), 0.0)
        with self.assertRaises(ValueError):
            optic.OpticLobe(self.c, self.r, plain_params(stream_rectify=[("^Mi1$", "^T3$", "relu")]), device="cpu")
        with self.assertRaises(ValueError):
            optic.OpticLobe(self.c, self.r, plain_params(stream_adapt=[("^Mi1$", "^T3$", 0.0, 1.0)]), device="cpu")

    def test_parse_kv_grammar(self):
        kv = common.parse_kv(["stream_rectify=^(Mi1|Tm3|Tm2)$,^T3$,pos;^(Tm1|Tm4)$,^T3$,neg", "spatial_suppress=^(Mi1|Tm1|Tm3|Tm4)$,0.5,10",
                              "stream_adapt=^Mi1$,^T3$,100,1", "fb_hold=.*,.*", "norm=l1", "gain_fb=0", 'pair_gain=[["^Mi1$", "^T3$", 2.0]]'])
        self.assertEqual(kv["stream_rectify"], [["^(Mi1|Tm3|Tm2)$", "^T3$", "pos"], ["^(Tm1|Tm4)$", "^T3$", "neg"]])
        self.assertEqual(kv["spatial_suppress"], [["^(Mi1|Tm1|Tm3|Tm4)$", 0.5, 10.0]])
        self.assertEqual(kv["stream_adapt"], [["^Mi1$", "^T3$", 100.0, 1.0]]); self.assertEqual(kv["fb_hold"], [[".*", ".*"]])
        self.assertEqual(kv["norm"], "l1"); self.assertEqual(kv["gain_fb"], 0); self.assertEqual(kv["pair_gain"], [["^Mi1$", "^T3$", 2.0]])
        p = optic.OpticParams(**{k: v for k, v in kv.items() if k in ("stream_rectify", "spatial_suppress", "stream_adapt", "fb_hold")})
        ol = optic.OpticLobe(self.c, self.r, p, device="cpu")
        self.assertTrue(ol._hooks); self.assertEqual(len(ol.streams), 2)          # (rect 0 + adapt 0) Mi1 -> T3, and Mi1 -> Pm1 suppressed only
        self.assertEqual([(x["mode"], x["adapt_id"]) for x in ol.streams], [(None, -1), ("pos", 0)])
        # the old behaviour is untouched: literals, JSON, plain strings, and a comma under a non-list key
        self.assertEqual(common.parse_kv(["receptor_model=None", "x=[1, 2]", "s=abc", "t=a,b"]), {"receptor_model": None, "x": [1, 2], "s": "abc", "t": "a,b"})


# ---------------------------------------------------------------------------------------------- the cached subset
@unittest.skipUnless(CACHE, "connectome cache absent")
class CachedSubsetTests(unittest.TestCase):
    TYPES = ["R1-R6", "R7p", "R7y", "R8p", "R8y", "L1", "L2", "Mi1", "Tm3", "Tm2", "Tm1", "Tm4", "T3", "T2", "Tm5Y", "LC11", "LC10a", "LoVC16"]
    RECT = [("^(Mi1|Tm3|Tm2)$", "^T3$", "pos"), ("^(Tm1|Tm4)$", "^T3$", "neg")]

    @classmethod
    def setUpClass(cls):
        c = cn.load(verbose=False)
        cls.sub = c.subset(c.select(type=cls.TYPES)); cls.r = rt.build_retina(cls.sub)
        cls.ol0 = optic.OpticLobe(cls.sub, cls.r, optic.OpticParams(), device="cpu")
        cls.types = cls.sub.neurons.type.fillna("").to_numpy()[cls.ol0.rate_idx]
        gen = torch.Generator().manual_seed(0)
        cls.rad = torch.rand(1, cls.r.n_columns, 4, generator=gen)
        cls.spk = torch.zeros(1, cls.sub.n); cls.spk[0, cls.sub.select(type="LoVC16")] = 60.0     # a feedback source

    def frames(self, ol, n=12):
        ol.reset(); ol.relax()
        out = []
        for k in range(n):
            out.append(ol.step_frame(self.rad * (1.0 + 0.5 * np.sin(k / 2.0)), self.spk, 10.0).clone())
        return out

    def assert_bit_identical(self, a, b):
        for x, y in zip(self.frames(a), self.frames(b)):
            self.assertTrue(torch.equal(x, y))
        self.assertTrue(torch.equal(a.v, b.v)); self.assertTrue(torch.equal(a.adapt, b.adapt)); self.assertTrue(torch.equal(a.delta_rate, b.delta_rate))

    def test_a_defaults_bit_identical_to_committed_optic(self):
        """The shipped defaults through the hook-aware code against the committed flyverse/optic.py (git HEAD), loaded as
        a sibling module: every frame's drive and the state tensors equal bit for bit."""
        try:
            src = subprocess.run(["git", "show", "HEAD:flyverse/optic.py"], cwd=ROOT, capture_output=True, text=True, timeout=30, check=True).stdout
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"git unavailable: {e!r}")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "_optic_head.py"; path.write_text(src, encoding="utf-8")
            spec = importlib.util.spec_from_file_location("flyverse._optic_head", path)
            mod = importlib.util.module_from_spec(spec); mod.__package__ = "flyverse"
            sys.modules["flyverse._optic_head"] = mod; spec.loader.exec_module(mod)
        head_fields = set(mod.OpticParams.__dataclass_fields__)
        kw = {k: v for k, v in optic.OpticParams().__dict__.items() if k in head_fields}
        ol_head = mod.OpticLobe(self.sub, self.r, mod.OpticParams(**kw), device="cpu")
        self.assert_bit_identical(self.ol0, ol_head)
        self.assertFalse(self.ol0._hooks); self.assertEqual(self.ol0.streams, [])

    def test_a_no_match_hook_is_bit_identical(self):
        """A hook that matches no type walks the split code path with everything in the rest block: bit-identical."""
        ol = optic.OpticLobe(self.sub, self.r, optic.OpticParams(stream_rectify=[("^NoSuchType$", "^T3$", "pos")]), device="cpu")
        self.assertTrue(ol._hooks); self.assertEqual(ol.streams, []); self.assertEqual(ol.hook_info["entries_rr_matched"], 0)
        self.assertEqual(ol.hook_info["entries_rr_rest"], to_scipy(self.ol0.W_rr).nnz)
        self.assert_bit_identical(self.ol0, ol)
        ol = optic.OpticLobe(self.sub, self.r, optic.OpticParams(fb_hold=[("^NoSuchType$", ".*")]), device="cpu")
        self.assertFalse(ol._hooks); self.assert_bit_identical(self.ol0, ol)

    def test_a_default_substep_matches_the_documented_expression(self):
        """One frame of the shipped model, recomputed from the docstring's equations with the lobe's own tensors."""
        ol = optic.OpticLobe(self.sub, self.r, optic.OpticParams(), device="cpu")
        ol.reset(); ol.relax(); p = ol.p
        ol.step_frame(self.rad, self.spk, 10.0); ol.step_frame(self.rad * 1.3, self.spk, 10.0)
        a_pr = ol.photoreceptor_activity(self.rad * 0.7, 10.0)
        s = (self.spk[:, ol.spk_idx_t] / 100.0).clamp(0, 3)
        pr_input = p.gain_in * optic._mv(ol.W_rp, a_pr); spk_input = p.gain_fb * optic._mv(ol.W_rs, s)
        v, adapt = ol.v.clone(), ol.adapt.clone()
        for _ in range(10):
            dr = (v + ol.b_vec[None]).clamp(0, 1) - ol.b_vec[None]
            inp = p.gain_rr * optic._mv(ol.W_rr, dr) + pr_input - p.adapt_gain * adapt
            inp = inp + spk_input
            v = inp + (v - inp) * ol._a[None]; adapt = dr + (adapt - dr) * ol._a_ad
        # replay the same frame on the lobe (its photoreceptor stage was advanced by the call above: rewind it)
        ol2 = optic.OpticLobe(self.sub, self.r, optic.OpticParams(), device="cpu"); ol2.reset(); ol2.relax()
        ol2.step_frame(self.rad, self.spk, 10.0); ol2.step_frame(self.rad * 1.3, self.spk, 10.0); ol2.step_frame(self.rad * 0.7, self.spk, 10.0)
        self.assertTrue(torch.equal(ol2.v, v)); self.assertTrue(torch.equal(ol2.adapt, adapt))

    def masks(self, rect):
        """Entry masks over the shipped W_rr (CSR order) for the stream blocks of `rect`."""
        W = to_scipy(self.ol0.W_rr).tocoo(); m = np.zeros(W.nnz, bool)
        for pre_re, post_re, _ in rect:
            m |= optic.OpticLobe._type_mask(pre_re, self.types)[W.col] & optic.OpticLobe._type_mask(post_re, self.types)[W.row]
        return W, m

    def test_b_rectify_changes_only_its_stream(self):
        ol = optic.OpticLobe(self.sub, self.r, optic.OpticParams(stream_rectify=self.RECT), device="cpu")
        self.assertTrue(ol._hooks); self.assertFalse(ol.cuda); self.assertFalse(ol.metal); self.assertEqual(len(ol.streams), 2)
        W, m = self.masks(self.RECT)
        # the split is exact: rest = the unmatched entries, blocks = the matched ones, weights untouched
        rest = to_scipy(ol.W_rr_rest).tocoo(); self.assertEqual(rest.nnz, int((~m).sum()))
        R = sp.csr_matrix((W.data[~m], (W.row[~m], W.col[~m])), shape=W.shape)
        self.assertEqual(abs(R - to_scipy(ol.W_rr_rest)).max(), 0.0)
        blk = sum(to_scipy(s["W_rr"]) for s in ol.streams); M = sp.csr_matrix((W.data[m], (W.row[m], W.col[m])), shape=W.shape)
        self.assertEqual(abs(M - blk).max(), 0.0); self.assertGreater(M.nnz, 10000)
        # the same dr: unmatched post rows bit-identical to the plain sum; T3 rows = rest + rectified blocks
        gen = torch.Generator().manual_seed(3); dr = (torch.rand(1, ol.n_rate, generator=gen) - 0.5) * 0.4
        plain = self.ol0._recurrent(dr); hooked = ol._recurrent(dr)
        t3 = self.types == "T3"
        self.assertTrue(torch.equal(plain[0, ~t3], hooked[0, ~t3]))
        self.assertFalse(torch.allclose(plain[0, t3], hooked[0, t3]))
        y = dr[0].numpy()
        M_pos = to_scipy(ol.streams[0]["W_rr"]) if ol.streams[0]["mode"] == "pos" else to_scipy(ol.streams[1]["W_rr"])
        M_neg = to_scipy(ol.streams[1]["W_rr"]) if ol.streams[1]["mode"] == "neg" else to_scipy(ol.streams[0]["W_rr"])
        expect = ol.p.gain_rr * (R @ y + M_pos @ np.maximum(y, 0) + M_neg @ np.maximum(-y, 0))
        np.testing.assert_allclose(hooked[0].numpy(), expect, atol=1e-5)
        # the rest product itself is bit-identical to the plain product on rows without a matched entry
        rest_rows = np.ones(ol.n_rate, bool); rest_rows[np.unique(W.row[m])] = False
        self.assertTrue(torch.equal(optic._mv(ol.W_rr_rest, dr)[0, rest_rows], optic._mv(self.ol0.W_rr, dr)[0, rest_rows]))
        # and through whole frames the non-T3 cells' inputs are the linear ones: the drive of LC11 differs only through T3
        f0 = self.frames(self.ol0); f1 = self.frames(ol)
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(f0, f1)))

    def test_b_adapt_changes_only_its_stream(self):
        ol = optic.OpticLobe(self.sub, self.r, optic.OpticParams(stream_adapt=[("^(Mi1|Tm3|Tm2|Tm1|Tm4)$", "^T3$", 100.0, 1.0)]), device="cpu")
        self.assertEqual(len(ol.streams), 1); self.assertEqual(ol.streams[0]["mode"], None); self.assertEqual(ol.streams[0]["adapt_id"], 0)
        gen = torch.Generator().manual_seed(4); dr = (torch.rand(1, ol.n_rate, generator=gen) - 0.5) * 0.4
        plain = self.ol0._recurrent(dr)
        first = ol._recurrent(dr, update=True)                          # state 0: x = dr, every row equals the plain sum
        np.testing.assert_allclose(first[0].numpy(), plain[0].numpy(), atol=1e-6)
        second = ol._recurrent(dr, update=True)                         # state moved: only T3 rows change
        t3 = self.types == "T3"
        self.assertTrue(torch.equal(second[0, ~t3], plain[0, ~t3])); self.assertFalse(torch.allclose(second[0, t3], plain[0, t3]))
        a = np.exp(-1.0 / 100.0); M = to_scipy(ol.streams[0]["W_rr"]); R = to_scipy(ol.W_rr_rest); y = dr[0].numpy()
        expect = ol.p.gain_rr * (R @ y + M @ (y - 1.0 * y * (1 - a)))
        np.testing.assert_allclose(second[0].numpy(), expect, atol=1e-5)

    def test_b_suppress_changes_only_its_stream(self):
        ol = optic.OpticLobe(self.sub, self.r, optic.OpticParams(spatial_suppress=[("^(Mi1|Tm1|Tm3|Tm4)$", 0.5, 10.0)]), device="cpu")
        info = ol.hook_info["spatial"]["entries"][0]
        self.assertGreater(info["cells_with_column"], 0.99 * info["cells"]); self.assertGreater(info["neighbours_mean"], 5)
        G = to_scipy(ol.G_supp).tocoo()
        pre_m = optic.OpticLobe._type_mask("^(Mi1|Tm1|Tm3|Tm4)$", self.types)
        self.assertTrue(pre_m[G.row].all()); self.assertTrue(pre_m[G.col].all())
        self.assertTrue((self.types[G.row] == self.types[G.col]).all())            # within a type only
        rs = np.asarray(to_scipy(ol.G_supp).sum(1)).ravel()
        np.testing.assert_allclose(rs[rs > 0], 0.5, atol=1e-6)                  # rows sum to k
        gen = torch.Generator().manual_seed(5); dr = (torch.rand(1, ol.n_rate, generator=gen) - 0.5) * 0.4
        plain = self.ol0._recurrent(dr); hooked = ol._recurrent(dr)
        W, m = self.masks([("^(Mi1|Tm1|Tm3|Tm4)$", ".*", None)])
        rows_hit = np.zeros(ol.n_rate, bool); rows_hit[np.unique(W.row[m])] = True
        self.assertTrue(torch.equal(plain[0, ~rows_hit], hooked[0, ~rows_hit]))
        y = dr[0].numpy(); u = y - to_scipy(ol.G_supp) @ y
        R = to_scipy(ol.W_rr_rest); M = to_scipy(ol.streams[0]["W_rr"])
        np.testing.assert_allclose(hooked[0].numpy(), ol.p.gain_rr * (R @ y + M @ u), atol=1e-5)
        # a uniform deviation over one type is halved at k = 0.5 (every neighbourhood is uniform)
        dr = torch.zeros(1, ol.n_rate); mi = self.types == "Mi1"; dr[0, torch.from_numpy(mi)] = 0.2
        u = ol._stream_signals(dr)[0][0].numpy()
        with_col = mi & (np.asarray(to_scipy(ol.G_supp).sum(1)).ravel() > 0)
        np.testing.assert_allclose(u[with_col], 0.1, atol=1e-6)

    def test_e_fb_hold(self):
        ol_fb0 = optic.OpticLobe(self.sub, self.r, optic.OpticParams(gain_fb=0.0), device="cpu")
        ol_all = optic.OpticLobe(self.sub, self.r, optic.OpticParams(fb_hold=[(".*", ".*")]), device="cpu")
        self.assertFalse(ol_all._hooks); self.assertEqual(to_scipy(ol_all.W_rs).nnz, 0); self.assertEqual(ol_all.hook_info_fb["entries_after"], 0)
        self.assertGreater(ol_all.hook_info_fb["entries_before"], 0)
        self.assert_bit_identical(ol_fb0, ol_all)
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(self.frames(self.ol0), self.frames(ol_fb0))))   # the feedback source does act
        ol_t3 = optic.OpticLobe(self.sub, self.r, optic.OpticParams(fb_hold=[("^LoVC16$", "^T3$")]), device="cpu")
        W0 = to_scipy(self.ol0.W_rs); W1 = to_scipy(ol_t3.W_rs); D = (W0 - W1).tocoo()
        st = self.sub.neurons.type.fillna("").to_numpy()[self.ol0.spk_idx]
        self.assertGreater(D.nnz, 0); self.assertTrue((self.types[D.row] == "T3").all()); self.assertTrue((st[D.col] == "LoVC16").all())
        self.assertEqual(ol_t3.hook_info_fb["held"][0]["entries"], D.nnz)
        t3 = self.types == "T3"
        s = (self.spk[:, self.ol0.spk_idx_t] / 100.0).clamp(0, 3)
        self.assertTrue(torch.equal(optic._mv(self.ol0.W_rs, s)[0, ~t3], optic._mv(ol_t3.W_rs, s)[0, ~t3]))


if __name__ == "__main__":
    unittest.main()
