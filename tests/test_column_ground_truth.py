"""The leave-one-type-out machinery of scripts/probe_column_ground_truth.py on a tiny synthetic hex graph (CPU, no data).

    CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python -m pytest tests/test_column_ground_truth.py -q

Graph: 7 columns (a centre and its six neighbours, one side), one photoreceptor and one Mi1 per column (annotated),
one Tm1 per column (annotated) driven by its own column's Mi1 (+10) and its clockwise neighbour's Mi1 (+3), and one
wide-field 'Dm' cell annotated at the centre that drives every Tm1 with +2. Withholding Tm1 must recover every Tm1's
column exactly from its strongest input; the within-type permutation must not; withholding Mi1 AND Tm1 leaves Tm1 on
the wide-field cell's centre column (error 1 column = 4.6 deg for the six ring cells); degrees come from the retina's
own col_az_el."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from flyverse import connectome as cn, retina as rt                 # noqa: E402
import probe_column_ground_truth as P                              # noqa: E402

HEX = [(0, 0), (1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1)]   # centre, then the six neighbours in ring order


def hex_graph() -> cn.Connectome:
    rows = []
    for k, (h1, h2) in enumerate(HEX):
        rows.append(dict(bodyId=100 + k, type="R1-R6", superclass="ol_sensory", nt="histamine", hex1=h1, hex2=h2, hex_side="L", hex_source="partner"))
    for k, (h1, h2) in enumerate(HEX):
        rows.append(dict(bodyId=200 + k, type="Mi1", superclass="ol_intrinsic", nt="acetylcholine", hex1=h1, hex2=h2, hex_side="L", hex_source="annotation"))
    for k, (h1, h2) in enumerate(HEX):
        rows.append(dict(bodyId=300 + k, type="Tm1", superclass="ol_intrinsic", nt="acetylcholine", hex1=h1, hex2=h2, hex_side="L", hex_source="annotation"))
    rows.append(dict(bodyId=400, type="Dm", superclass="ol_intrinsic", nt="glutamate", hex1=0, hex2=0, hex_side="L", hex_source="annotation"))
    rows.append(dict(bodyId=500, type="LC11", superclass="visual_projection", nt="acetylcholine", hex1=np.nan, hex2=np.nan, hex_side=None, hex_source=""))
    n = pd.DataFrame(rows)
    n["class"] = ""; n["subclass"] = ""; n["somaSide"] = "L"; n["instance"] = n.type + "_x"
    n["sign"] = np.array([cn.NT_SIGN[x] for x in n.nt], dtype=np.float32)
    N = len(n); i_pr = list(range(0, 7)); i_mi = list(range(7, 14)); i_tm = list(range(14, 21)); i_dm, i_lc = 21, 22
    post, pre, val = [], [], []
    for k in range(7):
        post.append(i_mi[k]); pre.append(i_pr[k]); val.append(-10.0)
        post.append(i_tm[k]); pre.append(i_mi[k]); val.append(10.0)                    # own column: the strongest input
        post.append(i_tm[k]); pre.append(i_mi[1 + k % 6 if k else 1]); val.append(3.0)  # a neighbour's Mi1, weaker
        post.append(i_tm[k]); pre.append(i_dm); val.append(2.0)                       # the wide-field cell, weakest
    post.append(i_lc); pre.append(i_tm[3]); val.append(20.0)
    W = sp.csr_matrix((np.array(val, np.float32), (post, pre)), shape=(N, N)); W.sort_indices()
    return cn.Connectome(n, W, pd.Series(np.arange(N), index=n.bodyId.to_numpy()))


class LeaveOneOutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = hex_graph(); cls.r = rt.build_retina(cls.c); cls.ridx = P.rate_index(cls.c)

    def test_retina_and_baseline(self):
        self.assertEqual(self.r.n_columns, 7)
        self.assertEqual(len(self.ridx), 15)                              # 7 Mi1 + 7 Tm1 + Dm; no photoreceptor, no LC11
        col = P.infer_columns(self.c, self.r, self.ridx)
        self.assertTrue((col[self.ridx] >= 0).all())
        truth = P.truth_table(self.c, self.r, self.ridx)
        self.assertTrue((truth.truth_col >= 0).all())
        sc = P.score_cells(truth, col, self.r)
        self.assertTrue(sc.exact.all()); self.assertTrue((sc.deg_err == 0).all())

    def test_withhold_removes_only_the_named_type(self):
        h = P.withhold(self.c, ["Tm1"])
        self.assertTrue(h.neurons.loc[h.neurons.type == "Tm1", "hex1"].isna().all())
        self.assertEqual((h.neurons.hex_source == "annotation").sum(), 8)  # 7 Mi1 + Dm
        self.assertIs(h.W, self.c.W)
        self.assertFalse(self.c.neurons.loc[self.c.neurons.type == "Tm1", "hex1"].isna().any())   # the original is untouched

    def test_leave_one_out_recovers_tm1_exactly_and_chance_does_not(self):
        rng = np.random.default_rng(0)
        pt, pc, fp = P.leave_one_out(self.c, self.r, {"loo:Tm1": ["Tm1"]}, rng, n_shuffles=20, ridx=self.ridx, log=lambda *_: None)
        both = pt[(pt.type == "Tm1") & (pt.side == "both")].iloc[0]
        self.assertEqual(both.n_truth, 7); self.assertEqual(both.frac_assigned, 1.0); self.assertEqual(both.frac_exact, 1.0)
        self.assertEqual(both.median_col_err, 0.0); self.assertEqual(both.median_deg_err, 0.0); self.assertEqual(both.frac_within_io_deg, 1.0)
        self.assertEqual(both.n_withheld_types, 1)
        # chance: a permutation of 7 distinct columns leaves 1 / 7 fixed on average and a median error of about a column or more
        self.assertLess(both.chance_frac_exact, 0.5); self.assertGreater(both.chance_median_deg_err, 0.0)
        self.assertGreater(both.chance_median_col_err, 0.5)
        self.assertEqual(set(pc.type), {"Tm1"}); self.assertEqual(len(pc), 7)
        self.assertEqual(fp.partner_types_top5.iloc[0], {"Mi1": 7}); self.assertEqual(fp.n_bad.iloc[0], 0); self.assertEqual(fp.frac_partner_annotated.iloc[0], 1.0)

    def test_withholding_the_partner_too_falls_back_to_the_wide_field_column(self):
        rng = np.random.default_rng(0)
        pt, pc, fp = P.leave_one_out(self.c, self.r, {"both": ["Mi1", "Tm1"]}, rng, n_shuffles=2, ridx=self.ridx, log=lambda *_: None)
        tm = pt[(pt.type == "Tm1") & (pt.side == "both")].iloc[0]
        self.assertEqual(tm.frac_assigned, 1.0)                                          # Dm is still annotated at the centre
        self.assertAlmostEqual(tm.frac_exact, 1 / 7)                                     # only the centre Tm1 is right
        self.assertAlmostEqual(tm.median_col_err, 1.0)                                   # the ring is one column off
        self.assertAlmostEqual(tm.median_deg_err, rt.EyeGeometry().interommatidial_deg, delta=0.01)   # great circle vs the planar pitch
        mi = pt[(pt.type == "Mi1") & (pt.side == "both")].iloc[0]
        self.assertEqual(mi.frac_assigned, 0.0)                                          # Mi1 has no rate inputs at all
        self.assertTrue(np.isnan(mi.median_deg_err))
        tmp = fp[fp.type == "Tm1"].iloc[0]; self.assertEqual(tmp.partner_types_top5, {"Dm": 7}); self.assertEqual(tmp.n_bad, 0)   # one column (4.6 deg) is not > 10

    def test_shuffle_is_a_permutation_within_type_and_side(self):
        col = P.infer_columns(self.c, self.r, self.ridx)
        truth = P.truth_table(self.c, self.r, self.ridx)
        sh = P.shuffle_within(col, truth, np.random.default_rng(1))
        for t in ("Mi1", "Tm1"):
            idx = truth[truth.type == t].model_index.to_numpy()
            self.assertEqual(sorted(sh[idx].tolist()), sorted(col[idx].tolist()))
        self.assertEqual(sh[22], col[22])                                                # LC11 is not a truth cell: untouched

    def test_lc_case_and_partner(self):
        col = P.infer_columns(self.c, self.r, self.ridx)
        pc, summ = P.lc_case(self.c, self.r, col, ("LC11",), np.random.default_rng(0), n_shuffles=2)
        self.assertEqual(int(pc.inferred_col.iloc[0]), int(col[14 + 3]))                 # the column of Tm1 #3
        self.assertEqual(pc.n_input_columns.iloc[0], 1); self.assertAlmostEqual(pc.centroid_to_col_deg.iloc[0], 0.0, places=3)
        s = summ[summ.side == "both"].iloc[0]
        self.assertEqual(s.n_distinct_columns, 1); self.assertEqual(s.frac_no_column, 0.0)
        part = P.strongest_partner(self.c, col, ("LC11", "Tm1"))
        self.assertEqual(part[part.type == "LC11"].partner_types_top5.iloc[0], {"Tm1": 1})
        self.assertEqual(part[part.type == "Tm1"].partner_types_top5.iloc[0], {"Mi1": 7})
        self.assertEqual(part[part.type == "Tm1"].frac_partner_annotated.iloc[0], 1.0)

    def test_hex_distance_matches_the_retina(self):
        # lattice neighbours are one column apart and 4.6 deg apart in the retina's embedding
        xy = P.hex_xy([h[0] for h in HEX], [h[1] for h in HEX])
        self.assertTrue(np.allclose(np.linalg.norm(xy[1:] - xy[0], axis=1), 1.0))
        cae = self.r.col_az_el; k = P.retina_key(self.r); c0 = k[("L", 0, 0)]
        for h in HEX[1:]:
            d = P.angular_distance_deg(cae[c0, 0], cae[c0, 1], cae[k[("L", *h)], 0], cae[k[("L", *h)], 1])
            self.assertAlmostEqual(float(d), 4.6, delta=0.05)


if __name__ == "__main__":
    unittest.main()
