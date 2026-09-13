"""Round-2 object assay tests (CPU, no connectome, no GPU). Each builder of the round appends a class here.

    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1 python -m unittest tests.test_object_round2 -v
"""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class MatchedSphereGeometryTests(unittest.TestCase):
    """scripts/probe_object_matched.py: the fly-relative arc geometry (constant elevation, angular diameter, distance
    and angular speed for three diameters), the substrate-clearance rule, the tracer's material switch, the footprint
    diagnostic, the summary statistics against probe_object_sweep's definitions, and the RF-map windowing."""

    @classmethod
    def setUpClass(cls):
        cls.m = _load_script("probe_object_matched")
        # a pinned pose facing -y on a horizontal face (probe_object_sweep's), eye raised
        cls.eye = np.array([-0.20, 0.10, 0.75 + 0.15]); cls.fwd = np.array([0.0, -1.0, 0.0]); cls.left = np.array([1.0, 0.0, 0.0]); cls.up = np.array([0.0, 0.0, 1.0])

    def test_arc_holds_elevation_diameter_distance_and_speed_for_three_diameters(self):
        m = self.m
        for diam, el in ((4.5, 0.0), (11.0, 0.0), (30.0, 12.5)):
            r = m.radius_for(diam, 0.05)
            self.assertAlmostEqual(m.angular_diameter_deg(r, 0.05), diam, places=9)
            tr = m.make_track(1000, 50.0, 40.0, el, 0.05, diam)                    # 10 s at 10 ms: 4 one-way legs of 2.5 s
            seen = [m.seen_from_eye(m.centre_world(self.eye, self.fwd, self.left, self.up, az, el, 0.05), self.eye, self.fwd, self.left, self.up, r)
                    for az in tr["az_deg"]]
            els = np.array([s["el_deg"] for s in seen]); dias = np.array([s["diam_deg"] for s in seen])
            dists = np.array([s["dist_m"] for s in seen]); azs = np.array([s["az_deg"] for s in seen])
            self.assertLess(np.abs(els - el).max(), 1e-9, f"elevation drifts at {diam} deg")
            self.assertLess(np.abs(dias - diam).max(), 1e-9, f"angular diameter drifts at {diam} deg")
            self.assertLess(np.abs(dists - 0.05).max(), 1e-12)
            self.assertLess(np.abs(azs - tr["az_deg"]).max(), 1e-9)                   # the azimuth seen is the azimuth commanded
            speed = np.abs(np.diff(azs)) / 0.01
            off_turn = ~tr["turn"][:-1]
            self.assertLess(np.abs(speed[off_turn] - 40.0).max(), 1e-6, "angular speed is not constant along the arc")
            self.assertEqual(int(tr["turn"].sum()), 4)                                # reversals at 2.5, 5, 7.5 s and at 10.0 s (the step out of the last frame)
            self.assertTrue(tr["turn"][249] and tr["turn"][499] and tr["turn"][749] and tr["turn"][999])
            self.assertAlmostEqual(tr["az_deg"].max(), 50.0); self.assertAlmostEqual(tr["az_deg"].min(), -50.0)
            self.assertEqual(tr["az_deg"][0], 50.0)                                   # starts left, moves right
            self.assertLess(tr["az_deg"][1], tr["az_deg"][0])
        # the radius is derived from the distance: the same diameter at another distance is another radius, same angle
        self.assertAlmostEqual(m.angular_diameter_deg(m.radius_for(20.0, 0.08), 0.08), 20.0, places=9)
        self.assertNotAlmostEqual(m.radius_for(20.0, 0.08), m.radius_for(20.0, 0.05))

    def test_substrate_clearance_names_the_table_clip_at_the_walking_eye_height(self):
        m = self.m
        for diam in (4.5, 11.0, 20.0, 30.0):
            walking = m.substrate_clearance(0.0, diam, 0.05, 0.0012)
            self.assertFalse(walking["clear"], f"{diam} deg at elevation 0 should be clipped by the table at 1.2 mm eye height")
            self.assertAlmostEqual(walking["substrate_horizon_deg"], -1.4, delta=0.1)
            raised = m.substrate_clearance(0.0, diam, 0.05, 0.15)
            self.assertTrue(raised["clear"]); self.assertEqual(raised["substrate_horizon_deg"], -90.0)
        # raising the elevation instead also clears it: a 30-deg ball at 20 deg has its lower limb at +5 deg
        self.assertTrue(m.substrate_clearance(20.0, 30.0, 0.05, 0.0012)["clear"])
        g = m.geometry_record(11.0, 0.05, 0.0, 50.0, 40.0, 0.15)
        self.assertAlmostEqual(g["one_way_s"], 2.5); self.assertAlmostEqual(g["angular_diameter_check_deg"], 11.0, places=9)

    def test_tracer_supports_dark_bright_and_emissive_ball_materials(self):
        import torch
        from flyverse import world
        w = world.World(device=torch.device("cpu"))
        w.planes.append(world.Plane((0, -2, 0), (0, 1, 0), "wall"))
        w.spheres.append(world.Sphere((0, -0.05, 0), (0.005, 0.005, 0.005), "black"))
        o = torch.zeros(2, 3); d = torch.tensor([[0.0, -1.0, 0.0], [0.3, -1.0, 0.0]]); d = d / d.norm(dim=1, keepdim=True)
        lum = {}
        for mat in self.m.MATERIALS:
            w.spheres[0].material = mat; w.invalidate()
            rad = w.trace(o, d)
            lum[mat] = float(rad[0].sum()); wall = float(rad[1].sum())
        self.assertLess(lum["black"], 0.05 * wall)             # a dark object: dimmed by > 95 %
        self.assertGreater(lum["lamp"], 1.5 * wall)             # an emissive object: brighter than the wall
        self.assertGreater(lum["plate"], lum["black"])
        # the headlamp: a point light at the eye lights the near face of the ball, hides its shadow
        w.light_pos = (0.0, 0.0, 0.0); w.spheres[0].material = "black"; w.invalidate()
        rad = w.trace(o, d)
        self.assertGreater(float(rad[0].sum()), lum["black"])

    def test_footprint_recovers_a_dimmed_disc_and_reads_zero_on_blank_vs_blank(self):
        m = self.m
        az, el = np.meshgrid(np.arange(-60, 61, 4.6), np.arange(-40, 41, 4.6)); col = np.c_[az.ravel(), el.ravel()]
        T = 5; blank = np.ones((T, len(col), 4)) * 0.25
        obj = blank.copy()
        for j in range(T):
            cen_az = 30 - 10 * j
            disc = np.hypot(col[:, 0] - cen_az, col[:, 1] - 5.0) < 6.0
            obj[j, disc] *= 0.02
        fp = m.footprint(obj, blank, col)
        s = fp["summary"]
        self.assertEqual(s["frames_with_dimmed"], T)
        self.assertAlmostEqual(s["centroid_el_mean_deg"], 5.0, delta=2.5)
        self.assertLess(s["centroid_el_sd_deg"], 2.5)
        self.assertGreater(s["centroid_az_max_deg"], s["centroid_az_min_deg"])
        self.assertAlmostEqual(s["blank_lum_under_object_mean"], 1.0)
        self.assertEqual(s["blank_static_max_abs_diff"], 0.0)
        self.assertGreater(s["n_dimmed_50pct_mean"], 0); self.assertGreaterEqual(s["n_changed_5pct_mean"], s["n_dimmed_50pct_mean"])
        z = m.footprint(blank, blank, col)["summary"]
        self.assertEqual(z["n_changed_5pct_mean"], 0.0); self.assertEqual(z["frames_with_dimmed"], 0)

    def test_summary_statistics_match_probe_object_sweep_definitions(self):
        m = self.m
        rng = np.random.default_rng(0)
        T, n = 40, 3
        A = {"drive": {"LC11": rng.normal(size=(T, n))}, "spk": {"LC11": (rng.random((T, n)) < 0.1).astype(np.float32)},
             "dr": {"T3": rng.normal(size=(T, 4)) * 0.1}, "cells": {"LC11": np.arange(n), "T3": np.arange(4) + 10},
             "bodies": {"LC11": np.array([101, 102, 103]), "T3": np.array([201, 202, 203, 204])}}
        B = {"drive": {"LC11": rng.normal(size=(T, n))}, "spk": {"LC11": (rng.random((T, n)) < 0.1).astype(np.float32)},
             "dr": {"T3": rng.normal(size=(T, 4)) * 0.1}, "cells": A["cells"], "bodies": A["bodies"]}
        track_az = m.azimuth_track(np.arange(T) * 0.01, 50.0, 40.0)
        per_type, per_body = m.summarize_arms(A, B, T * 0.01, track_az, 50.0)
        md = A["drive"]["LC11"].mean(0) - B["drive"]["LC11"].mean(0)
        self.assertAlmostEqual(per_type["LC11"]["diff_max_over_cells_mean_mv"], md.max())          # probe_object_sweep.py:277
        self.assertAlmostEqual(per_type["LC11"]["diff_mean_over_cells_mean_mv"], md.mean())
        hz = A["spk"]["LC11"].sum(0) / (T * 0.01) - B["spk"]["LC11"].sum(0) / (T * 0.01)
        self.assertAlmostEqual(per_type["LC11"]["diff_rate_hz_max_cell"], hz.max())
        ma, mb = np.abs(A["dr"]["T3"]).mean(0), np.abs(B["dr"]["T3"]).mean(0)
        self.assertAlmostEqual(per_type["T3"]["diff_abs_best_cell_mean"], (ma - mb).max())         # :291
        sd = A["dr"]["T3"].mean(0) - B["dr"]["T3"].mean(0)
        self.assertAlmostEqual(per_type["T3"]["diff_signed_best_cell"], np.abs(sd).max())           # :292 -- distinct from the |dev| statistic
        self.assertAlmostEqual(per_type["T3"]["diff_signed_mean"], sd.mean())
        self.assertNotAlmostEqual(per_type["T3"]["diff_signed_best_cell"], per_type["T3"]["diff_abs_best_cell_mean"])
        # per-body rows: every body, both quantities, diff = a - b
        self.assertEqual(len(per_body), 2 * n + 2 * 4)
        row = per_body[(per_body.bodyId == "102") & (per_body.quantity == "drive_mv")].iloc[0]
        self.assertAlmostEqual(row["diff"], md[1]); self.assertEqual(row["type"], "LC11"); self.assertEqual(row["n_frames"], T)

    def test_rf_map_loader_and_windowing(self):
        m = self.m
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rf.csv"
            pd.DataFrame({"bodyId": [101, 202], "az_deg": [20.0, -30.0], "el_deg": [0.0, 0.0], "width_deg": [10.0, 10.0]}).to_csv(p, index=False)
            rf = m.load_rf_map(p)
            self.assertEqual(list(rf.bodyId), ["101", "202"]); self.assertTrue((rf.height_deg == rf.width_deg).all())
            bad = Path(d) / "bad.csv"; pd.DataFrame({"bodyId": [1], "az_deg": [0.0]}).to_csv(bad, index=False)
            with self.assertRaises(ValueError):
                m.load_rf_map(bad)
        T = 500
        tr = m.make_track(T, 50.0, 40.0, 0.0, 0.05, 11.0)
        win = m.rf_frames(tr["az_deg"], tr["el_deg"], tr["diam_deg"], 20.0, 0.0, 10.0, 10.0)
        self.assertTrue(win.any()); self.assertLess(win.mean(), 0.5)
        self.assertTrue((np.abs(tr["az_deg"][win] - 20.0) <= 10.5 + 1e-9).all())      # (width + diam) / 2 = 10.5 deg
        fa = np.zeros((T, 1)); fa[win, 0] = 1.0; fb = np.zeros((T, 1))
        per_body = pd.DataFrame([{"bodyId": "101", "type": "LC11", "quantity": "drive_mv", "diff": float(fa.mean() - fb.mean())},
                                 {"bodyId": "999", "type": "LC11", "quantity": "drive_mv", "diff": 0.0}])
        out = m.rf_windowed(per_body, {("LC11", "drive_mv"): (fa, fb, np.array([101]))}, rf, tr)
        r = out[out.bodyId == "101"].iloc[0]
        self.assertEqual(int(r.rf_n_frames), int(win.sum())); self.assertAlmostEqual(r.rf_diff, 1.0)
        self.assertTrue(np.isnan(out[out.bodyId == "999"].iloc[0].rf_diff))         # no localizer row -> NaN, never 0
        stats = m.rf_population_stats(out)
        self.assertEqual(stats["LC11"]["drive_mv"]["n_bodies_windowed"], 1); self.assertAlmostEqual(stats["LC11"]["drive_mv"]["rf_diff_max_over_cells"], 1.0)

    def test_cli_parses_the_documented_flags_without_touching_a_gpu(self):
        ap = self.m.build_parser()
        a = ap.parse_args(["run", "--diam-deg", "30", "--elevation-deg", "0", "--deg-per-s", "40", "--az-max", "50", "--material", "lamp", "--light", "ceiling",
                           "--null", "--rf-map", "x.csv", "--out", "out/x", "--optic", "gain_fb=0"])
        self.assertEqual(a.diam_deg, 30.0); self.assertEqual(a.eye_height_m, 0.15); self.assertTrue(a.null); self.assertEqual(a.material, "lamp")
        self.assertEqual(a.elevation_deg, 0.0); self.assertEqual(a.light, "ceiling")
        g = ap.parse_args(["geometry"]); self.assertEqual(g.diam_deg, self.m.DIAMETERS_DEG)


if __name__ == "__main__":
    unittest.main()
