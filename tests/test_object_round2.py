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
        # a BRIGHT object (the lamp material) is a brightened set with the same centroid rule
        bright = blank.copy()
        for j in range(T):
            bright[j, np.hypot(col[:, 0] - (30 - 10 * j), col[:, 1] - 5.0) < 6.0] *= 4.0
        fb_ = m.footprint(bright, blank, col)["summary"]
        self.assertEqual(fb_["n_dimmed_50pct_mean"], 0.0); self.assertGreater(fb_["n_brightened_50pct_mean"], 0)
        self.assertAlmostEqual(fb_["centroid_el_mean_deg"], 5.0, delta=2.5); self.assertEqual(fb_["frames_with_dimmed"], T)

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
            # the synthetic task's rfmap CSV: unfitted rows (fitted False / NaN centre) carry no localizer
            synth = Path(d) / "rfmap.csv"
            pd.DataFrame({"bodyId": [11, 12, 13], "type": ["LC11"] * 3, "az_deg": [10.0, np.nan, 5.0], "el_deg": [0.0, np.nan, 1.0],
                          "width_deg": [8.0, np.nan, 9.0], "peak": [1.0, 0.0, 2.0], "fitted": [True, False, False]}).to_csv(synth, index=False)
            self.assertEqual(list(m.load_rf_map(synth).bodyId), ["11"])
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

    def test_batch_generator_writes_the_cluster_rule_into_every_job_line(self):
        m = self.m
        jobs = m.batch_jobs("smoke", "out/objm/x", 3.0, 3.0, [0], [4.5, 11.0, 20.0, 30.0])
        self.assertEqual(len(jobs), 7)                                            # 4 object + null + lamp + ceiling
        for j in jobs:
            self.assertTrue(j.startswith("mkdir -p out/objm/x && source .venv/bin/activate && "))
            self.assertIn("assert torch.cuda.is_available()", j); self.assertIn("> out/objm/x/", j)
        self.assertIn("--null", jobs[4]); self.assertIn("--material lamp", jobs[5]); self.assertIn("--light ceiling", jobs[6])
        ladder = m.batch_jobs("ladder", "out/objm/y", 12.0, 3.0, [0, 1, 2, 3, 4], [4.5, 11.0, 20.0])
        self.assertEqual(len(ladder), 3 * 5 + 5)
        self.assertEqual(sum("--null" in j for j in ladder), 5)


def _load_module(name: str):
    """Like _load_script, but registered in sys.modules first (a @dataclass under `from __future__ import annotations`
    looks its module up there)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class SyntheticStimuliTests(unittest.TestCase):
    """scripts/probe_synthetic_stimuli.py: every generator's shape / contrast / speed / position on a synthetic column
    grid (no connectome), the sampling kernel against Retina.ray_directions, the localizer sequence and its per-node
    reductions, the RF fit, the radiance-path player (no aliasing of the stored stimulus), the batch verifier, the
    statistics definitions, the RF-map CSV columns and the batch generators."""

    @classmethod
    def setUpClass(cls):
        cls.m = _load_module("probe_synthetic_stimuli")
        az, el = np.meshgrid(np.arange(-60, 60.1, 4.6), np.arange(-40, 40.1, 4.6))
        cls.col = np.c_[az.ravel(), el.ravel()]                       # 27 x 18 = 486 columns, 4.6-deg pitch (the retina's)
        cls.bg = np.asarray(cls.m.BACKGROUND, np.float32)

    def _dimmed(self, frame, thr=0.5):
        rel = frame.sum(1) / self.bg.sum()
        return np.flatnonzero(rel < thr) if rel.min() < 1 else np.flatnonzero(rel > 1 + (1 - thr))

    def test_sampling_kernel_equals_retina_ray_directions(self):
        from flyverse import retina as rmod
        m = self.m
        n = len(self.col)
        az, el = np.deg2rad(self.col[:, 0]), np.deg2rad(self.col[:, 1])
        col_dir = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)
        r = rmod.Retina(pr_index=np.arange(n), pr_column=np.arange(n), pr_sens=np.ones((n, 4), np.float32), col_side=np.array(["L"] * n),
                        col_hex=np.zeros((n, 2)), col_dir=col_dir, col_az_el=self.col.astype(float))
        dirs, w = r.ray_directions()
        s_az, s_el, w2 = m.sample_points(self.col, m.ACCEPTANCE_DEG, m.RAYS)
        self.assertTrue(np.allclose(w, w2))
        a, e = np.deg2rad(s_az), np.deg2rad(s_el)
        mine = np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], -1)
        self.assertLess(np.abs(mine - dirs).max(), 1e-9)                     # the same 7 rays per column, in degrees
        self.assertEqual(w.shape, (7,)); self.assertAlmostEqual(float(w.sum()), 1.0)

    def test_rectangle_shape_contrast_speed_and_position(self):
        m = self.m
        s = m.rectangle(self.col, seconds=1.0, width_deg=4.4, height_deg=8.8, contrast=-0.995, speed_deg_s=40.0, elevation_deg=0.0, half_span_deg=30.0)
        self.assertEqual(s.presented().shape, (100, len(self.col), 4)); self.assertEqual(s.blank.shape, (len(self.col), 4))
        self.assertTrue(np.array_equal(s.blank, np.tile(self.bg, (len(self.col), 1))))
        P = s.presented(); rel = P / self.bg[None, None, :]
        self.assertTrue(0.005 < float(rel.min()) < 0.5)                      # a 4.4-deg rectangle never covers a whole 4.5-deg kernel (max coverage ~0.77)
        self.assertAlmostEqual(float(rel.max()), 1.0, places=6)               # nothing brightens
        wide = m.rectangle(self.col, seconds=0.02, width_deg=20.0, height_deg=20.0, contrast=-0.995, half_span_deg=0.0, speed_deg_s=0.0)
        self.assertAlmostEqual(float((wide.presented() / self.bg).min()), 1 - 0.995, places=6)   # a fully covered column: background x (1 + contrast)
        far = np.hypot(self.col[:, 0] - s.track["az_deg"][0], self.col[:, 1]) > 20
        self.assertTrue(np.array_equal(P[0][far], s.blank[far]))              # far columns: the background exactly
        v = np.abs(np.diff(s.track["az_deg"])) / s.dt_s
        self.assertLess(np.abs(v - 40.0).max(), 1e-9)                         # constant angular speed, every frame
        self.assertEqual(s.track["az_deg"][0], 30.0); self.assertLess(s.track["az_deg"][1], 30.0)
        self.assertLessEqual(s.track["az_deg"].max(), 30.0); self.assertGreaterEqual(s.track["az_deg"].min(), -30.0)
        self.assertTrue((s.track["el_deg"] == 0.0).all())
        for k in (0, 37, 99):                                                 # the dimmed columns sit where the track says
            cov = 1 - P[k].sum(1) / self.bg.sum()
            self.assertAlmostEqual(float((cov * self.col[:, 0]).sum() / cov.sum()), s.track["az_deg"][k], delta=2.5)
            self.assertAlmostEqual(float((cov * self.col[:, 1]).sum() / cov.sum()), 0.0, delta=2.5)
        b = m.rectangle(self.col, seconds=0.02, width_deg=20.0, height_deg=20.0, contrast=+0.995, half_span_deg=0.0, speed_deg_s=0.0)
        self.assertAlmostEqual(float((b.presented() / self.bg).max()), 1.995, places=5)    # the bright mirror image
        self.assertAlmostEqual(float((b.presented() / self.bg).min()), 1.0, places=6)

    def test_height_and_width_ladders_change_one_dimension(self):
        m = self.m
        ext = {}
        for h in m.HEIGHT_LADDER["heights"]:
            s = m.rectangle(self.col, seconds=0.02, width_deg=m.HEIGHT_LADDER["width"], height_deg=h, half_span_deg=0.0, speed_deg_s=0.0)
            cov = 1 - s.presented()[0].sum(1) / self.bg.sum(); on = cov > 0.5
            ext[("h", h)] = (np.ptp(self.col[on, 0]) if on.any() else 0.0, np.ptp(self.col[on, 1]) if on.any() else 0.0, float(cov.sum()))
        for w in m.WIDTH_LADDER["widths"]:
            s = m.rectangle(self.col, seconds=0.02, width_deg=w, height_deg=m.WIDTH_LADDER["height"], half_span_deg=0.0, speed_deg_s=0.0)
            cov = 1 - s.presented()[0].sum(1) / self.bg.sum(); on = cov > 0.5
            ext[("w", w)] = (np.ptp(self.col[on, 0]) if on.any() else 0.0, np.ptp(self.col[on, 1]) if on.any() else 0.0, float(cov.sum()))
        hs = m.HEIGHT_LADDER["heights"]; ws = m.WIDTH_LADDER["widths"]
        self.assertTrue(all(ext[("h", hs[i + 1])][2] > ext[("h", hs[i])][2] for i in range(len(hs) - 1)))   # coverage grows with height
        self.assertTrue(all(ext[("w", ws[i + 1])][2] > ext[("w", ws[i])][2] for i in range(len(ws) - 1)))   # ... and with width
        self.assertLessEqual(ext[("h", 30.0)][0], 4.6 + 1e-9)               # the height ladder keeps the azimuth extent at one column
        self.assertGreaterEqual(ext[("h", 30.0)][1], 20.0)
        self.assertLessEqual(ext[("w", 30.0)][1], 9.2 + 1e-9)               # the width ladder keeps the elevation extent at ~8.8 deg
        self.assertGreaterEqual(ext[("w", 30.0)][0], 20.0)
        sq = m.rectangle(self.col, seconds=0.02, width_deg=30.0, height_deg=30.0, half_span_deg=0.0, speed_deg_s=0.0)
        cov = 1 - sq.presented()[0].sum(1) / self.bg.sum()
        self.assertAlmostEqual(float(cov.sum()), (30.0 / 4.6) ** 2, delta=8)  # a 30-deg square covers ~ (30 / 4.6)^2 columns of the grid

    def test_bar_covers_every_elevation_at_its_azimuth(self):
        m = self.m
        s = m.bar(self.col, seconds=0.6, width_deg=7.0, speed_deg_s=40.0, half_span_deg=30.0)
        self.assertEqual(s.name, "bar"); self.assertEqual(s.params["height_deg"], "full")
        for k in (6, 29, 52):                                                  # frames whose bar centre is within 0.2 deg of a grid azimuth
            cov = 1 - s.presented()[k].sum(1) / self.bg.sum()
            near = np.abs(self.col[:, 0] - s.track["az_deg"][k]) <= 3.5 - 2.25    # columns whose whole kernel is inside the bar
            self.assertTrue(near.any() and (cov[near] > 0.99).all()); self.assertEqual(len(set(np.round(self.col[near, 1], 3))), 18)   # every elevation row
            self.assertTrue((cov[np.abs(self.col[:, 0] - s.track["az_deg"][k]) > 3.5 + 2.25] == 0).all())

    def test_grating_matches_probe_motion_formula_and_speed(self):
        m = self.m
        s = m.grating(self.col, seconds=0.75, period_deg=30.0, contrast=0.5, speed_deg_s=40.0, direction="front->back")
        P = s.presented(); self.assertEqual(P.shape, (75, len(self.col), 4))
        x = np.abs(self.col[:, 0])
        for k in (0, 10, 74):
            expect = 1 + 0.5 * np.sin(2 * np.pi * (x - 40.0 * k * 0.01) / 30.0)
            self.assertLess(np.abs(P[k] / self.bg - expect[:, None]).max(), 1e-5)
        self.assertLess(np.abs(P.mean(0) / self.bg - 1.0).max(), 1e-5)         # one full period (0.75 s x 40 deg/s = 30 deg): mean = background
        self.assertAlmostEqual(float(s.track["phase_shift_deg"][-1]), 40.0 * 0.74)
        with self.assertRaises(ValueError):
            m.grating(self.col, direction="sideways")

    def test_flicker_square_wave_full_field_and_mean(self):
        m = self.m
        s = m.flicker(self.col, seconds=1.0, hz=2.0, contrast=0.5)
        P = s.presented(); self.assertEqual(P.shape, (100, len(self.col), 4))
        lev = P[:, 0, :] / self.bg
        self.assertTrue(np.allclose(lev[:25], 1.5) and np.allclose(lev[25:50], 0.5) and np.allclose(lev[50:75], 1.5))
        self.assertTrue(np.allclose(P, P[:, :1, :]))                            # full field: every column the same
        self.assertTrue(np.allclose(P.mean(0), self.bg, atol=1e-6))

    def test_flash_isolated_on_off_transitions_at_a_fixed_position(self):
        m = self.m
        for contrast, name in ((-0.995, "flash_off"), (+0.995, "flash_on")):
            s = m.flash(self.col, seconds=3.0, size_deg=8.8, contrast=contrast, az_deg=20.0, el_deg=-10.0, on_s=0.5, period_s=1.5, first_s=0.5)
            self.assertEqual(s.name, name); self.assertEqual(s.presented().shape, (300, len(self.col), 4))
            idx = s.index
            self.assertEqual(int((np.diff(idx) != 0).sum()), 4)               # 4 isolated transitions in 3 s
            self.assertTrue((idx[50:100] == 1).all() and (idx[100:200] == 0).all() and (idx[200:250] == 1).all() and (idx[:50] == 0).all())
            obj = s.pattern[1]; cov = np.abs(1 - obj.sum(1) / self.bg.sum())
            self.assertAlmostEqual(float((cov * self.col[:, 0]).sum() / cov.sum()), 20.0, delta=1.5)
            self.assertAlmostEqual(float((cov * self.col[:, 1]).sum() / cov.sum()), -10.0, delta=1.5)
            self.assertTrue(np.array_equal(s.pattern[0], s.blank))

    def test_transition_frames_split_the_run_at_the_right_edges(self):
        """The flash is a PERIODIC square, so every run holds both polarities: the ON / OFF measurement is the
        analysis-side split (`luminance_state` / `transition_frames`), and a bright run vs a dark run is bright-vs-dark."""
        m = self.m
        dark = m.flash(self.col, seconds=3.0, size_deg=8.8, contrast=-0.995, on_s=0.5, period_s=1.5, first_s=0.5)
        bright = m.flash(self.col, seconds=3.0, size_deg=8.8, contrast=+0.995, on_s=0.5, period_s=1.5, first_s=0.5)
        vis = dark.track["visible"]
        self.assertEqual(dark.params["transition"], "OFF at onset / ON at offset")       # the stimulus' own record
        self.assertEqual(bright.params["transition"], "ON at onset / OFF at offset")
        self.assertTrue(np.array_equal(m.luminance_state(dark.name, dark.params, dark.track), ~vis))     # a dark square IS the darker level
        self.assertTrue(np.array_equal(m.luminance_state(bright.name, bright.params, bright.track), vis))
        for s, on_edges, off_edges in ((dark, [100, 250], [50, 200]), (bright, [50, 200], [100, 250])):
            tf = m.transition_frames(m.luminance_state(s.name, s.params, s.track), s.dt_s, window_s=0.3)
            self.assertTrue(np.array_equal(tf["edges_on"], on_edges))      # the square appears at 0.5 / 2.0 s and vanishes at 1.0 / 2.5 s
            self.assertTrue(np.array_equal(tf["edges_off"], off_edges))
            self.assertEqual(tf["window_frames"], 30); self.assertAlmostEqual(tf["window_s"], 0.3)
            self.assertEqual(int(tf["on"].sum()), 60); self.assertEqual(int(tf["off"].sum()), 60)
            self.assertFalse((tf["on"] & tf["off"]).any())                 # a frame belongs to exactly one transition
            self.assertFalse(tf["on"][:50].any() or tf["off"][:50].any())  # the frames before the first edge belong to neither
            for e in on_edges:
                self.assertTrue(tf["on"][e:e + 30].all() and not tf["off"][e:e + 30].any())
            for e in off_edges:
                self.assertTrue(tf["off"][e:e + 30].all() and not tf["on"][e:e + 30].any())
        short = m.flash(self.col, seconds=2.0, contrast=+0.995, on_s=0.2, period_s=1.0, first_s=0.5)
        tf = m.transition_frames(m.luminance_state(short.name, short.params, short.track), short.dt_s, window_s=0.3)
        self.assertTrue(np.array_equal(tf["edges_on"], [50, 150]) and np.array_equal(tf["edges_off"], [70, 170]))
        self.assertEqual(int(tf["on"].sum()), 40)                          # 20 frames per ON edge: truncated at the next edge
        self.assertEqual(int(tf["off"].sum()), 60)
        fl = m.flicker(self.col, seconds=1.0, hz=2.0, contrast=0.5)        # the same split on the flicker's level
        tfl = m.transition_frames(m.luminance_state(fl.name, fl.params, fl.track), fl.dt_s, window_s=0.1)
        self.assertTrue(np.array_equal(tfl["edges_off"], [25, 75]) and np.array_equal(tfl["edges_on"], [50]))
        self.assertIsNone(m.luminance_state("rect", {"contrast": -0.995}, m.rectangle(self.col, seconds=0.1).track))

    def test_transition_stats_read_only_their_own_transition_frames(self):
        """On a synthetic recording whose drive IS the frame index, the split statistic equals the mean frame index of
        its own windows -- ON, OFF and pooled are three different numbers, and the pooled one is the whole window."""
        m = self.m
        from flyverse.interp import common
        s = m.flash(self.col, seconds=3.0, size_deg=8.8, contrast=-0.995, on_s=0.5, period_s=1.5, first_s=0.5)
        T = s.n_frames; ramp = np.arange(T, dtype=np.float32); zero = np.zeros(T, np.float32)
        types = np.array(["LC11", "T3"])

        def rec(active):
            return common.Recording(np.arange(T) * 10.0, np.arange(2), np.array([11, 33]), types,
                                    {"drive_mv": np.c_[ramp if active else zero, zero].astype(np.float32),
                                     "optic_dr": np.c_[np.full(T, np.nan, np.float32), (ramp if active else zero) * 0.001].astype(np.float32),
                                     "spike_count": np.c_[np.cumsum(np.ones(T) if active else zero), zero].astype(np.float32)})
        stim, blank = rec(True), rec(False)
        tf = m.transition_frames(m.luminance_state(s.name, s.params, s.track), s.dt_s, window_s=0.3)
        f_on = m.family_stats(stim, blank, frames=tf["on"])
        f_off = m.family_stats(stim, blank, frames=tf["off"])
        pooled = m.family_stats(stim, blank)
        self.assertAlmostEqual(f_on["LC11"]["diff_max_over_cells_mean_mv"], float(ramp[tf["on"]].mean()), places=3)
        self.assertAlmostEqual(f_off["LC11"]["diff_max_over_cells_mean_mv"], float(ramp[tf["off"]].mean()), places=3)
        self.assertAlmostEqual(pooled["LC11"]["diff_max_over_cells_mean_mv"], float(ramp.mean()), places=3)
        self.assertAlmostEqual(f_on["T3"]["diff_signed_best_cell"], float(ramp[tf["on"]].mean()) * 0.001, places=5)
        self.assertGreater(f_on["LC11"]["diff_max_over_cells_mean_mv"], f_off["LC11"]["diff_max_over_cells_mean_mv"])  # ON windows are later here
        n = int(tf["on"].sum())
        self.assertAlmostEqual(f_on["LC11"]["rate_hz_mean_stim"], (n - 1) / (n * 0.01), places=3)   # spikes re-accumulated over the kept frames only
        split = m.transition_stats(stim, blank, s.name, s.params, s.track, 0.3, s.dt_s)
        self.assertEqual(split["n_frames"], {"on": 60, "off": 60}); self.assertEqual(split["n_edges"], {"on": 2, "off": 2})
        self.assertEqual(sorted(split["per_type"]), ["off", "on"])
        self.assertAlmostEqual(split["per_type"]["on"]["LC11"]["diff_max_over_cells_mean_mv"],
                               f_on["LC11"]["diff_max_over_cells_mean_mv"], places=6)
        self.assertIsNone(m.transition_stats(stim, blank, "rect", {"contrast": -0.995}, {}, 0.3, s.dt_s))

    def test_localizer_grid_sequence_and_node_frames(self):
        m = self.m
        s = m.localizer(self.col, spacing_deg=20.0, size_deg=4.5, flash_s=0.2, blank_s=0.3, order_seed=0, az_range=(-80, 80), el_range=(-60, 60))
        nodes = np.c_[s.track["node_az_deg"], s.track["node_el_deg"]]
        n = len(nodes); self.assertEqual(s.params["n_nodes"], n); self.assertGreater(n, 10)
        for a, e in nodes:                                                     # every node reaches a column
            self.assertLessEqual(np.hypot(self.col[:, 0] - a, self.col[:, 1] - e).min(), 4.5 / 2 + 4.5 / 2)
        self.assertEqual(s.n_frames, n * 50 + 30); self.assertAlmostEqual(s.params["seconds"], n * 0.5 + 0.3)
        idx, node, on = s.index, s.track["node"], s.track["on"]
        self.assertEqual(int(on.sum()), 20 * n); self.assertTrue((idx[on] == node[on] + 1).all()); self.assertTrue((idx[~on] == 0).all())
        for i in range(n):
            self.assertEqual(int((node == i).sum()), 20)                       # each node once, 200 ms
        runs = np.flatnonzero(np.diff(on.astype(int)) == 1)
        runs = runs + 1
        self.assertTrue((np.diff(runs) == 50).all()); self.assertEqual(runs[0], 30)   # 300 ms blank before every flash
        for i in (0, n // 2, n - 1):
            cov = 1 - s.pattern[i + 1].sum(1) / self.bg.sum(); self.assertGreater(cov.sum(), 0)
            self.assertLess(np.hypot((cov * self.col[:, 0]).sum() / cov.sum() - nodes[i, 0], (cov * self.col[:, 1]).sum() / cov.sum() - nodes[i, 1]), 3.0)
        s2 = m.localizer(self.col, spacing_deg=20.0, order_seed=0, az_range=(-80, 80), el_range=(-60, 60))
        self.assertTrue(np.array_equal(s.track["order"], s2.track["order"]))   # a fixed order across runs
        self.assertGreaterEqual(len(m.localizer_grid(self.col, 10.0)), 4 * len(m.localizer_grid(self.col, 20.0)) * 0.8)

    def test_node_accumulator_windows_and_spikes_per_frame(self):
        m = self.m
        s = m.localizer(self.col, spacing_deg=40.0, size_deg=4.5, flash_s=0.2, blank_s=0.3, az_range=(-40, 40), el_range=(-40, 40))
        n = s.params["n_nodes"]; acc = m.NodeAccumulator(s, 2, ("drive_mv", "spike_count"))
        node = s.track["node"]; sc = 0.0
        for t in range(s.n_frames):
            sc += 1.0 if node[t] >= 0 else 0.0                                   # one spike per flash frame, none on blanks
            acc.add(t, {"drive_mv": np.array([float(node[t]), 100.0 + t]), "spike_count": np.array([sc, 0.0])})
        out = acc.finish()
        self.assertTrue((out["n_on"] == 20).all() and (out["n_base"] == 10).all() and (out["n_off"] == 10).all())
        self.assertTrue(np.allclose(out["on__drive_mv"][:, 0], np.arange(n)))   # the on window carries its node's value
        self.assertTrue(np.allclose(out["base__drive_mv"][:, 0], -1.0) and np.allclose(out["off__drive_mv"][:, 0], -1.0))
        self.assertTrue(np.allclose(out["on__spikes_per_frame"][:, 0], 1.0) and np.allclose(out["base__spikes_per_frame"][:, 0], 0.0))
        self.assertTrue((np.diff(out["on__drive_mv"][:, 1]) != 0).any())        # a monotone cell's window means follow the presentation order

    def test_fit_rf_recovers_centre_and_width_and_rejects_noise(self):
        m = self.m
        az, el = np.meshgrid(np.arange(-50, 51, 10.0), np.arange(-30, 31, 10.0)); az, el = az.ravel(), el.ravel()
        rng = np.random.default_rng(1)
        A = np.exp(-0.5 * ((az - 20) ** 2 + (el + 10) ** 2) / 8.0 ** 2) + 0.01 * rng.normal(size=len(az))
        B = 0.01 * rng.normal(size=len(az))
        C = -0.5 * np.exp(-0.5 * ((az + 30) ** 2 + el ** 2) / 6.0 ** 2) + 0.005 * rng.normal(size=len(az))
        df = m.fit_rf(np.c_[A, B, C], az, el, z_min=5.0, spacing_deg=10.0)
        self.assertTrue(df.fitted[0] and df.fitted[2] and not df.fitted[1])
        self.assertLess(np.hypot(df.az_deg[0] - 20, df.el_deg[0] + 10), 3.0); self.assertEqual(df.sign[0], 1)
        self.assertLess(np.hypot(df.az_deg[2] + 30, df.el_deg[2]), 3.0); self.assertEqual(df.sign[2], -1)
        self.assertTrue(10.0 <= df.width_deg[0] <= 30.0); self.assertGreater(df.n_nodes_above_threshold[0], 0)
        self.assertTrue(np.isnan(df.az_deg[1]) and df.n_nodes_above_threshold[1] == 0)
        m2 = m.merge_maps([df, df]); self.assertEqual(int(m2.n_runs_fitted[0]), 2); self.assertAlmostEqual(float(m2.centre_spread_deg[0]), 0.0, places=3)
        self.assertTrue(np.isnan(m2.az_deg[1]))
        d = m.angular_distance_deg(np.array([0.0, 90.0]), np.array([0.0, 0.0]), np.array([10.0, 0.0]), np.array([0.0, 0.0]))
        self.assertTrue(np.allclose(d, [10.0, 90.0]))

    def test_play_hands_the_stored_radiance_to_vision_without_aliasing(self):
        import torch
        m = self.m

        class StubFB:
            device = torch.device("cpu")

            def __init__(self):
                self.seen = []

            def vision(self, x):
                self.seen.append(x.clone().numpy())

            def step(self, ms):
                pass
        for s in (m.rectangle(self.col, seconds=0.1), m.localizer(self.col, spacing_deg=40.0, flash_s=0.05, blank_s=0.05, az_range=(-40, 40), el_range=(-40, 40))):
            before = s.presented().copy()
            fb = StubFB(); cs = m.play(fb, s, True)
            self.assertTrue(np.array_equal(np.stack(fb.seen), before))          # what vision received IS the stored stimulus
            self.assertTrue(np.array_equal(s.presented(), before))              # ... and the stored stimulus is untouched afterwards
            self.assertTrue(np.allclose(cs, before.astype(np.float64).sum((1, 2))))
            fb = StubFB(); cs = m.play(fb, s, False)
            self.assertTrue(all(np.array_equal(f, s.blank) for f in fb.seen)); self.assertTrue(np.allclose(cs, s.blank.astype(np.float64).sum()))

    def test_statistics_definitions_and_time_course(self):
        m = self.m
        from flyverse.interp import common
        rng = np.random.default_rng(0); T = 40
        types = np.array(["LC11"] * 3 + ["T3"] * 4); body = np.arange(7) + 100
        nan3 = np.full((T, 3), np.nan, np.float32)

        def rec(seed):
            r = np.random.default_rng(seed)
            return common.Recording(np.arange(T) * 10.0, np.arange(7), body, types,
                                    {"drive_mv": np.c_[r.normal(size=(T, 3)), np.zeros((T, 4))].astype(np.float32),
                                     "optic_dr": np.c_[nan3, r.normal(size=(T, 4)) * 0.1].astype(np.float32),
                                     "spike_count": np.cumsum(r.random((T, 7)) < 0.1, axis=0).astype(np.float32)})
        A, B = rec(1), rec(2)
        f = m.family_stats(A, B)
        md = A.quantities["drive_mv"][:, :3].mean(0) - B.quantities["drive_mv"][:, :3].mean(0)
        self.assertAlmostEqual(f["LC11"]["diff_max_over_cells_mean_mv"], md.max(), places=5)     # probe_object_sweep.py:277
        self.assertAlmostEqual(f["LC11"]["diff_mean_over_cells_mean_mv"], md.mean(), places=5)
        sd = A.quantities["optic_dr"][:, 3:].mean(0) - B.quantities["optic_dr"][:, 3:].mean(0)
        ma = np.abs(A.quantities["optic_dr"][:, 3:]).mean(0) - np.abs(B.quantities["optic_dr"][:, 3:]).mean(0)
        self.assertAlmostEqual(f["T3"]["diff_signed_best_cell"], np.abs(sd).max(), places=5)     # :292
        self.assertAlmostEqual(f["T3"]["diff_abs_best_cell_mean"], ma.max(), places=5)           # :291 -- a distinct statistic
        self.assertAlmostEqual(f["T3"]["diff_signed_mean"], sd.mean(), places=5)
        self.assertNotAlmostEqual(f["T3"]["diff_signed_best_cell"], f["T3"]["diff_abs_best_cell_mean"])
        tc = A.quantities["drive_mv"][:, :3].mean(1) - B.quantities["drive_mv"][:, :3].mean(1)
        self.assertAlmostEqual(f["LC11"]["tc_peak_pop_mean"], tc[np.argmax(np.abs(tc))], places=5)
        self.assertAlmostEqual(f["LC11"]["tc_peak_t_s"], 0.01 * np.argmax(np.abs(tc)))
        t = m.time_course(A, B, every=5)
        self.assertEqual(len(t), 2 * 8); self.assertAlmostEqual(float(t[(t.type == "LC11") & (t.t_s == 0.0)].diff_pop_mean.iloc[0]), tc[0], places=5)
        pb = m.per_body_rows(A, B, types=("LC11",))
        self.assertEqual(len(pb), 3); self.assertAlmostEqual(float(pb.diff_time_mean[1]), md[1], places=5); self.assertEqual(pb.bodyId[0], "100")
        hz = (A.quantities["spike_count"][-1, 0] - A.quantities["spike_count"][0, 0]) / (T * 0.01)
        self.assertAlmostEqual(float(pb.rate_hz_stim[0]), hz, places=5)

    def test_rf_map_csv_columns_and_the_sphere_probe_reads_it(self):
        m = self.m
        az, el = np.meshgrid(np.arange(-50, 51, 10.0), np.arange(-30, 31, 10.0)); az, el = az.ravel(), el.ravel()
        R = np.c_[np.exp(-0.5 * ((az - 20) ** 2 + (el + 10) ** 2) / 8.0 ** 2), np.zeros(len(az))]
        df = m.fit_rf(R, az, el); df.insert(0, "type", ["LC11", "LC10a"]); df.insert(0, "bodyId", ["101", "202"])
        df = df[m.RF_CSV_COLUMNS + [c for c in df.columns if c not in m.RF_CSV_COLUMNS]]
        self.assertEqual(list(df.columns[:7]), ["bodyId", "type", "az_deg", "el_deg", "width_deg", "peak", "n_nodes_above_threshold"])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rf.csv"; df.to_csv(p, index=False)
            back = pd.read_csv(p, dtype={"bodyId": str})
            self.assertEqual(list(back.bodyId), ["101", "202"]); self.assertTrue(np.isnan(back.az_deg[1]) and back.fitted[0] and not back.fitted[1])
            try:
                sphere = _load_script("probe_object_matched")
            except Exception:                                                   # the other builder's file: read against the documented interface only
                self.skipTest("probe_object_matched not importable")
            rf = sphere.load_rf_map(p)                                          # its loader drops rows without a fitted centre
            self.assertIn("101", list(rf.bodyId)); self.assertAlmostEqual(float(rf.set_index("bodyId").az_deg["101"]), float(df.az_deg[0]))
            self.assertTrue((rf.height_deg == rf.width_deg).all())

    def test_verify_dir_reads_summary_provenance_and_console(self):
        import json
        m = self.m
        with tempfile.TemporaryDirectory() as d:
            for name, dev, chk, console in (("a", "cuda:0", True, "FlyBrain ready: device cuda:0\nwritten x; device cuda:0\n"),
                                            ("b", "cpu", True, "FlyBrain ready: device cpu\nwritten x; device cpu\n"),
                                            ("c", "cuda:0", False, "FlyBrain ready: device cuda:0\nwritten x; device cuda:0\n")):
                stem = Path(d) / name
                (Path(d) / f"{name}_summary.json").write_text(json.dumps({"stimulus": "rect", "seed": 0, "arms": ["stim", "blank"], "device": dev, "optic_overrides": {},
                                                                          "checksum_per_arm": {"stim": chk, "blank": True}, "n_frames": 300}), encoding="utf-8")
                (Path(d) / f"{name}_prov.json").write_text(json.dumps({"execution": {"device": dev}}), encoding="utf-8")
                (Path(d) / f"{name}.txt").write_text(console, encoding="utf-8")
                for arm in ("stim", "blank"):
                    np.savez(str(stem) + f"_{arm}.npz", t_ms=np.zeros(1))
            df = m.verify_dir(d).set_index("run")
            self.assertTrue(df.ok["a"]); self.assertFalse(df.ok["b"]); self.assertFalse(df.ok["c"])
            self.assertEqual(len(m.verify_runs([d], require_cuda=True)), 2); self.assertEqual(len(m.verify_runs([d], require_cuda=False)), 1)

    def test_cli_and_batch_generators(self):
        m = self.m
        ap = m.build_parser()
        a = ap.parse_args(["record", "--stimulus", "rect", "--width", "4.4", "--height", "8.8", "--contrast", "-0.995", "--seed", "3", "--out", "out/x", "--optic", "gain_fb=0", "--null"])
        self.assertEqual(a.stimulus, "rect"); self.assertEqual(a.seed, 3); self.assertTrue(a.null); self.assertEqual(a.settle, 2.0); self.assertEqual(a.optic, ["gain_fb=0"])
        self.assertTrue(m._flag_set(["--contrast", "0.5"], "--contrast") and not m._flag_set(["--width", "1"], "--contrast"))
        jobs = m.batch_jobs(3.0, 2.0, 0, 3)
        self.assertEqual(len(jobs), 14); self.assertEqual(sum(j[0].startswith("loc_") for j in jobs), 4)
        self.assertEqual(sum("gain_fb=0" in j[1] for j in jobs), 1); self.assertEqual(sum("--null" in j[1] for j in jobs), 1)
        self.assertEqual(len({j[0] for j in jobs}), 14)
        fam = {j[1].split()[1] for j in jobs}; self.assertEqual(fam, {"localizer", "rect", "bar", "grating", "flicker", "flash"})
        bl = m.blank(self.col, seconds=0.5)
        self.assertEqual(bl.presented().shape, (50, len(self.col), 4)); self.assertTrue((bl.presented() == self.bg).all()); self.assertEqual(bl.name, "blank")
        pb = ap.parse_args(["preview", "--stimulus", "blank", "--seconds", "0.2"]); pb.contrast_set = pb.width_set = False
        self.assertEqual(m.make_stimulus("blank", self.col, pb).n_frames, 20)
        lad = m.ladder_jobs(3.0, 2.0, 0, 5)
        self.assertEqual(len(lad), (6 + 6 + 6) * 2 * 5 + 5)
        self.assertEqual(sum(j[0].startswith("hlad_") for j in lad), 60); self.assertTrue(all("--width 4.4" in j[1] for j in lad if j[0].startswith("hlad_")))
        self.assertTrue(all("--height 8.8" in j[1] for j in lad if j[0].startswith("wlad_")))
        with tempfile.TemporaryDirectory() as d:
            p = m.write_batch(d.replace("\\", "/"), "t", 10, jobs[:2], "test")
            txt = p.read_text(encoding="utf-8")
            self.assertIn("source .venv/bin/activate", txt); self.assertIn("--fetch", txt); self.assertIn("cluster_run.py --name t --minutes 10", txt)
            self.assertTrue((Path(d) / "batch_jobs.json").exists())
        pa = ap.parse_args(["preview", "--stimulus", "flicker", "--hz", "3"]); pa.contrast_set = False; pa.width_set = False
        st = m.make_stimulus("flicker", self.col, pa)
        self.assertEqual(st.params["hz"], 3.0); self.assertEqual(st.params["contrast"], 0.5)

    def test_make_stimulus_maps_the_command_line_of_every_family(self):
        """The CLI -> generator mapping for every family, with the batch's own argument strings (the first batch's flash
        jobs died on a positional `background` landing in `first_s`; this is the test that would have caught it)."""
        import json
        m = self.m
        ap = m.build_parser()
        for name, a in m.batch_jobs(0.5, 0.1, 0, 1) + m.ladder_jobs(0.5, 0.1, 0, 1)[:3]:
            argv = ["record", "--out", "x"] + a.split()
            if "--stimulus localizer" in a:
                argv += ["--grid-spacing", "40"]
            pa = ap.parse_args(argv); pa.contrast_set = m._flag_set(argv, "--contrast"); pa.width_set = m._flag_set(argv, "--width")
            st = m.make_stimulus(pa.stimulus, self.col, pa)
            self.assertEqual(st.presented().shape[1:], (len(self.col), 4), name)
            self.assertTrue(np.array_equal(st.blank, np.tile(self.bg, (len(self.col), 1))), name)
            self.assertTrue(np.allclose(json.loads(json.dumps(m.to_jsonable(st.params)))["background"], self.bg), name)
            if pa.stimulus == "flash":
                self.assertEqual(st.params["first_s"], 0.5); self.assertEqual(st.params["size_deg"], 8.8)
                self.assertEqual(st.name, "flash_on" if pa.contrast > 0 else "flash_off")
            if pa.stimulus == "rect":
                self.assertEqual(st.params["width_deg"], pa.width); self.assertEqual(st.params["height_deg"], pa.height); self.assertEqual(st.params["contrast"], pa.contrast)

    def test_localizer_passes_sensitivity_table_and_stationarity(self):
        m = self.m
        from flyverse.interp import common
        one = m.localizer(self.col, spacing_deg=40.0, flash_s=0.1, blank_s=0.1, az_range=(-40, 40), el_range=(-40, 40))
        three = m.localizer(self.col, spacing_deg=40.0, flash_s=0.1, blank_s=0.1, az_range=(-40, 40), el_range=(-40, 40), passes=3)
        n = one.params["n_nodes"]
        self.assertEqual(three.params["passes"], 3); self.assertEqual(three.n_frames, 3 * n * 20 + 10); self.assertEqual(one.n_frames, n * 20 + 10)
        self.assertTrue(np.array_equal(three.track["order"][:n], one.track["order"]))                     # pass 0 is the single-pass order
        self.assertFalse(np.array_equal(three.track["order"][n:2 * n], one.track["order"]))              # a new order per pass
        for i in range(n):
            self.assertEqual(int((three.track["node"] == i).sum()), 30)                                   # each node 3 x 100 ms
        acc = m.NodeAccumulator(three, 1, ("drive_mv",))
        for t in range(three.n_frames):
            acc.add(t, {"drive_mv": np.array([1.0 if three.track["on"][t] else 0.0])})
        self.assertTrue((acc.finish()["n_on"] == 30).all())                                               # the reductions average over the passes
        az, el = np.meshgrid(np.arange(-50, 51, 10.0), np.arange(-30, 31, 10.0)); az, el = az.ravel(), el.ravel()
        rng = np.random.default_rng(3)
        R = np.c_[np.exp(-0.5 * ((az - 20) ** 2 + (el + 10) ** 2) / 8.0 ** 2) + 0.01 * rng.normal(size=len(az)), 0.01 * rng.normal(size=len(az))]
        Rb = 0.01 * rng.normal(size=R.shape)
        sens = m.rf_sensitivity([R, R], [Rb], az, el, np.array(["A", "B"]))
        self.assertEqual(len(sens), 2 * 4)
        self.assertTrue((sens[sens.type == "A"].coverage_all_runs == 1.0).all()); self.assertTrue((sens[sens.type == "B"].coverage_all_runs == 0.0).all())
        self.assertTrue((sens.false_fit_rate_blank_any_run == 0.0).all())
        T = 400; t = np.arange(T) * 0.01
        osc = np.sin(2 * np.pi * 4.0 * t)[:, None] * np.array([[1.0, 0.5]])                                # a 4-Hz oscillation in two cells
        rec = common.Recording(t * 1000.0, np.arange(3), np.arange(3) + 7, np.array(["X", "X", "Y"]),
                               {"drive_mv": np.c_[osc, np.zeros(T)].astype(np.float32)}, {}, {"arm": "blank"})
        st = m.stationarity_stats(rec, skip_s=1.0)
        x = st[st.type == "X"].iloc[0]; y = st[st.type == "Y"].iloc[0]
        self.assertAlmostEqual(float(x.pop_spectral_peak_hz), 4.0, delta=0.4); self.assertGreater(float(x.cell_sd_over_frames_median), 0.3)
        self.assertEqual(float(y.cell_sd_over_frames_median), 0.0); self.assertEqual(float(y.frame_diff_abs_median), 0.0); self.assertEqual(int(x.n_frames), 300)
        pooled = common.Recording(t * 1000.0, np.arange(3), np.arange(3) + 7, np.array(["X", "X", "Y"]), {"pooled_drive_mv": np.c_[osc[:, :1], np.zeros(T)].astype(np.float32)},
                                  {}, {"pooled_keys": ["X", "Y"]})
        sp = m.stationarity_stats(pooled, skip_s=1.0)
        self.assertTrue(sp.pooled.all()); self.assertAlmostEqual(float(sp[sp.type == "X"].iloc[0].pop_spectral_peak_hz), 4.0, delta=0.4)
        # divergence: identical arms until the first flash, a perturbed arm afterwards
        on = np.zeros(T, bool); on[100:120] = True
        s_arm = osc[:, :1].copy(); s_arm[100:] = np.sin(2 * np.pi * 4.0 * t[100:] + 1.0)[:, None]
        stim_rec = common.Recording(t * 1000.0, np.arange(1), np.array([7]), np.array(["X"]), {"pooled_drive_mv": s_arm.astype(np.float32)}, {}, {"pooled_keys": ["X"]})
        blank_rec = common.Recording(t * 1000.0, np.arange(1), np.array([7]), np.array(["X"]), {"pooled_drive_mv": osc[:, :1].astype(np.float32)}, {}, {"pooled_keys": ["X"]})
        dv = m.divergence_stats(stim_rec, blank_rec, on).iloc[0]
        self.assertEqual(int(dv.first_flash_frame), 100); self.assertEqual(float(dv.before_first_flash_max), 0.0)
        self.assertGreater(float(dv.during_first_flash_max), 0.1); self.assertGreater(float(dv.last_third_mean), 0.1); self.assertLess(float(dv.corr_last_third), 0.9)


if __name__ == "__main__":
    unittest.main()
