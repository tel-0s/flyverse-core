"""The toolkit's front door (flyverse/interp/deficit.py, scripts/interp_deficit.py; docs/INTERP.md 10.1 and 11.2 item
14) on a synthetic graph and a synthetic expectation table: finding a row by id or check key, where it looks (readout
and sources, the declared tables and the overrides), step 2 end to end with the tools' own Result JSONs, the report and
the CLI. CPU only, no dataset. The validation on the shipped cache (`deficit.validate`) is the last class and skips
without the compiled cache.

    CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python -m pytest tests/test_interp_deficit.py -q
"""
from __future__ import annotations

import csv
import json
import os
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

from flyverse import connectome as cn                      # noqa: E402
from flyverse.brain import LIFParams                       # noqa: E402
from flyverse.connectome import CACHE_DIR, Connectome      # noqa: E402
from flyverse.interp import common                         # noqa: E402
from flyverse.interp import deficit as DF                  # noqa: E402
from flyverse.interp import ledger as L                    # noqa: E402

CELLS = [  # bodyId, type, superclass, class, subclass, somaSide, nt
    (1, "ORN_DM1", "cb_sensory", "olfactory", "", "L", "acetylcholine"),
    (2, "ORN_DM1", "cb_sensory", "olfactory", "", "R", "acetylcholine"),
    (3, "DM1_lPN", "cb_intrinsic", "", "", "L", "acetylcholine"),
    (4, "LHN1", "cb_intrinsic", "", "", "L", "acetylcholine"),
    (5, "DNa02", "descending_neuron", "", "", "L", "acetylcholine"),
    (6, "DNa02", "descending_neuron", "", "", "R", "acetylcholine"),
    (7, "OA-VUM", "cb_intrinsic", "", "", "L", "octopamine"),          # sign 0 onto DNa02_L
    (8, "GABAin", "cb_intrinsic", "", "", "L", "gaba"),
    (9, "LAL001", "cb_intrinsic", "", "", "L", "acetylcholine"),       # a ring rotation input ('~^LAL')
    (10, "GLNO", "cb_intrinsic", "", "", "L", "unknown"),              # sign 0 onto PEN_a
    (11, "PEN_a", "cb_intrinsic", "", "", "R", "acetylcholine"),
    (12, "EPG", "cb_intrinsic", "", "", "L", "acetylcholine"),
    (13, "PS196_b", "cb_intrinsic", "", "", "R", "acetylcholine"),     # a ring rotation input, direct onto EPG
    (14, "SpsP", "cb_intrinsic", "", "", "L", "acetylcholine"),        # a ring rotation input (exact type)
    (15, "IbSpsP", "cb_intrinsic", "", "", "L", "acetylcholine"),      # NOT one: only contains 'SpsP'
    (16, "DNp01", "descending_neuron", "", "", "L", "acetylcholine"),
    (17, "MDN", "descending_neuron", "", "", "L", "acetylcholine"),
    (18, "Fe reductor MN", "vnc_motor", "", "fl", "L", "acetylcholine"),
]
EDGES = [  # pre, post, synapses
    (1, 3, 30), (2, 3, 30), (3, 4, 40), (4, 5, 20), (4, 6, 20),       # ORN -> PN -> LHN -> DNa02 L / R
    (7, 5, 15),                                                         # OA-VUM -> DNa02_L (sign 0)
    (8, 5, 10), (8, 6, 10),                                             # GABAin -> DNa02
    (9, 10, 50), (10, 11, 200), (11, 12, 30),                           # LAL001 -> GLNO -> PEN_a -> EPG
    (13, 12, 5), (14, 11, 8), (15, 11, 9),                              # PS196_b -> EPG; SpsP / IbSpsP -> PEN_a
    (16, 5, 4), (17, 18, 20),                                           # DNp01 -> DNa02_L; MDN -> leg MN
]


def graph() -> tuple[Connectome, sp.csr_matrix]:
    """The synthetic connectome and its raw-count matrix (the sign-0 entries' counts included, as
    cache/sign0_counts.npz supplies them for the shipped cache: W stores them as explicit zeros)."""
    n = pd.DataFrame(CELLS, columns=["bodyId", "type", "superclass", "class", "subclass", "somaSide", "nt"])
    n["bodyId"] = n.bodyId.astype(np.int64)
    n["instance"] = n.type + "_" + n.somaSide
    n["sign"] = np.array([cn.NT_SIGN[x] for x in n.nt], dtype=np.float32)
    N, k = len(n), {b: i for i, b in enumerate(n.bodyId)}
    rows = [k[post] for _, post, _ in EDGES]; cols = [k[pre] for pre, _, _ in EDGES]
    cnt = np.array([s for *_, s in EDGES], np.float32)
    sign = n.sign.to_numpy()
    W = sp.csr_matrix((cnt * sign[cols], (rows, cols)), shape=(N, N), dtype=np.float32); W.sort_indices()
    C = sp.csr_matrix((cnt, (rows, cols)), shape=(N, N), dtype=np.float32); C.sort_indices()
    return Connectome(n, W, pd.Series(np.arange(N), index=n.bodyId)), C


ROWS = [
    dict(row_id="demo.DNa02.rate_hz", population="DNa02", stimulus="apple_odour", quantity="rate_hz", expected="2",
         op=">", bound="1", unit="Hz", gap="1", check_key="demo.dna02_check", source="a citation", model_reference="a probe"),
    dict(row_id="compass.demo.EPG.cells", population="EPG", stimulus="compass_pulse_after", quantity="cells_persisting",
         expected="8", op=">=", bound="6", unit="cells", gap="1"),
    dict(row_id="demo.struct", population="GLNO->PEN_a", stimulus="structural", quantity="sign", expected="-1",
         op="==", bound="-1", unit="sign", gap="1"),
    dict(row_id="demo.body.escapes", population="body", stimulus="loom_demo", quantity="escapes", expected="1",
         op=">=", bound="1", unit="count"),
    dict(row_id="demo.body.steps", population="body", stimulus="leg_cycle", quantity="step_frequency_hz_at_speed",
         expected="10", op="range", bound="5|15", unit="Hz"),
    dict(row_id="demo.MDN.leg", population="MDN", stimulus="MDN_stim_150hz", quantity="leg_L_hz", expected="3",
         op=">", bound="0.1", unit="Hz"),
    dict(row_id="demo.DNa02.self", population="DNa02", stimulus="DNa02_L_stim_150hz", quantity="top_rate_hz",
         expected="150", op="<", bound="250", unit="Hz", requires="demo.DNa02.rate_hz"),
    dict(row_id="demo.power", population="wing_power_MN", stimulus="walking", quantity="rate_max_hz", expected="20",
         op="<", bound="50", unit="Hz", check_key="walk.power_max_hz"),
    dict(row_id="demo.nosource", population="DNa02", stimulus="CSDn_serotonin", quantity="rate_hz", expected="1",
         op=">", bound="0", unit="Hz"),
]


def write_table(d: Path, rows=ROWS) -> Path:
    path = Path(d) / "expected.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=L.COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in L.COLUMNS})
    return path


def params():
    return LIFParams(receptor_model=None, event_driven=False)


class WhereToLookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.table = write_table(Path(self.tmp.name))
        self.c, self.C = graph()

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_row_by_id_or_by_the_check_it_mirrors(self):
        self.assertEqual(DF.find_row("demo.DNa02.rate_hz", self.table).row_id, "demo.DNa02.rate_hz")
        self.assertEqual(DF.find_row("demo.dna02_check", self.table).row_id, "demo.DNa02.rate_hz")
        with self.assertRaises(KeyError):
            DF.find_row("no.such.row", self.table)
        sub = Path(self.tmp.name) / "dup"; sub.mkdir()
        dup = write_table(sub, ROWS + [dict(ROWS[1], row_id="compass.demo.EPG.cells2", check_key="demo.dna02_check")])
        with self.assertRaisesRegex(KeyError, "2 rows"):
            DF.find_row("demo.dna02_check", dup)

    def test_readouts(self):
        row = lambda rid: DF.find_row(rid, self.table)                                  # noqa: E731
        self.assertEqual(DF.readout_of(self.c, row("demo.DNa02.rate_hz"))["spec"], "DNa02")
        edge = DF.readout_of(self.c, row("demo.struct"))
        self.assertEqual((edge["kind"], edge["pre"], edge["spec"]), ("edge", "GLNO", "PEN_a"))
        self.assertEqual(DF.readout_of(self.c, row("demo.body.escapes"))["spec"], "DNp01")
        self.assertEqual(DF.readout_of(self.c, row("demo.power"))["spec"], DF.POWER_MN_TYPES)
        self.assertEqual(DF.readout_of(self.c, row("demo.MDN.leg"))["spec"], DF.LEG_MN)
        self.assertEqual(DF.readout_of(self.c, row("demo.DNa02.self"))["kind"], "self")
        with self.assertRaisesRegex(ValueError, "readout="):
            DF.readout_of(self.c, row("demo.body.steps"))                              # the body, no declared neural readout
        self.assertEqual(DF.readout_of(self.c, row("demo.body.steps"), override="MDN")["spec"], "MDN")

    def test_sources(self):
        row = lambda rid: DF.find_row(rid, self.table)                                  # noqa: E731
        r = row("demo.DNa02.rate_hz")
        self.assertEqual(DF.sources_of(self.c, r, DF.readout_of(self.c, r)), [DF.ORNS])
        r = row("demo.struct")
        self.assertEqual(DF.sources_of(self.c, r, DF.readout_of(self.c, r))[0][1], "GLNO")
        r = row("demo.MDN.leg")
        self.assertEqual(DF.sources_of(self.c, r, DF.readout_of(self.c, r))[0][1], "MDN")
        r = row("demo.DNa02.self")                                                      # 'DNa02_L' is no type: the side is a clause
        self.assertEqual(DF.sources_of(self.c, r, DF.readout_of(self.c, r))[0][1], "type=DNa02&somaSide=L")
        r = row("demo.nosource")
        self.assertEqual(DF.sources_of(self.c, r, DF.readout_of(self.c, r)), [])
        self.assertEqual(DF.sources_of(self.c, r, DF.readout_of(self.c, r), override=[("mine", "GABAin")]), [("mine", "GABAin")])

    def test_a_joined_spec_is_read_as_the_cli_reads_it(self):
        """'LNO1|...|SpsP|PS196_b|~^LAL' passed whole is ONE unanchored regex on type: it would take IbSpsP (which only
        contains 'SpsP') and never LAL001 ('~^LAL' matches no type name). The front door reads it as interp_paths.py does."""
        from flyverse.interp import paths as P
        spec = DF.RING_ROTATION_INPUTS[1]
        ty = self.c.neurons.type.to_numpy()
        self.assertIn("IbSpsP", set(ty[P.resolve_loose(self.c, spec)]))                 # the trap, as the grammar has it
        self.assertEqual(set(ty[P.resolve_loose(self.c, DF.as_spec(spec))]), {"LAL001", "PS196_b", "SpsP"})


class DiagnoseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.d = Path(self.tmp.name); self.table = write_table(self.d)
        self.c, self.C = graph()

    def tearDown(self):
        self.tmp.cleanup()

    def diagnose(self, row, **kw):
        return DF.diagnose(self.c, row, table=self.table, params=params(), counts=self.C, out_dir=self.d / row,
                           log=lambda *a, **k: None, **kw)

    def test_odour_row_end_to_end(self):
        rep = self.diagnose("demo.dna02_check")
        self.assertEqual(rep["schema"], DF.SCHEMA)
        self.assertEqual(rep["row"]["row_id"], "demo.DNa02.rate_hz")
        self.assertIsNone(rep["status"])
        st = rep["structure"]
        self.assertEqual(st["readout"]["n_cells"], 2)
        src = st["sources"][0]
        self.assertEqual(src["label"], "ORNs"); self.assertEqual(src["reachable_k"], 3)
        self.assertEqual(src["top_signed"]["3"]["path"], "a:ORN_DM1 -> DM1_lPN -> LHN1 -> b")     # JSON keys
        first = rep["answers"]["first_signed_route_per_source"]["ORNs"]
        self.assertEqual(first["k"], 3)
        silent = {x["pre_type"]: x for x in st["readout_inputs"]["top_silent"]}
        self.assertIn("OA-VUM", silent)                                                # sign 0, counted from the raw counts
        self.assertEqual(silent["OA-VUM"]["raw_count"], 15.0)
        self.assertEqual(silent["OA-VUM"]["silent"], "sign0")
        self.assertAlmostEqual(st["readout_inputs"]["raw_input_total"], 20 + 20 + 15 + 10 + 10 + 4)
        ei = rep["answers"]["E_I_per_volley"]["DNa02"]
        self.assertGreater(ei["E_total"], 0); self.assertLess(ei["I_total"], 0)
        self.assertAlmostEqual(ei["net"], ei["E_total"] + ei["I_total"], places=9)
        self.assertEqual(ei["cancelling_pair"]["I"], "GABAin")
        # the tools' own Results, loadable and exportable, and the report files
        for f in [src["json"], st["decompose"]["json"]]:
            back = common.Result.load(f)
            self.assertEqual(back.check(), [])
        self.assertEqual(common.Result.load(src["json"]).tool, "paths")
        self.assertEqual(common.Result.load(st["decompose"]["json"]).tool, "decompose")
        md = Path(rep["files"]["markdown"]).read_text(encoding="utf-8")
        for needle in ("## 1. The readout and the expectation", "**(a) The strongest silent link per k**",
                       "**(b) E / I per volley", "**(c) Is any route", "## 3-7. What to run next",
                       "scripts/interp_trace.py record --protocol odour", "demo.dna02_check", "KNOWN GAP"):
            self.assertIn(needle, md)
        self.assertIn("OVERRIDDEN", md)                         # receptor_model=None is not the shipped model, and it says so
        self.assertFalse(rep["model"]["shipped"])
        self.assertTrue(st["readout_inputs"]["sign0_counts_available"])
        self.assertEqual(json.loads(Path(rep["files"]["report"]).read_text(encoding="utf-8"))["row"]["row_id"], "demo.DNa02.rate_hz")

    def test_compass_row_finds_the_sign0_link(self):
        rep = self.diagnose("compass.demo.EPG.cells")
        src = rep["structure"]["sources"][0]
        self.assertEqual(src["n_cells"], 3)                                            # LAL001, PS196_b, SpsP -- not IbSpsP
        best = rep["answers"]["strongest_silent_link_per_k"]
        self.assertEqual((best["3"]["pre"], best["3"]["post"], best["3"]["silent"]), ("GLNO", "PEN_a", "sign0"))
        self.assertEqual(best["3"]["raw_count"], 200.0)
        self.assertEqual(rep["answers"]["first_signed_route_per_source"][src["label"]]["path"], "a:PS196_b -> b")
        self.assertIn("docs/audits/deficit_rotation.md", rep["known_audit"])
        self.assertTrue(any("interp_health.py record --protocol compass" in x for x in rep["plan"]))

    def test_structural_row_is_one_link(self):
        rep = self.diagnose("demo.struct")
        self.assertEqual(rep["k_max"], 1)                                              # one link, k = 1
        src = rep["structure"]["sources"][0]
        self.assertEqual(src["n_walks"], {"1": 1})
        self.assertEqual(src["direct"]["raw_count"], 200.0)
        self.assertIsNone(rep["structure"]["decompose"])                               # an edge row decomposes nothing
        self.assertIn("structural row", rep["plan"][0])

    def test_rows_with_nothing_to_trace_say_so(self):
        rep = self.diagnose("demo.nosource")
        self.assertEqual(rep["structure"]["sources"], [])
        self.assertEqual([x["pre_type"] for x in rep["structure"]["readout_inputs"]["top_silent"]], ["OA-VUM"])   # no source needed
        self.assertTrue(any("no sensory entry is declared" in n for n in rep["structure"]["notes"]))
        self.assertIsNotNone(rep["structure"]["decompose"])                            # the readout's own inputs still come
        rep = self.diagnose("demo.DNa02.self")
        self.assertTrue(any("own rate" in n for n in rep["structure"]["notes"]))
        self.assertIn("Requires", DF.render(rep))

    def test_a_wide_readout_prints_the_first_types_and_keeps_all(self):
        from unittest.mock import patch
        rep = self.diagnose("demo.nosource", readout="~.")                               # every cell; 7 types receive input
        n = len(rep["structure"]["decompose"]["by_type"])
        self.assertEqual(n, 7)
        with patch.object(DF, "MAX_TYPES_SHOWN", 3):
            md = DF.render(rep)
        self.assertIn("4 more readout types in report.json", md)
        self.assertEqual(md.count("= net **"), 3)

    def test_status_is_scored_from_finished_results(self):
        bench = self.d / "bench.json"
        bench.write_text(json.dumps({"sections": {}, "config": {}, "checks": [{"key": "demo.dna02_check", "measured": 0.25}]}))
        rep = self.diagnose("demo.DNa02.rate_hz", results=[str(bench)])
        mine = [s for s in rep["status"] if s["row_id"] == "demo.DNa02.rate_hz"]
        self.assertEqual(len(mine), 1)
        self.assertEqual(mine[0]["measured"], 0.25)
        self.assertEqual(mine[0]["status"], "KNOWN GAP")                                # 0.25 > 1 fails on a gap row
        self.assertIn("KNOWN GAP -- measured 0.25", DF.render(rep))

    def test_plan_runs_nothing_and_keeps_the_exit_code(self):
        rep = self.diagnose("demo.DNa02.rate_hz")
        plan = "\n".join(rep["plan"])
        self.assertIn("st=\\$?", plan)                                                  # INTERP 10.4 item 4
        self.assertIn("predeclared.json", plan)
        self.assertIn("min(n_a, n_b) >= 4", plan)
        self.assertIn("interp_lesion.py plan", plan)                                   # the row mirrors a suite check


class CliTests(unittest.TestCase):
    def test_list_and_source_parsing(self):
        import interp_deficit as cli
        self.assertEqual(cli._source("DANs=~^(PAM|PPL1)"), ("DANs", "~^(PAM|PPL1)"))
        self.assertEqual(cli._source("type=DNa02&somaSide=L"), ("type=DNa02&somaSide=L", "type=DNa02&somaSide=L"))
        self.assertEqual(cli._source("class=olfactory"), ("class=olfactory", "class=olfactory"))
        with tempfile.TemporaryDirectory() as d:
            t = cli.list_rows(write_table(Path(d)))
        self.assertEqual(set(t.row_id[:3]), {"demo.DNa02.rate_hz", "compass.demo.EPG.cells", "demo.struct"})   # gap rows first
        self.assertTrue(t.gap.iloc[:3].all()); self.assertFalse(t.gap.iloc[3:].any())
        self.assertEqual(t.set_index("row_id").loc["demo.MDN.leg", "sources"], "the stimulated population")


@unittest.skipUnless((CACHE_DIR / "sign0_counts.npz").exists() and (CACHE_DIR / "W_post_pre.npz").exists(),
                     "needs the compiled MaleCNS cache with its sign-0 counts (python -m flyverse.connectome)")
class ShippedCacheValidation(unittest.TestCase):
    """VALIDATION on the shipped cache: GLNO -> PEN (sign 0) as the strongest silent link into the compass, DNa02 net
    excitatory per volley with LLPC1 its largest excitatory input -- from the ledger rows alone."""

    def test_validate(self):
        c = cn.load(verbose=False)
        if common.connectome_fingerprint(c)["md5"] != "ef23cc27bea13be7f6a96f3c04fd3737":
            self.skipTest("cache/ is not the shipped graph (compiled-W md5 differs): rebuild it")
        v = DF.validate(c, log=lambda *a, **k: None)
        self.assertEqual(v["status"], "reproduced", v["measured"])
