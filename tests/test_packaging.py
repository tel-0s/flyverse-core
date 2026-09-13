"""Release packaging contracts: the public import surface comes up with no MaleCNS data, no compiled
cache and no GPU; `import flyverse.interp` stays torch-free and cache-free; pyproject.toml and
CITATION.cff say what the release promises.

Everything here runs in a subprocess with FLYVERSE_DATA pointed at a path that does not exist, so a
regression that makes an import reach for the 3 GB download (or for cache/) fails here rather than in
someone's clean clone. No test in this file needs the data, a cache, a GPU or a network.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CITATION = ROOT / "CITATION.cff"
NO_DATA = str(ROOT / "no-such-malecns-directory")

# The public surface a newcomer is told to import (README "Use the brain in your own simulation").
PUBLIC_MODULES = ["flyverse.body", "flyverse.programs", "flyverse.env", "flyverse.batch_sim",
                  "flyverse.interp", "flyverse.nt_readout"]
PUBLIC_ATTRS = ["FlyBrain", "MotorRates", "StepResult", "AsyncFlyBrain", "BatchSim",
                "NTChannel", "NTSnapshot", "NTSource"]
# pyproject dependency name -> the module it provides.
DEP_IMPORTS = {"numpy": "numpy", "pandas": "pandas", "pyarrow": "pyarrow", "scipy": "scipy",
               "torch": "torch", "pygame": "pygame", "imageio": "imageio", "pillow": "PIL"}


def run_isolated(code: str):
    """Run `code` in a fresh interpreter with no MaleCNS data on the path. Returns CompletedProcess."""
    env = dict(os.environ, FLYVERSE_DATA=NO_DATA, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8",
               SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
    env.pop("FLYVERSE_CUDA_KERNELS", None)
    return subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                          capture_output=True, text=True, timeout=300)


def pyproject():
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)


class ImportSurfaceTests(unittest.TestCase):
    """The imports a clean clone has to survive before any data is fetched."""

    def test_public_surface_imports_without_the_dataset(self):
        code = ("import importlib, flyverse\n"
                f"for m in {PUBLIC_MODULES!r}: importlib.import_module(m)\n"
                f"for a in {PUBLIC_ATTRS!r}: assert getattr(flyverse, a) is not None, a\n"
                "from flyverse.fly import FlyBrain\n"
                "assert flyverse.FlyBrain is FlyBrain\n"
                "print('ok')\n")
        p = run_isolated(code)
        self.assertEqual(p.returncode, 0, f"public imports failed without data:\n{p.stdout}\n{p.stderr}")
        self.assertIn("ok", p.stdout)

    def test_import_flyverse_is_lazy_in_torch(self):
        """`import flyverse` alone must not drag in torch: the connectome CLI and fetch_data run without it."""
        p = run_isolated("import sys, flyverse\nprint('torch' in sys.modules)\n")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), "False", "import flyverse pulled in torch")

    def test_interp_imports_without_torch_or_the_cache(self):
        """The interpretability toolkit is pandas/pyarrow/scipy only at import time, and reads nothing."""
        code = r'''
import os, sys
opened = []
def hook(event, args):
    if event in ("open", "io.open") and args:
        try:
            opened.append(os.fspath(args[0]))
        except TypeError:
            pass
sys.addaudithook(hook)
import flyverse.interp
assert "torch" not in sys.modules, "flyverse.interp imported torch"
from flyverse import connectome as cn
CACHE = os.path.realpath(cn.CACHE_DIR)
touched = []
for p in opened:
    try:
        r = os.path.realpath(p)
    except (TypeError, ValueError):
        continue
    if isinstance(r, str) and (r.startswith(CACHE + os.sep) or "W_post_pre" in r or "sign0_counts" in r):
        touched.append(r)
assert not touched, "flyverse.interp read the connectome cache at import: %r" % (touched,)
print("ok")
'''
        p = run_isolated(code)
        self.assertEqual(p.returncode, 0, f"{p.stdout}\n{p.stderr}")
        self.assertIn("ok", p.stdout)

    def test_connectome_module_does_not_load_on_import(self):
        """flyverse.connectome exposes DATA_DIR/CACHE_DIR; it must not read either one at import time."""
        code = ("from flyverse import connectome as cn\n"
                "import os\n"
                f"assert str(cn.DATA_DIR) == {NO_DATA!r}, cn.DATA_DIR\n"
                "assert not os.path.exists(cn.DATA_DIR)\n"
                "print('ok')\n")
        p = run_isolated(code)
        self.assertEqual(p.returncode, 0, f"{p.stdout}\n{p.stderr}")
        self.assertIn("ok", p.stdout)


class PyprojectTests(unittest.TestCase):

    def test_requires_python_is_3_12_and_this_interpreter_satisfies_it(self):
        proj = pyproject()["project"]
        self.assertEqual(proj["requires-python"], ">=3.12")
        self.assertGreaterEqual(sys.version_info[:2], (3, 12),
                                "the test suite is running on an interpreter pyproject.toml excludes")

    def test_metadata_the_release_promises(self):
        proj = pyproject()["project"]
        self.assertEqual(proj["name"], "flyverse")
        self.assertEqual(proj["readme"], "README.md")
        self.assertEqual(proj["license"], "MIT")
        self.assertTrue((ROOT / "README.md").is_file())
        self.assertTrue((ROOT / "LICENSE").is_file())
        self.assertEqual([a["name"] for a in proj["authors"]], ["tel0s"])
        self.assertEqual(proj["urls"]["Repository"], "https://github.com/tel-0s/flyverse")
        self.assertEqual(proj["license-files"], ["LICENSE"])
        # PEP 639: the SPDX expression and a license classifier cannot coexist -- setuptools >= 77 fails
        # the build outright. Guard the combination that used to be the obvious thing to write.
        self.assertFalse([c for c in proj["classifiers"] if c.startswith("License ::")],
                         "a License:: classifier alongside the SPDX `license` breaks setuptools >= 77")
        for extra in ("ui", "cuda", "dev", "interp"):
            self.assertIn(extra, proj["optional-dependencies"])

    def test_every_declared_dependency_is_importable(self):
        deps = pyproject()["project"]["dependencies"]
        self.assertEqual(sorted(deps), sorted(DEP_IMPORTS), "update DEP_IMPORTS when dependencies change")
        for dep in deps:
            with self.subTest(dep=dep):
                __import__(DEP_IMPORTS[dep])

    def test_every_subpackage_is_shipped(self):
        """`packages` is explicit, so a new flyverse/<pkg>/__init__.py must be added to it or wheels lose it."""
        declared = set(tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["setuptools"]["packages"])
        found = {".".join(p.parent.relative_to(ROOT).parts)
                 for p in (ROOT / "flyverse").rglob("__init__.py") if "__pycache__" not in p.parts}
        self.assertEqual(found, declared)

    def test_pytest_markers_are_registered(self):
        ini = pyproject()["tool"]["pytest"]["ini_options"]
        self.assertEqual(ini["testpaths"], ["tests"])
        names = {m.split(":", 1)[0].strip() for m in ini["markers"]}
        self.assertEqual({"gpu", "data", "cluster"}, names)


class CitationTests(unittest.TestCase):

    def setUp(self):
        self.yaml = __import__("yaml") if self._has_yaml() else None
        if self.yaml is None:
            self.skipTest("PyYAML is not installed (it is in the [dev] extra)")
        self.cff = self.yaml.safe_load(CITATION.read_text(encoding="utf-8"))

    @staticmethod
    def _has_yaml():
        try:
            __import__("yaml")
            return True
        except ImportError:
            return False

    def test_citation_parses_and_is_cff_1_2_0(self):
        self.assertEqual(self.cff["cff-version"], "1.2.0")
        self.assertEqual(self.cff["type"], "software")
        for key in ("title", "authors", "message", "version", "date-released", "repository-code", "license"):
            self.assertIn(key, self.cff)
        self.assertEqual([a["name"] for a in self.cff["authors"]], ["tel0s"])
        self.assertEqual(self.cff["license"], "MIT")
        self.assertEqual(self.cff["repository-code"], "https://github.com/tel-0s/flyverse")

    def test_citation_version_tracks_pyproject(self):
        self.assertEqual(str(self.cff["version"]), pyproject()["project"]["version"])

    def test_the_data_and_model_references_are_present(self):
        refs = self.cff["references"]
        for ref in refs:                                  # CFF requires all three on every reference
            for key in ("type", "title", "authors"):
                self.assertIn(key, ref, ref.get("title"))
        malecns = [r for r in refs if "MaleCNS" in r["title"]]
        self.assertEqual(len(malecns), 1)
        self.assertEqual(malecns[0]["type"], "data")
        self.assertEqual(malecns[0]["url"], "https://storage.googleapis.com/flyem-male-cns/")
        shiu = [r for r in refs if any("Shiu" in a.get("family-names", "") for a in r["authors"])]
        self.assertEqual(len(shiu), 1)
        self.assertEqual(shiu[0]["journal"], "Nature")
        self.assertEqual(shiu[0]["year"], 2024)
        self.assertEqual(shiu[0]["doi"], "10.1038/s41586-024-07763-9")


if __name__ == "__main__":
    unittest.main()
