"""Auto-mark the test files that need something CI does not have, so the CPU job can select with

    python -m pytest -m "not gpu and not data and not cluster"

without editing any existing test file. Marks are applied by FILE NAME here rather than by decorator in
the files themselves; the markers are registered in pyproject.toml's [tool.pytest.ini_options].

  gpu      needs a CUDA or MPS device, or the runtime-compiled kernels (nvcc).
  data     needs the MaleCNS download and/or a compiled cache/ -- the ~3 GB that is not in the repo.
  cluster  needs a live GPU cluster (SSH + job manager). NOTHING carries this today: the cluster tests
           in tests/test_cluster_run.py are fully offline (canned in-process HTTP API, ssh/scp patched
           out, faked tunnel subprocess), so they run in CI. The marker is registered and applied here
           the moment a test does need a real cluster.

Everything not listed runs on a CPU-only machine with no data: the files that have cache-dependent
classes (test_receptor_model, test_optic_hooks, test_proprioception) already guard them with
`@unittest.skipUnless((CACHE_DIR / "W_post_pre.npz").exists(), ...)` and simply skip those cases, so
they stay in CI for the synthetic-graph coverage that is the bulk of each file.
"""
from __future__ import annotations

from pathlib import Path

import pytest

FILE_MARKS: dict[str, tuple[str, ...]] = {
    "test_cuda.py": ("gpu",),                    # opt-in FLYVERSE_CUDA_TESTS=1, compiles kernels with nvcc
    "test_metal.py": ("gpu",),                   # needs an MPS device
    "test_integration.py": ("gpu", "data"),      # opt-in FLYVERSE_INTEGRATION=1: full connectome cache + CUDA
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        path = getattr(item, "path", None) or Path(str(item.fspath))
        for name in FILE_MARKS.get(Path(path).name, ()):
            item.add_marker(getattr(pytest.mark, name))
