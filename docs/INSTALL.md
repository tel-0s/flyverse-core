# Install

From a clean clone to a fly walking on a table. Python **3.12+** (the code uses `X | None` at runtime and
the cluster venvs are 3.12; the development machine runs 3.13). Linux, Windows and macOS all work; a GPU is
optional for the demo and effectively required for the benchmark suite and batched sweeps.

The repository ships **no connectome data**: the MaleCNS files are ~3.7 GB and are fetched separately
(§2), and the third-party expression tables are other people's data and are not redistributed. Everything
in §1 works before any of that is downloaded, including the test suite.

```
git clone https://github.com/tel-0s/flyverse
cd flyverse
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
```

## 1. Install the package

### CPU

Plain `pip` gives a CPU torch on Windows and macOS; on Linux the default PyPI wheel is the CUDA build
(~2.5 GB), so ask for the CPU index explicitly if you do not want it:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu     # Linux, if you want CPU-only torch
pip install -e .
```

### CUDA

Install the CUDA build of torch **first**, from the PyTorch cu128 index, then the package — `pip install -e .`
sees the requirement already satisfied and leaves the wheel alone:

```
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # want True
```

Match the index to your driver (`cu126`, `cu128`, … at <https://pytorch.org/get-started/locally/>). cu128 is
what the benchmark numbers in `docs/PERFORMANCE.md` and the cluster runs were produced with.

### Extras

```
pip install -e ".[dev]"     # pytest, ruff, pyyaml -- what CI installs
pip install -e ".[ui]"      # pygame; ALREADY a hard dependency, the extra just names it
pip install -e ".[cuda]"    # empty on purpose: the CUDA wheel comes from the index above, not from PyPI
pip install -e ".[interp]"  # empty on purpose: the toolkit needs pandas/pyarrow/scipy, already required
```

`cuda` and `interp` install nothing. They exist so the documented commands are honest and so dependencies
can move into them later without changing anyone's install line. `flyverse.interp` imports **without torch**
and without reading `cache/` — the interpretability toolkit's import surface is pandas/pyarrow/scipy only.

### macOS

Apple-silicon torch (MPS) comes from the default PyPI wheel. `flyverse/metal.py` compiles the Metal kernels
at first use through `torch.mps.compile_shader`; set `FLYVERSE_METAL=0` for plain torch.

## 2. Fetch the data

`scripts/fetch_data.py` is driven by `flyverse/data/manifest.json` (URL, SHA-256, size and the citation and
licence of every file), streams to a `.part` file and renames only after the hash matches. Standard library
only — it runs before the package is installed if you like.

```
python scripts/fetch_data.py --list                 # what the manifest knows, and what is already present
python scripts/fetch_data.py --malecns              # the MaleCNS files flyverse reads (~3.7 GB)
python scripts/fetch_data.py --fafb                 # female brain v783, thresholded edges
python scripts/fetch_data.py --fafb --edges no_threshold  # also fetch the optional unthresholded pair table
python scripts/fetch_data.py --banc                 # female brain + VNC v888
python scripts/fetch_data.py --external all         # third-party expression/typing tables -> data/external/
python scripts/fetch_data.py --verify               # re-hash everything present
```

The MaleCNS files go to `flyverse.connectome.DATA_DIR`, which is `$FLYVERSE_DATA` (or a Windows default);
set it before fetching, or pass `--data-dir`:

```
export FLYVERSE_DATA=/data/male-cns-connectome-v1.0/flat-connectome     # Windows: setx FLYVERSE_DATA ...
python scripts/fetch_data.py --malecns
```

`--external` is only needed to *rebuild* the receptor/transmitter tables under `flyverse/data/` with the
`scripts/build_*.py` scripts; the built CSVs are committed, so running the model does not need it.
`data/external/` is git-ignored (Özel 2021, Davis 2020, Kurmangaliyev 2020, Nern 2025 and the typing tables
are redistributed under their own licences — cite them from the manifest, do not re-host them).

Female data paths are `FLYVERSE_DATA_FAFB` and `FLYVERSE_DATA_BANC`; the Windows defaults are
`D:\Datasets\flywire\Female Adult Fly Brain v783` and `D:\Datasets\flywire\BANC v888`.
The manifest pins public release URLs and SHA-256 hashes. No skeleton archive or per-synapse geometry
archive is fetched. FAFB's `labels.csv.gz` is used by the independent dorsal-rim validation.

```python
from flyverse import connectome
connectome.load(dataset="fafb")
connectome.load(dataset="banc")
```

These compile into separate `cache/fafb/` and `cache/banc/` directories. The MaleCNS files remain intact.
`FLYVERSE_CACHE` overrides the cache parent. BANC supports body experiments without an optic module;
FAFB supplies vision but lacks VNC motor/proprioceptive populations. See `docs/CONTROL_SURFACE.md`.

## 3. Build the cache

`cache/` (git-ignored, ~200 MB) holds the compiled graph: the neuron table plus the signed CSR weight
matrix. Build it once from the fetched files; it is a one-off cost of minutes and several GB of RAM (the
weights table alone is a ~1 GB feather that is read into memory):

```
python -c "from flyverse import connectome; connectome.load(rebuild=True)"
python -m flyverse.connectome            # the same thing, then prints N and nnz
```

Afterwards `connectome.load()` reads `cache/W_post_pre.npz`, `cache/neurons.parquet` and
`cache/sign0_counts.npz` in a second or two and the MaleCNS files are not touched again. The shipped
default cache fingerprint is `sum|W| = 121,460,584` — `scripts/hash_weights.py` prints yours.

## 4. Run the room demo

```
python scripts/room_demo.py                                        # live pygame window
python scripts/room_demo.py --fruit apple --fence                  # one apple, a fence around the table top
python scripts/room_demo.py --program cx --escape-gating           # simulated central-complex steering
python scripts/room_demo.py --headless --seconds 20 --loom-at 4 --gif out/room.gif
```

Keys and the rest of the flags: the module docstring at the top of `scripts/room_demo.py`, and
`docs/ROOM_UI.md`. `--headless` sets `SDL_VIDEODRIVER=dummy` for you, which is what you want over SSH or in
a container. On CPU the demo runs, slowly; the brain is 72k spiking neurons behind an 89k-unit graded optic
lobe and wants a GPU for anything real-time (`docs/PERFORMANCE.md`).

## 5. Run the tests

The CPU subset needs **no data, no cache and no GPU** — it is what CI runs
(`.github/workflows/ci.yml`):

```
pip install -e ".[dev]"
python -m pytest -m "not gpu and not data and not cluster" -q
```

`tests/conftest.py` applies the markers by file name; `pyproject.toml` registers them:

| marker    | what it means                                       | files |
|-----------|-----------------------------------------------------|-------|
| `gpu`     | needs CUDA/MPS or the nvcc-compiled kernels         | `test_cuda.py`, `test_metal.py` |
| `data`    | needs the MaleCNS download and/or a compiled cache  | `test_integration.py` (also `gpu`) |
| `cluster` | needs a live cluster                                | none — `test_cluster_run.py` is fully mocked and runs in CI |

Everything else runs on the synthetic graphs the tests build themselves. Three files
(`test_receptor_model.py`, `test_optic_hooks.py`, `test_proprioception.py`) have extra classes that pin
numbers against the real cache; those classes skip on their own when `cache/W_post_pre.npz` is absent and
run automatically once you have built the cache. With the cache and a GPU, run the lot:

```
python -m pytest -q                                       # everything the machine can do
FLYVERSE_CUDA_TESTS=1 python -m pytest tests/test_cuda.py -q
FLYVERSE_INTEGRATION=1 python -m pytest tests/test_integration.py -q
```

Windows note: `PYTHONIOENCODING=utf-8` if your console is not UTF-8 — several tests print the type names
from the connectome.

Lint the way CI does (syntax errors and undefined names only; style is not gated):

```
ruff check flyverse scripts tests --select E9,F63,F7,F82
```

## 6. Optional: the CUDA kernels

`flyverse/cuda.py` compiles `flyverse/kernels/neural.cu` with **nvcc** at first use into a small shared
library called with raw device pointers (no Torch C++ ABI dependency). It is opt-in and entirely optional —
everything works on the plain torch path without it. You need a CUDA toolkit on `PATH` (`nvcc --version`),
not just a CUDA torch wheel:

```
FLYVERSE_CUDA_KERNELS=1 python scripts/room_demo.py        # or FlyBrain(..., cuda_kernels=True)
python scripts/benchmark.py --fast --json out/bench.json   # native backend: kernels + graphs + event-driven
```

Scripts that take the backend flags spell them out, e.g. `--cuda-graphs --cuda-kernels --event-driven
--cuda-sparse torch` (warp CSR is batch-1 only; batched runs need `torch`). If nvcc is missing, ask for the
kernels explicitly and you get an error rather than a silent fallback; leave `FLYVERSE_CUDA_KERNELS` unset
and nothing is compiled.

## 7. Optional: the cluster runner

`scripts/cluster_run.py` ships this checkout's uncommitted diff to one or more GPU targets, submits each
command as a job through the job manager, waits, prints the logs and copies `--fetch` paths back — e.g.
`python scripts/cluster_run.py --name bench "python scripts/benchmark.py --fast --json out/bench.json"
--fetch out/bench.json`, with several commands in one call forming a batch that runs concurrently. Addresses,
filesystem paths and the submitting user come from `.cluster.json` at the repo root or the `FLYVERSE_CLUSTER`
environment variable (the same JSON); nothing infrastructural is hard-coded. Both `.cluster.json` and the
operator guide `docs/CLUSTER.md` are **git-ignored on purpose** — they describe private infrastructure, not
the model — so a public clone has neither and simply does not use this path; write your own `.cluster.json`
against the schema documented in the `cluster_run.py` docstring, which `tests/test_cluster_run.py` pins
offline (canned in-process API, `ssh`/`scp` patched out, no job submitted anywhere).

## Troubleshooting

| symptom | cause |
|---|---|
| `FileNotFoundError: ...-male-cns-v1.0-minconf-0.5.feather` | `FLYVERSE_DATA` is unset or points elsewhere; §2 |
| `connectome.load()` is slow every time | no `cache/`; run §3 once |
| `torch.cuda.is_available()` is `False` on a GPU box | a CPU wheel got installed; reinstall torch from the cu128 index *before* `pip install -e .` |
| `pip install -e .` fails on the `license` field | setuptools < 77; `pip install -U setuptools` (PEP 639 SPDX metadata) |
| pygame cannot open a display over SSH | `--headless`, or `SDL_VIDEODRIVER=dummy` |
| `nvcc: not found` with `FLYVERSE_CUDA_KERNELS=1` | a CUDA *toolkit* is needed, not just the CUDA torch wheel; §6 |

Architecture and what the model actually does: `README.md`, then `docs/ARCHITECTURE.md`,
`docs/NOTES.md` and `docs/BENCHMARK_BATTERY.md`.
