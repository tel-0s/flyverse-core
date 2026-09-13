"""lesion -- a manifest of lesion sets x the benchmark checks and probes -> a check x lesion delta matrix with scatter.

The counterfactual half of the toolkit (docs/INTERP.md section 4.4). It never edits the model: every lesion is an
override applied for the duration of one job -- a `LIFParams` / `OpticParams` value, a `--receptor-table` (the hold
tables of `scripts/build_hold_tables.py`), or an in-process mask on `brain._shaped_weights` (the hook pattern of
`scripts/retire_measures.py`) -- and the JSON records exactly what was applied, on which bodies, with how many
entries and synapses removed.

    plan     resolve the manifest on the connectome (bodies per lesion, entries per hold table), write
             <out_dir>/manifest.resolved.json and <out_dir>/batch.sh: ONE scripts/cluster_run.py call with one job
             per lesion x replicate plus `replicates` baseline jobs and --fetch <out_dir>/
    run      one job in-process: scripts/benchmark.py's sections (through a benchmark.Context subclass, the
             retire_measures pattern, so a weight mask can be installed) plus the manifest's probes through their
             CLIs; writes <out_dir>/<lesion>_r<k>.json
    analyse  the finished job JSONs -> the check x lesion matrix (baseline, value, delta, replicate values / sd / n,
             status change), the Neurome `sensitivity` table, and the double dissociations

A dissociation is (L1, L2) x (c1, c2) with c1 moved by L1 and not by L2 and c2 moved by L2 and not by L1 --
bit-identity when the check is deterministic in every arm (nine of the sixteen checks of
`docs/audits/receptor_integration.md` E.2 are), else beyond twice the pooled replicate scatter.

Validation (VALIDATION['lesion'], EXPECTED below): the round-4 / round-5 hold matrix -- `taste.MN9_hz` 10.9342 under
default / holdKC / holdDN1 / holdOptic and 5.8455 under holdBrain / off, `smell.KC_active` 816 / 1155 / 1109 / 816 /
1426, `walk.power_max_hz` non-monotone in the number of applied flips -- and, on the CPU, the E.4 double dissociation
(`holdBrainGlu` = default on taste and off on smell, `holdBrainHis` the mirror image, three brain seeds).
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import glob
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import common
from .common import MIN_REPLICATES, Result, body_str, print_table, to_jsonable

ROOT = common.ROOT
SCRIPTS = ROOT / "scripts"
JOB_SCHEMA = "flyverse.interp.lesion.job/1"
TOOL = "lesion"
DEFAULT_SECTIONS = "rest,taste,smell,walk,bitter"
KINDS = ("none", "population", "types", "module", "transmitter", "receptor_tier", "hold_table", "lif", "optic", "pair_gain")
WEIGHT_KINDS = ("population", "types", "module", "transmitter")
HOLD_GROUPS = ("KC", "DN1", "Brain", "Optic", "BrainGlu", "BrainHis")
MAX_BODIES_RECORDED = 5000          # above this a lesion records n_bodies + the spec, not every bodyId (schema rule 4)


# ------------------------------------------------------------------------------------------------ manifests
def _hold(group: str) -> dict:
    return {"id": f"hold{group}", "kind": "hold_table", "group": group,
            "spec": f"out/receptors_hold{group}.csv", "receptor_model": "sign", "receptor_net_rule": "abs",
            "note": f"scripts/build_hold_tables.py group {group}: those rows held at NT_SIGN, every other default entry kept"}


# The manifests the validation runs. `holds` is the round-4 / round-5 attribution batch (GPU); `holds_cpu` is the
# E.4 double dissociation (CPU, taste / smell only, no optic lobe, one brain seed per replicate).
BUILTIN_MANIFESTS = {
    "holds": {"name": "holds", "sections": DEFAULT_SECTIONS,
              "baseline": {"id": "baseline", "kind": "none", "receptor_model": "sign", "receptor_net_rule": "abs",
                           "note": "the shipped default (receptor_model sign, net rule abs), passed explicitly: "
                                   "--receptor-model default returns before --receptor-table is applied"},
              "lesions": [_hold("KC"), _hold("DN1"), _hold("Brain"), _hold("Optic"),
                          {"id": "off", "kind": "lif", "spec": {"receptor_model": None}, "receptor_model": "off",
                           "note": "the presynaptic NT_SIGN rule: no receptor entry applied at all"}],
              "probes": [],
              "source": "docs/audits/receptor_integration.md E.1-E.2 (r5-attr) and the round-4 re-score table"},
    "holds_cpu": {"name": "holds_cpu", "sections": "taste,smell",
                  "baseline": {"id": "baseline", "kind": "none", "receptor_model": "sign", "receptor_net_rule": "abs",
                               "note": "the shipped default"},
                  "lesions": [_hold("Brain"), _hold("Optic"), _hold("BrainGlu"), _hold("BrainHis"),
                              _hold("KC"), _hold("DN1"),
                              {"id": "off", "kind": "lif", "spec": {"receptor_model": None}, "receptor_model": "off",
                               "note": "the presynaptic NT_SIGN rule"}],
                  "brain_seeds": [0, 1, 2],
                  "probes": [],
                  "source": "docs/audits/receptor_integration.md E.4 (scripts/r5_attr_taste_cpu.py)"},
}

# What each arm must reproduce. Numbers are the audit's; a [lo, hi] entry is a scattering check quoted as a range.
EXPECTED = {
    "gpu": {
        "source": "docs/audits/receptor_integration.md E.2 (out/r5_attr_*.json, 4 draws per hold arm) and the "
                  "round-4 re-score table (out/r4_holdKC_*.json, out/r4_holdDN1_*.json, 2 draws each)",
        "checks": {
            "rest.spikes_per_step": {"baseline": 0.0, "holdBrain": 0.0, "holdOptic": 0.0, "off": 0.0},
            "taste.MN9_hz": {"baseline": 10.9342, "holdKC": 10.9342, "holdDN1": 10.9342, "holdOptic": 10.9342,
                             "holdBrain": 5.8455, "off": 5.8455},
            "smell.KC_active": {"baseline": 816, "holdKC": 1155, "holdDN1": 1109, "holdOptic": 816,
                                "holdBrain": 1426, "off": 1426},
            "smell.PN_hz": {"baseline": 7.8612, "holdKC": 11.64, "holdDN1": 10.64, "holdOptic": 7.8612,
                            "holdBrain": 11.1877, "off": 11.1877},
            "bitter.calibrated_sugar_MN9_hz": {"baseline": 5.5184, "holdKC": 5.52, "holdDN1": 5.52,
                                               "holdOptic": 5.5184, "holdBrain": 4.5659, "off": 4.5659},
            "bitter.shiu_sugar_MN9_hz": {"baseline": 139.8985, "holdKC": 139.90, "holdDN1": 129.27,
                                         "holdOptic": 139.8985, "holdBrain": 123.5394, "off": 123.5394},
            "bitter.shiu_sugar_bitter_MN9_hz": {"baseline": 0.8178, "holdOptic": 0.8178, "holdBrain": 2.1243, "off": 2.1243},
            "walk.GF_max_hz": {"baseline": 4.6292, "holdOptic": 13.3109, "holdBrain": 12.5174, "off": 4.9641},
            "walk.power_max_hz": {"baseline": 48.4805, "holdOptic": 64.9147, "holdBrain": 47.0012, "off": [95.5416, 97.1014]},
            "walk.power_sustained_hz": {"baseline": 20.1091, "holdOptic": 33.5072, "holdBrain": 21.3426, "off": [49.1802, 50.5960]},
            "loom.GF_peak_hz": {"baseline": [43.5929, 46.4969], "holdOptic": [31.7754, 32.0792],
                                "holdBrain": [50.0089, 60.0354], "off": [27.9926, 29.0034]},
        },
        "caveats": {
            "holdKC/holdDN1 walk.*": "round 4 measured the holdKC / holdDN1 arms under the PRE-retirement GF damping "
                                     "(walk.power_max default 79.47 there): their walk.* rows are not comparable and are "
                                     "not listed; taste / smell / bitter are bit-stable across that change and are.",
            "loom.GF_peak_hz": "one of the two checks E.2 found scattering (10.0 Hz within holdBrain): a range, not a value.",
        }},
    "cpu": {
        "source": "docs/audits/receptor_integration.md E.4 (scripts/r5_attr_taste_cpu.py, device cpu, brain seeds 0,1,2)",
        "per_seed": {
            "taste.MN9_hz": {"off": [1.554839, 4.341760, 2.309808], "baseline": [5.090923, 4.315772, 2.360074],
                             "holdBrain": [1.554839, 4.341760, 2.309808], "holdOptic": [5.090923, 4.315772, 2.360074],
                             "holdBrainGlu": [5.090923, 4.315772, 2.360074], "holdBrainHis": [1.554839, 4.341760, 2.309808],
                             "holdKC": [5.090923, 4.315772, 2.360074], "holdDN1": [5.090923, 4.315772, 2.360074]},
            "smell.KC_active": {"off": [1079, 427, 1178], "baseline": [486, 412, 543], "holdBrain": [1079, 427, 1178],
                                "holdOptic": [486, 412, 543], "holdBrainGlu": [1079, 427, 1178],
                                "holdBrainHis": [486, 412, 543], "holdKC": [1225, 464, 1006], "holdDN1": [525, 433, 540]},
            "smell.PN_hz": {"off": [11.5126, 2.4370, 4.9003], "baseline": [3.6941, 2.9017, 4.2780],
                            "holdBrain": [11.5126, 2.4370, 4.9003], "holdOptic": [3.6941, 2.9017, 4.2780],
                            "holdBrainGlu": [11.5126, 2.4370, 4.9003], "holdBrainHis": [3.6941, 2.9017, 4.2780],
                            "holdKC": [9.9661, 2.9307, 8.1874], "holdDN1": [3.9050, 2.4012, 4.2898]}},
        "dissociation": {"lesions": ["holdBrainGlu", "holdBrainHis"], "checks": ["smell.KC_active", "taste.MN9_hz"]}},
}
# The entry counts each hold table must change vs sign(W.data) on the shipped cache (E.0).
EXPECTED_ENTRIES = {"baseline": 48295, "holdBrain": 44463, "holdOptic": 3832, "holdBrainGlu": 44586,
                    "holdBrainHis": 48172, "holdKC": 45461, "holdDN1": 47420, "off": 0}


def load_manifest(manifest) -> dict:
    """A manifest from a dict, a builtin name ('holds', 'holds_cpu'), a JSON / YAML path, or a literal JSON string."""
    if isinstance(manifest, dict):
        man = copy.deepcopy(manifest)
    elif isinstance(manifest, str) and manifest in BUILTIN_MANIFESTS:
        man = copy.deepcopy(BUILTIN_MANIFESTS[manifest])
        man["builtin"] = manifest                      # so `plan` can name it on the command line instead of embedding it
    elif isinstance(manifest, (str, Path)) and str(manifest).strip().startswith("{"):
        man = json.loads(str(manifest))
    else:
        path = Path(manifest)
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            import yaml                                      # optional; JSON manifests need nothing
            man = yaml.safe_load(text)
        else:
            man = json.loads(text)
        man.setdefault("name", path.stem)
        man["file"] = str(path)
    man.setdefault("name", "manifest")
    man.setdefault("sections", DEFAULT_SECTIONS)
    man.setdefault("lesions", [])
    man.setdefault("probes", [])
    man.setdefault("baseline", {"id": "baseline", "kind": "none"})
    if isinstance(man["baseline"], str):
        man["baseline"] = {"id": man["baseline"], "kind": "none"}
    return man


@dataclass
class Lesion:
    """One lesion set: what is removed / held, and how it is applied (never by editing the model)."""
    id: str
    kind: str = "none"
    spec: object = None
    within: object = None
    group: str | None = None                 # hold_table shortcut: a scripts/build_hold_tables.py group
    scope: str = "lif"                       # weight kinds: 'lif' (brain._shaped_weights) | 'graph' (also c.W: reaches the optic lobe)
    receptor_model: str | None = None        # the benchmark flag this arm runs under ('sign' / 'off' / 'default' / None = unchanged)
    receptor_net_rule: str | None = None
    note: str = ""

    @classmethod
    def of(cls, d) -> "Lesion":
        if isinstance(d, Lesion):
            return d
        d = dict(d)
        known = {f.name for f in dataclasses.fields(cls)}
        extra = {k: v for k, v in d.items() if k not in known}
        les = cls(**{k: v for k, v in d.items() if k in known})
        if extra:
            les.note = (les.note + " " + json.dumps(extra)).strip()
        if les.kind not in KINDS:
            raise ValueError(f"lesion {les.id!r}: unknown kind {les.kind!r}; choose from {KINDS}")
        return les


def lesions_of(man: dict, *, with_baseline: bool = True) -> list[Lesion]:
    """[baseline] + the manifest's lesions, as Lesion objects."""
    out = [Lesion.of(man["baseline"])] if with_baseline else []
    return out + [Lesion.of(d) for d in man["lesions"]]


def find_lesion(man: dict, lesion_id: str) -> Lesion:
    for les in lesions_of(man):
        if les.id == lesion_id:
            return les
    raise SystemExit(f"no lesion {lesion_id!r} in the manifest; have " + ", ".join(l.id for l in lesions_of(man)))


# ------------------------------------------------------------------------------------------------ receptor tables
def _file_md5(path) -> str | None:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except OSError:
        return None


def _rel(path) -> str:
    """`path` relative to the repository root with forward slashes (what a cluster command line needs); the absolute
    path when the two are on different drives (a Windows temp directory in the tests)."""
    try:
        return os.path.relpath(str(path), ROOT).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def hold_table(les: Lesion, *, build: bool = True) -> str:
    """The receptors_by_type.csv a hold_table lesion runs under, built by scripts/build_hold_tables.py when missing
    (the cluster ships no out/: every job rebuilds its own copy, atomically, exactly as the round-5 batch did)."""
    group = les.group
    spec = les.spec if isinstance(les.spec, str) else None
    if spec is None and group is None:
        raise ValueError(f"lesion {les.id!r}: kind hold_table needs a table path in `spec` or a builder `group`")
    path = spec or str(ROOT / "out" / f"receptors_hold{group}.csv")
    if not os.path.isabs(path):
        path = str(ROOT / path)
    if os.path.isfile(path) or not build:
        return path
    m = re.match(r"receptors_hold([A-Za-z0-9]+)\.csv$", os.path.basename(path))
    group = group or (m.group(1) if m else None)
    if group is None:
        raise SystemExit(f"lesion {les.id!r}: {path} does not exist and its name names no build_hold_tables group")
    sys.path.insert(0, str(SCRIPTS))
    import build_hold_tables as bht                                    # the writer of the shipped hold tables
    header, table, n = bht.hold_table(group)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bht.write_atomic(path, header + table.to_csv(index=False, lineterminator="\n"))
    return path


def tier_hold_table(tier: str, out_dir, table_path=None) -> tuple[str, int]:
    """Write a receptor table with every row the lookup decided at `tier` held at the presynaptic prior
    (`fast_sign_abs = connectome.NT_SIGN[transmitter]`, gain class none, `fast_net_abs = 'held'`), the way
    scripts/build_hold_tables.py holds a type group. Returns (path, rows held)."""
    from .. import connectome as cn
    src = Path(table_path or cn.RECEPTOR_TABLE)
    if tier not in cn.RECEPTOR_TIERS:
        raise ValueError(f"unknown receptor tier {tier!r}; choose from {cn.RECEPTOR_TIERS}")
    header = "".join(l for l in open(src, encoding="utf-8") if l.startswith("#"))
    t = pd.read_csv(src, comment="#")
    prior = t.transmitter.map(cn.NT_SIGN).astype(float)
    sel = (t.tier.astype(str) == tier) & (t.fast_sign_abs.astype(float) != prior)
    t.loc[sel, "fast_sign_abs"] = prior[sel].astype(int)
    t.loc[sel, "fast_gain_class_abs"] = "none"
    t.loc[sel, "fast_net_abs"] = "held"
    header += (f"# flyverse/interp/lesion.py: {int(sel.sum())} rows decided at tier {tier} held at NT_SIGN "
               f"(fast_sign_abs = NT_SIGN[transmitter], gain none, fast_net_abs held)\n")
    path = str(Path(out_dir) / f"receptors_tier_{tier}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + t.to_csv(index=False, lineterminator="\n"))
    os.replace(tmp, path)
    return path, int(sel.sum())


def changed_entries(c, table_path=None, net_rule: str = "abs") -> int:
    """How many stored entries a receptor table changes vs sign(W.data) -- E.0's column, per table."""
    from .. import connectome as cn
    r = cn.receptor_signs(c, table_path=table_path, net_rule=net_rule)
    return int((r.fast_sign != np.sign(c.W.data)).sum())


# ------------------------------------------------------------------------------------------------ applying a lesion
def pre_indices(c, les: Lesion) -> np.ndarray:
    """The presynaptic cells a weight-mask lesion silences (model indices)."""
    if les.kind in ("population", "types"):
        idx = common.resolve(c, les.spec)
    elif les.kind == "module":
        idx = common.resolve(c, f"module={les.spec}")
    elif les.kind == "transmitter":
        idx = common.resolve(c, f"nt={les.spec}")
    else:
        return np.zeros(0, np.int64)
    if les.within is not None:
        idx = np.intersect1d(idx, common.resolve(c, les.within))
    return np.asarray(idx, np.int64)


def overrides(les: Lesion, *, build_tables: bool = True, out_dir=None) -> tuple[dict, dict, dict]:
    """(LIFParams overrides, OpticParams overrides, a record of what was applied) for one lesion. Nothing in
    flyverse/ is edited: a hold table is a receptor_table path, the rest are parameter values."""
    lif, optic, rec = {}, {}, {"id": les.id, "kind": les.kind, "note": les.note}
    if les.kind == "hold_table":
        path = hold_table(les, build=build_tables)
        lif["receptor_table"] = path
        rec.update({"receptor_table": path, "receptor_table_md5": _file_md5(path), "group": les.group,
                    "spec": _rel(path)})
    elif les.kind == "receptor_tier":
        path, n = tier_hold_table(str(les.spec), out_dir or (ROOT / "out"))
        lif["receptor_table"] = path
        rec.update({"receptor_table": path, "receptor_table_md5": _file_md5(path), "rows_held": n,
                    "spec": f"tier={les.spec}"})
    elif les.kind == "lif":
        lif.update(dict(les.spec or {}))
        rec["spec"] = json.dumps(to_jsonable(les.spec))
    elif les.kind == "optic":
        optic.update(dict(les.spec or {}))
        rec["spec"] = json.dumps(to_jsonable(les.spec))
    elif les.kind == "pair_gain":
        from .. import optic as optic_mod
        pre_re, post_re, factor = les.spec
        out, found = [], 0
        for a, b, g in optic_mod.DEFAULT_PAIR_GAIN:
            if (a, b) == (pre_re, post_re):
                found += 1
                out.append((a, b, float(factor)))
            else:
                out.append((a, b, g))
        if not found:
            out.append((pre_re, post_re, float(factor)))
        optic["pair_gain"] = out
        rec["spec"] = f"{pre_re} -> {post_re} x {factor}" + ("" if found else " (added)")
    elif les.kind in WEIGHT_KINDS:
        rec["spec"] = common.spec_repr(les.spec) + ("" if les.within is None else f" within {common.spec_repr(les.within)}")
        rec["scope"] = les.scope
    else:                                                              # 'none': the baseline arm
        rec["spec"] = ""
    return lif, optic, rec


def mask_matrix(W, pre_idx):
    """`W` with the presynaptic columns `pre_idx` zeroed (entries kept as explicit zeros: the sparsity pattern, and
    hence every receptor / count array aligned with it, is unchanged)."""
    import scipy.sparse as sp
    coo = sp.csr_matrix(W).tocoo()
    mask = np.zeros(coo.shape[1], bool)
    mask[np.asarray(pre_idx, np.int64)] = True
    coo.data = coo.data.copy()
    coo.data[mask[coo.col]] = 0.0
    return coo.tocsr()


@contextlib.contextmanager
def weight_mask(pre_idx, *, c=None, scope: str = "lif"):
    """Silence the output of `pre_idx` for the duration: scope 'lif' wraps brain._shaped_weights (the spiking model
    only -- optic.OpticLobe reads c.W directly and is NOT masked); scope 'graph' zeroes the same columns of c.W too,
    so the rate optic lobe and the fan-in totals see the lesion as well. The wrapper also copies c.W before shaping
    (scripts/retire_measures.py's `_protected`: the gain loops write into W.data)."""
    from .. import brain as brain_mod
    pre_idx = np.asarray(pre_idx, np.int64)
    orig = brain_mod._shaped_weights

    def shaped(cc, p, receptor=None):
        old = cc.W
        cc.W = old.copy()
        try:
            W = orig(cc, p, receptor)
        finally:
            cc.W = old
        return mask_matrix(W, pre_idx) if len(pre_idx) else W

    brain_mod._shaped_weights = shaped
    old_W = None
    if scope == "graph" and c is not None and len(pre_idx):
        old_W = c.W
        c.W = mask_matrix(old_W, pre_idx)
    try:
        yield
    finally:
        brain_mod._shaped_weights = orig
        if old_W is not None:
            c.W = old_W


def _device_name(dev) -> str | None:
    """The GPU model behind a realised torch device ('NVIDIA B200'), or None off the GPU."""
    try:
        import torch
        if dev is not None and getattr(dev, "type", str(dev)).startswith("cuda") and torch.cuda.is_available():
            return torch.cuda.get_device_name(dev)
    except Exception:  # noqa: BLE001 -- provenance must never break a run
        pass
    return None


@contextlib.contextmanager
def brain_probe(device=None, seed=None):
    """Record what the Brains of this job actually ran on (the REALISED device, backend flags, dt) and, when `device`
    / `seed` are given, inject them where the caller passed none -- read-only instrumentation of brain.Brain."""
    from .. import brain as brain_mod
    seen: dict = {}
    orig = brain_mod.Brain.__init__

    def init(self, c, params=None, *a, **kw):
        if device is not None and len(a) < 1 and kw.get("device") is None:
            kw["device"] = device
        if seed is not None and len(a) < 2 and "seed" not in kw:
            kw["seed"] = int(seed)
        orig(self, c, params, *a, **kw)
        this = {"device": str(getattr(self, "device", None)), "batch": int(getattr(self, "B", 1)),
                "dt_ms": float(getattr(getattr(self, "p", None), "dt", np.nan)),
                "event_driven": bool(getattr(self, "event_driven", False)),
                "cuda_kernels": bool(getattr(self, "cuda", False)),
                "cuda_sparse": getattr(self, "cuda_sparse", None),
                "metal": bool(getattr(self, "metal", False))}
        if not seen:
            seen.update(this)
            seen["device_name"] = _device_name(getattr(self, "device", None))
            seen["n_brains"], seen["variants"] = 0, []
        if this not in seen["variants"]:                  # a section builds its own Brain / FlyBrain: the backend flags
            seen["variants"].append(this)                 # of the first are not those of all (benchmark's taste vs walk)
        seen["n_brains"] += 1

    brain_mod.Brain.__init__ = init
    try:
        yield seen
    finally:
        brain_mod.Brain.__init__ = orig


def resolve_lesion(c, les: Lesion, *, out_dir=None, entries: bool = False, build_tables: bool = True) -> dict:
    """What this lesion resolves to on this connectome: bodies (weight masks), table + changed entries (holds)."""
    lif, optic, rec = overrides(les, build_tables=build_tables, out_dir=out_dir)
    rec["lif_overrides"] = to_jsonable(lif)
    rec["optic_overrides"] = to_jsonable(optic)
    rec["receptor_model"] = les.receptor_model
    rec["receptor_net_rule"] = les.receptor_net_rule
    if les.kind in WEIGHT_KINDS:
        idx = pre_indices(c, les)
        Wc = c.W.tocsr()[:, idx]
        rec.update({"n_bodies": int(len(idx)),
                    "bodies": body_str(c.neurons.bodyId.to_numpy()[idx]) if len(idx) <= MAX_BODIES_RECORDED else None,
                    "n_entries_removed": int(Wc.nnz), "synapses_removed": float(np.abs(Wc.data).sum()),
                    "types": sorted(set(c.neurons.type.fillna("").to_numpy()[idx]))[:50],
                    "applies_to": ("brain._shaped_weights (the spiking model); optic.OpticLobe reads c.W directly and is "
                                   "NOT masked under scope 'lif'" if les.scope == "lif" else
                                   "brain._shaped_weights and c.W (the rate optic lobe and the fan-in totals too)"),
                    "fan_in_note": "Brain normalises on the masked matrix, so the removed entries also leave the "
                                   "postsynaptic fan-in totals (this differs from screen.ablate, which keeps them)"})
    if entries and les.kind in ("hold_table", "receptor_tier", "none"):
        try:                                                  # E.0's column: what this table changes vs sign(W.data)
            rec["entries_changed_vs_sign_W"] = changed_entries(c, rec.get("receptor_table"),
                                                               les.receptor_net_rule or "abs")
        except Exception as e:                                # noqa: BLE001 -- a graph the receptor table cannot type
            rec["entries_changed_vs_sign_W"] = None
            rec["entries_error"] = repr(e)
    if entries and les.kind == "lif" and dict(les.spec or {}).get("receptor_model", "x") is None:
        rec["entries_changed_vs_sign_W"] = 0
    return rec


# ------------------------------------------------------------------------------------------------ one job
def _bench_args(bm_defaults, *, sections, seeds, eager, cache_dir, les: Lesion, lif_over: dict):
    """scripts/benchmark.py's argparse namespace for this arm (retire_measures.bench_args keeps it in sync with
    benchmark.py's own defaults)."""
    kw = dict(sections=sections, seeds=seeds, eager=eager, cache_dir=cache_dir)
    model = les.receptor_model
    if model is None and les.kind == "lif" and dict(les.spec or {}).get("receptor_model", "x") is None:
        model = "off"                       # receptor_model None must reach benchmark.Context as the flag, not a value
    if model is not None:
        kw["receptor_model"] = model
        kw["receptor_net_rule"] = les.receptor_net_rule or "abs"
    if "receptor_table" in lif_over:
        kw["receptor_table"] = lif_over["receptor_table"]
    return bm_defaults(**kw)


def _context(bm, args, lif_over: dict, optic_over: dict):
    """benchmark.Context with this lesion's LIFParams / OpticParams overrides (the retire_measures pattern)."""
    lif_over = {k: v for k, v in lif_over.items() if k != "receptor_table"}          # that one goes through the flag

    class LesionContext(bm.Context):
        @property
        def has_overrides(self):
            return bool(lif_over) or bool(optic_over) or bm.Context.has_overrides.fget(self)

        def _apply_lif(self, p):
            super()._apply_lif(p)
            for k, v in lif_over.items():
                setattr(p, k, copy.deepcopy(v))
            return p

        def _apply_optic(self, op):
            super()._apply_optic(op)
            for k, v in optic_over.items():
                setattr(op, k, copy.deepcopy(v))
            return op

    return LesionContext(args)


def _dig(d, path: str):
    """'summary.LC11.diff_max_over_cells_mean_mv' -> the value in a nested dict / list (None when absent)."""
    cur = d
    for part in str(path).split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.lstrip("-").isdigit():
            cur = cur[int(part)]
        else:
            return None
    return cur


def run_probe(probe: dict, les: Lesion, rec: dict, *, out_dir, seed, replicate, quiet=False) -> dict:
    """One probe CLI: `cmd` with {out} / {seed} / {lesion} / {replicate} / {table} / {receptor_flags} substituted,
    then `read` (a dotted path) pulled out of the JSON it wrote."""
    out_dir = Path(out_dir)
    out = out_dir / f"probe_{probe['id']}_{les.id}_r{replicate}.json"
    table = rec.get("receptor_table", "")
    flags = ""
    if les.receptor_model == "off" or rec.get("lif_overrides", {}).get("receptor_model", "x") is None:
        flags = "--receptor-model off"
    elif table:
        flags = f"--receptor-model {les.receptor_model or 'sign'} --receptor-net-rule {les.receptor_net_rule or 'abs'} --receptor-table {table}"
    cmd = probe["cmd"].format(out=str(out), seed=seed, lesion=les.id, replicate=replicate, table=table, receptor_flags=flags)
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, cwd=str(ROOT), capture_output=True, text=True)
    value, payload = None, None
    if os.path.isfile(out):
        try:
            with open(out, encoding="utf-8") as f:
                payload = json.load(f)
            value = _dig(payload, probe.get("read", ""))
        except (OSError, json.JSONDecodeError) as e:                    # a probe that wrote nothing readable
            payload = {"error": repr(e)}
    if not quiet:
        print(f"probe {probe['id']:20s} {les.id:12s} r{replicate}: {probe.get('read', '')} = {value}  ({time.time() - t0:.0f} s)", flush=True)
    return {"id": probe["id"], "cmd": cmd, "read": probe.get("read"), "value": value, "exit_code": r.returncode,
            "json": str(out), "seconds": round(time.time() - t0, 1),
            "stderr_tail": (r.stderr or "")[-400:] if r.returncode else ""}


def run_job(manifest, lesion_id: str, *, out_dir, replicate: int = 0, sections=None, device=None, seeds="0,1,2",
            brain_seed=None, probes=(), eager: bool = False, cache_dir=None, quiet: bool = False) -> dict:
    """One arm of the matrix, in this process: the benchmark sections under the lesion, then the probes. Writes
    <out_dir>/<lesion>_r<replicate>.json (JOB_SCHEMA) and returns it."""
    man = load_manifest(manifest)
    les = find_lesion(man, lesion_id)
    sections = sections or man.get("sections") or DEFAULT_SECTIONS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if brain_seed is None and man.get("brain_seeds"):
        seeds_list = man["brain_seeds"]
        brain_seed = int(seeds_list[replicate % len(seeds_list)])
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(SCRIPTS))
    import benchmark as bm                                              # the suite: sections, references, statuses
    import retire_measures as rm                                        # bench_args: benchmark.py's own flag defaults
    from .. import brain as brain_mod

    lif_over, optic_over, rec = overrides(les, out_dir=out_dir)
    g = man.get("global") or {}                                         # --lif / --optic: applied to EVERY arm
    if g:
        lif_over = dict(g.get("lif", {}), **lif_over)
        optic_over = dict(g.get("optic", {}), **optic_over)
        rec["global_overrides"] = to_jsonable(g)
    args = _bench_args(rm.bench_args, sections=sections, seeds=seeds, eager=eager, cache_dir=cache_dir,
                       les=les, lif_over=lif_over)
    t_all = time.time()
    ctx = _context(bm, args, lif_over, optic_over)
    ctx.c.reference._norm_cache.clear()                                 # several arms in one process must not share it
    rec.update(resolve_lesion(ctx.c, les, out_dir=out_dir, entries=True, build_tables=False))
    rec["lif_overrides"], rec["optic_overrides"] = to_jsonable(lif_over), to_jsonable(optic_over)
    pre_idx = pre_indices(ctx.c, les)
    lif, op = ctx.lif(), ctx.optic_params()
    if not quiet:
        print(f"lesion {les.id} (kind {les.kind}) replicate {replicate}: {json.dumps(to_jsonable(rec), default=str)[:600]}", flush=True)
        print(f"  receptor_model {lif.receptor_model!r} net_rule {lif.receptor_net_rule!r} table {lif.receptor_table!r}; "
              f"sections {sections}; device request {device}; brain seed {brain_seed}", flush=True)
    with weight_mask(pre_idx, c=ctx.c, scope=les.scope), brain_probe(device=device, seed=brain_seed) as seen:
        for letter, name, fn in bm.select_sections(sections):
            print(f"\n=== [{letter or '-'}] {name}", flush=True)
            t0 = time.time()
            try:
                ctx.results[name] = fn(ctx)
            except Exception as e:                                      # one broken section must not hide the others
                import traceback
                traceback.print_exc()
                ctx.results[name] = {"error": repr(e)}
                for key in bm.REFERENCES:
                    if key.split(".")[0] == name or (name == "walk" and key.split(".")[0] in ("walk", "loom", "rotate")):
                        ctx.report(key, None)
            ctx.runtime[name] = time.time() - t0
            print(f"    [{name}: {ctx.runtime[name]:.0f} s]", flush=True)
    probe_out = [run_probe(p, les, rec, out_dir=out_dir, seed=brain_seed if brain_seed is not None else replicate,
                           replicate=replicate, quiet=quiet)
                 for p in list(man.get("probes", [])) + list(probes or [])]
    shim = _device_shim(seen, lif)
    prov = common.provenance(ctx.c, lif, op, fb=shim, device=device,
                             seeds={"brain": brain_seed, "benchmark_seeds": ctx.seeds},
                             stimulus={"protocol": f"benchmark.py sections {sections}", "params": {"sections": sections},
                                       "control": man["baseline"]["id"]},
                             cache_dir=ctx.cache_dir, replicate_unit="runs")
    prov["execution"]["backend"].update({k: v for k, v in seen.items() if k not in ("device", "batch", "dt_ms", "device_name")})
    # benchmark.py's own label for its configuration (the string it writes into its JSONs). It describes the flags
    # ctx.sim() hands the room demo, NOT what every section builds: with rest/taste/smell/walk/bitter no Brain is
    # built with them, and `variants` below is the record of what the Brains of this job actually got.
    prov["execution"]["backend"]["benchmark_backend"] = ("eager torch" if eager else
                                                         "native (cuda_kernels, cuda_graphs, event_driven, warp)")
    prov["execution"]["backend"]["benchmark_backend_is"] = ("benchmark.py's configuration label (--eager or not); the "
                                                            "realised per-Brain flags are in `variants`")
    prov["execution"]["device_name"] = seen.get("device_name")         # the shim carries a device STRING, so
    prov["execution"]["backend"]["flags_are_of"] = ("the first Brain the job built; `variants` lists every distinct "
                                                    "backend configuration the sections used")
    job = {"schema": JOB_SCHEMA, "tool": TOOL, "manifest": man.get("name"), "manifest_source": man.get("file"),
           "lesion": rec, "replicate": int(replicate), "brain_seed": brain_seed, "sections": sections,
           "checks": ctx.checks, "results": ctx.results, "probes": probe_out, "provenance": prov,
           "runtime_s": ctx.runtime, "total_runtime_s": time.time() - t_all,
           "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "generator": "python " + " ".join(shlex.quote(a) for a in sys.argv)}
    path = out_dir / f"{les.id}_r{replicate}.json"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(to_jsonable(job), f, indent=1)
    if not quiet:
        print("\n" + bm.summary_table(ctx))
        print(f"\nwrote {path}  ({job['total_runtime_s'] / 60:.1f} min; realised device {seen.get('device')})", flush=True)
    job["file"] = str(path)
    return job


def _device_shim(seen: dict, lif):
    """What common.execution_record reads, from the Brains this job actually built (never the request)."""
    from types import SimpleNamespace
    b = SimpleNamespace(device=seen.get("device"), B=seen.get("batch", 1), p=SimpleNamespace(dt=seen.get("dt_ms", np.nan)),
                        event_driven=seen.get("event_driven", False), cuda=seen.get("cuda_kernels", False),
                        cuda_sparse=seen.get("cuda_sparse"), metal=seen.get("metal", False))
    return SimpleNamespace(brain=b, cuda_graphs=False)


# ------------------------------------------------------------------------------------------------ plan (the batch)
def manifest_argument(man: dict) -> str:
    """What every job's `--manifest` should say: the builtin's name when the manifest is one unchanged ('holds'), the
    file path when it came from a file the cluster copy will have (anything tracked under the repo, i.e. not out/),
    else the manifest itself as JSON -- out/ is not shipped, so an out/ manifest has to travel inside the command."""
    name = man.get("builtin")
    if name in BUILTIN_MANIFESTS:
        a, b = copy.deepcopy(man), load_manifest(name)
        a.pop("builtin", None); b.pop("builtin", None)
        if a == b:
            return name
    src = man.get("file")
    if src:
        rel = _rel(src)
        if not rel.startswith("..") and not rel.startswith("out/") and os.path.isfile(os.path.join(ROOT, rel)):
            return rel
    return json.dumps(man)


def job_command(manifest_arg: str, les_id: str, replicate: int, *, out_dir: str, sections: str, device=None,
                seeds="0,1,2", eager=False) -> str:
    """The `scripts/interp_lesion.py run` line of one job, with its own stdout file (one per job: the batch runs
    them concurrently). It makes its own output directory first: `out/` is git-ignored, so the cluster's run copy of
    the checkout does NOT have it and the redirection would fail before python starts (les-holds-80f423: 12 of 12
    jobs dead in 34 s with 'out/les_holds/off_r1.txt: No such file or directory')."""
    stem = f"{out_dir}/{les_id}_r{replicate}"
    cmd = (f"python scripts/interp_lesion.py run --manifest {shlex.quote(manifest_arg)} --one {les_id} "
           f"--replicate {replicate} --out {out_dir} --sections {sections} --seeds {seeds}")
    if device:
        cmd += f" --device {device}"
    if eager:
        cmd += " --eager"
    return (f"mkdir -p {out_dir} && python -c 'import torch; assert torch.cuda.is_available()' && "
            f"{cmd} > {stem}.txt; cat {stem}.txt")


def plan(manifest, *, out_dir, replicates: int = MIN_REPLICATES, sections=None, minutes: int = 60, name=None,
         device=None, seeds="0,1,2", eager: bool = False, c=None, quiet: bool = False) -> dict:
    """Resolve the manifest on the connectome and write <out_dir>/manifest.resolved.json + <out_dir>/batch.sh (ONE
    cluster_run.py call: one job per lesion x replicate plus `replicates` baseline jobs, --fetch <out_dir>/)."""
    man = load_manifest(manifest)
    sections = sections or man.get("sections") or DEFAULT_SECTIONS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_out = _rel(out_dir)
    if c is None:
        from .. import connectome as cn
        c = cn.load(verbose=False)
    resolved = [resolve_lesion(c, les, out_dir=out_dir, entries=True) for les in lesions_of(man)]
    manifest_arg = manifest_argument(man)
    jobs = []
    for les in lesions_of(man):
        for k in range(int(replicates)):
            jobs.append({"lesion": les.id, "replicate": k,
                         "command": job_command(manifest_arg, les.id, k, out_dir=rel_out, sections=sections,
                                                device=device, seeds=seeds, eager=eager),
                         "json": f"{rel_out}/{les.id}_r{k}.json"})
    short = name or f"les-{man.get('name', 'run')}"
    log = f"out/{short}_cluster.log"
    call = (f"python scripts/cluster_run.py --name {short} --minutes {int(minutes)} \\\n  "
            + " \\\n  ".join(shlex.quote(j["command"]) for j in jobs)
            + f" \\\n  --fetch {rel_out}/")
    batch = ("#!/bin/sh\n"
             f"# generated by flyverse/interp/lesion.py plan ({man.get('name')}, {len(jobs)} jobs = "
             f"{len(resolved)} arms x {replicates} replicates) on {time.strftime('%Y-%m-%d %H:%M')}\n"
             f"# every number the matrix quotes comes from one of the {len(jobs)} JSONs in {rel_out}/.\n"
             f"set -e\nmkdir -p {rel_out} out\n"
             f"# a second run of this batch keeps the first run's console log instead of overwriting it\n"
             f'if [ -f {log} ]; then mv {log} "{log[:-4]}.$(date +%Y%m%dT%H%M%S).log"; fi\n'
             f"{call} 2>&1 | tee {log}\n")
    (out_dir / "batch.sh").write_text(batch, encoding="utf-8", newline="\n")
    man_resolved = {"manifest": man, "sections": sections, "replicates": int(replicates), "out_dir": rel_out,
                    "lesions": resolved, "jobs": jobs, "cluster_log": log,
                    "connectome": common.connectome_fingerprint(c),
                    "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(out_dir / "manifest.resolved.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(to_jsonable(man_resolved), f, indent=1)
    if not quiet:
        print(f"{len(jobs)} job(s) planned -> {out_dir / 'batch.sh'} (fetch {rel_out}/, console log {log})")
    return man_resolved


# ------------------------------------------------------------------------------------------------ analyse
def load_jobs(out_dir) -> list[dict]:
    """Every finished job JSON in a run directory (JOB_SCHEMA), oldest first."""
    jobs = []
    for path in sorted(glob.glob(str(Path(out_dir) / "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("schema") == JOB_SCHEMA:
            d["file"] = path
            jobs.append(d)
    return jobs


def _arms(jobs: list[dict], checks="all") -> dict:
    """{check: {lesion: [values in replicate order]}} plus the status / criterion of each (check, lesion) and the
    brain seed of each replicate (arms that share their seed list are compared seed by seed)."""
    want = None if checks in (None, "all") else set(checks.split(",") if isinstance(checks, str) else checks)
    values: dict = {}
    status: dict = {}
    meta: dict = {}
    seeds: dict = {}
    for job in sorted(jobs, key=lambda j: (j["lesion"]["id"], j.get("replicate", 0))):
        seeds.setdefault(job["lesion"]["id"], []).append(job.get("brain_seed"))
        les = job["lesion"]["id"]
        for ch in job.get("checks", []):
            key = ch["key"]
            if want and key not in want:
                continue
            values.setdefault(key, {}).setdefault(les, []).append(ch["measured"])
            status.setdefault(key, {}).setdefault(les, []).append(ch["status"])
            meta.setdefault(key, {"criterion": ch.get("criterion"), "reference": ch.get("reference"),
                                  "session": ch.get("session")})
        for pr in job.get("probes", []):
            key = f"probe.{pr['id']}.{pr.get('read')}"
            if want and key not in want:
                continue
            values.setdefault(key, {}).setdefault(les, []).append(pr.get("value"))
            status.setdefault(key, {}).setdefault(les, []).append("PROBE")
            meta.setdefault(key, {"criterion": None, "reference": None, "session": None})
    paired = (len(seeds) > 1 and all(None not in v for v in seeds.values())
              and len({tuple(v) for v in seeds.values()}) == 1)
    return {"values": values, "status": status, "meta": meta, "seeds": seeds, "paired": bool(paired)}


def _num(values) -> np.ndarray:
    out = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.asarray(out, dtype=np.float64)


def matrix(jobs: list[dict], *, baseline: str = "baseline", checks="all", tolerance: float = 0.0,
           rel_tolerance: float = 0.0) -> pd.DataFrame:
    """The check x lesion table: baseline, value, delta, replicate values / sd / n, whether the check is bit-identical
    within every arm (or, for seed-paired arms, seed by seed), whether this lesion moved it, and the status change.

    Three criteria, in order. (1) The check is **bit-identical inside every arm** (nine of E.2's sixteen are, because
    `sec_walk` / `sec_motion` draw no RNG and the taste / smell / bitter draws are seed-locked): a lesion moved it iff
    the value differs at all. (2) The arms are **paired by brain seed** (the CPU protocol: every arm ran seeds
    0, 1, 2): moved iff every paired difference exceeds the threshold, not moved iff every paired difference is at or
    below it -- the E.4 criterion ('= off, every digit'), which assumes the protocol is deterministic given the seed
    (it is on the CPU: E.4's rerun reproduced every digit of the first run). (3) Otherwise the scatter rule: moved
    above twice the pooled replicate sd, not moved at or below one pooled sd, 'unclear' in between -- and
    **'underpowered', whatever the numbers, when either arm has fewer than MIN_REPLICATES runs** (docs/INTERP.md 2.4;
    an arm of two draws on a check that scatters is quoted, not called, and cannot enter a dissociation. E.2 refused
    to attribute `rotate.DNp20_flip_hz` for exactly this reason: its holdBrain draws span every condition).

    The threshold is max(`tolerance`, `rel_tolerance` x |baseline|) -- 0, i.e. bit-identity, unless a caller asks for
    a size (`rel_tolerance` 0.05 ignores moves under 5 % of the baseline value)."""
    arms = _arms(jobs, checks)
    lesion_meta = {j["lesion"]["id"]: j["lesion"] for j in jobs}
    rows = []
    for check, per_lesion in arms["values"].items():
        base_vals = _num(per_lesion.get(baseline, []))
        bit = all(np.nanmax(_num(v)) == np.nanmin(_num(v)) for v in per_lesion.values() if len(v) > 1) and \
            any(len(v) > 1 for v in per_lesion.values())
        for les, vals in per_lesion.items():
            v = _num(vals)
            sd = float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan")
            base_sd = float(np.nanstd(base_vals, ddof=1)) if len(base_vals) > 1 else float("nan")
            mean = float(np.nanmean(v)) if len(v) else float("nan")
            base_mean = float(np.nanmean(base_vals)) if len(base_vals) else float("nan")
            delta = mean - base_mean
            pooled = float(np.sqrt(np.nanmean([x ** 2 for x in (sd, base_sd) if np.isfinite(x)]))) \
                if any(np.isfinite(x) for x in (sd, base_sd)) else float("nan")
            paired = arms["paired"] and len(v) == len(base_vals) and len(v) > 0
            pdelta = (v - base_vals) if paired else np.zeros(0)
            thr = max(float(tolerance), float(rel_tolerance) * abs(base_mean) if np.isfinite(base_mean) else 0.0)
            if les == baseline:
                moved, criterion = "baseline", ""
            elif bit:
                moved = "yes" if (np.isfinite(delta) and abs(delta) > thr) else "no"
                criterion = f"bit-identical within every arm; threshold {thr:g}"
            elif paired:
                criterion = f"paired by brain seed {tuple(arms['seeds'][les])}; threshold {thr:g}"
                if np.all(np.abs(pdelta) <= thr):
                    moved = "no"
                elif np.all(np.abs(pdelta) > thr):
                    moved = "yes"
                else:
                    moved = "unclear"
            elif not np.isfinite(pooled):
                moved, criterion = "unclear", "one run per arm: no scatter"
            elif min(len(v), len(base_vals)) < MIN_REPLICATES:
                moved = "underpowered"
                criterion = (f"the check scatters (pooled sd {pooled:.4g}) and this arm has "
                             f"{min(len(v), len(base_vals))} run(s) < {MIN_REPLICATES}: |delta| {abs(delta):.4g} is "
                             f"quoted, not called")
            elif abs(delta) > max(2 * pooled, thr):
                moved, criterion = "yes", "|delta| > 2 x pooled replicate sd"
            elif abs(delta) <= max(pooled, thr):
                moved, criterion = "no", "|delta| <= pooled replicate sd"
            else:
                moved, criterion = "unclear", "between 1 and 2 pooled sd"
            st = arms["status"][check][les]
            base_st = arms["status"][check].get(baseline, [])
            rows.append({"check": check, "lesion_id": les, "lesion_kind": lesion_meta.get(les, {}).get("kind"),
                         "lesion_spec": lesion_meta.get(les, {}).get("spec"), "n_replicates": int(len(v)),
                         "value": mean, "replicate_values": [None if not np.isfinite(x) else x for x in v],
                         "replicate_sd": sd, "baseline": base_mean, "baseline_sd": base_sd,
                         "n_baseline": int(len(base_vals)), "delta": delta,
                         "paired_by_seed": bool(paired), "brain_seeds": list(arms["seeds"].get(les, [])),
                         "paired_deltas": [float(x) for x in pdelta] if paired else None,
                         "identical_to_baseline": bool(paired and np.all(pdelta == 0)) if paired else
                                                 (bool(bit and delta == 0) if les != baseline else None),
                         "bit_identical_within_arms": bool(bit), "pooled_sd": pooled, "moved": moved,
                         "moved_criterion": criterion,
                         "status": st[0] if len(set(st)) == 1 else "/".join(sorted(set(st))),
                         "baseline_status": base_st[0] if base_st and len(set(base_st)) == 1 else "/".join(sorted(set(base_st))),
                         "status_change": ("" if not base_st or (st[0] if len(set(st)) == 1 else None) ==
                                           (base_st[0] if len(set(base_st)) == 1 else None)
                                           else f"{base_st[0]} -> {st[0]}"),
                         "criterion": arms["meta"][check]["criterion"], "reference": arms["meta"][check]["reference"]})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["check", "lesion_id"]).reset_index(drop=True)
    return df


def dissociations(mat: pd.DataFrame, *, baseline: str = "baseline") -> pd.DataFrame:
    """Every (L1, L2) x (c1, c2) with c1 moved by L1 and not by L2 and c2 moved by L2 and not by L1 -- the
    double-dissociation extraction (bit-identity where the checks are deterministic, else twice the pooled scatter)."""
    if not len(mat):
        return pd.DataFrame()
    m = mat[mat.lesion_id != baseline]
    moved = {(r.check, r.lesion_id): r.moved for r in m.itertuples()}
    delta = {(r.check, r.lesion_id): r.delta for r in m.itertuples()}
    crit = {(r.check, r.lesion_id): r.moved_criterion for r in m.itertuples()}
    lesions = sorted(m.lesion_id.unique())
    checks = sorted(m.check.unique())
    seen, rows = set(), []
    for i, l1 in enumerate(lesions):
        for l2 in lesions[i + 1:]:
            for a, c1 in enumerate(checks):
                for c2 in checks[a + 1:]:
                    for (la, lb) in ((l1, l2), (l2, l1)):
                        if (moved.get((c1, la)) == "yes" and moved.get((c1, lb)) == "no" and
                                moved.get((c2, lb)) == "yes" and moved.get((c2, la)) == "no"):
                            key = (la, lb, c1, c2)
                            if key in seen:
                                continue
                            seen.add(key)
                            rows.append({"lesion_a": la, "lesion_b": lb, "check_a": c1, "check_b": c2,
                                         "delta_a_under_a": delta[(c1, la)], "delta_a_under_b": delta[(c1, lb)],
                                         "delta_b_under_a": delta[(c2, la)], "delta_b_under_b": delta[(c2, lb)],
                                         "criterion": crit.get((c1, la), "") or crit.get((c2, lb), "")})
    return pd.DataFrame(rows)


def _close(measured, expected, rtol: float = 5e-3) -> bool:
    if measured is None or not np.isfinite(measured):
        return False
    if isinstance(expected, (list, tuple)):
        lo, hi = min(expected), max(expected)
        span = max(hi - lo, abs(hi) * rtol)
        return lo - span * 0.5 <= measured <= hi + span * 0.5
    if float(expected) == 0.0:
        return abs(measured) <= 1e-6
    return abs(measured - float(expected)) <= max(rtol * abs(float(expected)), 1e-6)


def validate(mat: pd.DataFrame, jobs: list[dict], *, arm: str | None = None, baseline: str = "baseline") -> dict:
    """Compare the matrix to the audit's numbers (EXPECTED) row by row; `arm` 'gpu' / 'cpu'. The default picks 'cpu'
    only for the E.4 protocol -- every job on a cpu device AND every job carrying its own brain seed, since that
    table is quoted per seed (r5_attr_taste_cpu.py, seeds 0, 1, 2) -- and 'gpu' otherwise, where E.2's per-arm values
    are the reference. Returns VALIDATION['lesion'] filled in: every measured number beside the published one."""
    devices = sorted({(j.get("provenance", {}).get("execution", {}) or {}).get("device") for j in jobs} - {None})
    seeded = bool(jobs) and all(j.get("brain_seed") is not None for j in jobs)
    if arm is None:
        arm = "cpu" if (devices and seeded and all(str(d).startswith("cpu") for d in devices)) else "gpu"
    rows, per_seed_rows = [], []
    if arm == "gpu":
        for check, per_lesion in EXPECTED["gpu"]["checks"].items():
            sub = mat[mat.check == check]
            for les, exp in per_lesion.items():
                got = sub[sub.lesion_id == les]
                measured = float(got.value.iloc[0]) if len(got) else None
                rows.append({"check": check, "lesion": les, "expected": exp, "measured": measured,
                             "replicates": (got.replicate_values.iloc[0] if len(got) else None),
                             "n": int(got.n_replicates.iloc[0]) if len(got) else 0,
                             "status": "reproduced" if _close(measured, exp) else
                                       ("not run" if measured is None else "not reproduced")})
    else:
        for check, per_lesion in EXPECTED["cpu"]["per_seed"].items():
            sub = mat[mat.check == check]
            for les, exp in per_lesion.items():
                got = sub[sub.lesion_id == les]
                vals = list(got.replicate_values.iloc[0]) if len(got) else None
                ok = bool(vals) and len(vals) == len(exp) and all(_close(v, e, 2e-4) for v, e in zip(vals, exp))
                per_seed_rows.append({"check": check, "lesion": les, "expected_per_seed": exp, "measured_per_seed": vals,
                                      "status": "reproduced" if ok else ("not run" if not vals else "not reproduced")})
        rows = per_seed_rows
    # the entry counts of each table (E.0)
    entries = []
    for j in jobs:
        les = j["lesion"]
        if "entries_changed_vs_sign_W" in les:
            exp = EXPECTED_ENTRIES.get(les["id"])
            entries.append({"lesion": les["id"], "expected_entries": exp, "measured_entries": les["entries_changed_vs_sign_W"],
                            "table_md5": les.get("receptor_table_md5"),
                            "status": "reproduced" if exp is not None and les["entries_changed_vs_sign_W"] == exp
                                      else ("no reference" if exp is None else "not reproduced")})
    entries = pd.DataFrame(entries).drop_duplicates(subset=["lesion"]).to_dict("records") if entries else []
    statuses = [r["status"] for r in rows] + [e["status"] for e in entries if e["status"] != "no reference"]
    done = [s for s in statuses if s != "not run"]
    status = ("not run" if not done else "reproduced" if all(s == "reproduced" for s in done) else "not reproduced")
    out = dict(common.VALIDATION[TOOL])
    out.update({"arm": arm, "devices": devices, "source": EXPECTED[arm]["source"],
                "measured": {"checks": rows, "entries_changed": entries,
                             "n_reproduced": int(sum(s == "reproduced" for s in done)), "n_compared": len(done)},
                "status": status})
    if arm == "cpu":
        d = EXPECTED["cpu"]["dissociation"]
        out["measured"]["dissociation_target"] = d
    return out


# ------------------------------------------------------------------------------------------------ the tool
def lesion(manifest, *, out_dir, mode: str = "plan", replicates: int = MIN_REPLICATES, checks="all", probes=(),
           seeds=None, cluster: bool = False, minutes: int = 60, baseline: str = "baseline",
           c=None, one: str | None = None, replicate: int | None = None, device=None, sections=None,
           tolerance: float = 0.0, rel_tolerance: float = 0.0, brain_seed: int | None = None,
           eager: bool = False, cache_dir=None,
           name: str | None = None, quiet: bool = False) -> Result:
    """A manifest of lesion sets x the benchmark checks and probes -> a check x lesion delta matrix with scatter.

    `manifest`: a dict, a builtin name ('holds' = the round-4 / round-5 hold arms, 'holds_cpu' = the E.4 dissociation),
    a JSON / YAML path, or a JSON string -- {'sections', 'baseline', 'lesions': [{'id', 'kind', 'spec', ...}],
    'probes': [{'id', 'cmd', 'read'}]}. Kinds: 'population' / 'types' (a common.resolve spec, silenced by zeroing the
    presynaptic columns of the shaped weights in-process: the same effect as screen.ablate, no output), 'module',
    'transmitter', 'receptor_tier' (the entries the receptor lookup decided at that tier held at NT_SIGN, through a
    temporary table), 'hold_table' (a receptors_by_type.csv from scripts/build_hold_tables.py, passed as
    --receptor-table), 'lif' / 'optic' (parameter overrides), 'pair_gain' (an optic pair-gain factor), 'none' (the
    baseline arm). NOTHING in flyverse/ is edited by any of them.

    `mode`: 'plan' resolves the manifest on the connectome and writes <out_dir>/manifest.resolved.json and
    <out_dir>/batch.sh (ONE cluster_run.py call, one job per lesion x replicate plus `replicates` baseline jobs,
    --fetch <out_dir>/; `cluster` also submits it); 'run' executes one job in-process (`one` / `replicate`; without
    `one`, every arm x replicate serially -- the CPU protocol); 'analyse' reads the finished job JSONs in `out_dir`
    and builds the matrix, the Neurome `sensitivity` table and the double dissociations. `seeds` is the benchmark's
    --seeds (the demo sections); `brain_seed` (or the manifest's `brain_seeds`) sets the LIF Poisson seed per
    replicate; `device` is the request -- the JSON records the realised one. Validation: VALIDATION['lesion'].

    Tables: 'matrix' (check x lesion), 'sensitivity' (Neurome fields), 'dissociations', 'lesions', 'jobs'.
    """
    man = load_manifest(manifest)
    sections = sections or man.get("sections") or DEFAULT_SECTIONS
    seeds = seeds if seeds is not None else "0,1,2"
    if isinstance(seeds, (list, tuple)):
        seeds = ",".join(str(int(s)) for s in seeds)
    out_dir = Path(out_dir)
    if mode == "plan":
        planned = plan(man, out_dir=out_dir, replicates=replicates, sections=sections, minutes=minutes, name=name,
                       device=device, seeds=seeds, eager=eager, c=c, quiet=quiet)
        prov = common.provenance(c if c is not None else _load_c(), device=device, cache_dir=cache_dir)
        prov["execution"]["device"] = prov["execution"]["device"] or "not run (plan)"
        res = Result.new(TOOL, prov, summary={"mode": "plan", "n_jobs": len(planned["jobs"]), "out_dir": str(out_dir),
                                              "batch": str(out_dir / "batch.sh"), "cluster_log": planned["cluster_log"]})
        res.add_table("lesions", planned["lesions"])
        res.add_table("jobs", planned["jobs"])
        res.files = {"batch": str(out_dir / "batch.sh"), "resolved_manifest": str(out_dir / "manifest.resolved.json"),
                     "generator": "flyverse/interp/lesion.py::plan"}
        if cluster:
            log = ROOT / planned["cluster_log"]
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f"submitting {out_dir / 'batch.sh'} (log {log})", flush=True)
            r = subprocess.run(["sh", str(out_dir / "batch.sh")], cwd=str(ROOT))
            res.summary["cluster_exit_code"] = r.returncode
        return res
    if mode == "run":
        jobs = []
        if one is not None:
            jobs.append(run_job(man, one, out_dir=out_dir, replicate=int(replicate or 0), sections=sections,
                                device=device, seeds=seeds, brain_seed=brain_seed, probes=probes, eager=eager,
                                cache_dir=cache_dir, quiet=quiet))
        else:
            for les in lesions_of(man):
                for k in range(int(replicates)):
                    jobs.append(run_job(man, les.id, out_dir=out_dir, replicate=k, sections=sections, device=device,
                                        seeds=seeds, brain_seed=brain_seed, probes=probes, eager=eager,
                                        cache_dir=cache_dir, quiet=quiet))
        prov = copy.deepcopy(jobs[-1]["provenance"])
        res = Result.new(TOOL, prov, summary={"mode": "run", "n_jobs": len(jobs), "out_dir": str(out_dir),
                                              "lesions": sorted({j["lesion"]["id"] for j in jobs})})
        res.replicates = {"n": len(jobs), "unit": "runs",
                          "runs": [{"run_index": i, "lesion": j["lesion"]["id"], "replicate": j["replicate"],
                                    "seed": j.get("brain_seed"), "file": j.get("file"),
                                    "device": j["provenance"]["execution"]["device"]} for i, j in enumerate(jobs)],
                          "null": None}
        res.add_table("checks", [dict(ch, lesion_id=j["lesion"]["id"], replicate=j["replicate"])
                                 for j in jobs for ch in j["checks"]])
        res.add_table("lesions", [j["lesion"] for j in jobs])
        res.files = {"jobs": [j.get("file") for j in jobs], "generator": "flyverse/interp/lesion.py::run_job"}
        return res
    if mode != "analyse":
        raise ValueError(f"unknown mode {mode!r}; choose from plan, run, analyse")

    jobs = load_jobs(out_dir)
    if not jobs:
        raise SystemExit(f"no {JOB_SCHEMA} job files in {out_dir}")
    mat = matrix(jobs, baseline=baseline, checks=checks, tolerance=tolerance, rel_tolerance=rel_tolerance)
    dis = dissociations(mat, baseline=baseline)
    prov = copy.deepcopy(jobs[0]["provenance"])
    devices = sorted({j["provenance"]["execution"]["device"] for j in jobs})
    prov["execution"]["device"] = devices[0] if len(devices) == 1 else "; ".join(str(d) for d in devices)
    prov["execution"]["devices_per_job"] = devices
    prov["execution"]["replicate_unit"] = "runs"
    if str(prov.get("flyverse_commit", {}).get("commit")) in ("unknown", "None", ""):
        # a cluster run copy is a file tree, not a git checkout, so the JOB could not record a commit: substitute the
        # analysing checkout's (it is the tree cluster_run.py shipped) and say so rather than leave 'unknown' standing.
        prov["flyverse_commit"] = dict(common.git_state(),
                                       note="recorded on the analysing checkout: the cluster run copy is not a git "
                                            "checkout and every job recorded 'unknown'; the shipped file list is in "
                                            "the batch's console log",
                                       job_commit=jobs[0].get("provenance", {}).get("flyverse_commit", {}).get("commit"))
    res = Result.new(TOOL, prov)
    res.replicates = {"n": int(mat.n_replicates.max()) if len(mat) else 0, "unit": "runs",
                      "runs": [{"run_index": i, "lesion": j["lesion"]["id"], "replicate": j.get("replicate"),
                                "seed": j.get("brain_seed"), "file": j.get("file"),
                                "device": j["provenance"]["execution"]["device"]} for i, j in enumerate(jobs)],
                      "null": None}
    res.add_table("matrix", mat)
    res.add_table("dissociations", dis)
    res.add_table("lesions", [j["lesion"] for j in jobs])
    sens = mat[mat.lesion_id != baseline][["lesion_id", "lesion_kind", "lesion_spec", "check", "baseline", "value",
                                           "delta", "replicate_sd", "n_replicates", "replicate_values"]].copy()
    bodies = {j["lesion"]["id"]: (json.dumps(j["lesion"].get("bodies")) if j["lesion"].get("bodies") else
                                  j["lesion"].get("spec", "")) for j in jobs}
    sens.insert(3, "bodies", [bodies.get(x, "") for x in sens.lesion_id])
    res.add_table("sensitivity", sens)
    res.validation = validate(mat, jobs, baseline=baseline)
    moved = mat[(mat.moved == "yes")]
    res.summary = {"mode": "analyse", "out_dir": str(out_dir), "n_jobs": len(jobs),
                   "lesions": sorted({j["lesion"]["id"] for j in jobs}), "checks": sorted(mat.check.unique()),
                   "replicates_per_arm": {k: int(v) for k, v in mat.groupby("lesion_id").n_replicates.max().items()},
                   "n_moved": int(len(moved)), "n_dissociations": int(len(dis)),
                   "n_underpowered": int((mat.moved == "underpowered").sum()),
                   "underpowered_checks": sorted(mat[mat.moved == "underpowered"].check.unique()),
                   "tolerance": float(tolerance), "rel_tolerance": float(rel_tolerance),
                   "bit_identical_checks": sorted(mat[mat.bit_identical_within_arms].check.unique()),
                   "largest_delta": (moved.reindex(moved.delta.abs().sort_values(ascending=False).index)
                                     [["check", "lesion_id", "delta"]].head(10).to_dict("records") if len(moved) else []),
                   "scatter_rule": f"on a check that scatters, an arm of fewer than {MIN_REPLICATES} runs is quoted with "
                                   f"its replicate values and marked 'underpowered'; a difference is called only on "
                                   f"bit-identity, on seed pairing, or beyond twice the pooled scatter with "
                                   f"{MIN_REPLICATES}+ runs per arm"}
    res.files = {"jobs": [j.get("file") for j in jobs], "generator": "flyverse/interp/lesion.py::lesion(mode='analyse')"}
    return res


def _load_c():
    from .. import connectome as cn
    return cn.load(verbose=False)


def pivot(mat: pd.DataFrame, value: str = "value") -> pd.DataFrame:
    """The printed matrix: checks down, lesions across (the baseline column first)."""
    if not len(mat):
        return pd.DataFrame()
    p = mat.pivot_table(index="check", columns="lesion_id", values=value, aggfunc="mean")
    cols = [c for c in p.columns if c == "baseline"] + [c for c in p.columns if c != "baseline"]
    return p[cols].reset_index()
