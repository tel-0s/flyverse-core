"""What every interpretability tool shares (docs/INTERP.md, section 2).

The toolkit reads the model; it never changes it. This module holds the five things the eight tools agree on:

1. **Population selection** -- one grammar (`resolve`, `populations`) over `Connectome.select`, `regions.labels`
   and body ids, so a lesion manifest, a trace source and an atlas list name cells the same way.
2. **The shaped-weight accessor** -- `effective_weights` returns the matrix Brain installs (brain._shaped_weights x
   the fan-in scale x w_syn, mV per presynaptic spike, post x pre) and `links` turns any block of it into an edge
   table with the raw synapse count, the sign rule that decided each entry and the silence flags (sign 0 / frozen
   / pruned / never firing).  Nothing here re-implements a rule of brain.py: the accessor calls brain._shaped_weights.
3. **Recording** -- `Recorder` (reusing screen.TypeRecorder for the population pooling) captures per-frame
   quantities of a cell set from a FlyBrain / Sim / BatchSim into a `Recording` (npz + json) that the CPU half of
   every tool analyses; `record_frames` is the loop.
4. **Null / replicate helpers** -- `ArmStats`, `compare`, `p_floor`: the object-sweep statistic (z against a
   none-vs-none null, Welch, exact Mann-Whitney) and the scatter rule -- no 'result' below FOUR independent runs per
   arm (`CALL_REPLICATES`; five for a small effect), and none below the run count at which the exact rank test can
   reach alpha at all (`p_floor`). A deterministic null (SD 0) is 'undetermined', never a z of NaN or 1e41.
5. **The result schema** -- `Result` + `provenance`: one JSON for all tools, carrying everything the Neurome
   export (docs/NEUROME_INTERFACE.md) needs, so export is a serializer.

CPU only; torch is imported lazily (inside the functions that need brain.py).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .. import connectome as cn
from .. import regions
from ..screen import TypeRecorder

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "flyverse.interp.result/1"
RECORDING_SCHEMA = "flyverse.interp.recording/1"
TOOLS = ("decompose", "trace", "paths", "lesion", "atlas", "health", "ledger", "export")
DATASET_NAME = "male-cns"
DATASET_RELEASE = "v1.0 flat-connectome"
UNIT_KINDS = ("spiking", "graded", "photoreceptor")
CONTRIBUTION_KINDS = ("anatomical_count", "effective_weight_mV", "current", "voltage", "activity")
FRAME_MS, OPTIC_DT_MS = 10.0, 1.0
MIN_REPLICATES = 3          # the scatter rule: a difference is quoted only over >= 3 independent runs per arm
CALL_REPLICATES = 4         # the CALL rule (docs/INTERP.md 10.1(3), 10.2): min(n_a, n_b) >= 4, 5 for a small effect
#: A difference is CALLED only over `CALL_REPLICATES` runs in EVERY arm -- a rule about the smaller arm, not about the
#: symmetric exact-U floor: 3 v 3 floors at p 0.10, but 3 v 5 floors at 0.036 and would otherwise let three runs be
#: called. `compare` applies both (it reports `p_floor` and says 'underpowered' while it exceeds alpha), so three runs
#: buy the scatter and four per arm (five for a small effect) buy a result, whatever the other arm's size.
Z_RESULT = 3.0              # the object-sweep criterion (docs/audits/object_sweep.md 8.5)
NEVER_FIRING_HZ = 0.5       # a cell whose max rate over a rollout stays below this is 'never_firing'

# The unit-handling facts of the compiled model that every export must declare (docs/NEUROME_INTERFACE.md, section 2).
UNITS = [
    {"item": "node_set", "rule": "every MaleCNS body with status Traced plus every photoreceptor body regardless of "
                                 "status; bodies outside it are absent, not merged"},
    {"item": "graded", "rule": "ol_intrinsic cells run as rate units (optic.py); their LIF rows are frozen and pruned; "
                               "exported as unit_kind graded with rate [0-1] and received drive in mV"},
    {"item": "photoreceptor", "rule": "photoreceptor bodies are driven by the ray tracer and assigned to hex columns by "
                                      "their strongest hexed postsynaptic partner; the column map travels with the export"},
    {"item": "signs", "rule": "presynaptic NT_SIGN, corrected per (post type, pre transmitter) by receptors_by_type.csv "
                              "under LIFParams.receptor_model; TYPE_NT_OVERRIDE and UNKNOWN_NT_OVERRIDE_REGEX listed in "
                              "compiled_connectome; sign-0 presynaptic bodies are silent"},
    {"item": "counts", "rule": "synaptic_pair_count is the raw pre -> post synapse count, uncapped and unsigned; the "
                               "effective weight applies conn_cap, path / type / same-type gains, the fan-in scale and w_syn"},
    {"item": "nothing_combined", "rule": "no cells are merged; per-type rows are means over the named bodies"},
]

# Validation targets: the known localization each tool must reproduce before its numbers are trusted
# (the task brief and the audits it cites; every number from the file named).
VALIDATION = {
    "decompose": {
        "name": "walk.GF_max cancellation + the 282-synapse taste dependence",
        "reference": {"walk.GF_max_hz": {"off": 4.964, "default": 4.629, "holdBrain": 12.517, "holdOptic": 13.311},
                      "taste_carrier": {"entries": 123, "synapses": 282, "targets": ["OA-AL2i3", "TmY14", "DNge138",
                                                                                  "DNge149", "DNge150"],
                                        "pre": ["R8p", "R8_unclear", "R8y", "HBeyelet"], "transmitter": "histamine"}},
        "source": "docs/audits/receptor_integration.md G.4 (walk recount) and E.4 (double dissociation)"},
    "trace": {
        "name": "the object stage + the LH odour gate",
        "reference": {"Mi4_z": [22.3, 28.6], "Mi1_z": [7.8, 27.9], "Tm3_z": [7.8, 15.6],
                      "at_null": {"T2": [0.5, 1.2], "T3": [0.0, 0.7], "Tm5Y": [0.4, 2.6], "TmY21": [0.6, 1.1],
                                  "LC11": [-0.1, 0.4], "LC10a": [-0.1, 0.3]},
                      "LHPD4d1_hz": {"8cm": 20.6, "40cm": 12.5, "clean": 3.4, "d_prime": 4.51}},
        "source": "docs/audits/object_sweep.md 8.4 / 8.7; docs/NOTES.md session 8 (screen_odour --fruit apple)"},
    "paths": {
        "name": "GLNO as the silent link on rotation -> PEN + the ExR4/5/6 -> EPG loop",
        "reference": {"GLNO->PEN": {"entries": 84, "raw_synapses": 16371, "share_of_PEN_input": 0.194, "sign": 0,
                                    "mv_per_pair_if_signed": 16.5},
                      "EPG->PEN_mv_per_pair": 5.05, "EPG->PEN_mv_per_post_volley": 79.9, "Delta7->PEN_mv_per_pair": -4.70,
                      "ExR_loop_two_step_mv2_per_wedge": [-3210, -2760], "PEN_on_wedge_two_step_mv2": 2465,
                      "Delta7_peak_two_step_mv2": -433},
        "source": "docs/audits/cx_glno.md section 1; docs/audits/cx_wedge.md sections 2-3 (cx_wedge.json)"},
    "lesion": {
        "name": "the holdKC / holdDN1 / holdBrain / holdOptic double dissociation",
        "reference": {"taste.MN9_hz": {"off": [1.554839, 4.341760, 2.309808], "default": [5.090923, 4.315772, 2.360074],
                                       "holdBrain": "= off", "holdOptic": "= default", "holdBrainGlu": "= default",
                                       "holdBrainHis": "= off"},
                      "smell.KC_active": {"off": [1079, 427, 1178], "default": [486, 412, 543], "holdBrain": "= off",
                                          "holdOptic": "= default", "holdBrainGlu": "= off", "holdBrainHis": "= default",
                                          "holdKC": [1225, 464, 1006], "holdDN1": [525, 433, 540]}},
        "source": "docs/audits/receptor_integration.md E.4 (scripts/r5_attr_taste_cpu.py, CPU, seeds 0,1,2)"},
    "atlas": {
        "name": "the DNp18 +45 / DNp33 -49 wind flips and the PFL3 -> DNa02 steering result",
        "reference": {"wind.DNp18_flip_hz": 45.0, "wind.DNp33_flip_hz": -49.0,
                      "PFL3_L_80Hz": {"DNa02_R_hz": 22.6, "DNa02_L_hz": 0.0},
                      "DNa02_L_150Hz": {"leg_L_hz": 3.1, "leg_R_hz": 0.1}},
        "source": "docs/NOTES.md session 8 (screen_steering.py; FlyBrain.stimulate at the blueberry site); "
                  "scripts/benchmark.py REFERENCES wind.* and dn.*"},
    "health": {
        "name": "the 200 Hz refractory-limited bump and the sign-0 / silent populations of the NT audit",
        "reference": {"bump_hz": [180, 260], "t_ref_ms": 2.2, "PEN_hz": [40, 65], "Delta7_hz": [90, 112],
                      "sign0_presynaptic_bodies": 3312, "sign0_synapse_share": 0.022,
                      "mushroom_body_input_share_silenced": 0.098, "visual_centrifugal_output_silenced": 0.176,
                      "note": "3,312 is the SHIPPED cache's sign-0 body count (docs/audits/nt_audit.md; 2,683 of them "
                              "presynaptic, 2,701,289 of 124,161,873 synapses). The 3,407 this entry carried until "
                              "this revision is the pre-TYPE_NT_OVERRIDE cache's count (receptor_verification.md, "
                              "sum|W| 121,427,136) and is not this model's; docs/NEUROME_INTERFACE.md line 73 still "
                              "quotes it. The tool reports the count of the cache it loaded, with its fingerprint."},
        "source": "docs/audits/cx_wedge.md section 7 / cx_glno.md section 5; docs/audits/nt_audit.md"},
    "ledger": {
        "name": "the existing T4/T5 DS, loom, optomotor and sugar/bitter numbers",
        "reference": {"motion.min_dsi": 0.16, "loom.GF_peak_hz": [34, 56], "rotation_dprime": {"HSN": 4.2, "DNp20": 3.1,
                                                                                            "HSE": 2.2},
                      "bitter.calibrated_sugar_MN9_hz": 4.6, "bitter.calibrated_sugar_bitter_MN9_hz": 0.0,
                      "bitter.shiu_sugar_MN9_hz": 123.5, "bitter.shiu_sugar_bitter_MN9_hz": 2.1},
        "source": "scripts/benchmark.py REFERENCES; docs/BENCHMARK_BATTERY.md"},
    "export": {
        "name": "a round-trip: Result -> manifest + tables -> the same numbers, bodyId as decimal strings",
        "reference": {"key": ["dataset", "release", "bodyId"], "LC11_LC10a_rows_per_body": 2},
        "source": "docs/NEUROME_INTERFACE.md section 1"},
}


# ---------------------------------------------------------------------------------------------- 1. population selection
_REGEX_META = set("^$.*+?[](){}\\")


@dataclass
class Population:
    """A named cell set: the spec it came from, its model indices (sorted, unique) and body ids."""
    label: str
    spec: object
    idx: np.ndarray
    body_ids: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.idx))

    def record(self) -> dict:
        return {"label": self.label, "spec": spec_repr(self.spec), "n_cells": self.n,
                "body_ids": [str(int(b)) for b in self.body_ids]}


def spec_repr(spec) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        return "&".join(f"{k}={v}" if not (isinstance(v, str) and v.startswith("~")) else f"{k}{v}" for k, v in spec.items())
    if isinstance(spec, (list, tuple)):
        return " | ".join(spec_repr(s) for s in spec)
    a = np.asarray(spec)
    return f"<{a.dtype.kind}{a.size}>"


def _clause(c: cn.Connectome, clause: str) -> np.ndarray:
    """One 'key=value' / 'key~regex' / 'key:a|b' / bare clause -> boolean mask over c."""
    clause = clause.strip()
    if clause.startswith("~"):                                   # Connectome.select's convention: a regex on type
        return _clause(c, f"type~{clause[1:]}")
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(=|~|:)\s*(.*)$", clause)
    if m is None:                                                # bare token: a type name, a regex on type, or a|b|c
        if any(ch in _REGEX_META for ch in clause):
            return _clause(c, f"type~{clause}")
        if "|" in clause:
            return _clause(c, f"type:{clause}")
        return _clause(c, f"type={clause}")
    key, op, val = m.group(1), m.group(2), m.group(3).strip()
    if key == "module":
        if val not in regions.MODULES:
            raise ValueError(f"unknown module {val!r}; choose from {regions.MODULES}")
        return regions.labels(c) == val
    if key in ("body", "bodyId"):
        ids = np.array([int(x) for x in val.split("|")], dtype=np.int64)
        return np.isin(c.neurons.bodyId.to_numpy(), ids)
    if key == "index":
        idx = np.array([int(x) for x in val.split("|")], dtype=np.int64)
        mask = np.zeros(c.n, bool); mask[idx] = True
        return mask
    if key not in c.neurons.columns:
        raise ValueError(f"unknown selection key {key!r}; columns: {list(c.neurons.columns)}")
    col = c.neurons[key]
    if op == "~":
        return col.fillna("").astype(str).str.contains(val, regex=True).to_numpy()
    values = val.split("|") if op == ":" else [val]
    if pd.api.types.is_numeric_dtype(col.dtype):
        return col.isin([float(v) for v in values]).to_numpy()
    return col.fillna("").astype(str).isin(values).to_numpy()


def resolve(c: cn.Connectome, spec) -> np.ndarray:
    """Model indices (sorted, unique) of a population spec.

    spec: an index array / boolean mask / Connectome.select criteria dict (as Connectome.indices), a string
    `clause[&clause...]` (AND) where a clause is `key=value`, `key~regex`, `key:a|b|c` (isin), `module=<regions.MODULES>`,
    `body:123|456`, `index:0|5`, or a bare token (a type name; a regex on type when it contains a metacharacter;
    `a|b` = types a or b), or a list / tuple of specs (OR). Examples: "DNa02", "~^LC1(0a|1)$", "type=PEN_a&somaSide=L",
    "module=optic&nt=histamine", ["EPG", "PEN_a", "PEN_b"].
    """
    if isinstance(spec, Population):
        return spec.idx
    if isinstance(spec, (list, tuple)) and not (len(spec) and isinstance(spec[0], (int, np.integer, bool, np.bool_))):
        if len(spec) == 0:
            return np.zeros(0, np.int64)
        mask = np.zeros(c.n, bool)
        for s in spec:
            mask[resolve(c, s)] = True
        return np.flatnonzero(mask)
    if isinstance(spec, str):
        mask = np.ones(c.n, bool)
        for clause in spec.split("&"):
            mask &= _clause(c, clause)
        return np.flatnonzero(mask)
    return np.unique(c.indices(spec))


def population(c: cn.Connectome, spec, label: str | None = None) -> Population:
    idx = resolve(c, spec)
    return Population(label if label is not None else spec_repr(spec), spec, idx, c.neurons.bodyId.to_numpy()[idx])


def populations(c: cn.Connectome, specs) -> list[Population]:
    """A dict {label: spec} or a list of specs (labelled by their repr) -> [Population]."""
    if isinstance(specs, dict):
        return [population(c, s, k) for k, s in specs.items()]
    return [population(c, s) for s in specs]


def by_type(c: cn.Connectome, idx, by_side: bool = False) -> dict[str, np.ndarray]:
    """{type (or type_side): indices} over a cell set, in TypeRecorder's key order."""
    idx = np.asarray(idx)
    if len(idx) == 0:
        return {}
    t = c.neurons.type.fillna("").to_numpy()[idx]
    if by_side:
        s = c.neurons.somaSide.fillna("?").to_numpy()[idx]
        t = np.array([f"{a}_{b}" for a, b in zip(t, s)])
    keys, inv = np.unique(t, return_inverse=True)
    return {k: idx[inv == i] for i, k in enumerate(keys)}


def body_str(ids) -> list[str]:
    """bodyIds as decimal strings (the interchange key of docs/NEUROME_INTERFACE.md)."""
    return [str(int(b)) for b in np.asarray(ids).ravel()]


# ---------------------------------------------------------------------------------------------- 2. shaped weights
@dataclass
class EffectiveWeights:
    """A[post, pre] in mV per presynaptic spike exactly as Brain installs it (before the optic prune):
    brain._shaped_weights (receptor signs, cap, path / type / same-type gains) x the fan-in scale x w_syn.
    `scale` and `tot` are the fan-in factor and the shaped input total per postsynaptic cell; `md5` hashes A."""
    A: sp.csr_matrix
    scale: np.ndarray
    tot: np.ndarray
    params: object
    md5: str
    shaped_md5: str

    def block(self, post_idx, pre_idx) -> np.ndarray:
        return self.A[np.asarray(post_idx)][:, np.asarray(pre_idx)].toarray().astype(np.float64)

    def type_matrix(self, c: cn.Connectome, post_types, pre_types) -> pd.DataFrame:
        """M[post_type, pre_type] = mean over post cells of the summed input from every pre cell of the type
        (mV per post cell per volley of the presynaptic type -- cx_wedge's 'total_per_post')."""
        groups_post = {t: c.select(type=t) for t in post_types}
        groups_pre = {t: c.select(type=t) for t in pre_types}
        M = np.zeros((len(post_types), len(pre_types)))
        for i, tp in enumerate(post_types):
            rows = self.A[groups_post[tp]]
            for j, tq in enumerate(pre_types):
                if len(groups_post[tp]) and len(groups_pre[tq]):
                    M[i, j] = rows[:, groups_pre[tq]].sum(axis=1).mean()
        return pd.DataFrame(M, index=list(post_types), columns=list(pre_types))


def _md5_csr(W: sp.csr_matrix) -> str:
    W = W.tocsr()
    if not W.has_sorted_indices:
        W = W.copy(); W.sort_indices()
    m = hashlib.md5(); m.update(W.data.tobytes()); m.update(W.indices.tobytes()); m.update(W.indptr.tobytes())
    return m.hexdigest()


def effective_weights(c: cn.Connectome, params=None, receptor=None) -> EffectiveWeights:
    """The matrix Brain installs, computed on the CPU without building a Brain (same code path: brain._shaped_weights,
    then the fan-in scale of Brain.__init__ on the REFERENCE graph, then w_syn). `receptor` is an optional
    precomputed connectome.receptor_signs(c, ...)."""
    from .. import brain
    p = params or brain.LIFParams()
    W = brain._shaped_weights(c, p, receptor)
    shaped_md5 = _md5_csr(W)
    if p.input_norm_alpha > 0:
        ref = c.reference
        if ref is c:
            tot = np.asarray(abs(W).sum(axis=1)).ravel()
        else:
            tot = np.asarray(abs(brain._shaped_weights(ref, p)).sum(axis=1)).ravel()[ref.index_of(c.neurons.bodyId)]
        scale = np.clip((p.input_norm_ref / np.maximum(tot, 1.0)) ** p.input_norm_alpha, 0.02, 1.0).astype(np.float32)
    else:
        tot = np.asarray(abs(W).sum(axis=1)).ravel()
        scale = np.ones(c.n, np.float32)
    A = (sp.diags(scale) @ W).tocsr()
    A.data = (A.data * np.float32(p.w_syn)).astype(np.float32)
    A.sort_indices()
    return EffectiveWeights(A, scale, tot, p, _md5_csr(A), shaped_md5)


def raw_counts(c: cn.Connectome, with_sign0: bool = True, *, dtype=np.float32, build: bool = True) -> tuple[sp.csr_matrix, bool]:
    """The true raw synapse count of every stored entry of c.W (post x pre), unsigned and uncapped.

    The compiled cache holds the signed raw count of every signed entry in `W.data` and an explicit **zero** for every
    entry whose presynaptic transmitter carries no sign; those entries' counts live in `cache/sign0_counts.npz`
    (`connectome.sign0_counts`), an array aligned with `W.data` that is non-zero ONLY on them. The two are therefore
    **merged, never substituted**:

        C.data = maximum(|W.data| (already in count units), sign0_counts)

    (Substituting -- the pre-revision behaviour -- reported 0 synapses for every signed edge and wrote
    `synaptic_pair_count 0` into the mandatory Neurome column; `docs/INTERP.md` section 11, defect 1.)

    `with_sign0` says whether the **sign-0 entries are included**: True (the default) fills them from
    `sign0_counts`, so the matrix carries every synapse the connectome has; False leaves them at their stored 0, i.e.
    the count of the entries the LIF can actually carry. Signed entries are the same either way.

    `build` is passed to `connectome.sign0_counts` (build the npz from the raw weights table when it is absent);
    `dtype` is float32 by default and float64 where an exact whole-model total is wanted (the NT audit's
    124,161,873 synapses do not fit a float32 mantissa). Returns (counts, sign0_available); `sign0_available` is
    False when the npz is not on this machine, and the sign-0 entries then read 0 (a lower bound).
    """
    W = c.W.tocsr()
    C = W.copy()                                             # keep the structure, explicit zeros included
    C.data = np.abs(W.data).astype(dtype)
    if not with_sign0:
        return C, False
    try:
        cnt = cn.sign0_counts(c, W=W, build=build)
    except Exception:  # noqa: BLE001 -- the raw weights table is not on every machine
        cnt = None
    if cnt is None or len(np.asarray(cnt)) != W.nnz:
        return C, False
    C.data = np.maximum(C.data, np.asarray(cnt, dtype=dtype))
    return C, True


def silent_flags(c: cn.Connectome, pre_idx, frozen_idx=None, prune_frozen: bool = True, rates=None,
                 min_hz: float = NEVER_FIRING_HZ) -> pd.DataFrame:
    """Per presynaptic cell: sign0 (its transmitter carries no sign -> every output entry is an explicit zero),
    frozen (a rate unit of the optic lobe: never spikes in the LIF), pruned (frozen and dropped from the LIF matrix),
    never_firing (max recorded rate below `min_hz`).

    Every column is a **boolean**: with no `rates` the never_firing question was not asked, and the answer is False
    (not NaN, which `links` and every other `bool(flag)` reader turned into 'every cell never fires' --
    docs/INTERP.md section 11, defect 2). A structural table therefore carries no never_firing flag; say so with the
    table's own silent_rule when it matters. `rates`: (T, N) or (N,) Hz over the model indices, or a dict
    {index: max_hz}."""
    pre_idx = np.asarray(pre_idx)
    sign = c.neurons["sign"].to_numpy() if "sign" in c.neurons else np.ones(c.n)
    out = pd.DataFrame({"index": pre_idx, "sign0": (sign[pre_idx] == 0)})
    frozen = np.zeros(c.n, bool)
    if frozen_idx is not None:
        frozen[np.asarray(frozen_idx)] = True
    out["frozen"] = frozen[pre_idx]
    out["pruned"] = out.frozen & bool(prune_frozen)
    if rates is None:
        out["never_firing"] = np.zeros(len(pre_idx), bool)      # not evaluated: no rollout was given
    else:
        if isinstance(rates, dict):
            mx = np.array([rates.get(int(i), np.nan) for i in pre_idx])
        else:
            r = np.asarray(rates)
            mx = (r.max(axis=0) if r.ndim > 1 else r)[pre_idx]
        out["never_firing"] = mx < min_hz
    return out


def links(c: cn.Connectome, ew: EffectiveWeights, pre, post, receptor=None, counts=None, flags: pd.DataFrame | None = None,
          nonzero_only: bool = False) -> pd.DataFrame:
    """Edge table of the block pre -> post (Neurome field names): body_pre / body_post, pre_type / post_type, pre_nt,
    synaptic_pair_count (raw, unsigned, uncapped), effective_mv (A entry), sign, sign_rule ('nt_sign' or the
    receptor tier), gain_rule (the factors in force, one string), silent (flags joined by '|'). Entries that exist in
    the cache but are zero in A (sign 0) are kept unless nonzero_only."""
    pre_idx, post_idx = resolve(c, pre), resolve(c, post)
    if counts is None:
        counts, _ = raw_counts(c)
    Wc = c.W.tocsr()
    sub = Wc[post_idx][:, pre_idx].tocoo()
    Cn = counts[post_idx][:, pre_idx].tocsr()
    Ab = ew.A[post_idx][:, pre_idx].tocsr()
    n = c.neurons
    ty, nt = n.type.fillna("").to_numpy(), (n["nt"].fillna("unknown").to_numpy() if "nt" in n else np.array(["unknown"] * c.n))
    bid = n.bodyId.to_numpy()
    rows = post_idx[sub.row]; cols = pre_idx[sub.col]
    eff = np.asarray(Ab[sub.row, sub.col]).ravel()
    cnt = np.asarray(Cn[sub.row, sub.col]).ravel()
    tier = np.array(["nt_sign"] * len(rows), dtype=object)
    if receptor is not None:
        # receptor arrays are in CSR order of c.W: locate each (row, col) entry
        Wcsr = c.W.tocsr()
        pos = np.array([_entry_pos(Wcsr, r, q) for r, q in zip(rows, cols)])
        ok = pos >= 0
        t = np.asarray(receptor.tier)[pos[ok]]
        matched = t >= cn.RECEPTOR_TIERS.index("nt_class")
        tier[np.flatnonzero(ok)[matched]] = np.array(["receptor:" + cn.RECEPTOR_TIERS[i] for i in t[matched]], dtype=object)
    p = ew.params
    gain_rule = (f"conn_cap {p.conn_cap}; same_type_gain {p.same_type_gain}; input_norm ({p.input_norm_ref}, alpha {p.input_norm_alpha}); "
                 f"w_syn {p.w_syn}; path_gain + type_path_gain as resolved in provenance.model")
    df = pd.DataFrame({"pre_index": cols, "post_index": rows, "body_pre": body_str(bid[cols]), "body_post": body_str(bid[rows]),
                       "pre_type": ty[cols], "post_type": ty[rows], "pre_nt": nt[cols], "synaptic_pair_count": cnt,
                       "effective_mv": eff.astype(np.float64), "sign": np.sign(eff).astype(int), "sign_rule": tier,
                       "gain_rule": gain_rule, "fanin_scale_post": ew.scale[rows].astype(np.float64)})
    if flags is None:
        flags = silent_flags(c, np.unique(cols))
    f = flags.set_index("index")
    cols_f = ["sign0", "frozen", "pruned", "never_firing"]
    ff = f.reindex(cols).reset_index(drop=True)
    # a flag that was not evaluated (no rates) or whose cell is absent from `flags` reads False, never True:
    # bool(NaN) is True, which is what marked every structural row 'never_firing' (docs/INTERP.md 11, defect 2).
    fb = {k: ff[k].fillna(False).to_numpy().astype(bool) for k in cols_f}
    df["silent"] = ["|".join(k for k in cols_f if fb[k][i]) for i in range(len(df))]
    if nonzero_only:
        df = df[df.effective_mv != 0]
    return df.sort_values(["post_index", "pre_index"]).reset_index(drop=True)


def _entry_pos(W: sp.csr_matrix, r: int, q: int) -> int:
    a, b = W.indptr[r], W.indptr[r + 1]
    k = np.searchsorted(W.indices[a:b], q)
    return int(a + k) if k < b - a and W.indices[a + k] == q else -1


def frozen_indices(fb) -> np.ndarray | None:
    """The optic rate units of a FlyBrain (frozen in its LIF), or None when it has no optic lobe."""
    o = getattr(fb, "optic", None)
    return None if o is None else np.asarray(o.rate_idx)


def unit_kinds(c: cn.Connectome, fb=None) -> np.ndarray:
    """(N,) 'spiking' | 'graded' | 'photoreceptor' (docs/NEUROME_INTERFACE.md section 2). With a FlyBrain, graded = its
    optic rate units; without one, the static rule (superclass ol_intrinsic)."""
    out = np.full(c.n, "spiking", dtype=object)
    sc = c.neurons.superclass.fillna("").to_numpy()
    graded = (sc == "ol_intrinsic") & c.has_optic_columns
    fr = frozen_indices(fb) if fb is not None else None
    if fr is not None:
        graded = np.zeros(c.n, bool); graded[fr] = True
    out[graded] = "graded"
    out[np.isin(c.neurons.type.fillna("").to_numpy(), cn.PHOTORECEPTOR_TYPES)] = "photoreceptor"
    return out


# ---------------------------------------------------------------------------------------------- 3. recording
QUANTITIES = {"rate_hz": "Hz", "drive_mv": "mV", "v_mv": "mV", "adapt_mv": "mV", "refrac": "0/1", "spike_count": "spikes",
              "optic_dr": "rate units", "optic_rate": "rate units"}


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


@dataclass
class Recording:
    """Per-frame quantities of a cell set over a rollout: `quantities[name]` is (T, n) for a single fly or (T, B, n)
    for a batch; `motor[name]` is (T,) / (T, B). Saved as <path>.npz + <path>.json (RECORDING_SCHEMA)."""
    t_ms: np.ndarray
    idx: np.ndarray
    body_ids: np.ndarray
    types: np.ndarray
    quantities: dict = field(default_factory=dict)
    motor: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def batched(self) -> bool:
        return any(v.ndim == 3 for v in self.quantities.values())

    @property
    def n_frames(self) -> int:
        return int(len(self.t_ms))

    def window(self, start_s: float, end_s: float) -> "Recording":
        m = (self.t_ms >= start_s * 1000.0) & (self.t_ms < end_s * 1000.0)
        return Recording(self.t_ms[m], self.idx, self.body_ids, self.types,
                         {k: v[m] for k, v in self.quantities.items()}, {k: v[m] for k, v in self.motor.items()},
                         dict(self.meta, window_s=[start_s, end_s]))

    def row(self, b: int) -> "Recording":
        """One batch member as a single-fly recording."""
        q = {k: (v[:, b] if v.ndim == 3 else v) for k, v in self.quantities.items()}
        mo = {k: (v[:, b] if v.ndim == 2 else v) for k, v in self.motor.items()}
        return Recording(self.t_ms, self.idx, self.body_ids, self.types, q, mo, dict(self.meta, batch_row=b))

    def recorder(self, by_side: bool = False) -> TypeRecorder:
        """A screen.TypeRecorder over the recorded cells (keys = types, or type_side), indices into this recording."""
        labels = self.types.astype(str)
        if by_side:
            labels = np.array([f"{a}_{b}" for a, b in zip(labels, self.meta.get("sides", ["?"] * len(labels)))])
        keys, inv = np.unique(labels, return_inverse=True)
        return TypeRecorder(keys=keys, idx=np.arange(len(self.idx)), inv=inv, n_cells=np.bincount(inv))

    def per_type(self, quantity: str = "rate_hz", by_side: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """(keys, (T, n_keys)) population means of one quantity (single-fly recordings; use .row(b) for a batch)."""
        rec = self.recorder(by_side)
        x = self.quantities[quantity]
        if x.ndim == 3:
            raise ValueError("per_type on a batched recording: select a row first (Recording.row)")
        return rec.keys, np.stack([rec.snapshot(f) for f in x])

    def per_cell_mean(self, quantity: str = "rate_hz") -> np.ndarray:
        x = self.quantities[quantity]
        return x.mean(axis=0)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"t_ms": self.t_ms, "idx": self.idx, "body_ids": self.body_ids, "types": self.types.astype(str)}
        arrays.update({f"q__{k}": v for k, v in self.quantities.items()})
        arrays.update({f"m__{k}": v for k, v in self.motor.items()})
        np.savez(path.with_suffix(".npz"), **arrays)
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump({"schema": RECORDING_SCHEMA, "meta": to_jsonable(self.meta), "quantities": list(self.quantities),
                       "motor": list(self.motor), "n_frames": self.n_frames, "n_cells": int(len(self.idx)),
                       "units": {k: QUANTITIES.get(k, "") for k in self.quantities}}, f, indent=1)
        return path.with_suffix(".npz")

    @classmethod
    def load(cls, path) -> "Recording":
        path = Path(path)
        z = np.load(path.with_suffix(".npz"), allow_pickle=False)
        meta = {}
        if path.with_suffix(".json").exists():
            with open(path.with_suffix(".json"), encoding="utf-8") as f:
                meta = json.load(f).get("meta", {})
        q = {k[3:]: z[k] for k in z.files if k.startswith("q__")}
        mo = {k[3:]: z[k] for k in z.files if k.startswith("m__")}
        return cls(z["t_ms"], z["idx"], z["body_ids"], z["types"], q, mo, meta)


class Recorder:
    """Capture per-frame quantities of a cell set from a FlyBrain (or anything with `.brain` / `.optic`).

        rec = Recorder(fb.c, ["DNp01", "~^LC4$", "module=descending"], quantities=("rate_hz", "drive_mv"))
        for _ in range(n_frames):
            sim.step(); rec.capture(sim.fb, motor=sim.fb.motor())
        recording = rec.finish(meta={"protocol": ...})

    rate_hz reads brain.rate (rate_np() for B = 1, the cached per-step copy the demo uses); drive_mv / v_mv / adapt_mv /
    refrac / spike_count read the Brain tensors of the same names (spike_count is cumulative: diff it for spikes per
    frame); optic_dr / optic_rate read the OpticLobe's delta_rate / rates() for the recorded cells that are rate units
    (NaN elsewhere). Population pooling is screen.TypeRecorder's (Recording.recorder / per_type)."""

    def __init__(self, c: cn.Connectome, selection, quantities=("rate_hz",), by_side: bool = False):
        self.c = c
        self.idx = resolve(c, selection)
        self.quantities = tuple(quantities)
        unknown = set(self.quantities) - set(QUANTITIES)
        if unknown:
            raise ValueError(f"unknown quantities {sorted(unknown)}; choose from {list(QUANTITIES)}")
        self.by_side = by_side
        n = c.neurons
        self.types = n.type.fillna("").to_numpy()[self.idx].astype(str)
        self.sides = n.somaSide.fillna("?").to_numpy()[self.idx].astype(str) if "somaSide" in n else np.array(["?"] * len(self.idx))
        self.body_ids = n.bodyId.to_numpy()[self.idx]
        self._frames = {q: [] for q in self.quantities}
        self._motor = {}
        self._t = []
        self._optic_pos = None
        self._extension_frames = {}
        self._extension_model = None

    def _optic_positions(self, optic):
        if self._optic_pos is None:
            pos = -np.ones(self.c.n, np.int64); pos[np.asarray(optic.rate_idx)] = np.arange(len(optic.rate_idx))
            self._optic_pos = pos[self.idx]
        return self._optic_pos

    def capture(self, fb, t_ms: float | None = None, motor=None) -> None:
        brain = getattr(fb, "brain", fb)
        B = int(getattr(brain, "B", 1))
        self._t.append(float(brain.t if t_ms is None else t_ms))
        extension_inputs = fb.module_inputs() if hasattr(fb, "module_inputs") else {}
        shape = (B, len(self.idx)) if B > 1 else (len(self.idx),)
        for key in set(extension_inputs) | set(self._extension_frames):
            frames = self._extension_frames.setdefault(key, [np.zeros(shape, np.float32) for _ in self._t[:-1]])
            value = _np(extension_inputs[key])[..., self.idx] if key in extension_inputs else np.zeros(shape, np.float32)
            frames.append(np.asarray(value, np.float32).reshape(shape).copy())
        if extension_inputs or getattr(fb, "attached_modules", {}) or getattr(fb, "hooks", []):
            self._extension_model = {"hooks": fb.hooks, "modules": fb.module_records()}
        for q in self.quantities:
            if q == "rate_hz":
                x = np.asarray(brain.rate_np())[self.idx] if B == 1 and hasattr(brain, "rate_np") else _np(brain.rate)[..., self.idx]
            elif q in ("drive_mv", "v_mv", "adapt_mv", "refrac", "spike_count"):
                name = {"drive_mv": "drive", "v_mv": "v", "adapt_mv": "adapt", "refrac": "refrac", "spike_count": "spike_counts"}[q]
                x = _np(getattr(brain, name))[..., self.idx]
                if q == "refrac":
                    x = (x > 0).astype(np.float32)
            else:                                                   # optic_dr / optic_rate
                optic = getattr(fb, "optic", None)
                shape = (B, len(self.idx)) if B > 1 else (len(self.idx),)
                x = np.full(shape, np.nan, np.float32)
                if optic is not None:
                    pos = self._optic_positions(optic); ok = pos >= 0
                    src = _np(optic.delta_rate if q == "optic_dr" else optic.rates())
                    x[..., ok] = src[..., pos[ok]]
            x = np.asarray(x, dtype=np.float32)
            self._frames[q].append(x[0] if (B == 1 and x.ndim == 2) else x)
        if motor is not None:
            names = [f.name for f in dataclasses.fields(motor)] if dataclasses.is_dataclass(motor) else list(vars(motor))
            for name in names:
                v = getattr(motor, name)
                if isinstance(v, (int, float, np.ndarray, np.floating)) and name != "time_ms":
                    self._motor.setdefault(name, []).append(np.asarray(v, dtype=np.float32))
            for ch, v in getattr(motor, "lh_odour", {}).items():
                self._motor.setdefault(f"lh_odour.{ch}", []).append(np.asarray(v, dtype=np.float32))

    def finish(self, meta: dict | None = None) -> Recording:
        q = {k: np.stack(v) if v else np.zeros((0, len(self.idx)), np.float32) for k, v in self._frames.items()}
        q.update({k: np.stack(v) for k, v in self._extension_frames.items()})
        mo = {k: np.stack(v) for k, v in self._motor.items()}
        m = dict(meta or {}); m.setdefault("sides", self.sides.tolist()); m.setdefault("quantities", list(self.quantities))
        if self._extension_model is not None:
            m["extensions"] = self._extension_model
            m["input_classes"] = list(self._extension_frames)
        return Recording(np.asarray(self._t, dtype=np.float64), self.idx.copy(), self.body_ids.copy(), self.types.copy(), q, mo, m)


def module_input_table(records):
    """External input identities, kept distinct from synaptic edges in trace/decompose."""
    rows = []
    for m in records:
        for label, ids in m.get("writes", {}).items():
            rows.extend({"input_class": "module:" + m["name"], "channel": m["channel_out"],
                         "body_post": str(i), "label": label, "module_kind": m["kind"]} for i in ids)
    return pd.DataFrame(rows)


def record_frames(step, rec: Recorder, frames: int, fb=None, motor=None, every: int = 1) -> Recorder:
    """Run `step()` `frames` times, capturing every `every`-th frame. `step` returns the object to capture from (a
    FlyBrain, or anything with .brain) or None (then `fb` is captured); `motor` is an optional callable returning
    the MotorRates to store."""
    for k in range(int(frames)):
        out = step()
        if k % every == 0:
            rec.capture(out if out is not None else fb, motor=motor() if motor else None)
    return rec


# ---------------------------------------------------------------------------------------------- 4. null / replicates
@dataclass
class ArmStats:
    n: int
    mean: float
    sd: float
    values: list

    @classmethod
    def of(cls, values) -> "ArmStats":
        v = np.asarray([x for x in np.asarray(values, dtype=np.float64).ravel() if np.isfinite(x)])
        return cls(int(len(v)), float(v.mean()) if len(v) else float("nan"),
                   float(v.std(ddof=1)) if len(v) > 1 else float("nan"), v.tolist())

    def record(self) -> dict:
        return dataclasses.asdict(self)


def p_floor(n_a: int, n_b: int) -> float:
    """The smallest two-sided exact Mann-Whitney p that two arms of `n_a` and `n_b` runs can reach -- every draw of one
    arm above every draw of the other: 2 / C(n_a + n_b, n_a).

    3 v 3 runs floor at 0.10, 4 v 4 at 0.029, 5 v 5 at 0.0079 (docs/audits/object_sweep.md 8.4). The floor is
    SYMMETRIC and is therefore not the run-count rule: 3 v 5 floors at 0.036, which is under alpha and would let a
    three-run arm be called. `compare` applies the floor AND `CALL_REPLICATES` (>= 4 runs in the smaller arm);
    this function reports the floor alone."""
    from math import comb
    n_a, n_b = int(n_a), int(n_b)
    if n_a < 1 or n_b < 1:
        return float("nan")
    return 2.0 / comb(n_a + n_b, n_a)


def compare(stim, null, z_min: float = Z_RESULT, min_n: int = MIN_REPLICATES, alpha: float = 0.05) -> dict:
    """The object-sweep comparison of two replicate arms (docs/audits/object_sweep.md 8.4): z = (mean stim - mean null) /
    SD(null); Welch = the difference over the standard error of the two means; exact Mann-Whitney U and p when scipy
    can. Every value the verdict rests on is returned, `p_floor` (the exact-U floor at these two n) included.

    The verdict, in this order:

    * `'underpowered'` -- fewer than `max(min_n, CALL_REPLICATES)` runs in the SMALLER arm, **or** `p_floor > alpha`:
      at this many runs the rank test cannot reach alpha however large the effect, so no z can make the difference a
      result (3 v 3 floors at p 0.10). The two conditions are not the same one: the floor is symmetric, so 3 v 5
      floors at 0.036 and used to be callable on three runs -- the rule is `min(n_a, n_b) >= 4` (docs/INTERP.md
      10.1(3) / 10.2), and `min_n` is raised to `CALL_REPLICATES` whatever the caller asks for. `MIN_REPLICATES`
      stays 3 -- three runs still buy the scatter, and a bit-identical check is right to use them -- but four per arm
      is the smallest that can be CALLED, five when the effect is small. The applied floor is returned as `min_n` and
      the smaller arm as `n_min`.
    * `'undetermined'` -- the null arm is deterministic (SD 0, up to float noise: bit-identical draws, an all-silent
      readout, a `gain_fb=0` lobe), the arms differ, and the rank test does not settle it as null (p <= alpha, or no
      p at all). z is the criterion and z is not defined there, so neither a huge z nor a NaN one is a verdict: read
      `diff` (the magnitude) and `p`, plus the tool's own effect-size column (atlas `z_floor`) where it has one.
      This replaces the three private answers of the build round (docs/INTERP.md 11, defect 4). A deterministic null
      that the rank test DOES settle (p > alpha) is a plain 'null'.
    * `'result'` -- |z| >= `z_min` and p <= `alpha` (when a p exists).
    * `'null'` -- otherwise, the deterministic null with no difference at all included.
    """
    from scipy import stats
    a, b = ArmStats.of(stim), ArmStats.of(null)
    diff = a.mean - b.mean
    z = diff / b.sd if (b.n > 1 and b.sd > 0) else float("nan")
    se = np.sqrt((a.sd ** 2 / a.n if a.n > 1 else 0.0) + (b.sd ** 2 / b.n if b.n > 1 else 0.0)) if (a.n > 1 or b.n > 1) else float("nan")
    welch = diff / se if (se and np.isfinite(se) and se > 0) else float("nan")
    U = p = float("nan")
    if a.n >= 1 and b.n >= 1 and (a.n + b.n) >= 3:
        try:
            r = stats.mannwhitneyu(a.values, b.values, alternative="two-sided", method="exact" if (a.n + b.n) <= 40 else "asymptotic")
            U, p = float(r.statistic), float(r.pvalue)
        except ValueError:
            pass
    floor = p_floor(a.n, b.n) if (a.n >= 1 and b.n >= 1) else float("nan")
    # 'deterministic' is SD exactly 0 or below float noise on the arm's own scale -- the case that sent z to 1e41 on
    # near-zero groups as surely as the exactly-zero one sent it to NaN.
    det_null = bool(b.n >= 2 and np.isfinite(b.sd) and b.sd <= 1e-12 * max(1.0, abs(b.mean)))
    n_min = int(min(a.n, b.n))
    called_n = max(int(min_n), CALL_REPLICATES)      # the run-count rule is about the SMALLER arm, not about p_floor
    if n_min < called_n or (np.isfinite(floor) and floor > alpha):
        verdict = "underpowered"
    elif det_null and diff != 0 and not (np.isfinite(p) and p > alpha):
        verdict = "undetermined"
    elif np.isfinite(z) and abs(z) >= z_min and (not np.isfinite(p) or p <= alpha):
        verdict = "result"
    else:
        verdict = "null"
    return {"stim": a.record(), "null": b.record(), "diff": float(diff), "z": float(z), "welch": float(welch),
            "U": U, "p": p, "verdict": verdict, "z_min": z_min, "min_n": called_n, "min_n_requested": int(min_n),
            "min_replicates": MIN_REPLICATES, "n_min": n_min, "alpha": alpha,
            "p_floor": float(floor), "null_sd_zero": det_null}


def replicate_seeds(n: int, seed0: int = 0) -> list[int]:
    """Seeds of n independent runs (the replicate unit is 'runs', not batch rows: docs/BATCH_SIM.md)."""
    return [int(seed0 + k) for k in range(int(n))]


# ---------------------------------------------------------------------------------------------- 5. provenance / result
def to_jsonable(o):
    """numpy / dataclass / Path / set -> plain JSON types."""
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return to_jsonable(dataclasses.asdict(o))
    if isinstance(o, pd.DataFrame):
        return to_jsonable(o.to_dict("records"))
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def porcelain_path(line: str) -> str:
    """The path of one `git status --porcelain` line ('XY path', 'R  old -> new').

    Not `line[3:]`: the XY field is two columns and a space, but `_git` **strips** the command's output, so the FIRST
    line arrives without its leading space (' M docs/INTERP.md' -> 'M docs/INTERP.md') and the fixed slice ate a
    character of the path ('ocs/INTERP.md'), which no reader of `modified_files` could match against a checkout.
    Strip the status field by its shape instead, and keep the target of a rename."""
    m = re.match(r"^\s*[A-Z?!ADMRCU ]{1,2}\s+(.*)$", line.rstrip())
    path = (m.group(1) if m else line).strip()
    if " -> " in path:                                   # 'R  old -> new': the file that is there now
        path = path.split(" -> ", 1)[-1].strip()
    return path.strip('"')


def git_state() -> dict:
    """{'commit', 'dirty', 'modified_files'} of this checkout ('unknown' when git is unavailable)."""
    head = _git("rev-parse", "HEAD") or "unknown"
    status = _git("status", "--porcelain")
    files = [porcelain_path(ln) for ln in status.splitlines() if ln.strip()]
    return {"commit": head, "dirty": bool(files), "modified_files": files[:200]}


def source_fingerprint(git: dict | None = None, force: bool = False) -> dict:
    """The content identity of the code behind a Result when git cannot name it.

    A cluster job runs from an rsynced copy with no `.git`, so `git_state()` there says `commit 'unknown'` and the
    JSON names no code at all (docs/INTERP.md 11, defect 8: every GPU Result but the export's). This calls
    `export.source_fingerprint(include_loaded=True)` -- SHA-256 of every simulation / probe source under ROOT plus
    the modules the process really imported -- so `export.match_sources` can pin the run to a checkout by content.
    Computed when the commit is unresolved (or `force`); otherwise the block records that git answered, so a Result
    always carries the key and never pays for the hashing twice."""
    git = git_state() if git is None else git
    if not force and git.get("commit") not in (None, "", "unknown"):
        return {"computed": False, "reason": "git_state() resolved the commit", "commit": git.get("commit")}
    try:
        from . import export as _export
        fp = _export.source_fingerprint(include_loaded=True)
    except Exception as e:  # noqa: BLE001 -- never let provenance fail a run
        return {"computed": False, "reason": f"unavailable: {e!r}", "commit": git.get("commit", "unknown")}
    fp["computed"] = True
    fp["reason"] = "git_state() could not resolve the commit: identify this code by these hashes (export.match_sources)"
    return fp


def dataset_release(c=None) -> dict:
    """The MaleCNS release and its four file hashes from flyverse/data/manifest.json."""
    if c is not None and c.dataset != "malecns":
        return {"name": c.dataset, "release": c.release, "files": c._manifest.get("sources", []),
                "pair_threshold": c._manifest.get("pair_threshold"), "edges": c._manifest.get("edges")}
    with open(ROOT / "flyverse" / "data" / "manifest.json", encoding="utf-8") as f:
        m = json.load(f)
    files = [{"path": x["path"], "sha256": x.get("sha256")} for x in m["malecns"]["files"]]
    return {"name": DATASET_NAME, "release": DATASET_RELEASE, "citation": m["malecns"].get("citation"), "files": files}


_FP_CACHE: dict = {}


def connectome_fingerprint(c: cn.Connectome, cache_dir=None) -> dict:
    """md5 of the reference W's data / indices / indptr (= cache/W_post_pre.npz when c is the loaded cache), sum|W|,
    nnz, n_neurons, the NT overrides in force and, for a subset, its size. Memoised per object."""
    ref = c.reference
    key = id(ref)
    # Object IDs can be reused after a scratch graph is collected. Keep a weak
    # identity check so an unrelated graph never inherits its fingerprint.
    import weakref
    if key in _FP_CACHE and _FP_CACHE[key][0]() is not ref:
        del _FP_CACHE[key]
    if key not in _FP_CACHE:
        W = ref.W.tocsr()
        if not W.has_sorted_indices:
            W = W.copy(); W.sort_indices()
        fingerprint = {"md5_data": hashlib.md5(W.data.tobytes()).hexdigest(),
                          "md5_indices": hashlib.md5(W.indices.tobytes()).hexdigest(),
                          "md5_indptr": hashlib.md5(W.indptr.tobytes()).hexdigest(),
                          "md5": _md5_csr(W), "sum_abs_W": float(np.abs(W.data).sum()), "nnz": int(W.nnz), "n_neurons": int(ref.n),
                          "nt_counts": {str(k): int(v) for k, v in ref.neurons["nt"].value_counts().items()} if "nt" in ref.neurons else {}}
        _FP_CACHE[key] = (weakref.ref(ref, lambda unused, k=key: _FP_CACHE.pop(k, None)), fingerprint)
    fp = dict(_FP_CACHE[key][1])
    fp["cache_dir"] = str(cache_dir or getattr(c, "cache_dir", None) or os.environ.get("FLYVERSE_CACHE") or cn.CACHE_DIR)
    fp["type_nt_override"] = dict(cn.TYPE_NT_OVERRIDE) if cn.TYPE_NT_OVERRIDE_DEFAULT else {}
    fp["unknown_nt_override_regex"] = dict(cn.UNKNOWN_NT_OVERRIDE_REGEX)
    fp["subset"] = None if ref is c else {"n": int(c.n), "of": int(ref.n)}
    # Section 4.1 pins the entire legacy MaleCNS fingerprint, including its keys.
    # New releases carry their namespace and compile rules alongside the CSR hashes.
    if c.dataset != "malecns":
        fp.update(dataset=c.dataset, release=c.release, manifest=c._manifest,
                  type_nt_override={}, unknown_nt_override_regex={})
    if c._extension is not None:
        fp["extension"] = to_jsonable(c._extension)
    return fp


def _file_md5(path) -> str | None:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except OSError:
        return None


def model_record(lif=None, optic=None, body: bool = True) -> dict:
    """LIFParams / OpticParams with every default RESOLVED (path_gain, type_path_gain, adapt_by_type, std_u_by_type,
    receptor gain classes, pair_gain, tau_by_type, baseline_by_type), the receptor table md5, and the body thresholds
    a rollout applies (Flight.gf_hz / takeoff_power_hz / takeoff_hold_s, Locomotion.mdn_threshold)."""
    from .. import brain
    p = lif or brain.LIFParams()
    lif_d = dataclasses.asdict(p)
    lif_d["path_gain"] = brain.DEFAULT_PATH_GAIN if p.path_gain is None else p.path_gain
    lif_d["type_path_gain"] = brain.DEFAULT_TYPE_PATH_GAIN if p.type_path_gain is None else p.type_path_gain
    lif_d["adapt_by_type"] = brain.DEFAULT_ADAPT_BY_TYPE if p.adapt_by_type is None else p.adapt_by_type
    lif_d["std_u_by_type"] = brain.DEFAULT_STD_U_BY_TYPE if p.std_u_by_type is None else p.std_u_by_type
    lif_d["receptor_gain_resolved"] = brain._receptor_gain(p)
    lif_d["slow_gain_by_class_resolved"] = brain._slow_gains(p) if p.receptor_model == "full" else None
    lif_d["slow_tau_by_class_resolved"] = brain._slow_taus(p) if p.receptor_model == "full" else None
    table = p.receptor_table or str(cn.RECEPTOR_TABLE)
    out = {"lif": lif_d, "receptor_table": table, "receptor_table_md5": _file_md5(table) if p.receptor_model else None}
    try:
        from .. import optic as optic_mod
        op = optic or optic_mod.OpticParams()
        od = dataclasses.asdict(op)
        od["pair_gain"] = optic_mod.DEFAULT_PAIR_GAIN if op.pair_gain is None else op.pair_gain
        od["tau_by_type"] = optic_mod.DEFAULT_TAU_BY_TYPE if op.tau_by_type is None else op.tau_by_type
        od["baseline_by_type"] = optic_mod.DEFAULT_BASELINE_BY_TYPE if op.baseline_by_type is None else op.baseline_by_type
        out["optic"] = od
    except Exception as e:  # noqa: BLE001
        out["optic"] = {"error": repr(e)}
    if body:
        try:
            from .. import body as body_mod
            fl, lo = body_mod.Flight(), body_mod.Locomotion()
            out["body"] = {"gf_hz": float(fl.gf_hz), "takeoff_power_hz": float(fl.takeoff_power_hz),
                           "takeoff_hold_s": float(fl.takeoff_hold_s), "mdn_threshold_hz": float(lo.mdn_threshold),
                           "k_opto": float(lo.k_opto)}
        except Exception as e:  # noqa: BLE001
            out["body"] = {"error": repr(e)}
    return out


def execution_record(fb=None, device=None, seeds=None, env_seeds=None, batch=None, backend=None, replicate_unit="runs") -> dict:
    """The realised device (fb.brain.device, never the request), backend flags, dt, seeds, batch size, host.

    With no `fb` there is no brain to ask, and `execution.device` used to stay null while the tool's own config block
    held the device it really ran on (`scripts/retire_measures.py`'s cluster JSONs; `Result.check()` then rejects the
    result for a device it was told). A caller with no FlyBrain passes the device it RESOLVED, and it is recorded as
    the realised one -- `device_requested` keeps the string either way."""
    rec = {"device_requested": None if device is None else str(device), "device": None, "device_name": None,
           "backend": dict(backend or {}), "dt": {"lif_ms": None, "optic_ms": OPTIC_DT_MS, "frame_ms": FRAME_MS},
           "seeds": {"brain": to_jsonable(seeds), "env": to_jsonable(env_seeds)}, "batch": batch,
           "replicate_unit": replicate_unit, "host": socket.gethostname(), "platform": platform.platform()}
    if fb is None and device is not None:
        rec["device"] = str(device)                      # no brain to ask: the caller's resolved device IS the record
    try:
        import torch
        rec["torch"] = torch.__version__
        if fb is None and rec["device"] and "cuda" in rec["device"] and torch.cuda.is_available():
            rec["device_name"] = torch.cuda.get_device_name(torch.device(rec["device"]))
        if fb is not None:
            b = getattr(fb, "brain", fb)
            dev = getattr(b, "device", None)
            rec["device"] = str(dev)
            rec["dt"]["lif_ms"] = float(getattr(getattr(b, "p", None), "dt", np.nan))
            rec["batch"] = int(getattr(b, "B", batch or 1)) if batch is None else batch
            rec["backend"].update({"event_driven": bool(getattr(b, "event_driven", False)), "cuda_kernels": bool(getattr(b, "cuda", False)),
                                   "cuda_sparse": getattr(b, "cuda_sparse", None), "metal": bool(getattr(b, "metal", False)),
                                   "cuda_graphs": bool(getattr(fb, "cuda_graphs", False))})
            if dev is not None and getattr(dev, "type", "") == "cuda" and torch.cuda.is_available():
                rec["device_name"] = torch.cuda.get_device_name(dev)
    except Exception as e:  # noqa: BLE001
        rec["torch"] = f"unavailable: {e!r}"
    return rec


def provenance(c: cn.Connectome, lif=None, optic=None, fb=None, device=None, seeds=None, env_seeds=None, batch=None,
               backend=None, stimulus=None, retina=None, cache_dir=None, replicate_unit="runs") -> dict:
    """The mandatory provenance block of every Result (docs/INTERP.md section 3): git, dataset release, compiled
    connectome fingerprint, resolved model parameters, realised execution, stimulus spec, retina record, units.

    `source_fingerprint` is the fallback identity of the code: when `git_state()` cannot resolve the commit (a cluster
    copy with no `.git`), the SHA-256 of every source file the run loaded goes in, so 'commit unknown' is the last
    resort and not the only answer."""
    git = git_state()
    if fb is not None:
        candidate_lif = getattr(getattr(fb, "brain", None), "p", None)
        candidate_optic = getattr(getattr(fb, "optic", None), "p", None)
        lif = lif or (candidate_lif if dataclasses.is_dataclass(candidate_lif) else None)
        optic = optic or (candidate_optic if dataclasses.is_dataclass(candidate_optic) else None)
    model = model_record(lif, optic)
    model.update(dataset=c.dataset, release=c.release)
    if not c.has_optic_columns or (fb is not None and getattr(fb, "optic", None) is None):
        model["optic"] = None
    model["hooks"] = getattr(fb, "hooks", [])
    model["modules"] = fb.module_records() if hasattr(fb, "module_records") else []
    retina_record = to_jsonable(retina) if retina is not None else {"file": None, "n_columns": None, "column_to_bodies": None}
    live_retina = getattr(fb, "retina", None)
    if live_retina is not None and hasattr(live_retina, "coverage"):
        coverage = live_retina.coverage()
        if coverage["without_photoreceptors"]:
            retina_record["coverage"] = coverage
    return {"flyverse_commit": git, "source_fingerprint": source_fingerprint(git), "dataset_release": dataset_release(c),
            "compiled_connectome": connectome_fingerprint(c, cache_dir), "model": model,
            "execution": execution_record(fb, device, seeds, env_seeds, batch, backend, replicate_unit),
            "stimulus": to_jsonable(stimulus) if stimulus is not None else {"protocol": None, "params": {}, "control": None},
            "retina": retina_record,
            "units": UNITS if c.dataset == "malecns" else [
                {"item": "node_set", "rule": f"{c.dataset} {c.release}; retained release neurons; see compiled_connectome.manifest"},
                {"item": "graded", "rule": "ol_intrinsic rate units where an optic module exists; otherwise LIF"},
                {"item": "photoreceptor", "rule": "anatomical photoreceptor identity; optical drive exists only with a retina and optic module"},
                *UNITS[3:]]}


REQUIRED_PROVENANCE = ("flyverse_commit", "dataset_release", "compiled_connectome", "model", "execution", "stimulus", "retina", "units")
# Table names and the columns the export serialises from them (docs/NEUROME_INTERFACE.md section 1).
EXPORT_TABLES = {
    "readout_per_body": ["bodyId", "model_index", "type", "unit_kind", "quantity", "window_start_s", "window_end_s", "stimulus_value",
                         "control_value", "stimulus_minus_control", "unit", "n_trials", "trial_sd", "control_ids"],
    "contributions": ["body_pre", "body_post", "pre_type", "post_type", "value", "kind", "sign_rule", "gain_rule", "normalisation",
                      "reference_graph", "window", "synaptic_pair_count"],
    "sensitivity": ["lesion_id", "lesion_kind", "lesion_spec", "bodies", "check", "baseline", "value", "delta", "replicate_sd",
                    "n_replicates", "replicate_values"],
}


def new_run_id(tool: str) -> str:
    return f"{tool}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"


@dataclass
class Result:
    """The one result of every tool. `tables` are lists of records (DataFrame.to_dict('records')) named per
    EXPORT_TABLES plus tool-specific ones ('per_type', 'links', 'paths', 'matrix', ...); `summary` holds the tool's
    scalars; `validation` names the VALIDATION target and what was measured; `files` the generators and raw files."""
    tool: str
    run_id: str
    provenance: dict
    populations: list = field(default_factory=list)
    replicates: dict = field(default_factory=lambda: {"n": 0, "unit": "runs", "runs": [], "null": None})
    tables: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)
    created_utc: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    schema: str = SCHEMA
    tool_version: str = "0.1"

    @classmethod
    def new(cls, tool: str, provenance: dict, **kw) -> "Result":
        if tool not in TOOLS:
            raise ValueError(f"unknown tool {tool!r}; choose from {TOOLS}")
        r = cls(tool=tool, run_id=new_run_id(tool), provenance=provenance, **kw)
        r.validation = dict(VALIDATION[tool], measured=None, status="not run") | r.validation
        return r

    def add_table(self, name: str, df) -> None:
        self.tables[name] = to_jsonable(df if isinstance(df, list) else pd.DataFrame(df).to_dict("records"))

    def table(self, name: str) -> pd.DataFrame:
        return pd.DataFrame(self.tables.get(name, []))

    def add_population(self, pop: Population, unit_kind: str | None = None, keep_ids: bool = True) -> None:
        rec = pop.record()
        if not keep_ids:
            rec.pop("body_ids")
        rec["unit_kind"] = unit_kind
        self.populations.append(rec)

    def to_dict(self) -> dict:
        return to_jsonable(dataclasses.asdict(self))

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)
        return path

    @classmethod
    def load(cls, path) -> "Result":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("schema") != SCHEMA:
            raise ValueError(f"{path}: schema {d.get('schema')!r} is not {SCHEMA}")
        return cls(**{k: d[k] for k in d if k in {f.name for f in dataclasses.fields(cls)}})

    def check(self) -> list[str]:
        """Problems that would make the export refuse this result (empty list = exportable)."""
        problems = [f"provenance.{k} missing" for k in REQUIRED_PROVENANCE if k not in self.provenance]
        for name, cols in EXPORT_TABLES.items():
            if name in self.tables and self.tables[name]:
                missing = [col for col in cols if col not in self.tables[name][0]]
                if missing:
                    problems.append(f"tables.{name} lacks {missing}")
        ex = self.provenance.get("execution", {})
        if ex.get("device") is None:
            problems.append("execution.device is not the realised device (None)")
        if self.tool not in TOOLS:
            problems.append(f"unknown tool {self.tool!r}")
        return problems


def print_table(df: pd.DataFrame, floatfmt: str = "{:+.3f}", max_rows: int = 60, index: bool = False) -> str:
    """The printed table every CLI ends with (also returned)."""
    if isinstance(df, list):
        df = pd.DataFrame(df)
    s = df.head(max_rows).to_string(index=index, float_format=lambda v: floatfmt.format(v))
    print(s, flush=True)
    return s


# ---------------------------------------------------------------------------------------------- 6. CLI helpers
TUPLE_LIST_KEYS = ("stream_rectify", "stream_adapt", "spatial_suppress", "fb_hold", "pair_gain")


def parse_kv(items) -> dict:
    """['w_syn=0.3', 'path_gain=[]', 'receptor_model=None'] -> {'w_syn': 0.3, 'path_gain': [], 'receptor_model': None}
    (values parsed as JSON, else Python literals, else strings).

    List-of-tuple fields (the OpticParams stream hooks and pair_gain, `TUPLE_LIST_KEYS`) also accept a shell-friendly
    grammar when the value is neither JSON nor a Python literal: entries separated by ';', fields by ',', each field a
    number where it parses as one, else a string (a regex): `stream_rectify=^(Mi1|Tm3|Tm2)$,^T3$,pos;^(Tm1|Tm4)$,^T3$,neg`
    -> [['^(Mi1|Tm3|Tm2)$', '^T3$', 'pos'], ['^(Tm1|Tm4)$', '^T3$', 'neg']]; `spatial_suppress=^(Mi1|Tm1|Tm3|Tm4)$,0.5,10`
    -> [['^(Mi1|Tm1|Tm3|Tm4)$', 0.5, 10.0]]. A value containing ';' takes the grammar under any key. A regex that itself
    contains ',' or ';' needs the JSON form."""
    import ast
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"expected KEY=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        k = k.strip()
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            try:
                out[k] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                out[k] = _parse_tuple_list(v) if (";" in v or (k in TUPLE_LIST_KEYS and "," in v)) else v
    return out


def _parse_tuple_list(v: str) -> list:
    """'a,b,1;c,d,2.5' -> [['a', 'b', 1.0], ['c', 'd', 2.5]] (numbers where a field parses as float, else strings)."""
    def field(s: str):
        s = s.strip()
        try:
            return float(s)
        except ValueError:
            return s
    return [[field(f) for f in entry.split(",")] for entry in v.split(";") if entry.strip()]


def add_common_args(ap) -> None:
    """The flags every scripts/interp_<tool>.py accepts (docs/INTERP.md section 2.6).

    The null arm has ONE convention: `--null-runs GLOB [GLOB ...]` on an `analyse` subcommand names the finished
    control-vs-control runs (the wrappers that label arms accept `LABEL=GLOB`), and a `record` / `run` subcommand
    that has to GENERATE that arm declares its own bare `--null` switch. `--null-arm` / `--null-recordings` stay as
    hidden aliases of `--null-runs` so the round's command lines keep working (docs/INTERP.md 11, defect 5)."""
    import argparse
    ap.add_argument("--json", default=None, help="write the Result JSON here (default out/interp/<tool>/<run_id>.json)")
    ap.add_argument("--replicates", type=int, default=MIN_REPLICATES,
                    help=f"independent runs per arm (default {MIN_REPLICATES}; >= 4 to reach a 'result', 5 for a small effect)")
    ap.add_argument("--seed", type=int, default=0, help="first brain seed; replicates take consecutive seeds")
    ap.add_argument("--null-runs", nargs="*", default=[], metavar="GLOB",
                    help="the matched control-vs-control arm: glob(s) of its finished runs (LABEL=GLOB where arms are named)")
    ap.add_argument("--null-arm", "--null-recordings", nargs="*", dest="null_runs", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--device", default=None, help="torch device request; the JSON records the realised one")
    ap.add_argument("--cache-dir", default=None, help="connectome cache directory (default cache/)")
    ap.add_argument("--receptor-model", default="default", choices=["default", "off", "sign", "sign+gain", "full"])
    ap.add_argument("--receptor-net-rule", default="abs", choices=["class", "abs", "nonmda"])
    ap.add_argument("--receptor-table", default=None, help="a receptors_by_type.csv override (e.g. a hold table)")
    ap.add_argument("--lif", action="append", default=[], metavar="KEY=VALUE", help="LIFParams override (repeatable)")
    ap.add_argument("--optic", action="append", default=[], metavar="KEY=VALUE", help="OpticParams override (repeatable)")
    ap.add_argument("--quiet", action="store_true")


def params_from_args(args):
    """(LIFParams, OpticParams) from the common flags."""
    from .. import brain, optic
    lif_kw = parse_kv(args.lif)
    if args.receptor_model != "default":
        lif_kw["receptor_model"] = None if args.receptor_model == "off" else args.receptor_model
    if args.receptor_model not in ("default", "off"):
        lif_kw["receptor_net_rule"] = args.receptor_net_rule
    if args.receptor_table:
        lif_kw["receptor_table"] = args.receptor_table
    return brain.LIFParams(**lif_kw), optic.OpticParams(**parse_kv(args.optic))


def default_json_path(tool: str, run_id: str) -> Path:
    return ROOT / "out" / "interp" / tool / f"{run_id}.json"
