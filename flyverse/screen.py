"""Condition screens: which cell types (or which side of which type) report a variable of the world?

The method that found the lateral-horn odour population (NOTES, session 8), made reusable:

    rec = TypeRecorder(fb.c, pattern=r"^(LH|MBON|DN[a-z]|WED)")      # or by_side=True for L / R keys
    runs = {name: record(step_fn, rec, seconds=30) for name, step_fn in conditions.items()}
    table = rank(runs, positive=["fruit_into_wind", "fruit_away"], negative=["clean_into_wind", "clean_away"], rec)

`rank` scores every key by d' between the *worst* positive condition and the *best* negative one (so a
heading confound shows up as a low score), and by the fraction of time a midpoint threshold separates
them. `contrast` gives the plain difference of two conditions for interaction questions (e.g. does a
type's L - R asymmetry flip with wind side more under odour than without?). `ablate` / `restore`
silence a selection on a `FlyBrain` so a candidate can be tested causally, and `FlyBrain.stimulate`
does the converse. Everything works on `Brain.rate_np()` snapshots, so it costs nothing extra per frame.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TypeRecorder:
    keys: np.ndarray        # one label per recorded population
    idx: np.ndarray         # neuron indices (all recorded cells)
    inv: np.ndarray         # population id of each recorded cell
    n_cells: np.ndarray

    @classmethod
    def build(cls, c, pattern: str | None = None, types=None, by_side: bool = False) -> "TypeRecorder":
        t = c.neurons.type.fillna("").to_numpy()
        if types is not None:
            m = np.isin(t, list(types))
        elif pattern is not None:
            rx = re.compile(pattern); m = np.array([bool(rx.match(x)) for x in t])
        else:
            m = t != ""
        idx = np.flatnonzero(m)
        labels = t[idx]
        if by_side:
            side = c.neurons.somaSide.fillna("?").to_numpy()[idx]
            labels = np.array([f"{a}_{b}" for a, b in zip(labels, side)])
        keys, inv = np.unique(labels, return_inverse=True)
        return cls(keys=keys, idx=idx, inv=inv, n_cells=np.bincount(inv))

    def snapshot(self, rate_np: np.ndarray) -> np.ndarray:
        """(n_keys,) mean rate per population from one (N,) rate vector."""
        return np.bincount(self.inv, weights=rate_np[self.idx], minlength=len(self.keys)) / np.maximum(self.n_cells, 1)


def record(step_fn, rec: TypeRecorder, seconds: float, hz: float = 100.0) -> np.ndarray:
    """Call `step_fn()` (which advances the simulation one frame and returns an (N,) rate vector)
    `seconds * hz` times; returns (T, n_keys) population means."""
    n = int(round(seconds * hz)); out = np.zeros((n, len(rec.keys)), np.float32)
    for k in range(n):
        out[k] = rec.snapshot(step_fn())
    return out


def _smooth(x: np.ndarray, tau_s: float, hz: float = 100.0) -> np.ndarray:
    if tau_s <= 0:
        return x
    a = np.exp(-1.0 / (tau_s * hz)); y = np.empty_like(x); acc = x[0].copy()
    for k in range(len(x)):
        acc = a * acc + (1 - a) * x[k]; y[k] = acc
    return y


def rank(runs: dict, positive: list, negative: list, rec: TypeRecorder, smooth_s: float = 1.0, skip_s: float = 3.0,
         hz: float = 100.0, min_cells: int = 1, min_hz: float = 3.0) -> pd.DataFrame:
    """Score every population by how well it separates the positive conditions from the negative ones."""
    S = {k: _smooth(v, smooth_s, hz)[int(skip_s * hz):] for k, v in runs.items()}
    mu = {k: v.mean(0) for k, v in S.items()}; sd = {k: v.std(0) for k, v in S.items()}
    pos = np.min([mu[k] for k in positive], 0); neg = np.max([mu[k] for k in negative], 0)
    pooled = np.sqrt(np.mean([sd[k] ** 2 for k in S], 0)) + 1.0
    thr = 0.5 * (pos + neg)
    frac_pos = np.mean([np.mean(S[k] > thr, 0) for k in positive], 0)
    frac_neg = np.mean([np.mean(S[k] > thr, 0) for k in negative], 0)
    df = pd.DataFrame({"key": rec.keys, "cells": rec.n_cells, "d_prime": (pos - neg) / pooled,
                       "pos_min_hz": pos, "neg_max_hz": neg, "above_thr_pos": frac_pos, "above_thr_neg": frac_neg})
    for k in runs:
        df[k] = mu[k]
    df = df[(df.cells >= min_cells) & (df.pos_min_hz >= min_hz)]
    return df.sort_values("d_prime", ascending=False).reset_index(drop=True)


def contrast(runs: dict, a: str, b: str, rec: TypeRecorder, smooth_s: float = 1.0, skip_s: float = 3.0, hz: float = 100.0) -> pd.DataFrame:
    """Mean(a) - mean(b) per population, with a pooled-sd d'."""
    A = _smooth(runs[a], smooth_s, hz)[int(skip_s * hz):]; B = _smooth(runs[b], smooth_s, hz)[int(skip_s * hz):]
    pooled = np.sqrt(0.5 * (A.std(0) ** 2 + B.std(0) ** 2)) + 1.0
    return pd.DataFrame({"key": rec.keys, "cells": rec.n_cells, "diff_hz": A.mean(0) - B.mean(0), "d_prime": (A.mean(0) - B.mean(0)) / pooled,
                         a: A.mean(0), b: B.mean(0)}).sort_values("d_prime", ascending=False).reset_index(drop=True)


def lateral_pairs(rec: TypeRecorder):
    """For a by_side recorder: (type, index of _L key, index of _R key) for every type with both sides."""
    pos = {k: i for i, k in enumerate(rec.keys)}
    out = []
    for k, i in pos.items():
        if k.endswith("_L") and k[:-2] + "_R" in pos:
            out.append((k[:-2], i, pos[k[:-2] + "_R"]))
    return out


def ablate(fb, selection) -> np.ndarray:
    """Silence a selection (indices, boolean mask, or Connectome.select criteria) on a FlyBrain; returns the indices."""
    idx = _resolve(fb, selection)
    fb.brain.active[fb.brain._idx(idx)] = 0.0
    return idx


def restore(fb, idx) -> None:
    fb.brain.active[fb.brain._idx(np.asarray(idx))] = 1.0


def _resolve(fb, selection) -> np.ndarray:
    if isinstance(selection, dict):
        return fb.c.select(**selection)
    sel = np.asarray(selection)
    return np.flatnonzero(sel) if sel.dtype == bool else sel.astype(int)
