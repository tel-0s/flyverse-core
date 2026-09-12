"""Round-5 attribution table: the check values of the r5-attr batch side by side (Brain-side vs optic-side holds).

    PYTHONIOENCODING=utf-8 python scripts/r5_attr_report.py                      # markdown table + the attribution verdicts
    PYTHONIOENCODING=utf-8 python scripts/r5_attr_report.py --md out/r5_attr_table.md

Reads out/r5_attr_{default_1,holdBrain_1,holdBrain_2,holdOptic_1,holdOptic_2,off_1}.json (cluster batch
`r5-attr-6aa260`, generator scripts/r5_attr_batch.sh). Each arm's receptor header is printed so the condition
label is checkable in the file, as in round 4. For every check it prints the four conditions and, where the
default differs from off, which SIDE reproduces the default: a value is attributed to the Brain side when
holdOptic (Brain-side entries only) lands on the default and holdBrain (optic-side only) on off, and to the
optic side in the mirror case; anything else is 'neither / interacting'.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARMS = ["default_1", "holdBrain_1", "holdBrain_2", "holdOptic_1", "holdOptic_2", "off_1"]
# what each arm applies
APPLIES = {"default_1": "all 48,295", "holdBrain_1": "optic 44,463", "holdBrain_2": "optic 44,463",
           "holdOptic_1": "Brain 3,832", "holdOptic_2": "Brain 3,832", "off_1": "none"}


def load(out_dir: str) -> dict:
    d = {}
    for a in ARMS:
        p = os.path.join(out_dir, f"r5_attr_{a}.json")
        if not os.path.isfile(p):
            sys.exit(f"missing {p} -- run scripts/r5_attr_batch.sh and fetch out/")
        d[a] = json.load(open(p, encoding="utf-8"))
    return d


def condition(arm: str) -> str:
    return arm.rsplit("_", 1)[0]


def draws(dirs: list, key: str) -> dict:
    """Every draw of every condition, pooled over the given run directories: {condition: [values]}."""
    out = {}
    for out_dir in dirs:
        d = load(out_dir)
        for a in ARMS:
            v, _ = value(d[a], key)
            if v is not None:
                out.setdefault(condition(a), []).append(v)
    return out


def value(j, key):
    for c in j["checks"]:
        if c["key"] == key:
            return c["measured"], c["status"]
    return None, None


def side(default, brain_hold, optic_hold, off, tol):
    """Which side reproduces the default? brain_hold = holdBrain (optic-side entries only)."""
    if default is None or off is None or abs(default - off) <= tol:
        return "no default/off difference"
    near = lambda a, b: abs(a - b) <= tol
    # holdOptic applies the Brain side; holdBrain applies the optic side
    b_like_def = all(near(v, default) for v in optic_hold)
    b_like_off = all(near(v, off) for v in optic_hold)
    o_like_def = all(near(v, default) for v in brain_hold)
    o_like_off = all(near(v, off) for v in brain_hold)
    if b_like_def and o_like_off:
        return "BRAIN side"
    if o_like_def and b_like_off:
        return "OPTIC side"
    if b_like_def and o_like_def:
        return "either side alone (saturating)"
    if b_like_off and o_like_off:
        return "neither side alone"
    frac = lambda v: (v - off) / (default - off)
    return ("partial: holdOptic(Brain) " + "/".join(f"{frac(v):+.2f}" for v in optic_hold) +
            ", holdBrain(optic) " + "/".join(f"{frac(v):+.2f}" for v in brain_hold) + " of the gap")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "out"))
    ap.add_argument("--md", default="")
    ap.add_argument("--dirs", default="", help="extra out-dirs of independent run directories, comma separated (e.g. out/r5_attr_dup)")
    ap.add_argument("--tol", type=float, default=1e-4, help="two values count as equal within this (bit-stable checks)")
    a = ap.parse_args()
    d = load(a.out_dir)
    print("-- arms --")
    for arm in ARMS:
        r = d[arm]["config"]["receptor"]
        print(f"{arm:14s} applies {APPLIES[arm]:14s} model {str(r.get('model')):5s} net_rule {str(r.get('net_rule')):5s} "
              f"flag {r.get('flag')!r:10s} changed_entries {r.get('fast_sign_changed_entries', 0)} "
              f"table {os.path.basename(str(r.get('table')))} runtime {d[arm]['total_runtime_s']:.0f}s")
    keys = []
    for arm in ARMS:
        for c in d[arm]["checks"]:
            if c["key"] not in keys:
                keys.append(c["key"])
    hdr = ["check", "criterion", "default (all)", "holdBrain x2 (optic only)", "holdOptic x2 (Brain only)", "off", "carried by"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    print("\n-- checks --")
    for key in keys:
        vd, sd = value(d["default_1"], key)
        vo, so = value(d["off_1"], key)
        bh = [value(d[x], key)[0] for x in ("holdBrain_1", "holdBrain_2")]
        oh = [value(d[x], key)[0] for x in ("holdOptic_1", "holdOptic_2")]
        crit = next((c["criterion"] for c in d["default_1"]["checks"] if c["key"] == key), "")
        if None in bh or None in oh or vd is None or vo is None:
            verdict = "arm missing"
        else:
            verdict = side(vd, bh, oh, vo, a.tol)
        fmt = lambda v, s=None: "absent" if v is None else (f"{v:.4f}" + (f" {s[0]}" if s else ""))
        row = [f"`{key}`", crit, fmt(vd, sd),
               " / ".join(fmt(v) for v in bh), " / ".join(fmt(v) for v in oh), fmt(vo, so), verdict]
        lines.append("| " + " | ".join(row) + " |")
        print(f"{key:34s} {crit:12s} def {fmt(vd, sd):>12s}  holdBrain {' / '.join(fmt(v) for v in bh):>21s}  "
              f"holdOptic {' / '.join(fmt(v) for v in oh):>21s}  off {fmt(vo, so):>12s}   {verdict}")
    print("\n-- tallies --")
    for arm in ARMS:
        t = {}
        for c in d[arm]["checks"]:
            t[c["status"]] = t.get(c["status"], 0) + 1
        print(f"{arm:14s} {t}")
    if a.dirs:
        dirs = [a.out_dir] + [x for x in a.dirs.split(",") if x]
        print(f"\n-- scatter over {len(dirs)} independent run directories ({', '.join(dirs)}) --")
        for key in keys:
            dr = draws(dirs, key)
            print(f"{key:34s} " + "  ".join(
                f"{c} {'/'.join(f'{v:.4f}' for v in vs)}" for c, vs in dr.items()))
    if a.md:
        with open(a.md, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nwrote {a.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
