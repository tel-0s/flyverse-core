import json, pathlib
from collections import Counter
OUT = pathlib.Path(r"D:\Projects\flyverse\out")
def load(n): return json.loads((OUT/n).read_text(encoding="utf-8"))

arms = {"adopt": ["r5_adopt_default_1.json","r5_adopt_default_2.json","r5_adopt_default_3.json"],
        "r4def": ["r4_default_1.json","r4_default_2.json","r4_default_3.json"],
        "r4off": ["r4_off_1.json","r4_off_2.json","r4_off_3.json"]}
data = {a: [load(f) for f in fs] for a, fs in arms.items()}

# header sanity
for a, js in data.items():
    for f, j in zip(arms[a], js):
        c = j["config"]
        print(f"{f:28s} tpg={json.dumps(c['type_path_gain'])} gf_hz={c.get('gf_hz')} dev={c.get('device')} "
              f"rec={c['receptor'].get('model')}/{c['receptor'].get('net_rule')} n_entries="
              f"{c['receptor'].get('entries', c['receptor'].get('n_entries','?'))} rt={j['total_runtime_s']:.0f}s det={c.get('deterministic')} fast={c.get('fast')}")
print()
# receptor entries: find the count field
print("adopt receptor keys:", sorted(data["adopt"][0]["config"]["receptor"].keys()))
r = data["adopt"][0]["config"]["receptor"]
print({k: v for k, v in r.items() if not isinstance(v, (list, dict))})
print()

ORDER = {"PASS": 0, "KNOWN GAP": 1, "FAIL": 2}
def cmap(j): return {c["key"]: c for c in j["checks"]}

keys = sorted(set().union(*[set(cmap(j)) for js in data.values() for j in js]))
print(f"{'check':34s} {'adopt status x3':22s} {'r4def x3':22s} {'r4off x3':22s}")
worse = []
for k in keys:
    row = {}
    for a, js in data.items():
        row[a] = [cmap(j).get(k, {}).get("status", "-") for j in js]
    print(f"{k:34s} {','.join(row['adopt']):22s} {','.join(row['r4def']):22s} {','.join(row['r4off']):22s}")
    # worse in status than ANY r4def / r4off run?  (criterion (b): no check worse in status)
    a_max = max(ORDER.get(s, 9) for s in row["adopt"])
    for ref in ("r4def", "r4off"):
        ref_max = max(ORDER.get(s, -1) for s in row[ref])      # best case for adopt: compare to worst ref
        ref_min = min(ORDER.get(s, 9) for s in row[ref])
        if a_max > ref_max:
            worse.append((k, "adopt worst > %s worst" % ref, row["adopt"], row[ref]))
        if a_max > ref_min:
            worse.append((k, "adopt worst > %s best" % ref, row["adopt"], row[ref]))
print("\nchecks where adopt is worse in status:", worse if worse else "NONE")

# named numbers
named = {"walk.power_max": None, "walk.power_sustained": None, "walk_gf.GF_max": None, "loom.GF_peak": None,
         "loom_escape.GF_peak": None, "motion.min_dsi": None, "taste.MN9_hz": None, "smell.KC_active": None,
         "bitter.MN9_hz": None}
print("\n=== measured values per arm ===")
for k in sorted(set(keys)):
    vals = {}
    for a, js in data.items():
        vals[a] = [cmap(j).get(k, {}).get("measured") for j in js]
    def f(v):
        return ",".join("%.4f" % x if isinstance(x, float) else str(x) for x in v)
    print(f"{k:34s} adopt[{f(vals['adopt'])}] r4def[{f(vals['r4def'])}] r4off[{f(vals['r4off'])}]")
