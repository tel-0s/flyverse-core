"""Summarise games/mars.py run logs (no brain, no GPU): the outcome numbers the caption is written from.

    python games/mars_assets/summarize.py out/games/mars/seed0.json out/games/mars/nowind_seed0.json ...
    python games/mars_assets/summarize.py --events out/games/mars/seed0.json      # also list the log's events

Per log: device, brain / wall seconds, the tally (hazards passed clean / after a collision, collisions, route
completion), the giant fibre's maximum and crossings, the wind steer's onsets, veers (with direction, the DN rates and
whether they pointed away from dust devil D3), the overrides (held ticks and episodes, cap ticks), every devil's
closest approach and the D3 encounter's excursion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(path, events=False):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    m, s, ev = d["meta"], d["summary"], d["events"]
    arm = m.get("mars", {}).get("control", "?")
    print("=" * 110)
    print(f"{path}  arm={arm} seed={m.get('seed')}  device={m.get('device_name')}  torch={m.get('torch')}  "
          f"brain_s={m.get('brain_s')}  wall_s={m.get('wall_s')}  sources={m.get('sources')}")
    print(f"  eye_cuda_graph={m.get('eye_cuda_graph')}")
    print(f"  finished={s.get('finished')} x_end={s.get('x_end_m')}  passed {s.get('passed')}/{s.get('hazards')}: "
          f"clean {s.get('passed_clean')}, after collision {s.get('passed_after_collision')}; collisions "
          f"{s.get('collisions')} (hazard contacts {s.get('contacts')}); not reached {s.get('not_reached')}")
    print(f"  GF max {s.get('gf_max')} Hz, crossings {s.get('gf_crossings')}, stops applied {s.get('hazard_stops_applied')};"
          f" MDN max {s.get('mdn_max')} (t<1 s: {s.get('mdn_max_t_lt_1s')}); loom L/R max {s.get('loomL_max')} / "
          f"{s.get('loomR_max')}")
    print(f"  wind steer: onsets {s.get('steer_onsets')}, veers {s.get('veers')}, active ticks "
          f"{s.get('steer_active_ticks')}, u_s [{s.get('u_s_min')}, {s.get('u_s_max')}] Hz, |yaw cmd| max "
          f"{s.get('yaw_dec_abs_max_deg_s')} deg/s, applied turn {s.get('steer_applied_abs_deg')} deg; wind (Earth "
          f"equiv.) max {s.get('wind_equiv_max_m_s')} m/s")
    print(f"  overrides: held ticks {s.get('steer_held_ticks')} in {s.get('steer_held_episodes')} collision "
          f"recoveries ({s.get('steer_held_abs_deg')} deg of turn withheld); cap ticks {s.get('steer_cap_ticks')}")
    print(f"  devils' closest approach: {s.get('devils_closest')}")
    print(f"  D3 encounter: {s.get('D3_encounter')}")
    for h in s.get("per_hazard", []):
        print(f"    {h['name']} x {h['x']:.1f} y {h['y']:+.2f} r {h['r']:.2f}: {h['outcome']}, contacts {h['contacts']}, "
              f"min clearance {h['min_clear_m']}, passed at {h['t_passed']} s")
    for e in ev:
        if e["kind"] in ("veer", "gf_cross", "encounter_start"):
            print("   ", {k: v for k, v in e.items()})
    if events:
        for e in ev:
            print("     ", e)
    return d


def row(path):
    """One markdown table row of the caption's per-clip outcomes."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    m, s, ev = d["meta"], d["summary"], d["events"]
    cols = {c: k for k, c in enumerate(s["series_cols"])}
    ser = s["series_10hz"]
    veers = [e for e in ev if e["kind"] == "veer"]
    v = ["%.2f s %s%s%s" % (e["t_s"], e["direction"], "" if e["applied"] else (" (held)" if e["held"] else
                                                                              " (not applied)"),
                            ", away from D3" if e["away_from_D3"] else ("" if e["away_from_D3"] is None else
                                                                          ", towards D3's side")) for e in veers]
    first = veers[0] if veers else None
    dn = ("DNp18 %.0f / %.0f, DNp33 %.0f / %.0f" % (first["DNp18_L"], first["DNp18_R"], first["DNp33_L"],
                                                   first["DNp33_R"])) if first else "-"
    # the excursion between 30 s (the rover is back on the line after H2 in every arm) and the next collision (H3)
    tcol = min([e["t_s"] for e in ev if e["kind"] == "collision" and e["t_s"] > 30.0] + [1e9])
    seg = [r for r in ser if 30.0 <= r[0] <= tcol]
    exc = "heading %+.1f deg, y %+.2f m" % (max(r[cols["yaw_deg"]] for r in seg), max(r[cols["y_m"]] for r in seg))
    d3 = s["devils_closest"]["D3"]
    passed = [h for h in s["per_hazard"] if h["outcome"] is not None]
    return ("| %s | %s | %.1f / %.0f | %s (x %.1f m) | %d of %d, %d clean, %d after a collision | %d | %s | %s | %s | "
            "%.1f m at %.1f s (rover y %+.2f m) | %d ticks, %d episodes, %.1f deg | %d | %.1f Hz, %d | %.1f to %.1f Hz |"
            % (Path(path).name, m["mars"]["control"], m["brain_s"], m["wall_s"], "yes" if s["finished"] else "no",
               s["x_end_m"], len(passed), s["hazards"], s["passed_clean"], s["passed_after_collision"],
               s["collisions"], "; ".join(v) or "none", dn, exc, d3["distance_m"], d3["t_s"], d3["rover_y_m"],
               s["steer_held_ticks"], s["steer_held_episodes"], s["steer_held_abs_deg"], s["steer_cap_ticks"],
               s["gf_max"], s["gf_crossings"], s["u_s_min"], s["u_s_max"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--events", action="store_true")
    ap.add_argument("--table", action="store_true", help="print the caption's markdown rows instead")
    a = ap.parse_args()
    if a.table:
        print("| clip | arm | brain / wall s | finished | boulders passed | collisions | veers | DN rates at the "
              "first veer (L / R, Hz) | max heading and left offset, 30 s to the H3 collision | D3 closest | held | cap ticks | "
              "GF max, crossings | u_s range |")
        print("|" + "---|" * 14)
        for p in a.logs:
            print(row(p))
        return
    for p in a.logs:
        summarize(p, a.events)


if __name__ == "__main__":
    main()
