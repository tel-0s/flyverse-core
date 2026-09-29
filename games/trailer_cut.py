"""Build the announcement trailer's edit decision list (games/trailer_edl.json) from the soundtrack's beat map and the
games' recorded clips, then render it with games/trailer.py.

    python games/trailer_cut.py                 # writes games/trailer_edl.json
    python games/trailer.py games/trailer_edl.json

The soundtrack is not in the repository. Fetch it into out/trailer/music/ (the EDL's "music" path):

    curl -L -o out/trailer/music/zarathustra_macleod.ogg
        https://upload.wikimedia.org/wikipedia/commons/0/0b/Also_Sprach_Zarathustra_-_Einleitung.ogg
    ffmpeg -i out/trailer/music/zarathustra_macleod.ogg -ar 48000 -ac 2 -c:a pcm_s16le out/trailer/music/zarathustra.wav

(one curl command, split here for width). The copy the trailer was cut to has SHA-256
79d036e9f564450bf2b7973877be16131ca8b6174cb3971f5c173c1ce482a8ef (Vorbis, 44.1 kHz stereo, 86.06 s).

"Also Sprach Zarathustra" by Kevin MacLeod (incompetech.com), licensed under Creative Commons: By Attribution 3.0
(http://creativecommons.org/licenses/by/3.0/). The beat map (games/trailer_assets/beatmap.json) was measured from that
file by games/trailer_assets/analyze_track.py.

Every clip shot is placed by one rule: `at(event_time_in_clip, beat_time)`. The shot's source offset is chosen so a
logged event (a kill, a takeoff, a point) lands on a musical event from games/trailer_assets/beatmap.json. Speeds are
1.0. The clips each shot comes from, and their seeds and devices, are listed in games/README.md ("The trailer").
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEAT = json.loads((ROOT / "games/trailer_assets/beatmap.json").read_text(encoding="utf-8"))
EV = {e["id"]: e["t"] for e in BEAT["events"]}
TIMP1 = [s["t"] for s in BEAT["timpani_detail"]["after_call1"]["strokes"]]
TIMP2 = [s["t"] for s in BEAT["timpani_detail"]["after_call2"]["strokes"]]
DUR = 86.0


def clip(src, t0, t1, *, event=None, at=None, src_t0=None, speed=1.0, **kw):
    """A clip shot on [t0, t1). Either `src_t0` directly, or `event` (seconds in the clip) placed at `at` (seconds on
    the trailer timeline)."""
    if src_t0 is None:
        src_t0 = (event - (at - t0) * speed) if event is not None else 0.0
    if src_t0 < 0:
        raise ValueError(f"{src}: shot starts before the clip ({src_t0:.2f} s)")
    return {"kind": "clip", "src": str(src), "t0": round(t0, 3), "t1": round(t1, 3), "src_t0": round(src_t0, 3),
            "speed": speed, **kw}


def card(t0, t1, lines, style="title", **kw):
    return {"kind": "card", "t0": t0, "t1": t1, "lines": lines, "style": style, **kw}


def build(shots):
    shots = sorted(shots, key=lambda s: s["t0"])
    for a, b in zip(shots, shots[1:]):
        if b["t0"] < a["t1"] - 1e-6:
            raise ValueError(f"overlap: {a['t0']}-{a['t1']} and {b['t0']}-{b['t1']}")
    return {"fps": 50, "size": [1920, 1080], "duration": DUR, "music": "out/trailer/music/zarathustra.wav",
            "music_start": 0.0, "audio_fade_out": 2.0, "out": "out/trailer/flyverse_trailer.mp4", "shots": shots}


if __name__ == "__main__":
    import importlib
    spec = importlib.import_module("trailer_shots")          # games/trailer_shots.py: the actual cut (edited by hand)
    edl = build(spec.shots(EV, TIMP1, TIMP2, clip, card))
    out = ROOT / "games/trailer_edl.json"
    out.write_text(json.dumps(edl, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(edl['shots'])} shots, {edl['duration']} s)")
