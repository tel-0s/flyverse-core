"""The announcement trailer's cut: which recorded clip plays where, aligned to the beat map. Read by trailer_cut.py.

Every source is a recorded seed clip under out/games/<name>/ (the recording policy in games/README.md). Every shot
plays at 1x. Beat times come from out/trailer/music/beatmap.json (`ev`, `t1`, `t2`). Event times inside clips come
from each clip's own run log, and the log event is named next to each number.
"""

G = "out/games"

# Crops of each game's main view (x, y, w, h on the 1920x1080 canvas; 16:9), for the fast cuts.
CROP = {
    "doom": [12, 68, 1312, 738],
    "pong": [180, 76, 1560, 878],
    "minecraft": [0, 64, 1536, 864],
    "spacecraft": [300, 64, 1360, 765],
    "swarm": [0, 64, 1344, 756],
    "swat": [0, 64, 1380, 776],
    "hairdryer": [0, 64, 1380, 776],
}

# 3x2 grid cells are about 629x532 (1.18:1): crops of each main view with that aspect
CELL = {
    "spacecraft": [380, 70, 1000, 846], "pong": [400, 70, 1120, 948], "minecraft": [120, 64, 1100, 931],
    "doom": [96, 68, 1136, 962], "swarm": [120, 64, 1100, 931], "swat": [140, 64, 1100, 931],
}
SPLIT = [420, 64, 900, 1016]          # the 2x1 split's cells are about 947x1064 (0.89:1): the ship, portrait

# tron and mars, seed 0: filled from their verified run logs (placeholders until then)
TRON = "tron/rec2/seed0"   # the recorder's batch (out/games/tron/rec2/, fetched from the cluster)
TRON_TURN = 4.17         # seed 0: MaleCNS 'GIANT FIBRE 34 Hz -> TURN RIGHT', 1.3 m to the wall (event 'turn', player A)
TRON_REVEAL = 13.71      # MaleCNS turns right (33 Hz, 1.1 m to the wall); FAFB derezzes at 13.82 s, MaleCNS takes round 2
TRON_CLIMAX = 26.54      # MaleCNS turns right (35 Hz, 0.7 m); FAFB crashes at 26.77 s, MaleCNS takes round 3
MARS_VISTA = 0.0         # seed 0 opens on a moving crane shot (0.0-3.2 s): sun, dust devil, rover, title plate
MARS_EVENT = 32.24       # seed 0: 'WIND DNs -> VEER LEFT' (DNp18 L 35 / R 7 Hz), away from the dust devil

# hairdryer, seed 0: the brain second of first apple contact and of the FEEDING banner, from
# out/games/hairdryer/seed0.json (filled in once the run log exists)
HD_CONTACT = 20.48       # event "contact"
HD_FEED = 21.70          # the FEEDING banner first shows at clip 21.68 s (logged event at 21.47 s)


def shots(ev, t1, t2, clip, card):
    S = []
    W = (1920 / 2, 1080 / 2)

    # ---- the low C pedal: darkness, the numbers (sunrise, seed 0; its lamp switches on at 12.02 s, the first C)
    S.append(clip(f"{G}/sunrise/seed0.mp4", 0.0, ev["call1_G4"], src_t0=0.0, fade_in=2.4, texts=[
        {"t0": 1.8, "t1": 4.7, "text": "167,106 neurons", "size": 104},
        {"t0": 4.9, "t1": 7.8, "text": "25.6 million connections", "size": 104},
        {"t0": 8.0, "t1": 11.8, "text": "one fly's brain and nerve cord, synapse by synapse", "size": 64},
        {"t0": 8.3, "t1": 11.8, "text": "MaleCNS v1.0 · Janelia FlyEM + Google Research, 2026", "size": 30, "bold": False,
         "pos": [W[0], W[1] + 70], "color": [151, 171, 172]},
        {"t0": 12.25, "t1": 13.95, "text": "the room's light comes on · every glow is the model's own activity", "size": 30,
         "bold": False, "pos": [W[0], 1020], "color": [213, 223, 219]},
    ]))
    # ---- call 1, G: the fly on its table (swat, seed 0, before the first swing)
    S.append(clip(f"{G}/swat/seed0.mp4", ev["call1_G4"], ev["call1_C5"], src_t0=0.0,
                  label="A FLY ON A TABLE", sub="the same brain, given eyes, antennae, taste and a body"))
    # ---- call 1, held C: the ship (spacecraft seed 0: tumbling from t = 0; below 15 deg/s at clip 4.24 s, and the
    # ATTITUDE HOLD banner fades in from clip 4.32 s, so the shot runs 0.5-4.54 s to include it)
    S.append(clip(f"{G}/spacecraft/seed0.mp4", ev["call1_C5"], ev["chord1_hit"], src_t0=0.5,
                  label="SPACECRAFT", sub="declared decoders read its HS / H2 and VST2 cells and fire the attitude thrusters"))
    # ---- chord 1: the title
    S.append(card(ev["chord1_hit"], ev["chord1_release"], ["flyverse", "a whole fly connectome, in strange places",
                                                           "0.2.0 · open source"], style="end", fade_in=0.05, flash=True))
    S.append({"kind": "black", "t0": ev["chord1_release"], "t1": t1[0]})
    # ---- timpani 1: nine strokes, nine worlds
    fast1 = [("doom/seed0", "doom", 7.28), ("pong/seed2", "pong", 16.4), ("minecraft/seed0", "minecraft", 4.85),
             ("spacecraft/seed0", "spacecraft", 8.7), (TRON, None, 3.8), ("swat/seed0", "swat", 2.22),
             ("hairdryer/seed0", "hairdryer", HD_CONTACT - 3.0), ("sunrise/seed0", None, 12.15), ("doom/seed0", "doom", 15.3)]
    ends = t1[1:] + [ev["call2_C4"]]
    for (src, crop, at), a, b in zip(fast1, t1, ends):
        S.append(clip(f"{G}/{src}.mp4", a, b, src_t0=at, crop=CROP.get(crop) if crop else None, flash=(a == t1[0])))
    # ---- call 2: Pong, Minecraft, Doom
    S.append(clip(f"{G}/pong/seed2.mp4", ev["call2_C4"], ev["call2_G4"], src_t0=15.1, label="PONG",
                  sub="MaleCNS v1.0 vs FlyWire FAFB v783 · paddles follow decoders of each fly's own L2 + Mi1 columns"))
    # minecraft seed 0: the side phantom, GIANT FIBRE 38 Hz -> JUMP (gf_cross 26.54 s, jump 26.55 s), landed before the C
    S.append(clip(f"{G}/minecraft/seed0.mp4", ev["call2_G4"], ev["call2_C5"], event=26.55, at=ev["call2_C5"] - 0.45,
                  label="MINECRAFT", sub="a real 1.21.4 server · 1,466 ommatidia ray-cast through its blocks · giant fibre → jump"))
    # doom seed 0: trigger at 15.34 s (GF 47 Hz), KILL #2 at 15.45 s, YOU DIED at 15.59 s
    S.append(clip(f"{G}/doom/seed0.mp4", ev["call2_C5"], ev["chord2_hit"], event=15.34, at=ev["call2_C5"] + 2.2,
                  label="DOOM", sub="the gun fires when the fly's giant fibre does"))
    # ---- chord 2 (minor -> major): six worlds, desaturated, then in colour on the third's switch
    S.append({"kind": "grid", "t0": ev["chord2_hit"], "t1": ev["chord2_release"], "cols": 3, "rows": 2, "flash": True,
              "sat0": 0.12, "sat1": 1.0, "sat_ramp": 0.9,
              "cells": [{"src": f"{G}/spacecraft/seed0.mp4", "src_t0": 3.0, "tag": "SPACECRAFT", "crop": CELL["spacecraft"]},
                        {"src": f"{G}/pong/seed2.mp4", "src_t0": 17.0, "tag": "PONG", "crop": CELL["pong"]},
                        {"src": f"{G}/minecraft/seed0.mp4", "src_t0": 24.0, "tag": "MINECRAFT", "crop": CELL["minecraft"]},
                        {"src": f"{G}/doom/seed1.mp4", "src_t0": 17.2, "tag": "DOOM", "crop": CELL["doom"]},
                        {"src": f"{G}/{TRON}.mp4", "src_t0": TRON_TURN - 2.0, "tag": "TRON"},
                        {"src": f"{G}/swat/seed0.mp4", "src_t0": 4.5, "tag": "SWAT", "crop": CELL["swat"]}],
              "texts": [{"t0": 0.6, "t1": 2.55, "text": "raw connectomes · no weights touched", "size": 64},
                        {"t0": 2.7, "t1": 4.9, "text": "every game control is a declared read-out", "size": 64}]})
    S.append({"kind": "black", "t0": ev["chord2_release"], "t1": t2[0]})
    # ---- timpani 2: ten strokes
    fast2 = [("swat/seed0", "swat", 5.1), ("doom/seed1", "doom", 12.2), ("spacecraft/seed0", "spacecraft", 14.0),
             ("minecraft/seed0", "minecraft", 26.35), ("pong/seed0", "pong", 8.4), ("mars/seed0", None, MARS_EVENT - 0.5),
             ("hairdryer/seed0", "hairdryer", HD_CONTACT - 1.0), ("swat/seed0", "swat", 13.45), ("doom/seed2", "doom", 26.3),
             ("sunrise/seed0", None, 13.0)]
    ends = t2[1:] + [ev["call3_C4"]]
    for (src, crop, at), a, b in zip(fast2, t2, ends):
        S.append(clip(f"{G}/{src}.mp4", a, b, src_t0=at, crop=CROP.get(crop) if crop else None, flash=(a == t2[0])))
    # ---- call 3: hairdryer, swat, TRON through the held note; the Mars vista on the F-major tutti
    S.append(clip(f"{G}/hairdryer/seed0.mp4", ev["call3_C4"], ev["call3_G4"], event=HD_CONTACT, at=ev["call3_G4"] - 0.2,
                  label="HAIRDRYER", sub="a scripted hand aims the jet · the fly's wind DNs → a declared turn decoder · sugar → MN9"))
    # swat seed 0: swing 1 (0.5 m/s from above) takeoff at 2.29 s, 100 ms before the paddle reached its spot
    S.append(clip(f"{G}/swat/seed0.mp4", ev["call3_G4"], ev["call3_C5"], event=2.29, at=ev["call3_G4"] + 1.2,
                  label="SWAT", sub="no decoders: LC4 / LPLC2 → giant fibre → the body model's escape jump"))
    S.append(clip(f"{G}/{TRON}.mp4", ev["call3_C5"], ev["chord3_hit"], event=TRON_REVEAL, at=ev["chord3_hit"] - 0.6,
                  label="TRON", sub="MaleCNS vs FAFB · a declared decoder turns each cycle when its giant fibre crosses 33 Hz"))
    S.append(clip(f"{G}/mars/seed0.mp4", ev["chord3_hit"], ev["answer_change2"], src_t0=MARS_VISTA, flash=True,
                  label="MARS", sub="a dust devil's wind → the fly's wind neurons → a declared decoder veers the rover"))
    # ---- the answer phrase: best moments on the harmony changes
    S.append({"kind": "grid", "t0": ev["answer_change2"], "t1": ev["answer_change3"], "cols": 2, "rows": 1, "gap": 10,
              "cells": [{"src": f"{G}/spacecraft/b200_seed0.mp4", "src_t0": 2.6, "tag": "DECODERS ON", "crop": SPLIT},
                        {"src": f"{G}/spacecraft/control_seed0.mp4", "src_t0": 2.6, "tag": "CONTROL: THRUSTERS OFF", "crop": SPLIT}],
              "texts": [{"t0": 0.2, "t1": 1.85, "text": "same seed, same GPU model · thrusters on, and off", "size": 44,
                         "pos": [W[0], 1000]}]})
    # pong seed 0: RALLY 10 at the 200 deg/s cap (8.61 s), then MaleCNS misses by 0.18 deg (8.94 s)
    S.append(clip(f"{G}/pong/seed0.mp4", ev["answer_change3"], ev["answer_change4"], event=8.61, at=ev["answer_change4"] - 0.5))
    S.append(clip(f"{G}/doom/seed1.mp4", ev["answer_change4"], ev["build_start"], src_t0=17.4, texts=[
        {"t0": 0.1, "t1": 1.4, "text": "the chips say what is the fly and what is not: CONNECTOME · DECODER · GAME", "size": 48, "pos": [W[0], 990]}]))
    # ---- the build: accelerating cuts, ending on the swat approach
    bounds = [63.25, 64.62, 65.68, 66.62, 67.20, 67.98, 68.40, 68.98, 69.24, 69.68, 69.98, 70.28, 70.58, 70.88, 71.18]
    build = [("minecraft/seed0", None, 0.3), ("spacecraft/seed0", None, 13.4), ("pong/seed2", None, 25.4),
             ("doom/seed0", None, 24.5), ("mars/seed0", None, MARS_EVENT - 1.0), ("hairdryer/seed0", None, HD_CONTACT - 2.0),
             ("swat/seed0", None, 7.7), ("minecraft/seed0", "minecraft", 21.0), ("pong/seed2", "pong", 17.8),
             ("doom/seed2", "doom", 26.35), ("spacecraft/seed0", "spacecraft", 28.2), (TRON, None, TRON_TURN + 2.0),
             ("swat/seed0", "swat", 10.8), ("sunrise/seed0", None, 12.1)]
    for (src, crop, at), a, b in zip(build, bounds, bounds[1:]):
        S.append(clip(f"{G}/{src}.mp4", a, b, src_t0=at, crop=CROP.get(crop) if crop else None))
    S.append(clip(f"{G}/swat/seed0.mp4", 71.18, 72.32, event=2.29, at=ev["climax"]))
    S.append({"kind": "white", "t0": 72.32, "t1": ev["climax"]})
    # ---- the climax: one best moment per world
    c = ev["climax"]
    S.append(clip(f"{G}/swat/seed0.mp4", c, 73.52, event=2.29, at=c + 0.02, flash=True))
    # minecraft seed 2: the frontal phantom dive, GIANT FIBRE 44 Hz -> JUMP (21.10 s), GF peak 94.8 Hz
    S.append(clip(f"{G}/minecraft/seed2.mp4", 73.52, 74.44, event=21.10, at=73.80))
    # doom seed 1: GIANT FIBRE 40 Hz -> FIRE -> KILL #4 (17.62 s)
    S.append(clip(f"{G}/doom/seed1.mp4", 74.44, 75.32, event=17.62, at=74.72))
    S.append(clip(f"{G}/{TRON}.mp4", 75.32, 76.20, event=TRON_CLIMAX, at=75.50))
    # pong seed 2: RALLY 10 at 200 deg/s (17.56 s)
    S.append(clip(f"{G}/pong/seed2.mp4", 76.20, 77.00, event=17.56, at=76.30))
    S.append(clip(f"{G}/hairdryer/seed0.mp4", 77.00, 77.80, event=HD_FEED, at=77.10))
    S.append(clip(f"{G}/mars/seed0.mp4", 77.80, 78.56, event=MARS_EVENT, at=78.0))
    S.append(clip(f"{G}/sunrise/seed0.mp4", 78.56, ev["orchestra_release"], src_t0=12.0))
    # ---- the organ tail: the end card
    S.append(card(ev["orchestra_release"], 86.0, [
        "flyverse", "github.com/tel-0s/flyverse-core",
        "pip install -e .", "python scripts/fetch_data.py --malecns --fafb", "python games/pong.py",
        "every clip is seed 0, 1 or 2 · every game control is declared · the captions are in games/",
        "Connectome: MaleCNS v1.0 (Janelia FlyEM + Google Research) · FlyWire FAFB v783",
        "Music: \"Also Sprach Zarathustra\" Kevin MacLeod (incompetech.com) · CC BY 3.0"],
        style="end", n_big=5, commands=[2, 3, 4], fade_in=0.6, fade_out=0.8))
    return S
