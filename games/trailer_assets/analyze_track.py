"""Beat map of the trailer soundtrack: Kevin MacLeod, "Also Sprach Zarathustra (Sonnenaufgang)", CC BY 3.0.

    python games/trailer_assets/analyze_track.py      # reads out/trailer/music/zarathustra.wav

The WAV is fetched and decoded as games/trailer_cut.py's docstring says. Writes beatmap.json next to this file and a
diagnostic plot to out/trailer/music/beatmap.png. numpy / scipy / matplotlib only.

Method (a listening proxy):
* onsets: SuperFlux-style spectral flux (log magnitude, 3-bin max filter on the reference frame, 20 ms lag) on a
  2048-point STFT at a 10 ms hop, 30 Hz - 16 kHz, normalised to the track's maximum; adaptive peak picking (local
  max +-50 ms, above the +-0.5 s moving median plus max(0.025, 2 x the moving MAD), 80 ms refractory).
* trumpet line: harmonic-sum pitch salience (8192-point STFT, 5 ms hop, MIDI 58-86, 5 harmonics, octave check)
  plus the rise of each note's own partials.
* chord quality: the major third (E3/E4/E5) against the minor third (Eb3/Eb4/Eb5) in the chord.
* timpani: SuperFlux in 40-400 Hz at a 2 ms hop; each stroke's pitch from the spectrum gained after it
  (G2 ~98 Hz vs C3 ~131-135 Hz).
The search windows for each label come from the score's known order (pedal; three C-G-C calls, each followed by a
chord; timpani after calls 1 and 2; the third call rises E-F into the build; C-major climax; organ alone) and a first
look at the spectrogram. Every time inside a window is measured, not typed in.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wf
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import maximum_filter1d, median_filter, uniform_filter1d

HERE = Path(__file__).resolve().parent
MUSIC = HERE.parents[1] / "out/trailer/music"
WAV = MUSIC / "zarathustra.wav"

SR, X = wf.read(WAV)
X = X.astype(np.float32) / 32768.0
M = X.mean(1)
DUR = len(M) / SR


def stft_mag(sig, n, hop, fmax=None):
    """|STFT| with frame i centred at i*hop/SR; magnitude scaled so a full-scale sine peaks at ~1."""
    pad = np.pad(sig, (n // 2, n // 2))
    frames = sliding_window_view(pad, n)[::hop]
    win = np.hanning(n).astype(np.float32)
    f = np.fft.rfftfreq(n, 1 / SR)
    kmax = len(f) if fmax is None else int(np.searchsorted(f, fmax)) + 1
    out = np.empty((kmax, len(frames)), np.float32)
    for i in range(0, len(frames), 1024):
        blk = frames[i:i + 1024] * win
        out[:, i:i + len(blk)] = np.abs(np.fft.rfft(blk, axis=1))[:, :kmax].T
    out /= win.sum() / 2
    return f[:kmax], np.arange(len(frames)) * hop / SR, out


def superflux(S, f, lo, hi, lag):
    b = (f >= lo) & (f < hi)
    L = np.log1p(100 * S[b])
    ref = maximum_filter1d(L, size=3, axis=0)
    d = np.zeros(S.shape[1], np.float32)
    d[lag:] = np.maximum(0, L[:, lag:] - ref[:, :-lag]).mean(0)
    return d


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def nname(midi):
    return f"{NAMES[midi % 12]}{midi // 12 - 1}"


def r3(v):
    return round(float(v), 3)


# ----------------------------------------------------------------------------------------------------- onsets
fA, tA, SA = stft_mag(M, 2048, 480)
FPS_A = SR / 480
odf = superflux(SA, fA, 30, 16000, lag=2)
odf_low = superflux(SA, fA, 30, 250, lag=2)
odf_mid = superflux(SA, fA, 250, 2000, lag=2)
odf_high = superflux(SA, fA, 2000, 16000, lag=2)
odf_n = odf / odf.max()


def pick_peaks(env, fps, local=0.05, med_win=0.5, k=2.0, delta=0.025, wait=0.08):
    """Local maxima above a robust moving threshold: median(+-med_win) + max(delta, k * MAD)."""
    n = len(env)
    lmax = maximum_filter1d(env, size=2 * int(round(local * fps)) + 1)
    w = 2 * int(round(med_win * fps)) + 1
    med = median_filter(env, size=w, mode="nearest")
    mad = 1.4826 * median_filter(np.abs(env - med), size=w, mode="nearest")
    thr = med + np.maximum(delta, k * mad)
    peaks, last = [], -1e9
    for i in range(1, n - 1):
        if env[i] == lmax[i] and env[i] >= thr[i] and (i - last) / fps >= wait:
            peaks.append(i)
            last = i
    return np.array(peaks, int), thr


pk, odf_thr = pick_peaks(odf_n, FPS_A)
N_LOW, N_MID, N_HIGH = ((fA >= 30) & (fA < 250)).sum(), ((fA >= 250) & (fA < 2000)).sum(), ((fA >= 2000) & (fA < 16000)).sum()
onsets = []
for i in pk:
    parts = np.array([odf_low[i] * N_LOW, odf_mid[i] * N_MID, odf_high[i] * N_HIGH])   # summed flux per band
    onsets.append({"t": r3(tA[i]), "strength": round(float(odf_n[i]), 4),
                   "band": ["low", "mid", "high"][int(np.argmax(parts))]})

# ------------------------------------------------------------------------------------- fine spectra for labels
fB, tB, SB = stft_mag(M, 4096, 96, fmax=6000)        # 2 ms hop, 11.7 Hz bins
FPS_B = SR / 96
PB = SB ** 2
fC, tC, SC = stft_mag(M, 8192, 240, fmax=5000)       # 5 ms hop, 5.9 Hz bins (pitch)


def note_db(midi, harmonics=(1,), P=PB, f=fB):
    e = 0
    for h in harmonics:
        fm = hz(midi) * h
        b = (f > fm * 2 ** (-0.4 / 12)) & (f < fm * 2 ** (0.4 / 12))
        e = e + P[b].sum(0)
    return 10 * np.log10(e + 1e-14)


def band_db(lo, hi, P=PB, f=fB):
    b = (f >= lo) & (f < hi)
    return 10 * np.log10(P[b].sum(0) + 1e-14)


def smooth(s, sec, fps=FPS_B):
    return uniform_filter1d(s, max(1, int(sec * fps)))


def steepest(s, a, b, t=tB, rising=True, sec=0.05):
    """Time of the steepest rise (or fall) of a dB curve inside [a, b], and the slope in dB/s."""
    d = np.gradient(smooth(s, sec)) * FPS_B
    i = np.where((t >= a) & (t <= b))[0]
    j = i[np.argmax(d[i])] if rising else i[np.argmin(d[i])]
    return float(t[j]), float(d[j])


def level(s, a, b, t=tB):
    return float(np.median(s[(t >= a) & (t <= b)]))


def odf_peak(a, b, env=odf_n, t=tA):
    i = np.where((t >= a) & (t <= b))[0]
    j = i[np.argmax(env[i])]
    return float(t[j]), float(env[j])


# trumpet pitch track (harmonic-sum salience, reported where the trumpet is the top tonal line)
MIDI_T = np.arange(58, 87)          # Bb3 up: the trumpet register (a held G3 under call 2 is not the trumpet)
sal = np.zeros((len(MIDI_T), len(tC)), np.float32)
for k, mm in enumerate(MIDI_T):
    for h, w in zip(range(1, 6), (1.0, 0.8, 0.6, 0.45, 0.35)):
        fm = hz(mm) * h
        b = (fC > fm * 2 ** (-0.4 / 12)) & (fC < fm * 2 ** (0.4 / 12))
        if b.any():
            sal[k] += w * SC[b].max(0)
best = np.argmax(sal, 0)
# octave check: prefer the lower octave when its own fundamental is within 10 dB of the chosen note's
fund = np.stack([SC[np.argmin(np.abs(fC - hz(mm)))] for mm in MIDI_T])
for i in range(len(tC)):
    k = best[i]
    if k >= 12 and fund[k - 12, i] > 0.316 * fund[k, i]:
        best[i] = k - 12
pitch = MIDI_T[best].astype(float)
pitch_conf = sal.max(0) / (np.sort(sal, 0)[-2] + 1e-9)
pitch_med = np.round(uniform_filter1d(pitch, 21))  # ~0.1 s smoothing for display


def dominant_note(a, b):
    i = (tC >= a) & (tC <= b)
    vals, counts = np.unique(pitch[i], return_counts=True)
    j = np.argmax(counts)
    return int(vals[j]), float(counts[j] / i.sum())


C4, G4, C5, G5, C6 = (note_db(k) for k in (60, 67, 72, 79, 84))
E_maj = 10 * np.log10(sum(10 ** (note_db(k) / 10) for k in (52, 64, 76)))
E_min = 10 * np.log10(sum(10 ** (note_db(k) / 10) for k in (51, 63, 75)))
BB = band_db(200, 4000)
LOW = band_db(25, 120)

events, sections = [], []


def ev(id_, t, label, conf, evidence, **kw):
    d = {"id": id_, "t": r3(t), "frame_50fps": int(round(t * 50)), "label": label, "confidence": conf,
         "evidence": evidence}
    d.update(kw)
    events.append(d)
    return d


def sec(id_, t0, t1, label, conf, evidence, **kw):
    d = {"id": id_, "t_start": r3(t0), "t_end": r3(t1), "label": label, "confidence": conf, "evidence": evidence}
    d.update(kw)
    sections.append(d)
    return d


# ------------------------------------------------------------------------------------------------------ pedal
t_ped, _ = steepest(LOW, 0.0, 2.0)
ev("pedal_start", t_ped, "Low C pedal enters (C1 ~32.7 Hz with its harmonics; organ / contrabass register)", 0.95,
   f"25-120 Hz energy rises ~35 dB from the -62 dBFS floor; steepest at {t_ped:.2f} s; C1 fundamental dominates "
   "the spectrum through 12 s with no trumpet partials")

# anchors (s) from a first look at the spectrogram: (C4, G4, C5, chord) per call; each label is measured inside
# a window around its anchor
ANCH = [(12.05, 14.05, 16.0, 20.0), (29.1, 31.1, 33.1, 37.45), (46.9, 48.75, 50.6, 54.6)]


def hit_time(a, b):
    """A chord attack: the first picked onset in [a, b] reaching half the strongest one there (the attack's start,
    not the later peak of its upper partials). Returns (t, strength, t_peak, strength_peak)."""
    t_pk, s_pk = odf_peak(a, b)
    cands = [o for o in onsets if a <= o["t"] <= t_pk + 1e-6 and o["strength"] >= 0.5 * s_pk]
    if not cands:
        return t_pk, s_pk, t_pk, s_pk
    return cands[0]["t"], cands[0]["strength"], t_pk, s_pk


# chords after calls 1 and 2 (the order of the thirds is what distinguishes them)
chords = {}
for n, a in enumerate([ANCH[0][3], ANCH[1][3]], start=1):
    t_hit, s_hit, t_pk, s_pk = hit_time(a - 0.25, a + 0.25)
    t_bb, _ = steepest(BB, a - 0.3, a + 0.3)
    early = (t_hit + 0.05, t_hit + 0.35)
    late = (t_hit + 1.0, t_hit + 4.0)
    maj_e, min_e = level(E_maj, *early), level(E_min, *early)
    maj_l, min_l = level(E_maj, *late), level(E_min, *late)
    first = "major" if maj_e > min_e else "minor"
    second = "major" if maj_l > min_l else "minor"
    diff = smooth(E_maj - E_min, 0.04)
    i = np.where((tB >= t_hit) & (tB <= t_hit + 1.2))[0]
    sgn = np.sign(diff[i])
    flips = np.where(np.diff(sgn) != 0)[0]
    t_switch = float(tB[i][flips[0] + 1]) if len(flips) else float("nan")
    t_rel, _ = steepest(band_db(1500, 6000), t_hit + 4.0, t_hit + 6.0, rising=False, sec=0.1)
    chords[n] = (t_hit, t_switch, t_rel, first, second)
    ev(f"chord{n}_hit", t_hit, f"Chord {n}: tutti C {first} (third {'E' if first == 'major' else 'Eb'})", 0.95,
       f"attack onset {s_hit:.2f}, flux peak {s_pk:.2f} at {t_pk:.2f} s, steepest 200-4000 Hz rise {t_bb:.2f} s; "
       f"thirds just after the hit: E {maj_e:.0f} dB vs Eb {min_e:.0f} dB", quality=first, t_peak=r3(t_pk))
    ev(f"chord{n}_switch", t_switch, f"Chord {n}: third moves -> C {second}", 0.85,
       f"major-minus-minor third energy changes sign at {t_switch:.2f} s; held part: E {maj_l:.0f} dB vs "
       f"Eb {min_l:.0f} dB", quality=second)
    ev(f"chord{n}_release", t_rel, f"Chord {n} released (upper partials decay)", 0.7,
       "steepest fall of 1.5-6 kHz energy; the release is a decay, so +-0.2 s", quality=None)
    sec(f"chord{n}", t_hit, t_rel, f"Chord {n}: C {first} -> C {second}", 0.9,
        "hit, third switch and release as the events of the same name", quality=f"{first}->{second}")

# timpani after calls 1 and 2
fT, tT, ST = fB, tB, SB
tflux = superflux(ST, fT, 40, 400, lag=15)   # 30 ms lag at the 2 ms hop
tflux_s = uniform_filter1d(tflux, 5)


def stroke_pitch(t0):
    def spec(a, b):
        s = M[int(a * SR):int(b * SR)].astype(np.float64)
        F = np.abs(np.fft.rfft(s * np.hanning(len(s)), n=1 << 16)) ** 2
        return np.fft.rfftfreq(1 << 16, 1 / SR), F
    fr, A = spec(t0 + 0.01, t0 + 0.22)
    _, B = spec(t0 - 0.20, t0 - 0.01)

    def gain(f0):
        b = (fr > f0 * 0.97) & (fr < f0 * 1.04)
        return 10 * np.log10((A[b].sum() + 1e-12) / (B[b].sum() + 1e-12))
    g, c = gain(98.0), gain(133.0)
    if max(g, c) < 3 or abs(g - c) < 3:
        return None, g, c
    return ("G2" if g > c else "C3"), g, c


timp = {}
for n, (a, b) in enumerate([(chords[1][2], ANCH[1][0]), (chords[2][2], ANCH[2][0])], start=1):
    i = np.where((tT >= a - 0.4) & (tT <= b - 0.05))[0]
    env = tflux_s[i]
    lm = maximum_filter1d(env, size=int(0.2 * FPS_B) | 1)
    cand = [i[k] for k in range(1, len(env) - 1) if env[k] == lm[k] and env[k] > np.percentile(env, 80)]
    strokes, unpitched = [], []
    for j in cand:
        p, g, c = stroke_pitch(float(tT[j]))
        d = {"t": r3(tT[j]), "pitch": p, "gain_G2_db": round(g, 1), "gain_C3_db": round(c, 1),
             "flux": round(float(tflux_s[j] / tflux_s[i].max()), 2)}
        (strokes if p is not None else unpitched).append(d)
    # the regular figure: the longest run with alternating pitch and 0.3-0.45 s spacing
    best_run = []
    for s0 in range(len(strokes)):
        run = [strokes[s0]]
        for s in strokes[s0 + 1:]:
            dt = s["t"] - run[-1]["t"]
            if dt < 0.28:
                continue
            if 0.28 <= dt <= 0.46 and s["pitch"] != run[-1]["pitch"]:
                run.append(s)
            elif dt > 0.46:
                break
        if len(run) > len(best_run):
            best_run = run
    pickups = sorted([s for s in strokes + unpitched if s["t"] < best_run[0]["t"] - 0.1 and s["flux"] >= 0.25],
                     key=lambda s: s["t"])
    timp[n] = (best_run, pickups)
    for k, s in enumerate(best_run, start=1):
        ev(f"timp{n}_stroke{k}", s["t"], f"Timpani {n}, stroke {k} ({s['pitch']})", 0.85,
           f"40-400 Hz flux peak; pitch from the post-stroke spectral gain (G2 {s['gain_G2_db']:+.1f} dB, "
           f"C3 {s['gain_C3_db']:+.1f} dB)", pitch=s["pitch"], group=n)
    for s in pickups:
        ev(f"timp{n}_pickup_{s['t']:.2f}", s["t"], f"Timpani {n}: possible stroke ({s['pitch'] or 'pitch unclear'}) "
           "under the chord release", 0.4 if s["pitch"] else 0.3,
           "40-400 Hz flux peak before the regular figure, overlapping the chord's decay; the same two-stroke lead-in "
           "appears ~0.5 s before stroke 1 in both passages", pitch=s["pitch"], group=n)
    sec(f"timpani{n}", best_run[0]["t"], best_run[-1]["t"] + 0.2,
        f"Timpani {n}: {len(best_run)} strokes alternating G2 / C3, ending on {best_run[-1]['pitch']}", 0.85,
        f"mean spacing {np.mean(np.diff([s['t'] for s in best_run])):.3f} s")

# ---------------------------------------------------------------------------------------- three trumpet calls
calls = []
for n, (ac, ag, acc, ach) in enumerate(ANCH, start=1):
    # C4: the rise of its fundamental plus its 3rd partial (G5); the window opens after the last timpani stroke,
    # whose C3 shares the C4 partial
    lo = ac - 0.3 if n == 1 else max(ac - 0.3, timp[n - 1][0][-1]["t"] + 0.1)
    c4_sig = 10 * np.log10(10 ** (C4 / 10) + 10 ** (G5 / 10))
    t_c4, _ = steepest(c4_sig, lo, ac + 0.3)
    t_g4, _ = steepest(G4, ag - 0.3, ag + 0.3)
    t_c5, _ = steepest(10 * np.log10(10 ** (C5 / 10) + 10 ** (C6 / 10)), acc - 0.3, acc + 0.3)
    # pitch-track check: dominant salient note in the middle of each held note
    chk = [dominant_note(t_c4 + 0.3, t_g4 - 0.2), dominant_note(t_g4 + 0.3, t_c5 - 0.2),
           dominant_note(t_c5 + 0.3, t_c5 + 1.5)]
    calls.append({"call": n, "C4": t_c4, "G4": t_g4, "C5": t_c5, "track": chk})
    for key, t_, (mm, frac), want in zip(("C4", "G4", "C5"), (t_c4, t_g4, t_c5), chk, (60, 67, 72)):
        ok = mm % 12 == want % 12
        conf = 0.9 if ok and frac > 0.5 else (0.75 if ok else 0.6)
        ev(f"call{n}_{key}", t_, f"Trumpet call {n}: {key[0]} ({key}, {hz(want):.0f} Hz)", conf,
           f"steepest rise of the {key} partials; pitch track over the held note = {nname(mm)} "
           f"({frac:.0%} of frames)", call=n, note=key)

# ------------------------------------------------------------------------------- third call, build and climax
t_e, _ = steepest(E_maj, 54.2, 54.9)
t_f_hit, s_f, t_f_pk, s_f_pk = hit_time(54.8, 55.3)
t_f, _ = steepest(note_db(65, (1, 2)), 54.8, 55.3)
ev("call3_E", t_e, "Trumpet call 3 continues: E over a C-major swell (the tutti starts to enter)", 0.75,
   "rise of the E3/E4/E5 partials after the third C5; 200-4000 Hz energy starts rising 54.6 s", call=3, note="E")
ev("chord3_hit", t_f_hit, "Chord 3: tutti F major (trumpet E -> F; IV of C)", 0.85,
   f"flux {s_f:.2f}, one of the three strongest attacks in the track; F/A/C chroma, F2-F3 bass from 55.0 s; "
   f"F partial rise {t_f:.2f} s", quality="major (F, IV)", t_peak=r3(t_f_pk))


def chroma_novelty():
    b = (fC > 100) & (fC < 2000)
    fr = fC[b]
    pc = np.round(12 * np.log2(fr / 440.0) + 69).astype(int) % 12
    C = np.zeros((12, len(tC)), np.float32)
    for k in range(12):
        C[k] = (SC[b][pc == k] ** 2).sum(0)
    C = C / (np.linalg.norm(C, axis=0) + 1e-12)
    w = int(0.4 * 200)
    nov = np.zeros(len(tC))
    for i in range(w, len(tC) - w):
        a_ = C[:, i - w:i].mean(1)
        b_ = C[:, i:i + w].mean(1)
        nov[i] = 1 - a_ @ b_ / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-12)
    return C, nov


CHROMA, NOV = chroma_novelty()


def chord_name(a, b):
    i = (tC >= a) & (tC <= b)
    c = CHROMA[:, i].mean(1)
    top = np.argsort(-c)[:4]
    return " ".join(NAMES[k] for k in top)


# chord changes of the answer phrase between the F chord and the big hit that starts the build
t_build, s_build, t_build_pk, _ = hit_time(63.0, 63.6)
i = np.where((tC >= t_f_hit + 1.0) & (tC <= t_build - 0.4))[0]
lm = maximum_filter1d(NOV[i], size=int(1.2 * 200) | 1)
chg = [float(tC[i][k]) for k in range(len(i)) if NOV[i][k] == lm[k] and NOV[i][k] > 0.08]
answer_marks = [t_f_hit] + chg + [t_build]
for k, t_ in enumerate(chg, start=1):
    ev(f"answer_change{k}", t_, f"Answer phrase: harmony changes (top pitch classes after: "
       f"{chord_name(t_ + 0.1, t_ + 1.0)})", 0.6, "chroma novelty peak (0.4 s windows); harmonic change under "
       "sustained texture, no sharp attack")
sec("answer", t_f_hit, t_build, "Answer phrase after call 3: F major and its continuation (F - F/C - Am - Dm/G7 "
    "colours)", 0.65, "chroma per segment; chord names are approximate (dense synthesized orchestra)",
    changes=[r3(t) for t in chg])

ev("build_start", t_build, "Build begins: tutti hit, C/G (cadential 6-4) then a held dominant G with crescendo", 0.8,
   f"onset-strength peak {s_build:.2f} (one of the three strongest); G4/G5/G6 dominate 64-69.8 s over a G bass")
t_a, _ = steepest(note_db(69, (1, 2)), 69.6, 70.4, sec=0.12)
t_b, _ = steepest(note_db(71, (1, 2)), 71.0, 71.8, sec=0.12)
ev("build_A", t_a, "Build: top line rises G -> A", 0.7, "A4/A5 partials rise; A strongest pitch class 70.0-71.0 s")
ev("build_B", t_b, "Build: top line rises A -> B over G major (V)", 0.7,
   "B4/B5 partials rise in two steps (a soft one ~71.1 s, the main one ~71.35-71.6 s); B/D/G strongest 71.5-72.2 s")
t_clx, s_clx, t_clx_pk, s_clx_pk = hit_time(72.1, 72.7)
t_clx_c, _ = steepest(C5, 72.1, 72.7)
ev("climax", t_clx, "CLIMAX: C major, full orchestra + organ (top line B -> C)", 0.95,
   f"the strongest attack in the track (flux {s_clx_pk:.2f} at {t_clx_pk:.2f} s); C5 partial rise {t_clx_c:.2f} s; "
   "C/E/G chroma from 72.4 s; C2 enters in the bass", t_peak=r3(t_clx_pk))
sec("build", t_build, t_clx, "Build: dominant held, top line G -> A -> B", 0.8, "events build_*")
t_orch_off, _ = steepest(band_db(1500, 6000), 79.0, 80.5, rising=False, sec=0.1)
sec("climax_hold", t_clx, t_orch_off, "Climax chord held (C major)", 0.9, "C/E/G chroma throughout")
ev("orchestra_release", t_orch_off, "Orchestra releases; the organ holds on alone", 0.8,
   "steepest fall of 1.5-6 kHz energy; afterwards only the low C and G3 register sustain and decay smoothly")
w = int(0.05 * SR)
rms_db = 20 * np.log10(np.sqrt(np.convolve(M.astype(np.float64) ** 2, np.ones(w) / w, "same")) + 1e-9)
t_m40 = float(np.where(rms_db > -40)[0][-1] / SR)
t_m60 = float(np.where(rms_db > -60)[0][-1] / SR)
ev("tail_below_-40dBFS", t_m40, "Organ tail falls below -40 dBFS", 0.9, "50 ms RMS")
ev("audible_end", t_m60, "Organ tail falls below -60 dBFS (effective end)", 0.9, "50 ms RMS")
sec("organ_tail", t_orch_off, DUR, "Organ tail (organ alone, decaying to silence)", 0.75,
    "instrument identity (organ) is from the score; the audio shows a sustained low C + G3 decaying ~8.5 dB/s")
sec("pedal", t_ped, calls[0]["C4"], "Low C pedal (darkness)", 0.95, "events pedal_start, call1_C4")
for n, c in enumerate(calls, start=1):
    end = chords[n][0] if n < 3 else t_f_hit
    sec(f"call{n}", c["C4"], end, f"Trumpet call {n}: C-G-C" + (" (then E-F)" if n == 3 else ""), 0.9,
        f"pitch track {', '.join(nname(mm) for mm, _ in c['track'])}")

for s_ in sections:          # a timpani passage ends where the next trumpet call begins
    if s_["id"] in ("timpani1", "timpani2"):
        s_["t_end"] = min(s_["t_end"], r3(calls[int(s_["id"][-1])]["C4"]))
events.sort(key=lambda d: d["t"])
sections.sort(key=lambda d: (d["t_start"], d["t_end"]))
# cross-reference: which picked onsets coincide (+-40 ms) with a labelled event
for o in onsets:
    near = [e["id"] for e in events if abs(e["t"] - o["t"]) <= 0.04 and "pickup" not in e["id"]]
    if near:
        o["event"] = near[0]

out = {
    "track": {
        "file": "out/trailer/music/zarathustra.wav",
        "source_file": "out/trailer/music/zarathustra_macleod.ogg",
        "title": "Also Sprach Zarathustra (Sonnenaufgang)",
        "title_as_tagged_in_file": "Also Sprach Zarathustra - Sonnenaufgang",
        "commons_file_title": "Also Sprach Zarathustra - Einleitung",
        "composer": "Richard Strauss, Also sprach Zarathustra op. 30 (1896), Einleitung",
        "performer_arranger": "Kevin MacLeod",
        "year": 2010,
        "licence": "CC BY 3.0 (Creative Commons Attribution 3.0 Unported)",
        "licence_url": "https://creativecommons.org/licenses/by/3.0/",
        "source_url": "https://commons.wikimedia.org/wiki/File:Also_Sprach_Zarathustra_-_Einleitung.ogg",
        "original_source_url": "http://incompetech.com/music/royalty-free/index.html?keywords=Zarathustra",
        "attribution_line": ('"Also Sprach Zarathustra (Sonnenaufgang)" Kevin MacLeod (incompetech.com) / Licensed '
                             'under Creative Commons: By Attribution 3.0 License / '
                             'http://creativecommons.org/licenses/by/3.0/'),
        "attribution_notes": [
            "Commons page (checked 2026-09-28): Author Kevin MacLeod, Date 2010, Source incompetech.com (link above), "
            "licence template {{CC-BY-3.0}}; no custom attribution string is specified.",
            "incompetech.com's own credit template is: \"<Title>\" Kevin MacLeod (incompetech.com) / Licensed under "
            "Creative Commons: By Attribution 4.0 License / http://creativecommons.org/licenses/by/4.0/ ; the line "
            "above is that template with the 3.0 licence this copy was obtained under.",
            "The piece is no longer in incompetech.com's catalogue (pieces.json, 1,443 titles, checked 2026-09-28); "
            "only 'Fanfare for Space' (2013), described by MacLeod as 'a take-off on Strauss' Zarathustra theme that "
            "is legal for use in every country'. The Commons CC BY 3.0 grant stands; the remark concerns the "
            "composition's copyright in countries with terms longer than life+70 (Strauss died 1949).",
            "CC BY 3.0 treats synchronising the music to video as an Adaptation (s.1a); s.4b asks for the author, "
            "the title, the licence URI and a credit identifying the use, e.g. 'Music: ...' on the end card.",
        ],
    },
    "duration_s": r3(DUR),
    "sample_rate_hz": int(SR),
    "channels": 2,
    "peak_dbfs": round(float(20 * np.log10(np.abs(X).max() + 1e-12)), 2),
    "method": {
        "onsets": "SuperFlux spectral flux, 2048-pt STFT, 10 ms hop, 30 Hz-16 kHz, log1p(100|X|), 3-bin max "
                  "filter, 20 ms lag, normalised to the track maximum; peaks: local max +-50 ms, >= moving median "
                  "(+-0.5 s) + max(0.025, 2 x moving MAD), 80 ms refractory; band = low <250 Hz, mid 250-2000, "
                  "high >2000 (largest summed flux at the peak); event = labelled event within 40 ms",
        "labels": "see analyze_track.py docstring; each event carries its evidence and a 0-1 confidence",
        "times": "seconds from the first sample of zarathustra.wav (= the .ogg); frame_50fps = round(t*50)",
    },
    "onsets": onsets,
    "sections": sections,
    "events": events,
    "timpani_detail": {f"after_call{n}": {"strokes": v[0], "possible_pickups": v[1]} for n, v in timp.items()},
}
(HERE / "beatmap.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

# --------------------------------------------------------------------------------------------------- figure
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
COL = {"pedal": "#4a3aa7", "call": "#2a78d6", "chord": "#eb6834", "timp": "#1baf7a", "answer": "#e87ba4",
       "build": "#eda100", "climax": "#e34948", "tail": "#8a8984"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(26, 15.5), facecolor="#fcfcfb")
gs = fig.add_gridspec(5, 2, height_ratios=[1.35, 0.8, 1.2, 1.0, 1.0], hspace=0.42, wspace=0.08)
ax_w = fig.add_subplot(gs[0, :])
ax_o = fig.add_subplot(gs[1, :], sharex=ax_w)
ax_p = fig.add_subplot(gs[2, :], sharex=ax_w)
ax_t1 = fig.add_subplot(gs[3, 0])
ax_t2 = fig.add_subplot(gs[3, 1])
ax_c = fig.add_subplot(gs[4, :])

# 1. waveform envelope + labelled spans
hop = 480
nb = len(M) // hop
env = np.abs(M[:nb * hop]).reshape(nb, hop)
tt = (np.arange(nb) + 0.5) * hop / SR
ax_w.fill_between(tt, -env.max(1), env.max(1), color="#b7d3f6", lw=0)
ax_w.fill_between(tt, -np.sqrt((env ** 2).mean(1)), np.sqrt((env ** 2).mean(1)), color="#2a78d6", lw=0)
spans = [("pedal", "pedal", "LOW C PEDAL")] + [(f"call{n}", "call", f"CALL {n}\nC-G-C") for n in (1, 2, 3)] + \
    [(f"chord{n}", "chord", f"CHORD {n}\n{'maj->min' if n == 1 else 'min->maj'}") for n in (1, 2)] + \
    [(f"timpani{n}", "timp", f"TIMPANI {n}\n{len(timp[n][0])} strokes") for n in (1, 2)] + \
    [("answer", "answer", "F MAJOR +\nANSWER"), ("build", "build", "BUILD\nG->A->B"),
     ("climax_hold", "climax", "CLIMAX\nC MAJOR"), ("organ_tail", "tail", "ORGAN\nTAIL")]
secd = {s["id"]: s for s in sections}
for sid, ck, text in spans:
    s = secd[sid]
    ax_w.axvspan(s["t_start"], s["t_end"], ymin=0.86, ymax=1.0, color=COL[ck], alpha=0.9, lw=0)
    ax_w.text((s["t_start"] + s["t_end"]) / 2, 1.07, text, ha="center", va="bottom", fontsize=9, color=INK,
              transform=ax_w.get_xaxis_transform())
key_ids = ["pedal_start", "call1_C4", "call1_G4", "call1_C5", "chord1_hit", "call2_C4", "call2_G4", "call2_C5",
           "chord2_hit", "call3_C4", "call3_G4", "call3_C5", "chord3_hit", "build_start", "build_A", "build_B",
           "climax", "orchestra_release", "audible_end"]
evd = {e["id"]: e for e in events}
short = {"pedal_start": "pedal", "chord1_hit": "C maj", "chord2_hit": "C min", "chord3_hit": "F maj",
         "build_start": "C/G hit", "build_A": "A", "build_B": "B", "climax": "C MAJOR", "orchestra_release": "orch off",
         "audible_end": "-60 dB"}
prev_t, lvl = -9.0, 0
for k in key_ids:
    e = evd[k]
    lvl = (lvl + 1) % 3 if e["t"] - prev_t < 2.2 else 0
    prev_t = e["t"]
    ck = "call" if k.startswith("call") else ("chord" if "chord" in k else ("climax" if k == "climax" else
                                                                             ("build" if "build" in k else "tail")))
    if k == "pedal_start":
        ck = "pedal"
    ax_w.axvline(e["t"], color=COL[ck], lw=1.4, ymax=0.86)
    lab = (short[k] if k in short else k.split("_")[-1])
    ax_w.text(e["t"] + 0.12, -0.95 + 0.27 * lvl, f"{lab}\n{e['t']:.2f}", fontsize=8, color=INK, va="bottom")
ax_w.set_ylim(-1, 1.18)
ax_w.set_ylabel("amplitude")
ax_w.set_title("Also Sprach Zarathustra (Kevin MacLeod, CC BY 3.0) - beat map: waveform (peak / RMS per 10 ms) with "
               "labelled structure", loc="left", fontsize=13, color=INK, pad=38)

# 2. onset strength + picked onsets
ax_o.plot(tA, odf_n, color="#2a78d6", lw=0.7)
ot = np.array([o["t"] for o in onsets])
os_ = np.array([o["strength"] for o in onsets])
ax_o.plot(ot, os_, "o", ms=3, color="#eb6834", mec="none")
for e in events:
    if e["id"].startswith("timp") and "pickup" not in e["id"]:
        ax_o.axvline(e["t"], color=COL["timp"], lw=0.8, alpha=0.8, ymax=0.25)
ax_o.set_ylabel("onset strength\n(norm. flux)")
ax_o.set_title(f"Spectral-flux onset strength; orange dots = {len(onsets)} picked onsets; green ticks = timpani "
               "strokes", loc="left", fontsize=11, color=INK)

# 3. spectrogram 180-1400 Hz + trumpet pitch track
b = (fC >= 150) & (fC <= 1400)
Ld = 20 * np.log10(SC[b] + 1e-7)
dec = 4
ax_p.pcolormesh(tC[::dec], fC[b], Ld[:, ::dec], shading="auto", cmap="Greys", vmin=Ld.max() - 55, vmax=Ld.max() - 5,
                rasterized=True)
ax_p.set_yscale("log")
ax_p.set_ylim(150, 1400)
YT = (55, 60, 63, 67, 72, 76, 79, 84)
yt = [hz(k) for k in YT]
ax_p.set_yticks(yt)
ax_p.set_yticklabels([nname(k) for k in YT], fontsize=8)
ax_p.minorticks_off()
for n, c in enumerate(calls, start=1):
    a = c["C4"]
    e_ = chords[n][0] if n < 3 else t_f_hit
    i = (tC >= a) & (tC <= e_)
    ax_p.plot(tC[i], hz(pitch_med[i]), ".", ms=1.5, color="#e34948")
    for key in ("C4", "G4", "C5"):
        ax_p.axvline(c[key], color=COL["call"], lw=1)
        ax_p.text(c[key] + 0.1, 1250, key, fontsize=8, color=INK)
for n in (1, 2):
    t_hit, t_sw = chords[n][0], chords[n][1]
    ax_p.axvline(t_hit, color=COL["chord"], lw=1)
    ax_p.axvline(t_sw, color=COL["chord"], lw=1, ls="--")
    ax_p.text(t_sw + 0.1, 170, f"{chords[n][3][:3]}->{chords[n][4][:3]} {t_sw:.2f}", fontsize=8, color=INK)
ax_p.set_ylabel("pitch")
ax_p.set_title("Spectrogram 150-1400 Hz; red = trumpet pitch track (harmonic-sum salience) during the calls; "
               "solid orange = chord hit, dashed = third switches", loc="left", fontsize=11, color=INK)
ax_p.set_xlim(0, DUR)
ax_p.set_xticks(np.arange(0, 87, 2))
ax_p.set_xlabel("time (s)")

# 4. timpani zooms
for ax, n in ((ax_t1, 1), (ax_t2, 2)):
    run, pups = timp[n]
    a, b_ = run[0]["t"] - 0.9, run[-1]["t"] + 0.6
    i = (tT >= a) & (tT <= b_)
    ax.plot(tT[i], tflux_s[i] / tflux_s[i].max(), color="#1baf7a", lw=0.9)
    for s in run:
        ax.axvline(s["t"], color=INK2, lw=0.6, ls=":")
        ax.text(s["t"], 1.02, f"{s['pitch'][0]}\n{s['t']:.2f}", ha="center", fontsize=7.5, color=INK)
    for s in pups:
        ax.text(s["t"], 0.75, f"?{(s['pitch'] or '?')[0]}\n{s['t']:.2f}", ha="center", fontsize=7.5, color=MUTED)
    ax.set_ylim(0, 1.25)
    ax.set_title(f"Timpani {n}: 40-400 Hz flux; {len(run)} strokes (G = G2 ~98 Hz, C = C3 ~133 Hz); grey ? = "
                 "possible soft pickup", loc="left", fontsize=10, color=INK)
    ax.set_xlabel("time (s)")

# 5. third call -> climax -> tail detail: chroma
i = (tC >= 46) & (tC <= DUR)
ax_c.pcolormesh(tC[i][::dec], np.arange(12), CHROMA[:, i][:, ::dec], shading="auto", cmap="Blues", rasterized=True)
ax_c.set_yticks(range(12))
ax_c.set_yticklabels(NAMES, fontsize=8)
for k in ("call3_C4", "call3_G4", "call3_C5", "call3_E", "chord3_hit", "build_start", "build_A", "build_B", "climax",
          "orchestra_release", "audible_end"):
    e = evd[k]
    ax_c.axvline(e["t"], color=INK, lw=0.9)
    lvl = 1 if k in ("chord3_hit", "build_B") else 0
    ax_c.text(e["t"] + 0.08, 11.6 - 1.6 * lvl, f"{short[k] if k in short else k.split('_')[-1]} {e['t']:.2f}",
              fontsize=8, color=INK, rotation=0, va="bottom", clip_on=False,
              bbox=dict(boxstyle="square,pad=0.1", fc="#fcfcfb", ec="none", alpha=0.85) if lvl else None)
for t_ in chg:
    ax_c.axvline(t_, color=INK2, lw=0.8, ls="--")
ax_c.set_xlim(46, DUR)
ax_c.set_xticks(np.arange(46, 87, 1))
ax_c.set_title("Call 3 -> build -> climax -> tail: pitch-class energy (chroma, 100-2000 Hz); dashed = answer-phrase "
               "chord changes", loc="left", fontsize=11, color=INK, pad=16)
ax_c.set_xlabel("time (s)")
for ax in (ax_w, ax_o, ax_p, ax_t1, ax_t2, ax_c):
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_facecolor("#fcfcfb")
plt.setp(ax_w.get_xticklabels(), visible=False)
plt.setp(ax_o.get_xticklabels(), visible=False)
MUSIC.mkdir(parents=True, exist_ok=True)
fig.savefig(MUSIC / "beatmap.png", dpi=80, bbox_inches="tight", facecolor=fig.get_facecolor())

# ------------------------------------------------------------------------------------------------ console
print(f"duration {DUR:.3f} s, {len(onsets)} onsets, peak {out['peak_dbfs']} dBFS")
for e in events:
    if "pickup" in e["id"]:
        continue
    print(f"{e['t']:8.3f}  {e['confidence']:.2f}  {e['id']:<22} {e['label']}")
for n in (1, 2):
    print(f"timpani {n} pickups:", [(s['t'], s['pitch']) for s in timp[n][1]])
