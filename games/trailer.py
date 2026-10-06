"""Cut the announcement trailer from the games' recorded clips, to the soundtrack's beat map.

    python games/trailer.py games/trailer_edl.json            # -> out/trailer/flyverse_trailer.mp4

The edit decision list (EDL) is JSON:

    {"fps": 50, "size": [1920, 1080], "duration": 86.0,
     "music": "out/trailer/music/zarathustra.wav", "music_start": 0.0, "audio_fade_out": 2.5,
     "out": "out/trailer/flyverse_trailer.mp4",
     "shots": [
       {"t0": 0.0, "t1": 12.4, "kind": "clip", "src": "out/games/sunrise/seed0.mp4", "src_t0": 0.0, "speed": 1.0,
        "crop": [x, y, w, h] | null, "fade_in": 1.5, "fade_out": 0.0, "flash": false,
        "label": "PONG", "sub": "MaleCNS v1.0 vs FlyWire FAFB v783 ...", "label_t": [0.2, 3.0],
        "texts": [{"t0": 1.0, "t1": 4.0, "text": "167,106 neurons", "size": 84, "pos": "center"}]},
       {"t0": 80.0, "t1": 86.0, "kind": "card", "style": "end", "lines": ["flyverse", "..."]}
     ]}

Shots are placed on the output timeline by t0 / t1 (seconds). A clip shot plays its source from `src_t0` at
`speed` (source frames are repeated or skipped; nothing is interpolated). `crop` is a rectangle of the 1920x1080
source, scaled to fill the frame. Nothing is retimed inside a clip except by `speed`, and every clip's provenance
is in games/README.md; this script adds only cuts, fades, flashes, cards and text.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as gc  # noqa: E402


class ClipReader:
    """Sequential RGB frames of one source clip from `t0`, optionally cropped and scaled to `size`."""

    def __init__(self, path, t0, size, crop=None, src_fps=None):
        self.path, self.size = str(path), size
        w, h = size
        vf = []
        if crop:
            x, y, cw, ch = crop
            vf.append(f"crop={cw}:{ch}:{x}:{y}")
        vf.append(f"scale={w}:{h}:flags=lanczos")
        if src_fps:
            vf.append(f"fps={src_fps}")
        self.proc = subprocess.Popen([shutil.which("ffmpeg"), "-v", "error", "-ss", f"{t0:.4f}", "-i", self.path,
                                      "-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                     stdout=subprocess.PIPE)
        self.index, self.frame = -1, np.zeros((h, w, 3), np.uint8)

    def get(self, index):
        """Frame `index` (0 = the frame at t0); holds the last frame past the end of the source."""
        n = self.size[0] * self.size[1] * 3
        while self.index < index:
            buf = self.proc.stdout.read(n)
            if len(buf) < n:
                break
            self.frame = np.frombuffer(buf, np.uint8).reshape(self.size[1], self.size[0], 3)
            self.index += 1
        return self.frame

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


def probe_fps(path):
    out = subprocess.run([shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=r_frame_rate", "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
    try:
        a, b = out.split("/") if "/" in out else (out, 1)
        return float(a) / float(b)
    except ValueError:                                  # unreadable (still being written): previews show it dark
        print(f"  warning: cannot read {path}; drawn dark", flush=True)
        return 50.0


def smooth(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


class Compositor:
    def __init__(self, edl):
        import pygame
        pygame.init()
        self.pg = pygame
        self.edl = edl
        self.fps = int(edl.get("fps", 50))
        self.size = tuple(edl.get("size", gc.DEFAULT_SIZE))
        self.hud = gc.Hud()
        self.surface = pygame.Surface(self.size)
        self.readers = {}

    # ------------------------------------------------------------------------------------------ text helpers
    def text(self, s, pos, size, color=gc.TEXT, *, bold=False, display=True, anchor="center", alpha=1.0, shadow=True):
        font = self.hud.font(size, bold, display)
        img = font.render(str(s), True, color)
        if alpha < 1.0:
            img.set_alpha(int(255 * max(0.0, alpha)))
        rect = img.get_rect(**{anchor: (int(pos[0]), int(pos[1]))})
        if shadow:
            sh = font.render(str(s), True, (0, 0, 0))
            sh.set_alpha(int(150 * max(0.0, alpha)))
            self.surface.blit(sh, rect.move(2, 3))
        self.surface.blit(img, rect)
        return rect

    def lower_third(self, label, sub, alpha):
        if alpha <= 0:
            return
        w, h = self.size
        band = self.pg.Surface((w, 190), self.pg.SRCALPHA)
        for y in range(190):                                          # dark gradient for legibility
            band.fill((0, 0, 0, int(170 * alpha * (y / 190) ** 1.3)), (0, y, w, 1))
        self.surface.blit(band, (0, h - 190))
        x = 72
        r = self.text(label, (x, h - 118), 64, gc.WHITE, bold=True, anchor="topleft", alpha=alpha)
        self.pg.draw.rect(self.surface, gc.SAGE, (x, r.y - 14, 64, 5))
        if sub:
            self.text(sub, (x + 2, h - 50), 24, gc.TEXT, anchor="topleft", alpha=alpha, display=False)

    def card(self, shot, t):
        w, h = self.size
        self.surface.fill((5, 7, 8))
        style = shot.get("style", "title")
        lines = shot.get("lines", [])
        a = smooth((t - shot["t0"]) / max(shot.get("fade_in", 0.4), 1e-3))
        if style == "end":
            # line 0: the title; lines 1 .. n_big-1 large (the URL; `commands` lists the lines set as commands, in the
            # plain face at `cmd_size`); the rest small print. The block is measured, then stacked top-down and centred.
            n_big, commands = shot.get("n_big", 3), set(shot.get("commands", [2]))
            rows = []
            for i, line in enumerate(lines):
                if i == 0:
                    style_i = (150, gc.SAGE, True, True)
                elif i in commands:
                    style_i = (shot.get("cmd_size", 36), gc.TEXT, False, False)
                elif i < n_big:
                    style_i = (48, gc.TEXT, False, True)
                else:
                    style_i = (28, gc.MUTED, False, True)
                size, col, bold, disp = style_i
                rows.append((line, size, col, bold, disp, self.hud.font(size, bold, disp).size(line)[1]))

            def gap(i):                                   # space above line i
                if i == 1:
                    return 26
                if i in commands:
                    return 30 if (i - 1) not in commands else 6
                if i == n_big:
                    return 40
                return 6
            total = sum(r[-1] for r in rows) + sum(gap(i) for i in range(1, len(rows)))
            y = (h - total) / 2 - h * shot.get("lift", 0.03)
            for i, (line, size, col, bold, disp, lh) in enumerate(rows):
                y += gap(i) if i else 0
                self.text(line, (w / 2, y), size, col, bold=bold, display=disp, anchor="midtop", alpha=a)
                y += lh
        else:
            y = h / 2 - (len(lines) - 1) * 50
            for i, line in enumerate(lines):
                self.text(line, (w / 2, y + i * 100), 76 if i == 0 else 34, gc.WHITE if i == 0 else gc.MUTED,
                          bold=(i == 0), alpha=a)

    # ------------------------------------------------------------------------------------------ frames
    def reader(self, k, shot, size=None):
        if k not in self.readers:
            fps = probe_fps(shot["src"])
            self.readers[k] = (ClipReader(shot["src"], shot.get("src_t0", 0.0), size or self.size, shot.get("crop")), fps)
        return self.readers[k]

    def grid(self, k, shot, local):
        """A grid of clips: `cells` (each {src, src_t0, crop, speed}) laid out `cols` x `rows` with `gap` pixels;
        saturation ramps from `sat0` to `sat1` over `sat_ramp` seconds (a chord that turns major)."""
        cols, rows, gap = shot.get("cols", 3), shot.get("rows", 2), shot.get("gap", 8)
        W, H = self.size
        cw, ch = (W - gap * (cols + 1)) // cols, (H - gap * (rows + 1)) // rows
        cw -= cw % 2; ch -= ch % 2
        sat = shot.get("sat0", 1.0) + (shot.get("sat1", 1.0) - shot.get("sat0", 1.0)) * smooth(local / max(shot.get("sat_ramp", 0.5), 1e-3))
        self.surface.fill((5, 7, 8))
        for i, cell in enumerate(shot["cells"][: cols * rows]):
            # keep the source's (or its crop's) aspect: fit inside the cell, centred, never stretched
            src_w, src_h = (cell["crop"][2], cell["crop"][3]) if cell.get("crop") else (1920, 1080)
            scale = min(cw / src_w, ch / src_h)
            fw, fh = int(src_w * scale) // 2 * 2, int(src_h * scale) // 2 * 2
            rd, fps = self.reader((k, i), cell, (fw, fh))
            img = rd.get(int(np.floor(local * cell.get("speed", 1.0) * fps + 1e-6))).astype(np.float32)
            if sat < 0.999:
                grey = img.mean(axis=2, keepdims=True)
                img = grey + (img - grey) * sat
            x0, y0 = gap + (i % cols) * (cw + gap), gap + (i // cols) * (ch + gap)
            x, y = x0 + (cw - fw) // 2, y0 + (ch - fh) // 2
            self.surface.blit(self.pg.surfarray.make_surface(np.clip(img, 0, 255).astype(np.uint8).swapaxes(0, 1)), (x, y))
            if cell.get("tag"):
                self.text(cell["tag"], (x + 14, y + fh - 14), 26, gc.WHITE, bold=True, anchor="bottomleft")

    def frame(self, t):
        s = self.surface
        s.fill((0, 0, 0))
        shots = self.edl["shots"]
        live = [k for k, sh in enumerate(shots) if sh["t0"] <= t < sh["t1"]]
        for k in list(self.readers):                                   # free readers of finished shots
            if shots[k[0] if isinstance(k, tuple) else k]["t1"] <= t:
                self.readers.pop(k)[0].close()
        for k in live:
            sh = shots[k]
            local = t - sh["t0"]
            if sh["kind"] == "card":
                self.card(sh, t)
            elif sh["kind"] in ("black", "white"):
                s.fill((0, 0, 0) if sh["kind"] == "black" else (255, 252, 246))
            elif sh["kind"] == "grid":
                self.grid(k, sh, local)
            else:
                rd, fps = self.reader(k, sh)
                img = rd.get(int(np.floor(local * sh.get("speed", 1.0) * fps + 1e-6)))
                s.blit(self.pg.surfarray.make_surface(img.swapaxes(0, 1)), (0, 0))
            # fades (to black)
            fi, fo = sh.get("fade_in", 0.0), sh.get("fade_out", 0.0)
            dark = 0.0
            if fi > 0 and local < fi:
                dark = 1 - smooth(local / fi)
            if fo > 0 and sh["t1"] - t < fo:
                dark = max(dark, 1 - smooth((sh["t1"] - t) / fo))
            if dark > 0:
                veil = self.pg.Surface(self.size); veil.fill((0, 0, 0)); veil.set_alpha(int(255 * dark))
                s.blit(veil, (0, 0))
            for tx in sh.get("texts", []):
                if tx["t0"] <= local < tx["t1"]:
                    a = min(smooth((local - tx["t0"]) / 0.35), smooth((tx["t1"] - local) / 0.35))
                    pos = tx.get("pos", "center")
                    pos = (self.size[0] / 2, self.size[1] / 2) if pos == "center" else tuple(pos)
                    self.text(tx["text"], pos, tx.get("size", 72), tuple(tx.get("color", gc.WHITE)),
                              bold=tx.get("bold", True), alpha=a, anchor=tx.get("anchor", "center"))
            if sh.get("label"):
                lt0, lt1 = sh.get("label_t", [0.15, sh["t1"] - sh["t0"]])
                if lt0 <= local < lt1:
                    a = min(smooth((local - lt0) / 0.25), smooth((lt1 - local) / 0.3))
                    self.lower_third(sh["label"], sh.get("sub", ""), a)
            if sh.get("flash") and local < 0.12:                        # a white flash on the cut (a timpani hit)
                veil = self.pg.Surface(self.size); veil.fill((255, 250, 240)); veil.set_alpha(int(200 * (1 - local / 0.12)))
                s.blit(veil, (0, 0))
        return s

    def render(self, out_video):
        dur = float(self.edl["duration"])
        rec = gc.Recorder(out_video, self.size, self.fps, crf=int(self.edl.get("crf", 16)))
        n = int(round(dur * self.fps))
        try:
            for f in range(n):
                rec.write(self.frame(f / self.fps))
                if f % (self.fps * 5) == 0:
                    print(f"  trailer {f / self.fps:5.1f} / {dur:.1f} s", flush=True)
        finally:
            rec.close()
            for rd, _ in self.readers.values():
                rd.close()
        return rec.path


def mux(video, edl, out):
    music = edl["music"]
    dur = float(edl["duration"])
    fade = float(edl.get("audio_fade_out", 2.0))
    af = f"afade=t=out:st={max(0.0, dur - fade):.3f}:d={fade:.3f}"
    if edl.get("audio_fade_in"):
        af = f"afade=t=in:st=0:d={float(edl['audio_fade_in']):.3f}," + af
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-y", "-i", str(video), "-ss", f"{float(edl.get('music_start', 0.0)):.3f}",
                    "-i", music, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
                    "-af", af, "-t", f"{dur:.3f}", "-movflags", "+faststart", str(out)], check=True)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("edl", help="edit decision list (JSON)")
    ap.add_argument("--out", default=None, help="output mp4 (default: the EDL's 'out')")
    ap.add_argument("--from", dest="t_from", type=float, default=None, help="render only [from, to) for previews")
    ap.add_argument("--to", dest="t_to", type=float, default=None)
    args = ap.parse_args()
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    edl = json.loads(Path(args.edl).read_text(encoding="utf-8"))
    out = Path(args.out or edl.get("out", "out/trailer/flyverse_trailer.mp4"))
    if args.t_from is not None or args.t_to is not None:        # preview: shift the timeline
        a = args.t_from or 0.0; b = args.t_to or float(edl["duration"])
        shots = []
        for sh in edl["shots"]:
            if sh["t1"] <= a or sh["t0"] >= b:
                continue
            sh = dict(sh)
            if sh["t0"] < a and sh["kind"] == "clip":
                sh["src_t0"] = sh.get("src_t0", 0.0) + (a - sh["t0"]) * sh.get("speed", 1.0)
                sh["texts"] = [dict(tx, t0=tx["t0"] - (a - sh["t0"]), t1=tx["t1"] - (a - sh["t0"])) for tx in sh.get("texts", [])]
            shift = min(sh["t0"], a) if sh["t0"] < a else a
            sh["t0"], sh["t1"] = max(sh["t0"], a) - a, min(sh["t1"], b) - a
            del shift
            shots.append(sh)
        edl = dict(edl, shots=shots, duration=b - a, music_start=float(edl.get("music_start", 0.0)) + a)
    out.parent.mkdir(parents=True, exist_ok=True)
    silent = out.with_name(out.stem + "_silent.mp4")
    Compositor(edl).render(silent)
    mux(silent, edl, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
