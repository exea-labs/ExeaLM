"""Animate one real relay example and one real lookup example from a trained model (experiments/trace_memory.py).

Every number on screen is read from the trace: which board note agent B attends to, which tables the search pulls,
which retrieved token the read lands on, and what the model answers with the memory on and off.

python docs/figures/memory_gif.py --trace docs/figures/memory_trace.json --fonts <dir with DMMono and Fraunces .ttf>
"""
import argparse
import json
import os
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

PAPER, PANEL = "#111418", "#181c21"
INK, INK2, INK3 = "#ece6d8", "#c9c3b5", "#8f897c"
LINE = "#3a3d42"
ORANGE, AG, RG = "#ff6a3d", "#ff6fc8", "#ff8a8a"
W, H, FPS = 1280, 720, 12


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


class Frame:
    def __init__(self, mono, serif):
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)
        self.ax.axis("off")
        self.fig.patch.set_facecolor(PAPER)
        self.mono, self.serif = mono, serif

    def text(self, x, y, s, size=14, color=INK, alpha=1.0, serif=False, weight="normal", ha="left", va="baseline"):
        if alpha <= 0:
            return
        self.ax.text(x, y, s, fontsize=size * 0.75, color=color, alpha=alpha, ha=ha, va=va,
                     fontproperties=font_manager.FontProperties(fname=self.serif if serif else self.mono, weight=weight))

    def box(self, x, y, w, h, fc=PANEL, ec=LINE, alpha=1.0, lw=1.0, r=6):
        if alpha <= 0:
            return
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, alpha=alpha))

    def line(self, x0, y0, x1, y1, color=INK, lw=1.0, alpha=1.0):
        if alpha > 0 and lw > 0:
            self.ax.plot([x0, x1], [y0, y1], color=color, lw=lw, alpha=alpha, solid_capstyle="round")

    def save(self):
        self.fig.canvas.draw()
        img = Image.frombytes("RGBA", self.fig.canvas.get_width_height(), bytes(self.fig.canvas.buffer_rgba())).convert("RGB")
        plt.close(self.fig)
        return img


def pct(v):
    return f"{100 * v:.0f}%" if v >= 0.01 else "<1%"


# ---------------------------------------------------------------- scene 1: the relay through the board
def relay_frames(tr, fonts):
    r = tr["relay"]
    C = r["chunk"]
    n = len(r["A"]) // C
    A = [r["A"][i * C:(i + 1) * C] for i in range(n)]
    B = [r["B"][i * C:(i + 1) * C] for i in range(n)]
    qs = r["questions"]
    cw, gap, x0 = 136, 8, 120
    cx = lambda i: x0 + i * (cw + gap)
    yA, yN, yB = 150, 300, 470
    beats = [("title", 1.2), ("posts", 2.4), ("asker", 1.0)] + [(f"q{j}", 3.2) for j in range(len(qs))] + [("hold", 1.6)]
    total = sum(d for _, d in beats)
    frames = []
    for f in range(int(total * FPS)):
        t = f / FPS
        acc, cur, local = 0.0, None, 0.0
        for name, d in beats:
            if t < acc + d:
                cur, local = name, (t - acc) / d
                break
            acc += d
        stage = [b for b, _ in beats].index(cur)
        fr = Frame(*fonts)
        fr.text(40, 52, "Agents talk through the board", 30, INK, serif=True)
        fr.text(40, 84, "A trained 18M ExeaLM (seed 0). Real attention from its first TONN layer.", 15, INK3)
        a_show = ease(local / 0.6) if cur == "title" else 1.0
        fr.text(40, yA + 30, "agent A", 15, INK2, a_show)
        for i, ch in enumerate(A):
            fact = ch[0] == "FACT"
            fr.box(cx(i), yA, cw, 46, PANEL, ORANGE if fact else LINE, a_show, 1.2 if fact else 0.8)
            fr.text(cx(i) + cw / 2, yA + 29, " ".join(ch[:3]) if fact else "· · · · ·", 14, INK if fact else INK3, a_show, ha="center")
        # notes: one per 8-token stretch A read; the four that hold facts are drawn
        fr.text(40, yN + 30, "board", 15, AG, 1.0 if stage >= 1 else 0.0)
        for i in range(4):
            p = 1.0 if stage > 1 else (ease((local * 4 - i) / 0.8) if cur == "posts" else 0.0)
            if p <= 0:
                continue
            fr.line(cx(i) + cw / 2, yA + 48, cx(i) + cw / 2, yA + 48 + (yN - yA - 50) * p, AG, 1.2, 0.5 * p)
            fr.box(cx(i) + 10, yN, cw - 20, 46, "#2a1726", AG, p, 1.2)
            fr.text(cx(i) + cw / 2, yN + 21, f"note {i + 1}", 13, AG, p, ha="center")
            fr.text(cx(i) + cw / 2, yN + 38, "32 numbers", 11, INK3, p, ha="center")
        b_show = 1.0 if stage > 2 else (ease(local / 0.6) if cur == "asker" else 0.0)
        fr.text(40, yB + 30, "agent B", 15, INK2, b_show)
        for i, ch in enumerate(B):
            ask = ch[0] == "ASK"
            fr.box(cx(i), yB, cw, 46, PANEL, LINE, b_show, 0.8)
            fr.text(cx(i) + cw / 2, yB + 29, f"ASK {ch[1]} ?" if ask else "· · · · ·", 14, INK if ask else INK3, b_show, ha="center")
        if b_show > 0:
            fr.text(cx(4), yB + 80, "B never sees A's text. It can only read the notes.", 13, INK3, b_show)
        for j, q in enumerate(qs):
            name = f"q{j}"
            if name not in [b for b, _ in beats][:stage + 1]:
                continue
            k = 4 + j
            done = cur != name
            prog = 1.0 if done else local
            att = q["layers"][0]["A"]
            if not done:
                fr.box(cx(k) - 3, yB - 3, cw + 6, 52, "none", ORANGE, 1.0, 2.0)
                g = ease(prog / 0.45)
                for i in range(4):
                    w = att[i]
                    fr.line(cx(k) + cw / 2, yB - 4, cx(i) + cw / 2, yN + 48, AG, 1 + 9 * w * g, (0.15 + 0.85 * w) * g)
                best = max(range(4), key=lambda i: att[i])
                if g > 0.6:
                    fr.box(cx(best) + 6, yN - 4, cw - 12, 54, "none", INK, ease((g - 0.6) / 0.4), 1.6)
                    fr.text(cx(best) + cw / 2, yN - 12, f"reads {pct(att[best])}", 13, INK, ease((g - 0.6) / 0.4), ha="center")
            show = 1.0 if done else ease((prog - 0.5) / 0.3)
            dim = 0.55 if done else 1.0
            top, cut = q["shared"][0], q["cut"][0]
            fr.text(cx(k) + cw / 2, yB + 112, f"{top[0]} {pct(top[1])}", 17, ORANGE if top[0] == q["answer"] else INK2, show * dim, ha="center", weight="medium")
            fr.text(cx(k) + cw / 2, yB + 136, "board shared", 11, INK3, show * dim, ha="center")
            fr.text(cx(k) + cw / 2, yB + 168, f"{cut[0]} {pct(cut[1])}", 15, ORANGE if cut[0] == q["answer"] else INK3, show * dim, ha="center")
            fr.text(cx(k) + cw / 2, yB + 190, "board cut", 11, INK3, show * dim, ha="center")
        if cur in ("q0",) and local < 0.25:
            pass
        fr.text(40, H - 24, "Each note is made from one 8-token stretch of A's text. Line width: how much B's board read attends to each note.", 12, INK3)
        frames.append(fr.save())
    return frames


# ---------------------------------------------------------------- scene 2: the lookup through the library
def lookup_frames(tr, fonts):
    L = tr["lookup"]
    lay = L["layers"][0]
    tables, weights = lay["tables"], lay["weights"]
    right = L["right_table"]
    rows = lay["table_tokens"][0]                                       # the top table's 64 tokens
    facts = [rows[4 * i:4 * i + 3] for i in range(len(rows) // 4)]
    qs = L["questions"]
    pick = [q for q in qs if q["attached"][0][0] == q["answer"]][:2] + [q for q in qs if q["attached"][0][0] != q["answer"]][:1]
    gx, gy, cell = 380, 160, 34
    tx, ty = 820, 132
    beats = [("title", 1.2), ("search", 2.2), ("open", 1.2)] + [(f"q{j}", 3.4) for j in range(len(pick))] + [("hold", 1.6)]
    total = sum(d for _, d in beats)
    frames = []
    for f in range(int(total * FPS)):
        t = f / FPS
        acc, cur, local = 0.0, None, 0.0
        for name, d in beats:
            if t < acc + d:
                cur, local = name, (t - acc) / d
                break
            acc += d
        names = [b for b, _ in beats]
        stage = names.index(cur)
        fr = Frame(*fonts)
        fr.text(40, 52, "The model looks it up in its own library", 30, INK, serif=True)
        fr.text(40, 84, f"{L['library_tables']} tables, 1,024 facts it never saw in training. Real search weights and read attention.", 15, INK3)
        # the question chunk
        a0 = ease(local / 0.6) if cur == "title" else 1.0
        fr.text(40, 150, "the question chunk", 13, INK2, a0)
        qtxt = " ".join(L["question"]).split("ASK")[1:7]
        for i, s in enumerate(qtxt):
            fr.text(40, 182 + 26 * i, "ASK" + s.replace(" .", "").rstrip(), 14, INK, a0)
        fr.text(40, 182 + 26 * 6, "... 16 keys in all", 12, INK3, a0)
        # the library grid
        fr.text(gx, 150, "library: 64 tables", 13, RG, a0)
        s_on = 1.0 if stage > 1 else (ease(local) if cur == "search" else 0.0)
        for i in range(L["library_tables"]):
            x, y = gx + (i % 8) * (cell + 4), gy + 12 + (i // 8) * (cell + 4)
            if i in tables:
                w = weights[tables.index(i)]
                hot = s_on * (0.25 + 0.75 * w)
                fr.box(x, y, cell, cell, "#3a1d1d" if w > 0.5 else PANEL, RG, max(a0, hot), 1.0 + 1.8 * w * s_on)
                if s_on > 0.4:
                    fr.text(x + cell / 2, y + cell + 13, pct(w), 10, RG if w > 0.5 else INK3, ease((s_on - 0.4) / 0.6), ha="center")
            else:
                fr.box(x, y, cell, cell, PANEL, LINE, a0, 0.6)
        if s_on > 0:
            fr.line(250, 230, gx - 10, 230, RG, 2.0, 0.6 * s_on)
            fr.text(gx - 14, 222, "search", 11, RG, s_on, ha="right")
            fr.text(gx, gy + 12 + 8 * (cell + 4) + 30, "the search's top 4, with their weights", 12, INK3, s_on)
            fr.text(gx, gy + 12 + 8 * (cell + 4) + 50, f"the right table is #{right + 1}", 12, INK3, s_on)
        # the top table, opened
        o_on = 1.0 if stage > 2 else (ease(local) if cur == "open" else 0.0)
        if o_on > 0:
            fr.text(tx, ty + 18, f"table #{tables[0] + 1}, as retrieved", 13, RG, o_on)
            for i, fct in enumerate(facts):
                fr.text(tx + (i // 8) * 210, ty + 50 + (i % 8) * 26, f"{fct[0]} {fct[1]}  {fct[2]}", 14, INK2, o_on)
        for j, q in enumerate(pick):
            name = f"q{j}"
            if name not in names[:stage + 1]:
                continue
            done = cur != name
            if done:
                continue
            fr.box(tx - 8, ty + 270, 430, 120, PANEL, ORANGE, 1.0, 1.4)
            fr.text(tx + 8, ty + 298, f"answer chunk: ANS {q['keys'][0]} {q['keys'][1]} ?", 15, INK, 1.0)
            fr.text(tx + 418, ty + 298, f"true value {q['answer']}", 12, INK3, 1.0, ha="right")
            g = ease(local / 0.45)
            for slot, pos, w in q["read"][0]:
                if slot != 0 or w < 0.05:
                    continue
                fi = pos // 4
                fx, fy = tx + (fi // 8) * 210, ty + 50 + (fi % 8) * 26        # the fact's text: x, baseline
                fr.box(fx - 6, fy - 18, 150, 25, "none", ORANGE, g * (0.3 + 0.7 * w), 1 + 1.5 * w)
                fr.line(tx + 120, ty + 268, fx + 69, fy + 8, ORANGE, 1 + 6 * w * g, (0.2 + 0.8 * w) * g)
                if g > 0.6:
                    fr.text(fx + 150, fy, f"{pct(w)}", 12, ORANGE, ease((g - 0.6) / 0.4))
            sh = ease((local - 0.5) / 0.3)
            top, rem = q["attached"][0], q["removed"][0]
            good = top[0] == q["answer"]
            fr.text(tx + 8, ty + 332, f"library attached: {top[0]} {pct(top[1])}  " + ("right" if good else "wrong"),
                    15, ORANGE if good else INK2, sh)
            fr.text(tx + 8, ty + 360, f"library removed:  {rem[0]} {pct(rem[1])}  wrong", 14, INK3, sh)
        fr.text(40, H - 24, "Line width: how much the read at the answer attends to each retrieved token.", 12, INK3)
        frames.append(fr.save())
    return frames


def end_frames(fonts):
    frames = []
    for f in range(int(3.0 * FPS)):
        a = ease(f / 8)
        fr = Frame(*fonts)
        fr.text(W / 2, 250, "Over many examples, two seeds", 30, INK, a, serif=True, ha="center")
        fr.text(W / 2, 330, "relay through the board: 100% (1% with the board cut)", 20, AG, a, ha="center")
        fr.text(W / 2, 380, "lookup in a library it never saw: 94% (1.6% without it)", 20, RG, a, ha="center")
        fr.text(W / 2, 450, "Synthetic tasks. On real text these paths do not help yet (issue #27).", 15, INK3, a, ha="center")
        fr.text(W / 2, 480, "blog.exealabs.org  ·  github.com/exea-labs/ExeaLM", 14, INK3, a, ha="center")
        frames.append(fr.save())
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="docs/figures/memory_trace.json")
    ap.add_argument("--fonts", required=True, help="folder with DMMono-Regular.ttf, DMMono-Medium.ttf and Fraunces.ttf")
    ap.add_argument("--out", default="docs/figures/memory")
    a = ap.parse_args()
    for f in os.listdir(a.fonts):
        if f.endswith(".ttf"):
            font_manager.fontManager.addfont(os.path.join(a.fonts, f))
    fonts = (os.path.join(a.fonts, "DMMono-Regular.ttf"), os.path.join(a.fonts, "Fraunces.ttf"))
    tr = json.load(open(a.trace))
    frames = relay_frames(tr, fonts) + lookup_frames(tr, fonts) + end_frames(fonts)
    tmp = a.out + "_frames"
    os.makedirs(tmp, exist_ok=True)
    for i, im in enumerate(frames):
        im.save(os.path.join(tmp, f"{i:04d}.png"))
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(tmp, "%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out + ".mp4"], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(tmp, "%04d.png"),
                        "-vf", "scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=none",
                        a.out + ".gif"], check=True)
    else:
        small = [im.resize((960, 540)) for im in frames]
        small[0].save(a.out + ".gif", save_all=True, append_images=small[1:], duration=int(1000 / FPS), loop=0)
    shutil.rmtree(tmp)
    print("wrote", a.out + ".gif", "and", a.out + ".mp4", len(frames), "frames")


if __name__ == "__main__":
    main()
