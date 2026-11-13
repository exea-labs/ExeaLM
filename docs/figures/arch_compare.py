"""Draw docs/figures/arch_compare.png: Llama, a Qwen3.5-style model and the ExeaLM hybrid with gated attention and
QK-norm, as compared at 13M parameters in results/README.md section 12. Top: every layer of each stack and its
residual connections. Bottom: the inside of every block type. Sizes, circuits and scores are the ones of those runs.

python docs/figures/arch_compare.py
"""
import html
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1600, 1420
SANS = "'Helvetica Neue', Helvetica, Arial, sans-serif"
SERIF = "'STIX Two Text', 'Times New Roman', serif"
INK, MID, LIGHT = "#1a1a1a", "#555", "#888"
NORM, LIN, ATT, DEL, TONN, ACT, MLP, EMB = "#fff1b8", "#d6e6f7", "#f9d9c2", "#d5edd5", "#e6ddf4", "#f4f4f4", "#ececec", "#e6e6e6"
els = []


def t(x, y, s, size=12, color=INK, anchor="middle", weight=400, serif=False, italic=False):
    els.append(f'<text x="{x}" y="{y}" font-family="{SERIF if serif else SANS}" font-size="{size}" font-weight="{weight}" '
               f'fill="{color}" text-anchor="{anchor}"{" font-style=" + chr(34) + "italic" + chr(34) if italic else ""}>{html.escape(s)}</text>')


def box(x, y, w, h, label="", fill=LIN, size=11.5, sub=None, sub_size=10, weight=400, italic_sub=False):
    els.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" stroke="{INK}" stroke-width="1"/>')
    if label and sub:
        t(x + w / 2, y + h / 2 - 2, label, size, weight=weight)
        t(x + w / 2, y + h / 2 + 11, sub, sub_size, MID, serif=italic_sub, italic=italic_sub)
    elif label:
        t(x + w / 2, y + h / 2 + size * 0.36, label, size, weight=weight)


def line(points, arrow=True, color=INK, w=1.1, dash=None):
    d = "M" + " L".join(f"{x},{y}" for x, y in points)
    els.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w}"' + (f' stroke-dasharray="{dash}"' if dash else "")
               + (' marker-end="url(#a)"' if arrow else "") + "/>")


def op(x, y, sym, r=8):
    els.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="white" stroke="{INK}" stroke-width="1"/>')
    if sym == "+":
        line([(x - 4.5, y), (x + 4.5, y)], False)
        line([(x, y - 4.5), (x, y + 4.5)], False)
    elif sym == "x":
        line([(x - 3.5, y - 3.5), (x + 3.5, y + 3.5)], False)
        line([(x - 3.5, y + 3.5), (x + 3.5, y - 3.5)], False)
    else:
        t(x, y + 4, sym, 11, serif=True)


def up(x, y0, y1):
    line([(x, y0), (x, y1)])


# ------------------------------------------------------------------ title
t(W / 2, 34, "Three 13M-parameter language models, layer by layer", 22, serif=True, weight=600)
t(W / 2, 56, "Width 256, 6 layers, 32,768-token vocabulary with tied embeddings, 4 heads of 64, SwiGLU MLP of width 680. "
  "3,000 steps, seed 0; bits per byte on English, Japanese and code, lower is better.", 12, MID)

# ------------------------------------------------------------------ top: the three stacks
STACKS = [
    ("Llama", "13.10M parameters", "1.543", ["a"] * 6),
    ("Qwen3.5-style", "13.51M parameters", "1.520", ["c", "c", "c", "b", "c", "b"]),
    ("ExeaLM hybrid + gated attention + QK-norm", "13.36M parameters", "1.511", ["d0", "b", "b", "d3", "b", "b"]),
]
NAMES = {"a": ("(a) Attention", ATT), "b": ("(b) Gated attention + QK-norm", ATT), "c": ("(c) Gated DeltaNet", DEL),
         "d0": ("(d) TONN layer: 8 circuits", TONN), "d3": ("(d) TONN layer: 8 circuits", TONN)}
CW = 500
ROW, GAP = 26, 7
for si, (name, params, score, kinds) in enumerate(STACKS):
    x0 = 30 + si * (CW + 25)
    rx = x0 + 70                                                         # the residual stream
    bx, bw = x0 + 120, CW - 140
    t(x0 + CW / 2, 90, name, 15, weight=600, serif=True)
    t(x0 + CW / 2, 108, f"{params}   ·   {score} bits per byte", 12, MID)
    # output head
    yo = 146
    box(bx, yo, bw, ROW, "Linear 256 → 32,768 (tied with the embedding), softmax", EMB, 11)
    box(bx, yo + ROW + GAP, bw, ROW, "RMSNorm", NORM, 11)
    line([(rx, yo + 2 * ROW + 2 * GAP + 10), (rx, yo + ROW + GAP + ROW / 2), (bx, yo + ROW + GAP + ROW / 2)])
    up(bx + bw / 2, yo + ROW + GAP, yo + ROW + 1)
    t(bx + bw / 2, yo - 6, "next-token probabilities", 10.5, MID)
    up(bx + bw / 2, yo, yo - 2) if False else None
    # layers, top (layer 5) to bottom (layer 0): MLP above mixer
    y = yo + 2 * (ROW + GAP) + 14
    rows = []
    for li in range(5, -1, -1):
        rows.append(("mlp", li))
        rows.append((kinds[li], li))
    STEP = ROW + 16
    for k, (kind, li) in enumerate(rows):
        yt = y + k * STEP
        yb = yt + ROW
        if kind == "mlp":
            box(bx, yt, bw, ROW, "(e) SwiGLU MLP", MLP, 11)
            els.append(f'<path d="M{x0 + 44},{yt + 2} h-6 v{STEP + ROW - 4} h6" fill="none" stroke="#bbb" stroke-width="1"/>')
            t(x0 + 34, yt + (STEP + ROW) / 2 + 4, f"layer {li}", 10.5, MID, anchor="end")
        else:
            label, fill = NAMES[kind]
            box(bx, yt, bw, ROW, label + ("" if kind[0] != "d" else ("   (RP, MI, EP, MIM, MI, OS, MI, SM)" if kind == "d0" else "   (G, EP, OS, SM, RP, GI, C, C)")), fill, 11)
        ya = yt - 8                                                       # the residual add after this block
        op(rx, ya, "+", 7)
        line([(rx, yb + 8), (bx - 14, yb + 8), (bx - 14, yb - ROW / 2), (bx, yb - ROW / 2)])   # stream into the block
        line([(bx + bw / 2, yt), (bx + bw / 2, ya), (rx + 7, ya)])        # block out, into the add
        up(rx, yb + 8, ya + 7)
    ylast = y + len(rows) * STEP - 8
    box(bx, ylast + 10, bw, ROW, "Token embedding 32,768 × 256", EMB, 11)
    line([(bx, ylast + 10 + ROW / 2), (rx, ylast + 10 + ROW / 2), (rx, ylast)], True)
    t(bx + bw / 2, ylast + 10 + ROW + 18, "tokens", 11, MID)
    up(bx + bw / 2, ylast + 10 + ROW + 6, ylast + 10 + ROW + 1)
TOP_END = 805
els.append(f'<line x1="20" x2="{W - 20}" y1="{TOP_END}" y2="{TOP_END}" stroke="#ccc" stroke-width="1"/>')

# ------------------------------------------------------------------ bottom: inside each block
PY0, PY1 = TOP_END + 62, H - 78                                          # panel top (output) and bottom (input)


def panel(x0, w, title, subtitle):
    t(x0 + w / 2, TOP_END + 28, title, 13.5, weight=600, serif=True)
    if subtitle:
        t(x0 + w / 2, TOP_END + 44, subtitle, 10.5, MID)


def frame(x0, w, cx, out_y, in_y):
    """Input x at the bottom, the residual skip on the left, the add at the top, the output above it."""
    t(cx, in_y + 22, "x", 13, serif=True, italic=True)
    sx = x0 + 14
    line([(cx, in_y + 8), (cx, in_y - 4)], True)
    line([(cx, in_y + 2), (sx, in_y + 2), (sx, out_y), (cx - 9, out_y)], True)
    op(cx, out_y, "+")
    up(cx, out_y - 8, out_y - 20)


# (a) attention
x0, w = 20, 250
cx = x0 + 135
panel(x0, w, "(a) Attention", "Llama")
frame(x0, w, cx, PY0 + 22, PY1)
ys = {}
bw = 180
box(cx - bw / 2, PY1 - 40, bw, 28, "RMSNorm", NORM)
box(cx - bw / 2, PY1 - 92, bw, 28, "Linear 256 → 768", LIN)
up(cx, PY1 - 4, PY1 - 12)
up(cx, PY1 - 40, PY1 - 64)
for i, (lab, dx) in enumerate((("q", -60), ("k", 0), ("v", 60))):
    line([(cx, PY1 - 92), (cx + dx, PY1 - 112)], False)
    t(cx + dx, PY1 - 118, lab, 12, serif=True, italic=True)
for lab, dx in (("q", -60), ("k", 0)):
    box(cx + dx - 26, PY1 - 172, 52, 26, "RoPE", LIN, 11)
    up(cx + dx, PY1 - 130, PY1 - 146)
line([(cx + 60, PY1 - 130), (cx + 60, PY1 - 210)])
box(cx - bw / 2, PY1 - 262, bw, 50, "causal softmax(QKᵀ/√64)V", ATT, 11.5, "4 heads × 64")
for dx in (-60, 0):
    up(cx + dx, PY1 - 172, PY1 - 212)
box(cx - bw / 2, PY1 - 330, bw, 28, "Linear 256 → 256", LIN)
up(cx, PY1 - 262, PY1 - 302)
up(cx, PY1 - 330, PY0 + 30)

# (b) gated attention + QK-norm
x0, w = 285, 300
cx = x0 + 130
panel(x0, w, "(b) Gated attention + QK-norm", "Qwen3.5-style and ExeaLM")
frame(x0, w, cx, PY0 + 22, PY1)
bw = 170
box(cx - bw / 2, PY1 - 40, bw, 28, "RMSNorm", NORM)
up(cx, PY1 - 4, PY1 - 12)
t(cx + bw / 2 + 8, PY1 - 22, "h", 12, serif=True, italic=True, anchor="start")
box(cx - bw / 2, PY1 - 92, bw, 28, "Linear 256 → 768", LIN)
up(cx, PY1 - 40, PY1 - 64)
for lab, dx in (("q", -55), ("k", 0), ("v", 55)):
    line([(cx, PY1 - 92), (cx + dx, PY1 - 110)], False)
    t(cx + dx, PY1 - 116, lab, 12, serif=True, italic=True)
for lab, dx in (("q", -55), ("k", 0)):
    box(cx + dx - 25, PY1 - 162, 50, 24, "RMSNorm", NORM, 9.5)
    up(cx + dx, PY1 - 128, PY1 - 138)
    box(cx + dx - 25, PY1 - 202, 50, 24, "RoPE", LIN, 10.5)
    up(cx + dx, PY1 - 162, PY1 - 178)
line([(cx + 55, PY1 - 128), (cx + 55, PY1 - 230)])
box(cx - bw / 2, PY1 - 280, bw, 48, "causal softmax(QKᵀ/√64)V", ATT, 11, "4 heads × 64")
for dx in (-55, 0):
    up(cx + dx, PY1 - 202, PY1 - 232)
gx = cx + bw / 2 + 40                                                    # the gate branch, from h
line([(cx + bw / 2, PY1 - 26), (gx, PY1 - 26), (gx, PY1 - 270)])
box(gx - 36, PY1 - 304, 72, 32, "Linear", LIN, 9.5, "256 → 256", 8.5)
op(gx, PY1 - 322, "σ")
up(gx, PY1 - 300, PY1 - 314)
op(cx, PY1 - 322, "x")
up(cx, PY1 - 280, PY1 - 314)
line([(gx - 8, PY1 - 322), (cx + 9, PY1 - 322)])
t(gx + 14, PY1 - 330, "gate", 10, MID, anchor="start")
box(cx - bw / 2, PY1 - 376, bw, 28, "Linear 256 → 256", LIN)
up(cx, PY1 - 330, PY1 - 348)
up(cx, PY1 - 376, PY0 + 30)

# (c) Gated DeltaNet
x0, w = 600, 380
cx = x0 + 140
panel(x0, w, "(c) Gated DeltaNet", "Qwen3.5-style, layers 0, 1, 2 and 4")
frame(x0, w, cx, PY0 + 22, PY1)
bw = 180
box(cx - bw / 2, PY1 - 40, bw, 28, "RMSNorm", NORM)
up(cx, PY1 - 4, PY1 - 12)
t(cx + bw / 2 + 8, PY1 - 22, "h", 12, serif=True, italic=True, anchor="start")
box(cx - bw / 2, PY1 - 90, bw, 26, "Linear 256 → 768", LIN, 11)
up(cx, PY1 - 40, PY1 - 64)
box(cx - bw / 2, PY1 - 128, bw, 26, "causal conv1d, kernel 4, then SiLU", ACT, 10.5)
up(cx, PY1 - 90, PY1 - 102)
for lab, dx in (("q", -60), ("k", 0), ("v", 60)):
    line([(cx, PY1 - 128), (cx + dx, PY1 - 144)], False)
    t(cx + dx, PY1 - 150, lab, 12, serif=True, italic=True)
for dx, lab in ((-60, "L2-norm ÷ 8"), (0, "L2-norm")):
    box(cx + dx - 30, PY1 - 186, 60, 22, lab, NORM, 9)
    up(cx + dx, PY1 - 158, PY1 - 164)
line([(cx + 60, PY1 - 158), (cx + 60, PY1 - 210)])
box(cx - bw / 2 - 10, PY1 - 280, bw + 20, 68, "", DEL)
t(cx, PY1 - 263, "gated delta rule, 4 heads × 64", 11)
t(cx, PY1 - 244, "Sₜ = αₜ Sₜ₋₁ (I − βₜ kₜkₜᵀ) + βₜ vₜkₜᵀ", 11.5, INK, serif=True, italic=True)
t(cx, PY1 - 226, "oₜ = Sₜ qₜ", 11.5, INK, serif=True, italic=True)
for dx in (-60, 0):
    up(cx + dx, PY1 - 186, PY1 - 212)
ax = cx + bw / 2 + 52                                                    # alpha and beta, from h
line([(cx + bw / 2, PY1 - 30), (ax, PY1 - 30), (ax, PY1 - 76)])
box(ax - 40, PY1 - 104, 80, 26, "Linear 256 → 8", LIN, 9.5)
box(ax - 62, PY1 - 166, 124, 48, "", ACT)
t(ax, PY1 - 147, "α = exp(−eᴬ·softplus(a + δ))", 9.5, serif=True, italic=True)
t(ax, PY1 - 129, "β = σ(b), one per head", 9.5, serif=True, italic=True)
up(ax, PY1 - 104, PY1 - 118)
line([(ax, PY1 - 166), (ax, PY1 - 246), (cx + bw / 2 + 12, PY1 - 246)])
box(cx - bw / 2, PY1 - 316, bw, 26, "RMSNorm per head", NORM, 10.5)
up(cx, PY1 - 276, PY1 - 288)
gx = cx + bw / 2 + 60                                                    # output gate, from h
line([(cx + bw / 2 + 4, PY1 - 18), (gx + 68, PY1 - 18), (gx + 68, PY1 - 340), (gx + 34, PY1 - 340)])
box(gx - 6, PY1 - 356, 40, 32, "Linear", LIN, 8.5, "256²", 8)
box(gx - 56, PY1 - 352, 40, 24, "SiLU", ACT, 9.5)
line([(gx - 6, PY1 - 340), (gx - 16, PY1 - 340)])
op(cx, PY1 - 340, "x")
up(cx, PY1 - 316, PY1 - 332)
line([(gx - 56, PY1 - 340), (cx + 9, PY1 - 340)])
box(cx - bw / 2, PY1 - 392, bw, 26, "Linear 256 → 256", LIN, 11)
up(cx, PY1 - 348, PY1 - 366)
up(cx, PY1 - 392, PY0 + 30)

# (d) TONN layer
x0, w = 995, 405
cx = x0 + 215
panel(x0, w, "(d) TONN layer", "ExeaLM, layers 0 and 3 (layer 0 shown)")
frame(x0, w, cx, PY0 + 22, PY1)
box(cx - 90, PY1 - 40, 180, 28, "RMSNorm", NORM)
up(cx, PY1 - 4, PY1 - 12)
box(cx - 150, PY1 - 92, 300, 28, "Linear 256 → 768 (+ bias): 8 slots × 3 streams × 32", LIN, 10)
up(cx, PY1 - 40, PY1 - 64)
SLOTS = [("RP", "lag 8", "rotary", "attention"), ("MI", "lag 16", "Σλ A(t−16)", "× B"), ("EP", "", "softmax", "pooling"),
         ("MIM", "lag 3", "C × Σλ", "A(t−3)B"), ("MI", "lag 1", "Σλ A(t−1)", "× B"), ("OS", "", "oscillating", "memory"),
         ("MI", "lag 128", "Σλ A(t−128)", "× B"), ("SM", "", "selective", "state space")]
sw, sg = 44, 3.5
sx0 = cx - (8 * sw + 7 * sg) / 2
for i, (ty, lag, f1, f2) in enumerate(SLOTS):
    x = sx0 + i * (sw + sg)
    mx = x + sw / 2
    for dx, lab in ((-11, "A"), (0, "B"), (11, "C")):
        line([(cx + (i - 3.5) * 30, PY1 - 92), (mx + dx, PY1 - 124)], True, MID, 0.7)
    if i == 0:
        t(mx - 11, PY1 - 128, "A", 8, MID)
        t(mx, PY1 - 128, "B", 8, MID)
        t(mx + 11, PY1 - 128, "C", 8, MID)
    box(x, PY1 - 214, sw, 82, "", TONN)
    t(mx, PY1 - 196, ty, 11.5, weight=600)
    t(mx, PY1 - 183, lag, 8.5, MID)
    t(mx, PY1 - 166, f1, 8, INK)
    t(mx, PY1 - 156, f2, 8, INK)
    box(x + 3, PY1 - 250, sw - 6, 22, "÷ rms", ACT, 8.5)
    up(mx, PY1 - 214, PY1 - 228)
    line([(mx, PY1 - 250), (cx + (i - 3.5) * 12, PY1 - 282)], True, MID, 0.7)
box(cx - 150, PY1 - 310, 300, 26, "concatenate 8 × 32 → Linear 256 → 256", LIN, 10)
up(cx, PY1 - 310, PY0 + 30)
for i, s_ in enumerate(("Each slot: streams A, B, C (32 ch)", "→ one circuit → 32 channels.", "λ: a learned decay per channel.",
                        "÷ rms: the slot's running size,", "from past batches.", "Layer 3 holds G (gated unit), EP,",
                        "OS, SM, RP lag 16, GI (gated", "memory), C lag 6, C lag 128.")):
    t(x0 + 26, PY0 + 34 + i * 13.5 + (8 if i >= 5 else 0) + (4 if i >= 2 else 0), s_, 9.5, MID, anchor="start")

# (e) SwiGLU MLP
x0, w = 1415, 170
cx = x0 + 95
panel(x0, w, "(e) SwiGLU MLP", "after every layer, in all three")
frame(x0, w, cx, PY0 + 22, PY1)
bw = 130
box(cx - bw / 2, PY1 - 40, bw, 28, "RMSNorm", NORM)
up(cx, PY1 - 4, PY1 - 12)
box(cx - bw / 2, PY1 - 92, bw, 28, "Linear 256 → 1,360", LIN, 10.5)
up(cx, PY1 - 40, PY1 - 64)
line([(cx, PY1 - 92), (cx - 30, PY1 - 112)], False)
line([(cx, PY1 - 92), (cx + 30, PY1 - 112)], False)
t(cx - 30, PY1 - 118, "a", 12, serif=True, italic=True)
t(cx + 30, PY1 - 118, "b", 12, serif=True, italic=True)
box(cx - 52, PY1 - 162, 44, 24, "SiLU", ACT, 10.5)
up(cx - 30, PY1 - 128, PY1 - 138)
op(cx, PY1 - 196, "x")
line([(cx - 30, PY1 - 162), (cx - 30, PY1 - 196), (cx - 9, PY1 - 196)])
line([(cx + 30, PY1 - 128), (cx + 30, PY1 - 196), (cx + 9, PY1 - 196)])
t(cx, PY1 - 210, "680", 9, MID)
box(cx - bw / 2, PY1 - 256, bw, 28, "Linear 680 → 256", LIN, 10.5)
up(cx, PY1 - 204, PY1 - 228)
up(cx, PY1 - 256, PY0 + 30)

# legend and caption
lx, ly = 30, H - 28
for i, (lab, fill) in enumerate((("norm", NORM), ("linear", LIN), ("attention", ATT), ("delta rule", DEL), ("TONN circuit", TONN), ("other", ACT))):
    els.append(f'<rect x="{lx + i * 125}" y="{ly - 11}" width="16" height="12" rx="2" fill="{fill}" stroke="{INK}" stroke-width="0.8"/>')
    t(lx + i * 125 + 22, ly, lab, 11, MID, anchor="start")
t(W - 30, ly, "⊕ residual add   ⊗ elementwise product   σ sigmoid", 11, MID, anchor="end")

markers = (f'<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
           f'<path d="M0,1 L10,5 L0,9 z" fill="{INK}"/></marker>')
page = f"""<!doctype html><html><head><meta charset="utf-8"><style>html,body{{margin:0;background:#fff}}</style></head><body>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><defs>{markers}</defs>
<rect width="{W}" height="{H}" fill="#fff"/>{chr(10).join(els)}</svg></body></html>"""
src = os.path.join(HERE, "arch_compare.html")
open(src, "w").write(page)
subprocess.run(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--headless=new", "--disable-gpu",
                "--hide-scrollbars", "--force-device-scale-factor=2", f"--window-size={W},{H}", "--virtual-time-budget=5000",
                f"--screenshot={os.path.join(HERE, 'arch_compare.png')}", "file://" + src],
               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("wrote", os.path.join(HERE, "arch_compare.png"))
