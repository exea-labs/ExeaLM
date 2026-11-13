"""Draw docs/figures/arch_17m.png: the 17M ExeaLM next to the same-size Llama it is tested against.

Writes an SVG page (arch_17m.html) and renders it with headless Chrome:
  python docs/figures/arch_17m.py
Circuits shown are seed 0's starting draw (the 17M runs keep it: the circuit search was off). Parameter counts come
from building both models with the arguments of experiments/full_small.py.
"""
import html
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1800, 1140
PAPER, PANEL = "#111418", "#181c21"
INK, INK2, INK3 = "#ece6d8", "#c9c3b5", "#a29c8e"
LINE, LINE2 = "rgba(236,230,216,0.16)", "rgba(236,230,216,0.34)"
ORANGE, RG, AG = "#ff6a3d", "#ff8a8a", "#ff6fc8"
MONO, SERIF = "'DM Mono', ui-monospace, monospace", "'Fraunces', Georgia, serif"

els = []


def text(x, y, s, size=14, fill=INK, serif=False, weight=400, anchor="start"):
    els.append(f'<text x="{x}" y="{y}" font-family="{SERIF if serif else MONO}" font-size="{size}" '
               f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{html.escape(s)}</text>')


def rect(x, y, w, h, fill=PANEL, stroke=LINE, rx=8, sw=1):
    els.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def path(d, stroke=LINE2, sw=1.5, arrow=None, fill="none"):
    m = f' marker-end="url(#{arrow})"' if arrow else ""
    els.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{m}/>')


def tint(c, a):
    r, g, b = (int(c[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{a})"


# ------------------------------------------------------------------ header
text(64, 96, "ExeaLM and a same-size Llama", 46, INK, serif=True, weight=500)
text(64, 136, "The 17M test models. Same width, depth, vocabulary, MLPs and norms. Two of the six layers differ.", 16, INK3)

# ------------------------------------------------------------------ the two stacks
CW = 440
STACKS = {
    64: ("Llama", "17.83M parameters", ["attn"] * 6),
    584: ("ExeaLM", "18.08M parameters", ["tonn", "attn", "attn", "tonn", "attn", "attn"]),
}
ROW0, RH, GAP = 372, 62, 10
for x, (name, params, kinds) in STACKS.items():
    text(x, 214, name, 30, INK, serif=True, weight=500)
    text(x, 242, params, 15, INK3)
    cx = x + CW / 2
    text(cx, 284, "tokens", 14, INK3, anchor="middle")
    path(f"M{cx},291 V306", arrow="a")
    rect(x, 310, CW, 44)
    text(x + 16, 338, "embedding", 16)
    text(x + CW - 16, 338, "32,768 × 320, shared with the output", 13, INK3, anchor="end")
    path(f"M{cx},356 V368", arrow="a")
    for i, k in enumerate(kinds):
        y = ROW0 + i * (RH + GAP)
        text(x + 18, y + 37, str(i), 15, INK3, anchor="middle")
        if k == "tonn":
            rect(x + 42, y, 230, RH, fill=tint(ORANGE, 0.13), stroke=ORANGE, sw=1.5)
            text(x + 58, y + 27, "TONN layer", 16, ORANGE, weight=500)
            text(x + 58, y + 48, "10 circuits × 32 channels", 13, INK2)
        else:
            rect(x + 42, y, 230, RH)
            text(x + 58, y + 27, "attention", 16)
            text(x + 58, y + 48, "5 heads × 64, RoPE", 13, INK3)
        rect(x + 278, y, CW - 278, RH)
        text(x + 294, y + 27, "SwiGLU MLP", 16)
        text(x + 294, y + 48, "848 wide", 13, INK3)
    yb = ROW0 + 6 * (RH + GAP) - GAP
    path(f"M{cx},{yb + 2} V{yb + 22}", arrow="a")
    rect(x, yb + 26, CW, 44)
    text(cx, yb + 54, "RMSNorm → next-token probabilities", 15, anchor="middle")

# ------------------------------------------------------------------ inside the TONN layers
X0, XW = 1124, 612
text(X0, 214, "Inside the two TONN layers", 30, INK, serif=True, weight=500)
text(X0, 242, "10 small circuits side by side. Seed 0's draw.", 15, INK3)
for j, s in enumerate(("norm, project in", "10 circuits", "project out, add")):
    bx = X0 + j * 218
    rect(bx, 270, 176, 40, fill="none", stroke=LINE2)
    text(bx + 88, 295, s, 14, INK2, anchor="middle")
    if j < 2:
        path(f"M{bx + 180},290 H{bx + 212}", arrow="a")

CELL, CG, CX0 = 50, 4, 1200
LAYERS = {0: ["RP", "MIM", "MI", "MI", "MI", "EP", "SM", "OS"], 3: ["RP", "C", "C", "G", "GI", "EP", "SM", "OS"]}
for r, (li, others) in enumerate(LAYERS.items()):
    y = 352 + r * 48
    for c, t in enumerate(["RG"] + others + ["AG"]):
        cx = CX0 + c * (CELL + CG)
        col = {"RG": RG, "AG": AG}.get(t)
        rect(cx, y, CELL, 40, fill=tint(col, 0.16) if col else PANEL, stroke=col or LINE2, rx=6, sw=1.3 if col else 1)
        text(cx + CELL / 2, y + 26, t, 14, col or INK2, weight=500 if col else 400, anchor="middle")
# connectors from ExeaLM's TONN rows (layers 0 and 3) to their slot rows
ex_right = 584 + CW
for li, (yrow, yslot, xv) in {0: (ROW0 + RH / 2, 372, 1080), 3: (ROW0 + 3 * (RH + GAP) + RH / 2, 420, 1100)}.items():
    path(f"M{ex_right + 2},{yrow} H{xv} V{yslot} H{CX0 - 6}", stroke=ORANGE, sw=1.5, arrow="ao")
    text(1112 if li == 0 else 1110, yslot - 8, f"layer {li}", 12, INK3)

# library and board panels
PY, PH, PW = 480, 290, 296
rgx, agx = CX0 + CELL / 2, CX0 + 9 * (CELL + CG) + CELL / 2
path(f"M{rgx},442 V{PY - 4}", stroke=RG, sw=1.5, arrow="ar")
path(f"M{agx},442 V{PY - 4}", stroke=AG, sw=1.5, arrow="ag")

lx = X0
rect(lx, PY, PW, PH, stroke=tint(RG, 0.55))
text(lx + 18, PY + 34, "Library search (RG)", 16, RG, weight=500)
hit = {1, 3, 4, 6}
for i in range(7):
    by = PY + 58 + i * 13
    on = i in hit
    els.append(f'<rect x="{lx + 18}" y="{by}" width="120" height="9" rx="2" fill="{tint(RG, 0.75) if on else "rgba(236,230,216,0.14)"}"/>')
    if on:
        path(f"M{lx + 142},{by + 4.5} C{lx + 175},{by + 4.5} {lx + 175},{PY + 104} {lx + 198},{PY + 104}", stroke=tint(RG, 0.6), sw=1.2)
for k in range(8):
    els.append(f'<rect x="{lx + 200 + k * 10}" y="{PY + 99}" width="8" height="10" rx="1.5" fill="rgba(236,230,216,0.55)"/>')
text(lx + 239, PY + 130, "this chunk", 11, INK3, anchor="middle")
text(lx + 78, PY + 162, "library", 11, INK3, anchor="middle")
for i, s in enumerate(("Read-only store of 64-token", "chunks. Once per 64 tokens it", "encodes what it just read and",
                       "takes the 4 closest chunks.", "Every token attends over them.")):
    text(lx + 18, PY + 192 + i * 20, s, 13, INK2)

bx = X0 + PW + 20
rect(bx, PY, PW, PH, stroke=tint(AG, 0.55))
text(bx + 18, PY + 34, "Agent board (AG)", 16, AG, weight=500)
bcx, bcy = bx + PW / 2, PY + 104
rect(bcx - 46, bcy - 24, 92, 48, fill=tint(AG, 0.12), stroke=AG, rx=6)
for k in range(5):
    els.append(f'<rect x="{bcx - 36 + k * 15}" y="{bcy - 12}" width="11" height="24" rx="2" fill="{tint(AG, 0.55)}"/>')
for (ax, ay), lab in zip(((bx + 46, PY + 70), (bx + 46, PY + 138), (bx + PW - 46, PY + 70), (bx + PW - 46, PY + 138)), "ABCD"):
    els.append(f'<circle cx="{ax}" cy="{ay}" r="15" fill="{PANEL}" stroke="{LINE2}" stroke-width="1.2"/>')
    text(ax, ay + 4.5, lab, 13, INK2, anchor="middle")
    ex = bcx - 50 if ax < bcx else bcx + 50
    sx = ax + 18 if ax < bcx else ax - 18
    path(f"M{sx},{ay} L{ex},{bcy + (-8 if ay < bcy else 8)}", stroke=tint(AG, 0.6), sw=1.2)
text(bcx, PY + 172, "copies of the model", 11, INK3, anchor="middle")
for i, s in enumerate(("Shared by a swarm of copies.", "Every 8 tokens each posts one", "message summing up those 8",
                       "tokens. Every token reads the", "earlier posts by attention.")):
    text(bx + 18, PY + 192 + i * 20, s, 13, INK2)

# legend of circuit codes
LEG = [("RG", "library search", RG), ("AG", "agent board", AG), ("RP", "attention circuit", None),
       ("MIM", "induction-like memory", None), ("MI", "co-occurrence count", None), ("EP", "soft pooling", None),
       ("SM", "selective state space", None), ("OS", "oscillating memory", None), ("GI", "LSTM-like memory", None),
       ("G", "gated unit", None), ("C", "short convolution", None)]
for n, (code, desc, col) in enumerate(LEG):
    lx_, ly_ = X0 + (0, 222, 420)[n % 3], 806 + (n // 3) * 24
    text(lx_, ly_, code, 13, col or INK, weight=500)
    text(lx_ + 36, ly_, desc, 13, INK3)

text(X0, 934, "Speed", 22, INK, serif=True, weight=500)
for i, s in enumerate(("1.30× the Llama's time per step on the Mac Studio, same batches.",
                       "Not profiled yet. TONN's circuits run as plain PyTorch;", "attention uses a fused kernel.")):
    text(X0, 964 + i * 22, s, 14, INK2)

# ------------------------------------------------------------------ where the parameters are
text(64, 934, "Where the parameters are", 22, INK, serif=True, weight=500)
text(504, 934, "Llama", 13, INK3, anchor="end")
text(1024, 934, "ExeaLM", 13, INK3, anchor="end")
ROWS = [("embedding, shared with the output", "10.49M", "10.49M", False), ("MLPs", "4.89M", "4.89M", False),
        ("attention layers", "2.46M (6)", "1.64M (4)", False), ("TONN layers", "–", "0.84M (2)", True),
        ("library encoder", "–", "0.23M", True)]
for i, (lab, a, b, new) in enumerate(ROWS):
    y = 964 + i * 22
    text(64, y, lab, 14, INK2)
    text(504, y, a, 14, INK2, anchor="end")
    text(1024, y, b, 14, ORANGE if new else INK2, anchor="end")
path(f"M64,{964 + 4 * 22 + 10} H1024", stroke=LINE2, sw=1)
text(64, 964 + 5 * 22 + 8, "total", 14, INK, weight=500)
text(504, 964 + 5 * 22 + 8, "17.83M", 14, INK, weight=500, anchor="end")
text(1024, 964 + 5 * 22 + 8, "18.08M", 14, INK, weight=500, anchor="end")

markers = "".join(
    f'<marker id="{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
    for i, c in (("a", LINE2), ("ao", ORANGE), ("ar", RG), ("ag", AG)))
page = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,400;9..144,500&display=block" rel="stylesheet">
<style>html,body{{margin:0;background:{PAPER};}} svg{{display:block}}</style></head><body>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>{markers}</defs><rect width="{W}" height="{H}" fill="{PAPER}"/>
{chr(10).join(els)}
</svg></body></html>"""
src = os.path.join(HERE, "arch_17m.html")
open(src, "w").write(page)
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                f"--window-size={W},{H}", "--virtual-time-budget=10000",
                f"--screenshot={os.path.join(HERE, 'arch_17m.png')}", "file://" + src],
               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("wrote", os.path.join(HERE, "arch_17m.png"))
