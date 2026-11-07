"""Figure 1: multisubject coupled sparse autoencoder and one real semantic-cortical component (c18).

v6e: saturated gold only on the component-18 thread (columns, tags, leaders); W_brain drawn as four
delay blocks of 50 columns with z-tilde as four matching segments; active code cells select their
columns with chevrons and value-scaled tints.

Run:
  MPLCONFIGDIR="$TMPDIR/mpl" uv run --no-project --with matplotlib --with numpy --with pillow python -I figure1.py
Writes figure1.pdf and figure1.png (600 dpi) next to this script. Print width 5.5 in, height 3.0 in.
All coordinates are inches on that canvas, so every font size is the size on paper.

Real: component 18's LLM-judge description, most-activating text, delay weights and cortical maps.
Schematic: all tensors and matrices in panel a, and the activity trace in the "when" strip (the
response there is that schematic trace convolved with the real kernel).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, FancyArrowPatch
from matplotlib.path import Path
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
for name in ("Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf", "Arial Bold Italic.ttf"):
    fm.fontManager.addfont(os.path.join("/System/Library/Fonts/Supplemental", name))
plt.rcParams.update({
    "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42, "text.hinting": "none",
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.bfit": "Arial:bold:italic", "mathtext.default": "it",
})

W, H, DPI = 5.5, 3.0, 600
INK, GREY, RULE = "#222222", "#666666", "#9A9A9A"
# v6d palette (OKLCH-designed: blue L=0.47 C=0.10 h=250, orange L=0.57 C=0.12 h=50, gold L=0.83 C=0.15 h=90;
# tints are the same hues blended toward white). Copy these constants into fig2_v2/strip.py if adopted.
BLUE, ORANGE = "#295E90", "#AE5F2F"           # semantic side / brain side (text on white 6.8:1 / 4.7:1)
GOLD = "#EDC23E"                              # the one highlight colour: component 18
BLUE_T, ORANGE_T = "#E1E8EF", "#F2E5DE"       # matrix fills   = mix(BLUE, 0.14), mix(ORANGE, 0.16)
BLUE_A, ORANGE_A = "#94AEC8", "#DBB7A1"       # active columns = mix(BLUE, 0.50), mix(ORANGE, 0.45)
GOLD_T = "#F7E4A8"                            # response fill  = mix(GOLD, 0.45)
OPER = "#B8B8B8"                              # large light operators  =  ≈  ∗
LINE = "#8C8C8C"                              # hollow (echo) stems in the "when" strip
FILL = "#EEEEEE"
LW = 0.6
FS_MATH, FS, FS_S = 7.2, 6.2, 5.4            # math + panel letters (scripts stay >= 5 pt), titles, labels
FS_XS = FS_S                                  # v6e: one label size only

fig = plt.figure(figsize=(W, H), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
REND = fig.canvas.get_renderer()
rng = np.random.default_rng(7)


# ---------------------------------------------------------------- helpers
def T(x, y, s, fs=FS, color=INK, ha="left", va="baseline", weight="normal", style="normal",
      rot=0, z=6):
    return ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, fontweight=weight,
                   fontstyle=style, rotation=rot, rotation_mode="anchor", zorder=z)


def tw(s, fs=FS, weight="normal", style="normal"):
    t = ax.text(0, 0, s, fontsize=fs, fontweight=weight, fontstyle=style)
    w = t.get_window_extent(REND).width / fig.dpi
    t.remove()
    return w


HLPAD = 0.02                        # extra space on each side of a highlighted run


def hl_box(x, y, w, fs, z=3.5):
    """Borderless pale-gold highlighter behind a text run with baseline y."""
    em = fs / 72
    r = (x - 0.014, y - 0.30 * em, w + 0.028, 1.17 * em)
    ax.add_patch(Rectangle(r[:2], r[2], r[3], fc=GOLD_T, ec="none", zorder=z))
    return r


def runs(x, y, parts, fs=FS, ha="left", dry=False):
    """Text runs side by side on one baseline. parts: (text, kwargs); kwargs may hold 'hl' (gold
    highlighter) and 'fs'. Returns the x where the line ends."""
    if ha == "center":
        x -= runs(0, 0, parts, fs, dry=True) / 2
    elif ha == "right":
        x -= runs(0, 0, parts, fs, dry=True)
    for s, kw in parts:
        kw = dict(kw); hl = kw.pop("hl", False); f = kw.pop("fs", fs)
        sp = tw("a a", f) - tw("aa", f)
        lead = len(s) - len(s.lstrip(" ")); trail = len(s) - len(s.rstrip(" "))
        core = s.strip(" ")
        x += lead * sp + (HLPAD if hl else 0)
        w = tw(core, f, kw.get("weight", "normal"), kw.get("style", "normal"))
        if not dry:
            if hl:
                hl_box(x, y, w, f)
            T(x, y, core, fs=f, **kw)
        x += w + trail * sp + (HLPAD if hl else 0)
    return x


def rbox(x0, y0, w, h, fc="white", ec=INK, lw=LW, r=0.03, z=3):
    p = FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                       lw=lw, zorder=z, joinstyle="round")
    ax.add_patch(p)
    return p


def arrow(pts, color=INK, lw=0.7, r=0.05, head=True, z=4):
    """Polyline with rounded corners and an optional filled head at the last point."""
    pts = [np.array(p, float) for p in pts]
    verts, codes = [pts[0]], [Path.MOVETO]
    for i in range(1, len(pts) - 1):
        d1 = pts[i] - pts[i - 1]; d1 /= np.linalg.norm(d1)
        d2 = pts[i + 1] - pts[i]; d2 /= np.linalg.norm(d2)
        verts += [pts[i] - d1 * r, pts[i], pts[i] + d2 * r]
        codes += [Path.LINETO, Path.CURVE3, Path.CURVE3]
    verts.append(pts[-1]); codes.append(Path.LINETO)
    style = "-|>,head_length=3.0,head_width=1.25" if head else "-"
    ax.add_patch(FancyArrowPatch(path=Path(verts, codes), arrowstyle=style, mutation_scale=1,
                                 lw=lw, ec=color, fc=color, shrinkA=0, shrinkB=0, zorder=z,
                                 capstyle="butt", joinstyle="round"))


def vstrip(x0, y0, w, h, colors, ec=INK, z=3, lw=LW):
    """Vertical tensor: stacked cells, index 0 on top."""
    n = len(colors); ch = h / n
    for i, c in enumerate(colors):
        ax.add_patch(Rectangle((x0, y0 + h - (i + 1) * ch), w, ch, fc=c, ec="none", zorder=z))
    ax.add_patch(Rectangle((x0, y0), w, h, fc="none", ec=ec, lw=lw, zorder=z + 0.02))


def mix(c, t):
    """Blend colour c toward white: t=0 white, t=1 full colour."""
    c = np.array(matplotlib.colors.to_rgb(c))
    return tuple(1 - t * (1 - c))


def gline(xs, ys, lw=1.0, z=5):
    """Plain gold line: the component-18 leader."""
    ax.plot(xs, ys, color=GOLD, lw=lw, zorder=z, solid_capstyle="butt", solid_joinstyle="miter")


def tag(xc, yc, s="c18", fs=FS_XS):
    """Gold pill that names component 18."""
    w = tw(s, fs) + 0.07; h = 0.1
    rbox(xc - w / 2, yc - h / 2, w, h, fc=GOLD, ec="none", lw=0, r=0.028, z=5)
    T(xc, yc - 0.358 * fs / 72, s, fs=fs, ha="center")
    return xc - w / 2, xc + w / 2


def oper(x, y, s, fs=9.5):
    """Large, light operator between blocks."""
    T(x, y, s, fs=fs, color=OPER, ha="center", va="center")


def swatch(x, y):
    """Small gold bar that opens each face of panel b (same gold as the column it describes)."""
    ax.add_patch(Rectangle((x, y - 0.03), 0.03, 0.115, fc=GOLD, ec="none", zorder=4))


def place_image(path, x0, y0, w_in, h_box):
    """Place a raster resized to exact 600-dpi pixels, centred vertically in a box of height h_box
    whose lower-left corner is (x0, y0)."""
    a = np.asarray(Image.open(path).convert("RGB")).copy()
    for edge, nxt in ((0, 1), (-1, -2)):          # drop a detached 1-px sliver on an image edge
        if (a[:, nxt] == 255).all():
            a[:, edge] = 255
    im = Image.fromarray(a)
    wpx = int(round(w_in * DPI)); hpx = int(round(wpx * im.height / im.width))
    im = im.resize((wpx, hpx), Image.LANCZOS)
    x0 = round(x0 * DPI) / DPI; y0 = round((y0 + (h_box - hpx / DPI) / 2) * DPI) / DPI
    ax.imshow(np.asarray(im), extent=(x0, x0 + wpx / DPI, y0, y0 + hpx / DPI),
              interpolation="none", zorder=2, aspect="auto")
    ax.set_xlim(0, W); ax.set_ylim(0, H)



# ---------------------------------------------------------------- geometry of panel a
NK, K = 50, 18                      # 50 components; c18 drawn at index 18 (labels are 0-based, c0..c49)
CW = 0.015                          # one latent unit = one column of W_sem = 0.015 in
MW = NK * CW
NDEL, CWB, BGAP = 4, 0.007, 0.02    # W_brain: four delay blocks of 50 columns (0.007 in each), thin gaps
BW = NK * CWB
SWB = NDEL * BW + (NDEL - 1) * BGAP
YM0, YM1 = 1.888, 2.388             # rows of every decoder matrix and output tensor
YC = (YM0 + YM1) / 2
GAP = 0.035                         # gap between a code strip and the matrix under it (holds the chevrons)
ZH = 0.08                           # height of the code strips
Y_TOP, Y_HDR, Y_LAB, Y_BOT = 2.825, 2.72, 2.595, 1.728   # baselines: objective, decoder names, code labels, names
Y_BUS = YM1 + GAP + ZH / 2

X_IN0, X_IN1 = 0.10, 0.77          # story card / language model
X_X = 0.84                          # x_t
SW = 0.07
X_E0, X_E1 = 0.995, 1.315          # encoder
X_S = 1.485                         # W_sem left edge
X_B = 3.32                          # W_brain stack left edge
XK_S = X_S + (K + 0.5) * CW
XB_J = [X_B + j * (BW + BGAP) for j in range(NDEL)]          # left edge of delay block j
XK_BJ = [x + (K + 0.5) * CWB for x in XB_J]                  # column 18 inside each delay block
XC_B = float(np.mean(XK_BJ))                                 # where the four merge into one leader
Y_BAR = YM0 - 0.028                                          # the merge bar under the stack
NSUB, CGAP = 6, 0.018
CARD_H = (YM1 - YM0 - (NSUB - 1) * CGAP) / NSUB
SHOWN = ("UTS02", "UTS03", "UTS06")

# schematic code values (0..1): active entry -> weight; c18 is the peak at t
ACT_T = {3: 0.5, K: 1.0, 26: 0.3, 33: 0.8, 46: 0.6}
LAM = np.array([0.917, 1.122, 1.008, 0.892])                 # real delay scales of c18 (2, 4, 6, 8 s)
LV = 0.15 + 0.85 * (LAM - LAM.min()) / (LAM.max() - LAM.min())   # -> 0.24, 1.0, 0.58, 0.15
SEGF = (0.65, 1.0, 0.8, 0.45)                                # schematic overall level of each segment
ACT_D = tuple({i: (LV[j] if i == K else min(1.0, v * SEGF[j])) for i, v in d.items()} for j, d in enumerate((
    {3: 0.8, K: 1, 26: 0.45, 39: 1.0, 46: 0.6},              # z_{t-1} (2 s)
    {3: 0.4, K: 1, 22: 0.75, 39: 1.0, 46: 0.3},              # z_{t-2} (4 s)
    {8: 0.9, K: 1, 22: 0.5, 30: 1.0, 44: 0.65},              # z_{t-3} (6 s)
    {8: 0.6, K: 1, 13: 1.0, 30: 0.45, 44: 0.8})))            # z_{t-4} (8 s)


def cell_col(v):
    """Grey of an active code cell, darker for a larger value."""
    return mix("#000000", 0.15 + 0.8 * v)


def col_col(base, v):
    """Tint of a selected decoder column, stronger for a larger code value."""
    return mix(base, 0.15 + 0.7 * v)


def gold_col(v):
    """Component 18 keeps its hue; the value sets how saturated the gold is."""
    return mix(GOLD, 0.62 + 0.38 * v)


def code_row(x0, y0, w, h, active, line=("#C4C4C4", 0.2), z=3):
    """One row of 50 code cells of width w; active ones grey scaled by value, component 18 gold
    (paler for a smaller value). Cells stay inside the frame; the frame is drawn last."""
    ax.add_patch(Rectangle((x0, y0), NK * w, h, fc="white", ec="none", zorder=z))
    for i, v in active.items():
        ax.add_patch(Rectangle((x0 + i * w, y0), w, h, fc=gold_col(v) if i == K else cell_col(v),
                               ec="none", zorder=z + 0.01 + (0.02 if i == K else 0)))
    if line:
        for i in range(1, NK):
            ax.plot([x0 + i * w] * 2, [y0, y0 + h], color=line[0], lw=line[1], zorder=z + 0.02,
                    solid_capstyle="butt")
    ax.add_patch(Rectangle((x0, y0), NK * w, h, fc="none", ec=INK, lw=LW, zorder=z + 0.05))


def matrix(x0, y0, w, h, active, tint, base, line=("white", 0.25), z=3, lw=LW):
    """A decoder block of 50 columns of width w (column slots as hairlines); selected columns tinted
    by code value; column 18 gold, paler where the code value is smaller."""
    ax.add_patch(Rectangle((x0, y0), NK * w, h, fc=tint, ec="none", zorder=z))
    for i, v in active.items():
        if i != K:
            ax.add_patch(Rectangle((x0 + i * w, y0), w, h, fc=col_col(base, v), ec="none", zorder=z + 0.01))
    ax.add_patch(Rectangle((x0 + K * w, y0), w, h, fc=gold_col(active.get(K, 1.0)), ec="none",
                           zorder=z + 0.015))
    if line:
        for i in range(1, NK):
            ax.plot([x0 + i * w] * 2, [y0, y0 + h], color=line[0], lw=line[1], zorder=z + 0.02,
                    solid_capstyle="butt")
    ax.add_patch(Rectangle((x0, y0), NK * w, h, fc="none", ec=INK, lw=lw, zorder=z + 0.05))


def chevrons(x0, w, active, z=2.95):
    """Small downward arrowheads in the gap: each active code cell selects the column below it.
    They start just under the strip's frame and end just above the matrix's frame."""
    hw = max(0.5 * w, 0.005)
    e = 0.006                                     # half frame width plus a hair
    for i, v in active.items():
        xc = x0 + (i + 0.5) * w
        ax.add_patch(Polygon([(xc - hw, YM1 + GAP - e), (xc + hw, YM1 + GAP - e), (xc, YM1 + e)],
                             closed=True, fc=gold_col(v) if i == K else cell_col(v), ec="none", zorder=z))


# ================================================================ (a) model
T(0.05, H - 0.045, "a", fs=FS_MATH, weight="bold", va="top")

# --- joint objective with both loss terms written out
obj = [(r"$L=$", {}), (r"$L_{\mathrm{brain}}$", dict(color=ORANGE)), (r"$\,+\,\alpha\,$", {}),
       (r"$L_{\mathrm{sem}}$", dict(color=BLUE)),
       ("      ", {}),
       (r"$L_{\mathrm{sem}}$", dict(color=BLUE)),
       (r"$=\Vert\mathbf{x}_t-\hat{\mathbf{x}}_t\Vert^2$", {}),
       ("      ", {}),
       (r"$L_{\mathrm{brain}}$", dict(color=ORANGE)),
       (r"$=\sum_s\Vert\mathbf{y}^{(s)}_t-\hat{\mathbf{y}}^{(s)}_t\Vert^2$", {})]
x_obj = W - 0.05 - runs(0, 0, obj, fs=FS_MATH, dry=True)
runs(x_obj, Y_TOP, obj, fs=FS_MATH)
T(x_obj - 0.06, Y_TOP, "trained jointly:", fs=FS_S, color=GREY, ha="right")
T(0.19, Y_TOP, "Shared code, two linear decoders", fs=FS, weight="bold")

# --- story text and language model (fixed feature extractor)
rbox(X_IN0, 2.328, X_IN1 - X_IN0, 0.44, fc="white", ec=RULE, lw=0.5)
story = [[("… my mother put", {})],
         [("sausages ", {}), ("and french", dict(hl=True))],
         [("fries on the table", dict(hl=True))],
         [("for dinner", dict(hl=True)), (" …", {})]]
for i, parts in enumerate(story):
    runs(X_IN0 + 0.035, 2.668 - i * 0.1, parts, fs=FS_S)
arrow([(0.26, 2.328), (0.26, 2.203)])
T(0.33, 2.245, "per TR (2 s)", fs=FS_XS, color=GREY)
rbox(X_IN0, 1.918, X_IN1 - X_IN0, 0.28, fc=FILL, ec=INK)
T((X_IN0 + X_IN1) / 2, 2.075, "Gemma-2-2b", fs=FS_S, ha="center")
T((X_IN0 + X_IN1) / 2, 1.965, "layer 25", fs=FS_S, ha="center", color=GREY)
arrow([(X_IN1, 2.058), (X_X, 2.058)])

# --- x_t (dense)
dense = rng.uniform(0.08, 0.72, 30)
vstrip(X_X, YM0, SW, YM1 - YM0, [mix("#000000", v) for v in dense])
T(X_X + SW / 2, YM1 + 0.045, r"$\mathbf{x}_t$", fs=FS_MATH, ha="center")
T(X_X + SW / 2, Y_BOT, "2304-d", fs=FS_S, color=GREY, ha="center")

# --- shared encoder
arrow([(X_X + SW, YC), (X_E0, YC)])
ax.add_patch(Polygon([(X_E0, YM0), (X_E0, YM1), (X_E1, YM1 - 0.08), (X_E1, YM0 + 0.08)],
                     closed=True, fc=FILL, ec=INK, lw=LW, zorder=3, joinstyle="round"))
for i, (s_, kw_) in enumerate((("shared", {}), ("encoder", {}),
                               ("ReLU,", dict(color=GREY)), ("TopK", dict(color=GREY)))):
    T((X_E0 + X_E1) / 2, YC + 0.10 - i * 0.085 - (0.012 if i > 1 else 0), s_, fs=FS_XS, ha="center", **kw_)
arrow([(X_E1, YC), (X_E1 + 0.085, YC), (X_E1 + 0.085, Y_BUS), (X_S - 0.012, Y_BUS)], r=0.04)

# --- decoder names: the same two names head the two faces of panel b
T(X_S, Y_HDR, "semantic decoder", fs=FS_S, color=BLUE)
T(X_B + SWB, Y_HDR, "brain decoder", fs=FS_S, color=ORANGE, ha="right")

# --- shared code z_t laid along the column axis of W_sem: entry k sits over column k and selects it
code_row(X_S, YM1 + GAP, CW, ZH, ACT_T)
matrix(X_S, YM0, CW, YM1 - YM0, ACT_T, BLUE_T, BLUE)
chevrons(X_S, CW, ACT_T)
T(X_S, Y_LAB, r"$\mathbf{z}_t$", fs=FS_MATH)
ax.plot([XK_S, XK_S], [YM1 + GAP + ZH + 0.006, Y_LAB], color=GOLD, lw=0.9, zorder=4)
_, xe = tag(XK_S, Y_LAB + 0.03)
T(xe + 0.05, Y_LAB + 0.045, "50 components,", fs=FS_S, color=GREY)
T(xe + 0.05, Y_LAB - 0.045, r"$k{=}5$ active", fs=FS_S, color=GREY)

# --- semantic readout as matrix times vector: W_sem z_t = sum_j z_j w_j = x_hat ~ x_t
xq = X_S + MW
oper(xq + 0.06, YC, r"$=$")
X_XH = xq + 0.12
recon = np.clip(dense + rng.normal(0, 0.07, 30), 0.05, 1)
vstrip(X_XH, YM0, 0.06, YM1 - YM0, [mix(BLUE, 0.15 + 0.8 * v) for v in recon])
oper(xq + 0.24, YC, r"$\approx$")
X_XT = xq + 0.30
vstrip(X_XT, YM0, 0.06, YM1 - YM0, [mix("#000000", v) for v in dense])
arrow([(X_XT + 0.06, YC), (X_XT + 0.13, YC)])
T(X_XT + 0.15, YC - 0.025, r"$L_{\mathrm{sem}}$", fs=FS_MATH, color=BLUE)
T(XK_S - 0.035, Y_BOT, r"$\mathbf{W}_{\mathrm{sem}}$", fs=FS_MATH, color=BLUE, ha="right")
T(XK_S + 0.035, Y_BOT, r"$\mathbf{z}_t=\Sigma_j\,z_j\,\mathbf{w}_j$", fs=FS_MATH)
T(X_XH + 0.03, Y_BOT, r"$\hat{\mathbf{x}}_t$", fs=FS_MATH, ha="center")
T(X_XT + 0.03, Y_BOT, r"$\mathbf{x}_t$", fs=FS_MATH, ha="center")

# --- delay step: z-tilde is one strip of four segments, segment delta = lambda_delta * z_{t-delta}
arrow([(xq, Y_BUS), (X_B - 0.012, Y_BUS)])
T(X_B - 0.09, Y_LAB - 0.045, r"$\tilde{\mathbf{z}}_{t,\delta}=\mathbf{\lambda}_{\delta}\odot\mathbf{z}_{t-\delta}$",
  fs=FS_MATH, ha="right")
for j in range(NDEL):
    code_row(XB_J[j], YM1 + GAP, CWB, ZH, ACT_D[j], line=("#DCDCDC", 0.15))
    T(XB_J[j] + BW - 0.01, Y_LAB - 0.045, f"{2 * (j + 1)} s", fs=FS_XS, color=GREY, ha="right")  # delay
ax.plot([XK_BJ[0]] * 2, [YM1 + GAP + ZH + 0.006, Y_LAB + 0.06], color=GOLD, lw=0.9, zorder=4)
tag(XK_BJ[0], Y_LAB + 0.09)

# --- brain readout: six stacked decoders, one per subject, each four delay blocks of the same 50 columns
for s in range(NSUB):
    y0 = YM1 - (s + 1) * CARD_H - s * CGAP
    for j in range(NDEL):
        matrix(XB_J[j], y0, CWB, CARD_H, ACT_D[j], ORANGE_T, ORANGE, line=("white", 0.18), lw=0.5)
    sid = f"UTS0{s + 1}"
    shown = sid in SHOWN
    T(X_B - 0.035, y0 + CARD_H / 2 - 0.019, sid, fs=FS_S, ha="right", color=INK if shown else GREY)
ax.plot([X_B - 0.275, X_B - 0.30, X_B - 0.30, X_B - 0.275], [YM0, YM0, YM1, YM1], color=RULE, lw=0.5,
        zorder=3, solid_capstyle="butt", solid_joinstyle="miter")      # thin bracket: six decoders
T(X_B - 0.335, YC, "6 subjects", fs=FS_XS, color=GREY, ha="center", va="center", rot=90)
for j in range(NDEL):
    chevrons(XB_J[j], CWB, ACT_D[j])
    ax.add_patch(Rectangle((XK_BJ[j] - CWB / 2, YM0), CWB, YM1 - YM0, fc=gold_col(LV[j]), ec="none", zorder=2.9))
    ax.plot([XK_BJ[j]] * 2, [YM0 - 0.006, Y_BAR], color=GOLD, lw=0.55, zorder=2.5, solid_capstyle="butt")
ax.plot([XK_BJ[0], XK_BJ[-1]], [Y_BAR, Y_BAR], color=GOLD, lw=0.9, zorder=2.5, solid_capstyle="butt")

xq = X_B + SWB
oper(xq + 0.06, YC, r"$=$")
X_YH = xq + 0.12
oper(xq + 0.24, YC, r"$\approx$")
X_YT = xq + 0.30
for s in range(NSUB):
    y0 = YM1 - (s + 1) * CARD_H - s * CGAP
    v = rng.uniform(0.05, 1, 4) ** 1.4
    vstrip(X_YH, y0, 0.06, CARD_H, [mix(ORANGE, 0.12 + 0.8 * a) for a in v], lw=0.5)
    vstrip(X_YT, y0, 0.06, CARD_H, [mix("#000000", 0.08 + 0.62 * a) for a in
                                    np.clip(v + rng.normal(0, 0.1, 4), 0, 1)], lw=0.5)
arrow([(X_YT + 0.06, YC), (X_YT + 0.13, YC)])
T(X_YT + 0.15, YC - 0.025, r"$L_{\mathrm{brain}}$", fs=FS_MATH, color=ORANGE)
T(XC_B - 0.04, Y_BOT, "≥ 0;  30,000 × 200", fs=FS_S, color=GREY, ha="right")
runs(XC_B + 0.04, Y_BOT, [(r"$\mathbf{W}^{(s)}_{\mathrm{brain}}$", dict(color=ORANGE)),
                          (r"$\;\tilde{\mathbf{z}}_t$", {})], fs=FS_MATH)
T(X_YH + 0.03, Y_BOT, r"$\hat{\mathbf{y}}^{(s)}_t$", fs=FS_MATH, ha="center")
T(X_YT + 0.05, Y_BOT, r"$\mathbf{y}^{(s)}_t$", fs=FS_MATH, ha="center")

# ================================================================ (b) one component
Y_HEAD, Y_RULE = 1.548, 1.473
T(0.05, Y_HEAD, "b", fs=FS_MATH, weight="bold")
xe = runs(0.19, Y_HEAD, [("One semantic–cortical component", dict(weight="bold"))], fs=FS)
tag(XK_S, Y_HEAD + 0.022)                     # the pill sits on the thread like a bead

XS0, XS1 = 0.045, 2.40          # semantic face
X_TXT = X_IN0                   # text column of panels b and 'when', aligned with the story card
XB0, XB1 = 2.50, W - 0.05       # cortical face
for (x0, x1, xk, ytop, col) in ((XS0, XS1, XK_S, YM0 - 0.006, BLUE), (XB0, XB1, XC_B, Y_BAR, ORANGE)):
    ax.plot([x0, x1], [Y_RULE, Y_RULE], color=col, lw=1.0, zorder=2, solid_capstyle="butt")
    gline([xk, xk], [ytop, Y_RULE - 0.007], lw=1.0, z=2.5)

# --- semantic face
yh = 1.341
runs(X_TXT, yh, [("semantic decoder: column 18 of ", dict(color=BLUE)), (r"$\mathbf{W}_{\mathrm{sem}}$", dict(color=BLUE))],
     fs=FS_MATH)
T(X_TXT, 1.236, "a direction in Gemma space", fs=FS_S, color=GREY)
runs(X_TXT, 1.131, [("LLM judge: ", dict(color=GREY)),
                         ("“food and drink items and meals”", {})],
     fs=FS_S)
T(X_TXT, 1.026, "top excerpts", fs=FS_S, color=GREY)
snips = [("… sausages ", "and french fries on the table for dinner", ""),
         ("… ", "plate of pasta and my two glasses of", " cheap red wine"),
         ("… ", "with bunch of yellow rice on it", " and some vegetable curry")]
for i, (a, b, c) in enumerate(snips):
    runs(X_TXT, 0.921 - i * 0.105, [(a, dict(color=GREY)), (b, dict(hl=True)),
                                         (c, dict(color=GREY))], fs=FS_S)

# --- cortical face
xe = runs(XB0 + 0.075, yh, [("brain decoder: c18 in each ", dict(color=ORANGE)),
                            (r"$\mathbf{W}^{(s)}_{\mathrm{brain}}$", dict(color=ORANGE))], fs=FS_MATH)
T(xe + 0.05, yh, "per subject, no anatomical alignment", fs=FS_S, color=GREY)
mw = 0.94
mh = mw * 477 / 937                     # common frame: the tallest of the three flatmaps
gap = (XB1 - XB0 - 3 * mw) / 2
y_map = 0.773
for i, sid in enumerate(("02", "03", "06")):
    x0, y0, w_, h_ = XB0 + i * (mw + gap), y_map, mw, mh
    place_image(os.path.join(HERE, "assets", f"c18_UTS{sid}.png"), x0, y0, mw, mh)
    T(x0 + w_ / 2, y_map - 0.093, f"UTS{sid}", fs=FS_S, ha="center")

# --- when it acts: the delay step is a convolution in time
Y_SR = 0.61
ax.plot([XS0, XB1], [Y_SR, Y_SR], color=RULE, lw=1.0, zorder=2, solid_capstyle="butt")
T(X_TXT, 0.465, "When it acts", fs=FS, weight="bold")

lam = LAM                                        # real: learned delay weights of component 18
zk = np.zeros(10); zk[1], zk[2] = 1.0, 0.6       # schematic activity, one sample per TR
resp = np.convolve(zk, np.r_[0, lam])[:10]       # r_t = sum_d lam_d z_(t-d)
YB, U, P = 0.175, 0.15, 0.125                    # baseline, inches per unit, inches per TR


def stem(x, v, col=GOLD, z=4):
    if col == GOLD:
        ax.plot([x, x], [YB, YB + v * U], color=GOLD, lw=0.7, zorder=z, solid_capstyle="butt")
        ax.plot([x], [YB + v * U], marker="o", ms=2.2, mfc=GOLD, mec="none", zorder=z + 0.02)
    else:                                     # hollow echo of an activity stem
        ax.plot([x, x], [YB, YB + v * U], color=LINE, lw=0.6, zorder=z, solid_capstyle="butt")
        ax.plot([x], [YB + v * U], marker="o", ms=2.6, mfc="white", mec=LINE, mew=0.6, zorder=z + 0.02)


def taxis(x0, n=9):
    ax.plot([x0, x0 + n * P + 0.03], [YB, YB], color=INK, lw=0.5, zorder=3)
    for i in range(n + 1):
        ax.plot([x0 + i * P] * 2, [YB, YB - 0.022], color=INK, lw=0.4, zorder=3)


# activity
XA = 0.85
taxis(XA, 9)
for i in np.nonzero(zk)[0]:
    stem(XA + i * P, zk[i])
T(XA, 0.465, r"$z^{(18)}(t)$", fs=FS_MATH)
for i, s in enumerate(("… and french fries", "on the table for dinner")):
    runs(XA + 3.0 * P, 0.335 - i * 0.097, [(s, dict(hl=True))], fs=FS_S)
T(XA, 0.06, "TR = 2 s", fs=FS_XS, color=GREY)

# kernel
oper(2.155, 0.295, r"$\ast$", fs=11)
XKER = 2.305
ax.plot([XKER, XKER + 0.73], [YB, YB], color=INK, lw=0.5, zorder=3)
for j, v in enumerate(lam):
    xs = XKER + 0.095 + j * 0.18
    stem(xs, v)
    T(xs, YB + v * U + 0.04, f"{v:.2f}", fs=FS_XS, ha="center")
    T(xs - tw(str(2 * (j + 1)), FS_XS) / 2, 0.06, f"{2 * (j + 1)}" + (" s" if j == 3 else ""),
      fs=FS_XS, color=GREY)
runs(XKER, 0.465, [(r"$\lambda^{(18)}$", {}), ("  delay scales", dict(color=GREY, fs=FS_S))], fs=FS_MATH)

# response
oper(3.185, 0.295, r"$=$", fs=11)
XR = 3.335
taxis(XR, 9)
for i in np.nonzero(zk)[0]:
    stem(XR + i * P, zk[i], col=INK, z=4.3)
nz = np.nonzero(resp)[0]
xs = np.r_[XR + (nz[0] - 1) * P, XR + nz * P, XR + (nz[-1] + 1) * P]
ys = np.r_[YB, YB + resp[nz] * U, YB]
ax.add_patch(Polygon(np.c_[xs, ys], closed=False, fc=mix(GOLD, 0.18), ec=GOLD, lw=0.8, zorder=4,
                     joinstyle="round"))
ax.plot(xs[1:-1], ys[1:-1], ls="none", marker="o", ms=1.8, mfc=GOLD, mec="none", zorder=4.1)
xt = 4.72
T(xt, 0.465, "delayed response", fs=FS_S)
T(xt, 0.355, "2–8 s after the activity", fs=FS_XS, color=GREY)

# ---------------------------------------------------------------- layout self-check
fig.canvas.draw()
boxes = []
for t in ax.texts:
    bb = t.get_window_extent(REND)
    s_ = t.get_text()
    eff = t.get_fontsize() * (0.7 if ("$" in s_ and ("_" in s_ or "^" in s_)) else 1.0)
    boxes.append((s_, bb.x0 / DPI, bb.y0 / DPI, bb.x1 / DPI, bb.y1 / DPI, eff))
nbad = 0
for s_, x0, y0, x1, y1, fs_ in boxes:
    if x0 < 0.02 or y0 < 0.02 or x1 > W - 0.02 or y1 > H - 0.02:
        print("NEAR EDGE:", repr(s_), round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)); nbad += 1
    if fs_ < 5.0:
        print("SMALL FONT:", repr(s_), round(fs_, 2)); nbad += 1
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        a_, b_ = boxes[i], boxes[j]
        e = 0.004      # inches; ink boxes of neighbouring runs may touch
        if a_[1] + e < b_[3] and b_[1] + e < a_[3] and a_[2] + e < b_[4] and b_[2] + e < a_[4]:
            print("TEXT OVERLAP:", repr(a_[0]), "|", repr(b_[0])); nbad += 1
for s_, x0, y0, x1, y1, fs_ in boxes:          # no text may sit on a component-18 leader (the c18 pill
    for xk in (XK_S, XC_B):                     # in the panel-b title is pinned on it by design)
        if s_ != "c18" and x0 - 0.015 < xk < x1 + 0.015 and y0 < YM0 and y1 > Y_RULE:
            print("TEXT ON LEADER:", repr(s_)); nbad += 1
print(f"checked {len(boxes)} text items; smallest effective size "
      f"{min(b[5] for b in boxes):.2f} pt; problems: {nbad}")

fig.savefig(os.path.join(HERE, "figure1.pdf"))
fig.savefig(os.path.join(HERE, "figure1.png"), dpi=DPI)
print("saved", fig.get_size_inches())
