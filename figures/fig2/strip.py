"""Figure 2c: per-component stability across 20 seeds (cortical map vs semantic decoder).

Writes strip.pdf, strip.png (600 dpi) and layout_preview.png (whole-figure mock-up) next to this file.
Style constants copied from fig1_v6/figure1.py.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "..", "catalog", "components.csv")

SHOWN = [8, 23, 2, 45]          # components shown as flatmaps above the strip (placeholders)
THRESH = 0.5                    # cortical-map stability below this: grey tick label

plt.rcParams.update({
    "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42, "text.hinting": "none",
})
W, H, DPI = 5.5, 1.4, 600
INK, GREY, RULE = "#222222", "#666666", "#9A9A9A"
BLUE, ORANGE = "#0072B2", "#D55E00"          # semantic / brain
GOLD = "#F7C61C"                              # highlight
FILL = "#EEEEEE"
LW = 0.6
FS, FS_S, FS_XS = 6.2, 5.4, 5.1


def draw_strip(fig, rect, letter="c"):
    """rect = (x0, y0, w, h) in inches of the whole panel, including labels."""
    fw, fh = fig.get_size_inches()
    x0, y0, w, h = rect
    L, R, B, T = 0.40, 0.04, 0.21, 0.10                # inner margins (in)
    ax = fig.add_axes([(x0 + L) / fw, (y0 + B) / fh, (w - L - R) / fw, (h - B - T) / fh])
    df = pd.read_csv(CSV).rename(columns={"cortical_map_stability_20seeds": "brn_cos", "semantic_decoder_stability_20seeds": "dec_cos"})
    df["atom"] = df.component.str[1:].astype(int)
    df = df.sort_values("brn_cos", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))
    b, s = df.brn_cos.values, df.dec_cos.values
    ax.axhline(THRESH, color=RULE, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
    ax.vlines(x, np.minimum(b, s), np.maximum(b, s), color=RULE, lw=0.5, zorder=2)
    ax.plot(x, s, "o", ms=2.8, mfc="white", mec=BLUE, mew=0.6, zorder=3, clip_on=False)
    ax.plot(x, b, "o", ms=2.8, mfc=ORANGE, mec=ORANGE, mew=0.0, zorder=4, clip_on=False)
    ax.set_xlim(-0.8, len(df) - 0.2)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_ylabel("similarity\nacross seeds", fontsize=FS_S, color=INK, labelpad=2, linespacing=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"c{a}" for a in df.atom], rotation=90, fontsize=FS_XS)
    for lab, a, bc in zip(ax.get_xticklabels(), df.atom, b):
        if a in SHOWN:
            lab.set_fontweight("bold"); lab.set_color(INK)
            lab.set_bbox(dict(boxstyle="round,pad=0.18,rounding_size=0.35", fc=GOLD, ec=INK, lw=0.5))
        else:
            lab.set_color(INK if bc >= THRESH else RULE)
    ax.tick_params(axis="both", labelsize=FS_XS, width=0.5, length=2, color=INK, labelcolor=INK, pad=1.5)
    ax.tick_params(axis="x", length=0, pad=2.5)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        ax.spines[k].set_linewidth(0.5); ax.spines[k].set_color(INK)
    ax.text(0.995, 1.03, "cortical map", color=ORANGE, fontsize=FS_S, ha="right", va="top",
            transform=ax.transAxes)
    ax.text(0.995, 0.91, "semantic decoder", color=BLUE, fontsize=FS_S, ha="right", va="top",
            transform=ax.transAxes)
    fig.text((x0 + 0.03) / fw, (y0 + h - 0.02) / fh, letter, fontsize=8, fontweight="bold",
             ha="left", va="top", color=INK)
    return ax


def tag(ax, xc, yc, s):
    ax.text(xc, yc, s, fontsize=FS_XS, fontweight="bold", ha="center", va="center", color=INK,
            bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.35", fc=GOLD, ec=INK, lw=0.5))


def layout_preview(path):
    HT, HS = 1.85, H                                  # top row, strip
    fh = HT + HS
    fig = plt.figure(figsize=(W, fh), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, fh); ax.axis("off")
    ax.text(0.03, fh - 0.02, "b", fontsize=8, fontweight="bold", va="top", color=INK)
    x_l, x_r, gap = 0.20, W - 0.04, 0.10
    cw = (x_r - x_l - 3 * gap) / 4
    for i, a in enumerate(SHOWN):
        xc0 = x_l + i * (cw + gap); xm = xc0 + cw / 2
        tag(ax, xm, fh - 0.10, f"c{a}")
        ax.text(xm, fh - 0.25, "one-line description placeholder", fontsize=FS_S, ha="center",
                va="center", color=INK)
        ax.add_patch(Rectangle((xc0, HS + 0.20), cw, fh - 0.34 - HS - 0.20, fc=FILL, ec="none"))
        ax.text(xm, HS + 0.20 + (fh - 0.34 - HS - 0.20) / 2, "flatmap", fontsize=FS_S,
                ha="center", va="center", color=GREY)
        ax.text(xm, HS + 0.09, "0.71 · 0.22 · 0.91 · 3.4%", fontsize=FS_XS, ha="center",
                va="center", color=GREY)
    draw_strip(fig, (0, 0, W, HS), "c")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    draw_strip(fig, (0, 0, W, H), "c")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"strip.{ext}"), dpi=DPI)
    plt.close(fig)
    layout_preview(os.path.join(HERE, "layout_preview.png"))
