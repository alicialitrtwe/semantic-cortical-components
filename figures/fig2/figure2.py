"""Figure 2, assembled as one figure at the text width.

Panels a-d: components c8, c23, c2, c45 (three subject flatmaps, the fsaverage map, four excerpts, a side
column of per-component values). Panel e: all 50 components on one 0-1 similarity axis (cross-seed
stability of the cortical map and of the semantic decoder, cross-subject consistency).
All coordinates are inches on a 5.5 in canvas, so every font size is the printed size. The brain maps are
the pycortex renders cached by figures/fig2/component_panel.py --cache_dir <cache> (render those first);
everything else is vector. Values come from catalog/components.csv and, for panel e, the per-component
output of scripts/31_cross_subject_consistency.py (seed 0, final epoch).
Colour scales: the three subject maps of a component share one scale (0 to the 99th percentile of the
subjects' weights, as rendered); the fsaverage map is an average over six subjects and was rendered on
its own 99th percentile, so it gets its own colour bar.

Usage:
    uv run python figures/fig2/figure2.py --model_dir <seed-0 run> --consistency_json <file> [--cache_dir outputs/cache/figure2]
"""
import glob
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
import argparse
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--model_dir", required=True, help="run folder of the seed-0 coupled SAE")
_ap.add_argument("--consistency_json", required=True,
                 help="output of scripts/31_cross_subject_consistency.py (with the per-component list)")
_ap.add_argument("--cache_dir", default=os.path.join(PROJECT, "outputs", "cache", "figure2"),
                 help="render cache written by component_panel.py --cache_dir")
_ap.add_argument("--catalog", default=os.path.join(PROJECT, "catalog", "components.csv"))
_ap.add_argument("--out", default=os.path.join(HERE, "figure2"))
_CLI = _ap.parse_args()
sys.argv = sys.argv[:1]
_spec = importlib.util.spec_from_file_location("panel", os.path.join(HERE, "component_panel.py"))
panel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(panel)          # catalog reader, contexts, per-subject weights (imports pycortex)
panel.MODEL_DIR = os.path.abspath(_CLI.model_dir)
panel.MODEL_NAME = os.path.basename(panel.MODEL_DIR.rstrip("/"))
panel.FIGS_DIR = os.path.join(PROJECT, "reports", "figures", panel.MODEL_NAME)
panel.CATALOG = _CLI.catalog
panel.FSAV_CACHE = os.path.join(os.path.abspath(_CLI.cache_dir), "fsaverage_atom_maps")
panel.FLATMAP_CACHE = os.path.join(os.path.abspath(_CLI.cache_dir), "flatmap_pngs")

CACHE = os.path.abspath(_CLI.cache_dir)
XSUBJ_JSON = _CLI.consistency_json
OUT = _CLI.out

SHOWN = [8, 23, 2, 45]
SUBJECTS = ["UTS02", "UTS03", "UTS06"]
N_EXC = 4

plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42, "text.hinting": "none"})
MONO = "DejaVu Sans Mono"
W, DPI = 5.5, 600
INK, GREY, RULE = "#222222", "#666666", "#9A9A9A"
BLUE, ORANGE, GOLD = "#0072B2", "#D55E00", "#F7C61C"
FS_TITLE, FS_LAB, FS_VAL, FS_EXC, FS_TICK, FS_LETTER = 7.0, 5.5, 6.5, 5.5, 5.0, 8.0
CMAP = panel.BRAIN_CMAP

ROW_H, E_H, GAP = 1.42, 1.36, 0.04
H = len(SHOWN) * ROW_H + GAP + E_H
X_L, X_SIDE, X_R = 0.17, 4.12, W - 0.02            # content left edge, side column start, right edge


def subject_maps(a):
    out = {}
    for s in SUBJECTS:
        f = glob.glob(os.path.join(CACHE, "flatmap_pngs", f"*_atom{a}_{s}_w1200.npy"))
        assert f, f"no cached flatmap for c{a} {s}: run figures/fig2/component_panel.py --component {a} --cache_dir {CACHE}"
        im = Image.fromarray((np.clip(np.load(f[0]), 0, 1) * 255).round().astype(np.uint8))
        out[s] = np.array(im.resize((700, round(im.height * 700 / im.width)), Image.LANCZOS))  # >600 dpi printed
    return out


def fsav_map(a, width_px=700):
    """The cached render is 5212 px wide for a map printed ~1.1 in wide; 700 px is >600 dpi at print size."""
    im = Image.open(os.path.join(CACHE, "fsaverage_atom_maps", f"atom{a}_fsav_panels_300dpi.png")).convert("RGBA")
    return np.array(im.resize((width_px, round(im.height * width_px / im.width)), Image.LANCZOS))


def vmaxes(a, sd):
    w = panel.get_per_subject_atom_weights(sd, None, a)
    subj = max(max(np.percentile(np.abs(x), 99) for x in w.values()), 0.01)   # as in the subject renders
    f = np.load(glob.glob(os.path.join(CACHE, "fsaverage_atom_maps", f"*_atom{a}_fsav.npy"))[0])
    d = f.copy(); d[d < np.percentile(f[f > 0], 5)] = np.nan
    return subj, float(np.nanpercentile(d, 99))                                 # as in the fsaverage render


def fmt_scale(v):
    return f"{v:.2f}" if v >= 0.1 else f"{v:.3f}"


def main():
    import torch
    sd = torch.load(os.path.join(panel.MODEL_DIR, "model_last.pt"), map_location="cpu", weights_only=False)["state_dict"]
    cat = pd.read_csv(panel.CATALOG)
    stab = pd.DataFrame({"atom": cat.component.str[1:].astype(int),
                         "brn_cos": cat.cortical_map_stability_20seeds,
                         "dec_cos": cat.semantic_decoder_stability_20seeds}).set_index("atom")
    xs = {d["component"]: d for d in json.load(open(XSUBJ_JSON))["first_model_null"]["per_component"]}
    null = json.load(open(XSUBJ_JSON))["first_model_null"]

    fig = plt.figure(figsize=(W, H), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    rend = fig.canvas.get_renderer()

    def T(x, y, s, fs, **kw):
        kw.setdefault("color", INK)
        return ax.text(x, y, s, fontsize=fs, **kw)

    def width(s, fs, **kw):
        t = ax.text(0, 0, s, fontsize=fs, **kw); w = t.get_window_extent(rend).width / DPI; t.remove(); return w

    def scale(x_right, yc, vmax, label, bw=0.22, bh=0.045):
        """Inline colour scale ending at x_right: '<label> 0 [bar] <vmax>'. Returns its left edge."""
        hi = fmt_scale(vmax)
        x = x_right - width(hi, FS_TICK)
        T(x, yc, hi, FS_TICK, va="center")
        x -= 0.03 + bw
        cax = fig.add_axes([x / W, (yc - bh / 2) / H, bw / W, bh / H])
        cax.imshow(np.linspace(0, 1, 256)[None, :], cmap=CMAP, aspect="auto", interpolation="none"); cax.axis("off")
        x -= 0.03 + width("0", FS_TICK)
        T(x, yc, "0", FS_TICK, va="center")
        x -= 0.04 + width(label, FS_TICK)
        T(x, yc, label, FS_TICK, va="center", color=GREY)
        return x

    letters = "abcd"
    for i, a in enumerate(SHOWN):
        top = H - i * ROW_H
        m = panel.get_metrics(a)
        T(0.02, top - 0.02, letters[i], FS_LETTER, fontweight="bold", va="top")
        # title line: gold tag + judge description
        tag = T(X_L, top - 0.085, f"c{a}", FS_TITLE, fontweight="bold", va="center",
                bbox=dict(boxstyle="round,pad=0.16,rounding_size=0.3", fc=GOLD, ec=INK, lw=0.4))
        T(X_L + width(f"c{a}", FS_TITLE, fontweight="bold") + 0.07, top - 0.085, m["description"], FS_TITLE, va="center")
        # maps: three subjects and fsaverage at one height, filling X_L .. X_SIDE - 0.06
        subj, fs_img = subject_maps(a), fsav_map(a)
        ar_s = subj[SUBJECTS[0]].shape[1] / subj[SUBJECTS[0]].shape[0]
        ar_f = fs_img.shape[1] / fs_img.shape[0]
        gap = 0.035
        avail = (X_SIDE - 0.06) - X_L - 3 * gap
        mh = avail / (3 * ar_s + ar_f)
        y_maps_top = top - 0.255
        y_maps = y_maps_top - mh
        x = X_L
        for s in SUBJECTS:
            iax = fig.add_axes([x / W, y_maps / H, mh * ar_s / W, mh / H]); iax.imshow(subj[s], interpolation="none"); iax.axis("off")
            T(x + mh * ar_s / 2, y_maps_top + 0.012, s, FS_LAB, ha="center", va="bottom", color=GREY)
            x += mh * ar_s + gap
        iax = fig.add_axes([x / W, y_maps / H, mh * ar_f / W, mh / H]); iax.imshow(fs_img, interpolation="none"); iax.axis("off")
        T(x + mh * ar_f / 2, y_maps_top + 0.012, "fsaverage (mean of 6 subjects)", FS_LAB, ha="center", va="bottom", color=GREY)
        # colour bars on the title line, right-aligned to the map area
        vs, vf = vmaxes(a, sd)
        xl = scale(X_R, top - 0.085, vf, "fsaverage")
        scale(xl - 0.12, top - 0.085, vs, "decoder weight: subjects")
        # excerpts: 4 lines, gold highlight on the firing words, centred under the subject maps
        ctx = panel.parse_contexts(a)[:N_EXC]
        y = y_maps - 0.085
        x_mid = X_L + (3 * mh * ar_s + 2 * gap) / 2
        for before, target, after in ctx:
            b, t, af = panel.trim_context(before, target, after, max_len=60)
            wb, wt, wa = width(b + " ", FS_EXC, family=MONO), width(t, FS_EXC, family=MONO, fontweight="bold"), width(" " + af, FS_EXC, family=MONO)
            pad = 0.02
            x0 = max(X_L, x_mid - (wb + wt + wa + 2 * pad) / 2)
            T(x0, y, b + " ", FS_EXC, family=MONO, color=GREY, va="baseline")
            em = FS_EXC / 72
            ax.add_patch(Rectangle((x0 + wb, y - 0.30 * em), wt + 2 * pad, 1.17 * em, fc=GOLD, ec=INK, lw=0.3, zorder=3))
            T(x0 + wb + pad, y, t, FS_EXC, family=MONO, fontweight="bold", va="baseline", zorder=4)
            T(x0 + wb + wt + 2 * pad, y, " " + af, FS_EXC, family=MONO, color=GREY, va="baseline")
            y -= 0.112
        # side column: label (grey) above value (black)
        dec = stab.loc[a, "dec_cos"]
        items = [
            [("Nameability AUC", m["auc"]), ("Cross-subject r", m["xsubj_r"])],
            [("Cross-seed stability", [(m["stability"], True), (" map   ", False), (f"{dec:.2f}", True), (" decoder", False)])],
            [("Active (% of TRs)", m["active"]), ("Yeo 7 network", m["networks"].replace(", ", " · "))],
            [("Neurosynth term (r)", f"{m['neurosynth_term']} ({m['neurosynth_r'].lstrip('+')}{'*' if m['neurosynth_q'].endswith('*') else ''})")],
        ]
        cw = (X_R - X_SIDE)
        y = top - 0.24
        for row in items:
            xs_ = [X_SIDE + cw / 2] if len(row) == 1 else [X_SIDE + cw / 4, X_SIDE + 3 * cw / 4]
            for xc, (lab, val) in zip(xs_, row):
                T(xc, y, lab, FS_LAB, ha="center", va="baseline", color=GREY)
                if isinstance(val, list):        # runs: (text, bold); bold values 6.5 pt, words 5.5 pt grey
                    ws = [width(t_, FS_VAL if b_ else FS_LAB, fontweight="bold" if b_ else "normal") for t_, b_ in val]
                    xx = xc - sum(ws) / 2
                    for (t_, b_), w_ in zip(val, ws):
                        T(xx, y - 0.105, t_, FS_VAL if b_ else FS_LAB, va="baseline",
                          fontweight="bold" if b_ else "normal", color=INK if b_ else GREY)
                        xx += w_
                else:
                    T(xc, y - 0.105, val, FS_VAL, ha="center", va="baseline", fontweight="bold")
            y -= 0.27

    # panel e: all 50 components on one similarity axis
    y0 = 0.0
    T(0.02, y0 + E_H - 0.02, "e", FS_LETTER, fontweight="bold", va="top")
    L, R, B, Tm = 0.42, 0.04, 0.22, 0.20
    eax = fig.add_axes([L / W, (y0 + B) / H, (W - L - R) / W, (E_H - B - Tm) / H])
    df = stab.reset_index().sort_values("brn_cos", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))
    b, s = df.brn_cos.values, df.dec_cos.values
    r = np.array([xs[int(c)]["r"] for c in df.atom]); sig = np.array([xs[int(c)]["sig_q05"] for c in df.atom])
    eax.axhspan(null["null_mean"] - null["null_std"], null["null_mean"] + null["null_std"], color="#E4E4E4", lw=0, zorder=0)
    eax.axhline(0.5, color=RULE, lw=0.5, ls=(0, (2.5, 2)), zorder=1)
    eax.vlines(x, np.minimum(b, s), np.maximum(b, s), color=RULE, lw=0.5, zorder=2)
    eax.plot(x, s, "o", ms=2.6, mfc="white", mec=BLUE, mew=0.6, zorder=3, clip_on=False)
    eax.plot(x, b, "o", ms=2.6, mfc=ORANGE, mec=ORANGE, mew=0, zorder=4, clip_on=False)
    eax.plot(x[sig], r[sig], "D", ms=2.2, mfc="#444444", mec="#444444", mew=0, zorder=5, clip_on=False)
    eax.plot(x[~sig], r[~sig], "D", ms=2.2, mfc="white", mec="#444444", mew=0.5, zorder=5, clip_on=False)
    eax.set_xlim(-0.8, len(df) - 0.2); eax.set_ylim(0, 1)
    eax.set_yticks([0, 0.5, 1]); eax.set_yticklabels(["0", "0.5", "1"])
    eax.set_ylabel("similarity", fontsize=6.0, color=INK, labelpad=2)
    eax.set_xticks(x); eax.set_xticklabels([f"c{c}" for c in df.atom], rotation=90, fontsize=FS_TICK)
    for lab, c in zip(eax.get_xticklabels(), df.atom):
        if c in SHOWN:
            lab.set_fontweight("bold"); lab.set_bbox(dict(boxstyle="round,pad=0.16,rounding_size=0.3", fc=GOLD, ec=INK, lw=0.4))
    eax.tick_params(axis="both", labelsize=FS_TICK, width=0.5, length=2, color=INK, labelcolor=INK, pad=1.5)
    eax.tick_params(axis="x", length=0, pad=2.5)
    for k in ("top", "right"):
        eax.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        eax.spines[k].set_linewidth(0.5); eax.spines[k].set_color(INK)
    from matplotlib.lines import Line2D
    hd = [Line2D([], [], ls="none", marker="o", ms=2.6, mfc=ORANGE, mec=ORANGE, mew=0),
          Line2D([], [], ls="none", marker="o", ms=2.6, mfc="white", mec=BLUE, mew=0.6),
          Line2D([], [], ls="none", marker="D", ms=2.2, mfc="#444444", mec="#444444", mew=0)]
    lg = eax.legend(hd, ["cross-seed stability, cortical map", "cross-seed stability, semantic decoder",
                         "cross-subject r (open: q ≥ 0.05; band: null)"],
                    loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=FS_LAB,
                    handletextpad=0.3, columnspacing=1.2, borderaxespad=0.15)
    for t_, c_ in zip(lg.get_texts(), [ORANGE, BLUE, "#444444"]):
        t_.set_color(c_)

    # checks: font sizes, overlaps, height
    fig.canvas.draw()                        # lay out legends and tick labels before measuring
    rend = fig.canvas.get_renderer()
    texts = [t for a_ in fig.axes for t in a_.texts]
    for a_ in fig.axes:
        if a_.axison:
            texts += [t for t in a_.get_xticklabels() + a_.get_yticklabels() if t.get_text() and t.get_visible()]
            if a_.yaxis.label.get_text():
                texts.append(a_.yaxis.label)
        if a_.get_legend():
            texts += list(a_.get_legend().get_texts())
    sizes = sorted({round(t.get_fontsize(), 2) for t in texts})
    print("font sizes used (pt):", sizes, "| smallest:", min(sizes))
    boxes = [(t, t.get_window_extent(rend)) for t in texts if t.get_visible() and t.get_text().strip()]
    over = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            bi, bj = boxes[i][1], boxes[j][1]
            if bi.overlaps(bj) and (min(bi.x1, bj.x1) - max(bi.x0, bj.x0)) > 1 and (min(bi.y1, bj.y1) - max(bi.y0, bj.y0)) > 1:
                over.append((boxes[i][0].get_text()[:25], boxes[j][0].get_text()[:25]))
    print("overlapping text pairs:", len(over)); [print("   ", o) for o in over[:20]]
    print(f"figure: {W} x {H:.2f} in (limit 7.4)")
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=DPI)
    plt.close(fig)
    print("wrote", OUT + ".pdf/.png")


if __name__ == "__main__":
    main()
