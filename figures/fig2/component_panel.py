#!/usr/bin/env python
"""One component panel of Figure 2.

Layout (vertical stack, full width):
  Header:  component label and description. Side column: AUC, cross-subject r, cross-seed stability,
           active fraction, Neurosynth term, Yeo 7 networks, all read from catalog/components.csv
  Middle:  3 subject flatmaps + 1 fsaverage average (4 panels)
  Bottom:  8 text excerpts with yellow-highlighted target words

Usage:
    uv run python figures/fig2/component_panel.py --model_dir <seed-0 run> --component 18
"""
import argparse
import csv
import os
import re
import sys
import tempfile
from os.path import join

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import h5py
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)
from sparsebrain import paths

PYCORTEX_DB = paths.pycortex_db
os.environ.setdefault("PYCORTEX_STORE", str(PYCORTEX_DB))
import cortex
# pycortex 1.4 reads its store from ~/.config/pycortex/options.cfg and ignores PYCORTEX_STORE, so
# point it at the dataset's store here (UTS0x subjects and fsaverage with the atlas_2mm transform).
if os.path.isdir(join(str(PYCORTEX_DB), "fsaverage")):
    cortex.db.filestore = str(PYCORTEX_DB)
    cortex.db.reload_subjects()
# The WebGL viewer reads its colormap folder from the same global config; use this environment's.
CMAP_DIR = join(sys.prefix, "share", "pycortex", "colormaps")
if os.path.isdir(CMAP_DIR):
    cortex.options.config.set("webgl", "colormaps", CMAP_DIR)

# ── Config ──────────────────────────────────────────────────────────────────

# Set from --model_dir in __main__ (the seed-0 coupled SAE in the paper).
MODEL_NAME = None
MODEL_DIR = None
FIGS_DIR = None
MAPPER_DIR = join(PROJECT_DIR, "outputs", "mappers")
FSAV_CACHE = join(PROJECT_DIR, "outputs", "cache", "fsaverage_atom_maps")
FLATMAP_CACHE = join(PROJECT_DIR, "outputs", "cache", "flatmap_pngs")
YEO7_DIR = None
CATALOG = join(PROJECT_DIR, "catalog", "components.csv")

NETWORK_ABBREV = {
    "Visual": "VIS", "Somatomotor": "SMN", "DorsalAttn": "DAN",
    "VentralAttn": "VAN", "Limbic": "LIM", "Frontoparietal": "FPN",
    "Default": "DMN",
}

N_VOXELS_FULL = {
    "UTS01": 81126, "UTS02": 94251, "UTS03": 95556,
    "UTS04": 109469, "UTS05": 99322, "UTS06": 92198,
}

SHOW_SUBJECTS = ["UTS02", "UTS03", "UTS06"]  # 3 representative subjects
ALL_SUBJECTS = list(N_VOXELS_FULL.keys())
BRAIN_CMAP = "inferno"
N_EXCERPTS = 6


# ── Brain maps ──────────────────────────────────────────────────────────────

def load_mapper(subject):
    path = join(MAPPER_DIR, f"{subject}_voxel_to_fsaverage.hdf")
    with h5py.File(path, "r") as f:
        return scipy.sparse.csr_matrix(
            (f["data"][:], f["indices"][:], f["indptr"][:]),
            shape=tuple(f["shape"][:]),
        )


def get_per_subject_atom_weights(sd, ckpt, atom_idx):
    """Per-voxel atom weight for each subject (delay-summed magnitude)."""
    n_atoms = sd["encoder.weight"].shape[0]
    n_delays = sd["delay_scales"].shape[0]
    delay_scales = sd["delay_scales"].numpy()
    weights = {}
    for subj in ALL_SUBJECTS:
        if f"brain_decoders.{subj}.weight" not in sd:
            continue
        W = sd[f"brain_decoders.{subj}.weight"].numpy()
        W_re = W.reshape(W.shape[0], n_delays, n_atoms)
        weighted = (W_re * delay_scales[None, :, :]).sum(axis=1)
        weights[subj] = np.abs(weighted[:, atom_idx])
    return weights


def compute_fsaverage_atom_map(atom_weights, voxel_indices, atom_idx):
    """Average atom brain map across all 6 subjects in fsaverage space.

    Caches to disk for reuse.
    """
    os.makedirs(FSAV_CACHE, exist_ok=True)
    cache_path = join(FSAV_CACHE, f"{MODEL_NAME}_atom{atom_idx}_fsav.npy")
    if os.path.exists(cache_path):
        print(f"  Loading cached fsaverage map for atom {atom_idx}")
        return np.load(cache_path)

    print(f"  Computing fsaverage map for atom {atom_idx}...")
    maps = []
    for subj in ALL_SUBJECTS:
        if subj not in atom_weights:
            continue
        D_full = np.zeros(N_VOXELS_FULL[subj], dtype=np.float32)
        D_full[voxel_indices[subj]] = atom_weights[subj]
        mapper = load_mapper(subj)
        D_fsav = np.clip(np.asarray(mapper.dot(D_full)).flatten(), 0, None)
        maps.append(D_fsav)

    avg = np.mean(maps, axis=0)
    np.save(cache_path, avg)
    return avg


def render_flatmap(weights, vox_idx, subject, vmax=None, target_width=1200,
                   atom_idx=None):
    """Render pycortex flatmap, crop whitespace, scale to target_width.

    Results cached to FLATMAP_CACHE as .npy arrays keyed by (atom, subject).
    """
    from PIL import Image as PILImage

    if atom_idx is not None:
        os.makedirs(FLATMAP_CACHE, exist_ok=True)
        cache_path = join(FLATMAP_CACHE,
                          f"{MODEL_NAME}_atom{atom_idx}_{subject}_w{target_width}.npy")
        if os.path.exists(cache_path):
            print(" (cached)", end="")
            return np.load(cache_path)
    else:
        cache_path = None

    n_full = N_VOXELS_FULL[subject]
    full_w = np.full(n_full, np.nan)
    full_w[vox_idx] = weights

    if vmax is None:
        vmax = max(np.nanpercentile(np.abs(weights), 95), 0.01)

    vol = cortex.Volume(full_w, subject=subject, xfmname=f"{subject}_auto",
                        vmin=0, vmax=vmax, cmap=BRAIN_CMAP)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        fig_c = cortex.quickflat.make_figure(
            vol, with_curvature=True, with_colorbar=False,
            with_rois=False, with_labels=False,
            curvature_brightness=0.2, curvature_contrast=0.2,
            thick=32, sampler='nearest', nanmean=True)
        fig_c.savefig(tmp, dpi=300, bbox_inches="tight",
                      facecolor="white", pad_inches=0.01)
        plt.close(fig_c)

        pil_img = PILImage.open(tmp).convert("RGB")
        # Crop whitespace
        bg = PILImage.new("RGB", pil_img.size, (255, 255, 255))
        diff = PILImage.fromarray(
            np.abs(np.array(pil_img).astype(int)
                   - np.array(bg).astype(int)).astype(np.uint8))
        bbox = diff.getbbox()
        if bbox:
            pil_img = pil_img.crop(bbox)
        # Scale preserving aspect ratio
        w, h = pil_img.size
        new_h = int(h * target_width / w)
        pil_img = pil_img.resize((target_width, new_h), PILImage.LANCZOS)
        arr = np.array(pil_img) / 255.0
        if cache_path is not None:
            np.save(cache_path, arr)
        return arr
    except Exception as e:
        print(f"  Flatmap failed for {subject}: {e}")
        return np.ones((200, target_width, 3))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def render_fsaverage_flatmap(fsav_map, vmax=None, target_width=1200):
    """Render fsaverage surface map using pycortex."""
    from PIL import Image as PILImage

    if vmax is None:
        vmax = max(np.nanpercentile(fsav_map[fsav_map > 0], 95), 0.01)

    # Mask near-zero vertices with NaN so curvature shows through
    # (otherwise inferno renders them as dark, hiding curvature)
    data = fsav_map.copy()
    threshold = np.percentile(data[data > 0], 5)  # bottom 5% of nonzero
    data[data < threshold] = np.nan

    vol = cortex.Vertex(data, "fsaverage", vmin=0, vmax=vmax,
                        cmap=BRAIN_CMAP)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        fig_c = cortex.quickflat.make_figure(
            vol, with_curvature=True, with_colorbar=False,
            with_rois=False, with_labels=False,
            curvature_brightness=0.2, curvature_contrast=0.2,
            thick=32, sampler='nearest', nanmean=True)
        fig_c.savefig(tmp, dpi=300, bbox_inches="tight",
                      facecolor="white", pad_inches=0.01)
        plt.close(fig_c)

        pil_img = PILImage.open(tmp).convert("RGB")
        bg = PILImage.new("RGB", pil_img.size, (255, 255, 255))
        diff = PILImage.fromarray(
            np.abs(np.array(pil_img).astype(int)
                   - np.array(bg).astype(int)).astype(np.uint8))
        bbox = diff.getbbox()
        if bbox:
            pil_img = pil_img.crop(bbox)
        w, h = pil_img.size
        new_h = int(h * target_width / w)
        pil_img = pil_img.resize((target_width, new_h), PILImage.LANCZOS)
        return np.array(pil_img) / 255.0
    except Exception as e:
        print(f"  fsaverage flatmap failed: {e}")
        return np.ones((200, target_width, 3))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ── Context parsing ─────────────────────────────────────────────────────────

def parse_contexts(atom_idx):
    """Parse (before, target, after) from atom_contexts/ files."""
    ctx_dir = join(FIGS_DIR, "atom_contexts")
    if not os.path.isdir(ctx_dir):
        return []

    for fn in sorted(os.listdir(ctx_dir)):
        if not fn.endswith(".txt"):
            continue
        with open(join(ctx_dir, fn)) as f:
            header = f.readline()
        m = re.match(r"=== Atom (\d+)", header)
        if m and int(m.group(1)) == atom_idx:
            break
    else:
        return []

    contexts = []
    lines = open(join(ctx_dir, fn)).readlines()
    i = 0
    while i < len(lines):
        if "[act=" in lines[i] and i + 1 < len(lines):
            text = lines[i + 1].strip()
            bm = re.search(r"\*\*(.+?)\*\*", text)
            if bm:
                contexts.append((text[:bm.start()].strip(),
                                 bm.group(1).strip(),
                                 text[bm.end():].strip()))
            i += 2
        else:
            i += 1
    return contexts


def trim_context(before, target, after, max_len=120):
    """Trim before/after/target to fit max_len."""
    max_target = 40
    if len(target) > max_target:                     # cut at a word boundary
        cut = target[:max_target - 1].rsplit(" ", 1)[0]
        target = (cut if len(cut) >= max_target // 2 else target[:max_target - 1]) + "…"
    avail = max_len - len(target)
    half = avail // 2
    if len(before) > half:                           # keep whole words only
        tail = before[-(half - 1):]
        tail = tail.split(" ", 1)[1] if " " in tail else tail
        before = "…" + tail
    if len(after) > half:
        head = after[:half - 1]
        head = head.rsplit(" ", 1)[0] if " " in head else head
        after = head + "…"
    return before, target, after


# ── Metrics ─────────────────────────────────────────────────────────────────

def get_catalog_row(atom_idx):
    """The component's row of catalog/components.csv, formatted as printed in the paper's catalog table."""
    for r in csv.DictReader(open(CATALOG)):
        if r["component"] != f"c{atom_idx}":
            continue
        star = lambda sig: "*" if sig else ""
        q_ns = float(r["neurosynth_q"])
        return dict(
            description=r["description"], auc=f"{float(r['autointerp_auc']):.2f}", coherence_z=r["coherence_z"],
            stability=f"{float(r['cortical_map_stability_20seeds']):.2f}",
            xsubj_r=f"{float(r['cross_subject_r']):.2f}" + star(r["cross_subject_sig_q05"] == "True"),
            active=f"{round(100 * float(r['active_fraction']))}%",
            networks=r["yeo7_networks"].replace(";", ", ") or "—",
            ns_term=r["neurosynth_term"], ns_r=f"{float(r['neurosynth_r']):+.3f}",
            ns_q=f"{q_ns:.3f}" + star(q_ns < 0.05))
    raise SystemExit(f"component c{atom_idx} not found in {CATALOG}")


def get_metrics(atom_idx):
    """Per-component values of the paper's catalog (catalog/components.csv)."""
    row = get_catalog_row(atom_idx)
    return dict(description=row["description"], auc=row["auc"], coherence_z=row["coherence_z"],
                stability=row["stability"], xsubj_r=row["xsubj_r"], active=row["active"],
                networks=row["networks"], neurosynth_term=row["ns_term"], neurosynth_r=row["ns_r"],
                neurosynth_q=row["ns_q"])


_YEO7_NETWORKS = ["Visual", "Somatomotor", "DorsalAttn", "VentralAttn",
                   "Limbic", "Frontoparietal", "Default"]


def get_yeo7_sig_networks(atom_idx):
    """Return list of significant Yeo7 networks (sorted by cosine) from pre-computed CSVs."""
    cos_path = join(YEO7_DIR, "atom_yeo7_cos.csv")
    pval_path = join(YEO7_DIR, "atom_yeo7_pvalues.csv")
    if not (os.path.exists(cos_path) and os.path.exists(pval_path)):
        return []

    cos_row = next((r for r in csv.DictReader(open(cos_path))
                    if int(r["atom"]) == atom_idx), None)
    pval_row = next((r for r in csv.DictReader(open(pval_path))
                     if int(r["atom"]) == atom_idx), None)
    if cos_row is None or pval_row is None:
        return []

    sig = [(net, float(cos_row[f"{net}_cos"]))
           for net in _YEO7_NETWORKS
           if int(pval_row[f"{net}_sig"]) == 1]
    sig.sort(key=lambda x: -x[1])
    return sig


# ── Main ────────────────────────────────────────────────────────────────────

def _trim_svg_bottom(svg_path, png_path):
    """Trim SVG viewBox to match the PNG's cropped content height."""
    import xml.etree.ElementTree as ET
    from PIL import Image as _PILImage

    # Measure content extent from the (already-cropped) PNG
    _img = _PILImage.open(png_path).convert("RGB")
    _arr = np.array(_img)
    _is_white = (_arr == 255).all(axis=(1, 2))
    _last_row = next((i for i in range(_arr.shape[0] - 1, -1, -1)
                       if not _is_white[i]), _arr.shape[0] - 1)
    _top_row = next((i for i in range(_arr.shape[0]) if not _is_white[i]), 0)
    content_frac = (_last_row + 1) / _arr.shape[0]

    # Adjust SVG viewBox height to match
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    tree = ET.parse(svg_path)
    root = tree.getroot()
    vb = root.get("viewBox").split()
    old_h = float(vb[3])
    new_h = old_h * content_frac
    root.set("viewBox", f"{vb[0]} {vb[1]} {vb[2]} {new_h:.2f}")
    root.set("height", f"{new_h:.2f}pt")
    tree.write(svg_path, xml_declaration=True, encoding="utf-8")


def render_fsaverage_panels(fsav_map, vmax, atom_idx):
    """Render fsaverage inflated panel views via xvfb-run subprocess.

    Saves to a cached PNG and returns as numpy image array.
    Must be run separately because plot_panels needs a display.
    """
    from PIL import Image as PILImage
    import subprocess

    cache_png = join(FSAV_CACHE, f"atom{atom_idx}_fsav_panels_300dpi.png")
    cache_svg = join(FSAV_CACHE, f"atom{atom_idx}_fsav_panels_300dpi.svg")
    if os.path.exists(cache_png):
        print("  Loading cached fsaverage panels")
        return np.array(PILImage.open(cache_png).convert("RGBA")) / 255.0

    # Write a temp script for xvfb-run
    script = f"""
import cortex
import os
if os.path.isdir('{join(str(PYCORTEX_DB), 'fsaverage')}'):
    cortex.db.filestore = '{PYCORTEX_DB}'
    cortex.db.reload_subjects()
if os.path.isdir('{CMAP_DIR}'):
    cortex.options.config.set('webgl', 'colormaps', '{CMAP_DIR}')
from cortex.export import plot_panels, params_flatmap_inflated_lateral_medial_ventral
import numpy as np
import matplotlib.pyplot as plt
import copy

fsav = np.load('{join(FSAV_CACHE, f"{MODEL_NAME}_atom{atom_idx}_fsav.npy")}')
threshold = np.percentile(fsav[fsav > 0], 5)
data = fsav.copy()
data[data < threshold] = np.nan
vmax = np.nanpercentile(data, 99)

vert = cortex.Vertex(data, 'fsaverage')
mapper = cortex.get_mapper('fsaverage', 'atlas_2mm')
vol_data = mapper.backwards(vert)
vd = vol_data.data.astype(float)
vd[~(vd >= threshold)] = np.nan  # back-projection turns NaN vertices into 0; keep them transparent
vol = cortex.Volume(vd, 'fsaverage', 'atlas_2mm',
                    vmin=0, vmax=vmax, cmap='inferno')

params = copy.deepcopy(params_flatmap_inflated_lateral_medial_ventral)
params['panels'][3]['extent'] = [-0.06, 0.19, 0.2, 0.2]
params['panels'][4]['extent'] = [0.86, 0.19, 0.2, 0.2]

fig = plot_panels(vol, **params,
    viewer_params={{'curvature_brightness': 0.2, 'curvature_contrast': 0.2,
                    'labels_visible': [], 'overlays_visible': []}},
    layers=32, headless=True)
fig.savefig('{cache_png}', dpi=300, bbox_inches='tight', transparent=True)
fig.savefig('{cache_svg}', bbox_inches='tight', transparent=True)
plt.close(fig)
print('done')
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                     delete=False, dir="/tmp") as f:
        f.write(script)
        script_path = f.name

    print("  Rendering fsaverage panels (xvfb-run)...", end="", flush=True)
    result = subprocess.run(
        ["xvfb-run", "-a", sys.executable, script_path],  # this environment; `uv run` would re-sync it
        capture_output=True, text=True, cwd=PROJECT_DIR, timeout=300)
    os.unlink(script_path)

    if result.returncode != 0:
        print(f" FAILED: {result.stderr[-200:]}")
        return np.ones((400, 800, 3))

    print(" done")
    return np.array(PILImage.open(cache_png).convert("RGBA")) / 255.0


OUT_PATH = None


def plot_atom_example(atom_idx):
    # Load model
    print("Loading model...")
    ckpt = torch.load(join(MODEL_DIR, "model.pt"), map_location="cpu",
                       weights_only=False)
    sd = torch.load(join(MODEL_DIR, "model_last.pt"), map_location="cpu",
                     weights_only=False)["state_dict"]
    voxel_indices = ckpt["voxel_indices"]

    # Per-subject weights
    atom_weights = get_per_subject_atom_weights(sd, ckpt, atom_idx)

    # Global vmax (99th percentile)
    global_vmax = max(np.percentile(np.abs(w), 99)
                      for w in atom_weights.values())
    global_vmax = max(global_vmax, 0.01)

    # Fsaverage map (cached)
    fsav_map = compute_fsaverage_atom_map(atom_weights, voxel_indices, atom_idx)

    # Contexts + metrics
    contexts = parse_contexts(atom_idx)
    metrics = get_metrics(atom_idx)
    description = metrics.get("description", f"component c{atom_idx}")

    # ── Render 3 subject flatmaps ──
    print("Rendering flatmaps...")
    brain_imgs = {}
    for subj in SHOW_SUBJECTS:
        print(f"  {subj}...", end="", flush=True)
        brain_imgs[subj] = render_flatmap(
            atom_weights[subj], voxel_indices[subj], subj,
            vmax=global_vmax, atom_idx=atom_idx)
        print(" done")

    # Fsaverage inflated panels (rendered via xvfb subprocess)
    fsav_panels_img = render_fsaverage_panels(fsav_map, global_vmax, atom_idx)

    # ── Figure ──
    fig_w = 14
    fig_h = 5.5
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",          # embed text as SVG <text>, not paths
    })

    # Column geometry — subjects 10% wider, right column proportionally tighter
    left_x = 0.03
    left_w = 0.68        # +10% vs 0.62
    right_x = 0.71       # shifted left to close the gap
    right_w = 0.27       # ~12% wider than before

    # Title
    fig.text(0.5, 0.985,
             f'Component c{atom_idx}: “{description}”',
             fontsize=14, fontstyle="italic", fontweight="bold",
             va="top", ha="center")

    # ── Left top: 3 subject flatmaps ──
    brain_top = 0.91   # closer to title → tighter title-label gap
    cell_w = left_w / 3
    cell_gap = 0.001             # reduced from 0.012 → flatmaps ~5% wider within each cell
    # Compute brain_h from image aspect ratio so axes fits image exactly (no letterboxing)
    _img = brain_imgs[SHOW_SUBJECTS[0]]
    _ih, _iw = _img.shape[:2]
    brain_h = (cell_w - cell_gap) * fig_w * _ih / (_iw * fig_h)
    brain_bot = brain_top - brain_h
    label_gap = -0.018  # labels shift down toward maps, total title-to-map gap unchanged
    for j, subj in enumerate(SHOW_SUBJECTS):
        ax = fig.add_axes([left_x + j * cell_w, brain_bot,
                           cell_w - cell_gap, brain_h])
        ax.imshow(brain_imgs[subj])
        ax.axis("off")

    # ── Right top: fsaverage panels ──
    _fh, _fw = fsav_panels_img.shape[:2]
    fsav_h = right_w * fig_w * _fh / (_fw * fig_h)
    fsav_bot = brain_top - fsav_h
    ax_fsav = fig.add_axes([right_x, fsav_bot, right_w, fsav_h])
    ax_fsav.imshow(fsav_panels_img)
    ax_fsav.axis("off")

    # ── Labels: all on the same horizontal line, centered above each panel ──
    label_y = brain_top + label_gap   # fixed y for horizontal alignment
    for j, subj in enumerate(SHOW_SUBJECTS):
        cx = left_x + (j + 0.5) * cell_w
        fig.text(cx, label_y, subj, fontsize=11, ha="center", va="bottom")
    fig.text(right_x + right_w / 2, label_y, "fsaverage",
             fontsize=11, fontweight="bold", ha="center", va="bottom")

    # ── Left bottom: text excerpts, full-width → exactly aligned with subjects ──
    text_gap_top = 0.043
    text_top = brain_bot - text_gap_top
    # Compute text_bot so the last text line lands near the figure bottom
    _n_show = min(N_EXCERPTS, len(contexts))
    _dy = 0.8 / (_n_show + 1.2)
    _y_last = 0.99 - (_n_show - 1) * _dy  # last line position in axes fraction
    _bottom_margin = 0.005
    text_bot = (_bottom_margin - _y_last * text_top) / (1 - _y_last)
    text_bot = max(0.0, text_bot)
    text_h = text_top - text_bot
    ax_txt = fig.add_axes([left_x, text_bot, left_w, text_h])
    ax_txt.axis("off")

    # ── Right bottom: metrics, full right-column width → centered on fsaverage ──
    # Values as printed in Table tab:catalog; '*' = q < 0.05 (BH-FDR), as in the table.
    nets = metrics["networks"].replace("SM,", "SMN,") if metrics["networks"] != "SM" else "SMN"
    # Rows of the side column; a row with two items is drawn as two half-width columns.
    metric_items = [
        [("AutoInterp AUC", metrics["auc"]), ("Cross-subject r", metrics["xsubj_r"])],
        [("Cross-seed stability", metrics["stability"]), ("Active", metrics["active"])],
        [("Neurosynth term (r, q)",
          f"{metrics['neurosynth_term']} ({metrics['neurosynth_r']}, q = {metrics['neurosynth_q']})")],
        [("Yeo 7 networks", nets.replace(", ", " · "))],
    ]

    # ── Colorbar below fsaverage panels (narrow swatch, ticks on top) ──
    import matplotlib.colors as _mcolors
    cbar_gap = 0.008
    cbar_h = 0.013
    cbar_top = fsav_bot - cbar_gap
    cbar_bot_y = cbar_top - cbar_h
    cbar_w = right_w * 0.09
    ax_cbar = fig.add_axes([right_x + (right_w - cbar_w) / 2, cbar_bot_y,
                             cbar_w, cbar_h])
    _sm = plt.cm.ScalarMappable(cmap=BRAIN_CMAP,
                                norm=_mcolors.Normalize(vmin=0, vmax=global_vmax))
    _sm.set_array([])
    cb = fig.colorbar(_sm, cax=ax_cbar, orientation="horizontal")
    cb.set_ticks([0, global_vmax])
    cb.set_ticklabels(["0", f"{global_vmax:.2f}"])
    cb.ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False,
                      labelsize=6, pad=0, length=2)

    # "brain decoder weights" immediately below bar
    cbar_label_gap = 0.004
    cbar_label_y = cbar_bot_y - cbar_label_gap
    fig.text(right_x + right_w / 2, cbar_label_y, "Brain decoder weights",
             fontsize=7, ha="center", va="top", color="#444444")

    # Metrics: top close below label; last item aligned with last context line
    met_gap_top = 0.018   # label text height + tight pad
    met_top = cbar_bot_y - met_gap_top
    met_bot = text_bot
    met_h = met_top - met_bot
    met_dy = 0.11
    met_val_offset = 0.048
    ax_met = fig.add_axes([right_x, met_bot, right_w, met_h])
    ax_met.axis("off")

    # y_m: last value center aligns with last context line
    n_metrics = len(metric_items)
    _last_ctx_fig_y = text_bot + _y_last * text_h
    _target_axes_y = (_last_ctx_fig_y - met_bot) / met_h
    met_dy = min(0.17, (0.93 - met_val_offset - _target_axes_y) / max(n_metrics - 1, 1))
    y_m = _target_axes_y + (n_metrics - 1) * met_dy + met_val_offset
    for row in metric_items:
        xs = [0.5] if len(row) == 1 else [0.27, 0.73]
        for xc, (label, value) in zip(xs, row):
            ax_met.text(xc, y_m, label, fontsize=8.5, color="#555555",
                        va="top", ha="center", transform=ax_met.transAxes)
            ax_met.text(xc, y_m - met_val_offset, value, fontsize=11, fontweight="bold",
                        va="top", ha="center", transform=ax_met.transAxes)
        y_m -= met_dy

    renderer = fig.canvas.get_renderer()
    n_show = min(N_EXCERPTS, len(contexts))
    dy = 0.8 / (n_show + 1.2)   # 80% spacing — tighter context lines
    y = 0.99  # start near the top of the text axes to close visual gap
    fontsize = 9.5
    max_chars = 90

    for i in range(n_show):
        before, target, after = contexts[i]
        before, target, after = trim_context(before, target, after,
                                             max_len=max_chars)

        # Pass 1: measure total line width invisibly to compute center offset
        inv = ax_txt.transAxes.inverted()
        t1 = ax_txt.text(0, y, before + " ", fontsize=fontsize,
                         fontfamily="monospace", color="none",
                         va="center", transform=ax_txt.transAxes)
        w1 = t1.get_window_extent(renderer).transformed(inv).width
        t1.remove()

        t2 = ax_txt.text(0, y, target, fontsize=fontsize + 0.5,
                         fontfamily="monospace", fontweight="bold",
                         color="none", va="center",
                         bbox=dict(boxstyle="round,pad=0.15",
                                   facecolor="none", edgecolor="none"),
                         transform=ax_txt.transAxes)
        w2 = t2.get_window_extent(renderer).transformed(inv).width
        t2.remove()

        t3 = ax_txt.text(0, y, " " + after, fontsize=fontsize,
                         fontfamily="monospace", color="none",
                         va="center", transform=ax_txt.transAxes)
        w3 = t3.get_window_extent(renderer).transformed(inv).width
        t3.remove()

        total_w = w1 + w2 + 0.012 + w3
        x = max(0.0, 0.5 - total_w / 2)

        # Pass 2: render visible, chaining positions
        t = ax_txt.text(x, y, before + " ", fontsize=fontsize,
                        fontfamily="monospace", color="#888888",
                        va="center", transform=ax_txt.transAxes)
        bb = t.get_window_extent(renderer=renderer)
        x = bb.transformed(inv).x1

        t = ax_txt.text(x, y, target, fontsize=fontsize + 0.5,
                        fontfamily="monospace", fontweight="bold",
                        color="#000000", va="center",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="#FFEB3B",
                                  edgecolor="#FFC107", alpha=0.7),
                        transform=ax_txt.transAxes)
        bb = t.get_window_extent(renderer=renderer)
        x = bb.transformed(inv).x1 + 0.012

        ax_txt.text(x, y, " " + after, fontsize=fontsize,
                    fontfamily="monospace", color="#888888",
                    va="center", transform=ax_txt.transAxes)

        y -= dy

    # Save full figure first, then crop bottom whitespace from both formats.
    out = OUT_PATH or join(os.path.dirname(os.path.abspath(__file__)), f"component_c{atom_idx}")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.2)
    # SVG: brain maps rasterized at 300 DPI, text/labels as vector
    for ax in fig.get_axes():
        for im in ax.get_images():
            im.set_rasterized(True)
    fig.savefig(out + ".svg", bbox_inches="tight", pad_inches=0.2,
                dpi=300)
    plt.close(fig)

    # Crop bottom whitespace from PNG (same as original working version)
    from PIL import Image as _PILImage
    _img = _PILImage.open(out + ".png").convert("RGB")
    _arr = np.array(_img)
    _is_white = (_arr == 255).all(axis=(1, 2))
    _top_margin = next((i for i in range(_arr.shape[0]) if not _is_white[i]), 0)
    _last_row = next((i for i in range(_arr.shape[0] - 1, -1, -1) if not _is_white[i]), 0)
    _crop_h = _last_row + _top_margin
    _img.crop((0, 0, _img.width, _crop_h)).save(out + ".png", dpi=(300, 300))

    # Apply the same proportional crop to SVG viewBox
    crop_frac = _crop_h / _arr.shape[0]
    import xml.etree.ElementTree as ET
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    tree = ET.parse(out + ".svg")
    root = tree.getroot()
    vb = root.get("viewBox").split()
    old_h = float(vb[3])
    new_h = old_h * crop_frac
    root.set("viewBox", f"{vb[0]} {vb[1]} {vb[2]} {new_h:.2f}")
    root.set("height", f"{new_h:.2f}pt")
    tree.write(out + ".svg", xml_declaration=True, encoding="utf-8")

    print(f"\nSaved to {out}.png (cropped to {crop_frac:.1%} of height)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--component", "--atom", dest="atom", type=int, default=18, help="component index j (cj)")
    p.add_argument("--model_dir", required=True, help="run folder of the seed-0 coupled SAE")
    p.add_argument("--catalog", default=CATALOG)
    p.add_argument("--out", default=None, help="output path without extension (.png and .svg are written)")
    p.add_argument("--cache_dir", default=None,
                   help="render cache folder (flatmap_pngs/ and fsaverage_atom_maps/ inside); default: "
                        "outputs/cache/")
    args = p.parse_args()
    if args.cache_dir:
        FSAV_CACHE = join(os.path.abspath(args.cache_dir), "fsaverage_atom_maps")
        FLATMAP_CACHE = join(os.path.abspath(args.cache_dir), "flatmap_pngs")
    MODEL_DIR = os.path.abspath(args.model_dir)
    MODEL_NAME = os.path.basename(MODEL_DIR.rstrip("/"))
    FIGS_DIR = join(PROJECT_DIR, "reports", "figures", MODEL_NAME)
    YEO7_DIR = join(PROJECT_DIR, "reports", "figures", f"yeo7_{MODEL_NAME}")
    CATALOG = args.catalog
    OUT_PATH = os.path.abspath(args.out) if args.out else None
    plot_atom_example(args.atom)
