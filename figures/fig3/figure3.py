#!/usr/bin/env python
"""Figure 3: transfer to new subjects (UTS07, UTS08), frozen coupled SAE vs matched ridge, n = 0.5–22 stories.

Reads all per-seed CSVs, collapses story seeds that drew the same training set
into one observation, averages the two subjects within each training-story draw
(both subjects use the same seed, hence the same stories), and plots mean ± 1 SEM
over draws
with secondary time axis. At n = 22 the story pool is exhausted, so every seed
trains on the same stories: only the mean is drawn there (no band, no test).

Reads the results.csv files written by scripts/34_learning_curve.py under reports/figures/learning_curve*/.

Usage:
    uv run python figures/fig3/figure3.py --out figures/fig3/figure3.png

Set SPARSEBRAIN_DIR to read the results from a different checkout.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR = os.environ.get(
    "SPARSEBRAIN_DIR",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ── Constants ────────────────────────────────────────────────────────────────

ZERO_SHOT_R     = 0.057   # mean of UTS07=0.055, UTS08=0.060 (fsaverage baseline)

# Story draws collide across seeds. 34_learning_curve.py seeds
# np.random.default_rng(story_seed) and draws n stories from a pool of ~21-22,
# so at small n different seeds often select the SAME training set ({0,2} at
# n=0.5 and n=1; {0,5} and {1,6} at n=2), and at n=22 the pool is exhausted so
# every seed trains on the same stories. Such runs are not independent
# replicates: they differ only by SAE initialisation. We therefore deduplicate
# on the training set itself. Runs that record a `draw_key` column use it;
# older CSVs fall back to fingerprinting the training set by the ridge score,
# which is deterministic given its data, so equal ridge r within a
# (subject, n) cell means the same stories. The same keep-set is applied to
# every condition. (Replaces the old DUPLICATE_SEEDS={2}, which caught only the
# seed-0/seed-2 collision and wrongly dropped seed 2 at n>=2.)
DEDUP_FINGERPRINT_DECIMALS = 9

MEDIAN_TRS_PER_STORY = 355   # median across 27 UTS07 stories, TR=2 s → 11.8 min
TR_S = 2

# Split-complementary, colorblind-friendly palette
COLOR_FRT   = "#F4845F"   # salmon coral  — coupled SAE (primary, warm/prominent)
COLOR_RIDGE = "#2A9D8F"   # teal green    — ridge (cool contrast)
COLOR_RAND  = "#457B9D"   # steel blue    — random encoder (neutral third)

# Top-axis tick positions — all n values fit at full column width
TOP_AXIS_N = [0.5, 1, 2, 4, 8, 16, 22]


# ── Data ─────────────────────────────────────────────────────────────────────

def load_all_results():
    lc = os.path.join(PROJECT_DIR, "reports", "figures")
    paths = []
    main = os.path.join(lc, "learning_curve", "results.csv")
    if os.path.exists(main):
        paths.append(main)
    for p in sorted(glob.glob(os.path.join(lc, "learning_curve_s[0-9]*", "results.csv"))):
        paths.append(p)
    for p in sorted(glob.glob(os.path.join(lc, "learning_curve_half_s[0-9]*", "results.csv"))):
        paths.append(p)
    for p in sorted(glob.glob(os.path.join(lc, "learning_curve_rand_s[0-9]*", "results.csv"))):
        paths.append(p)
    if not paths:
        raise SystemExit(f"no results.csv found under {lc}")

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["n_stories"] = df["n_stories"].astype(float)

    # De-duplicate: rand runs also emit ridge/freeze_retrain rows — keep one copy per
    # (subject, n_stories, story_seed, sae_seed, condition).
    df = df.drop_duplicates(
        subset=["subject", "n_stories", "story_seed", "sae_seed", "condition"]
    ).reset_index(drop=True)
    return df


def assign_draw_ids(df):
    """Label every (subject, n_stories, story_seed) with the training set it drew.

    Uses the exact `draw_key` when every row carries one; otherwise fingerprints
    the training set by the deterministic ridge score. The resulting draw_id is
    attached to all conditions of that seed-run, so the keep-set is shared.
    """
    key = ["subject", "n_stories", "story_seed"]
    if "draw_key" in df.columns and df["draw_key"].notna().all():
        ids = df.groupby(key, dropna=False)["draw_key"].first()
    else:
        ridge = df[df.condition == "ridge"].groupby(key)["brain_r"].mean()
        ids = ridge.round(DEDUP_FINGERPRINT_DECIMALS)
    ids = ids.rename("draw_id").reset_index()
    out = df.merge(ids, on=key, how="left")
    missing = out["draw_id"].isna()
    if missing.any():
        raise SystemExit("seed-runs without a ridge fingerprint:\n"
                         + str(out.loc[missing, key].drop_duplicates()))
    return out


def dedup_training_sets(df):
    """One observation per distinct training set.

    Averages over sae_seeds, then collapses story_seeds that drew the same
    training set into a single observation. Returns one row per
    (subject, n_stories, draw_id, condition).
    """
    df = assign_draw_ids(df)
    per_seed = (df.groupby(["subject", "n_stories", "draw_id", "story_seed", "condition"])
                  ["brain_r"].mean().reset_index())
    unit = (per_seed.groupby(["subject", "n_stories", "draw_id", "condition"])
                    .agg(brain_r=("brain_r", "mean"), draw_seed=("story_seed", "min"))
                    .reset_index())
    return unit


def pool_subjects(unit):
    """One observation per training-story draw: average the subjects within it.

    A given story_seed draws the same stories for every subject, so the subjects
    of one draw are not independent samples of the training set. Draws are
    matched across subjects by `draw_seed` (smallest story_seed of the draw);
    the seed-to-draw partition must agree across subjects.
    """
    parts = (unit[unit.condition == "ridge"]
             .groupby(["n_stories", "subject"])["draw_seed"]
             .apply(lambda s: tuple(sorted(s))).unstack("subject"))
    bad = parts[parts.nunique(axis=1) > 1]
    if not bad.empty:
        raise SystemExit("training-set partitions differ across subjects:\n" + str(bad))
    return (unit.groupby(["n_stories", "draw_seed", "condition"])["brain_r"]
                .mean().reset_index())


def coverage_report(df):
    """Nominal seed-runs vs distinct training sets, per (subject, n)."""
    df = assign_draw_ids(df)
    rows = []
    for (subj, n), g in df.groupby(["subject", "n_stories"]):
        rows.append((subj, n, g.story_seed.nunique(), g.draw_id.nunique()))
    return pd.DataFrame(rows, columns=["subject", "n_stories", "seeds", "distinct"])


def compute_stats(unit):
    """Mean ± SEM per (n_stories, condition) over the rows of `unit`.

    Pass pool_subjects(unit) for the pooled curve (unit = training-story draw)
    or one subject's rows for a per-subject curve (unit = that subject's draw).
    """
    agg = (unit.groupby(["n_stories", "condition"])["brain_r"]
               .agg(mean="mean", std="std", count="count").reset_index())
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])
    return agg


def is_single_draw(unit, n):
    """True when every subject has exactly one training set at this n."""
    sub = unit[(unit.n_stories == n) & (unit.condition == "ridge")]
    return (sub.groupby("subject")["draw_id"].nunique() == 1).all()


def paired_p(unit, n, direction="frt_gt_ridge"):
    """One-sided paired t-test, coupled SAE > ridge, unit = training-story draw.

    Subjects are averaged within each draw first (pool_subjects), the same unit
    as the error bands; subject x draw pairs are not independent.
    """
    pooled = pool_subjects(unit)
    sub = pooled[pooled.n_stories == n].set_index("draw_seed")
    ridge = sub[sub.condition == "ridge"]["brain_r"]
    frt   = sub[sub.condition == "freeze_retrain"]["brain_r"]
    frt, ridge = frt.align(ridge, join="inner")
    if len(frt) < 3:
        return np.nan
    t, p_two = stats.ttest_rel(frt, ridge)
    return p_two / 2 if t > 0 else 1.0


def sig_star(p):
    """Standard neuroscience significance glyphs."""
    if not np.isfinite(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.1:   return "+"   # not a dagger: the tables use † for other marks
    return ""


# ── Plot ─────────────────────────────────────────────────────────────────────

def make_plot(unit, out_path):
    stats_df = compute_stats(pool_subjects(unit))
    n_vals = sorted(stats_df["n_stories"].unique())
    single = {n for n in n_vals if is_single_draw(unit, n)}

    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.size":        9,
        "axes.linewidth":   0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size":  3,
        "ytick.major.size":  3,
    })

    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    has_rand = "random_encoder" in stats_df["condition"].values

    # Collect per-condition series for fill-between
    lines = {}
    cond_styles = [
        ("ridge",          COLOR_RIDGE, "--", "o", "Dense feature ridge regression"),
        ("freeze_retrain", COLOR_FRT,   "-",  "s", "Coupled SAE, retrain decoder"),
    ]
    if has_rand:
        cond_styles.append(
            ("random_encoder", COLOR_RAND, ":", "^", "Random encoder, retrain decoder")
        )
    for cond, color, ls, marker, label in cond_styles:
        sub = stats_df[stats_df.condition == cond].sort_values("n_stories")
        x   = sub["n_stories"].values
        y   = sub["mean"].values
        sem = sub["sem"].values
        lines[cond] = (x, y, sem)

        # Band only where there is more than one training set per subject.
        band = np.array([n not in single for n in x])
        ax.fill_between(x[band], (y - sem)[band], (y + sem)[band],
                        color=color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(x, y,
                color=color, linestyle=ls, marker=marker,
                markersize=5.5, linewidth=2.0, markeredgewidth=0,
                label=label, zorder=3)

    # Subtle fill between means: highlight which model leads
    x_r, y_r, _ = lines["ridge"]
    x_f, y_f, _ = lines["freeze_retrain"]
    ax.fill_between(x_f, y_r, y_f,
                    where=(y_f >= y_r), color=COLOR_FRT,   alpha=0.08,
                    linewidth=0, zorder=1)
    ax.fill_between(x_f, y_r, y_f,
                    where=(y_f <  y_r), color=COLOR_RIDGE, alpha=0.08,
                    linewidth=0, zorder=1)

    # ── Primary x-axis ──
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_vals)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Number of training stories", fontsize=10)

    # ── Secondary x-axis: wall time (skip n=16 to avoid crowding) ──
    ax2 = ax.twiny()
    ax2.set_xscale("log", base=2)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(TOP_AXIS_N)

    def fmt_time(n):
        mins = n * MEDIAN_TRS_PER_STORY * TR_S / 60
        return f"{mins:.0f} min" if mins < 60 else f"{mins/60:.1f} hr"

    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: fmt_time(v)))
    ax2.tick_params(axis="x", labelsize=8, rotation=0)
    ax2.set_xlabel("Estimated scan time", fontsize=9, labelpad=6)

    # ── Y-axis: from zero ──
    ax.set_ylabel("Brain prediction (Pearson r)", fontsize=10)
    ymin = 0
    ymax = 0.12
    ax.set_ylim(ymin, ymax)

    # Light horizontal grid at y-axis ticks only
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    ax.grid(axis="y", color="0.88", linewidth=0.6, zorder=0)

    # Significance markers above the freeze-retrain line (not at single-draw n)
    frt_stats = stats_df[stats_df.condition == "freeze_retrain"].set_index("n_stories")
    for n in n_vals:
        if n in single:
            continue
        p = paired_p(unit, n)
        star = sig_star(p)
        if star:
            y_top = frt_stats.loc[n, "mean"] + frt_stats.loc[n, "sem"]
            ax.text(n, y_top + 0.002, star, ha="center", va="bottom",
                    fontsize=9.5, color="0.25", zorder=4)

    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left",
              handlelength=2.4, handleheight=1.0)

    note = "* p < 0.05   + p < 0.1"
    if single:
        n_single = ", ".join(f"{n:g}" for n in sorted(single))
        note = (f"Bands: ±1 SEM over training-story draws (subjects averaged). "
                f"n = {n_single}: one training set per subject, mean only.\n" + note)
    ax.text(0.99, 0.02, note,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="0.4", linespacing=1.4)

    fig.tight_layout(pad=0.8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(PROJECT_DIR, "figures", "fig3", "figure3.png"))
    args = p.parse_args()

    df = load_all_results()
    print(f"Loaded {len(df)} rows: {df.story_seed.nunique()} seeds, "
          f"n_stories={sorted(df.n_stories.unique())}")

    cov = coverage_report(df)
    print("\nSeed coverage: nominal seed-runs vs DISTINCT training sets")
    print(f"{'subject':>9}{'n':>7}{'seeds':>7}{'distinct':>10}")
    for _, r in cov.sort_values(["subject", "n_stories"]).iterrows():
        flag = "  <- collisions" if r.distinct < r.seeds else ""
        print(f"{r.subject:>9}{r.n_stories:>7g}{r.seeds:>7}{r.distinct:>10}{flag}")
    print(f"  TOTAL {cov.seeds.sum()} seed-runs -> {cov.distinct.sum()} distinct")
    per_n = cov.groupby("n_stories")["distinct"].sum()
    print("  distinct per n (both subjects): "
          + ", ".join(f"n={n:g}: {d}" for n, d in per_n.items()))

    unit = dedup_training_sets(df)
    stats_df = compute_stats(pool_subjects(unit))

    print("\nMean ± SEM per (n, condition), unit = training-story draw (subjects averaged):")
    print(f"{'n':>6}{'condition':>16}{'obs':>5}{'mean':>9}{'sem':>9}")
    for _, r in stats_df.sort_values(["n_stories", "condition"]).iterrows():
        print(f"{r.n_stories:>6g}{r.condition:>16}{r['count']:>5}"
              f"{r['mean']:>9.4f}{r['sem']:>9.4f}")

    print("\nOne-sided paired t-test, coupled SAE > ridge (unit = training-story draw, subjects averaged):")
    print(f"{'n':>6}{'pairs':>7}{'p(1s)':>9}")
    for n in sorted(unit.n_stories.unique()):
        sub = unit[(unit.n_stories == n) & (unit.condition == "ridge")]
        if is_single_draw(unit, n):
            print(f"{n:>6g}{sub.draw_seed.nunique():>7}{'n/a (one training set per subject)':>9}")
        else:
            print(f"{n:>6g}{sub.draw_seed.nunique():>7}{paired_p(unit, n):>9.4f}")

    n_max = unit.n_stories.max()
    print(f"\nEndpoint n={n_max:g}, per subject:")
    for subj, g in unit[unit.n_stories == n_max].groupby("subject"):
        r = g.groupby("condition")["brain_r"].mean()
        print(f"  {subj}  ridge={r['ridge']:.4f}  frozen={r['freeze_retrain']:.4f}  "
              f"{100*r['freeze_retrain']/r['ridge']:.1f}% of ridge")

    make_plot(unit, args.out)


if __name__ == "__main__":
    main()
