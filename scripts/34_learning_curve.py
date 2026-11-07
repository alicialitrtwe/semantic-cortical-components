#!/usr/bin/env python
"""34_learning_curve.py — Learning curve: freeze-retrain vs matched ridge.

For UTS07/UTS08, varies the number of training stories [0.5, 1, 2, 4, 8, 16, 22] and
compares brain_r on the test story for:
  - Freeze-retrain: frozen SAE encoder + new linear brain decoder trained with AdamW
  - Ridge: himalaya KernelRidgeCV on raw Gemma-2 features, per-voxel alpha

Validation scheme: same 4 fixed val stories (val_seed=0) used for
freeze-retrain early stopping and ridge alpha selection across all counts/seeds.
Training stories are randomly sampled from the 22-story pool at each count/seed.
Both conditions evaluate on the same 30k selected voxels (from existing
transfer checkpoint).

Usage:
    uv run python scripts/34_learning_curve.py \\
        --base_model_dirs outputs/models/<sae_s0> outputs/models/<sae_s1> outputs/models/<sae_s2> \\
        --transfer_ref_dir outputs/models/<freeze_retrain_s0>
"""
import argparse
import math
import os
import sys
from os.path import join
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import zscore as sp_zscore
from himalaya.backend import set_backend
from himalaya.kernel_ridge import KernelRidgeCV, KernelRidge

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from sparsebrain.data import load_features, load_responses
from sparsebrain.models.coupled_sae import MultiSubjectCoupledSAE
from sparsebrain.training.utils import apply_delays_np
from sparsebrain import STORIES_TRAIN

TEST_STORY = "wheretheressmoke"
DELAYS = [1, 2, 3, 4]
ALPHAS = np.logspace(1, 20, 20)
STORY_COUNTS = [0.5, 1, 2, 4, 8, 16, 22]
N_STORY_SEEDS = 3
VAL_N_STORIES = 4
VAL_SEED = 0
N_TARGETS_BATCH = 40000

# Freeze-retrain hypers — match 17_fit_multisubject_sae.py defaults
FRT_EPOCHS = 600
FRT_PATIENCE = 10
FRT_LR = 1e-3
FRT_BATCH = 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_draws(n_stories, pool_len, n_draws, seed_start, mode):
    """Return [(draw_id, tuple_of_pool_indices), ...] — the training sets to run.

    mode="seed" (default) reproduces the original behaviour exactly: one draw per
    story_seed via np.random.default_rng(story_seed).

    mode="enumerate" fixes a real problem with that. The generator is seeded only
    by story_seed, and at small n the space of possible training sets is tiny
    (for n<=1 it is just `pool_len` single-story choices, since the fractional
    case picks one story and truncates it). Different seeds therefore collide by
    birthday: measured 203 nominal seed-runs -> 160 distinct training sets. In
    enumerate mode we instead:
      * enumerate ALL draws when the space is no larger than the number asked for
        (exhaustive, zero redundancy, and the story-sampling spread becomes exact
        rather than estimated); otherwise
      * sample DISTINCT sets by rejection, so k draws give k distinct sets.
    """
    from itertools import combinations
    n_int = 1 if n_stories < 1 else int(n_stories)
    n_int = min(n_int, pool_len)

    if mode == "seed":
        out = []
        for sd in range(seed_start, seed_start + n_draws):
            rng = np.random.default_rng(sd)
            idx = tuple(sorted(rng.choice(pool_len, size=n_int, replace=False).tolist()))
            out.append((sd, idx))
        return out

    # --- enumerate mode ---
    total = math.comb(pool_len, n_int)
    if total <= n_draws:
        print(f"  [draws] exhaustive: all {total} possible training sets")
        return list(enumerate(tuple(c) for c in combinations(range(pool_len), n_int)))
    print(f"  [draws] {total} possible sets; sampling {n_draws} distinct. "
          f"Pass --n_story_seeds {total} for exhaustive coverage.")
    rng = np.random.default_rng(seed_start)
    seen, out = set(), []
    while len(out) < n_draws:
        idx = tuple(sorted(rng.choice(pool_len, size=n_int, replace=False).tolist()))
        if idx in seen:
            continue
        seen.add(idx)
        out.append((len(out), idx))
    return out


def get_val_and_pool(all_stories):
    """Val/pool split using fixed val_seed=0 — same logic as training script."""
    rng = np.random.default_rng(VAL_SEED)
    val_idx = sorted(rng.choice(len(all_stories), size=VAL_N_STORIES, replace=False).tolist())
    val = [all_stories[i] for i in val_idx]
    pool = [s for i, s in enumerate(all_stories) if i not in set(val_idx)]
    return val, pool


def brain_r(Y_pred, Y_true):
    """Mean per-voxel Pearson r on test story."""
    p = np.nan_to_num(sp_zscore(Y_pred, axis=0))
    a = np.nan_to_num(sp_zscore(Y_true, axis=0))
    return float(np.nan_to_num((p * a).mean(axis=0)).mean())


def apply_delays(X):
    """Delay [1,2,3,4] within a single story; wrapper around canonical apply_delays_np."""
    return apply_delays_np(X, DELAYS).astype(np.float32)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_encoder(model_dir, device):
    """Load frozen encoder from model_last.pt. Returns model in eval mode."""
    lp = join(model_dir, "model_last.pt")
    if not os.path.exists(lp):
        raise SystemExit(f"{lp} not found: the transfer uses the final-epoch encoder")
    ckpt = torch.load(lp, map_location="cpu", weights_only=False)
    main = torch.load(join(model_dir, "model.pt"), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    n_atoms = sd["encoder.weight"].shape[0]
    n_sem = sd["encoder.weight"].shape[1]
    args = main["args"]
    k = args.get("k", 5) if isinstance(args, dict) else getattr(args, "k", 5)
    n_delays = sd["delay_scales"].shape[0]

    model = MultiSubjectCoupledSAE(
        n_sem=n_sem,
        n_voxels_per_subject=main["n_voxels_per_subject"],
        n_atoms=n_atoms,
        k=k,
        n_delays=n_delays,
        tie_semantic_decoder="decoder_sem.weight" not in sd,
        encoder_bias="encoder_bias" in sd,
        brain_voxel_scale=False,
    )
    model.load_state_dict_compat(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    return model, n_atoms, n_delays


# ---------------------------------------------------------------------------
# Sparse code precomputation
# ---------------------------------------------------------------------------

def compute_codes_story(model, X_rms, device):
    """Return Z_delayed (n_TRs, n_atoms * n_delays) for one story.

    X_rms: (n_TRs, n_sem) RMS-normalised features.
    """
    X_del = apply_delays(X_rms)  # (n_TRs, n_delays * n_sem)
    with torch.no_grad():
        out = model(
            torch.tensor(X_rms, dtype=torch.float32, device=device),
            torch.tensor(X_del, dtype=torch.float32, device=device),
            subject=None,
        )
    return out["s_delayed"].cpu().numpy()  # (n_TRs, n_atoms * n_delays)


# ---------------------------------------------------------------------------
# Freeze-retrain linear decoder
# ---------------------------------------------------------------------------

def train_decoder(S_train, Y_train, S_val, Y_val, S_test, Y_test, device):
    """Train linear brain decoder W: (n_atoms*n_delays -> n_voxels) with AdamW.

    Early stops on val brain_r with patience FRT_PATIENCE.
    Returns test brain_r at best val epoch.
    """
    n_in = S_train.shape[1]
    n_vox = Y_train.shape[1]

    S_tr = torch.tensor(S_train, dtype=torch.float32, device=device)
    Y_tr = torch.tensor(Y_train, dtype=torch.float32, device=device)
    S_vl = torch.tensor(S_val, dtype=torch.float32, device=device)
    Y_vl = torch.tensor(Y_val, dtype=torch.float32, device=device)

    decoder = nn.Linear(n_in, n_vox, bias=False).to(device)
    nn.init.xavier_uniform_(decoder.weight)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=FRT_LR, weight_decay=0.0)

    n_train = S_tr.shape[0]
    batch_size = min(FRT_BATCH, n_train)
    n_batches = max(1, n_train // batch_size)

    best_val_r = -np.inf
    best_W = decoder.weight.data.clone()
    patience_count = 0

    for epoch in range(FRT_EPOCHS):
        decoder.train()
        perm = torch.randperm(n_train, device=device)
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            loss = nn.functional.mse_loss(decoder(S_tr[idx]), Y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        decoder.eval()
        with torch.no_grad():
            Y_val_pred = decoder(S_vl).cpu().numpy()
        val_r = brain_r(Y_val_pred, Y_val)

        if val_r > best_val_r:
            best_val_r = val_r
            best_W = decoder.weight.data.clone()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= FRT_PATIENCE:
                break

    # Evaluate at best val epoch
    decoder.weight.data.copy_(best_W)
    decoder.eval()
    with torch.no_grad():
        Y_test_pred = decoder(
            torch.tensor(S_test, dtype=torch.float32, device=device)
        ).cpu().numpy()
    return brain_r(Y_test_pred, Y_test)


# ---------------------------------------------------------------------------
# Ridge with himalaya — single train/val fold, per-voxel alpha
# ---------------------------------------------------------------------------

def fit_ridge(X_train, Y_train, X_val, Y_val, X_test, Y_test, backend):
    """Per-voxel alpha via single val fold; final fit on train only.

    Step 1: KernelRidgeCV on (train+val) with a single (train→val) fold to
    select per-voxel best_alphas_.  Val is used only for alpha selection.
    Step 2: Refit KernelRidge with those fixed alphas on X_train alone, so
    ridge and freeze-retrain both use exactly n training stories.
    """
    X_all = np.vstack([X_train, X_val]).astype(np.float32)
    Y_all = np.vstack([Y_train, Y_val]).astype(np.float32)
    n_tr = X_train.shape[0]
    n_vl = X_val.shape[0]
    cv = [(np.arange(n_tr), np.arange(n_tr, n_tr + n_vl))]

    cv_model = KernelRidgeCV(
        alphas=ALPHAS,
        cv=cv,
        solver_params=dict(n_targets_batch=N_TARGETS_BATCH, diagonalize_method="svd"),
    )
    cv_model.fit(backend.asarray(X_all), backend.asarray(Y_all))
    best_alphas = backend.to_numpy(cv_model.best_alphas_)  # shape (n_voxels,)

    # Refit on train only — one KernelRidge per unique alpha
    Y_pred = np.empty((X_test.shape[0], Y_train.shape[1]), dtype=np.float32)
    for alpha_val in np.unique(best_alphas):
        vox_mask = best_alphas == alpha_val
        m = KernelRidge(alpha=float(alpha_val), kernel="precomputed")
        K_tr = backend.asarray(X_train.astype(np.float32)) @ backend.asarray(X_train.astype(np.float32)).T
        m.fit(K_tr, backend.asarray(Y_train[:, vox_mask].astype(np.float32)))
        K_te = backend.asarray(X_test.astype(np.float32)) @ backend.asarray(X_train.astype(np.float32)).T
        Y_pred[:, vox_mask] = backend.to_numpy(m.predict(K_te))

    return brain_r(Y_pred, Y_test)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_all_responses(subject, stories, vox_idx):
    """Load z-scored responses on selected voxels for each story.
    Returns dict: story -> np.array (n_TRs, n_voxels).
    """
    out = {}
    for story in stories:
        Y = np.nan_to_num(load_responses([story], subject))  # (n_TRs, n_vox_full)
        out[story] = Y[:, vox_idx].astype(np.float32)
    return out


def load_all_features_rms(stories, rms):
    """Load and RMS-normalise features for each story.
    Returns dict: story -> np.array (n_TRs, n_sem).
    """
    return {s: (load_features([s], embedding="gemma2") / rms).astype(np.float32)
            for s in stories}


def load_all_features_delayed_raw(stories):
    """Load raw delayed features for each story (for ridge).
    Returns dict: story -> np.array (n_TRs, n_sem * n_delays).
    """
    return {s: apply_delays(load_features([s], embedding="gemma2").astype(np.float32))
            for s in stories}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve(path):
    return join(PROJECT_DIR, path) if not os.path.isabs(path) else path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_model_dirs", nargs="+", required=True,
                   help="SAE base model dirs (3 seeds for frozen encoder)")
    p.add_argument("--random_encoder_dirs", nargs="+", default=None,
                   help="Random-encoder model dirs (same arch, random frozen weights)")
    p.add_argument("--transfer_ref_dir", required=True,
                   help="Existing UTS07/08 freeze-retrain dir (for voxel_indices)")
    p.add_argument("--subjects", nargs="+", default=["UTS07", "UTS08"])
    p.add_argument("--story_counts", nargs="+", type=float, default=STORY_COUNTS)
    p.add_argument("--n_story_seeds", type=int, default=N_STORY_SEEDS)
    p.add_argument("--story_seed_start", type=int, default=0)
    p.add_argument("--draw_mode", choices=["seed", "enumerate"], default="seed",
                   help="'seed': one draw per story_seed (original behaviour). "
                        "'enumerate': exhaustive when the draw space is small "
                        "(n<=1 has only len(pool) options), else distinct sets "
                        "by rejection sampling. See make_draws().")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--out_dir", default="reports/figures/learning_curve")
    p.add_argument("--input_rms", type=float, default=None,
                   help="RMS the encoder was trained with (printed in its training log as "
                        "'dividing all features by rms=...'; 11.32 for the reported model). "
                        "Default: recompute over all of STORIES_TRAIN, which gives 11.86 and "
                        "does not match training.")
    p.add_argument("--val_stories", nargs="+", default=None,
                   help="Pin the validation stories (the main model uses fromboyhoodtofatherhood "
                        "itsabox myfirstdaywiththeyankees souls). Default: a random draw per subject.")
    args = p.parse_args()

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    out_dir = resolve(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # himalaya backend
    if device == "cuda":
        hbackend = set_backend("torch_cuda", on_error="warn")
    else:
        hbackend = set_backend("torch", on_error="warn")

    # --- Load voxel indices for held-out subjects ---
    ref_ckpt = torch.load(
        join(resolve(args.transfer_ref_dir), "model_last.pt"),
        map_location="cpu", weights_only=False,
    )
    vox_idx = {s: ref_ckpt["voxel_indices"][s] for s in args.subjects}
    for s in args.subjects:
        print(f"  {s}: {len(vox_idx[s])} selected voxels")

    # --- All 26 stories (from transfer checkpoint) ---
    all_stories = sorted(ref_ckpt["train_stories"])
    if args.val_stories:
        missing = [v for v in args.val_stories if v not in all_stories]
        if missing:
            raise SystemExit(f"--val_stories not in this subject's stories: {missing}")
        val_stories = list(args.val_stories)
        pool = [s for s in all_stories if s not in set(val_stories)]
    else:
        val_stories, pool = get_val_and_pool(all_stories)
    print(f"\nAll stories: {len(all_stories)}")
    print(f"Val stories (fixed): {val_stories}")
    print(f"Pool (for training): {len(pool)} stories")

    # --- Training RMS from STORIES_TRAIN (UTS01-06) ---
    if args.input_rms is not None:
        rms = float(args.input_rms)
        print(f"\nUsing the training RMS given on the command line: {rms:.4f}")
    else:
        print("\nComputing RMS from STORIES_TRAIN (NOTE: differs from the training RMS; pass --input_rms)...")
        X_tmp = load_features(STORIES_TRAIN, embedding="gemma2")
        rms = float(np.sqrt((X_tmp ** 2).mean()))
        del X_tmp
        print(f"  RMS = {rms:.4f}")

    # --- Precompute features and responses for all stories + test ---
    all_needed = list(all_stories) + [TEST_STORY]
    print(f"\nLoading features for {len(all_needed)} stories...")
    feat_rms = load_all_features_rms(all_needed, rms)
    feat_del_raw = load_all_features_delayed_raw(all_needed)

    print("\nLoading responses...")
    resp = {}
    for subj in args.subjects:
        print(f"  {subj}...")
        resp[subj] = load_all_responses(subj, all_needed, vox_idx[subj])

    Y_test = {subj: resp[subj][TEST_STORY] for subj in args.subjects}
    X_test_raw_del = feat_del_raw[TEST_STORY]

    # --- Load SAE encoders ---
    print(f"\nLoading {len(args.base_model_dirs)} SAE encoders...")
    encoders = []
    for i, mdir in enumerate(args.base_model_dirs):
        enc, n_atoms, n_delays = load_encoder(resolve(mdir), device)
        encoders.append(enc)
        print(f"  seed {i}: {n_atoms} atoms, {n_delays} delays ({n_atoms * n_delays}-d codes)")

    # --- Load random encoders (optional) ---
    rand_encoders = []
    if args.random_encoder_dirs:
        print(f"\nLoading {len(args.random_encoder_dirs)} random encoders...")
        for i, mdir in enumerate(args.random_encoder_dirs):
            enc, n_atoms, n_delays = load_encoder(resolve(mdir), device)
            rand_encoders.append(enc)
            print(f"  seed {i}: {n_atoms} atoms, {n_delays} delays (random weights)")

    # --- Precompute sparse codes for all stories ---
    def precompute_codes(enc_list, label):
        all_codes = []
        print(f"\nPrecomputing {label} codes for all stories...")
        for seed_i, enc in enumerate(enc_list):
            seed_codes = {s: compute_codes_story(enc, feat_rms[s], device) for s in all_needed}
            all_codes.append(seed_codes)
            print(f"  seed {seed_i}: done ({seed_codes[TEST_STORY].shape[1]}-d)")
        return all_codes

    codes      = precompute_codes(encoders,      "SAE")
    rand_codes = precompute_codes(rand_encoders, "random") if rand_encoders else []

    # --- Prepare val data (fixed across all runs) ---
    X_val_raw_del = np.vstack([feat_del_raw[s] for s in val_stories])
    Z_val      = [np.vstack([codes[ss][s]      for s in val_stories]) for ss in range(len(encoders))]
    Z_val_rand = [np.vstack([rand_codes[ss][s] for s in val_stories]) for ss in range(len(rand_encoders))]
    Y_val = {subj: np.vstack([resp[subj][s] for s in val_stories]) for subj in args.subjects}

    # --- Main loop ---
    records = []
    for n_stories in args.story_counts:
        # Loud failure on the silent-clamp trap: rng.choice used min(n, len(pool)),
        # so asking for more stories than the pool holds trained on len(pool) while
        # still labelling the row with the requested n.
        if n_stories >= 1 and int(n_stories) > len(pool):
            print(f"\n[WARN] n_stories={n_stories} exceeds pool size {len(pool)}; "
                  f"rows will be recorded with n_stories_actual={len(pool)}.")
        draws = make_draws(n_stories, len(pool), args.n_story_seeds,
                           args.story_seed_start, args.draw_mode)
        print(f"\n[draws] n_stories={n_stories}: {len(draws)} distinct training "
              f"set(s) via draw_mode={args.draw_mode}")
        for story_seed, chosen in draws:
            if n_stories < 1:
                # Fractional story: pick 1 story, keep first (fraction * n_TRs) rows
                chosen_idx = int(chosen[0])
                train_stories_n = [pool[chosen_idx]]
                story = train_stories_n[0]
                n_full = feat_del_raw[story].shape[0]
                n_keep = max(1, int(round(n_stories * n_full)))
                print(f"\nn_stories={n_stories}, story_seed={story_seed}: "
                      f"train=['{story}'] first {n_keep}/{n_full} TRs "
                      f"({n_keep*2/60:.1f} min)")
                X_train_raw_del = feat_del_raw[story][:n_keep]
                Z_train      = [codes[ss][story][:n_keep]      for ss in range(len(encoders))]
                Z_train_rand = [rand_codes[ss][story][:n_keep] for ss in range(len(rand_encoders))]
                tr_resp = {subj: resp[subj][story][:n_keep] for subj in args.subjects}
            else:
                # Integer story count
                train_stories_n = [pool[i] for i in chosen]
                print(f"\nn_stories={n_stories}, story_seed={story_seed}: "
                      f"train={train_stories_n[:3]}{'...' if len(train_stories_n)>3 else ''}")
                X_train_raw_del = np.vstack([feat_del_raw[s] for s in train_stories_n])
                Z_train      = [np.vstack([codes[ss][s]      for s in train_stories_n])
                                for ss in range(len(encoders))]
                Z_train_rand = [np.vstack([rand_codes[ss][s] for s in train_stories_n])
                                for ss in range(len(rand_encoders))]
                tr_resp = {subj: np.vstack([resp[subj][s] for s in train_stories_n])
                           for subj in args.subjects}

            # Exact identity of this training set, so downstream analysis can
            # deduplicate on the draw itself instead of inferring it from the
            # ridge score. n_stories_actual records what was really used.
            draw_key = ",".join(sorted(train_stories_n))
            n_actual = (n_stories if n_stories < 1 else len(train_stories_n))

            for subj in args.subjects:
                Y_train_subj = tr_resp[subj]

                # --- Ridge ---
                r_ridge = fit_ridge(
                    X_train_raw_del, Y_train_subj,
                    X_val_raw_del, Y_val[subj],
                    X_test_raw_del, Y_test[subj],
                    hbackend,
                )
                records.append(dict(
                    subject=subj, n_stories=n_stories, story_seed=story_seed,
                    sae_seed=-1, condition="ridge", brain_r=r_ridge,
                    n_stories_actual=n_actual, draw_key=draw_key,
                ))
                print(f"  {subj} ridge: {r_ridge:.4f}")

                # --- Freeze-retrain (per SAE seed) ---
                for sae_seed in range(len(encoders)):
                    r_frt = train_decoder(
                        Z_train[sae_seed], Y_train_subj,
                        Z_val[sae_seed], Y_val[subj],
                        codes[sae_seed][TEST_STORY], Y_test[subj],
                        device,
                    )
                    records.append(dict(
                        subject=subj, n_stories=n_stories, story_seed=story_seed,
                        sae_seed=sae_seed, condition="freeze_retrain", brain_r=r_frt,
                        n_stories_actual=n_actual, draw_key=draw_key,
                    ))
                    print(f"  {subj} frt sae_seed={sae_seed}: {r_frt:.4f}")

                # --- Random encoder freeze-retrain (per random seed) ---
                for rand_seed in range(len(rand_encoders)):
                    r_rand = train_decoder(
                        Z_train_rand[rand_seed], Y_train_subj,
                        Z_val_rand[rand_seed], Y_val[subj],
                        rand_codes[rand_seed][TEST_STORY], Y_test[subj],
                        device,
                    )
                    records.append(dict(
                        subject=subj, n_stories=n_stories, story_seed=story_seed,
                        sae_seed=rand_seed, condition="random_encoder", brain_r=r_rand,
                        n_stories_actual=n_actual, draw_key=draw_key,
                    ))
                    print(f"  {subj} rand sae_seed={rand_seed}: {r_rand:.4f}")

    # --- Save ---
    df = pd.DataFrame(records)
    csv_path = join(out_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows to {csv_path}")

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("SUMMARY: mean ± std across story_seeds × sae_seeds")
    print("=" * 60)
    conds = ["ridge", "freeze_retrain"] + (["random_encoder"] if rand_encoders else [])
    for cond in conds:
        sub_df = df[df.condition == cond]
        for n in args.story_counts:
            vals = sub_df[sub_df.n_stories == n]["brain_r"].values
            print(f"  {cond:<16} n={n}: {vals.mean():.4f} ± {vals.std():.4f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
