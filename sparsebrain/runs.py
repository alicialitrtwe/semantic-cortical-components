"""Find training runs by their saved settings, not by folder names.

Every run of scripts/train_multisubj_hydra.py writes ``args.json`` next to its checkpoints. The
analysis scripts select runs by those settings (for example ``n_atoms=50, k=5, alpha=1.0``) and
read the seed from them, so they do not depend on how run folders are named. For runs without
``args.json`` the settings are read from the checkpoint itself.
"""
from __future__ import annotations

import glob
import json
import os

DEFAULT_ROOT = os.path.join("outputs", "models")

# Settings added after some runs were trained; a run without them behaved as the value given here.
MISSING_MEANS = {"val_masked": False, "zscore_own_rows": False}

# The coupled SAE of the paper (Table 1); the 20-seed stability runs use the same settings.
PAPER_COUPLED = dict(subjects="UTS01,UTS02,UTS03,UTS04,UTS05,UTS06", all_stories=True, n_atoms=50, k=5, alpha=1.0, train_mode="coupled", freeze_from=None,
                     no_encoder_norm=False, brain_voxel_scale=False, encoder_mode="linear",
                     matryoshka_levels=None, val_masked=False, zscore_own_rows=False)


def run_args(run_dir: str) -> dict:
    """Training settings of one run (``args.json``, else the checkpoint's ``args``)."""
    path = os.path.join(run_dir, "args.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    import torch

    for name in ("model_last.pt", "model.pt"):
        ckpt = os.path.join(run_dir, name)
        if os.path.exists(ckpt):
            a = torch.load(ckpt, map_location="cpu", weights_only=False)["args"]
            return a if isinstance(a, dict) else vars(a)
    raise FileNotFoundError(f"no args.json or checkpoint in {run_dir}")


def _matches(args: dict, want: dict) -> bool:
    for key, value in want.items():
        have = args.get(key, MISSING_MEANS.get(key))
        if isinstance(value, float) or isinstance(have, float):
            try:
                if abs(float(have) - float(value)) > 1e-9:
                    return False
            except (TypeError, ValueError):
                return False
        elif have != value:
            return False
    return True


def find_runs(pattern: str | None = None, root: str = DEFAULT_ROOT, require: str = "model_last.pt",
              seeds=None, name_contains: str | None = None, **want) -> list[str]:
    """Run folders that contain ``require`` and whose settings match ``want``, sorted by seed.

    pattern        optional glob of run folders (default: every folder under ``root``)
    seeds          optional iterable of seeds to keep; one run per seed (the first by name)
    name_contains  optional substring of the run's ``name_suffix`` setting
    want           settings that must match, e.g. n_atoms=50, k=5, alpha=1.0, train_mode="coupled"
    """
    dirs = sorted(glob.glob(pattern) if pattern else glob.glob(os.path.join(root, "*")))
    picked = {}
    for d in dirs:
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, require)):
            continue
        a = run_args(d)
        if name_contains and name_contains not in str(a.get("name_suffix", "")):
            continue
        if not _matches(a, want):
            continue
        s = int(a.get("seed", -1))
        if seeds is not None and s not in set(seeds):
            continue
        picked.setdefault(s, d)
    return [picked[s] for s in sorted(picked)]


def seed_of(run_dir: str) -> int:
    return int(run_args(run_dir).get("seed", -1))
