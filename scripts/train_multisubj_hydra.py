#!/usr/bin/env python
"""Hydra wrapper around scripts/17_fit_multisubject_sae.py.

Runs a single config:
    uv run python scripts/train_multisubj_hydra.py

Overrides on the CLI:
    uv run python scripts/train_multisubj_hydra.py alpha=0.3 seed=1

Multi-run sweeps (creates one job per combination):
    uv run python scripts/train_multisubj_hydra.py -m alpha=0.3,1.0,3.0,10.0
    uv run python scripts/train_multisubj_hydra.py -m alpha=0.3,1.0 seed=0,1,2

The config is converted to the same argparse.Namespace that
17_fit_multisubject_sae.main accepts, so logic stays in one place.
"""
import os
import sys
from argparse import Namespace

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "scripts"))

# Import the original training script's main() (imports work because
# the sys.path entries above include scripts/).
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_train17",
    os.path.join(PROJECT_DIR, "scripts", "17_fit_multisubject_sae.py"),
)
_train17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train17)


@hydra.main(
    version_base=None,
    config_path=os.path.join(PROJECT_DIR, "configs"),
    config_name="multisubj",
)
def run(cfg: DictConfig) -> None:
    """Convert hydra config to argparse Namespace and delegate to main()."""
    plain = OmegaConf.to_container(cfg, resolve=True)
    plain.pop("hydra", None)

    # Build a clean, readable name_suffix from key parameters.
    suffix = plain.get("name_suffix") or ""
    for key, fmt in [
        ("embedding", "{}"),
        ("n_atoms", "{}at"),
        ("k", "k{}"),
        ("alpha", "a{}"),
        ("seed", "s{}"),
        ("train_mode", "{}"),
    ]:
        val = plain.get(key)
        if val is not None:
            tag = fmt.format(val).replace(".", "p")
            if key == "train_mode" and val == "coupled":
                continue
            if tag not in suffix:
                suffix = f"{suffix}_{tag}" if suffix else tag

    # Add subject info if non-default
    subj = plain.get("subjects", "")
    if subj and subj != "UTS01,UTS02,UTS03":
        subj_tag = subj.replace(",", "_")
        if subj_tag not in suffix:
            suffix = f"{suffix}_{subj_tag}" if suffix else subj_tag

    if plain.get("all_stories"):
        if "allstories" not in suffix:
            suffix = f"{suffix}_allstories" if suffix else "allstories"

    plain["name_suffix"] = suffix

    args = Namespace(**plain)
    _train17.main(args)


if __name__ == "__main__":
    run()
