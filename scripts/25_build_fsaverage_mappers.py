#!/usr/bin/env python
"""25_build_fsaverage_mappers.py — Build voxel→fsaverage mappers for each subject.

Creates sparse matrices that map from subject-specific voxel space to
fsaverage surface vertices. Adapted from compose/interpretation/save_mappers.py.

Requires FreeSurfer installation and subject reconstructions.

Usage:
    uv run python scripts/25_build_fsaverage_mappers.py
"""
import os
import sys
from os.path import join

import h5py
import numpy as np
import scipy.sparse

# FreeSurfer setup
if "FREESURFER_HOME" not in os.environ:
    raise SystemExit("Set FREESURFER_HOME to your FreeSurfer installation.")
os.environ["PATH"] = join(os.environ["FREESURFER_HOME"], "bin") + ":" + os.environ["PATH"]

# FreeSurfer subjects dir (deep-fMRI-dataset)
FS_SUBJECTS_DIR = join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "external", "deep-fMRI-dataset", "data", "ds003020",
    "derivative", "freesurfer_subjdir",
)
os.environ["SUBJECTS_DIR"] = FS_SUBJECTS_DIR

import cortex

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

SUBJECTS = ["UTS01", "UTS02", "UTS03", "UTS04", "UTS05", "UTS06"]
OUT_DIR = join(PROJECT_DIR, "outputs", "mappers")


def build_mapper(subject: str):
    """Build voxel→fsaverage sparse matrix for one subject."""
    xfm_name = f"{subject}_auto"
    print(f"\n{'='*60}")
    print(f"Building mapper for {subject}")
    print(f"{'='*60}")

    # Step 1: Voxel → native surface (nearest neighbor)
    print("  Step 1: voxel → native surface...")
    mapper = cortex.get_mapper(subject, xfm_name, type="nearest", recache=True)
    voxel_to_surface_L, voxel_to_surface_R = mapper.masks

    # Get the voxel mask (which voxels are in the brain)
    mask = cortex.db.get_mask(subject, xfm_name)
    n_voxels_total = mask.sum()
    print(f"  Total voxels in mask: {n_voxels_total}")

    # Constrain to masked voxels only
    voxel_to_surface_L = voxel_to_surface_L[:, mask.flatten()]
    voxel_to_surface_R = voxel_to_surface_R[:, mask.flatten()]

    # Step 2: Native surface → fsaverage
    print("  Step 2: native surface → fsaverage (may take ~5 min)...")
    freesurfer_surface = subject

    surface_L_to_fsaverage_L = cortex.freesurfer.get_mri_surf2surf_matrix(
        freesurfer_surface, hemi="lh", surface_type="fiducial",
        target_subj="fsaverage", subjects_dir=FS_SUBJECTS_DIR,
    )
    surface_R_to_fsaverage_R = cortex.freesurfer.get_mri_surf2surf_matrix(
        freesurfer_surface, hemi="rh", surface_type="fiducial",
        target_subj="fsaverage", subjects_dir=FS_SUBJECTS_DIR,
    )

    # Combine: voxel → fsaverage
    voxel_to_fsaverage_L = surface_L_to_fsaverage_L * voxel_to_surface_L
    voxel_to_fsaverage_R = surface_R_to_fsaverage_R * voxel_to_surface_R

    voxel_to_fsaverage = scipy.sparse.vstack(
        [voxel_to_fsaverage_L, voxel_to_fsaverage_R]
    ).tocsr()
    print(f"  Mapper shape: {voxel_to_fsaverage.shape}")
    print(f"  (n_fsaverage_vertices x n_voxels)")

    return voxel_to_fsaverage, n_voxels_total


def save_mapper(subject: str, mapper: scipy.sparse.csr_matrix, n_voxels: int):
    """Save mapper to HDF5."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = join(OUT_DIR, f"{subject}_voxel_to_fsaverage.hdf")

    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=mapper.data)
        f.create_dataset("indices", data=mapper.indices)
        f.create_dataset("indptr", data=mapper.indptr)
        f.create_dataset("shape", data=np.array(mapper.shape))
        f.attrs["subject"] = subject
        f.attrs["n_voxels"] = n_voxels

    print(f"  Saved to {path}")


def load_mapper(subject: str) -> scipy.sparse.csr_matrix:
    """Load saved mapper from HDF5."""
    path = join(OUT_DIR, f"{subject}_voxel_to_fsaverage.hdf")
    with h5py.File(path, "r") as f:
        data = f["data"][:]
        indices = f["indices"][:]
        indptr = f["indptr"][:]
        shape = tuple(f["shape"][:])
    return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)


def main():
    print(f"FreeSurfer subjects dir: {FS_SUBJECTS_DIR}")
    print(f"Output dir: {OUT_DIR}")

    for subject in SUBJECTS:
        # Check if FreeSurfer data exists
        sphere_path = join(FS_SUBJECTS_DIR, subject, "surf", "lh.sphere.reg")
        if not os.path.exists(sphere_path) or os.path.getsize(sphere_path) < 100:
            print(f"\nSkipping {subject}: FreeSurfer sphere.reg not available")
            continue

        try:
            mapper, n_voxels = build_mapper(subject)
            save_mapper(subject, mapper, n_voxels)
        except Exception as e:
            print(f"\nERROR building mapper for {subject}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
