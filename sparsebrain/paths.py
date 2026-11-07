"""Centralized path management for the sparsebrain project.

All paths to data, outputs, and external resources are defined here
to avoid hard-coded paths scattered across scripts.
"""
import os
import warnings
from dataclasses import dataclass
from functools import cached_property

# Project root (where pyproject.toml lives)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mapping from short embedding names to filenames used on disk.
# 02_save_features.py writes files with these names.
_EMBEDDING_FILE_NAMES = {
    "eng1000": "english1000",
    "glove_factor": "glove-factor",
}


@dataclass
class Paths:
    """Centralized path management for all data and outputs.

    Usage:
        from sparsebrain.paths import paths
        responses_dir = paths.preprocessed_data / "UTS03"
    """

    project_dir: str = PROJECT_DIR

    @cached_property
    def dataset_dir(self) -> str:
        return os.path.join(self.project_dir, "external", "deep-fMRI-dataset")

    @cached_property
    def data_dir(self) -> str:
        return os.path.join(self.dataset_dir, "data")

    @cached_property
    def derivative_dir(self) -> str:
        return os.path.join(self.data_dir, "ds003020", "derivative")

    @cached_property
    def preprocessed_data(self) -> str:
        return os.path.join(self.derivative_dir, "preprocessed_data")

    @cached_property
    def pycortex_db(self) -> str:
        return os.path.join(self.derivative_dir, "pycortex-db")

    @cached_property
    def features_dir(self) -> str:
        return os.path.join(self.project_dir, "outputs", "features")

    @cached_property
    def models_dir(self) -> str:
        return os.path.join(self.project_dir, "outputs", "models")

    @cached_property
    def reports_dir(self) -> str:
        return os.path.join(self.project_dir, "reports")

    @cached_property
    def figures_dir(self) -> str:
        return os.path.join(self.reports_dir, "figures")

    @cached_property
    def encoding_dir(self) -> str:
        """Directory containing encoding model outputs from the dataset."""
        return os.path.join(self.dataset_dir, "encoding")

    @cached_property
    def compose_dir(self) -> str:
        """External compose project directory (for GloVe vectors).

        Set via SPARSEBRAIN_COMPOSE_DIR environment variable,
        """
        path = os.environ.get("SPARSEBRAIN_COMPOSE_DIR", os.path.join(self.project_dir, "external", "compose"))
        if not os.path.isdir(path):
            warnings.warn(
                f"compose_dir does not exist: {path}\n"
                f"Set SPARSEBRAIN_COMPOSE_DIR environment variable to the correct path.",
                UserWarning,
                stacklevel=2,
            )
        return path

    @cached_property
    def glove_vectors_path(self) -> str:
        """Path to pre-trained GloVe word vectors.

        Uses compose project's transformed GloVe vectors.
        """
        return os.path.join(
            self.compose_dir, "data", "wordvectors", "glove-transformed-v2.kv"
        )

    def subject_data_dir(self, subject: str) -> str:
        """Get preprocessed data directory for a subject."""
        return os.path.join(self.preprocessed_data, subject)

    def subject_pycortex_dir(self, subject: str) -> str:
        """Get pycortex directory for a subject."""
        return os.path.join(self.pycortex_db, subject)

    def thick_mask_path(self, subject: str) -> str:
        """Get path to thick mask NIfTI for a subject."""
        return os.path.join(
            self.pycortex_db, subject, "transforms",
            f"{subject}_auto", "mask_thick.nii.gz"
        )

    def features_path(self, embedding: str, subject: str = None) -> str:
        """Get path to pre-computed features file.

        Parameters
        ----------
        embedding : str
            Embedding type ('glove', 'eng1000', 'glove_factor', etc.)
        subject : str, optional
            Subject ID for subject-specific features.

        Returns
        -------
        str
            Path to features file (may or may not exist).
        """
        file_name = _EMBEDDING_FILE_NAMES.get(embedding, embedding)
        if subject:
            return os.path.join(self.features_dir, f"stories-lebel-{file_name}-{subject}.npz")
        return os.path.join(self.features_dir, f"stories-lebel-{file_name}.npz")


    # ------------------------------------------------------------------ #
    # Narratives dataset (Nastase et al., 2021, ds002345)
    # Source: hasson lab DataLad collection with fMRIPrep + AFNI derivatives
    # ------------------------------------------------------------------ #

    @cached_property
    def narratives_dir(self) -> str:
        return os.path.join(self.project_dir, "external", "narratives")

    @cached_property
    def narratives_raw_dir(self) -> str:
        """OpenNeuro BIDS clone (raw data + hasson lab derivatives)."""
        return os.path.join(self.narratives_dir, "data", "ds002345")

    @cached_property
    def narratives_hasson_dir(self) -> str:
        """Hasson lab DataLad collection (has preprocessed derivatives)."""
        return os.path.join(self.narratives_raw_dir, "narratives")

    @cached_property
    def narratives_afni_dir(self) -> str:
        """AFNI-cleaned BOLD derivatives (confound-regressed, fsaverage6 surface)."""
        return os.path.join(self.narratives_hasson_dir, "derivatives", "afni-nosmooth")

    @cached_property
    def narratives_fmriprep_dir(self) -> str:
        """fMRIPrep derivatives directory."""
        return os.path.join(self.narratives_hasson_dir, "derivatives", "fmriprep")

    @cached_property
    def narratives_gentle_dir(self) -> str:
        """Gentle forced-aligner word timing JSON files."""
        return os.path.join(self.narratives_hasson_dir, "stimuli", "gentle")

    @cached_property
    def narratives_preprocessed_data(self) -> str:
        """Our custom HDF5 files (n_TRs × n_vertices) after preprocessing."""
        return os.path.join(self.narratives_dir, "derivative", "preprocessed_data")

    @cached_property
    def narratives_textgrids_dir(self) -> str:
        """Word-timing TextGrids converted from Gentle JSON."""
        return os.path.join(self.narratives_dir, "derivative", "TextGrids")

    def narratives_subject_data_dir(self, subject: str) -> str:
        return os.path.join(self.narratives_preprocessed_data, subject)

    def narratives_afni_path(self, subject: str, story: str,
                             hemi: str, run: str = None) -> str:
        """Path to AFNI-cleaned surface GIfTI for a Narratives subject."""
        run_tag = f"_{run}" if run else ""
        fn = f"{subject}_task-{story}{run_tag}_space-fsaverage6_hemi-{hemi}_desc-clean.func.gii"
        return os.path.join(self.narratives_afni_dir, subject, "func", fn)

    def narratives_gentle_path(self, story: str) -> str:
        return os.path.join(self.narratives_gentle_dir, story, "align.json")

    # ------------------------------------------------------------------ #
    # HCP dataset (Human Connectome Project, WU-Minn)
    # Language task: story listening + math, TR=0.72s, ~1200 subjects
    # Requires HCP data use agreement; download via ConnectomeDB or S3
    # ------------------------------------------------------------------ #

    @cached_property
    def hcp_dir(self) -> str:
        return os.path.join(self.project_dir, "external", "hcp")

    @cached_property
    def hcp_data_dir(self) -> str:
        """Root of HCP_1200 subject directories."""
        return os.path.join(self.hcp_dir, "data", "HCP_1200")

    @cached_property
    def hcp_behavioral_dir(self) -> str:
        """HCP behavioral battery CSV files."""
        return os.path.join(self.hcp_dir, "derivative", "behavioral")

    @cached_property
    def hcp_preprocessed_data(self) -> str:
        """Our custom HDF5 files after extracting HCP story BOLD."""
        return os.path.join(self.hcp_dir, "derivative", "preprocessed_data")

    def hcp_subject_bold_dir(self, subject: str) -> str:
        """MNINonLinear/Results/tfMRI_LANGUAGE directory for a subject."""
        return os.path.join(self.hcp_data_dir, subject,
                            "MNINonLinear", "Results", "tfMRI_LANGUAGE")

    def hcp_subject_data_dir(self, subject: str) -> str:
        return os.path.join(self.hcp_preprocessed_data, subject)


# Global singleton instance
paths = Paths()
