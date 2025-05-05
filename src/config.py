"""Configuration file for the l-pbf-dataset."""

from pathlib import Path

# Constants for segmentation
CLASS_ID_STREAK = 3
CLASS_ID_SPATTER = 8

# Define the base path where datasets are mounted inside the container
# This should match the 'target' in devcontainer.json mounts, plus any subdirectories
DATASET_BASE_PATH = Path("/mnt/ssd/l-pbf-dataset")

# Define dataset paths relative to the base path
DATASET_PATHS = {
    "tcr_phase1_build1": [
        DATASET_BASE_PATH / "2021-07-13 TCR Phase 1 Build 1.hdf5",
    ],
    "tcr_phase1_build2": [
        DATASET_BASE_PATH / "2021-04-16 TCR Phase 1 Build 2.hdf5",
    ],
    "tcr_phase1_build3": [
        DATASET_BASE_PATH / "2021-07-13 TCR Phase 1 Build 3.hdf5",
    ],
    "tcr_phase1_build4": [
        DATASET_BASE_PATH / "2021-08-03 TCR Phase 1 Build 4.hdf5",
    ],
    "tcr_phase1_build5": [
        DATASET_BASE_PATH / "2021-08-23 TCR Phase 1 Build 5.hdf5",
    ],
}


def get_dataset_path(dataset_key: str) -> Path | None:
    """Get the full path for a dataset."""
    if dataset_key not in DATASET_PATHS:
        raise KeyError(f"Dataset key '{dataset_key}' not found in configuration")

    expected_path = DATASET_PATHS[dataset_key][0]

    if expected_path.exists():
        return expected_path

    print(
        f"Warning: Dataset '{dataset_key}' not found at the expected location:\n"
        f" - {expected_path}"
    )
    return None
