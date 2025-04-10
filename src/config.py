"""Configuration file for the l-pbf-dataset."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants for segmentation
CLASS_ID_STREAK = 3
CLASS_ID_SPATTER = 8

DATASET_ROOT = os.getenv("HOST_DATASET_PATH", "/data")

# Define dataset paths with single location
DATASET_PATHS = {
    "tcr_phase1_build1": [
        Path(DATASET_ROOT) / "2021-07-13 TCR Phase 1 Build 1.hdf5",
    ],
    "tcr_phase1_build2": [
        Path(DATASET_ROOT) / "2021-04-16 TCR Phase 1 Build 2.hdf5",
    ],
    "tcr_phase1_build3": [
        Path(DATASET_ROOT) / "2021-05-03 TCR Phase 1 Build 3.hdf5",
    ],
    "tcr_phase1_build4": [
        Path(DATASET_ROOT) / "2021-05-17 TCR Phase 1 Build 4.hdf5",
    ],
    "tcr_phase1_build5": [
        Path(DATASET_ROOT) / "2021-06-01 TCR Phase 1 Build 5.hdf5",
    ],
}


def get_dataset_path(dataset_key: str) -> Path | None:
    """Get the full path for a dataset."""
    if dataset_key not in DATASET_PATHS:
        raise KeyError(f"Dataset key '{dataset_key}' not found in configuration")

    # Try each possible path for this dataset
    for path in DATASET_PATHS[dataset_key]:
        if path.exists():
            return path

    # If we get here, none of the paths existed
    paths_str = "\n - ".join([str(p) for p in DATASET_PATHS[dataset_key]])
    print(
        f"Warning: Dataset '{dataset_key}' not found in any of the configured locations:\n"  # noqa: E501
        f" - {paths_str}"
    )
    return None
