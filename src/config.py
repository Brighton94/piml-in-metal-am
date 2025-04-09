"""Configuration settings for the project.

This module handles the dataset paths and constants used throughout the project.
It loads environment variables for dataset locations and provides a function to
retrieve the dataset path.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants for segmentation
CLASS_ID_STREAK = 3
CLASS_ID_SPATTER = 8

DATASET_ROOT = os.getenv("DATASET_ROOT", "/data")
EXTERNAL_DATASET_PATH = os.getenv("EXTERNAL_DRIVE_PATH", "/external/l-pbf-dataset")

# Define dataset paths with multiple possible locations
DATASET_PATHS = {
    "tcr_phase1_build1": [
        Path(DATASET_ROOT) / "2021-07-13 TCR Phase 1 Build 1.hdf5",
        Path(EXTERNAL_DATASET_PATH) / "2021-07-13 TCR Phase 1 Build 1.hdf5",
    ],
    "tcr_phase1_build2": [
        Path(DATASET_ROOT) / "2021-04-16 TCR Phase 1 Build 2.hdf5",
        Path(EXTERNAL_DATASET_PATH) / "2021-04-16 TCR Phase 1 Build 2.hdf5",
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
