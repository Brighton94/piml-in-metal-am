"""Exports images and segmentation masks from an HDF5 file for YOLO training."""

import argparse
import logging
import os

import cv2
import h5py
import imageio
import numpy as np
from src.config import (
    CLASS_ID_SPATTER,
    CLASS_ID_STREAK,
    DATASET_PATHS,
    get_dataset_path,
)
from src.utils.yolo_segmentation import load_hdf5_slice
from tqdm import tqdm  # Add this import

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_PATH: str | None = None
IMG_TRAIN_DIR: str | None = None
LBL_TRAIN_DIR: str | None = None

CAMERA_PATH = "slices/camera_data/visible/0"

SPATTER_MASK_HDF5_PATH = f"slices/segmentation_results/{CLASS_ID_SPATTER}"
STREAK_MASK_HDF5_PATH = f"slices/segmentation_results/{CLASS_ID_STREAK}"


def _create_output_directories() -> None:
    """Create output directories for images and labels if they don't exist."""
    if not IMG_TRAIN_DIR or not LBL_TRAIN_DIR:
        logger.error("Output directories are not set. Cannot create them.")
        return
    logger.info(f"Target training image directory: {IMG_TRAIN_DIR}")
    logger.info(f"Target training label directory: {LBL_TRAIN_DIR}")
    os.makedirs(IMG_TRAIN_DIR, exist_ok=True)
    os.makedirs(LBL_TRAIN_DIR, exist_ok=True)


def _validate_hdf5_file() -> h5py.File | None:
    """Validate HDF5 file existence and camera path, return HDF5 file object or None."""
    if not DATA_PATH:
        logger.error("DATA_PATH is not set. Cannot validate HDF5 file.")
        return None

    if not os.path.exists(DATA_PATH):
        logger.error(f"HDF5 file not found at {DATA_PATH}.")
        return None

    h5_file = h5py.File(DATA_PATH, "r")
    if CAMERA_PATH not in h5_file:
        logger.error(
            f"CAMERA_PATH '{CAMERA_PATH}' not found in HDF5 file. "
            f"Available top-level keys: {list(h5_file.keys())}"
        )
        h5_file.close()
        return None
    return h5_file


def _check_mask_paths(h5_file: h5py.File) -> tuple[bool, bool]:
    """Check for spatter and streak mask paths in the HDF5 file."""
    has_spatter_mask = SPATTER_MASK_HDF5_PATH in h5_file
    has_streak_mask = STREAK_MASK_HDF5_PATH in h5_file

    if not has_spatter_mask:
        logger.warning(
            f"Spatter mask path '{SPATTER_MASK_HDF5_PATH}' "
            f"(derived from CLASS_ID_SPATTER={CLASS_ID_SPATTER}) not found."
        )
    if not has_streak_mask:
        logger.warning(
            f"Streak mask path '{STREAK_MASK_HDF5_PATH}' "
            f"(derived from CLASS_ID_STREAK={CLASS_ID_STREAK}) not found."
        )

    if not has_spatter_mask and not has_streak_mask:
        logger.error(
            "Neither spatter nor streak mask paths found. Cannot export labels."
        )
        if "slices/segmentation_results" in h5_file:
            logger.info(
                "Available segmentation classes in HDF5: "
                f"{list(h5_file['slices/segmentation_results'].keys())}"
            )
        return False, False

    return has_spatter_mask, has_streak_mask


def _process_layer(
    h5_file: h5py.File,
    layer_index: int,
    has_spatter_mask: bool,
    has_streak_mask: bool,
) -> None:
    """Load, process, and save a single layer's image and label."""
    if not DATA_PATH or not IMG_TRAIN_DIR or not LBL_TRAIN_DIR:
        logger.error(
            "Essential paths (DATA_PATH, IMG_TRAIN_DIR, LBL_TRAIN_DIR) not set."
        )
        return

    img = load_hdf5_slice(DATA_PATH, layer_index, CAMERA_PATH)
    image_filename = os.path.join(IMG_TRAIN_DIR, f"{layer_index:05d}.png")
    imageio.imwrite(image_filename, img)

    label = np.zeros(img.shape[:2], dtype=np.uint8)  # Background = 0

    if has_spatter_mask:
        sp_mask_data = h5_file[SPATTER_MASK_HDF5_PATH][layer_index]
        if sp_mask_data.shape != label.shape:
            logger.warning(
                f"Spatter mask shape {sp_mask_data.shape} does not match image "
                f"shape {label.shape} for layer {layer_index}. Resizing."
            )
            sp_mask_data = cv2.resize(
                sp_mask_data.astype(np.uint8),
                (label.shape[1], label.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        label[sp_mask_data.astype(bool)] = 1  # Spatter = 1

    if has_streak_mask:
        st_mask_data = h5_file[STREAK_MASK_HDF5_PATH][layer_index]
        if st_mask_data.shape != label.shape:
            logger.warning(
                f"Streak mask shape {st_mask_data.shape} does not match image "
                f"shape {label.shape} for layer {layer_index}. Resizing."
            )
            st_mask_data = cv2.resize(
                st_mask_data.astype(np.uint8),
                (label.shape[1], label.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        label[st_mask_data.astype(bool)] = 2  # Streak = 2

    label_filename = os.path.join(LBL_TRAIN_DIR, f"{layer_index:05d}.png")
    imageio.imwrite(label_filename, label)


def export_data_for_yolo() -> None:
    """Exports images & segmentation masks from the HDF5 file for a configured build."""
    if not DATA_PATH or not IMG_TRAIN_DIR or not LBL_TRAIN_DIR:
        logger.error(
            "DATA_PATH, IMG_TRAIN_DIR, or LBL_TRAIN_DIR not configured. "
            "Exiting export process."
        )
        return

    _create_output_directories()
    logger.info(f"Reading HDF5 data from: {DATA_PATH}")

    h5_file = _validate_hdf5_file()
    if not h5_file:
        return

    try:
        has_spatter_mask, has_streak_mask = _check_mask_paths(h5_file)
        if not has_spatter_mask and not has_streak_mask:
            logger.error("Required mask paths not found in HDF5. Aborting export.")
            return

        num_layers = h5_file[CAMERA_PATH].shape[0]
        logger.info(f"Exporting {num_layers} layers as images and masks...")

        for layer_idx in tqdm(range(num_layers), desc="Processing layers"):
            _process_layer(h5_file, layer_idx, has_spatter_mask, has_streak_mask)

        logger.info(
            f"Export complete: {num_layers} layers exported to "
            f"{IMG_TRAIN_DIR} and {LBL_TRAIN_DIR}"
        )

    except Exception as e:
        logger.exception(
            f"An error occurred during HDF5 processing or file export: {e}"
        )
    finally:
        if h5_file:
            h5_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export image and label data from HDF5 for YOLO training."
    )
    parser.add_argument(
        "--build_key",
        type=str,
        required=False,  # Changed to False to allow --list_builds alone
        help="Dataset build key (e.g., tcr_phase1_build1). "
        "Use --list_builds to see available keys.",
    )
    parser.add_argument(
        "--list_builds",
        action="store_true",
        help="List available dataset build keys and exit.",
    )
    args = parser.parse_args()

    if args.list_builds:
        print("Available dataset build keys:")
        for key in DATASET_PATHS:
            print(f"  - {key}")
        exit(0)

    if not args.build_key:
        parser.error("the following arguments are required: --build_key")
        exit(1)  # Redundant due to parser.error, but for clarity

    build_key = args.build_key

    data_path_obj = get_dataset_path(build_key)
    if data_path_obj is None:
        logger.error(
            f"Failed to get dataset path for '{build_key}' from config. "
            "Please check 'src/config.py' and data availability."
        )
        exit(1)
    DATA_PATH = str(data_path_obj)

    IMG_TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", build_key, "images", "train")
    LBL_TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", build_key, "labels", "train")

    logger.info(f"Starting YOLO data export process for build: {build_key}")
    export_data_for_yolo()
    logger.info("YOLO data export process finished.")
