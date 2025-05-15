"""Helper functions for YOLO-based segmentation on build slices with batch support."""

import logging
import os
from typing import Any

import cv2
import h5py
import numpy as np
from ultralytics import YOLO  # pip install ultralytics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_hdf5_slice(hdf5_path: str, layer: int, dataset_path: str) -> np.ndarray:
    """Load a single slice (layer) from an HDF5 file."""
    if not os.path.exists(hdf5_path):
        logger.error(f"HDF5 file not found: {hdf5_path}")
        raise FileNotFoundError(hdf5_path)
    with h5py.File(hdf5_path, "r") as h5:
        arr = h5[dataset_path][layer]
        img = np.array(arr, dtype=np.uint8)
        # ensure 3-channel
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.concatenate([img] * 3, axis=2)
    return img


def load_hdf5_stack(hdf5_path: str, dataset_path: str) -> np.ndarray:
    """Load entire 3D image stack from HDF5 as a (N,H,W,3) uint8 array."""
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(hdf5_path)
    with h5py.File(hdf5_path, "r") as h5:
        raw = np.array(h5[dataset_path], dtype=np.uint8)  # (N,H,W) or (N,H,W,1)
    # convert to (N,H,W,3)
    if raw.ndim == 3:
        raw = np.stack([raw] * 3, axis=-1)
    elif raw.ndim == 4 and raw.shape[-1] == 1:
        raw = np.concatenate([raw] * 3, axis=-1)
    return raw


def load_yolo_model(weights_path: str = "yolov8s-seg.pt") -> Any:
    """Load a YOLOv8 segmentation model."""
    logger.info(f"Loading YOLO model from {weights_path}")
    return YOLO(weights_path)


def run_yolo_segmentation(model: Any, images: np.ndarray, **kwargs) -> Any:
    """Run YOLO segmentation on one image or a batch of images."""
    return model(images, imgsz=kwargs.get("imgsz", 640), conf=kwargs.get("conf", 0.25))


def extract_anomaly_mask(
    results: Any, class_ids: list[int], image_shape: tuple[int, int]
) -> np.ndarray:
    """Combine YOLO masks into a binary anomaly mask."""
    mask = np.zeros(image_shape, dtype=np.uint8)
    for idx, cls in enumerate(results.boxes.cls.cpu().numpy().astype(int)):
        if cls in class_ids:
            single = results.masks.data[idx].cpu().numpy().astype(np.uint8)
            mask = np.maximum(mask, single)
    return mask


def compute_anomaly_area(mask: np.ndarray, pixel_size_mm2: float = 1.0) -> float:
    """Compute the total anomaly area in square millimeters."""
    px = int(mask.sum())
    mm2 = px * pixel_size_mm2
    logger.info(f"Anomaly area: {px} px → {mm2} mm²")
    return mm2


def batch_predict_and_compute_areas(
    model: Any,
    images: np.ndarray,
    class_ids: list[int],
    pixel_size_mm2: float = 1.0,
    imgsz: int = 640,
    conf: float = 0.25,
) -> np.ndarray:
    """Run batch inference and return per-image anomaly areas (mm²)."""
    # results is a list of Results objects, one per image
    results_batch = model(images, imgsz=imgsz, conf=conf)
    n = len(results_batch)
    H, W = images.shape[1], images.shape[2]
    areas = np.zeros(n, dtype=float)

    for i, res in enumerate(results_batch):
        mask = np.zeros((H, W), dtype=np.uint8)
        for idx, cls in enumerate(res.boxes.cls.cpu().numpy().astype(int)):
            if cls in class_ids:
                m = res.masks.data[idx].cpu().numpy().astype(np.uint8)
                mask = np.maximum(mask, m)
        areas[i] = int(mask.sum()) * pixel_size_mm2

    return areas


def visualize_detections(image: np.ndarray, results: Any) -> np.ndarray:
    """Overlay segmentation masks and boxes on the original image."""
    annotated = image.copy()
    return results.plot(annotated)
