# tests/test_export_yolo_training_data.py

import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import src.config as config_mod
from src.utils.export_to_yolo_data import (
    CAMERA_PATH,
    export,
    export_layers,
    prepare_output_directories,
    validate_and_get_data_path,
    validate_hdf5_file,
)

# Ensure project src is on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

TEST_BUILD_KEY = "test_build"
INVALID_KEY = "invalid_build"


@pytest.fixture(autouse=True)
def tmp_dataset(tmp_path, monkeypatch):
    """Create a dummy HDF5 file."""
    # Build dummy HDF5
    h5_path = tmp_path / "dummy.hdf5"
    with h5py.File(str(h5_path), "w") as h5:
        # camera data: 3 layers of 5×5 grayscale
        h5.create_dataset(
            CAMERA_PATH, data=np.random.randint(0, 255, (3, 5, 5), dtype=np.uint8)
        )
        # mask channels
        grp = h5.require_group("slices/segmentation_results")
        grp.create_dataset("8", data=np.random.randint(0, 2, (3, 5, 5), dtype=np.uint8))
        grp.create_dataset("3", data=np.random.randint(0, 2, (3, 5, 5), dtype=np.uint8))

    # Monkeypatch config to point to that file
    monkeypatch.setattr(config_mod, "DATASETS", {TEST_BUILD_KEY: str(h5_path)})
    return h5_path


def test_validate_and_get_data_path_success(tmp_dataset):
    path = validate_and_get_data_path(TEST_BUILD_KEY)
    assert isinstance(path, Path)
    assert path.exists()
    assert str(path).endswith("dummy.hdf5")


def test_validate_and_get_data_path_invalid():
    result = validate_and_get_data_path(INVALID_KEY)
    assert result is None


def test_prepare_output_directories(tmp_path):
    out_root = tmp_path / "out"
    # call, without existing dirs
    out_dir, img_tr, img_val, lbl_tr, lbl_val = prepare_output_directories(
        "foo", out_root
    )
    # All dirs should now exist
    for d in (img_tr, img_val, lbl_tr, lbl_val):
        assert d.exists()
        assert d.is_dir()
    # out_dir should be out_root / build_key
    assert out_dir == out_root / "foo"


def test_validate_hdf5_file_success(tmp_dataset):
    with h5py.File(str(tmp_dataset), "r") as h5:
        img_ds, mask_dss, h, w = validate_hdf5_file(h5, tmp_dataset)
    # camera dataset and two masks
    assert img_ds.shape == (3, 5, 5)
    assert set(mask_dss.keys()) == {"spatter", "streak"}
    # height and width
    assert (h, w) == (5, 5)


def test_validate_hdf5_file_missing_camera(tmp_dataset):
    with h5py.File(str(tmp_dataset), "r") as h5:
        # remove camera path
        del h5[CAMERA_PATH]
        img_ds, mask_dss, h, w = validate_hdf5_file(h5, tmp_dataset)
    assert img_ds is None
    assert mask_dss is None


def test_validate_hdf5_file_missing_masks(tmp_dataset):
    with h5py.File(str(tmp_dataset), "r") as h5:
        # remove segmentation_results group
        del h5["slices/segmentation_results"]
        img_ds, mask_dss, h, w = validate_hdf5_file(h5, tmp_dataset)
    assert img_ds is None
    assert mask_dss is None


def test_export_layers_creates_files(tmp_path, tmp_dataset, caplog):
    """Test that export_layers writes the correct number of image and label files."""
    # Prepare directories
    out_root = tmp_path / "out"
    img_tr = out_root / "images" / "train"
    img_val = out_root / "images" / "val"
    lbl_tr = out_root / "labels" / "train"
    lbl_val = out_root / "labels" / "val"
    for d in (img_tr, img_val, lbl_tr, lbl_val):
        d.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(tmp_dataset), "r") as h5:
        img_ds, mask_dss, h, w = validate_hdf5_file(h5, tmp_dataset)
        caplog.set_level(logging.INFO)
        # export with val_split=0.5
        export_layers(
            img_ds=img_ds,
            mask_dss=mask_dss,
            h=h,
            w=w,
            val_split=0.5,
            workers=2,
            img_train=img_tr,
            img_val=img_val,
            lbl_train=lbl_tr,
            lbl_val=lbl_val,
        )
    # We had 3 layers total, so train+val count = 3
    img_train_files = list(img_tr.glob("*.jpg"))
    img_val_files = list(img_val.glob("*.jpg"))
    label_train = list(lbl_tr.glob("*.txt"))
    label_val = list(lbl_val.glob("*.txt"))
    assert len(img_train_files) + len(img_val_files) == 3
    assert len(label_train) + len(label_val) == 3
    # Check log message
    assert "Exporting 3 layers" in caplog.text


def test_export_integration(tmp_path, tmp_dataset, caplog, monkeypatch):
    """High-level integration: call export() & verify directories under DEFAULT_OUT."""
    caplog.set_level(logging.INFO)
    export(TEST_BUILD_KEY, val_split=0.0, workers=1, out_root=tmp_path)
    root = tmp_path / TEST_BUILD_KEY
    # train dirs should exist
    assert (root / "images" / "train").exists()
    assert (root / "labels" / "train").exists()
    # three jpg + three txt
    imgs = list((root / "images" / "train").glob("*.jpg"))
    txts = list((root / "labels" / "train").glob("*.txt"))
    assert len(imgs) == 3
    assert len(txts) == 3
    # Done log
    assert "Done – images & labels stored in" in caplog.text
