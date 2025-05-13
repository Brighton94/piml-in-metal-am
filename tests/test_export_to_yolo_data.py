"""Tests for src.utils.export_yolo_training_data script."""

import argparse
import logging
import subprocess
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pytest
from src.utils import export_to_yolo_data

TEST_BUILD_KEY = "test_build_01"
DUMMY_HDF5_PATH = "/tmp/dummy_test_data.hdf5"
PROJECT_ROOT_FOR_TESTS = Path(export_to_yolo_data.PROJECT_ROOT)


@pytest.fixture
def mock_config_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock DATASET_PATHS in the config module used by the script."""
    mock_dataset_paths = {
        TEST_BUILD_KEY: Path(DUMMY_HDF5_PATH),
        "another_build": Path("/tmp/another.hdf5"),
    }
    monkeypatch.setattr(export_to_yolo_data, "DATASET_PATHS", mock_dataset_paths)
    monkeypatch.setattr(export_to_yolo_data.config, "DATASET_PATHS", mock_dataset_paths)


@pytest.fixture
def mock_hdf5_file() -> mock.MagicMock:
    """Create a mock HDF5 file object with expected datasets."""
    mock_file = mock.MagicMock(spec=h5py.File)
    mock_file.__enter__.return_value = mock_file

    # Mock camera data
    mock_camera_dataset = mock.MagicMock(spec=h5py.Dataset)
    mock_camera_dataset.shape = (5, 100, 100, 3)  # 5 layers, 100x100 RGB images
    mock_camera_dataset.__getitem__.return_value = np.random.randint(
        0, 256, (100, 100, 3), dtype=np.uint8
    )
    mock_file.__contains__.side_effect = lambda key: key in [
        export_to_yolo_data.CAMERA_PATH,
        export_to_yolo_data.SPATTER_MASK_HDF5_PATH,
        export_to_yolo_data.STREAK_MASK_HDF5_PATH,
        "slices/segmentation_results",  # for _check_mask_paths
    ]
    mock_file[export_to_yolo_data.CAMERA_PATH] = mock_camera_dataset

    # Mock mask data
    mock_spatter_mask_dataset = mock.MagicMock(spec=h5py.Dataset)
    mock_spatter_mask_dataset.__getitem__.return_value = np.random.randint(
        0, 2, (100, 100), dtype=np.uint8
    )
    mock_spatter_mask_dataset.shape = (100, 100)
    mock_file[export_to_yolo_data.SPATTER_MASK_HDF5_PATH] = mock_spatter_mask_dataset

    mock_streak_mask_dataset = mock.MagicMock(spec=h5py.Dataset)
    mock_streak_mask_dataset.__getitem__.return_value = np.random.randint(
        0, 2, (100, 100), dtype=np.uint8
    )
    mock_streak_mask_dataset.shape = (100, 100)
    mock_file[export_to_yolo_data.STREAK_MASK_HDF5_PATH] = mock_streak_mask_dataset

    # Mock segmentation results group
    mock_segmentation_results = mock.MagicMock(spec=h5py.Group)
    mock_segmentation_results.keys.return_value = [
        str(export_to_yolo_data.CLASS_ID_SPATTER),
        str(export_to_yolo_data.CLASS_ID_STREAK),
    ]
    mock_file["slices/segmentation_results"] = mock_segmentation_results

    return mock_file


def test_list_builds(mock_config_paths: None, capsys: pytest.CaptureFixture) -> None:
    """Test the --list_builds functionality."""
    with (
        mock.patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(list_builds=True, build_key=None),
        ),
        pytest.raises(SystemExit) as e,
    ):
        export_to_yolo_data.main()
    assert e.value.code == 0
    captured = capsys.readouterr()
    assert "Available dataset build keys:" in captured.out
    assert f"  - {TEST_BUILD_KEY}" in captured.out
    assert "  - another_build" in captured.out


def test_main_no_build_key_error(
    mock_config_paths: None, capsys: pytest.CaptureFixture
) -> None:
    """Test that an error is raised if --build_key is not provided and --list_builds is not used."""  # noqa: E501
    with (
        mock.patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(list_builds=False, build_key=None),
        ),
        pytest.raises(SystemExit) as e,
    ):
        export_to_yolo_data.main()
    assert e.value.code != 0
    captured = capsys.readouterr()
    assert "the following arguments are required: --build_key" in captured.err


@mock.patch("src.utils.export_yolo_training_data.os.path.exists")
@mock.patch("src.utils.export_yolo_training_data.h5py.File")
@mock.patch("src.utils.export_yolo_training_data.os.makedirs")
@mock.patch("src.utils.export_yolo_training_data.imageio.imwrite")
def test_export_data_for_yolo_success(
    mock_imwrite: mock.MagicMock,
    mock_makedirs: mock.MagicMock,
    mock_h5py_file_constructor: mock.MagicMock,
    mock_os_path_exists: mock.MagicMock,
    mock_config_paths: None,
    mock_hdf5_file: mock.MagicMock,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test successful data export for a valid build key."""
    caplog.set_level(logging.INFO)
    mock_os_path_exists.return_value = True
    mock_h5py_file_constructor.return_value = mock_hdf5_file

    monkeypatch.setattr(export_to_yolo_data, "DATA_PATH", DUMMY_HDF5_PATH)
    img_train_dir = (
        PROJECT_ROOT_FOR_TESTS / "data" / TEST_BUILD_KEY / "images" / "train"
    )
    lbl_train_dir = (
        PROJECT_ROOT_FOR_TESTS / "data" / TEST_BUILD_KEY / "labels" / "train"
    )
    monkeypatch.setattr(export_to_yolo_data, "IMG_TRAIN_DIR", str(img_train_dir))
    monkeypatch.setattr(export_to_yolo_data, "LBL_TRAIN_DIR", str(lbl_train_dir))

    export_to_yolo_data.export_data_for_yolo()

    mock_makedirs.assert_any_call(str(img_train_dir), exist_ok=True)
    mock_makedirs.assert_any_call(str(lbl_train_dir), exist_ok=True)
    mock_h5py_file_constructor.assert_called_once_with(DUMMY_HDF5_PATH, "r")

    num_layers = mock_hdf5_file[export_to_yolo_data.CAMERA_PATH].shape[0]
    assert mock_imwrite.call_count == num_layers * 2

    expected_img_path = str(img_train_dir / "00000.png")
    expected_lbl_path = str(lbl_train_dir / "00000.png")

    image_call_found = any(
        call[0][0] == expected_img_path for call in mock_imwrite.call_args_list
    )
    label_call_found = any(
        call[0][0] == expected_lbl_path for call in mock_imwrite.call_args_list
    )
    assert image_call_found
    assert label_call_found

    assert f"Exporting {num_layers} layers as images and masks..." in caplog.text
    assert "Export complete" in caplog.text
    mock_hdf5_file.close.assert_called_once()


def test_main_invalid_build_key(
    mock_config_paths: None, caplog: pytest.LogCaptureFixture
) -> None:
    """Test main function with an invalid build key."""
    caplog.set_level(logging.ERROR)
    invalid_key = "non_existent_build"
    with (
        mock.patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(list_builds=False, build_key=invalid_key),
        ),
        pytest.raises(SystemExit) as e,
    ):
        export_to_yolo_data.main()
    assert e.value.code == 1
    assert f"Failed to get dataset path for '{invalid_key}'" in caplog.text


@mock.patch("src.utils.export_yolo_training_data.os.path.exists", return_value=False)
def test_export_data_hdf5_not_found(
    mock_os_path_exists: mock.MagicMock,
    mock_config_paths: None,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test export when HDF5 file does not exist."""
    caplog.set_level(logging.ERROR)
    monkeypatch.setattr(export_to_yolo_data, "DATA_PATH", DUMMY_HDF5_PATH)
    monkeypatch.setattr(export_to_yolo_data, "IMG_TRAIN_DIR", "/tmp/img")
    monkeypatch.setattr(export_to_yolo_data, "LBL_TRAIN_DIR", "/tmp/lbl")

    export_to_yolo_data.export_data_for_yolo()
    assert f"HDF5 file not found at {DUMMY_HDF5_PATH}" in caplog.text


@mock.patch("src.utils.export_yolo_training_data.os.path.exists", return_value=True)
@mock.patch("src.utils.export_yolo_training_data.h5py.File")
def test_export_data_hdf5_missing_camera_path(
    mock_h5py_file_constructor: mock.MagicMock,
    mock_os_path_exists: mock.MagicMock,
    mock_config_paths: None,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test export when HDF5 file is missing the camera data path."""
    caplog.set_level(logging.ERROR)
    mock_file = mock.MagicMock(spec=h5py.File)
    mock_file.__enter__.return_value = mock_file
    mock_file.__contains__.return_value = False
    mock_file.keys.return_value = ["some_other_key"]
    mock_h5py_file_constructor.return_value = mock_file

    monkeypatch.setattr(export_to_yolo_data, "DATA_PATH", DUMMY_HDF5_PATH)
    monkeypatch.setattr(export_to_yolo_data, "IMG_TRAIN_DIR", "/tmp/img")
    monkeypatch.setattr(export_to_yolo_data, "LBL_TRAIN_DIR", "/tmp/lbl")

    export_to_yolo_data.export_data_for_yolo()
    assert (
        f"CAMERA_PATH '{export_to_yolo_data.CAMERA_PATH}' not found in HDF5 file."
        in caplog.text
    )
    mock_file.close.assert_called_once()


@mock.patch("src.utils.export_yolo_training_data.os.path.exists", return_value=True)
@mock.patch("src.utils.export_yolo_training_data.h5py.File")
def test_export_data_hdf5_missing_mask_paths(
    mock_h5py_file_constructor: mock.MagicMock,
    mock_os_path_exists: mock.MagicMock,
    mock_config_paths: None,
    mock_hdf5_file: mock.MagicMock,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test export when HDF5 file is missing mask data paths."""
    caplog.set_level(logging.ERROR)

    mock_hdf5_file.__contains__.side_effect = (
        lambda key: key == export_to_yolo_data.CAMERA_PATH
    )
    if "slices/segmentation_results" in mock_hdf5_file:
        del mock_hdf5_file["slices/segmentation_results"]

    mock_h5py_file_constructor.return_value = mock_hdf5_file

    monkeypatch.setattr(export_to_yolo_data, "DATA_PATH", DUMMY_HDF5_PATH)
    monkeypatch.setattr(export_to_yolo_data, "IMG_TRAIN_DIR", "/tmp/img")
    monkeypatch.setattr(export_to_yolo_data, "LBL_TRAIN_DIR", "/tmp/lbl")

    export_to_yolo_data.export_data_for_yolo()
    assert (
        "Neither spatter nor streak mask paths found. Cannot export labels."
        in caplog.text
    )
    assert "Required mask paths not found in HDF5. Aborting export." in caplog.text
    mock_hdf5_file.close.assert_called_once()


def test_script_execution_list_builds(
    mock_config_paths: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test script execution via subprocess for --list_builds."""
    script_path = (
        PROJECT_ROOT_FOR_TESTS / "src" / "utils" / "export_yolo_training_data.py"
    )
    result = subprocess.run(
        ["python", str(script_path), "--list_builds"],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT_FOR_TESTS,
    )
    assert result.returncode == 0
    assert "Available dataset build keys:" in result.stdout
    assert f"  - {TEST_BUILD_KEY}" in result.stdout
