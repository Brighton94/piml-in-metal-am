"""Tests for the configuration module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from src.config import (
    CLASS_ID_SPATTER,
    CLASS_ID_STREAK,
    DATASET_PATHS,
    get_dataset_path,
)


def test_constants():
    """Test that constants are defined correctly."""
    assert CLASS_ID_STREAK == 3
    assert CLASS_ID_SPATTER == 8


def test_environment_variables():
    """Test that environment variables are loaded correctly."""
    # Test default values
    with patch.dict(os.environ, {}, clear=True):
        import importlib

        import src.config

        importlib.reload(src.config)

        assert str(src.config.DATASET_ROOT) == "/data"

    # Test custom values
    with patch.dict(
        os.environ,
        {
            "HOST_DATASET_PATH": "/custom/data",
            "EXTERNAL_DRIVE_PATH": "/custom/external",
        },
    ):
        import importlib

        import src.config

        importlib.reload(src.config)

        assert str(src.config.DATASET_ROOT) == "/custom/data"


def test_dataset_paths_structure():
    """Test the structure of the DATASET_PATHS dictionary."""
    assert "tcr_phase1_build1" in DATASET_PATHS
    assert "tcr_phase1_build2" in DATASET_PATHS

    for _key, paths in DATASET_PATHS.items():
        assert isinstance(paths, list)
        assert len(paths) >= 1
        for path in paths:
            assert isinstance(path, Path)


def test_get_dataset_path_valid_key_file_exists():
    """Test retrieving a dataset path when the file exists."""
    test_key = "tcr_phase1_build1"

    # Mock the Path.exists method to return True for the first path
    with patch.object(Path, "exists", return_value=True):
        # Get fresh dataset paths after mocking
        import importlib

        import src.config

        importlib.reload(src.config)

        result = get_dataset_path(test_key)
        fresh_paths = src.config.DATASET_PATHS

    # Should return the first path in the list for this key
    assert result == fresh_paths[test_key][0]


@pytest.mark.skip(reason="No second path available in config; test disabled")
def test_get_dataset_path_second_path_exists():
    """Test retrieving a dataset path when only the second path exists (skipped)."""
    pass


def test_get_dataset_path_valid_key_no_file():
    """Test retrieving a dataset when no files exist at any path."""
    test_key = "tcr_phase1_build2"

    # Mock the Path.exists method to return False for all paths
    with (
        patch.object(Path, "exists", return_value=False),
        patch("builtins.print") as mock_print,
    ):
        result = get_dataset_path(test_key)

    # Should return None if no file exists
    assert result is None
    # Should print a warning message
    mock_print.assert_called_once()


def test_get_dataset_path_invalid_key():
    """Test retrieving a dataset with an invalid key."""
    test_key = "invalid_key"

    # Should raise KeyError for invalid key
    with pytest.raises(
        KeyError, match=f"Dataset key '{test_key}' not found in configuration"
    ):
        get_dataset_path(test_key)
