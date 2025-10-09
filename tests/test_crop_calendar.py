import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ftw_tools.download.crop_calendar import (
    CROP_CALENDAR_FILES,
    download_crop_calendar_files,
    ensure_crop_calendar_exists,
    get_crop_calendar_cache_dir,
)


# Using flat fixture instead of class to match existing test pattern
@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp()
    original_cache_dir = os.environ.get("FTW_CACHE_DIR")
    os.environ["FTW_CACHE_DIR"] = temp_dir

    yield temp_dir

    if original_cache_dir:
        os.environ["FTW_CACHE_DIR"] = original_cache_dir
    else:
        os.environ.pop("FTW_CACHE_DIR", None)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_crop_calendar_download(temp_cache_dir):
    """Test download of crop calendar files from GitHub."""
    cache_dir = get_crop_calendar_cache_dir()

    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    download_crop_calendar_files()

    assert cache_dir.exists()

    for filename in CROP_CALENDAR_FILES:
        file_path = cache_dir / filename
        assert file_path.exists()
        assert file_path.stat().st_size > 0


def test_ensure_crop_calendar_exists_uses_cached_files(temp_cache_dir):
    """Test that existing files are not re-downloaded."""
    cache_dir = get_crop_calendar_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    for filename in CROP_CALENDAR_FILES:
        (cache_dir / filename).touch()

    with patch(
        "ftw_tools.download.crop_calendar.download_crop_calendar_files"
    ) as mock_download:
        result = ensure_crop_calendar_exists()

        mock_download.assert_not_called()
        assert result == cache_dir


def test_force_redownload_integration(temp_cache_dir):
    """Test that force=True actually re-downloads files."""
    cache_dir = get_crop_calendar_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    for filename in CROP_CALENDAR_FILES:
        file_path = cache_dir / filename
        file_path.write_text("dummy content")

    original_sizes = {
        filename: (cache_dir / filename).stat().st_size
        for filename in CROP_CALENDAR_FILES
    }

    download_crop_calendar_files(force=True)

    for filename in CROP_CALENDAR_FILES:
        new_size = (cache_dir / filename).stat().st_size
        assert new_size > original_sizes[filename]


def test_get_crop_calendar_cache_dir(temp_cache_dir):
    """Test cache directory path generation."""
    cache_dir = get_crop_calendar_cache_dir()
    assert str(cache_dir).startswith(temp_cache_dir)
    assert "crop_calendar" in str(cache_dir)


def test_custom_cache_directory():
    """Test using a custom cache directory."""
    temp_dir = tempfile.mkdtemp()
    custom_dir = Path(temp_dir) / "custom_cache"

    try:
        with patch.dict(os.environ, {"FTW_CACHE_DIR": str(custom_dir)}):
            cache_dir = get_crop_calendar_cache_dir()

            assert str(cache_dir).startswith(str(custom_dir))
            assert "crop_calendar" in str(cache_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@patch("ftw_tools.download.crop_calendar.wget.download")
def test_download_crop_calendar_files_failure(mock_wget, temp_cache_dir):
    """Test handling of download failure."""
    mock_wget.side_effect = Exception("Network error")

    with pytest.raises(Exception, match="Network error"):
        download_crop_calendar_files()
