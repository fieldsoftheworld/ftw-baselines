import os
from pathlib import Path

import wget

from ftw_tools.settings import CROP_CALENDAR_BASE_URL, CROP_CALENDAR_FILES


def get_crop_calendar_cache_dir() -> Path:
    """Get the cache directory for crop calendar files.

    Returns:
        The cache directory path for crop calendar files.
    """
    cache_base = os.environ.get("FTW_CACHE_DIR")
    if not cache_base:
        cache_base = Path.home() / ".cache" / "ftw-tools"
    else:
        cache_base = Path(cache_base)

    cache_dir = cache_base / "crop_calendar"
    return cache_dir


def ensure_crop_calendar_exists() -> Path:
    """Ensure crop calendar files exist, downloading if necessary.

    Returns:
        The cache directory containing the crop calendar files.
    """
    cache_dir = get_crop_calendar_cache_dir()

    all_files_exist = cache_dir.exists() and all(
        (cache_dir / filename).exists() for filename in CROP_CALENDAR_FILES
    )

    if not all_files_exist:
        print("Downloading crop calendar files (first-time setup)...")
        download_crop_calendar_files()

    return cache_dir


def download_crop_calendar_files(force: bool = False) -> None:
    """Download all crop calendar files.

    Args:
        force: If True, re-download even if files exist.
    """
    cache_dir = get_crop_calendar_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    for filename in CROP_CALENDAR_FILES:
        file_path = cache_dir / filename

        if file_path.exists() and not force:
            continue

        url = CROP_CALENDAR_BASE_URL + filename
        print(f"Downloading {filename}...")

        if file_path.exists():
            file_path.unlink()

        wget.download(url, str(file_path.resolve()))

    print(f"Files cached at {cache_dir}")
