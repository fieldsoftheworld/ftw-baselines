import os
from pathlib import Path

import wget

CROP_CALENDAR_BASE_URL = "https://github.com/fieldsoftheworld/ftw-baselines/raw/main/assets/global_crop_calendar/"
CROP_CALENDAR_FILES = [
    "sc_sos_3x3_v2_cog.tiff",
    "sc_eos_3x3_v2_cog.tiff",
]


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
        success = download_crop_calendar_files()
        if not success:
            raise RuntimeError("Failed to download crop calendar files")

    return cache_dir


def download_crop_calendar_files(force: bool = False) -> bool:
    """Download all crop calendar files.

    Args:
        force: If True, re-download even if files exist.

    Returns:
        True if successful, False otherwise.
    """
    cache_dir = get_crop_calendar_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
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
        return True

    except Exception as e:
        print(f"Error downloading crop calendar files: {e}")
        return False
