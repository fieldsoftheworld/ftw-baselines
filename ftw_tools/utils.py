import hashlib
import logging
import os

import click
import pandas as pd
import scipy
import scipy.stats
import xarray as xr

from ftw_tools.download.crop_calendar import ensure_crop_calendar_exists
from ftw_tools.settings import CROP_CAL_SUMMER_END, CROP_CAL_SUMMER_START

logger = logging.getLogger()


def compute_md5(file_path: str) -> str | None:
    """Compute the MD5 checksum of a file.

    Args:
        file_path: Path to the file to compute checksum for.

    Returns:
        The MD5 checksum as a hexadecimal string, or None if file not found.
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    except FileNotFoundError:
        return None
    return hash_md5.hexdigest()


def validate_checksums(checksum_file: str, root_directory: str) -> bool:
    """Validate checksums stored in a checksum file.

    Args:
        checksum_file: Path to the checksum file.
        root_directory: Root directory for resolving relative file paths.

    Returns:
        True if all checksums match, False otherwise.
    """
    if not os.path.isfile(checksum_file):
        print(f"Checksum file not found: {checksum_file}")
        return False

    with open(checksum_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 2:
            continue

        stored_checksum, file_path = parts
        file_path = os.path.join(root_directory, file_path)
        current_checksum = compute_md5(file_path)

        if current_checksum != stored_checksum:
            print("Checksum mismatch: {file_path}")
            return False
    return True


def harvest_to_datetime(harvest_day: int, year: int) -> pd.Timestamp:
    """Convert a harvest integer (day of the year) to a datetime object.

    Args:
        harvest_day: Day of the year (1-365).
        year: The year for which the date is to be calculated.

    Returns:
        Corresponding datetime object.
    """
    return pd.to_datetime(f"{year}-{harvest_day}", format="%Y-%j")


def get_harvest_integer_from_bbox(
    bbox: list[float],
    start_year_raster_path: str = None,
    end_year_raster_path: str = None,
) -> list[int]:
    """Gets harvest integer from a user-provided bounding box. Note currently just uses summer crops.

    Args:
        bbox: Bounding box in the format [minx, miny, maxx, maxy].
        start_year_raster_path: Optional path to start of year raster.
        end_year_raster_path: Optional path to end of year raster.

    Returns:
        Start and end harvest integer (day of the year).
    """

    if start_year_raster_path is None:
        cache_dir = ensure_crop_calendar_exists()
        start_year_raster_path = str(cache_dir / CROP_CAL_SUMMER_START)
    if end_year_raster_path is None:
        cache_dir = ensure_crop_calendar_exists()
        end_year_raster_path = str(cache_dir / CROP_CAL_SUMMER_END)

    start_harvest_dset = xr.open_dataset(start_year_raster_path, engine="rasterio")
    end_harvest_dset = xr.open_dataset(end_year_raster_path, engine="rasterio")

    # Clip the datasets to the bounding box
    start_value = start_harvest_dset.rio.clip_box(
        bbox[0], bbox[1], bbox[2], bbox[3], allow_one_dimensional_raster=True
    )

    if len(start_value["band_data"][0][0]) > 1:
        start_days = start_value["band_data"].values[0][0]
        logger.info(
            f"Multiple dates found in area of interest {start_days}. Using circular mean to determine harvest day."
        )
        start_value = int(round(scipy.stats.circmean(start_days, high=365, low=1)))
    else:
        start_value = int(start_value["band_data"].values[0][0][0])

    end_value = end_harvest_dset.rio.clip_box(
        bbox[0], bbox[1], bbox[2], bbox[3], allow_one_dimensional_raster=True
    )
    if len(end_value["band_data"][0][0]) > 1:
        end_days = end_value["band_data"].values[0][0]
        logger.info(
            f"Multiple dates found in area of interest {end_days}. Using circular mean to determine harvest day."
        )
        end_value = int(round(scipy.stats.circmean(end_days, high=365, low=1)))

    else:
        end_value = int(end_value["band_data"].values[0][0][0])

    return [start_value, end_value]


def parse_bbox(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> list[float] | None:
    """Parse and validate a bounding box string.

    Args:
        ctx: Click context (unused but required for callback).
        param: Click parameter (unused but required for callback).
        value: Bounding box string in format 'minx,miny,maxx,maxy'.

    Returns:
        List of four float values [minx, miny, maxx, maxy], or None if value is None.

    Raises:
        click.BadParameter: If the bounding box format is invalid.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise click.BadParameter("Bounding box must be a string")
    values = value.split(",")
    if len(values) != 4:
        raise click.BadParameter("Bounding box must contain exactly 4 values")
    for i, v in enumerate(values):
        try:
            values[i] = float(v)
        except ValueError:
            raise click.BadParameter(
                f"Invalid value for element {i} in bounding box: {v}"
            )
    return values
