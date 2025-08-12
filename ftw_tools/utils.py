import hashlib
import logging
import os

import numpy as np
import pandas as pd
import scipy
import scipy.stats
import xarray as xr

# Harvest day raster paths from https://github.com/ucg-uv/research_products/tree/main
SUMMER_START_RASTER_PATH = "assets/global_crop_calendar/sc_sos_3x3_v2_cog.tiff"
SUMMER_END_RASTER_PATH = "assets/global_crop_calendar/sc_eos_3x3_v2_cog.tiff"

logger = logging.getLogger()


def compute_md5(file_path):
    """Compute the MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    except FileNotFoundError:
        return None
    return hash_md5.hexdigest()


def validate_checksums(checksum_file, root_directory):
    """Validate checksums stored in a checksum file."""
    if not os.path.isfile(checksum_file):
        print(f"Checksum file not found: {checksum_file}")
        return

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
    """
    Convert a harvest integer (day of the year) to a datetime object.

    Args:
        harvest_day (int): Day of the year (1-365).
        year (int): The year for which the date is to be calculated.

    Returns:
        pd.Timestamp: Corresponding datetime object.
    """
    return pd.to_datetime(f"{year}-{harvest_day}", format="%Y-%j")


# to-do func to get harvest integer from user provided bbox
def get_harvest_integer_from_bbox(
    bbox: list[int],
    start_year_raster_path: str = SUMMER_START_RASTER_PATH,
    end_year_raster_path: str = SUMMER_END_RASTER_PATH,
) -> list[int]:
    """
    Gets harvest integer from a user-provided bounding box. Note currently just uses summer crops.

    Args:
        bbox (str): Bounding box in the format 'minx,miny,maxx,maxy'.

    Returns:
        list: start and end harvest integer (day of the year).
    """

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
