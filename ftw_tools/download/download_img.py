import logging
import os
import time
from typing import Tuple

import dask.diagnostics.progress
import odc.stac
import pandas as pd
import planetary_computer as pc
import pystac
import pystac_client
import rioxarray  # seems unused but is needed
import xarray as xr
from pystac.extensions.eo import EOExtension as eo
from shapely.geometry import shape
from tenacity import retry, stop_after_attempt, wait_random_exponential

from ftw_tools.settings import BANDS_OF_INTEREST, COLLECTION_ID, MSPC_URL
from ftw_tools.utils import get_harvest_integer_from_bbox, harvest_to_datetime

logger = logging.getLogger()


@retry(wait=wait_random_exponential(max=3), stop=stop_after_attempt(2))
def get_item(id):
    if "/" not in id:
        uri = MSPC_URL + "/collections/" + COLLECTION_ID + "/items/" + id
    else:
        uri = id

    item = pystac.Item.from_file(uri)

    if uri.startswith(MSPC_URL):
        item = pc.sign(item)

    return item


def scene_selection(
    bbox: list[int], year: int, cloud_cover_max: int = 20
) -> Tuple[str, str]:
    """
    Returns sentinel 2 image id for start and end date within +/- 1 week
    of crop calendar indicated dates. If there are multiple images within the date
    range, lowest cloud cover will be returned.

    Args:
        bbox (list[int]): Bounding box in [minx, miny, maxx, maxy] format.
        year (int): Year for filtering scenes.
        cloud_cover_max (int, optional): Maximum allowed cloud cover percentage. Defaults to 20.

    Returns:
        tuple: Sentinel2 image ids to be used as input into the 2 image crop model
    """
    # get crop calendar days
    start_day, end_day = get_harvest_integer_from_bbox(bbox=bbox)

    start_dt = harvest_to_datetime(harvest_day=start_day, year=year)
    end_dt = harvest_to_datetime(
        harvest_day=end_day, year=year + 1 if end_day < start_day else year
    )  # to account for southern hemisphere harvest

    # search for +/- 1 week of the crop calendar indicated start and end days

    win_a = query_stac(bbox=bbox, date=start_dt, cloud_cover_max=cloud_cover_max)
    win_b = query_stac(bbox=bbox, date=end_dt, cloud_cover_max=cloud_cover_max)

    return (win_a, win_b)


def query_stac(bbox: list[int], date: pd.Timestamp, cloud_cover_max: int = 20) -> str:
    """
    Queries Sentinel-2 imagery hosted on planetary computer via pystac.
    sentinel 2 image id for start and end date within +/- 1 week
    of crop calendar indicated dates, with the lowest percent cloud cover is returned.

    Args:
        bbox: Bounding box in [minx, miny, maxx, maxy] format.
        date: crop calendar indicated date
        cloud_cover_max: threshold for maximum percent cloud cover.

    Returns:
        Sentinel-2 image id.
    """
    # make +/- 1 week datetime range to query over
    start = (date - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    end = (date + pd.Timedelta(days=14)).strftime("%Y-%m-%d")

    # Format as string
    date_range = f"{start}/{end}"

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
    )

    search = catalog.search(
        collections=[COLLECTION_ID],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": cloud_cover_max}},
    )

    items = search.item_collection()
    logger.info(f"Returned {len(items)} Items")

    if len(items) == 0:
        raise ValueError(
            f"No sentinel scenes within this area for {date_range} with {cloud_cover_max}"
        )
    # check if aoi spans multiple S2 tiles
    if len(items) > 1:
        s2_tile_ids = [item.properties["s2:mgrs_tile"] for item in items]
        if len(set(s2_tile_ids)) > 1:
            raise ValueError(
                f"Multiple MGRS tiles found: {set(s2_tile_ids)}. Please chose a smaller "
                f"search area, support coming soon for multiple Sentinel 2 scenes."
            )
    # sort by percent cloud cover
    least_cloudy_item = min(items, key=lambda item: eo.ext(item).cloud_cover)

    logger.info(
        f"Choosing {least_cloudy_item.id} from {least_cloudy_item.datetime.date()}"
        f" with {eo.ext(least_cloudy_item).cloud_cover}% cloud cover"
    )
    return least_cloudy_item.id


def create_input(win_a, win_b, out, overwrite, bbox=None):
    """Main function for creating input for inference."""
    out = os.path.abspath(out)
    if os.path.exists(out) and not overwrite:
        print("Output file already exists, use -f to overwrite them. Exiting.")
        return

    # Ensure that the base directory exists
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # Parse the bounding box
    if bbox is not None:
        bbox = list(map(float, bbox.split(",")))

    # Load the items
    identifiers = [win_a, win_b]
    items = []
    timestamp = None
    version = 0
    for i in identifiers:
        item = get_item(i)
        items.append(item)

        datetime = item.datetime or item.start_datetime or item.end_datetime
        if datetime and (not timestamp or datetime > timestamp):
            timestamp = datetime

        proc_version = item.properties.get(
            "processing:version", item.properties.get("s2:processing_baseline", 0)
        )
        try:
            proc_version = float(proc_version)
        except TypeError:
            proc_version = 0
        if proc_version > 0:
            if version > 0 and version != proc_version:
                print(
                    f"Warning: Processing version of imagery differs: {version} vs {proc_version}. Continuing anyway."
                )
            version = proc_version

    shapes = [shape(item.geometry) for item in items]
    # Ensure the images intersect
    if not shapes[0].intersects(shapes[1]):
        print("The provided images do not intersect. Exiting.")
        return

    tic = time.time()
    data = odc.stac.load(
        [items[0], items[1]],
        bands=BANDS_OF_INTEREST,
        dtype="uint16",
        resampling="bilinear",
        bbox=bbox,
        chunks={"x": "auto", "y": "auto"},
    )

    data = (
        data.to_array(dim="band")
        .stack(bands=("time", "band"))
        .drop_vars("band")
        .transpose("bands", "y", "x")
    )

    if version < 3 or version >= 4:
        print(
            f"Processing version {version} unknown or untested (< 3.0 or >= 6.0). Inference quality might decrease."
        )
    if version >= 4:
        print(
            f"Rescaling data to processing version 3.0 from processing version {version}."
        )
        data = (data.astype("int32") - 1000).clip(min=0).astype("uint16")

    print("Writing output")
    with dask.diagnostics.progress.ProgressBar():
        data.rio.to_raster(
            out,
            driver="GTiff",
            compress="deflate",
            dtype="uint16",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            tags={"TIFFTAG_DATETIME": timestamp.strftime("%Y:%m:%d %H:%M:%S")},
        )

    print(f"Finished merging and writing output in {time.time() - tic:0.2f} seconds")
