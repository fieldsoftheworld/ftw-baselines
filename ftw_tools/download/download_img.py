import os
import time
from urllib.parse import urlparse

import dask.diagnostics.progress
import odc.stac
import pystac
import rioxarray  # seems unused but is needed
import xarray as xr
from shapely.geometry import shape
from tenacity import retry, stop_after_attempt, wait_random_exponential

from ftw_tools.settings import AWS_SENTINEL_URL, BANDS_OF_INTEREST


@retry(wait=wait_random_exponential(max=3), stop=stop_after_attempt(2))
def get_item(id: str) -> pystac.Item:
    """Get a STAC item from a given ID or S3 URL."""

    if "s3://" in id:
        parsed = urlparse(id)
        path = parsed.path.strip("/")  # remove leading slash
        item_id = os.path.basename(path)

        uri = f"{AWS_SENTINEL_URL}/{path}/{item_id}.json"

    elif "/" not in id:
        # Convert ID into full URL

        parts = id.split("_")
        mgrs_tile = parts[1]
        date_str = parts[2]

        utm_zone = mgrs_tile[:2]
        lat_band = mgrs_tile[2]
        grid_square = mgrs_tile[3:]

        year = date_str[:4]
        month = date_str[4:6]

        uri = (
            f"{AWS_SENTINEL_URL}/"
            f"sentinel-s2-l2a-cogs/{utm_zone}/{lat_band}/{grid_square}/"
            f"{year}/{month}/{id}/{id}.json"
        )

    else:
        uri = id

    item = pystac.Item.from_file(uri)

    return item


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
                print("Processing version of imagery differs. Exiting.")
                return
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
