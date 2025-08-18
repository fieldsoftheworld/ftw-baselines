import os
import time

import dask.diagnostics.progress
import odc.stac
import planetary_computer as pc
import pystac
import rioxarray  # seems unused but is needed
import xarray as xr
from shapely.geometry import shape
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .cfg import BANDS_OF_INTEREST, COLLECTION_ID, MSPC_URL


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
            f"Processing version {version} unknown or untested (< 3.0 or >= 4.0). Inference quality might decrease."
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
