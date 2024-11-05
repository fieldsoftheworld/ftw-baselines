import os
import time

import odc.stac
import planetary_computer as pc
import pystac
import rioxarray  # seems unused but is needed
import xarray as xr
from tqdm.auto import tqdm
from shapely.geometry import shape

from .cfg import BANDS_OF_INTEREST, COLLECTION_ID, MSPC_URL


def get_item(id):
    if "/" not in id:
        uri = MSPC_URL + "/collections/" + COLLECTION_ID + "/items/" + id
    else:
        uri = id
    
    item = pystac.Item.from_file(uri)

    if uri.startswith(MSPC_URL):
        item = pc.sign(item)

    return item


def create_input(win_a, win_b, out, overwrite, bbox = None):
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
    item_win_a = get_item(win_a)
    item_win_b = get_item(win_b)

    # Ensure the images intersect
    geometry1 = shape(item_win_a.geometry)
    geometry2 = shape(item_win_b.geometry)
    if not geometry1.intersects(geometry2):
        print("The provided images do not intersect. Exiting.")
        return

    # Get the latest timestamp
    timestamps = list(filter(lambda x: x is not None, [item_win_a.datetime, item_win_b.datetime]))
    timestamp = max(timestamps) if len(timestamps) > 0 else None

    print("Loading data")
    tic = time.time()
    data = odc.stac.load(
        [item_win_a, item_win_b],
        bands=BANDS_OF_INTEREST,
        dtype="uint16",
        resampling="bilinear",
        bbox=bbox,
        progress=tqdm
    )

    print("Merging data")
    data = data.to_array(dim="band").stack(bands=("time", "band")).drop_vars("band").transpose('bands', 'y', 'x')

    print("Writing output")
    data.rio.to_raster(
        out,
        driver="GTiff",
        compress="deflate",
        dtype="uint16",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        tags={
            "TIFFTAG_DATETIME": timestamp.strftime("%Y:%m:%d %H:%M:%S")
        }
    )

    print(f"Finished merging and writing output in {time.time()-tic:0.2f} seconds")