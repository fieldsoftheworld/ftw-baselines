import os
import tempfile
import time

import numpy as np
import odc.stac
import planetary_computer as pc
import pystac
import rasterio
import rasterio.features
import rioxarray  # seems unused but is needed

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


def create_input(win_a, win_b, out, overwrite):
    """Main function for creating input for inference."""
    if os.path.exists(out) and not overwrite:
        print("Output file already exists, use -f to overwrite them. Exiting.")
        return

    # Ensure that the base directory exists
    os.makedirs(os.path.dirname(out), exist_ok=True)

    item_win_a = get_item(win_a)
    item_win_b = get_item(win_b)

    # TODO: Check that items are spatially aligned, or implement a way to only download the intersection

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_win_a_fn = os.path.join(tmpdirname, "tmp_win_a.tif")
        tmp_win_b_fn = os.path.join(tmpdirname, "tmp_win_b.tif")

        print("Loading window A data")
        tic = time.time()
        data = odc.stac.stac_load(
            [item_win_a],
            bands=BANDS_OF_INTEREST,
            dtype="uint16",
            resampling="bilinear",
        ).isel(time=0)

        data.rio.to_raster(
            tmp_win_a_fn,
            driver="GTiff",
            dtype="uint16",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        print(f"Finished saving window A to file in {time.time()-tic:0.2f} seconds")

        print("Loading window B data")
        tic = time.time()
        data = odc.stac.stac_load(
            [item_win_b],
            bands=BANDS_OF_INTEREST,
            dtype="uint16",
            resampling="bilinear",
        ).isel(time=0)

        data.rio.to_raster(
            tmp_win_b_fn,
            driver="GTiff",
            dtype="uint16",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        print(f"Finished saving window B to file in {time.time()-tic:0.2f} seconds")

        print("Merging data and writing output")
        tic = time.time()
        with rasterio.open(tmp_win_a_fn) as f:
            profile = f.profile
            data1 = f.read()

        with rasterio.open(tmp_win_b_fn) as f:
            data2 = f.read()

        data = np.concatenate([data1, data2], axis=0)
        profile["count"] = data.shape[0]
        profile["compress"] = "deflate"
        profile["tiled"] = True
        profile["blockxsize"] = 256
        profile["blockysize"] = 256
        profile["BIGTIFF"] = "YES"

        with rasterio.open(out, "w", **profile) as f:
            f.write(data)
        print(f"Finished merging and writing output in {time.time()-tic:0.2f} seconds")