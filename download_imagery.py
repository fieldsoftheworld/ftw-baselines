# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Script for downloading and stacking a pair of S2 scenes for inference."""

import argparse
import os
import time
import tempfile

import pystac
import planetary_computer as pc
import odc.stac
import rasterio
import numpy as np
import rioxarray


def get_parser() -> argparse.ArgumentParser:
    """Creates argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--win_a",
        type=str,
        help="Path to a Sentinel-2 STAC item for the window A image",
    )
    parser.add_argument(
        "--win_b",
        type=str,
        help="Path to a Sentinel-2 STAC item for the window B image",
    )
    parser.add_argument(
        "--output_fn",
        type=str,
        help="Filename to save results to",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrites the outputs if they exist"
    )

    return parser


def main(args) -> None:
    """Main function for the download_imagery.py script."""
    if os.path.exists(args.output_fn) and not args.overwrite:
        print(
            "Output file already exists, use --overwrite to overwrite them."
            + " Exiting."
        )
        return

    # Ensure that the base directory exists
    os.makedirs(os.path.dirname(args.output_fn), exist_ok=True)

    BANDS_OF_INTEREST = ["B04", "B03", "B02", "B08"]

    item_win_a = pc.sign(pystac.Item.from_file(args.win_a))
    item_win_b = pc.sign(pystac.Item.from_file(args.win_b))

    # TODO: Check that items are spatially aligned, or implement a way to only download the intersection

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_win_a_fn = os.path.join(tmpdirname, "tmp_win_a.tif")
        tmp_win_b_fn = os.path.join(tmpdirname, "tmp_win_b.tif")

        print("Loading window A data")
        tic = time.time()
        # TODO: Can we just load both items at the same time?
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

        with rasterio.open(args.output_fn, "w", **profile) as f:
            f.write(data)
        print(f"Finished merging and writing output in {time.time()-tic:0.2f} seconds")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
