"field processing with lulc"

import logging
import os
from typing import Union

import planetary_computer
import pystac_client
import rasterio as rio
import rioxarray
import xarray as xr
from rasterio import warp

logger = logging.getLogger(__name__)


class RasterLULCFilter:
    """
    Filters raster by LULC class
    """

    LULC_CLASS_IO = 5  # IO LULC class for agriculture
    LULC_CLASS_ESA = 40  # ESA LULC class for agriculture
    LULC_PROVIDER = ["io-lulc-annual-v02", "esa-worldcover"]  # Providers of LULC data

    def __init__(
        self,
        input_path: str,
        output_path: str,
        collection_name: str = "io-lulc-annual-v02",
        save_lulc_tif: bool = False,
    ):
        assert collection_name in self.LULC_PROVIDER, (
            f"Collection name must be one of {self.LULC_PROVIDER}"
        )
        self.LULC_CLASS = (
            self.LULC_CLASS_IO
            if collection_name == "io-lulc-annual-v02"
            else self.LULC_CLASS_ESA
        )
        self.input_path = input_path
        self.catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        # Get bounds from input geotiff file with classification results for fields
        with rio.open(self.input_path) as src:
            self.src_bounds = src.bounds
            self.bounds_4326 = rio.warp.transform_bounds(
                src.crs, "EPSG:4326", *src.bounds
            )
            self.src_crs = src.crs
        lulc = self.load_lulc(collection_name)
        self.filter_raster_by_lulc(self.input_path, lulc, output_path, save_lulc_tif)

    def load_lulc(self, collection_name: str) -> xr.Dataset:
        """
        Loads data for the specified collection.

        First item from collection used for IO LULC couse it is most resent year (2023).
        First item from collection used for ESA LULC because it is the most recent version (200).

        Args:
            collection_name (str): Name of the collection ('io-lulc-annual-v02' or 'esa-worldcover')
        """
        search = self.catalog.search(
            collections=[collection_name], bbox=self.bounds_4326, limit=10
        )
        items = search.item_collection()

        # Define asset key depending on collection
        asset_key = "data" if collection_name == "io-lulc-annual-v02" else "map"

        # Here we get the href of the asset.
        # First item used for IO LULC couse it is most resent year.
        # First item used for ESA LULC because it is the most recent version.
        asset_href = items.items[0].assets[asset_key].href

        # Load data from asset
        ds = rioxarray.open_rasterio(asset_href)
        # Transform bounds to the size of the raster
        minx, miny, maxx, maxy = warp.transform_bounds(
            self.src_crs, ds.rio.crs, *self.src_bounds
        )
        clipped_ds = ds.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)  # type: ignore

        return clipped_ds

    def filter_raster_by_lulc(
        self,
        input_path: str,
        lulc: xr.Dataset,
        output_path: str,
        save_lulc_tif: bool,
    ) -> str:
        """
        Filters raster by LULC class

        Args:
            input_path (str): Path to the input raster
            lulc (xr.Dataset): LULC raster
        """

        # Read Geotiff with classification reults with rioxarray
        inference = rioxarray.open_rasterio(input_path)

        # Reproject LULC to the size of inference
        lulc = lulc.rio.reproject_match(inference)

        # Convert to numpy arrays
        inference_array = inference.values[0]  # Take first channel
        lulc_array = lulc.values[0]  # Take first channel

        # Create mask where LULC is equal to agro class
        lulc_mask = lulc_array == self.LULC_CLASS

        # Optionally save the LULC mask
        if save_lulc_tif:
            lulc_output_path = output_path.split(".")[0] + "-mask.tif"
            with rio.open(
                lulc_output_path,
                "w",
                driver="GTiff",
                height=lulc_mask.shape[0],
                width=lulc_mask.shape[1],
                count=1,
                dtype=lulc.dtype,
                crs=lulc.rio.crs,
                compress="lzw",
                blockxsize=512,
                blockysize=512,
                interleave="band",
                transform=lulc.rio.transform(),
            ) as dst:
                dst.write(lulc_mask, 1)
                print(f"Saved LULC mask to {lulc_output_path}")

        # Create copy of inference array
        inference_modified = inference_array.copy()

        # Set 0 where LULC is not equal to agro class
        inference_modified[~lulc_mask] = 0

        # Save result to new file
        with rio.open(
            output_path,
            "w",
            driver="GTiff",
            height=inference_modified.shape[0],
            width=inference_modified.shape[1],
            count=1,
            dtype=inference_array.dtype,
            crs=inference.rio.crs,
            nodata=0,
            compress="lzw",
            blockxsize=512,
            blockysize=512,
            interleave="band",
            transform=inference.rio.transform(),
        ) as dst:
            dst.write(inference_modified, 1)
        return output_path


def lulc_filtering(
    input: str,
    out: str,
    overwrite: bool = False,
    collection_name: str = "io-lulc-annual-v02",
    save_lulc_tif: bool = False,
) -> Union[str, None]:
    if os.path.exists(out) and not overwrite:
        logger.info(f"Output file {out} already exists. Use -f to overwrite.")
        return None
    elif os.path.exists(out) and overwrite:
        os.remove(out)  # GPKGs are sometimes weird about overwriting in-place
    RasterLULCFilter(
        input_path=input,
        output_path=out,
        collection_name=collection_name,
        save_lulc_tif=save_lulc_tif,
    )
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="""
    Processing fields with filtering by agricultural lands according to LULC data.

    Available collections:
    - io-lulc-annual-v02
    - esa-worldcover

    Usage:
    python lulc_filtering.py --input /path/to/input.tif --out /path/to/output.tif --overwrite False --collection_name io-lulc-annual-v02
    """
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--overwrite", type=bool, required=False, default=False)
    parser.add_argument(
        "--collection_name", type=str, required=False, default="io-lulc-annual-v02"
    )
    parser.add_argument("--save_lulc_tif", type=bool, required=False, default=False)
    args = parser.parse_args()
    input_file = args.input
    out = args.out
    overwrite = args.overwrite
    collection_name = args.collection_name
    save_lulc_tif = args.save_lulc_tif
    output_path = lulc_filtering(
        input=input_file,
        out=out,
        overwrite=overwrite,
        collection_name=collection_name,
        save_lulc_tif=save_lulc_tif,
    )
    print(f"Output path: {output_path}")
