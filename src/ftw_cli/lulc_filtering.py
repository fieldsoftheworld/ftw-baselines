"field processing with lulc"

import math
import os
import time
from typing import Union

import geopandas as gpd
import planetary_computer
import pystac
import rasterio as rio
import rioxarray
from loguru import logger
from rasterio.features import shapes
from shapely.geometry import box, shape


class LulcFiltering:
    """Class for processing fields using LULC data."""

    LULC_CLASS = 5

    def __init__(self, fields_path: str, minimal_area_m2: int, lulc_year: int):
        """
        Field processor initialization.

        Args:
            fields_path (str): Path to the field geodata file
        """
        self.lulc_year = lulc_year
        self._fields = gpd.read_file(fields_path)
        self._fields = LulcFiltering.convert_to_utm(self._fields)
        self._fields = self._fields[self._fields.geometry.area > minimal_area_m2]

    @staticmethod
    def convert_to_utm(fields: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Converts geodata to UTM projection.

        Args:
            fields (gpd.GeoDataFrame): Source geodata

        Returns:
            gpd.GeoDataFrame: Geodata in UTM projection
        """
        return fields.to_crs(epsg=fields.estimate_utm_crs().to_epsg())

    def get_lulc(self, lulc_path: str) -> str:
        """
        Gets and processes land use data (LULC) for specified fields.

        Args:
            fields (gpd.GeoDataFrame): Field geodata

        Returns:
            str: Path to saved LULC raster
        """
        fields = self._fields
        # Get fields bbox
        start_time = time.time()
        minx, miny, maxx, maxy = fields.total_bounds

        centroid = box(*fields.to_crs(epsg=4326).total_bounds).centroid
        lon, lat = centroid.x, centroid.y
        zone_number = math.floor((lon + 180) / 6) + 1
        hemisphere = "N" if lat >= 0 else "S"
        utm_zone = f"{zone_number}{'U' if hemisphere == 'N' else 'C'}"

        # Form URL for required zone
        item_url = (
            "https://planetarycomputer.microsoft.com/api/stac/v1/collections"
            f"/io-lulc-annual-v02/items/{utm_zone}-{self.lulc_year}"
        )

        # Load metadata and sign assets
        for _ in range(5):
            try:
                item = pystac.Item.from_file(item_url)
                signed_item = planetary_computer.sign(item)
                break
            except Exception as e:
                logger.error(f"Error loading item: {e}")
                logger.info("Retrying in 1 second...")
                time.sleep(1)

        # Open raster using rioxarray
        asset_href = signed_item.assets["data"].href
        ds = rioxarray.open_rasterio(asset_href)

        # Clip by bbox
        clipped_ds = ds.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)  # type: ignore

        # Optionally: save clipped raster
        clipped_ds.rio.to_raster(lulc_path)
        total_time = time.time() - start_time
        logger.info(f"Total time for getting LULC data: {total_time:.2f} sec")

        return lulc_path

    def get_fields_intersecting_lulc(self, lulc_path: str, output_path: str) -> str:
        """
        Finds fields that intersect with agricultural lands according to LULC data.

        Args:
            fields (gpd.GeoDataFrame): Field geodata
            lulc_path (str): Path to LULC raster
            output_path (str): Path for saving results

        Returns:
            gpd.GeoDataFrame: Filtered fields
        """
        start_time = time.time()
        fields = self._fields
        with rio.open(lulc_path) as src:
            logger.info("Reading LULC raster...")
            lulc = src.read(1)
            transform = src.transform
            crs = src.crs

        logger.info("Creating mask for all geometries...")
        mask = lulc == self.LULC_CLASS
        shapes_gen = shapes(lulc, mask=mask, transform=transform, connectivity=8)

        # Find intersections for all fields at once
        logger.info("Finding intersections...")

        lulc_polygons = gpd.GeoDataFrame(
            geometry=[shape(geom) for geom, value in shapes_gen], crs=crs
        )
        lulc_polygons_utm = lulc_polygons.to_crs(
            epsg=fields.estimate_utm_crs().to_epsg()
        )

        # Filter polygons by area
        mask_area_lulc = lulc_polygons_utm.geometry.area > 1000
        lulc_polygons_utm = lulc_polygons_utm[mask_area_lulc]

        # Spatial join
        logger.info("Filtering fields...")

        filtered_fields = fields.sjoin(lulc_polygons_utm, predicate="intersects")

        # Save result
        assert output_path.endswith(".geojson"), "output_path must end with .geojson"
        filtered_fields.to_file(output_path, driver="GeoJSON")
        total_time = time.time() - start_time
        logger.info(f"Total field processing time: {total_time:.2f} sec")
        logger.info(f"Number of fields: {len(filtered_fields)}")
        return output_path


def lulc_filtering(
    input: str,
    out: str,
    minimal_area_m2: int,
    lulc_year: int = 2023,
    lulc_path: str = "lulc.tif",
    overwrite: bool = False,
) -> Union[str, None]:

    lulc_filter = LulcFiltering(
        fields_path=input, minimal_area_m2=minimal_area_m2, lulc_year=lulc_year
    )

    if os.path.exists(out) and not overwrite:
        logger.info(f"Output file {out} already exists. Use -f to overwrite.")
        return None
    elif os.path.exists(out) and overwrite:
        os.remove(out)  # GPKGs are sometimes weird about overwriting in-place
    if os.path.exists(lulc_path) and not overwrite:
        logger.info(f"LULC file already exists: {lulc_path}")
    else:
        lulc_path = lulc_filter.get_lulc(lulc_path=lulc_path)
    output_path = lulc_filter.get_fields_intersecting_lulc(lulc_path, out)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="""
    Processing fields with filtering by minimal area and intersection with agricultural lands according to LULC data.

    Usage:
    python field_processing.py --fields_path /path/to/fields.geojson --output_path /path/to/output.geojson --minimal_area_m2 1000 --lulc_year 2023 --lulc_path /path/to/lulc.tif
    """
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--minimal_area_m2", type=int, required=False, default=1000)
    parser.add_argument("--lulc_year", type=int, required=False, default=2023)
    parser.add_argument("--lulc_path", type=str, required=False, default="LULC.tif")
    parser.add_argument("--overwrite", type=bool, required=False, default=False)
    args = parser.parse_args()
    input_file = args.input
    out = args.out
    minimal_area_m2 = args.minimal_area_m2
    lulc_year = args.lulc_year
    lulc_path = args.lulc_path
    overwrite = args.overwrite
    output_path = lulc_filtering(
        input=input_file,
        out=out,
        minimal_area_m2=minimal_area_m2,
        lulc_year=lulc_year,
        lulc_path=lulc_path,
        overwrite=overwrite,
    )
