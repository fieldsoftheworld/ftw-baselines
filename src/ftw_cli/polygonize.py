import math
import os
import re
import time

import fiona
import fiona.transform
import numpy as np
import rasterio
import rasterio.features
import shapely.geometry
from affine import Affine
from fiboa_cli.parquet import create_parquet, features_to_dataframe
from pyproj import CRS
from tqdm import tqdm

from .cfg import SUPPORTED_POLY_FORMATS_TXT


def polygonize(input, out, simplify, min_size, overwrite, close_interiors):
    """Polygonize the output from inference."""

    print(f"Polygonizing input file: {input}")

    # TODO: Get this warning working right, based on the CRS of the input file
    # if simplify is not None and simplify > 1:
    #    print("WARNING: You are passing a value of `simplify` greater than 1 for a geographic coordinate system. This is probably **not** what you want.")

    if not out:
        out = os.path.splitext(input)[0] + ".parquet"

    if os.path.exists(out) and not overwrite:
        print(f"Output file {out} already exists. Use -f to overwrite.")
        return
    elif os.path.exists(out) and overwrite:
        os.remove(out)  # GPKGs are sometimes weird about overwriting in-place

    tic = time.time()
    rows = []
    schema = {
        'geometry': 'Polygon',
        'properties': {
            'id': 'str',
            'area': 'float',
            'perimeter': 'float',
        }
    }
    i = 1
    # read the input file as a mask
    with rasterio.open(input) as src:
        original_crs = src.crs.to_string()
        is_meters = src.crs.linear_units in ["m", "metre", "meter"]
        transform = src.transform
        equal_area_crs = CRS.from_epsg(6933) # Define the equal-area projection using EPSG:6933
        tags = src.tags()

        input_height, input_width = src.shape
        mask = (src.read(1) == 1).astype(np.uint8)
        polygonization_stride = 2048
        total_iterations = math.ceil(input_height / polygonization_stride) * math.ceil(input_width / polygonization_stride)

        if out.endswith(".gpkg"):
            format = "GPKG"
        elif out.endswith(".parquet"):
            format = "Parquet"
        elif out.endswith(".fgb"):
            format = "FlatGeobuf"
        elif out.endswith(".geojson") or out.endswith(".json"):
            format = "GeoJSON"
        else:
            raise ValueError("Output format not supported. " + SUPPORTED_POLY_FORMATS_TXT)

        rows = []
        with tqdm(total=total_iterations, desc="Processing mask windows") as pbar:
            for y in range(0, input_height, polygonization_stride):
                for x in range(0, input_width, polygonization_stride):
                    new_transform = transform * Affine.translation(x, y)
                    mask_window = mask[y:y+polygonization_stride, x:x+polygonization_stride]
                    for geom_geojson, val in rasterio.features.shapes(mask_window, transform=new_transform):
                        if val != 1:
                            continue
                            
                        geom = shapely.geometry.shape(geom_geojson)

                        if close_interiors:
                            geom = shapely.geometry.Polygon(geom.exterior)
                        if simplify > 0:
                            geom = geom.simplify(simplify)
                        
                        # Calculate the area of the reprojected geometry
                        if is_meters:
                            geom_proj_meters = geom
                        else:
                            # Reproject the geometry to the equal-area projection
                            # if the CRS is not in meters
                            geom_proj_meters = shapely.geometry.shape(
                                fiona.transform.transform_geom(
                                    original_crs, equal_area_crs, geom_geojson
                                )
                            )

                        area = geom_proj_meters.area
                        perimeter = geom_proj_meters.length
                        
                        # Only include geometries that meet the minimum size requirement
                        if area < min_size:
                            continue

                        rows.append({
                            "geometry": shapely.geometry.mapping(geom),
                            "properties": {
                                "id": str(i),
                                "area": area * 0.0001, # Add the area in hectares
                                "perimeter": perimeter, # Add the perimeter in meters
                            }
                        })
                        i += 1
                    
                    pbar.update(1)

    if format == "Parquet":
        timestamp = tags.get("TIFFTAG_DATETIME", None)
        if timestamp is not None:
            pattern = re.compile(r"^(\d{4})[:-](\d{2})[:-](\d{2})[T\s](\d{2}):(\d{2}):(\d{2}).*$")
            if pattern.match(timestamp):
                timestamp = re.sub(pattern, r"\1-\2-\3T\4:\5:\6Z", timestamp)
            else:
                print("WARNING: Unable to parse timestamp from TIFFTAG_DATETIME tag.")
                timestamp = None
    
        config = collection = {"fiboa_version": "0.2.0"}
        columns = ["geometry", "determination_method"] + list(schema["properties"].keys())
        gdf = features_to_dataframe(rows, columns)
        gdf.set_crs(original_crs, inplace=True, allow_override=True)
        gdf["determination_method"] = "auto-imagery"
        if timestamp is not None:
            gdf["determination_datetime"] = timestamp
            columns.append("determination_datetime")
        
        create_parquet(gdf, columns, collection, out, config, compression = "brotli")
    else:
        print("WARNING: The fiboa-compliant GeoParquet output format is recommended for field boundaries.")
        with fiona.open(out, 'w', format, schema=schema, crs=original_crs) as dst:
            dst.writerecords(rows)

    print(f"Finished polygonizing output at {out} in {time.time() - tic:.2f}s")
