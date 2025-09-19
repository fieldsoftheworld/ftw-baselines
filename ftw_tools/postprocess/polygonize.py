import math
import os
import re
import time
from typing import Optional

import fiona
import fiona.transform
import numpy as np
import rasterio
import rasterio.features
import shapely.geometry
from affine import Affine
try:
    from fiboa_cli.parquet import create_parquet, features_to_dataframe
    FIBOA_AVAILABLE = True
except ImportError:
    FIBOA_AVAILABLE = False
    print("Warning: fiboa_cli.parquet not available. Parquet output will use alternative implementation.")
    
    def create_parquet(gdf, columns, collection, out, config, compression="brotli"):
        import geopandas as gpd
        gdf.to_parquet(out, compression=compression)
    
    def features_to_dataframe(rows, columns):
        import geopandas as gpd
        import pandas as pd
        
        if not rows:
            return gpd.GeoDataFrame(columns=columns)
        
        data = []
        geometries = []
        for row in rows:
            data.append(row['properties'])
            geometries.append(shapely.geometry.shape(row['geometry']))
        
        df = pd.DataFrame(data)
        gdf = gpd.GeoDataFrame(df, geometry=geometries)
        return gdf
from pyproj import CRS, Transformer
from scipy.ndimage import maximum_filter, minimum_filter
from shapely.ops import transform
from tqdm import tqdm

import higra as hg

from ftw_tools.settings import SUPPORTED_POLY_FORMATS_TXT


def InstSegm(extent, boundary, t_ext=0.5, t_bound=0.2):
    extent = np.asarray(extent).squeeze().astype(np.float32)
    boundary = np.asarray(boundary).squeeze().astype(np.float32)

    if extent.shape != boundary.shape:
        raise ValueError(f"extent and boundary must have same shape. Got {extent.shape} vs {boundary.shape}")

    ext_binary = (extent >= t_ext).astype(np.uint8)
    input_hws = boundary.copy()
    input_hws[ext_binary == 0] = 1.0

    size = input_hws.shape[:2]
    graph = hg.get_8_adjacency_graph(size)
    edge_weights = hg.weight_graph(graph, input_hws, hg.WeightFunction.mean)
    tree, altitudes = hg.watershed_hierarchy_by_dynamics(graph, edge_weights)

    instances = hg.labelisation_horizontal_cut_from_threshold(
        tree, altitudes, threshold=t_bound
    ).astype(float)

    instances[ext_binary == 0] = np.nan
    return instances


def get_boundary(mask):
    m = mask.copy()
    m[m == 3] = 0
    field_mask = (m > 0).astype(np.uint8)

    local_max = maximum_filter(m, size=3)
    local_min = minimum_filter(m, size=3)
    boundary = ((local_max != local_min) & (field_mask > 0)).astype(np.float32)
    
    return boundary


def polygonize(
    input,
    out,
    simplify=True,
    min_size=500,
    max_size=None,
    overwrite=False,
    close_interiors=False,
    algorithm="simple",
    t_ext=0.5,
    t_bound=0.2,
):
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
        "geometry": "Polygon",
        "properties": {
            "id": "str",
            "area": "float",
            "perimeter": "float",
        },
    }
    i = 1
    # read the input file as a mask
    with rasterio.open(input) as src:
        original_crs = src.crs.to_string()
        is_meters = src.crs.linear_units in ["m", "metre", "meter"]
        equal_area_crs = CRS.from_epsg(
            6933
        )  # Define the equal-area projection using EPSG:6933
        tags = src.tags()

        input_height, input_width = src.shape
        mask = src.read(1)
        
        if algorithm == "watershed":
            print("Using watershed instance segmentation...")
            extent = (mask == 1).astype(np.float32)
            boundary = get_boundary(mask)
            instances = InstSegm(extent, boundary, t_ext, t_bound)
            unique_instances = np.unique(instances[~np.isnan(instances)])
            print(f"Found {len(unique_instances)} instances")
        else:
            mask = (mask == 1).astype(np.uint8)
            polygonization_stride = 2048
            total_iterations = math.ceil(input_height / polygonization_stride) * math.ceil(
                input_width / polygonization_stride
            )

        if out.endswith(".gpkg"):
            format = "GPKG"
        elif out.endswith(".parquet"):
            format = "Parquet"
        elif out.endswith(".fgb"):
            format = "FlatGeobuf"
        elif out.endswith(".geojson") or out.endswith(".json"):
            format = "GeoJSON"
        elif out.endswith(".ndjson"):
            format = "GeoJSONSeq"
        else:
            raise ValueError(
                "Output format not supported. " + SUPPORTED_POLY_FORMATS_TXT
            )

        is_geojson = format.startswith("GeoJSON")
        if is_geojson:
            epsg4326 = CRS.from_epsg(4326)
            affine = Transformer.from_crs(
                original_crs, epsg4326, always_xy=True
            ).transform

        rows = []
        
        if algorithm == "watershed":
            for instance_id in unique_instances:
                instance_mask = (instances == instance_id).astype(np.uint8)
                
                for geom_geojson, val in rasterio.features.shapes(
                    instance_mask, transform=src.transform
                ):
                    if val != 1:
                        continue
                        
                    geom = shapely.geometry.shape(geom_geojson)
                    
                    if close_interiors:
                        geom = shapely.geometry.Polygon(geom.exterior)
                    if simplify > 0:
                        geom = geom.simplify(simplify)
                    
                    if is_meters:
                        geom_proj_meters = geom
                    else:
                        geom_proj_meters = shapely.geometry.shape(
                            fiona.transform.transform_geom(
                                original_crs, equal_area_crs, geom_geojson
                            )
                        )
                    
                    area = geom_proj_meters.area
                    perimeter = geom_proj_meters.length
                    
                    if area < min_size or (max_size is not None and area > max_size):
                        continue
                    
                    if is_geojson:
                        geom = transform(affine, geom)
                    
                    rows.append({
                        "geometry": shapely.geometry.mapping(geom),
                        "properties": {
                            "id": str(i),
                            "area": area * 0.0001,
                            "perimeter": perimeter,
                        },
                    })
                    i += 1
        else:
            with tqdm(total=total_iterations, desc="Processing mask windows") as pbar:
                for y in range(0, input_height, polygonization_stride):
                    for x in range(0, input_width, polygonization_stride):
                        new_transform = src.transform * Affine.translation(x, y)
                        mask_window = mask[
                            y : y + polygonization_stride, x : x + polygonization_stride
                        ]
                        for geom_geojson, val in rasterio.features.shapes(
                            mask_window, transform=new_transform
                        ):
                            if val != 1:
                                continue

                            geom = shapely.geometry.shape(geom_geojson)

                            if close_interiors:
                                geom = shapely.geometry.Polygon(geom.exterior)
                            if simplify > 0:
                                geom = geom.simplify(simplify)

                            if is_meters:
                                geom_proj_meters = geom
                            else:
                                geom_proj_meters = shapely.geometry.shape(
                                    fiona.transform.transform_geom(
                                        original_crs, equal_area_crs, geom_geojson
                                    )
                                )

                            area = geom_proj_meters.area
                            perimeter = geom_proj_meters.length

                            if area < min_size or (
                                max_size is not None and area > max_size
                            ):
                                continue

                            if is_geojson:
                                geom = transform(affine, geom)

                            rows.append(
                                {
                                    "geometry": shapely.geometry.mapping(geom),
                                    "properties": {
                                        "id": str(i),
                                        "area": area * 0.0001,
                                        "perimeter": perimeter,
                                    },
                                }
                            )
                            i += 1

                        pbar.update(1)

    if format == "Parquet":
        timestamp = tags.get("TIFFTAG_DATETIME", None)
        if timestamp is not None:
            pattern = re.compile(
                r"^(\d{4})[:-](\d{2})[:-](\d{2})[T\s](\d{2}):(\d{2}):(\d{2}).*$"
            )
            if pattern.match(timestamp):
                timestamp = re.sub(pattern, r"\1-\2-\3T\4:\5:\6Z", timestamp)
            else:
                print("WARNING: Unable to parse timestamp from TIFFTAG_DATETIME tag.")
                timestamp = None

        config = collection = {"fiboa_version": "0.2.0"}
        columns = ["geometry", "determination_method"] + list(
            schema["properties"].keys()
        )
        gdf = features_to_dataframe(rows, columns)
        gdf.set_crs(original_crs, inplace=True, allow_override=True)
        gdf["determination_method"] = "auto-imagery"
        if timestamp is not None:
            gdf["determination_datetime"] = timestamp
            columns.append("determination_datetime")

        create_parquet(gdf, columns, collection, out, config, compression="brotli")
    else:
        print(
            "WARNING: The fiboa-compliant GeoParquet output format is recommended for field boundaries."
        )
        if is_geojson:
            original_crs = epsg4326
        with fiona.open(out, "w", format, schema=schema, crs=original_crs) as dst:
            dst.writerecords(rows)

    print(f"Finished polygonizing output at {out} in {time.time() - tic:.2f}s")
