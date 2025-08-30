import re

import geopandas as gpd
import numpy as np
import shapely
from fiboa_cli.parquet import create_parquet


def merge_polygons(
    polygons: gpd.GeoDataFrame,
    iou_thresh: float = 0.2,
    grid_size: float = 5.0,
) -> gpd.GeoDataFrame:
    """Merge overlapping polygons in a GeoDataFrame while keeping singletons.

    Args:
        polygons: The input GeoDataFrame containing the polygons.
        iou_thresh: Merge polygons with IoU greater than this threshold.
        grid_size: The grid size for snapping the polygons.

    Returns:
        The merged polygons.
    """
    gdf = polygons.copy()
    gdf["geometry"] = gdf.geometry.apply(shapely.make_valid)
    gdf["geometry"] = gdf.geometry.set_precision(grid_size)  # snap helps edge seams

    # self-join to get candidate duplicate polygons
    pairs = (
        gpd.sjoin(
            gdf[["geometry"]], gdf[["geometry"]], predicate="overlaps", how="inner"
        )
        .reset_index()
        .rename(columns={"index": "left", "index_right": "right"})
    )
    pairs = pairs[pairs["left"] < pairs["right"]][["left", "right"]]

    # IoU filter
    if len(pairs):
        L = gdf.geometry.values[pairs["left"].values]
        R = gdf.geometry.values[pairs["right"].values]
        inter_a = np.fromiter((a.intersection(b).area for a, b in zip(L, R)), float)
        area_a = np.fromiter((a.area for a in L), float)
        area_b = np.fromiter((b.area for b in R), float)
        union_a = area_a + area_b - inter_a
        iou = inter_a / np.maximum(union_a, 1e-12)
        pairs = pairs[iou >= iou_thresh]

    # Connected components over edges (singletons remain their own component)
    n = len(gdf)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def unite(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pj] = pi

    for i, j in pairs.itertuples(index=False):
        unite(i, j)

    labels = [find(i) for i in range(n)]
    gdf["_merge_id"] = gpd.pd.Series(labels).astype("category").cat.codes

    # Union per component; singletons just pass through
    merged_geom = gdf.groupby("_merge_id", group_keys=False)["geometry"].apply(
        lambda s: s.union_all(method="unary", grid_size=None)
    )
    out = gpd.GeoDataFrame(geometry=merged_geom, crs=gdf.crs).reset_index(drop=True)
    out["geometry"] = out.geometry.apply(shapely.make_valid)
    out = out[out.geometry.is_valid & ~out.geometry.is_empty]

    # Convert from GeometryCollection and/or MultiPolygon to Polygons only
    # drop any other geometry types
    out = out.explode(index_parts=False, ignore_index=True)
    out = out.explode(index_parts=False, ignore_index=True)
    out = out[out.geometry.geom_type == "Polygon"].reset_index(drop=True)
    out = out[out.geometry.is_valid & ~out.geometry.is_empty]
    return out


def postprocess_instance_polygons(
    polygons: gpd.GeoDataFrame,
    simplify: int = 0,
    min_size: int | None = None,
    max_size: int | None = None,
    close_interiors: bool = True,
    overlap_threshold: float = 0.2,
) -> gpd.GeoDataFrame:
    """Postprocess polygons to remove small polygons, simplify them, and compute area and perimeter.

    Args:
        polygons: The polygons to postprocess.
        simplify: The simplification factor.
        min_size: The minimum size of the polygons.
        max_size: The maximum size of the polygons.
        close_interiors: Whether to close the interiors of the polygons.

    Returns:
        The postprocessed polygons.
    """
    # Convert polygons to a meter based CRS
    src_crs = polygons.crs
    polygons.to_crs("EPSG:6933", inplace=True)

    if close_interiors:
        polygons.geometry = polygons.geometry.exterior
        polygons.geometry = polygons.geometry.apply(
            lambda x: shapely.geometry.Polygon(x)
        )

    if simplify > 0:
        polygons.geometry = polygons.geometry.simplify(simplify)

    if min_size is not None:
        polygons.loc[polygons.geometry.area <= min_size, "geometry"] = None
    if max_size is not None:
        polygons.loc[polygons.geometry.area >= max_size, "geometry"] = None
    polygons.dropna(subset=["geometry"], inplace=True)
    polygons.reset_index(drop=True, inplace=True)

    if overlap_threshold > 0:
        polygons = merge_polygons(polygons, iou_thresh=overlap_threshold)

    # Convert to hectares
    polygons["area"] = polygons.geometry.area * 0.0001
    polygons["perimeter"] = polygons.geometry.length

    # Convert back to original CRS
    polygons.to_crs(src_crs, inplace=True)

    polygons.reset_index(drop=True, inplace=True)
    polygons["id"] = polygons.index + 1

    return polygons


def convert_to_fiboa(
    polygons: gpd.GeoDataFrame,
    output: str,
    timestamp: str | None,
) -> gpd.GeoDataFrame:
    """Convert polygons to fiboa parquet format.

    Args:
        polygons: The polygons to convert.
        output: The output file path.
        timestamp: The timestamp of the image.

    Returns:
        The converted polygons.
    """
    polygons["determination_method"] = "auto-imagery"

    config = collection = {"fiboa_version": "0.2.0"}
    columns = ["id", "area", "perimeter", "determination_method", "geometry"]

    if timestamp is not None:
        pattern = re.compile(
            r"^(\d{4})[:-](\d{2})[:-](\d{2})[T\s](\d{2}):(\d{2}):(\d{2}).*$"
        )
        if pattern.match(timestamp):
            timestamp = re.sub(pattern, r"\1-\2-\3T\4:\5:\6Z", timestamp)
            polygons["determination_datetime"] = timestamp
            columns.append("determination_datetime")
        else:
            print("WARNING: Unable to parse timestamp from TIFFTAG_DATETIME tag.")

    create_parquet(polygons, columns, collection, output, config, compression="brotli")
