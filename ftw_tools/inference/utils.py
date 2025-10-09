import re

import geopandas as gpd
import numpy as np
import shapely
from fiboa_cli.parquet import create_parquet


def merge_polygons(
    polygons: gpd.GeoDataFrame,
    iou_thresh: float = 0.2,
    contain_thresh: float = 0.8,
    grid_size: float = 10.0,
) -> gpd.GeoDataFrame:
    """Merge overlapping polygons in a GeoDataFrame.

    Args:
        polygons: The GeoDataFrame containing the polygons.
        iou_thresh: The IoU threshold for merging polygons.
        contain_thresh: The containment threshold for merging polygons.
        grid_size: The grid size for merging polygons. Defaults to 10m for Sentinel-2.

    Returns:
        The merged GeoDataFrame.
    """
    gdf = polygons.copy()
    gdf["geometry"] = gdf.geometry.apply(shapely.make_valid)

    # Select pairs of polygons that intersect or contained within each other
    pairs = gpd.sjoin(
        gpd.GeoDataFrame(geometry=gdf.geometry, crs=gdf.crs),
        gdf[["geometry"]],
        how="inner",
        predicate="intersects",
    )
    left = pairs.index.to_numpy(copy=False)
    right = pairs["index_right"].to_numpy(copy=False)
    m = left < right
    left, right = left[m], right[m]
    if left.size == 0:
        out = gpd.GeoDataFrame(geometry=gdf.geometry.copy(), crs=gdf.crs)
        return _to_polygons_only(out)

    # Compute overlap and containment metrics
    geoms = gdf.geometry.to_numpy()
    L, R = geoms[left], geoms[right]
    inter = shapely.intersection(L, R)
    inter_a = shapely.area(inter)
    a = shapely.area(L)
    b = shapely.area(R)
    union_a = a + b - inter_a
    iou = inter_a / np.maximum(union_a, 1e-12)
    containment = inter_a / np.maximum(np.minimum(a, b), 1e-12)

    # Filter pairs by threshold
    keep = (iou >= iou_thresh) | (containment >= contain_thresh)
    left, right = left[keep], right[keep]

    # Merge polygons
    n = len(gdf)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(left, right):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pj] = pi
    labels = np.fromiter((find(i) for i in range(n)), int, count=n)
    _, comp_ids = np.unique(labels, return_inverse=True)
    gdf["_merge_id"] = comp_ids
    merged = (
        gdf.groupby("_merge_id", group_keys=False)["geometry"]
        .apply(lambda s: s.union_all(method="unary", grid_size=grid_size))
        .reset_index(drop=True)
    )
    merged = gpd.GeoDataFrame(geometry=merged, crs=gdf.crs)
    merged["geometry"] = merged.geometry.apply(shapely.make_valid)

    # Convert merged multipolygons or geometry collections to polygons
    merged = merged.explode(index_parts=False, ignore_index=True)
    merged = merged.explode(index_parts=False, ignore_index=True)
    merged = merged[merged.geometry.geom_type == "Polygon"].reset_index(drop=True)
    merged = merged[merged.geometry.is_valid & ~merged.geometry.is_empty]
    return merged


def postprocess_instance_polygons(
    polygons: gpd.GeoDataFrame,
    simplify: int = 0,
    padding: int = 0,
    min_size: int | None = None,
    max_size: int | None = None,
    close_interiors: bool = True,
    overlap_iou_threshold: float = 0.2,
    overlap_contain_threshold: float = 0.8,
) -> gpd.GeoDataFrame:
    """Postprocess polygons to remove small polygons, simplify them, and compute area and perimeter.

    Args:
        polygons: The polygons to postprocess.
        simplify: The simplification factor.
        padding: The padding to used for inference.
        min_size: The minimum size of the polygons.
        max_size: The maximum size of the polygons.
        close_interiors: Whether to close the interiors of the polygons.
        overlap_iou_threshold: The overlap IoU threshold for merging polygons.
        overlap_contain_threshold: The overlap containment threshold for merging polygons.

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

    if padding > 0 and (overlap_iou_threshold > 0 or overlap_contain_threshold > 0):
        print(
            f"Merging polygons with overlap IoU threshold {overlap_iou_threshold} and overlap containment threshold {overlap_contain_threshold}. This may take awhile for a large number of polygons..."
        )
        polygons = merge_polygons(
            polygons,
            iou_thresh=overlap_iou_threshold,
            contain_thresh=overlap_contain_threshold,
        )

    # Convert to hectares
    polygons["area"] = polygons.geometry.area * 0.0001
    polygons["perimeter"] = polygons.geometry.length

    # Convert back to original CRS
    polygons.to_crs(src_crs, inplace=True)

    polygons.reset_index(drop=True, inplace=True)
    polygons["id"] = polygons.index + 1

    return polygons


def convert_to_fiboa(
    polygons: gpd.GeoDataFrame, output: str, timestamp: str | None
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
