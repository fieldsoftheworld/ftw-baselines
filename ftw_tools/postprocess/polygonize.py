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
from pyproj import CRS, Transformer
from shapely.ops import transform, unary_union
from skimage.morphology import dilation, erosion
from tqdm import tqdm

from ftw_tools.settings import SUPPORTED_POLY_FORMATS_TXT


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def merge_adjacent_polygons(features, ratio):
    """Merge polygons when they overlap or touch sufficiently."""
    print(f"Merging polygons that overlap at least {ratio * 100}%")
    geoms, ids = [], []
    for f in features:
        g = shapely.geometry.shape(f["geometry"])
        if g.is_empty:
            continue
        geoms.append(g)
        ids.append(str(f.get("properties", {}).get("id", "")))

    n = len(geoms)
    if n == 0:
        return []

    perims = [g.length for g in geoms]
    bboxes = [g.bounds for g in geoms]  # (minx, miny, maxx, maxy)

    uf = UnionFind(n)
    for i in range(n):
        minx_i, miny_i, maxx_i, maxy_i = bboxes[i]
        gi = geoms[i]
        for j in range(i + 1, n):
            minx_j, miny_j, maxx_j, maxy_j = bboxes[j]
            # quick bbox reject
            if maxx_i < minx_j or maxx_j < minx_i or maxy_i < miny_j or maxy_j < miny_i:
                continue

            gj = geoms[j]
            if not gi.intersects(gj):
                continue

            inter = gi.intersection(gj)
            # if overlapping (positive area), then merge
            if getattr(inter, "area", 0.0) > 0.0:
                uf.union(i, j)
                continue

            # if touching, then check shared boundary ratio
            shared = gi.boundary.intersection(gj.boundary).length
            if shared > 0:
                mperim = min(perims[i], perims[j])
                if (mperim > 0 and (shared / mperim) >= ratio) or (
                    ratio == 0 and gi.touches(gj)
                ):
                    uf.union(i, j)

    # Group by connected components
    comps = {}
    for k in range(n):
        r = uf.find(k)
        comps.setdefault(r, []).append(k)

    # Dissolve components (assumes union is a Polygon)
    out = []
    for idxs in comps.values():
        u = unary_union([geoms[k] for k in idxs])
        props = {
            "id": ",".join([ids[k] for k in idxs if ids[k]]),
            "area": float(u.area),
            "perimeter": float(u.length),
        }
        out.append({"geometry": shapely.geometry.mapping(u), "properties": props})

    return out


def zhang_suen_thinning(img: np.ndarray, max_iters: int = 0) -> np.ndarray:
    """Zhang-Suen thinning on a binary image.

    Old school!
    Zhang, Tongjie Y., and Ching Y. Suen. "A fast parallel algorithm for thinning digital patterns." Communications of the ACM 27, no. 3 (1984): 236-239.

    Args:
        img (np.ndarray): 2D binary image (nonzero pixels are foreground).
        max_iters (int): If > 0, cap the number of full iterations (each has 2 sub-steps).
            If 0 (default), run until convergence.

    Returns:
        np.ndarray: Thinned binary image of same shape and dtype as input.
    """
    if img.ndim != 2:
        raise ValueError("img must be a 2D array")

    # Work in uint8 {0,1} with a 1-pixel zero pad to avoid wraparound.
    I = (img != 0).astype(np.uint8)
    I = np.pad(I, 1, mode="constant")

    def neighbors_views(A):
        # Using the standard Zhang–Suen neighbor naming:
        #   p9 p2 p3
        #   p8 p1 p4
        #   p7 p6 p5
        P2 = A[:-2, 1:-1]
        P3 = A[:-2, 2:]
        P4 = A[1:-1, 2:]
        P5 = A[2:, 2:]
        P6 = A[2:, 1:-1]
        P7 = A[2:, :-2]
        P8 = A[1:-1, :-2]
        P9 = A[:-2, :-2]
        C = A[1:-1, 1:-1]  # center p1
        return C, (P2, P3, P4, P5, P6, P7, P8, P9)

    def B_count(Ps):
        # B(p1) = number of non-zero neighbors
        return sum(Ps)

    def A_count(Ps):
        # A(p1) = number of 0->1 transitions in sequence p2,p3,...,p9,p2
        P2, P3, P4, P5, P6, P7, P8, P9 = Ps
        terms = [
            ((P2 == 0) & (P3 == 1)).astype(np.uint8),
            ((P3 == 0) & (P4 == 1)).astype(np.uint8),
            ((P4 == 0) & (P5 == 1)).astype(np.uint8),
            ((P5 == 0) & (P6 == 1)).astype(np.uint8),
            ((P6 == 0) & (P7 == 1)).astype(np.uint8),
            ((P7 == 0) & (P8 == 1)).astype(np.uint8),
            ((P8 == 0) & (P9 == 1)).astype(np.uint8),
            ((P9 == 0) & (P2 == 1)).astype(np.uint8),
        ]
        out = terms[0]
        for t in terms[1:]:
            out = out + t
        return out

    changed = True
    iters = 0
    while changed and (max_iters == 0 or iters < max_iters):
        changed = False

        # ----- Sub-iteration 1 -----
        C, Ps = neighbors_views(I)
        B = B_count(Ps)
        A = A_count(Ps)
        P2, P3, P4, P5, P6, P7, P8, P9 = Ps

        m1 = (
            (C == 1)
            & (B >= 2)
            & (B <= 6)
            & (A == 1)
            & ((P2 * P4 * P6) == 0)
            & ((P4 * P6 * P8) == 0)
        )
        if np.any(m1):
            C[m1] = 0
            changed = True

        # ----- Sub-iteration 2 -----
        C, Ps = neighbors_views(I)  # recompute after deletion
        B = B_count(Ps)
        A = A_count(Ps)
        P2, P3, P4, P5, P6, P7, P8, P9 = Ps

        m2 = (
            (C == 1)
            & (B >= 2)
            & (B <= 6)
            & (A == 1)
            & ((P2 * P4 * P8) == 0)
            & ((P2 * P6 * P8) == 0)
        )
        if np.any(m2):
            C[m2] = 0
            changed = True

        iters += 1

    # Remove padding, cast back to original dtype
    out = I[1:-1, 1:-1].astype(img.dtype)
    return out


def thin_boundary_preserving_fields(
    field_mask: np.ndarray, boundary_mask: np.ndarray, max_iters: int = 0
):
    """Thin class-2 boundary pixels (~1-pixel skeleton) while preserving connectivity.
    Class-1 field pixels are left unchanged.

    Args:
        field_mask (array-like): 2D array (H, W) with class labels {0,1} where
            0 = background, 1 = field.
        boundary_mask (array-like): 2D array (H, W) with class labels {0,1} where
            0 = background, 1 = boundary to be thinned.
        max_iter (int | None): Maximum number of Zhang–Suen full iterations
            (each has 2 sub-steps). If None or 0, run until convergence.
    """
    field_mask = np.asarray(field_mask)
    out = field_mask.copy()

    # prevent a frame line at image edges
    boundary_mask[[0, -1], :] = 0
    boundary_mask[:, [0, -1]] = 0

    # Skeletonize/thin the boundary
    skel = zhang_suen_thinning(boundary_mask, max_iters=max_iters)
    skel = skel.astype(np.bool_)
    assert skel.shape == field_mask.shape

    out[boundary_mask == 1] = 1
    out[skel] = 0

    field_mask = (out == 1).astype(np.uint8)

    return field_mask


def multi_erosion(mask, iterations):
    for _ in range(iterations):
        kernel = [
            [0, 1, 0],  # image morphology filter
            [1, 1, 1],
            [0, 1, 0],
        ]
        mask = erosion(mask, np.array(kernel))
    return mask


def multi_dilation(mask, iterations):
    for _ in range(iterations):
        kernel = [
            [0, 1, 0],  # image morphology filter
            [1, 1, 1],
            [0, 1, 0],
        ]
        mask = dilation(mask, np.array(kernel))
    return mask


def polygonize(
    input,
    out,
    simplify=True,
    min_size=500,
    max_size=None,
    overwrite=False,
    close_interiors=False,
    polygonization_stride=2048,
    softmax_threshold=None,
    merge_adjacent=None,
    erode_dilate=0,
    dilate_erode=0,
    erode_dilate_raster=0,
    dilate_erode_raster=0,
    thin_boundaries=False,
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
        "properties": {"id": "str", "area": "float", "perimeter": "float"},
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

        if softmax_threshold:
            assert src.count == 3, (
                "Input tif should have 3 bands (background, interior, boundary scores)."
            )
            # softmax scores were quantized to [0,255], so convert threshold similarly
            softmax_threshold *= 255
            # 1st channel: scores for background class, 2nd: interior, 3rd: boundary
            mask = (src.read(2) >= softmax_threshold).astype(np.uint8)
            boundary_mask = (src.read(3) >= softmax_threshold).astype(np.uint8)
        else:
            assert src.count == 1, "Input tif should be single-band (predicted class)."
            mask = (src.read(1) == 1).astype(np.uint8)
            boundary_mask = (src.read(1) == 2).astype(np.uint8)

        if thin_boundaries:
            mask = thin_boundary_preserving_fields(mask, boundary_mask)

        if erode_dilate_raster > 0:
            # morphological opening before polygonization
            mask = multi_erosion(mask, erode_dilate_raster)
            mask = multi_dilation(mask, erode_dilate_raster)
        elif dilate_erode_raster > 0:
            # morphological closing before polygonization
            mask = multi_dilation(mask, dilate_erode_raster)
            mask = multi_erosion(mask, dilate_erode_raster)

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

                        if erode_dilate > 0:
                            # morphological opening
                            geom = geom.buffer(
                                -erode_dilate,
                                join_style=shapely.geometry.JOIN_STYLE.mitre,
                            ).buffer(
                                +erode_dilate,
                                join_style=shapely.geometry.JOIN_STYLE.mitre,
                            )
                        if dilate_erode > 0:
                            # morphological closing
                            geom = geom.buffer(
                                +dilate_erode,
                                join_style=shapely.geometry.JOIN_STYLE.mitre,
                            ).buffer(
                                -dilate_erode,
                                join_style=shapely.geometry.JOIN_STYLE.mitre,
                            )
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
                        if area < min_size or (
                            max_size is not None and area > max_size
                        ):
                            continue

                        if is_geojson:
                            geom = transform(affine, geom)

                        # explode MultiPolygons if needed
                        if isinstance(geom, shapely.geometry.MultiPolygon):
                            for g in geom.geoms:
                                rows.append(
                                    {
                                        "geometry": shapely.geometry.mapping(g),
                                        "properties": {
                                            "id": str(i),
                                            "area": g.area
                                            * 0.0001,  # Add the area in hectares
                                            "perimeter": g.length,  # Add the perimeter in meters
                                        },
                                    }
                                )
                                i += 1
                        else:
                            rows.append(
                                {
                                    "geometry": shapely.geometry.mapping(geom),
                                    "properties": {
                                        "id": str(i),
                                        "area": area
                                        * 0.0001,  # Add the area in hectares
                                        "perimeter": perimeter,  # Add the perimeter in meters
                                    },
                                }
                            )
                            i += 1

                    pbar.update(1)

    # Merge adjacent polygons
    if merge_adjacent != None:
        rows = merge_adjacent_polygons(rows, merge_adjacent)

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
