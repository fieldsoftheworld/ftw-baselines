import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from pyproj import CRS, Transformer
from rasterio.transform import from_origin
from shapely.geometry import box

from ftw_tools.inference.utils import (
    metric_crs_for_geographic_bounds,
    postprocess_instance_polygons,
)
from ftw_tools.postprocess.polygonize import polygonize


def write_mask(path: Path, crs: CRS, transform) -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:4, 1:4] = 1  # 900 m² at the nominal 10 m resolution
    mask[5:7, 5:7] = 1  # 400 m²: removed by min_size

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=mask.shape[1],
        height=mask.shape[0],
        count=1,
        dtype=mask.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(mask, 1)


@pytest.mark.parametrize(
    ("crs", "transform"),
    [
        (
            CRS.from_epsg(4326),
            from_origin(
                15.0, 48.0, 10 / (111_320 * math.cos(math.radians(48))), 10 / 111_320
            ),
        ),
        (
            CRS.from_epsg(32633),
            from_origin(
                *Transformer.from_crs(4326, 32633, always_xy=True).transform(15, 48),
                10,
                10,
            ),
        ),
    ],
    ids=["geographic", "projected"],
)
def test_polygonize_uses_metre_scale_and_preserves_crs(
    tmp_path: Path, crs: CRS, transform
) -> None:
    mask_path = tmp_path / "mask.tif"
    output_path = tmp_path / "fields.gpkg"
    write_mask(mask_path, crs, transform)

    polygonize(
        str(mask_path),
        str(output_path),
        simplify=2,
        min_size=500,
        erode_dilate=2,
        polygonization_stride=8,
    )

    fields = gpd.read_file(output_path)
    assert fields.crs == crs
    assert len(fields) == 1
    assert 700 < fields.iloc[0]["metrics:area"] < 1_000
    assert fields.geometry.is_valid.all()


@pytest.mark.parametrize(
    ("bounds", "expected_epsg"),
    [((-10, 84.5, 10, 85), 32661), ((-10, -85, 10, -84.5), 32761)],
)
def test_metric_crs_uses_polar_projection(bounds, expected_epsg) -> None:
    assert metric_crs_for_geographic_bounds(4326, bounds).to_epsg() == expected_epsg


def test_shared_postprocessing_uses_metric_distance_and_restores_crs() -> None:
    polygons = gpd.GeoDataFrame(
        geometry=[box(15, 48, 15.0004, 48.0004)], crs="EPSG:4326"
    )

    result = postprocess_instance_polygons(
        polygons, simplify=2, min_size=500, close_interiors=False
    )

    assert result.crs == polygons.crs
    assert len(result) == 1
    assert result.iloc[0]["metrics:area"] > 500


def test_geographic_geojson_merge_keeps_metric_measurements(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.tif"
    output_path = tmp_path / "fields.geojson"
    transform = from_origin(
        15.0, 48.0, 10 / (111_320 * math.cos(math.radians(48))), 10 / 111_320
    )
    write_mask(mask_path, CRS.from_epsg(4326), transform)

    polygonize(
        str(mask_path),
        str(output_path),
        simplify=0,
        min_size=0,
        merge_adjacent=0.5,
        polygonization_stride=8,
    )

    fields = gpd.read_file(output_path)
    assert fields.crs == CRS.from_epsg(4326)
    assert len(fields) == 2
    assert sorted(fields["metrics:area"]) == pytest.approx([400, 900], rel=0.02)
