import geopandas as gpd
import numpy as np
import pytest
import rasterio
import torch

pytest.importorskip("ultralytics")


def test_delineate_anything():
    from ultralytics.engine.results import Results

    from ftw_tools.models.delineate_anything import DelineateAnything

    device = "cpu"
    model = DelineateAnything(
        model="DelineateAnything-S",
        resize_factor=2,
        max_detections=50,
        iou_threshold=0.5,
        conf_threshold=0.05,
        device=device,
    )

    # test model inference
    with rasterio.open("./tests/data-files/inference-img.tif") as src:
        x = torch.from_numpy(src.read().astype(np.float32)).unsqueeze(0)

    with torch.inference_mode():
        results = model(x)

    assert len(results) == 1
    assert isinstance(results[0], Results)

    # test conversion of results to polygons
    transform = rasterio.Affine(1, 0, 0, 0, -1, 0)
    crs = rasterio.CRS.from_epsg(4326)
    gdf = model.polygonize(results[0], transform, crs)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs == crs
