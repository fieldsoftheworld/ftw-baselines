import geopandas as gpd
import pytest
import rasterio
import torch

pytest.importorskip("ultralytics")


def test_delineate_anything():
    import ultralytics

    from ftw.models.delineate_anything import DelineateAnything

    device = "cpu"
    model = DelineateAnything(
        model="DelineateAnything-S",
        image_size=(320, 320),
        max_detections=50,
        iou_threshold=0.6,
        conf_threshold=0.1,
        device=device,
    )

    # test model inference
    x = torch.randn(2, 3, 256, 256, requires_grad=False, device=device)
    with torch.inference_mode():
        results = model(x)

    assert len(results) == 2
    assert isinstance(results[0], ultralytics.engine.results.Results)

    # test conversion of results to polygons
    transform = rasterio.Affine(1, 0, 0, 0, -1, 0)
    crs = rasterio.CRS.from_epsg(4326)
    gdf = model.polygonize(results[0], transform, crs)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs == crs
