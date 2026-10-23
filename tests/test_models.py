from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import torch

pytest.importorskip("ultralytics")


def test_delineate_anything_loads_from_hub_cache(tmp_path: Path):
    from ftw_tools.inference.models import DelineateAnything

    url = DelineateAnything.checkpoints["DelineateAnything-S"]
    expected = tmp_path / "checkpoints" / url.rsplit("/", 1)[-1]

    def fake_download(url, dst, progress=True):
        Path(dst).write_bytes(b"weights")

    with (
        patch("torch.hub.get_dir", return_value=str(tmp_path)),
        patch("torch.hub.download_url_to_file", side_effect=fake_download) as dl,
        patch("ultralytics.YOLO") as yolo,
    ):
        DelineateAnything(model="DelineateAnything-S", device="cpu")
        DelineateAnything(model="DelineateAnything-S", device="cpu")

    assert dl.call_count == 1
    assert [c.args[0] for c in yolo.call_args_list] == [str(expected)] * 2


def test_delineate_anything():
    from ultralytics.engine.results import Results

    from ftw_tools.inference.models import DelineateAnything

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
