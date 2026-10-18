import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pystac
import pytest
import rasterio
import torch
from click.testing import CliRunner
from torchgeo.models import FCN
from vecorel_cli.vecorel import schemas

from ftw_tools.cli import (
    ftw_inference_all,
    inference_download,
    inference_polygonize,
    inference_run,
    inference_run_instance_segmentation,
    inference_run_instance_segmentation_all,
    scene_selection,
)
from ftw_tools.inference.models import DelineateAnything
from ftw_tools.settings import EARTHSEARCH_URL, MSPC_URL


@pytest.fixture(scope="session")
def mock_crop_calendar_downloads():
    with patch(
        "ftw_tools.download.crop_calendar.get_crop_calendar_cache_dir",
        return_value=Path(__file__).parent / "data-files" / "crop-calendar",
    ):
        yield


@pytest.fixture(scope="session")
def field_checkpoint(tmp_path_factory):
    hparams = {"model": "fcn", "in_channels": 8, "num_classes": 3, "num_filters": 8}
    model = FCN(in_channels=8, classes=3, num_filters=8)
    state_dict = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    state_dict["last.bias"] = torch.tensor([0.0, 1.0, 0.0])
    path = tmp_path_factory.mktemp("model") / "field.ckpt"
    torch.save(
        {
            "hyper_parameters": hparams,
            "state_dict": {f"model.{k}": v for k, v in state_dict.items()},
        },
        path,
    )
    return path


@pytest.fixture
def recorded_stac(monkeypatch):
    recordings = {
        url: json.loads(Path(f"./tests/data-files/stac/{name}.json").read_text())
        for name, url in [("earthsearch", EARTHSEARCH_URL), ("mspc", MSPC_URL)]
    }

    class Catalog:
        def __init__(self, host):
            self.searches = recordings[host]

        def search(self, datetime, **kwargs):
            items = [pystac.Item.from_dict(d) for d in self.searches[datetime]]
            return SimpleNamespace(item_collection=lambda: pystac.ItemCollection(items))

    monkeypatch.setattr("pystac_client.Client.open", Catalog)
    return recordings


@pytest.fixture
def recorded_items(monkeypatch, recorded_stac, tmp_path):
    items = {
        d["id"]: d
        for searches in recorded_stac.values()
        for results in searches.values()
        for d in results
    }

    def from_file(href, *args, **kwargs):
        item = pystac.Item.from_dict(items[href.rsplit("/", 1)[-1]])
        crs = item.properties["proj:code"]
        for asset in item.assets.values():
            path = tmp_path / f"{crs.replace(':', '_')}.tif"
            if not path.exists():
                height, width = asset.extra_fields["proj:shape"]
                transform = rasterio.Affine(*asset.extra_fields["proj:transform"][:6])
                with rasterio.open(
                    path,
                    "w",
                    driver="GTiff",
                    width=width,
                    height=height,
                    count=1,
                    dtype="uint16",
                    crs=crs,
                    transform=transform,
                    tiled=True,
                    sparse_ok=True,
                ):
                    pass
            asset.href = path.as_uri()
        return item

    monkeypatch.setattr(pystac.Item, "from_file", from_file)


@pytest.fixture
def recorded_schemas(monkeypatch):
    load_file = schemas.load_file

    def local_file(uri):
        return load_file(Path("./tests/data-files/schemas") / uri.split("://", 1)[1])

    monkeypatch.setattr(schemas, "load_file", local_file)


@pytest.fixture(scope="session")
def yolo_checkpoint(tmp_path_factory):
    ultralytics = pytest.importorskip("ultralytics")
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = ultralytics.YOLO("yolo11n-seg.yaml")
    for branch in model.model.model[-1].cv3:
        branch[-1].bias.data.fill_(10.0)
    path = tmp_path_factory.mktemp("yolo") / "yolo11n-seg.pt"
    model.save(path)
    return path


@pytest.fixture
def local_delineate_anything(monkeypatch, yolo_checkpoint):
    monkeypatch.setitem(
        DelineateAnything.checkpoints, "DelineateAnything-S", str(yolo_checkpoint)
    )


@pytest.mark.integration
def test_scene_selection_earthsearch():
    runner = CliRunner()

    result = runner.invoke(
        scene_selection,
        [
            "--bbox=-93.68708939,41.9530844,-93.64078526,41.98070608",
            "--year=2023",
            "--cloud_cover_max=20",
            "--buffer_days=14",
            "--stac_host=earthsearch",
        ],
    )
    assert result.exit_code == 0, (
        f"Exited with {result.exit_code}. Output: {result.stdout} {result.stderr}"
    )
    assert "S2A_T15TVG_20230406T170330_L2A" in result.output  # window a
    assert "S2A_T15TVG_20231112T170622_L2A" in result.output  # window b


@pytest.mark.integration
def test_scene_selection_mspc():
    runner = CliRunner()

    result = runner.invoke(
        scene_selection,
        [
            "--bbox=-93.68708939,41.9530844,-93.64078526,41.98070608",
            "--year=2022",
            "--cloud_cover_max=20",
            "--buffer_days=14",
            "--stac_host=mspc",
        ],
    )
    assert result.exit_code == 0, (
        f"Exited with {result.exit_code}. Output: {result.stdout} {result.stderr}"
    )
    assert (
        "S2A_MSIL2A_20220315T171051_R112_T15TVG_20220316T083617" in result.output
    )  # window a
    assert (
        "S2B_MSIL2A_20221125T171639_R112_T15TVG_20221202T081730" in result.output
    )  # window b


@pytest.mark.parametrize(
    ("stac_host", "year", "win_a", "win_b"),
    [
        (
            "earthsearch",
            2023,
            "S2A_T15TVG_20230406T170330_L2A",
            "S2A_T15TVG_20231112T170622_L2A",
        ),
        (
            "mspc",
            2022,
            "S2A_MSIL2A_20220315T171051_R112_T15TVG_20220316T083617",
            "S2B_MSIL2A_20221125T171639_R112_T15TVG_20221202T081730",
        ),
    ],
)
def test_scene_selection_recorded(
    stac_host, year, win_a, win_b, recorded_stac, mock_crop_calendar_downloads
):
    args = [
        "--bbox=-93.68708939,41.9530844,-93.64078526,41.98070608",
        f"--year={year}",
        "--cloud_cover_max=20",
        "--buffer_days=14",
        f"--stac_host={stac_host}",
    ]
    result = CliRunner().invoke(scene_selection, args)
    assert result.exit_code == 0, result.output
    assert win_a in result.output
    assert win_b in result.output


@pytest.mark.integration
def test_inference_download_via_earthsearch(tmp_path: Path):
    runner = CliRunner()

    # Download imagery by ID from EarthSearch
    inference_image = tmp_path / "austria_example.tif"
    result = runner.invoke(
        inference_download,
        [
            "--win_a=S2B_33UUP_20210617_1_L2A",
            "--win_b=S2B_33UUP_20210925_1_L2A",
            "--bbox=13.0,48.0,13.2,48.2",
            "-o",
            str(inference_image),
            "-f",
            "--stac_host=earthsearch",
        ],
    )
    assert result.exit_code == 0, (
        f"Exited with {result.exit_code}. Output: {result.stdout} {result.stderr}"
    )
    assert "Writing output" in result.output
    assert "Finished merging and writing output" in result.output
    assert os.path.exists(str(inference_image))


@pytest.mark.integration
def test_inference_download_via_mspc(tmp_path: Path):
    runner = CliRunner()

    # Download imagery by ID from Microsoft Planetary Computer
    inference_image = tmp_path / "austria_example.tif"
    args = [
        "--win_a=S2B_MSIL2A_20210617T100559_R022_T33UUP_20210624T063729",
        "--win_b=S2B_MSIL2A_20210925T101019_R022_T33UUP_20210926T121923",
        "--bbox=13.0,48.0,13.2,48.2",
        "-o",
        str(inference_image),
        "-f",
        "--stac_host=mspc",
    ]
    result = runner.invoke(inference_download, args)
    assert result.exit_code == 0, (
        f"Exited with {result.exit_code}. Output: {result.stdout} {result.stderr}"
    )
    assert "Writing output" in result.output
    assert "Finished merging and writing output" in result.output
    assert os.path.exists(str(inference_image))


@pytest.mark.integration
def test_inference_download_single(tmp_path: Path):
    runner = CliRunner()

    # Download imagery by ID from Microsoft Planetary Computer
    inference_image = tmp_path / "austria_example.tif"
    args = [
        "--win_a=S2B_MSIL2A_20210617T100559_R022_T33UUP_20210624T063729",
        "--bbox=13.0,48.0,13.2,48.2",
        "-o",
        str(inference_image),
        "-f",
        "--stac_host=mspc",
    ]
    result = runner.invoke(inference_download, args)
    assert result.exit_code == 0, result.output
    assert "Writing output" in result.output
    assert "Finished merging and writing output" in result.output
    assert os.path.exists(str(inference_image))


@pytest.mark.parametrize(
    ("stac_host", "windows"),
    [
        (
            "earthsearch",
            [
                f"--win_a={EARTHSEARCH_URL}/collections/sentinel-2-c1-l2a/items/S2A_T15TVG_20230406T170330_L2A",
                f"--win_b={EARTHSEARCH_URL}/collections/sentinel-2-c1-l2a/items/S2A_T15TVG_20231112T170622_L2A",
            ],
        ),
        ("mspc", ["--win_a=S2A_MSIL2A_20220315T171051_R112_T15TVG_20220316T083617"]),
    ],
)
def test_inference_download_recorded(
    tmp_path: Path, stac_host, windows, recorded_items
):
    out = tmp_path / "image.tif"
    args = [
        *windows,
        "--bbox=-93.68708939,41.9530844,-93.64078526,41.98070608",
        "-o",
        str(out),
        "-f",
        f"--stac_host={stac_host}",
    ]
    result = CliRunner().invoke(inference_download, args)
    assert result.exit_code == 0, result.output
    assert "Finished merging and writing output" in result.output
    assert out.exists()


@pytest.mark.integration
def test_inference_run(tmp_path: Path):
    runner = CliRunner()

    # Check required files are present
    inf_input_path = "./tests/data-files/inference-img.tif"
    assert os.path.exists(inf_input_path)

    # Run inference
    inf_output_path = tmp_path / "austria_example_output_full.tif"
    args = [
        inf_input_path,
        "--model",
        "FTW_v1_3_Class_FULL",
        "--out",
        str(inf_output_path),
        "--gpu",
        "0",
        "--overwrite",
        "--resize_factor",
        "2",
        "--overwrite",
    ]
    result = runner.invoke(inference_run, args)
    assert result.exit_code == 0, (
        f"Exited with {result.exit_code}. Output: {result.stdout} {result.stderr}"
    )
    assert "Finished inference and saved output" in result.output
    assert os.path.exists(str(inf_output_path))


def test_inference_run_local_checkpoint(tmp_path: Path, field_checkpoint):
    out = tmp_path / "output.tif"
    args = [
        "./tests/data-files/inference-img.tif",
        "--model",
        str(field_checkpoint),
        "--out",
        str(out),
        "--gpu",
        "0",
        "--resize_factor",
        "2",
        "--overwrite",
    ]
    result = CliRunner().invoke(inference_run, args)
    assert result.exit_code == 0, result.output
    assert "Finished inference and saved output" in result.output
    assert out.exists()


def test_inference_polygonize(tmp_path: Path):
    runner = CliRunner()

    # Check required files are present
    mask = "./tests/data-files/mask.tif"
    assert os.path.exists(mask)

    # Polygonize the file
    out_path = tmp_path / "polygons.gpkg"
    result = runner.invoke(inference_polygonize, [mask, "-o", str(out_path), "-f"])
    assert result.exit_code == 0, (
        f"Exited with {result.exit_code}. Output: {result.stdout} {result.stderr}"
    )
    assert "Polygonizing input file:" in result.output
    assert "Finished polygonizing output" in result.output
    assert os.path.exists(str(out_path))


@pytest.mark.integration
def test_ftw_inference_all(tmp_path: Path, mock_crop_calendar_downloads):
    runner = CliRunner()
    out_path = tmp_path / "inference_output"
    args = [
        "--bbox=13.0,48.0,13.2,48.2",
        "--year=2024",
        f"--out={str(out_path)}",
        "--cloud_cover_max=20",
        "--buffer_days=14",
        "--model=FTW_v1_3_Class_FULL",
        "--resize_factor=2",
        "--overwrite",
    ]
    result = runner.invoke(ftw_inference_all, args)
    assert result.exit_code == 0, (
        f"Exited with {result.exit_code}. Output: {result.stdout} {result.stderr}"
    )
    assert "Finished inference and saved output" in result.output
    assert os.path.exists(str(out_path))


def test_ftw_inference_all_recorded(
    tmp_path: Path,
    recorded_items,
    recorded_schemas,
    field_checkpoint,
    mock_crop_calendar_downloads,
):
    out = tmp_path / "inference_output"
    args = [
        "--bbox=-93.68708939,41.9530844,-93.64078526,41.98070608",
        "--year=2022",
        f"--out={out}",
        "--cloud_cover_max=20",
        "--buffer_days=14",
        f"--model={field_checkpoint}",
        "--resize_factor=2",
        "--overwrite",
    ]
    result = CliRunner().invoke(ftw_inference_all, args)
    assert result.exit_code == 0, result.output
    assert "Finished inference and saved output" in result.output
    assert (out / "polygons.parquet").exists()


@pytest.mark.integration
def test_instance_segmentation_inference(tmp_path: Path):
    runner = CliRunner()

    # Check required files are present
    inf_input_path = "./tests/data-files/inference-img.tif"
    assert os.path.exists(inf_input_path)

    # Run inference
    inf_output_path = tmp_path / "austria_example_output_full.parquet"
    args = [
        inf_input_path,
        "--model",
        "DelineateAnything-S",
        "--out",
        str(inf_output_path),
        "--gpu",
        "0",
        "--overwrite",
    ]
    result = runner.invoke(inference_run_instance_segmentation, args)
    assert result.exit_code == 0, result.output
    assert os.path.exists(str(inf_output_path))


@pytest.mark.integration
def test_instance_segmentation_inference_all(tmp_path: Path):
    runner = CliRunner()

    out_dir = tmp_path / "inference_output"
    downloaded_path = out_dir / "inference_data.tif"
    output_path = out_dir / "inference_output.parquet"

    args = [
        "S2B_MSIL2A_20210617T100559_R022_T33UUP_20210624T063729",
        "--bbox=13.0,48.0,13.2,48.2",
        "--model",
        "DelineateAnything-S",
        "--out_dir",
        str(out_dir),
        "--gpu",
        "0",
        "--overwrite",
    ]
    result = runner.invoke(inference_run_instance_segmentation_all, args)
    assert result.exit_code == 0, result.output
    assert os.path.exists(str(downloaded_path))
    assert os.path.exists(str(output_path))


def test_instance_segmentation_inference_local_model(
    tmp_path: Path, local_delineate_anything, recorded_schemas
):
    out = tmp_path / "output.parquet"
    args = [
        "./tests/data-files/inference-img.tif",
        "--model",
        "DelineateAnything-S",
        "--out",
        str(out),
        "--gpu",
        "0",
        "--overwrite",
    ]
    result = CliRunner().invoke(inference_run_instance_segmentation, args)
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_instance_segmentation_inference_all_recorded(
    tmp_path: Path, recorded_items, recorded_schemas, local_delineate_anything
):
    out_dir = tmp_path / "inference_output"
    args = [
        "S2A_MSIL2A_20220315T171051_R112_T15TVG_20220316T083617",
        "--bbox=-93.68708939,41.9530844,-93.64078526,41.98070608",
        "--model",
        "DelineateAnything-S",
        "--out_dir",
        str(out_dir),
        "--gpu",
        "0",
        "--overwrite",
    ]
    result = CliRunner().invoke(inference_run_instance_segmentation_all, args)
    assert result.exit_code == 0, result.output
    assert (out_dir / "inference_data.tif").exists()
    assert (out_dir / "inference_output.parquet").exists()
