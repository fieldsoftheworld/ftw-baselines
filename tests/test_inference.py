import os
from pathlib import Path

from click.testing import CliRunner

from ftw_tools.cli import (
    ftw_inference_all,
    inference_download,
    inference_polygonize,
    inference_run,
    inference_run_instance_segmentation,
    inference_run_instance_segmentation_all,
    scene_selection,
)


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
        "3_Class_FULL_v1",
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
    assert "Using custom trainer" in result.output
    assert "Finished inference and saved output" in result.output
    assert os.path.exists(str(inf_output_path))


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


def test_ftw_inference_all(tmp_path: Path):
    runner = CliRunner()
    out_path = tmp_path / "inference_output"
    args = [
        "--bbox=13.0,48.0,13.2,48.2",
        "--year=2024",
        f"--out={str(out_path)}",
        "--cloud_cover_max=20",
        "--buffer_days=14",
        "--model=3_Class_FULL_v1",
        "--resize_factor=2",
        "--overwrite",
    ]
    result = runner.invoke(ftw_inference_all, args)
    assert result.exit_code == 0, (
        f"Exited with {result.exit_code}. Output: {result.stdout} {result.stderr}"
    )
    assert "Finished inference and saved output" in result.output
    assert os.path.exists(str(out_path))


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
