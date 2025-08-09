import os

from click.testing import CliRunner

from ftw_tools.cli import (
    ftw_inference_all,
    inference_download,
    inference_polygonize,
    inference_run,
    model_download,
    scene_selection,
)


def test_scene_selection():
    runner = CliRunner()

    result = runner.invoke(
        scene_selection,
        [
            "--bbox=-93.68708939,  41.9530844 , -93.64078526,  41.98070608",
            "--year=2022",
            "--cloud_cover_max=20",
            "--buffer_days=14",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "S2B_15TVG_20220327_0_L2A" in result.output  # window a
    assert "S2B_15TVG_20221125_0_L2A" in result.output  # window b


def test_inference_download():  # create_input
    runner = CliRunner()

    # Download imagery by ID from EarthSearch
    inference_image = "inference_imagery/austria_example.tif"
    result = runner.invoke(
        inference_download,
        [
            "--win_a=S2B_33UUP_20210617_1_L2A",
            "--win_b=S2B_33UUP_20210925_1_L2A",
            "--bbox=13.0,48.0,13.2,48.2",
            "-o",
            inference_image,
            "-f",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Writing output" in result.output
    assert "Finished merging and writing output" in result.output
    assert os.path.exists(inference_image)


def test_inference_run():
    runner = CliRunner()

    # Download the pretrained model
    runner.invoke(model_download, ["--type=THREE_CLASS_FULL"])
    model_path = "3_Class_FULL_FTW_Pretrained.ckpt"
    assert os.path.exists(model_path)

    # Check required files are present
    inf_input_path = "./tests/data-files/inference-img.tif"
    assert os.path.exists(inf_input_path)

    # Run inference
    inf_output_path = "austria_example_output_full.tif"
    result = runner.invoke(
        inference_run,
        [
            inf_input_path,
            "--model",
            model_path,
            "--out",
            inf_output_path,
            "--gpu",
            "0",
            "--overwrite",
            "--resize_factor",
            "2",
            "--overwrite",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Using custom trainer" in result.output
    assert "Finished inference and saved output" in result.output
    assert os.path.exists(inf_output_path)
    os.remove(model_path)


def test_inference_polygonize():
    runner = CliRunner()

    # Check required files are present
    mask = "./tests/data-files/mask.tif"
    assert os.path.exists(mask)

    # Polygonize the file
    out_path = "polygons.gpkg"
    result = runner.invoke(inference_polygonize, [mask, "-o", out_path, "-f"])
    assert result.exit_code == 0, result.output
    assert "Polygonizing input file:" in result.output
    assert "Finished polygonizing output" in result.output
    assert os.path.exists(out_path)
    os.remove(out_path)


def test_ftw_inference_all():
    runner = CliRunner()

    # Download the pretrained model
    runner.invoke(model_download, ["--type=THREE_CLASS_FULL"])
    model_path = "3_Class_FULL_FTW_Pretrained.ckpt"

    # Run the full inference pipeline
    out_path = "./tests/data-files/inference_all.tif"
    result = runner.invoke(
        ftw_inference_all,
        [
            "--bbox=13.0,48.0,13.2,48.2",
            "--year=2024",
            "--out=" + out_path,
            "--cloud_cover_max=20",
            "--buffer_days=14",
            "--model=" + model_path,
            "--resize_factor=2",
            "--overwrite",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Finished inference and saved output" in result.output
    assert os.path.exists(out_path)
    os.remove(out_path)
