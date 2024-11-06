
import os

from click.testing import CliRunner

from ftw_cli.cli import inference_download, inference_polygonize, inference_run


def test_inference_download(): # create_input
    runner = CliRunner()

    # Download imagery by ID from MS Planetary Computer
    inference_image = "inference_imagery/austria_example.tif"
    result = runner.invoke(inference_download, [
        "--win_a=S2B_MSIL2A_20210617T100559_R022_T33UUP_20210624T063729",
        "--win_b=S2B_MSIL2A_20210925T101019_R022_T33UUP_20210926T121923",
        "-o", inference_image,
        "-f"
    ])
    assert result.exit_code == 0, result.output
    assert "Loading data" in result.output
    assert "Merging data" in result.output
    assert "Writing output" in result.output
    assert "Finished merging and writing output" in result.output
    assert os.path.exists(inference_image)

def test_inference_run():
    runner = CliRunner()

    # Check required files are present
    model_path = "3_Class_FULL_FTW_Pretrained.ckpt"
    assert os.path.exists(model_path)
    inf_input_path = "./src/tests/data-files/inference-img.tif"
    assert os.path.exists(inf_input_path)

    # Run inference
    inf_output_path = "austria_example_output_full.tif"
    result = runner.invoke(inference_run, [
        inf_input_path,
        "--model", model_path,
        "--out", inf_output_path,
        "--gpu", "0",
        "--overwrite",
        "--resize_factor", "2",
        "--overwrite"
    ])
    assert result.exit_code == 0, result.output
    assert "Using custom trainer" in result.output
    assert "Finished inference and saved output" in result.output
    assert os.path.exists(inf_output_path)

def test_inference_polygonize():
    runner = CliRunner()

    # Check required files are present
    mask = "./src/tests/data-files/mask.tif"
    assert os.path.exists(mask)

    # Polygonize the file
    out_path = "polygons.gpkg"
    result = runner.invoke(inference_polygonize, [
        mask,
        "-o", out_path,
        "-f"
    ])
    assert result.exit_code == 0, result.output
    assert "Polygonizing input file:" in result.output
    assert "Finished polygonizing output" in result.output
    assert os.path.exists(out_path)
