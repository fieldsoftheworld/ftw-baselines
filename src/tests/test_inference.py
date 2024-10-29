
import os

from click.testing import CliRunner

from ftw_cli.cli import inference_download, inference_polygonize, inference_run

INF_IMG_PATH = "inference_imagery/austria_example.tif"
INF_OUT_PATH = "austria_example_output_full.tif"

def test_inference_download(): # create_input
    runner = CliRunner()

    # Check help
    result = runner.invoke(inference_download, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: download [OPTIONS]" in result.output

    # Download imagery by ID from MS Planetary Computer
    result = runner.invoke(inference_download, [
        "--win_a=S2B_MSIL2A_20210617T100559_R022_T33UUP_20210624T063729",
        "--win_b=S2B_MSIL2A_20210925T101019_R022_T33UUP_20210926T121923",
        "-o", INF_IMG_PATH,
        "-f"
    ])
    assert result.exit_code == 0, result.output
    assert "Finished saving window A to file" in result.output
    assert "Finished saving window B to file" in result.output
    assert "Finished merging and writing output" in result.output
    assert os.path.exists(INF_IMG_PATH)

def test_inference_run():
    runner = CliRunner()

    # Check help
    result = runner.invoke(inference_run, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: run [OPTIONS] INPUT" in result.output

    # Run inference
    model_path = "3_Class_FULL_FTW_Pretrained.ckpt"
    assert os.path.exists(model_path)
    result = runner.invoke(inference_run, [
        INF_IMG_PATH,
        "--model", model_path,
        "--out", INF_OUT_PATH,
        "--gpu", "0",
        "--overwrite",
        "--resize_factor", "2",
        "--overwrite"
    ])
    assert result.exit_code == 0, result.output
    assert "Using custom trainer" in result.output
    assert "100%|" in result.output
    assert "Finished inference and saved output" in result.output
    assert os.path.exists(INF_IMG_PATH)


def test_inference_polygonize():
    runner = CliRunner()

    # Check help
    result = runner.invoke(inference_polygonize, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: polygonize [OPTIONS] INPUT" in result.output

    # TODO: Activate once https://github.com/fieldsoftheworld/ftw-baselines/issues/27#issuecomment-2445119067 has been fixed
    # Polygonize the file
    # out_path = "austria_example_output_full.gpkg"
    # result = runner.invoke(inference_polygonize, [INF_OUT_PATH, "-o", out_path, "-f"])
    # assert result.exit_code == 0, result.output
    # assert "Polygonizing input file:" in result.output
    # assert "100%|" in result.output
    # assert "Finished polygonizing output" in result.output
    # assert os.path.exists(out_path)
