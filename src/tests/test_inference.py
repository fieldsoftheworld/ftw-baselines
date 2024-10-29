
from click.testing import CliRunner

from ftw_cli.cli import inference_download, inference_polygonize, inference_run


def test_inference_download(): # create_input
    runner = CliRunner()

    # Check help
    result = runner.invoke(inference_download, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: download [OPTIONS]" in result.output

    # TODO: Add more tests

def test_inference_polygonize():
    runner = CliRunner()

    # Check help
    result = runner.invoke(inference_polygonize, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: polygonize [OPTIONS] INPUT" in result.output

    # TODO: Add more tests

def test_inference_run():
    runner = CliRunner()

    # Check help
    result = runner.invoke(inference_run, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: run [OPTIONS] INPUT" in result.output

    # TODO: Add more tests
