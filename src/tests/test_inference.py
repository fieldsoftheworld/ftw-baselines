
from ftw_cli.inference import create_input, polygonize, run
from click.testing import CliRunner
import pytest

def test_inference_download(): # create_input
    runner = CliRunner()

    # Check help
    result = runner.invoke(create_input, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: download [OPTIONS]" in result.output

    # TODO: Add more tests

def test_inference_polygonize():
    runner = CliRunner()

    # Check help
    result = runner.invoke(polygonize, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: polygonize [OPTIONS] INPUT" in result.output

    # TODO: Add more tests

def test_inference_run():
    runner = CliRunner()

    # Check help
    result = runner.invoke(run, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: run [OPTIONS] INPUT" in result.output

    # TODO: Add more tests
