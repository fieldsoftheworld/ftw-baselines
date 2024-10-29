from click.testing import CliRunner

from ftw_cli.cli import data_download as download
from ftw_cli.cli import data_unpack as unpack
from ftw_cli.cli import inference_download, inference_polygonize, inference_run
from ftw_cli.cli import model_fit, model_test

# Run all the help functions for a basic CLI check
# Just ensures imports are correct and that there are no syntax errors in Python files etc

def test_data_download():
    runner = CliRunner()
    result = runner.invoke(download, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: download [OPTIONS]" in result.output

def test_data_unpack():
    runner = CliRunner()
    result = runner.invoke(unpack, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: unpack [OPTIONS] [INPUT]" in result.output

def test_inference_download(): # create_input
    runner = CliRunner()
    result = runner.invoke(inference_download, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: download [OPTIONS]" in result.output

def test_inference_polygonize():
    runner = CliRunner()
    result = runner.invoke(inference_polygonize, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: polygonize [OPTIONS] INPUT" in result.output

def test_inference_run():
    runner = CliRunner()
    result = runner.invoke(inference_run, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: run [OPTIONS] INPUT" in result.output

def test_model_fit():
    runner = CliRunner()
    result = runner.invoke(model_fit, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: fit [OPTIONS] [CLI_ARGS]..." in result.output

def test_model_test():
    runner = CliRunner()
    result = runner.invoke(model_test, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: test [OPTIONS] [CLI_ARGS]..." in result.output
