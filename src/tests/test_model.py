
from ftw_cli.model import fit, test
from click.testing import CliRunner
import pytest

def test_model_fit():
    runner = CliRunner()

    # Check help
    result = runner.invoke(fit, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: fit [OPTIONS] [CLI_ARGS]..." in result.output

    # TODO: Add more tests

def test_model_test():
    runner = CliRunner()

    # Check help
    result = runner.invoke(test, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: test [OPTIONS] [CLI_ARGS]..." in result.output

    # TODO: Add more tests
