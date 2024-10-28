
from click.testing import CliRunner

from ftw_cli.cli import model_fit, model_test


def test_model_fit():
    runner = CliRunner()

    # Check help
    result = runner.invoke(model_fit, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: fit [OPTIONS] [CLI_ARGS]..." in result.output

    # TODO: Add more tests

def test_model_test():
    runner = CliRunner()

    # Check help
    result = runner.invoke(model_test, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: test [OPTIONS] [CLI_ARGS]..." in result.output

    # TODO: Add more tests
