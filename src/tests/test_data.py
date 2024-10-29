import os
from click.testing import CliRunner

from ftw_cli.cli import data_download as download
from ftw_cli.cli import data_unpack as unpack


def test_data_download():
    runner = CliRunner()

    # Clean download
    result = runner.invoke(download, ["-f", "--countries=Rwanda", "--no-unpack"])
    assert result.exit_code == 0, result.output
    assert "Downloading selected countries: ['rwanda']" in result.output
    assert "Overall Download Progress: 100%" in result.output
    assert "Unpacking files:" not in result.output
    assert os.path.exists("data/rwanda.zip")
    assert not os.path.exists("data/ftw/rwanda")

    # Try again and expect to skip the download, but unpack files
    result = runner.invoke(download, ["--countries=Rwanda"])
    assert result.exit_code == 0, result.output
    assert "already exists, skipping download." in result.output
    assert "Unpacking files: 100%" in result.output
    assert os.path.exists("data/rwanda.zip")
    assert os.path.exists("data/ftw/rwanda")

    # Try again and expect to skip the download and not unpack
    result = runner.invoke(download, ["--countries=Rwanda"])
    assert result.exit_code == 0, result.output
    assert "already exists, skipping download." in result.output
    assert "Unpacking files:" in result.output

def test_data_unpack():
    runner = CliRunner()

    # Unpack the files
    result = runner.invoke(unpack, [])
    assert result.exit_code == 0, result.output
    assert "Unpacking files: 100%" in result.output

    # Error with non-existing folder
    result = runner.invoke(unpack, ["./invalid_folder"])
    assert result.exit_code == 1, result.output
    assert "Folder ./invalid_folder does not exist." in result.output
