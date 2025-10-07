import os
from pathlib import Path

from click.testing import CliRunner

from ftw_tools.cli import data_download as download
from ftw_tools.cli import data_unpack as unpack


def test_data_download(tmp_path: Path):
    runner = CliRunner()

    # Clean download
    result = runner.invoke(
        download, ["-f", "--countries=Rwanda", "--no-unpack", "-o", str(tmp_path)]
    )
    assert result.exit_code == 0, (
        f"Exited with {result.exit_code}. Output: {result.stdout} {result.stderr}"
    )
    assert "Downloading selected countries: ['rwanda']" in result.output
    assert "Overall Download Progress: 100%" in result.output
    assert "Unpacking files:" not in result.output
    assert os.path.exists(str(tmp_path / "rwanda.zip"))
    assert not os.path.exists(str(tmp_path / "ftw" / "rwanda"))

    # Try again and expect to skip the download and not unpack
    result = runner.invoke(download, ["--countries=Rwanda", "-o", str(tmp_path)])
    assert result.exit_code == 0, (
        f"Exited with {result.exit_code}. Output: {result.stdout} {result.stderr}"
    )
    assert "already exists, skipping download." in result.output
    assert "Unpacking files:" in result.output


def test_data_unpack(tmp_path: Path):
    runner = CliRunner()

    out_dir = tmp_path / "data"
    # First download without unpacking so we have something to unpack
    result = runner.invoke(
        download, ["-f", "--countries=Rwanda", "--no-unpack", "-o", str(out_dir)]
    )
    assert result.exit_code == 0, (
        f"Exited with {result.exit_code}. Output: {result.stdout} {result.stderr}"
    )
    assert os.path.exists(str(out_dir / "rwanda.zip"))

    # Now unpack the files
    result = runner.invoke(unpack, [str(out_dir)])
    assert result.exit_code == 0, (
        f"Exited with {result.exit_code}. Output: {result.stdout} {result.stderr}"
    )
    assert "Unpacking files: 100%" in result.output

    # Error with non-existing folder
    result = runner.invoke(unpack, ["invalid_folder"])
    assert result.exit_code == 2, result.output
    assert "Directory 'invalid_folder' does not exist." in result.output
