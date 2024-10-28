from ftw_cli.download import download
from ftw_cli.unpack import unpack
from click.testing import CliRunner
import pytest

@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_data_download():
    runner = CliRunner()

    # Check help
    result = runner.invoke(download, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: download [OPTIONS]" in result.output

    # Clean download
    result = runner.invoke(download, ["-f", "--countries=Rwanda"])
    assert result.exit_code == 0, result.output
    assert "Downloading selected countries: ['rwanda']" in result.output
    assert "Overall Download Progress: 100%" in result.output

    # Try again and expect to skip the download
    result = runner.invoke(download, ["--countries=Rwanda"])
    assert result.exit_code == 0, result.output
    assert "already exists, skipping download." in result.output

def test_data_unpack():
    runner = CliRunner()
    
    # Check help
    result = runner.invoke(unpack, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage: unpack [OPTIONS] [INPUT]" in result.output

    # Unpack the files
    result = runner.invoke(unpack, [])
    assert result.exit_code == 0, result.output
    assert "Unpacking files: 100%" in result.output

    # Error with non-existing folder
    result = runner.invoke(unpack, ["./invalid_folder"])
    assert result.exit_code == 1, result.output
    assert "Folder ./invalid_folder does not exist." in result.output
