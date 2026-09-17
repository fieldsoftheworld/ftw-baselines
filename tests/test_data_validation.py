import hashlib
from pathlib import Path

import geopandas as gpd
import pytest
from click.testing import CliRunner
from shapely.geometry import Point

from ftw_tools.cli import data_validate
from ftw_tools.data_validation import parse_countries, validate_dataset
from ftw_tools.training.datasets import FTW


@pytest.fixture()
def ftw_dataset(tmp_path):
    """Build a complete, one-country FTW dataset with three samples."""
    country = "france"
    country_root = tmp_path / country
    directories = (
        country_root / "s2_images" / "window_a",
        country_root / "s2_images" / "window_b",
        country_root / "label_masks" / "semantic_2class",
        country_root / "label_masks" / "semantic_3class",
    )
    for directory in directories:
        directory.mkdir(parents=True)

    aoi_ids = ["train_0", "val_0", "test_0"]
    for aoi_id in aoi_ids:
        for directory in directories:
            (directory / f"{aoi_id}.tif").write_bytes(aoi_id.encode())

    chips = gpd.GeoDataFrame(
        {"aoi_id": aoi_ids, "split": ["train", "val", "test"]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs="EPSG:4326",
    )
    chips.to_parquet(country_root / f"chips_{country}.parquet")
    return tmp_path


def _write_checksum_manifest(root: Path, country: str, name: str, target: Path):
    checksum = hashlib.md5(target.read_bytes()).hexdigest()
    relative_target = target.relative_to(root)
    (root / country / name).write_text(
        f"{checksum}  {relative_target}\n", encoding="utf-8"
    )


def test_validate_dataset_reports_split_counts(ftw_dataset):
    report = validate_dataset(ftw_dataset, ["france"])

    assert report.valid
    assert report.split_counts == {"train": 1, "val": 1, "test": 1}
    assert report.countries[0].checksum_files_checked == 0


def test_validate_dataset_reports_missing_sample_files(ftw_dataset):
    missing_file = ftw_dataset / "france/s2_images/window_a/val_0.tif"
    missing_file.unlink()

    report = validate_dataset(ftw_dataset, ["france"])

    assert not report.valid
    assert report.countries[0].missing_files["window_a"] == [missing_file]


def test_validate_dataset_checks_available_manifests(ftw_dataset):
    country_root = ftw_dataset / "france"
    target = country_root / "s2_images/window_a/train_0.tif"
    manifest = country_root / "window_a_checksums.md5"
    manifest.write_text(
        f"{'0' * 32}  {target.relative_to(ftw_dataset)}\n", encoding="utf-8"
    )

    report = validate_dataset(ftw_dataset, ["france"])

    assert not report.valid
    assert report.countries[0].checksum_files_checked == 1
    assert report.countries[0].errors == [f"Checksum mismatch: {target}"]


def test_parse_all_countries_uses_only_downloaded_countries(ftw_dataset):
    assert parse_countries("all", ftw_dataset) == ["france"]


def test_data_validate_command_succeeds(ftw_dataset):
    result = CliRunner().invoke(
        data_validate, [str(ftw_dataset), "--countries", "france"]
    )

    assert result.exit_code == 0, result.output
    assert "france: OK" in result.output
    assert "train=1, val=1, test=1" in result.output
    assert "Validation passed" in result.output


def test_data_validate_command_fails_for_missing_file(ftw_dataset):
    (ftw_dataset / "france/label_masks/semantic_3class/test_0.tif").unlink()

    result = CliRunner().invoke(
        data_validate, [str(ftw_dataset), "--countries", "france"]
    )

    assert result.exit_code == 1
    assert "missing semantic_3class files: 1" in result.output
    assert "Validation failed" in result.output


def test_ftw_checksum_checks_only_selected_countries(ftw_dataset):
    country = "france"
    target = ftw_dataset / country / "s2_images/window_a/train_0.tif"
    for checksum_name in (
        "distances_checksums.md5",
        "masks_checksums.md5",
        "window_b_checksums.md5",
        "window_a_checksums.md5",
    ):
        _write_checksum_manifest(ftw_dataset, country, checksum_name, target)

    dataset = FTW(
        root=str(ftw_dataset),
        countries=country,
        split="train",
        checksum=True,
        verbose=False,
    )

    assert len(dataset) == 1
