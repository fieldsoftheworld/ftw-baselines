from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from ftw_tools.training.datasets import FTW


def _make_country_dir(root, country, aoi_id="sample_0", mask_dirnames=()):
    """Build a minimal fake FTW dataset directory for a single country/sample.

    Creates the window_a/window_b image files (always) plus the
    ``label_masks/<name>`` mask file for each name in ``mask_dirnames``. The
    corresponding mask directories are always created (even if empty) so that
    ``_check_integrity`` passes regardless of which mask type is requested.
    """
    country_dir = root / country
    (country_dir / "s2_images" / "window_a").mkdir(parents=True)
    (country_dir / "s2_images" / "window_b").mkdir(parents=True)
    (country_dir / "label_masks" / "semantic_2class").mkdir(parents=True)
    (country_dir / "label_masks" / "semantic_3class").mkdir(parents=True)

    (country_dir / "s2_images" / "window_a" / f"{aoi_id}.tif").touch()
    (country_dir / "s2_images" / "window_b" / f"{aoi_id}.tif").touch()
    for name in mask_dirnames:
        (country_dir / "label_masks" / name / f"{aoi_id}.tif").touch()

    chips = gpd.GeoDataFrame(
        {"aoi_id": [aoi_id], "split": ["train"]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    chips.to_parquet(country_dir / f"chips_{country}.parquet")

    return country_dir


@pytest.mark.parametrize(
    ("load_boundaries", "mask_dirname"),
    [(False, "semantic_2class"), (True, "semantic_3class")],
)
def test_ftw_requires_only_selected_mask_type(tmp_path, load_boundaries, mask_dirname):
    """A sample with only its requested mask type present should be included."""
    country = "france"
    aoi_id = "sample_0"
    _make_country_dir(tmp_path, country, aoi_id, mask_dirnames=(mask_dirname,))

    dataset = FTW(
        root=str(tmp_path),
        countries=country,
        split="train",
        load_boundaries=load_boundaries,
        verbose=False,
    )
    assert len(dataset) == 1
    assert Path(dataset.filenames[0]["mask"]).parts[-2:] == (
        mask_dirname,
        f"{aoi_id}.tif",
    )


def test_ftw_skips_sample_missing_selected_mask_type(tmp_path):
    """Only the 2-class mask exists for the sample. Requesting the 2-class mask
    (load_boundaries=False) should find it, while requesting the 3-class mask
    (load_boundaries=True) should skip the sample -- demonstrating that only the
    selected mask type is required, not both.
    """
    country = "france"
    aoi_id = "sample_0"
    _make_country_dir(tmp_path, country, aoi_id, mask_dirnames=("semantic_2class",))

    dataset_2class = FTW(
        root=str(tmp_path),
        countries=country,
        split="train",
        load_boundaries=False,
        verbose=False,
    )
    assert len(dataset_2class) == 1

    dataset_3class = FTW(
        root=str(tmp_path),
        countries=country,
        split="train",
        load_boundaries=True,
        verbose=False,
    )
    assert len(dataset_3class) == 0
