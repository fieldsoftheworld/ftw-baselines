"""Validation helpers for local Fields of The World datasets."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import geopandas as gpd

from ftw_tools.settings import ALL_COUNTRIES
from ftw_tools.utils import checksum_errors, compute_md5, load_archive_checksums

VALID_SPLITS = ("train", "val", "test")
MASK_DIRECTORIES = ("semantic_2class", "semantic_3class")
CHECKSUM_FILES = (
    "distances_checksums.md5",
    "masks_checksums.md5",
    "window_b_checksums.md5",
    "window_a_checksums.md5",
)


@dataclass
class CountryValidationResult:
    """Validation details for one country directory."""

    country: str
    split_counts: dict[str, int] = field(
        default_factory=lambda: {split: 0 for split in VALID_SPLITS}
    )
    missing_files: dict[str, list[Path]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    checksum_files_checked: int = 0

    @property
    def sample_count(self) -> int:
        """Return the number of samples declared by the chips file."""
        return sum(self.split_counts.values())

    @property
    def valid(self) -> bool:
        """Return whether the country passed validation."""
        return not self.errors and not self.missing_files


@dataclass
class DatasetValidationResult:
    """Validation details for an FTW dataset root."""

    root: Path
    countries: list[CountryValidationResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Return whether the complete validation passed."""
        return not self.errors and all(result.valid for result in self.countries)

    @property
    def split_counts(self) -> dict[str, int]:
        """Aggregate split counts across all selected countries."""
        extra_splits = sorted(
            {
                split
                for result in self.countries
                for split in result.split_counts
                if split not in VALID_SPLITS
            }
        )
        return {
            split: sum(result.split_counts.get(split, 0) for result in self.countries)
            for split in (*VALID_SPLITS, *extra_splits)
        }


def countries_in_dataset(root: str | Path) -> list[str]:
    """Return supported country directories found under an FTW dataset root."""
    root_path = Path(root)
    return [country for country in ALL_COUNTRIES if (root_path / country).is_dir()]


def parse_countries(value: str, root: str | Path) -> list[str]:
    """Parse a comma-separated country selection for dataset validation."""
    if value.lower() == "all":
        countries = countries_in_dataset(root)
        if not countries:
            raise ValueError(f"No supported country directories found in {root}")
        return countries

    countries = []
    for country in value.split(","):
        country = country.strip().lower()
        if not country:
            continue
        if country not in ALL_COUNTRIES:
            raise ValueError(f"Invalid country: {country}")
        if country not in countries:
            countries.append(country)
    if not countries:
        raise ValueError("Please select at least one country")
    return countries


def validate_dataset(
    root: str | Path,
    countries: Sequence[str],
    *,
    mask_directories: Sequence[str] = MASK_DIRECTORIES,
    check_samples: bool = True,
    check_checksums: bool = True,
    require_checksum_files: bool = False,
) -> DatasetValidationResult:
    """Validate selected countries in an unpacked FTW dataset.

    Checksum manifests are validated when present. Set ``require_checksum_files``
    when all known manifests must exist, as for ``FTW(checksum=True)``.
    """
    root_path = Path(root)
    report = DatasetValidationResult(root=root_path)
    if not root_path.is_dir():
        report.errors.append(f"Dataset root not found: {root_path}")
        return report

    normalized_countries = []
    for country in countries:
        normalized_country = country.lower()
        if normalized_country not in ALL_COUNTRIES:
            report.errors.append(f"Invalid country: {country}")
        elif normalized_country not in normalized_countries:
            normalized_countries.append(normalized_country)

    if not normalized_countries:
        report.errors.append("No countries selected")
        return report

    invalid_masks = set(mask_directories) - set(MASK_DIRECTORIES)
    if invalid_masks:
        report.errors.append(
            f"Invalid mask directories: {', '.join(sorted(invalid_masks))}"
        )
        return report

    for country in normalized_countries:
        report.countries.append(
            _validate_country(
                root_path,
                country,
                mask_directories=mask_directories,
                check_samples=check_samples,
                check_checksums=check_checksums,
                require_checksum_files=require_checksum_files,
            )
        )
    return report


def _validate_country(
    root: Path,
    country: str,
    *,
    mask_directories: Sequence[str],
    check_samples: bool,
    check_checksums: bool,
    require_checksum_files: bool,
) -> CountryValidationResult:
    result = CountryValidationResult(country=country)
    country_root = root / country
    if not country_root.is_dir():
        result.errors.append(f"Country directory not found: {country_root}")
        return result

    required_directories = {
        "window_a": country_root / "s2_images" / "window_a",
        "window_b": country_root / "s2_images" / "window_b",
        **{
            mask_directory: country_root / "label_masks" / mask_directory
            for mask_directory in mask_directories
        },
    }
    for label, directory in required_directories.items():
        if not directory.is_dir():
            result.errors.append(f"Missing {label} directory: {directory}")

    chips_path = country_root / f"chips_{country}.parquet"
    if not chips_path.is_file():
        result.errors.append(f"Missing chips file: {chips_path}")
    elif check_samples:
        _validate_samples(chips_path, required_directories, result)

    if check_checksums:
        available_checksum_files = []
        for checksum_name in CHECKSUM_FILES:
            checksum_path = country_root / checksum_name
            if checksum_path.is_file():
                available_checksum_files.append(checksum_path)
                result.checksum_files_checked += 1
                result.errors.extend(checksum_errors(str(checksum_path), str(root)))

        if available_checksum_files:
            if require_checksum_files:
                missing_checksum_files = set(CHECKSUM_FILES) - {
                    path.name for path in available_checksum_files
                }
                for checksum_name in sorted(missing_checksum_files):
                    result.errors.append(
                        f"Missing checksum file: {country_root / checksum_name}"
                    )
        else:
            archive_checked = _validate_archive_checksum(root, country, result)
            if require_checksum_files and not archive_checked:
                result.errors.append(
                    f"No checksum source found for {country}; expected country "
                    f"manifests or {root.parent / 'checksum.md5'} with "
                    f"{root.parent / f'{country}.zip'}"
                )

    return result


def _validate_samples(
    chips_path: Path,
    required_directories: dict[str, Path],
    result: CountryValidationResult,
) -> None:
    try:
        chips = gpd.read_parquet(chips_path)
    except Exception as error:
        result.errors.append(f"Could not read chips file {chips_path}: {error}")
        return

    required_columns = {"aoi_id", "split"}
    missing_columns = required_columns - set(chips.columns)
    if missing_columns:
        result.errors.append(
            f"Chips file {chips_path} is missing columns: "
            f"{', '.join(sorted(missing_columns))}"
        )
        return

    split_counts = chips["split"].value_counts(dropna=False)
    for split in VALID_SPLITS:
        result.split_counts[split] = int(split_counts.get(split, 0))
    for split, count in split_counts.items():
        split_name = str(split)
        if split_name not in result.split_counts:
            result.split_counts[split_name] = int(count)

    for aoi_id in chips["aoi_id"]:
        filename = f"{aoi_id}.tif"
        for label, directory in required_directories.items():
            if not directory.is_dir():
                continue
            path = directory / filename
            if not path.is_file():
                result.missing_files.setdefault(label, []).append(path)


def _validate_archive_checksum(
    root: Path, country: str, result: CountryValidationResult
) -> bool:
    archive_root = root.parent
    checksum_path = archive_root / "checksum.md5"
    archive_path = archive_root / f"{country}.zip"
    if not checksum_path.is_file() or not archive_path.is_file():
        return False

    result.checksum_files_checked += 1
    try:
        checksums = load_archive_checksums(checksum_path)
    except (OSError, ValueError) as error:
        result.errors.append(
            f"Could not read archive checksums {checksum_path}: {error}"
        )
        return True

    expected_checksum = checksums.get(country)
    if expected_checksum is None:
        result.errors.append(
            f"No archive checksum found for {country} in {checksum_path}"
        )
        return True

    current_checksum = compute_md5(str(archive_path))
    if current_checksum != expected_checksum:
        result.errors.append(f"Checksum mismatch: {archive_path}")
    return True


def format_validation_report(report: DatasetValidationResult) -> str:
    """Format a dataset validation report for CLI output."""
    lines = [f"Validating FTW dataset at {report.root}"]
    for country in report.countries:
        status = "OK" if country.valid else "FAILED"
        counts = ", ".join(
            f"{split}={count}" for split, count in country.split_counts.items()
        )
        lines.extend(
            [
                "",
                f"{country.country}: {status}",
                f"  samples: {counts}",
                f"  checksum sources checked: {country.checksum_files_checked}",
            ]
        )
        for label, paths in country.missing_files.items():
            lines.append(f"  missing {label} files: {len(paths)}")
            lines.extend(f"    - {path}" for path in paths[:5])
            if len(paths) > 5:
                lines.append(f"    - ... and {len(paths) - 5} more")
        lines.extend(f"  error: {error}" for error in country.errors)

    lines.extend(f"error: {error}" for error in report.errors)
    totals = ", ".join(
        f"{split}={count}" for split, count in report.split_counts.items()
    )
    outcome = "passed" if report.valid else "failed"
    lines.extend(["", f"Validation {outcome} ({totals})."])
    return "\n".join(lines)
