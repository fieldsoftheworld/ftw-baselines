import concurrent.futures
import functools
import hashlib
import logging
import os
import shutil

import boto3
import smart_open
from botocore import UNSIGNED
from botocore.config import Config

from .cfg import ALL_COUNTRIES

logger = logging.getLogger()
client = boto3.client(
    "s3", config=Config(signature_version=UNSIGNED), region_name="us-west-1"
)


def _load_checksums(local_md5_file_path):
    """
    Load the checksum data from a local md5 file.

    :param local_md5_file_path: Path to the local checksum.md5 file
    :return: Dictionary with country name as key and checksum hash as value
    """
    checksum_data = {}
    with open(local_md5_file_path, "r") as file:
        for line in file:
            country, checksum = line.strip().split(",")
            checksum_data[country.lower()] = checksum
    return checksum_data


def _calculate_md5(file_path):
    """Calculate the MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _download_file(key: str, fpath: str):
    # if key.endswith(".zip"):
    print(f"Downloading {key} to {fpath}")
    with smart_open.open(
        f"s3://us-west-2.opendata.source.coop/{key}",
        "rb",
        transport_params={"client": client},
    ) as f:
        with open(fpath, "wb") as outf:
            for chunk in f:
                outf.write(chunk)


def _download_country_file(
    country_name: str, root_folder: str, checksum_data: dict[str, str]
):
    key = f"kerner-lab/fields-of-the-world-archive/{country_name}.zip"
    local_file_path = os.path.join(root_folder, f"{country_name}.zip")

    # Check if the file already exists locally
    if os.path.exists(local_file_path):
        print(f"File {local_file_path} already exists, skipping download.")
        return

    # Otherwise download the file.
    expected_md5 = checksum_data[country_name]
    try:
        _download_file(key, local_file_path)

        # Verify checksum (md5 hash)
        actual_md5 = _calculate_md5(local_file_path)
        if country_name not in checksum_data:
            print(f"No checksum found for {country_name}, skipping verification.")
            return
        if actual_md5 == expected_md5:
            logger.info(f"Checksum verification passed for {local_file_path}")
            print(f"Checksum verification passed for {country_name}.")
        else:
            logger.error(f"Checksum verification failed for {local_file_path}")
            print(
                f"Checksum verification failed for {country_name}. Expected: {expected_md5}, Found: {actual_md5}"
            )
    except Exception:
        logger.exception(f"Error downloading {key}")
        print(f"Failed to download {key}.")


def download(out, clean_download, countries):
    root_folder_path = os.path.abspath(out)

    # Deletes all files and directories in the root folder.
    if clean_download and os.path.exists(root_folder_path):
        try:
            shutil.rmtree(root_folder_path)
            print(f"Deleted old data from {root_folder_path}")
        except Exception as e:
            print(f"Failed to delete {root_folder_path}. Reason: {e}")

    # Ensure the root folder exists
    os.makedirs(root_folder_path, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(root_folder_path, "download.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    # Step 1: Download the checksum.md5 file
    local_md5_file_path = os.path.join(root_folder_path, "checksum.md5")
    key = "kerner-lab/fields-of-the-world-archive/checksum.md5"
    try:
        _download_file(key, local_md5_file_path)
        print(f"Downloaded checksum.md5 to {local_md5_file_path}")
    except Exception as exc:
        print("Failed to download checksum.md5 file.")
        raise exc

    # Step 2: Load the checksum data
    checksum_data = _load_checksums(local_md5_file_path)

    # Step 3: Handle country selection (all or specific countries)
    if countries == "all":
        country_names = ALL_COUNTRIES
        print("Downloading all available countries...")
    else:
        country_names = [
            country.lower().strip()
            for country in countries.split(",")
            if country.lower().strip() in ALL_COUNTRIES
        ]
        print(f"Downloading selected countries: {country_names}")

    # Step 4: Run the download
    func = functools.partial(
        _download_country_file,
        root_folder=root_folder_path,
        checksum_data=checksum_data,
    )
    with concurrent.futures.ThreadPoolExecutor() as exec:
        exec.map(func, country_names)
