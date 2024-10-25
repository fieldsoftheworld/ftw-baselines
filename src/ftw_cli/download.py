import hashlib
import logging
import os
import shutil
import time

import click
import wget
from tqdm import tqdm


@click.group()
def ftw():
    pass

@click.command(help="Download the FTW dataset.")
@click.option('--clean_download', '-f', is_flag=True, help="If set, the script will delete the root folder before downloading.")
@click.option('--root_folder', type=str, default="./data", help="Root folder where the files will be downloaded. Defaults to './data'.")
@click.option('--countries', type=str, default="all", help="Comma-separated list of countries to download. If 'all' is passed, downloads all available countries.")
def download(clean_download, root_folder, countries):
    # Use the provided root_folder or prompt the user for it
    if root_folder is None:
        root_folder = click.prompt(
            "Please enter the root folder for downloaded files",
            default=os.path.abspath('./data'),
            show_default=True
        )
    else:
        root_folder = os.path.abspath(root_folder)

    # Root folder where the files will be downloaded
    root_folder_path = root_folder

    # Ensure the root folder exists
    os.makedirs(root_folder_path, exist_ok=True)

    # Path for the log file inside the data folder
    log_file = os.path.join(root_folder_path, 'download.log')

    # Initialize logging
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logger = logging.getLogger()

    def calculate_md5(file_path):
        """Calculate the MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def custom_progress_bar(file_name):
        start_time = time.time()
        print(f"Downloading {file_name}...")

        def bar(current, total, width=None):
            if width is None:
                width = shutil.get_terminal_size((80, 20)).columns
            elapsed_time = time.time() - start_time
            speed = current / elapsed_time if elapsed_time > 0 else 0
            progress = current / total
            bar_width = int((width - 40))  # Adjust width for the text and make the bar 30% shorter
            block = int(round(bar_width * progress))
            progress_message = f"{current / (1024**3):.3f} GB / {total / (1024**3):.3f} GB"
            speed_message = f"Speed: {speed / (1024**2):.2f} MB/s"
            bar = f"[{'#' * block}{'-' * (bar_width - block)}]"
            print(f"{progress_message} {bar} {speed_message}", end="\r")

        return bar

    def download_file(url, local_file_path):
        """
        Downloads a file from the given URL using wget.

        :param url: URL of the file to be downloaded
        :param local_file_path: Path to the local file where the URL will be downloaded
        :return: True if download was successful, False otherwise
        """
        try:
            # Use wget to download the file with a custom progress bar
            wget.download(url, local_file_path, bar=custom_progress_bar(os.path.basename(local_file_path)))
            logger.info(f"Downloaded {url} to {local_file_path}")
            print(f"\nDownloaded {local_file_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            print(f"Error downloading {url}: {str(e)}")
            return False

    def download_and_verify_md5(zip_file_path, country, checksum_data):
        """
        Download a .zip file for a country and verify its checksum.

        :param zip_file_path: Path to the local .zip file
        :param country: Name of the country
        :param checksum_data: Dictionary of country to checksum hash
        :return: True if checksum matches, False otherwise
        """
        print(f"Verifying {zip_file_path}...")

        if country not in checksum_data:
            print(f"No checksum found for {country}, skipping verification.")
            return False

        # Calculate the MD5 checksum of the downloaded file
        calculated_checksum = calculate_md5(zip_file_path)
        expected_checksum = checksum_data[country]

        if calculated_checksum == expected_checksum:
            print(f"Checksum verification passed for {country}.")
            logger.info(f"Checksum verification passed for {zip_file_path}")
            return True
        else:
            print(f"Checksum verification failed for {country}. Expected: {expected_checksum}, Found: {calculated_checksum}")
            logger.error(f"Checksum verification failed for {zip_file_path}")
            return False

    def load_checksums(local_md5_file_path):
        """
        Load the checksum data from a local md5 file.

        :param local_md5_file_path: Path to the local checksum.md5 file
        :return: Dictionary with country name as key and checksum hash as value
        """
        checksum_data = {}
        with open(local_md5_file_path, 'r') as file:
            for line in file:
                country, checksum = line.strip().split(',')
                checksum_data[country.lower()] = checksum
        return checksum_data

    def clean_root_folder(root_folder_path):
        """
        Deletes all files and directories in the root folder.
        """
        if os.path.exists(root_folder_path):
            for filename in os.listdir(root_folder_path):
                file_path = os.path.join(root_folder_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        os.rmdir(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")

    def download_all_country_files(root_folder, country_names, checksum_data):
        """
        Download all .zip files for the specified countries and verify their checksums using basic URL downloads.

        :param root_folder: Path to the root folder where files will be downloaded
        :param country_names: List of country names whose files need to be downloaded
        :param checksum_data: Dictionary of country to checksum hash
        """
        base_url = 'https://data.source.coop/kerner-lab/fields-of-the-world-archive/'

        # tqdm progress for the entire dataset
        overall_progress = tqdm(total=len(country_names), desc="Overall Download Progress", unit="country")

        for country in country_names:
            url = f"{base_url}{country}.zip"
            local_file_path = os.path.join(root_folder, f"{country}.zip")

            # Check if the file already exists locally
            if os.path.exists(local_file_path):
                logger.info(f"File {local_file_path} already exists, skipping download.")
                print(f"File {local_file_path} already exists, skipping download.")
            else:
                # Download the file
                if download_file(url, local_file_path):
                    # Verify the file checksum
                    download_and_verify_md5(local_file_path, country.lower(), checksum_data)
            
            # Update the overall progress
            overall_progress.update(1)

        overall_progress.close()

    # List of all available countries
    all_countries = [
        "belgium", 
        "cambodia", 
        "croatia", 
        "estonia", 
        "portugal", 
        "slovakia", 
        "south_africa", 
        "sweden", 
        "austria", 
        "brazil", 
        "corsica", 
        "denmark", 
        "france", 
        "india", 
        "latvia", 
        "luxembourg", 
        "finland", 
        "germany", 
        "kenya", 
        "lithuania", 
        "netherlands", 
        "rwanda", 
        "slovenia", 
        "spain", 
        "vietnam"
    ]

    # Main script
    # Step 1: Clean the dataset folder if --clean_download is specified
    if clean_download:
        print(f"Performing a clean download. Deleting contents of {root_folder_path}...")
        clean_root_folder(root_folder_path)

    # Step 2: Download the checksum.md5 file
    local_md5_file_path = os.path.join(root_folder_path, 'checksum.md5')
    md5_file_url = 'https://data.source.coop/kerner-lab/fields-of-the-world-archive/checksum.md5'

    if download_file(md5_file_url, local_md5_file_path):
        print(f"Downloaded checksum.md5 to {local_md5_file_path}")

        # Step 3: Load the checksum data
        checksum_data = load_checksums(local_md5_file_path)

        # Step 4: Handle country selection (all or specific countries)
        if countries == 'all':
            country_names = all_countries
            print("Downloading all available countries...")
        else:
            country_names = [country.lower().strip() for country in countries.split(',') if country.lower().strip() in all_countries]
            print(f"Downloading selected countries: {country_names}")

        # Step 5: Download all .zip files for the specified countries and verify their checksums
        download_all_country_files(root_folder_path, country_names, checksum_data)

    else:
        print("Failed to download checksum.md5 file.")

# Add the download command to the ftw click group
ftw.add_command(download)

if __name__ == "__main__":
    ftw()
