import multiprocessing
import os
import shutil
import sys
import zipfile

from tqdm import tqdm

def unpack(input):
    ftw_folder_path = os.path.join(input, 'ftw')

    # Ensure the input folder exists
    if not os.path.exists(input):
        print(f"Folder {input} does not exist.")
        sys.exit(1)

    clean_and_create_ftw_folder(ftw_folder_path)
    unpack_zip_files(input, ftw_folder_path)

def clean_and_create_ftw_folder(ftw_folder_path):
    """
    Delete the existing ftw folder if it exists and recreate it.

    :param ftw_folder_path: Path to the ftw folder.
    """
    if os.path.exists(ftw_folder_path):
        print(f"Deleting existing folder {ftw_folder_path}...")
        shutil.rmtree(ftw_folder_path)

    os.makedirs(ftw_folder_path, exist_ok=True)
    print(f"Created new folder {ftw_folder_path}.")

def unpack_zip_file(zip_file_path, extract_folder_path):
    """
    Unpack a single .zip file into the specified folder.

    :param zip_file_path: Path to the .zip file.
    :param extract_folder_path: Folder where the contents will be extracted.
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder_path)
    return zip_file_path

def unpack_zip_files(root_folder_path, ftw_folder_path):
    """
    Unpack all .zip files in the given root folder into subfolders under the ftw folder using multiprocessing.

    :param root_folder_path: Path to the root folder where the .zip files are located.
    :param ftw_folder_path: Path to the ftw folder where unpacked files will be placed.
    """
    # Find all .zip files in the root folder
    zip_files = [f for f in os.listdir(root_folder_path) if f.endswith('.zip')]

    if not zip_files:
        print(f"No .zip files found in {root_folder_path}")
        return

    # Prepare for multiprocessing
    cpu_count = multiprocessing.cpu_count()
    print(f"Using {cpu_count} CPUs for unpacking.")
    
    # Define tasks for each zip file
    tasks = []
    for zip_file in zip_files:
        zip_file_path = os.path.join(root_folder_path, zip_file)
        folder_name = os.path.splitext(zip_file)[0]  # Remove .zip extension to get folder name
        extract_folder_path = os.path.join(ftw_folder_path, folder_name)

        # Create the folder where contents will be extracted
        os.makedirs(extract_folder_path, exist_ok=True)

        tasks.append((zip_file_path, extract_folder_path))

    # Unpack the .zip files using multiprocessing with tqdm
    with multiprocessing.Pool(cpu_count) as pool:
        for _ in tqdm(pool.starmap(unpack_zip_file, tasks), total=len(tasks), desc="Unpacking files"):
            pass
