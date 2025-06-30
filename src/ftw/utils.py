import hashlib
import os


def compute_md5(file_path):
    """Compute the MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    except FileNotFoundError:
        return None
    return hash_md5.hexdigest()


def validate_checksums(checksum_file, root_directory):
    """Validate checksums stored in a checksum file."""
    if not os.path.isfile(checksum_file):
        print(f"Checksum file not found: {checksum_file}")
        return

    with open(checksum_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        
        stored_checksum, file_path = parts
        file_path = os.path.join(root_directory, file_path)
        current_checksum = compute_md5(file_path)
        
        if current_checksum != stored_checksum:
            print("Checksum mismatch: {file_path}")
            return False
    return True

