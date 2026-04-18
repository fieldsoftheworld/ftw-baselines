#!/usr/bin/env python3
"""
Download .npz files from source.coop.
Provide file paths in a file or via command line.
"""

from pathlib import Path
from urllib.request import urlopen, Request
import sys

S3_ENDPOINT = "https://data.source.coop"
PREFIX = "mvrl/ftw-inference-gfm/precomputed_feats/clay/austria"
OUTPUT_DIR = "./source_coop_download"


def download_file(url, output_path):
    """Download a single file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        size = output_path.stat().st_size
        print(f"  [OK] Exists ({size:,} bytes): {output_path.name}")
        return True

    try:
        print(f"  [DL] Downloading: {output_path.name}...", end=" ", flush=True)
        req = Request(url)
        req.add_header(
            "User-Agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        )
        with urlopen(req, timeout=120) as response:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(output_path, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    f.write(chunk)

        size = output_path.stat().st_size
        print(f"OK ({size:,} bytes)")
        return True
    except Exception as e:
        print(f"Error: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def main():
    # Try to find file list
    files = []

    # Check if file list was provided
    if len(sys.argv) > 1:
        list_file = sys.argv[1]
        if Path(list_file).exists():
            with open(list_file) as f:
                files = [line.strip() for line in f if line.strip()]
    else:
        # Try to load from files_to_download.txt if it exists
        if Path("files_to_download.txt").exists():
            with open("files_to_download.txt") as f:
                files = [line.strip() for line in f if line.strip()]
        else:
            # Use example file as default
            files = [
                f"{PREFIX}/window_a/clay_g77_00002_10.npz",
            ]

    print("FTW Source.coop Downloader")
    print(f"Endpoint: {S3_ENDPOINT}")
    print(f"Output: {OUTPUT_DIR}\n")

    if not files:
        print("No files specified!")
        print("\nUsage:")
        print("  python download_source_coop.py [file_list.txt]")
        print("\nOr create files_to_download.txt with one relative path per line:")
        print("  mvrl/ftw-inference-gfm/precomputed_feats/clay/austria/window_a/file.npz")
        return

    print(f"Found {len(files)} files to download\n")

    success = 0
    failed = 0

    for i, file_path in enumerate(files, 1):
        # Construct full URL
        if file_path.startswith("http"):
            url = file_path
            # Extract relative path
            rel_path = file_path.replace(S3_ENDPOINT + "/", "")
        else:
            url = f"{S3_ENDPOINT}/{file_path}"
            rel_path = file_path

        # Get output path (strip the prefix for local storage)
        if PREFIX in rel_path:
            local_rel = rel_path[len(PREFIX) :].lstrip("/")
        else:
            local_rel = rel_path.split("/")[-1]

        output_path = Path(OUTPUT_DIR) / local_rel

        print(f"[{i}/{len(files)}]", end=" ")
        if download_file(url, output_path):
            success += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(
        f"Complete! Downloaded {success}, Failed {failed}"
    )
    print(f"Files saved to: {OUTPUT_DIR}")

    if failed > 0:
        print("\nNote: If all files failed, you may need to:")
        print("  - Check file paths are correct")
        print("  - Verify network connectivity")


if __name__ == "__main__":
    main()
