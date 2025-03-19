#!/usr/bin/env python3
'''
Compute band-wise statistics of mean and std for all GeoTIFF raster images in a directory.
Example usage: python3 compute_band_stats.py /ftw/austria/s2_images/ --output austria_stats.txt
TODO: need to integrate into cli
'''

import os
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
from typing import Tuple, List

def setup_logger():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def is_valid_geotiff(filepath: Path) -> bool:
    """
    Check if a file is a valid GeoTIFF.
    
    Args:
        filepath: Path to the file
        
    Returns:
        bool: True if file is a valid GeoTIFF
    """
    try:
        with rasterio.open(filepath) as src:
            return True
    except Exception as e:
        return False

def compute_band_stats(image_dir: Path, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-band mean and standard deviation for all GeoTIFFs in directory.
    Uses Welford's online algorithm for numerical stability.
    
    Args:
        image_dir: Path to directory containing GeoTIFFs
        logger: Logger instance
        
    Returns:
        Tuple containing:
            - np.ndarray: Per-band means
            - np.ndarray: Per-band standard deviations
    """
    # Get list of GeoTIFF files
    tiff_files = [f for f in image_dir.glob('**/*') if f.suffix.lower() in ('.tif', '.tiff')]
    logger.info(f"Found {len(tiff_files)} potential GeoTIFF files")
    
    # Filter for valid GeoTIFFs
    valid_files = [f for f in tiff_files if is_valid_geotiff(f)]
    logger.info(f"{len(valid_files)} valid GeoTIFF files will be processed")
    
    if not valid_files:
        raise ValueError("No valid GeoTIFF files found in directory")
    
    # Initialize variables for Welford's algorithm
    with rasterio.open(valid_files[0]) as src:
        num_bands = src.count
        logger.info(f"Processing {num_bands} bands per image")
        means = np.zeros(num_bands)
        M2s = np.zeros(num_bands)  # For variance computation
    
    total_pixels = 0
    
    # Process each image
    for filepath in tqdm(valid_files, desc="Processing images"):
        try:
            with rasterio.open(filepath) as src:
                # Read all bands
                image = src.read()
                
                # Skip any fully masked or invalid areas
                if src.nodata is not None:
                    valid_mask = image != src.nodata
                else:
                    valid_mask = ~np.isnan(image)
                
                # Update statistics for each band
                for band_idx in range(num_bands):
                    band_data = image[band_idx][valid_mask[band_idx]]
                    if len(band_data) == 0:
                        continue
                        
                    # Remove outliers (optional)
                    # q1, q3 = np.percentile(band_data, [25, 75])
                    # iqr = q3 - q1
                    # band_data = band_data[
                    #     (band_data >= q1 - 1.5 * iqr) & 
                    #     (band_data <= q3 + 1.5 * iqr)
                    # ]
                    
                    n = len(band_data)
                    if n == 0:
                        continue
                        
                    delta = band_data - means[band_idx]
                    means[band_idx] += np.sum(delta) / (total_pixels + n)
                    delta2 = band_data - means[band_idx]
                    M2s[band_idx] += np.sum(delta * delta2)
                    
                total_pixels += n
                
        except Exception as e:
            logger.error(f"Error processing {filepath}: {str(e)}")
            continue
    
    # Calculate final standard deviations
    stds = np.sqrt(M2s / (total_pixels - 1))
    
    return means, stds

def save_stats(means: np.ndarray, stds: np.ndarray, output_file: Path):
    """Save computed statistics to a file."""
    with output_file.open('w') as f:
        f.write("Band Statistics:\n")
        f.write("-" * 40 + "\n")
        for i, (mean, std) in enumerate(zip(means, stds), 1):
            f.write(f"Band {i}:\n")
            f.write(f"  Mean: {mean:.6f}\n")
            f.write(f"  Std:  {std:.6f}\n")
        f.write("\n")
        f.write("Config Format:\n")
        f.write("PIXEL_MEAN: " + str(means.tolist()) + "\n")
        f.write("PIXEL_STD: " + str(stds.tolist()) + "\n")

def main():
    parser = argparse.ArgumentParser(description='Compute per-band mean and std for GeoTIFFs')
    parser.add_argument('image_dir', type=str, help='Directory containing GeoTIFF files')
    parser.add_argument('--output', type=str, default='band_stats.txt',
                      help='Output file for statistics (default: band_stats.txt)')
    args = parser.parse_args()
    
    logger = setup_logger()
    image_dir = Path(args.image_dir)
    output_file = Path(args.output)
    
    if not image_dir.exists():
        raise ValueError(f"Directory {image_dir} does not exist")
    
    try:
        logger.info("Computing band statistics...")
        means, stds = compute_band_stats(image_dir, logger)
        
        logger.info("Saving results...")
        save_stats(means, stds, output_file)
        
        logger.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise

if __name__ == '__main__':
    main()