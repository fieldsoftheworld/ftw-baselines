#!/usr/bin/env python3
"""
Convert FTW dataset to COCO detection format using instance masks, with HPC support.
Example usage:
python ftw_to_coco.py \
    --geotiff_dir /ftw/austria/s2_images/window_a \
    --instance_mask_dir /ftw/austria/label_masks/instance \
    --output_dir /ftw/austria/coco \
    --chips /ftw/austria/chips_austria.parquet \
    --num_workers $SLURM_CPUS_PER_TASK \
    --debug
"""

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.features import rasterize, shapes
from shapely.geometry import shape, mapping, box
import warnings
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import math
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
import fsspec
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InstanceInfo(NamedTuple):
    id: int
    mask: np.ndarray
    bbox: List[float]
    area: float

class COCOConverter:
    def __init__(
        self,
        output_dir: str,
        min_area: int = 100,
        num_workers: int = None,
        debug: bool = False
    ):
        self.output_dir = Path(output_dir)
        self.min_area = min_area
        self.num_workers = num_workers or mp.cpu_count()
        self.debug = debug
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
            
        self._setup_directories()
        
        # Define category for agriculture fields (just one category for now)
        self.categories = [
            {
                "id": 1,
                "name": "ag_field",
                "supercategory": "landcover"
            }
        ]

    def _setup_directories(self):
        """Create necessary output directories with proper permissions"""
        for path in [
            self.output_dir,
            self.output_dir / "annotations",
        ]:
            path.mkdir(exist_ok=True, parents=True)
            os.chmod(str(path), 0o775)

    def _load_split_data(self, chips_path: str) -> Dict[str, List[str]]:
        """Load split data from chips file"""
        logger.info("Loading split data from chips file...")
        
        # Read chips geoparquet
        try:
            chips_gdf = gpd.read_parquet(chips_path)
            logger.debug(f"Chips GeoDataFrame loaded: {chips_gdf.shape}")
            logger.debug(f"Chips columns: {chips_gdf.columns}")
        except Exception as e:
            logger.error(f"Failed to load chips file: {e}")
            raise
        
        # Create dictionary of split -> image IDs
        splits = {}
        for split in ['train', 'val', 'test']:
            mask = chips_gdf['split'] == split
            splits[split] = chips_gdf.loc[mask, 'aoi_id'].tolist()
        
        logger.info(f"Split statistics:")
        for split, ids in splits.items():
            logger.info(f"  {split}: {len(ids)} images")
            
        return splits

    def _create_instance(
        self,
        geom,
        transform: rio.transform.Affine,
        height: int,
        width: int,
        instance_id: int
    ) -> Optional[InstanceInfo]:
        """Create instance from geometry"""
        try:
            mask = rasterize(
                [(geom, 1)],
                out_shape=(height, width),
                transform=transform,
                dtype=np.uint8,
                all_touched=True
            )

            if mask is None or not mask.any() or float(mask.sum()) < self.min_area:
                return None
            
            # Compute bbox
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not rows.any() or not cols.any():
                return None
                
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            
            return InstanceInfo(
                id=instance_id,
                mask=mask,
                bbox=[float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)],
                area=float(mask.sum())
            )
            
        except Exception as e:
            warnings.warn(f"Failed to create instance: {str(e)}")
            return None

    def _get_geometries_from_instance_mask(
        self, 
        instance_mask_path: str
    ) -> List[Dict]:
        """
        Extract geometries from an instance mask file
        
        Args:
            instance_mask_path: Path to instance mask GeoTIFF
            
        Returns:
            List of geometries with properties
        """
        if not Path(instance_mask_path).exists():
            logger.warning(f"Instance mask not found: {instance_mask_path}")
            return []
        
        try:
            with rio.open(instance_mask_path) as src:
                # Read the first band (instance mask)
                instance_data = src.read()[0]
                transform = src.transform
                
                # Get unique instance IDs (excluding 0 which is background)
                instance_ids = np.unique(instance_data)
                instance_ids = instance_ids[instance_ids > 0]
                
                if len(instance_ids) == 0:
                    logger.warning(f"No instances found in {instance_mask_path}")
                    logger.warning(f"instance_data: {np.unique(instance_data, return_counts=True)}")
                    return []
                
                logger.debug(f"Found {len(instance_ids)} instances in {instance_mask_path}")
                
                # Extract shapes for each instance ID
                geometries = []
                for instance_id in instance_ids:
                    # Create a binary mask for this instance
                    binary_mask = (instance_data == instance_id).astype(np.uint8)
                    
                    # Skip if area is too small
                    if binary_mask.sum() < self.min_area:
                        continue
                    
                    # Extract polygon shapes
                    for geom, value in shapes(binary_mask, mask=binary_mask > 0, transform=transform):
                        # Convert to shapely geometry
                        polygon = shape(geom)
                        
                        # Skip invalid geometries
                        if not polygon.is_valid or polygon.is_empty:
                            continue
                        
                        # Add to results
                        geometries.append({
                            'geometry': polygon,
                            'properties': {'instance_id': int(instance_id)}
                        })
                
                logger.debug(f"Extracted {len(geometries)} valid geometries from {instance_mask_path}")
                return geometries
                
        except Exception as e:
            logger.error(f"Error processing instance mask {instance_mask_path}: {str(e)}")
            return []

    def _process_single_image(
        self,
        geotiff_path: str,
        instance_mask_path: str,
        image_id: int,
        aoi_id: str
    ) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
        """Process single image"""
        try:
            # Check if instance mask exists
            if not Path(instance_mask_path).exists():
                logger.warning(f"Instance mask not found: {instance_mask_path}")
                return None
                
            # Open and read image data
            with rio.open(geotiff_path) as src:
                height, width = src.height, src.width
                transform = src.transform
                bounds = src.bounds
                
                # Create bbox from bounds
                image_bbox = (bounds.left, bounds.bottom, bounds.right, bounds.top)
            
            # Get geometries from instance mask
            geometries = self._get_geometries_from_instance_mask(instance_mask_path)
            
            # Skip if no fields
            if not geometries:
                logger.warning(f"No field geometries found for {geotiff_path}")
                return None
            
            instance_anns = []
            next_instance_id = 1  # Start with instance ID 1
            
            # Create instances from geometries
            for geom_data in geometries:
                geom = geom_data['geometry']
                
                instance = self._create_instance(
                    geom, transform, height, width, next_instance_id
                )
                if instance is None:
                    continue
                
                instance_ann = {
                    "id": next_instance_id,
                    "image_id": image_id,
                    "category_id": 1,  # Only one category - "ag_field"
                    "area": instance.area,
                    "bbox": instance.bbox,
                    "iscrowd": 0
                }
                instance_anns.append(instance_ann)
                next_instance_id += 1
            
            if not instance_anns:
                logger.warning(f"No valid instances found in {geotiff_path}")
                return None
            
            image_info = {
                "file_name": os.path.basename(geotiff_path),
                "id": image_id,
                "height": height,
                "width": width,
                "aoi_id": aoi_id
            }
            
            return image_info, instance_anns
            
        except Exception as e:
            logger.error(f"Error processing {geotiff_path}: {str(e)}")
            return None

    def _chunk_data(self, data_list: List[str], chunk_size: int) -> List[List[str]]:
        """Split data into chunks for parallel processing"""
        n_chunks = math.ceil(len(data_list) / chunk_size)
        return [
            data_list[i * chunk_size:(i + 1) * chunk_size]
            for i in range(n_chunks)
        ]

    def _process_chunk(
        self,
        chunk: List[str],
        geotiff_dir: str,
        instance_mask_dir: str,
        chunk_id: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """Process a chunk of images by aoi_id"""
        images = []
        instance_anns = []
        failed_images = []
        
        base_id = chunk_id * len(chunk) * 1000  # Ensure unique IDs across chunks
        
        for idx, aoi_id in enumerate(chunk):
            try:
                # Construct the path to the geotiff and instance mask
                geotiff_path = Path(geotiff_dir) / f"{aoi_id}.tif"
                instance_mask_path = Path(instance_mask_dir) / f"{aoi_id}.tif"
                
                if not geotiff_path.exists():
                    failed_images.append(f"Geotiff not found: {geotiff_path}")
                    continue
                    
                if not instance_mask_path.exists():
                    failed_images.append(f"Instance mask not found: {instance_mask_path}")
                    continue
                
                image_id = base_id + idx
                
                result = self._process_single_image(
                    str(geotiff_path), 
                    str(instance_mask_path), 
                    image_id,
                    aoi_id
                )
                
                if result:
                    image_info, inst_anns = result
                    images.append(image_info)
                    instance_anns.extend(inst_anns)
                else:
                    failed_images.append(f"No valid instances found in {geotiff_path}")
                    
            except Exception as e:
                failed_images.append(f"Error processing {aoi_id}: {str(e)}")
                continue
        
        # Save failed images for this chunk
        if failed_images:
            chunk_fail_file = Path(self.output_dir) / f"failed_images_chunk{chunk_id}.txt"
            with open(chunk_fail_file, "w") as f:
                f.write("\n".join(failed_images))
        
        return images, instance_anns

    def convert_dataset(
        self, 
        geotiff_dir: str, 
        instance_mask_dir: str, 
        chips_path: str
    ):
        """Convert dataset using parallel processing"""
        logger.info(f"Starting conversion with {self.num_workers} workers")
        
        # Load split data
        splits = self._load_split_data(chips_path)
        
        # Process each split
        for split, aoi_ids in splits.items():
            if not aoi_ids:
                logger.info(f"Skip {split} split: no images")
                continue
                
            logger.info(f"\nProcessing {split} split with {len(aoi_ids)} images")
            
            # Calculate chunk size
            chunk_size = math.ceil(len(aoi_ids) / (self.num_workers * 2))  # 2 chunks per worker
            chunks = self._chunk_data(aoi_ids, chunk_size)
            logger.info(f"Split data into {len(chunks)} chunks of ~{chunk_size} images each")
            
            # If in debug mode, limit to first chunk
            if self.debug and len(chunks) > 1:
                logger.debug("Debug mode: limiting to first chunk")
                chunks = [chunks[0]]
            
            # Initialize result containers
            all_images = []
            all_instance_anns = []
            
            # Create argument tuples for each chunk
            chunk_args = [
                (chunk, geotiff_dir, instance_mask_dir, i) 
                for i, chunk in enumerate(chunks)
            ]
            
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Map the function directly with arguments
                futures = [
                    executor.submit(self._process_chunk, *args)
                    for args in chunk_args
                ]
                
                for future in tqdm(futures, total=len(chunks), desc=f"Processing {split} chunks"):
                    try:
                        images, instance_anns = future.result()
                        all_images.extend(images)
                        all_instance_anns.extend(instance_anns)
                    except Exception as e:
                        logger.error(f"Chunk processing failed: {str(e)}")
            
            logger.info(f"\nConversion summary for {split} split:")
            logger.info(f"Total images in split: {len(aoi_ids)}")
            logger.info(f"Successfully processed: {len(all_images)}")
            logger.info(f"Failed to process: {len(aoi_ids) - len(all_images)}")
            
            # Skip empty splits
            if not all_images:
                logger.warning(f"No valid images processed for {split} split. Skipping...")
                continue
            
            # Create final dictionary
            instances_dict = {
                "images": all_images,
                "annotations": all_instance_anns,
                "categories": self.categories
            }
            
            # Save annotation file
            logger.info(f"Saving annotation file for {split} split...")
            with open(self.output_dir / "annotations" / f"instances_{split}.json", "w") as f:
                json.dump(instances_dict, f)
        
        logger.info("Conversion complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert Fields of the World dataset to COCO format with parallel processing"
    )
    parser.add_argument("--geotiff_dir", required=True, help="Directory with GeoTIFF image files")
    parser.add_argument("--instance_mask_dir", required=True, help="Directory with instance mask GeoTIFF files")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--chips", required=True, help="Path to chips GeoParquet with split information")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--min_area", type=int, default=10, help="Minimum instance area")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()

    logger.info("Parsed args: %s", args)
    
    converter = COCOConverter(
        args.output_dir,
        min_area=args.min_area,
        num_workers=args.num_workers,
        debug=args.debug
    )
    
    logger.info("Initialized converter")

    converter.convert_dataset(
        args.geotiff_dir,
        args.instance_mask_dir,
        args.chips
    )