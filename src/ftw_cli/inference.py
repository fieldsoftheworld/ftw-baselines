import math
import os
import tempfile
import time

import fiona
import fiona.transform
import kornia.augmentation as K
import numpy as np
import odc.stac
import planetary_computer as pc
import pystac
import rasterio
import rasterio.features
import rioxarray  # seems unused but is needed
import shapely.geometry
import torch
from affine import Affine
from kornia.constants import Resample
from pyproj import CRS
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from torchgeo.samplers import GridGeoSampler
from tqdm import tqdm

from ftw.datamodules import preprocess
from ftw.datasets import SingleRasterDataset
from ftw.trainers import CustomSemanticSegmentationTask

from .cfg import BANDS_OF_INTEREST, COLLECTION_ID, MSPC_URL, SUPPORTED_POLY_FORMATS_TXT


def get_item(id):
    if "/" not in id:
        uri = MSPC_URL + "/collections/" + COLLECTION_ID + "/items/" + id
    else:
        uri = id
    
    item = pystac.Item.from_file(uri)

    if uri.startswith(MSPC_URL):
        item = pc.sign(item)

    return item


def create_input(win_a, win_b, out, overwrite):
    """Main function for creating input for inference."""
    if os.path.exists(out) and not overwrite:
        print("Output file already exists, use -f to overwrite them. Exiting.")
        return

    # Ensure that the base directory exists
    os.makedirs(os.path.dirname(out), exist_ok=True)

    item_win_a = get_item(win_a)
    item_win_b = get_item(win_b)

    # TODO: Check that items are spatially aligned, or implement a way to only download the intersection

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_win_a_fn = os.path.join(tmpdirname, "tmp_win_a.tif")
        tmp_win_b_fn = os.path.join(tmpdirname, "tmp_win_b.tif")

        print("Loading window A data")
        tic = time.time()
        data = odc.stac.stac_load(
            [item_win_a],
            bands=BANDS_OF_INTEREST,
            dtype="uint16",
            resampling="bilinear",
        ).isel(time=0)

        data.rio.to_raster(
            tmp_win_a_fn,
            driver="GTiff",
            dtype="uint16",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        print(f"Finished saving window A to file in {time.time()-tic:0.2f} seconds")

        print("Loading window B data")
        tic = time.time()
        data = odc.stac.stac_load(
            [item_win_b],
            bands=BANDS_OF_INTEREST,
            dtype="uint16",
            resampling="bilinear",
        ).isel(time=0)

        data.rio.to_raster(
            tmp_win_b_fn,
            driver="GTiff",
            dtype="uint16",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        print(f"Finished saving window B to file in {time.time()-tic:0.2f} seconds")

        print("Merging data and writing output")
        tic = time.time()
        with rasterio.open(tmp_win_a_fn) as f:
            profile = f.profile
            data1 = f.read()

        with rasterio.open(tmp_win_b_fn) as f:
            data2 = f.read()

        data = np.concatenate([data1, data2], axis=0)
        profile["count"] = data.shape[0]
        profile["compress"] = "deflate"
        profile["tiled"] = True
        profile["blockxsize"] = 256
        profile["blockysize"] = 256
        profile["BIGTIFF"] = "YES"

        with rasterio.open(out, "w", **profile) as f:
            f.write(data)
        print(f"Finished merging and writing output in {time.time()-tic:0.2f} seconds")


def run(input, model, out, resize_factor, gpu, patch_size, batch_size, padding, overwrite, mps_mode):
    # Sanity checks
    assert os.path.exists(model), f"Model file {model} does not exist."
    assert model.endswith(".ckpt"), "Model file must be a .ckpt file."
    assert os.path.exists(input), f"Input file {input} does not exist."
    assert input.endswith(".tif") or input.endswith(".vrt"), "Input file must be a .tif or .vrt file."
    assert int(math.log(patch_size, 2)) == math.log(patch_size, 2), "Patch size must be a power of 2."

    stride = patch_size - padding * 2

    if os.path.exists(out) and not overwrite:
        print(f"Output file {out} already exists. Use -f to overwrite.")
        return

    # Determine the device: GPU, MPS, or CPU
    if mps_mode:
        assert torch.backends.mps.is_available(), "MPS mode is not available."
        device = torch.device("mps")
    elif gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    else:
        print("Neither GPU nor MPS mode is enabled, defaulting to CPU.")
        device = torch.device("cpu")

    # Load task and data
    tic = time.time()
    task = CustomSemanticSegmentationTask.load_from_checkpoint(model, map_location="cpu")
    task.freeze()
    model = task.model.eval().to(device)

    if mps_mode:
        up_sample = K.Resize((patch_size * resize_factor, patch_size * resize_factor)).to("cpu")
        down_sample = K.Resize((patch_size, patch_size), resample=Resample.NEAREST.name).to(device).to("cpu")
    else:
        up_sample = K.Resize((patch_size * resize_factor, patch_size * resize_factor)).to(device)
        down_sample = K.Resize((patch_size, patch_size), resample=Resample.NEAREST.name).to(device)

    dataset = SingleRasterDataset(input, transforms=preprocess)
    sampler = GridGeoSampler(dataset, size=patch_size, stride=stride)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=6, collate_fn=stack_samples)

    # Run inference
    with rasterio.open(input) as f:
        input_height, input_width = f.shape
        profile = f.profile
        transform = profile["transform"]

    output_mask = np.zeros((input_height, input_width), dtype=np.uint8)
    dl_enumerator = tqdm(dataloader)

    for batch in dl_enumerator:
        images = batch["image"].to(device)
        images = up_sample(images)

        # torchgeo>=0.6 refers to the bounding box as "bounds" instead of "bbox"
        if "bounds" in batch and batch["bounds"] is not None:
            bboxes = batch["bounds"]
        else:
            bboxes = batch["bbox"]

        with torch.inference_mode():
            predictions = model(images)
            predictions = predictions.argmax(axis=1).unsqueeze(0)
            predictions = down_sample(predictions.float()).int().cpu().numpy()[0]

        for i in range(len(bboxes)):
            bb = bboxes[i]
            left, top = ~transform * (bb.minx, bb.maxy)
            right, bottom = ~transform * (bb.maxx, bb.miny)
            left, right, top, bottom = int(np.round(left)), int(np.round(right)), int(np.round(top)), int(np.round(bottom))
            destination_height, destination_width = output_mask[top + padding:bottom - padding, left + padding:right - padding].shape

            inp = predictions[i][padding:padding + destination_height, padding:padding + destination_width]
            output_mask[top + padding:bottom - padding, left + padding:right - padding] = inp

    # Save predictions
    profile.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": "uint8",
        "compress": "lzw",
        "nodata": 0,
        "blockxsize": 512,
        "blockysize": 512,
        "tiled": True,
        "interleave": "pixel"
    })

    with rasterio.open(out, "w", **profile) as f:
        f.write(output_mask, 1)

    print(f"Finished inference and saved output to {out} in {time.time() - tic:.2f}s")


def polygonize(input, out, simplify, min_size, overwrite, close_interiors):
    """Polygonize the output from inference."""

    print(f"Polygonizing input file: {input}")

    # TODO: Get this warning working right, based on the CRS of the input file
    # if simplify is not None and simplify > 1:
    #    print("WARNING: You are passing a value of `simplify` greater than 1 for a geographic coordinate system. This is probably **not** what you want.")

    if os.path.exists(out) and not overwrite:
        print(f"Output file {out} already exists. Use -f to overwrite.")
        return
    elif os.path.exists(out) and overwrite:
        os.remove(out)  # GPKGs are sometimes weird about overwriting in-place

    tic = time.time()
    rows = []
    schema = {'geometry': 'Polygon', 'properties': {'id': 'str', 'area': 'float'}}
    i = 1
    # read the input file as a mask
    with rasterio.open(input) as src:
        input_height, input_width = src.shape
        original_crs = src.crs.to_string()
        is_meters = src.crs.linear_units in ["m", "metre", "meter"]
        transform = src.transform 
        mask = (src.read(1) == 1).astype(np.uint8)
        polygonization_stride = 10980
        total_iterations = math.ceil(input_height / polygonization_stride) * math.ceil(input_width / polygonization_stride)
        
        # Define the equal-area projection using EPSG:6933
        equal_area_crs = CRS.from_epsg(6933)

        if out.endswith(".gpkg"):
            format = "GPKG"
        elif out.endswith(".parquet"):
            format = "Parquet"
        elif out.endswith(".fgb"):
            format = "FlatGeobuf"
        elif out.endswith(".geojson") or out.endswith(".json"):
            format = "GeoJSON"
        else:
            raise ValueError("Output format not supported. " + SUPPORTED_POLY_FORMATS_TXT)

        with (
            fiona.open(out, 'w', format, schema=schema, crs=original_crs) as dst,
            tqdm(total=total_iterations, desc="Processing mask windows") as pbar
        ):
            for y in range(0, input_height, polygonization_stride):
                for x in range(0, input_width, polygonization_stride):
                    new_transform = transform * Affine.translation(x, y)
                    mask_window = mask[y:y+polygonization_stride, x:x+polygonization_stride]
                    rows = []
                    for geom_geojson, val in rasterio.features.shapes(mask_window, transform=new_transform):
                        if val != 1:
                            continue
                            
                        geom = shapely.geometry.shape(geom_geojson)

                        if close_interiors:
                            geom = shapely.geometry.Polygon(geom.exterior)
                        if simplify is not None:
                            geom = geom.simplify(simplify)
                        
                        # Calculate the area of the reprojected geometry
                        if is_meters:
                            area = geom.area
                        else:
                            # Reproject the geometry to the equal-area projection
                            # if the CRS is not in meters
                            geom_area_proj = fiona.transform.transform_geom(
                                original_crs, equal_area_crs, geom_geojson
                            )
                            area = shapely.geometry.shape(geom_area_proj).area  
                        
                        # Only include geometries that meet the minimum size requirement
                        if area < min_size:
                            continue

                        # Keep the geometry in the original projection for output
                        geom = shapely.geometry.mapping(geom)

                        rows.append({
                            "geometry": geom,
                            "properties": {
                                "id": str(i),
                                "area": area  # Add the area in m² to the properties
                            }
                        })
                        i += 1
                    
                    dst.writerecords(rows)
                    pbar.update(1)

    print(f"Finished polygonizing output at {out} in {time.time() - tic:.2f}s")
