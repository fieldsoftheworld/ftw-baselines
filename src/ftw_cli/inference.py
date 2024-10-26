import math
import os
import tempfile
import time

import click
import fiona
import fiona.transform
import kornia.augmentation as K
import numpy as np
import odc.stac
import planetary_computer as pc
import pystac
import rasterio
import rasterio.features
import rioxarray # seems unused but is needed
import shapely.geometry
import torch
from affine import Affine
from kornia.constants import Resample
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from torchgeo.samplers import GridGeoSampler
from tqdm import tqdm

from ftw.datamodules import preprocess
from ftw.datasets import SingleRasterDataset
from ftw.trainers import CustomSemanticSegmentationTask


MSPC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION_ID = "sentinel-2-l2a"

@click.group()
def inference():
    """Inference-related commands."""
    pass

def get_item(id):
    if "/" not in id:
        uri = MSPC_URL + "/collections/" + COLLECTION_ID + "/items/" + id
    else:
        uri = id
    
    item = pystac.Item.from_file(uri)

    if uri.startswith(MSPC_URL):
        item = pc.sign(item)

    return item

WIN_HELP = "URL to or Microsoft Planetary Computer ID of an Sentinel-2 L2A STAC item for the window {x} image"

@inference.command(name="download", help="Download 2 Sentinel-2 scenes & stack them in a single file for inference.")
@click.option('--win_a', type=str, required=True, help=WIN_HELP.format(x="A"))
@click.option('--win_b', type=str, required=True, help=WIN_HELP.format(x="B"))
@click.option('--out', '-o', type=str, required=True, help="Filename to save results to")
@click.option('--overwrite', '-f', is_flag=True, help="Overwrites the outputs if they exist")
def create_input(win_a, win_b, out, overwrite):
    """Main function for creating input for inference."""
    if os.path.exists(out) and not overwrite:
        print("Output file already exists, use -f to overwrite them. Exiting.")
        return

    # Ensure that the base directory exists
    os.makedirs(os.path.dirname(out), exist_ok=True)

    BANDS_OF_INTEREST = ["B04", "B03", "B02", "B08"]

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

@inference.command(name="run", help="Run inference on the stacked Sentinel-2 L2A satellite images specified via INPUT.")
@click.argument('input', type=click.Path(exists=True), required=True)
@click.option('--model', '-m', type=click.Path(exists=True), required=True, help="Path to the model checkpoint.")
@click.option('--out', '-o', type=str, required=True, help="Output filename.")
@click.option('--resize_factor', type=int, default=2, help="Resize factor to use for inference.")
@click.option('--gpu', type=int, help="GPU ID to use. If not provided, CPU will be used by default.")
@click.option('--patch_size', type=int, default=1024, help="Size of patch to use for inference.")
@click.option('--batch_size', type=int, default=2, help="Batch size.")
@click.option('--padding', type=int, default=64, help="Pixels to discard from each side of the patch.")
@click.option('--overwrite', '-f', is_flag=True, help="Overwrite outputs if they exist.")
@click.option('--mps_mode', is_flag=True, help="Run inference in MPS mode (Apple GPUs).")
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

    out = np.zeros((input_height, input_width), dtype=np.uint8)
    dl_enumerator = tqdm(dataloader)

    for batch in dl_enumerator:
        images = batch["image"].to(device)
        images = up_sample(images)
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
            destination_height, destination_width = out[top + padding:bottom - padding, left + padding:right - padding].shape

            inp = predictions[i][padding:padding + destination_height, padding:padding + destination_width]
            out[top + padding:bottom - padding, left + padding:right - padding] = inp

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
        f.write(out, 1)
        f.write_colormap(1, {1: (255, 0, 0)})
        f.colorinterp = [ColorInterp.palette]

    print(f"Finished inference and saved output to {out} in {time.time() - tic:.2f}s")

@inference.command(name="polygonize", help="Polygonize the output from inference for the raster image given via INPUT.")
@click.argument('input', type=click.Path(exists=True), required=True)
@click.option('--out', '-o', type=str, required=True, help="Output filename for the polygonized data.")
@click.option('--simplify', type=float, default=None, help="Simplification factor to use when polygonizing.")
@click.option('--min_size', type=float, default=500, help="Minimum area size in square meters to include in the output.")
@click.option('--overwrite', '-f', is_flag=True, help="Overwrite outputs if they exist.")
def polygonize(input, out, simplify, min_size, overwrite):
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
    i = 0
    # read the input file as a mask
    with rasterio.open(input) as src:
        input_height, input_width = src.shape
        original_crs = src.crs.to_string()
        transform = src.transform 
        mask = (src.read(1) == 1).astype(np.uint8)
        polygonization_stride = 2048
        total_iterations = (input_height // polygonization_stride) * (input_width // polygonization_stride)
        
        # Define the equal-area projection using EPSG:6933
        equal_area_crs = CRS.from_epsg(6933)

        with tqdm(total=total_iterations, desc="Processing mask windows") as pbar:
            for y in range(0, input_height, polygonization_stride):
                for x in range(0, input_width, polygonization_stride):
                    new_transform = transform * Affine.translation(x, y)
                    mask_window = mask[y:y+polygonization_stride, x:x+polygonization_stride]
                    for geom, val in rasterio.features.shapes(mask_window, transform=new_transform):
                        if val == 1:
                            geom = shapely.geometry.shape(geom)
                            if simplify is not None:
                                geom = geom.simplify(simplify)
                            
                            # Convert the geometry to a GeoJSON-like format for transformation
                            geom_geojson = shapely.geometry.mapping(geom)
                            
                            # Reproject the geometry to the equal-area projection for accurate area calculation
                            geom_area_proj = fiona.transform.transform_geom(original_crs, equal_area_crs, geom_geojson)
                            geom_area_proj = shapely.geometry.shape(geom_area_proj)
                            area = geom_area_proj.area  # Calculate the area of the reprojected geometry
                            
                            # Only include geometries that meet the minimum size requirement
                            if area >= min_size:
                                # Keep the geometry in the original projection for output
                                geom = shapely.geometry.mapping(geom)

                                rows.append({
                                    "geometry": geom,
                                    "properties": {
                                        "idx": i,
                                        "area": area  # Add the area to the properties
                                    }
                                })
                                i += 1
                    pbar.update(1)

    schema = {'geometry': 'Polygon', 'properties': {'idx': 'int', 'area': 'float'}}
    with fiona.open(out, 'w', 'GPKG', schema=schema, crs=original_crs) as dst:
        for row in rows:
            dst.write(row)

    print(f"Finished polygonizing output at {out} in {time.time() - tic:.2f}s")
