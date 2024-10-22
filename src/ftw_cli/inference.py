import os
import time
import tempfile
import click
import pystac
import planetary_computer as pc
import odc.stac
import rasterio
import numpy as np
import rioxarray
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import kornia.augmentation as K
from kornia.constants import Resample
from ftw.datasets import SingleRasterDataset
from ftw.trainers import CustomSemanticSegmentationTask
from torchgeo.samplers import GridGeoSampler
from torchgeo.datasets import stack_samples
from ftw.datamodules import preprocess

@click.group()
def inference():
    """Running inference on satellite images plus data prep."""
    pass

@inference.command(name="download", help="Download 2 Sentinel-2 scenes & stack them in a single file for inference.")
@click.option('--win_a', type=str, required=True, help="Path to a Sentinel-2 STAC item for the window A image")
@click.option('--win_b', type=str, required=True, help="Path to a Sentinel-2 STAC item for the window B image")
@click.option('--output_fn', type=str, required=True, help="Filename to save results to")
@click.option('--overwrite', is_flag=True, help="Overwrites the outputs if they exist")
def create_input(win_a, win_b, output_fn, overwrite):
    """Main function for creating input for inference."""
    if os.path.exists(output_fn) and not overwrite:
        print("Output file already exists, use --overwrite to overwrite them. Exiting.")
        return

    # Ensure that the base directory exists
    os.makedirs(os.path.dirname(output_fn), exist_ok=True)

    BANDS_OF_INTEREST = ["B04", "B03", "B02", "B08"]

    item_win_a = pc.sign(pystac.Item.from_file(win_a))
    item_win_b = pc.sign(pystac.Item.from_file(win_b))

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

        with rasterio.open(output_fn, "w", **profile) as f:
            f.write(data)
        print(f"Finished merging and writing output in {time.time()-tic:0.2f} seconds")

@inference.command(name="run", help="Run inference on the stacked satellite images")
@click.option('--input_fn', type=click.Path(exists=True), required=True, help="Input raster file (Sentinel-2 L2A stack).")
@click.option('--model_fn', type=click.Path(exists=True), required=True, help="Path to the model checkpoint.")
@click.option('--output_fn', type=str, required=True, help="Output filename.")
@click.option('--resize_factor', type=int, default=2, help="Resize factor to use for inference.")
@click.option('--gpu', type=int, help="GPU ID to use. If not provided, CPU will be used by default.")
@click.option('--patch_size', type=int, default=1024, help="Size of patch to use for inference.")
@click.option('--batch_size', type=int, default=2, help="Batch size.")
@click.option('--padding', type=int, default=64, help="Pixels to discard from each side of the patch.")
@click.option('--overwrite', is_flag=True, help="Overwrite outputs if they exist.")
@click.option('--mps_mode', is_flag=True, help="Run inference in MPS mode (Apple GPUs).")
def run(input_fn, model_fn, output_fn, resize_factor, gpu, patch_size, batch_size, padding, overwrite, mps_mode):
    # Sanity checks
    assert os.path.exists(model_fn), f"Model file {model_fn} does not exist."
    assert model_fn.endswith(".ckpt"), "Model file must be a .ckpt file."
    assert os.path.exists(input_fn), f"Input file {input_fn} does not exist."
    assert input_fn.endswith(".tif") or input_fn.endswith(".vrt"), "Input file must be a .tif or .vrt file."
    assert int(math.log(patch_size, 2)) == math.log(patch_size, 2), "Patch size must be a power of 2."

    stride = patch_size - padding * 2

    if os.path.exists(output_fn) and not overwrite:
        print(f"Output file {output_fn} already exists. Use --overwrite to overwrite.")
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
    task = CustomSemanticSegmentationTask.load_from_checkpoint(model_fn, map_location="cpu")
    task.freeze()
    model = task.model.eval().to(device)

    if mps_mode:
        up_sample = K.Resize((patch_size * resize_factor, patch_size * resize_factor)).to("cpu")
        down_sample = K.Resize((patch_size, patch_size), resample=Resample.NEAREST.name).to(device).to("cpu")
    else:
        up_sample = K.Resize((patch_size * resize_factor, patch_size * resize_factor)).to(device)
        down_sample = K.Resize((patch_size, patch_size), resample=Resample.NEAREST.name).to(device)

    dataset = SingleRasterDataset(input_fn, transforms=preprocess)
    sampler = GridGeoSampler(dataset, size=patch_size, stride=stride)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=6, collate_fn=stack_samples)

    # Run inference
    with rasterio.open(input_fn) as f:
        input_height, input_width = f.shape
        profile = f.profile
        transform = profile["transform"]

    output = np.zeros((input_height, input_width), dtype=np.uint8)
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
            destination_height, destination_width = output[top + padding:bottom - padding, left + padding:right - padding].shape

            inp = predictions[i][padding:padding + destination_height, padding:padding + destination_width]
            output[top + padding:bottom - padding, left + padding:right - padding] = inp

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

    with rasterio.open(output_fn, "w", **profile) as f:
        f.write(output, 1)
        f.write_colormap(1, {1: (255, 0, 0)})
        f.colorinterp = [ColorInterp.palette]

    print(f"Finished inference and saved output to {output_fn} in {time.time() - tic:.2f}s")
