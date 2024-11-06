import math
import os
import time

import kornia.augmentation as K
import numpy as np
import rasterio
import rasterio.features
import torch
from kornia.constants import Resample
from rasterio.enums import ColorInterp
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from torchgeo.samplers import GridGeoSampler
from tqdm import tqdm

from ftw.datamodules import preprocess
from ftw.datasets import SingleRasterDataset
from ftw.trainers import CustomSemanticSegmentationTask


def run(input, model, out, resize_factor, gpu, patch_size, batch_size, padding, overwrite, mps_mode):
    # IO related sanity checks
    assert os.path.exists(model), f"Model file {model} does not exist."
    assert model.endswith(".ckpt"), "Model file must be a .ckpt file."
    assert os.path.exists(input), f"Input file {input} does not exist."
    assert input.endswith(".tif") or input.endswith(".vrt"), "Input file must be a .tif or .vrt file."
    assert overwrite or os.path.exists(out), f"Output file {out} already exists. Use -f to overwrite."

    # Determine the device: GPU, MPS, or CPU
    if mps_mode:
        assert torch.backends.mps.is_available(), "MPS mode is not available."
        device = torch.device("mps")
    elif gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    else:
        print("Neither GPU nor MPS mode is enabled, defaulting to CPU.")
        device = torch.device("cpu")

    # Load the input raster
    with rasterio.open(input) as src:
        input_height, input_width = src.shape
        profile = src.profile
        transform = profile["transform"]
        tags = src.tags()

    # Determine the default patch size
    if patch_size is None:
        steps = [1024, 512, 256, 128]
        for step in steps:
            if step <= min(input_height, input_width):
                patch_size = step
                break
    stride = patch_size - padding * 2
    print("Patch size:", patch_size)
    
    assert patch_size is not None, "Input image is too small"
    assert patch_size % 32 == 0, "Patch size must be a multiple of 32."
    assert stride > 64, "Patch size minus two times the padding must be greater than 64."

    # Load task
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
            pleft = left + padding
            pright = right - padding
            ptop = top + padding
            pbottom = bottom - padding
            destination_height, destination_width = output_mask[ptop:pbottom, pleft:pright].shape
            inp = predictions[i][padding:padding + destination_height, padding:padding + destination_width]
            output_mask[ptop:pbottom, pleft:pright] = inp

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

    with rasterio.open(out, "w", **profile) as dst:
        dst.update_tags(**tags)
        dst.write(output_mask, 1)
        dst.write_colormap(1, {1: (255, 0, 0), 2:(0, 255, 0)})
        dst.colorinterp = [ColorInterp.palette]

    print(f"Finished inference and saved output to {out} in {time.time() - tic:.2f}s")
