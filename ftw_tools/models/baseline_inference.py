import math
import os
import time

import kornia.augmentation as K
import numpy as np
import rasterio
import torch
import torchgeo
from einops import rearrange
from packaging.version import Version, parse
from kornia.constants import Resample
from rasterio.enums import ColorInterp
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from torchgeo.samplers import GridGeoSampler
from tqdm import tqdm
import shapely.geometry
import fiona

from ftw_tools.torchgeo.datamodules import preprocess
from ftw_tools.torchgeo.datasets import SingleRasterDataset
from ftw_tools.torchgeo.trainers import CustomSemanticSegmentationTask

TORCHGEO_06 = Version("0.6.0")
TORCHGEO_08 = Version("0.8.0.dev0")
TORCHGEO_CURRENT = parse(torchgeo.__version__)

def run(
    input,
    model,
    out,
    resize_factor,
    gpu,
    patch_size,
    batch_size,
    padding,
    overwrite,
    mps_mode,
):
    if not out:
        out = os.path.join(
            os.path.dirname(input), "inference." + os.path.basename(input)
        )
    if gpu is None:
        gpu = -1

    # IO related sanity checks
    assert os.path.exists(model), f"Model file {model} does not exist."
    assert model.endswith(".ckpt"), "Model file must be a .ckpt file."
    assert os.path.exists(input), f"Input file {input} does not exist."
    assert input.endswith(".tif") or input.endswith(".vrt"), (
        "Input file must be a .tif or .vrt file."
    )
    assert overwrite or not os.path.exists(out), (
        f"Output file {out} already exists. Use -f to overwrite."
    )

    # Determine the device: GPU, MPS, or CPU
    if mps_mode:
        assert torch.backends.mps.is_available(), "MPS mode is not available."
        device = torch.device("mps")
    elif torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f"cuda:{gpu}")
    else:
        print("Neither GPU nor MPS mode is enabled, defaulting to CPU.")
        device = torch.device("cpu")

    # Load the input raster
    with rasterio.open(input) as src:
        input_height, input_width = src.shape
        print(f"Input image size: {input_height}x{input_width} pixels (HxW)")
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
    print("Patch size:", patch_size)
    assert patch_size is not None, "Input image is too small"
    assert patch_size % 32 == 0, "Patch size must be a multiple of 32."
    assert patch_size <= min(input_height, input_width), (
        "Patch size must not be larger than the input image dimensions."
    )

    if padding is None:
        # 64 for patch sizes >= 1024, otherwise smaller paddings
        padding = math.ceil(min(1024, patch_size) / 16)
    print("Padding:", padding)

    stride = patch_size - padding * 2
    assert stride > 64, (
        "Patch size minus two times the padding must be greater than 64."
    )

    # Load task
    tic = time.time()
    task = CustomSemanticSegmentationTask.load_from_checkpoint(
        model, map_location="cpu"
    )
    task.freeze()
    model = task.model.eval().to(device)
    model_type = task.hparams["model"]

    if mps_mode:
        up_sample = K.Resize(
            (patch_size * resize_factor, patch_size * resize_factor)
        ).to("cpu")
        down_sample = (
            K.Resize((patch_size, patch_size), resample=Resample.NEAREST.name)
            .to(device)
            .to("cpu")
        )
    else:
        up_sample = K.Resize(
            (patch_size * resize_factor, patch_size * resize_factor)
        ).to(device)
        down_sample = K.Resize(
            (patch_size, patch_size), resample=Resample.NEAREST.name
        ).to(device)

    dataset = SingleRasterDataset(input, transforms=preprocess)
    sampler = GridGeoSampler(dataset, size=patch_size, stride=stride)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=6,
        collate_fn=stack_samples,
    )

    # Run inference
    output_mask = np.zeros((input_height, input_width), dtype=np.uint8)
    dl_enumerator = tqdm(dataloader)

    inference_geoms = []

    for batch in dl_enumerator:
        images = batch["image"]
        images = up_sample(images)

        if model_type in ["fcsiamdiff", "fcsiamconc", "fcsiamavg"]:
            images = rearrange(images, "b (t c) h w -> b t c h w", t=2)
        images = images.to(device)

        # torchgeo>=0.8 switched from BoundingBox to slices
        # torchgeo>=0.6 refers to the bounding box as "bounds" instead of "bbox"
        bboxes = []
        if TORCHGEO_CURRENT >= TORCHGEO_08:
            for slices in batch["bounds"]:
                minx = slices[0].start
                maxx = slices[0].stop
                miny = slices[1].start
                maxy = slices[1].stop
                bboxes.append((minx, miny, maxx, maxy))
        elif TORCHGEO_CURRENT >= TORCHGEO_06:
            for bbox in batch["bounds"]:
                bboxes.append((bbox.minx, bbox.miny, bbox.maxx, bbox.maxy))
        else:
            for bbox in batch["bbox"]:
                bboxes.append((bbox.minx, bbox.miny, bbox.maxx, bbox.maxy))

        with torch.inference_mode():
            predictions = model(images)
            predictions = predictions.argmax(axis=1).unsqueeze(0)
            predictions = down_sample(predictions.float()).int().cpu().numpy()[0]

        for i in range(len(bboxes)):
            minx, miny, maxx, maxy = bboxes[i]

            # Save the polygon of this patch for debugging/visualization
            geom = shapely.geometry.mapping(shapely.geometry.box(minx, miny, maxx, maxy))
            inference_geoms.append(geom)

            left, bottom = ~transform * (minx, miny)
            right, top = ~transform * (maxx, maxy)
            left, right, top, bottom = (
                int(np.round(left)),
                int(np.round(right)),
                int(np.round(top)),
                int(np.round(bottom)),
            )

            # Determine per-side effective padding (no padding when on image border)
            effective_left_pad = 0 if left <= 0 else padding
            effective_right_pad = 0 if right >= input_width else padding
            effective_top_pad = 0 if top <= 0 else padding
            effective_bottom_pad = 0 if bottom >= input_height else padding

            # Interior (after trimming padding) in destination image coordinates
            pleft = left + effective_left_pad
            pright = right - effective_right_pad
            ptop = top + effective_top_pad
            pbottom = bottom - effective_bottom_pad

            # Clamp to image bounds to avoid negative or overflow indices
            dst_left = max(pleft, 0)
            dst_top = max(ptop, 0)
            dst_right = min(pright, input_width)
            dst_bottom = min(pbottom, input_height)

            # Source indices within prediction patch.
            src_left = effective_left_pad + (dst_left - pleft)
            src_right = effective_left_pad + (dst_right - pleft)
            src_top = effective_top_pad + (dst_top - ptop)
            src_bottom = effective_top_pad + (dst_bottom - ptop)

            h, w = predictions[i].shape
            src_left = max(0, min(src_left, w))
            src_right = max(0, min(src_right, w))
            src_top = max(0, min(src_top, h))
            src_bottom = max(0, min(src_bottom, h))
            if src_right <= src_left or src_bottom <= src_top:
                continue

            inp = predictions[i][src_top:src_bottom, src_left:src_right]
            output_mask[dst_top:dst_bottom, dst_left:dst_right] = inp

    # Some code to save prediction footprints
    # with fiona.open("inference_footprints.geojson", "w", driver="GeoJSON", crs=profile["crs"], schema={"geometry": "Polygon", "properties": {}}) as dst:
    #     for geom in inference_geoms:
    #         dst.write({"geometry": geom, "properties": {}})

    # Save predictions
    profile.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": "uint8",
            "compress": "lzw",
            "nodata": 0,
            "blockxsize": 512,
            "blockysize": 512,
            "tiled": True,
            "interleave": "pixel",
        }
    )

    with rasterio.open(out, "w", **profile) as dst:
        dst.update_tags(**tags)
        dst.write_colormap(1, {1: (255, 0, 0), 2: (0, 255, 0)})
        dst.colorinterp = [ColorInterp.palette]
        dst.write(output_mask, 1)

    print(f"Finished inference and saved output to {out} in {time.time() - tic:.2f}s")
