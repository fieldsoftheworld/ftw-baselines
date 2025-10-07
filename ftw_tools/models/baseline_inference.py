import math
import os
import time
from pathlib import Path
from typing import Literal

import geopandas as gpd
import kornia.augmentation as K
import numpy as np
import pandas as pd
import rasterio
import shapely.geometry
import torch
import torch.nn.functional as F
import torchgeo
from einops import rearrange
from kornia.constants import Resample
from packaging.version import Version, parse
from rasterio.enums import ColorInterp
from rasterio.transform import from_bounds
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from torchgeo.samplers import GridGeoSampler
from tqdm import tqdm

from ftw_tools.models.model_registry import MODEL_REGISTRY
from ftw_tools.models.utils import convert_to_fiboa, postprocess_instance_polygons
from ftw_tools.torchgeo.datamodules import preprocess
from ftw_tools.torchgeo.datasets import SingleRasterDataset
from ftw_tools.torchgeo.trainers import CustomSemanticSegmentationTask

TORCHGEO_06 = Version("0.6.0")
TORCHGEO_08 = Version("0.8.0.dev0")
TORCHGEO_CURRENT = parse(torchgeo.__version__)


def setup_inference(input, out, gpu, patch_size, padding, overwrite, mps_mode):
    if not out:
        out = os.path.join(
            os.path.dirname(input), "inference." + os.path.basename(input)
        )
    if gpu is None:
        gpu = -1

    # IO related sanity checks
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
        input_shape = src.shape
        input_height, input_width = input_shape[0], input_shape[1]
        print(f"Input image size: {input_height}x{input_width} pixels (HxW)")
        profile = src.profile
        transform = profile["transform"]

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

    return device, transform, input_shape, patch_size, stride, padding


def run(
    input,
    model,
    out,
    resize_factor,
    gpu,
    patch_size,
    batch_size,
    num_workers,
    padding,
    overwrite,
    mps_mode,
    save_scores,
):
    device, transform, input_shape, patch_size, stride, padding = setup_inference(
        input, out, gpu, patch_size, padding, overwrite, mps_mode
    )

    # Load model
    if model in MODEL_REGISTRY.keys():
        print(f"Downloading model {model} from {MODEL_REGISTRY[model].url}")
        model_path = MODEL_REGISTRY[model].url

        # Ensure cache directory exists
        cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_ckpt_path = str(cache_dir / f"{model}.ckpt")

        # Only download if not already cached
        if not Path(model_ckpt_path).exists():
            torch.hub.download_url_to_file(model_path, model_ckpt_path, progress=True)

    else:
        assert model.endswith(".ckpt"), "Model file must be a .ckpt file."
        assert os.path.exists(model), f"Model file {model} does not exist."
        model_ckpt_path = model

    # Load task
    tic = time.time()
    task = CustomSemanticSegmentationTask.load_from_checkpoint(
        model_ckpt_path, map_location="cpu"
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
        num_workers=num_workers,
        collate_fn=stack_samples,
    )

    # Run inference
    input_height, input_width = input_shape[0], input_shape[1]
    if save_scores:
        out_channels = 3
    else:
        out_channels = 1
    output_mask = np.zeros([out_channels, input_height, input_width], dtype=np.uint8)
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
            if save_scores:
                # compute softmax to interpret logits as probabilities [0, 1]
                predictions = F.softmax(predictions, dim=1)
                predictions = (
                    down_sample(predictions.float()).cpu().numpy().astype(np.float32)
                )
                # rescale probabilities from [0, 1] to [0, 255] and store as uint8
                predictions = (predictions * 255).clip(0, 255).astype(np.uint8)
            else:
                predictions = predictions.argmax(axis=1).unsqueeze(1)
                predictions = down_sample(predictions.float()).int().cpu().numpy()

        for i in range(len(bboxes)):
            minx, miny, maxx, maxy = bboxes[i]

            # Save the polygon of this patch for debugging/visualization
            geom = shapely.geometry.mapping(
                shapely.geometry.box(minx, miny, maxx, maxy)
            )
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

            _, h, w = predictions[i].shape
            src_left = max(0, min(src_left, w))
            src_right = max(0, min(src_right, w))
            src_top = max(0, min(src_top, h))
            src_bottom = max(0, min(src_bottom, h))
            if src_right <= src_left or src_bottom <= src_top:
                continue

            inp = predictions[i, :, src_top:src_bottom, src_left:src_right]
            output_mask[:, dst_top:dst_bottom, dst_left:dst_right] = inp

    # Some code to save prediction footprints
    # with fiona.open("inference_footprints.geojson", "w", driver="GeoJSON", crs=profile["crs"], schema={"geometry": "Polygon", "properties": {}}) as dst:
    #     for geom in inference_geoms:
    #         dst.write({"geometry": geom, "properties": {}})

    with rasterio.open(input) as src:
        profile = src.profile
        tags = src.tags()

    # Save predictions
    profile.update(
        {
            "driver": "GTiff",
            "count": out_channels,
            "dtype": "uint8",
            "compress": "lzw",
            "predictor": 2,
            "blockxsize": 512,
            "blockysize": 512,
            "tiled": True,
            "interleave": "pixel",
        }
    )
    if save_scores:
        profile["nodata"] = None
    else:
        profile["nodata"] = 0  # background class

    with rasterio.open(out, "w", **profile) as dst:
        dst.update_tags(**tags)
        if save_scores:
            # write all logit channels
            dst.write(output_mask)
        else:
            # palette only for single-band labels
            dst.write_colormap(1, {1: (255, 0, 0), 2: (0, 255, 0)})
            dst.colorinterp = [ColorInterp.palette]
            dst.write(output_mask[0], 1)

    print(f"Finished inference and saved output to {out} in {time.time() - tic:.2f}s")


@torch.inference_mode()
def run_instance_segmentation(
    input: str,
    model: Literal["DelineateAnything-S", "DelineateAnything"],
    out: str,
    gpu: int | None = None,
    num_workers: int = 4,
    patch_size: int = 256,
    resize_factor: int = 2,
    batch_size: int = 4,
    max_detections: int = 100,
    iou_threshold: float = 0.3,
    conf_threshold: float = 0.05,
    padding: int | None = None,
    overwrite: bool = False,
    mps_mode: bool = False,
    simplify: int = 2,
    min_size: int | None = 500,
    max_size: int | None = None,
    close_interiors: bool = True,
    overlap_iou_threshold: float = 0.3,
    overlap_contain_threshold: float = 0.8,
):
    """Run instance segmentation inference on an image.

    Args:
        input: The input image file path.
        model: The model to use for inference.
        out: The output file path.
        gpu: The GPU device to use for inference.
        num_workers: The number of workers to use for inference.
        patch_size: The size of the patch to use for inference.
        resize_factor: The resize factor to use for inference.
        batch_size: The batch size to use for inference.
        max_detections: The maximum number of detections to use for inference.
        iou_threshold: The IoU threshold to use for inference (lower values filter out more detections).
        conf_threshold: The confidence threshold to use for inference (higher values filter out more detections).
        padding: The padding to use for inference (in pixels).
        overwrite: Whether to overwrite the output file if it already exists.
        mps_mode: Whether to use MPS mode for inference.
        simplify: The simplification factor to use for inference (in pixels).
        min_size: The minimum size of the polygons to use for inference (in hectares).
        max_size: The maximum size of the polygons to use for inference (in hectares).
        close_interiors: Whether to close the interiors of the polygons to use for inference.
        overlap_iou_threshold: Merge polygons with IoU greater than this threshold.
        overlap_contain_threshold: Merge polygons with contain greater than this threshold.

    Raises:
        AssertionError: If the model is not DelineateAnything or DelineateAnything-S.

    Returns:
        None
    """
    from .delineate_anything import DelineateAnything

    assert model in ["DelineateAnything", "DelineateAnything-S"], (
        "Model must be either DelineateAnything or DelineateAnything-S."
    )

    padding = padding if padding is not None else patch_size // 4
    device, _, _, patch_size, stride, _ = setup_inference(
        input, out, gpu, patch_size, padding, overwrite, mps_mode
    )

    # Load task
    tic = time.time()
    model = DelineateAnything(
        model=model,
        patch_size=patch_size,
        resize_factor=resize_factor,
        max_detections=max_detections,
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
        device=device,
    )

    # Load dataset
    dataset = SingleRasterDataset(input)
    sampler = GridGeoSampler(dataset, size=patch_size, stride=stride)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=stack_samples,
    )

    # Run inference
    polygons = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        images = batch["image"].to(device)
        predictions = model(images)

        # torchgeo>=0.6 refers to the bounding box as "bounds" instead of "bbox"
        if "bounds" in batch and batch["bounds"] is not None:
            bboxes = batch["bounds"]
        else:
            bboxes = batch["bbox"]

        # Convert instance predictions to polygons
        for image, pred, bounds, crs in zip(images, predictions, bboxes, batch["crs"]):
            _, h, w = image.shape
            transform = from_bounds(
                west=bounds.minx,
                south=bounds.miny,
                east=bounds.maxx,
                north=bounds.maxy,
                height=h,
                width=w,
            )
            polygons.append(model.polygonize(pred, transform, crs))

    polygons = [p for p in polygons if p is not None]
    polygons = gpd.GeoDataFrame(pd.concat(polygons), crs=dataset.crs)

    polygons = postprocess_instance_polygons(
        polygons=polygons,
        padding=padding,
        simplify=simplify,
        min_size=min_size,
        max_size=max_size,
        close_interiors=close_interiors,
        overlap_iou_threshold=overlap_iou_threshold,
        overlap_contain_threshold=overlap_contain_threshold,
    )

    # Save polygons
    ext = os.path.splitext(out)[1]
    if ext == ".parquet":
        with rasterio.open(input) as src:
            timestamp = src.tags().get("TIFFTAG_DATETIME", None)
        convert_to_fiboa(polygons, out, timestamp)
    elif ext == ".gpkg":
        polygons.to_file(out, driver="GPKG")
    elif ext == ".geojson":
        polygons.to_file(out, driver="GeoJSON")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    print(f"Finished inference and saved output to {out} in {time.time() - tic:.2f}s")
