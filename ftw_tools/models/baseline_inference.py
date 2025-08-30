import math
import os
import re
import time
from typing import Literal

import geopandas as gpd
import kornia.augmentation as K
import numpy as np
import pandas as pd
import rasterio
import shapely
import torch
from fiboa_cli.parquet import create_parquet
from kornia.constants import Resample
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
from rasterio.transform import from_bounds
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from torchgeo.samplers import GridGeoSampler
from tqdm import tqdm

from ftw_tools.torchgeo.datamodules import preprocess
from ftw_tools.torchgeo.datasets import SingleRasterDataset
from ftw_tools.torchgeo.trainers import CustomSemanticSegmentationTask


def setup_inference(
    input,
    out,
    gpu,
    patch_size,
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
):
    device, transform, input_shape, patch_size, stride, padding = setup_inference(
        input, out, gpu, patch_size, padding, overwrite, mps_mode
    )

    assert os.path.exists(model), f"Model file {model} does not exist."
    assert model.endswith(".ckpt"), "Model file must be a .ckpt file."

    # Load task
    tic = time.time()
    task = CustomSemanticSegmentationTask.load_from_checkpoint(
        model, map_location="cpu"
    )
    task.freeze()
    model = task.model.eval().to(device)

    if mps_mode:
        up_sample = K.Resize(
            (
                patch_size * resize_factor,
                patch_size * resize_factor,
            )
        ).to("cpu")
        down_sample = (
            K.Resize((patch_size, patch_size), resample=Resample.NEAREST.name)
            .to(device)
            .to("cpu")
        )
    else:
        up_sample = K.Resize(
            (
                patch_size * resize_factor,
                patch_size * resize_factor,
            )
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
    output_mask = np.zeros(input_shape, dtype=np.uint8)
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
            left, right, top, bottom = (
                int(np.round(left)),
                int(np.round(right)),
                int(np.round(top)),
                int(np.round(bottom)),
            )
            pleft = left + padding
            pright = right - padding
            ptop = top + padding
            pbottom = bottom - padding
            destination_height, destination_width = output_mask[
                ptop:pbottom, pleft:pright
            ].shape
            inp = predictions[i][
                padding : padding + destination_height,
                padding : padding + destination_width,
            ]
            output_mask[ptop:pbottom, pleft:pright] = inp

    with rasterio.open(input) as src:
        profile = src.profile
        tags = src.tags()

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


@torch.inference_mode()
def run_instance_segmentation(
    input: str,
    model: Literal["DelineateAnything-S", "DelineateAnything"],
    out: str,
    gpu: int | None = None,
    num_workers: int = 4,
    image_size: int = 320,
    patch_size: int = 256,
    batch_size: int = 2,
    max_detections: int = 50,
    iou_threshold: float = 0.6,
    conf_threshold: float = 0.1,
    padding: int | None = None,
    overwrite: bool = False,
    mps_mode: bool = False,
    simplify: int = 15,
    min_size: int | None = 500,
    max_size: int | None = None,
    close_interiors: bool = True,
):
    """Run instance segmentation inference on an image.

    Args:
        input: The input image file path.
        model: The model to use for inference.
        out: The output file path.
        gpu: The GPU device to use for inference.
        num_workers: The number of workers to use for inference.
        image_size: The size of the image to use for inference.
        patch_size: The size of the patch to use for inference.
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

    Raises:
        AssertionError: If the model is not DelineateAnything or DelineateAnything-S.

    Returns:
        None
    """
    from .delineate_anything import DelineateAnything

    assert model in ["DelineateAnything", "DelineateAnything-S"], (
        "Model must be either DelineateAnything or DelineateAnything-S."
    )

    device, _, _, patch_size, stride, _ = setup_inference(
        input, out, gpu, patch_size, padding, overwrite, mps_mode
    )

    # Load task
    tic = time.time()
    model = DelineateAnything(
        model=model,
        image_size=image_size,
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
    for batch in tqdm(
        dataloader,
        total=len(dataloader),
    ):
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

    # Convert polygons to fiboa format before writing to file
    with rasterio.open(input) as src:
        timestamp = src.tags().get("TIFFTAG_DATETIME", None)
        bounds = tuple(src.bounds)

    polygons = postprocess_instance_polygons(
        polygons, bounds, simplify, min_size, max_size, close_interiors
    )

    # Save polygons
    ext = os.path.splitext(out)[1]
    if ext == ".parquet":
        convert_to_fiboa(polygons, out, timestamp)
    elif ext == ".gpkg":
        polygons.to_file(out, driver="GPKG")
    elif ext == ".geojson":
        polygons.to_file(out, driver="GeoJSON")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    print(f"Finished inference and saved output to {out} in {time.time() - tic:.2f}s")


def postprocess_instance_polygons(
    polygons: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    simplify: int = 0,
    min_size: int | None = None,
    max_size: int | None = None,
    close_interiors: bool = True,
) -> gpd.GeoDataFrame:
    """Postprocess polygons to remove small polygons, simplify them, and compute area and perimeter.

    Args:
        polygons: The polygons to postprocess.
        simplify: The simplification factor.
        min_size: The minimum size of the polygons.
        max_size: The maximum size of the polygons.
        close_interiors: Whether to close the interiors of the polygons.

    Returns:
        The postprocessed polygons.
    """
    # Clip any polygons outside of image bounds
    polygons = polygons.clip(bounds)

    # Convert polygons to a meter based CRS
    src_crs = polygons.crs
    polygons.to_crs("EPSG:6933", inplace=True)

    if close_interiors:
        polygons.geometry = polygons.geometry.exterior
        polygons.geometry = polygons.geometry.apply(
            lambda x: shapely.geometry.Polygon(x)
        )

    if simplify > 0:
        polygons.geometry = polygons.geometry.simplify(simplify)

    polygons["area"] = polygons.geometry.area
    polygons["perimeter"] = polygons.geometry.length

    if min_size is not None:
        polygons.drop(polygons[polygons["area"] < min_size].index, inplace=True)
    if max_size is not None:
        polygons = polygons[polygons["area"] <= max_size]

    # Convert to hectares
    polygons["area"] = polygons["area"] * 0.0001

    # Convert back to original CRS
    polygons.to_crs(src_crs, inplace=True)

    polygons.reset_index(drop=True, inplace=True)
    polygons["id"] = polygons.index + 1

    return polygons


def convert_to_fiboa(
    polygons: gpd.GeoDataFrame,
    output: str,
    timestamp: str | None,
) -> gpd.GeoDataFrame:
    """Convert polygons to fiboa parquet format.

    Args:
        polygons: The polygons to convert.
        output: The output file path.
        timestamp: The timestamp of the image.

    Returns:
        The converted polygons.
    """
    polygons["determination_method"] = "auto-imagery"

    config = collection = {"fiboa_version": "0.2.0"}
    columns = ["id", "area", "perimeter", "determination_method", "geometry"]

    if timestamp is not None:
        pattern = re.compile(
            r"^(\d{4})[:-](\d{2})[:-](\d{2})[T\s](\d{2}):(\d{2}):(\d{2}).*$"
        )
        if pattern.match(timestamp):
            timestamp = re.sub(pattern, r"\1-\2-\3T\4:\5:\6Z", timestamp)
            polygons["determination_datetime"] = timestamp
            columns.append("determination_datetime")
        else:
            print("WARNING: Unable to parse timestamp from TIFFTAG_DATETIME tag.")

    create_parquet(polygons, columns, collection, output, config, compression="brotli")
