# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Script for running inference on a satellite image with a torchgeo segmentation model."""

import argparse
import math
import os
import time

import numpy as np
import rasterio
import torch
import tqdm
from rasterio.enums import ColorInterp
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from torchgeo.samplers import GridGeoSampler

from src.ftw.datasets import SingleRasterDataset
from src.ftw.trainers import CustomSemanticSegmentationTask

import kornia.augmentation as K
from kornia.constants import Resample


def preprocess(sample):
    sample["image"] = sample["image"] / 3000
    return sample


def get_parser() -> argparse.ArgumentParser:
    """Creates argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_fn",
        type=str,
        help="Input raster file. Should be a 8 channel stack of two Sentinel-2 L2A scenes with B04, B03, B02, B08 band ordering and original uint16 values.",
    )
    parser.add_argument(
        "--model_fn",
        type=str,
        help="Model checkpoint to load",
    )
    parser.add_argument(
        "--output_fn",
        type=str,
        help="Output filename",
    )
    parser.add_argument(
        "--resize_factor",
        type=int,
        default=2,
        help="Resize factor to use for inference",
    )
    parser.add_argument("--gpu_id", type=int, required=True, help="GPU id to use")
    parser.add_argument(
        "--patch_size",
        default=1024,
        type=int,
        help="Size of patch to use for inference",
    )
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size")
    parser.add_argument(
        "--padding",
        type=int,
        default=64,
        help="Number of pixels to throw away from each side of the patch after inference",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrites the outputs if they exist"
    )
    parser.add_argument(
        "--mps_mode",
        action="store_true",
        help="Use this flag to run inference in MPS mode on latest Apple GPUs",
    )

    return parser


def main(args) -> None:
    """Main function for the inference.py script."""
    assert os.path.exists(args.model_fn)
    input_model_checkpoint = args.model_fn
    print(input_model_checkpoint)
    input_image_fn = args.input_fn
    patch_size = args.patch_size
    padding = args.padding
    output_fn = args.output_fn

    # Sanity checks
    assert os.path.exists(input_model_checkpoint)
    assert input_model_checkpoint.endswith(".ckpt")
    assert os.path.exists(input_image_fn)
    assert input_image_fn.endswith(".tif") or input_image_fn.endswith(".vrt")
    assert int(math.log(patch_size, 2)) == math.log(patch_size, 2)
    stride = patch_size - padding * 2

    if os.path.exists(output_fn) and not args.overwrite:
        print(
            "Experiment output files already exist, use --overwrite to overwrite them."
            + " Exiting."
        )
        return

    if args.mps_mode:
        assert torch.backends.mps.is_available()
        device = torch.device("mps")
    else:
        assert torch.cuda.is_available()
        device = torch.device(f"cuda:{args.gpu_id}")

    # Load task and data
    tic = time.time()
    task = CustomSemanticSegmentationTask.load_from_checkpoint(input_model_checkpoint, map_location="cpu")
    task.freeze()
    model = task.model
    model = model.eval().to(device)

    up_sample = K.Resize(
        (patch_size * args.resize_factor, patch_size * args.resize_factor)
    ).to(device)
    down_sample = K.Resize((patch_size, patch_size), resample=Resample.NEAREST.name).to(
        device
    )

    dataset = SingleRasterDataset(input_image_fn, transforms=preprocess)
    sampler = GridGeoSampler(dataset, size=patch_size, stride=stride)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=6,
        collate_fn=stack_samples,
    )

    # Run inference
    tic = time.time()
    with rasterio.open(input_image_fn) as f:
        input_height, input_width = f.shape
        profile = f.profile
        transform = profile["transform"]

    print(f"Input size: {input_height} x {input_width}")
    assert patch_size <= input_height
    assert patch_size <= input_width
    output = np.zeros((input_height, input_width), dtype=np.uint8)

    # NOTE: we can make output quiet by adding a flag to set `dl_enumerator = dataloader`
    dl_enumerator = tqdm.tqdm(dataloader)

    for batch in dl_enumerator:
        images = batch["image"].to(device)
        images = up_sample(images)
        bboxes = batch["bbox"]
        with torch.inference_mode():
            predictions = model(images)
            # TODO: investigate doing bilinear interpolation before argmax
            predictions = predictions.argmax(axis=1).unsqueeze(0)

            # Don't squeeze here as somtimes the batch size is one
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
            assert right - left == patch_size
            assert bottom - top == patch_size

            destination_height, destination_width = output[
                top + padding : bottom - padding, left + padding : right - padding
            ].shape
            if (
                destination_height < patch_size - padding * 2
                and destination_width < patch_size - padding * 2
            ):
                inp = predictions[i][
                    padding : destination_height + padding,
                    padding : destination_width + padding,
                ]
            elif destination_height < patch_size - padding * 2:
                inp = predictions[i][
                    padding : destination_height + padding, padding:-padding
                ]
            elif destination_width < patch_size - padding * 2:
                inp = predictions[i][
                    padding:-padding, padding : destination_width + padding
                ]
            else:
                inp = predictions[i][padding:-padding, padding:-padding]
            output[
                top + padding : bottom - padding, left + padding : right - padding
            ] = inp

    print(f"Finished running model in {time.time()-tic:0.2f} seconds")

    # Save predictions
    tic = time.time()
    profile["driver"] = "GTiff"
    profile["count"] = 1
    profile["dtype"] = "uint8"
    profile["compress"] = "lzw"
    profile["predictor"] = 2
    profile["nodata"] = 0
    profile["blockxsize"] = 512
    profile["blockysize"] = 512
    profile["tiled"] = True
    profile["interleave"] = "pixel"

    with rasterio.open(output_fn, "w", **profile) as f:
        f.write(output, 1)
        f.write_colormap(
            1,
            {
                1: (0, 255, 0),
                2: (255, 0, 0),
            },
        )
        f.colorinterp = [ColorInterp.palette]

    print(f"Finished saving predictions in {time.time()-tic:0.2f} seconds")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
