import math
import os
import time

import click
import fiona
import fiona.transform
import kornia.augmentation as K
import numpy as np
import rasterio
import rasterio.features
import shapely.geometry
import torch
from affine import Affine
from kornia.constants import Resample
from lightning.pytorch.cli import LightningCLI
from pyproj import CRS
from rasterio.enums import ColorInterp
from torch.utils.data import DataLoader
from torchgeo.datamodules import BaseDataModule
from torchgeo.datasets import stack_samples
from torchgeo.samplers import GridGeoSampler
from torchgeo.trainers import BaseTask
from torchmetrics import JaccardIndex, MetricCollection, Precision, Recall
from tqdm import tqdm

from ftw.datamodules import preprocess
from ftw.datasets import FTW, SingleRasterDataset
from ftw.metrics import get_object_level_metrics
from ftw.trainers import CustomSemanticSegmentationTask


# Define the main click group
@click.group()
def ftw():
    """CLI group for FTW commands."""
    pass


# Define the 'model' click group
@click.group()
def model():
    """Training and testing FTW models."""
    pass


# Define the 'fit' command under 'model'
@click.command(help="Fit the model")
@click.option('--config', required=True, type=click.Path(exists=True), help='Path to the config file (required)')
@click.option('--ckpt_path', type=click.Path(exists=True), help='Path to a checkpoint file to resume training from')
@click.argument('cli_args', nargs=-1, type=click.UNPROCESSED)  # Capture all remaining arguments
def fit(config, ckpt_path, cli_args):
    """Command to fit the model."""
    print("Running fit command")

    # Construct the arguments for PyTorch Lightning CLI
    cli_args = ["fit", f"--config={config}"] + list(cli_args)

    # If a checkpoint path is provided, append it to the CLI arguments
    if ckpt_path:
        cli_args += [f"--ckpt_path={ckpt_path}"]

    print(f"CLI arguments: {cli_args}")

    # Best practices for Rasterio environment variables
    rasterio_best_practices = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
        "GDAL_SWATH_SIZE": "200000000",
        "VSI_CURL_CACHE_SIZE": "200000000",
    }
    os.environ.update(rasterio_best_practices)

    # Run the LightningCLI with the constructed arguments
    cli = LightningCLI(
        model_class=BaseTask,
        datamodule_class=BaseDataModule,
        seed_everything_default=0,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        args=cli_args,  # Pass the constructed cli_args
    )



# Define the 'test' command under 'model'
@click.command(help="Test the model")
@click.option('--model', '-m', required=True, type=str, help='Path to model checkpoint')
@click.option('--root_dir', type=str, default="data/ftw", help='Root directory of dataset')
@click.option('--gpu', type=int, default=0, help='GPU to use')
@click.option('--countries', type=str, multiple=True, required=True, help='Countries to evaluate on')
@click.option('--postprocess', is_flag=True, help='Apply postprocessing to the model output')
@click.option('--iou_threshold', type=float, default=0.5, help='IoU threshold for matching predictions to ground truths')
@click.option('--output', '-o', type=str, default="metrics.json", help='Output file for metrics')
@click.option('--model_predicts_3_classes', is_flag=True, help='Whether the model predicts 3 classes or 2 classes')
@click.option('--test_on_3_classes', is_flag=True, help='Whether to test on 3 classes or 2 classes')
@click.option('--temporal_options', type=str, default="stacked", help='Temporal option (stacked, windowA, windowB, etc.)')
@click.argument('cli_args', nargs=-1, type=click.UNPROCESSED)  # Capture all remaining arguments
def test(checkpoint, root_dir, gpu, countries, postprocess, iou_threshold, output, model_predicts_3_classes, test_on_3_classes, temporal_options, cli_args):
    """Command to test the model."""
    print("Running test command")

    # Merge `test_model` function into this test command
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    print("Loading model")
    tic = time.time()
    trainer = CustomSemanticSegmentationTask.load_from_checkpoint(checkpoint, map_location="cpu")
    model = trainer.model.eval().to(device)
    print(f"Model loaded in {time.time() - tic:.2f}s")

    print("Creating dataloader")
    tic = time.time()

    ds = FTW(
        root=root_dir,
        countries=countries,
        split="test",
        transforms=preprocess,
        load_boundaries=test_on_3_classes,
        temporal_options=temporal_options
    )
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=12)
    print(f"Created dataloader with {len(ds)} samples in {time.time() - tic:.2f}s")

    if test_on_3_classes:
        metrics = MetricCollection([
            JaccardIndex(task="multiclass", average="none", num_classes=3, ignore_index=3),
            Precision(task="multiclass", average="none", num_classes=3, ignore_index=3),
            Recall(task="multiclass", average="none", num_classes=3, ignore_index=3)
        ]).to(device)
    else:
        metrics = MetricCollection([
            JaccardIndex(task="multiclass", average="none", num_classes=2, ignore_index=3),
            Precision(task="multiclass", average="none", num_classes=2, ignore_index=3),
            Recall(task="multiclass", average="none", num_classes=2, ignore_index=3)
        ]).to(device)

    all_tps = 0
    all_fps = 0
    all_fns = 0
    for batch in tqdm(dl):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        with torch.inference_mode():
            outputs = model(images)

        outputs = outputs.argmax(dim=1)

        if model_predicts_3_classes:
            new_outputs = torch.zeros(outputs.shape[0], outputs.shape[1], outputs.shape[2], device=device)
            new_outputs[outputs == 2] = 0  # Boundary pixels
            new_outputs[outputs == 0] = 0  # Background pixels
            new_outputs[outputs == 1] = 1  # Crop pixels
            outputs = new_outputs
        else:
            if test_on_3_classes:
                raise ValueError("Cannot test on 3 classes when the model was trained on 2 classes")

        metrics.update(outputs, masks)
        outputs = outputs.cpu().numpy().astype(np.uint8)
        masks = masks.cpu().numpy().astype(np.uint8)

        for i in range(len(outputs)):
            output = outputs[i]
            mask = masks[i]
            if postprocess:
                post_processed_output = output.copy()
                output = post_processed_output
            tps, fps, fns = get_object_level_metrics(mask, output, iou_threshold=iou_threshold)
            all_tps += tps
            all_fps += fps
            all_fns += fns

    results = metrics.compute()
    pixel_level_iou = results["MulticlassJaccardIndex"][1].item()
    pixel_level_precision = results["MulticlassPrecision"][1].item()
    pixel_level_recall = results["MulticlassRecall"][1].item()

    if all_tps + all_fps > 0:
        object_precision = all_tps / (all_tps + all_fps)
    else:
        object_precision = float('nan')

    if all_tps + all_fns > 0:
        object_recall = all_tps / (all_tps + all_fns)
    else:
        object_recall = float('nan')

    print(f"Pixel level IoU: {pixel_level_iou:.4f}")
    print(f"Pixel level precision: {pixel_level_precision:.4f}")
    print(f"Pixel level recall: {pixel_level_recall:.4f}")
    print(f"Object level precision: {object_precision:.4f}")
    print(f"Object level recall: {object_recall:.4f}")

    if output is not None:
        if not os.path.exists(output):
            with open(output, "w") as f:
                f.write("train_checkpoint,test_countries,pixel_level_iou,pixel_level_precision,pixel_level_recall,object_level_precision,object_level_recall\n")
        with open(output, "a") as f:
            f.write(f"{checkpoint},{countries},{pixel_level_iou},{pixel_level_precision},{pixel_level_recall},{object_precision},{object_recall}\n")


# Add 'fit' and 'test' commands under the 'model' group
model.add_command(fit)
model.add_command(test)

# Add the 'model' group under the 'ftw' group
ftw.add_command(model)


if __name__ == "__main__":
    ftw()
