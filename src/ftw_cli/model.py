import os
import time
import torch
import numpy as np
import click
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import rasterio
from rasterio.enums import ColorInterp
import rasterio.features
import shapely.geometry
import fiona
import fiona.transform
from torchmetrics import JaccardIndex, Precision, Recall, MetricCollection
from lightning.pytorch.cli import ArgsType, LightningCLI
from ftw.datamodules import preprocess
from ftw.datasets import FTW
from ftw.metrics import get_object_level_metrics
from ftw.trainers import CustomSemanticSegmentationTask
from torchgeo.datamodules import BaseDataModule
from torchgeo.trainers import BaseTask
from torchgeo.datasets import stack_samples
from torchgeo.samplers import GridGeoSampler
from ftw.datasets import SingleRasterDataset
from ftw.trainers import CustomSemanticSegmentationTask
import kornia.augmentation as K
from kornia.constants import Resample   


# Define the main click group
@click.group()
def ftw():
    """CLI group for FTW commands."""
    pass


# Define the 'model' click group
@click.group()
def model():
    """Model-related commands."""
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
@click.option('--checkpoint_fn', required=True, type=str, help='Path to model checkpoint')
@click.option('--root_dir', type=str, default="data/ftw", help='Root directory of dataset')
@click.option('--gpu', type=int, default=0, help='GPU to use')
@click.option('--countries', type=str, multiple=True, required=True, help='Countries to evaluate on')
@click.option('--postprocess', is_flag=True, help='Apply postprocessing to the model output')
@click.option('--iou_threshold', type=float, default=0.5, help='IoU threshold for matching predictions to ground truths')
@click.option('--output_fn', type=str, default="metrics.json", help='Output file for metrics')
@click.option('--model_predicts_3_classes', is_flag=True, help='Whether the model predicts 3 classes or 2 classes')
@click.option('--test_on_3_classes', is_flag=True, help='Whether to test on 3 classes or 2 classes')
@click.option('--temporal_options', type=str, default="stacked", help='Temporal option (stacked, windowA, windowB, etc.)')
@click.argument('cli_args', nargs=-1, type=click.UNPROCESSED)  # Capture all remaining arguments
def test(checkpoint_fn, root_dir, gpu, countries, postprocess, iou_threshold, output_fn, model_predicts_3_classes, test_on_3_classes, temporal_options, cli_args):
    """Command to test the model."""
    print("Running test command")

    # Merge `test_model` function into this test command
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    print("Loading model")
    tic = time.time()
    trainer = CustomSemanticSegmentationTask.load_from_checkpoint(checkpoint_fn, map_location="cpu")
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

    if output_fn is not None:
        if not os.path.exists(output_fn):
            with open(output_fn, "w") as f:
                f.write("train_checkpoint,test_countries,pixel_level_iou,pixel_level_precision,pixel_level_recall,object_level_precision,object_level_recall\n")
        with open(output_fn, "a") as f:
            f.write(f"{checkpoint_fn},{countries},{pixel_level_iou},{pixel_level_precision},{pixel_level_recall},{object_precision},{object_recall}\n")


# Define the 'inference' command under 'model'
@click.command(help="Run inference on a satellite image")
@click.option('--input_fn', type=click.Path(exists=True), required=True, help="Input raster file (Sentinel-2 L2A stack).")
@click.option('--model_fn', type=click.Path(exists=True), required=True, help="Path to the model checkpoint.")
@click.option('--output_fn', type=str, required=True, help="Output filename.")
@click.option('--resize_factor', type=int, default=2, help="Resize factor to use for inference.")
@click.option('--gpu', type=int, help="GPU ID to use. If not provided, CPU will be used by default.")
@click.option('--patch_size', type=int, default=1024, help="Size of patch to use for inference.")
@click.option('--batch_size', type=int, default=2, help="Batch size.")
@click.option('--padding', type=int, default=64, help="Pixels to discard from each side of the patch.")
@click.option('--overwrite', is_flag=True, help="Overwrite outputs if they exist.")
@click.option('--polygonize', is_flag=True, help="Additionally polygonize the output (a GPKG file identical to `output_fn` will be created).")
@click.option('--simplify', type=float, default=None, help="Simplification factor to use when polygonizing.")
@click.option('--mps_mode', is_flag=True, help="Run inference in MPS mode (Apple GPUs).")
def inference(input_fn, model_fn, output_fn, resize_factor, gpu, patch_size, batch_size, padding, overwrite, polygonize, simplify, mps_mode):
    """Main function for the inference command."""

    # Sanity checks
    assert os.path.exists(model_fn), f"Model file {model_fn} does not exist."
    assert model_fn.endswith(".ckpt"), "Model file must be a .ckpt file."
    assert os.path.exists(input_fn), f"Input file {input_fn} does not exist."
    assert input_fn.endswith(".tif") or input_fn.endswith(".vrt"), "Input file must be a .tif or .vrt file."
    assert int(math.log(patch_size, 2)) == math.log(patch_size, 2), "Patch size must be a power of 2."
    assert output_fn.endswith(".tif"), "Output file must be a .tif file."

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
        crs = f.crs.to_string()
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

    if polygonize:
        print("Polygonizing output")
        tic = time.time()
        rows = []
        i = 0
        mask = (output==1).astype(np.uint8)

        mask = mask[:1024, :1024]

        # TODO: this can be very slow if there are many small objects
        for geom, val in rasterio.features.shapes(mask, transform=transform):
            if val == 1:
                rows.append({
                    "geometry": geom,
                    "properties": {
                        "idx": i
                    }
                })
                i += 1
        schema = {
            "geometry": "Polygon",
            "properties": {"idx": "int"}
        }
        with fiona.open(output_fn.replace(".tif", ".gpkg"), "w", driver="GPKG", crs=crs, schema=schema) as f:
            f.writerecords(rows)

        print(f"Finished polygonizing output in {time.time() - tic:.2f}s")



# Add 'fit' and 'test' commands under the 'model' group
model.add_command(fit)
model.add_command(test)
model.add_command(inference)

# Add the 'model' group under the 'ftw' group
ftw.add_command(model)


if __name__ == "__main__":
    ftw()
