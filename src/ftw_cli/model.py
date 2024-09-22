import os
import time
import torch
import numpy as np
import click
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics import JaccardIndex, Precision, Recall, MetricCollection
from lightning.pytorch.cli import ArgsType, LightningCLI
from ftw.datamodules import preprocess
from ftw.datasets import FTW
from ftw.metrics import get_object_level_metrics
from ftw.trainers import CustomSemanticSegmentationTask
from torchgeo.datamodules import BaseDataModule
from torchgeo.trainers import BaseTask


# Define the main click group
@click.group()
def ftw():
    """CLI group for FTW commands."""
    pass


# Define the unified 'model' command for fit or test
@click.command(help="Model Training and Testing", context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True
))
@click.argument('mode', type=click.Choice(['fit', 'test'], case_sensitive=False))  # Capture 'fit' or 'test' as argument
@click.argument('cli_args', nargs=-1, type=click.UNPROCESSED)  # Capture all remaining arguments
@click.pass_context  # Pass click context to capture extra arguments
def model(ctx, mode, cli_args):
    """Command to fit or test the model."""
    print(f"Running model command in {mode} mode with cli_args: {cli_args}")

    if mode == "fit":
        print("Running fit command")
        cli_args = ["fit"] + list(cli_args)
        run_cli(cli_args, mode="fit")
    elif mode == "test":
        print("Running test command")
        test_model(cli_args)


def test_model(cli_args):
    """Function to handle the test command logic."""
    import argparse

    # Use argparse to parse additional cli_args for the test mode
    parser = argparse.ArgumentParser(description='Evaluates a trained model on a set of test data')
    parser.add_argument('--checkpoint_fn', required=True, type=str, help='Path to model checkpoint')
    parser.add_argument('--root_dir', type=str, default="data/ftw", help='Root directory of dataset')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    parser.add_argument('--countries', type=str, nargs='+', required=True, help='Countries to evaluate on')
    parser.add_argument('--postprocess', action='store_true', help='Apply postprocessing to the model output')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for matching predictions to ground truths')
    parser.add_argument('--output_fn', type=str, default="metrics.json", help='Output file for metrics')
    parser.add_argument('--model_predicts_3_classes', action='store_true', help='Whether the model predicts 3 classes or 2 classes')
    parser.add_argument('--test_on_3_classes', action='store_true', help='Whether to test on 3 classes or 2 classes')
    parser.add_argument('--temporal_options', type=str, default="stacked", help='Temporal option (stacked, windowA, windowB, etc.)')

    args = parser.parse_args(cli_args)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    print("Loading model")
    tic = time.time()
    trainer = CustomSemanticSegmentationTask.load_from_checkpoint(args.checkpoint_fn, map_location="cpu")
    model = trainer.model.eval().to(device)
    print(f"Model loaded in {time.time() - tic:.2f}s")

    print("Creating dataloader")
    tic = time.time()

    ds = FTW(
        root=args.root_dir,
        countries=args.countries,
        split="test",
        transforms=preprocess,
        load_boundaries=args.test_on_3_classes,
        temporal_options=args.temporal_options
    )
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=12)
    print(f"Created dataloader with {len(ds)} samples in {time.time() - tic:.2f}s")

    if args.test_on_3_classes:
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

        if args.model_predicts_3_classes:
            new_outputs = torch.zeros(outputs.shape[0], outputs.shape[1], outputs.shape[2], device=device)
            new_outputs[outputs == 2] = 0  # Boundary pixels
            new_outputs[outputs == 0] = 0  # Background pixels
            new_outputs[outputs == 1] = 1  # Crop pixels
            outputs = new_outputs
        else:
            if args.test_on_3_classes:
                raise ValueError("Cannot test on 3 classes when the model was trained on 2 classes")

        metrics.update(outputs, masks)
        outputs = outputs.cpu().numpy().astype(np.uint8)
        masks = masks.cpu().numpy().astype(np.uint8)

        for i in range(len(outputs)):
            output = outputs[i]
            mask = masks[i]
            if args.postprocess:
                post_processed_output = output.copy()
                output = post_processed_output
            tps, fps, fns = get_object_level_metrics(mask, output, iou_threshold=args.iou_threshold)
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

    if args.output_fn is not None:
        if not os.path.exists(args.output_fn):
            with open(args.output_fn, "w") as f:
                f.write("train_checkpoint,test_countries,pixel_level_iou,pixel_level_precision,pixel_level_recall,object_level_precision,object_level_recall\n")
        with open(args.output_fn, "a") as f:
            f.write(f"{args.checkpoint_fn},{args.countries},{pixel_level_iou},{pixel_level_precision},{pixel_level_recall},{object_precision},{object_recall}\n")


def run_cli(cli_args: ArgsType = None, mode: str = "fit") -> None:
    """Runs the TorchGeo LightningCLI for fit or validation mode."""
    print(cli_args)

    rasterio_best_practices = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
        "GDAL_SWATH_SIZE": "200000000",
        "VSI_CURL_CACHE_SIZE": "200000000",
    }
    os.environ.update(rasterio_best_practices)

    cli = LightningCLI(
        model_class=BaseTask,
        datamodule_class=BaseDataModule,
        seed_everything_default=0,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        args=list(cli_args),
    )


# Add the unified 'model' command to the click group
ftw.add_command(model)


if __name__ == "__main__":
    ftw()
