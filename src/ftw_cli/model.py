import os
import time

import numpy as np
import torch
from lightning.pytorch.cli import LightningCLI
from torch.utils.data import DataLoader
from torchgeo.datamodules import BaseDataModule
from torchgeo.trainers import BaseTask
from torchmetrics import JaccardIndex, MetricCollection, Precision, Recall
from tqdm import tqdm

from ftw.datamodules import preprocess
from ftw.datasets import FTW
from ftw.metrics import get_object_level_metrics
from ftw.trainers import CustomSemanticSegmentationTask


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

    print("Finished")


def test(model, dir, gpu, countries, postprocess, iou_threshold, out, model_predicts_3_classes, test_on_3_classes, temporal_options, cli_args):
    """Command to test the model."""
    print("Running test command")

    # Merge `test_model` function into this test command
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    print("Loading model")
    tic = time.time()
    trainer = CustomSemanticSegmentationTask.load_from_checkpoint(model, map_location="cpu")
    model = trainer.model.eval().to(device)
    print(f"Model loaded in {time.time() - tic:.2f}s")

    print("Creating dataloader")
    tic = time.time()

    ds = FTW(
        root=dir,
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
                post_processed_output = out.copy()
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

    if out is not None:
        if not os.path.exists(out):
            with open(out, "w") as f:
                f.write("train_checkpoint,test_countries,pixel_level_iou,pixel_level_precision,pixel_level_recall,object_level_precision,object_level_recall\n")
        with open(out, "a") as f:
            f.write(f"{model},{countries},{pixel_level_iou},{pixel_level_precision},{pixel_level_recall},{object_precision},{object_recall}\n")
