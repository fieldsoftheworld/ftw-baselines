import os
import time
from contextlib import contextmanager
from typing import Sequence

import kornia.augmentation as K
import numpy as np
import torch
from einops import rearrange
from kornia.constants import Resample
from lightning.pytorch.cli import LightningCLI
from torch.utils.data import DataLoader
from torchgeo.tasks import BaseTask
from torchmetrics import JaccardIndex, MetricCollection, Precision, Recall
from tqdm import tqdm

from ftw_tools.settings import FULL_DATA_COUNTRIES
from ftw_tools.training.datasets import FTW
from ftw_tools.training.metrics import get_object_level_metrics
from ftw_tools.training.trainers import CustomSemanticSegmentationTask


def expand_countries(countries: Sequence[str]) -> list[str]:
    """Expand the 'full_data' placeholder to the full list of countries.
    Args:
        countries: List of country names, may contain 'full_data' placeholder
    Returns:
        List of country names with 'full_data' expanded to FULL_DATA_COUNTRIES.
        Always returns a new list to avoid modifying the original.
    Examples:
        >>> expand_countries(['full_data'])
        ['austria', 'belgium', ...]
        >>> expand_countries(['rwanda', 'kenya'])
        ['rwanda', 'kenya']
        >>> expand_countries(['rwanda', 'full_data', 'kenya'])
        ['austria', 'belgium', ...]  # full_data replaces entire list
    """
    countries = list(countries)  # Make sure this is a list
    if "full_data" in countries:
        return FULL_DATA_COUNTRIES.copy()
    return countries


@contextmanager
def gpu_memory_manager():
    """Context manager for efficient GPU memory management."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def pixel_confusion_counts(outputs: torch.Tensor, masks: torch.Tensor) -> np.ndarray:
    """Per-sample (TP, FP, FN) pixel counts for class 1, ignoring mask value 3."""
    pred = (outputs == 1) & (masks != 3)
    true = masks == 1
    counts = torch.stack([pred & true, pred & ~true, ~pred & true], dim=1)
    return counts.flatten(2).sum(2).cpu().numpy().astype(np.int64)


def bootstrap_confidence_intervals(
    sample_counts: np.ndarray,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Bootstrap CIs from per-sample pixel and object (TP, FP, FN) counts."""
    rng = np.random.RandomState(seed)
    n_samples = len(sample_counts)
    totals = np.empty((n_resamples, 6))
    for i in range(n_resamples):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        totals[i] = np.bincount(indices, minlength=n_samples) @ sample_counts

    pixel_tp, pixel_fp, pixel_fn, object_tp, object_fp, object_fn = totals.T
    object_precision = object_tp / (object_tp + object_fp)
    object_recall = object_tp / (object_tp + object_fn)
    resampled = {
        "pixel_iou": pixel_tp / (pixel_tp + pixel_fp + pixel_fn),
        "pixel_precision": pixel_tp / (pixel_tp + pixel_fp),
        "pixel_recall": pixel_tp / (pixel_tp + pixel_fn),
        "object_precision": object_precision,
        "object_recall": object_recall,
        "object_f1": 2
        * object_precision
        * object_recall
        / (object_precision + object_recall),
    }
    alpha = 1 - confidence_level
    return {
        name: tuple(np.nanpercentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
        for name, values in resampled.items()
    }


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
        seed_everything_default=0,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        args=cli_args,  # Pass the constructed cli_args
    )

    print("Finished")


def test(
    model_path: str,
    dir: str,
    gpu: int,
    countries: Sequence[str],
    iou_threshold: float,
    out: str,
    model_predicts_3_classes: bool,
    test_on_3_classes: bool,
    temporal_options: str,
    use_val_set: bool,
    swap_order: bool,
    norm_constant: float | None,
    resize_factor: int,
    num_workers: int,
    bootstrap: bool = False,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
):
    """Command to test the model."""
    target_split = "val" if use_val_set else "test"
    print(f"Running test command on the {target_split} set")
    if gpu is None:
        gpu = -1

    countries = expand_countries(countries)

    # Merge `test_model` function into this test command
    if torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    print("Loading model")
    tic = time.time()
    trainer = CustomSemanticSegmentationTask.load_from_checkpoint(
        model_path, map_location="cpu"
    )
    model_type = trainer.hparams["model"]
    model = trainer.model.eval().to(device)
    print(f"Model loaded in {time.time() - tic:.2f}s")

    print("Creating dataloader")
    tic = time.time()

    norm = 3000.0  # Default normalization constant
    if norm_constant is not None:
        norm = norm_constant

    ds = FTW(
        root=dir,
        countries=countries,
        split=target_split,
        load_boundaries=test_on_3_classes,
        temporal_options=temporal_options,
        swap_order=swap_order,
    )
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=num_workers)
    print(f"Created dataloader with {len(ds)} samples in {time.time() - tic:.2f}s")

    if test_on_3_classes:
        metrics = MetricCollection(
            [
                JaccardIndex(
                    task="multiclass", average="none", num_classes=3, ignore_index=3
                ),
                Precision(
                    task="multiclass", average="none", num_classes=3, ignore_index=3
                ),
                Recall(
                    task="multiclass", average="none", num_classes=3, ignore_index=3
                ),
            ]
        ).to(device)
    else:
        metrics = MetricCollection(
            [
                JaccardIndex(
                    task="multiclass", average="none", num_classes=2, ignore_index=3
                ),
                Precision(
                    task="multiclass", average="none", num_classes=2, ignore_index=3
                ),
                Recall(
                    task="multiclass", average="none", num_classes=2, ignore_index=3
                ),
            ]
        ).to(device)

    patch_size = 256
    up_sample = K.Resize((patch_size * resize_factor, patch_size * resize_factor)).to(
        device
    )
    down_sample = K.Resize((patch_size, patch_size), resample=Resample.NEAREST.name).to(
        device
    )

    pixel_counts = []
    object_counts = []
    for batch in tqdm(dl):
        images = batch["image"].to(device) / norm
        masks = batch["mask"].to(device)

        if resize_factor != 1:
            images = up_sample(images)

        if model_type in ["fcsiamdiff", "fcsiamconc", "fcsiamavg"]:
            images = rearrange(images, "b (t c) h w -> b t c h w", t=2)

        with torch.inference_mode():
            outputs = model(images).argmax(dim=1)
        if resize_factor != 1:
            outputs = down_sample(outputs.unsqueeze(1).float()).squeeze().long()
        if model_predicts_3_classes:
            new_outputs = torch.zeros(
                outputs.shape[0], outputs.shape[1], outputs.shape[2], device=device
            )
            new_outputs[outputs == 2] = 0  # Boundary pixels
            new_outputs[outputs == 0] = 0  # Background pixels
            new_outputs[outputs == 1] = 1  # Crop pixels
            outputs = new_outputs
        else:
            if test_on_3_classes:
                raise ValueError(
                    "Cannot test on 3 classes when the model was trained on 2 classes"
                )

        metrics.update(outputs, masks)
        if bootstrap:
            pixel_counts.append(pixel_confusion_counts(outputs, masks))
        outputs_np = outputs.cpu().numpy().astype(np.uint8)
        masks_np = masks.cpu().numpy().astype(np.uint8)
        for output_np, mask_np in zip(outputs_np, masks_np):
            object_counts.append(
                get_object_level_metrics(
                    mask_np, output_np, iou_threshold=iou_threshold
                )
            )

    # Compute overall metrics
    results = metrics.compute()
    pixel_level_iou = results["MulticlassJaccardIndex"][1].item()
    pixel_level_precision = results["MulticlassPrecision"][1].item()
    pixel_level_recall = results["MulticlassRecall"][1].item()
    all_tps, all_fps, all_fns = np.sum(object_counts, axis=0).tolist()

    if all_tps + all_fps > 0:
        object_precision = all_tps / (all_tps + all_fps)
    else:
        object_precision = float("nan")

    if all_tps + all_fns > 0:
        object_recall = all_tps / (all_tps + all_fns)
    else:
        object_recall = float("nan")

    # Compute object F1
    if object_precision + object_recall > 0 and not (
        np.isnan(object_precision) or np.isnan(object_recall)
    ):
        object_f1 = (
            2 * object_precision * object_recall / (object_precision + object_recall)
        )
    else:
        object_f1 = float("nan")

    confidence_intervals = None
    if bootstrap:
        sample_counts = np.hstack([np.concatenate(pixel_counts), object_counts])
        confidence_intervals = bootstrap_confidence_intervals(
            sample_counts, n_bootstrap, confidence_level
        )

    point_estimates = {
        "pixel_iou": ("Pixel level IoU", pixel_level_iou),
        "pixel_precision": ("Pixel level precision", pixel_level_precision),
        "pixel_recall": ("Pixel level recall", pixel_level_recall),
        "object_precision": ("Object level precision", object_precision),
        "object_recall": ("Object level recall", object_recall),
        "object_f1": ("Object level F1", object_f1),
    }
    for name, (label, value) in point_estimates.items():
        line = f"{label}: {value:.4f}"
        if confidence_intervals is not None:
            lower, upper = confidence_intervals[name]
            line += f" [{lower:.4f}, {upper:.4f}]"
        print(line)

    country_str = ";".join(countries)
    if set(countries) == set(FULL_DATA_COUNTRIES):
        country_str = "all"

    if out is not None:
        if not os.path.exists(out):
            with open(out, "w") as f:
                if bootstrap:
                    f.write(
                        "train_checkpoint,countries,pixel_level_iou,pixel_level_precision,pixel_level_recall,object_level_precision,object_level_recall,object_level_f1,"
                        "pixel_level_iou_ci_lower,pixel_level_iou_ci_upper,"
                        "pixel_level_precision_ci_lower,pixel_level_precision_ci_upper,"
                        "pixel_level_recall_ci_lower,pixel_level_recall_ci_upper,"
                        "object_level_precision_ci_lower,object_level_precision_ci_upper,"
                        "object_level_recall_ci_lower,object_level_recall_ci_upper,"
                        "object_level_f1_ci_lower,object_level_f1_ci_upper\n"
                    )
                else:
                    f.write(
                        "train_checkpoint,countries,pixel_level_iou,pixel_level_precision,pixel_level_recall,object_level_precision,object_level_recall,object_level_f1\n"
                    )

        with open(out, "a") as f:
            if confidence_intervals is not None:
                ci = confidence_intervals
                f.write(
                    f"{model_path},{country_str},{pixel_level_iou},{pixel_level_precision},{pixel_level_recall},{object_precision},{object_recall},{object_f1},"
                    f"{ci['pixel_iou'][0]},{ci['pixel_iou'][1]},"
                    f"{ci['pixel_precision'][0]},{ci['pixel_precision'][1]},"
                    f"{ci['pixel_recall'][0]},{ci['pixel_recall'][1]},"
                    f"{ci['object_precision'][0]},{ci['object_precision'][1]},"
                    f"{ci['object_recall'][0]},{ci['object_recall'][1]},"
                    f"{ci['object_f1'][0]},{ci['object_f1'][1]}\n"
                )
            else:
                f.write(
                    f"{model_path},{country_str},{pixel_level_iou},{pixel_level_precision},{pixel_level_recall},{object_precision},{object_recall},{object_f1}\n"
                )
