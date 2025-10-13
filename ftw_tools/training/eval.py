import os
import time
from typing import Sequence, Dict, Tuple


import numpy as np
import torch
from einops import rearrange
from lightning.pytorch.cli import LightningCLI
from torch.utils.data import DataLoader
from torchgeo.trainers import BaseTask
from torchmetrics import JaccardIndex, MetricCollection, Precision, Recall
from tqdm import tqdm

from ftw_tools.settings import FULL_DATA_COUNTRIES
from ftw_tools.training.datamodules import preprocess
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


def compute_metrics_from_samples(
    pixel_ious: list[float],
    pixel_precisions: list[float],
    pixel_recalls: list[float],
    object_tps: list[int],
    object_fps: list[int],
    object_fns: list[int],
) -> dict[str, float]:
    """Compute metrics from lists of per-sample values."""
    pixel_iou = np.mean(pixel_ious)
    pixel_precision = np.mean(pixel_precisions)
    pixel_recall = np.mean(pixel_recalls)

    total_tps = sum(object_tps)
    total_fps = sum(object_fps)
    total_fns = sum(object_fns)

    if total_tps + total_fps > 0:
        object_precision = total_tps / (total_tps + total_fps)
    else:
        object_precision = float("nan")

    if total_tps + total_fns > 0:
        object_recall = total_tps / (total_tps + total_fns)
    else:
        object_recall = float("nan")

    if object_precision + object_recall > 0 and not (
        np.isnan(object_precision) or np.isnan(object_recall)
    ):
        object_f1 = (
            2 * object_precision * object_recall / (object_precision + object_recall)
        )
    else:
        object_f1 = float("nan")

    return {
        "pixel_iou": float(pixel_iou),
        "pixel_precision": float(pixel_precision),
        "pixel_recall": float(pixel_recall),
        "object_precision": float(object_precision),
        "object_recall": float(object_recall),
        "object_f1": float(object_f1),
    }


def bootstrap_confidence_intervals(
    pixel_ious: list[float],
    pixel_precisions: list[float],
    pixel_recalls: list[float],
    object_tps: list[int],
    object_fps: list[int],
    object_fns: list[int],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """
    Compute bootstrap confidence intervals for metrics.

    Returns:
        metrics: Point estimates of metrics
        confidence_intervals: Dict mapping metric names to (lower, upper) bounds
    """
    np.random.seed(42)  # For reproducibility
    n_samples = len(pixel_ious)

    # Compute point estimates
    metrics = compute_metrics_from_samples(
        pixel_ious, pixel_precisions, pixel_recalls, object_tps, object_fps, object_fns
    )

    # Bootstrap sampling
    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        boot_pixel_ious = [pixel_ious[i] for i in indices]
        boot_pixel_precisions = [pixel_precisions[i] for i in indices]
        boot_pixel_recalls = [pixel_recalls[i] for i in indices]
        boot_object_tps = [object_tps[i] for i in indices]
        boot_object_fps = [object_fps[i] for i in indices]
        boot_object_fns = [object_fns[i] for i in indices]

        boot_metrics = compute_metrics_from_samples(
            boot_pixel_ious,
            boot_pixel_precisions,
            boot_pixel_recalls,
            boot_object_tps,
            boot_object_fps,
            boot_object_fns,
        )
        bootstrap_metrics.append(boot_metrics)

    # Compute confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    confidence_intervals = {}
    for metric_name in metrics.keys():
        if metric_name in bootstrap_metrics[0]:
            values = [boot_metrics[metric_name] for boot_metrics in bootstrap_metrics]
            # Filter out NaN values for percentile computation
            values = [v for v in values if not np.isnan(v)]
            if values:
                lower = np.percentile(values, lower_percentile)
                upper = np.percentile(values, upper_percentile)
                confidence_intervals[metric_name] = (lower, upper)
            else:
                confidence_intervals[metric_name] = (float("nan"), float("nan"))

    return metrics, confidence_intervals


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
    model_path,
    dir,
    gpu,
    countries: Sequence[str],
    iou_threshold,
    out,
    model_predicts_3_classes,
    test_on_3_classes,
    temporal_options,
    use_val_set,
    swap_order,
    num_workers,
    bootstrap=False,
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

    ds = FTW(
        root=dir,
        countries=countries,
        split=target_split,
        transforms=preprocess,
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

    # For bootstrap: collect per-sample metrics
    pixel_ious = []
    pixel_precisions = []
    pixel_recalls = []
    object_tps = []
    object_fps = []
    object_fns = []

    all_tps = 0
    all_fps = 0
    all_fns = 0

    for batch in tqdm(dl):
        images = batch["image"]
        masks = batch["mask"].to(device)

        if model_type in ["fcsiamdiff", "fcsiamconc", "fcsiamavg"]:
            images = rearrange(images, "b (t c) h w -> b t c h w", t=2)
        images = images.to(device)

        with torch.inference_mode():
            outputs = model(images).argmax(dim=1)

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

        # Update global metrics (for backward compatibility)
        metrics.update(outputs, masks)
        outputs_np = outputs.cpu().numpy().astype(np.uint8)
        masks_np = masks.cpu().numpy().astype(np.uint8)

        # Compute per-sample metrics for bootstrap
        for i in range(len(outputs)):
            output = outputs_np[i]
            mask = masks_np[i]

            # Pixel-level metrics per sample
            if bootstrap:
                # Create temporary metrics for this sample
                if test_on_3_classes:
                    sample_metrics = MetricCollection(
                        [
                            JaccardIndex(
                                task="multiclass",
                                average="none",
                                num_classes=3,
                                ignore_index=3,
                            ),
                            Precision(
                                task="multiclass",
                                average="none",
                                num_classes=3,
                                ignore_index=3,
                            ),
                            Recall(
                                task="multiclass",
                                average="none",
                                num_classes=3,
                                ignore_index=3,
                            ),
                        ]
                    ).to(device)
                else:
                    sample_metrics = MetricCollection(
                        [
                            JaccardIndex(
                                task="multiclass",
                                average="none",
                                num_classes=2,
                                ignore_index=3,
                            ),
                            Precision(
                                task="multiclass",
                                average="none",
                                num_classes=2,
                                ignore_index=3,
                            ),
                            Recall(
                                task="multiclass",
                                average="none",
                                num_classes=2,
                                ignore_index=3,
                            ),
                        ]
                    ).to(device)

                sample_outputs = torch.from_numpy(output).unsqueeze(0).to(device)
                sample_masks = torch.from_numpy(mask).unsqueeze(0).to(device)
                sample_metrics.update(sample_outputs, sample_masks)
                sample_results = sample_metrics.compute()

                pixel_ious.append(sample_results["MulticlassJaccardIndex"][1].item())
                pixel_precisions.append(sample_results["MulticlassPrecision"][1].item())
                pixel_recalls.append(sample_results["MulticlassRecall"][1].item())

            # Object-level metrics per sample
            tps, fps, fns = get_object_level_metrics(
                mask, output, iou_threshold=iou_threshold
            )
            if bootstrap:
                object_tps.append(tps)
                object_fps.append(fps)
                object_fns.append(fns)

            all_tps += tps
            all_fps += fps
            all_fns += fns

    # Compute overall metrics
    results = metrics.compute()
    pixel_level_iou = results["MulticlassJaccardIndex"][1].item()
    pixel_level_precision = results["MulticlassPrecision"][1].item()
    pixel_level_recall = results["MulticlassRecall"][1].item()

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

    # Bootstrap confidence intervals if requested
    confidence_intervals = None
    if bootstrap:
        bootstrap_metrics, confidence_intervals = bootstrap_confidence_intervals(
            pixel_ious,
            pixel_precisions,
            pixel_recalls,
            object_tps,
            object_fps,
            object_fns,
        )

        print(
            f"Pixel level IoU: {pixel_level_iou:.4f} [{confidence_intervals['pixel_iou'][0]:.4f}, {confidence_intervals['pixel_iou'][1]:.4f}]"
        )
        print(
            f"Pixel level precision: {pixel_level_precision:.4f} [{confidence_intervals['pixel_precision'][0]:.4f}, {confidence_intervals['pixel_precision'][1]:.4f}]"
        )
        print(
            f"Pixel level recall: {pixel_level_recall:.4f} [{confidence_intervals['pixel_recall'][0]:.4f}, {confidence_intervals['pixel_recall'][1]:.4f}]"
        )
        print(
            f"Object level precision: {object_precision:.4f} [{confidence_intervals['object_precision'][0]:.4f}, {confidence_intervals['object_precision'][1]:.4f}]"
        )
        print(
            f"Object level recall: {object_recall:.4f} [{confidence_intervals['object_recall'][0]:.4f}, {confidence_intervals['object_recall'][1]:.4f}]"
        )
        print(
            f"Object level F1: {object_f1:.4f} [{confidence_intervals['object_f1'][0]:.4f}, {confidence_intervals['object_f1'][1]:.4f}]"
        )
    else:
        print(f"Pixel level IoU: {pixel_level_iou:.4f}")
        print(f"Pixel level precision: {pixel_level_precision:.4f}")
        print(f"Pixel level recall: {pixel_level_recall:.4f}")
        print(f"Object level precision: {object_precision:.4f}")
        print(f"Object level recall: {object_recall:.4f}")
        print(f"Object level F1: {object_f1:.4f}")

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
            if bootstrap and confidence_intervals is not None:
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
