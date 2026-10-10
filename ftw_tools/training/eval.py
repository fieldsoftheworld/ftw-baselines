import os
import time
from contextlib import contextmanager
from typing import Literal, Sequence

import kornia.augmentation as K
import numpy as np
import torch
from einops import rearrange
from kornia.constants import Resample
from lightning.pytorch.cli import LightningCLI
from torch.utils.data import DataLoader
from torchgeo.trainers import BaseTask
from torchmetrics import JaccardIndex, MetricCollection, Precision, Recall
from tqdm import tqdm

from ftw_tools.settings import FULL_DATA_COUNTRIES
from ftw_tools.training.datasets import FTW
from ftw_tools.training.metrics import get_object_level_metrics
from ftw_tools.training.trainers import CustomSemanticSegmentationTask
from ftw_tools.inference.models import DelineateAnything


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


def compute_metrics_from_aggregated_data(
    all_outputs: torch.Tensor,
    all_masks: torch.Tensor,
    all_object_tps: list[int],
    all_object_fps: list[int],
    all_object_fns: list[int],
    test_on_3_classes: bool = False,
) -> dict[str, float]:
    """Compute metrics from aggregated predictions and targets."""

    # Create metrics collection
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
        ).to(all_outputs.device)
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
        ).to(all_outputs.device)

    # Compute pixel-level metrics on GPU
    metrics.update(all_outputs, all_masks)
    results = metrics.compute()

    pixel_iou = results["MulticlassJaccardIndex"][1].item()
    pixel_precision = results["MulticlassPrecision"][1].item()
    pixel_recall = results["MulticlassRecall"][1].item()

    # Compute object-level metrics
    total_tps = sum(all_object_tps)
    total_fps = sum(all_object_fps)
    total_fns = sum(all_object_fns)

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
    all_outputs_list: list[np.ndarray],
    all_masks_list: list[np.ndarray],
    all_object_tps: list[int],
    all_object_fps: list[int],
    all_object_fns: list[int],
    test_on_3_classes: bool = False,
    device: torch.device | None = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """
    Compute bootstrap confidence intervals for metrics using global aggregation.

    Returns:
        metrics: Point estimates of metrics
        confidence_intervals: Dict mapping metric names to (lower, upper) bounds
    """
    np.random.seed(42)  # For reproducibility
    n_samples = len(all_outputs_list)

    print("Creating tensor list for bootstrapping")
    all_outputs_tensor_list = [
        torch.from_numpy(arr).to(device) for arr in all_outputs_list
    ]
    all_masks_tensor_list = [torch.from_numpy(arr).to(device) for arr in all_masks_list]

    # Compute point estimates using all data
    print("Concatenating all data for point estimates")
    all_outputs = torch.cat(all_outputs_tensor_list, dim=0)
    all_masks = torch.cat(all_masks_tensor_list, dim=0)
    print("Computing point estimates")
    metrics = compute_metrics_from_aggregated_data(
        all_outputs,
        all_masks,
        all_object_tps,
        all_object_fps,
        all_object_fns,
        test_on_3_classes,
    )
    del all_outputs, all_masks
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Bootstrap sampling
    bootstrap_metrics = []
    print("Running bootstrapping")
    for _ in tqdm(range(n_bootstrap)):
        # Sample with replacement at the sample level
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        # Aggregate bootstrap sample data
        boot_outputs_list = [all_outputs_tensor_list[i] for i in indices]
        boot_masks_list = [all_masks_tensor_list[i] for i in indices]
        boot_object_tps = [all_object_tps[i] for i in indices]
        boot_object_fps = [all_object_fps[i] for i in indices]
        boot_object_fns = [all_object_fns[i] for i in indices]

        # Concatenate bootstrap sample
        boot_outputs = torch.cat(boot_outputs_list, dim=0)
        boot_masks = torch.cat(boot_masks_list, dim=0)

        boot_metrics = compute_metrics_from_aggregated_data(
            boot_outputs,
            boot_masks,
            boot_object_tps,
            boot_object_fps,
            boot_object_fns,
            test_on_3_classes,
        )
        bootstrap_metrics.append(boot_metrics)

        # Free memory after each bootstrap iteration
        del boot_outputs, boot_masks
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

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

    # For bootstrap: collect per-sample data
    all_outputs_list = []
    all_masks_list = []
    object_tps = []
    object_fps = []
    object_fns = []

    all_tps = 0
    all_fps = 0
    all_fns = 0

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

        # Update global metrics (for backward compatibility)
        metrics.update(outputs, masks)
        outputs_np = outputs.cpu().numpy().astype(np.uint8)
        masks_np = masks.cpu().numpy().astype(np.uint8)

        # Store data for bootstrap if requested
        for i in range(len(outputs)):
            output_np = outputs_np[i]
            mask_np = masks_np[i]

            if bootstrap:
                # Store per-sample tensors for bootstrap
                all_outputs_list.append(output_np)
                all_masks_list.append(mask_np)

            # Object-level metrics per sample
            tps, fps, fns = get_object_level_metrics(
                mask_np, output_np, iou_threshold=iou_threshold
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
        del metrics, model  # Free memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        bootstrap_metrics, confidence_intervals = bootstrap_confidence_intervals(
            all_outputs_list,
            all_masks_list,
            object_tps,
            object_fps,
            object_fns,
            test_on_3_classes,
            device,
        )

        print(
            f"Pixel level IoU: {pixel_level_iou:.4f}\t{bootstrap_metrics['pixel_iou']} [{confidence_intervals['pixel_iou'][0]:.4f}, {confidence_intervals['pixel_iou'][1]:.4f}]"
        )
        print(
            f"Pixel level precision: {pixel_level_precision:.4f}\t{bootstrap_metrics['pixel_precision']} [{confidence_intervals['pixel_precision'][0]:.4f}, {confidence_intervals['pixel_precision'][1]:.4f}]"
        )
        print(
            f"Pixel level recall: {pixel_level_recall:.4f}\t{bootstrap_metrics['pixel_recall']} [{confidence_intervals['pixel_recall'][0]:.4f}, {confidence_intervals['pixel_recall'][1]:.4f}]"
        )
        print(
            f"Object level precision: {object_precision:.4f}\t{bootstrap_metrics['object_precision']} [{confidence_intervals['object_precision'][0]:.4f}, {confidence_intervals['object_precision'][1]:.4f}]"
        )
        print(
            f"Object level recall: {object_recall:.4f}\t{bootstrap_metrics['object_recall']} [{confidence_intervals['object_recall'][0]:.4f}, {confidence_intervals['object_recall'][1]:.4f}]"
        )
        print(
            f"Object level F1: {object_f1:.4f}\t{bootstrap_metrics['object_f1']} [{confidence_intervals['object_f1'][0]:.4f}, {confidence_intervals['object_f1'][1]:.4f}]"
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


def test_delineate_anything(
    dir: str,
    gpu: int,
    countries: Sequence[str],
    iou_threshold: float,
    out: str | None = None,
    model_variant: Literal["DelineateAnything-S", "DelineateAnything"] = "DelineateAnything-S",
    percentile_low: float | None = 0.02,
    percentile_high: float | None = 0.98,
    norm_constant: float | None = None,
    resize_factor: int = 2,
    patch_size: int = 256,
    max_detections: int = 100,
    conf_threshold: float = 0.05,
    model_iou_threshold: float = 0.3,
    num_workers: int = 4,
    batch_size: int = 16,
    use_val_set: bool = False,
):
    """Test DelineateAnything model on FTW dataset.

    Args:
        dir: Root directory of FTW dataset.
        gpu: GPU device index (-1 for CPU).
        countries: List of countries to test on.
        iou_threshold: IoU threshold for object-level metrics.
        out: Output CSV file path.
        model_variant: "DelineateAnything-S" or "DelineateAnything".
        percentile_low: Lower percentile for normalization. Mutually exclusive with norm_constant.
        percentile_high: Upper percentile for normalization. Mutually exclusive with norm_constant.
        norm_constant: If provided, divide by this value and clip to [0, 1] instead of
            using percentile normalization. Mutually exclusive with percentile_low/percentile_high.
        resize_factor: Factor to resize input images.
        patch_size: Input patch size.
        max_detections: Maximum detections per image.
        conf_threshold: Confidence threshold for detections.
        model_iou_threshold: IoU threshold for NMS in model.
        num_workers: Number of dataloader workers.
        batch_size: Batch size for inference.
        use_val_set: If True, use validation set instead of test set.

    Raises:
        ValueError: If both norm_constant and percentile options are specified.
    """
    from scipy.ndimage import binary_erosion

    # Validate mutually exclusive normalization options
    if norm_constant is not None and (percentile_low is not None or percentile_high is not None):
        raise ValueError(
            "norm_constant is mutually exclusive with percentile_low/percentile_high. "
            "Set percentile_low=None and percentile_high=None when using norm_constant."
        )

    target_split = "val" if use_val_set else "test"
    print(f"Running DelineateAnything test on the {target_split} set")

    if gpu is None:
        gpu = -1

    countries = expand_countries(countries)

    if torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    print(f"Loading DelineateAnything model ({model_variant})")
    tic = time.time()
    model = DelineateAnything(
        model=model_variant,
        patch_size=patch_size,
        resize_factor=resize_factor,
        max_detections=max_detections,
        iou_threshold=model_iou_threshold,
        conf_threshold=conf_threshold,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
        norm_constant=norm_constant,
        device=str(device),
    )
    print(f"Model loaded in {time.time() - tic:.2f}s")

    print("Creating dataloader")
    tic = time.time()
    ds = FTW(
        root=dir,
        countries=countries,
        split=target_split,
        load_boundaries=True,  # 3-class masks for comparison
        temporal_options="stacked",
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Created dataloader with {len(ds)} samples in {time.time() - tic:.2f}s")

    # Metrics for 3-class evaluation (0=bg, 1=field, 2=boundary)
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

    all_tps = 0
    all_fps = 0
    all_fns = 0

    print("Running inference")
    for batch in tqdm(dl):
        images = batch["image"]  # (B, C, H, W)
        masks = batch["mask"].to(device)  # (B, H, W)

        # Run inference
        with torch.inference_mode():
            results_list = model(images)

        # Convert instance segmentation results to 3-class masks
        pred_masks = []
        for results in results_list:
            h, w = patch_size, patch_size
            instance_mask = np.zeros((h, w), dtype=np.int32)

            if results.masks is not None:
                result_masks = results.masks.data.cpu().numpy()
                from scipy.ndimage import zoom

                for idx, mask in enumerate(result_masks, start=1):
                    if mask.shape != (h, w):
                        scale_y = h / mask.shape[0]
                        scale_x = w / mask.shape[1]
                        mask = zoom(mask, (scale_y, scale_x), order=0)
                    instance_mask[mask > 0.5] = idx

            # Convert to 3-class: 0=bg, 1=interior, 2=boundary
            boundary_mask = np.zeros((h, w), dtype=np.uint8)
            for idx in range(1, instance_mask.max() + 1):
                instance_binary = instance_mask == idx
                if not np.any(instance_binary):
                    continue
                interior = binary_erosion(instance_binary, iterations=1)
                boundary = instance_binary & ~interior
                boundary_mask[interior] = 1
                boundary_mask[boundary] = 2

            pred_masks.append(boundary_mask)

        pred_masks = torch.from_numpy(np.stack(pred_masks)).to(device)

        # Update metrics
        metrics.update(pred_masks, masks)

        # Object-level metrics
        pred_masks_np = pred_masks.cpu().numpy().astype(np.uint8)
        masks_np = masks.cpu().numpy().astype(np.uint8)

        for i in range(len(pred_masks_np)):
            # Convert to binary for object metrics (field vs background)
            pred_binary = (pred_masks_np[i] > 0).astype(np.uint8)
            mask_binary = (masks_np[i] > 0).astype(np.uint8)
            tps, fps, fns = get_object_level_metrics(
                mask_binary, pred_binary, iou_threshold=iou_threshold
            )
            all_tps += tps
            all_fps += fps
            all_fns += fns

    # Compute metrics
    results = metrics.compute()
    pixel_level_iou = results["MulticlassJaccardIndex"][1].item()
    pixel_level_precision = results["MulticlassPrecision"][1].item()
    pixel_level_recall = results["MulticlassRecall"][1].item()

    boundary_iou = results["MulticlassJaccardIndex"][2].item()
    boundary_precision = results["MulticlassPrecision"][2].item()
    boundary_recall = results["MulticlassRecall"][2].item()

    if all_tps + all_fps > 0:
        object_precision = all_tps / (all_tps + all_fps)
    else:
        object_precision = float("nan")

    if all_tps + all_fns > 0:
        object_recall = all_tps / (all_tps + all_fns)
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

    print(f"\n{'='*60}")
    print(f"DelineateAnything ({model_variant}) Results")
    if norm_constant is not None:
        print(f"Constant normalization: {norm_constant}")
    else:
        print(f"Percentile normalization: [{percentile_low}, {percentile_high}]")
    print(f"{'='*60}")
    print(f"Field pixel IoU: {pixel_level_iou:.4f}")
    print(f"Field pixel precision: {pixel_level_precision:.4f}")
    print(f"Field pixel recall: {pixel_level_recall:.4f}")
    print(f"Boundary pixel IoU: {boundary_iou:.4f}")
    print(f"Boundary pixel precision: {boundary_precision:.4f}")
    print(f"Boundary pixel recall: {boundary_recall:.4f}")
    print(f"Object precision: {object_precision:.4f}")
    print(f"Object recall: {object_recall:.4f}")
    print(f"Object F1: {object_f1:.4f}")

    country_str = ";".join(countries)
    if set(countries) == set(FULL_DATA_COUNTRIES):
        country_str = "all"

    if out is not None:
        if not os.path.exists(out):
            with open(out, "w") as f:
                f.write(
                    "model,countries,percentile_low,percentile_high,norm_constant,resize_factor,conf_threshold,model_iou_threshold,"
                    "field_pixel_iou,field_pixel_precision,field_pixel_recall,"
                    "boundary_pixel_iou,boundary_pixel_precision,boundary_pixel_recall,"
                    "object_precision,object_recall,object_f1\n"
                )
        with open(out, "a") as f:
            norm_const_str = str(norm_constant) if norm_constant is not None else ""
            percentile_low_str = str(percentile_low) if percentile_low is not None else ""
            percentile_high_str = str(percentile_high) if percentile_high is not None else ""
            f.write(
                f"{model_variant},{country_str},{percentile_low_str},{percentile_high_str},{norm_const_str},{resize_factor},{conf_threshold},{model_iou_threshold},"
                f"{pixel_level_iou},{pixel_level_precision},{pixel_level_recall},"
                f"{boundary_iou},{boundary_precision},{boundary_recall},"
                f"{object_precision},{object_recall},{object_f1}\n"
            )
        print(f"\nResults saved to {out}")
