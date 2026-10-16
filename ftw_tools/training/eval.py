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
from torchgeo.trainers import BaseTask
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


COUNT_COLUMNS = (
    "pixel_tp",
    "pixel_fp",
    "pixel_fn",
    "object_tp",
    "object_fp",
    "object_fn",
)


def pixel_confusion_counts(
    outputs: torch.Tensor,
    masks: torch.Tensor,
    positive_class: int = 1,
    ignore_index: int = 3,
) -> np.ndarray:
    """Count per-sample pixel TP, FP and FN for one class.

    These match the per-class stats torchmetrics' multiclass metrics use with
    ``ignore_index``, so summing the rows reproduces the global pixel metrics.

    Args:
        outputs: Predicted labels of shape (B, H, W).
        masks: Target labels of shape (B, H, W).
        positive_class: Class to count.
        ignore_index: Target value whose pixels are skipped.

    Returns:
        Array of shape (B, 3) with columns (tp, fp, fn).
    """
    valid = masks != ignore_index
    pred_pos = (outputs == positive_class) & valid
    true_pos = masks == positive_class
    counts = torch.stack(
        [
            (pred_pos & true_pos).flatten(1).sum(1),
            (pred_pos & ~true_pos).flatten(1).sum(1),
            (~pred_pos & true_pos).flatten(1).sum(1),
        ],
        dim=1,
    )
    return counts.cpu().numpy().astype(np.int64)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide elementwise, returning 0 where the denominator is 0 (as torchmetrics does)."""
    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )


def metrics_from_counts(totals: np.ndarray) -> dict[str, np.ndarray]:
    """Compute pixel and object metrics from summed counts.

    Args:
        totals: Array of shape (..., 6) with columns ordered as ``COUNT_COLUMNS``.

    Returns:
        Dict mapping metric names to arrays of shape ``totals.shape[:-1]``. Undefined
        pixel metrics are 0 (as in torchmetrics) and undefined object metrics are NaN.
    """
    pixel_tp, pixel_fp, pixel_fn, object_tp, object_fp, object_fn = np.moveaxis(
        np.asarray(totals, dtype=np.float64), -1, 0
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        object_precision = np.where(
            object_tp + object_fp > 0, object_tp / (object_tp + object_fp), np.nan
        )
        object_recall = np.where(
            object_tp + object_fn > 0, object_tp / (object_tp + object_fn), np.nan
        )
        pr_sum = object_precision + object_recall
        object_f1 = np.where(
            pr_sum > 0, 2 * object_precision * object_recall / pr_sum, np.nan
        )
    return {
        "pixel_iou": _safe_ratio(pixel_tp, pixel_tp + pixel_fp + pixel_fn),
        "pixel_precision": _safe_ratio(pixel_tp, pixel_tp + pixel_fp),
        "pixel_recall": _safe_ratio(pixel_tp, pixel_tp + pixel_fn),
        "object_precision": object_precision,
        "object_recall": object_recall,
        "object_f1": object_f1,
    }


def bootstrap_confidence_intervals(
    sample_counts: np.ndarray,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Compute percentile bootstrap confidence intervals over samples.

    Each resample draws samples with replacement and sums their counts, so memory is
    O(n_samples) regardless of image size or number of resamples.

    Args:
        sample_counts: Array of shape (n_samples, 6) with per-sample counts ordered as
            ``COUNT_COLUMNS``.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level of the intervals, in (0, 1).
        seed: Seed for resampling; a fixed seed gives identical intervals.

    Returns:
        Dict mapping metric names to (lower, upper) bounds, or (nan, nan) for a metric
        that is undefined in every resample.
    """
    sample_counts = np.asarray(sample_counts, dtype=np.int64)
    if sample_counts.ndim != 2 or sample_counts.shape[1] != len(COUNT_COLUMNS):
        raise ValueError(
            f"sample_counts must have shape (n_samples, {len(COUNT_COLUMNS)}), "
            f"got {sample_counts.shape}"
        )
    n_samples = len(sample_counts)
    if n_samples == 0:
        raise ValueError("Cannot bootstrap an empty set of samples")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be at least 1, got {n_resamples}")
    if not 0 < confidence_level < 1:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

    # Legacy RandomState keeps the resampled indices identical to earlier releases
    rng = np.random.RandomState(seed)
    totals = np.empty((n_resamples, len(COUNT_COLUMNS)), dtype=np.int64)
    for i in tqdm(range(n_resamples), desc="Bootstrapping"):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        totals[i] = np.bincount(indices, minlength=n_samples) @ sample_counts

    alpha = 1 - confidence_level
    percentiles = (100 * alpha / 2, 100 * (1 - alpha / 2))
    confidence_intervals = {}
    for name, values in metrics_from_counts(totals).items():
        values = values[~np.isnan(values)]
        if values.size == 0:
            confidence_intervals[name] = (float("nan"), float("nan"))
        else:
            lower, upper = np.percentile(values, percentiles)
            confidence_intervals[name] = (float(lower), float(upper))
    return confidence_intervals


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
    bootstrap_seed: int = 42,
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

    # For bootstrap: per-sample counts ordered as COUNT_COLUMNS
    sample_counts = []

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

        batch_object_counts = []
        for i in range(len(outputs)):
            # Object-level metrics per sample
            tps, fps, fns = get_object_level_metrics(
                masks_np[i], outputs_np[i], iou_threshold=iou_threshold
            )
            batch_object_counts.append((tps, fps, fns))

            all_tps += tps
            all_fps += fps
            all_fns += fns

        if bootstrap:
            sample_counts.append(
                np.hstack(
                    [
                        pixel_confusion_counts(outputs, masks),
                        np.array(batch_object_counts, dtype=np.int64),
                    ]
                )
            )

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
        confidence_intervals = bootstrap_confidence_intervals(
            np.concatenate(sample_counts),
            n_resamples=n_bootstrap,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
        )
        ci = confidence_intervals
        print(
            f"Bootstrap: {n_bootstrap} resamples, "
            f"{confidence_level:.0%} confidence, seed {bootstrap_seed}"
        )
        print(
            f"Pixel level IoU: {pixel_level_iou:.4f} [{ci['pixel_iou'][0]:.4f}, {ci['pixel_iou'][1]:.4f}]"
        )
        print(
            f"Pixel level precision: {pixel_level_precision:.4f} [{ci['pixel_precision'][0]:.4f}, {ci['pixel_precision'][1]:.4f}]"
        )
        print(
            f"Pixel level recall: {pixel_level_recall:.4f} [{ci['pixel_recall'][0]:.4f}, {ci['pixel_recall'][1]:.4f}]"
        )
        print(
            f"Object level precision: {object_precision:.4f} [{ci['object_precision'][0]:.4f}, {ci['object_precision'][1]:.4f}]"
        )
        print(
            f"Object level recall: {object_recall:.4f} [{ci['object_recall'][0]:.4f}, {ci['object_recall'][1]:.4f}]"
        )
        print(
            f"Object level F1: {object_f1:.4f} [{ci['object_f1'][0]:.4f}, {ci['object_f1'][1]:.4f}]"
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
