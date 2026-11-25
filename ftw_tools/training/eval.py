import csv
import os
import time
from contextlib import contextmanager
from typing import Sequence

import kornia.augmentation as K
import numpy as np
import rasterio
import torch
from einops import rearrange
from kornia.constants import Resample
from lightning.pytorch.cli import LightningCLI
from pyproj import CRS
from torch.utils.data import DataLoader
from torchgeo.tasks import BaseTask
from torchmetrics import JaccardIndex, MetricCollection, Precision, Recall
from tqdm import tqdm

from ftw_tools.settings import FULL_DATA_COUNTRIES
from ftw_tools.training.datasets import FTW
from ftw_tools.training.metrics import (
    InstanceStatistics,
    chamfer_statistics,
    get_object_level_metrics,
    instance_scores_for_reporting,
    instance_statistics,
)
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


def _reference_pixel_spacing(reference, shape):
    """Return (row, column) metre spacing on the original reference grid.

    Geographic grids use geodesic one-pixel steps at the chip centre: a local,
    constant-spacing approximation, not exact geodesic distances across a chip.
    Projected grids use their CRS linear units. No resampling is performed.
    """
    with rasterio.open(reference) as source:
        if source.shape != shape:
            raise ValueError(
                f"Reference grid shape differs from scored mask: {reference}"
            )
        transform = source.transform
        if (
            source.crs is None
            or not np.isfinite(tuple(transform)).all()
            or transform.b != 0
            or transform.d != 0
            or transform.a == 0
            or transform.e == 0
        ):
            raise ValueError(
                f"Boundary metrics require a known CRS and unrotated, "
                f"nondegenerate grid: {reference}"
            )
        crs = CRS.from_user_input(source.crs)
        if crs.is_projected:
            _, factor = source.crs.linear_units_factor
            spacing = (abs(transform.e) * factor, abs(transform.a) * factor)
        elif crs.is_geographic:
            x, y = transform * (source.width / 2, source.height / 2)
            geod = crs.get_geod()
            spacing = (
                abs(geod.inv(x, y, x, y + transform.e)[2]),
                abs(geod.inv(x, y, x + transform.a, y)[2]),
            )
        else:
            raise ValueError(f"Unsupported CRS for boundary metrics: {reference}")
    if not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError(f"Invalid metre pixel spacing: {reference}")
    return spacing


def _extended_object_scores(instance_stats, boundary_stats, *, background_known):
    """Pool chip statistics, never chip means; keep legacy score names separate."""
    pooled = InstanceStatistics()
    for stats in instance_stats:
        pooled += stats
    scores = instance_scores_for_reporting(pooled, background_known=background_known)
    for prefix in ("threshold_chamfer_m", "best_iou_chamfer_m"):
        count = sum(stats[prefix + "_count"] for stats in boundary_stats)
        total = sum(stats[prefix + "_sum"] for stats in boundary_stats)
        scores[prefix] = total / count if count else float("nan")
        scores[prefix + "_count"] = count
    reference_count = sum(stats["reference_count"] for stats in boundary_stats)
    best_count = sum(stats["best_iou_reference_count"] for stats in boundary_stats)
    scores.update(
        reference_count=reference_count,
        best_iou_reference_count=best_count,
        best_iou_reference_coverage=best_count / reference_count
        if reference_count
        else float("nan"),
    )
    return {f"extended_{name}": value for name, value in scores.items()}


def _test_csv_fields(*, extended_object_metrics, bootstrap):
    fields = [
        "train_checkpoint",
        "countries",
        "pixel_level_iou",
        "pixel_level_precision",
        "pixel_level_recall",
        "object_level_precision",
        "object_level_recall",
        "object_level_f1",
    ]
    if bootstrap:
        for name in fields[2:8]:
            fields.extend((name + "_ci_lower", name + "_ci_upper"))
    if extended_object_metrics:
        names = _extended_object_scores([], [], background_known=True)
        fields.extend(names)
        if bootstrap:
            for name in names:
                fields.extend((name + "_ci_lower", name + "_ci_upper"))
    return fields


def _check_test_csv_header(out, fields):
    """Do not mix legacy, extended, or bootstrap schemas in one results file."""
    if out is not None and os.path.exists(out) and os.path.getsize(out):
        with open(out, newline="") as source:
            if next(csv.reader(source), None) != fields:
                raise ValueError(
                    f"Metrics CSV header does not match requested metrics: {out}. "
                    "Choose a separate output file."
                )


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
    *,
    instance_stats: list[InstanceStatistics] | None = None,
    boundary_stats: list[dict] | None = None,
    background_known: bool = True,
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """
    Compute bootstrap confidence intervals for metrics using global aggregation.

    Returns:
        metrics: Point estimates of metrics
        confidence_intervals: Dict mapping metric names to (lower, upper) bounds
    """
    np.random.seed(42)  # For reproducibility
    n_samples = len(all_outputs_list)
    if (instance_stats is None) != (boundary_stats is None):
        raise ValueError("Provide both instance and boundary statistics")
    if instance_stats is not None and (
        len(instance_stats) != n_samples or len(boundary_stats) != n_samples
    ):
        raise ValueError("Extended statistics must align with bootstrap samples")

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
    if instance_stats is not None:
        metrics.update(
            _extended_object_scores(
                instance_stats, boundary_stats, background_known=background_known
            )
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
        if instance_stats is not None:
            boot_metrics.update(
                _extended_object_scores(
                    [instance_stats[i] for i in indices],
                    [boundary_stats[i] for i in indices],
                    background_known=background_known,
                )
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
    *,
    extended_object_metrics: bool = False,
):
    """Test a checkpoint, optionally adding the reference-based instance suite.

    Extended results use semantic reference components, mask ignored pixels on
    both sides, and pool sufficient statistics before taking ratios. They do not
    replace legacy scores or modify training/validation. Chamfer distances are
    object-weighted metres with matching counts and reference coverage reported.
    FP-dependent extended scores are NaN when any country is presence-only.
    """
    csv_fields = _test_csv_fields(
        extended_object_metrics=extended_object_metrics, bootstrap=bootstrap
    )
    _check_test_csv_header(out, csv_fields)
    target_split = "val" if use_val_set else "test"
    print(f"Running test command on the {target_split} set")
    if gpu is None:
        gpu = -1

    countries = expand_countries(countries)
    background_known = all(
        country.lower() in FULL_DATA_COUNTRIES for country in countries
    )

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
    extended_stats = []
    boundary_stats = []
    sample_offset = 0
    if extended_object_metrics:
        print(
            "Extended object metrics: four-connected semantic references; ignored "
            "pixels excluded from both masks. Chamfer distances use per-chip "
            "metre spacing (local centre approximation for geographic grids)."
        )
        if not background_known:
            print(
                "Presence-only labels included: extended precision, F1, RQ and PQ "
                "are undefined; legacy results are unchanged."
            )

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
            outputs = down_sample(outputs.unsqueeze(1).float()).squeeze(1).long()
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
            if extended_object_metrics:
                # Read georeferencing only when explicitly requested; no dataset
                # changes or instance-mask files are required.
                reference = ds.filenames[sample_offset + i]["mask"]
                spacing = _reference_pixel_spacing(reference, mask_np.shape)
                truth = masks[i].cpu().numpy()
                ignore_index = trainer.hparams.get("ignore_index")
                extended_stats.append(
                    instance_statistics(
                        truth,
                        output_np,
                        iou_threshold=iou_threshold,
                        ignore_index=ignore_index,
                    )
                )
                boundary_stats.append(
                    chamfer_statistics(
                        truth,
                        output_np,
                        pixel_size_m=spacing,
                        iou_threshold=iou_threshold,
                        ignore_index=ignore_index,
                    )
                )

            if bootstrap:
                object_tps.append(tps)
                object_fps.append(fps)
                object_fns.append(fns)

            all_tps += tps
            all_fps += fps
            all_fns += fns
        sample_offset += len(outputs)

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
            instance_stats=extended_stats if extended_object_metrics else None,
            boundary_stats=boundary_stats if extended_object_metrics else None,
            background_known=background_known,
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

    if extended_object_metrics:
        extended_scores = _extended_object_scores(
            extended_stats, boundary_stats, background_known=background_known
        )
        for name, score in extended_scores.items():
            interval = (
                f" [{confidence_intervals[name][0]:.4f}, "
                f"{confidence_intervals[name][1]:.4f}]"
                if confidence_intervals is not None
                else ""
            )
            print(f"{name}: {score:.4f}{interval}")
        if out is not None:
            row = dict(
                zip(
                    csv_fields[:8],
                    (
                        model_path,
                        country_str,
                        pixel_level_iou,
                        pixel_level_precision,
                        pixel_level_recall,
                        object_precision,
                        object_recall,
                        object_f1,
                    ),
                )
            )
            row.update(extended_scores)
            if confidence_intervals is not None:
                for name, interval in confidence_intervals.items():
                    column = (
                        name
                        if name.startswith("extended_")
                        else name.replace("_", "_level_", 1)
                    )
                    row[column + "_ci_lower"], row[column + "_ci_upper"] = interval
            _check_test_csv_header(out, csv_fields)
            write_header = not os.path.exists(out) or not os.path.getsize(out)
            with open(out, "a", newline="") as target:
                writer = csv.DictWriter(
                    target, fieldnames=csv_fields, lineterminator="\n"
                )
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        return

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
