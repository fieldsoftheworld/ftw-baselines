import datetime
import json
import os

import click

# torchvision.ops.nms is not supported on MPS yet
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from ftw_tools.models.model_registry import MODEL_REGISTRY
from ftw_tools.settings import (
    ALL_COUNTRIES,
    LULC_COLLECTIONS,
    S2_COLLECTIONS,
    SUPPORTED_POLY_FORMATS_TXT,
    TEMPORAL_OPTIONS,
)
from ftw_tools.utils import parse_bbox

# Imports are in the functions below to speed-up CLI startup time
# Some of the ML related imports (presumable torch) are very slow
# See https://github.com/fieldsoftheworld/ftw-baselines/issues/40

COUNTRIES_CHOICE = ALL_COUNTRIES.copy()
COUNTRIES_CHOICE.append("all")

# All commands are meant to use dashes as separator for words.
# All parameters are meant to use underscores as separator for words.


# Common parameter definitions for shared CLI options
def common_bbox_option():
    """Common bbox option for inference commands."""
    return click.option(
        "--bbox",
        type=str,
        default=None,
        help="Bounding box to use for the download in the format 'minx,miny,maxx,maxy'",
        callback=parse_bbox,
    )


def common_year_option():
    """Common year option for inference commands."""
    return click.option(
        "--year",
        type=click.IntRange(min=2015, max=datetime.date.today().year),
        required=True,
        help="Year to run model inference over",
    )


def common_cloud_cover_option():
    """Common cloud cover option for inference commands."""
    return click.option(
        "--cloud_cover_max",
        "-ccx",
        type=click.IntRange(min=0, max=100),
        default=20,
        show_default=True,
        help="Maximum percentage of cloud cover allowed in the Sentinel-2 scene",
    )


def common_buffer_days_option():
    """Common buffer days option for inference commands."""
    return click.option(
        "--buffer_days",
        "-b",
        type=click.IntRange(min=0),
        default=14,
        show_default=True,
        help="Number of days to buffer the date for querying to help balance decreasing cloud cover "
        "and selecting a date near the crop calendar indicated date.",
    )


def common_stac_host_option():
    """Common STAC host option for inference commands."""
    return click.option(
        "--stac_host",
        "-h",
        type=click.Choice(["mspc", "earthsearch"]),
        default="mspc",
        show_default=True,
        help="The host to download the imagery from. mspc = Microsoft Planetary Computer, earthsearch = EarthSearch (Element84/AWS).",
    )


def common_s2_collection_option():
    """Common S2 collection option for inference commands."""
    return click.option(
        "--s2_collection",
        "-s2",
        type=click.Choice(list(S2_COLLECTIONS.keys())),
        default="c1",
        show_default=True,
        help="Sentinel-2 collection to use with EarthSearch only: 'old-baseline' = sentinel-2-l2a, 'c1' = sentinel-2-c1-l2a (default). Ignored when using MSPC.",
    )


def common_verbose_option():
    """Common verbose option for inference commands."""
    return click.option(
        "--verbose",
        "-v",
        is_flag=True,
        default=False,
        show_default=True,
        help="Enable verbose output showing STAC calls, scene details, and download URLs.",
    )


@click.group()
def ftw():
    """Fields of The World (FTW) - Command Line Interface"""
    pass


## Data group


@ftw.group()
def data():
    """Downloading, unpacking, and preparing the FTW dataset."""
    pass


@data.command("download", help="Download and unpack the FTW dataset.")
@click.option(
    "--out",
    "-o",
    type=click.Path(exists=False),
    default="./data",
    show_default=True,
    help="Folder where the files will be downloaded to.",
)
@click.option(
    "--clean_download",
    "--clean",
    "-f",
    is_flag=True,
    default=False,
    show_default=True,
    help="If set, the script will delete the folder before downloading.",
)
@click.option(
    "--countries",
    type=click.Choice(COUNTRIES_CHOICE, case_sensitive=False),
    default="all",
    show_default=True,
    help="Comma-separated list of countries to download. The default value 'all' downloads all available countries.",
)
@click.option(
    "--no-unpack",  # deprecated
    "--no_unpack",
    is_flag=True,
    default=False,
    show_default=True,
    help="If set, the script will NOT unpack the downloaded files.",
)
def data_download(out, clean_download, countries, no_unpack):
    from ftw_tools.download.download_ftw import download
    from ftw_tools.download.unpack import unpack

    download(out, clean_download, countries)
    if not no_unpack:
        unpack(out)


@data.command(
    "unpack",
    help="Unpack the downloaded FTW dataset. Specify the folder where the data is located via INPUT, which defaults to './data'.",
)
@click.argument(
    "input",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default="./data",
    required=False,
)
def data_unpack(input):
    from ftw_tools.download.unpack import unpack

    unpack(input)


### Model group


@ftw.group()
def model():
    """Training and testing FTW models."""
    pass


@model.command("fit", help="Fit the model")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the config file",
)
@click.option(
    "--ckpt_path",
    "-m",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    show_default=True,
    help="Path to a checkpoint file to resume training from",
)
@click.argument(
    "cli_args", nargs=-1, type=click.UNPROCESSED
)  # Capture all remaining arguments
def model_fit(config, ckpt_path, cli_args):
    from ftw_tools.models.baseline_eval import fit

    fit(config, ckpt_path, cli_args)


@model.command("test", help="Test the model")
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to model checkpoint",
)
@click.option(
    "--countries",
    "-c",
    type=click.Choice(COUNTRIES_CHOICE, case_sensitive=False),
    multiple=True,
    required=True,
    help="Countries to evaluate on",
)
@click.option(
    "--dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="./data/ftw",
    show_default=True,
    help="Directory of the FTW dataset",
)
@click.option(
    "--gpu",
    type=int,
    default=0,
    show_default=True,
    help="GPU to use, zero-based index. Set to -1 to use CPU. CPU is also always used if CUDA is not available.",
)
@click.option(
    "--iou_threshold",
    "-iou",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.5,
    show_default=True,
    help="IoU threshold for matching predictions to ground truths",
)
@click.option(
    "--out",
    "-o",
    type=click.Path(exists=False),
    default="metrics.json",
    show_default=True,
    help="Output file for metrics",
)
@click.option(
    "--model_predicts_3_classes",
    "-p3",
    is_flag=True,
    default=False,
    show_default=True,
    help="Whether the model predicts 3 classes or 2 classes (default)",
)
@click.option(
    "--test_on_3_classes",
    "-t3",
    is_flag=True,
    default=False,
    show_default=True,
    help="Whether to test on 3 classes or 2 classes (default)",
)
@click.option(
    "--temporal_options",
    "-t",
    type=click.Choice(TEMPORAL_OPTIONS),
    default="stacked",
    show_default=True,
    help="Temporal option",
)
@click.option(
    "--use_val_set",
    is_flag=True,
    default=False,
    show_default=True,
    help="Whether to run evaluation on the val set or test set (default).",
)
@click.option(
    "--swap_order",
    is_flag=True,
    default=False,
    show_default=True,
    help="Whether to run inference on (window_a, window_b) instead of the default (window_b, window_a).",
)
@click.option(
    "--num_workers",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Number of workers to use for inference.",
)
def model_test(
    model,
    countries,
    dir,
    gpu,
    iou_threshold,
    out,
    model_predicts_3_classes,
    test_on_3_classes,
    temporal_options,
    use_val_set,
    swap_order,
    num_workers,
):
    from ftw_tools.models.baseline_eval import test

    test(
        model,
        dir,
        gpu,
        countries,
        iou_threshold,
        out,
        model_predicts_3_classes,
        test_on_3_classes,
        temporal_options,
        use_val_set,
        swap_order,
        num_workers,
    )


### Inference group

WIN_HELP = "URL to a Sentinel-2 L2A STAC item for the window {x} image. Alternatively, an ID of a STAC Item on Microsoft Planetary Computer."


@ftw.group()
def inference():
    """Inference-related commands."""
    pass


@inference.command(
    "all",
    help="Run all inference commands from crop calendar scene selection,"
    "then download, inference and polygonize.",
)
@click.option(
    "--out",
    "-o",
    type=click.Path(exists=False),
    required=True,
    help="Directory to save downloaded inference imagery, and inference output to",
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(MODEL_REGISTRY.keys()),
    required=True,
    help="Short model name corresponding to a .ckpt file in github.",
)
@common_year_option()
@common_bbox_option()
@common_cloud_cover_option()
@common_buffer_days_option()
@click.option(
    "--overwrite",
    "-f",
    is_flag=True,
    default=False,
    show_default=True,
    help="Overwrites the outputs if they exist",
)
@click.option(
    "--resize_factor",
    "-r",
    type=click.IntRange(min=1),
    default=2,
    show_default=True,
    help="Resize factor to use for inference.",
)
@click.option(
    "--gpu",
    type=int,
    default=-1,
    show_default=True,
    help="GPU to use, zero-based index. Set to -1 to use CPU. CPU is also always used if CUDA or MPS is not available.",
)
@click.option(
    "--patch_size",
    "-ps",
    type=click.IntRange(min=128),
    default=None,
    help="Size of patch to use for inference. Defaults to 1024 unless the image is < 1024x1024px and a smaller value otherwise.",
)
@click.option(
    "--batch_size",
    "-bs",
    type=click.IntRange(min=1),
    default=2,
    show_default=True,
    help="Batch size.",
)
@click.option(
    "--num_workers",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Number of workers to use for inference.",
)
@click.option(
    "--padding",
    "-p",
    type=click.IntRange(min=0),
    default=None,
    help="Pixels to discard from each side of the patch. Defaults to 64 unless the image is < 1024x1024px and a smaller value otherwise.",
)
@click.option(
    "--mps_mode",
    "-mps",
    is_flag=True,
    default=False,
    show_default=True,
    help="Run inference in MPS mode (Apple GPUs).",
)
@click.option(
    "--save_scores",
    is_flag=True,
    default=False,
    show_default=True,
    help="Save segmentation softmax scores (rescaled to [0,255]) instead of classes (argmax of scores)",
)
@common_stac_host_option()
@common_s2_collection_option()
@common_verbose_option()
def ftw_inference_all(
    out,
    model,
    year,
    bbox,
    cloud_cover_max,
    buffer_days,
    overwrite,
    resize_factor,
    gpu,
    patch_size,
    batch_size,
    num_workers,
    padding,
    mps_mode,
    save_scores,
    stac_host,
    s2_collection,
    verbose,
):
    """Run all inference commands from crop calendar scene selection, then download, inference and polygonize."""
    from ftw_tools.download.download_img import create_input, scene_selection
    from ftw_tools.models.baseline_inference import run
    from ftw_tools.postprocess.polygonize import polygonize

    # Ensure output directory exists
    if not os.path.exists(out):
        os.makedirs(out)

    # Set output paths
    inference_data = os.path.join(out, "inference_data.tif")
    inf_output_path = os.path.join(out, "inference_output.tif")

    # Scene selection
    win_a, win_b = scene_selection(
        bbox=bbox,
        year=year,
        stac_host=stac_host,
        cloud_cover_max=cloud_cover_max,
        buffer_days=buffer_days,
        s2_collection=s2_collection,
        verbose=verbose,
    )

    # Download imagery
    create_input(
        win_a=win_a,
        win_b=win_b,
        out=inference_data,
        overwrite=overwrite,
        bbox=bbox,
        stac_host=stac_host,
        s2_collection=s2_collection,
        verbose=verbose,
    )

    # Run inference
    run(
        input=inference_data,
        model=model,
        out=inf_output_path,
        resize_factor=resize_factor,
        gpu=gpu,
        patch_size=patch_size,
        batch_size=batch_size,
        num_workers=num_workers,
        padding=padding,
        overwrite=overwrite,
        mps_mode=mps_mode,
        save_scores=save_scores,
    )

    # Polygonize the output
    polygonize(
        input=inf_output_path, out=f"{out}/polygons.parquet", overwrite=overwrite
    )


@inference.command(
    "scene-selection", help="Select Sentinel-2 scenes for inference with crop calendar"
)
@common_year_option()
@common_bbox_option()
@common_cloud_cover_option()
@common_buffer_days_option()
@click.option(
    "--out",
    "-o",
    type=click.Path(exists=False),
    default=None,
    help="Output JSON file to save the scene selection results. If not provided, prints to stdout.",
)
@common_stac_host_option()
@common_s2_collection_option()
@common_verbose_option()
def scene_selection(
    year, bbox, cloud_cover_max, buffer_days, out, stac_host, s2_collection, verbose
):
    """Download Sentinel-2 scenes for inference."""
    from ftw_tools.download.download_img import scene_selection

    win_a, win_b = scene_selection(
        bbox=bbox,
        year=year,
        stac_host=stac_host,
        cloud_cover_max=cloud_cover_max,
        buffer_days=buffer_days,
        s2_collection=s2_collection,
        verbose=verbose,
    )
    if out:
        # persist results to json
        result = {"window_a": win_a, "window_b": win_b}
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {out}")
    else:
        print(f"Window A: {win_a}, Window B: {win_b}")


@inference.command(
    "download",
    help="Download 2 Sentinel-2 scenes & stack them in a single file for inference.",
)
@click.option("--win_a", "-a", type=str, required=True, help=WIN_HELP.format(x="A"))
@click.option("--win_b", "-b", type=str, default=None, help=WIN_HELP.format(x="B"))
@click.option(
    "--out",
    "-o",
    type=click.Path(exists=False),
    required=True,
    help="Filename to save results to",
)
@click.option(
    "--overwrite",
    "-f",
    is_flag=True,
    default=False,
    show_default=True,
    help="Overwrites the outputs if they exist",
)
@common_bbox_option()
@common_stac_host_option()
@common_s2_collection_option()
@common_verbose_option()
def inference_download(
    win_a, win_b, out, overwrite, bbox, stac_host, s2_collection, verbose
):
    from ftw_tools.download.download_img import create_input

    create_input(
        win_a=win_a,
        win_b=win_b,
        out=out,
        overwrite=overwrite,
        bbox=bbox,
        stac_host=stac_host,
        s2_collection=s2_collection,
        verbose=verbose,
    )


@inference.command(
    "run",
    help="Run inference on the stacked Sentinel-2 L2A satellite images specified via INPUT.",
)
@click.argument("input", type=click.Path(exists=True), required=True)
@click.option(
    "--model",
    "-m",
    type=click.Choice(list(MODEL_REGISTRY.keys())),
    required=True,
    help="Short model name corresponding to a released model",
)
@click.option(
    "--out",
    "-o",
    type=click.Path(exists=False),
    default=None,
    help="Output filename for the inference imagery. Defaults to the name of the input file name with 'inference.' prefix.",
)
@click.option(
    "--resize_factor",
    "-r",
    type=click.IntRange(min=1),
    default=2,
    show_default=True,
    help="Resize factor to use for inference.",
)
@click.option(
    "--gpu",
    type=int,
    default=-1,
    show_default=True,
    help="GPU to use, zero-based index. Set to -1 to use CPU. CPU is also always used if CUDA or MPS is not available.",
)
@click.option(
    "--patch_size",
    "-ps",
    type=click.IntRange(min=128),
    default=None,
    help="Size of patch to use for inference. Defaults to 1024 unless the image is < 1024x1024px and a smaller value otherwise.",
)
@click.option(
    "--batch_size",
    "-bs",
    type=click.IntRange(min=1),
    default=2,
    show_default=True,
    help="Batch size.",
)
@click.option(
    "--num_workers",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Number of workers to use for inference.",
)
@click.option(
    "--padding",
    "-p",
    type=click.IntRange(min=0),
    default=None,
    help="Pixels to discard from each side of the patch. Defaults to 64 unless the image is < 1024x1024px and a smaller value otherwise.",
)
@click.option(
    "--overwrite",
    "-f",
    is_flag=True,
    default=False,
    show_default=True,
    help="Overwrite outputs if they exist.",
)
@click.option(
    "--mps_mode",
    "-mps",
    is_flag=True,
    default=False,
    show_default=True,
    help="Run inference in MPS mode (Apple GPUs).",
)
@click.option(
    "--save_scores",
    is_flag=True,
    default=False,
    show_default=True,
    help="Save segmentation softmax scores (rescaled to [0,255]) instead of classes (argmax of scores)",
)
def inference_run(
    input,
    model,
    out,
    resize_factor,
    gpu,
    patch_size,
    batch_size,
    num_workers,
    padding,
    overwrite,
    mps_mode,
    save_scores,
):
    from ftw_tools.models.baseline_inference import run

    run(
        input,
        model,
        out,
        resize_factor,
        gpu,
        patch_size,
        batch_size,
        num_workers,
        padding,
        overwrite,
        mps_mode,
        save_scores,
    )


@inference.command(
    "run-instance-segmentation",
    help="Run an instance segmentation model inference on a single Sentinel-2 L2A satellite images specified via INPUT.",
)
@click.argument("input", type=click.Path(exists=True), required=True)
@click.option(
    "--model",
    "-m",
    type=click.Choice(["DelineateAnything", "DelineateAnything-S"]),
    default="DelineateAnything",
    show_default=True,
    help="The model to use for inference.",
)
@click.option(
    "--out",
    "-o",
    type=click.Path(exists=False),
    default=None,
    help="Output filename for the polygonized data. Defaults to the name of the input file with parquet extension. "
    + SUPPORTED_POLY_FORMATS_TXT,
)
@click.option(
    "--gpu",
    type=click.IntRange(min=0),
    default=None,
    help="GPU ID to use. If not provided, CPU will be used by default.",
)
@click.option(
    "--resize_factor",
    "-r",
    type=click.IntRange(min=1),
    default=2,
    show_default=True,
    help="Resize factor to use for inference.",
)
@click.option(
    "--patch_size",
    "-ps",
    type=click.IntRange(min=128),
    default=256,
    help="Size of patch to use for inference.",
)
@click.option(
    "--batch_size",
    "-bs",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Batch size.",
)
@click.option(
    "--num_workers",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Number of workers to use for inference.",
)
@click.option(
    "--max_detections",
    type=click.IntRange(min=1),
    default=100,
    show_default=True,
    help="Maximum number of detections to keep per patch.",
)
@click.option(
    "--iou_threshold",
    "-iou",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.3,
    show_default=True,
    help="IoU threshold for matching predictions to ground truths",
)
@click.option(
    "--conf_threshold",
    "-ct",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.05,
    show_default=True,
    help="Confidence threshold for keeping detections.",
)
@click.option(
    "--padding",
    "-p",
    type=click.IntRange(min=0),
    default=None,
    help="Pixels to discard from each side of the patch.",
)
@click.option(
    "--overwrite",
    "-f",
    is_flag=True,
    default=False,
    show_default=True,
    help="Overwrites the outputs if they exist",
)
@click.option(
    "--mps_mode",
    "-mps",
    is_flag=True,
    default=False,
    show_default=True,
    help="Run inference in MPS mode (Apple GPUs).",
)
@click.option(
    "--simplify",
    "-s",
    type=click.FloatRange(min=0.0),
    default=2,
    show_default=True,
    help="Simplification factor to use when polygonizing in the unit of the CRS, e.g. meters for Sentinel-2 imagery in UTM. Set to 0 to disable simplification.",
)
@click.option(
    "--min_size",
    "-sn",
    type=click.FloatRange(min=0.0),
    default=500,
    show_default=True,
    help="Minimum area size in square meters to include in the output. Set to 0 to disable.",
)
@click.option(
    "--max_size",
    "-sx",
    type=click.FloatRange(min=0.0),
    default=100000,
    show_default=True,
    help="Maximum area size in square meters to include in the output. Disabled by default.",
)
@click.option(
    "--close_interiors",
    is_flag=True,
    default=True,
    show_default=True,
    help="Remove the interiors holes in the polygons.",
)
@click.option(
    "--overlap_iou_threshold",
    "-oit",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.2,
    show_default=True,
    help="Overlap IoU threshold for merging polygons.",
)
@click.option(
    "--overlap_contain_threshold",
    "-cot",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.8,
    show_default=True,
    help="Overlap containment threshold for merging polygons.",
)
def inference_run_instance_segmentation(
    input,
    model,
    out,
    gpu,
    resize_factor,
    patch_size,
    batch_size,
    num_workers,
    max_detections,
    iou_threshold,
    conf_threshold,
    padding,
    overwrite,
    mps_mode,
    simplify,
    min_size,
    max_size,
    close_interiors,
    overlap_iou_threshold,
    overlap_contain_threshold,
):
    from ftw_tools.models.baseline_inference import run_instance_segmentation

    run_instance_segmentation(
        input=input,
        model=model,
        out=out,
        gpu=gpu,
        num_workers=num_workers,
        patch_size=patch_size,
        resize_factor=resize_factor,
        batch_size=batch_size,
        max_detections=max_detections,
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
        padding=padding,
        overwrite=overwrite,
        mps_mode=mps_mode,
        simplify=simplify,
        min_size=min_size,
        max_size=max_size,
        close_interiors=close_interiors,
        overlap_iou_threshold=overlap_iou_threshold,
        overlap_contain_threshold=overlap_contain_threshold,
    )


@inference.command(
    "instance-segmentation-all",
    help="Run all inference instance segmentation commands from download and inference.",
)
@click.argument("input", type=str, required=True)
@click.option(
    "--bbox",
    type=str,
    default=None,
    help="Bounding box to use for the download in the format 'minx,miny,maxx,maxy'",
    callback=parse_bbox,
)
@click.option(
    "--out_dir",
    "-o",
    type=str,
    required=True,
    help="Directory to save downloaded inference imagery, and inference output to",
)
@click.option(
    "--stac_host",
    "-h",
    type=click.Choice(["mspc", "earthsearch"]),
    default="mspc",
    show_default=True,
    help="The host to download the imagery from. mspc = Microsoft Planetary Computer, earthsearch = EarthSearch (Element84/AWS).",
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(["DelineateAnything", "DelineateAnything-S"]),
    default="DelineateAnything",
    show_default=True,
    help="The model to use for inference.",
)
@click.option(
    "--gpu",
    type=click.IntRange(min=0),
    default=None,
    help="GPU ID to use. If not provided, CPU will be used by default.",
)
@click.option(
    "--resize_factor",
    "-r",
    type=click.IntRange(min=1),
    default=2,
    show_default=True,
    help="Resize factor to use for inference.",
)
@click.option(
    "--patch_size",
    "-ps",
    type=click.IntRange(min=128),
    default=256,
    help="Size of patch to use for inference.",
)
@click.option(
    "--batch_size",
    "-bs",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Batch size.",
)
@click.option(
    "--num_workers",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Number of workers to use for inference.",
)
@click.option(
    "--max_detections",
    type=click.IntRange(min=1),
    default=100,
    show_default=True,
    help="Maximum number of detections to keep per patch.",
)
@click.option(
    "--iou_threshold",
    "-iou",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.3,
    show_default=True,
    help="IoU threshold for matching predictions to ground truths",
)
@click.option(
    "--conf_threshold",
    "-ct",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.05,
    show_default=True,
    help="Confidence threshold for keeping detections.",
)
@click.option(
    "--padding",
    "-p",
    type=click.IntRange(min=0),
    default=None,
    help="Pixels to discard from each side of the patch.",
)
@click.option(
    "--overwrite",
    "-f",
    is_flag=True,
    default=False,
    show_default=True,
    help="Overwrites the outputs if they exist",
)
@click.option(
    "--mps_mode",
    "-mps",
    is_flag=True,
    default=False,
    show_default=True,
    help="Run inference in MPS mode (Apple GPUs).",
)
@click.option(
    "--simplify",
    "-s",
    type=click.FloatRange(min=0.0),
    default=2,
    show_default=True,
    help="Simplification factor to use when polygonizing in the unit of the CRS, e.g. meters for Sentinel-2 imagery in UTM. Set to 0 to disable simplification.",
)
@click.option(
    "--min_size",
    "-sn",
    type=click.FloatRange(min=0.0),
    default=500,
    show_default=True,
    help="Minimum area size in square meters to include in the output. Set to 0 to disable.",
)
@click.option(
    "--max_size",
    "-sx",
    type=click.FloatRange(min=0.0),
    default=100000,
    show_default=True,
    help="Maximum area size in square meters to include in the output. Disabled by default.",
)
@click.option(
    "--close_interiors",
    is_flag=True,
    default=True,
    show_default=True,
    help="Remove the interiors holes in the polygons.",
)
@click.option(
    "--overlap_iou_threshold",
    "-oit",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.2,
    show_default=True,
    help="Overlap IoU threshold for merging polygons.",
)
@click.option(
    "--overlap_contain_threshold",
    "-cot",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.8,
    show_default=True,
    help="Overlap containment threshold for merging polygons.",
)
def inference_run_instance_segmentation_all(
    input,
    bbox,
    out_dir,
    stac_host,
    model,
    gpu,
    resize_factor,
    patch_size,
    batch_size,
    num_workers,
    max_detections,
    iou_threshold,
    conf_threshold,
    padding,
    overwrite,
    mps_mode,
    simplify,
    min_size,
    max_size,
    close_interiors,
    overlap_iou_threshold,
    overlap_contain_threshold,
):
    """Run all inference instance segmentation commands from download and inference."""
    from ftw_tools.download.download_img import create_input
    from ftw_tools.models.baseline_inference import run_instance_segmentation

    # Ensure output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Set output paths
    inference_data = os.path.join(out_dir, "inference_data.tif")
    inf_output_path = os.path.join(out_dir, "inference_output.parquet")

    # Download imagery
    create_input(
        win_a=input,
        win_b=None,
        out=inference_data,
        overwrite=overwrite,
        bbox=bbox,
        stac_host=stac_host,
    )

    # Run inference
    run_instance_segmentation(
        input=inference_data,
        model=model,
        out=inf_output_path,
        gpu=gpu,
        num_workers=num_workers,
        patch_size=patch_size,
        resize_factor=resize_factor,
        batch_size=batch_size,
        max_detections=max_detections,
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
        padding=padding,
        overwrite=overwrite,
        mps_mode=mps_mode,
        simplify=simplify,
        min_size=min_size,
        max_size=max_size,
        close_interiors=close_interiors,
        overlap_iou_threshold=overlap_iou_threshold,
        overlap_contain_threshold=overlap_contain_threshold,
    )


@inference.command(
    "polygonize",
    help="Polygonize the output from inference for the raster image given via INPUT. Results are in the CRS of the given raster image.",
)
@click.argument("input", type=click.Path(exists=True), required=True)
@click.option(
    "--out",
    "-o",
    type=click.Path(exists=False),
    default=None,
    help="Output filename for the polygonized data. Defaults to the name of the input file with '.parquet' file extension. "
    + SUPPORTED_POLY_FORMATS_TXT,
)
@click.option(
    "--simplify",
    "-s",
    type=click.FloatRange(min=0.0),
    default=15,
    show_default=True,
    help="Simplification factor to use when polygonizing in the unit of the CRS, e.g. meters for Sentinel-2 imagery in UTM. Set to 0 to disable simplification.",
)
@click.option(
    "--min_size",
    "-sn",
    type=click.FloatRange(min=0.0),
    default=500,
    show_default=True,
    help="Minimum area size in square meters to include in the output. Set to 0 to disable.",
)
@click.option(
    "--max_size",
    "-sx",
    type=click.FloatRange(min=0.0),
    default=None,
    show_default=True,
    help="Maximum area size in square meters to include in the output. Disabled by default.",
)
@click.option(
    "--overwrite",
    "-f",
    is_flag=True,
    default=False,
    show_default=True,
    help="Overwrite output if it exists.",
)
@click.option(
    "--close_interiors",
    is_flag=True,
    default=False,
    show_default=True,
    help="Remove the interiors holes in the polygons.",
)
@click.option(
    "--stride",
    "-st",
    type=click.IntRange(min=0),
    default=2048,
    show_default=True,
    help="Stride size (in pixels) for cutting tif into smaller tiles for polygonizing. Helps avoid OOM errors.",
)
@click.option(
    "--softmax_threshold",
    type=click.FloatRange(min=0, max=1),
    default=None,
    show_default=True,
    help="Threshold on softmax scores for class predictions. Note: To use this option, you must pass a tif of scores (using `--save_scores` option from `ftw inference run`).",
)
@click.option(
    "--merge_adjacent",
    "-ma",
    type=click.FloatRange(min=0.0, max=1.0),
    default=None,
    show_default=True,
    help="Threshold for merging adjacent polygons. Threshold is the percent of a polygon's perimeter touching another polygon.",
)
@click.option(
    "--erode_dilate",
    "-ed",
    type=click.FloatRange(min=0.0),
    default=0,
    show_default=True,
    help="Distance (in CRS units, e.g., meters) for a morphological opening (erode then dilate) applied to each polygon to shave spurs and remove thin slivers. Set 0 to disable. A good starting value is 0.5–1x the raster pixel size.",
)
@click.option(
    "--dilate_erode",
    "-de",
    type=click.FloatRange(min=0.0),
    default=0,
    show_default=True,
    help="Distance (in CRS units, e.g., meters) for a morphological closing (dilate then erode) applied to each polygon to seal hairline gaps, fill pinholes, and connect near-touching parts without net growth. Set 0 to disable. A good starting value is 0.5–1x the raster pixel size.",
)
@click.option(
    "--erode_dilate_raster",
    "-edr",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Number of iterations for a morphological opening (erode then dilate) applied to raster mask before polygonization. Set to 0 to disable.",
)
@click.option(
    "--dilate_erode_raster",
    "-der",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Number of iterations for a morphological closing (dilate then erode) applied to raster mask before polygonization. Set to 0 to disable.",
)
@click.option(
    "--thin_boundaries",
    "-tb",
    is_flag=True,
    default=False,
    show_default=True,
    help="Thin boundaries before polygonization using Zhang-Suen thinning algorithm.",
)
def inference_polygonize(
    input,
    out,
    simplify,
    min_size,
    max_size,
    overwrite,
    close_interiors,
    stride,
    softmax_threshold,
    merge_adjacent,
    erode_dilate,
    dilate_erode,
    erode_dilate_raster,
    dilate_erode_raster,
    thin_boundaries,
):
    from ftw_tools.postprocess.polygonize import polygonize

    polygonize(
        input,
        out,
        simplify,
        min_size,
        max_size,
        overwrite,
        close_interiors,
        stride,
        softmax_threshold,
        merge_adjacent,
        erode_dilate,
        dilate_erode,
        erode_dilate_raster,
        dilate_erode_raster,
        thin_boundaries,
    )


@inference.command(
    "filter-by-lulc", help="Filter the output raster in GeoTIFF format by LULC mask."
)
@click.argument("input", type=click.Path(exists=True), required=True)
@click.option(
    "--out",
    "-o",
    type=click.Path(exists=False),
    default=None,
    help="Output filename for the (filtered) polygonized data. Defaults to the name of the input file with '.parquet' file extension. "
    + SUPPORTED_POLY_FORMATS_TXT,
)
@click.option(
    "--overwrite",
    "-f",
    is_flag=True,
    default=False,
    show_default=True,
    help="Overwrite output if it exists.",
)
@click.option(
    "--collection_name",
    "-c",
    type=click.Choice(LULC_COLLECTIONS),
    default="io-lulc-annual-v02",
    show_default=True,
    help="Name of the LULC collection to use.",
)
@click.option(
    "--save_lulc_tif",
    "-tiff",
    is_flag=True,
    default=False,
    show_default=True,
    help="Save the LULC mask as a GeoTIFF.",
)
def inference_lulc_filtering(input, out, overwrite, collection_name, save_lulc_tif):
    from ftw_tools.postprocess.lulc_filtering import lulc_filtering

    lulc_filtering(
        input=input,
        out=out,
        overwrite=overwrite,
        collection_name=collection_name,
        save_lulc_tif=save_lulc_tif,
    )


if __name__ == "__main__":
    ftw()
