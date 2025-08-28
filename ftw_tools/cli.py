import enum
import json
import os

import click
import wget

from ftw_tools.download.download_img import scene_selection
from ftw_tools.settings import ALL_COUNTRIES, SUPPORTED_POLY_FORMATS_TXT

# Imports are in the functions below to speed-up CLI startup time
# Some of the ML related imports (presumable torch) are very slow
# See https://github.com/fieldsoftheworld/ftw-baselines/issues/40


class ModelVersions(enum.StrEnum):
    """Mapping from short_name to .ckpt file in github."""

    TWO_CLASS_CCBY = "2_Class_CCBY_FTW_Pretrained.ckpt"
    TWO_CLASS_FULL = "2_Class_FULL_FTW_Pretrained.ckpt"
    THREE_CLASS_CCBY = "3_Class_CCBY_FTW_Pretrained.ckpt"
    THREE_CLASS_FULL = "3_Class_FULL_FTW_Pretrained.ckpt"


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
    type=str,
    default="./data",
    help="Folder where the files will be downloaded to. Defaults to './data'.",
)
@click.option(
    "--clean_download",
    "-f",
    is_flag=True,
    help="If set, the script will delete the root folder before downloading.",
)
@click.option(
    "--countries",
    type=str,
    default="all",
    help="Comma-separated list of countries to download. If 'all' (default) is passed, downloads all available countries. Available countries: "
    + ", ".join(ALL_COUNTRIES),
)
@click.option(
    "--no-unpack",
    is_flag=True,
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
    help="Unpack the downloaded FTW dataset. Specify the folder where the data is located via INPUT. Defaults to './data'.",
)
@click.argument("input", type=str, default="./data")
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
    type=click.Path(exists=True),
    help="Path to the config file (required)",
)
@click.option(
    "--ckpt_path",
    type=click.Path(exists=True),
    help="Path to a checkpoint file to resume training from",
)
@click.argument(
    "cli_args", nargs=-1, type=click.UNPROCESSED
)  # Capture all remaining arguments
def model_fit(config, ckpt_path, cli_args):
    from ftw_tools.models.baseline_eval import fit

    fit(config, ckpt_path, cli_args)


@model.command("test", help="Test the model")
@click.option("--model", "-m", required=True, type=str, help="Path to model checkpoint")
@click.option("--dir", type=str, default="data/ftw", help="Directory of dataset")
@click.option("--gpu", type=int, default=0, help="GPU to use")
@click.option(
    "--countries",
    type=str,
    multiple=True,
    required=True,
    help="Countries to evaluate on",
)
@click.option(
    "--postprocess", is_flag=True, help="Apply postprocessing to the model output"
)
@click.option(
    "--iou_threshold",
    type=float,
    default=0.5,
    help="IoU threshold for matching predictions to ground truths",
)
@click.option(
    "--out", "-o", type=str, default="metrics.json", help="Output file for metrics"
)
@click.option(
    "--model_predicts_3_classes",
    is_flag=True,
    help="Whether the model predicts 3 classes or 2 classes",
)
@click.option(
    "--test_on_3_classes",
    is_flag=True,
    help="Whether to test on 3 classes or 2 classes",
)
@click.option(
    "--temporal_options",
    type=str,
    default="stacked",
    help="Temporal option (stacked, windowA, windowB, etc.)",
)
@click.argument(
    "cli_args", nargs=-1, type=click.UNPROCESSED
)  # Capture all remaining arguments
def model_test(
    model,
    dir,
    gpu,
    countries,
    postprocess,
    iou_threshold,
    out,
    model_predicts_3_classes,
    test_on_3_classes,
    temporal_options,
    cli_args,
):
    from ftw_tools.models.baseline_eval import test

    test(
        model,
        dir,
        gpu,
        countries,
        postprocess,
        iou_threshold,
        out,
        model_predicts_3_classes,
        test_on_3_classes,
        temporal_options,
        cli_args,
    )


@model.command("download", help="Download model checkpoints")
@click.option(
    "--type",
    type=click.Choice(ModelVersions),
    required=True,
    help="Short model name corresponding to a .ckpt file in github.",
)
def model_download(type: ModelVersions):
    github_url = f"https://github.com/fieldsoftheworld/ftw-baselines/releases/download/v1/{type.value}"
    print(f"Downloading {github_url} to {type.value}")
    wget.download(github_url)


### Inference group

WIN_HELP = "URL to or Microsoft Planetary Computer ID of an Sentinel-2 L2A STAC item for the window {x} image"


@ftw.group()
def inference():
    """Inference-related commands."""
    pass


@inference.command(
    "ftw-inference-all",
    help="Run all inference commands from crop calendar scene selection,"
    "then download, inference and polygonize.",
)
@click.option(
    "--bbox",
    type=str,
    default=None,
    help="Bounding box to use for the download in the format 'minx,miny,maxx,maxy'",
    required=True,
)
@click.option(
    "--year", type=int, required=True, help="Year to run model inference over"
)
@click.option(
    "--cloud_cover_max",
    type=int,
    default=20,
    help="Max percent cloud cover in sentinel2 scene",
)
@click.option(
    "--buffer_days",
    type=int,
    default=14,
    help="Number of days to buffer the date for querying to help balance decreasing cloud cover "
    "and selecting a date near the crop calendar indicated date.",
)
@click.option(
    "--out_dir",
    "-o",
    type=str,
    required=True,
    help="Directory to save downloaded inference imagery, and inference output to",
)
@click.option(
    "--overwrite", "-f", is_flag=True, help="Overwrites the outputs if they exist"
)
@click.option(
    "--model",
    "-m",
    type=click.Path(exists=True),
    required=True,
    help="Path to the model checkpoint.",
)
@click.option(
    "--resize_factor",
    type=int,
    default=2,
    show_default=True,
    help="Resize factor to use for inference.",
)
@click.option(
    "--gpu",
    type=int,
    help="GPU ID to use. If not provided, CPU will be used by default.",
)
@click.option(
    "--patch_size",
    type=int,
    default=None,
    help="Size of patch to use for inference. Defaults to 1024 unless the image is < 1024x1024px.",
)
@click.option(
    "--batch_size", type=int, default=2, show_default=True, help="Batch size."
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
    type=int,
    default=None,
    help="Pixels to discard from each side of the patch.",
)
@click.option(
    "--mps_mode", is_flag=True, help="Run inference in MPS mode (Apple GPUs)."
)
@click.option(
    "--stac_host",
    type=click.Choice(["mspc", "earthsearch"]),
    default="earthsearch",
    show_default=True,
    help="The host to download the imagery from. mspc = Microsoft Planetary Computer, earthsearch = EarthSearch (Element84/AWS).",
)
def ftw_inference_all(
    bbox,
    year,
    cloud_cover_max,
    buffer_days,
    out_dir,
    overwrite,
    model,
    resize_factor,
    gpu,
    patch_size,
    batch_size,
    num_workers,
    padding,
    mps_mode,
    stac_host,
):
    """Run all inference commands from crop calendar scene selection, then download, inference and polygonize."""
    from ftw_tools.download.download_img import create_input, scene_selection
    from ftw_tools.models.baseline_inference import run
    from ftw_tools.postprocess.polygonize import polygonize

    bbox_formatted = [float(x) for x in bbox.split(",")]

    # Ensure output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Set output paths
    inference_data = os.path.join(out_dir, "inference_data.tif")
    inf_output_path = os.path.join(out_dir, "inference_output.tif")

    # Scene selection
    win_a, win_b = scene_selection(
        bbox=bbox_formatted,
        year=year,
        cloud_cover_max=cloud_cover_max,
        buffer_days=buffer_days,
    )

    # Download imagery
    create_input(
        win_a=win_a,
        win_b=win_b,
        out=inference_data,
        overwrite=overwrite,
        bbox=bbox,
        stac_host=stac_host,
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
    )

    # Polygonize the output
    polygonize(
        input=inf_output_path,
        out=f"{out_dir}/polygons.parquet",
        simplify=True,
        min_size=15,
        max_size=500,
        overwrite=overwrite,
        close_interiors=True,
    )


@inference.command(
    "scene_selection", help="Select Sentinel-2 scenes for inference with crop calendar"
)
@click.option(
    "--bbox",
    type=str,
    default=None,
    help="Bounding box to use for the download in the format 'minx,miny,maxx,maxy'",
)
@click.option(
    "--year", type=int, required=True, help="Year to run model inference over"
)
@click.option(
    "--cloud_cover_max",
    type=int,
    default=20,
    help="Max percent cloud cover in sentinel2 scene",
)
@click.option(
    "--buffer_days",
    type=int,
    default=14,
    help="Number of days to buffer the date for querying to help balance decreasing cloud cover "
    "and selecting a date near the crop calendar indicated date.",
)
@click.option(
    "--out",
    "-o",
    type=str,
    default=None,
    help="Output JSON file to save the scene selection results. If not provided, prints to stdout.",
)
def scene_selection(year, cloud_cover_max, bbox, buffer_days, out):
    """Download Sentinel-2 scenes for inference."""
    from ftw_tools.download.download_img import scene_selection

    bbox_formatted = [float(x) for x in bbox.split(",")]
    win_a, win_b = scene_selection(
        bbox=bbox_formatted,
        year=year,
        cloud_cover_max=cloud_cover_max,
        buffer_days=buffer_days,
    )
    if out:
        # persist results to json
        result = {
            "window_a": win_a,
            "window_b": win_b,
        }
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {out}")
    else:
        print(f"Window A: {win_a}, Window B: {win_b}")


@inference.command(
    "download",
    help="Download 2 Sentinel-2 scenes & stack them in a single file for inference.",
)
@click.option("--win_a", type=str, required=True, help=WIN_HELP.format(x="A"))
@click.option("--win_b", type=str, default=None, help=WIN_HELP.format(x="B"))
@click.option(
    "--out", "-o", type=str, required=True, help="Filename to save results to"
)
@click.option(
    "--overwrite", "-f", is_flag=True, help="Overwrites the outputs if they exist"
)
@click.option(
    "--bbox",
    type=str,
    default=None,
    help="Bounding box to use for the download in the format 'minx,miny,maxx,maxy'",
)
@click.option(
    "--stac_host",
    "-h",
    type=click.Choice(["mspc", "earthsearch"]),
    default="mspc",
    show_default=True,
    help="The host to download the imagery from. mspc = Microsoft Planetary Computer, earthsearch = EarthSearch (Element84/AWS).",
)
def inference_download(win_a, win_b, out, overwrite, bbox, stac_host):
    from ftw_tools.download.download_img import create_input

    create_input(
        win_a=win_a,
        win_b=win_b,
        out=out,
        overwrite=overwrite,
        bbox=bbox,
        stac_host=stac_host,
    )


@inference.command(
    "run",
    help="Run inference on the stacked Sentinel-2 L2A satellite images specified via INPUT.",
)
@click.argument("input", type=click.Path(exists=True), required=True)
@click.option(
    "--model",
    "-m",
    type=click.Path(exists=True),
    required=True,
    help="Path to the model checkpoint.",
)
@click.option(
    "--out",
    "-o",
    type=str,
    default=None,
    help="Output filename for the inference imagery. Defaults to the name of the input file name with 'inference.' prefix.",
)
@click.option(
    "--resize_factor",
    type=int,
    default=2,
    show_default=True,
    help="Resize factor to use for inference.",
)
@click.option(
    "--gpu",
    type=int,
    help="GPU ID to use. If not provided, CPU will be used by default.",
)
@click.option(
    "--patch_size",
    type=int,
    default=None,
    help="Size of patch to use for inference. Defaults to 1024 unless the image is < 1024x1024px.",
)
@click.option(
    "--batch_size", type=int, default=2, show_default=True, help="Batch size."
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
    type=int,
    default=None,
    help="Pixels to discard from each side of the patch.",
)
@click.option(
    "--overwrite", "-f", is_flag=True, help="Overwrite outputs if they exist."
)
@click.option(
    "--mps_mode", is_flag=True, help="Run inference in MPS mode (Apple GPUs)."
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
    required=True,
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
    "--image_size",
    type=click.IntRange(min=1),
    default=320,
    show_default=True,
    help="Image size to use for inference.",
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
    "--max_detections",
    type=click.IntRange(min=1),
    default=50,
    show_default=True,
    help="Maximum number of detections to keep.",
)
@click.option(
    "--iou_threshold",
    type=float,
    default=0.6,
    show_default=True,
    help="IoU threshold for matching detections to ground truths.",
)
@click.option(
    "--conf_threshold",
    type=float,
    default=0.1,
    show_default=True,
    help="Confidence threshold for keeping detections.",
)
@click.option(
    "--padding",
    type=int,
    default=None,
    help="Pixels to discard from each side of the patch.",
)
@click.option(
    "--overwrite", "-f", is_flag=True, help="Overwrite outputs if they exist."
)
@click.option(
    "--mps_mode", is_flag=True, help="Run inference in MPS mode (Apple GPUs)."
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
    "--close_interiors",
    is_flag=True,
    default=False,
    show_default=True,
    help="Remove the interiors holes in the polygons.",
)
def inference_run_instance_segmentation(
    input,
    model,
    out,
    gpu,
    image_size,
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
):
    from ftw_tools.models.baseline_inference import run_instance_segmentation

    run_instance_segmentation(
        input=input,
        model=model,
        out=out,
        gpu=gpu,
        num_workers=num_workers,
        image_size=image_size,
        patch_size=patch_size,
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
    required=True,
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
    type=click.Choice(["mspc", "earthsearch"]),
    default="earthsearch",
    show_default=True,
    help="The host to download the imagery from. mspc = Microsoft Planetary Computer, earthsearch = EarthSearch (Element84/AWS).",
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(["DelineateAnything", "DelineateAnything-S"]),
    required=True,
    help="The model to use for inference.",
)
@click.option(
    "--gpu",
    type=int,
    help="GPU ID to use. If not provided, CPU will be used by default.",
)
@click.option(
    "--image_size",
    type=int,
    default=320,
    show_default=True,
    help="Image size to use for inference.",
)
@click.option(
    "--patch_size",
    type=int,
    default=None,
    help="Size of patch to use for inference. Defaults to 1024 unless the image is < 1024x1024px.",
)
@click.option(
    "--batch_size", type=int, default=2, show_default=True, help="Batch size."
)
@click.option(
    "--num_workers",
    type=int,
    default=2,
    show_default=True,
    help="Number of workers to use for inference.",
)
@click.option(
    "--max_detections",
    type=int,
    default=50,
    show_default=True,
    help="Maximum number of detections to keep.",
)
@click.option(
    "--iou_threshold",
    type=float,
    default=0.6,
    show_default=True,
    help="IoU threshold for matching detections to ground truths.",
)
@click.option(
    "--conf_threshold",
    type=float,
    default=0.1,
    show_default=True,
    help="Confidence threshold for keeping detections.",
)
@click.option(
    "--padding",
    type=int,
    default=None,
    help="Pixels to discard from each side of the patch.",
)
@click.option(
    "--overwrite", "-f", is_flag=True, help="Overwrite outputs if they exist."
)
@click.option(
    "--mps_mode", is_flag=True, help="Run inference in MPS mode (Apple GPUs)."
)
@click.option(
    "--simplify",
    type=float,
    default=15,
    show_default=True,
    help="Simplification factor to use when polygonizing in the unit of the CRS, e.g. meters for Sentinel-2 imagery in UTM. Set to 0 to disable simplification.",
)
@click.option(
    "--min_size",
    type=float,
    default=500,
    show_default=True,
    help="Minimum area size in square meters to include in the output. Set to 0 to disable.",
)
@click.option(
    "--max_size",
    type=float,
    default=None,
    show_default=True,
    help="Maximum area size in square meters to include in the output. Disabled by default.",
)
@click.option(
    "--close_interiors",
    is_flag=True,
    help="Remove the interiors holes in the polygons.",
)
def ftw_inference_instance_segmentation_all(
    input,
    bbox,
    out_dir,
    stac_host,
    model,
    gpu,
    image_size,
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
        image_size=image_size,
        patch_size=patch_size,
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
    )


@inference.command(
    "polygonize",
    help="Polygonize the output from inference for the raster image given via INPUT. Results are in the CRS of the given raster image.",
)
@click.argument("input", type=click.Path(exists=True), required=True)
@click.option(
    "--out",
    "-o",
    type=str,
    default=None,
    help="Output filename for the polygonized data. Defaults to the name of the input file with parquet extension. "
    + SUPPORTED_POLY_FORMATS_TXT,
)
@click.option(
    "--simplify",
    type=float,
    default=15,
    show_default=True,
    help="Simplification factor to use when polygonizing in the unit of the CRS, e.g. meters for Sentinel-2 imagery in UTM. Set to 0 to disable simplification.",
)
@click.option(
    "--min_size",
    type=float,
    default=500,
    show_default=True,
    help="Minimum area size in square meters to include in the output. Set to 0 to disable.",
)
@click.option(
    "--max_size",
    type=float,
    default=None,
    show_default=True,
    help="Maximum area size in square meters to include in the output. Disabled by default.",
)
@click.option(
    "--overwrite", "-f", is_flag=True, help="Overwrite outputs if they exist."
)
@click.option(
    "--close_interiors",
    is_flag=True,
    help="Remove the interiors holes in the polygons.",
)
def inference_polygonize(
    input, out, simplify, min_size, max_size, overwrite, close_interiors
):
    from ftw_tools.postprocess.polygonize import polygonize

    polygonize(input, out, simplify, min_size, max_size, overwrite, close_interiors)


@inference.command(
    "filter_by_lulc", help="Filter the output raster in GeoTIFF format by LULC mask."
)
@click.argument("input", type=click.Path(exists=True), required=True)
@click.option(
    "--out",
    "-o",
    type=str,
    default=None,
    help="Output filename for the (filtered) polygonized data. Defaults to the name of the input file with parquet extension. "
    + SUPPORTED_POLY_FORMATS_TXT,
)
@click.option(
    "--overwrite", "-f", is_flag=True, help="Overwrite outputs if they exist."
)
@click.option(
    "--collection_name",
    type=str,
    default="io-lulc-annual-v02",
    help="Name of the LULC collection to use. Available collections: io-lulc-annual-v02 (default) and esa-worldcover",
)
@click.option(
    "--save_lulc_tif",
    is_flag=True,
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
