import datetime
import enum
import json
import os
from pathlib import Path
from typing import Optional

import click
import wget

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


class ModelVersions(enum.StrEnum):
    """Mapping from short_name to .ckpt file in github."""

    TWO_CLASS_CCBY = "2_Class_CCBY_FTW_Pretrained.ckpt"
    TWO_CLASS_FULL = "2_Class_FULL_FTW_Pretrained.ckpt"
    THREE_CLASS_CCBY = "3_Class_CCBY_FTW_Pretrained.ckpt"
    THREE_CLASS_FULL = "3_Class_FULL_FTW_Pretrained.ckpt"


# All commands are meant to use dashes as separator for words.
# All parameters are meant to use underscores as separator for words.


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
    "--postprocess",
    "-pp",
    is_flag=True,
    default=False,
    show_default=True,
    help="Apply postprocessing to the model output",
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
def model_test(
    model,
    countries,
    dir,
    gpu,
    postprocess,
    iou_threshold,
    out,
    model_predicts_3_classes,
    test_on_3_classes,
    temporal_options,
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
    )


@model.command("download", help="Download model checkpoints")
@click.option(
    "--type",
    type=click.Choice(ModelVersions),
    required=True,
    help="Short model name corresponding to a .ckpt file in github.",
)
@click.option(
    "--out",
    "-o",
    type=click.Path(exists=False),
    default=None,
    show_default=True,
    help="File where the file will be stored to. Defaults to the original filename of the selected model.",
)
def model_download(type: ModelVersions, out: Optional[str] = None):
    github_url = f"https://github.com/fieldsoftheworld/ftw-baselines/releases/download/v1/{type.value}"
    target = Path(out or type.value)
    if target.exists():
        print(f"File {target} already exists, skipping download.")
        return

    print(f"Downloading {github_url} to {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    wget.download(github_url, str(target.resolve()))


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
    type=click.Path(exists=True),
    required=True,
    help="Path to the model checkpoint.",
)
@click.option(
    "--year",
    type=click.IntRange(min=2015, max=datetime.date.today().year),
    required=True,
    help="Year to run model inference over",
)
@click.option(
    "--bbox",
    type=str,
    default=None,
    help="Bounding box to use for the download in the format 'minx,miny,maxx,maxy'",
    callback=parse_bbox,
)
@click.option(
    "--cloud_cover_max",
    "-ccx",
    type=click.IntRange(min=0, max=100),
    default=20,
    show_default=True,
    help="Maximum percentage of cloud cover allowed in the Sentinel-2 scene",
)
@click.option(
    "--buffer_days",
    "-b",
    type=click.IntRange(min=0),
    default=14,
    show_default=True,
    help="Number of days to buffer the date for querying to help balance decreasing cloud cover "
    "and selecting a date near the crop calendar indicated date.",
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
    "--stac_host",
    "-h",
    type=click.Choice(["mspc", "earthsearch"]),
    default="mspc",
    show_default=True,
    help="The host to download the imagery from. mspc = Microsoft Planetary Computer, earthsearch = EarthSearch (Element84/AWS).",
)
@click.option(
    "--s2_collection",
    "-s2",
    type=click.Choice(list(S2_COLLECTIONS.keys())),
    default="c1",
    show_default=True,
    help="Sentinel-2 collection to use with EarthSearch only: 'old-baseline' = sentinel-2-l2a, 'c1' = sentinel-2-c1-l2a (default). Ignored when using MSPC.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    show_default=True,
    help="Enable verbose output showing STAC calls, scene details, and download URLs.",
)
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
    padding,
    mps_mode,
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
        padding=padding,
        overwrite=overwrite,
        mps_mode=mps_mode,
    )

    # Polygonize the output
    polygonize(
        input=inf_output_path,
        out=f"{out}/polygons.parquet",
        overwrite=overwrite,
    )


@inference.command(
    "scene-selection", help="Select Sentinel-2 scenes for inference with crop calendar"
)
@click.option(
    "--year",
    type=click.IntRange(min=2015, max=datetime.date.today().year),
    required=True,
    help="Year to run model inference over",
)
@click.option(
    "--bbox",
    type=str,
    default=None,
    help="Bounding box to use for the download in the format 'minx,miny,maxx,maxy'",
    callback=parse_bbox,
)
@click.option(
    "--cloud_cover_max",
    "-ccx",
    type=click.IntRange(min=0, max=100),
    default=20,
    show_default=True,
    help="Maximum percentage of cloud cover allowed in the Sentinel-2 scene",
)
@click.option(
    "--buffer_days",
    "-b",
    type=click.IntRange(min=0),
    default=14,
    show_default=True,
    help="Number of days to buffer the date for querying to help balance decreasing cloud cover "
    "and selecting a date near the crop calendar indicated date.",
)
@click.option(
    "--out",
    "-o",
    type=click.Path(exists=False),
    default=None,
    help="Output JSON file to save the scene selection results. If not provided, prints to stdout.",
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
    "--s2_collection",
    "-s2",
    type=click.Choice(list(S2_COLLECTIONS.keys())),
    default="c1",
    show_default=True,
    help="Sentinel-2 collection to use with EarthSearch only: 'old-baseline' = sentinel-2-l2a, 'c1' = sentinel-2-c1-l2a (default). Ignored when using MSPC.",
)
def scene_selection(
    year, bbox, cloud_cover_max, buffer_days, out, stac_host, s2_collection
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
@click.option("--win_a", "-a", type=str, required=True, help=WIN_HELP.format(x="A"))
@click.option("--win_b", "-b", type=str, required=True, help=WIN_HELP.format(x="B"))
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
@click.option(
    "--bbox",
    type=str,
    default=None,
    help="Bounding box to use for the download in the format 'minx,miny,maxx,maxy'",
    callback=parse_bbox,
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
    "--s2_collection",
    "-s2",
    type=click.Choice(list(S2_COLLECTIONS.keys())),
    default="c1",
    show_default=True,
    help="Sentinel-2 collection to use with EarthSearch only: 'old-baseline' = sentinel-2-l2a, 'c1' = sentinel-2-c1-l2a (default). Ignored when using MSPC.",
)
def inference_download(win_a, win_b, out, overwrite, bbox, stac_host, s2_collection):
    from ftw_tools.download.download_img import create_input

    create_input(
        win_a=win_a,
        win_b=win_b,
        out=out,
        overwrite=overwrite,
        bbox=bbox,
        stac_host=stac_host,
        s2_collection=s2_collection,
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
def inference_run(
    input,
    model,
    out,
    resize_factor,
    gpu,
    patch_size,
    batch_size,
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
        padding,
        overwrite,
        mps_mode,
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
def inference_polygonize(
    input, out, simplify, min_size, max_size, overwrite, close_interiors
):
    from ftw_tools.postprocess.polygonize import polygonize

    polygonize(input, out, simplify, min_size, max_size, overwrite, close_interiors)


@inference.command(
    "filter-by-lulc",
    help="Filter the output raster in GeoTIFF format by LULC mask.",
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
