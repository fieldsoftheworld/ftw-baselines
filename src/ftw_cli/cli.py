from pathlib import Path
from typing import Optional

import click
import wget

from .cfg import ALL_COUNTRIES, SUPPORTED_POLY_FORMATS_TXT
from .types import ModelVersions

# Imports are in the functions below to speed-up CLI startup time
# Some of the ML related imports (presumable torch) are very slow
# See https://github.com/fieldsoftheworld/ftw-baselines/issues/40


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
    from ftw_cli.download_ftw import download
    from ftw_cli.unpack import unpack

    download(out, clean_download, countries)
    if not no_unpack:
        unpack(out)


@data.command(
    "unpack",
    help="Unpack the downloaded FTW dataset. Specify the folder where the data is located via INPUT. Defaults to './data'.",
)
@click.argument("input", type=str, default="./data")
def data_unpack(input):
    from ftw_cli.unpack import unpack

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
    from ftw_cli.model import fit

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
    from ftw_cli.model import test

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

WIN_HELP = "URL to or Microsoft Planetary Computer ID of an Sentinel-2 L2A STAC item for the window {x} image"


@ftw.group()
def inference():
    """Inference-related commands."""
    pass


@inference.command(
    "download",
    help="Download 2 Sentinel-2 scenes & stack them in a single file for inference.",
)
@click.option("--win_a", type=str, required=True, help=WIN_HELP.format(x="A"))
@click.option("--win_b", type=str, required=True, help=WIN_HELP.format(x="B"))
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
def inference_download(win_a, win_b, out, overwrite, bbox):
    from ftw_cli.download_img import create_input

    create_input(win_a, win_b, out, overwrite, bbox)


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
    padding,
    overwrite,
    mps_mode,
):
    from ftw_cli.inference import run

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
    from ftw_cli.polygonize import polygonize

    polygonize(input, out, simplify, min_size, max_size, overwrite, close_interiors)


@inference.command(
    "filter-by-lulc", help="Filter the output raster in GeoTIFF format by LULC mask."
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
    from ftw_cli.lulc_filtering import lulc_filtering

    lulc_filtering(
        input=input,
        out=out,
        overwrite=overwrite,
        collection_name=collection_name,
        save_lulc_tif=save_lulc_tif,
    )


if __name__ == "__main__":
    ftw()
