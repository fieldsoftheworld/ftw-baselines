
# Fields of The World (FTW) - Baselines Codebase <!-- omit in toc -->

[**Fields of The World (FTW)**](https://fieldsofthe.world/) is a large-scale benchmark dataset designed to advance machine learning models for instance segmentation of agricultural field boundaries. This dataset supports the need for accurate and scalable field boundary data, which is essential for global agricultural monitoring, land use assessments, and environmental studies.

This repository provides the codebase for working with the [FTW dataset](https://beta.source.coop/repositories/kerner-lab/fields-of-the-world/description/), including tools for data pre-processing, model training, and evaluation.

> [!NOTE]
> The Fields of The World Command Line Inferface (FTW CLI), published under the name `ftw-tools`, currently lives in this `ftw-baselines` repository due to legacy reasons. We plan to migrate the FTW CLI and related tools into an `ftw-tools` repository soon. Until then, the latest and most complete version of the FTW CLI still lives in `ftw-baselines`.

## Table of Contents <!-- omit in toc -->

- [System setup](#system-setup)
  - [Using uv (Recommended)](#using-uv-recommended)
    - [Installation](#installation)
    - [Environment Setup](#environment-setup)
    - [Usage](#usage)
    - [Development Setup](#development-setup)
    - [Verify Installation](#verify-installation)
- [Predicting field boundaries](#predicting-field-boundaries)
  - [FTW Semantic Segmentation Baseline Model](#ftw-semantic-segmentation-baseline-model)
    - [1. Decide which model you want to use](#1-decide-which-model-you-want-to-use)
    - [2. FTW Inference all (using `ftw inference all`)](#2-ftw-inference-all-using-ftw-inference-all)
    - [3. Download S2 image scene (using `ftw inference download`)](#3-download-s2-image-scene-using-ftw-inference-download)
    - [4. Run inference (using `ftw inference run`)](#4-run-inference-using-ftw-inference-run)
    - [5. Filter predictions by land cover (using `ftw inference filter-by-lulc`)](#5-filter-predictions-by-land-cover-using-ftw-inference-filter-by-lulc)
    - [6. Polygonize the output (using `ftw inference polygonize`)](#6-polygonize-the-output-using-ftw-inference-polygonize)
  - [Delineate Anything](#delineate-anything)
    - [1. End-to-end inference (using `ftw inference instance-segmentation-all`)](#1-end-to-end-inference-using-ftw-inference-instance-segmentation-all)
- [FTW Baseline Dataset](#ftw-baseline-dataset)
  - [Download the FTW Baseline Dataset](#download-the-ftw-baseline-dataset)
  - [Visualize the FTW Baseline Dataset](#visualize-the-ftw-baseline-dataset)
- [CC-BY vs. the full model](#cc-by-vs-the-full-model)
- [Experimentation](#experimentation)
- [Notes](#notes)
- [Upcoming features](#upcoming-features)
- [Contributing](#contributing)
- [License](#license)

## System setup

To ensure consistent behavior and compatibility, use a dedicated Python virtual environment to isolate the dependencies for the FTW CLI (ftw-tools).

### Using uv (Recommended)


#### Installation

First, install `uv` if you haven't already:

```bash
# On macOS and Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip:
pip install uv
```

#### Environment Setup

Create a virtual environment:
```bash
uv venv
```

Activate your virtual environment:
```bash
# On macOS and Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

Install ftw-tools in development mode:
```bash
uv sync --all-extras --dev
```

#### Usage

Using the FTW CLI:

```bash
# Use the FTW CLI directly (no prefix needed)
ftw --help

# Run any ftw command
ftw data download --countries=Rwanda
ftw model fit -c configs/example_config.yaml
```

#### Development Setup

For development work with testing and linting tools:

```bash

# Run tests
uv run pytest tests/

# Set up pre-commit hooks (only run this once)
uv run pre-commit install

# Run pre-commit hooks
uv run pre-commit run --all-files
```

To install the optional delineate-anything feature:

```bash
uv sync --extra delineate-anything
```

To install everything (all optional dependencies):

```bash
uv sync --all-extras
```

#### Verify Installation

To confirm the FTW CLI is properly installed:

```bash
# Check FTW CLI
ftw --help

# Check PyTorch
uv run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Check geospatial stack
uv run python -c "import rasterio, geopandas; print('Geospatial stack working')"

# Check FTW CLI import
uv run python -c "from ftw_tools.cli import ftw; print('FTW CLI ready')"
```

You should see:

You should see:

```text
Usage: ftw [OPTIONS] COMMAND [ARGS]...

  Fields of The World (FTW) - Command Line Interface

Options:
  --help  Show this message and exit.

Commands:
  data       Downloading, unpacking, and preparing the FTW dataset.
  inference  Running inference on satellite images plus data prep.
  model      Training and testing FTW models.
```

## Predicting field boundaries

The following commands show the steps for using the FTW CLI to obtain the FTW model and data, and then run an inference using that model on that data, and finally polygonizing that output. This example uses a pair of Sentinel-2 (S2) scenes over Austria.

> **Note**: Make sure you have activated your Python virtual environment before running these commands (e.g., `source venv/bin/activate`).

### FTW Semantic Segmentation Baseline Model

#### 1. Decide which model you want to use

In order to use `ftw inference` cli command you need to select one of the existing pre-trained models.
The pre-trained models with descriptions are in the releases portion of the repo, see [here](https://github.com/fieldsoftheworld/ftw-baselines/releases) for more details.

The string representations of the models released are defined in `models/model_registry.py` and are:
* 2_Class_CCBY_v1
* 2_Class_FULL_v1
* 3_Class_CCBY_v1
* 3_Class_FULL_v1
* 3_Class_FULL_singleWindow_v2
* 3_Class_FULL_multiWindow_v2

**Note**: If you want more control ie provide specific Sentinel2 scenes to work with follow steps 3-6 to run each part of the inference pipeline sequentially. There is the option to run step 2 `all` which links together the distinct inference steps. If you decide to run step 2 you will get extracted field boundaries as polygons and don't need to proceed with steps 3-6.

#### 2. FTW Inference all (using `ftw inference all`)

This single CLI call handles the complete inference pipeline: Sentinel-2 scene selection, imagery download, model inference, and polygonization. Sentinel-2 data is selected based on the crop [calendar harvest dates](https://github.com/ucg-uv/research_products/tree/main?tab=readme-ov-file#-citation).

```text
ftw inference all --help

Usage: ftw inference all [OPTIONS]

  Run all inference commands from crop calendar scene selection,then download,
  inference and polygonize.

Options:
  -o, --out PATH                  Directory to save downloaded inference
                                  imagery, and inference output to  [required]
  -m, --model str                String representation of released model name.  [required]
  --year INTEGER RANGE            Year to run model inference over
                                  [2015<=x<=2025; required]
  --bbox TEXT                     Bounding box to use for the download in the
                                  format 'minx,miny,maxx,maxy'
  -ccx, --cloud_cover_max INTEGER RANGE
                                  Maximum percentage of cloud cover allowed in
                                  the Sentinel-2 scene  [default: 20;
                                  0<=x<=100]
  -b, --buffer_days INTEGER RANGE
                                  Number of days to buffer the date for
                                  querying to help balance decreasing cloud
                                  cover and selecting a date near the crop
                                  calendar indicated date.  [default: 14;
                                  x>=0]
  -f, --overwrite                 Overwrites the outputs if they exist
  -r, --resize_factor INTEGER RANGE
                                  Resize factor to use for inference.
                                  [default: 2; x>=1]
  --gpu INTEGER                   GPU to use, zero-based index. Set to -1 to
                                  use CPU. CPU is also always used if CUDA or
                                  MPS is not available.  [default: -1]
  -ps, --patch_size INTEGER RANGE
                                  Size of patch to use for inference. Defaults
                                  to 1024 unless the image is < 1024x1024px
                                  and a smaller value otherwise.  [x>=128]
  -bs, --batch_size INTEGER RANGE
                                  Batch size.  [default: 2; x>=1]
  --num_workers INTEGER RANGE     Number of workers to use for inference.
                                  [default: 4; x>=1]
  -p, --padding INTEGER RANGE     Pixels to discard from each side of the
                                  patch. Defaults to 64 unless the image is <
                                  1024x1024px and a smaller value otherwise.
                                  [x>=0]
  -mps, --mps_mode                Run inference in MPS mode (Apple GPUs).
  --save_scores                   Save segmentation softmax scores (rescaled to [0,255])
                                  instead of classes (argmax of scores)
  -h, --stac_host [mspc|earthsearch]
                                  The host to download the imagery from. mspc
                                  = Microsoft Planetary Computer, earthsearch
                                  = EarthSearch (Element84/AWS).  [default:
                                  mspc]
  -s2, --s2_collection [old-baseline|c1]
                                  Sentinel-2 collection to use with
                                  EarthSearch only: 'old-baseline' =
                                  sentinel-2-l2a, 'c1' = sentinel-2-c1-l2a
                                  (default). Ignored when using MSPC.
                                  [default: c1]
  -v, --verbose                   Enable verbose output showing STAC calls,
                                  scene details, and download URLs.
  --help                          Show this message and exit.
```

Example usage:

```bash
ftw inference all \
    --bbox=13.0,48.0,13.2,48.2 \
    --year=2024 \
    --out=/path/to/output \
    --cloud_cover_max=20 \
    --buffer_days=14 \
    --model=3_Class_FULL_singleWindow_v2 \
    --resize_factor=2 \
    --overwrite
```

This will create the following files in the output directory:

- `inference_data.tif` - The downloaded and stacked Sentinel-2 imagery
- `inference_output.tif` - The raw model inference output
- `polygons.parquet` - The final polygonized field boundaries

#### 3. Download S2 image scene (using `ftw inference download`)

Steps 3-5 all use `ftw inference`. We provide the `inference` CLI commands to allow users to run models that have been pre-trained on FTW on any temporal pair of S2 images.

```text
ftw inference --help

Usage: ftw inference [OPTIONS] COMMAND [ARGS]...

  Inference-related commands.

Options:
  --help  Show this message and exit.

Commands:
  download    Download 2 Sentinel-2 scenes & stack them in a single file...
  polygonize  Polygonize the output from inference
  run         Run inference on the stacked satellite images
```

You need to concatenate the bands of two aligned Sentinel-2 scenes that show your area of interest in two seasons (e.g. planting and harvesting seasons) in the following order: B04_t1, BO3_t1, BO2_t1, B08_t1, B04_t2, BO3_t2, BO2_t2, B08_t2 (t1 and t2 represent two different points in time). The `ftw inference download` command does this automatically given two STAC items. The Microsoft [Planetary Computer Explorer](https://planetarycomputer.microsoft.com/explore?d=sentinel-2-l2a) is a convenient tool for finding relevant scenes and their corresponding STAC items.

To select the timeframe for the two images (Window A and Window B), we looked at the [crop calendar](https://ipad.fas.usda.gov/ogamaps/cropcalendar.aspx) by USDA and found the approximate time for planting and harvesting. For example, if you open the crop calendar and select [China](https://ipad.fas.usda.gov/rssiws/al/crop_calendar/che.aspx), you will find that most of the crops are planted from Feb to May, and harvested from Aug to Nov. We then put these dates as filtering parameters in the Planetary Computer Explorer. Set the cloud threshold to 10% or less. Then select a clear observation that covers the full tile.

```text
ftw inference download --help

Usage: ftw inference download [OPTIONS]

  Download 2 Sentinel-2 scenes & stack them in a single file for inference.

Options:
  --win_a TEXT     URL to or Microsoft Planetary Computer ID of an Sentinel-2
                   L2A STAC item for the window A image  [required]
  --win_b TEXT     URL to or Microsoft Planetary Computer ID of an Sentinel-2
                   L2A STAC item for the window B image  [required]
  -o, --out TEXT   Filename to save results to  [required]
  -f, --overwrite  Overwrites the outputs if they exist
  --bbox TEXT      Bounding box to use for the download in the format
                   'minx,miny,maxx,maxy'
  --help           Show this message and exit.
```

Run this line to download our S2 scenes of interest. This line specifies a bounding box (bbox) to download a smaller subset of the data, with `--bbox 13.0,48.0,13.3,48.3`. If you leave that off you'll get the full S2 scenes downloaded.

  ```bash
  ftw inference download --win_a S2B_MSIL2A_20210617T100559_R022_T33UUP_20210624T063729 --win_b S2B_MSIL2A_20210925T101019_R022_T33UUP_20210926T121923 --out inference_imagery/austria_example.tif --bbox 13.0,48.0,13.3,48.3
  ```

  If you are looking to download data from the FTW Baseline Dataset, you would use `ftw data download`. You can see an example of this lower on this README in the [FTW Baseline Dataset](#ftw-baseline-dataset) section.

#### 4. Run inference (using `ftw inference run`)

`ftw inference run` is the command that will run a given model on overlapping patches of input imagery (i.e. the output of `ftw inference download`) and stitch the results together in GeoTIFF format.

```text
ftw inference run --help

Usage: ftw inference run [OPTIONS] INPUT

  Run inference on the stacked Sentinel-2 L2A satellite images specified via
  INPUT.

Options:
  -m, --model str                 String name of released model, valid model names are defined in `model_registry.py`.  [required]
  -o, --out PATH                  Output filename for the inference imagery.
                                  Defaults to the name of the input file name
                                  with 'inference.' prefix.
  -r, --resize_factor INTEGER RANGE
                                  Resize factor to use for inference.
                                  [default: 2; x>=1]
  --gpu INTEGER                   GPU to use, zero-based index. Set to -1 to
                                  use CPU. CPU is also always used if CUDA or
                                  MPS is not available.  [default: -1]
  -ps, --patch_size INTEGER RANGE
                                  Size of patch to use for inference. Defaults
                                  to 1024 unless the image is < 1024x1024px
                                  and a smaller value otherwise.  [x>=128]
  -bs, --batch_size INTEGER RANGE
                                  Batch size.  [default: 2; x>=1]
  --num_workers INTEGER RANGE     Number of workers to use for inference.
                                  [default: 4; x>=1]
  -p, --padding INTEGER RANGE     Pixels to discard from each side of the
                                  patch. Defaults to 64 unless the image is <
                                  1024x1024px and a smaller value otherwise.
                                  [x>=0]
  -f, --overwrite                 Overwrite outputs if they exist.
  -mps, --mps_mode                Run inference in MPS mode (Apple GPUs).
  --save_scores                   Save segmentation softmax scores (rescaled to [0,255])
                                  instead of classes (argmax of scores)
  --help                          Show this message and exit.
```

Let's run inference on the entire downloaded scene.

  ```bash
  ftw inference run inference_imagery/austria_example.tif --model 3_Class_FULL_FTW_Pretrained.ckpt --out austria_example_output_full.tif --gpu 0 --overwrite
  ```

#### 5. Filter predictions by land cover (using `ftw inference filter-by-lulc`)

FTW models are known to make some errors where land parcels that are not cropland (for example, pasture) are segmented as fields. You can try to filter out these errors by filtering the predicted map using a land cover/land use map. The `ftw inference filter-by-lulc` command filters the GeoTIFF predictions raster to only include pixels that are cropland in the land cover map.

```text
ftw inference filter-by-lulc --help

Usage: ftw inference filter-by-lulc [OPTIONS] INPUT

  Filter the output raster in GeoTIFF format by LULC mask.

Options:
  -o, --out TEXT          Output filename for the (filtered) polygonized data.
                          Defaults to the name of the input file with parquet
                          extension. Available file extensions: .parquet
                          (GeoParquet, fiboa-compliant), .fgb (FlatGeoBuf),
                          .gpkg (GeoPackage), .geojson / .json / .ndjson
                          (GeoJSON)
  -f, --overwrite         Overwrite outputs if they exist.
  --collection_name TEXT  Name of the LULC collection to use. Available
                          collections: io-lulc-annual-v02 (default) and esa-
                          worldcover
  --save_lulc_tif         Save the LULC mask as a GeoTIFF.
  --help                  Show this message and exit.
```

#### 6. Polygonize the output (using `ftw inference polygonize`)

You can then use the `ftw inference polygonize` command to convert the output of the inference into a vector format (defaults to GeoParquet/[fiboa](https://github.com/fiboa/), with GeoPackage, FlatGeobuf and GeoJSON as other options).

```text
ftw inference polygonize --help

Usage: ftw inference polygonize [OPTIONS] INPUT

  Polygonize the output from inference for the raster image given via INPUT.
  Results are in the CRS of the given raster image.

Options:
  -o, --out PATH                  Output filename for the polygonized data.
                                  Defaults to the name of the input file with
                                  '.parquet' file extension. Available file
                                  extensions: .parquet (GeoParquet, fiboa-
                                  compliant), .fgb (FlatGeoBuf), .gpkg
                                  (GeoPackage), .geojson / .json / .ndjson
                                  (GeoJSON)
  -s, --simplify FLOAT RANGE      Simplification factor to use when
                                  polygonizing in the unit of the CRS, e.g.
                                  meters for Sentinel-2 imagery in UTM. Set to
                                  0 to disable simplification.  [default: 15;
                                  x>=0.0]
  -sn, --min_size FLOAT RANGE     Minimum area size in square meters to
                                  include in the output. Set to 0 to disable.
                                  [default: 500; x>=0.0]
  -sx, --max_size FLOAT RANGE     Maximum area size in square meters to
                                  include in the output. Disabled by default.
                                  [x>=0.0]
  -f, --overwrite                 Overwrite output if it exists.
  --close_interiors               Remove the interiors holes in the polygons.
  -st, --stride INTEGER RANGE     Stride size (in pixels) for cutting tif into
                                  smaller tiles for polygonizing. Helps avoid
                                  OOM errors.  [default: 2048; x>=0]
  --softmax_threshold FLOAT RANGE
                                  Threshold on softmax scores for class
                                  predictions. Note: To use this option, you
                                  must pass a tif of scores (using
                                  `--save_scores` option from `ftw inference
                                  run`).  [0<=x<=1]
  -ma, --merge_adjacent FLOAT RANGE
                                  Threshold for merging adjacent polygons.
                                  Threshold is the percent of a polygon's
                                  perimeter touching another polygon.
                                  [0.0<=x<=1.0]
  -ed, --erode_dilate FLOAT RANGE
                                  Distance (in CRS units, e.g., meters) for a
                                  morphological opening (erode then dilate)
                                  applied to each polygon to shave spurs and
                                  remove thin slivers. Set 0 to disable. A
                                  good starting value is 0.5–1x the raster
                                  pixel size.  [default: 0; x>=0.0]
  -de, --dilate_erode FLOAT RANGE
                                  Distance (in CRS units, e.g., meters) for a
                                  morphological closing (dilate then erode)
                                  applied to each polygon to seal hairline
                                  gaps, fill pinholes, and connect near-
                                  touching parts without net growth. Set 0 to
                                  disable. A good starting value is 0.5–1x the
                                  raster pixel size.  [default: 0; x>=0.0]
  -edr, --erode_dilate_raster INTEGER RANGE
                                  Number of iterations for a morphological
                                  opening (erode then dilate) applied to
                                  raster mask before polygonization. Set to 0
                                  to disable.  [default: 0; x>=0]
  -der, --dilate_erode_raster INTEGER RANGE
                                  Number of iterations for a morphological
                                  closing (dilate then erode) applied to
                                  raster mask before polygonization. Set to 0
                                  to disable.  [default: 0; x>=0]
  -tb, --thin_boundaries          Thin boundaries before polygonization using
                                  Zhang-Suen thinning algorithm.
  --help                          Show this message and exit.
```

Simplification factor is measured in the units of the coordinate reference system (CRS), and for Sentinel-2 this is meters, so a simplification factor of 15 or 20 is usually sufficient (and recommended, or the vector file will be as large as the raster file).

  ```bash
  ftw inference polygonize austria_example_output_full.tif --simplify 20
  ```

This results in a fiboa-compliant file named `austria_example_output_full.parquet`. You can then view this file in QGIS to see something similar to the following image of the sample prediction output. The polygons in red are the predicted fields.

![Sample Prediction Output](/assets/austria_prediction.png)

And that's it! In 4 lines of code, you obtained an FTW model, downloaded S2 data, ran model inference on that data, and polygonized the output to have a final parquet product.

### Delineate Anything

#### 1. End-to-end inference (using `ftw inference instance-segmentation-all`)

[Delineate Anything](https://lavreniuk.github.io/Delineate-Anything/) is a pretrained instance segmentation which can detect and segment out individual field boundaries directly to polygons without an intermediate predictions raster. It's trained on the [FBIS-22M](https://huggingface.co/datasets/MykolaL/FBIS-22M) which is a large-scale, multi-resolution dataset comprising 672,909 high-resolution satellite image patches (0.25 m – 10 m) and 22,926,427 instance masks of individual fields. The model comes in two variants: `DelineateAnything` and `DelineateAnything-S`. `DelineateAnything` is the full model and `DelineateAnything-S` is a smaller model that is faster to run (see table below for details). If you use this model in your research, please cite the [Delineate Anything paper](https://arxiv.org/abs/2504.02534).

| Method                   | mAP@0.5 | mAP@0.5:0.95 | Latency (ms) | Size    |
| ------------------------ | ------- | ------------ | ------------ | ------- |
| **Delineate Anything-S** | 0.632   | 0.383        | 16.8         | 17.6 MB |
| **Delineate Anything**   | 0.720   | 0.477        | 25.0         | 125 MB  |

You can run Delineate Anything on a single scene using the `ftw inference instance-segmentation-all` command or optionally on an existing local file using `ftw inference run-instance-segmentation`. See below for examples.

Note that inference uses patching with overlap which will result in duplicate polygons in the overlapping regions. Postprocessing is used to merge polygons via IoU and containment thresholds which are defined by the `--overlap_iou_threshold` and `--overlap_contain_threshold` parameters. For large scenes with many polygons or using a low confidence threshold, this can become computationally slow.

Example usage:

```bash
ftw inference instance-segmentation-all \
    S2B_MSIL2A_20210617T100559_R022_T33UUP_20210624T063729 \
    --bbox=13.0,48.0,13.2,48.2 \
    --out_dir=instance-segmentation-output \
    --gpu=0 \
    --model=DelineateAnything \
    --resize_factor=2 \
    --patch_size=256 \
    --max_detections=100 \
    --iou_threshold=0.3 \
    --conf_threshold=0.05 \
    --simplify=2 \
    --min_size=500 \
    --close_interiors \
    --overlap_iou_threshold=0.2 \
    --overlap_contain_threshold=0.8 \
    --overwrite
```

Usage:

```text
ftw inference instance-segmentation-all --help
Usage: ftw inference instance-segmentation-all [OPTIONS] INPUT

  Run all inference instance segmentation commands from download and
  inference.

Options:
  --bbox TEXT                     Bounding box to use for the download in the
                                  format 'minx,miny,maxx,maxy'
  -o, --out_dir TEXT              Directory to save downloaded inference
                                  imagery, and inference output to  [required]
  -h, --stac_host [mspc|earthsearch]
                                  The host to download the imagery from. mspc
                                  = Microsoft Planetary Computer, earthsearch
                                  = EarthSearch (Element84/AWS).  [default:
                                  mspc]
  -m, --model [DelineateAnything|DelineateAnything-S]
                                  The model to use for inference.  [default:
                                  DelineateAnything]
  --gpu INTEGER RANGE             GPU ID to use. If not provided, CPU will be
                                  used by default.  [x>=0]
  -r, --resize_factor INTEGER RANGE
                                  Resize factor to use for inference.
                                  [default: 2; x>=1]
  -ps, --patch_size INTEGER RANGE
                                  Size of patch to use for inference.
                                  [x>=128]
  -bs, --batch_size INTEGER RANGE
                                  Batch size.  [default: 4; x>=1]
  --num_workers INTEGER RANGE     Number of workers to use for inference.
                                  [default: 4; x>=1]
  --max_detections INTEGER RANGE  Maximum number of detections to keep per
                                  patch.  [default: 100; x>=1]
  -iou, --iou_threshold FLOAT RANGE
                                  IoU threshold for matching predictions to
                                  ground truths  [default: 0.1; 0.0<=x<=1.0]
  -ct, --conf_threshold FLOAT RANGE
                                  Confidence threshold for keeping detections.
                                  [default: 0.1; 0.0<=x<=1.0]
  -p, --padding INTEGER RANGE     Pixels to discard from each side of the
                                  patch.  [x>=0]
  -f, --overwrite                 Overwrites the outputs if they exist
  -mps, --mps_mode                Run inference in MPS mode (Apple GPUs).
  -s, --simplify FLOAT RANGE      Simplification factor to use when
                                  polygonizing in the unit of the CRS, e.g.
                                  meters for Sentinel-2 imagery in UTM. Set to
                                  0 to disable simplification.  [default: 2;
                                  x>=0.0]
  -sn, --min_size FLOAT RANGE     Minimum area size in square meters to
                                  include in the output. Set to 0 to disable.
                                  [default: 500; x>=0.0]
  -sx, --max_size FLOAT RANGE     Maximum area size in square meters to
                                  include in the output. Disabled by default.
                                  [default: 100000; x>=0.0]
  --close_interiors               Remove the interiors holes in the polygons.
                                  [default: True]
  -oit, --overlap_iou_threshold FLOAT RANGE
                                  Overlap IoU threshold for merging polygons.
                                  [default: 0.2; 0.0<=x<=1.0]
  -cot, --overlap_contain_threshold FLOAT RANGE
                                  Overlap containment threshold for merging polygons.
                                  [default: 0.5; 0.0<=x<=1.0]
                                  patch.  [default: 100; x>=1]
  -iou, --iou_threshold FLOAT RANGE
                                  IoU threshold for matching predictions to
                                  ground truths  [default: 0.3; 0.0<=x<=1.0]
  -ct, --conf_threshold FLOAT RANGE
                                  Confidence threshold for keeping detections.
                                  [default: 0.05; 0.0<=x<=1.0]
  -p, --padding INTEGER RANGE     Pixels to discard from each side of the
                                  patch.  [x>=0]
  -f, --overwrite                 Overwrites the outputs if they exist
  -mps, --mps_mode                Run inference in MPS mode (Apple GPUs).
  -s, --simplify FLOAT RANGE      Simplification factor to use when
                                  polygonizing in the unit of the CRS, e.g.
                                  meters for Sentinel-2 imagery in UTM. Set to
                                  0 to disable simplification.  [default: 2;
                                  x>=0.0]
  -sn, --min_size FLOAT RANGE     Minimum area size in square meters to
                                  include in the output. Set to 0 to disable.
                                  [default: 500; x>=0.0]
  -sx, --max_size FLOAT RANGE     Maximum area size in square meters to
                                  include in the output. Disabled by default.
                                  [default: 100000; x>=0.0]
  --close_interiors               Remove the interiors holes in the polygons.
                                  [default: True]
  -oit, --overlap_iou_threshold FLOAT RANGE
                                  Overlap IoU threshold for merging polygons.
                                  [default: 0.2; 0.0<=x<=1.0]
  -cot, --overlap_contain_threshold FLOAT RANGE
                                  Overlap containment threshold for merging
                                  polygons.  [default: 0.8; 0.0<=x<=1.0]
  --help                          Show this message and exit.
```

## FTW Baseline Dataset

Download and unpack the FTW Baseline Dataset using the FTW CLI.
This will create a `ftw` folder under the given folder after unpacking.

```text
ftw data download --help
Usage: ftw data download [OPTIONS]

  Download and unpack the FTW dataset.

Options:
  -o, --out TEXT        Folder where the files will be downloaded to. Defaults
                        to './data'.
  -f, --clean_download  If set, the script will delete the root folder before
                        downloading.
  --countries TEXT      Comma-separated list of countries to download. If
                        'all' (default) is passed, downloads all available
                        countries.
  --no-unpack           If set, the script will NOT unpack the downloaded
                        files.
  --help                Show this message and exit.
```

If you had `--no-unpack` enabled during download, you can manually unpack the downloaded files using the `unpack` command.
This will create a `ftw` folder under the given folder after unpacking.

```text
Usage: ftw data unpack [OPTIONS] [INPUT]

  Unpack the downloaded FTW dataset. Specify the folder where the data is
  located via INPUT. Defaults to './data'.

Options:
  --help  Show this message and exit.
```

### Download the FTW Baseline Dataset

To download and unpack the complete FTW Baseline Dataset, use following command:

```bash
ftw data download
```

To download and unpack the specific country or set of countries, use following command:

```bash
ftw data download --countries belgium,kenya,vietnam
```

*Note:* Make sure to avoid adding any space in between the list of comma seperated countries.

### Visualize the FTW Baseline Dataset

Explore `visualize_dataset.ipynb` to know more about the dataset.

![Sample 1](/assets/sample1.png)
![Sample 2](/assets/sample2.png)

## CC-BY vs. the full model

Consider using CC-BY FTW Trained Checkpoints from the release file for Commercial Purpose. For Non-Commercial Purpose and Academic purpose, you can use the FULL FTW Trained Checkpoints (See the graph below for perfrmance comparison).

We have also made FTW model checkpoints available that are pretrained only on CC-BY (or equivalent open licenses) datasets. You can download these checkpoints using the following command:

- 3 Class

  ```bash
  wget https://github.com/fieldsoftheworld/ftw-baselines/releases/download/v1/3_Class_CCBY_FTW_Pretrained.ckpt
  ```

- 2 Class

  ```bash
  https://github.com/fieldsoftheworld/ftw-baselines/releases/download/v1/2_Class_CCBY_FTW_Pretrained.ckpt
  ```

![3 Class IoU](/assets/3%20Class%20IoU%20Comparison.png)
![2 Class IoU](/assets/2%20Class%20IoU%20Comparison.png)

## Experimentation

For details on the experimentation process, see [Experimentation section](./EXPERIMENTS.md).

## Notes

If you see any warnings in this format:

```bash
/home/byteboogie/miniforge3/envs/ftw/lib/python3.12/site-packages/kornia/feature/lightglue.py:44: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
```

This is due to outdated libraries that rely on an older version of pytorch.
Rest assured `ftw` won't face any issue in experimentation and dataset exploration.

## Upcoming features

Check out the [Issues Section](https://github.com/fieldsoftheworld/ftw-baselines/issues) to see what we are working on and to suggest desired features.

## Contributing

We welcome contributions! Please fork the repository, make your changes, and submit a pull request. For any issues, feel free to open an issue ticket.

## License

This codebase is released under the MIT License. See the [LICENSE](LICENSE) file for details.
