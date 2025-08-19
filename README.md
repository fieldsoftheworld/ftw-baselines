
# Fields of The World (FTW) - Baselines Codebase <!-- omit in toc -->

[**Fields of The World (FTW)**](https://fieldsofthe.world/) is a large-scale benchmark dataset designed to advance machine learning models for instance segmentation of agricultural field boundaries. This dataset supports the need for accurate and scalable field boundary data, which is essential for global agricultural monitoring, land use assessments, and environmental studies.

This repository provides the codebase for working with the [FTW dataset](https://beta.source.coop/repositories/kerner-lab/fields-of-the-world/description/), including tools for data pre-processing, model training, and evaluation.

> [!NOTE]  
> The Fields of The World Command Line Inferface (FTW CLI), published under the name `ftw-tools`, currently lives in this `ftw-baselines` repository due to legacy reasons. We plan to migrate the FTW CLI and related tools into an `ftw-tools` repository soon. Until then, the latest and most complete version of the FTW CLI still lives in `ftw-baselines`.

## Table of Contents <!-- omit in toc -->

- [System setup](#system-setup)
  - [(Ana)conda](#anaconda)
  - [Mamba](#mamba)
  - [Setup FTW CLI](#setup-ftw-cli)
  - [Verify PyTorch installation and CUDA availability](#verify-pytorch-installation-and-cuda-availability)
  - [Development](#development)
- [Predicting field boundaries](#predicting-field-boundaries)
  - [1. Download the model (using `wget`)](#1-download-the-model-using-wget)
  - [2. Download S2 image scene (using `ftw inference download`)](#2-download-s2-image-scene-using-ftw-inference-download)
  - [3. Run inference (using `ftw inference run`)](#3-run-inference-using-ftw-inference-run)
  - [4. Filter predictions by land cover (using `ftw inference filter-by-lulc`)](#4-filter-predictions-by-land-cover-using-ftw-inference-filter-by-lulc)
  - [5. Polygonize the output (using `ftw inference polygonize`)](#5-polygonize-the-output-using-ftw-inference-polygonize)
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

**NOTE:** You need to install Python 3.11 or 3.12 and GDAL with libgdal-arrow-parquet.

To ensure consistent behavior and compatibility, use a dedicated environment to isolate the system requirements to run the FTW CLI (ftw-tools). You can do this through Anaconda or Mamba. Set up the environment using the provided `env.yml` file:

### (Ana)conda

```bash
conda env create -f env.yml
conda activate ftw
```

### Mamba

```bash
mamba env create -f env.yml
mamba activate ftw
```

### Setup FTW CLI

To install the `ftw` command-line tool on your computer, run the following:

```bash
pip install ftw-tools
```

### Verify PyTorch installation and CUDA availability

If you are using GPU, verify that PyTorch and CUDA are installed correctly:

```bash
python -c 'import torch; print(torch.cuda.is_available())'
```

### Development

If you plan to make changes to the FTW CLI at all, you will run one of the following commands from within your cloned repository:

```bash
pip install -e .
```

or for development that includes testing:

```bash
pip install -e .[dev]

# setup pre-commit
pre-commit install
```

This repo uses pre-commit to automatically lint code as you write commits.  You may manually run the linter with `pre-commit run --all-files`.  To confirm you properly downloaded the FTW CLI, run `ftw` in your command line, and you should see the following output:

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

### 1. Download the model (using `wget`)

In order to use `ftw inference`, you need a trained model. You can either download a pre-trained model (FTW pre-trained models can be found in the [Releases](https://github.com/fieldsoftheworld/ftw-baselines/releases) list) or you can train your own model as explained in the [Training](./EXPERIMENTS.md#training) section. This example will use an FTW pre-trained model (with options for either 3 Class or 2 Class).

- Download pretrained checkpoint from [v1](https://github.com/fieldsoftheworld/ftw-baselines/releases/tag/v1).
  - 3 Class

    ```bash
    ftw model download --type THREE_CLASS_FULL
    ```

  - 2 Class

    ```bash
    ftw model download --type TWO_CLASS_FULL
    ```

### 2. Download S2 image scene (using `ftw inference download`)

Steps 2-4 all use `ftw inference`. We provide the `inference` CLI commands to allow users to run models that have been pre-trained on FTW on any temporal pair of S2 images.

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

  If you are looking to download data from the FTW Baseline Dataset, you would use `ftw data download`. You can see an example of this lower on this README at [Dataset setup](#dataset-setup).

### 3. Run inference (using `ftw inference run`)

`ftw inference run` is the command that will run a given model on overlapping patches of input imagery (i.e. the output of `ftw inference download`) and stitch the results together in GeoTIFF format.

```text
ftw inference run --help

Usage: ftw inference run [OPTIONS] INPUT

  Run inference on the stacked Sentinel-2 L2A satellite images specified via
  INPUT.

Options:
  -m, --model PATH         Path to the model checkpoint.  [required]
  -o, --out TEXT           Output filename.  [required]
  --resize_factor INTEGER  Resize factor to use for inference.  [default: 2]
  --gpu INTEGER            GPU ID to use. If not provided, CPU will be used by
                           default.
  --patch_size INTEGER     Size of patch to use for inference. Defaults to
                           1024 unless the image is < 1024x1024px.
  --batch_size INTEGER     Batch size.  [default: 2]
  --padding INTEGER        Pixels to discard from each side of the patch.
                           Defaults to 64 unless the image is < 1024x1024px.
  -f, --overwrite          Overwrite outputs if they exist.
  --mps_mode               Run inference in MPS mode (Apple GPUs).
  --help                   Show this message and exit.
```

Let's run inference on the entire downloaded scene.
  
  ```bash
  ftw inference run inference_imagery/austria_example.tif --model 3_Class_FULL_FTW_Pretrained.ckpt --out austria_example_output_full.tif --gpu 0 --overwrite
  ```

### 4. Filter predictions by land cover (using `ftw inference filter-by-lulc`)

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

### 5. Polygonize the output (using `ftw inference polygonize`)

You can then use the `ftw inference polygonize` command to convert the output of the inference into a vector format (defaults to GeoParquet/[fiboa](https://github.com/fiboa/), with GeoPackage, FlatGeobuf and GeoJSON as other options).

```text
ftw inference polygonize --help

Usage: ftw inference polygonize [OPTIONS] INPUT

  Polygonize the output from inference for the raster image given via INPUT.
  Results are in the CRS of the given raster image.

Options:
  -o, --out TEXT     Output filename for the polygonized data. If not given
                     defaults to the name of the input file with parquet
                     extension. Available file extensions: .parquet
                     (GeoParquet, fiboa-compliant), .fgb (FlatGeoBuf), .gpkg
                     (GeoPackage), .geojson and .json (GeoJSON)
  --simplify FLOAT   Simplification factor to use when polygonizing in the
                     unit of the CRS, e.g. meters for Sentinel-2 imagery in
                     UTM. Set to 0 to disable simplification.  [default: 15]
  --min_size FLOAT   Minimum area size in square meters to include in the
                     output. Set to 0 to disable.  [default: 500]
  --max_size FLOAT   Maximum area size in square meters to include in the
                     output. Disabled by default.
  -f, --overwrite    Overwrite outputs if they exist.
  --close_interiors  Remove the interiors holes in the polygons.
  --help             Show this message and exit.
```

Simplification factor is measured in the units of the coordinate reference system (CRS), and for Sentinel-2 this is meters, so a simplification factor of 15 or 20 is usually sufficient (and recommended, or the vector file will be as large as the raster file).
  
  ```bash
  ftw inference polygonize austria_example_output_full.tif --simplify 20
  ```

This results in a fiboa-compliant file named `austria_example_output_full.parquet`. You can then view this file in QGIS to see something similar to the following image of the sample prediction output. The polygons in red are the predicted fields.

![Sample Prediction Output](/assets/austria_prediction.png)

And that's it! In 4 lines of code, you obtained an FTW model, downloaded S2 data, ran model inference on that data, and polygonized the output to have a final parquet product.

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
