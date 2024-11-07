
# Fields of The World (FTW) - Baselines Codebase <!-- omit in toc -->

[**Fields of The World (FTW)**](https://fieldsofthe.world/) is a large-scale benchmark dataset designed to advance machine learning models for instance segmentation of agricultural field boundaries. This dataset supports the need for accurate and scalable field boundary data, which is essential for global agricultural monitoring, land use assessments, and environmental studies.

This repository provides the codebase for working with the [FTW dataset](https://beta.source.coop/repositories/kerner-lab/fields-of-the-world/description/), including tools for data pre-processing, model training, and evaluation.

## Table of Contents <!-- omit in toc -->

- [System setup](#system-setup)
  - [(Ana)conda](#anaconda)
  - [Mamba](#mamba)
  - [Verify PyTorch installation and CUDA availability](#verify-pytorch-installation-and-cuda-availability)
  - [Setup FTW CLI](#setup-ftw-cli)
- [Dataset setup](#dataset-setup)
  - [Examples](#examples)
- [Dataset visualization](#dataset-visualization)
- [Inference](#inference)
  - [Sample Prediction Output (Austria Patch, Red - Fields)](#sample-prediction-output-austria-patch-red---fields)
  - [CC-BY (or equivalent) trained models](#cc-by-or-equivalent-trained-models)
- [Experimentation](#experimentation)
- [Notes](#notes)
- [Upcoming features](#upcoming-features)
- [Contributing](#contributing)
- [License](#license)

## System setup

You need to install Python 3.9 or later and GDAL with libgdal-arrow-parquet.

As a simple way to install the required software you can use Anaconda/Mamba.
Set up the environment using the provided `env.yml` file:

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

### Verify PyTorch installation and CUDA availability

Verify that PyTorch and CUDA are installed correctly (if using a GPU):

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Setup FTW CLI

This creates the `ftw` command-line tool, which is used to download and unpack the data.

```bash
pip install -e .
```

or for development purposes:

```bash
pip install -e .[dev]
```

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

## Dataset setup

Download and unpack the dataset using the FTW CLI.
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

### Examples

To download and unpack the complete dataset use following commands:

```bash
ftw data download
```

To download and unpack the specific set of countries use following commands:

```bash
ftw data download --countries belgium,kenya,vietnam
```

*Note:* Make sure to avoid adding any space in between the list of comma seperated countries.

## Dataset visualization

Explore `visualize_dataset.ipynb` to know more about the dataset.

![Sample 1](/assets/sample1.png)
![Sample 2](/assets/sample2.png)

## Inference

We provide the `inference` cli commands to allow users to run models that have been pre-trained on FTW on any temporal pair of S2 images.

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

First, you need a trained model - either download a pre-trained model (we provide an example pre-trained model in the [Releases](https://github.com/fieldsoftheworld/ftw-baselines/releases) list), or train your own model as explained in the [Training](./EXPERIMENTS.md#training) section.

Second, you need to concatenate the bands of two aligned Sentinel-2 scenes that show your area of interest in two seasons (e.g. planting and harvesting seasons) in the following order: B04_t1, BO3_t1, BO2_t1, B08_t1, B04_t2, BO3_t2, BO2_t2, B08_t2 (t1 and t2 represent two different points in time). The `ftw inference download` command does this automatically given two STAC items. The Microsoft [Planetary Computer Explorer](https://planetarycomputer.microsoft.com/explore?d=sentinel-2-l2a) is a convenient tool for finding relevant scenes and their corresponding STAC items.

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

Then `ftw inference run` is the command that will run a given model on overlapping patches of input imagery (i.e. the output of `ftw inference download`) and stitch the results together in GeoTIFF format.

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
                           [default: 64]
  -f, --overwrite          Overwrite outputs if they exist.
  --mps_mode               Run inference in MPS mode (Apple GPUs).
  --help                   Show this message and exit.
```

You can then use the `ftw inference polygonize` command to convert the output of the inference into a vector format (defaults to GeoParquet/[Fiboa](https://github.com/fiboa/), with GeoPackage, FlatGeobuf and GeoJSON as other options).

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
                     output.  [default: 500]
  -f, --overwrite    Overwrite outputs if they exist.
  --close_interiors  Remove the interiors holes in the polygons.
  --help             Show this message and exit.
```

Simplification factor is measured in the units of the coordinate reference system (CRS), and for Sentinel-2 this is meters, so a simplification factor of 15 or 20 is usually sufficient (and recommended, or the vector file will be as large as the raster file).

The following commands show these four steps for a pair of Sentinel-2 scenes over Austria:

- Download pretrained checkpoint from [Pretrained-Models](https://github.com/fieldsoftheworld/ftw-baselines/releases/tag/Pretrained-Models).
  - 3 Class
    ```bash
    wget https://github.com/fieldsoftheworld/ftw-baselines/releases/download/Pretrained-Models/3_Class_FULL_FTW_Pretrained.ckpt
    ```

  - 2 Class
    ```bash
    wget https://github.com/fieldsoftheworld/ftw-baselines/releases/download/Pretrained-Models/2_Class_FULL_FTW_Pretrained.ckpt
    ```

- Download S2 Image scene.
  
  ```bash
  ftw inference download --win_a S2B_MSIL2A_20210617T100559_R022_T33UUP_20210624T063729 --win_b S2B_MSIL2A_20210925T101019_R022_T33UUP_20210926T121923 --out inference_imagery/austria_example.tif
  ```

  You can also specify a bbox to download a smaller subset of the data, e.g. add `--bbox 13.0,48.0,13.3,48.3`

- Run inference on the entire scene.
  
  ```bash
  ftw inference run inference_imagery/austria_example.tif --model 3_Class_FULL_FTW_Pretrained.ckpt --out austria_example_output_full.tif --gpu 0 --overwrite
  ```

### Sample Prediction Output (Austria Patch, Red - Fields)

![Sample Prediction Output](/assets/austria_prediction.png)

- Polygonize the output.
  
  ```bash
  ftw inference polygonize austria_example_output_full.tif --simplify 20
  ```

This results in a fiboa-compliant file named `austria_example_output_full.parquet`.

### CC-BY (or equivalent) trained models

Consider using CC-BY FTW Trained Checkpoints from the release file for Commercial Purpose, For Non-Commercial Purpose and Academic purpose you can use the FULL FTW Trained Checkpoints (See the Images below for perfrmance comparison)

We have also made FTW model checkpoints available that are pretrained only on CC-BY (or equivalent open licenses) datasets. You can download these checkpoints using the following command:
  
- 3 Class
  
  ```bash
  wget https://github.com/fieldsoftheworld/ftw-baselines/releases/download/Pretrained-Models/3_Class_CCBY_FTW_Pretrained.ckpt
  ```

- 2 Class
  
  ```bash
  https://github.com/fieldsoftheworld/ftw-baselines/releases/download/Pretrained-Models/2_Class_CCBY_FTW_Pretrained.ckpt
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

We have made the dataset compatible with torchgeo for ease of use, and [TorchGeo](https://github.com/microsoft/torchgeo) release 0.7 will include both the dataset and pre-trained models for streamlined integration. To get started, you can install the development version of TorchGeo and load the Fields of the World dataset with the following code:

```bash
pip install git+https://github.com/Microsoft/torchgeo.git  # to get version 0.7 dev
```

```python
from torchgeo.datasets import FieldsOfTheWorld
ds = FieldsOfTheWorld("dataset/", countries="austria", split="train", download=True)
```

## Contributing

We welcome contributions! Please fork the repository, make your changes, and submit a pull request. For any issues, feel free to open an issue ticket.

## License

This codebase is released under the MIT License. See the [LICENSE](LICENSE) file for details.
