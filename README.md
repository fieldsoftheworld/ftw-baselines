
# Fields of The World (FTW) - Baselines Codebase

**Fields of The World (FTW)** is a large-scale benchmark dataset designed to advance machine learning models for instance segmentation of agricultural field boundaries. This dataset supports the need for accurate and scalable field boundary data, which is essential for global agricultural monitoring, land use assessments, and environmental studies.

This repository provides the codebase for working with the [FTW dataset](https://beta.source.coop/repositories/kerner-lab/fields-of-the-world/description/), including tools for data pre-processing, model training, and evaluation.

## Table of Contents

- [Fields of The World (FTW) - Baselines Codebase](#fields-of-the-world-ftw---baselines-codebase)
  - [Table of Contents](#table-of-contents)
  - [Folder structure](#folder-structure)
  - [System setup](#system-setup)
    - [Create Conda/Mamba environment](#create-condamamba-environment)
    - [Verify PyTorch installation and CUDA availability](#verify-pytorch-installation-and-cuda-availability)
    - [Setup FTW CLI](#setup-ftw-cli)
  - [Dataset setup](#dataset-setup)
    - [Examples](#examples)
  - [Dataset visualization](#dataset-visualization)
  - [Pre-requisites for experimentation](#pre-requisites-for-experimentation)
- [Experimentation](#experimentation)
  - [Training](#training)
    - [Train a model from scratch](#train-a-model-from-scratch)
    - [Resume training from a checkpoint](#resume-training-from-a-checkpoint)
    - [Visualizing the training process](#visualizing-the-training-process)
  - [Testing](#testing)
    - [Test a model](#test-a-model)
  - [Parallel experimentation](#parallel-experimentation)
    - [Run experiments in parallel](#run-experiments-in-parallel)
  - [Inference](#inference)
    - [Sample Prediction Output (Austria Patch, Red - Fields)](#sample-prediction-output-austria-patch-red---fields)
    - [CC-BY(or equivalent) trained models](#cc-byor-equivalent-trained-models)
  - [Notes](#notes)
  - [Upcoming features](#upcoming-features)
  - [Contributing](#contributing)
  - [License](#license)

## Folder structure

```
Fields-of-The-World
├── .flake8
├── .gitignore
├── CHANGELOGS.md
├── LICENSE
├── README.md
├── assets
├── configs
│   └── example_config.yaml
├── environment.yml
├── inference.py
├── notebooks
│   └── visualize_dataset.ipynb
├── pyproject.toml
└── src
   ├── ftw
   │   ├── __init__.py
   │   ├── datamodules.py
   │   ├── datasets.py
   │   ├── metrics.py
   │   ├── trainers.py
   │   └── utils.py
   └── ftw_cli
       ├── __init__.py
       ├── cli.py
       ├── download.py
       ├── model.py
       └── unpack.py
```

## System setup

### Create Conda/Mamba environment

To set up the environment using the provided `env.yml` file:

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

```
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

Download the dataset using the `FTW Cli`, `root_folder` defaults to `./data` and `clean_download` is to freshly download the entire dataset(deletes default local folder):

```
ftw data download --help
Usage: ftw data download [OPTIONS]

  Download the FTW dataset.

Options:
  -f, --clean_download  If set, the script will delete the root folder before
                        downloading.
  --root_folder TEXT    Root folder where the files will be downloaded.
                        Defaults to './data'.
  --countries TEXT      Comma-separated list of countries to download. If
                        'all' is passed, downloads all available countries.
  --help                Show this message and exit.
```

Unpack the dataset using the `unpack.py` script, this will create a `ftw` folder under the `data` after unpacking.

```
ftw data unpack --help
    Usage: ftw data unpack [OPTIONS]

  Unpack the downloaded FTW dataset.

Options:
  --root_folder TEXT  Root folder where the .zip files are located. Defaults
                      to './data'.
  --help              Show this message and exit.
```

### Examples

To download and unpack the complete dataset use following commands:

```bash
ftw data download 
ftw data unpack
```

To download and unpack the specific set of countries use following commands:

```bash
ftw data download --countries belgium,kenya,vietnam
ftw data unpack
```

*Note:* Make sure to avoid adding any space in between the list of comma seperated countries.

## Dataset visualization

Explore `visualize_dataset.ipynb` to know more about the dataset.

![Sample 1](/assets/sample1.png)
![Sample 2](/assets/sample2.png)

## Pre-requisites for experimentation

Before running experiments, make sure to create configuration files in the `configs` directory. These files should specify the root directory of the dataset. Additionally, update the `root` argument in `datasets.py` to reflect the correct dataset path.

`example_config` gives an idea of the parameters that can be changed to spin out experiments.

```yaml
trainer:
  max_epochs: <E.G. 100, NUMBER OF EPOCHS>
  log_every_n_steps: <E.G. 10, LOGGING FREQUENCY>
  accelerator: <E.G. "gpu", ACCELERATOR>
  default_root_dir: <LOGS DIRECTORY>
  devices:
    - <DEVICE ID>
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        monitor: val_loss
        mode: min
        save_top_k: <E.G. 0, NUMBER OF MODELS TO SAVE>
        save_last: <TRUE / FALSE, WHETHER TO SAVE THE LAST MODEL OR NOT>
        filename: "{epoch}-{val_loss:.2f}"
model:
  class_path: ftw.trainers.CustomSemanticSegmentationTask
  init_args:
    loss: <E.G. "jaccard", LOSS FUNCTION>
    model: <E.G. "unet", MODEL>
    backbone: <E.G. "efficientnet-b3", BACKBONE MODEL>
    weights: <TRUE / FALSE, WHETHER TO USE PRETRAINED WEIGHTS OR NOT>
    patch_weights : <TRUE / FALSE, WHETHER TO PATCH THE WEIGHTS IN A CUSTOM FORMAT OR NOT>
    in_channels: <E.G. 8, NUMBER OF INPUT CHANNELS>
    num_classes: <E.G. 3, NUMBER OF CLASSES>
    num_filters: <E.G. 64, NUMBER OF FILTERS>
    ignore_index: null
    lr: <E.G. 1e-3, LEARNING RATE>
    patience: <E.G. 100, PATIENCE FOR COSINE ANNEALING>
data:
  class_path: ftw.datamodules.FTWDataModule
  init_args:
    batch_size: 32
    num_workers: 8
    num_samples: -1
    train_countries:
      - country 1
      - country 2
    val_countries:
      - country 1
      - country 2
    test_countries:
      - country 1
      - country 2
  dict_kwargs:
    root: <ROOT FOLDER OF THE DATASET>
    load_boundaries: <TRUE / FALSE WHETHER TO LOAD 3 CLASS MASKS OR NOT>
seed_everything: <SEED VALUE>
```

# Experimentation

This section provides guidelines for running model training, testing, and experimentation using multiple GPUs and configuration files.

```
ftw model --help
  Usage: ftw model [OPTIONS] COMMAND [ARGS]...

  Model-related commands.

Options:
  --help  Show this message and exit.

Commands:
  fit        Fit the model
  test       Test the model
```

**Our current best model architecture comprises `unet` with `efficientnet-b3` backbone trained with weighted `cross-entropy` loss. Please see the config files under `configs/FTW-Release` folder for more information.**

## Training

We use [LightningCLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html) to streamline the training process, leveraging configuration files to define the model architecture, dataset, and training parameters.

### Train a model from scratch

```
ftw model fit --help

Usage: ftw model fit [OPTIONS] [CLI_ARGS]...

  Fit the model

Options:
  --config PATH     Path to the config file (required)  [required]
  --ckpt_path PATH  Path to a checkpoint file to resume training from
  --help            Show this message and exit.
```

You can train your model using a configuration file as follows:

```bash
ftw model fit --config configs/example_config.yaml
```

### Resume training from a checkpoint

If training has been interrupted or if you wish to fine-tune a pre-trained model, you can resume training from a checkpoint:

```bash
ftw model fit --config configs/example_config.yaml --ckpt_path <Checkpoint File Path>
```

### Visualizing the training process

Tensorboard is used to log the training and validation steps. The logs are situated in folders specified in the `config.yaml` files. Search for this line in the configuration. 

```
default_root_dir: logs/<experiment name>
```

To visualize the the logs run:

```
tensorboard --logdir=<log directory of experiments>
```

Currently logged informations are:
- Train multi class accuracy
- Train multi class jaccard index
- Train loss
- Validation multi class accuracy
- Validation multi class jaccard index
- Validation loss
- 10 prediction results (images)
- Learning rate

## Testing

Once your model has been trained, you can evaluate it on the test set specified in your datamodule. This can be done using the same configuration file used for training.

```
ftw model test --help

Usage: ftw model test [OPTIONS] [CLI_ARGS]...

  Test the model

Options:
  --checkpoint TEXT           Path to model checkpoint  [required]
  --root_dir TEXT             Root directory of dataset
  --gpu INTEGER               GPU to use
  --countries TEXT            Countries to evaluate on  [required]
  --postprocess               Apply postprocessing to the model output
  --iou_threshold FLOAT       IoU threshold for matching predictions to ground
                              truths
 -o, --output TEXT            Output file for metrics
  --model_predicts_3_classes  Whether the model predicts 3 classes or 2
                              classes
  --test_on_3_classes         Whether to test on 3 classes or 2 classes
  --temporal_options TEXT     Temporal option (stacked, windowA, windowB,
                              etc.)
  --help                      Show this message and exit.
```


### Test a model

Using FTW cli commands to test the model, you can pass specific options, such as selecting the GPUs, providing checkpoints, specifying countries for testing, and postprocessing results:

```bash
ftw model test --gpu 0 --root_dir /path/to/dataset --checkpoint logs/path_to_model/checkpoints/last.ckpt --countries country_to_test_on --output results.csv
```

This will output test results into `results.csv` after running on the selected GPUs and processing the specified countries.

Note: If data directory path is custom (not default ./data/) then make sure to pass custom data directory path in testing using ```--root_dir custom_dir/ftw```.

## Parallel experimentation

For running multiple experiments across different GPUs in parallel, the provided Python script `run_experiments.py` can be used. It efficiently manages and distributes training tasks across available GPUs by using multiprocessing and queuing mechanisms.

### Run experiments in parallel

1. Define the list of experiment configuration files in the `experiment_configs` list.
2. Specify the list of GPUs in the `GPUS` variable (e.g., `[0,1,2,3]`).
3. Set `DRY_RUN = False` to execute the experiments.

The script automatically detects the available GPUs and runs the specified experiments on them. Each experiment will use the configuration file specified in `experiment_configs`.

```bash
python run_experiments.py
```

The script will distribute the experiments across the specified GPUs using a queue, with each experiment running on the corresponding GPU.

## Inference

We provide the `inference` cli commands to allow users to run models that have been pre-trained on FTW on any temporal pair of S2 images. 

```
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

First, you need a trained model - either download a pre-trained model (we provide an example pre-trained model in the [Releases](https://github.com/fieldsoftheworld/ftw-baselines/releases) list), or train your own model as explained in the [Training](#training) section. 

Second, you need to concatenate the bands of two aligned Sentinel-2 scenes that show your area of interest in two seasons (e.g. planting and harvesting seasons) in the following order: B04_t1, BO3_t1, BO2_t1, B08_t1, B04_t2, BO3_t2, BO2_t2, B08_t2 (t1 and t2 represent two different points in time). The `ftw inference download` command does this automatically given two STAC items. The Microsoft [Planetary Computer Explorer](https://planetarycomputer.microsoft.com/explore?d=sentinel-2-l2a) is a convenient tool for finding relevant scenes and their corresponding STAC items. 

```
ftw inference download --help

Usage: ftw inference download [OPTIONS]

  Download 2 Sentinel-2 scenes & stack them in a single file for inference.

Options:
  --win_a TEXT       Path to a Sentinel-2 STAC item for the window A image
                     [required]
  --win_b TEXT       Path to a Sentinel-2 STAC item for the window B image
                     [required]
  -o, --output TEXT  Filename to save results to  [required]
  -f, --overwrite    Overwrites the outputs if they exist
  --help             Show this message and exit.
```

Then `ftw inference run` is the command that will run a given model on overlapping patches of input imagery (i.e. the output of `ftw inference download`) and stitch the results together in GeoTIFF format. 

```
ftw inference run --help

Usage: ftw inference run [OPTIONS] INPUT

  Run inference on the stacked Sentinel-2 L2A satellite images specified in
  INPUT.

Options:
  -m, --model PATH         Path to the model checkpoint.  [required]
  -o, --output TEXT        Output filename.  [required]
  --resize_factor INTEGER  Resize factor to use for inference.
  --gpu INTEGER            GPU ID to use. If not provided, CPU will be used by
                           default.
  --patch_size INTEGER     Size of patch to use for inference.
  --batch_size INTEGER     Batch size.
  --padding INTEGER        Pixels to discard from each side of the patch.
  -f, --overwrite          Overwrite outputs if they exist.
  --mps_mode               Run inference in MPS mode (Apple GPUs).
  --help                   Show this message and exit.
```

You can then use the `ftw inference polygonize` command to convert the output of the inference into a vector format (initially GeoPackage, GeoParquet/Fiboa coming soon).

```
ftw inference polygonize --help

Usage: ftw inference polygonize [OPTIONS] INPUT

  Polygonize the output from inference for the raster image given via INPUT.

Options:
  -o, --output TEXT  Output filename for the polygonized data.  [required]
  --simplify FLOAT   Simplification factor to use when polygonizing.
  -f, --overwrite    Overwrite outputs if they exist.
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
  ftw inference download --win_a "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2B_MSIL2A_20210617T100559_R022_T33UUP_20210624T063729" --win_b "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2B_MSIL2A_20210925T101019_R022_T33UUP_20210926T121923" --output inference_imagery/austria_example.tif
  ```

- Run inference on the entire scene.
  ```bash
  ftw inference run inference_imagery/austria_example.tif --model 3_Class_FULL_FTW_Pretrained.ckpt --output austria_example_output_full.tif --gpu 0 --overwrite --resize_factor 2
  ```

### Sample Prediction Output (Austria Patch, Red - Fields)

![Sample Prediction Output](/assets/austria_prediction.png)

- Polygonize the output.
  ```bash
  ftw inference polygonize austria_example_output_full.tif --output austria_example_output_full.gpkg --simplify 20
  ```

### CC-BY(or equivalent) trained models

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

## Notes

If you see any warnings in this format,

```bash
/home/byteboogie/miniforge3/envs/ftw/lib/python3.12/site-packages/kornia/feature/lightglue.py:44: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
```

this is due to a PR in official PyTorch `PyTorch 2.4 deprecated the use of torch.cuda.amp.autocast in favor of torch.amp.autocast("cuda", ...), but this change has missed updating internal uses in PyTorch` [Link](https://github.com/pytorch/pytorch/issues/130659), rest assured `ftw` won't face any issue in experimentation and dataset exploration.

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
