
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
      - [Examples:](#examples)
  - [Dataset visualization](#dataset-visualization)
  - [Pre-requisites for experimentation](#pre-requisites-for-experimentation)
- [Experimentation](#experimentation)
  - [Training](#training)
    - [To train a model from scratch:](#to-train-a-model-from-scratch)
    - [To resume training from a checkpoint:](#to-resume-training-from-a-checkpoint)
    - [Visualizing the training process](#visuaizing-training-process)
  - [Testing](#testing)
    - [To test a model:](#to-test-a-model)
  - [Parallel experimentation](#parallel-experimentation)
    - [To run experiments in parallel:](#to-run-experiments-in-parallel)
  - [Inference](#inference)
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

```bash
Usage: ftw [OPTIONS] COMMAND [ARGS]...

  Fields of The World (FTW) - Command Line Interface

Options:
  --help  Show this message and exit.

Commands:
  download  Download the FTW dataset.
  model     Model-related commands.
  unpack    Unpack the downloaded FTW dataset.
```

## Dataset setup

Download the dataset using the `FTW Cli`, `root_folder` defaults to `./data` and `clean_download` is to freshly download the entire dataset(deletes default local folder):

```bash
ftw download --help
Usage: ftw download [OPTIONS]

  Download the FTW dataset.

Options:
  --clean_download    If set, the script will delete the root folder before
                      downloading.
  --root_folder TEXT  Root folder where the files will be downloaded. Defaults
                      to './data'.
  --countries TEXT    Comma-separated list of countries to download. If 'all'
                      is passed, downloads all available countries.
  --help              Show this message and exit.
```

Unpack the dataset using the `unpack.py` script, this will create a `ftw` folder under the `data` after unpacking.

```bash
ftw unpack --help
    Usage: ftw unpack [OPTIONS]

  Unpack the downloaded FTW dataset.

Options:
  --root_folder TEXT  Root folder where the .zip files are located. Defaults
                      to './data'.
  --help              Show this message and exit.
```

#### Examples:
To download and unpack the complete dataset use following commands:

```bash
ftw download 
ftw unpack
```

To download and unpack the specific set of countries use following commands:

```bash
ftw download --countries belgium,kenya,vietnam
ftw unpack
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

```bash
ftw model --help
  Usage: ftw model [OPTIONS] COMMAND [ARGS]...

  Model-related commands.

Options:
  --help  Show this message and exit.

Commands:
  fit        Fit the model
  inference  Run inference on a satellite image
  test       Test the model
```

**Our current best model architecture comprises `unet` with `efficientnet-b3` backbone trained with weighted `cross-entropy` loss. Please see the config files under `configs/FTW-Release` folder for more information.**

## Training

We use [LightningCLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html) to streamline the training process, leveraging configuration files to define the model architecture, dataset, and training parameters.

### To train a model from scratch:

```bash
ftw model fit --help

Usage: ftw model fit [OPTIONS] [CLI_ARGS]...

  Fit the model

Options:
  --config PATH  Path to the config file (required)  [required]
  --help         Show this message and exit.
```

You can train your model using a configuration file as follows:

```bash
ftw model fit --config configs/example_config.yaml
```

### To resume training from a checkpoint:

If training has been interrupted or if you wish to fine-tune a pre-trained model, you can resume training from a checkpoint:

```bash
ftw model fit --config configs/example_config.yaml --ckpt_path <Checkpoint File Path>
```

### VIsuaizing training process

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

```bash
ftw model test --help

Usage: ftw model test [OPTIONS] [CLI_ARGS]...

  Test the model

Options:
  --checkpoint_fn TEXT        Path to model checkpoint  [required]
  --root_dir TEXT             Root directory of dataset
  --gpu INTEGER               GPU to use
  --countries TEXT            Countries to evaluate on  [required]
  --postprocess               Apply postprocessing to the model output
  --iou_threshold FLOAT       IoU threshold for matching predictions to ground
                              truths
  --output_fn TEXT            Output file for metrics
  --model_predicts_3_classes  Whether the model predicts 3 classes or 2
                              classes
  --test_on_3_classes         Whether to test on 3 classes or 2 classes
  --temporal_options TEXT     Temporal option (stacked, windowA, windowB,
                              etc.)
  --help                      Show this message and exit.
```


### To test a model:

Using FTW cli commands to test the model, you can pass specific options, such as selecting the GPUs, providing checkpoints, specifying countries for testing, and postprocessing results:

```bash
ftw model test --gpu 0 --root_dir /path/to/dataset --checkpoint_fn logs/path_to_model/checkpoints/last.ckpt --countries country_to_test_on --output_fn results.csv
```

This will output test results into `results.csv` after running on the selected GPUs and processing the specified countries.

Note: If data directory path is custom (not default ./data/) then make sure to pass custom data directory path in testing using ```--root_dir custom_dir/ftw```.

## Parallel experimentation

For running multiple experiments across different GPUs in parallel, the provided Python script `run_experiments.py` can be used. It efficiently manages and distributes training tasks across available GPUs by using multiprocessing and queuing mechanisms.

### To run experiments in parallel:

1. Define the list of experiment configuration files in the `experiment_configs` list.
2. Specify the list of GPUs in the `GPUS` variable (e.g., `[0,1,2,3]`).
3. Set `DRY_RUN = False` to execute the experiments.

The script automatically detects the available GPUs and runs the specified experiments on them. Each experiment will use the configuration file specified in `experiment_configs`.

```bash
python run_experiments.py
```

The script will distribute the experiments across the specified GPUs using a queue, with each experiment running on the corresponding GPU.

## Inference

We provide two scripts that allow users to run models that have been pre-trained on FTW on any temporal pair of S2 images. First, you need a trained model - either download a pre-trained model (we provide an example pre-trained model in the [Releases](https://github.com/fieldsoftheworld/ftw-baselines/releases) list), or train your own model as explained in the [Training](#training) section. Second, you need to concatenate the bands of two aligned Sentinel-2 scenes that show your area of interest in two seasons (e.g. planting and harvesting seasons) in the following order: B04_t1, BO3_t1, BO2_t1, B08_t1, B04_t2, BO3_t2, BO2_t2, B08_t2 (t1 and t2 represent two different points in time). The `download_imagery.py` script does this automatically given two STAC items. The Microsoft [Planetary Computer Explorer](https://planetarycomputer.microsoft.com/explore?d=sentinel-2-l2a) is a convenient tool for finding relevant scenes and their corresponding STAC items. Finally, `ftw model inference` is a script that will run a given model on overlapping patches of input imagery (i.e. the output of `download_imagery.py`) and stitch the results together in GeoTIFF format.

```bash
ftw model inference --help

Usage: ftw model inference [OPTIONS]

  Run inference on a satellite image

Options:
  --input_fn PATH          Input raster file (Sentinel-2 L2A stack).
                           [required]
  --model_fn PATH          Path to the model checkpoint.  [required]
  --output_fn TEXT         Output filename.  [required]
  --resize_factor INTEGER  Resize factor to use for inference.
  --gpu INTEGER            GPU ID to use. If not provided, CPU will be used by
                           default.
  --patch_size INTEGER     Size of patch to use for inference.
  --batch_size INTEGER     Batch size.
  --padding INTEGER        Pixels to discard from each side of the patch.
  --overwrite              Overwrite outputs if they exist.
  --mps_mode               Run inference in MPS mode (Apple GPUs).
  --help                   Show this message and exit.
```

The following commands show these three steps for a pair of Sentinel-2 scenes over Austria:

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
  python download_imagery.py --win_a "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2B_MSIL2A_20210617T100559_R022_T33UUP_20210624T063729" --win_b "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2B_MSIL2A_20210925T101019_R022_T33UUP_20210926T121923" --output_fn inference_imagery/austria_example.tif
  ```

- Run inference on the entire scene.
  ```bash
  ftw model inference --input_fn inference_imagery/austria_example.tif --model_fn 3_Class_FULL_FTW_Pretrained.ckpt --output_fn austria_example_output_full.tif --gpu 0 --overwrite --resize_factor 2
  ```

### Sample Prediction Output (Austria Patch, Red - Fields)
![Sample Prediction Output](/assets/austria_prediction.png)

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
````

## Contributing

We welcome contributions! Please fork the repository, make your changes, and submit a pull request. For any issues, feel free to open an issue ticket.

## License

This codebase is released under the MIT License. See the [LICENSE](LICENSE) file for details.
