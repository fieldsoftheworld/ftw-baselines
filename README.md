
# Fields of The World (FTW) - Baselines Codebase

**Fields of The World (FTW)** is a large-scale benchmark dataset designed to advance machine learning models for instance segmentation of agricultural field boundaries. This dataset supports the need for accurate and scalable field boundary data, which is essential for global agricultural monitoring, land use assessments, and environmental studies.

This repository provides the codebase for working with the [FTW dataset](https://beta.source.coop/repositories/kerner-lab/fields-of-the-world/description/), including tools for data pre-processing, model training, and evaluation.

## Table of Contents
- [Fields of The World (FTW) - Baselines Codebase](#fields-of-the-world-ftw---baselines-codebase)
  - [Table of Contents](#table-of-contents)
  - [Folder Structure](#folder-structure)
  - [System Setup](#system-setup)
    - [Create Conda/Mamba Environment](#create-condamamba-environment)
    - [Verify PyTorch Installation and CUDA Availability](#verify-pytorch-installation-and-cuda-availability)
    - [Setup FTW CLI](#setup-ftw-cli)
  - [Dataset Setup](#dataset-setup)
    - [Option 1: Download and Unpack the Zipped Version (Recommended)](#option-1-download-and-unpack-the-zipped-version-recommended)
    - [Option 2: Download Using AWS CLI](#option-2-download-using-aws-cli)
      - [AWS CLI Pre-requisites](#aws-cli-pre-requisites)
      - [Obtain AWS Credentials](#obtain-aws-credentials)
  - [Dataset Visualization](#dataset-visualization)
  - [Pre-Requisites for Experimentation](#pre-requisites-for-experimentation)
- [Experimentation](#experimentation)
  - [Training](#training)
    - [To train a model from scratch:](#to-train-a-model-from-scratch)
    - [To resume training from a checkpoint:](#to-resume-training-from-a-checkpoint)
  - [Parallel Experimentation](#parallel-experimentation)
    - [To run experiments in parallel:](#to-run-experiments-in-parallel)
  - [Testing](#testing)
    - [To test a model:](#to-test-a-model)
  - [Contributing](#contributing)
  - [License](#license)

## Folder Structure

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
├── main.py
├── notebooks
│   └── visualize_dataset.ipynb
├── pyproject.toml
├── src
│   ├── ftw
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── download_dataset.py
│   │   └── unpack_dataset.py
│   ├── __init__.py
│   ├── datamodules.py
│   ├── datasets.py
│   ├── metrics.py
│   ├── trainers.py
│   └── utils.py
└── test.py
```

## System Setup

### Create Conda/Mamba Environment
To set up the environment using the provided `env.yml` file:

```bash
mamba env create -f env.yml
mamba activate ftw
```

### Verify PyTorch Installation and CUDA Availability
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
  model     Model Training and Testing
  unpack    Unpack the downloaded FTW dataset.
```

## Dataset Setup

You can download the FTW dataset using one of two methods:

### Option 1: Download and Unpack the Zipped Version (Recommended)

1. Download the dataset using the `FTW Cli`, `root_folder` defaults to `./data` and `clean_download` is to freshly download the entire dataset(deletes default local folder):

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

2. Unpack the dataset using the `unpack_dataset.py` script, this will create a `ftw` folder under the `data` after unpacking.

    ```bash
    ftw unpack --help
        Usage: ftw unpack [OPTIONS]

      Unpack the downloaded FTW dataset.

    Options:
      --root_folder TEXT  Root folder where the .zip files are located. Defaults
                          to './data'.
      --help              Show this message and exit.
    ```

### Option 2: Download Using AWS CLI

#### AWS CLI Pre-requisites
To download the dataset, install AWS CLI 2 by following the [AWS CLI installation guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html). Alternatively, you can install it via the command line:

  ```bash
  curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
  unzip awscliv2.zip
  sudo ./aws/install
  ```

#### Obtain AWS Credentials
To access the dataset, generate AWS credentials (AWS Access Key ID, AWS Secret Access Key, and AWS Session Token) from the Source Cooperative. Follow the instructions on the [Source Cooperative website](https://beta.source.coop/repositories/kerner-lab/fields-of-the-world/download/). To sync the dataset to your local machine using AWS CLI, replace `/path/to/local/
folder` with your preferred location.

```bash
aws s3 sync s3://us-west-2.opendata.source.coop/kerner-lab/fields-of-the-world/ /path/to/local/folder
```

## Dataset Visualization

Explore `visualize_dataset.ipynb` to know more about the dataset.

![Sample 1](/assets/sample1.png)
![Sample 2](/assets/sample2.png)


## Pre-Requisites for Experimentation

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
  Usage: ftw model [OPTIONS] {fit|test} [CLI_ARGS]...

  Model Training and Testing

Options:
  --help  Show this message and exit.
```

## Training

We use [LightningCLI](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html) to streamline the training process, leveraging configuration files to define the model architecture, dataset, and training parameters.

### To train a model from scratch:

You can train your model using a configuration file as follows:

```bash
ftw model fit --config configs/example_config.yaml
```

### To resume training from a checkpoint:

If training has been interrupted or if you wish to fine-tune a pre-trained model, you can resume training from a checkpoint:

```bash
ftw model fit --config configs/example_config.yaml --ckpt_path <Checkpoint File Path>
```

## Parallel Experimentation

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

## Testing

Once your model has been trained, you can evaluate it on the test set specified in your datamodule. This can be done using the same configuration file used for training.

### To test a model:

Using FTW cli commands to test the model, you can pass specific options, such as selecting the GPUs, providing checkpoints, specifying countries for testing, and postprocessing results:

```bash
ftw model test --gpu 0 --checkpoint_fn logs/path_to_model/checkpoints/last.ckpt --countries denmark finland --postprocess --output_fn results.csv
```

This will output test results into `results.csv` after running on the selected GPUs and processing the specified countries.


## Contributing

We welcome contributions! Please fork the repository, make your changes, and submit a pull request. For any issues, feel free to open an issue ticket.

## License

This codebase is released under the MIT License. See the [LICENSE](LICENSE) file for details.
