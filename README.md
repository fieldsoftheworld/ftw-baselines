
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

# Experimentation

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
