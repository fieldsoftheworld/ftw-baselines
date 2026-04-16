
# Fields of The World (FTW) - Baselines Codebase <!-- omit in toc -->

## System setup

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
uv venv --python 3.12
```

Activate your virtual environment:
```bash
# On gitbash:
source .venv/Scripts/activate
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
ftw data download --countries=Austria
ftw model fit -c configs/example_config.yaml
```

#### Development Setup

For development work with testing and linting tools:

```bash

# Run tests
uv run pytest tests/  # uses xdist via default -n auto

# Set up pre-commit hooks (only run this once)
uv run pre-commit install

# Run pre-commit hooks
uv run pre-commit run --all-files
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


## Predicting field boundaries


> **Note**: Make sure you have activated your Python virtual environment before running these commands (e.g., `source venv/bin/activate`).

### FTW Semantic Segmentation Workflow

#### 1. Decide which model you want to use

modify the model yaml file based on `configs\example_config.yaml`

#### 2. Train your own model and save it to the registry 

ftw model fit -c configs/example_config.yaml

#### 3. FTW Inference/Testing

scipts/inference.py

#### 4. Filter predictions by land cover (using `ftw inference filter-by-lulc`)

FTW models are known to make some errors where land parcels that are not cropland (for example, pasture) are segmented as fields. You can try to filter out these errors by filtering the predicted map using a land cover/land use map. The `ftw inference filter-by-lulc` command filters the GeoTIFF predictions raster to only include pixels that are cropland in the land cover map.

```text
ftw inference filter-by-lulc --help

Usage: ftw inference filter-by-lulc [OPTIONS] INPUT

```

#### 6. Polygonize the output (using `ftw inference polygonize`)

You can then use the `ftw inference polygonize` command to convert the output of the inference into a vector format (defaults to GeoParquet/[fiboa](https://github.com/fiboa/), with GeoPackage, FlatGeobuf and GeoJSON as other options).


Simplification factor is measured in the units of the coordinate reference system (CRS), and for Sentinel-2 this is meters, so a simplification factor of 15 or 20 is usually sufficient (and recommended, or the vector file will be as large as the raster file).

  ```bash
  ftw inference polygonize austria_example_output_full.tif --simplify 20
  ```

This results in a fiboa-compliant file named `austria_example_output_full.parquet`. You can then view this file in QGIS to see something similar to the following image of the sample prediction output. The polygons in red are the predicted fields.

#### 7. Post-processing: merging adjacent polygons, removing small polygons, etc.


#### 8. Print results, metrics, and folium plot 
