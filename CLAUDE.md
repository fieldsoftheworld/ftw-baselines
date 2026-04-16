# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Fields of The World (FTW)** is a large-scale benchmark dataset and toolkit for instance segmentation of agricultural field boundaries. This repository contains the FTW CLI (`ftw-tools`), a command-line interface for data management, model training, and satellite image inference.

The primary focus is:
- Downloading and preprocessing satellite data (Sentinel-2 imagery)
- Training semantic segmentation and instance segmentation models
- Running inference on satellite imagery to detect field boundaries
- Post-processing results into vector formats (polygons)

## Development Setup

### Environment

The project uses `uv` for dependency management and Python 3.12+:

```bash
# Create and activate environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies in development mode (includes all optional extras and dev tools)
uv sync --all-extras --dev

# To install without delineate-anything feature
uv sync --dev
```

### Building and Installing

```bash
# Build distribution
uv build

# Install/reinstall package in development mode
uv pip install -e .
```

## Common Commands

### Testing

```bash
# Run all non-integration tests with parallel execution (default via pytest.ini)
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_inference.py

# Run a single test function
uv run pytest tests/test_inference.py::test_function_name

# Run only integration tests
uv run pytest tests/ -m integration

# Run without parallelization (remove -n auto)
uv run pytest tests/ -n 0
```

### Linting and Code Quality

```bash
# Set up pre-commit hooks (run once)
uv run pre-commit install

# Run pre-commit on all files
uv run pre-commit run --all-files

# Run ruff linter and formatter
uv run ruff check ftw_tools tests --fix
uv run ruff format ftw_tools tests
```

### Running the CLI

```bash
# Show help for main command
ftw --help

# Show help for specific command group
ftw data --help
ftw inference --help
ftw model --help

# Examples
ftw model list                    # List available models
ftw data download --countries=Rwanda   # Download dataset
ftw inference run --help          # See inference options
```

## Codebase Architecture

### Core Modules

**CLI Entry Point (`ftw_tools/cli.py`)**
- Click-based CLI with three main command groups: `data`, `model`, `inference`
- Lazy imports for ML modules to minimize startup time
- Custom parameter types for validation (e.g., `ModelOrCheckpointParamType`)
- Common option decorators to reduce duplication (e.g., `common_bbox_option()`)

**Download Module (`ftw_tools/download/`)**
- `download_img.py` - Core logic for fetching Sentinel-2 imagery from STAC providers (MSPC, EarthSearch)
- `download_ftw.py` - Dataset downloading and unpacking
- `crop_calendar.py` - Temporal filtering using crop harvest calendars
- Uses `odc.stac`, `pystac`, `planetary_computer`, and `rioxarray` for STAC integration

**Inference Module (`ftw_tools/inference/`)**
- `inference.py` - Core inference pipeline: patching, model execution, stitching results
- `models.py` - Model architectures and loading logic (supports various backbones and checkpoint formats)
- `model_registry.py` - Registry of released models with metadata (task type, inputs, outputs, license)
- `utils.py` - Post-processing utilities like fiboa conversion and polygon merging
- Uses `torchgeo` for geospatial data handling and `torch` for GPU support

**Training Module (`ftw_tools/training/`)**
- `trainers.py` - PyTorch Lightning trainer (`CustomSemanticSegmentationTask`)
- `datasets.py` - Dataset classes for training (RasterDataset subclasses)
- `datamodules.py` - Lightning DataModules for train/val/test splits
- `losses.py` - Custom loss functions (focal losses, Dice variants, etc.)
- `metrics.py` - Custom metrics (object-level metrics beyond pixel-level)
- `eval.py` - Evaluation and metric computation
- Uses `segmentation_models_pytorch`, `lightning`, and `torchmetrics`

**Post-processing Module (`ftw_tools/postprocess/`)**
- `polygonize.py` - Converts raster predictions to vector polygons (GeoParquet, GeoPackage, GeoJSON)
- `lulc_filtering.py` - Filters predictions using land cover/land use masks
- Uses `fiboa` for standardized field output format

### Data Flow

1. **Download**: Sentinel-2 scenes fetched from STAC providers → stacked TIF (bands from two timepoints)
2. **Inference**: Input TIF → patching with overlap → model inference → stitching → output TIF
3. **Post-processing**: Output TIF → polygonization → vector output (default: GeoParquet)
4. **Training**: Paired TIFs + labels → Lightning DataModule → trainer → checkpoint

### Key Dependencies

- **Geospatial**: `rasterio`, `geopandas`, `rioxarray`, `torchgeo`, `odc.stac`, `pystac`
- **ML**: `torch`, `torchvision`, `lightning`, `segmentation_models_pytorch`, `kornia`
- **CLI**: `click`
- **Data**: `dask`, `numpy`, `pandas`, `pyarrow`

## Important Implementation Notes

### Lazy Imports in CLI

The CLI module (`ftw_tools/cli.py`) deliberately avoids importing heavy modules (torch, geopandas, etc.) at the top level. Imports are deferred to individual command functions to minimize startup time. Follow this pattern when adding new commands.

### Configuration and Settings

`ftw_tools/settings.py` centralizes all configuration:
- STAC endpoints and collection IDs
- Supported output formats (polygon, LULC collections)
- Band definitions for different Sentinel-2 collections
- Temporal options and constants

Modify settings here rather than hardcoding values throughout the code.

### Model Registry

The model registry in `ftw_tools/inference/model_registry.py` is the single source of truth for released models. It maps model names to checkpoint URLs, architectures, and metadata. When adding new models:
1. Add entry to the registry
2. Update in `cli_models.py` if needed for visibility

### Testing Pattern

Tests use `conftest.py` for fixtures and shared setup. Integration tests are marked with `@pytest.mark.integration` and excluded from default test runs. When writing tests:
- Use parametrize for testing multiple scenarios
- Fixtures handle temporary files and cleanup
- Integration tests require external resources (STAC APIs, etc.)

### Checkpoint Format

Models are saved as PyTorch Lightning checkpoints (`.ckpt`). The code supports:
- Loading from checkpoint path (must be `.ckpt` extension)
- Loading by model name from registry (automatically downloads if needed)

## Release Process

Follow these steps to publish a new version:

1. Update version in `pyproject.toml`
2. Run `uv build` to create distribution
3. Run full test suite: `uv run pytest tests/`
4. Run `uv publish` (requires credentials)
5. Create GitHub release with `.ckpt` model files
6. Verify installed package works: `pip install ftw-tools==<version>`

See `pyproject.toml` [tool.hatch] section for build exclusions and configurations.
