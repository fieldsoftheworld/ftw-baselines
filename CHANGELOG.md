# Changelog

## Model

### v1

- First release

## CLI

### Unreleased

### 1.3.0

- Fixed the overwrite check in `ftw run inference`
- Make inference more robust against Sentinel-2 processing versions 4 and 5
- Warn if the input imagery for inference is not processing version 3 (which was used to train the models)
- Handle patch size and padding inputs more gracefully for smaller images/bboxes < 1024x1024px
- Changed default value of the `padding` parameter in `ftw run inference` (for images < 1024x1024px)

### 1.2.0

- Requires Python 3.10 or 3.11
- Updates dependencies (especially torchgeo and odc-stac)
- Inderence: Parameter `out` is not required any longer
- Polygonization: Reprojects polygons for GeoJSON output to WGS84
- Polygonization: Add `max_size` parameter
- Polygonization: Support NDJSON

### 1.1.1

- Fixed `fiboa inference run` to not show `GDAL signalled an error: Cannot modify tag "PhotometricInterpretation" while writing` any longer

### 1.1.0

- First public release to pypi

### internal legacy non-pypi releases

#### Version 1.0.0 (Public Release)

- Added Dataset download and Unpacking script.
- Using WGET to download the zipped version of the data.
- Visualize notebook is updated to display stats about the images.
- Updated folder structure in README.md

#### Version 1.0.1 (Public Release)

- FTW Cli now supports downloading partial datasets (country wise) alongside downloading entire dataset.
- Updated FTW Cli to support model training and testing.
- `main.py` and `test.py` are removed and merged within the FTW Cli for seamless experimentation.
- Configuration files are updated to use the new `ftw` and `ftw_cli` modules.

#### Version 1.0.2 (Public Release)

- Separate click grouping for model class, which has two separate commands fit and test. Checkout `README.md` for more details.

#### Version 1.0.3 (Public Release)

- `README.md` is cleaned up to keep instruction simple and easy to use.

#### Version 1.0.4 (Public Release)

- Added configuration and evaluation script for Full FTW and CC-BY Countries training.
- Added tensorboard for monitoring training progress.
- Added torchgeo 0.7 version information in the `upcoming features` section.
- Updated ReadMe for inference section for Commercial and Non-Commercial purposes.
- Fixed [Issue 11](https://github.com/fieldsoftheworld/ftw-baselines/issues/11).
- Added Prediction output sample in the `README.md`.
- Integrated inference pipeline with ftw cli.
