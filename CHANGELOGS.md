# Changelogs

## Version 1.0.0 (Public Release)

- Added Dataset download and Unpacking script.
- Using WGET to download the zipped version of the data.
- Visualize notebook is updated to display stats about the images.
- Updated folder structure in README.md

## Version 1.0.1 (Public Release)

- FTW Cli now supports downloading partial datasets (country wise) alongside downloading entire dataset.
- Updated FTW Cli to support model training and testing.
- `main.py` and `test.py` are removed and merged within the FTW Cli for seamless experimentation.
- Configuration files are updated to use the new `ftw` and `ftw_cli` modules.

## Version 1.0.2 (Public Release)

- Separate click grouping for model class, which has two separate commands fit and test. Checkout `README.md` for more details.

## Version 1.0.3 (Public Release)

- `README.md` is cleaned up to keep instruction simple and easy to use.

## Version 1.0.4 (Public Release)

- Added configuration and evaluation script for Full FTW and CC-BY Countries training.
- Added tensorboard for monitoring training progress.
- Added torchgeo 0.7 version information in the `upcoming features` section.
- Updated ReadMe for inference section for Commercial and Non-Commercial purposes.
- Fixed [Issue 11](https://github.com/fieldsoftheworld/ftw-baselines/issues/11).
- Added Prediction output sample in the `README.md`.
- Integrated inference pipeline with ftw cli.
