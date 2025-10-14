import os
import shutil
from pathlib import Path

import pytest
import torch
import yaml
from click.testing import CliRunner

from ftw_tools.cli import data_download, model_fit, model_test

CONFIG_FILE = "tests/data-files/min_config.yaml"


@pytest.fixture(scope="session")
def shared_data_dir(tmp_path_factory):
    """Download data once and reuse across all tests in this module."""
    import time

    tmp_dir = tmp_path_factory.mktemp("ftw_data")

    runner = CliRunner()

    # Download required data for all tests with exponential backoff
    max_retries = 3
    backoff_delay = 1  # Start with 1 second
    result = None

    for attempt in range(max_retries):
        result = runner.invoke(
            data_download, ["--out", str(tmp_dir / "data"), "--countries=Rwanda"]
        )

        # Check if download was successful
        if result.exit_code == 0:
            break

        # Check if it's a Bad Gateway error that we should retry
        if "502: Bad Gateway" in str(result.stdout) or "502: Bad Gateway" in str(
            result.stderr
        ):
            if attempt < max_retries - 1:  # Don't wait after the last attempt
                print(
                    f"Bad Gateway error on attempt {attempt + 1}, retrying in {backoff_delay} seconds..."
                )
                time.sleep(backoff_delay)
                backoff_delay *= 2  # Exponential backoff: 1s, 2s, 4s...
                continue

        # If we get here, either it's not a retryable error or we've exhausted retries
        break

    assert result is not None and result.exit_code == 0, (
        f"Data download failed with exit code {result.exit_code if result else 'unknown'} after {max_retries} attempts. "
        f"Output: {result.stdout if result else 'N/A'} {result.stderr if result else 'N/A'}"
    )

    # Verify the data was downloaded
    assert (tmp_dir / "data" / "ftw" / "rwanda").exists()

    return tmp_dir


@pytest.fixture(scope="session")
def trained_model_checkpoint(shared_data_dir):
    """Fixture that provides the path to a trained model checkpoint."""
    return (
        shared_data_dir
        / "tests"
        / "logs"
        / "FTW-CI"
        / "lightning_logs"
        / "version_0"
        / "checkpoints"
        / "last.ckpt"
    )


def _create_modified_config_and_train(shared_data_dir):
    """Helper function to create a modified config and train the model."""
    runner = CliRunner()

    # Read the original config file
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    # Update the data root to use our shared data directory
    config["data"]["dict_kwargs"]["root"] = str(shared_data_dir / "data" / "ftw")

    # Update the default_root_dir to use our shared directory for logs
    config["trainer"]["default_root_dir"] = str(
        shared_data_dir / "tests" / "logs" / "FTW-CI"
    )

    # Save the modified config to a temporary file in shared directory
    config_dst = shared_data_dir / "modified_config.yaml"
    with open(config_dst, "w") as f:
        yaml.dump(config, f)

    # Run the training
    result = runner.invoke(model_fit, ["-c", str(config_dst)])

    return result, config_dst


def test_model_fit(caplog, shared_data_dir, trained_model_checkpoint):
    # Create modified config and train the model
    result, config_dst = _create_modified_config_and_train(shared_data_dir)

    # Assert training completed successfully
    assert result.exit_code == 0, (
        f"Exited with {result.exit_code}. Output: {result.stdout} {result.stderr}"
    )
    assert "Train countries: ['rwanda']" in result.output
    assert "`Trainer.fit` stopped: `max_epochs=1` reached." in caplog.text
    assert "Finished" in result.output

    # Verify the checkpoint was created
    assert trained_model_checkpoint.exists()


def test_model_test(shared_data_dir, trained_model_checkpoint, caplog):
    # Ensure we have a trained model - if not, train it first
    if not trained_model_checkpoint.exists():
        result, config_dst = _create_modified_config_and_train(shared_data_dir)
        assert result.exit_code == 0, f"Model training failed: {result.stderr}"
        assert trained_model_checkpoint.exists(), (
            f"Checkpoint was not created at {trained_model_checkpoint}"
        )

    runner = CliRunner()

    # Create results directory
    results_dir = shared_data_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Check model for Rwanda
    result = runner.invoke(
        model_test,
        [
            "--gpu",
            "0",
            "--dir",
            str(shared_data_dir / "data" / "ftw"),
            "--model",
            str(trained_model_checkpoint),
            "--countries",
            "Rwanda",  # should be "rwanda", but let's test case insensitivity
            "--out",
            str(results_dir / "results.csv"),
        ],
    )

    assert "Running test command" in result.output
    assert "Created dataloader" in result.output
    assert "Object level recall" in result.output
    assert (results_dir / "results.csv").exists()


@pytest.mark.parametrize(
    "arch", ["unet", "deeplabv3+", "fcn", "segformer", "dpt", "upernet"]
)
@torch.inference_mode()
def test_model_archs(arch: str):
    from ftw_tools.training.trainers import CustomSemanticSegmentationTask

    params = {
        "class_weights": [0.04, 0.08, 0.88],
        "loss": "ce",
        "backbone": "efficientnet-b3",
        "weights": False,
        "patch_weights": False,
        "in_channels": 8,
        "num_classes": 3,
        "num_filters": 64,
        "ignore_index": 3,
        "lr": 1e-3,
        "patience": 100,
        "model_kwargs": {},
    }
    params["model"] = arch

    if arch == "dpt":
        params["backbone"] = "tu-resnet18"

    model = CustomSemanticSegmentationTask(**params)
    model.eval()
    x = torch.randn(1, 8, 128, 128)
    y = model(x)
    assert y.shape == (1, 3, 128, 128), f"Output shape mismatch for {arch}: {y.shape}"
