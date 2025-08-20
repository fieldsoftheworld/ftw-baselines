import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from click.testing import CliRunner

from ftw_tools.cli import data_download, model_download, model_fit, model_test

# todo: if you are not running tests in a clean environment, there might already be a version_0
# and your results may reside in version_1 or version_2 or so, which make the tests fail.
CKPT_FILE = Path("logs/FTW-CI/lightning_logs/version_0/checkpoints/last.ckpt")
CONFIG_FILE = "tests/data-files/min_config.yaml"


def test_model_download1():
    target = "test.ckpt"
    assert not os.path.exists(target), (
        f"{target} should not exist before running the test"
    )

    # Download the model
    runner = CliRunner()
    runner.invoke(model_download, ["--type=TWO_CLASS_FULL", "-o", target])
    assert os.path.exists(target), f"Failed to download model to {target}"

    # cleanup
    os.remove(target)


def test_model_download2():
    filepath = "3_Class_CCBY_FTW_Pretrained.ckpt"
    assert not os.path.exists(filepath), (
        f"{filepath} should not exist before running the test"
    )

    # Download the model
    runner = CliRunner()
    runner.invoke(model_download, ["--type=THREE_CLASS_CCBY"])
    assert os.path.exists(filepath), f"Failed to download model to {filepath}"

    # Test that it does not download again if the file already exists
    result = runner.invoke(model_download, ["--type=THREE_CLASS_CCBY"])
    assert f"File {filepath} already exists, skipping download." in result.output

    # cleanup
    os.remove(filepath)


def test_model_fit(caplog):
    versioned_folder = CKPT_FILE.parent.parent
    assert not versioned_folder.exists(), (
        f"{versioned_folder} should not exist before running the test"
    )

    runner = CliRunner()

    # Download required data for the fit command
    runner.invoke(data_download, ["--countries=Rwanda"])
    assert os.path.exists("data/ftw/rwanda")
    assert os.path.exists(CONFIG_FILE)

    # Run minimal fit
    result = runner.invoke(model_fit, ["-c", CONFIG_FILE])
    assert result.exit_code == 0, result.output
    assert "Train countries: ['rwanda']" in result.output
    assert "`Trainer.fit` stopped: `max_epochs=1` reached." in caplog.text
    assert "Finished" in result.output
    assert CKPT_FILE.exists()


def test_model_test():
    assert CKPT_FILE.exists()

    runner = CliRunner()

    # Check model for Rwanda
    result = runner.invoke(
        model_test,
        [
            "--gpu",
            "0",
            "--model",
            CKPT_FILE,
            "--countries",
            "Rwanda",  # should be "rwanda", but let's test case insensitivity
            "--out",
            "results.csv",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Running test command" in result.output
    assert "Created dataloader" in result.output
    assert "Object level recall: 0.0000" in result.output
    assert os.path.exists("results.csv")
    os.remove("results.csv")

    # cleanup
    versioned_folder = CKPT_FILE.parent.parent
    if versioned_folder.exists():
        shutil.rmtree(versioned_folder)


@pytest.mark.parametrize(
    "arch", ["unet", "deeplabv3+", "fcn", "segformer", "dpt", "upernet"]
)
@torch.inference_mode()
def test_model_archs(arch: str):
    from ftw_tools.torchgeo.trainers import CustomSemanticSegmentationTask

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
    x = torch.randn(1, 8, 256, 256)
    y = model(x)
    assert y.shape == (1, 3, 256, 256), f"Output shape mismatch for {arch}: {y.shape}"


def test_cuda_installation():
    """Test that CUDA is properly installed if GPU hardware is present."""

    def has_nvidia_gpu():
        """Check for NVIDIA GPU hardware independent of PyTorch."""
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    torch_cuda_available = torch.cuda.is_available()
    hardware_gpu_present = has_nvidia_gpu()

    if hardware_gpu_present and not torch_cuda_available:
        pytest.fail(
            "GPU hardware detected via nvidia-smi but PyTorch CUDA not available. "
            "This indicates CUDA libraries may not be properly installed or "
            "PyTorch was not installed with CUDA support."
        )

    if torch_cuda_available:
        assert torch.version.cuda is not None, "CUDA version not detected"
        assert torch.cuda.device_count() > 0, "No CUDA devices found"

        try:
            x = torch.tensor([1.0, 2.0]).cuda()
            y = x * 2
            assert y.is_cuda, "CUDA tensor operations not working"
        except Exception as e:
            pytest.fail(f"CUDA operations failed: {e}")
