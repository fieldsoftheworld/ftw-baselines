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
    assert result.exit_code == 0, (
        f"Exited with {result.exit_code}. Output: {result.stdout} {result.stderr}"
    )
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
    assert result.exit_code == 0, (
        f"Exited with {result.exit_code}. Output: {result.stdout} {result.stderr}"
    )
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


def test_expand_countries_with_full_data():
    """Test that expand_countries() correctly expands 'full_data' to all FULL_DATA_COUNTRIES."""
    from ftw_tools.models.baseline_eval import FULL_DATA_COUNTRIES, expand_countries

    # Test basic expansion
    result = expand_countries(["full_data"])

    # Verify that result is now the full list
    assert isinstance(result, list)
    assert len(result) == len(FULL_DATA_COUNTRIES)
    assert set(result) == set(FULL_DATA_COUNTRIES)

    # Verify specific expected countries are included
    expected_countries = ["austria", "belgium", "france", "germany", "netherlands"]
    for country in expected_countries:
        assert country in result, f"Expected country '{country}' not in expanded list"

    # Verify no unexpected values remain
    assert "full_data" not in result, "'full_data' should be replaced, not kept in list"


def test_expand_countries_without_full_data():
    """Test that expand_countries() preserves specific country names when 'full_data' is absent."""
    from ftw_tools.models.baseline_eval import expand_countries

    # Test with specific countries (no full_data)
    input_countries = ["rwanda", "kenya", "belgium"]
    result = expand_countries(input_countries)

    # Should return the same list
    assert result == input_countries
    assert len(result) == 3
    assert "rwanda" in result
    assert "kenya" in result
    assert "belgium" in result


def test_expand_countries_mixed_with_full_data():
    """Test that 'full_data' replaces the entire list when mixed with specific countries."""
    from ftw_tools.models.baseline_eval import FULL_DATA_COUNTRIES, expand_countries

    # Test when full_data is mixed with other countries
    result = expand_countries(["rwanda", "full_data", "kenya"])

    # When full_data is present, it should replace the entire list
    assert set(result) == set(FULL_DATA_COUNTRIES)
    assert len(result) == len(FULL_DATA_COUNTRIES)

    # Original specific countries should not be preserved (full_data replaces everything)
    # This is the current behavior documented by the function


def test_expand_countries_does_not_modify_original():
    """Test that expand_countries() does not modify the original input list."""
    from ftw_tools.models.baseline_eval import expand_countries

    # Test immutability
    original = ["rwanda", "kenya"]
    original_copy = original.copy()
    result = expand_countries(original)

    # Original should be unchanged
    assert original == original_copy
    assert result == original
    assert result is not original  # Should be a different object


def test_full_data_countries_constant():
    """Test that FULL_DATA_COUNTRIES constant is properly defined."""
    from ftw_tools.models.baseline_eval import FULL_DATA_COUNTRIES

    # Verify it's a non-empty list
    assert isinstance(FULL_DATA_COUNTRIES, list)
    assert len(FULL_DATA_COUNTRIES) > 0

    # Verify no duplicates
    assert len(FULL_DATA_COUNTRIES) == len(set(FULL_DATA_COUNTRIES))

    # Verify all entries are strings
    assert all(isinstance(country, str) for country in FULL_DATA_COUNTRIES)

    # Verify all entries are lowercase (consistent with FTW dataset convention)
    assert all(country.islower() for country in FULL_DATA_COUNTRIES)

    # Verify specific known countries are present
    assert "austria" in FULL_DATA_COUNTRIES
    assert "belgium" in FULL_DATA_COUNTRIES
    assert "vietnam" in FULL_DATA_COUNTRIES


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
