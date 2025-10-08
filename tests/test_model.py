import os
import shutil
from pathlib import Path

import pytest
import torch
from click.testing import CliRunner

from ftw_tools.cli import data_download, model_fit, model_test

CKPT_FILE = Path("tests/logs/FTW-CI/lightning_logs/version_0/checkpoints/last.ckpt")
CONFIG_FILE = "tests/data-files/min_config.yaml"


def test_model_fit(caplog):
    versioned_folder = CKPT_FILE.parent.parent
    if versioned_folder.exists():
        shutil.rmtree(versioned_folder)

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


def test_model_test(tmp_path: Path):
    assert CKPT_FILE.exists()

    runner = CliRunner()

    # Check model for Rwanda
    result = runner.invoke(
        model_test,
        [
            "--gpu",
            "0",
            "--model",
            str(CKPT_FILE),
            "--countries",
            "Rwanda",  # should be "rwanda", but let's test case insensitivity
            "--out",
            str(tmp_path / "results.csv"),
        ],
    )
    assert result.exit_code == 0, (
        f"Exited with {result.exit_code}. Output: {result.stdout} {result.stderr}"
    )
    assert "Running test command" in result.output
    assert "Created dataloader" in result.output
    assert "Object level recall: 0.0000" in result.output
    assert os.path.exists(str(tmp_path / "results.csv"))

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
    x = torch.randn(1, 8, 128, 128)
    y = model(x)
    assert y.shape == (1, 3, 128, 128), f"Output shape mismatch for {arch}: {y.shape}"
