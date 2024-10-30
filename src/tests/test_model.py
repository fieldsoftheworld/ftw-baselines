import os

from click.testing import CliRunner

from ftw_cli.cli import model_fit, model_test, data_download

CKPT_FILE = "logs/FTW-CI/lightning_logs/version_0/checkpoints/last.ckpt"
CONFIG_FILE = "src/tests/data-files/min_config.yaml"

def test_model_fit():
    runner = CliRunner()

    # Download required data for the fit command
    runner.invoke(data_download, ["--countries=Rwanda"])
    assert os.path.exists("data/ftw/rwanda")
    assert os.path.exists(CONFIG_FILE)
    
    # Run minimal fit
    result = runner.invoke(model_fit, ["-c", CONFIG_FILE])
    assert result.exit_code == 0, result.output
    assert "Train countries: ['rwanda']" in result.output
    assert "`Trainer.fit` stopped: `max_epochs=1` reached." in result.output
    assert "Finished" in result.output
    assert os.path.exists(CKPT_FILE)

def test_model_test():
    runner = CliRunner()

    # Check model for Rwanda
    result = runner.invoke(model_test, [
        "--gpu", "0",
        "--model", CKPT_FILE,
        "--countries", "Rwanda", # should be "rwanda", but let's test case insensitivity
        "--out", "results.csv"
    ])
    assert result.exit_code == 0, result.output
    assert "Running test command" in result.output
    assert "Created dataloader" in result.output
    assert "Object level recall: 0.0000" in result.output
    assert os.path.exists("results.csv")
