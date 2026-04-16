"""Wrapper script for WandB hyperparameter sweeps.

Reads sweep parameters from wandb.config and calls ftw_tools training
with the appropriate CLI overrides.

Usage:
    # 1. Create the sweep (run once):
    wandb sweep configs/dwei/wandb_sweep.yaml

    # 2. Launch one or more agents (each runs one sweep trial):
    wandb agent <entity>/<project>/<sweep-id>
"""

import wandb

from ftw_tools.training import eval as ftw_eval

BASE_CONFIG = "configs/dwei/3_class/full-ftw.yaml"


def train():
    run = wandb.init()
    cfg = wandb.config

    # Start with the base config, then override swept parameters
    overrides = []

    if hasattr(cfg, "lr"):
        overrides += ["--model.init_args.lr", str(cfg.lr)]
    if hasattr(cfg, "loss"):
        overrides += ["--model.init_args.loss", cfg.loss]
    if hasattr(cfg, "backbone"):
        overrides += ["--model.init_args.backbone", cfg.backbone]
    if hasattr(cfg, "omega"):
        # class_weights = [background, interior, boundary] = [0.05, 0.95-ω, ω]
        omega = cfg.omega
        weights = [0.05, round(0.95 - omega, 2), round(omega, 2)]
        overrides += ["--model.init_args.class_weights", str(weights)]

    ftw_eval.fit(config=BASE_CONFIG, ckpt_path=None, cli_args=overrides)

    run.finish()


if __name__ == "__main__":
    train()
