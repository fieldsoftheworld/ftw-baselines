"""Dev run: aug sanity check + short train + test + inference + polygonize.

Usage:
    python scripts/dev_run.py              # full pipeline, no wandb
    python scripts/dev_run.py --wandb      # full pipeline, with wandb
    python scripts/dev_run.py --skip-train # only aug samples (skips all later stages)
"""

import argparse
import os
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_INPUT_TIF = REPO_ROOT / "tests" / "data-files" / "inference-img.tif"

warnings.filterwarnings(
    "ignore",
    message="Default grid_sample and affine_grid behavior has changed",
    category=UserWarning,
    module=r"torch\.nn\.functional",
)


def _rgb_from_tensor(img: torch.Tensor) -> np.ndarray:
    """Take first 3 channels, min-max normalize, return HxWx3 numpy."""
    rgb = img[:3].float().cpu()
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return rgb.permute(1, 2, 0).numpy()


def _load_data_kwargs(config_path: str) -> dict:
    """Read a LightningCLI-style YAML and merge data.init_args + data.dict_kwargs
    so we can instantiate the datamodule with the exact flags from the config.
    """
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {}) or {}
    kwargs = {}
    kwargs.update(data_cfg.get("init_args", {}) or {})
    kwargs.update(data_cfg.get("dict_kwargs", {}) or {})
    return kwargs


def plot_aug_samples(config_path: str, out_dir: Path, n_samples: int = 3, n_reps: int = 4) -> None:
    """Render raw samples and each augmentation in isolation so the effect of
    `preprocess_aug`, `resize_aug`, `random_shuffle`, and `brightness_aug` is
    individually visible. Each per-aug column is forced on (p=1.0). The final
    column(s) show the real `dm.train_aug` pipeline from the config (stochastic).
    """
    import kornia.augmentation as K

    from ftw_tools.training.datamodules import (
        FTWDataModule,
        randomDivisorNormalize,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    data_kwargs = _load_data_kwargs(config_path)
    data_kwargs["num_workers"] = 0
    data_kwargs["batch_size"] = n_samples
    if "num_samples" not in data_kwargs or data_kwargs["num_samples"] in (-1, None):
        data_kwargs["num_samples"] = 32

    print(f"plot_aug_samples data kwargs: {data_kwargs}")

    dm = FTWDataModule(**data_kwargs)
    dm.setup("fit")

    raw_loader = torch.utils.data.DataLoader(
        dm.train_dataset, batch_size=n_samples, shuffle=True, num_workers=0
    )
    raw_batch = next(iter(raw_loader))
    raw_img = raw_batch["image"].float()

    normalize = K.Normalize(mean=dm.mean, std=dm.std)

    def aug_raw(x):
        return normalize(x)

    def aug_preprocess(x):
        return randomDivisorNormalize(x)

    def aug_resize(x):
        return K.RandomResizedCrop(
            (256, 256), scale=(0.3, 0.9), ratio=(0.75, 1.33), p=1.0
        )(normalize(x))

    def aug_shuffle(x):
        # Force the shuffle branch (randomChannelShuffle internally has a coin flip).
        return torch.cat([x[:, 4:8], x[:, :4]], dim=1).contiguous() / 3000.0

    def aug_brightness(x):
        return K.RandomBrightness(p=1.0, brightness=(0.5, 1.5))(normalize(x))

    # Only isolate augs actually enabled in the config. `raw` is always shown
    # as a reference. Each enabled aug is forced p=1.0 so its effect is visible.
    isolated = [("raw", aug_raw)]
    if data_kwargs.get("preprocess_aug"):
        isolated.append(("preprocess_aug", aug_preprocess))
    if data_kwargs.get("resize_aug"):
        isolated.append(("resize_aug", aug_resize))
    if data_kwargs.get("random_shuffle"):
        isolated.append(("random_shuffle", aug_shuffle))
    if data_kwargs.get("brightness_aug"):
        isolated.append(("brightness_aug", aug_brightness))

    n_cols = len(isolated) + n_reps
    fig, axes = plt.subplots(
        n_samples, n_cols, figsize=(4 * n_cols, 4 * n_samples)
    )
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for col, (name, fn) in enumerate(isolated):
        with torch.no_grad():
            out = fn(raw_img.clone())
        for row in range(n_samples):
            ax = axes[row, col]
            ax.imshow(_rgb_from_tensor(out[row]))
            ax.set_title(name if row == 0 else "")
            ax.axis("off")

    # Trailing columns: real stochastic train_aug from the config. Good for
    # spotting the actual mix the trainer will see.
    for rep in range(n_reps):
        batch = {k: v.clone() if torch.is_tensor(v) else v for k, v in raw_batch.items()}
        with torch.no_grad():
            aug_batch = dm.train_aug(batch)
        img = aug_batch["image"] if isinstance(aug_batch, dict) else aug_batch
        col = len(isolated) + rep
        for row in range(n_samples):
            ax = axes[row, col]
            ax.imshow(_rgb_from_tensor(img[row]))
            ax.set_title(f"train_aug #{rep + 1}" if row == 0 else "")
            ax.axis("off")

    # Unverified-but-likely behavior: randomChannelShuffle swaps window A/B
    # channels (indices 0:4 ↔ 4:8). In the shuffle column we apply it
    # deterministically so you can compare the RGB against the raw column —
    # they should look like different dates of the same scene.

    plt.tight_layout()
    grid_path = out_dir / "aug_grid.png"
    fig.savefig(grid_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved aug grid: {grid_path}")


def run_training(config: str, use_wandb: bool) -> None:
    if not use_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    from ftw_tools.training import eval as ftw_eval

    cli_args = ["--trainer.max_epochs=2"]
    ftw_eval.fit(config=config, ckpt_path=None, cli_args=cli_args)


def find_latest_checkpoint() -> Path:
    """Find the most recently written .ckpt anywhere under wandb/ or logs/."""
    candidates = []
    for root in (REPO_ROOT / "wandb", REPO_ROOT / "logs"):
        if root.exists():
            candidates.extend(root.rglob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError("No .ckpt found under wandb/ or logs/ after training")
    # Prefer best.ckpt if present, otherwise most recent file.
    best = [p for p in candidates if p.name == "best.ckpt"]
    if best:
        return max(best, key=lambda p: p.stat().st_mtime)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run(cmd: list[str]) -> None:
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_test(ckpt: Path, out_dir: Path) -> None:
    out_csv = out_dir / "test_metrics.csv"
    _run(
        [
            "ftw", "model", "test",
            "--model", str(ckpt),
            "--countries", "austria",
            "--model_predicts_3_classes",
            "--test_on_3_classes",
            "--out", str(out_csv),
        ]
    )
    print(f"Test metrics: {out_csv}")


def run_inference(ckpt: Path, out_dir: Path) -> Path:
    out_tif = out_dir / "inference.tif"
    _run(
        [
            "ftw", "inference", "run",
            str(SAMPLE_INPUT_TIF),
            "--model", str(ckpt),
            "--out", str(out_tif),
            "--overwrite",
            "--resize_factor", "2",
            "--gpu", "0",
        ]
    )
    print(f"Inference raster: {out_tif}")
    return out_tif


def run_polygonize(inference_tif: Path, out_dir: Path) -> None:
    out_parquet = out_dir / "inference.parquet"
    _run(
        [
            "ftw", "inference", "polygonize",
            str(inference_tif),
            "--out", str(out_parquet),
            "--overwrite",
        ]
    )
    print(f"Polygons: {out_parquet}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dwei/dev.yaml")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging (default off)")
    parser.add_argument("--skip-samples", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-polygonize", action="store_true")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--n-reps", type=int, default=4, help="Aug repetitions per sample")
    parser.add_argument("--out-dir", default="outputs/dev", help="Root output dir for all dev artifacts")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_samples:
        print("=== [1/5] Plotting augmentation samples ===")
        plot_aug_samples(args.config, out_dir, n_samples=args.n_samples, n_reps=args.n_reps)

    if args.skip_train:
        print("skip-train set, stopping before training.")
        return

    print(f"\n=== [2/5] Training 2 epochs (wandb={'on' if args.wandb else 'off'}) ===")
    run_training(args.config, use_wandb=args.wandb)

    ckpt = find_latest_checkpoint()
    print(f"\nUsing checkpoint: {ckpt}")

    if not args.skip_test:
        print("\n=== [3/5] Testing ===")
        run_test(ckpt, out_dir)

    inference_tif = None
    if not args.skip_inference:
        print("\n=== [4/5] Inference ===")
        if not SAMPLE_INPUT_TIF.exists():
            print(f"WARNING: sample input {SAMPLE_INPUT_TIF} missing — skipping inference/polygonize.")
        else:
            inference_tif = run_inference(ckpt, out_dir)

    if not args.skip_polygonize and inference_tif is not None:
        print("\n=== [5/5] Polygonize ===")
        run_polygonize(inference_tif, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
