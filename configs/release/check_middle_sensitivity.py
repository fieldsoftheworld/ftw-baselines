import argparse
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import segmentation_models_pytorch as smp
import torch
from einops import rearrange
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm

from ftw_tools.inference.models import load_model_from_checkpoint
from ftw_tools.settings import ALL_COUNTRIES
from ftw_tools.training.datasets import FTW
from ftw_tools.training.datamodules import preprocess


def get_corner(data, corner, padding):
    if corner == 0:  # top-left
        return data[:, -(2 * padding) :, -(2 * padding) :]
    elif corner == 1:  # top-right
        return data[:, -(2 * padding) :, : (2 * padding)]
    elif corner == 2:  # bottom-left
        return data[:, : (2 * padding), -(2 * padding) :]
    elif corner == 3:  # bottom-right
        return data[:, : (2 * padding), : (2 * padding)]
    else:
        raise ValueError("Invalid corner")


def main(args: argparse.Namespace):
    device = torch.device(f"cuda:{args.gpu}")

    # Load model
    model, model_type = load_model_from_checkpoint(args.model)
    model = model.eval().to(device)

    # List to store all results
    all_results = []

    # Process each country
    for country in tqdm(ALL_COUNTRIES, desc="Processing countries"):
        print(f"\nProcessing {country}...")

        ds = FTW(
            root="data/ftw",
            countries=[country],
            split=args.split,
            load_boundaries=True,
            verbose=False,
            transforms=preprocess,
        )

        if len(ds) == 0:
            print(f"Skipping {country} - no test data available")
            continue

        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=8)

        # create 4 images of size 128+32 anchored at each corner
        size = 128
        padding = 64

        patch_idx = 0
        for batch in tqdm(dl, desc=f"Processing {country} batches", leave=False):
            images = batch["image"]

            if model_type in ["fcsiamdiff", "fcsiamconc", "fcsiamavg"]:
                images = rearrange(images, "b (t c) h w -> b t c h w", t=2)

            for i in range(images.shape[0]):
                # Support both 4D (B,C,H,W) and 5D (B,T,C,H,W) inputs.
                if images.ndim == 4:  # (B,C,H,W)
                    img1 = images[i : i + 1, :, 0 : size + padding, 0 : size + padding]  # top-left
                    img2 = images[i : i + 1, :, 0 : size + padding, -size - padding :]  # top-right
                    img3 = images[i : i + 1, :, -size - padding :, 0 : size + padding]  # bottom-left
                    img4 = images[i : i + 1, :, -size - padding :, -size - padding :]  # bottom-right
                elif images.ndim == 5:  # (B,T,C,H,W) for fcsiam* models
                    # Correct indexing keeps temporal and channel dimensions intact.
                    img1 = images[i : i + 1, :, :, 0 : size + padding, 0 : size + padding]  # top-left
                    img2 = images[i : i + 1, :, :, 0 : size + padding, -size - padding :]  # top-right
                    img3 = images[i : i + 1, :, :, -size - padding :, 0 : size + padding]  # bottom-left
                    img4 = images[i : i + 1, :, :, -size - padding :, -size - padding :]  # bottom-right
                else:
                    raise ValueError(f"Unsupported image ndim {images.ndim}, expected 4 or 5.")

                batch_tensor = torch.cat([img1, img2, img3, img4], dim=0).to(device)

                with torch.inference_mode():
                    outputs = model(batch_tensor).softmax(dim=1).cpu().numpy()

                out1 = get_corner(outputs[0], 0, padding)
                out2 = get_corner(outputs[1], 1, padding)
                out3 = get_corner(outputs[2], 2, padding)
                out4 = get_corner(outputs[3], 3, padding)
                stacked_outputs = np.stack([out1, out2, out3, out4], axis=0)
                hard_output = stacked_outputs.argmax(axis=1)
                consensus = (
                    (hard_output[0] == hard_output[1])
                    & (hard_output[0] == hard_output[2])
                    & (hard_output[0] == hard_output[3])
                )
                consensus_score = consensus.mean()

                # Store result for this patch
                all_results.append(
                    {
                        "model_checkpoint": args.model,
                        "country": country,
                        "patch_idx": patch_idx,
                        "consensus_score": consensus_score,
                        "split": args.split,
                    }
                )
                patch_idx += 1

        print(f"Processed {patch_idx} patches for {country}")

    # Convert to DataFrame and save
    df = pd.DataFrame(all_results)
    df.to_csv(args.output_fn, index=False)

    print(f"\nResults saved to {args.output_fn}")
    print(f"Total patches processed: {len(df)}")

    # Print summary statistics
    if len(df) > 0:
        print(f"\nOverall statistics:")
        print(
            f"Mean consensus: {df['consensus_score'].mean():.4f} +/- {df['consensus_score'].std():.4f}"
        )
        print(f"Min consensus: {df['consensus_score'].min():.4f}")
        print(f"Max consensus: {df['consensus_score'].max():.4f}")
        print(f"Median consensus: {df['consensus_score'].median():.4f}")

        print(f"\nPer-country statistics:")
        country_stats = (
            df.groupby("country")["consensus_score"]
            .agg(["count", "mean", "std"])
            .round(4)
        )
        print(country_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_fn", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument(
        "--name",
        type=str,
        help="Name to use for output file instead of model checkpoint name",
    )

    args = parser.parse_args()
    main(args)
