import argparse

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from ftw_tools.inference.models import load_model_from_checkpoint
from ftw_tools.settings import ALL_COUNTRIES
from ftw_tools.training.datamodules import preprocess
from ftw_tools.training.datasets import FTW


def extract_overlap_region(
    pred_patch: np.ndarray | torch.Tensor, corner: int, padding: int, overlap_size: int
):
    """Extract the central overlap region of size ``overlap_size`` from a corner patch prediction.

    Let the full image be size D x D. We create corner crops of size C = (D + S)/2, where S is the desired
    central overlap size (``overlap_size``). The padding from each edge before entering the overlap region is
    ``padding = D - C = (D - S)/2``. For each corner, the coordinates of the overlap region within the local
    corner crop are:

    - top-left:     rows [padding : padding + S], cols [padding : padding + S]
    - top-right:    rows [padding : padding + S], cols [0 : S]
    - bottom-left:  rows [0 : S],                 cols [padding : padding + S]
    - bottom-right: rows [0 : S],                 cols [0 : S]
    """
    if corner == 0:  # top-left
        return pred_patch[
            :, padding : padding + overlap_size, padding : padding + overlap_size
        ]
    elif corner == 1:  # top-right
        return pred_patch[:, padding : padding + overlap_size, :overlap_size]
    elif corner == 2:  # bottom-left
        return pred_patch[:, :overlap_size, padding : padding + overlap_size]
    elif corner == 3:  # bottom-right
        return pred_patch[:, :overlap_size, :overlap_size]
    else:
        raise ValueError(f"Invalid corner {corner}; expected 0-3")


def main(args: argparse.Namespace):
    device = torch.device(f"cuda:{args.gpu}")
    overlap_size = args.size

    model, model_type = load_model_from_checkpoint(args.model)
    model = model.eval().to(device)

    all_results = []
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

        # Determine image dimensions from a single sample (avoids building an iterator early).
        sample = ds[0]["image"]
        if model_type in ["fcsiamdiff", "fcsiamconc", "fcsiamavg"]:
            sample = rearrange(sample, "(t c) h w -> t c h w", t=2)
        img_dim = sample.shape[-1]
        if sample.shape[-2] != img_dim:
            raise ValueError("Input images must be square for overlap computation.")
        if not (0 < overlap_size < img_dim):
            raise ValueError(
                f"--size (overlap size) must be between 1 and {img_dim - 1}, got {overlap_size}."
            )
        if (img_dim - overlap_size) % 2 != 0:
            raise ValueError(
                f"Image dim {img_dim} minus overlap size {overlap_size} must be even; try an overlap size with same parity as {img_dim}."
            )

        corner_crop_size = (img_dim + overlap_size) // 2
        padding = img_dim - corner_crop_size  # equals (img_dim - overlap_size)//2

        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=8)
        patch_idx = 0

        for batch in tqdm(dl, desc=f"Processing {country} batches", leave=False):
            images = batch["image"]
            if model_type in ["fcsiamdiff", "fcsiamconc", "fcsiamavg"]:
                images = rearrange(images, "b (t c) h w -> b t c h w", t=2)

            for i in range(images.shape[0]):
                # Support both 4D (B,C,H,W) and 5D (B,T,C,H,W) inputs.
                if images.ndim == 4:  # (B,C,H,W)
                    img1 = images[
                        i : i + 1, :, 0:corner_crop_size, 0:corner_crop_size
                    ]  # top-left
                    img2 = images[
                        i : i + 1, :, 0:corner_crop_size, -corner_crop_size:
                    ]  # top-right
                    img3 = images[
                        i : i + 1, :, -corner_crop_size:, 0:corner_crop_size
                    ]  # bottom-left
                    img4 = images[
                        i : i + 1, :, -corner_crop_size:, -corner_crop_size:
                    ]  # bottom-right
                elif images.ndim == 5:  # (B,T,C,H,W) for fcsiam* models
                    # Preserve temporal dimension.
                    img1 = images[
                        i : i + 1, :, :, 0:corner_crop_size, 0:corner_crop_size
                    ]
                    img2 = images[
                        i : i + 1, :, :, 0:corner_crop_size, -corner_crop_size:
                    ]
                    img3 = images[
                        i : i + 1, :, :, -corner_crop_size:, 0:corner_crop_size
                    ]
                    img4 = images[
                        i : i + 1, :, :, -corner_crop_size:, -corner_crop_size:
                    ]
                else:
                    raise ValueError(
                        f"Unsupported image ndim {images.ndim}, expected 4 or 5."
                    )

                batch_tensor = torch.cat([img1, img2, img3, img4], dim=0).to(device)

                with torch.inference_mode():
                    outputs = model(batch_tensor).softmax(dim=1).cpu().numpy()

                out1 = extract_overlap_region(outputs[0], 0, padding, overlap_size)
                out2 = extract_overlap_region(outputs[1], 1, padding, overlap_size)
                out3 = extract_overlap_region(outputs[2], 2, padding, overlap_size)
                out4 = extract_overlap_region(outputs[3], 3, padding, overlap_size)
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

    df = pd.DataFrame(all_results)
    df.to_csv(args.output_fn, index=False)

    print(f"\nResults saved to {args.output_fn}")
    print(f"Total patches processed: {len(df)}")
    print(f"\nOverall statistics:")
    print(
        f"Mean consensus: {df['consensus_score'].mean():.4f} +/- {df['consensus_score'].std():.4f}"
    )
    print(f"Min consensus: {df['consensus_score'].min():.4f}")
    print(f"Max consensus: {df['consensus_score'].max():.4f}")
    print(f"Median consensus: {df['consensus_score'].median():.4f}")

    print(f"\nPer-country statistics:")
    country_stats = (
        df.groupby("country")["consensus_score"].agg(["count", "mean", "std"]).round(4)
    )
    print(country_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_fn", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Desired size (in pixels) of the central overlap region shared by the four corner crops",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name to use for output file instead of model checkpoint name",
    )

    args = parser.parse_args()
    main(args)
