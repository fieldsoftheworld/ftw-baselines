import os

import cv2
import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ftw_tools.settings import ALL_COUNTRIES
from ftw_tools.training.datasets import FTW


def compute_canny_edge_sum(
    images: torch.Tensor, low_threshold: float = 50, high_threshold: float = 150
) -> torch.Tensor:
    """Compute Canny edge detection for each band and sum over all bands.

    Args:
        images: Input tensor of shape (B, 8, H, W) where B is batch size,
                8 is number of channels/bands, H and W are height and width
        low_threshold: Lower threshold for Canny edge detection (default: 50)
        high_threshold: Upper threshold for Canny edge detection (default: 150)

    Returns:
        Tensor of shape (B, H, W) containing summed edge maps across all bands
    """
    B, C, H, W = images.shape
    assert C == 8, f"Expected 8 channels, got {C}"

    # Initialize output tensor
    edge_sum = torch.zeros((B, H, W), dtype=torch.float32, device=images.device)

    # Convert to numpy for OpenCV processing
    images_np = images.cpu().numpy()

    # Process each sample in the batch
    for b in range(B):
        # Process each band/channel
        for c in range(C):
            # Get single band and normalize to 0-255 range for Canny
            band = images_np[b, c, :, :]

            # Normalize to 0-255 range
            band_min, band_max = band.min(), band.max()
            if band_max > band_min:
                band_norm = ((band - band_min) / (band_max - band_min) * 255).astype(
                    np.uint8
                )
            else:
                band_norm = np.zeros_like(band, dtype=np.uint8)

            band_norm = np.clip(band_norm, 0, 255).astype(np.uint8)

            # Apply Canny edge detection
            edges = cv2.Canny(band_norm, low_threshold, high_threshold)

            # Add to sum (convert back to float and normalize to 0-1)
            edge_sum[b] += torch.from_numpy(edges).float().to(images.device) / 255.0

    return edge_sum


if __name__ == "__main__":
    for country in ALL_COUNTRIES:
        os.makedirs(f"data/ftw/{country}/label_masks/edges", exist_ok=True)

        ds = FTW(root="data/ftw", countries=country, load_boundaries=True)

        dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=8)

        i = 0
        for batch in tqdm(dl):
            images = batch["image"]  # (B, 8, H, W)
            bs = images.shape[0]
            edges = compute_canny_edge_sum(images)  # (B, H, W)
            for j in range(bs):
                mask_fn = ds.filenames[i]["mask"]
                edge_fn = mask_fn.replace("semantic_3class", "edges")

                if os.path.exists(edge_fn):
                    i += 1
                    continue

                with rasterio.open(mask_fn) as src:
                    profile = src.profile
                profile.update(count=1, dtype=rasterio.uint8, compress="deflate")
                with rasterio.open(edge_fn, "w", **profile) as dst:
                    dst.write((edges[j].cpu().numpy()).astype(np.uint8), 1)
                i += 1
