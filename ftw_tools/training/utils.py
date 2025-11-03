from __future__ import annotations

from typing import Optional

import torch


@torch.no_grad()
def compute_corner_consensus_from_model(
    model: torch.nn.Module,
    image: torch.Tensor,
    size: int = 128,
    padding: int = 64,
) -> Optional[float]:
    """Compute corner consensus by re-running the model on four corner crops.

    This differs from :func:`compute_corner_consensus` which slices a single
    full-image logits tensor. Here we run the model on each corner crop so that
    receptive field / context truncation effects are represented.

    Args:
        model: Segmentation model returning logits of same spatial size as input.
        image: Tensor of shape (C,H,W) on an arbitrary device.
        size: Inner patch size (non-overlapped region extent inside a corner crop).
        padding: Overlap padding. Overlap side length is 2*padding.

    Returns:
        float | None: Consensus in [0,1] or None if the image is too small.
    """
    if image.ndim != 3:
        raise ValueError(f"image must be (C,H,W); got {tuple(image.shape)}")

    _, H, W = image.shape
    patch_side = size + padding
    overlap_side = 2 * padding
    if H < patch_side or W < patch_side or H < overlap_side or W < overlap_side:
        return None

    # Extract corner crops (C, patch_side, patch_side)
    tl_img = image[:, :patch_side, :patch_side]
    tr_img = image[:, :patch_side, -patch_side:]
    bl_img = image[:, -patch_side:, :patch_side]
    br_img = image[:, -patch_side:, -patch_side:]

    device = next(model.parameters()).device
    crops = [tl_img, tr_img, bl_img, br_img]
    batch = torch.stack(crops, dim=0).to(device)
    with torch.inference_mode():
        logits = model(batch)  # (4, C, patch_side, patch_side)
    if logits.ndim != 4 or logits.shape[-1] != patch_side:
        # Allow models that return tuple (logits, aux); pick first tensor
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
    if logits.shape[2] != patch_side or logits.shape[3] != patch_side:
        raise ValueError(
            "Model output spatial size does not match crop size. "
            f"Expected {patch_side}, got {tuple(logits.shape[2:])}."
        )

    tl, tr, bl, br = logits
    r1 = tl[:, -overlap_side:, -overlap_side:]
    r2 = tr[:, -overlap_side:, :overlap_side]
    r3 = bl[:, :overlap_side, -overlap_side:]
    r4 = br[:, :overlap_side, :overlap_side]

    h1 = r1.argmax(dim=0)
    h2 = r2.argmax(dim=0)
    h3 = r3.argmax(dim=0)
    h4 = r4.argmax(dim=0)
    consensus = (h1 == h2) & (h1 == h3) & (h1 == h4)
    return consensus.float().mean().item()


@torch.no_grad()
def batch_corner_consensus_from_model(
    model: torch.nn.Module,
    images: torch.Tensor,
    size: int = 128,
    padding: int = 64,
) -> list[Optional[float]]:
    """Batch version of :func:`compute_corner_consensus_from_model`.

    Args:
        model: Segmentation model.
        images: Tensor (B,C,H,W).
        size: Inner patch size.
        padding: Overlap padding.

    Returns:
        list[float | None]: Per-sample consensus scores.
    """
    scores: list[Optional[float]] = []
    for i in range(images.shape[0]):
        scores.append(
            compute_corner_consensus_from_model(
                model=model, image=images[i], size=size, padding=padding
            )
        )
    return scores
