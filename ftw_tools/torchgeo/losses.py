from typing import Optional, Tuple, Union

import kornia as K
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss


class Dice(DiceLoss):
    def __init__(self, class_weights=None, use_log_cosh=True, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        )
        self.use_log_cosh = use_log_cosh

    def aggregate_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if self.class_weights is not None:
            weights = self.class_weights.to(loss.device)
            loss = loss * weights

        loss = loss.mean()

        if self.use_log_cosh:
            loss = torch.log(torch.cosh(loss))
        return loss


class DiceCE(nn.Module):
    def __init__(self, weight_ce=0.5, weight_dice=0.5, 
                 ignore_index=None, class_weights=None, 
                 mode="multiclass", classes=None, use_log_cosh=True):
        super().__init__()
        # cross entropy
        ignore_value = -1000 if ignore_index is None else ignore_index
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_value,
            weight=class_weights
        )
        # logcosh dice
        self.dice_loss = Dice(mode=mode, classes=classes, 
                              use_log_cosh=use_log_cosh, 
                              class_weights=class_weights,
                              ignore_index=ignore_index)

        # weighting
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, inputs, targets):
        loss_ce = self.ce_loss(inputs, targets)
        loss_dice = self.dice_loss(inputs, targets)
        return self.weight_ce * loss_ce + self.weight_dice * loss_dice


class PixelWeightedCE(nn.Module):
    """
    Pixel-weighted Cross-Entropy.
    Weights come from blurring a binary map of the target class with a Gaussian.

    Args
    ----
    kernel_size : int or (int, int)
        Gaussian kernel size, e.g. 5 or (5,5).
    sigma : float or (float, float)
        Gaussian sigma in pixels, e.g. 3.0 or (3.0, 3.0).
    scale : float
        Weight multiplier for the target class neighborhood (default 1.0).
    class_weights : Optional[torch.Tensor]
        Per-class weights for cross-entropy (shape [C]).
    target_class : int
        Class index whose neighborhood should be upweighted.
    ignore_index : Optional[int]
        Label to ignore in loss/weight computation (e.g. 255).
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = 5,
        sigma: Union[float, Tuple[float, float]] = 3,
        scale: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        target_class: int = 2,
        ignore_index: Optional[int] = None,
    ):
        super().__init__()
        # normalize args to tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(sigma, (int, float)):
            sigma = (float(sigma), float(sigma))

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.scale = scale
        # Register class_weights as a buffer so it moves with .to(device)
        # If provided, store as float tensor via register_buffer; otherwise None
        if class_weights is not None:
            cw = class_weights.clone().detach().float()
            self.register_buffer("class_weights", cw)
        else:
            self.class_weights = None
        self.target_class = int(target_class)
        self.ignore_index = ignore_index

        # Build the blur op (created once; runs on whatever device/tensor you pass)
        self.blur = K.filters.GaussianBlur2d(
            kernel_size=self.kernel_size, sigma=self.sigma, border_type="reflect"
        )

    @torch.no_grad()
    def _make_weights(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Build per-pixel weights:
          1) E = 1 where masks == target_class else 0 (ignores are 0)
          2) weights = blur(E)
        Returns weights shaped (N, H, W) on masks.device.
        """
        assert masks.ndim == 3, "masks should be (N, H, W) with class indices"
        dtype = torch.float32

        # Binary map for the chosen class
        E = masks == self.target_class
        if self.ignore_index is not None:
            E = E & (masks != self.ignore_index)

        E = E.to(dtype=dtype).unsqueeze(1)  # (N,1,H,W)
        # Kornia modules track device via input; no manual .to needed
        weights = self.blur(E).squeeze(1)  # (N,H,W)

        # If everything is ignored/absent, avoid all-zero weights
        if weights.sum() <= 0:
            weights = torch.zeros_like(weights)

        return (weights * self.scale) + 1.0  # add 1 to ensure min weight is 1.0

    def forward(self, preds: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        preds: (N, C, H, W) logits
        masks: (N, H, W)  int64 class indices
        """
        assert preds.ndim == 4 and masks.ndim == 3, "preds (N,C,H,W), masks (N,H,W)"
        assert preds.shape[0] == masks.shape[0] and preds.shape[2:] == masks.shape[1:]

        # Per-pixel CE, no reduction
        ce = F.cross_entropy(
            preds,
            masks,
            weight=self.class_weights,
            reduction="none",
            ignore_index=-100 if self.ignore_index is None else self.ignore_index,
        )  # (N,H,W), float

        # Build weights from blurred target-class mask
        with torch.no_grad():
            w = self._make_weights(masks)  # (N,H,W)

            # Zero out weights on ignore pixels to be safe
            if self.ignore_index is not None:
                w = torch.where(masks == self.ignore_index, torch.zeros_like(w), w)

        # Weighted average: sum(w * ce) / sum(w)
        num = (w * ce).sum()
        den = w.sum().clamp_min(1e-8)
        return num / den
